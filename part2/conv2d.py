import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    c_out_pmax = nl.tile_size.pmax
    n_tiles_c_out = out_channels // c_out_pmax

    # - load in the weights into an SBUF array of shape (n_tiles_out_channels, nl.par_dim(c_out_pmax), n_tiles_in_channels, 128, kernel_height, kernel_width)
    W_sbuf = nl.ndarray(
        shape=(n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )


    for inpt_c in nl.sequential_range(n_tiles_c_out):
        for outpt_c in nl.sequential_range(n_tiles_c_in):
            # load using nl.load
            i_x = inpt_c * c_out_pmax
            i_y = outpt_c * c_in_pmax
            W_sbuf[inpt_c, :, outpt_c, :, :, :] = nl.load(W[i_x:i_x + c_out_pmax, i_y:i_y + c_in_pmax, :, :])


            # W_sbuf[inpt_c, :, outpt_c, :, :, :] = nl.load(W[inpt_c * c_out_pmax:(inpt_c + 1) * c_out_pmax, outpt_c * c_in_pmax:(outpt_c + 1) * c_in_pmax, :, :])




    # - move data around using nl.copy to get an array of shape (kernel_height, kernel_width, n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_out_pmax), c_in_pmax)
    moved_W = nl.ndarray(
        shape=(filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )



    for filter_row in nl.sequential_range(filter_height):
        for filter_col in nl.sequential_range(filter_width):
            for inpt_c in nl.sequential_range(n_tiles_c_out):
                for outpt_c in nl.sequential_range(n_tiles_c_in):
                    moved_W[filter_row, filter_col, inpt_c, outpt_c, :, :] = nl.copy(W_sbuf[inpt_c, :, outpt_c, :, filter_row, filter_col])
            

    # - transpose that to get an array of shape (kernel_height, kernel_width, n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_in_pmax), c_out_pmax), call this w
    for filter_row in nl.sequential_range(filter_height):
        for filter_col in nl.sequential_range(filter_width):
            for inpt_c in nl.sequential_range(n_tiles_c_out):
                for outpt_c in nl.sequential_range(n_tiles_c_in):
                    moved_W[filter_row, filter_col, inpt_c, outpt_c, :, :] = nl.transpose(moved_W[filter_row, filter_col, inpt_c, outpt_c, :, :])


    #  the weights now are prepared for matrix multiply





    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # TODO: Perform the convolution of X[b] with the weights W and bias b, followed by a maxpool
        # and store the result in X_out[b]

        # broad algorithm:
        # W.shape (out_channels, in_channels, kernel_height, kernel_width)
        # X.shape (batch_size, in_channels, image_hegiht, image_width)


        # First of all, convince yourself that the matmuls would be carried out much more naturally
        # if the weights are in the shape (kernel_height, kernel_width, in_channels, out_channels)

        # both c_in_pmax and c_out_pmax are just 128, they are just used to represent which dimension they are encoding.
        # n_tiles_c_in in in_channels // 128, n_tiles_c_out is out_channels // 128

        

        # Once we have this, we have prepared the weights for matrix multiply.

        # loop over batch:
        #     - assign space in SBUF to store entire image, call it x
            # shape : (n_tiles_c_in, nl.par_dim(c_in_pmax), image_height, image_width)
            # loop over n_tiles_c_in:
            #     - load corresponding part of input image

            # loop over n_tiles_c_out:
            #     - assign space in SBUF to store output
                # shape : (nl.par_dim(c_out_pmax), out_height, out_width)
                # loop over output_rows:
                #     - assign space in PSUM to store output row
                #     for filter_row in nl.affine_range(kernel_height): # WAS: loop over kernel_height:
                #         for filter_col in nl.affine_range(kernel_width):    # WAS: loop over kernel_width:
                #             for curr_c_in_tile in nl.affine_range(n_tiles_c_in):    # WAS: loop over n_tiles_c_in:
                #                 - matmul w[filter_row, filter_col, n_tile_c_out, n_tile_cin, :, :].T with
                #                 x[curr_c_in_tile, :, out_row + filter_row, kernel_width:kernel_width + filter_col]
                                
                                # MATMUL WAS: matmul w[kernel_height, kernel_width, n_tile_c_out, n_tile_cin, :, :].T with x[n_tiles_c_in, :, out_row + kernel_height, kernel_width:kernel_width + out_width]
                #     - copy stuff from PSUM back to SBUF
                # - copy stuff from SBUF back to HBM

        
        # assign space in SBUF to store entire image, call it x
        # shape : (n_tiles_c_in, nl.par_dim(c_in_pmax), image_height, image_width)
        x = nl.ndarray(
            shape=(n_tiles_c_in, nl.par_dim(c_in_pmax), input_height, input_width),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )

        # loop over n_tiles_c_in:
        for inpt_c in nl.sequential_range(n_tiles_c_in):
            # load corresponding part of input image
            x[inpt_c, :, :, :] = nl.load(X[b, inpt_c * c_in_pmax:(inpt_c + 1) * c_in_pmax, :, :])

        # assign space in SBUF to store output
        # shape : (nl.par_dim(c_out_pmax), out_height, out_width)
        out = nl.ndarray(
            shape=(nl.par_dim(c_out_pmax), out_height, out_width),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )

        # loop over n_tiles_c_out:
        for outpt_c in nl.sequential_range(n_tiles_c_out):
            # assign space in PSUM to store output row
            psum = nl.ndarray(
                shape=(out_height, out_width),
                dtype=X.dtype,
                buffer=nl.psum,
            )

            # loop over output_rows:
            for out_row in nl.affine_range(out_height):
                for filter_row in nl.affine_range(filter_height):
                    for filter_col in nl.affine_range(filter_width):
                        for curr_c_in_tile in nl.affine_range(n_tiles_c_in):
                            # matmul w[filter_row, filter_col, n_tile_c_out, n_tile_cin, :, :].T with
                            # x[curr_c_in_tile, :, out_row + filter_row, kernel_width:kernel_width + filter_col]
                            psum[out_row, :] += nl.matmul(moved_W[filter_row, filter_col, outpt_c, curr_c_in_tile, :, :].T, x[curr_c_in_tile, :, out_row + filter_row, out_width:out_width + filter_col])

            # copy stuff from PSUM back to SBUF


        



        continue

    return X_out
