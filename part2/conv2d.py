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

    # W = W.reshape((n_tiles_c_out, c_out_pmax, n_tiles_c_in, c_in_pmax, filter_height, filter_width))
    
    # weight_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    # weight_copy = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax), dtype=W.dtype, buffer=nl.sbuf)
    # w = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax), dtype=W.dtype, buffer=nl.sbuf)

    # for c_out_tile in nl.affine_range(n_tiles_c_out):
    #     weight_sbuf[c_out_tile] = nl.load(W[c_out_tile])

    # for c_out_tile in nl.affine_range(n_tiles_c_out):
    #     for c_in_tile in nl.affine_range(n_tiles_c_in):
    #         for i in nl.affine_range(filter_height):
    #             for j in nl.affine_range(filter_width):
    #                 weight_copy[i, j, c_out_tile, c_in_tile, :, :] = nl.copy(weight_sbuf[c_out_tile, :, c_in_tile, :, i, j], dtype = W.dtype)
    #                 w[i, j, c_out_tile, c_in_tile] = nisa.nc_transpose(weight_copy[i, j, c_out_tile, c_in_tile])

    #  the weights now are prepared for matrix multiply

    # variables for larger images loading
    #n_chunks = (input_height + c_in_pmax - 1) // c_in_pmax    # 1 for small, 2 for large
    #out_rows = (out_height + n_chunks - 1) // n_chunks        # how many output rows will be covered in one chunk
    out_rows = 2
    n_chunks = (out_height + out_rows - 1) // out_rows
    in_rows = out_rows + filter_height - 1                    # how many input rows will be covered in one chunk

    bias_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), 1), dtype=W.dtype, buffer=nl.sbuf)
    for c_out_tile in nl.affine_range(n_tiles_c_out):
        bias_sbuf[c_out_tile] = nl.load(bias[c_out_tile * c_out_pmax:(c_out_tile + 1) * c_out_pmax])

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # assign space in SBUF to store entire image, call it x
        x = nl.ndarray(
            shape=(n_tiles_c_in, nl.par_dim(c_in_pmax), in_rows, input_width),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )

        for chunk in nl.sequential_range(n_chunks):
            # loop over n_tiles_c_in:
            for inpt_c in nl.sequential_range(n_tiles_c_in):
                # load corresponding part of input image
                i_c, i_h, i_w  = nl.mgrid[0:c_in_pmax, 0:in_rows, 0:input_width]
                x[inpt_c, :, :, :] = nl.load(
                    X[b, i_c + (inpt_c * c_in_pmax), (chunk * out_rows) + i_h, i_w], 
                    mask=((chunk * out_rows) + i_h) < input_height
                )

            # assign space in SBUF to store output
            out = nl.ndarray(
                shape=(nl.par_dim(c_out_pmax), out_rows, out_width),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )

            # loop over n_tiles_c_out:
            for outpt_c in nl.sequential_range(n_tiles_c_out):
                # assign space in PSUM to store output 
                psum_out = nl.ndarray(
                    shape=(out_rows, out_width),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )

                # loop over output_rows:
                for out_row in nl.affine_range(out_rows):
                    # Allocate PSUM for the entire row
                    psum_row = nl.zeros(
                        shape=(nl.par_dim(c_out_pmax), out_width),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )

                    for filter_row in nl.affine_range(filter_height):
                        for filter_col in nl.affine_range(filter_width):
                            for curr_c_in_tile in nl.affine_range(n_tiles_c_in):
                                # Load the relevant weight tile
                                weights_tile = moved_W[
                                    filter_row,
                                    filter_col,
                                    outpt_c,
                                    curr_c_in_tile,
                                    :, :,
                                ]

                                # Load the corresponding input slice
                                input_tile = nl.ndarray(
                                    shape=(nl.par_dim(c_in_pmax), out_width),
                                    dtype=X.dtype,
                                    buffer=nl.sbuf,
                                )
                                input_tile[...] = x[
                                    curr_c_in_tile,
                                    :,
                                    out_row + filter_row,
                                    filter_col:filter_col + out_width,
                                ]

                                # Perform the matmul for the tile
                                partial_sum = nl.matmul(weights_tile, input_tile, transpose_x=True)

                                # Accumulate the results in PSUM
                                psum_row += partial_sum

                    # Store the computed row back to SBUF
                    out[:, out_row, :] = psum_row

                out_bias = nisa.tensor_scalar(out, np.add, bias_sbuf[outpt_c])
                # Store the output tile from SBUF back to HBM
                nl.store(
                    X_out[b, outpt_c * c_out_pmax:(outpt_c + 1) * c_out_pmax, chunk * out_rows:(chunk + 1) * out_rows, :],
                    out_bias,
                )

    return X_out
