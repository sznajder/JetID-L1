#ifndef NNET_CONV1D_LATENCY_H_
#define NNET_CONV1D_LATENCY_H_

#include "nnet_common.h"
#include <cstdlib>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_latency_sparse_type1(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt],
    // weights and biases are not used 
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    assert(CONFIG_T::filt_width == 1);

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    //#pragma HLS function_instantiate variable=weights,biases

    // Parallel mode
    //#pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    #pragma HLS PIPELINE II=1
    //#pragma HLS ARRAY_PARTITION variable=biases complete dim=0

    // Limit multipliers to control parallelization
    //const int multiplier_limit = compute_multiplier_limit<CONFIG_T>(weights);
    //#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    assert(CONFIG_T::n_chan * (CONFIG_T::n_chan-1) == CONFIG_T::n_filt);
    assert(CONFIG_T::in_width == CONFIG_T::out_width);
    // check this paper ( arxiv.org/abs/2209.14065 ) for details
    ConvOut: for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        ConvChan: for(int jj = 0; jj < CONFIG_T::n_chan; jj++) {
            ConvChanM1: for(int kk = 0; kk < CONFIG_T::n_chan-1; kk++) {
                int index_res = ii*CONFIG_T::n_chan*(CONFIG_T::n_chan-1) + jj*(CONFIG_T::n_chan-1) + kk;
                int index_data = ii*CONFIG_T::n_chan + jj;
                res[index_res] = data[index_data];

            }
        }
    }

}


template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_latency_sparse_type2(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt],
    // weights and biases are not used 
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    assert(CONFIG_T::filt_width == 1);

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    //#pragma HLS function_instantiate variable=weights,biases

    // Parallel mode
    //#pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    #pragma HLS PIPELINE II=1
    //#pragma HLS ARRAY_PARTITION variable=biases complete dim=0

    // Limit multipliers to control parallelization
    //const int multiplier_limit = compute_multiplier_limit<CONFIG_T>(weights);
    //#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    assert(CONFIG_T::n_chan * (CONFIG_T::n_chan-1) == CONFIG_T::n_filt);
    assert(CONFIG_T::in_width == CONFIG_T::out_width);

    // check this paper ( arxiv.org/abs/2209.14065 ) for details
    ConvOut: for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        ConvChan: for(int jj = 0; jj < CONFIG_T::n_chan; jj++) {
            ConvChanM1: for(int kk = 0; kk < CONFIG_T::n_chan-1; kk++) {
                int index_res = ii*CONFIG_T::n_chan*(CONFIG_T::n_chan-1) + jj*(CONFIG_T::n_chan-1) + kk;
                int index_data = ii*CONFIG_T::n_chan + ((kk<jj)?kk:(kk+1));
                res[index_res] = data[index_data];

            }
        }
    }

}

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_latency_sparse_type3(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt],
    // weights and biases are not used 
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    assert(CONFIG_T::filt_width == 1);

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    //#pragma HLS function_instantiate variable=weights,biases

    // Parallel mode
    //#pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    #pragma HLS PIPELINE II=1
    //#pragma HLS ARRAY_PARTITION variable=biases complete dim=0

    // Limit multipliers to control parallelization
    //const int multiplier_limit = compute_multiplier_limit<CONFIG_T>(weights);
    //#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    assert(CONFIG_T::n_chan == CONFIG_T::n_filt*(CONFIG_T::n_filt-1));
    assert(CONFIG_T::in_width == CONFIG_T::out_width);

    // check this paper ( arxiv.org/abs/2209.14065 ) for details
    ConvOut: for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        ConvChan: for(int jj = 0; jj < CONFIG_T::n_filt; jj++) {

            typename CONFIG_T::accum_t acc = (typename CONFIG_T::accum_t) 0; 
            ConvChanM1: for(int kk = 0; kk < CONFIG_T::n_filt-1; kk++) {
                int index_data = ii*CONFIG_T::n_filt*(CONFIG_T::n_filt-1) + jj*(CONFIG_T::n_filt-1) + kk;
                acc = acc + data[index_data];
            }

            int index_res = ii*CONFIG_T::n_filt + jj;
            res[index_res] = (res_T) acc; 
        }
    }


}
}

