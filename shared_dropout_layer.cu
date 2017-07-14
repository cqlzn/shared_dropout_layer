#include <vector>

#include "caffe/layers/shared_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SharedDropoutForward(const int n, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}

template <typename Dtype>
void SharedDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_uniform(count, mask);
    // set thresholds
    for(int i = 0; i < top.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* top_data = top[i]->mutable_gpu_data();
      if (scale_train_) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        SharedDropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, mask, uint_thres_, scale_, top_data);
      } else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        SharedDropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count),
          CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, mask, uint_thres_, 1.f, top_data);
      }
      CUDA_POST_KERNEL_CHECK;
    }
  } else {
    for(int i = 0; i < top.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* top_data = top[i]->mutable_gpu_data();
      caffe_copy(count, bottom_data, top_data);
      if (!scale_train_) {
        caffe_gpu_scal<Dtype>(count, 1. / scale_, top_data);
      }
    }
  }
}

template <typename Dtype>
__global__ void SharedDropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}

template <typename Dtype>
void SharedDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    if (this->phase_ == TRAIN) {
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      for(int i = 0; i < top.size(); ++i) {
        const Dtype* top_diff = top[i]->gpu_diff();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        if (scale_train_) {
          // NOLINT_NEXT_LINE(whitespace/operators)
          SharedDropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(
              count, top_diff, mask, uint_thres_, scale_, bottom_diff);
        } else {
          // NOLINT_NEXT_LINE(whitespace/operators)
          SharedDropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(
             count, top_diff, mask, uint_thres_, 1.f, bottom_diff);
        }
        CUDA_POST_KERNEL_CHECK;
      }
    } else {
      for(int i = 0; i < top.size(); ++i) {
        const Dtype* top_diff = top[i]->gpu_diff();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        caffe_copy(top[0]->count(), top_diff, bottom_diff);
        if (!scale_train_) {
          caffe_gpu_scal<Dtype>(top[0]->count(), 1. / scale_, bottom_diff);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SharedDropoutLayer);

}  // namespace caffe
