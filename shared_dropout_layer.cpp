// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/shared_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SharedDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
  scale_train_ = this->layer_param_.dropout_param().scale_train();
}

template <typename Dtype>
void SharedDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for(int i = 0; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape()) <<
    "The same shape of bottom blobs is required in SharedDropout Layer.";    	
    top[i]->ReshapeLike(*bottom[i]);
  }
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void SharedDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    for(int i = 0; i < top.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data();
      if (scale_train_) {
        for (int n = 0; n < count; ++n) {
          top_data[n] = bottom_data[n] * mask[n] * scale_;
        }
      } else {
        for (int n = 0; n < count; ++n) {
          top_data[n] = bottom_data[n] * mask[n];
        }
      }
    }
  } else {
    for(int i = 0; i < top.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data(); 
      caffe_copy(bottom[0]->count(), bottom_data, top_data);
      if (!scale_train_) {
        caffe_scal<Dtype>(count, 1. / scale_, top_data);
      }
    }
  }
}

template <typename Dtype>
void SharedDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int count = bottom[0]->count();
      for(int i = 0; i < top.size(); ++i) {
        const Dtype* top_diff = top[i]->cpu_diff();
        Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
        if (scale_train_) {
          for (int n = 0; n < count; ++n) {
            bottom_diff[n] = top_diff[n] * mask[n] * scale_;
          }
        } else {
          for (int n = 0; n < count; ++n) {
            bottom_diff[n] = top_diff[n] * mask[n];
          }
        }
      }
    } else {
      for(int i = 0; i < top.size(); ++i) {
        const Dtype* top_diff = top[i]->cpu_diff();
        Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
        caffe_copy(top[0]->count(), top_diff, bottom_diff);
        if (!scale_train_) {
          caffe_scal<Dtype>(top[0]->count(), 1. / scale_, bottom_diff);
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SharedDropoutLayer);
#endif

INSTANTIATE_CLASS(SharedDropoutLayer);
REGISTER_LAYER_CLASS(SharedDropout);

}  // namespace caffe
