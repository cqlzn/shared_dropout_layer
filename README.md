# shared_dropout_layer
shared_dropout_layer for Caffe
shared_dropout_layer is used in Siamese Net. If 2 dropout layers are used in different branches, they will drop different neurons.That is not expected in a Siaseme Net.

# USAGE:
(1) Copy `shared_dropout_layer.cpp` and `shared_dropout_layer.cu` under `<caffe_root>/src/caffe/layers`.  

(2) Copy `shared_dropout_layer.hpp` under `<caffe_root>/include/caffe/layers`.  

(3) Rebuild your Caffe.  

(4) Now, use your shared_dropout_layer according to the folloing way:
```
layer {
  name: "shareddrop"
  type: "SharedDropout"
  bottom: "ip5"
  bottom: "ip5_p"
  top: "dp5"
  top: "dp5_p"
  dropout_param{
  dropout_ratio : 0.5
  }
}
```
 
