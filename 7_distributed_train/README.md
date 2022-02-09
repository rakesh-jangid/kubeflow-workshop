# Distributed Training using tf-operator and pytorch-operator

TFJob is a Kubernetes custom resource that you can use to run TensorFlow training jobs on Kubernetes. 
The Kubeflow implementation of TFJob is in tf-operator. 

You can also use PyTorch Job by defining a PyTorchJob config file and pytorch-operator will create PyTorch job, monitor and track it.

Kubeflow also support MXNet, XGBoost and MPI.

More: https://www.kubeflow.org/docs/components/training/

Pytorch: https://pytorch.org/tutorials/beginner/dist_overview.html

Tensorflow: https://www.tensorflow.org/guide/distributed_training
