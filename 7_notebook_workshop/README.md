# Notebook introduction

What is Kubeflow Notebooks?
Kubeflow Notebooks provides a way to run web-based development environments inside your Kubernetes cluster by running them inside Pods.

Some key features include:

* Native support for JupyterLab, RStudio, and Visual Studio Code (code-server).
* Users can create notebook containers directly in the cluster, rather than locally on their workstations.
* Admins can provide standard notebook images for their organization with required packages pre-installed.
* Access control is managed by Kubeflow’s RBAC, enabling easier notebook sharing across the organization.


# Container images

Kubeflow Notebooks natively supports three types of notebooks, JupyterLab, RStudio, and Visual Studio Code (code-server), but any web-based IDE should work.
Notebook servers run as containers inside a Kubernetes Pod, which means the type of IDE (and which packages are installed) is determined by the Docker image you pick for your server.

# Images
Kubeflow community provides a number of example container images to get you started. See https://github.com/kubeflow/kubeflow/tree/master/components/example-notebook-servers

# Notebook Pod ServiceAccount
Kubeflow assigns the default-editor Kubernetes ServiceAccount to the Notebook Pods. 
The Kubernetes default-editor ServiceAccount is bound to the kubeflow-edit ClusterRole, which has namespace-scoped permissions to many Kubernetes resources.

You can get the full list of RBAC for ClusterRole/kubeflow-edit using:

>kubectl describe clusterrole kubeflow-edit

# Kubectl in Notebook Pod 
Because every Notebook Pod has the highly-privileged default-editor Kubernetes ServiceAccount bound to it, you can run kubectl inside it without providing additional authentication.

For example, the following command will create the resources defined in test.yaml:

>kubectl create -f "test.yaml" --namespace "MY_PROFILE_NAMESPACE"

# Iam role for service account
You can also tag default-editor service account with IRSA tag and assing IAM role directly to your Notebook.
See: https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html

# Simple demo

Let's start our first notebook.
Use custom image:
```
gcr.io/kubeflow-images-public/tensorflow-1.14.0-notebook-cpu:v1.0.0
```

Create notebook with 1.0 cpu and 1gib memory

Connect to the notebook and click on New -> Python3
Copy sample code to the notebook cell. This Python sample code uses TensorFlow to create a training model for MNIST database
```
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
# Helper libraries
import numpy as np
import os
import subprocess
import argparse

# Reduce spam logs from s3 client
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def preprocessing():
  fashion_mnist = keras.datasets.fashion_mnist
  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

  # scale the values to 0.0 to 1.0
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  # reshape for feeding into the model
  train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
  test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

  print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
  print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

  return train_images, train_labels, test_images, test_labels

def train(train_images, train_labels, epochs, model_summary_path):
  if model_summary_path:
    logdir=model_summary_path # + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

  model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3,
                        strides=2, activation='relu', name='Conv1'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
  ])
  model.summary()

  model.compile(optimizer=tf.train.AdamOptimizer(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
                )
  if model_summary_path:
    model.fit(train_images, train_labels, epochs=epochs, callbacks=[tensorboard_callback])
  else:
    model.fit(train_images, train_labels, epochs=epochs)

  return model

def eval(model, test_images, test_labels):
  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('\nTest accuracy: {}'.format(test_acc))

def export_model(model, model_export_path):
  version = 1
  export_path = os.path.join(model_export_path, str(version))

  tf.saved_model.simple_save(
    keras.backend.get_session(),
    export_path,
    inputs={'input_image': model.input},
    outputs={t.name:t for t in model.outputs})

  print('\nSaved model: {}'.format(export_path))


def main(argv=None):
  parser = argparse.ArgumentParser(description='Fashion MNIST Tensorflow Example')
  parser.add_argument('--model_export_path', type=str, help='Model export path')
  parser.add_argument('--model_summary_path', type=str,  help='Model summry files for Tensorboard visualization')
  parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
  args = parser.parse_args(args=[])

  train_images, train_labels, test_images, test_labels = preprocessing()
  model = train(train_images, train_labels, args.epochs, args.model_summary_path)
  eval(model, test_images, test_labels)

  if args.model_export_path:
    export_model(model, args.model_export_path)
```

Add code to the next cell and run
```
main()
```

You should see in a while:
```

train_images.shape: (60000, 28, 28, 1), of float64
test_images.shape: (10000, 28, 28, 1), of float64
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Conv1 (Conv2D)               (None, 13, 13, 8)         80        
_________________________________________________________________
flatten (Flatten)            (None, 1352)              0         
_________________________________________________________________
Softmax (Dense)              (None, 10)                13530     
=================================================================
Total params: 13,610
Trainable params: 13,610
Non-trainable params: 0
_________________________________________________________________
Epoch 1/5
60000/60000 [==============================] - 5s 89us/sample - loss: 0.5310 - acc: 0.8149
Epoch 2/5
60000/60000 [==============================] - 5s 87us/sample - loss: 0.4028 - acc: 0.8604
Epoch 3/5
60000/60000 [==============================] - 5s 86us/sample - loss: 0.3730 - acc: 0.8697
Epoch 4/5
60000/60000 [==============================] - 5s 87us/sample - loss: 0.3571 - acc: 0.8757
Epoch 5/5
60000/60000 [==============================] - 5s 87us/sample - loss: 0.3464 - acc: 0.8777
10000/10000 [==============================] - 0s 42us/sample - loss: 0.3842 - acc: 0.8613

Test accuracy: 0.861299991607666
```

The first few lines shows that TensorFlow and Keras dataset is downloaded. Training data set is 60k images and test data set is 10k images. Hyperparameters used for the training, outputs from five epochs, and finally the model accuracy is shown.

# Troubleshooting
Problems and solutions for common problems with Kubeflow Notebooks

ISSUE: notebook not starting

SOLUTION: check events of Notebook

Run the following command then check the events section to make sure that there are no errors:

>kubectl describe notebooks "${MY_NOTEBOOK_NAME}" --namespace "${MY_PROFILE_NAMESPACE}"

SOLUTION: check events of Pod

Run the following command then check the events section to make sure that there are no errors:

>kubectl describe pod "${MY_NOTEBOOK_NAME}-0" --namespace "${MY_PROFILE_NAMESPACE}"

SOLUTION: check YAML of Pod

Run the following command and check the Pod YAML looks as expected:

>kubectl get pod "${MY_NOTEBOOK_NAME}-0" --namespace "${MY_PROFILE_NAMESPACE}" -o yaml

SOLUTION: check logs of Pod

Run the following command to get the logs from the Pod:

>kubectl logs "${MY_NOTEBOOK_NAME}-0" --namespace "${MY_PROFILE_NAMESPACE}"

ISSUE: manually delete notebook

SOLUTION: use kubectl to delete Notebook resource

Run the following command to delete a Notebook resource manually:

>kubectl delete notebook "${MY_NOTEBOOK_NAME}" --namespace "${MY_PROFILE_NAMESPACE}"
