# Model Inference
After the model is trained and stored in S3 bucket, the next step is to use that model for inference.

Update S3 directories to point to the model from the previous excercise:

inference.yaml
```
apiVersion: v1
kind: Service
metadata:
  labels:
    app: mnist
    type: inference
    framework: tensorflow
  name: mnist-inference
  namespace: default
spec:
  ports:
  - name: grpc-tf-serving
    port: 9000
    targetPort: 9000
  - name: http-tf-serving
    port: 8500
    targetPort: 8500
  selector:
    app: mnist
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: mnist
    type: inference
    framework: tensorflow
  name: mnist-inference
  namespace: default
spec:
  selector:
    matchLabels:
      app: mnist
  template:
    metadata:
      labels:
        app: mnist
        version: v1
        type: inference
        framework: tensorflow
    spec:
      containers:
      - args:
        - --port=9000
        - --rest_api_port=8500
        - --model_name=mnist
        - --model_base_path=s3://<s3-bucket-with-pretrainde-model>/mnist/tf_saved_model
        command:
        - /usr/bin/tensorflow_model_server
        env:
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              key: AWS_ACCESS_KEY_ID
              name: aws-secret
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              key: AWS_SECRET_ACCESS_KEY
              name: aws-secret
        - name: AWS_REGION
          value: us-west-2
        - name: S3_USE_HTTPS
          value: "true"
        - name: S3_VERIFY_SSL
          value: "true"
        - name: S3_ENDPOINT
          value: s3.us-west-2.amazonaws.com
        image: tensorflow/serving:1.11.1
        imagePullPolicy: IfNotPresent
        livenessProbe:
          initialDelaySeconds: 30
          periodSeconds: 30
          tcpSocket:
            port: 9000
        name: mnist
        ports:
        - containerPort: 9000
        - containerPort: 8500
        resources:
          limits:
            cpu: "4"
            memory: 4Gi
          requests:
            cpu: "1"
            memory: 1Gi
```

Apply it to the cluster:
```
kubectl apply -f inference.yaml
```

Wait for the containers to start and run the next command to check its status
```
$ kubectl get pods -l app=mnist,type=inference

NAME                               READY   STATUS    RESTARTS   AGE
mnist-inference-5c9d8f89b9-g48gv   1/1     Running   0          6m27s
```

Now, we are going to use Kubernetes port forward for the inference endpoint to do local testing:
```
kubectl port-forward `kubectl get pods -l=app=mnist,type=inference -o jsonpath='{.items[0].metadata.name}' --field-selector=status.phase=Running` 8500:8500
```

# Install packages for local testing
Open new terminal and install tensorflow with pypi
```buildoutcfg
#curl -O https://bootstrap.pypa.io/get-pip.py
#python3 get-pip.py
pip3 install requests tensorflow
```


inference_client.py
```
# TensorFlow and tf.keras
# Workaround to suppress FutureWarnings messages ref: https://www.cicoria.com/tensorflow-suppressing-futurewarning-numpy-messages-in-jupyter-notebooks/
import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
#import tensorflow as tf
#from tensorflow import keras

# Helper libraries
import numpy as np
import os
import subprocess
import argparse

import random
import json
import requests


def main(argv=None):
  parser = argparse.ArgumentParser(description='Fashion MNIST Tensorflow Serving Client')
  parser.add_argument('--endpoint', type=str, default='http://localhost:8500/v1/models/mnist:predict', help='Model serving endpoint')
  args = parser.parse_args()

  # Prepare test dataset
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

  # Random generate one image
  rando = random.randint(0,len(test_images)-1)
  data = json.dumps({"signature_name": "serving_default", "instances": test_images[rando:rando+1].tolist()})
  print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

  # HTTP call
  headers = {"content-type": "application/json"}
  json_response = requests.post(args.endpoint, data=data, headers=headers)
  predictions = json.loads(json_response.text)['predictions']

  title = 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
    class_names[np.argmax(predictions[0])], test_labels[rando], class_names[np.argmax(predictions[0])], test_labels[rando])
  print(title)

if __name__ == "__main__":
  main()
```

Run inference client:
```
python inference_client.py --endpoint http://localhost:8500/v1/models/mnist:predict
```

It will randomly pick one image from test data set and make prediction
```
Data: {"signature_name": "serving_default", "instances": ... 62744], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]]}
The model thought this was a T-shirt/top (class 0), and it was actually a T-shirt/top (class 0)
```

# Cleanup
```
kubectl delete -f mnist-training.yaml
kubectl delete -f inference.yaml
```
