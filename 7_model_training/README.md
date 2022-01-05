# Model Training
While Jupyter notebook is good for interactive model training, you may want to package the training code as Docker image and run it in Amazon EKS cluster.
We will explain here how to train ML model for Fashion-MNIST dataset using TensorFlow and Keras on EKS. 
Our dataset contains 70,000 grayscale images in 10 categories and is meant to be a drop-in replace of MNIST.

# Docker Image
We will use a pre-built Docker image seedjeffwan/mnist_tensorflow_keras:1.13.1 for this exercise. This image uses tensorflow/tensorflow:1.13.1 as the base image. The image has training code and downloads training and test data sets. It also stores the generated model in an S3 bucket.
Alternatively, you can use Dockerfile to build the image and push it to the ECR repository.

Dockerfile
```
FROM tensorflow/tensorflow:1.13.1

COPY mnist-tensorflow-docker.py /mnist-tensorflow-docker.py

CMD [ "python", "mnist-tensorflow-docker.py" ]
```

Create S3 bucket
```buildoutcfg
aws s3 mb s3://$S3_BUCKET --region $AWS_REGION
```

This name will be used in the pod specification later. This bucket is also used for serving the model.

If you want to use an existing bucket in a different region, then make sure to specify the exact region as the value of AWS_REGION environment variable in mnist-training.yaml.

# AWS credentials for EKS
```buildoutcfg
aws iam create-user --user-name s3user
aws iam attach-user-policy --user-name s3user --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam create-access-key --user-name s3user > /tmp/create_output.json
```
Then
```buildoutcfg
export AWS_ACCESS_KEY_ID_VALUE=$(jq -j .AccessKey.AccessKeyId /tmp/create_output.json | base64)
export AWS_SECRET_ACCESS_KEY_VALUE=$(jq -j .AccessKey.SecretAccessKey /tmp/create_output.json | base64)
```
And apply it to he Kubernetes clustr
```buildoutcfg
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: aws-secret
type: Opaque
data:
  AWS_ACCESS_KEY_ID: $AWS_ACCESS_KEY_ID_VALUE
  AWS_SECRET_ACCESS_KEY: $AWS_SECRET_ACCESS_KEY_VALUE
EOF
```

# POD
model.yaml
```
apiVersion: v1
kind: Pod
metadata:
  name: mnist-training
  labels:
    app: mnist
    type: training
    framework: tensorflow
spec:
  restartPolicy: OnFailure
  containers:
  - name: mnist-training
    image: seedjeffwan/mnist_tensorflow_keras:1.13.1
    command:
      - "python"
      - "mnist.py"
      - "--model_export_path"
      - "s3://<your_s3_bucket>/mnist/tf_saved_model"
      - "--model_summary_path"
      - "s3://<your_s3_bucket>/mnist/tf_summary"
      - "--epochs"
      - "40"
    env:
      - name: AWS_REGION
        value: "us-west-2"
      - name: S3_REGION
        value: "us-west-2"
      - name: S3_ENDPOINT
        value: "s3.us-west-2.amazonaws.com"
      - name: S3_USE_HTTPS
        value: "1"
      - name: S3_VERIFY_SSL
        value: "0"
      - name: AWS_ACCESS_KEY_ID
        valueFrom:
          secretKeyRef:
            name: aws-secret
            key: AWS_ACCESS_KEY_ID
      - name: AWS_SECRET_ACCESS_KEY
        valueFrom:
          secretKeyRef:
            name: aws-secret
            key: AWS_SECRET_ACCESS_KEY

```

Apply pod to the cluster
```
kubectl apply -f model.yaml -n <your_namespace>
```

After a while you should see similar logs in the pod
```
2021-12-30 12:02:10.373060: I tensorflow/core/platform/s3/aws_logging.cc:54] Connection has been released. Continuing.
2021-12-30 12:02:10.385022: I tensorflow/core/platform/s3/aws_logging.cc:54] Found secret key
2021-12-30 12:02:10.385146: I tensorflow/core/platform/s3/aws_logging.cc:54] Connection has been released. Continuing.
2021-12-30 12:02:10.401522: I tensorflow/core/platform/s3/aws_logging.cc:54] Found secret key
2021-12-30 12:02:10.401702: I tensorflow/core/platform/s3/aws_logging.cc:54] Connection has been released. Continuing.
2021-12-30 12:02:10.506891: I tensorflow/core/platform/s3/aws_logging.cc:54] Found secret key
2021-12-30 12:02:10.507546: I tensorflow/core/platform/s3/aws_logging.cc:54] Connection has been released. Continuing.
2021-12-30 12:02:10.528373: I tensorflow/core/platform/s3/aws_logging.cc:54] Found secret key
2021-12-30 12:02:10.528477: I tensorflow/core/platform/s3/aws_logging.cc:54] Connection has been released. Continuing.
2021-12-30 12:02:10.608650: I tensorflow/core/platform/s3/aws_logging.cc:54] Deleting file: /tmp/s3_filesystem_XXXXXX20211230T1202101640865730528
2021-12-30 12:02:10.609168: I tensorflow/core/platform/s3/aws_logging.cc:54] Found secret key
2021-12-30 12:02:10.609430: I tensorflow/core/platform/s3/aws_logging.cc:54] Connection has been released. Continuing.
2021-12-30 12:02:10.632788: I tensorflow/core/platform/s3/aws_logging.cc:54] Found secret key
2021-12-30 12:02:10.632914: I tensorflow/core/platform/s3/aws_logging.cc:54] Connection has been released. Continuing.
2021-12-30 12:02:10.647010: I tensorflow/core/platform/s3/aws_logging.cc:54] Found secret key
2021-12-30 12:02:10.647149: I tensorflow/core/platform/s3/aws_logging.cc:54] Connection has been released. Continuing.
2021-12-30 12:02:10.714881: I tensorflow/core/platform/s3/aws_logging.cc:54] Found secret key
2021-12-30 12:02:10.715070: I tensorflow/core/platform/s3/aws_logging.cc:54] Connection has been released. Continuing.
2021-12-30 12:02:10.725820: E tensorflow/core/platform/s3/aws_logging.cc:60] No response body. Response code: 404
2021-12-30 12:02:10.725880: W tensorflow/core/platform/s3/aws_logging.cc:57] If the signature check failed. This could be because of a time skew. Attempting to adjust the signer.
2021-12-30 12:02:10.726171: I tensorflow/core/platform/s3/aws_logging.cc:54] Found secret key
2021-12-30 12:02:10.726312: I tensorflow/core/platform/s3/aws_logging.cc:54] Connection has been released. Continuing.
2021-12-30 12:02:10.747072: I tensorflow/core/platform/s3/aws_logging.cc:54] Found secret key
2021-12-30 12:02:10.747257: I tensorflow/core/platform/s3/aws_logging.cc:54] Connection has been released. Continuing.
2021-12-30 12:02:10.777961: I tensorflow/core/platform/s3/aws_logging.cc:54] Deleting file: /tmp/s3_filesystem_XXXXXX20211230T1202101640865730746

Test accuracy: 0.888899981976

Saved model: s3://noble-123-123/mnist/tf_saved_model/1
```

Your model should be save in the S3 bucket.
