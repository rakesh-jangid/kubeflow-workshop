Attach polices to EKS Node group or IAM role for service account(if enabled).
We will need S3 and ECR access
For example for role name: eksctl-noble-nodegroup-ng-1-NodeInstanceRole-1E922WF5PB8FI 
```
aws iam attach-role-policy --role-name eksctl-noble-nodegroup-ng-1-NodeInstanceRole-1E922WF5PB8FI --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam attach-role-policy --role-name eksctl-noble-nodegroup-ng-1-NodeInstanceRole-1E922WF5PB8FI --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess
```
Create Norebook with the following image with 1CPU and 1Gi memory.
ECR image for notebook:
```
527798164940.dkr.ecr.us-west-2.amazonaws.com/tensorflow-1.15.2-notebook-cpu:1.0.0
```

fairing_into.ipynb

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Kubeflow Fairing Introduction\n",
    "\n",
    "Kubeflow Fairing is a Python package that streamlines the process of `building`, `training`, and `deploying` machine learning (ML) models in a hybrid cloud environment. By using Kubeflow Fairing and adding a few lines of code, you can run your ML training job locally or in the cloud, directly from Python code or a Jupyter notebook. After your training job is complete, you can use Kubeflow Fairing to deploy your trained model as a prediction endpoint.\n",
    "\n",
    "\n",
    "# How does Kubeflow Fairing work\n",
    "\n",
    "Kubeflow Fairing \n",
    "1. Packages your Jupyter notebook, Python function, or Python file as a Docker image\n",
    "2. Deploys and runs the training job on Kubeflow or AI Platform. \n",
    "3. Deploy your trained model as a prediction endpoint on Kubeflow after your training job is complete.\n",
    "\n",
    "\n",
    "# Goals of Kubeflow Fairing project\n",
    "\n",
    "- Easily package ML training jobs: Enable ML practitioners to easily package their ML model training code, and their code’s dependencies, as a Docker image.\n",
    "- Easily train ML models in a hybrid cloud environment: Provide a high-level API for training ML models to make it easy to run training jobs in the cloud, without needing to understand the underlying infrastructure.\n",
    "- Streamline the process of deploying a trained model: Make it easy for ML practitioners to deploy trained ML models to a hybrid cloud environment.\n",
    "\n",
    "\n",
    "> Note: Before fairing workshop, please read `README.md` under `02_01_fairing_introduction`\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install latest Fairing from github repository\n",
    "!pip install kubeflow-fairing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# check fairing is installed \n",
    "!pip show kubeflow-fairing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Basic Example\n",
    "\n",
    "If you see any issues, please restart notebook. It's probably because of new installed packages.\n",
    "\n",
    "Click `Kernel` -> `Restart & Clear Output`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "from kubeflow import fairing\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "\n",
    "def train():\n",
    "    # Genrating random linear data \n",
    "    # There will be 50 data points ranging from 0 to 50 \n",
    "    x = np.linspace(0, 50, 50) \n",
    "    y = np.linspace(0, 50, 50) \n",
    "\n",
    "    # Adding noise to the random linear data \n",
    "    x += np.random.uniform(-4, 4, 50) \n",
    "    y += np.random.uniform(-4, 4, 50) \n",
    "\n",
    "    n = len(x) # Number of data points \n",
    "\n",
    "    X = tf.placeholder(\"float\") \n",
    "    Y = tf.placeholder(\"float\")\n",
    "    W = tf.Variable(np.random.randn(), name = \"W\") \n",
    "    b = tf.Variable(np.random.randn(), name = \"b\") \n",
    "    learning_rate = 0.01\n",
    "    training_epochs = 1000\n",
    "    \n",
    "    # Hypothesis \n",
    "    y_pred = tf.add(tf.multiply(X, W), b) \n",
    "\n",
    "    # Mean Squared Error Cost Function \n",
    "    cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n)\n",
    "\n",
    "    # Gradient Descent Optimizer \n",
    "    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) \n",
    "\n",
    "    # Global Variables Initializer \n",
    "    init = tf.global_variables_initializer() \n",
    "\n",
    "\n",
    "    sess = tf.Session()\n",
    "    sess.run(init) \n",
    "      \n",
    "    # Iterating through all the epochs \n",
    "    for epoch in range(training_epochs): \n",
    "          \n",
    "        # Feeding each data point into the optimizer using Feed Dictionary \n",
    "        for (_x, _y) in zip(x, y): \n",
    "            sess.run(optimizer, feed_dict = {X : _x, Y : _y}) \n",
    "          \n",
    "        # Displaying the result after every 50 epochs \n",
    "        if (epoch + 1) % 50 == 0: \n",
    "            # Calculating the cost a every epoch \n",
    "            c = sess.run(cost, feed_dict = {X : x, Y : y}) \n",
    "            print(\"Epoch\", (epoch + 1), \": cost =\", c, \"W =\", sess.run(W), \"b =\", sess.run(b)) \n",
    "      \n",
    "    # Storing necessary values to be used outside the Session \n",
    "    training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) \n",
    "    weight = sess.run(W) \n",
    "    bias = sess.run(b) \n",
    "\n",
    "    print('Weight: ', weight, 'Bias: ', bias)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Local training for development\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Remote training\n",
    "\n",
    "We will show you how to remotely run training job in kubernetes cluster. You can use `ECR` as your container image registry."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Authenticate ECR\n",
    "# This command retrieves a token that is valid for a specified registry for 12 hours, \n",
    "# and then it prints a docker login command with that authorization token. \n",
    "# Then we executate this command to login ECR\n",
    "\n",
    "REGION='us-west-2'\n",
    "!eval $(aws ecr get-login --no-include-email --region=$REGION)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create an ECR repository in the same region\n",
    "# If you receive \"RepositoryAlreadyExistsException\" error, it means the repository already\n",
    "# exists. You can move to the next step\n",
    "!aws ecr create-repository --repository-name fairing-job --region=$REGION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setting up AWS Elastic Container Registry (ECR) for storing output containers\n",
    "# You can use any docker container registry istead of ECR\n",
    "AWS_ACCOUNT_ID=fairing.cloud.aws.guess_account_id()\n",
    "AWS_REGION='us-west-2'\n",
    "DOCKER_REGISTRY = '{}.dkr.ecr.{}.amazonaws.com'.format(AWS_ACCOUNT_ID, AWS_REGION)\n",
    "\n",
    "fairing.config.set_builder('append', base_image='tensorflow/tensorflow:1.14.0-py3', registry=DOCKER_REGISTRY, push=True)\n",
    "fairing.config.set_deployer('job')\n",
    "    \n",
    "if __name__ == '__main__':\n",
    "    remote_train = fairing.config.fn(train)\n",
    "    remote_train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install tensorflow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install tensorflow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!sudo pip install --upgrade pip "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}


fairing_job_backend.ipynb

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Packages your Python function, Python file or Jupyter notebook as a Docker image"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Environment Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import time\n",
    "from kubeflow import fairing\n",
    "from kubeflow.fairing import TrainJob\n",
    "from kubeflow.fairing.backends import KubeflowAWSBackend"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "AWS_ACCOUNT_ID=fairing.cloud.aws.guess_account_id()\n",
    "AWS_REGION='us-west-2'\n",
    "DOCKER_REGISTRY = '{}.dkr.ecr.{}.amazonaws.com'.format(AWS_ACCOUNT_ID, AWS_REGION)\n",
    "\n",
    "PY_VERSION = \".\".join([str(x) for x in sys.version_info[0:3]])\n",
    "BASE_IMAGE = 'python:{}'.format(PY_VERSION)\n",
    "\n",
    "# TODO: There's a bug in the code. python:3.6.8 won't work for this. Has to use org/repo:tag format\n",
    "BASE_IMAGE = 'tensorflow/tensorflow:1.14.0-py3'\n",
    "\n",
    "# Setting up AWS Elastic Container Registry (ECR) for storing output containers\n",
    "# You can use any docker container registry istead of ECR\n",
    "AWS_ACCOUNT_ID=fairing.cloud.aws.guess_account_id()\n",
    "AWS_REGION='us-west-2'\n",
    "DOCKER_REGISTRY = '{}.dkr.ecr.{}.amazonaws.com'.format(AWS_ACCOUNT_ID, AWS_REGION)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Convert Python function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[W 211230 14:07:48 tasks:62] Using builder: <class 'kubeflow.fairing.builders.append.append.AppendBuilder'>\n",
      "[I 211230 14:07:48 tasks:66] Building the docker image.\n",
      "[W 211230 14:07:48 append:50] Building image using Append builder...\n",
      "[I 211230 14:07:48 base:107] Creating docker context: /tmp/fairing_context_cz5kl1qt\n",
      "[W 211230 14:07:48 base:94] /usr/local/lib/python3.6/dist-packages/kubeflow/fairing/__init__.py already exists in Fairing context, skipping...\n",
      "[I 211230 14:07:48 docker_creds_:234] Loading Docker credentials for repository 'tensorflow/tensorflow:1.14.0-py3'\n",
      "[W 211230 14:07:49 append:54] Image successfully built in 0.9025515989997075s.\n",
      "[W 211230 14:07:49 append:94] Pushing image 444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:D7A0F403...\n",
      "[I 211230 14:07:49 docker_creds_:234] Loading Docker credentials for repository '444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:D7A0F403'\n",
      "[W 211230 14:07:49 append:81] Uploading 444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:D7A0F403\n",
      "[I 211230 14:07:49 docker_session_:280] Layer sha256:b054a26005b7f3b032577f811421fab5ec3b42ce45a4012dfa00cf6ed6191b0f exists, skipping\n",
      "[I 211230 14:07:49 docker_session_:280] Layer sha256:016724bbd2c9643f24eff7c1e86d9202d7c04caddd7fdd4375a77e3998ce8203 exists, skipping\n",
      "[I 211230 14:07:49 docker_session_:280] Layer sha256:5b7339215d1d5f8e68622d584a224f60339f5bef41dbd74330d081e912f0cddd exists, skipping\n",
      "[I 211230 14:07:49 docker_session_:280] Layer sha256:8832e37735788665026956430021c6d1919980288c66c4526502965aeb5ac006 exists, skipping\n",
      "[I 211230 14:07:49 docker_session_:280] Layer sha256:5bd1cb59702536c10e96bb14e54846922c9b257580d4e2c733076a922525240b exists, skipping\n",
      "[I 211230 14:07:49 docker_session_:280] Layer sha256:14ca88e9f6723ce82bc14b241cda8634f6d19677184691d086662641ab96fe68 exists, skipping\n",
      "[I 211230 14:07:49 docker_session_:280] Layer sha256:2b940936f9933b7737cf407f2149dd7393998d7a0bee5acf1c4a57b0487cef79 exists, skipping\n",
      "[I 211230 14:07:49 docker_session_:280] Layer sha256:a31c3b1caad473a474d574283741f880e37c708cc06ee620d3e93fa602125ee0 exists, skipping\n",
      "[I 211230 14:07:49 docker_session_:280] Layer sha256:5e671b828b2af02924968841e5d12084fa78e8722e9510402aaee80dc5d7a6db exists, skipping\n",
      "[I 211230 14:07:49 docker_session_:280] Layer sha256:68543864d6442a851eaff0500161b92e4a151051cf7ed2649b3790a3f876bada exists, skipping\n",
      "[I 211230 14:07:49 docker_session_:284] Layer sha256:5c42d8cb949dd8a180c618bd39f14f8de5a6017193088a5c5bacc3f6a8b8bfa0 pushed.\n",
      "[I 211230 14:07:49 docker_session_:284] Layer sha256:e242c1b5bb003d2d6ff45700e23939b0f128a8db1aed72cb8fe482856d725f4a pushed.\n",
      "[I 211230 14:07:49 docker_session_:334] Finished upload of: 444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:D7A0F403\n",
      "[W 211230 14:07:49 append:99] Pushed image 444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:D7A0F403 in 0.5919458120006311s.\n",
      "[W 211230 14:07:49 aws:70] Not able to find aws credentials secret: aws-secret\n",
      "[W 211230 14:07:49 job:90] The job fairing-job-pm7fw launched.\n",
      "[W 211230 14:07:50 manager:255] Waiting for fairing-job-pm7fw-wbsvc to start...\n",
      "[W 211230 14:07:50 manager:255] Waiting for fairing-job-pm7fw-wbsvc to start...\n",
      "[W 211230 14:07:50 manager:255] Waiting for fairing-job-pm7fw-wbsvc to start...\n",
      "[I 211230 14:07:52 manager:261] Pod started running False\n",
      "[W 211230 14:07:52 job:162] Cleaning up job fairing-job-pm7fw...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "simple train job!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'fairing-job-pm7fw'"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def train():\n",
    "    print(\"simple train job!\")\n",
    "\n",
    "job = TrainJob(train, base_docker_image=BASE_IMAGE, docker_registry=DOCKER_REGISTRY, backend=KubeflowAWSBackend())\n",
    "job.submit() "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Convert Python file "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Overwriting train.py\n"
     ]
    }
   ],
   "source": [
    "%%writefile train.py\n",
    "print(\"hello world!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[W 211230 14:11:29 tasks:62] Using builder: <class 'kubeflow.fairing.builders.append.append.AppendBuilder'>\n",
      "[I 211230 14:11:29 tasks:66] Building the docker image.\n",
      "[W 211230 14:11:29 append:50] Building image using Append builder...\n",
      "[I 211230 14:11:29 base:107] Creating docker context: /tmp/fairing_context_2ml3t_92\n",
      "[I 211230 14:11:29 docker_creds_:234] Loading Docker credentials for repository 'tensorflow/tensorflow:1.14.0-py3'\n",
      "[W 211230 14:11:30 append:54] Image successfully built in 0.8774582259993622s.\n",
      "[W 211230 14:11:30 append:94] Pushing image 444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:7B94EC06...\n",
      "[I 211230 14:11:30 docker_creds_:234] Loading Docker credentials for repository '444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:7B94EC06'\n",
      "[W 211230 14:11:30 append:81] Uploading 444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:7B94EC06\n",
      "[I 211230 14:11:30 docker_session_:280] Layer sha256:b054a26005b7f3b032577f811421fab5ec3b42ce45a4012dfa00cf6ed6191b0f exists, skipping\n",
      "[I 211230 14:11:30 docker_session_:280] Layer sha256:016724bbd2c9643f24eff7c1e86d9202d7c04caddd7fdd4375a77e3998ce8203 exists, skipping\n",
      "[I 211230 14:11:30 docker_session_:280] Layer sha256:8832e37735788665026956430021c6d1919980288c66c4526502965aeb5ac006 exists, skipping\n",
      "[I 211230 14:11:30 docker_session_:280] Layer sha256:5bd1cb59702536c10e96bb14e54846922c9b257580d4e2c733076a922525240b exists, skipping\n",
      "[I 211230 14:11:30 docker_session_:280] Layer sha256:5b7339215d1d5f8e68622d584a224f60339f5bef41dbd74330d081e912f0cddd exists, skipping\n",
      "[I 211230 14:11:30 docker_session_:280] Layer sha256:5e671b828b2af02924968841e5d12084fa78e8722e9510402aaee80dc5d7a6db exists, skipping\n",
      "[I 211230 14:11:30 docker_session_:280] Layer sha256:14ca88e9f6723ce82bc14b241cda8634f6d19677184691d086662641ab96fe68 exists, skipping\n",
      "[I 211230 14:11:30 docker_session_:280] Layer sha256:a31c3b1caad473a474d574283741f880e37c708cc06ee620d3e93fa602125ee0 exists, skipping\n",
      "[I 211230 14:11:30 docker_session_:280] Layer sha256:68543864d6442a851eaff0500161b92e4a151051cf7ed2649b3790a3f876bada exists, skipping\n",
      "[I 211230 14:11:30 docker_session_:280] Layer sha256:2b940936f9933b7737cf407f2149dd7393998d7a0bee5acf1c4a57b0487cef79 exists, skipping\n",
      "[I 211230 14:11:31 docker_session_:284] Layer sha256:c846414c9a5d715889e61b7cdb2a3987e56fe44cf8977f2b9cc66c9bbec74f89 pushed.\n",
      "[I 211230 14:11:31 docker_session_:284] Layer sha256:119f46f6e5b09e5325337e1bd016d7a84cdc1449b4e7ffd6e84b58cb3d5028bc pushed.\n",
      "[I 211230 14:11:31 docker_session_:334] Finished upload of: 444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:7B94EC06\n",
      "[W 211230 14:11:31 append:99] Pushed image 444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:7B94EC06 in 0.653451522000978s.\n",
      "[W 211230 14:11:31 aws:70] Not able to find aws credentials secret: aws-secret\n",
      "[W 211230 14:11:31 job:90] The job fairing-job-sq7st launched.\n",
      "[W 211230 14:11:31 manager:255] Waiting for fairing-job-sq7st-9txfm to start...\n",
      "[W 211230 14:11:31 manager:255] Waiting for fairing-job-sq7st-9txfm to start...\n",
      "[W 211230 14:11:31 manager:255] Waiting for fairing-job-sq7st-9txfm to start...\n",
      "[I 211230 14:11:33 manager:261] Pod started running False\n",
      "[W 211230 14:11:33 job:162] Cleaning up job fairing-job-sq7st...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "hello world!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'fairing-job-sq7st'"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "job = TrainJob(\"train.py\", base_docker_image=BASE_IMAGE, docker_registry=DOCKER_REGISTRY, backend=KubeflowAWSBackend())\n",
    "job.submit() "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Convert Jupyter Notebook"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Overwriting requirements.txt\n"
     ]
    }
   ],
   "source": [
    "%%writefile requirements.txt\n",
    "papermill\n",
    "jupyter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[W 211230 14:12:08 tasks:62] Using builder: <class 'kubeflow.fairing.builders.cluster.cluster.ClusterBuilder'>\n",
      "[I 211230 14:12:08 tasks:66] Building the docker image.\n",
      "[I 211230 14:12:08 cluster:46] Building image using cluster builder.\n",
      "[I 211230 14:12:08 base:107] Creating docker context: /tmp/fairing_context_z7huf3au\n",
      "[W 211230 14:12:09 aws:70] Not able to find aws credentials secret: aws-secret\n",
      "[W 211230 14:12:09 manager:255] Waiting for fairing-builder-ph6hl-fklh2 to start...\n",
      "[W 211230 14:12:09 manager:255] Waiting for fairing-builder-ph6hl-fklh2 to start...\n",
      "[W 211230 14:12:09 manager:255] Waiting for fairing-builder-ph6hl-fklh2 to start...\n",
      "[I 211230 14:12:13 manager:261] Pod started running True\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[36mINFO\u001b[0m[0000] Resolved base name tensorflow/tensorflow:1.14.0-py3 to tensorflow/tensorflow:1.14.0-py3\n",
      "\u001b[36mINFO\u001b[0m[0000] Resolved base name tensorflow/tensorflow:1.14.0-py3 to tensorflow/tensorflow:1.14.0-py3\n",
      "\u001b[36mINFO\u001b[0m[0000] Downloading base image tensorflow/tensorflow:1.14.0-py3\n",
      "\u001b[36mINFO\u001b[0m[0001] Error while retrieving image from cache: getting file info: stat /cache/sha256:7c05dfab9ea82c95b571ca7dc3d92d4e0b289eeec6b3abf51f189caca0684488: no such file or directory\n",
      "\u001b[36mINFO\u001b[0m[0001] Downloading base image tensorflow/tensorflow:1.14.0-py3\n",
      "\u001b[36mINFO\u001b[0m[0002] Built cross stage deps: map[]\n",
      "\u001b[36mINFO\u001b[0m[0002] Downloading base image tensorflow/tensorflow:1.14.0-py3\n",
      "\u001b[36mINFO\u001b[0m[0002] Error while retrieving image from cache: getting file info: stat /cache/sha256:7c05dfab9ea82c95b571ca7dc3d92d4e0b289eeec6b3abf51f189caca0684488: no such file or directory\n",
      "\u001b[36mINFO\u001b[0m[0002] Downloading base image tensorflow/tensorflow:1.14.0-py3\n",
      "\u001b[36mINFO\u001b[0m[0003] Unpacking rootfs as cmd COPY /app//requirements.txt /app/ requires it.\n",
      "\u001b[36mINFO\u001b[0m[0024] Taking snapshot of full filesystem...\n",
      "\u001b[36mINFO\u001b[0m[0033] WORKDIR /app/\n",
      "\u001b[36mINFO\u001b[0m[0033] cmd: workdir\n",
      "\u001b[36mINFO\u001b[0m[0033] Changed working directory to /app/\n",
      "\u001b[36mINFO\u001b[0m[0033] Creating directory /app/\n",
      "\u001b[36mINFO\u001b[0m[0033] Taking snapshot of files...\n",
      "\u001b[36mINFO\u001b[0m[0033] ENV FAIRING_RUNTIME 1\n",
      "\u001b[36mINFO\u001b[0m[0033] Using files from context: [/kaniko/buildcontext/app/requirements.txt]\n",
      "\u001b[36mINFO\u001b[0m[0033] COPY /app//requirements.txt /app/\n",
      "\u001b[36mINFO\u001b[0m[0033] Taking snapshot of files...\n",
      "\u001b[36mINFO\u001b[0m[0033] RUN if [ -e requirements.txt ];then pip install --no-cache -r requirements.txt; fi\n",
      "\u001b[36mINFO\u001b[0m[0033] cmd: /bin/sh\n",
      "\u001b[36mINFO\u001b[0m[0033] args: [-c if [ -e requirements.txt ];then pip install --no-cache -r requirements.txt; fi]\n",
      "Collecting papermill (from -r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/ec/3b/55dbb2017142340a57937f1fad78c7d9373552540b42740e37fd6c40d761/papermill-2.3.3-py3-none-any.whl\n",
      "Collecting jupyter (from -r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/83/df/0f5dd132200728a86190397e1ea87cd76244e42d39ec5e88efd25b2abd7e/jupyter-1.0.0-py2.py3-none-any.whl\n",
      "Collecting nbformat>=5.1.2 (from papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/e7/c7/dd50978c637a7af8234909277c4e7ec1b71310c13fb3135f3c8f5b6e045f/nbformat-5.1.3-py3-none-any.whl (178kB)\n",
      "Collecting tqdm>=4.32.2 (from papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/63/f3/b7a1b8e40fd1bd049a34566eb353527bb9b8e9b98f8b6cf803bb64d8ce95/tqdm-4.62.3-py2.py3-none-any.whl (76kB)\n",
      "Collecting requests (from papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/92/96/144f70b972a9c0eabbd4391ef93ccd49d0f2747f4f6a2a2738e99e5adc65/requests-2.26.0-py2.py3-none-any.whl (62kB)\n",
      "Collecting nbclient>=0.2.0 (from papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/27/70/69c3561f43ea305da4b360820e67b57244c5308faf1fa890bc444e7cf842/nbclient-0.5.9-py3-none-any.whl (69kB)\n",
      "Collecting pyyaml (from papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/b3/85/79b9e5b4e8d3c0ac657f4e8617713cca8408f6cdc65d2ee6554217cedff1/PyYAML-6.0-cp36-cp36m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (603kB)\n",
      "Collecting ansiwrap (from papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/03/50/43e775a63e0d632d9be3b3fa1c9b2cbaf3b7870d203655710a3426f47c26/ansiwrap-0.8.4-py2.py3-none-any.whl\n",
      "Collecting tenacity (from papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/f2/a5/f86bc8d67c979020438c8559cc70cfe3a1643fd160d35e09c9cca6a09189/tenacity-8.0.1-py3-none-any.whl\n",
      "Collecting entrypoints (from papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/ac/c6/44694103f8c221443ee6b0041f69e2740d89a25641e62fb4f2ee568f2f9c/entrypoints-0.3-py2.py3-none-any.whl\n",
      "Collecting click (from papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/48/58/c8aa6a8e62cc75f39fee1092c45d6b6ba684122697d7ce7d53f64f98a129/click-8.0.3-py3-none-any.whl (97kB)\n",
      "Collecting black (from papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/9b/27/b2f98b627738b02dcac06ae9e2ab13f14ab906fe6dd6366050c76883d4b5/black-21.12b0-py3-none-any.whl (156kB)\n",
      "Collecting qtconsole (from jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/67/17/d12ae86393682eaa65741808aa5207dfe85b8504874d09f1bbd1feebdc2a/qtconsole-5.2.2-py3-none-any.whl (120kB)\n",
      "Collecting nbconvert (from jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/13/2f/acbe7006548f3914456ee47f97a2033b1b2f3daf921b12ac94105d87c163/nbconvert-6.0.7-py3-none-any.whl (552kB)\n",
      "Collecting jupyter-console (from jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/59/cd/aa2670ffc99eb3e5bbe2294c71e4bf46a9804af4f378d09d7a8950996c9b/jupyter_console-6.4.0-py3-none-any.whl\n",
      "Collecting ipywidgets (from jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/6b/bb/285066ddd710779cb69f03d42fa72fbfe4352b4895eb6abab551eae1535a/ipywidgets-7.6.5-py2.py3-none-any.whl (121kB)\n",
      "Collecting ipykernel (from jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/e9/ad/9101e0ab5e84dd117462bb3a1379d31728a849b6886458452e3d97dc6bba/ipykernel-5.5.6-py3-none-any.whl (121kB)\n",
      "Collecting notebook (from jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/3d/26/bbbd933a180c9e59b1c001e4aa3085faf144471ce923dd7198549b0a38fe/notebook-6.4.6-py3-none-any.whl (9.9MB)\n",
      "Collecting jupyter-core (from nbformat>=5.1.2->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/d5/8e/fad835e31e3f54ea39d2b76027348a347433dcbc674a841ffe0716091c2d/jupyter_core-4.9.1-py3-none-any.whl (86kB)\n",
      "Collecting jsonschema!=2.5.0,>=2.4 (from nbformat>=5.1.2->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/e0/d9/05587ac378b9fd2c352c6f024f13240168365bd753a7e8007522b7025267/jsonschema-4.0.0-py3-none-any.whl (69kB)\n",
      "Collecting traitlets>=4.1 (from nbformat>=5.1.2->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/ca/ab/872a23e29cec3cf2594af7e857f18b687ad21039c1f9b922fac5b9b142d5/traitlets-4.3.3-py2.py3-none-any.whl (75kB)\n",
      "Collecting ipython-genutils (from nbformat>=5.1.2->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/fa/bc/9bd3b5c2b4774d5f33b2d544f1460be9df7df2fe42f352135381c347c69a/ipython_genutils-0.2.0-py2.py3-none-any.whl\n",
      "Collecting urllib3<1.27,>=1.21.1 (from requests->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/af/f4/524415c0744552cce7d8bf3669af78e8a069514405ea4fcbd0cc44733744/urllib3-1.26.7-py2.py3-none-any.whl (138kB)\n",
      "Requirement already satisfied: idna<4,>=2.5; python_version >= \"3\" in /usr/lib/python3/dist-packages (from requests->papermill->-r requirements.txt (line 1)) (2.6)\n",
      "Collecting charset-normalizer~=2.0.0; python_version >= \"3\" (from requests->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/47/84/b06f6729fac8108c5fa3e13cde19b0b3de66ba5538c325496dbe39f5ff8e/charset_normalizer-2.0.9-py3-none-any.whl\n",
      "Collecting certifi>=2017.4.17 (from requests->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/37/45/946c02767aabb873146011e665728b680884cd8fe70dde973c640e45b775/certifi-2021.10.8-py2.py3-none-any.whl (149kB)\n",
      "Collecting async-generator; python_version < \"3.7\" (from nbclient>=0.2.0->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/71/52/39d20e03abd0ac9159c162ec24b93fbcaa111e8400308f2465432495ca2b/async_generator-1.10-py3-none-any.whl\n",
      "Collecting jupyter-client>=6.1.5 (from nbclient>=0.2.0->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/ab/0a/c54391499d614a973f78f8bed12f56d25b0cceb041e88191df4ce553d6f4/jupyter_client-7.1.0-py3-none-any.whl (129kB)\n",
      "Collecting nest-asyncio (from nbclient>=0.2.0->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/06/e0/93453ebab12f5ce9a9ceda2ff71648b30e5f2ce5bba19ee3c95cbd0aaa67/nest_asyncio-1.5.4-py3-none-any.whl\n",
      "Collecting textwrap3>=0.9.2 (from ansiwrap->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/77/9c/a53e561d496ee5866bbeea4d3a850b3b545ed854f8a21007c1e0d872e94d/textwrap3-0.9.2-py2.py3-none-any.whl\n",
      "Collecting importlib-metadata; python_version < \"3.8\" (from click->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/a0/a1/b153a0a4caf7a7e3f15c2cd56c7702e2cf3d89b1b359d1f1c5e59d68f4ce/importlib_metadata-4.8.3-py3-none-any.whl\n",
      "Collecting typed-ast>=1.4.2; python_version < \"3.8\" and implementation_name == \"cpython\" (from black->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/2f/75/46eecff3e4f7d845cd61f0acb5c1174c8aa0fe9744a7af9255696ac61161/typed_ast-1.5.1-cp36-cp36m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (831kB)\n",
      "Collecting typing-extensions>=3.10.0.0 (from black->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/05/e4/baf0031e39cf545f0c9edd5b1a2ea12609b7fcba2d58e118b11753d68cf0/typing_extensions-4.0.1-py3-none-any.whl\n",
      "Collecting pathspec<1,>=0.9.0 (from black->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/42/ba/a9d64c7bcbc7e3e8e5f93a52721b377e994c22d16196e2b0f1236774353a/pathspec-0.9.0-py2.py3-none-any.whl\n",
      "Collecting tomli<2.0.0,>=0.2.6 (from black->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/05/e4/74f9440db36734d7ba83c574c1e7024009ce849208a41f90e94a134dc6d1/tomli-1.2.3-py3-none-any.whl\n",
      "Collecting mypy-extensions>=0.4.3 (from black->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/5c/eb/975c7c080f3223a5cdaff09612f3a5221e4ba534f7039db34c35d95fa6a5/mypy_extensions-0.4.3-py2.py3-none-any.whl\n",
      "Collecting platformdirs>=2 (from black->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/b1/78/dcfd84d3aabd46a9c77260fb47ea5d244806e4daef83aa6fe5d83adb182c/platformdirs-2.4.0-py3-none-any.whl\n",
      "Collecting dataclasses>=0.6; python_version < \"3.7\" (from black->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/fe/ca/75fac5856ab5cfa51bbbcefa250182e50441074fdc3f803f6e76451fab43/dataclasses-0.8-py3-none-any.whl\n",
      "Collecting pygments (from qtconsole->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/07/4c/8add7ee4771c3e217f4d3a7180c60954b9dcab8cee02161fc44044cb1c32/Pygments-2.11.0-py3-none-any.whl (1.1MB)\n",
      "Collecting pyzmq>=17.1 (from qtconsole->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/4b/d3/0cf139d9149bbf3f052b385842afa4e202ea743d85632815d98d0e67685a/pyzmq-22.3.0-cp36-cp36m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.1MB)\n",
      "Collecting qtpy (from qtconsole->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/66/da/b1f7184b82eed20a76066aa0c8482eeababc816012836e47f24090c9f655/QtPy-2.0.0-py3-none-any.whl (62kB)\n",
      "Collecting jinja2>=2.4 (from nbconvert->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/20/9a/e5d9ec41927401e41aea8af6d16e78b5e612bca4699d417f646a9610a076/Jinja2-3.0.3-py3-none-any.whl (133kB)\n",
      "Collecting pandocfilters>=1.4.1 (from nbconvert->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/5e/a8/878258cffd53202a6cc1903c226cf09e58ae3df6b09f8ddfa98033286637/pandocfilters-1.5.0-py2.py3-none-any.whl\n",
      "Collecting mistune<2,>=0.8.1 (from nbconvert->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/09/ec/4b43dae793655b7d8a25f76119624350b4d65eb663459eb9603d7f1f0345/mistune-0.8.4-py2.py3-none-any.whl\n",
      "Collecting defusedxml (from nbconvert->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/07/6c/aa3f2f849e01cb6a001cd8554a88d4c77c5c1a31c95bdf1cf9301e6d9ef4/defusedxml-0.7.1-py2.py3-none-any.whl\n",
      "Collecting bleach (from nbconvert->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/64/cc/74d634e1e5659742973a23bb441404c53a7bedb6cd3962109ca5efb703e8/bleach-4.1.0-py2.py3-none-any.whl (157kB)\n",
      "Collecting jupyterlab-pygments (from nbconvert->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/a8/6f/c34288766797193b512c6508f5994b830fb06134fdc4ca8214daba0aa443/jupyterlab_pygments-0.1.2-py2.py3-none-any.whl\n",
      "Collecting testpath (from nbconvert->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/ac/87/5422f6d056bfbded920ccf380a65de3713a3b95a95ba2255be2a3fb4f464/testpath-0.5.0-py3-none-any.whl (84kB)\n",
      "Collecting prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 (from jupyter-console->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/fb/37/4f9ae5a6cd0ebdfc1fbafcfd03e812df1ed92a92bf0bee09441c52164f58/prompt_toolkit-3.0.24-py3-none-any.whl (374kB)\n",
      "Collecting ipython (from jupyter-console->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/27/2b/547334a7763cb0e0016c2756fbda7b0eb862524e2860ec3f0913d8c432a5/ipython-7.16.2-py3-none-any.whl (782kB)\n",
      "Collecting jupyterlab-widgets>=1.0.0; python_version >= \"3.6\" (from ipywidgets->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/18/4d/22a93473bca99c80f2d23f867ebbfee2f6c8e186bf678864eec641500910/jupyterlab_widgets-1.0.2-py3-none-any.whl (243kB)\n",
      "Collecting widgetsnbextension~=3.5.0 (from ipywidgets->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/d7/31/7c1107fa30c621cd1d36410e9bbab86f6a518dc208aaec01f02ac6d5c2d2/widgetsnbextension-3.5.2-py2.py3-none-any.whl (1.6MB)\n",
      "Collecting tornado>=4.2 (from ipykernel->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/85/26/e710295dcb4aac62b08f22d07efc899574476db37532159a7f71713cdaf2/tornado-6.1-cp36-cp36m-manylinux2010_x86_64.whl (427kB)\n",
      "Collecting argon2-cffi (from notebook->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/a8/07/946d5a9431bae05a776a59746ec385fbb79b526738d25e4202d3e0bbf7f4/argon2_cffi-21.3.0-py3-none-any.whl\n",
      "Collecting terminado>=0.8.3 (from notebook->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/cb/17/b1162b39786c44e14d30ee557fbf41276c4a966dab01106c15fb70f5c27a/terminado-0.12.1-py3-none-any.whl\n",
      "Collecting Send2Trash>=1.8.0 (from notebook->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/47/26/3435896d757335ea53dce5abf8d658ca80757a7a06258451b358f10232be/Send2Trash-1.8.0-py3-none-any.whl\n",
      "Collecting prometheus-client (from notebook->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/df/6c/6c5f9404977f8f9caa30c1a408f6cc5ea6e0c1949761f24d0a33239b49c5/prometheus_client-0.12.0-py2.py3-none-any.whl (57kB)\n",
      "Collecting pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 (from jsonschema!=2.5.0,>=2.4->nbformat>=5.1.2->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/6c/19/1af501f6f388a40ede6d0185ba481bdb18ffc99deab0dd0d092b173bc0f4/pyrsistent-0.18.0-cp36-cp36m-manylinux1_x86_64.whl (117kB)\n",
      "Collecting attrs>=17.4.0 (from jsonschema!=2.5.0,>=2.4->nbformat>=5.1.2->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/be/be/7abce643bfdf8ca01c48afa2ddf8308c2308b0c3b239a44e57d020afa0ef/attrs-21.4.0-py2.py3-none-any.whl (60kB)\n",
      "Requirement already satisfied: six in /usr/lib/python3/dist-packages (from traitlets>=4.1->nbformat>=5.1.2->papermill->-r requirements.txt (line 1)) (1.11.0)\n",
      "Collecting decorator (from traitlets>=4.1->nbformat>=5.1.2->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/3d/cc/d7b758e54779f7e465179427de7e78c601d3330d6c411ea7ba9ae2f38102/decorator-5.1.0-py3-none-any.whl\n",
      "Collecting python-dateutil>=2.1 (from jupyter-client>=6.1.5->nbclient>=0.2.0->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/36/7a/87837f39d0296e723bb9b62bbb257d0355c7f6128853c78955f57342a56d/python_dateutil-2.8.2-py2.py3-none-any.whl (247kB)\n",
      "Collecting zipp>=0.5 (from importlib-metadata; python_version < \"3.8\"->click->papermill->-r requirements.txt (line 1))\n",
      "  Downloading https://files.pythonhosted.org/packages/bd/df/d4a4974a3e3957fd1c1fa3082366d7fff6e428ddb55f074bf64876f8e8ad/zipp-3.6.0-py3-none-any.whl\n",
      "Collecting packaging (from qtpy->qtconsole->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/05/8e/8de486cbd03baba4deef4142bd643a3e7bbe954a784dc1bb17142572d127/packaging-21.3-py3-none-any.whl (40kB)\n",
      "Collecting MarkupSafe>=2.0 (from jinja2>=2.4->nbconvert->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/08/dc/a5ed54fcc61f75343663ee702cbf69831dcec9b1a952ae21cf3d1fbc56ba/MarkupSafe-2.0.1-cp36-cp36m-manylinux2010_x86_64.whl\n",
      "Collecting webencodings (from bleach->nbconvert->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/f4/24/2a3e3df732393fed8b3ebf2ec078f05546de641fe1b667ee316ec1dcf3b7/webencodings-0.5.1-py2.py3-none-any.whl\n",
      "Collecting wcwidth (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->jupyter-console->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/59/7c/e39aca596badaf1b78e8f547c807b04dae603a433d3e7a7e04d67f2ef3e5/wcwidth-0.2.5-py2.py3-none-any.whl\n",
      "Collecting jedi<=0.17.2,>=0.10 (from ipython->jupyter-console->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/c3/d4/36136b18daae06ad798966735f6c3fb96869c1be9f8245d2a8f556e40c36/jedi-0.17.2-py2.py3-none-any.whl (1.4MB)\n",
      "Requirement already satisfied: setuptools>=18.5 in /usr/local/lib/python3.6/dist-packages (from ipython->jupyter-console->jupyter->-r requirements.txt (line 2)) (41.0.1)\n",
      "Collecting pexpect; sys_platform != \"win32\" (from ipython->jupyter-console->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/39/7b/88dbb785881c28a102619d46423cb853b46dbccc70d3ac362d99773a78ce/pexpect-4.8.0-py2.py3-none-any.whl (59kB)\n",
      "Collecting pickleshare (from ipython->jupyter-console->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/9a/41/220f49aaea88bc6fa6cba8d05ecf24676326156c23b991e80b3f2fc24c77/pickleshare-0.7.5-py2.py3-none-any.whl\n",
      "Collecting backcall (from ipython->jupyter-console->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/4c/1c/ff6546b6c12603d8dd1070aa3c3d273ad4c07f5771689a7b69a550e8c951/backcall-0.2.0-py2.py3-none-any.whl\n",
      "Collecting argon2-cffi-bindings (from argon2-cffi->notebook->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/b9/e9/184b8ccce6683b0aa2fbb7ba5683ea4b9c5763f1356347f1312c32e3c66e/argon2-cffi-bindings-21.2.0.tar.gz (1.8MB)\n",
      "  Installing build dependencies: started\n",
      "  Installing build dependencies: finished with status 'done'\n",
      "  Getting requirements to build wheel: started\n",
      "  Getting requirements to build wheel: finished with status 'done'\n",
      "    Preparing wheel metadata: started\n",
      "    Preparing wheel metadata: finished with status 'done'\n",
      "Collecting ptyprocess; os_name != \"nt\" (from terminado>=0.8.3->notebook->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/22/a6/858897256d0deac81a172289110f31629fc4cee19b6f01283303e18c8db3/ptyprocess-0.7.0-py2.py3-none-any.whl\n",
      "Collecting pyparsing!=3.0.5,>=2.0.2 (from packaging->qtpy->qtconsole->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/a0/34/895006117f6fce0b4de045c87e154ee4a20c68ec0a4c9a36d900888fb6bc/pyparsing-3.0.6-py3-none-any.whl (97kB)\n",
      "Collecting parso<0.8.0,>=0.7.0 (from jedi<=0.17.2,>=0.10->ipython->jupyter-console->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/93/d1/e635bdde32890db5aeb2ffbde17e74f68986305a4466b0aa373b861e3f00/parso-0.7.1-py2.py3-none-any.whl (109kB)\n",
      "Collecting cffi>=1.0.1 (from argon2-cffi-bindings->argon2-cffi->notebook->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/49/7b/449daf9cacfd7355cea1b4106d2be614315c29ac16567e01756167f6daab/cffi-1.15.0-cp36-cp36m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (405kB)\n",
      "Collecting pycparser (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->notebook->jupyter->-r requirements.txt (line 2))\n",
      "  Downloading https://files.pythonhosted.org/packages/62/d5/5f610ebe421e85889f2e55e33b7f9a6795bd982198517d912eb1c76e1a53/pycparser-2.21-py2.py3-none-any.whl (118kB)\n",
      "Building wheels for collected packages: argon2-cffi-bindings\n",
      "  Building wheel for argon2-cffi-bindings (PEP 517): started\n",
      "  Building wheel for argon2-cffi-bindings (PEP 517): finished with status 'done'\n",
      "  Stored in directory: /tmp/pip-ephem-wheel-cache-ejj5ik8e/wheels/26/f9/3c/bbbffbf3f8fbe065773fdfba1a9ae8a40d317b03e9feffc93e\n",
      "Successfully built argon2-cffi-bindings\n",
      "Installing collected packages: ipython-genutils, decorator, traitlets, jupyter-core, zipp, typing-extensions, importlib-metadata, pyrsistent, attrs, jsonschema, nbformat, tqdm, urllib3, charset-normalizer, certifi, requests, async-generator, pyzmq, nest-asyncio, entrypoints, python-dateutil, tornado, jupyter-client, nbclient, pyyaml, textwrap3, ansiwrap, tenacity, click, typed-ast, pathspec, tomli, mypy-extensions, platformdirs, dataclasses, black, papermill, pygments, parso, jedi, wcwidth, prompt-toolkit, ptyprocess, pexpect, pickleshare, backcall, ipython, ipykernel, pyparsing, packaging, qtpy, qtconsole, MarkupSafe, jinja2, pandocfilters, mistune, defusedxml, webencodings, bleach, jupyterlab-pygments, testpath, nbconvert, jupyter-console, jupyterlab-widgets, pycparser, cffi, argon2-cffi-bindings, argon2-cffi, terminado, Send2Trash, prometheus-client, notebook, widgetsnbextension, ipywidgets, jupyter\n",
      "Successfully installed MarkupSafe-2.0.1 Send2Trash-1.8.0 ansiwrap-0.8.4 argon2-cffi-21.3.0 argon2-cffi-bindings-21.2.0 async-generator-1.10 attrs-21.4.0 backcall-0.2.0 black-21.12b0 bleach-4.1.0 certifi-2021.10.8 cffi-1.15.0 charset-normalizer-2.0.9 click-8.0.3 dataclasses-0.8 decorator-5.1.0 defusedxml-0.7.1 entrypoints-0.3 importlib-metadata-4.8.3 ipykernel-5.5.6 ipython-7.16.2 ipython-genutils-0.2.0 ipywidgets-7.6.5 jedi-0.17.2 jinja2-3.0.3 jsonschema-4.0.0 jupyter-1.0.0 jupyter-client-7.1.0 jupyter-console-6.4.0 jupyter-core-4.9.1 jupyterlab-pygments-0.1.2 jupyterlab-widgets-1.0.2 mistune-0.8.4 mypy-extensions-0.4.3 nbclient-0.5.9 nbconvert-6.0.7 nbformat-5.1.3 nest-asyncio-1.5.4 notebook-6.4.6 packaging-21.3 pandocfilters-1.5.0 papermill-2.3.3 parso-0.7.1 pathspec-0.9.0 pexpect-4.8.0 pickleshare-0.7.5 platformdirs-2.4.0 prometheus-client-0.12.0 prompt-toolkit-3.0.24 ptyprocess-0.7.0 pycparser-2.21 pygments-2.11.0 pyparsing-3.0.6 pyrsistent-0.18.0 python-dateutil-2.8.2 pyyaml-6.0 pyzmq-22.3.0 qtconsole-5.\n",
      "2.2 qtpy-2.0.0 requests-2.26.0 tenacity-8.0.1 terminado-0.12.1 testpath-0.5.0 textwrap3-0.9.2 tomli-1.2.3 tornado-6.1 tqdm-4.62.3 traitlets-4.3.3 typed-ast-1.5.1 typing-extensions-4.0.1 urllib3-1.26.7 wcwidth-0.2.5 webencodings-0.5.1 widgetsnbextension-3.5.2 zipp-3.6.0\n",
      "WARNING: You are using pip version 19.1.1, however version 21.3.1 is available.\n",
      "You should consider upgrading via the 'pip install --upgrade pip' command.\n",
      "\u001b[36mINFO\u001b[0m[0053] Taking snapshot of full filesystem...\n",
      "\u001b[36mINFO\u001b[0m[0060] Using files from context: [/kaniko/buildcontext/app]\n",
      "\u001b[36mINFO\u001b[0m[0060] COPY /app/ /app/\n",
      "\u001b[36mINFO\u001b[0m[0060] Taking snapshot of files...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[W 211230 14:13:19 aws:70] Not able to find aws credentials secret: aws-secret\n",
      "[W 211230 14:13:19 job:90] The job fairing-job-lh4b9 launched.\n",
      "[W 211230 14:13:19 manager:255] Waiting for fairing-job-lh4b9-4h7mp to start...\n",
      "[W 211230 14:13:19 manager:255] Waiting for fairing-job-lh4b9-4h7mp to start...\n",
      "[W 211230 14:13:19 manager:255] Waiting for fairing-job-lh4b9-4h7mp to start...\n",
      "[I 211230 14:13:26 manager:261] Pod started running True\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Input Notebook:  train.ipynb\n",
      "Output Notebook: fairing_output_notebook.ipynb\n",
      "Executing notebook with kernel: python3\n",
      "Executing Cell 1---------------------------------------\n",
      "training in notebook!\n",
      "\n",
      "Ending Cell 1------------------------------------------\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[W 211230 14:13:28 job:162] Cleaning up job fairing-job-lh4b9...\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'fairing-job-lh4b9'"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# We already have a train.ipynb in the same folder\n",
    "job = TrainJob(\"train.ipynb\", input_files=[\"requirements.txt\"], base_docker_image=BASE_IMAGE, docker_registry=DOCKER_REGISTRY, backend=KubeflowAWSBackend())\n",
    "job.submit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```

fairing_resources.ipnyb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Allocate resources in remote container"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from kubeflow import fairing\n",
    "from kubeflow.fairing import TrainJob\n",
    "from kubeflow.fairing.backends import KubeflowAWSBackend\n",
    "from kubeflow.fairing.kubernetes.utils import get_resource_mutator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU count: 2\n",
      "Memory: 7.674312591552734\n"
     ]
    }
   ],
   "source": [
    "import multiprocessing\n",
    "import os\n",
    "import sys\n",
    "\n",
    "def train():\n",
    "    print(\"CPU count: {}\".format(multiprocessing.cpu_count()))\n",
    "    print(\"Memory: {}\".format(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')/(1024.**3)))\n",
    "\n",
    "train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[W 211230 14:38:02 tasks:62] Using builder: <class 'kubeflow.fairing.builders.append.append.AppendBuilder'>\n",
      "[I 211230 14:38:02 tasks:66] Building the docker image.\n",
      "[W 211230 14:38:02 append:50] Building image using Append builder...\n",
      "[I 211230 14:38:02 base:107] Creating docker context: /tmp/fairing_context_zwxucbpb\n",
      "[W 211230 14:38:02 base:94] /usr/local/lib/python3.6/dist-packages/kubeflow/fairing/__init__.py already exists in Fairing context, skipping...\n",
      "[I 211230 14:38:02 docker_creds_:234] Loading Docker credentials for repository 'tensorflow/tensorflow:1.14.0-py3'\n",
      "[W 211230 14:38:02 append:54] Image successfully built in 0.9029710159993556s.\n",
      "[W 211230 14:38:02 append:94] Pushing image 444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:97624C1B...\n",
      "[I 211230 14:38:02 docker_creds_:234] Loading Docker credentials for repository '444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:97624C1B'\n",
      "[W 211230 14:38:02 append:81] Uploading 444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:97624C1B\n",
      "[I 211230 14:38:03 docker_session_:280] Layer sha256:5e671b828b2af02924968841e5d12084fa78e8722e9510402aaee80dc5d7a6db exists, skipping\n",
      "[I 211230 14:38:03 docker_session_:280] Layer sha256:14ca88e9f6723ce82bc14b241cda8634f6d19677184691d086662641ab96fe68 exists, skipping\n",
      "[I 211230 14:38:03 docker_session_:280] Layer sha256:b054a26005b7f3b032577f811421fab5ec3b42ce45a4012dfa00cf6ed6191b0f exists, skipping\n",
      "[I 211230 14:38:03 docker_session_:280] Layer sha256:68543864d6442a851eaff0500161b92e4a151051cf7ed2649b3790a3f876bada exists, skipping\n",
      "[I 211230 14:38:03 docker_session_:280] Layer sha256:016724bbd2c9643f24eff7c1e86d9202d7c04caddd7fdd4375a77e3998ce8203 exists, skipping\n",
      "[I 211230 14:38:03 docker_session_:280] Layer sha256:a31c3b1caad473a474d574283741f880e37c708cc06ee620d3e93fa602125ee0 exists, skipping\n",
      "[I 211230 14:38:03 docker_session_:280] Layer sha256:5bd1cb59702536c10e96bb14e54846922c9b257580d4e2c733076a922525240b exists, skipping\n",
      "[I 211230 14:38:03 docker_session_:280] Layer sha256:2b940936f9933b7737cf407f2149dd7393998d7a0bee5acf1c4a57b0487cef79 exists, skipping\n",
      "[I 211230 14:38:03 docker_session_:280] Layer sha256:8832e37735788665026956430021c6d1919980288c66c4526502965aeb5ac006 exists, skipping\n",
      "[I 211230 14:38:03 docker_session_:280] Layer sha256:5b7339215d1d5f8e68622d584a224f60339f5bef41dbd74330d081e912f0cddd exists, skipping\n",
      "[I 211230 14:38:03 docker_session_:284] Layer sha256:36055d56a1c4b153d66a95c38263ff7032980d04676037f3ceddb34eb5dabea6 pushed.\n",
      "[I 211230 14:38:03 docker_session_:284] Layer sha256:601cef744423366ad3ddedfa06e4bda575be7d78e5a84d52666d773b3401462d pushed.\n",
      "[I 211230 14:38:03 docker_session_:334] Finished upload of: 444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:97624C1B\n",
      "[W 211230 14:38:03 append:99] Pushed image 444175659137.dkr.ecr.us-west-2.amazonaws.com/fairing-job:97624C1B in 0.7037836420004169s.\n",
      "[W 211230 14:38:03 aws:70] Not able to find aws credentials secret: aws-secret\n",
      "[W 211230 14:38:03 aws:70] Not able to find aws credentials secret: aws-secret\n",
      "[W 211230 14:38:03 job:90] The job fairing-job-qjcmr launched.\n",
      "[W 211230 14:38:03 manager:255] Waiting for fairing-job-qjcmr-fxvtg to start...\n",
      "[W 211230 14:38:03 manager:255] Waiting for fairing-job-qjcmr-fxvtg to start...\n"
     ]
    }
   ],
   "source": [
    "AWS_ACCOUNT_ID=fairing.cloud.aws.guess_account_id()\n",
    "AWS_REGION='us-west-2'\n",
    "DOCKER_REGISTRY = '{}.dkr.ecr.{}.amazonaws.com'.format(AWS_ACCOUNT_ID, AWS_REGION)\n",
    "PY_VERSION = \".\".join([str(x) for x in sys.version_info[0:3]])\n",
    "BASE_IMAGE = '{}/python:{}'.format(DOCKER_REGISTRY, PY_VERSION)\n",
    "# TODO: bug to fix. use tensorflow image temporarily\n",
    "BASE_IMAGE = 'tensorflow/tensorflow:1.14.0-py3'\n",
    "\n",
    "job = TrainJob(train, base_docker_image=BASE_IMAGE, docker_registry=DOCKER_REGISTRY,  backend=KubeflowAWSBackend(), \n",
    "              pod_spec_mutators=[get_resource_mutator(cpu=1, memory=2)])\n",
    "job.submit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}



fairing_e2e.ipnyb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Train and deploy model on Kubeflow in Notebooks\n",
    "\n",
    "This examples comes from a upstream fairing [example](https://github.com/kubeflow/fairing/tree/master/examples/prediction).\n",
    "\n",
    "\n",
    "Please check Kaggle competiton [\n",
    "House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)\n",
    "for details about the ML problem we want to resolve.\n",
    "\n",
    "This notebook introduces you to using Kubeflow Fairing to train and deploy a model to Kubeflow on Amazon EKS. This notebook demonstrate how to:\n",
    "\n",
    "* Train an XGBoost model in a local notebook,\n",
    "* Use Kubeflow Fairing to train an XGBoost model remotely on Kubeflow,\n",
    "* Use Kubeflow Fairing to deploy a trained model to Kubeflow,\n",
    "* Call the deployed endpoint for predictions.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Install python dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Overwriting requirements.txt\n"
     ]
    }
   ],
   "source": [
    "%%writefile requirements.txt\n",
    "pandas\n",
    "joblib\n",
    "numpy\n",
    "xgboost\n",
    "scikit-learn>=0.21.0\n",
    "seldon-core\n",
    "tornado>=6.0.3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (from -r requirements.txt (line 1)) (0.24.2)\n",
      "Collecting joblib\n",
      "  Downloading joblib-1.1.0-py2.py3-none-any.whl (306 kB)\n",
      "\u001b[K     |████████████████████████████████| 306 kB 6.8 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from -r requirements.txt (line 3)) (1.18.3)\n",
      "Requirement already satisfied: xgboost in /usr/local/lib/python3.6/dist-packages (from -r requirements.txt (line 4)) (1.1.1)\n",
      "Collecting scikit-learn>=0.21.0\n",
      "  Downloading scikit_learn-0.24.2-cp36-cp36m-manylinux2010_x86_64.whl (22.2 MB)\n",
      "\u001b[K     |████████████████████████████████| 22.2 MB 44.8 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting seldon-core\n",
      "  Downloading seldon_core-1.12.0-py3-none-any.whl (133 kB)\n",
      "\u001b[K     |████████████████████████████████| 133 kB 37.3 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: tornado>=6.0.3 in /usr/local/lib/python3.6/dist-packages (from -r requirements.txt (line 7)) (6.0.4)\n",
      "Requirement already satisfied: pytz>=2011k in /usr/local/lib/python3.6/dist-packages (from pandas->-r requirements.txt (line 1)) (2019.3)\n",
      "Requirement already satisfied: python-dateutil>=2.5.0 in /usr/local/lib/python3.6/dist-packages (from pandas->-r requirements.txt (line 1)) (2.8.0)\n",
      "Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from xgboost->-r requirements.txt (line 4)) (1.2.2)\n",
      "Collecting threadpoolctl>=2.0.0\n",
      "  Downloading threadpoolctl-3.0.0-py3-none-any.whl (14 kB)\n",
      "Collecting grpcio-reflection<1.35.0\n",
      "  Downloading grpcio-reflection-1.34.1.tar.gz (11 kB)\n",
      "Requirement already satisfied: protobuf<4.0.0 in /usr/local/lib/python3.6/dist-packages (from seldon-core->-r requirements.txt (line 6)) (3.19.1)\n",
      "Collecting PyYAML<5.5,>=5.4\n",
      "  Downloading PyYAML-5.4.1-cp36-cp36m-manylinux1_x86_64.whl (640 kB)\n",
      "\u001b[K     |███████████████████████████████▊| 634 kB 25.3 MB/s eta 0:00:01     |████████████████████████████████| 640 kB 25.3 MB/s \n",
      "\u001b[?25hRequirement already satisfied: jsonschema<4.0.0 in /usr/local/lib/python3.6/dist-packages (from seldon-core->-r requirements.txt (line 6)) (3.2.0)\n",
      "Collecting flatbuffers<2.0.0\n",
      "  Downloading flatbuffers-1.12-py2.py3-none-any.whl (15 kB)\n",
      "Collecting grpcio-opentracing<1.2.0,>=1.1.4\n",
      "  Downloading grpcio_opentracing-1.1.4-py3-none-any.whl (14 kB)\n",
      "Collecting urllib3==1.26.5\n",
      "  Downloading urllib3-1.26.5-py2.py3-none-any.whl (138 kB)\n",
      "\u001b[K     |████████████████████████████████| 138 kB 63.0 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: Flask<2.0.0 in /usr/local/lib/python3.6/dist-packages (from seldon-core->-r requirements.txt (line 6)) (1.1.1)\n",
      "Collecting click<8.1,>=8.0.0a1\n",
      "  Downloading click-8.0.3-py3-none-any.whl (97 kB)\n",
      "\u001b[K     |████████████████████████████████| 97 kB 13.5 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.6/dist-packages (from seldon-core->-r requirements.txt (line 6)) (46.1.3)\n",
      "Collecting jaeger-client<4.5.0,>=4.1.0\n",
      "  Downloading jaeger-client-4.4.0.tar.gz (83 kB)\n",
      "\u001b[K     |████████████████████████████████| 83 kB 4.2 MB/s  eta 0:00:01\n",
      "\u001b[?25hCollecting Flask-OpenTracing<1.2.0,>=1.1.0\n",
      "  Downloading Flask-OpenTracing-1.1.0.tar.gz (8.2 kB)\n",
      "Requirement already satisfied: grpcio<2.0.0 in /usr/local/lib/python3.6/dist-packages (from seldon-core->-r requirements.txt (line 6)) (1.43.0)\n",
      "Collecting Flask-cors<4.0.0\n",
      "  Downloading Flask_Cors-3.0.10-py2.py3-none-any.whl (14 kB)\n",
      "Requirement already satisfied: gunicorn<20.2.0,>=19.9.0 in /usr/local/lib/python3.6/dist-packages (from seldon-core->-r requirements.txt (line 6)) (20.0.4)\n",
      "Requirement already satisfied: prometheus-client<0.9.0,>=0.7.1 in /usr/local/lib/python3.6/dist-packages (from seldon-core->-r requirements.txt (line 6)) (0.8.0)\n",
      "Collecting cryptography<3.5,>=3.4\n",
      "  Downloading cryptography-3.4.8-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.2 MB)\n",
      "\u001b[K     |████████████████████████████████| 3.2 MB 73.0 MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting opentracing<2.5.0,>=2.2.0\n",
      "  Downloading opentracing-2.4.0.tar.gz (46 kB)\n",
      "\u001b[K     |████████████████████████████████| 46 kB 8.3 MB/s  eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: requests<3.0.0 in /usr/local/lib/python3.6/dist-packages (from seldon-core->-r requirements.txt (line 6)) (2.22.0)\n",
      "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.6/dist-packages (from python-dateutil>=2.5.0->pandas->-r requirements.txt (line 1)) (1.14.0)\n",
      "Requirement already satisfied: pyrsistent>=0.14.0 in /usr/local/lib/python3.6/dist-packages (from jsonschema<4.0.0->seldon-core->-r requirements.txt (line 6)) (0.16.0)\n",
      "Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.6/dist-packages (from jsonschema<4.0.0->seldon-core->-r requirements.txt (line 6)) (19.3.0)\n",
      "Requirement already satisfied: importlib-metadata; python_version < \"3.8\" in /usr/local/lib/python3.6/dist-packages (from jsonschema<4.0.0->seldon-core->-r requirements.txt (line 6)) (1.6.0)\n",
      "Requirement already satisfied: Werkzeug>=0.15 in /usr/local/lib/python3.6/dist-packages (from Flask<2.0.0->seldon-core->-r requirements.txt (line 6)) (1.0.1)\n",
      "Requirement already satisfied: Jinja2>=2.10.1 in /usr/local/lib/python3.6/dist-packages (from Flask<2.0.0->seldon-core->-r requirements.txt (line 6)) (2.11.2)\n",
      "Requirement already satisfied: itsdangerous>=0.24 in /usr/local/lib/python3.6/dist-packages (from Flask<2.0.0->seldon-core->-r requirements.txt (line 6)) (1.1.0)\n",
      "Collecting threadloop<2,>=1\n",
      "  Downloading threadloop-1.0.2.tar.gz (4.9 kB)\n",
      "Collecting thrift\n",
      "  Downloading thrift-0.15.0.tar.gz (59 kB)\n",
      "\u001b[K     |████████████████████████████████| 59 kB 11.6 MB/s eta 0:00:01\n",
      "\u001b[?25hRequirement already satisfied: cffi>=1.12 in /usr/local/lib/python3.6/dist-packages (from cryptography<3.5,>=3.4->seldon-core->-r requirements.txt (line 6)) (1.14.0)\n",
      "Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0->seldon-core->-r requirements.txt (line 6)) (2.8)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0->seldon-core->-r requirements.txt (line 6)) (2020.4.5.1)\n",
      "Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0->seldon-core->-r requirements.txt (line 6)) (3.0.4)\n",
      "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata; python_version < \"3.8\"->jsonschema<4.0.0->seldon-core->-r requirements.txt (line 6)) (3.1.0)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.6/dist-packages (from Jinja2>=2.10.1->Flask<2.0.0->seldon-core->-r requirements.txt (line 6)) (1.1.1)\n",
      "Requirement already satisfied: pycparser in /usr/local/lib/python3.6/dist-packages (from cffi>=1.12->cryptography<3.5,>=3.4->seldon-core->-r requirements.txt (line 6)) (2.20)\n",
      "Building wheels for collected packages: grpcio-reflection, jaeger-client, Flask-OpenTracing, opentracing, threadloop, thrift\n",
      "  Building wheel for grpcio-reflection (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25h  Created wheel for grpcio-reflection: filename=grpcio_reflection-1.34.1-py3-none-any.whl size=14412 sha256=00834c6f24df9830b578d34a313178c4a10ccfedb8f89fa6f8982563eea3e2e8\n",
      "  Stored in directory: /home/jovyan/.cache/pip/wheels/9b/84/80/81f4dc4afff82cce892567df37472395abf1a1bb675caec1ec\n",
      "  Building wheel for jaeger-client (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25h  Created wheel for jaeger-client: filename=jaeger_client-4.4.0-py3-none-any.whl size=64326 sha256=3529436379b979b4e7ae6f2ceb24ec444d4ba4f2ac3056b70a497fd81d8956fe\n",
      "  Stored in directory: /home/jovyan/.cache/pip/wheels/6e/ea/ec/560f1cffece42c70b9c68b1dc3519c769883650396adcb90b7\n",
      "  Building wheel for Flask-OpenTracing (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25h  Created wheel for Flask-OpenTracing: filename=Flask_OpenTracing-1.1.0-py3-none-any.whl size=9070 sha256=35b9f565b28cfa942ed982f106bdacb4e49d080209c2d145b98f8229e3780638\n",
      "  Stored in directory: /home/jovyan/.cache/pip/wheels/ad/4b/2d/24ff0da0a0b53c7c77ce59b843bcceaf644c88703241e59615\n",
      "  Building wheel for opentracing (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25h  Created wheel for opentracing: filename=opentracing-2.4.0-py3-none-any.whl size=51400 sha256=95039a0c8646b29927bedc489f07398000ad8026cc8b3d46607f178020f461af\n",
      "  Stored in directory: /home/jovyan/.cache/pip/wheels/83/73/4c/0e331f57d4702becb1fca9d9148277aca96d127bd838faf85e\n",
      "  Building wheel for threadloop (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25h  Created wheel for threadloop: filename=threadloop-1.0.2-py3-none-any.whl size=3423 sha256=6771eeeca10570f2cd3cf84998a60adc837681b52ba73fe45cfabd6396a3237f\n",
      "  Stored in directory: /home/jovyan/.cache/pip/wheels/02/54/65/9f87de48fe8fcaaee30f279973d946ad55f9df56b93b3e78da\n",
      "  Building wheel for thrift (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25h  Created wheel for thrift: filename=thrift-0.15.0-cp36-cp36m-linux_x86_64.whl size=258868 sha256=55b26292faa6805b97a75bb40ad4129497c5c8b480664a2fdd58beb59bca00ea\n",
      "  Stored in directory: /home/jovyan/.cache/pip/wheels/2a/91/a0/aea72c790bd97f9c2f31ac5ca0b20b25992010a31e4357f50b\n",
      "Successfully built grpcio-reflection jaeger-client Flask-OpenTracing opentracing threadloop thrift\n",
      "\u001b[31mERROR: requests 2.22.0 has requirement urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1, but you'll have urllib3 1.26.5 which is incompatible.\u001b[0m\n",
      "\u001b[31mERROR: kubeflow-fairing 0.7.1 has requirement urllib3==1.24.2, but you'll have urllib3 1.26.5 which is incompatible.\u001b[0m\n",
      "\u001b[31mERROR: kfserving 0.3.0.1 has requirement kubernetes==10.0.1, but you'll have kubernetes 11.0.0 which is incompatible.\u001b[0m\n",
      "\u001b[31mERROR: botocore 1.15.42 has requirement urllib3<1.26,>=1.20; python_version != \"3.4\", but you'll have urllib3 1.26.5 which is incompatible.\u001b[0m\n",
      "\u001b[31mERROR: awscli 1.18.42 has requirement PyYAML<5.4,>=3.10; python_version != \"3.4\", but you'll have pyyaml 5.4.1 which is incompatible.\u001b[0m\n",
      "Installing collected packages: joblib, threadpoolctl, scikit-learn, grpcio-reflection, PyYAML, flatbuffers, opentracing, grpcio-opentracing, urllib3, click, threadloop, thrift, jaeger-client, Flask-OpenTracing, Flask-cors, cryptography, seldon-core\n",
      "  Attempting uninstall: scikit-learn\n",
      "    Found existing installation: scikit-learn 0.20.3\n",
      "    Uninstalling scikit-learn-0.20.3:\n",
      "      Successfully uninstalled scikit-learn-0.20.3\n",
      "  Attempting uninstall: PyYAML\n",
      "    Found existing installation: PyYAML 5.3.1\n",
      "    Uninstalling PyYAML-5.3.1:\n",
      "      Successfully uninstalled PyYAML-5.3.1\n",
      "  Attempting uninstall: urllib3\n",
      "    Found existing installation: urllib3 1.24.2\n",
      "    Uninstalling urllib3-1.24.2:\n",
      "      Successfully uninstalled urllib3-1.24.2\n",
      "  Attempting uninstall: click\n",
      "    Found existing installation: click 7.1.1\n",
      "    Uninstalling click-7.1.1:\n",
      "      Successfully uninstalled click-7.1.1\n",
      "  Attempting uninstall: cryptography\n",
      "    Found existing installation: cryptography 2.9.1\n",
      "    Uninstalling cryptography-2.9.1:\n",
      "      Successfully uninstalled cryptography-2.9.1\n",
      "Successfully installed Flask-OpenTracing-1.1.0 Flask-cors-3.0.10 PyYAML-5.4.1 click-8.0.3 cryptography-3.4.8 flatbuffers-1.12 grpcio-opentracing-1.1.4 grpcio-reflection-1.34.1 jaeger-client-4.4.0 joblib-1.1.0 opentracing-2.4.0 scikit-learn-0.24.2 seldon-core-1.12.0 threadloop-1.0.2 threadpoolctl-3.0.0 thrift-0.15.0 urllib3-1.26.5\n",
      "\u001b[33mWARNING: You are using pip version 20.0.2; however, version 21.3.1 is available.\n",
      "You should consider upgrading via the '/usr/bin/python3 -m pip install --upgrade pip' command.\u001b[0m\n"
     ]
    }
   ],
   "source": [
    "!pip install -r requirements.txt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Develop your model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import argparse\n",
    "import logging\n",
    "import joblib\n",
    "import sys\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.impute import SimpleImputer\n",
    "from xgboost import XGBRegressor\n",
    "\n",
    "logging.basicConfig(format='%(message)s')\n",
    "logging.getLogger().setLevel(logging.INFO)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def read_input(file_name, test_size=0.25):\n",
    "    \"\"\"Read input data and split it into train and test.\"\"\"\n",
    "    data = pd.read_csv(file_name)\n",
    "    data.dropna(axis=0, subset=['SalePrice'], inplace=True)\n",
    "\n",
    "    y = data.SalePrice\n",
    "    X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])\n",
    "\n",
    "    train_X, test_X, train_y, test_y = train_test_split(X.values,\n",
    "                                                      y.values,\n",
    "                                                      test_size=test_size,\n",
    "                                                      shuffle=False)\n",
    "\n",
    "    imputer = SimpleImputer()\n",
    "    train_X = imputer.fit_transform(train_X)\n",
    "    test_X = imputer.transform(test_X)\n",
    "\n",
    "    return (train_X, train_y), (test_X, test_y)\n",
    "\n",
    "def train_model(train_X,\n",
    "                train_y,\n",
    "                test_X,\n",
    "                test_y,\n",
    "                n_estimators,\n",
    "                learning_rate):\n",
    "    \"\"\"Train the model using XGBRegressor.\"\"\"\n",
    "    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate)\n",
    "\n",
    "    model.fit(train_X,\n",
    "            train_y,\n",
    "            early_stopping_rounds=40,\n",
    "            eval_set=[(test_X, test_y)])\n",
    "\n",
    "    print(\"Best RMSE on eval: %.2f with %d rounds\" %\n",
    "               (model.best_score,\n",
    "                model.best_iteration+1))\n",
    "    return model\n",
    "\n",
    "def eval_model(model, test_X, test_y):\n",
    "    \"\"\"Evaluate the model performance.\"\"\"\n",
    "    predictions = model.predict(test_X)\n",
    "    logging.info(\"mean_absolute_error=%.2f\", mean_absolute_error(predictions, test_y))\n",
    "\n",
    "def save_model(model, model_file):\n",
    "    \"\"\"Save XGBoost model for serving.\"\"\"\n",
    "    joblib.dump(model, model_file)\n",
    "    logging.info(\"Model export success: %s\", model_file)\n",
    "    \n",
    "    \n",
    "class HousingServe(object):\n",
    "    \n",
    "    def __init__(self):\n",
    "        self.train_input = \"ames_dataset/train.csv\"\n",
    "        self.n_estimators = 50\n",
    "        self.learning_rate = 0.1\n",
    "        self.model_file = \"trained_ames_model.dat\"\n",
    "        self.model = None\n",
    "\n",
    "    def train(self):\n",
    "        (train_X, train_y), (test_X, test_y) = read_input(self.train_input)\n",
    "        model = train_model(train_X,\n",
    "                          train_y,\n",
    "                          test_X,\n",
    "                          test_y,\n",
    "                          self.n_estimators,\n",
    "                          self.learning_rate)\n",
    "\n",
    "        eval_model(model, test_X, test_y)\n",
    "        save_model(model, self.model_file)\n",
    "\n",
    "    def predict(self, X, feature_names=None):\n",
    "        \"\"\"Predict using the model for given ndarray.\"\"\"\n",
    "        if not self.model:\n",
    "            self.model = joblib.load(self.model_file)\n",
    "        # Do any preprocessing\n",
    "        prediction = self.model.predict(data=X)\n",
    "        # Do any postprocessing\n",
    "        return prediction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Train an XGBoost model in a notebook"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0]\tvalidation_0-rmse:177565.34375\n",
      "Will train until validation_0-rmse hasn't improved in 40 rounds.\n",
      "[1]\tvalidation_0-rmse:161967.20312\n",
      "[2]\tvalidation_0-rmse:148001.89062\n",
      "[3]\tvalidation_0-rmse:135010.17188\n",
      "[4]\tvalidation_0-rmse:123514.68750\n",
      "[5]\tvalidation_0-rmse:113210.39062\n",
      "[6]\tvalidation_0-rmse:103914.61719\n",
      "[7]\tvalidation_0-rmse:95352.96094\n",
      "[8]\tvalidation_0-rmse:87878.77344\n",
      "[9]\tvalidation_0-rmse:81683.14062\n",
      "[10]\tvalidation_0-rmse:75828.78906\n",
      "[11]\tvalidation_0-rmse:70085.50000\n",
      "[12]\tvalidation_0-rmse:65076.06641\n",
      "[13]\tvalidation_0-rmse:60899.83203\n",
      "[14]\tvalidation_0-rmse:57354.22266\n",
      "[15]\tvalidation_0-rmse:54106.52734\n",
      "[16]\tvalidation_0-rmse:51402.42578\n",
      "[17]\tvalidation_0-rmse:48774.04688\n",
      "[18]\tvalidation_0-rmse:46360.19141\n",
      "[19]\tvalidation_0-rmse:44304.82031\n",
      "[20]\tvalidation_0-rmse:42618.65625\n",
      "[21]\tvalidation_0-rmse:41219.88672\n",
      "[22]\tvalidation_0-rmse:39885.14453\n",
      "[23]\tvalidation_0-rmse:38977.95703\n",
      "[24]\tvalidation_0-rmse:37856.47656\n",
      "[25]\tvalidation_0-rmse:36739.78125\n",
      "[26]\tvalidation_0-rmse:35847.46094\n",
      "[27]\tvalidation_0-rmse:35350.00781\n",
      "[28]\tvalidation_0-rmse:34857.17578\n",
      "[29]\tvalidation_0-rmse:34342.78516\n",
      "[30]\tvalidation_0-rmse:33752.83594\n",
      "[31]\tvalidation_0-rmse:33613.74609\n",
      "[32]\tvalidation_0-rmse:33468.86328\n",
      "[33]\tvalidation_0-rmse:33188.18359\n",
      "[34]\tvalidation_0-rmse:32825.68750\n",
      "[35]\tvalidation_0-rmse:32538.23242\n",
      "[36]\tvalidation_0-rmse:32235.01562\n",
      "[37]\tvalidation_0-rmse:31968.84180\n",
      "[38]\tvalidation_0-rmse:31724.04492\n",
      "[39]\tvalidation_0-rmse:31474.02539\n",
      "[40]\tvalidation_0-rmse:31259.08008\n",
      "[41]\tvalidation_0-rmse:31034.54492\n",
      "[42]\tvalidation_0-rmse:30885.66406\n",
      "[43]\tvalidation_0-rmse:30737.64844\n",
      "[44]\tvalidation_0-rmse:30562.84180\n",
      "[45]\tvalidation_0-rmse:30434.94922\n",
      "[46]\tvalidation_0-rmse:30314.07617\n",
      "[47]\tvalidation_0-rmse:30231.97070\n",
      "[48]\tvalidation_0-rmse:30100.18945\n",
      "[49]\tvalidation_0-rmse:29988.56445\n",
      "Best RMSE on eval: 29988.56 with 50 rounds\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "mean_absolute_error=17379.71\n",
      "Model export success: trained_ames_model.dat\n"
     ]
    }
   ],
   "source": [
    "model = HousingServe()\n",
    "model.train()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Create an S3 bucket to store pipeline data\n",
    "> Note: Be sure to change the HASH variable to random hash before running next cell\n",
    "\n",
    "> Note: if you use `us-east-1`, please use command `!aws s3 mb s3://{HASH}'-kubeflow-pipeline-data' --region $AWS_REGION --endpoint-url https://s3.us-east-1.amazonaws.com`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "make_bucket: kqfnqxqaadsogyxb3137689148272660-kubeflow-pipeline-data\n"
     ]
    }
   ],
   "source": [
    "import random, string\n",
    "HASH = ''.join([random.choice(string.ascii_lowercase) for n in range(16)] + [random.choice(string.digits) for n in range(16)])\n",
    "AWS_REGION = 'us-west-2'\n",
    "!aws s3 mb s3://{HASH}'-kubeflow-pipeline-data' --region $AWS_REGION"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Set up Kubeflow Fairing for training and predictions\n",
    "\n",
    "> Note: remember to change `kubeflow-pipeline-data` to your own s3 bucket."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.6/dist-packages/requests/__init__.py:91: RequestsDependencyWarning: urllib3 (1.26.5) or chardet (3.0.4) doesn't match a supported version!\n",
      "  RequestsDependencyWarning)\n"
     ]
    }
   ],
   "source": [
    "from kubeflow import fairing\n",
    "from kubeflow.fairing import TrainJob\n",
    "from kubeflow.fairing.backends import KubeflowAWSBackend\n",
    "\n",
    "\n",
    "from kubeflow import fairing\n",
    "\n",
    "FAIRING_BACKEND = 'KubeflowAWSBackend'\n",
    "\n",
    "AWS_ACCOUNT_ID = fairing.cloud.aws.guess_account_id()\n",
    "AWS_REGION = 'us-west-2'\n",
    "DOCKER_REGISTRY = '{}.dkr.ecr.{}.amazonaws.com'.format(AWS_ACCOUNT_ID, AWS_REGION)\n",
    "S3_BUCKET = f'{HASH}-kubeflow-pipeline-data'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "import importlib\n",
    "\n",
    "if FAIRING_BACKEND == 'KubeflowAWSBackend':\n",
    "    from kubeflow.fairing.builders.cluster.s3_context import S3ContextSource\n",
    "    BuildContext = S3ContextSource(\n",
    "        aws_account=AWS_ACCOUNT_ID, region=AWS_REGION,\n",
    "        bucket_name=S3_BUCKET\n",
    "    )\n",
    "\n",
    "BackendClass = getattr(importlib.import_module('kubeflow.fairing.backends'), FAIRING_BACKEND)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Train an XGBoost model remotely on Kubeflow\n",
    "Import the `TrainJob` and use the configured backend class. Kubeflow Fairing packages the `HousingServe` class, the training data, and the training job's software prerequisites as a Docker image. Then Kubeflow Fairing deploys and runs the training job on Kubeflow.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using default base docker image: registry.hub.docker.com/library/python:3.6.9\n",
      "Using builder: <class 'kubeflow.fairing.builders.cluster.cluster.ClusterBuilder'>\n",
      "Building the docker image.\n",
      "Building image using cluster builder.\n",
      "/usr/local/lib/python3.6/dist-packages/kubeflow/fairing/__init__.py already exists in Fairing context, skipping...\n",
      "Creating docker context: /tmp/fairing_context_3ms370ej\n",
      "/usr/local/lib/python3.6/dist-packages/kubeflow/fairing/__init__.py already exists in Fairing context, skipping...\n",
      "Not able to find aws credentials secret: aws-secret\n",
      "Waiting for fairing-builder-5t7x2-grbph to start...\n",
      "Waiting for fairing-builder-5t7x2-grbph to start...\n",
      "Waiting for fairing-builder-5t7x2-grbph to start...\n",
      "Pod started running True\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[36mINFO\u001b[0m[0000] Resolved base name registry.hub.docker.com/library/python:3.6.9 to registry.hub.docker.com/library/python:3.6.9\n",
      "\u001b[36mINFO\u001b[0m[0000] Resolved base name registry.hub.docker.com/library/python:3.6.9 to registry.hub.docker.com/library/python:3.6.9\n",
      "\u001b[36mINFO\u001b[0m[0000] Downloading base image registry.hub.docker.com/library/python:3.6.9\n",
      "\u001b[36mINFO\u001b[0m[0001] Error while retrieving image from cache: getting file info: stat /cache/sha256:036d4ab50fa49df89e746cf1b5369c88db46e8af2fbd08531788e7d920e9a491: no such file or directory\n",
      "\u001b[36mINFO\u001b[0m[0001] Downloading base image registry.hub.docker.com/library/python:3.6.9\n",
      "\u001b[36mINFO\u001b[0m[0001] Built cross stage deps: map[]\n",
      "\u001b[36mINFO\u001b[0m[0001] Downloading base image registry.hub.docker.com/library/python:3.6.9\n",
      "\u001b[36mINFO\u001b[0m[0002] Error while retrieving image from cache: getting file info: stat /cache/sha256:036d4ab50fa49df89e746cf1b5369c88db46e8af2fbd08531788e7d920e9a491: no such file or directory\n",
      "\u001b[36mINFO\u001b[0m[0002] Downloading base image registry.hub.docker.com/library/python:3.6.9\n",
      "\u001b[36mINFO\u001b[0m[0003] Unpacking rootfs as cmd COPY /app//requirements.txt /app/ requires it.\n",
      "\u001b[36mINFO\u001b[0m[0020] Taking snapshot of full filesystem...\n",
      "\u001b[36mINFO\u001b[0m[0026] WORKDIR /app/\n",
      "\u001b[36mINFO\u001b[0m[0026] cmd: workdir\n",
      "\u001b[36mINFO\u001b[0m[0026] Changed working directory to /app/\n",
      "\u001b[36mINFO\u001b[0m[0026] Creating directory /app/\n",
      "\u001b[36mINFO\u001b[0m[0026] Taking snapshot of files...\n",
      "\u001b[36mINFO\u001b[0m[0026] ENV FAIRING_RUNTIME 1\n",
      "\u001b[36mINFO\u001b[0m[0026] Using files from context: [/kaniko/buildcontext/app/requirements.txt]\n",
      "\u001b[36mINFO\u001b[0m[0026] COPY /app//requirements.txt /app/\n",
      "\u001b[36mINFO\u001b[0m[0026] Taking snapshot of files...\n",
      "\u001b[36mINFO\u001b[0m[0026] RUN if [ -e requirements.txt ];then pip install --no-cache -r requirements.txt; fi\n",
      "\u001b[36mINFO\u001b[0m[0026] cmd: /bin/sh\n",
      "\u001b[36mINFO\u001b[0m[0026] args: [-c if [ -e requirements.txt ];then pip install --no-cache -r requirements.txt; fi]\n",
      "Collecting pandas\n",
      "  Downloading https://files.pythonhosted.org/packages/c3/e2/00cacecafbab071c787019f00ad84ca3185952f6bb9bca9550ed83870d4d/pandas-1.1.5-cp36-cp36m-manylinux1_x86_64.whl (9.5MB)\n",
      "Collecting joblib\n",
      "  Downloading https://files.pythonhosted.org/packages/3e/d5/0163eb0cfa0b673aa4fe1cd3ea9d8a81ea0f32e50807b0c295871e4aab2e/joblib-1.1.0-py2.py3-none-any.whl (306kB)\n",
      "Collecting numpy\n",
      "  Downloading https://files.pythonhosted.org/packages/14/32/d3fa649ad7ec0b82737b92fefd3c4dd376b0bb23730715124569f38f3a08/numpy-1.19.5-cp36-cp36m-manylinux2010_x86_64.whl (14.8MB)\n",
      "Collecting xgboost\n",
      "  Downloading https://files.pythonhosted.org/packages/36/16/3a81d29dea691882bc95151879217a0c21c07740d0355dc90fe11836e10d/xgboost-1.5.1-py3-none-manylinux2014_x86_64.whl (173.5MB)\n",
      "Collecting scikit-learn>=0.21.0\n",
      "  Downloading https://files.pythonhosted.org/packages/d3/eb/d0e658465c029feb7083139d9ead51000742e88b1fb7f1504e19e1b4ce6e/scikit_learn-0.24.2-cp36-cp36m-manylinux2010_x86_64.whl (22.2MB)\n",
      "Collecting seldon-core\n",
      "  Downloading https://files.pythonhosted.org/packages/af/61/b35e01d6a68d0b197fd7f40a3cbdacac8281efac4b3c43996081e8fef3c3/seldon_core-1.12.0-py3-none-any.whl (133kB)\n",
      "Collecting tornado>=6.0.3\n",
      "  Downloading https://files.pythonhosted.org/packages/85/26/e710295dcb4aac62b08f22d07efc899574476db37532159a7f71713cdaf2/tornado-6.1-cp36-cp36m-manylinux2010_x86_64.whl (427kB)\n",
      "Collecting pytz>=2017.2\n",
      "  Downloading https://files.pythonhosted.org/packages/d3/e3/d9f046b5d1c94a3aeab15f1f867aa414f8ee9d196fae6865f1d6a0ee1a0b/pytz-2021.3-py2.py3-none-any.whl (503kB)\n",
      "Collecting python-dateutil>=2.7.3\n",
      "  Downloading https://files.pythonhosted.org/packages/36/7a/87837f39d0296e723bb9b62bbb257d0355c7f6128853c78955f57342a56d/python_dateutil-2.8.2-py2.py3-none-any.whl (247kB)\n",
      "Collecting scipy\n",
      "  Downloading https://files.pythonhosted.org/packages/c8/89/63171228d5ced148f5ced50305c89e8576ffc695a90b58fe5bb602b910c2/scipy-1.5.4-cp36-cp36m-manylinux1_x86_64.whl (25.9MB)\n",
      "Collecting threadpoolctl>=2.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/ff/fe/8aaca2a0db7fd80f0b2cf8a16a034d3eea8102d58ff9331d2aaf1f06766a/threadpoolctl-3.0.0-py3-none-any.whl\n",
      "Collecting grpcio-opentracing<1.2.0,>=1.1.4\n",
      "  Downloading https://files.pythonhosted.org/packages/db/82/2fcad380697c3dab25de76ee590bcab3eb9bbfb4add916044d7e83ec2b10/grpcio_opentracing-1.1.4-py3-none-any.whl\n",
      "Collecting Flask<2.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/e8/6d/994208daa354f68fd89a34a8bafbeaab26fda84e7af1e35bdaed02b667e6/Flask-1.1.4-py2.py3-none-any.whl (94kB)\n",
      "Collecting grpcio-reflection<1.35.0\n",
      "  Downloading https://files.pythonhosted.org/packages/16/4c/7b05077d9a3d7e60df742ab507578c501cc05861f4328ba4919fc799d72b/grpcio-reflection-1.34.1.tar.gz\n",
      "Collecting cryptography<3.5,>=3.4\n",
      "  Downloading https://files.pythonhosted.org/packages/96/07/4d23f8e34e56d8eeb2c37cd5924928a01c3dd756a1d99e470181bc57551e/cryptography-3.4.8-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.2MB)\n",
      "Collecting protobuf<4.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/95/35/ddae33187bb5c7b6a39cab5b59f07951f1fc3e5c72dd522b8d5f52d112c0/protobuf-3.19.1-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1MB)\n",
      "Collecting gunicorn<20.2.0,>=19.9.0\n",
      "  Downloading https://files.pythonhosted.org/packages/e4/dd/5b190393e6066286773a67dfcc2f9492058e9b57c4867a95f1ba5caf0a83/gunicorn-20.1.0-py3-none-any.whl (79kB)\n",
      "Collecting jsonschema<4.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/c5/8f/51e89ce52a085483359217bc72cdbf6e75ee595d5b1d4b5ade40c7e018b8/jsonschema-3.2.0-py2.py3-none-any.whl (56kB)\n",
      "Collecting Flask-cors<4.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/db/84/901e700de86604b1c4ef4b57110d4e947c218b9997adf5d38fa7da493bce/Flask_Cors-3.0.10-py2.py3-none-any.whl\n",
      "Collecting click<8.1,>=8.0.0a1\n",
      "  Downloading https://files.pythonhosted.org/packages/48/58/c8aa6a8e62cc75f39fee1092c45d6b6ba684122697d7ce7d53f64f98a129/click-8.0.3-py3-none-any.whl (97kB)\n",
      "Collecting jaeger-client<4.5.0,>=4.1.0\n",
      "  Downloading https://files.pythonhosted.org/packages/cb/45/aa60d2fe5e727e718c2cd2e9fa5b240f59c2c12cb9213b1c8709e4e115eb/jaeger-client-4.4.0.tar.gz (83kB)\n",
      "Collecting urllib3==1.26.5\n",
      "  Downloading https://files.pythonhosted.org/packages/0c/cd/1e2ec680ec7b09846dc6e605f5a7709dfb9d7128e51a026e7154e18a234e/urllib3-1.26.5-py2.py3-none-any.whl (138kB)\n",
      "Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.6/site-packages (from seldon-core->-r requirements.txt (line 6)) (41.6.0)\n",
      "Collecting flatbuffers<2.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/eb/26/712e578c5f14e26ae3314c39a1bdc4eb2ec2f4ddc89b708cf8e0a0d20423/flatbuffers-1.12-py2.py3-none-any.whl\n",
      "Collecting prometheus-client<0.9.0,>=0.7.1\n",
      "  Downloading https://files.pythonhosted.org/packages/3f/0e/554a265ffdc56e1494ef08e18f765b0cdec78797f510c58c45cf37abb4f4/prometheus_client-0.8.0-py2.py3-none-any.whl (53kB)\n",
      "Collecting Flask-OpenTracing<1.2.0,>=1.1.0\n",
      "  Downloading https://files.pythonhosted.org/packages/58/6c/6417701ba5ecc8854670c6db3207bcc3e5fbc96289a7cb18d5516d99a1c6/Flask-OpenTracing-1.1.0.tar.gz\n",
      "Collecting opentracing<2.5.0,>=2.2.0\n",
      "  Downloading https://files.pythonhosted.org/packages/51/28/2dba4e3efb64cc59d4311081a5ddad1dde20a19b69cd0f677cdb2f2c29a6/opentracing-2.4.0.tar.gz (46kB)\n",
      "Collecting PyYAML<5.5,>=5.4\n",
      "  Downloading https://files.pythonhosted.org/packages/7a/5b/bc0b5ab38247bba158504a410112b6c03f153c652734ece1849749e5f518/PyYAML-5.4.1-cp36-cp36m-manylinux1_x86_64.whl (640kB)\n",
      "Collecting requests<3.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/92/96/144f70b972a9c0eabbd4391ef93ccd49d0f2747f4f6a2a2738e99e5adc65/requests-2.26.0-py2.py3-none-any.whl (62kB)\n",
      "Collecting grpcio<2.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/39/99/3ec65ac96cdac250f4aeb2c052c98826755801890288cd4198dbdd6926dc/grpcio-1.43.0-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.1MB)\n",
      "Collecting six>=1.5\n",
      "  Downloading https://files.pythonhosted.org/packages/d9/5a/e7c31adbe875f2abbb91bd84cf2dc52d792b5a01506781dbcf25c91daf11/six-1.16.0-py2.py3-none-any.whl\n",
      "Collecting itsdangerous<2.0,>=0.24\n",
      "  Downloading https://files.pythonhosted.org/packages/76/ae/44b03b253d6fade317f32c24d100b3b35c2239807046a4c953c7b89fa49e/itsdangerous-1.1.0-py2.py3-none-any.whl\n",
      "Collecting Jinja2<3.0,>=2.10.1\n",
      "  Downloading https://files.pythonhosted.org/packages/7e/c2/1eece8c95ddbc9b1aeb64f5783a9e07a286de42191b7204d67b7496ddf35/Jinja2-2.11.3-py2.py3-none-any.whl (125kB)\n",
      "Collecting Werkzeug<2.0,>=0.15\n",
      "  Downloading https://files.pythonhosted.org/packages/cc/94/5f7079a0e00bd6863ef8f1da638721e9da21e5bacee597595b318f71d62e/Werkzeug-1.0.1-py2.py3-none-any.whl (298kB)\n",
      "Collecting cffi>=1.12\n",
      "  Downloading https://files.pythonhosted.org/packages/49/7b/449daf9cacfd7355cea1b4106d2be614315c29ac16567e01756167f6daab/cffi-1.15.0-cp36-cp36m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (405kB)\n",
      "Collecting attrs>=17.4.0\n",
      "  Downloading https://files.pythonhosted.org/packages/be/be/7abce643bfdf8ca01c48afa2ddf8308c2308b0c3b239a44e57d020afa0ef/attrs-21.4.0-py2.py3-none-any.whl (60kB)\n",
      "Collecting pyrsistent>=0.14.0\n",
      "  Downloading https://files.pythonhosted.org/packages/6c/19/1af501f6f388a40ede6d0185ba481bdb18ffc99deab0dd0d092b173bc0f4/pyrsistent-0.18.0-cp36-cp36m-manylinux1_x86_64.whl (117kB)\n",
      "Collecting importlib-metadata; python_version < \"3.8\"\n",
      "  Downloading https://files.pythonhosted.org/packages/a0/a1/b153a0a4caf7a7e3f15c2cd56c7702e2cf3d89b1b359d1f1c5e59d68f4ce/importlib_metadata-4.8.3-py3-none-any.whl\n",
      "Collecting threadloop<2,>=1\n",
      "  Downloading https://files.pythonhosted.org/packages/d3/1d/8398c1645b97dc008d3c658e04beda01ede3d90943d40c8d56863cf891bd/threadloop-1.0.2.tar.gz\n",
      "Collecting thrift\n",
      "  Downloading https://files.pythonhosted.org/packages/6e/97/a73a1a62f62375b21464fa45a0093ef0b653cb14f7599cffce35d51c9161/thrift-0.15.0.tar.gz (59kB)\n",
      "Collecting charset-normalizer~=2.0.0; python_version >= \"3\"\n",
      "  Downloading https://files.pythonhosted.org/packages/47/84/b06f6729fac8108c5fa3e13cde19b0b3de66ba5538c325496dbe39f5ff8e/charset_normalizer-2.0.9-py3-none-any.whl\n",
      "Collecting certifi>=2017.4.17\n",
      "  Downloading https://files.pythonhosted.org/packages/37/45/946c02767aabb873146011e665728b680884cd8fe70dde973c640e45b775/certifi-2021.10.8-py2.py3-none-any.whl (149kB)\n",
      "Collecting idna<4,>=2.5; python_version >= \"3\"\n",
      "  Downloading https://files.pythonhosted.org/packages/04/a2/d918dcd22354d8958fe113e1a3630137e0fc8b44859ade3063982eacd2a4/idna-3.3-py3-none-any.whl (61kB)\n",
      "Collecting MarkupSafe>=0.23\n",
      "  Downloading https://files.pythonhosted.org/packages/08/dc/a5ed54fcc61f75343663ee702cbf69831dcec9b1a952ae21cf3d1fbc56ba/MarkupSafe-2.0.1-cp36-cp36m-manylinux2010_x86_64.whl\n",
      "Collecting pycparser\n",
      "  Downloading https://files.pythonhosted.org/packages/62/d5/5f610ebe421e85889f2e55e33b7f9a6795bd982198517d912eb1c76e1a53/pycparser-2.21-py2.py3-none-any.whl (118kB)\n",
      "Collecting typing-extensions>=3.6.4; python_version < \"3.8\"\n",
      "  Downloading https://files.pythonhosted.org/packages/05/e4/baf0031e39cf545f0c9edd5b1a2ea12609b7fcba2d58e118b11753d68cf0/typing_extensions-4.0.1-py3-none-any.whl\n",
      "Collecting zipp>=0.5\n",
      "  Downloading https://files.pythonhosted.org/packages/bd/df/d4a4974a3e3957fd1c1fa3082366d7fff6e428ddb55f074bf64876f8e8ad/zipp-3.6.0-py3-none-any.whl\n",
      "Building wheels for collected packages: grpcio-reflection, jaeger-client, Flask-OpenTracing, opentracing, threadloop, thrift\n",
      "  Building wheel for grpcio-reflection (setup.py): started\n",
      "  Building wheel for grpcio-reflection (setup.py): finished with status 'done'\n",
      "  Created wheel for grpcio-reflection: filename=grpcio_reflection-1.34.1-cp36-none-any.whl size=14413 sha256=d8012d6f6fff93adc41aeae7dbcae673502e3277e88459d31bd62f2be3e66701\n",
      "  Stored in directory: /tmp/pip-ephem-wheel-cache-yqjybced/wheels/85/0e/79/919373e994613ef41ec9ffcd6ee9dd7952ab0dc2bbf963d209\n",
      "  Building wheel for jaeger-client (setup.py): started\n",
      "  Building wheel for jaeger-client (setup.py): finished with status 'done'\n",
      "  Created wheel for jaeger-client: filename=jaeger_client-4.4.0-cp36-none-any.whl size=64325 sha256=b4dbbd42471648a7284e0e260ea46c6fc7f7a71c0e6dfad2852da6d60a85ee4d\n",
      "  Stored in directory: /tmp/pip-ephem-wheel-cache-yqjybced/wheels/fa/d4/ec/1ad2c2de6b5cdd43a6557226e398c8c0ee8615569a3f9b291a\n",
      "  Building wheel for Flask-OpenTracing (setup.py): started\n",
      "  Building wheel for Flask-OpenTracing (setup.py): finished with status 'done'\n",
      "  Created wheel for Flask-OpenTracing: filename=Flask_OpenTracing-1.1.0-cp36-none-any.whl size=9071 sha256=614f7aa6896f7605a6caa903a1734a9a6f04aea72861c2ca022242d652f4aea5\n",
      "  Stored in directory: /tmp/pip-ephem-wheel-cache-yqjybced/wheels/7b/dc/25/3cf0b35c129232ee596c413f13d1d1f5a8e38c427266276dfd\n",
      "  Building wheel for opentracing (setup.py): started\n",
      "  Building wheel for opentracing (setup.py): finished with status 'done'\n",
      "  Created wheel for opentracing: filename=opentracing-2.4.0-cp36-none-any.whl size=51401 sha256=ed019c54caf261a500dade24dcd4e0dbc07b8fe4f5c773d5fa1619246955bc1f\n",
      "  Stored in directory: /tmp/pip-ephem-wheel-cache-yqjybced/wheels/27/b6/a0/d0309988a0dd5623c34469b151e4d7b0e6271b28a8bcccb440\n",
      "  Building wheel for threadloop (setup.py): started\n",
      "  Building wheel for threadloop (setup.py): finished with status 'done'\n",
      "  Created wheel for threadloop: filename=threadloop-1.0.2-cp36-none-any.whl size=3425 sha256=463543fddabe2d4932f7cf8be20292deecb1024eeb287e2bc79b9fb317ed69e0\n",
      "  Stored in directory: /tmp/pip-ephem-wheel-cache-yqjybced/wheels/d7/7a/30/d212623a4cd34f6cce400f8122b1b7af740d3440c68023d51f\n",
      "  Building wheel for thrift (setup.py): started\n",
      "  Building wheel for thrift (setup.py): finished with status 'done'\n",
      "  Created wheel for thrift: filename=thrift-0.15.0-cp36-cp36m-linux_x86_64.whl size=484190 sha256=d0aba596aca859c5fcf4b1b470bbc9513fc9cdab5a721b709219fee90566fc8c\n",
      "  Stored in directory: /tmp/pip-ephem-wheel-cache-yqjybced/wheels/ed/98/a6/f324d326f5ebc20cf4aa06f0a1cffc29f0c31ed34830db24be\n",
      "Successfully built grpcio-reflection jaeger-client Flask-OpenTracing opentracing threadloop thrift\n",
      "ERROR: flask 1.1.4 has requirement click<8.0,>=5.1, but you'll have click 8.0.3 which is incompatible.\n",
      "Installing collected packages: numpy, pytz, six, python-dateutil, pandas, joblib, scipy, xgboost, threadpoolctl, scikit-learn, grpcio, opentracing, grpcio-opentracing, typing-extensions, zipp, importlib-metadata, click, itsdangerous, MarkupSafe, Jinja2, Werkzeug, Flask, protobuf, grpcio-reflection, pycparser, cffi, cryptography, gunicorn, attrs, pyrsistent, jsonschema, Flask-cors, tornado, threadloop, thrift, jaeger-client, urllib3, flatbuffers, prometheus-client, Flask-OpenTracing, PyYAML, charset-normalizer, certifi, idna, requests, seldon-core\n",
      "Successfully installed Flask-1.1.4 Flask-OpenTracing-1.1.0 Flask-cors-3.0.10 Jinja2-2.11.3 MarkupSafe-2.0.1 PyYAML-5.4.1 Werkzeug-1.0.1 attrs-21.4.0 certifi-2021.10.8 cffi-1.15.0 charset-normalizer-2.0.9 click-8.0.3 cryptography-3.4.8 flatbuffers-1.12 grpcio-1.43.0 grpcio-opentracing-1.1.4 grpcio-reflection-1.34.1 gunicorn-20.1.0 idna-3.3 importlib-metadata-4.8.3 itsdangerous-1.1.0 jaeger-client-4.4.0 joblib-1.1.0 jsonschema-3.2.0 numpy-1.19.5 opentracing-2.4.0 pandas-1.1.5 prometheus-client-0.8.0 protobuf-3.19.1 pycparser-2.21 pyrsistent-0.18.0 python-dateutil-2.8.2 pytz-2021.3 requests-2.26.0 scikit-learn-0.24.2 scipy-1.5.4 seldon-core-1.12.0 six-1.16.0 threadloop-1.0.2 threadpoolctl-3.0.0 thrift-0.15.0 tornado-6.1 typing-extensions-4.0.1 urllib3-1.26.5 xgboost-1.5.1 zipp-3.6.0\n",
      "WARNING: You are using pip version 19.3.1; however, version 21.3.1 is available.\n",
      "You should consider upgrading via the 'pip install --upgrade pip' command.\n",
      "\u001b[36mINFO\u001b[0m[0069] Taking snapshot of full filesystem...\n",
      "\u001b[36mINFO\u001b[0m[0097] Using files from context: [/kaniko/buildcontext/app]\n",
      "\u001b[36mINFO\u001b[0m[0097] COPY /app/ /app/\n",
      "\u001b[36mINFO\u001b[0m[0097] Taking snapshot of files...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Not able to find aws credentials secret: aws-secret\n",
      "The job fairing-job-vqtsg launched.\n",
      "Waiting for fairing-job-vqtsg-t5v44 to start...\n",
      "Waiting for fairing-job-vqtsg-t5v44 to start...\n",
      "Waiting for fairing-job-vqtsg-t5v44 to start...\n",
      "Pod started running True\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0]\tvalidation_0-rmse:177565.34375\n",
      "[1]\tvalidation_0-rmse:161967.20312\n",
      "[2]\tvalidation_0-rmse:148001.89062\n",
      "[3]\tvalidation_0-rmse:135010.17188\n",
      "[4]\tvalidation_0-rmse:123514.68750\n",
      "[5]\tvalidation_0-rmse:113210.39062\n",
      "[6]\tvalidation_0-rmse:103914.61719\n",
      "[7]\tvalidation_0-rmse:95352.96094\n",
      "[8]\tvalidation_0-rmse:87878.77344\n",
      "[9]\tvalidation_0-rmse:81683.14062\n",
      "[10]\tvalidation_0-rmse:75828.78906\n",
      "[11]\tvalidation_0-rmse:70085.50000\n",
      "[12]\tvalidation_0-rmse:65076.06641\n",
      "[13]\tvalidation_0-rmse:60899.83203\n",
      "[14]\tvalidation_0-rmse:57354.22266\n",
      "[15]\tvalidation_0-rmse:54106.52734\n",
      "[16]\tvalidation_0-rmse:51402.42578\n",
      "[17]\tvalidation_0-rmse:48774.04688\n",
      "[18]\tvalidation_0-rmse:46360.19141\n",
      "[19]\tvalidation_0-rmse:44304.82031\n",
      "[20]\tvalidation_0-rmse:42618.65625\n",
      "[21]\tvalidation_0-rmse:41219.88672\n",
      "[22]\tvalidation_0-rmse:39885.14453\n",
      "[23]\tvalidation_0-rmse:38977.95703\n",
      "[24]\tvalidation_0-rmse:37856.47656\n",
      "[25]\tvalidation_0-rmse:36739.78125\n",
      "[26]\tvalidation_0-rmse:35847.46094\n",
      "[27]\tvalidation_0-rmse:35350.00781\n",
      "[28]\tvalidation_0-rmse:34857.17578\n",
      "[29]\tvalidation_0-rmse:34342.78516\n",
      "[30]\tvalidation_0-rmse:33752.83594\n",
      "[31]\tvalidation_0-rmse:33613.74609\n",
      "[32]\tvalidation_0-rmse:33468.86328\n",
      "[33]\tvalidation_0-rmse:33188.18359\n",
      "[34]\tvalidation_0-rmse:32825.68750\n",
      "[35]\tvalidation_0-rmse:32538.23242\n",
      "[36]\tvalidation_0-rmse:32235.01562\n",
      "[37]\tvalidation_0-rmse:31968.84180\n",
      "[38]\tvalidation_0-rmse:31724.04492\n",
      "[39]\tvalidation_0-rmse:31474.02539\n",
      "[40]\tvalidation_0-rmse:31259.08008\n",
      "[41]\tvalidation_0-rmse:31034.54492\n",
      "[42]\tvalidation_0-rmse:30885.66406\n",
      "[43]\tvalidation_0-rmse:30737.64844\n",
      "[44]\tvalidation_0-rmse:30562.84180\n",
      "[45]\tvalidation_0-rmse:30434.94922\n",
      "[46]\tvalidation_0-rmse:30314.07617\n",
      "[47]\tvalidation_0-rmse:30231.97070\n",
      "[48]\tvalidation_0-rmse:30100.18945\n",
      "[49]\tvalidation_0-rmse:29988.56445\n",
      "mean_absolute_error=17379.71\n",
      "Model export success: trained_ames_model.dat\n",
      "Best RMSE on eval: 29988.56 with 50 rounds\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Cleaning up job fairing-job-vqtsg...\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'fairing-job-vqtsg'"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from kubeflow.fairing import TrainJob\n",
    "train_job = TrainJob(HousingServe, input_files=['ames_dataset/train.csv', \"requirements.txt\"],\n",
    "                     docker_registry=DOCKER_REGISTRY,\n",
    "                     backend=BackendClass(build_context_source=BuildContext))\n",
    "train_job.submit()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Deploy the trained model to Kubeflow for predictions\n",
    "\n",
    "Import the `PredictionEndpoint` and use the configured backend class. Kubeflow Fairing packages the `HousingServe` class, the trained model, and the prediction endpoint's software prerequisites as a Docker image. Then Kubeflow Fairing deploys and runs the prediction endpoint on Kubeflow."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using default base docker image: registry.hub.docker.com/library/python:3.6.9\n",
      "Using builder: <class 'kubeflow.fairing.builders.cluster.cluster.ClusterBuilder'>\n",
      "Building the docker image.\n",
      "Building image using cluster builder.\n",
      "/usr/local/lib/python3.6/dist-packages/kubeflow/fairing/__init__.py already exists in Fairing context, skipping...\n",
      "Creating docker context: /tmp/fairing_context__88a3ijr\n",
      "/usr/local/lib/python3.6/dist-packages/kubeflow/fairing/__init__.py already exists in Fairing context, skipping...\n",
      "Not able to find aws credentials secret: aws-secret\n",
      "Waiting for fairing-builder-tzg8c-8shwc to start...\n",
      "Waiting for fairing-builder-tzg8c-8shwc to start...\n",
      "Waiting for fairing-builder-tzg8c-8shwc to start...\n",
      "Pod started running True\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[36mINFO\u001b[0m[0000] Resolved base name registry.hub.docker.com/library/python:3.6.9 to registry.hub.docker.com/library/python:3.6.9\n",
      "\u001b[36mINFO\u001b[0m[0000] Resolved base name registry.hub.docker.com/library/python:3.6.9 to registry.hub.docker.com/library/python:3.6.9\n",
      "\u001b[36mINFO\u001b[0m[0000] Downloading base image registry.hub.docker.com/library/python:3.6.9\n",
      "\u001b[36mINFO\u001b[0m[0001] Error while retrieving image from cache: getting file info: stat /cache/sha256:036d4ab50fa49df89e746cf1b5369c88db46e8af2fbd08531788e7d920e9a491: no such file or directory\n",
      "\u001b[36mINFO\u001b[0m[0001] Downloading base image registry.hub.docker.com/library/python:3.6.9\n",
      "\u001b[36mINFO\u001b[0m[0002] Built cross stage deps: map[]\n",
      "\u001b[36mINFO\u001b[0m[0002] Downloading base image registry.hub.docker.com/library/python:3.6.9\n",
      "\u001b[36mINFO\u001b[0m[0002] Error while retrieving image from cache: getting file info: stat /cache/sha256:036d4ab50fa49df89e746cf1b5369c88db46e8af2fbd08531788e7d920e9a491: no such file or directory\n",
      "\u001b[36mINFO\u001b[0m[0002] Downloading base image registry.hub.docker.com/library/python:3.6.9\n",
      "\u001b[36mINFO\u001b[0m[0003] Unpacking rootfs as cmd COPY /app//requirements.txt /app/ requires it.\n",
      "\u001b[36mINFO\u001b[0m[0021] Taking snapshot of full filesystem...\n",
      "\u001b[36mINFO\u001b[0m[0028] WORKDIR /app/\n",
      "\u001b[36mINFO\u001b[0m[0028] cmd: workdir\n",
      "\u001b[36mINFO\u001b[0m[0028] Changed working directory to /app/\n",
      "\u001b[36mINFO\u001b[0m[0028] Creating directory /app/\n",
      "\u001b[36mINFO\u001b[0m[0028] Taking snapshot of files...\n",
      "\u001b[36mINFO\u001b[0m[0028] ENV FAIRING_RUNTIME 1\n",
      "\u001b[36mINFO\u001b[0m[0028] Using files from context: [/kaniko/buildcontext/app/requirements.txt]\n",
      "\u001b[36mINFO\u001b[0m[0028] COPY /app//requirements.txt /app/\n",
      "\u001b[36mINFO\u001b[0m[0028] Taking snapshot of files...\n",
      "\u001b[36mINFO\u001b[0m[0028] RUN if [ -e requirements.txt ];then pip install --no-cache -r requirements.txt; fi\n",
      "\u001b[36mINFO\u001b[0m[0028] cmd: /bin/sh\n",
      "\u001b[36mINFO\u001b[0m[0028] args: [-c if [ -e requirements.txt ];then pip install --no-cache -r requirements.txt; fi]\n",
      "Collecting pandas\n",
      "  Downloading https://files.pythonhosted.org/packages/c3/e2/00cacecafbab071c787019f00ad84ca3185952f6bb9bca9550ed83870d4d/pandas-1.1.5-cp36-cp36m-manylinux1_x86_64.whl (9.5MB)\n",
      "Collecting joblib\n",
      "  Downloading https://files.pythonhosted.org/packages/3e/d5/0163eb0cfa0b673aa4fe1cd3ea9d8a81ea0f32e50807b0c295871e4aab2e/joblib-1.1.0-py2.py3-none-any.whl (306kB)\n",
      "Collecting numpy\n",
      "  Downloading https://files.pythonhosted.org/packages/14/32/d3fa649ad7ec0b82737b92fefd3c4dd376b0bb23730715124569f38f3a08/numpy-1.19.5-cp36-cp36m-manylinux2010_x86_64.whl (14.8MB)\n",
      "Collecting xgboost\n",
      "  Downloading https://files.pythonhosted.org/packages/36/16/3a81d29dea691882bc95151879217a0c21c07740d0355dc90fe11836e10d/xgboost-1.5.1-py3-none-manylinux2014_x86_64.whl (173.5MB)\n",
      "Collecting scikit-learn>=0.21.0\n",
      "  Downloading https://files.pythonhosted.org/packages/d3/eb/d0e658465c029feb7083139d9ead51000742e88b1fb7f1504e19e1b4ce6e/scikit_learn-0.24.2-cp36-cp36m-manylinux2010_x86_64.whl (22.2MB)\n",
      "Collecting seldon-core\n",
      "  Downloading https://files.pythonhosted.org/packages/af/61/b35e01d6a68d0b197fd7f40a3cbdacac8281efac4b3c43996081e8fef3c3/seldon_core-1.12.0-py3-none-any.whl (133kB)\n",
      "Collecting tornado>=6.0.3\n",
      "  Downloading https://files.pythonhosted.org/packages/85/26/e710295dcb4aac62b08f22d07efc899574476db37532159a7f71713cdaf2/tornado-6.1-cp36-cp36m-manylinux2010_x86_64.whl (427kB)\n",
      "Collecting pytz>=2017.2\n",
      "  Downloading https://files.pythonhosted.org/packages/d3/e3/d9f046b5d1c94a3aeab15f1f867aa414f8ee9d196fae6865f1d6a0ee1a0b/pytz-2021.3-py2.py3-none-any.whl (503kB)\n",
      "Collecting python-dateutil>=2.7.3\n",
      "  Downloading https://files.pythonhosted.org/packages/36/7a/87837f39d0296e723bb9b62bbb257d0355c7f6128853c78955f57342a56d/python_dateutil-2.8.2-py2.py3-none-any.whl (247kB)\n",
      "Collecting scipy\n",
      "  Downloading https://files.pythonhosted.org/packages/c8/89/63171228d5ced148f5ced50305c89e8576ffc695a90b58fe5bb602b910c2/scipy-1.5.4-cp36-cp36m-manylinux1_x86_64.whl (25.9MB)\n",
      "Collecting threadpoolctl>=2.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/ff/fe/8aaca2a0db7fd80f0b2cf8a16a034d3eea8102d58ff9331d2aaf1f06766a/threadpoolctl-3.0.0-py3-none-any.whl\n",
      "Collecting grpcio-reflection<1.35.0\n",
      "  Downloading https://files.pythonhosted.org/packages/16/4c/7b05077d9a3d7e60df742ab507578c501cc05861f4328ba4919fc799d72b/grpcio-reflection-1.34.1.tar.gz\n",
      "Collecting prometheus-client<0.9.0,>=0.7.1\n",
      "  Downloading https://files.pythonhosted.org/packages/3f/0e/554a265ffdc56e1494ef08e18f765b0cdec78797f510c58c45cf37abb4f4/prometheus_client-0.8.0-py2.py3-none-any.whl (53kB)\n",
      "Collecting cryptography<3.5,>=3.4\n",
      "  Downloading https://files.pythonhosted.org/packages/96/07/4d23f8e34e56d8eeb2c37cd5924928a01c3dd756a1d99e470181bc57551e/cryptography-3.4.8-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.2MB)\n",
      "Collecting Flask<2.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/e8/6d/994208daa354f68fd89a34a8bafbeaab26fda84e7af1e35bdaed02b667e6/Flask-1.1.4-py2.py3-none-any.whl (94kB)\n",
      "Collecting jsonschema<4.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/c5/8f/51e89ce52a085483359217bc72cdbf6e75ee595d5b1d4b5ade40c7e018b8/jsonschema-3.2.0-py2.py3-none-any.whl (56kB)\n",
      "Collecting Flask-OpenTracing<1.2.0,>=1.1.0\n",
      "  Downloading https://files.pythonhosted.org/packages/58/6c/6417701ba5ecc8854670c6db3207bcc3e5fbc96289a7cb18d5516d99a1c6/Flask-OpenTracing-1.1.0.tar.gz\n",
      "Collecting click<8.1,>=8.0.0a1\n",
      "  Downloading https://files.pythonhosted.org/packages/48/58/c8aa6a8e62cc75f39fee1092c45d6b6ba684122697d7ce7d53f64f98a129/click-8.0.3-py3-none-any.whl (97kB)\n",
      "Collecting gunicorn<20.2.0,>=19.9.0\n",
      "  Downloading https://files.pythonhosted.org/packages/e4/dd/5b190393e6066286773a67dfcc2f9492058e9b57c4867a95f1ba5caf0a83/gunicorn-20.1.0-py3-none-any.whl (79kB)\n",
      "Collecting flatbuffers<2.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/eb/26/712e578c5f14e26ae3314c39a1bdc4eb2ec2f4ddc89b708cf8e0a0d20423/flatbuffers-1.12-py2.py3-none-any.whl\n",
      "Collecting Flask-cors<4.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/db/84/901e700de86604b1c4ef4b57110d4e947c218b9997adf5d38fa7da493bce/Flask_Cors-3.0.10-py2.py3-none-any.whl\n",
      "Collecting grpcio<2.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/39/99/3ec65ac96cdac250f4aeb2c052c98826755801890288cd4198dbdd6926dc/grpcio-1.43.0-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.1MB)\n",
      "Collecting jaeger-client<4.5.0,>=4.1.0\n",
      "  Downloading https://files.pythonhosted.org/packages/cb/45/aa60d2fe5e727e718c2cd2e9fa5b240f59c2c12cb9213b1c8709e4e115eb/jaeger-client-4.4.0.tar.gz (83kB)\n",
      "Collecting requests<3.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/92/96/144f70b972a9c0eabbd4391ef93ccd49d0f2747f4f6a2a2738e99e5adc65/requests-2.26.0-py2.py3-none-any.whl (62kB)\n",
      "Collecting protobuf<4.0.0\n",
      "  Downloading https://files.pythonhosted.org/packages/95/35/ddae33187bb5c7b6a39cab5b59f07951f1fc3e5c72dd522b8d5f52d112c0/protobuf-3.19.1-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1MB)\n",
      "Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.6/site-packages (from seldon-core->-r requirements.txt (line 6)) (41.6.0)\n",
      "Collecting opentracing<2.5.0,>=2.2.0\n",
      "  Downloading https://files.pythonhosted.org/packages/51/28/2dba4e3efb64cc59d4311081a5ddad1dde20a19b69cd0f677cdb2f2c29a6/opentracing-2.4.0.tar.gz (46kB)\n",
      "Collecting urllib3==1.26.5\n",
      "  Downloading https://files.pythonhosted.org/packages/0c/cd/1e2ec680ec7b09846dc6e605f5a7709dfb9d7128e51a026e7154e18a234e/urllib3-1.26.5-py2.py3-none-any.whl (138kB)\n",
      "Collecting PyYAML<5.5,>=5.4\n",
      "  Downloading https://files.pythonhosted.org/packages/7a/5b/bc0b5ab38247bba158504a410112b6c03f153c652734ece1849749e5f518/PyYAML-5.4.1-cp36-cp36m-manylinux1_x86_64.whl (640kB)\n",
      "Collecting grpcio-opentracing<1.2.0,>=1.1.4\n",
      "  Downloading https://files.pythonhosted.org/packages/db/82/2fcad380697c3dab25de76ee590bcab3eb9bbfb4add916044d7e83ec2b10/grpcio_opentracing-1.1.4-py3-none-any.whl\n",
      "Collecting six>=1.5\n",
      "  Downloading https://files.pythonhosted.org/packages/d9/5a/e7c31adbe875f2abbb91bd84cf2dc52d792b5a01506781dbcf25c91daf11/six-1.16.0-py2.py3-none-any.whl\n",
      "Collecting cffi>=1.12\n",
      "  Downloading https://files.pythonhosted.org/packages/49/7b/449daf9cacfd7355cea1b4106d2be614315c29ac16567e01756167f6daab/cffi-1.15.0-cp36-cp36m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (405kB)\n",
      "Collecting Jinja2<3.0,>=2.10.1\n",
      "  Downloading https://files.pythonhosted.org/packages/7e/c2/1eece8c95ddbc9b1aeb64f5783a9e07a286de42191b7204d67b7496ddf35/Jinja2-2.11.3-py2.py3-none-any.whl (125kB)\n",
      "Collecting itsdangerous<2.0,>=0.24\n",
      "  Downloading https://files.pythonhosted.org/packages/76/ae/44b03b253d6fade317f32c24d100b3b35c2239807046a4c953c7b89fa49e/itsdangerous-1.1.0-py2.py3-none-any.whl\n",
      "Collecting Werkzeug<2.0,>=0.15\n",
      "  Downloading https://files.pythonhosted.org/packages/cc/94/5f7079a0e00bd6863ef8f1da638721e9da21e5bacee597595b318f71d62e/Werkzeug-1.0.1-py2.py3-none-any.whl (298kB)\n",
      "Collecting pyrsistent>=0.14.0\n",
      "  Downloading https://files.pythonhosted.org/packages/6c/19/1af501f6f388a40ede6d0185ba481bdb18ffc99deab0dd0d092b173bc0f4/pyrsistent-0.18.0-cp36-cp36m-manylinux1_x86_64.whl (117kB)\n",
      "Collecting attrs>=17.4.0\n",
      "  Downloading https://files.pythonhosted.org/packages/be/be/7abce643bfdf8ca01c48afa2ddf8308c2308b0c3b239a44e57d020afa0ef/attrs-21.4.0-py2.py3-none-any.whl (60kB)\n",
      "Collecting importlib-metadata; python_version < \"3.8\"\n",
      "  Downloading https://files.pythonhosted.org/packages/a0/a1/b153a0a4caf7a7e3f15c2cd56c7702e2cf3d89b1b359d1f1c5e59d68f4ce/importlib_metadata-4.8.3-py3-none-any.whl\n",
      "Collecting threadloop<2,>=1\n",
      "  Downloading https://files.pythonhosted.org/packages/d3/1d/8398c1645b97dc008d3c658e04beda01ede3d90943d40c8d56863cf891bd/threadloop-1.0.2.tar.gz\n",
      "Collecting thrift\n",
      "  Downloading https://files.pythonhosted.org/packages/6e/97/a73a1a62f62375b21464fa45a0093ef0b653cb14f7599cffce35d51c9161/thrift-0.15.0.tar.gz (59kB)\n",
      "Collecting certifi>=2017.4.17\n",
      "  Downloading https://files.pythonhosted.org/packages/37/45/946c02767aabb873146011e665728b680884cd8fe70dde973c640e45b775/certifi-2021.10.8-py2.py3-none-any.whl (149kB)\n",
      "Collecting charset-normalizer~=2.0.0; python_version >= \"3\"\n",
      "  Downloading https://files.pythonhosted.org/packages/47/84/b06f6729fac8108c5fa3e13cde19b0b3de66ba5538c325496dbe39f5ff8e/charset_normalizer-2.0.9-py3-none-any.whl\n",
      "Collecting idna<4,>=2.5; python_version >= \"3\"\n",
      "  Downloading https://files.pythonhosted.org/packages/04/a2/d918dcd22354d8958fe113e1a3630137e0fc8b44859ade3063982eacd2a4/idna-3.3-py3-none-any.whl (61kB)\n",
      "Collecting pycparser\n",
      "  Downloading https://files.pythonhosted.org/packages/62/d5/5f610ebe421e85889f2e55e33b7f9a6795bd982198517d912eb1c76e1a53/pycparser-2.21-py2.py3-none-any.whl (118kB)\n",
      "Collecting MarkupSafe>=0.23\n",
      "  Downloading https://files.pythonhosted.org/packages/08/dc/a5ed54fcc61f75343663ee702cbf69831dcec9b1a952ae21cf3d1fbc56ba/MarkupSafe-2.0.1-cp36-cp36m-manylinux2010_x86_64.whl\n",
      "Collecting zipp>=0.5\n",
      "  Downloading https://files.pythonhosted.org/packages/bd/df/d4a4974a3e3957fd1c1fa3082366d7fff6e428ddb55f074bf64876f8e8ad/zipp-3.6.0-py3-none-any.whl\n",
      "Collecting typing-extensions>=3.6.4; python_version < \"3.8\"\n",
      "  Downloading https://files.pythonhosted.org/packages/05/e4/baf0031e39cf545f0c9edd5b1a2ea12609b7fcba2d58e118b11753d68cf0/typing_extensions-4.0.1-py3-none-any.whl\n",
      "Building wheels for collected packages: grpcio-reflection, Flask-OpenTracing, jaeger-client, opentracing, threadloop, thrift\n",
      "  Building wheel for grpcio-reflection (setup.py): started\n",
      "  Building wheel for grpcio-reflection (setup.py): finished with status 'done'\n",
      "  Created wheel for grpcio-reflection: filename=grpcio_reflection-1.34.1-cp36-none-any.whl size=14413 sha256=b14bd361c3c73061a19581d2c522c09b83bdeadf8795637e1b5485cc1b93c8dd\n",
      "  Stored in directory: /tmp/pip-ephem-wheel-cache-ia6txjwc/wheels/85/0e/79/919373e994613ef41ec9ffcd6ee9dd7952ab0dc2bbf963d209\n",
      "  Building wheel for Flask-OpenTracing (setup.py): started\n",
      "  Building wheel for Flask-OpenTracing (setup.py): finished with status 'done'\n",
      "  Created wheel for Flask-OpenTracing: filename=Flask_OpenTracing-1.1.0-cp36-none-any.whl size=9071 sha256=6b479be3ec7bf22c358f58462e6e2aa55296eb4ba31bae2d1e48d4869dc006eb\n",
      "  Stored in directory: /tmp/pip-ephem-wheel-cache-ia6txjwc/wheels/7b/dc/25/3cf0b35c129232ee596c413f13d1d1f5a8e38c427266276dfd\n",
      "  Building wheel for jaeger-client (setup.py): started\n",
      "  Building wheel for jaeger-client (setup.py): finished with status 'done'\n",
      "  Created wheel for jaeger-client: filename=jaeger_client-4.4.0-cp36-none-any.whl size=64325 sha256=160e6f6bc49ccbad401d84c3e150303af788be90034995e41fd73570fb716f7e\n",
      "  Stored in directory: /tmp/pip-ephem-wheel-cache-ia6txjwc/wheels/fa/d4/ec/1ad2c2de6b5cdd43a6557226e398c8c0ee8615569a3f9b291a\n",
      "  Building wheel for opentracing (setup.py): started\n",
      "  Building wheel for opentracing (setup.py): finished with status 'done'\n",
      "  Created wheel for opentracing: filename=opentracing-2.4.0-cp36-none-any.whl size=51401 sha256=d3d900b64dd0f5903db9d5729be0c4b47b5a1039ea849835d21559aa5a03c850\n",
      "  Stored in directory: /tmp/pip-ephem-wheel-cache-ia6txjwc/wheels/27/b6/a0/d0309988a0dd5623c34469b151e4d7b0e6271b28a8bcccb440\n",
      "  Building wheel for threadloop (setup.py): started\n",
      "  Building wheel for threadloop (setup.py): finished with status 'done'\n",
      "  Created wheel for threadloop: filename=threadloop-1.0.2-cp36-none-any.whl size=3425 sha256=926025b7bcce470bf2938c276503ae7c74cab247cc714e587d7e7994bf9ba6aa\n",
      "  Stored in directory: /tmp/pip-ephem-wheel-cache-ia6txjwc/wheels/d7/7a/30/d212623a4cd34f6cce400f8122b1b7af740d3440c68023d51f\n",
      "  Building wheel for thrift (setup.py): started\n",
      "  Building wheel for thrift (setup.py): finished with status 'done'\n",
      "  Created wheel for thrift: filename=thrift-0.15.0-cp36-cp36m-linux_x86_64.whl size=484204 sha256=8922dd8c49d34e20d6e6b08a36236c0824d9c3117e4720f8ee49ddf86fff734c\n",
      "  Stored in directory: /tmp/pip-ephem-wheel-cache-ia6txjwc/wheels/ed/98/a6/f324d326f5ebc20cf4aa06f0a1cffc29f0c31ed34830db24be\n",
      "Successfully built grpcio-reflection Flask-OpenTracing jaeger-client opentracing threadloop thrift\n",
      "ERROR: flask 1.1.4 has requirement click<8.0,>=5.1, but you'll have click 8.0.3 which is incompatible.\n",
      "Installing collected packages: pytz, numpy, six, python-dateutil, pandas, joblib, scipy, xgboost, threadpoolctl, scikit-learn, protobuf, grpcio, grpcio-reflection, prometheus-client, pycparser, cffi, cryptography, MarkupSafe, Jinja2, itsdangerous, zipp, typing-extensions, importlib-metadata, click, Werkzeug, Flask, pyrsistent, attrs, jsonschema, opentracing, Flask-OpenTracing, gunicorn, flatbuffers, Flask-cors, tornado, threadloop, thrift, jaeger-client, certifi, charset-normalizer, idna, urllib3, requests, PyYAML, grpcio-opentracing, seldon-core\n",
      "Successfully installed Flask-1.1.4 Flask-OpenTracing-1.1.0 Flask-cors-3.0.10 Jinja2-2.11.3 MarkupSafe-2.0.1 PyYAML-5.4.1 Werkzeug-1.0.1 attrs-21.4.0 certifi-2021.10.8 cffi-1.15.0 charset-normalizer-2.0.9 click-8.0.3 cryptography-3.4.8 flatbuffers-1.12 grpcio-1.43.0 grpcio-opentracing-1.1.4 grpcio-reflection-1.34.1 gunicorn-20.1.0 idna-3.3 importlib-metadata-4.8.3 itsdangerous-1.1.0 jaeger-client-4.4.0 joblib-1.1.0 jsonschema-3.2.0 numpy-1.19.5 opentracing-2.4.0 pandas-1.1.5 prometheus-client-0.8.0 protobuf-3.19.1 pycparser-2.21 pyrsistent-0.18.0 python-dateutil-2.8.2 pytz-2021.3 requests-2.26.0 scikit-learn-0.24.2 scipy-1.5.4 seldon-core-1.12.0 six-1.16.0 threadloop-1.0.2 threadpoolctl-3.0.0 thrift-0.15.0 tornado-6.1 typing-extensions-4.0.1 urllib3-1.26.5 xgboost-1.5.1 zipp-3.6.0\n",
      "WARNING: You are using pip version 19.3.1; however, version 21.3.1 is available.\n",
      "You should consider upgrading via the 'pip install --upgrade pip' command.\n",
      "\u001b[36mINFO\u001b[0m[0072] Taking snapshot of full filesystem...\n",
      "\u001b[36mINFO\u001b[0m[0101] Using files from context: [/kaniko/buildcontext/app]\n",
      "\u001b[36mINFO\u001b[0m[0101] COPY /app/ /app/\n",
      "\u001b[36mINFO\u001b[0m[0101] Taking snapshot of files...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Deploying the endpoint.\n",
      "Cluster endpoint: http://fairing-service-n5l9b.kubeflow-user-example-com.svc.cluster.local:5000/predict\n",
      "Prediction endpoint: http://fairing-service-n5l9b.kubeflow-user-example-com.svc.cluster.local:5000/predict\n"
     ]
    }
   ],
   "source": [
    "from kubeflow.fairing import PredictionEndpoint\n",
    "endpoint = PredictionEndpoint(HousingServe, input_files=['trained_ames_model.dat', \"requirements.txt\"],\n",
    "                              docker_registry=DOCKER_REGISTRY,\n",
    "                              service_type='ClusterIP',\n",
    "                              backend=BackendClass(build_context_source=BuildContext))\n",
    "endpoint.create()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Call the prediction endpoint\n",
    "Create a test dataset, then call the endpoint on Kubeflow for predictions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/bin/sh: 1: nc: not found\n"
     ]
    }
   ],
   "source": [
    "# Wait service a while to be ready and replace `<endpoint>` with the output from last step.\n",
    "# Here's an example !nc -vz fairing-service-srwh2.anonymous.svc.cluster.local 5000\n",
    "\n",
    "!nc -vz fairing-service-n5l9b.kubeflow-user-example-com.svc.cluster.local 5000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1096.           20.           78.         ...    0.\n",
      "     3.         2007.        ]\n",
      " [1097.           70.           60.         ...    0.\n",
      "     3.         2007.        ]\n",
      " [1098.          120.           69.62831858 ...    0.\n",
      "    10.         2007.        ]\n",
      " ...\n",
      " [1458.           70.           66.         ... 2500.\n",
      "     5.         2010.        ]\n",
      " [1459.           20.           68.         ...    0.\n",
      "     4.         2010.        ]\n",
      " [1460.           20.           75.         ...    0.\n",
      "     6.         2008.        ]]\n"
     ]
    },
    {
     "ename": "JSONDecodeError",
     "evalue": "Expecting value: line 1 column 1 (char 0)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mJSONDecodeError\u001b[0m                           Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-18-ac8d35b2842a>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0mendpoint\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0murl\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'http://fairing-service-n5l9b.kubeflow-user-example-com.svc.cluster.local:5000/predict'\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 8\u001b[0;31m \u001b[0mendpoint\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpredict_nparray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtest_X\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/kubeflow/fairing/ml_tasks/tasks.py\u001b[0m in \u001b[0;36mpredict_nparray\u001b[0;34m(self, data, feature_names)\u001b[0m\n\u001b[1;32m    124\u001b[0m         \u001b[0mserialized_data\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mjson\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdumps\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    125\u001b[0m         \u001b[0mr\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mrequests\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpost\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0murl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m{\u001b[0m\u001b[0;34m'json'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mserialized_data\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 126\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mjson\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mloads\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mr\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtext\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    127\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    128\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mdelete\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/lib/python3.6/json/__init__.py\u001b[0m in \u001b[0;36mloads\u001b[0;34m(s, encoding, cls, object_hook, parse_float, parse_int, parse_constant, object_pairs_hook, **kw)\u001b[0m\n\u001b[1;32m    352\u001b[0m             \u001b[0mparse_int\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0mparse_float\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m \u001b[0;32mand\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    353\u001b[0m             parse_constant is None and object_pairs_hook is None and not kw):\n\u001b[0;32m--> 354\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0m_default_decoder\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdecode\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0ms\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    355\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mcls\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    356\u001b[0m         \u001b[0mcls\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mJSONDecoder\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/lib/python3.6/json/decoder.py\u001b[0m in \u001b[0;36mdecode\u001b[0;34m(self, s, _w)\u001b[0m\n\u001b[1;32m    337\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    338\u001b[0m         \"\"\"\n\u001b[0;32m--> 339\u001b[0;31m         \u001b[0mobj\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mend\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mraw_decode\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0ms\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0midx\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0m_w\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0ms\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    340\u001b[0m         \u001b[0mend\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_w\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0ms\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mend\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    341\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mend\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0ms\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/lib/python3.6/json/decoder.py\u001b[0m in \u001b[0;36mraw_decode\u001b[0;34m(self, s, idx)\u001b[0m\n\u001b[1;32m    355\u001b[0m             \u001b[0mobj\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mend\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mscan_once\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0ms\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0midx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    356\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mStopIteration\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 357\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mJSONDecodeError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Expecting value\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0ms\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0merr\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalue\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    358\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mobj\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mend\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mJSONDecodeError\u001b[0m: Expecting value: line 1 column 1 (char 0)"
     ]
    }
   ],
   "source": [
    "# Get sample data and query endpoint\n",
    "(train_X, train_y), (test_X, test_y) = read_input(\"ames_dataset/train.csv\")\n",
    "print(test_X)\n",
    "# PR https://github.com/kubeflow/fairing/pull/376\n",
    "# Add `:5000/predict` to mitigate the issue.\n",
    "endpoint.url='http://fairing-service-n5l9b.kubeflow-user-example-com.svc.cluster.local:5000/predict'\n",
    "\n",
    "endpoint.predict_nparray(test_X)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Clean up the prediction endpoint\n",
    "Delete the prediction endpoint created by this notebook."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "endpoint.delete()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Clean up S3 bucket and ECR Repository\n",
    "Delete S3 bucket and ECR Repository that was created for this exercise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!aws s3 rb s3://$S3_BUCKET --force\n",
    "!aws ecr delete-repository --repository-name fairing-job --region $AWS_REGION --force"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

```
