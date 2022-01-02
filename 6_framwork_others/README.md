# User provisioning

User provisioning can be integrated with CI/CD system and can track external user database/directory group/k/v store etc.
Then it is recommended to disable UI user registration flow and deploy user namespace on demand. Namespace can be also integrated with IAM role for service account, providing granular access from the namespace to the AWS resources(like S3, ECR, SageMaker, etc).
Example user profile CRD:
profile.yaml
```
apiVersion: kubeflow.org/v1
kind: Profile
metadata:
  name: useralfa
spec:
  owner:
    kind: User
    name: useralfa@domain.com
```
After running ```kubectl apply -f profile.yaml``` Kubeflow will create dedicated namespace for the user.

## Sharing namespace with other namespace
You can create role binding giving access from 'useralfa' namespace to 'othernamespace'
binding.yaml
```
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  annotations:
    role: edit
    user: useralfa@domain.com
  name: user-useralfa-domain-com-clusterrole-edit
  namespace: othernamespace 
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: kubeflow-edit
subjects:
- apiGroup: rbac.authorization.k8s.io
  kind: User
  name: useralfa@domain.com
```
You have to add also additional istio authorization policy as follows:
policy.yaml
```
apiVersion: v1
items:
- apiVersion: security.istio.io/v1beta1
  kind: AuthorizationPolicy
  metadata:
    name: ml-pipeline-visualizationserver
    namespace: othernamespace
  spec:
    rules:
    - from:
      - source:
          principals:
          - cluster.local/ns/kubeflow/sa/ml-pipeline
    selector:
      matchLabels:
        app: ml-pipeline-visualizationserver
- apiVersion: security.istio.io/v1beta1
  kind: AuthorizationPolicy
  metadata:
    annotations:
      role: admin
      user: useralfa@domain.com
    name: ns-owner-access-istio-useralfa
    namespace: othernamespace
  spec:
    rules:
    - when:
      - key: request.headers[kubeflow-userid]
        values:
        - useralfa@domain.com
    - when:
      - key: source.namespace
        values:
        - useralfa
kind: List
```

In order to grant access to IAM role you can use Iam Role fro Service Account.
You have to create first IAM role with assume policy matching OIDC provider to the service account:
https://docs.aws.amazon.com/eks/latest/userguide/create-service-account-iam-policy-and-role.html

Then edit ServiceAccount and add annotation to user IRSA
```
kubectl edit ServiceAccount default-editor -n useralfa
```
and add metadata annotation
```
apiVersion: v1
kind: ServiceAccount
metadata:
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::<AWS_ACCOUNT_ID>:role/useralfa_role_name
```

RDS as a DB
## Deploy RDS using CloudFormation as described here:
https://www.kubeflow.org/docs/distributions/aws/customizing-aws/rds/

## Edit metadata mysql configuration
Metadata mysql configuration is passed to the metadata-grpc-deployment as a env vars taken from pipeline-install-config configmap and mysql-secret secret.
Update those values and restart deployment.
```
kubectl edit deployment metadata-grpc-deployment -n kubeflow
```

```
...
        env:
        - name: DBCONFIG_USER
          valueFrom:
            secretKeyRef:
              key: username
              name: mysql-secret
        - name: DBCONFIG_PASSWORD
          valueFrom:
            secretKeyRef:
              key: password
              name: mysql-secret
        - name: MYSQL_DATABASE
          valueFrom:
            configMapKeyRef:
              key: mlmdDb
              name: pipeline-install-config
        - name: MYSQL_HOST
          valueFrom:
            configMapKeyRef:
              key: dbHost
              name: pipeline-install-config
        - name: MYSQL_PORT
          valueFrom:
            configMapKeyRef:
              key: dbPort
              name: pipeline-install-config

```
```
kubectl edit configmap pipeline-install-config -n kubeflow
```
```
apiVersion: v1
data:
  appName: pipeline
  appVersion: 1.5.0
  autoUpdatePipelineDefaultVersion: "true"
  bucketName: mlpipeline
  cacheDb: cachedb
  cacheImage: gcr.io/google-containers/busybox
  cronScheduleTimezone: UTC
  dbHost: mysql
  dbPort: "3306"
  mlmdDb: metadb
  pipelineDb: mlpipeline
kind: ConfigMap

```

```
kubectl edit secret mysql-secret -n kubeflow
```
Secrets are stored as a base64 strings
```
apiVersion: v1
data:
  password: ""
  username: cm9vdA==
kind: Secret
```

## Edit pipelines rds connection
Kubeflow pipeline uses the same config map and secret as metadata-grpc configuration
```
kubectl edit deployment ml-pipeline -n kubeflow
```
``` 
....
        - name: DBCONFIG_USER
          valueFrom:
            secretKeyRef:
              key: username
              name: mysql-secret
        - name: DBCONFIG_PASSWORD
          valueFrom:
            secretKeyRef:
              key: password
              name: mysql-secret
        - name: DBCONFIG_DBNAME
          valueFrom:
            configMapKeyRef:
              key: pipelineDb
              name: pipeline-install-config
        - name: DBCONFIG_HOST
          valueFrom:
            configMapKeyRef:
              key: dbHost
              name: pipeline-install-config
        - name: DBCONFIG_PORT
          valueFrom:
            configMapKeyRef:
              key: dbPort
              name: pipeline-install-config
```

# S3 instead of minio
Currently Kubeflow does not work with IAM Roles for Service accounts for accessing S3. User have to pass AWS static credentials to access S3.
```
kubectl edit deployment ml-pipeline -n kubeflow
```
```
....
        - name: OBJECTSTORECONFIG_ACCESSKEY
          valueFrom:
            secretKeyRef:
              key: accesskey
              name: mlpipeline-minio-artifact
        - name: OBJECTSTORECONFIG_SECRETACCESSKEY
          valueFrom:
            secretKeyRef:
              key: secretkey
              name: mlpipeline-minio-artifact
....
```

Add S3 access credentials as a base64 encoded strings here:
```
kubectl edit secret mlpipeline-minio-artifact -n kubeflow
```
Change S3 bucket name by editing pipeline-install-config configmap value 'bucketName'
```
kubectl edit configmap pipeline-install-config -n kubeflow
```

Patch ml pipelines deployments with the following changes:
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-pipeline-ui
spec:
  template:
    metadata:
      labels:
        app: ml-pipeline-ui
    spec:
      volumes:
        - name: config-volume
          configMap:
            name: ml-pipeline-ui-configmap
      containers:
        - name: ml-pipeline-ui
          env:
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: mlpipeline-minio-artifact
                  key: accesskey
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: mlpipeline-minio-artifact
                  key: secretkey

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-pipeline
spec:
  template:
    metadata:
      labels:
        app: ml-pipeline
    spec:
      containers:
        - env:
            - name: OBJECTSTORECONFIG_SECURE
              value: "true"
            - name: OBJECTSTORECONFIG_BUCKETNAME
              valueFrom:
                configMapKeyRef:
                  name: pipeline-install-config
                  key: bucketName
            - name: OBJECTSTORECONFIG_HOST
              valueFrom:
                configMapKeyRef:
                  name: pipeline-install-config
                  key: minioServiceHost
            - name: OBJECTSTORECONFIG_REGION
              valueFrom:
                configMapKeyRef:
                  name: pipeline-install-config
                  key: minioServiceRegion
            - name: OBJECTSTORECONFIG_PORT
              value: ""
          name: ml-pipeline-api-server
```
There is already prepared kustomization for S3 and RDS for Pipelines in the manifest repository.
See: manifests-1.3-branch/apps/pipeline/upstream/env/aws for details

# Notebooks, pipelines on dedicated spot instances
## Add configuration to the Notebooks
Notebook UI can be customized per user needs. You can add your custom ecr image, add affinityConfig and tollerationGroup to run Notebook on GPU instance node or Spot.
Edit configmap. Create kustomization.
```
kubectl get configmap -n kubeflow | grep jupyter-web-app-parameters
```

# Autoscaling
See: https://github.com/malawskim/kubeflow-workshop/tree/main/5_A_autoscaling_eks_cluster

# Monitoring
See: https://github.com/malawskim/kubeflow-workshop/tree/main/5_B_monitoring_eks_cluster

# EFS
See: https://www.kubeflow.org/docs/distributions/aws/customizing-aws/storage/
and: https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html

# SSM agent as deamon set
https://github.com/aws-samples/ssm-agent-daemonset-installer/blob/main/setup.yaml

# Cognito
https://github.com/awslabs/kubeflow-manifests/tree/v1.3-branch/distributions/aws/examples

# Terraform 
See: https://github.com/malawskim/kubeflow-workshop/tree/main/6_terraform_for_kubeflow
