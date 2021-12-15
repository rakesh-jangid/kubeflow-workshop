# GPU workloads on EKS


GPU workload on Kubernetes cluster requires a non-trivial configuration and comes with high cost tag (GPU instances are quite expensive).
In this example we will show how to use efectively EKS cluster with auto scaling, spot instance on GPU enabled instances 


Eksctl supports selecting GPU instance types for nodegroups. Simply supply a compatible instance type to the create command, or via the config file.
```
eksctl create cluster --node-type=p2.xlarge
```
The AMI resolvers (auto and auto-ssm) will see that you want to use a GPU instance type and they will select the correct EKS optimized accelerated AMI.

Eksctl will detect that an AMI with a GPU-enabled instance type has been selected and will install the NVIDIA Kubernetes device plugin automatically.

With eksctl automation this is super simple but not effective in terms of costs.

Let's dig deeper and see how we can design cluster for production machine learning workflows in details.


## Prepare EKS

### 1. Deploy eks cluster
Cluster consist of one node group for general workload and 3 gpu nodegroups each per one AZ scalable to 0.
Currently EKS managed  node groups does not scale to/from zero so we use unmanaged node groups.

Nodes are using EKS Optimized AMI image with GPU support.
GPU AMI comes with preinstalled NVIDIA drivers, nvidia-docker2 package and default nvidia-container-runtime.

cluster.yaml
```
---
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: noble-gpu-example
  region: us-west-2
  version: "1.19"

availabilityZones: ["us-west-2a", "us-west-2b", "us-west-2c"]
nodeGroups:
  - name: nodeg-1 # 100% spot
    minSize: 1
    maxSize: 4
    desiredCapacity: 1
    volumeSize: 20
    volumeType: gp2
    iam:
      withAddonPolicies:
        autoScaler: true
    ssh:
      enableSsm: true
    instancesDistribution:
      maxPrice: 1.0
      instanceTypes: ["m5.large", "t3.large"] # At least one instance type should be specified
      onDemandBaseCapacity: 0
      onDemandPercentageAboveBaseCapacity: 0
      spotInstancePools: 2
  # spot GPU NG - west-2c AZ, scale from 0
  - name: gpu-spot-ng-a
    ami: auto
    instanceType: mixed
    desiredCapacity: 0
    minSize: 0
    maxSize: 3
    volumeSize: 20
    volumeType: gp2
    iam:
      withAddonPolicies:
        autoScaler: true
    ssh:
      enableSsm: true
    instancesDistribution:
      onDemandPercentageAboveBaseCapacity: 0
      instanceTypes:
        - p2.xlarge
        - p3.2xlarge
      spotInstancePools: 2
    tags:
      k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu=true
      k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu: 'true'
      k8s.io/cluster-autoscaler/enabled: 'true'
    labels:
      lifecycle: Ec2Spot
      nvidia.com/gpu: 'true'
      k8s.amazonaws.com/accelerator: nvidia-tesla
    taints:
      spotInstance: "true:PreferNoSchedule"
      nvidia.com/gpu: "true:NoSchedule"
    privateNetworking: true
    availabilityZones: ["us-west-2a"]

  # spot GPU NG - west-2b AZ, scaled from 0
  - name: gpu-spot-ng-b
    ami: auto
    instanceType: mixed
    desiredCapacity: 0
    minSize: 0
    maxSize: 3
    volumeSize: 20
    volumeType: gp2
    iam:
      withAddonPolicies:
        autoScaler: true
    ssh:
      enableSsm: true
    instancesDistribution:
      onDemandPercentageAboveBaseCapacity: 0
      instanceTypes:
        - p2.xlarge
        - p3.2xlarge
      spotInstancePools: 2
    tags:
      k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu=true
      k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu: 'true'
      k8s.io/cluster-autoscaler/enabled: 'true'
    labels:
      lifecycle: Ec2Spot
      nvidia.com/gpu: 'true'
      k8s.amazonaws.com/accelerator: nvidia-tesla
    taints:
      spotInstance: "true:PreferNoSchedule"
      nvidia.com/gpu: "true:NoSchedule"
    privateNetworking: true
    availabilityZones: ["us-west-2b"]

  # spot GPU NG - west-2c AZ, scale from 0
  - name: gpu-spot-ng-c
    ami: auto
    instanceType: mixed
    desiredCapacity: 0
    minSize: 0
    maxSize: 3
    volumeSize: 20
    volumeType: gp2
    iam:
      withAddonPolicies:
        autoScaler: true
    ssh:
      enableSsm: true
    instancesDistribution:
      onDemandPercentageAboveBaseCapacity: 0
      instanceTypes:
        - p2.xlarge
        - p3.2xlarge
      spotInstancePools: 2
    tags:
      k8s.io/cluster-autoscaler/node-template/taint/dedicated: nvidia.com/gpu=true
      k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu: 'true'
      k8s.io/cluster-autoscaler/enabled: 'true'
    labels:
      lifecycle: Ec2Spot
      nvidia.com/gpu: 'true'
      k8s.amazonaws.com/accelerator: nvidia-tesla
    taints:
      spotInstance: "true:PreferNoSchedule"
      nvidia.com/gpu: "true:NoSchedule"
    privateNetworking: true
    availabilityZones: ["us-west-2c"]

```


```
eksctl create cluster -f cluster.yaml
```
### 2. Update kubeconfig
Update kubeconfig for kubectl

```
aws eks update-kubeconfig --name noble-gpu-example

```

### 3. Deploy autoscaler
Autoscaler will add/remove nodes when needed. 
It is very important to run gpu nodes only when needed.
GPU workload are very pricy.

autoscaler.yaml
```
---
apiVersion: v1
kind: ServiceAccount
metadata:
  labels:
    k8s-addon: cluster-autoscaler.addons.k8s.io
    k8s-app: cluster-autoscaler
  name: cluster-autoscaler
  namespace: kube-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cluster-autoscaler
  labels:
    k8s-addon: cluster-autoscaler.addons.k8s.io
    k8s-app: cluster-autoscaler
rules:
  - apiGroups: [""]
    resources: ["events", "endpoints"]
    verbs: ["create", "patch"]
  - apiGroups: [""]
    resources: ["pods/eviction"]
    verbs: ["create"]
  - apiGroups: [""]
    resources: ["pods/status"]
    verbs: ["update"]
  - apiGroups: [""]
    resources: ["endpoints"]
    resourceNames: ["cluster-autoscaler"]
    verbs: ["get", "update"]
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["watch", "list", "get", "update"]
  - apiGroups: [""]
    resources:
      - "pods"
      - "services"
      - "replicationcontrollers"
      - "persistentvolumeclaims"
      - "persistentvolumes"
    verbs: ["watch", "list", "get"]
  - apiGroups: ["extensions"]
    resources: ["replicasets", "daemonsets"]
    verbs: ["watch", "list", "get"]
  - apiGroups: ["policy"]
    resources: ["poddisruptionbudgets"]
    verbs: ["watch", "list"]
  - apiGroups: ["apps"]
    resources: ["statefulsets", "replicasets", "daemonsets"]
    verbs: ["watch", "list", "get"]
  - apiGroups: ["storage.k8s.io"]
    resources: ["storageclasses", "csinodes"]
    verbs: ["watch", "list", "get"]
  - apiGroups: ["batch", "extensions"]
    resources: ["jobs"]
    verbs: ["get", "list", "watch", "patch"]
  - apiGroups: ["coordination.k8s.io"]
    resources: ["leases"]
    verbs: ["create"]
  - apiGroups: ["coordination.k8s.io"]
    resourceNames: ["cluster-autoscaler"]
    resources: ["leases"]
    verbs: ["get", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    k8s-addon: cluster-autoscaler.addons.k8s.io
    k8s-app: cluster-autoscaler
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["create","list","watch"]
  - apiGroups: [""]
    resources: ["configmaps"]
    resourceNames: ["cluster-autoscaler-status", "cluster-autoscaler-priority-expander"]
    verbs: ["delete", "get", "update", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: cluster-autoscaler
  labels:
    k8s-addon: cluster-autoscaler.addons.k8s.io
    k8s-app: cluster-autoscaler
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-autoscaler
subjects:
  - kind: ServiceAccount
    name: cluster-autoscaler
    namespace: kube-system

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    k8s-addon: cluster-autoscaler.addons.k8s.io
    k8s-app: cluster-autoscaler
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: cluster-autoscaler
subjects:
  - kind: ServiceAccount
    name: cluster-autoscaler
    namespace: kube-system

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    app: cluster-autoscaler
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
      annotations:
        prometheus.io/scrape: 'true'
        prometheus.io/port: '8085'
        cluster-autoscaler.kubernetes.io/safe-to-evict: 'false'
    spec:
      serviceAccountName: cluster-autoscaler
      containers:
        - image: us.gcr.io/k8s-artifacts-prod/autoscaling/cluster-autoscaler:v1.19.1
          name: cluster-autoscaler
          resources:
            limits:
              cpu: 100m
              memory: 300Mi
            requests:
              cpu: 100m
              memory: 300Mi
          command:
            - ./cluster-autoscaler
            - --v=4
            - --stderrthreshold=info
            - --cloud-provider=aws
            - --skip-nodes-with-local-storage=false
            - --expander=random
            - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/noble-gpu-example
            - --balance-similar-node-groups
            - --skip-nodes-with-system-pods=false
            - --aws-use-static-instance-list=true
          volumeMounts:
            - name: ssl-certs
              mountPath: /etc/ssl/certs/ca-certificates.crt
              readOnly: true
          imagePullPolicy: "Always"
      volumes:
        - name: ssl-certs
          hostPath:
            path: "/etc/ssl/certs/ca-bundle.crt"

```

```
kubectl apply -f autoscaler.yaml
```

### 4. Install deviceplugin (for prod use helm dist)
The NVIDIA device plugin for Kubernetes exposes the number of GPUs on each nodes of your cluster. 
Once the plugin is installed, it’s possible to use nvidia/gpu Kubernetes resource on GPU nodes and for Kubernetes workloads.
Run this command to apply the Nvidia Kubernetes device plugin as a daemonset running only on AWS GPU-powered worker nodes,
 using tolerations and nodeAffinity
 
nvidiaplugin.yaml
```
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin-ds
  updateStrategy:
    type: RollingUpdate
  template:
    metadata:
      # This annotation is deprecated. Kept here for backward compatibility
      # See https://kubernetes.io/docs/tasks/administer-cluster/guaranteed-scheduling-critical-addon-pods/
      annotations:
        scheduler.alpha.kubernetes.io/critical-pod: ""
      labels:
        name: nvidia-device-plugin-ds
    spec:
      tolerations:
      # This toleration is deprecated. Kept here for backward compatibility
      # See https://kubernetes.io/docs/tasks/administer-cluster/guaranteed-scheduling-critical-addon-pods/
      - key: CriticalAddonsOnly
        operator: Exists
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      # Mark this pod as a critical add-on; when enabled, the critical add-on
      # scheduler reserves resources for critical add-on pods so that they can
      # be rescheduled after a failure.
      # See https://kubernetes.io/docs/tasks/administer-cluster/guaranteed-scheduling-critical-addon-pods/
      priorityClassName: "system-node-critical"
      containers:
      - image: nvcr.io/nvidia/k8s-device-plugin:v0.10.0
        name: nvidia-device-plugin-ctr
        args: ["--fail-on-init-error=false"]
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop: ["ALL"]
        volumeMounts:
          - name: device-plugin
            mountPath: /var/lib/kubelet/device-plugins
      volumes:
        - name: device-plugin
          hostPath:
            path: /var/lib/kubelet/device-plugins
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: beta.kubernetes.io/instance-type
                operator: In
                values:
                - p3.2xlarge
                - p3.8xlarge
                - p3.16xlarge
                - p3dn.24xlarge
                - p2.xlarge
                - p2.8xlarge
                - p2.16xlarge

```


```
kubectl apply -f nvidiplugin.yaml
```

## Example workloads
4. Run gpu pod

Kubernetes taints allow a node to reject a set of pods. Taints and tolerations work together to ensure that pods are not scheduled onto inappropriate nodes.
One or more taints are applied to a node; this marks that the node should not accept any pods that do not tolerate the taints. 
Tolerations are applied to pods, and allow (but do not require) the pods to schedule onto nodes with matching taints.
See Kubernetes Taints and Tolerations documentation for more details.
To run a GPU workload on GPU-powered Spot instance nodes, with nvidia.com/gpu: "true:NoSchedule" taint, the workload must include both matching tolerations and nodeSelector configurations.
Kubernetes deployment with 10 pod replicas with nvidia/gpu: 1 limit:

cuda_example.yaml
```
apiVersion: v1
kind: Pod
metadata:
  name: cuda-vector-add-no-r
spec:
  restartPolicy: OnFailure
  # affinity:
  #   nodeAffinity:
  #     requiredDuringSchedulingIgnoredDuringExecution:
  #       nodeSelectorTerms:
  #       - matchExpressions:
  #         - key: k8s.amazonaws.com/accelerator
  #           operator: In
  #           values:
  #           - nvidia-tesla-k80
  #           - nvidia-tesla-v100
  nodeSelector:
    nvidia.com/gpu: "true"
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
  - key: "spotInstance"
    operator: "Exists"
    effect: "NoSchedule"
  containers:
    - name: cuda-vector-add
      # https://github.com/kubernetes/kubernetes/blob/v1.7.11/test/images/nvidia-cuda/Dockerfile
      image: "k8s.gcr.io/cuda-vector-add:v0.1"
      resources:
        limits:
          nvidia.com/gpu: 1 # requesting 1 GPU
```


```
kubectl apply -f cuda_example.yaml
```

Our EKS cluster has 3 gpu node groups that can be scaled from zero to serve requested gpu tasks and go back to zero after work's done.
Cluster autoscaler uses k8s.io/cluster-autoscaler/enabled ASG tag for auto-discovery across multiple AZ zones (using --balance-similar-node-groups flag).
Cluster autoscaler scales workload based on requested resource, nvidia/gpu for GPU workload.
All GPU EKS node are protected with taint: nvidia.com/gpu: "true:NoSchedule" and in order to run a GPU workload on these nodes, the workload must define a corresponding tollerations, like following:
