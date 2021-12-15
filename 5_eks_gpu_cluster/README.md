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
```
kubectl apply -f autoscaler
```

### 4. Install deviceplugin (for prod use helm dist)
The NVIDIA device plugin for Kubernetes exposes the number of GPUs on each nodes of your cluster. 
Once the plugin is installed, it’s possible to use nvidia/gpu Kubernetes resource on GPU nodes and for Kubernetes workloads.
Run this command to apply the Nvidia Kubernetes device plugin as a daemonset running only on AWS GPU-powered worker nodes,
 using tolerations and nodeAffinity
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

```
kubectl apply -f cudaexample.yaml
```

Our EKS cluster has 3 gpu node groups that can be scaled from zero to serve requested gpu tasks and go back to zero after work's done.
Cluster autoscaler uses k8s.io/cluster-autoscaler/enabled ASG tag for auto-discovery across multiple AZ zones (using --balance-similar-node-groups flag).
Cluster autoscaler scales workload based on requested resource, nvidia/gpu for GPU workload.
All GPU EKS node are protected with taint: nvidia.com/gpu: "true:NoSchedule" and in order to run a GPU workload on these nodes, the workload must define a corresponding tollerations, like following:
