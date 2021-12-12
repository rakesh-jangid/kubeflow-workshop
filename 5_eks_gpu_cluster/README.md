TODO:
gpu cluster one node per az scalable to zero

EKS with GPU


GPU workload on Kubernetes cluster requires a non-trivial configuration and comes with high cost tag (GPU instances are quite expensive).
In this example we will show how to use efectively EKS cluster with auto scaling, spot instance on GPU enabled instances 


Eksctl supports selecting GPU instance types for nodegroups. Simply supply a compatible instance type to the create command, or via the config file.
```
eksctl create cluster --node-type=p2.xlarge
```
The AMI resolvers (auto and auto-ssm) will see that you want to use a GPU instance type and they will select the correct EKS optimized accelerated AMI.

Eksctl will detect that an AMI with a GPU-enabled instance type has been selected and will install the NVIDIA Kubernetes device plugin automatically.

With eksctl automation this is super simple but not effective in terms of costs.
