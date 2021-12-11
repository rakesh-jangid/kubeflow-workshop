# Autoscaling EKS Cluster
## Prepare EKS
cluster.yaml
```
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: dev-cluster
  region: us-west-2

nodeGroups:
  - name: scale-west2b
    instanceType: t2.small
    desiredCapacity: 1
    maxSize: 3
    availabilityZones: ["us-west-2b"]
    iam:
      withAddonPolicies:
        autoScaler: true
    labels:
      nodegroup-type: stateful-west1c
      instance-type: onDemand
    ssh:
      enableSsm: true
  - name: scale-west2c
    instanceType: t2.small
    desiredCapacity: 1
    maxSize: 3
    availabilityZones: ["us-west-2c"]
    iam:
      withAddonPolicies:
        autoScaler: true
    labels:
      nodegroup-type: stateful-west2c
      instance-type: onDemand
    ssh:
      enableSsm: true
  - name: scale-spot-2d
    desiredCapacity: 1
    maxSize: 3
    instancesDistribution:
      instanceTypes: ["t2.small", "t3.small"]
      onDemandBaseCapacity: 0
      onDemandPercentageAboveBaseCapacity: 0
    availabilityZones: ["us-west-2d"]
    iam:
      withAddonPolicies:
        autoScaler: true
    labels:
      nodegroup-type: stateless-workload
      instance-type: spot
    ssh: 
      enableSsm: true
```

Deploy cluster or update nodegroups if it already exist
```
ekscrl create cluster -f cluster.yaml
```

update kube config
```
aws eks update-kubeconfig --name dev-cluster
```

test
```
kubectl get pod -A
```
## Deploy autoscaler

```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml
```
  
put required annotation to the autoscaler deployment:

```
kubectl -n kube-system annotate deployment.apps/cluster-autoscaler cluster-autoscaler.kubernetes.io/safe-to-evict="false"
```

check kubernetes version:
```
kubectl version
```

edit deployment and set your EKS cluster name:

```
kubectl -n kube-system edit deployment.apps/cluster-autoscaler
```

Open https://github.com/kubernetes/autoscaler/releases and search latest match realise for your cluster version.

For Kubernetes 1.19 use https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.19.1

=> set the image version at property ```image=k8s.gcr.io/cluster-autoscaler:vx.yy.z``` for example 1.19.1

=> set your EKS cluster name at the end of property ```- --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/<<EKS cluster name>>``` for example dev-cluster

Check status:
```
kubectl get deployment -A
```
```
kubectl -n kube-system logs deployment.apps/cluster-autoscaler
```
## Deploy sample app and play with it

### create a deployment of nginx

nginx-deploymnt.yaml
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-autoscaler
spec:
  selector:
    matchLabels:
      app: nginx
  replicas: 1
  template:
    metadata:
      labels:
        service: nginx
        app: nginx
    spec:
      containers:
      - image: nginx
        name: test-autoscaler
        resources:
          limits:
            cpu: 300m
            memory: 512Mi
          requests:
            cpu: 300m
            memory: 512Mi
      nodeSelector:
        instance-type: spot
```

```
kubectl apply -f nginx-deployment.yaml
```

### scale the deployment

```
kubectl scale --replicas=3 deployment/test-autoscaler
```

### check pods

```
kubectl get pods -o wide --watch
```

### check nodes 

```
kubectl get nodes
```

### view cluster autoscaler logs

```
kubectl -n kube-system logs deployment.apps/cluster-autoscaler | grep "Expanding Node Group"
```
```
kubectl -n kube-system logs deployment.apps/cluster-autoscaler | grep "removing node"
```


### scale down deployment
```
kubectl scale --replicas=1 deployment/test-autoscaler
```

## delete eks nodes (or cluster) if you finish for today
```
eksctl delete nodes -f cluster.yaml
```
or
```
eksctl delete cluster -f cluster.yaml 
```

# Extra materials
If you want to scale your pod automatically please take a look at the horizontal pod autoscaler: 
https://docs.aws.amazon.com/eks/latest/userguide/horizontal-pod-autoscaler.html
