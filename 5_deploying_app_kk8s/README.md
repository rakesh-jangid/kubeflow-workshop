# Deploying the application


## Prepare cluster

Create EKS cluster or add node group to the existing one

Create file cluster.yaml with the following content
```
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: dev-cluster
  region: us-west-2

nodeGroups:
  - name: ng-1-workers
    labels: { role: workers }
    instanceType: t3.small
    minSize: 1
    desiredCapacity: 3
    maxSize: 4
    volumeSize: 20
    privateNetworking: true
 ```
 
and execute it with:
```
eksctl create [cluster|nodegroup] -f cluster.yaml
```
## Create Pod

Update kube config:
```
aws eks update-kubeconfig --name dev-cluster
```
Check cluster
```
kubectl cluster-info
```

...much more info here:
```
kubectl cluster-info dump
```

Check kluster nodes:
```
 kubectl get nodes
```

List all pods in all namespaces:
```
kubectl get pods -A
```

Let's create simple pod from our python app, update <account_id> with your account id:

pod.yaml
```
apiVersion: v1
kind: Pod
metadata:
  name: static-web
  labels:
    role: myrole
spec:
  containers:
    - name: web
      image: <account_id>.dkr.ecr.us-west-2.amazonaws.com/repo1:latest
      ports:
        - name: web
          containerPort: 8001
          protocol: TCP
```

Create namespace test
```
kubectl create ns test
```

Deploy pod in the namespace test:
```
kubectl apply -f pod.yaml -n test
```
See the pod:
```
kubectl get pod -n test
```

Check the logs:
```
kubectl logs -f static-web -n test
```

Check IP and Node IP assigned by CNI plugin. Check EC2 instance network section from AWS console
```
kubectl get pod -n test -owide
```

Now let's delete pod and a namespace:

pod:
```
kubectl delete pod static-web -n test
```

ns:
```
kubectl delete ns test
```

## Create Deployment

deployment.yaml
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: static-web
spec:
  selector:
    matchLabels:
      app: webapp
      role: myrole
      tier: backend
  replicas: 1
  template:
    metadata:
      labels:
        app: webapp
        role: myrole
        tier: backend
    spec:
      containers:
      - name: webapp
        image: <account_id>.dkr.ecr.us-west-2.amazonaws.com/repo1:latest
        resources:
          requests:
            cpu: 100m
            memory: 100Mi
        ports:
        - containerPort: 8001
```

Create namespace deployment
```
kubectl create ns deployment
```

Deploy our app
```
kubectl apply -f deployment.yaml -n deployment
```

Check pod:
```
kubectl get pod -n deployment
```

ReplicaSet was created automatically
```
 kubectl get replicaset -n deployment
```

And there is our deployment:
```
kubectl get deployment -n deployment
```

Let's scale our deployment by editing <replicas> parameter of deployment object

```  
kubectl edit deployment -n deployment
 ```
  
and set ** replicas: 2 **

deployment.yaml
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: static-web
spec:
  selector:
    matchLabels:
      app: webapp
      role: myrole
      tier: backend
  replicas: 2 
  template:
    metadata:
      labels:
        app: webapp
        role: myrole
        tier: backend
    spec:
      containers:
      - name: webapp
        image: <account_id>.dkr.ecr.us-west-2.amazonaws.com/repo1:latest
        resources:
          requests:
            cpu: 100m
            memory: 100Mi
        ports:
        - containerPort: 8001
```

Inspect deployment
```
kubectl describe deployment static-web -n deployment
```
  
to make it more visible grep scaling acrtivities
```
kubectl describe deployment static-web -n deployment | grep ScalingReplicaSet
```
  
Now let's kill pod and see what happens:
```
kubectl get pod -n deployment
```
  
```
kubectl delete pod static-web-5df7d86566-6zwdd  -n deployment
```  


Now check pods AGE
```
kubectl get pod -owide -n deployment
```

Let's do the cleanup
```
kubectl delete deployment static-web -n deployment
```

```
kubectl delete ns deployment
```
  
# Create Service
## NodePort
service.yaml:
```
---
apiVersion: v1
kind: Namespace
metadata:
  name: webapp
---
apiVersion: apps/v1
kind: Deployment
metadata:
  namespace: webapp
  name: deployment-webapp
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: app-webapp
  replicas: 1
  template:
    metadata:
      labels:
        app.kubernetes.io/name: app-webapp
    spec:
      containers:
      - image: <account_id>.dkr.ecr.us-west-2.amazonaws.com/repo1:latest
        imagePullPolicy: Always
        name: app-webapp
        ports:
        - containerPort: 8001
---
apiVersion: v1
kind: Service
metadata:
  namespace: webapp
  name: service-webapp
spec:
  ports:
    - port: 8001
      targetPort: 8001
      protocol: TCP
  type: NodePort
  selector:
    app.kubernetes.io/name: app-webapp
```
  
```
kubectl get service -n webapp

webapp         service-webapp                      NodePort    10.100.246.160   <none>        8001:31994/TCP   96s
```
  
```
kubectl get pod -n webapp
NAME                                 READY   STATUS    RESTARTS   AGE
deployment-webapp-79ff5f6669-smbxr   1/1     Running   0          2m16s
```

```
kubectl port-forward deployment-webapp-79ff5f6669-smbxr 8001:8001
```
  
Open webbrowser on localhost:8001. You should see "Hello, world!"

Now delete service, deployment and pods:
```
kubectl delete -f service.yaml
```
  
## ALB ingress controler
# Install required software

IAM policy
```
curl -o iam_policy.json https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.2.0/docs/install/iam_policy.json
```

Create policy
```
aws iam create-policy \
    --policy-name AWSLoadBalancerControllerIAMPolicy \
    --policy-document file://iam_policy.json
```
  
Associate oidc provider
```
eksctl utils associate-iam-oidc-provider --region=us-west-2 --cluster=dev-cluster
```

Create IAM role for service account
```
eksctl create iamserviceaccount \
  --cluster=dev-cluster \
  --namespace=kube-system \
  --name=aws-load-balancer-controller \
  --attach-policy-arn=arn:aws:iam::<account_id>:policy/AWSLoadBalancerControllerIAMPolicy \
  --override-existing-serviceaccounts \
  --approve 
```
 
Install cert manager
```
kubectl apply \
    --validate=false \
    -f https://github.com/jetstack/cert-manager/releases/download/v1.1.1/cert-manager.yaml
```

Download alb controller:
```
curl -o v2_2_0_full.yaml https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.2.0/docs/install/v2_2_0_full.yaml
```
Make the following edits to the v2_2_0_full.yaml file:

Delete the ServiceAccount section of the file. Deleting this section prevents the annotation with the IAM role from being overwritten when the controller is deployed and preserves the service account that you created in step 4 if you delete the controller.

Replace your-cluster-name to the Deployment spec section of the file with the name of your cluster.

Apply the file:
```
kubectl apply -f v2_2_0_full.yaml
```
  
Check ingress controller
```
kubectl get pod -A
```

 
# Ingress
Now we can deploy our ingress service
 
ingress.yaml
```
---
apiVersion: v1
kind: Namespace
metadata:
  name: webapp
---
apiVersion: apps/v1
kind: Deployment
metadata:
  namespace: webapp
  name: deployment-webapp
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: app-webapp
  replicas: 1
  template:
    metadata:
      labels:
        app.kubernetes.io/name: app-webapp
    spec:
      containers:
      - image: <account_id>.dkr.ecr.us-west-2.amazonaws.com/repo1:latest
        imagePullPolicy: Always
        name: app-webapp
        ports:
        - containerPort: 8001
---
apiVersion: v1
kind: Service
metadata:
  namespace: webapp
  name: service-webapp
spec:
  ports:
    - port: 8001
      targetPort: 8001
      protocol: TCP
  type: NodePort
  selector:
    app.kubernetes.io/name: app-webapp
---
apiVersion: networking.k8s.io/v1beta1
kind: Ingress
metadata:
  namespace: webapp
  name: ingress-webapp
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
spec:
  rules:
    - http:
        paths:
          - path: /*
            backend:
              serviceName: service-webapp
              servicePort: 8001
```

Deploy ingress:
```
kubectl apply -f ingress.yaml 
```

Check ingress
```
kubectl get ingress -A
``` 

Copy load balancer address to your browser and Voila!

Now let's delete it:
```
kubectl delete -f ingress.yaml 
```
