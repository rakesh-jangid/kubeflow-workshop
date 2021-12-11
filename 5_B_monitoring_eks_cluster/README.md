# Monitoring EKS

## Prepare the cluster

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
    desiredCapacity: 2
    maxSize: 4
    volumeSize: 20
    privateNetworking: true
 ```
 
and execute it with:
```
eksctl create [cluster|nodegroup] -f cluster.yaml
```

## Metrics server

https://github.com/kubernetes-sigs/metrics-server

### Deploy the Metrics Server with the following command:
```
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### Verify that the metrics-server deployment is running the desired number of pods with the following command.
```
kubectl get deployment metrics-server -n kube-system
```
## Kubernetes Dashboard
https://github.com/kubernetes/dashboard

### Deploy Kubernetes Dashboard
```
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.0.5/aio/deploy/recommended.yaml
```

### Service account
Create an eks-admin service account and cluster role binding.
Create a file called eks-admin-service-account.yaml with the text below. 
This manifest defines a service account and cluster role binding called eks-admin.

eks-admin-service-account.yaml
```
apiVersion: v1
kind: ServiceAccount
metadata:
  name: eks-admin
  namespace: kube-system
---
apiVersion: rbac.authorization.k8s.io/v1beta1
kind: ClusterRoleBinding
metadata:
  name: eks-admin
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-admin
subjects:
- kind: ServiceAccount
  name: eks-admin
  namespace: kube-system
```


### Apply the service account and cluster role binding to your cluster.
```
kubectl apply -f eks-admin-service-account.yaml
```

### Connect to the dashboard

Retrieve an authentication token for the eks-admin service account. Copy the <authentication_token> value from the output. You use this token to connect to the dashboard.
```
kubectl -n kube-system describe secret $(kubectl -n kube-system get secret | grep eks-admin | awk '{print $1}')
```

### Start the kubectl proxy
```
kubectl proxy
```

### Open web browser and type:
```
http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/#!/login
```

In the browser paste authentication token from the previous command and SIGN IN. Let's review the Dashboard

## Helm
If you're using macOS with Homebrew, install the binaries with the following command.
```
brew install helm
```
Confirm that Helm is running with the following command.
```
helm help
```
## Prometheus

Control plane metrics with Prometheus

The Kubernetes API server exposes a number of metrics that are useful for monitoring and analysis. These metrics are exposed internally through a metrics endpoint that refers to the /metrics HTTP API. Like other endpoints, this endpoint is exposed on the Amazon EKS control plane. This topic explains some of the ways you can use this endpoint to view and analyze what your cluster is doing.

To view the raw metrics output, use kubectl with the --raw flag. This command allows you to pass any HTTP path and returns the raw response.

```
kubectl get --raw /metrics
```
This raw output returns verbatim what the API server exposes. These metrics are represented in a Prometheus format. This format allows the API server to expose different metrics broken down by line. Each line includes a metric name, tags, and a value.
```
<metric_name>{"<tag>"="<value>"[<,...>]} <value>
```
While this endpoint is useful if you are looking for a specific metric, you typically want to analyze these metrics over time. To do this, you can deploy Prometheus into your cluster. Prometheus is a monitoring and time series database that scrapes exposed endpoints and aggregates data, allowing you to filter, graph, and query the results.

### Deploying Prometheus
https://prometheus.io/docs/introduction/overview/

#### Create a Prometheus namespace
```
kubectl create namespace prometheus
```

#### Add the prometheus-community chart repository.
```
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
```

#### Deploy Prometheus
```
helm upgrade -i prometheus prometheus-community/prometheus \
    --namespace prometheus \
    --set alertmanager.persistentVolume.storageClass="gp2",server.persistentVolume.storageClass="gp2"
```

#### Verify that all of the pods in the prometheus namespace are in the READY state.
```
kubectl get pods -n prometheus
```

#### Use kubectl to port forward the Prometheus console to your local machine.
```
kubectl --namespace=prometheus port-forward deploy/prometheus-server 9090
```

#### Point a web browser to localhost:9090 to view the Prometheus console.

#### Search for some metrics
```
container_memory_usage_bytes
```

## Grafana

### Create namespace grafana
````
kubectl create namespace grafana
````

### Create a file called grafana.yaml, then paste the contents below.
```
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: grafana
  name: grafana
spec:
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      securityContext:
        fsGroup: 472
        supplementalGroups:
        - 0    
      containers:
        - name: grafana
          image: grafana/grafana:7.5.2
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 3000
              name: http-grafana
              protocol: TCP
          readinessProbe:
            failureThreshold: 3
            httpGet:
              path: /robots.txt
              port: 3000
              scheme: HTTP
            initialDelaySeconds: 10
            periodSeconds: 30
            successThreshold: 1
            timeoutSeconds: 2
          livenessProbe:
            failureThreshold: 3
            initialDelaySeconds: 30
            periodSeconds: 10
            successThreshold: 1
            tcpSocket:
              port: 3000
            timeoutSeconds: 1            
          resources:
            requests:
              cpu: 250m
              memory: 750Mi
          volumeMounts:
            - mountPath: /var/lib/grafana
              name: grafana-pv
      volumes:
        - name: grafana-pv
          persistentVolumeClaim:
            claimName: grafana-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
spec:
  ports:
    - port: 3000
      protocol: TCP
      targetPort: http-grafana
  selector:
    app: grafana
  sessionAffinity: None
  type: LoadBalancer
```

### Apply manifest with:
```
kubectl apply -f grafana.yaml -n grafana
```

### Check that it worked by running the following:
```
kubectl port-forward service/grafana 3000:3000 -n grafana
```

### Navigate to localhost:3000 in your browser. You should see a Grafana login page.

### Use admin for both the username and password to login.

### As a datasource use: 
prometheus: prometheus-server.prometheus.svc.cluster.local

### Let's install some cool dashboard
https://grafana.com/grafana/dashboards

for example 3119 or 10000

## Delete

### Delete grafana
```
kubectl delete -f grafana.yaml -n grafana
```

### Delete prometheus with helm
```
 helm uninstall prometheus --namespace prometheus
```

### Delete eks cluster
```
eksctl delete cluster -f cluster.yaml
```
