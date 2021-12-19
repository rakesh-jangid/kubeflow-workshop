# Manage Kubernetes Resources via Terraform

Kubernetes (K8S) is an open-source workload scheduler with focus on containerized applications. You can use the Terraform Kubernetes provider to interact with resources supported by Kubernetes.

In this tutorial, you will learn how to interact with Kubernetes using Terraform, by scheduling and exposing a NGINX deployment on a Kubernetes cluster.

The final Terraform configuration files used in this tutorial can be found in the Deploy NGINX on Kubernetes via Terraform GitHub repository.


# Prereqs

You will need existing EKS cluster.


# Create a directory named learn-terraform-deploy-nginx-kubernetes

```
$ mkdir learn-terraform-deploy-nginx-kubernetes
```

# Then, navigate into it

```
$ cd learn-terraform-deploy-nginx-kubernetes
```
Note: This directory is only used for managing Kubernetes cluster resources with Terraform. By keeping the Terraform configuration for provisioning a Kubernetes cluster and managing a Kubernetes resources separate, changes in one repository doesn't affect the other. In addition, the modularity makes the configuration more readable and enables you to scope different permissions to each workspace.


# Configure the provider

Before you can schedule any Kubernetes services using Terraform, you need to configure the Terraform Kubernetes provider.
In this tutorial we will use eks get-token to to this.
Create a new file named kubernetes.tf.
It will contain provider information and k8s deployment of the nginx server.

# Init
```
$ terraform init
```

# Apply
```
$ terraform apply
```

# Check pods
```
$ kubectl get pods -A
```

# Check deployment
```
$ kubectl get deployments -A
```

# Play with parameters, e.g. change number of replicas to 1 and apply changes

# Clean up
```
$ terraform destroy
```
