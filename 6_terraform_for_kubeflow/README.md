# Infrastructure as a Code with Terraform and AWS

The Amazon Elastic Kubernetes Service (EKS) is the AWS service for deploying, managing, and scaling containerized applications with Kubernetes.

In this tutorial, you will deploy an EKS cluster using Terraform. Then, you will configure kubectl using Terraform output to deploy a Kubernetes dashboard on the cluster.


# Why Terraform?

While you could use the built-in AWS provisioning processes (UI, CLI, CloudFormation) for EKS clusters, Terraform provides you with several benefits:

- Unified Workflow - If you are already deploying infrastructure to AWS with Terraform, your EKS cluster can fit into that workflow. You can also deploy applications into your EKS cluster using Terraform.

- Full Lifecycle Management - Terraform doesn't only create resources, it updates, and deletes tracked resources without requiring you to inspect the API to identify those resources.

- Graph of Relationships - Terraform understands dependency relationships between resources. For example, if an AWS Kubernetes cluster needs a specific VPC and subnet configurations, Terraform won't attempt to create the cluster if the VPC and subnets failed to create with the proper configuration.


# Prereqs

## Terraform cli
https://learn.hashicorp.com/tutorials/terraform/install-cli

The tutorial assumes some basic familiarity with Kubernetes and kubectl but does not assume any pre-existing deployment.

It also assumes that you are familiar with the usual Terraform plan/apply workflow. If you're new to Terraform itself, refer first to the Getting Started tutorial.

## For this tutorial, you will need:

- an AWS account with the IAM permissions listed on the EKS module documentation,
- a configured AWS CLI
- AWS IAM Authenticator
- kubectl
- wget (required for the eks module)

# Terraform modules
- vpc.tf provisions a VPC, subnets and availability zones using the AWS VPC Module. A new VPC is created for this tutorial so it doesn't impact your existing cloud environment and resources.

- security-groups.tf provisions the security groups used by the EKS cluster.

- eks-cluster.tf provisions all the resources (AutoScaling Groups, etc...) required to set up an EKS cluster using the AWS EKS Module.

- outputs.tf defines the output configuration.

- versions.tf sets the Terraform version to at least 0.14. It also sets versions for the providers used in this sample.


# Initialize Terraform workspace

Create versions.tf.
Initialize your Terraform workspace, which will download and configure the providers.
```
$ terraform init
```
Now let's add vpc.tf

```
$ terraform plan
```

We have install vpc module, so type again:

```
$ terraform init
```

OK. Now run plan again and see what will be created:
```
$ terraform plan
```
Now we can deploy VPC, confirm with 'yes'
```
$ terraform apply
```

Check in AWS Console VPC and VPC subnets.

Let's add  security-groups.tf
```
$ terraform plan
```
```
$ terraform apply
```

Now create kubernetes.tf, outputs.tf and eks.tf
```
$ terraform init
```
```
$ terraform plan
```
```
$ terraform apply
```
# Provision the EKS cluster

In your initialized directory, run terraform apply and review the planned actions. Your terminal output should indicate the plan is running and what resources will be created.
```
$ terrafrom apply
```
This terraform apply will provision a total of 51 resources (VPC, Security Groups, AutoScaling Groups, EKS Cluster, etc...). Confirm the apply with a yes.

This process should take approximately 10 minutes. Upon successful application, your terminal prints the outputs defined in outputs.tf.


# Configure kubectl
Now that you've provisioned your EKS cluster, you need to configure kubectl.

Run the following command to retrieve the access credentials for your cluster and automatically configure kubectl.
```
$ aws eks --region $(terraform output -raw region) update-kubeconfig --name $(terraform output -raw cluster_name)
```
``

# Manage Kubernetes Resources via Terraform
You can also user Terraform to manage K8s native resouces. Navigate to ./k8s_resources_with_terraform and see how to schedule Kubernetes deployment with Terraform.
```
$ cd k8s_resources_with_terraform/
```

# Clean up your workspace
Congratulations, you have provisioned an EKS cluster, configured kubectl, and deployed the Kubernetes dashboard.
You have also deployed native Kubernetes objects with Terraform! Go back to the root dir and delete EKS cluster and it's resouces.
```
$ terraform destroy
```
Navigate to the "Cluster" page by clicking on "Cluster" in the left navigation bar. You should see a list of nodes in your cluster.

The Kubernetes cluster name and region correspond to the output variables showed after the successful Terraform run.



