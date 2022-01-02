# Introduction

Kubeflow installation with Ansible and Kustomize

## Design directory structure

1 Create 'ansible' directory. It will be our main directory.

2 In 'ansible' dir create 'group_vars', 'inventory' and 'roles' dirs

3 create file 'ansible/group_vars/kubeflow' with content:
```
aws_region: us-west-2
cluster_name: "noble"
```

4 create ansible/inventory/local if not exist
```
[local]
localhost ansible_connection=local ansible_python_interpreter=/path/to/venv/bin/python
```

5 create ansible/roles/kubeflow/tasks/main.yaml
```
- name: Update kubeconfig
  shell: "aws eks update-kubeconfig --name {{ cluster_name }}"

- name: Deploy kubeflow with kustomize (version 3.5 recommended)
  shell: "./roles/kubeflow/app/noble/install.sh"
```

6 create ansible/ansible.cfg if not exist
```
[defaults]
bin_ansible_callbacks =True
display_skipped_hosts = no
host_key_checking = False
stdout_callback = yaml


[inventory]
enable_plugins = host_list, script, yaml, ini, auto
```

7 create ansible/noble_kubeflow.yaml
```
- hosts: local

  vars_files:
  - group_vars/kubeflow

  roles:
  - kubeflow
```

8 create ansible/kubeflow/app/noble/kubeflow1.3/kustomization.yaml
```
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
# Cert-Manager
#- ../manifests-1.3-branch/common/cert-manager/cert-manager-kube-system-resources/base
#- ../manifests-1.3-branch/common/cert-manager/cert-manager-crds/base
#- ../manifests-1.3-branch/common/cert-manager/cert-manager/overlays/self-signed
- ../manifests-1.3-branch/common/cert-manager/cert-manager/base
- ../manifests-1.3-branch/common/cert-manager/kubeflow-issuer/base

# Istio
- ../manifests-1.3-branch/common/istio-1-9/istio-crds/base
- ../manifests-1.3-branch/common/istio-1-9/istio-namespace/base
- ../manifests-1.3-branch/common/istio-1-9/istio-install/base
- ../manifests-1.3-branch/common/istio-1-9/cluster-local-gateway/base


# OIDC Authservice
- ../manifests-1.3-branch/common/oidc-authservice/base


# Dex
- ../manifests-1.3-branch/common/dex/overlays/istio

# KNative
- ../manifests-1.3-branch/common/knative/knative-serving/base

#- ../manifests-1.3-branch/common/knative/knative-serving-install/base

- ../manifests-1.3-branch/common/knative/knative-eventing/base

#- ../manifests-1.3-branch/common/knative/knative-eventing-install/base


# Kubeflow namespace
- ../manifests-1.3-branch/common/kubeflow-namespace/base

# Kubeflow Roles
- ../manifests-1.3-branch/common/kubeflow-roles/base

# Kubeflow Istio Resources
- ../manifests-1.3-branch/common/istio-1-9/kubeflow-istio-resources/base

# Kubeflow Pipelines
- ../manifests-1.3-branch/apps/pipeline/upstream/env/platform-agnostic-multi-user

# KFServing
- ../manifests-1.3-branch/apps/kfserving/upstream/overlays/kubeflow

# Central Dashboard
- ../manifests-1.3-branch/apps/centraldashboard/upstream/overlays/istio

# Admission Webhook
- ../manifests-1.3-branch/apps/admission-webhook/upstream/overlays/cert-manager

# Notebook Controller
- ../manifests-1.3-branch/apps/jupyter/jupyter-web-app/upstream/overlays/istio

# Jupyter Web App
- ../manifests-1.3-branch/apps/jupyter/notebook-controller/upstream/overlays/kubeflow

#Profiles + KFAM
- ../manifests-1.3-branch/apps/profiles/upstream/overlays/kubeflow

# Volumes Web App
- ../manifests-1.3-branch/apps/volumes-web-app/upstream/overlays/istio

# User namespace
- ../manifests-1.3-branch/common/user-namespace/base


# Katib
- ../manifests-1.3-branch/apps/katib/upstream/installs/katib-with-kubeflow
# Tensorboards Web App
-  ../manifests-1.3-branch/apps/tensorboard/tensorboard-controller/upstream/overlays/kubeflow
# Tensorboard Controller
-  ../manifests-1.3-branch/apps/tensorboard/tensorboards-web-app/upstream/overlays/istio
# TFJob Operator
- ../manifests-1.3-branch/apps/tf-training/upstream/overlays/kubeflow
# Pytorch Operator
#- ../manifests-1.3-branch/apps/pytorch-job/upstream/overlays/kubeflow
# MPI Operator
#- ../manifests-1.3-branch/apps/mpi-job/upstream/overlays/kubeflow
# MXNet Operator
#- ../manifests-1.3-branch/apps/mxnet-job/upstream/overlays/kubeflow
# XGBoost Operator
#- ../manifests-1.3-branch/apps/xgboost-job/upstream/overlays/kubeflow
```

8 create ansible/kubeflow/app/noble/download.sh
```
#!/bin/bash
# See
# https://github.com/kubeflow/manifests/tree/v1.3-branch
wget https://github.com/kubeflow/manifests/archive/v1.3-branch.tar.gz
tar -xvf v1.3-branch.tar.gz
```

9 create ansible/kubeflow/app/noble/install.sh
```
while ! kustomize build kubeflow1.3 | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done

```

You should have the following structure:
```
.
├── Makefile
├── ansible
│   ├── ansible.cfg
│   ├── group_vars
│   │   ├── kubeflow
│   ├── inventory
│   │   └── local
│   ├── noble_kubeflow.yaml
│   └── roles
│   │  ├── kubeflow
│   │  │   ├── app
│   │  │   ├── ├──  noble
│   │  │   ├── ├──  ├──  download.sh
│   │  │   ├── ├──  ├──  install.sh
│   │  │   ├── ├──  ├── kubeflow1.3
│   │  │   ├── ├──  ├── ├── kustomization
    
│   │  │   ├── tasks
│   │  │   ├── ├──  main.yaml

```

8 Run it with(you can add it to Makefile):
```
cd ansible && ansible-playbook -i inventory/local noble_kubeflow.yaml --verbose
```
