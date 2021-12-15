# Kubeflow on Microk8s

Microk8s is built by the Kubernetes team at Canonical. 
Build for the zero-ops user experience.  It comes with many usefull Addons and Charmed Operators.

## Minimum requirements:
16GB RAM
15GB Storage

You can do it on your own laptop or you can use AWS Ec2

## AWS Ec2 Setup
AMI Ubuntu 18.04
Ec2 instance type: t2.xlarge
EBS 60GB
ssh key

Donwload key.pem and change access
```
chmod 600 key.pem 
```

SSH to the machine
```
ssh -i key.pem ubuntu@<ec2-PublicDNS>.us-west-2.compute.amazonaws.com
```
## Installation
Login to the instance using SSM (or ssh)

Install Microk8s
```
sudo snap install microk8s --classic --channel=1.21/stable
```

Check status:
```
microk8s status
```

```
sudo usermod -a -G microk8s ubuntu
sudo chown -f -R ubuntu ~/.kube
```

Exit and login again

Type again:
```
microk8s status
```

Install core components
```
microk8s enable dns storage dashboard
```

Run again and check if installation succeed:
```
microk8s status
```

We can use also kubectl commands:
```
microk8s.kubectl get pod -A
```

Now let's run Kubeflow!
```
microk8s.enable kubeflow
```

Let's wait some time since Kubeflow is quite heavy...

In the end you will see endpoint to the dashboard as well as credentials.

## Access with SOCKS proxy
To access Ec2 VM we have to setup SOCKS proxy
```
ssh -i "my.pem" -D9999 ubuntu@<Ec2DNS>.amazonaws.com
```

Now enable on you localhost SOCKS proxy(on mac go to wifi/advanced/proxies/SOCKS)
127.0.0.1:9999


Now you can try to access to the Kubeflow dashboard using link provided during installation

The dashboard is available at http://Ec2_IP.nip.io
