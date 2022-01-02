# Introduction
AWS CloudFormation gives you an easy way to model a collection of related AWS and third-party resources, provision them quickly and consistently, and manage them throughout their lifecycles, by treating infrastructure as code. A CloudFormation template describes your desired resources and their dependencies so you can launch and configure them together as a stack. You can use a template to create, update, and delete an entire stack as a single unit, as often as you need to, instead of managing resources individually. You can manage and provision stacks across multiple AWS accounts and AWS Regions.
More: https://aws.amazon.com/cloudformation/


## Design directory structure

1 Create 'ansible' directory. It will be our main directory.

2 In 'ansible' dir create 'group_vars', 'inventory' and 'roles' dirs

3 create file 'ansible/group_vars/eks' with content:
```
aws_region: us-west-2
aws_account: TODO
# choose one from eksctl or cloudformation
cloudformation_installation: "true"
eksctl_installation: "false"

# cloudformation cluster settings
cluster_name: "noble"
k8s_version: "1.19"

# manage node group
node_group_name: "group1"
ami_id_gr_1: "ami-0c105638948525c1c"
```

4 create ansible/inventory/local if does not exist already
```
[local]
localhost ansible_connection=local ansible_python_interpreter=/path/to/venv/bin/python
```

5 create ansible/roles/eks/tasks/main.yaml
```
- name: setup infrastructure with cloudformation
  import_tasks: "cloudformation.yaml"

```

6 create ansible/roles/eks/tasks/cloudformation.yaml
```
- name: Deploy VPC
  cloudformation:
    stack_name: "{{ cluster_name }}-vpc"
    state: "present"
    region: "{{ aws_region }}"
    template: "roles/eks/app/cloudformation/VPC.yaml"

- name: get vpc params from cf
  cloudformation_info:
    stack_name: "{{ cluster_name }}-vpc"
  register: vpc_params

- name: Deploy IAM
  cloudformation:
    stack_name: "{{ cluster_name }}-iam"
    state: "present"
    region: "{{ aws_region }}"
    template: "roles/eks/app/cloudformation/IAM.yaml"

- name: get iam params from cf
  cloudformation_info:
    stack_name: "{{ cluster_name }}-iam"
  register: iam_params

- debug:
    msg: "{{ vpc_params['cloudformation'][cluster_name + '-vpc'].stack_outputs['VpcId'] }}"
- set_fact:
    VPCId: "{{ vpc_params['cloudformation'][cluster_name + '-vpc'].stack_outputs['VpcId'] }}"
- set_fact:
    PrivSub1: "{{ vpc_params['cloudformation'][cluster_name + '-vpc'].stack_outputs['PrivSub1'] }}"
- set_fact:
    PrivSub2: "{{ vpc_params['cloudformation'][cluster_name + '-vpc'].stack_outputs['PrivSub2'] }}"
- set_fact:
    PubSub1: "{{ vpc_params['cloudformation'][cluster_name + '-vpc'].stack_outputs['PubSub1'] }}"
- set_fact:
    PubSub2: "{{ vpc_params['cloudformation'][cluster_name + '-vpc'].stack_outputs['PubSub2'] }}"
- set_fact:
    EKSServiceRoleARN: "{{ iam_params['cloudformation'][cluster_name + '-iam'].stack_outputs['EKSServiceRoleARN'] }}"

- name: Deploy Cluster
  cloudformation:
    stack_name: "{{ cluster_name }}-cluster"
    state: "present"
    region: "{{ aws_region }}"
    template: "roles/eks/app/cloudformation/Cluster.yaml"
    template_parameters:
      EKSClusterName: "{{ cluster_name }}"
      KubernetesVersion: "{{ k8s_version }}"
      VPCID: "{{ VPCId }}"
      PrivSubnet1: "{{ PrivSub1 }}"
      PrivSubnet2: "{{ PrivSub2 }}"
      PubSubnet1: "{{ PubSub1 }}"
      PubSubnet2: "{{ PubSub2 }}"
      EKSServiceRoleARN: "{{ EKSServiceRoleARN }}"

- name: wait for cluster to become active
  shell: "aws eks wait cluster-active --name={{ cluster_name }}"

- name: Update aws-auth config map to allow nodes join cluster
  shell: |
    aws eks update-kubeconfig --name {{ cluster_name }}
    kubectl cluster-info
    cat << EOF | kubectl apply -f -
    ---
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: aws-auth
      namespace: kube-system
    data:
      mapRoles: |
        - rolearn: {{ iam_params['cloudformation'][cluster_name + '-iam'].stack_outputs['NodeInstanceRoleARN'] }}
          username: system:node:{{ '{{' }}EC2PrivateDNSName{{ '}}' }}
          groups:
            - system:bootstrappers
            - system:nodes
    EOF

- name: Deploy Node gr - managed
  cloudformation:
    stack_name: "{{ cluster_name }}-{{ node_group_name }}-nodegroup"
    state: "present"
    region: "{{ aws_region }}"
    template: "roles/eks/app/cloudformation/Node.yaml"
    template_parameters:
      EKSClusterName: "{{ cluster_name }}"
      NodeImageId: "{{ ami_id_gr_1 }}"
      NodeInstanceType: t2.micro
      NodeAutoScalingGroupMinSize: 1
      NodeAutoScalingGroupMaxSize: 3
      NodeAutoScalingGroupDesiredCapacity: 2
      BootstrapArguments: "--kubelet-extra-args --node-labels=kubeflow.node.type=core,nodegroup={{ node_group_name }}"
      NodeGroupName: "{{ node_group_name }}"
      Subnet1: "{{ PrivSub1 }}"
      Subnet2: "{{ PrivSub2 }}"
      VpcID: "{{ VPCId }}"

## To extend cluster with autoscalable workloads you can add nodegroups with GPU and CPU using spot intances
## Managed node groups currently does not scale from/to 0
## For stateful workloads (for example Notebooks) it is required to have one node group per AZ
## ADD spot unmanaged
## ADD GPU spot unmanaged

# Required by Iam roles for SA
- name: add iam oidc with the cluster using eksctl
  shell: "eksctl utils associate-iam-oidc-provider --cluster {{ cluster_name }} --approve"
  args:
    executable: /bin/bash

```

7 create ansible/roles/eks/app/cloudformation/VPC.yaml
```
---
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Amazon EKS Sample VPC - Private and Public subnets'

Parameters:

  VpcBlock:
    Type: String
    Default: 192.168.0.0/16
    Description: The CIDR range for the VPC. This should be a valid private (RFC 1918) CIDR range.

  PublicSubnet01Block:
    Type: String
    Default: 192.168.0.0/18
    Description: CidrBlock for public subnet 01 within the VPC

  PublicSubnet02Block:
    Type: String
    Default: 192.168.64.0/18
    Description: CidrBlock for public subnet 02 within the VPC

  PrivateSubnet01Block:
    Type: String
    Default: 192.168.128.0/18
    Description: CidrBlock for private subnet 01 within the VPC

  PrivateSubnet02Block:
    Type: String
    Default: 192.168.192.0/18
    Description: CidrBlock for private subnet 02 within the VPC

Metadata:
  AWS::CloudFormation::Interface:
    ParameterGroups:
      -
        Label:
          default: "Worker Network Configuration"
        Parameters:
          - VpcBlock
          - PublicSubnet01Block
          - PublicSubnet02Block
          - PrivateSubnet01Block
          - PrivateSubnet02Block

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock:  !Ref VpcBlock
      EnableDnsSupport: true
      EnableDnsHostnames: true
      Tags:
      - Key: Name
        Value: !Sub '${AWS::StackName}-VPC'

  InternetGateway:
    Type: "AWS::EC2::InternetGateway"

  VPCGatewayAttachment:
    Type: "AWS::EC2::VPCGatewayAttachment"
    Properties:
      InternetGatewayId: !Ref InternetGateway
      VpcId: !Ref VPC

  PublicRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
      - Key: Name
        Value: Public Subnets
      - Key: Network
        Value: Public

  PrivateRouteTable01:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
      - Key: Name
        Value: Private Subnet AZ1
      - Key: Network
        Value: Private01

  PrivateRouteTable02:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
      - Key: Name
        Value: Private Subnet AZ2
      - Key: Network
        Value: Private02

  PublicRoute:
    DependsOn: VPCGatewayAttachment
    Type: AWS::EC2::Route
    Properties:
      RouteTableId: !Ref PublicRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  PrivateRoute01:
    DependsOn:
    - VPCGatewayAttachment
    - NatGateway01
    Type: AWS::EC2::Route
    Properties:
      RouteTableId: !Ref PrivateRouteTable01
      DestinationCidrBlock: 0.0.0.0/0
      NatGatewayId: !Ref NatGateway01

  PrivateRoute02:
    DependsOn:
    - VPCGatewayAttachment
    - NatGateway02
    Type: AWS::EC2::Route
    Properties:
      RouteTableId: !Ref PrivateRouteTable02
      DestinationCidrBlock: 0.0.0.0/0
      NatGatewayId: !Ref NatGateway02

  NatGateway01:
    DependsOn:
    - NatGatewayEIP1
    - PublicSubnet01
    - VPCGatewayAttachment
    Type: AWS::EC2::NatGateway
    Properties:
      AllocationId: !GetAtt 'NatGatewayEIP1.AllocationId'
      SubnetId: !Ref PublicSubnet01
      Tags:
      - Key: Name
        Value: !Sub '${AWS::StackName}-NatGatewayAZ1'

  NatGateway02:
    DependsOn:
    - NatGatewayEIP2
    - PublicSubnet02
    - VPCGatewayAttachment
    Type: AWS::EC2::NatGateway
    Properties:
      AllocationId: !GetAtt 'NatGatewayEIP2.AllocationId'
      SubnetId: !Ref PublicSubnet02
      Tags:
      - Key: Name
        Value: !Sub '${AWS::StackName}-NatGatewayAZ2'

  NatGatewayEIP1:
    DependsOn:
    - VPCGatewayAttachment
    Type: 'AWS::EC2::EIP'
    Properties:
      Domain: vpc

  NatGatewayEIP2:
    DependsOn:
    - VPCGatewayAttachment
    Type: 'AWS::EC2::EIP'
    Properties:
      Domain: vpc

  PublicSubnet01:
    Type: AWS::EC2::Subnet
    Metadata:
      Comment: Subnet 01
    Properties:
      MapPublicIpOnLaunch: true
      AvailabilityZone:
        Fn::Select:
        - '0'
        - Fn::GetAZs:
            Ref: AWS::Region
      CidrBlock:
        Ref: PublicSubnet01Block
      VpcId:
        Ref: VPC
      Tags:
      - Key: Name
        Value: !Sub "${AWS::StackName}-PublicSubnet01"
      - Key: kubernetes.io/role/elb
        Value: 1

  PublicSubnet02:
    Type: AWS::EC2::Subnet
    Metadata:
      Comment: Subnet 02
    Properties:
      MapPublicIpOnLaunch: true
      AvailabilityZone:
        Fn::Select:
        - '1'
        - Fn::GetAZs:
            Ref: AWS::Region
      CidrBlock:
        Ref: PublicSubnet02Block
      VpcId:
        Ref: VPC
      Tags:
      - Key: Name
        Value: !Sub "${AWS::StackName}-PublicSubnet02"
      - Key: kubernetes.io/role/elb
        Value: 1

  PrivateSubnet01:
    Type: AWS::EC2::Subnet
    Metadata:
      Comment: Subnet 03
    Properties:
      AvailabilityZone:
        Fn::Select:
        - '0'
        - Fn::GetAZs:
            Ref: AWS::Region
      CidrBlock:
        Ref: PrivateSubnet01Block
      VpcId:
        Ref: VPC
      Tags:
      - Key: Name
        Value: !Sub "${AWS::StackName}-PrivateSubnet01"
      - Key: kubernetes.io/role/internal-elb
        Value: 1

  PrivateSubnet02:
    Type: AWS::EC2::Subnet
    Metadata:
      Comment: Private Subnet 02
    Properties:
      AvailabilityZone:
        Fn::Select:
        - '1'
        - Fn::GetAZs:
            Ref: AWS::Region
      CidrBlock:
        Ref: PrivateSubnet02Block
      VpcId:
        Ref: VPC
      Tags:
      - Key: Name
        Value: !Sub "${AWS::StackName}-PrivateSubnet02"
      - Key: kubernetes.io/role/internal-elb
        Value: 1

  PublicSubnet01RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      SubnetId: !Ref PublicSubnet01
      RouteTableId: !Ref PublicRouteTable

  PublicSubnet02RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      SubnetId: !Ref PublicSubnet02
      RouteTableId: !Ref PublicRouteTable

  PrivateSubnet01RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      SubnetId: !Ref PrivateSubnet01
      RouteTableId: !Ref PrivateRouteTable01

  PrivateSubnet02RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      SubnetId: !Ref PrivateSubnet02
      RouteTableId: !Ref PrivateRouteTable02

  ControlPlaneSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Cluster communication with worker nodes
      VpcId: !Ref VPC

Outputs:

  SubnetIds:
    Description: Subnets IDs in the VPC
    Value: !Join [ ",", [ !Ref PublicSubnet01, !Ref PublicSubnet02, !Ref PrivateSubnet01, !Ref PrivateSubnet02 ] ]

  PubSub1:
    Description: Public subnet1
    Value: !Ref PublicSubnet01

  PubSub2:
    Description: Public subnet2
    Value: !Ref PublicSubnet02

  PrivSub1:
    Description: Private subnet1
    Value: !Ref PrivateSubnet01

  PrivSub2:
    Description: Private subnet2
    Value: !Ref PrivateSubnet02

  SecurityGroups:
    Description: SG cluster control plane communication with worker nodes
    Value: !Join [ ",", [ !Ref ControlPlaneSecurityGroup ] ]

  VpcId:
    Description: The VPC Id
    Value: !Ref VPC

```

8 create ansible/roles/eks/app/cloudformation/IAM.yaml
```
AWSTemplateFormatVersion: "2010-09-09"
Description: "EKS IAM"

Resources:
  EKSServiceRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Action: sts:AssumeRole
            Principal:
              Service: eks.amazonaws.com
      Policies:
        - PolicyName: eks-lb-access
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action:
                  - ec2:DescribeAccountAttributes
                  - ec2:DescribeInternetGateways
                Resource: "*"
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonEKSClusterPolicy

  NodeInstanceProfile:
    Type: AWS::IAM::InstanceProfile
    Properties:
      Path: "/"
      Roles:
        - !Ref NodeInstanceRole

  NodeInstanceRole:
    Type: AWS::IAM::Role
    Properties:
      Policies:
        - PolicyName: eks-autoscaler
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action:
                  - autoscaling:DescribeAutoScalingGroups
                  - autoscaling:DescribeAutoScalingInstances
                  - autoscaling:DescribeLaunchConfigurations
                  - autoscaling:DescribeTags
                  - autoscaling:SetDesiredCapacity
                  - autoscaling:TerminateInstanceInAutoScalingGroup
                  - ec2:DescribeLaunchTemplateVersions
                Resource: "*"

        - PolicyName: ECRFullTemp
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: "Allow"
                Action:
                  - ecr:*
                Resource: "*"

        - PolicyName: S3FullTemp
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: "Allow"
                Action:
                  - s3:*
                Resource: "*"

      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Service:
                - ec2.amazonaws.com
            Action:
              - sts:AssumeRole
      Path: "/"

      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
        - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
        - arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
        - arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
        - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
        - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

Outputs:

  NodeInstanceRole:
    Export:
      Name: !Sub "${AWS::StackName}-NodeInstanceRole"
    Description: The node instance role
    Value: !Ref NodeInstanceRole

  NodeInstanceRoleARN:
    Export:
      Name: !Sub "${AWS::StackName}-NodeInstanceRoleARN"
    Description: The node instance role arn
    Value: !GetAtt NodeInstanceRole.Arn

  NodeInstanceProfileARN:
    Export:
      Name: !Sub "${AWS::StackName}-NodeInstanceProfileARN"
    Description: Instance profile arn
    Value: !GetAtt NodeInstanceProfile.Arn

  EKSServiceRoleARN:
    Export:
      Name: !Sub "${AWS::StackName}-EKSServiceRoleARN"
    Value: !GetAtt EKSServiceRole.Arn

```
9 create ansible/roles/eks/app/cloudformation/Cluster.yaml
```
AWSTemplateFormatVersion: "2010-09-09"
Description: "EKS Kubernetes cluster"

Parameters:

  EKSClusterName:
    Type: String
    MinLength: 3
    Default: "noble"

  KubernetesVersion:
    Type: String
    Default: "1.19"

  VPCID:
    Description: "VPC id"
    Type: String

  PrivSubnet1:
    Description: "Private subnet 1"
    Type: String

  PrivSubnet2:
    Description: "Private subnet 2"
    Type: String

  PubSubnet1:
    Description: "Public subnet1"
    Type: String

  PubSubnet2:
    Description: "Public subnet2"
    Type: String

  EKSServiceRoleARN:
    Description: "EKS service role arn"
    Type: String

Resources:
  ClusterControlPlaneSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Cluster connection with k8s nodes
      VpcId: !Ref VPCID
      SecurityGroupEgress:
        - Description: Cluster connection with k8s nodes
          CidrIp: 0.0.0.0/0
          IpProtocol: "-1"
          FromPort: 0
          ToPort: 65535

  EKSCluster:
    Type: AWS::EKS::Cluster
    Properties:
      Name: !Ref EKSClusterName
      RoleArn: !Ref EKSServiceRoleARN
      Version: !Ref KubernetesVersion
      ResourcesVpcConfig:
        SecurityGroupIds:
          - !GetAtt ClusterControlPlaneSecurityGroup.GroupId
        SubnetIds:
            - !Ref PrivSubnet1
            - !Ref PrivSubnet2
            - !Ref PubSubnet1
            - !Ref PubSubnet2


Outputs:
  ClusterControlPlaneSecurityGroup:
    Export:
      Name: !Sub "${AWS::StackName}-ClusterControlPlaneSecurityGroup"
    Value: !Ref ClusterControlPlaneSecurityGroup
  EKSClusterName:
    Export:
      Name: !Sub "${AWS::StackName}-EKSCluster"
    Value: !Ref EKSCluster
  EKSClusterEndpoint:
    Export:
      Name: !Sub "${AWS::StackName}-EKSClusterEndpoint"
    Value: !GetAtt EKSCluster.Endpoint

```

10 create ansible/roles/eks/app/cloudformation/Node.yaml
```
AWSTemplateFormatVersion: "2010-09-09"
Description: "EKS Node Group(managed)"

Parameters:
  EKSClusterName:
    Type: String
    MinLength: 3

  NodeImageId:
    Type: String
    Default:  ami-0c105638948525c1c
    Description: us-west-2 eks ami

  NodeInstanceType:
    Description: Node instance type
    Type: String
    Default: t3.medium

  NodeAutoScalingGroupMinSize:
    Description: Min size of ASG
    Type: Number
    Default: 1

  NodeAutoScalingGroupMaxSize:
    Description: Maximum size of ASG
    Type: Number
    Default: 3

  NodeAutoScalingGroupDesiredCapacity:
    Description: Desired capacity of ASG
    Type: Number
    Default: 2

  BootstrapArguments:
    Description: Arguments to pass to the bootstrap script. See files/bootstrap.sh in https://github.com/awslabs/amazon-eks-ami
    Type: String
    Default: ""

  NodeGroupName:
    Description: Unique name for  node group
    Type: String
    Default: nodegroup1

  Subnet1:
    Description: Private subnet 1
    Type: String

  Subnet2:
    Description: Private subnet 2
    Type: String

  VpcID:
    Description: VPC id
    Type: String

Resources:
  NodeSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for all nodes in the cluster
      VpcId: !Ref VpcID
      SecurityGroupEgress:
        - Description: Worker nodes communication to any service
          CidrIp: 0.0.0.0/0
          IpProtocol: "-1"
          FromPort: 0
          ToPort: 65535

  NodeSecurityGroupIngress:
    Type: AWS::EC2::SecurityGroupIngress
    Properties:
      Description: Allow node to communicate with each other
      GroupId: !Ref NodeSecurityGroup
      SourceSecurityGroupId: !Ref NodeSecurityGroup
      IpProtocol: "-1"
      FromPort: 0
      ToPort: 65535

  NodeSecurityGroupFromControlPlaneIngress:
    Type: AWS::EC2::SecurityGroupIngress
    Properties:
      Description: Allow worker Kubelets and pods to receive communication from the cluster control plane
      GroupId: !Ref NodeSecurityGroup
      SourceSecurityGroupId:
        Fn::ImportValue: !Sub ${EKSClusterName}-cluster-ClusterControlPlaneSecurityGroup
      IpProtocol: tcp
      FromPort: 1025
      ToPort: 65535

  NodeSecurityGroupFromControlPlaneOn443Ingress:
    Type: AWS::EC2::SecurityGroupIngress
    Properties:
      Description: Allow pods running extension API servers on port 443 to receive communication from cluster control plane
      GroupId: !Ref NodeSecurityGroup
      SourceSecurityGroupId:
        Fn::ImportValue: !Sub ${EKSClusterName}-cluster-ClusterControlPlaneSecurityGroup
      IpProtocol: tcp
      FromPort: 443
      ToPort: 443

  ControlPlaneEgressToNodeSecurityGroupOn443:
    Type: AWS::EC2::SecurityGroupEgress
    Properties:
      Description: Allow the cluster control plane to communicate with pods running extension API servers on port 443
      GroupId:
        Fn::ImportValue: !Sub ${EKSClusterName}-cluster-ClusterControlPlaneSecurityGroup
      DestinationSecurityGroupId: !Ref NodeSecurityGroup
      IpProtocol: tcp
      FromPort: 443
      ToPort: 443

  ClusterControlPlaneSecurityGroupIngress:
    Type: AWS::EC2::SecurityGroupIngress
    Properties:
      Description: Allow pods to communicate with the cluster API Server
      GroupId:
        Fn::ImportValue: !Sub ${EKSClusterName}-cluster-ClusterControlPlaneSecurityGroup
      SourceSecurityGroupId: !Ref NodeSecurityGroup
      IpProtocol: tcp
      ToPort: 443
      FromPort: 443

  ControlPlaneEgressToNodeSecurityGroup:
    Type: AWS::EC2::SecurityGroupEgress
    Properties:
      Description: Allow the cluster control plane to communicate with worker Kubelet and pods
      GroupId:
        Fn::ImportValue: !Sub ${EKSClusterName}-cluster-ClusterControlPlaneSecurityGroup
      DestinationSecurityGroupId: !Ref NodeSecurityGroup
      IpProtocol: tcp
      FromPort: 1025
      ToPort: 65535

  ManagedNodeGroup:
    Type: AWS::EKS::Nodegroup
    Properties:
      ClusterName: !Ref EKSClusterName
      NodegroupName: !Ref NodeGroupName
      ScalingConfig:
        DesiredSize: !Ref NodeAutoScalingGroupDesiredCapacity
        MaxSize: !Ref NodeAutoScalingGroupMaxSize
        MinSize: !Ref NodeAutoScalingGroupMinSize
      NodeRole:
        Fn::ImportValue: !Sub ${EKSClusterName}-iam-NodeInstanceRoleARN
      LaunchTemplate:
        Id: !Ref NodeLaunchTemplate
        Version: !GetAtt NodeLaunchTemplate.LatestVersionNumber
      ForceUpdateEnabled: true
      Subnets:
          - !Ref Subnet1
          - !Ref Subnet2
      Tags:
        Name: !Sub "${EKSClusterName}-${NodeGroupName}-Node"
  NodeLaunchTemplate:
    Type: AWS::EC2::LaunchTemplate
    Properties:
      LaunchTemplateData:
        ImageId: !Ref NodeImageId
        InstanceType: !Ref NodeInstanceType
        SecurityGroupIds:
          - !Ref NodeSecurityGroup
        UserData:
          Fn::Base64: !Sub |
            #!/bin/bash -xe

            # bootstrap cluster
            /etc/eks/bootstrap.sh ${EKSClusterName} ${BootstrapArguments}
            /opt/aws/bin/cfn-signal --exit-code $? \
                    --stack  ${AWS::StackName} \
                    --resource ManagedNodeGroup  \
                    --region ${AWS::Region}

Outputs:
  NodeSecurityGroup:
    Export:
      Name: !Sub "${AWS::StackName}-NodeSecurityGroup"
    Description: The security group for the node group
    Value: !Ref NodeSecurityGroup

```

11 create ansible/ansible.cfg if does not exist already
```
[defaults]
bin_ansible_callbacks =True
display_skipped_hosts = no
host_key_checking = False
stdout_callback = yaml


[inventory]
enable_plugins = host_list, script, yaml, ini, auto
```

12 create ansible/noble_eksctl_cluster.yaml
```
- hosts: local

  vars_files:
  - group_vars/eks

  roles:
  - eks
```
You should have the following structure:



13 Run it with(you can add it to Makefile):
```
cd ansible && ansible-playbook -i inventory/local noble_eksctl_cluster.yaml --verbose
```
