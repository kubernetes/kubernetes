# AWS Cloudformation based Kubernetes cluster definition

This module provides a real-world template for creating and maintaining a Kubernetes cluster in an existing AWS VPC alongside other AWS resources.

It is intended that the user checks a customized copy of this configuration into source control alongside their other cluster configuration.


## Purpose

Use declarative configuration for creation and maintenance of a Kubernetes cluster in a real-world deployment.  Creating, maintaining, and updating cluster components using this structure should be a simple matter of making a config change and running an update.

Typically, this structure will live alongside similar configuration for other portions of the user's application stack that rely on AWS native services (e.g., SQS, RDS, Route53, etc.).

Proper hygeine dictates that these configuration files are used to drive changes to your AWS resources.  Doing so will give the user a clear history of changes via their source control system, comments regarding why infrastructure is setup the way it is, and a map of infrastructure useful in architectural discussions.


## Description

This set of configuration files and tools represents a series of CloudFormation stacks which form the layers of a typical infrastructure stack.  These are comprised of global, regional, and zonal infrastructure as follows:

### Global

These configurations will be globally unique.

- iam_config.jsonnet - Global IAM configuration including users, roles, and security policies for Kubernetes cluster resources.

### Regional

These configurations will be copied per deployment region.

- notifications.jsonnet - SNS configuration for notifications of events
- network.jsonnet - VPC definition (includes subnets & routing tables, per-zone EC2 bastion hosts, and associated security groups)
- cert_storage.jsonnet - S3 bucket for hosting your cryptographic certificates used for internal/external ETCD & Kubernetes API access.
- common_infra.jsonnet - Per-region common infrastructure; includes CloudTrail auditing= & Kubernetes master/node security groups.
- etcd_cluster.jsonnet - Manages a distributed ETCD cluster spanning zones in the given region.

### Zonal

These configurations will be copied per deployment zone.
TODO: Consider migrating to multi-zone clusters and move all of these configs to the regional level.

- kube_cluster.jsonnet - Manages a Kubernetes cluster in the zone


## Prerequisites

These configurations assume the user has already created an AWS account, has gotten the AWS command line tool working, created a SSH KeyPair for debugging nodes, and has a set of certificates to be deployed to the cluster.

### AWS CLI

### KeyPair

### Certificates

Setting up regional authentication & encryption certificates is the most complicated prerequisite.

Assuming your VPC internal DNS name is something like ```internal.mycorp.io``` (as defined in ```utils/constants.jsonnet```, you'll need:
- A trusted CA certificate
- A CA signed certificate and private key for ETCD
  - Common Name: ```etcd.internal.mycorp.io``
  - Alternates:
    - DNS: ```*.us-west-2.internal.mycorp.io```
- A CA signed certificate and private key for Kubernetes nodes to share
  - Common Name: ```kubernetes-node-us-west-2```
  - Alternates:
    - DNS: ```kubelet```
    - DNS: ```kubelet.internal.mycorp.io```
    - DNS: ```*.us-west-2.compute.internal``` - AWS-internal-dns resolvable
- A CA signed certificate and private key for Kubernetes master components
  - Common Name: ```kubernetes-master```
  - Alternates:
    - IP Address: ```10.240.0.1```
    - DNS: ```kubernetes```
    - DNS: ```kubernetes.default```
    - DNS: ```kubernetes.default.svc```
    - DNS: ```kubernetes.default.svc.cluster.local```
    - DNS: ```kubernetes-master```
    - DNS: ```*.us-west-2.internal.mycorp.io```
- CA signed certificates and private keys for each user connecting to Kubernetes
  - Common Name: ```me@mycorp.io```

If your company doesn't already have certificate infrastructure set up, you can try a free service such as tinycert.org to manage a CA and certificates.


## First Deployment

The first step is to make a local copy of these files and prepare them for checkin to your source repository. These configuration files will dictate the desired state of your cluster at all times.

### Customize

Edit utils/constants.jsonnet and utils/network_cidrs.jsonnet for your environment

### Certificates

Ensure you have created certificates (instructions TBD)

### Deployment

Perform first-time deployment of the cluster in layer order.

#### Lower layers
- ```./cloud_formation_config.py global/iam_config.jsonnet create```
- ```./cloud_formation_config.py us-west-2/notifications.jsonnet create```
- ```./cloud_formation_config.py us-west-2/network.jsonnet create```
- ```./cloud_formation_config.py us-west-2/cert_storage.jsonnet create```

#### Certificate upload

At this stage, you'll want to upload your certifiates to the S3 bucket created for you by the cert_storage layer.
Your certificates should be uploaded in the following pattern into your certs S3 bucket:
- ```etcd```
  - ```ca.crt``` - Your CA certificate
  - ```etcd-server.crt``` - Your ETCD certificate chain
  - ```etcd-server.key``` - Your ETCD certificate private key
- ```kubelet```
  - ```ca.crt``` - Your CA certificate
  - ```kube-kubelet.crt``` - Your Kubelet certificate chain
  - ```kube-kubelet.key``` - Your kubelet certificate private key
- ```kubemaster```
  - ```ca.crt``` - Your CA certificate
  - ```kube-kubelet.crt``` - Your Kubelet certificate chain
  - ```kube-kubelet.key``` - Your kubelet certificate private key
  - ```kube-master.crt``` - Your Kubernetes master certificate chain
  - ```kube-master.key``` - Your Kubernetes master certificate private key

#### Upper layers
- ```./cloud_formation_config.py us-west-2/common_infra.jsonnet create```
- ```./cloud_formation_config.py us-west-2/etcd_cluster.jsonnet create```
- ```./cloud_formation_config.py us-west-2a/kube_cluster.jsonnet create```


### Review

At this point, you'll have deployed:

- Global IAM configuration
- SNS notifications for cluster events
- CloudTrail auditing for historical records of changes to your cluster(s)
- An EC2 bastion host in each zone which allows external Internet access for your CoreOS hosts to self-upgrade and restricted access for you to SSH into your cluster
- An EC2 ETCD node in each zone, participating in a region-wide distributed cluster
- An EC2 instance hosting the Kubernetes master components in each zone hosting a cluster
- A self-healing autoscaling group managing a set of EC2 instances which host your self-registering Kubernetes nodes.
- A set of SecurityGroups which enable access to your Kubernetes master and cluster components only from your ```corp_offices``` defined public IPs.


## Example operations

Examples of some common updates to the cluster

### Update Kubernetes version

- Edit ```templates/kube_cluster_template.jsonnet``` and change the ```kube_version```
- Re-deploy your cluster ```./cloud_formation_config.py us-west-2a/kube_cluster.jsonnet update```

You'll see cloudformation updating your Kubernetes master and Kubernetes node auto-scaling group, which will slowly roll-out updated Kubernetes node hosts over the course of a few minutes - ensuring minimal to zero downtime for your cluster.

### Change the public IP of one of your development sites

- Edit ```utils/network_cidrs.jsonnet``` and change the ```corp_offices``` map.
- Re-deploy your network layer with ```/cloud_formation_config.py us-west-2/network.jsonnet update```

You should see no downtime while Cloudformation applies the change to your VPC and related security groups.

