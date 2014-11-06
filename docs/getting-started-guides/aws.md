# Getting started on Amazon EC2

The example below creates an elastic Kubernetes cluster with 3 worker nodes and a master.

<<<<<<< HEAD
## Highlights

* Cluster bootstrapping using [cloud-config](https://coreos.com/docs/cluster-management/setup/cloudinit-cloud-config)
* Cross container networking with [flannel](https://github.com/coreos/flannel#flannel)
* Auto worker registration with [kube-register](https://github.com/kelseyhightower/kube-register#kube-register)
* Kubernetes v0.4.2 [official binaries](https://github.com/GoogleCloudPlatform/kubernetes/releases/tag/v0.4.2)

## Prerequisites

* [kubecfg CLI](aws/kubecfg.md)
* [aws CLI](http://aws.amazon.com/cli)
* CoreOS 490.0.0+

## Starting a Cluster

### Cloud Formation

The [cloudformation-template.json](aws/cloudformation-template.json) can be used to bootstrap a Kubernetes cluster with a single command.

```
aws cloudformation create-stack --stack-name kubernetes --region us-west-2 \
--template-body file://aws/cloudformation-template.json \
--parameters ParameterKey=KeyPair,ParameterValue=<keypair>
```

It will take a few minutes for the entire stack to come up. You can monitor the stack progress with the following command:

```
aws cloudformation describe-stack-events --stack-name kubernetes
```

> Record the Kubernetes Master IP address
=======
1. You need an AWS account. Visit [http://aws.amazon.com](http://aws.amazon.com) to get started
2. Install and configure [AWS Command Line Interface](http://aws.amazon.com/cli)

### Run from a binary release
>>>>>>> Update the AWS config.

1. Download the [binary release](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/getting-started-guides/binary_release.md)
2. Unpack the archive and ```cd kubernetes```
3. Turn up the cluster:
```
<<<<<<< HEAD
aws cloudformation describe-stacks --stack-name kubernetes
=======
export KUBERNETES_PROVIDER=aws
cluster/kube-up.sh
>>>>>>> Update the AWS config.
```

[Skip to kubecfg client configuration](#configure-the-kubecfg-ssh-tunnel)

<<<<<<< HEAD
### Manually

The following commands use the CoreOS 490.0.0 alpha AMI `ami-e18dc5d1` from the `us-west-2` region. For a list of different regions and corresponding AMI IDs see the [CoreOS EC2 cloud provider documentation](https://coreos.com/docs/running-coreos/cloud-providers/ec2/#choosing-a-channel).

#### Create the Kubernetes Security Group

```
aws ec2 create-security-group --group-name kubernetes --description "Kubernetes Security Group"
aws ec2 authorize-security-group-ingress --group-name kubernetes --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name kubernetes --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name kubernetes --source-security-group-name kubernetes
```

#### Save the master and node cloud-configs

* [master.yaml](aws/cloud-configs/master.yaml)
* [node.yaml](aws/cloud-configs/node.yaml)

#### Launch the master

```
aws ec2 run-instances --image-id ami-e18dc5d1 --key-name <keypair> \
--region us-west-2 --security-groups kubernetes --instance-type m3.medium \
--user-data file://master.yaml
```

> Record the `InstanceId` for the master.

Gather the public and private IPs for the master node:

```
aws ec2 describe-instances --instance-id <instance-id>
```

```
{
    "Reservations": [
        {
            "Instances": [
                {
                    "PublicDnsName": "ec2-54-68-97-117.us-west-2.compute.amazonaws.com", 
                    "RootDeviceType": "ebs", 
                    "State": {
                        "Code": 16, 
                        "Name": "running"
                    }, 
                    "PublicIpAddress": "54.68.97.117", 
                    "PrivateIpAddress": "172.31.9.9", 
...
```

#### Update the node.yaml cloud-config

Edit `node.yaml` and replace all instances of `<master-private-ip>` with the **private** IP address of the master node.

### Launch 3 worker nodes
=======
### Running examples

Take a look at [next steps](https://github.com/GoogleCloudPlatform/kubernetes#where-to-go-next)
>>>>>>> Update the AWS config.

### Tearing down the cluster
```
<<<<<<< HEAD
aws ec2 run-instances --count 3 --image-id ami-e18dc5d1 --key-name <keypair> \
--region us-west-2 --security-groups kubernetes --instance-type m3.medium \
--user-data file://node.yaml
```

### Add additional worker nodes

```
aws ec2 run-instances --count 1 --image-id ami-e18dc5d1 --key-name <keypair> \
--region us-west-2 --security-groups kubernetes --instance-type m3.medium \
--user-data file://node.yaml
```

### Configure the kubecfg SSH tunnel

This command enables secure communication between the kubecfg client and the Kubernetes API.

```
ssh -f -nNT -L 8080:127.0.0.1:8080 core@<master-public-ip>
```

### Listing worker nodes

Once the worker instances have fully booted, they will be automatically registered with the Kubernetes API server by the kube-register service running on the master node. It may take a few mins.

```
kubecfg list minions
```

## Starting a simple pod

Create a pod manifest: `pod.json`

```
{
  "id": "hello",
  "kind": "Pod",
  "apiVersion": "v1beta1",
  "desiredState": {
    "manifest": {
      "version": "v1beta1",
      "id": "hello",
      "containers": [{
        "name": "hello",
        "image": "quay.io/kelseyhightower/hello",
        "ports": [{
          "containerPort": 80,
          "hostPort": 80 
        }]
      }]
    }
  },
  "labels": {
    "name": "hello",
    "environment": "testing"
  }
}
```

### Create the pod using the kubecfg command line tool

```
kubecfg -c pod.json create pods
```

### Testing

```
kubecfg list pods
```

> Record the **Host** of the pod, which should be the private IP address.

Gather the public IP address for the worker node. 

```
aws ec2 describe-instances --filters 'Name=private-ip-address,Values=<host>'
```

```
{
    "Reservations": [
        {
            "Instances": [
                {
                    "PublicDnsName": "ec2-54-68-97-117.us-west-2.compute.amazonaws.com", 
                    "RootDeviceType": "ebs", 
                    "State": {
                        "Code": 16, 
                        "Name": "running"
                    }, 
                    "PublicIpAddress": "54.68.97.117", 
...
```

Visit the public IP address in your browser to view the running pod.

### Delete the pod

```
kubecfg delete pods/hello
```
=======
cd kubernetes
cluster/kube-down.sh
```

### Cloud Formation
There is a contributed example from [CoreOS](http://www.coreos.com) using Cloud Formation.
>>>>>>> Update the AWS config.
