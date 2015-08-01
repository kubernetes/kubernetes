<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/aws-coreos.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Getting started on Amazon EC2 with CoreOS

The example below creates an elastic Kubernetes cluster with a custom number of worker nodes and a master.

**Warning:** contrary to the [supported procedure](aws.md), the examples below provision Kubernetes with an insecure API server (plain HTTP,
no security tokens, no basic auth). For demonstration purposes only.

## Highlights

* Cluster bootstrapping using [cloud-config](https://coreos.com/docs/cluster-management/setup/cloudinit-cloud-config/)
* Cross container networking with [flannel](https://github.com/coreos/flannel#flannel)
* Auto worker registration with [kube-register](https://github.com/kelseyhightower/kube-register#kube-register)
* Kubernetes v0.19.3 [official binaries](https://github.com/GoogleCloudPlatform/kubernetes/releases/tag/v0.19.3)

## Prerequisites

* [aws CLI](http://aws.amazon.com/cli)
* [CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/)
* [kubectl CLI](aws/kubectl.md) ([installation](aws.md#command-line-administration-tool-kubectl))

## Starting a Cluster

### CloudFormation

The [cloudformation-template.json](aws/cloudformation-template.json) can be used to bootstrap a Kubernetes cluster with a single command:

```bash
aws cloudformation create-stack --stack-name kubernetes --region us-west-2 \
--template-body file://aws/cloudformation-template.json \
--parameters ParameterKey=KeyPair,ParameterValue=<keypair> \
             ParameterKey=ClusterSize,ParameterValue=<cluster_size> \
             ParameterKey=VpcId,ParameterValue=<vpc_id> \
             ParameterKey=SubnetId,ParameterValue=<subnet_id> \
             ParameterKey=SubnetAZ,ParameterValue=<subnet_az>
```

It will take a few minutes for the entire stack to come up. You can monitor the stack progress with the following command:

```bash
aws cloudformation describe-stack-events --stack-name kubernetes
```

Record the Kubernetes Master IP address:

```bash
aws cloudformation describe-stacks --stack-name kubernetes
```

[Skip to kubectl client configuration](#configure-the-kubectl-ssh-tunnel)

### AWS CLI

The following commands shall use the latest CoreOS alpha AMI for the `us-west-2` region. For a list of different regions and corresponding AMI IDs see the [CoreOS EC2 cloud provider documentation](https://coreos.com/docs/running-coreos/cloud-providers/ec2/#choosing-a-channel).

#### Create the Kubernetes Security Group

```bash
aws ec2 create-security-group --group-name kubernetes --description "Kubernetes Security Group"
aws ec2 authorize-security-group-ingress --group-name kubernetes --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name kubernetes --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name kubernetes --source-security-group-name kubernetes
```

#### Save the master and node cloud-configs

* [master.yaml](aws/cloud-configs/master.yaml)
* [node.yaml](aws/cloud-configs/node.yaml)

#### Launch the master

*Attention:* replace `<ami_image_id>` below for a [suitable version of CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/).

```bash
aws ec2 run-instances --image-id <ami_image_id> --key-name <keypair> \
--region us-west-2 --security-groups kubernetes --instance-type m3.medium \
--user-data file://master.yaml
```

Record the `InstanceId` for the master.

Gather the public and private IPs for the master node:

```bash
aws ec2 describe-instances --instance-id <instance-id>
```

```json
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
```

#### Update the node.yaml cloud-config

Edit `node.yaml` and replace all instances of `<master-private-ip>` with the **private** IP address of the master node.

### Launch 3 worker nodes

*Attention:* Replace `<ami_image_id>` below for a [suitable version of CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/#choosing-a-channel).

```bash
aws ec2 run-instances --count 3 --image-id <ami_image_id> --key-name <keypair> \
--region us-west-2 --security-groups kubernetes --instance-type m3.medium \
--user-data file://node.yaml
```

### Add additional worker nodes

*Attention:* replace `<ami_image_id>` below for a [suitable version of CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/#choosing-a-channel).

```bash
aws ec2 run-instances --count 1 --image-id <ami_image_id> --key-name <keypair> \
--region us-west-2 --security-groups kubernetes --instance-type m3.medium \
--user-data file://node.yaml
```

### Configure the kubectl SSH tunnel

This command enables secure communication between the kubectl client and the Kubernetes API.

```bash
ssh -f -nNT -L 8080:127.0.0.1:8080 core@<master-public-ip>
```

### Listing worker nodes

Once the worker instances have fully booted, they will be automatically registered with the Kubernetes API server by the kube-register service running on the master node. It may take a few mins.

```bash
kubectl get nodes
```

## Starting a simple pod

Create a pod manifest: `pod.json`

```json
{
  "apiVersion": "v1",
  "kind": "Pod",
  "metadata": {
    "name": "hello",
    "labels": {
      "name": "hello",
      "environment": "testing"
    }
  },
  "spec": {
    "containers": [{
      "name": "hello",
      "image": "quay.io/kelseyhightower/hello",
      "ports": [{
        "containerPort": 80,
        "hostPort": 80
      }]
    }]
  }
}
```

### Create the pod using the kubectl command line tool

```bash
kubectl create -f ./pod.json
```

### Testing

```bash
kubectl get pods
```

Record the **Host** of the pod, which should be the private IP address.

Gather the public IP address for the worker node.

```bash
aws ec2 describe-instances --filters 'Name=private-ip-address,Values=<host>'
```

```json
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
```

Visit the public IP address in your browser to view the running pod.

### Delete the pod

```bash
kubectl delete pods hello
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/aws-coreos.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
