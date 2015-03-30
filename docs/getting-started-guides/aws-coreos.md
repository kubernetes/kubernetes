__Note (11/21/2014): This mostly works, but doesn't currently register minions correctly.__


# Getting started on Amazon EC2

The example below creates an elastic LMKTFY cluster with 3 worker nodes and a master.

## Highlights

* Cluster bootstrapping using [cloud-config](https://coreos.com/docs/cluster-management/setup/cloudinit-cloud-config)
* Cross container networking with [flannel](https://github.com/coreos/flannel#flannel)
* Auto worker registration with [lmktfy-register](https://github.com/kelseyhightower/lmktfy-register#lmktfy-register)
* LMKTFY v0.10.1 [official binaries](https://github.com/GoogleCloudPlatform/lmktfy/releases/tag/v0.10.1)

## Prerequisites

* [lmktfyctl CLI](aws/lmktfyctl.md)
* [aws CLI](http://aws.amazon.com/cli)
* [CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/#choosing-a-channel)

## Starting a Cluster

### Cloud Formation

The [cloudformation-template.json](aws/cloudformation-template.json) can be used to bootstrap a LMKTFY cluster with a single command.

```
aws cloudformation create-stack --stack-name lmktfy --region us-west-2 \
--template-body file://aws/cloudformation-template.json \
--parameters ParameterKey=KeyPair,ParameterValue=<keypair>
```

It will take a few minutes for the entire stack to come up. You can monitor the stack progress with the following command:

```
aws cloudformation describe-stack-events --stack-name lmktfy
```

> Record the LMKTFY Master IP address

```
aws cloudformation describe-stacks --stack-name lmktfy
```

[Skip to lmktfyctl client configuration](#configure-the-lmktfyctl-ssh-tunnel)

### Manually

The following commands shall use the latest CoreOS alpha AMI for the `us-west-2` region. For a list of different regions and corresponding AMI IDs see the [CoreOS EC2 cloud provider documentation](https://coreos.com/docs/running-coreos/cloud-providers/ec2/#choosing-a-channel).

#### Create the LMKTFY Security Group

```
aws ec2 create-security-group --group-name lmktfy --description "LMKTFY Security Group"
aws ec2 authorize-security-group-ingress --group-name lmktfy --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name lmktfy --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name lmktfy --source-security-group-name lmktfy
```

#### Save the master and node cloud-configs

* [master.yaml](aws/cloud-configs/master.yaml)
* [node.yaml](aws/cloud-configs/node.yaml)

#### Launch the master

*Attention:* Replace ```<ami_image_id>``` below for a [suitable version of CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/#choosing-a-channel).

```
aws ec2 run-instances --image-id <ami_image_id> --key-name <keypair> \
--region us-west-2 --security-groups lmktfy --instance-type m3.medium \
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

*Attention:* Replace ```<ami_image_id>``` below for a [suitable version of CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/#choosing-a-channel).

```
aws ec2 run-instances --count 3 --image-id <ami_image_id> --key-name <keypair> \
--region us-west-2 --security-groups lmktfy --instance-type m3.medium \
--user-data file://node.yaml
```

### Add additional worker nodes

*Attention:* Replace ```<ami_image_id>``` below for a [suitable version of CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/#choosing-a-channel).

```
aws ec2 run-instances --count 1 --image-id <ami_image_id> --key-name <keypair> \
--region us-west-2 --security-groups lmktfy --instance-type m3.medium \
--user-data file://node.yaml
```

### Configure the lmktfyctl SSH tunnel

This command enables secure communication between the lmktfyctl client and the LMKTFY API.

```
ssh -f -nNT -L 8080:127.0.0.1:8080 core@<master-public-ip>
```

### Listing worker nodes

Once the worker instances have fully booted, they will be automatically registered with the LMKTFY API server by the lmktfy-register service running on the master node. It may take a few mins.

```
lmktfyctl get nodes
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

### Create the pod using the lmktfyctl command line tool

```
lmktfyctl create -f pod.json
```

### Testing

```
lmktfyctl get pods
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
lmktfyctl delete pods hello
```
