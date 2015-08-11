---
layout: docwithnav
title: "Getting started on Amazon EC2 with CoreOS"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Getting started on Amazon EC2 with CoreOS

The example below creates an elastic Kubernetes cluster with a custom number of worker nodes and a master.

**Warning:** contrary to the [supported procedure](aws.html), the examples below provision Kubernetes with an insecure API server (plain HTTP,
no security tokens, no basic auth). For demonstration purposes only.

## Highlights

* Cluster bootstrapping using [cloud-config](https://coreos.com/docs/cluster-management/setup/cloudinit-cloud-config/)
* Cross container networking with [flannel](https://github.com/coreos/flannel#flannel)
* Auto worker registration with [kube-register](https://github.com/kelseyhightower/kube-register#kube-register)
* Kubernetes v0.19.3 [official binaries](https://github.com/GoogleCloudPlatform/kubernetes/releases/tag/v0.19.3)

## Prerequisites

* [aws CLI](http://aws.amazon.com/cli)
* [CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/)
* [kubectl CLI](aws/kubectl.html) ([installation](aws.html#command-line-administration-tool-kubectl))

## Starting a Cluster

### CloudFormation

The [cloudformation-template.json](aws/cloudformation-template.json) can be used to bootstrap a Kubernetes cluster with a single command:

{% highlight bash %}
{% raw %}
aws cloudformation create-stack --stack-name kubernetes --region us-west-2 \
--template-body file://aws/cloudformation-template.json \
--parameters ParameterKey=KeyPair,ParameterValue=<keypair> \
             ParameterKey=ClusterSize,ParameterValue=<cluster_size> \
             ParameterKey=VpcId,ParameterValue=<vpc_id> \
             ParameterKey=SubnetId,ParameterValue=<subnet_id> \
             ParameterKey=SubnetAZ,ParameterValue=<subnet_az>
{% endraw %}
{% endhighlight %}

It will take a few minutes for the entire stack to come up. You can monitor the stack progress with the following command:

{% highlight bash %}
{% raw %}
aws cloudformation describe-stack-events --stack-name kubernetes
{% endraw %}
{% endhighlight %}

Record the Kubernetes Master IP address:

{% highlight bash %}
{% raw %}
aws cloudformation describe-stacks --stack-name kubernetes
{% endraw %}
{% endhighlight %}

[Skip to kubectl client configuration](#configure-the-kubectl-ssh-tunnel)

### AWS CLI

The following commands shall use the latest CoreOS alpha AMI for the `us-west-2` region. For a list of different regions and corresponding AMI IDs see the [CoreOS EC2 cloud provider documentation](https://coreos.com/docs/running-coreos/cloud-providers/ec2/#choosing-a-channel).

#### Create the Kubernetes Security Group

{% highlight bash %}
{% raw %}
aws ec2 create-security-group --group-name kubernetes --description "Kubernetes Security Group"
aws ec2 authorize-security-group-ingress --group-name kubernetes --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name kubernetes --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name kubernetes --source-security-group-name kubernetes
{% endraw %}
{% endhighlight %}

#### Save the master and node cloud-configs

* [master.yaml](aws/cloud-configs/master.yaml)
* [node.yaml](aws/cloud-configs/node.yaml)

#### Launch the master

*Attention:* replace `<ami_image_id>` below for a [suitable version of CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/).

{% highlight bash %}
{% raw %}
aws ec2 run-instances --image-id <ami_image_id> --key-name <keypair> \
--region us-west-2 --security-groups kubernetes --instance-type m3.medium \
--user-data file://master.yaml
{% endraw %}
{% endhighlight %}

Record the `InstanceId` for the master.

Gather the public and private IPs for the master node:

{% highlight bash %}
{% raw %}
aws ec2 describe-instances --instance-id <instance-id>
{% endraw %}
{% endhighlight %}

{% highlight json %}
{% raw %}
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
{% endraw %}
{% endhighlight %}

#### Update the node.yaml cloud-config

Edit `node.yaml` and replace all instances of `<master-private-ip>` with the **private** IP address of the master node.

### Launch 3 worker nodes

*Attention:* Replace `<ami_image_id>` below for a [suitable version of CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/#choosing-a-channel).

{% highlight bash %}
{% raw %}
aws ec2 run-instances --count 3 --image-id <ami_image_id> --key-name <keypair> \
--region us-west-2 --security-groups kubernetes --instance-type m3.medium \
--user-data file://node.yaml
{% endraw %}
{% endhighlight %}

### Add additional worker nodes

*Attention:* replace `<ami_image_id>` below for a [suitable version of CoreOS image for AWS](https://coreos.com/docs/running-coreos/cloud-providers/ec2/#choosing-a-channel).

{% highlight bash %}
{% raw %}
aws ec2 run-instances --count 1 --image-id <ami_image_id> --key-name <keypair> \
--region us-west-2 --security-groups kubernetes --instance-type m3.medium \
--user-data file://node.yaml
{% endraw %}
{% endhighlight %}

### Configure the kubectl SSH tunnel

This command enables secure communication between the kubectl client and the Kubernetes API.

{% highlight bash %}
{% raw %}
ssh -f -nNT -L 8080:127.0.0.1:8080 core@<master-public-ip>
{% endraw %}
{% endhighlight %}

### Listing worker nodes

Once the worker instances have fully booted, they will be automatically registered with the Kubernetes API server by the kube-register service running on the master node. It may take a few mins.

{% highlight bash %}
{% raw %}
kubectl get nodes
{% endraw %}
{% endhighlight %}

## Starting a simple pod

Create a pod manifest: `pod.json`

{% highlight json %}
{% raw %}
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
{% endraw %}
{% endhighlight %}

### Create the pod using the kubectl command line tool

{% highlight bash %}
{% raw %}
kubectl create -f ./pod.json
{% endraw %}
{% endhighlight %}

### Testing

{% highlight bash %}
{% raw %}
kubectl get pods
{% endraw %}
{% endhighlight %}

Record the **Host** of the pod, which should be the private IP address.

Gather the public IP address for the worker node. 

{% highlight bash %}
{% raw %}
aws ec2 describe-instances --filters 'Name=private-ip-address,Values=<host>'
{% endraw %}
{% endhighlight %}

{% highlight json %}
{% raw %}
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
{% endraw %}
{% endhighlight %}

Visit the public IP address in your browser to view the running pod.

### Delete the pod

{% highlight bash %}
{% raw %}
kubectl delete pods hello
{% endraw %}
{% endhighlight %}


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/aws-coreos.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

