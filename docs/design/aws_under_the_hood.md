# Peeking under the hood of Kubernetes on AWS

This document provides high-level insight into how Kubernetes works on AWS and
maps to AWS objects. We assume that you are familiar with AWS.

We encourage you to use [kube-up](../getting-started-guides/aws.md) to create
clusters on AWS. We recommend that you avoid manual configuration but are aware
that sometimes it's the only option.

Tip: You should open an issue and let us know what enhancements can be made to
the scripts to better suit your needs.

That said, it's also useful to know what's happening under the hood when
Kubernetes clusters are created on AWS. This can be particularly useful if
problems arise or in circumstances where the provided scripts are lacking and
you manually created or configured your cluster.

**Table of contents:**
 * [Architecture overview](#architecture-overview)
 * [Storage](#storage)
 * [Auto Scaling group](#auto-scaling-group)
 * [Networking](#networking)
 * [NodePort and LoadBalancer services](#nodeport-and-loadbalancer-services)
 * [Identity and access management (IAM)](#identity-and-access-management-iam)
 * [Tagging](#tagging)
 * [AWS objects](#aws-objects)
 * [Manual infrastructure creation](#manual-infrastructure-creation)
 * [Instance boot](#instance-boot)

### Architecture overview

Kubernetes is a cluster of several machines that consists of a Kubernetes
master and a set number of nodes (previously known as 'nodes') for which the
master is responsible. See the [Architecture](architecture.md) topic for
more details.

By default on AWS:

* Instances run Ubuntu 15.04 (the official AMI).  It includes a sufficiently
  modern kernel that pairs well with Docker and doesn't require a
  reboot. (The default SSH user is `ubuntu` for this and other ubuntu images.)
* Nodes use aufs instead of ext4 as the filesystem / container storage (mostly
  because this is what Google Compute Engine uses).

You can override these defaults by passing different environment variables to
kube-up.

### Storage

AWS supports persistent volumes by using [Elastic Block Store (EBS)](../user-guide/volumes.md#awselasticblockstore).
These can then be attached to pods that should store persistent data (e.g. if
you're running a database).

By default, nodes in AWS use [instance storage](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/InstanceStorage.html)
unless you create pods with persistent volumes
[(EBS)](../user-guide/volumes.md#awselasticblockstore). In general, Kubernetes
containers do not have persistent storage unless you attach a persistent
volume, and so nodes on AWS use instance storage.  Instance storage is cheaper,
often faster, and historically more reliable. Unless you can make do with
whatever space is left on your root partition, you must choose an instance type
that provides you with sufficient instance storage for your needs.

To configure Kubernetes to use EBS storage, pass the environment variable
`KUBE_AWS_STORAGE=ebs` to kube-up.

Note: The master uses a persistent volume ([etcd](architecture.md#etcd)) to
track its state. Similar to nodes, containers are mostly run against instance
storage, except that we repoint some important data onto the persistent volume.

The default storage driver for Docker images is aufs. Specifying btrfs (by
passing the environment variable `DOCKER_STORAGE=btrfs` to kube-up) is also a
good choice for a filesystem. btrfs is relatively reliable with Docker and has
improved its reliability with modern kernels. It can easily span multiple
volumes, which is particularly useful when we are using an instance type with
multiple ephemeral instance disks.

### Auto Scaling group

Nodes (but not the master) are run in an
[Auto Scaling group](http://docs.aws.amazon.com/AutoScaling/latest/DeveloperGuide/AutoScalingGroup.html)
on AWS. Currently auto-scaling (e.g. based on CPU) is not actually enabled
([#11935](http://issues.k8s.io/11935)). Instead, the Auto Scaling group means
that AWS will relaunch any nodes that are terminated.

We do not currently run the master in an AutoScalingGroup, but we should
([#11934](http://issues.k8s.io/11934)).

### Networking

Kubernetes uses an IP-per-pod model. This means that a node, which runs many
pods, must have many IPs. AWS uses virtual private clouds (VPCs) and advanced
routing support so each pod is assigned a /24 CIDR. The assigned CIDR is then
configured to route to an instance in the VPC routing table.

It is also possible to use overlay networking on AWS, but that is not the
default configuration of the kube-up script.

### NodePort and LoadBalancer services

Kubernetes on AWS integrates with [Elastic Load Balancing
(ELB)](http://docs.aws.amazon.com/AutoScaling/latest/DeveloperGuide/US_SetUpASLBApp.html).
When you create a service with `Type=LoadBalancer`, Kubernetes (the
kube-controller-manager) will create an ELB, create a security group for the
ELB which allows access on the service ports, attach all the nodes to the ELB,
and modify the security group for the nodes to allow traffic from the ELB to
the nodes.  This traffic reaches kube-proxy where it is then forwarded to the
pods.

ELB has some restrictions:
* ELB requires that all nodes listen on a single port,
* ELB acts as a forwarding proxy (i.e. the source IP is not preserved, but see below
on ELB annotations for pods speaking HTTP).

To work with these restrictions, in Kubernetes, [LoadBalancer
services](../user-guide/services.md#type-loadbalancer) are exposed as
[NodePort services](../user-guide/services.md#type-nodeport).  Then
kube-proxy listens externally on the cluster-wide port that's assigned to
NodePort services and forwards traffic to the corresponding pods.

For example, if we configure a service of Type LoadBalancer with a
public port of 80:
* Kubernetes will assign a NodePort to the service (e.g. port 31234)
* ELB is configured to proxy traffic on the public port 80 to the NodePort
assigned to the service (in this example port 31234).
* Then any in-coming traffic that ELB forwards to the NodePort (31234)
is recognized by kube-proxy and sent to the correct pods for that service.

Note that we do not automatically open NodePort services in the AWS firewall
(although we do open LoadBalancer services). This is because we expect that
NodePort services are more of a building block for things like inter-cluster
services or for LoadBalancer. To consume a NodePort service externally, you
will likely have to open the port in the node security group
(`kubernetes-node-<clusterid>`).

For SSL support, starting with 1.3 two annotations can be added to a service:

```
service.beta.kubernetes.io/aws-load-balancer-ssl-cert=arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012
```

The first specifies which certificate to use. It can be either a
certificate from a third party issuer that was uploaded to IAM or one created
within AWS Certificate Manager.

```
service.beta.kubernetes.io/aws-load-balancer-backend-protocol=(https|http|ssl|tcp)
```

The second annotation specifies which protocol a pod speaks. For HTTPS and
SSL, the ELB will expect the pod to authenticate itself over the encrypted
connection.

HTTP and HTTPS will select layer 7 proxying: the ELB will terminate
the connection with the user, parse headers and inject the `X-Forwarded-For`
header with the user's IP address (pods will only see the IP address of the
ELB at the other end of its connection) when forwarding requests.

TCP and SSL will select layer 4 proxying: the ELB will forward traffic without
modifying the headers.

### Identity and Access Management (IAM)

kube-proxy sets up two IAM roles, one for the master called
[kubernetes-master](../../cluster/aws/templates/iam/kubernetes-master-policy.json)
and one for the nodes called
[kubernetes-node](../../cluster/aws/templates/iam/kubernetes-minion-policy.json).

The master is responsible for creating ELBs and configuring them, as well as
setting up advanced VPC routing. Currently it has blanket permissions on EC2,
along with rights to create and destroy ELBs.

The nodes do not need a lot of access to the AWS APIs. They need to download
a distribution file, and then are responsible for attaching and detaching EBS
volumes from itself.

The node policy is relatively minimal. In 1.2 and later, nodes can retrieve ECR
authorization tokens, refresh them every 12 hours if needed, and fetch Docker
images from it, as long as the appropriate permissions are enabled. Those in
[AmazonEC2ContainerRegistryReadOnly](http://docs.aws.amazon.com/AmazonECR/latest/userguide/ecr_managed_policies.html#AmazonEC2ContainerRegistryReadOnly),
without write access, should suffice. The master policy is probably overly
permissive. The security conscious may want to lock-down the IAM policies
further ([#11936](http://issues.k8s.io/11936)).

We should make it easier to extend IAM permissions and also ensure that they
are correctly configured ([#14226](http://issues.k8s.io/14226)).

### Tagging

All AWS resources are tagged with a tag named "KubernetesCluster", with a value
that is the unique cluster-id. This tag is used to identify a particular
'instance' of Kubernetes, even if two clusters are deployed into the same VPC.
Resources are considered to belong to the same cluster if and only if they have
the same value in the tag named "KubernetesCluster". (The kube-up script is
not configured to create multiple clusters in the same VPC by default, but it
is possible to create another cluster in the same VPC.)

Within the AWS cloud provider logic, we filter requests to the AWS APIs to
match resources with our cluster tag. By filtering the requests, we ensure
that we see only our own AWS objects.

**Important:** If you choose not to use kube-up, you must pick a unique
cluster-id value, and ensure that all AWS resources have a tag with
`Name=KubernetesCluster,Value=<clusterid>`.

### AWS objects

The kube-up script does a number of things in AWS:
* Creates an S3 bucket (`AWS_S3_BUCKET`) and then copies the Kubernetes
distribution and the salt scripts into it. They are made world-readable and the
HTTP URLs are passed to instances; this is how Kubernetes code gets onto the
machines.
* Creates two IAM profiles based on templates in [cluster/aws/templates/iam](../../cluster/aws/templates/iam/):
    * `kubernetes-master` is used by the master.
    * `kubernetes-node` is used by nodes.
* Creates an AWS SSH key named `kubernetes-<fingerprint>`. Fingerprint here is
the OpenSSH key fingerprint, so that multiple users can run the script with
different keys and their keys will not collide (with near-certainty). It will
use an existing key if one is found at `AWS_SSH_KEY`, otherwise it will create
one there. (With the default Ubuntu images, if you have to SSH in: the user is
`ubuntu` and that user can `sudo`).
* Creates a VPC for use with the cluster (with a CIDR of 172.20.0.0/16) and
enables the `dns-support` and `dns-hostnames` options.
* Creates an internet gateway for the VPC.
* Creates a route table for the VPC, with the internet gateway as the default
route.
* Creates a subnet (with a CIDR of 172.20.0.0/24) in the AZ `KUBE_AWS_ZONE`
(defaults to us-west-2a). Currently, each Kubernetes cluster runs in a
single AZ on AWS. Although, there are two philosophies in discussion on how to
achieve High Availability (HA):
    * cluster-per-AZ: An independent cluster for each AZ, where each cluster
is entirely separate.
    * cross-AZ-clusters: A single cluster spans multiple AZs.
The debate is open here, where cluster-per-AZ is discussed as more robust but
cross-AZ-clusters are more convenient.
* Associates the subnet to the route table
* Creates security groups for the master (`kubernetes-master-<clusterid>`)
and the nodes (`kubernetes-node-<clusterid>`).
* Configures security groups so that masters and nodes can communicate. This
includes intercommunication between masters and nodes, opening SSH publicly
for both masters and nodes, and opening port 443 on the master for the HTTPS
API endpoints.
* Creates an EBS volume for the master of size `MASTER_DISK_SIZE` and type
`MASTER_DISK_TYPE`.
* Launches a master with a fixed IP address (172.20.0.9) that is also
configured for the security group and all the necessary IAM credentials. An
instance script is used to pass vital configuration information to Salt. Note:
The hope is that over time we can reduce the amount of configuration
information that must be passed in this way.
* Once the instance is up, it attaches the EBS volume and sets up a manual
routing rule for the internal network range (`MASTER_IP_RANGE`, defaults to
10.246.0.0/24).
* For auto-scaling, on each nodes it creates a launch configuration and group.
The name for both is <*KUBE_AWS_INSTANCE_PREFIX*>-node-group. The default
name is kubernetes-node-group. The auto-scaling group has a min and max size
that are both set to NUM_NODES. You can change the size of the auto-scaling
group to add or remove the total number of nodes from within the AWS API or
Console. Each nodes self-configures, meaning that they come up; run Salt with
the stored configuration; connect to the master; are assigned an internal CIDR;
and then the master configures the route-table with the assigned CIDR. The
kube-up script performs a health-check on the nodes but it's a self-check that
is not required.

If attempting this configuration manually, it is recommend to follow along
with the kube-up script, and being sure to tag everything with a tag with name
`KubernetesCluster` and value set to a unique cluster-id. Also, passing the
right configuration options to Salt when not using the script is tricky: the
plan here is to simplify this by having Kubernetes take on more node
configuration, and even potentially remove Salt altogether.

### Manual infrastructure creation

While this work is not yet complete, advanced users might choose to manually
create certain AWS objects while still making use of the kube-up script (to
configure Salt, for example). These objects can currently be manually created:
* Set the `AWS_S3_BUCKET` environment variable to use an existing S3 bucket.
* Set the `VPC_ID` environment variable to reuse an existing VPC.
* Set the `SUBNET_ID` environment variable to reuse an existing subnet.
* If your route table has a matching `KubernetesCluster` tag, it will be reused.
* If your security groups are appropriately named, they will be reused.

Currently there is no way to do the following with kube-up:
* Use an existing AWS SSH key with an arbitrary name.
* Override the IAM credentials in a sensible way
([#14226](http://issues.k8s.io/14226)).
* Use different security group permissions.
* Configure your own auto-scaling groups.

If any of the above items apply to your situation, open an issue to request an
enhancement to the kube-up script. You should provide a complete description of
the use-case, including all the details around what you want to accomplish.

### Instance boot

The instance boot procedure is currently pretty complicated, primarily because
we must marshal configuration from Bash to Salt via the AWS instance script.
As we move more post-boot configuration out of Salt and into Kubernetes, we
will hopefully be able to simplify this.

When the kube-up script launches instances, it builds an instance startup
script which includes some configuration options passed to kube-up, and
concatenates some of the scripts found in the cluster/aws/templates directory.
These scripts are responsible for mounting and formatting volumes, downloading
Salt and Kubernetes from the S3 bucket, and then triggering Salt to actually
install Kubernetes.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/aws_under_the_hood.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
