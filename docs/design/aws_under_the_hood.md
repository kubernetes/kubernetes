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
[here](http://releases.k8s.io/release-1.0/docs/design/aws_under_the_hood.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Peeking under the hood of kubernetes on AWS

We encourage you to use kube-up (or CloudFormation) to create a cluster.  But
it is useful to know what is being created: for curiosity, to understand any
problems that may arise, or if you have to create things manually because the
scripts are unsuitable for any reason.  We don't recommend manual configuration
(please file an issue and let us know what's missing if there's something you
need) but sometimes it is the only option.

This document sets out to document how kubernetes on AWS maps to AWS objects.
Familiarity with AWS is assumed.

### Top-level

Kubernetes consists of a single master node, and a collection of minion nodes.
Other documents describe the general architecture of Kubernetes (all nodes run
Docker; the kubelet agent runs on each node and launches containers; the
kube-proxy relays traffic between the nodes etc).

By default on AWS:

* Instances run Ubuntu 15.04 (the official AMI).  It includes a sufficiently
  modern kernel to give a good experience with Docker, it doesn't require a
  reboot.  (The default SSH user is `ubuntu` for this and other ubuntu images)
* By default we run aufs over ext4 as the filesystem / container storage on the
  nodes (mostly because this is what GCE uses).

These defaults can be changed by passing different environment variables to
kube-up.

### Storage

AWS does support persistent volumes via EBS.  These can then be attached to
pods that should store persistent data (e.g. if you're running a database).

Minions do not have persistent volumes otherwise.  In general, kubernetes
containers do not have persistent storage unless you attach a persistent
volume, and so minions on AWS use instance storage.  Instance storage is
cheaper, often faster, and historically more reliable.  This does mean that you
should pick an instance type that has sufficient instance storage, unless you
can make do with whatever space is left on your root partition.

The master _does_ have a persistent volume attached to it.  Containers are
mostly run against instance storage, just like the minions, except that we
repoint some important data onto the peristent volume.

By default we use aufs over ext4.  `DOCKER_STORAGE=btrfs` is also a good choice
for a filesystem: it is relatively reliable with Docker; btrfs itself is much
more reliable than it used to be with modern kernels.  It can easily span
multiple volumes, which is particularly useful when we are using an instance
type with multiple ephemeral instance disks.

### AutoScaling

We run the minions in an AutoScalingGroup.  Currently auto-scaling (e.g. based
on CPU) is not actually enabled (#11935).  Instead, the auto-scaling group
means that AWS will relaunch any minions that are terminated.

We do not currently run the master in an AutoScalingGroup, but we should
(#11934)

### Networking

Kubernetes uses an IP-per-pod model.  This means that a node, which runs many
pods, must have many IPs.  The way we implement this on AWS is to use VPCs and
the advanced routing support that it allows.  Each pod is assigned a /24 CIDR;
then this CIDR is configured to route to an instance in the VPC routing table.

It is also possible to use overlay networking on AWS, but the default kube-up
configuration does not.

### NodePort & LoadBalancing

Kubernetes on AWS integrates with ELB.  When you create a service with
Type=LoadBalancer, kubernetes (the kube-controller-manager) will create an ELB,
create a security group for the ELB which allows access on the service ports,
attach all the minions to the ELB, and modify the security group for the
minions to allow traffic from the ELB to the minions.  This traffic reaches
kube-proxy where it is then forwarded to the pods.

ELB requires that all minions listen on a single port, and it acts as a layer-7
forwarding proxy (i.e. the source IP is not preserved).  It is not trivial for
kube-proxy to recognize the traffic therefore.  So, LoadBalancer services are
also exposed as NodePort services.  For NodePort services, a cluster-wide port
is assigned by kubernetes to the service, and kube-proxy listens externally on
that port on every minion, and forwards traffic to the pods.  So for a
load-balanced service, ELB is configured to proxy traffic on the public port
(e.g. port 80) to the NodePort assigned to the service (e.g. 31234), kube-proxy
recognizes the traffic coming to the NodePort by the inbound port number, and
send it to the correct pods for the service.

Note that we do not automatically open NodePort services in the AWS firewall
(although we do open LoadBalancer services).  This is because we expect that
NodePort services are more of a building block for things like inter-cluster
services or for LoadBalancer.  To consume a NodePort service externally, you
will likely have to open the port in the minion security group
(`kubernetes-minion-<clusterid>`).

### IAM

kube-proxy sets up two IAM roles, one for the master called
(kubernetes-master)[cluster/aws/templates/iam/kubernetes-master-policy.json]
and one for the minions called
(kubernetes-minion)[cluster/aws/templates/iam/kubernetes-minion-policy.json].

The master is responsible for creating ELBs and configuring them, as well as
setting up advanced VPC routing.  Currently it has blanket permissions on EC2,
along with rights to create and destroy ELBs.

The minion does not need a lot of access to the AWS APIs.  It needs to download
a distribution file, and then it is responsible for attaching and detaching EBS
volumes to itself.

The minion policy is relatively minimal.  The master policy is probably overly
permissive.  The security concious may want to lock-down the IAM policies
further (#11936)

We should make it easier to extend IAM permissions and also ensure that they
are correctly configured (#???)

### Tagging

All AWS resources are tagged with a tag named "KuberentesCluster".  This tag is
used to identify a particular 'instance' of Kubernetes, even if two clusters
are deployed into the same VPC.  (The script doesn't do this by default, but it
can be done.)

Within the AWS cloud provider logic, we filter requests to the AWS APIs to
match resources with our cluster tag.  So we only see our own AWS objects.

If you choose not to use kube-up, you must tag everything with a
KubernetesCluster tag with a unique per-cluster value.


# AWS Objects

The kube-up script does a number of things in AWS:

* Creates an S3 bucket (`AWS_S3_BUCKET`) and copy the kubernetes distribution
  and the salt scripts into it.  They are made world-readable and the HTTP URLs
are passed to instances; this is how kubernetes code gets onto the machines.
* Creates two IAM profiles based on templates in `cluster/aws/templates/iam`.
  `kubernetes-master` is used by the master node; `kubernetes-minion` is used
by minion nodes.
* Creates an AWS SSH key named `kubernetes-<fingerprint>`.  Fingerprint here is
  the OpenSSH key fingerprint, so that multiple users can run the script with
different keys and their keys will not collide (with near-certainty) It will
use an existing key if one is found at `AWS_SSH_KEY`, otherwise it will create
one there.  (With the default ubuntu images, if you have to SSH in: the user is
`ubuntu` and that user can `sudo`)
* Creates a VPC for use with the cluster (with a CIDR of 172.20.0.0/16)., and
  enables the `dns-support` and `dns-hostnames` options.
* Creates an internet gateway for the VPC.
* Creates a route table for the VPC, with the internet gateway as the default
  route
* Creates a subnet (with a CIDR of 172.20.0.0/24) in the AZ `KUBE_AWS_ZONE`
  (defaults to us-west-2a).  Currently kubernetes runs in a single AZ; there
are two philosophies on how to achieve HA: cluster-per-AZ and
cross-AZ-clusters.  cluster-per-AZ says you should have an independent cluster
for each AZ, they are entirely separate.  cross-AZ-clusters allows a single
cluster to span multiple AZs.  The debate is open here: cluster-per-AZ is more
robust but cross-AZ-clusters are more convenient.  For now though, each AWS
kuberentes cluster lives in one AZ.
* Associates the subnet to the route table
* Creates security groups for the master node (`kubernetes-master-<clusterid>`)
  and the minion nodes (`kubernetes-minion-<clusterid>`)
* Configures security groups so that masters & minions can intercommunicate,
  and opens SSH to the world on master & minions, and opens port 443 to the
world on the master (for the HTTPS API endpoint)
* Creates an EBS volume for the master node of size `MASTER_DISK_SIZE` and type
  `MASTER_DISK_TYPE`
* Launches a master node with a fixed IP address (172.20.0.9), with the
  security group, IAM credentials etc.  An instance script is used to pass
vital configuration information to Salt.  The hope is that over time we can
reduce the amount of configuration information that must be passed in this way.
* Once the instance is up, it attaches the EBS volume & sets up a manual
  routing rule for the internal network range (`MASTER_IP_RANGE`, defaults to
10.246.0.0/24)
* Creates an auto-scaling launch-configuration and group for the minions.  The
  name for both is `<KUBE_AWS_INSTANCE_PREFIX>-minion-group`, defaults to
`kubernetes-minion-group`.  The auto-scaling group has size min & max both set
to `NUM_MINIONS`.  You can change the size of the auto-scaling group to add or
remove minions (directly though the AWS API/Console).  The minion nodes
self-configure: they come up, run Salt with the stored configuration; connect
to the master and are assigned an internal CIDR; the master configures the
route-table with the minion CIDR.  The script does health-check the minions,
but this is a self-check, it is not required.

If attempting this configuration manually, I highly recommend following along
with the kube-up script, and being sure to tag everything with a
`KubernetesCluster`=`<clusterid>` tag.  Also, passing the right configuration
options to Salt when not using the script is tricky: the plan here is to
simplify this by having Kubernetes take on more node configuration, and even
potentially remove Salt altogether.


## Manual infrastructure creation

While this work is not yet complete, advanced users may choose to create (some)
AWS objects themselves, and still make use of the kube-up script (to configure
Salt, for example).

* `AWS_S3_BUCKET` will use an existing S3 bucket
* `VPC_ID` will reuse an existing VPC
* `SUBNET_ID` will reuse an existing subnet
* If your route table is tagged with the correct `KubernetesCluster`, it will
  be reused
* If your security groups are appropriately named, they will be reused.

Currently there is no way to do the following with kube-up.  If these affect
you, please open an issue with a description of what you're trying to do (your
use-case) and we'll see what we can do:

* Use an existing AWS SSH key with an arbitrary name
* Override the IAM credentials in a sensible way (but this is in-progress)
* Use different security group permissions
* Configure your own auto-scaling groups

# Instance boot

The instance boot procedure is currently pretty complicated, primarily because
we must marshal configuration from Bash to Salt via the AWS instance script.
As we move more post-boot configuration out of Salt and into Kubernetes, we
will hopefully be able to simplify this.

When the kube-up script launches instances, it builds an instance startup
script which includes some configuration options passed to kube-up, and
concatenates some of the scripts found in the cluster/aws/templates directory.
These scripts are responsible for mounting and formatting volumes, downloading
Salt & Kubernetes from the S3 bucket, and then triggering Salt to actually
install Kubernetes.




<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/aws_under_the_hood.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
