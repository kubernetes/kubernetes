# AWS specific configuration options

These options can be set as environment variables to customize how your cluster is created.  Only options
specific to AWS are documented here, for cross-provider options see [this document](../options.md).

This is a work-in-progress; not all options are documented yet!

**KUBE_AWS_ZONE**

The AWS availability zone to deploy to.  Defaults to us-west-2a.

**AWS_IMAGE**

The AMI to use.  If not specified, the image will be selected based on the AWS region.

**AWS_S3_BUCKET**, **AWS_S3_REGION**

The bucket name to use, and the region where the bucket should be created, or where the bucket is located if it exists already.

If not specified, defaults to AWS_S3_REGION us-east-1, because buckets are globally named and you probably
want to share a bucket across all regions; us-east-1 is a sensible (relatively arbitrary) default.

AWS_S3_BUCKET will default to a uniquely generated name, so you won't collide with other kubernetes users.
(Currently this uses the hash of your AWS Access key to produce a per-user unique value).

It is not a bad idea to set AWS_S3_BUCKET to something more human friendly.

AWS_S3_REGION is useful for people that want to control their data location, because of regulatory restrictions for example.

**MASTER_SIZE**, **NODE_SIZE**

The instance type to use for creating the master/minion.  Defaults to auto-sizing based on the number of nodes (see below).

For production usage, we recommend bigger instances, for example:

```
export MASTER_SIZE=c4.large
export NODE_SIZE=r3.large
```

If you don't specify master and minion sizes, the scripts will attempt to guess the correct size of the master and worker
nodes based on `${NUM_NODES}`. See [Getting started on AWS EC2](../../docs/getting-started-guides/aws.md) for details.

Please note: `kube-up` utilizes ephemeral storage available on instances for docker storage. EBS-only instance types do not
support ephemeral storage and will default to docker storage on the root disk which is usually only 8GB.
EBS-only instance types include `t2`, `c4`, and `m4`.

**KUBE_ENABLE_NODE_PUBLIC_IP**

Should a public IP automatically assigned to the minions? "true" or "false"  
Defaults to: "true"

Please note: Do not set this to "false" unless you...

- ... already configured a NAT instance in the kubernetes VPC that will enable internet access for the new minions
- ... already configured a route for "0.0.0.0/0" to this NAT instance
- ... already configured a route for "YOUR_IP/32" to an AWS internet gateway (for the master instance to reach your
  client directly during setup)

**DOCKER_STORAGE**

Choose the docker storage driver to use.  This is an advanced option; most people should leave it as the default aufs
for parity with GCE.

Supported values: btrfs, aufs, devicemapper, aufs-nolvm

This will also configure your ephemeral storage in a compatible way, and your Docker containers
will run on this storage if available, as typically the root disk is comparatively small.

* `btrfs` will combine your ephemeral disks into a btrfs volume.  This is a good option if you have a recent kernel
  with a reliable btrfs.
* `aufs` uses the aufs driver, but also installs LVM to combine your disks. `aufs-nolvm` will not use LVM,
 meaning that only your first ephemeral disk will be used.
* `devicemapper` sets up LVM across all your ephemeral disks and sets Docker to drive it directly.  This is a
  similar option to btrfs, but without relying on the btrfs filesystem.  Sadly, it does not work with most
  configurations - see [this docker bug](https://github.com/docker/docker/issues/4036)

If your machines don't have any ephemeral disks, this will default to the aufs driver on your root disk (with no LVM).

**KUBE_OS_DISTRIBUTION**

The distribution to use.  Defaults to `jessie`

Supported options:

* `jessie`: Debian Jessie, running a custom kubernetes-optimized image.  Should
  be supported until 2018 by the debian-security team, and until 2020 by the
  debian-LTS team.
* `wily`: Ubuntu Wily.  Wily is not an LTS release, and OS support is due to
  end in July 2016.

No longer supported as of 1.3:

* `vivid`: Ubuntu Vivid.  Vivid OS support ended in early February 2016.
  Docker no longer provides packages for vivid.

Given the support situation, we recommend using Debian Jessie.  In Kubernetes
1.3 Ubuntu should have their next LTS release out, so we should be able to
recommend Ubuntu again at that time.

Using kube-up with other operating systems is neither supported nor
recommended.  But we would welcome increased OS support for kube-up, so please
contribute!

**NON_MASQUERADE_CIDR**

The 'internal' IP range which Kubernetes will use, which will therefore not
use IP masquerade.  By default kubernetes runs an internal network for traffic
between pods (and between pods and services), and by default this uses the
`10.0.0.0/8` range.  However, this sometimes overlaps with a range that you may
want to use; in particular the range cannot be used with EC2 ClassicLink.  You
may also want to run kubernetes in an existing VPC where you have chosen a CIDR
in the `10.0.0.0/8` range.

Setting this flag allows you to change this internal network CIDR.  Note that
you must set other values consistently within the CIDR that you choose.

For example, you might choose `172.16.0.0/14`; and you could then choose to
configure like this:

```
export NON_MASQUERADE_CIDR="172.16.0.0/14"
export SERVICE_CLUSTER_IP_RANGE="172.16.0.0/16"
export DNS_SERVER_IP="172.16.0.10"
export MASTER_IP_RANGE="172.17.0.0/24"
export CLUSTER_IP_RANGE="172.18.0.0/16"
```

When choosing a CIDR in the 172.20/12 reserved range you should be careful not
to choose a CIDR that overlaps your VPC CIDR (the kube-up script sets the VPC
CIDR to 172.20.0.0/16 by default, so you should not overlap that).  If you want
to allow inter-VPC traffic you should be careful to avoid your other VPCs as
well.

There is also a 100.64/10 address block which is reserved for "Carrier Grade
NAT", and which some users have reported success using.  While we haven't seen
any problems, or conflicts with any AWS networks, we can't guarantee it.  If you
decide you are comfortable using 100.64, you might use:

```
export NON_MASQUERADE_CIDR="100.64.0.0/10"
export SERVICE_CLUSTER_IP_RANGE="100.64.0.0/16"
export DNS_SERVER_IP="100.64.0.10"
export MASTER_IP_RANGE="100.65.0.0/24"
export CLUSTER_IP_RANGE="100.66.0.0/16"
```

**KUBE_VPC_CIDR_BASE**

By default `kube-up.sh` will create a VPC with CIDR 172.20.0.0/16. `KUBE_VPC_CIDR_BASE` allows to configure
this CIDR. For example you may choose to use `172.21.0.0/16`:

```
export KUBE_VPC_CIDR_BASE=172.21
```

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/aws/options.md?pixel)]()
