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

**MASTER_SIZE**, **MINION_SIZE**

The instance type to use for creating the master/minion.  Defaults to auto-sizing based on the number of nodes (see below).

For production usage, we recommend bigger instances, for example:

```
export MASTER_SIZE=c4.large
export MINION_SIZE=r3.large
```

If you don't specify master and minion sizes, the scripts will attempt to guess the correct size of the master and worker nodes based on `${NUM_MINIONS}`.
In particular for clusters less than 50 nodes it will 
use a `t2.micro` for clusters between 50 and 150 nodes it will use a `t2.small` and for clusters with greater than 150 nodes it will use a `t2.medium`.

Please note: `kube-up` utilizes ephemeral storage available on instances for docker storage. EBS-only instance types do not
support ephemeral storage and will default to docker storage on the root disk which is usually only 8GB.
EBS-only instance types include `t2`, `c4`, and `m4`.

**KUBE_ENABLE_MINION_PUBLIC_IP**

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

The distribution to use.  Valid options: `trusty`, `vivid`, `coreos`, `wheezy`, `jessie`

Defaults to vivid (Ubuntu Vivid Vervet), which has a modern kernel and does not require updating or a reboot.

`coreos` is also a good option.

Other options may require reboots, updates or configuration, and should be used only if you have a compelling
requirement to do so.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/aws/options.md?pixel)]()
