# AWS specific configuration options

These options can be set as environment variables to customize how your cluster is created.  Only options
specific to AWS are documented here, for cross-provider options see [this document](../options.md).

This is a work-in-progress; not all options are documented yet!

## ZONE

The AWS availability zone to deploy to.  Defaults to us-west-2a.

## AWS_IMAGE

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

The instance type to use for creating the master/minion.  Defaults to t2.micro.

For production usage, we recommend bigger instances, for example:

```
export MASTER_SIZE=c4.large
export MINION_SIZE=r3.large
```

**KUBE_ENABLE_MINION_PUBLIC_IP**

Should a public IP automatically assigned to the minions? "true" or "false"  
Defaults to: "true"

Please note: Do not set this to "false" unless you...

- ... already configured a NAT instance in the kubernetes VPC that will enable internet access for the new minions
- ... already configured a route for "0.0.0.0/0" to this NAT instance
- ... already configured a route for "YOUR_IP/32" to an AWS internet gateway (for the master instance to reach your
  client directly during setup)
