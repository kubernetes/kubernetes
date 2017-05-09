# Kubernetes networking setup for AWS

This document proposes different ways of satisfying the Kubernetes
networking model as described in
[networking](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/networking.md)
using the facilities provided by AWS.

## Proposal A

### Summary

- Kuberenetes instances run in a VPC
- [Source/destination check](http://docs.aws.amazon.com/AmazonVPC/latest/UserGuide/VPC_NAT_Instance.html#EIP_Disable_SrcDestCheck)
is disabled for all minion instances allowing them to send/receive traffic
not just destined for their primary IP.
- A route table entry is added for each minion instance, routing a
  `10.244.X.0/24` subnet to that instance.


### Implementation Notes

**VPC setup**:

Incomplete, but important parts of VPC setup:

```shell
aws ec2 create-vpc --cidr-block 172.20.0.0/16
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-support "{\"Value\": true}" > /dev/null
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-hostnames "{\"Value\": true}" > /dev/null
aws ec2 create-subnet --cidr-block 172.20.0.0/24 --vpc-id $VPC_ID
```

- I choose a different private address block (172.20.0.0/16) for
  instance IPs to avoid confusion with the 10.244.0.0/16 block for pod
  communication.
- AWS DNS support and hostname resolution is enabled

#### Minions

Minion instances need this extra bit of configuration.

**Disable source/destination check**:

```shell
aws ec2 modify-instance-attribute --instance-id ${INSTANCE_ID} --source-dest-check "{\"Value\": false}"
```

**Route 10.244.X.0/24 subnet to instance**:

```shell
aws ec2 create-route --route-table-id $ROUTE_TABLE_ID --destination-cidr-block "10.244.X.0/24" --instance-id $INSTANCE_ID
```
