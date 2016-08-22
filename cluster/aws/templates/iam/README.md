# IAM Roles and Policies for Kubernetes Master and Minion EC2 Nodes

In lieu of comments support in JSON, this document describes the IAM Roles and Policies required

## Master

### kubernetes-master-role.json

This establishes the trust relationship between the master role and EC2 service. The Role describes `ec2.amazonaws.com` can assume the role of `KubernetesMasterRole`.

### kubernetes-master-policy.json

`ec2:Describe*` is used reading instance metadata such as instance size and AZ information.

`elasticloadbalancing:*` are used for services with [Type=LoadBalancer](http://kubernetes.io/docs/user-guide/services/#type-loadbalancer).

`route53:*` are used for federation.

`s3:*` on `arn:aws:s3:::kubernetes-*` is specific to `kube-up` where kube binaries are stored in S3. This is not required for other setups.

## Minion

### kubernetes-minion-role.json

This establishes the trust relationship between the minion role and EC2 service. The Role describes `ec2.amazonaws.com` can assume the role of `KubernetesMinionRole`.

### kubernetes-minion-policy.json

Specific policies are granted to allow kubelet to perform various AWS-related tasks:

`s3:*` on `arn:aws:s3:::kubernetes-*` is specific to `kube-up` where kube binaries are stored in S3. This is not required for other setups.

`ec2:Describe*` is used reading instance metadata such as instance size and AZ information.

`ec2:AttachVolume` and `ec2:DetachVolume` are used for EBS support. This is not required for newer versions of Kubernetes on the minion as the logic has been moved to the controller.

`route53:*` are used for federation.

`ecr:*` are used for pulling images from EC2 Container Registry. This is not required for those not using ECR.
