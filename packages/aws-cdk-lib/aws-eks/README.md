# Amazon EKS Construct Library


This construct library allows you to define [Amazon Elastic Container Service for Kubernetes (EKS)](https://aws.amazon.com/eks/) clusters.
In addition, the library also supports defining Kubernetes resource manifests within EKS clusters.

## Table Of Contents

- [Amazon EKS Construct Library](#amazon-eks-construct-library)
  - [Table Of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [Architectural Overview](#architectural-overview)
  - [Provisioning clusters](#provisioning-clusters)
    - [Provisioned Control Plane](#provisioned-control-plane)
    - [Managed node groups](#managed-node-groups)
      - [Node Groups with IPv6 Support](#node-groups-with-ipv6-support)
      - [Spot Instances Support](#spot-instances-support)
      - [Launch Template Support](#launch-template-support)
    - [Update clusters](#update-clusters)
    - [Fargate profiles](#fargate-profiles)
    - [Self-managed nodes](#self-managed-nodes)
      - [Spot Instances](#spot-instances)
      - [Bottlerocket](#bottlerocket)
    - [Endpoint Access](#endpoint-access)
    - [Alb Controller](#alb-controller)
    - [VPC Support](#vpc-support)
      - [Kubectl Handler](#kubectl-handler)
      - [Cluster Handler](#cluster-handler)
    - [IPv6 Support](#ipv6-support)
    - [Kubectl Support](#kubectl-support)
      - [Environment](#environment)
      - [Runtime](#runtime)
      - [Memory](#memory)
    - [ARM64 Support](#arm64-support)
    - [Masters Role](#masters-role)
    - [Encryption](#encryption)
    - [Hybrid nodes](#hybrid-nodes)
  - [Permissions and Security](#permissions-and-security)
    - [AWS IAM Mapping](#aws-iam-mapping)
    - [Access Config](#access-config)
    - [Access Entry](#access-mapping)
    - [Cluster Security Group](#cluster-security-group)
    - [Node SSH Access](#node-ssh-access)
    - [Service Accounts](#service-accounts)
    - [Pod Identities](#pod-identities)
  - [Applying Kubernetes Resources](#applying-kubernetes-resources)
    - [Kubernetes Manifests](#kubernetes-manifests)
      - [ALB Controller Integration](#alb-controller-integration)
      - [Adding resources from a URL](#adding-resources-from-a-url)
      - [Dependencies](#dependencies)
      - [Resource Pruning](#resource-pruning)
      - [Manifests Validation](#manifests-validation)
    - [Helm Charts](#helm-charts)
    - [OCI Charts](#oci-charts)
    - [CDK8s Charts](#cdk8s-charts)
      - [Custom CDK8s Constructs](#custom-cdk8s-constructs)
      - [Manually importing k8s specs and CRD's](#manually-importing-k8s-specs-and-crds)
  - [Patching Kubernetes Resources](#patching-kubernetes-resources)
  - [Querying Kubernetes Resources](#querying-kubernetes-resources)
  - [Add-ons](#add-ons)
  - [Using existing clusters](#using-existing-clusters)
  - [Logging](#logging)
  - [Known Issues and Limitations](#known-issues-and-limitations)

## Quick Start

This example defines an Amazon EKS cluster with the following configuration:

* Dedicated VPC with default configuration (Implicitly created using [ec2.Vpc](https://docs.aws.amazon.com/cdk/api/latest/docs/aws-ec2-readme.html#vpc))
* A Kubernetes pod with a container based on the [paulbouwer/hello-kubernetes](https://github.com/paulbouwer/hello-kubernetes) image.

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

// provisioning a cluster
const cluster = new eks.Cluster(this, 'hello-eks', {
  version: eks.KubernetesVersion.V1_36,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});

// apply a kubernetes manifest to the cluster
cluster.addManifest('mypod', {
  apiVersion: 'v1',
  kind: 'Pod',
  metadata: { name: 'mypod' },
  spec: {
    containers: [
      {
        name: 'hello',
        image: 'paulbouwer/hello-kubernetes:1.5',
        ports: [ { containerPort: 8080 } ],
      },
    ],
  },
});
```

## Architectural Overview

The following is a qualitative diagram of the various possible components involved in the cluster deployment.

```text
 +-----------------------------------------------+               +-----------------+
 | EKS Cluster | kubectl |  |
 | ----------- |<-------------+| Kubectl Handler |
 |                                               |               |                 |
 |                                               |               +-----------------+
 | +--------------------+    +-----------------+ |
 | |                    |    |                 | |
 | | Managed Node Group |    | Fargate Profile | |               +-----------------+
 | |                    |    |                 | |               |                 |
 | +--------------------+    +-----------------+ |               | Cluster Handler |
 |                                               |               |                 |
 +-----------------------------------------------+               +-----------------+
    ^                                   ^                          +
    |                                   |                          |
    | connect self managed capacity     |                          | aws-sdk
    |                                   | create/update/delete     |
    +                                   |                          v
 +--------------------+                 +              +-------------------+
 |                    |                 --------------+| eks.amazonaws.com |
 | Auto Scaling Group |                                +-------------------+
 |                    |
 +--------------------+
```

In a nutshell:

* `EKS Cluster` - The cluster endpoint created by EKS.
* `Managed Node Group` - EC2 worker nodes managed by EKS.
* `Fargate Profile` - Fargate worker nodes managed by EKS.
* `Auto Scaling Group` - EC2 worker nodes managed by the user.
* `KubectlHandler` - Lambda function for invoking `kubectl` commands on the cluster - created by CDK.
* `ClusterHandler` - Lambda function for interacting with EKS API to manage the cluster lifecycle - created by CDK.

A more detailed breakdown of each is provided further down this README.

## Provisioning clusters

Creating a new cluster is done using the `Cluster` or `FargateCluster` constructs. The only required properties are the kubernetes `version` and `kubectlLayer`.

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

You can control what happens to the resources created by the cluster construct when they are no longer managed by CloudFormation by specifying a `removalPolicy`.

This can happen in one of three situations:
- The resource is removed from the template, so CloudFormation stops managing it;
- A change to the resource is made that requires it to be replaced, so CloudFormation stops managing it;
- The stack is deleted, so CloudFormation stops managing all resources in it.

This affects the EKS cluster itself, the custom resource that created the cluster, associated IAM roles, node groups, security groups, VPC and any other CloudFormation resources managed by this construct.

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';
import * as core from 'aws-cdk-lib/core';

new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
  removalPolicy: core.RemovalPolicy.RETAIN, // Keep all resources created by the construct.
});
```

You can also use `FargateCluster` to provision a cluster that uses only fargate workers.

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

new eks.FargateCluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

You can enable deletion protection for your cluster to prevent accidental deletion. When deletion protection is enabled,
the cluster cannot be deleted until protection is disabled. This setting only applies to clusters in an active state.

> For more details visit [Deletion protection](https://docs.aws.amazon.com/eks/latest/userguide/deletion-protection.html).

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
  deletionProtection: true,
});
```

### Provisioned Control Plane

Amazon EKS Provisioned Control Plane allows you to select a scaling tier to ensure high and predictable performance for demanding workloads such as AI training/inference, high-performance computing, or large-scale data processing.

```ts
import { KubectlV35Layer } from '@aws-cdk/lambda-layer-kubectl-v35';

new eks.Cluster(this, 'HighPerformanceCluster', {
  version: eks.KubernetesVersion.V1_35,
  kubectlLayer: new KubectlV35Layer(this, 'kubectl'),
  controlPlaneScalingTier: eks.ControlPlaneScalingTier.TIER_XL,
});
```

Available scaling tiers:

- `STANDARD` - Standard control plane (default, no additional cost)
- `TIER_XL` - Extra-large provisioned tier
- `TIER_2XL` - 2x extra-large provisioned tier
- `TIER_4XL` - 4x extra-large provisioned tier
- `TIER_8XL` - 8x extra-large provisioned tier

> For more details visit [Amazon EKS Provisioned Control Plane](https://docs.aws.amazon.com/eks/latest/userguide/eks-provisioned-control-plane.html).

> **NOTE: Only 1 cluster per stack is supported.** If you have a use-case for multiple clusters per stack, or would like to understand more about this limitation, see <https://github.com/aws/aws-cdk/issues/10073>.

Below you'll find a few important cluster configuration options. First of which is Capacity.
Capacity is the amount and the type of worker nodes that are available to the cluster for deploying resources. Amazon EKS offers 3 ways of configuring capacity, which you can combine as you like:

### Managed node groups

Amazon EKS managed node groups automate the provisioning and lifecycle management of nodes (Amazon EC2 instances) for Amazon EKS Kubernetes clusters.
With Amazon EKS managed node groups, you don't need to separately provision or register the Amazon EC2 instances that provide compute capacity to run your Kubernetes applications. You can create, update, or terminate nodes for your cluster with a single operation. Nodes run using the latest Amazon EKS optimized AMIs in your AWS account while node updates and terminations gracefully drain nodes to ensure that your applications stay available.

> For more details visit [Amazon EKS Managed Node Groups](https://docs.aws.amazon.com/eks/latest/userguide/managed-node-groups.html).

**Managed Node Groups are the recommended way to allocate cluster capacity.**

By default, this library will allocate a managed node group with 2 *m5.large* instances (this instance type suits most common use-cases, and is good value for money).

At cluster instantiation time, you can customize the number of instances and their type:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  defaultCapacity: 5,
  defaultCapacityInstance: ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.SMALL),
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

To access the node group that was created on your behalf, you can use `cluster.defaultNodegroup`.

Additional customizations are available post instantiation. To apply them, set the default capacity to 0, and use the `cluster.addNodegroupCapacity` method:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

const cluster = new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  defaultCapacity: 0,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});

cluster.addNodegroupCapacity('custom-node-group', {
  instanceTypes: [new ec2.InstanceType('m5.large')],
  minSize: 4,
  diskSize: 100,
});
```

To set node taints, you can set `taints` option.

```ts
declare const cluster: eks.Cluster;
cluster.addNodegroupCapacity('custom-node-group', {
  instanceTypes: [new ec2.InstanceType('m5.large')],
  taints: [
    {
      effect: eks.TaintEffect.NO_SCHEDULE,
      key: 'foo',
      value: 'bar',
    },
  ],
});
```

To define the type of the AMI for the node group, you may explicitly define `amiType` according to your requirements, supported amiType could be found [HERE](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_eks.NodegroupAmiType.html).

```ts
declare const cluster: eks.Cluster;

// X86_64 based AMI managed node group
cluster.addNodegroupCapacity('custom-node-group', {
  instanceTypes: [new ec2.InstanceType('m5.large')], // NOTE: if amiType is x86_64-based image, the instance types here must be x86_64-based.
  amiType: eks.NodegroupAmiType.AL2023_X86_64_STANDARD,
});

// ARM_64 based AMI managed node group
cluster.addNodegroupCapacity('custom-node-group', {
  instanceTypes: [new ec2.InstanceType('m6g.medium')], // NOTE: if amiType is ARM-based image, the instance types here must be ARM-based.
  amiType: eks.NodegroupAmiType.AL2023_ARM_64_STANDARD,
});
```

To define the maximum number of instances which can be simultaneously replaced in a node group during a version update you can set `maxUnavailable` or `maxUnavailablePercentage` options.

> For more details visit [Updating a managed node group](https://docs.aws.amazon.com/eks/latest/userguide/update-managed-node-group.html)

```ts
declare const cluster: eks.Cluster;
cluster.addNodegroupCapacity('custom-node-group', {
  instanceTypes: [new ec2.InstanceType('m5.large')],
  maxSize: 5,
  maxUnavailable: 2,
});
```

```ts
declare const cluster: eks.Cluster;
cluster.addNodegroupCapacity('custom-node-group', {
  instanceTypes: [new ec2.InstanceType('m5.large')],
  maxUnavailablePercentage: 33,
});
```

> **NOTE:** If you add instances with the inferentia class (`inf1` or `inf2`) or trainium class (`trn1`, `trn1n`, or `trn2`) 
> the [neuron plugin](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/containers/dlc-then-eks-devflow.html)
> will be automatically installed in the kubernetes cluster.

##### Default AMI type (under feature flag)

By default, managed node groups that do not set `amiType` use `AL2_X86_64` (or `AL2_ARM_64` for
ARM instances). Amazon Linux 2 EKS-optimized AMIs reached end of support on **November 26, 2025**. 
AL2023 is the AWS-recommended default.

New applications should enable the `@aws-cdk/aws-eks:defaultToAL2023` feature flag in `cdk.json`:

```json
{
  "context": {
    "@aws-cdk/aws-eks:defaultToAL2023": true
  }
}
```

When the flag is enabled, the default AMI type for x86_64 instances becomes
`AL2023_X86_64_STANDARD`, and for ARM instances it becomes `AL2023_ARM_64_STANDARD`. GPU
instances continue to default to `AL2_X86_64_GPU` because AL2023 splits GPU support into
separate NVIDIA and Neuron AMI variants — GPU users must pick a variant explicitly.

**Migration for existing applications.** Enabling this flag on an existing app will cause
managed node groups that previously defaulted to AL2 to be replaced with AL2023 on the next
deploy, which terminates running pods. To roll out safely, pin every existing node group to its
current AMI type first, and only then enable the flag as shown below. Then gradually unpin the
AMI for the nodes you want to upgrade.

```ts
declare const cluster: eks.Cluster;

// Pin existing node groups to AL2 explicitly before enabling the flag.
cluster.addNodegroupCapacity('workers', {
  instanceTypes: [new ec2.InstanceType('m5.large')],
  amiType: eks.NodegroupAmiType.AL2_X86_64,
});
```

Explicitly setting `amiType` will pin it — it is not affected by the feature flag.

#### Node Groups with IPv6 Support

Node groups are available with IPv6 configured networks.  For custom roles assigned to node groups additional permissions are necessary in order for pods to obtain an IPv6 address.  The default node role will include these permissions.

> For more details visit [Configuring the Amazon VPC CNI plugin for Kubernetes to use IAM roles for service accounts](https://docs.aws.amazon.com/eks/latest/userguide/cni-iam-role.html#cni-iam-role-create-role)

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

const ipv6Management = new iam.PolicyDocument({
    statements: [new iam.PolicyStatement({
    resources: ['arn:aws:ec2:*:*:network-interface/*'],
    actions: [
        'ec2:AssignIpv6Addresses',
        'ec2:UnassignIpv6Addresses',
    ],
    })],
});

const eksClusterNodeGroupRole = new iam.Role(this, 'eksClusterNodeGroupRole', {
  roleName: 'eksClusterNodeGroupRole',
  assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
  managedPolicies: [
    iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKSWorkerNodePolicy'),
    iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryReadOnly'),
    iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKS_CNI_Policy'),
  ],
    inlinePolicies: {
    ipv6Management,
  },
});

const cluster = new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  defaultCapacity: 0,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});

cluster.addNodegroupCapacity('custom-node-group', {
  instanceTypes: [new ec2.InstanceType('m5.large')],
  minSize: 2,
  diskSize: 100,
  nodeRole: eksClusterNodeGroupRole,
});
```

#### Spot Instances Support

Use `capacityType` to create managed node groups comprised of spot instances. To maximize the availability of your applications while using
Spot Instances, we recommend that you configure a Spot managed node group to use multiple instance types with the `instanceTypes` property.

> For more details visit [Managed node group capacity types](https://docs.aws.amazon.com/eks/latest/userguide/managed-node-groups.html#managed-node-group-capacity-types).


```ts
declare const cluster: eks.Cluster;
cluster.addNodegroupCapacity('extra-ng-spot', {
  instanceTypes: [
    new ec2.InstanceType('c5.large'),
    new ec2.InstanceType('c5a.large'),
    new ec2.InstanceType('c5d.large'),
  ],
  minSize: 3,
  capacityType: eks.CapacityType.SPOT,
});

```

#### Launch Template Support

You can specify a launch template that the node group will use. For example, this can be useful if you want to use
a custom AMI or add custom user data.

When supplying a custom user data script, it must be encoded in the MIME multi-part archive format, since Amazon EKS merges with its own user data. Visit the [Launch Template Docs](https://docs.aws.amazon.com/eks/latest/userguide/launch-templates.html#launch-template-user-data)
for mode details.

```ts
declare const cluster: eks.Cluster;

const userData = `MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="==MYBOUNDARY=="

--==MYBOUNDARY==
Content-Type: text/x-shellscript; charset="us-ascii"

#!/bin/bash
echo "Running custom user data script"

--==MYBOUNDARY==--\\
`;
const lt = new ec2.CfnLaunchTemplate(this, 'LaunchTemplate', {
  launchTemplateData: {
    instanceType: 't3.small',
    userData: Fn.base64(userData),
  },
});

cluster.addNodegroupCapacity('extra-ng', {
  launchTemplateSpec: {
    id: lt.ref,
    version: lt.attrLatestVersionNumber,
  },
});

```

Note that when using a custom AMI, Amazon EKS doesn't merge any user data. Which means you do not need the multi-part encoding. and are responsible for supplying the required bootstrap commands for nodes to join the cluster.
In the following example, `/ect/eks/bootstrap.sh` from the AMI will be used to bootstrap the node.

```ts
declare const cluster: eks.Cluster;
const userData = ec2.UserData.forLinux();
userData.addCommands(
  'set -o xtrace',
  `/etc/eks/bootstrap.sh ${cluster.clusterName}`,
);
const lt = new ec2.CfnLaunchTemplate(this, 'LaunchTemplate', {
  launchTemplateData: {
    imageId: 'some-ami-id', // custom AMI
    instanceType: 't3.small',
    userData: Fn.base64(userData.render()),
  },
});
cluster.addNodegroupCapacity('extra-ng', {
  launchTemplateSpec: {
    id: lt.ref,
    version: lt.attrLatestVersionNumber,
  },
});
```

You may specify one `instanceType` in the launch template or multiple `instanceTypes` in the node group, **but not both**.

> For more details visit [Launch Template Support](https://docs.aws.amazon.com/eks/latest/userguide/launch-templates.html).

Graviton 2 instance types are supported including `c6g`, `m6g`, `r6g` and `t4g`.
Graviton 3 instance types are supported including `c7g`.

### Update clusters

When you rename the cluster name and redeploy the stack, the cluster replacement will be triggered and
the existing one will be deleted after the new one is provisioned. As the cluster resource ARN has been changed, 
the cluster resource handler would not be able to delete the old one as the resource ARN in the IAM policy
has been changed. As a workaround, you need to add a temporary policy to the cluster admin role for 
successful replacement. Consider this example if you are renaming the cluster from `foo` to `bar`:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

const cluster = new eks.Cluster(this, 'cluster-to-rename', {
  clusterName: 'foo', // rename this to 'bar'
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
  version: eks.KubernetesVersion.V1_36,
});

// allow the cluster admin role to delete the cluster 'foo'
cluster.adminRole.addToPolicy(new iam.PolicyStatement({
  actions: [
    'eks:DeleteCluster',
    'eks:DescribeCluster',
  ],
  resources: [ 
    Stack.of(this).formatArn({ service: 'eks', resource: 'cluster', resourceName: 'foo' }),
]
}))
```

### Fargate profiles

AWS Fargate is a technology that provides on-demand, right-sized compute
capacity for containers. With AWS Fargate, you no longer have to provision,
configure, or scale groups of virtual machines to run containers. This removes
the need to choose server types, decide when to scale your node groups, or
optimize cluster packing.

You can control which pods start on Fargate and how they run with Fargate
Profiles, which are defined as part of your Amazon EKS cluster.

See [Fargate Considerations](https://docs.aws.amazon.com/eks/latest/userguide/fargate.html#fargate-considerations) in the AWS EKS User Guide.

You can add Fargate Profiles to any EKS cluster defined in your CDK app
through the `addFargateProfile()` method. The following example adds a profile
that will match all pods from the "default" namespace:

```ts
declare const cluster: eks.Cluster;
cluster.addFargateProfile('MyProfile', {
  selectors: [ { namespace: 'default' } ],
});
```

You can also directly use the `FargateProfile` construct to create profiles under different scopes:

```ts
declare const cluster: eks.Cluster;
new eks.FargateProfile(this, 'MyProfile', {
  cluster,
  selectors: [ { namespace: 'default' } ],
});
```

To create an EKS cluster that **only** uses Fargate capacity, you can use `FargateCluster`.
The following code defines an Amazon EKS cluster with a default Fargate Profile that matches all pods from the "kube-system" and "default" namespaces. It is also configured to [run CoreDNS on Fargate](https://docs.aws.amazon.com/eks/latest/userguide/fargate-getting-started.html#fargate-gs-coredns).

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

const cluster = new eks.FargateCluster(this, 'MyCluster', {
  version: eks.KubernetesVersion.V1_36,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

`FargateCluster` will create a default `FargateProfile` which can be accessed via the cluster's `defaultProfile` property. The created profile can also be customized by passing options as with `addFargateProfile`.

**NOTE**: Classic Load Balancers and Network Load Balancers are not supported on
pods running on Fargate. For ingress, we recommend that you use the [ALB Ingress
Controller](https://docs.aws.amazon.com/eks/latest/userguide/alb-ingress.html)
on Amazon EKS (minimum version v1.1.4).

### Self-managed nodes

Another way of allocating capacity to an EKS cluster is by using self-managed nodes.
EC2 instances that are part of the auto-scaling group will serve as worker nodes for the cluster.
This type of capacity is also commonly referred to as *EC2 Capacity** or *EC2 Nodes*.

For a detailed overview please visit [Self Managed Nodes](https://docs.aws.amazon.com/eks/latest/userguide/worker.html).

Creating an auto-scaling group and connecting it to the cluster is done using the `cluster.addAutoScalingGroupCapacity` method:

```ts
declare const cluster: eks.Cluster;
cluster.addAutoScalingGroupCapacity('frontend-nodes', {
  instanceType: new ec2.InstanceType('t2.medium'),
  minCapacity: 3,
  vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
});
```

To connect an already initialized auto-scaling group, use the `cluster.connectAutoScalingGroupCapacity()` method:

```ts
declare const cluster: eks.Cluster;
declare const asg: autoscaling.AutoScalingGroup;
cluster.connectAutoScalingGroupCapacity(asg, {});
```

To connect a self-managed node group to an imported cluster, use the `cluster.connectAutoScalingGroupCapacity()` method:

```ts
declare const cluster: eks.Cluster;
declare const asg: autoscaling.AutoScalingGroup;
const importedCluster = eks.Cluster.fromClusterAttributes(this, 'ImportedCluster', {
  clusterName: cluster.clusterName,
  clusterSecurityGroupId: cluster.clusterSecurityGroupId,
});

importedCluster.connectAutoScalingGroupCapacity(asg, {});
```

In both cases, the [cluster security group](https://docs.aws.amazon.com/eks/latest/userguide/sec-group-reqs.html#cluster-sg) will be automatically attached to
the auto-scaling group, allowing for traffic to flow freely between managed and self-managed nodes.

> **Note:** The default `updateType` for auto-scaling groups does not replace existing nodes. Since security groups are determined at launch time, self-managed nodes that were provisioned with version `1.78.0` or lower, will not be updated.
> To apply the new configuration on all your self-managed nodes, you'll need to replace the nodes using the `UpdateType.REPLACING_UPDATE` policy for the [`updateType`](https://docs.aws.amazon.com/cdk/api/latest/docs/@aws-cdk_aws-autoscaling.AutoScalingGroup.html#updatetypespan-classapi-icon-api-icon-deprecated-titlethis-api-element-is-deprecated-its-use-is-not-recommended%EF%B8%8Fspan) property.

You can customize the [/etc/eks/boostrap.sh](https://github.com/awslabs/amazon-eks-ami/blob/master/files/bootstrap.sh) script, which is responsible
for bootstrapping the node to the EKS cluster. For example, you can use `kubeletExtraArgs` to add custom node labels or taints.

```ts
declare const cluster: eks.Cluster;
cluster.addAutoScalingGroupCapacity('spot', {
  instanceType: new ec2.InstanceType('t3.large'),
  minCapacity: 2,
  bootstrapOptions: {
    kubeletExtraArgs: '--node-labels foo=bar,goo=far',
    awsApiRetryAttempts: 5,
  },
});
```

To disable bootstrapping altogether (i.e. to fully customize user-data), set `bootstrapEnabled` to `false`.
You can also configure the cluster to use an auto-scaling group as the default capacity:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

const cluster = new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  defaultCapacityType: eks.DefaultCapacityType.EC2,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

This will allocate an auto-scaling group with 2 *m5.large* instances (this instance type suits most common use-cases, and is good value for money).
To access the `AutoScalingGroup` that was created on your behalf, you can use `cluster.defaultCapacity`.
You can also independently create an `AutoScalingGroup` and connect it to the cluster using the `cluster.connectAutoScalingGroupCapacity` method:

```ts
declare const cluster: eks.Cluster;
declare const asg: autoscaling.AutoScalingGroup;
cluster.connectAutoScalingGroupCapacity(asg, {});
```

This will add the necessary user-data to access the apiserver and configure all connections, roles, and tags needed for the instances in the auto-scaling group to properly join the cluster.

#### Spot Instances

When using self-managed nodes, you can configure the capacity to use spot instances, greatly reducing capacity cost.
To enable spot capacity, use the `spotPrice` property:

```ts
declare const cluster: eks.Cluster;
cluster.addAutoScalingGroupCapacity('spot', {
  spotPrice: '0.1094',
  instanceType: new ec2.InstanceType('t3.large'),
  maxCapacity: 10,
});
```

> Spot instance nodes will be labeled with `lifecycle=Ec2Spot` and tainted with `PreferNoSchedule`.

The [AWS Node Termination Handler](https://github.com/aws/aws-node-termination-handler) `DaemonSet` will be
installed from [Amazon EKS Helm chart repository](https://github.com/aws/eks-charts/tree/master/stable/aws-node-termination-handler) on these nodes.
The termination handler ensures that the Kubernetes control plane responds appropriately to events that
can cause your EC2 instance to become unavailable, such as [EC2 maintenance events](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/monitoring-instances-status-check_sched.html)
and [EC2 Spot interruptions](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-interruptions.html) and helps gracefully stop all pods running on spot nodes that are about to be
terminated.

> Handler Version: [1.7.0](https://github.com/aws/aws-node-termination-handler/releases/tag/v1.7.0)
>
> Chart Version: [0.9.5](https://github.com/aws/eks-charts/blob/v0.0.28/stable/aws-node-termination-handler/Chart.yaml)

To disable the installation of the termination handler, set the `spotInterruptHandler` property to `false`. This applies both to `addAutoScalingGroupCapacity` and `connectAutoScalingGroupCapacity`.

#### Bottlerocket

[Bottlerocket](https://aws.amazon.com/bottlerocket/) is a Linux-based open-source operating system that is purpose-built by Amazon Web Services for running containers on virtual machines or bare metal hosts.

`Bottlerocket` is supported when using managed nodegroups or self-managed auto-scaling groups.

To create a Bottlerocket managed nodegroup:

```ts
declare const cluster: eks.Cluster;
cluster.addNodegroupCapacity('BottlerocketNG', {
  amiType: eks.NodegroupAmiType.BOTTLEROCKET_X86_64,
});
```

The following example will create an auto-scaling group of 2 `t3.small` Linux instances running with the `Bottlerocket` AMI.

```ts
declare const cluster: eks.Cluster;
cluster.addAutoScalingGroupCapacity('BottlerocketNodes', {
  instanceType: new ec2.InstanceType('t3.small'),
  minCapacity:  2,
  machineImageType: eks.MachineImageType.BOTTLEROCKET,
});
```

The specific Bottlerocket AMI variant will be auto selected according to the k8s version for the `x86_64` architecture.
For example, if the Amazon EKS cluster version is `1.17`, the Bottlerocket AMI variant will be auto selected as
`aws-k8s-1.17` behind the scene.

> See [Variants](https://github.com/bottlerocket-os/bottlerocket/blob/develop/README.md#variants) for more details.

Please note Bottlerocket does not allow to customize bootstrap options and `bootstrapOptions` properties is not supported when you create the `Bottlerocket` capacity.

To create a Bottlerocket managed nodegroup with Nvidia-based EC2 instance types use the `BOTTLEROCKET_X86_64_NVIDIA` or
`BOTTLEROCKET_ARM_64_NVIDIA` AMIs:

```ts
declare const cluster: eks.Cluster;
cluster.addNodegroupCapacity('BottlerocketNvidiaNG', {
  amiType: eks.NodegroupAmiType.BOTTLEROCKET_X86_64_NVIDIA,
  instanceTypes: [new ec2.InstanceType('g4dn.xlarge')],
});
```

For more details about Bottlerocket, see [Bottlerocket FAQs](https://aws.amazon.com/bottlerocket/faqs/) and [Bottlerocket Open Source Blog](https://aws.amazon.com/blogs/opensource/announcing-the-general-availability-of-bottlerocket-an-open-source-linux-distribution-purpose-built-to-run-containers/).

### Endpoint Access

When you create a new cluster, Amazon EKS creates an endpoint for the managed Kubernetes API server that you use to communicate with your cluster (using Kubernetes management tools such as `kubectl`)

By default, this API server endpoint is public to the internet, and access to the API server is secured using a combination of
AWS Identity and Access Management (IAM) and native Kubernetes [Role Based Access Control](https://kubernetes.io/docs/reference/access-authn-authz/rbac/) (RBAC).

You can configure the [cluster endpoint access](https://docs.aws.amazon.com/eks/latest/userguide/cluster-endpoint.html) by using the `endpointAccess` property:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

const cluster = new eks.Cluster(this, 'hello-eks', {
  version: eks.KubernetesVersion.V1_36,
  endpointAccess: eks.EndpointAccess.PRIVATE, // No access outside of your VPC.
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

The default value is `eks.EndpointAccess.PUBLIC_AND_PRIVATE`. Which means the cluster endpoint is accessible from outside of your VPC, but worker node traffic and `kubectl` commands issued by this library stay within your VPC.

### Alb Controller

Some Kubernetes resources are commonly implemented on AWS with the help of the [ALB Controller](https://kubernetes-sigs.github.io/aws-load-balancer-controller/latest/).

From the docs:

> AWS Load Balancer Controller is a controller to help manage Elastic Load Balancers for a Kubernetes cluster.
>
> * It satisfies Kubernetes Ingress resources by provisioning Application Load Balancers.
> * It satisfies Kubernetes Service resources by provisioning Network Load Balancers.

To deploy the controller on your EKS cluster, configure the `albController` property:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  albController: {
    version: eks.AlbControllerVersion.V3_2_2,
  },
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

To provide additional Helm chart values supported by `albController` in CDK, use the `additionalHelmChartValues` property. For example, the following code snippet shows how to set the `enableWafV2` flag:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  albController: {
    version: eks.AlbControllerVersion.V3_2_2,
    additionalHelmChartValues: {
      enableWafv2: false
    }
  },
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

To overwrite an existing ALB controller service account, use the `overwriteServiceAccount` property:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  albController: {
    version: eks.AlbControllerVersion.V3_2_2,
    overwriteServiceAccount: true
  },
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

The `albController` requires `defaultCapacity` or at least one nodegroup. If there's no `defaultCapacity` or available
nodegroup for the cluster, the `albController` deployment would fail.

Querying the controller pods should look something like this:

```console
❯ kubectl get pods -n kube-system
NAME                                            READY   STATUS    RESTARTS   AGE
aws-load-balancer-controller-76bd6c7586-d929p   1/1     Running   0          109m
aws-load-balancer-controller-76bd6c7586-fqxph   1/1     Running   0          109m
...
...
```

Every Kubernetes manifest that utilizes the ALB Controller is effectively dependent on the controller.
If the controller is deleted before the manifest, it might result in dangling ELB/ALB resources.
Currently, the EKS construct library does not detect such dependencies, and they should be done explicitly.

For example:

```ts
declare const cluster: eks.Cluster;
const manifest = cluster.addManifest('manifest', {/* ... */});
if (cluster.albController) {
  manifest.node.addDependency(cluster.albController);
}
```

### VPC Support

You can specify the VPC of the cluster using the `vpc` and `vpcSubnets` properties:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

declare const vpc: ec2.Vpc;

new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  vpc,
  vpcSubnets: [{ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }],
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

> Note: Isolated VPCs (i.e with no internet access) are not fully supported. See https://github.com/aws/aws-cdk/issues/12171. Check out [this aws-cdk-example](https://github.com/aws-samples/aws-cdk-examples/tree/master/java/eks/private-cluster) for reference.

If you do not specify a VPC, one will be created on your behalf, which you can then access via `cluster.vpc`. The cluster VPC will be associated to any EKS managed capacity (i.e Managed Node Groups and Fargate Profiles).

Please note that the `vpcSubnets` property defines the subnets where EKS will place the _control plane_ ENIs. To choose
the subnets where EKS will place the worker nodes, please refer to the **Provisioning clusters** section above.

If you allocate self managed capacity, you can specify which subnets should the auto-scaling group use:

```ts
declare const vpc: ec2.Vpc;
declare const cluster: eks.Cluster;
cluster.addAutoScalingGroupCapacity('nodes', {
  vpcSubnets: { subnets: vpc.privateSubnets },
  instanceType: new ec2.InstanceType('t2.medium'),
});
```

There are two additional components you might want to provision within the VPC.

#### Kubectl Handler

The `KubectlHandler` is a Lambda function responsible to issuing `kubectl` and `helm` commands against the cluster when you add resource manifests to the cluster.

The handler association to the VPC is derived from the `endpointAccess` configuration. The rule of thumb is: *If the cluster VPC can be associated, it will be*.

Breaking this down, it means that if the endpoint exposes private access (via `EndpointAccess.PRIVATE` or `EndpointAccess.PUBLIC_AND_PRIVATE`), and the VPC contains **private** subnets, the Lambda function will be provisioned inside the VPC and use the private subnets to interact with the cluster. This is the common use-case.

If the endpoint does not expose private access (via `EndpointAccess.PUBLIC`) **or** the VPC does not contain private subnets, the function will not be provisioned within the VPC.

If your use-case requires control over the IAM role that the KubeCtl Handler assumes, a custom role can be passed through the ClusterProps (as `kubectlLambdaRole`) of the EKS Cluster construct.

#### Cluster Handler

The `ClusterHandler` is a set of Lambda functions (`onEventHandler`, `isCompleteHandler`) responsible for interacting with the EKS API in order to control the cluster lifecycle. To provision these functions inside the VPC, set the `placeClusterHandlerInVpc` property to `true`. This will place the functions inside the private subnets of the VPC based on the selection strategy specified in the [`vpcSubnets`](https://docs.aws.amazon.com/cdk/api/latest/docs/@aws-cdk_aws-eks.Cluster.html#vpcsubnetsspan-classapi-icon-api-icon-experimental-titlethis-api-element-is-experimental-it-may-change-without-noticespan) property.

You can configure the environment of the Cluster Handler functions by specifying it at cluster instantiation. For example, this can be useful in order to configure an http proxy:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

declare const proxyInstanceSecurityGroup: ec2.SecurityGroup;
const cluster = new eks.Cluster(this, 'hello-eks', {
  version: eks.KubernetesVersion.V1_36,
  clusterHandlerEnvironment: {
    https_proxy: 'http://proxy.myproxy.com',
  },
  /**
   * If the proxy is not open publicly, you can pass a security group to the
   * Cluster Handler Lambdas so that it can reach the proxy.
   */
  clusterHandlerSecurityGroup: proxyInstanceSecurityGroup,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

### IPv6 Support

You can optionally choose to configure your cluster to use IPv6 using the [`ipFamily`](https://docs.aws.amazon.com/eks/latest/APIReference/API_KubernetesNetworkConfigRequest.html#AmazonEKS-Type-KubernetesNetworkConfigRequest-ipFamily) definition for your cluster.  Note that this will require the underlying subnets to have an associated IPv6 CIDR.

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';
declare const vpc: ec2.Vpc;

function associateSubnetWithV6Cidr(vpc: ec2.Vpc, count: number, subnet: ec2.ISubnet) {
  const cfnSubnet = subnet.node.defaultChild as ec2.CfnSubnet;
  cfnSubnet.ipv6CidrBlock = Fn.select(count, Fn.cidr(Fn.select(0, vpc.vpcIpv6CidrBlocks), 256, (128 - 64).toString()));
  cfnSubnet.assignIpv6AddressOnCreation = true;
}

// make an ipv6 cidr
const ipv6cidr = new ec2.CfnVPCCidrBlock(this, 'CIDR6', {
  vpcId: vpc.vpcId,
  amazonProvidedIpv6CidrBlock: true,
});

// connect the ipv6 cidr to all vpc subnets
let subnetcount = 0;
const subnets = vpc.publicSubnets.concat(vpc.privateSubnets);
for (let subnet of subnets) {
  // Wait for the ipv6 cidr to complete
  subnet.node.addDependency(ipv6cidr);
  associateSubnetWithV6Cidr(vpc, subnetcount, subnet);
  subnetcount = subnetcount + 1;
}

const cluster = new eks.Cluster(this, 'hello-eks', {
  version: eks.KubernetesVersion.V1_36,
  vpc: vpc,
  ipFamily: eks.IpFamily.IP_V6,
  vpcSubnets: [{ subnets: vpc.publicSubnets }],
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

### Kubectl Support

The resources are created in the cluster by running `kubectl apply` from a python lambda function.

By default, CDK will create a new python lambda function to apply your k8s manifests. If you want to use an existing kubectl provider function, for example with tight trusted entities on your IAM Roles - you can import the existing provider and then use the imported provider when importing the cluster:

```ts
const handlerRole = iam.Role.fromRoleArn(this, 'HandlerRole', 'arn:aws:iam::123456789012:role/lambda-role');
// get the serviceToken from the custom resource provider
const functionArn = lambda.Function.fromFunctionName(this, 'ProviderOnEventFunc', 'ProviderframeworkonEvent-XXX').functionArn;
const kubectlProvider = eks.KubectlProvider.fromKubectlProviderAttributes(this, 'KubectlProvider', {
  functionArn,
  kubectlRoleArn: 'arn:aws:iam::123456789012:role/kubectl-role',
  handlerRole,
});

const cluster = eks.Cluster.fromClusterAttributes(this, 'Cluster', {
  clusterName: 'cluster',
  kubectlProvider,
});
```

#### Environment

You can configure the environment of this function by specifying it at cluster instantiation. For example, this can be useful in order to configure an http proxy:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

const cluster = new eks.Cluster(this, 'hello-eks', {
  version: eks.KubernetesVersion.V1_36,
  kubectlEnvironment: {
    'http_proxy': 'http://proxy.myproxy.com',
  },
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

#### Runtime

The kubectl handler uses `kubectl`, `helm` and the `aws` CLI in order to
interact with the cluster. These are bundled into AWS Lambda layers included in
the `@aws-cdk/lambda-layer-awscli` and `@aws-cdk/lambda-layer-kubectl` modules.

The version of kubectl used must be compatible with the Kubernetes version of the
cluster. kubectl is supported within one minor version (older or newer) of Kubernetes
(see [Kubernetes version skew policy](https://kubernetes.io/releases/version-skew-policy/#kubectl)).
Depending on which version of kubernetes you're targeting, you will need to use one of 
the `@aws-cdk/lambda-layer-kubectl-vXY` packages.

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

const cluster = new eks.Cluster(this, 'hello-eks', {
  version: eks.KubernetesVersion.V1_36,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

You can also specify a custom `lambda.LayerVersion` if you wish to use a
different version of these tools, or a version not available in any of the
`@aws-cdk/lambda-layer-kubectl-vXY` packages. The handler expects the layer to
include the following two executables:

```text
helm/helm
kubectl/kubectl
```

See more information in the
[Dockerfile](https://github.com/aws/aws-cdk/tree/main/packages/%40aws-cdk/lambda-layer-awscli/layer) for @aws-cdk/lambda-layer-awscli
and the
[Dockerfile](https://github.com/aws/aws-cdk/tree/main/packages/%40aws-cdk/lambda-layer-kubectl/layer) for @aws-cdk/lambda-layer-kubectl.

```ts
const layer = new lambda.LayerVersion(this, 'KubectlLayer', {
  code: lambda.Code.fromAsset('layer.zip'),
});
```

Now specify when the cluster is defined:

```ts
declare const layer: lambda.LayerVersion;
declare const vpc: ec2.Vpc;

const cluster1 = new eks.Cluster(this, 'MyCluster', {
  kubectlLayer: layer,
  vpc,
  clusterName: 'cluster-name',
  version: eks.KubernetesVersion.V1_36,
});

// or
const cluster2 = eks.Cluster.fromClusterAttributes(this, 'MyCluster', {
  kubectlLayer: layer,
  vpc,
  clusterName: 'cluster-name',
});
```

#### Memory

By default, the kubectl provider is configured with 1024MiB of memory. You can use the `kubectlMemory` option to specify the memory size for the AWS Lambda function:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

new eks.Cluster(this, 'MyCluster', {
  kubectlMemory: Size.gibibytes(4),
  version: eks.KubernetesVersion.V1_36,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});

// or
declare const vpc: ec2.Vpc;
eks.Cluster.fromClusterAttributes(this, 'MyCluster', {
  kubectlMemory: Size.gibibytes(4),
  vpc,
  clusterName: 'cluster-name',
});
```

### ARM64 Support

Instance types with `ARM64` architecture are supported in both managed nodegroup and self-managed capacity. Simply specify an ARM64 `instanceType` (such as `m6g.medium`), and the latest
Amazon Linux 2 AMI for ARM64 will be automatically selected.

```ts
declare const cluster: eks.Cluster;
// add a managed ARM64 nodegroup
cluster.addNodegroupCapacity('extra-ng-arm', {
  instanceTypes: [new ec2.InstanceType('m6g.medium')],
  minSize: 2,
});

// add a self-managed ARM64 nodegroup
cluster.addAutoScalingGroupCapacity('self-ng-arm', {
  instanceType: new ec2.InstanceType('m6g.medium'),
  minCapacity: 2,
})
```

### Masters Role

When you create a cluster, you can specify a `mastersRole`. The `Cluster` construct will associate this role with the `system:masters` [RBAC](https://kubernetes.io/docs/reference/access-authn-authz/rbac/) group, giving it super-user access to the cluster.

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

declare const role: iam.Role;
new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  mastersRole: role,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

In order to interact with your cluster through `kubectl`, you can use the `aws eks update-kubeconfig` [AWS CLI command](https://docs.aws.amazon.com/cli/latest/reference/eks/update-kubeconfig.html)
to configure your local kubeconfig. The EKS module will define a CloudFormation output in your stack which contains the command to run. For example:

```plaintext
Outputs:
ClusterConfigCommand43AAE40F = aws eks update-kubeconfig --name cluster-xxxxx --role-arn arn:aws:iam::112233445566:role/yyyyy
```

Execute the `aws eks update-kubeconfig ...` command in your terminal to create or update a local kubeconfig context:

```console
$ aws eks update-kubeconfig --name cluster-xxxxx --role-arn arn:aws:iam::112233445566:role/yyyyy
Added new context arn:aws:eks:rrrrr:112233445566:cluster/cluster-xxxxx to /home/boom/.kube/config
```

And now you can simply use `kubectl`:

```console
$ kubectl get all -n kube-system
NAME                           READY   STATUS    RESTARTS   AGE
pod/aws-node-fpmwv             1/1     Running   0          21m
pod/aws-node-m9htf             1/1     Running   0          21m
pod/coredns-5cb4fb54c7-q222j   1/1     Running   0          23m
pod/coredns-5cb4fb54c7-v9nxx   1/1     Running   0          23m
...
```

If you do not specify it, you won't have access to the cluster from outside of the CDK application.

> Note that `cluster.addManifest` and `new KubernetesManifest` will still work.

### Encryption

When you create an Amazon EKS cluster, envelope encryption of Kubernetes secrets using the AWS Key Management Service (AWS KMS) can be enabled.
The documentation on [creating a cluster](https://docs.aws.amazon.com/eks/latest/userguide/create-cluster.html)
can provide more details about the customer master key (CMK) that can be used for the encryption.

You can use the `secretsEncryptionKey` to configure which key the cluster will use to encrypt Kubernetes secrets. By default, an AWS Managed key will be used.

> This setting can only be specified when the cluster is created and cannot be updated.

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

const secretsKey = new kms.Key(this, 'SecretsKey');
const cluster = new eks.Cluster(this, 'MyCluster', {
  secretsEncryptionKey: secretsKey,
  version: eks.KubernetesVersion.V1_36,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

You can also use a similar configuration for running a cluster built using the FargateCluster construct.

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

const secretsKey = new kms.Key(this, 'SecretsKey');
const cluster = new eks.FargateCluster(this, 'MyFargateCluster', {
  secretsEncryptionKey: secretsKey,
  version: eks.KubernetesVersion.V1_36,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

The Amazon Resource Name (ARN) for that CMK can be retrieved.

```ts
declare const cluster: eks.Cluster;
const clusterEncryptionConfigKeyArn = cluster.clusterEncryptionConfigKeyArn;
```

### Hybrid Nodes

When you create an Amazon EKS cluster, you can configure it to leverage the [EKS Hybrid Nodes](https://aws.amazon.com/eks/hybrid-nodes/) feature, allowing you to use your on-premises and edge infrastructure as nodes in your EKS cluster. Refer to the Hyrid Nodes [networking documentation](https://docs.aws.amazon.com/eks/latest/userguide/hybrid-nodes-networking.html) to configure your on-premises network, node and pod CIDRs, access control, etc before creating your EKS Cluster.

Once you have identified the on-premises node and pod (optional) CIDRs you will use for your hybrid nodes and the workloads running on them, you can specify them during cluster creation using the `remoteNodeNetworks` and `remotePodNetworks` (optional) properties:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

new eks.Cluster(this, 'Cluster', {
  version: eks.KubernetesVersion.V1_36,
  kubectlLayer: new KubectlV36Layer(this, 'KubectlLayer'),
  remoteNodeNetworks: [
    {
      cidrs: ['10.0.0.0/16'],
    },
  ],
  remotePodNetworks: [
    {
      cidrs: ['192.168.0.0/16'],
    },
  ],
});
```

### Self-Managed Add-ons

Amazon EKS automatically installs self-managed add-ons such as the Amazon VPC CNI plugin for Kubernetes, kube-proxy, and CoreDNS for every cluster. You can change the default configuration of the add-ons and update them when desired. If you wish to create a cluster without the default add-ons, set `bootstrapSelfManagedAddons` as `false`. When this is set to false, make sure to install the necessary alternatives which provide functionality that enables pod and service operations for your EKS cluster.

> Changing the value of `bootstrapSelfManagedAddons` after the EKS cluster creation will result in a replacement of the cluster.

## Permissions and Security

Amazon EKS provides several mechanism of securing the cluster and granting permissions to specific IAM users and roles.

### AWS IAM Mapping

As described in the [Amazon EKS User Guide](https://docs.aws.amazon.com/en_us/eks/latest/userguide/add-user-role.html), you can map AWS IAM users and roles to [Kubernetes Role-based access control (RBAC)](https://kubernetes.io/docs/reference/access-authn-authz/rbac).

The Amazon EKS construct manages the *aws-auth* `ConfigMap` Kubernetes resource on your behalf and exposes an API through the `cluster.awsAuth` for mapping
users, roles and accounts.

Furthermore, when auto-scaling group capacity is added to the cluster, the IAM instance role of the auto-scaling group will be automatically mapped to RBAC so nodes can connect to the cluster. No manual mapping is required.

For example, let's say you want to grant an IAM user administrative privileges on your cluster:

```ts
declare const cluster: eks.Cluster;
const adminUser = new iam.User(this, 'Admin');
cluster.awsAuth.addUserMapping(adminUser, { groups: [ 'system:masters' ]});
```

A convenience method for mapping a role to the `system:masters` group is also available:

```ts
declare const cluster: eks.Cluster;
declare const role: iam.Role;
cluster.awsAuth.addMastersRole(role);
```

To access the Kubernetes resources from the console, make sure your viewing principal is defined
in the `aws-auth` ConfigMap. Some options to consider:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';
declare const cluster: eks.Cluster;
declare const your_current_role: iam.Role;
declare const vpc: ec2.Vpc;

// Option 1: Add your current assumed IAM role to system:masters. Make sure to add relevant policies.
cluster.awsAuth.addMastersRole(your_current_role);

your_current_role.addToPolicy(new iam.PolicyStatement({
  actions: [
    'eks:AccessKubernetesApi',
    'eks:Describe*',
    'eks:List*',
],
  resources: [ cluster.clusterArn ],
}));
```

```ts
// Option 2: create your custom mastersRole with scoped assumeBy arn as the Cluster prop. Switch to this role from the AWS console.
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';
declare const vpc: ec2.Vpc;


const mastersRole = new iam.Role(this, 'MastersRole', {
  assumedBy: new iam.ArnPrincipal('arn_for_trusted_principal'),
});

const cluster = new eks.Cluster(this, 'EksCluster', {
  vpc,
  version: eks.KubernetesVersion.V1_36,
  kubectlLayer: new KubectlV36Layer(this, 'KubectlLayer'),
  mastersRole,
});

mastersRole.addToPolicy(new iam.PolicyStatement({
  actions: [
    'eks:AccessKubernetesApi',
    'eks:Describe*',
    'eks:List*',
],
  resources: [ cluster.clusterArn ],
}));
```

```ts
// Option 3: Create a new role that allows the account root principal to assume. Add this role in the `system:masters` and witch to this role from the AWS console.
declare const cluster: eks.Cluster;

const consoleReadOnlyRole = new iam.Role(this, 'ConsoleReadOnlyRole', {
  assumedBy: new iam.ArnPrincipal('arn_for_trusted_principal'),
});
consoleReadOnlyRole.addToPolicy(new iam.PolicyStatement({
  actions: [
    'eks:AccessKubernetesApi',
    'eks:Describe*',
    'eks:List*',
],
  resources: [ cluster.clusterArn ],
}));

// Add this role to system:masters RBAC group
cluster.awsAuth.addMastersRole(consoleReadOnlyRole)
```

### Access Config

Amazon EKS supports three modes of authentication: `CONFIG_MAP`, `API_AND_CONFIG_MAP`, and `API`. You can enable cluster
to use access entry APIs by using authenticationMode `API` or `API_AND_CONFIG_MAP`. Use authenticationMode `CONFIG_MAP`
to continue using aws-auth configMap exclusively. When `API_AND_CONFIG_MAP` is enabled, the cluster will source authenticated
AWS IAM principals from both Amazon EKS access entry APIs and the aws-auth configMap, with priority given to the access entry API.

To specify the `authenticationMode`:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';
declare const vpc: ec2.Vpc;

new eks.Cluster(this, 'Cluster', {
  vpc,
  version: eks.KubernetesVersion.V1_36,
  kubectlLayer: new KubectlV36Layer(this, 'KubectlLayer'),
  authenticationMode: eks.AuthenticationMode.API_AND_CONFIG_MAP,
});
```

> **Note** - Switching authentication modes on an existing cluster is a one-way operation. You can switch from
> `CONFIG_MAP` to `API_AND_CONFIG_MAP`. You can then switch from `API_AND_CONFIG_MAP` to `API`.
> You cannot revert these operations in the opposite direction. Meaning you cannot switch back to
> `CONFIG_MAP` or `API_AND_CONFIG_MAP` from `API`. And you cannot switch back to `CONFIG_MAP` from `API_AND_CONFIG_MAP`.

Read [A deep dive into simplified Amazon EKS access management controls
](https://aws.amazon.com/blogs/containers/a-deep-dive-into-simplified-amazon-eks-access-management-controls/) for more details.

You can disable granting the cluster admin permissions to the cluster creator role on bootstrapping by setting 
`bootstrapClusterCreatorAdminPermissions` to false. 

> **Note** - Switching `bootstrapClusterCreatorAdminPermissions` on an existing cluster would cause cluster replacement and should be avoided in production.


### Access Entry

An access entry is a cluster identity—directly linked to an AWS IAM principal user or role that is used to authenticate to
an Amazon EKS cluster. An Amazon EKS access policy authorizes an access entry to perform specific cluster actions.

Access policies are Amazon EKS-specific policies that assign Kubernetes permissions to access entries. Amazon EKS supports
only predefined and AWS managed policies. Access policies are not AWS IAM entities and are defined and managed by Amazon EKS.
Amazon EKS access policies include permission sets that support common use cases of administration, editing, or read-only access
to Kubernetes resources. See [Access Policy Permissions](https://docs.aws.amazon.com/eks/latest/userguide/access-policies.html#access-policy-permissions) for more details.

Use `AccessPolicy` to include predefined AWS managed policies:

```ts
// AmazonEKSClusterAdminPolicy with `cluster` scope
eks.AccessPolicy.fromAccessPolicyName('AmazonEKSClusterAdminPolicy', {
  accessScopeType: eks.AccessScopeType.CLUSTER,
});
// AmazonEKSAdminPolicy with `namespace` scope
eks.AccessPolicy.fromAccessPolicyName('AmazonEKSAdminPolicy', {
  accessScopeType: eks.AccessScopeType.NAMESPACE,
  namespaces: ['foo', 'bar'] } );
```

Use `grantAccess()` to grant the AccessPolicy to an IAM principal:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';
declare const vpc: ec2.Vpc;

const clusterAdminRole = new iam.Role(this, 'ClusterAdminRole', {
  assumedBy: new iam.ArnPrincipal('arn_for_trusted_principal'),
});

const eksAdminRole = new iam.Role(this, 'EKSAdminRole', {
  assumedBy: new iam.ArnPrincipal('arn_for_trusted_principal'),
});

const eksAdminViewRole = new iam.Role(this, 'EKSAdminViewRole', {
  assumedBy: new iam.ArnPrincipal('arn_for_trusted_principal'),
});

const cluster = new eks.Cluster(this, 'Cluster', {
  vpc,
  mastersRole: clusterAdminRole,
  version: eks.KubernetesVersion.V1_36,
  kubectlLayer: new KubectlV36Layer(this, 'KubectlLayer'),
  authenticationMode: eks.AuthenticationMode.API_AND_CONFIG_MAP,
});

// Cluster Admin role for this cluster
cluster.grantAccess('clusterAdminAccess', clusterAdminRole.roleArn, [
  eks.AccessPolicy.fromAccessPolicyName('AmazonEKSClusterAdminPolicy', {
    accessScopeType: eks.AccessScopeType.CLUSTER,
  }),
]);

// EKS Admin role for specified namespaces of this cluster
cluster.grantAccess('eksAdminRoleAccess', eksAdminRole.roleArn, [
  eks.AccessPolicy.fromAccessPolicyName('AmazonEKSAdminPolicy', { 
    accessScopeType: eks.AccessScopeType.NAMESPACE,
    namespaces: ['foo', 'bar'],
  }),
]);

// EKS Admin Viewer role for specified namespaces of this cluster
cluster.grantAccess('eksAdminViewRoleAccess', eksAdminViewRole.roleArn, [
  eks.AccessPolicy.fromAccessPolicyName('AmazonEKSAdminViewPolicy', {
    accessScopeType: eks.AccessScopeType.NAMESPACE,
    namespaces: ['foo', 'bar'],
  }),
]);
```

You can optionally specify an access entry type when granting access:

```ts
declare const cluster: eks.Cluster;
declare const nodeRole: iam.Role;

// For EKS Auto Mode node roles
cluster.grantAccess('NodeAccess', nodeRole.roleArn, [], { accessEntryType: eks.AccessEntryType.EC2 });
```

Supported types: `STANDARD` (default), `FARGATE_LINUX`, `EC2_LINUX`, `EC2_WINDOWS`, `EC2`, `HYBRID_LINUX`, `HYPERPOD_LINUX`.

**Note**: `EC2`, `HYBRID_LINUX`, and `HYPERPOD_LINUX` types cannot have access policies attached.

### Migrating from ConfigMap to Access Entry

If the cluster is created with the `authenticationMode` property left undefined,
it will default to `CONFIG_MAP`. 

The update path is: 

`undefined`(`CONFIG_MAP`) -> `API_AND_CONFIG_MAP` -> `API`

If you have explicitly declared `AwsAuth` resources and then try to switch to the `API` mode, which no longer supports the
`ConfigMap`, AWS CDK will throw an error as a protective measure to prevent you from losing all the access entries in the `ConfigMap`. In this case, you will need to remove all the declared `AwsAuth` resources explicitly and define the access entries before you are allowed to transition to the `API` mode.

> **Note** - This is a one-way transition. Once you switch to the `API` mode,
you will not be able to switch back. Therefore, it is crucial to ensure that you have defined all the necessary access entries before making the switch to the `API` mode.

### Cluster Security Group

When you create an Amazon EKS cluster, a [cluster security group](https://docs.aws.amazon.com/eks/latest/userguide/sec-group-reqs.html)
is automatically created as well. This security group is designed to allow all traffic from the control plane and managed node groups to flow freely
between each other.

The ID for that security group can be retrieved after creating the cluster.

```ts
declare const cluster: eks.Cluster;
const clusterSecurityGroupId = cluster.clusterSecurityGroupId;
```

### Node SSH Access

If you want to be able to SSH into your worker nodes, you must already have an SSH key in the region you're connecting to and pass it when
you add capacity to the cluster. You must also be able to connect to the hosts (meaning they must have a public IP and you
should be allowed to connect to them on port 22):

See [SSH into nodes](test/example.ssh-into-nodes.lit.ts) for a code example.

If you want to SSH into nodes in a private subnet, you should set up a bastion host in a public subnet. That setup is recommended, but is
unfortunately beyond the scope of this documentation.

### Service Accounts

With services account you can provide Kubernetes Pods access to AWS resources.

```ts
declare const cluster: eks.Cluster;
// add service account
const serviceAccount = cluster.addServiceAccount('MyServiceAccount');

const bucket = new s3.Bucket(this, 'Bucket');
bucket.grantReadWrite(serviceAccount);

const mypod = cluster.addManifest('mypod', {
  apiVersion: 'v1',
  kind: 'Pod',
  metadata: { name: 'mypod' },
  spec: {
    serviceAccountName: serviceAccount.serviceAccountName,
    containers: [
      {
        name: 'hello',
        image: 'paulbouwer/hello-kubernetes:1.5',
        ports: [ { containerPort: 8080 } ],
      },
    ],
  },
});

// create the resource after the service account.
mypod.node.addDependency(serviceAccount);

// print the IAM role arn for this service account
new CfnOutput(this, 'ServiceAccountIamRole', { value: serviceAccount.role.roleArn });
```

Note that using `serviceAccount.serviceAccountName` above **does not** translate into a resource dependency.
This is why an explicit dependency is needed. See <https://github.com/aws/aws-cdk/issues/9910> for more details.

It is possible to pass annotations and labels to the service account.

```ts
declare const cluster: eks.Cluster;
// add service account with annotations and labels
const serviceAccount = cluster.addServiceAccount('MyServiceAccount', {
  annotations: {
    'eks.amazonaws.com/sts-regional-endpoints': 'false',
  },
  labels: {
    'some-label': 'with-some-value',
  },
});
```

To overwrite an existing service account, use the `overwriteServiceAccount` property:

```ts
declare const cluster: eks.Cluster;
// overwrite existing service account
const serviceAccount = cluster.addServiceAccount('MyServiceAccount', {
  overwriteServiceAccount: true,
});
```

You can also add service accounts to existing clusters.
To do so, pass the `openIdConnectProvider` property when you import the cluster into the application.

```ts
// you can import an existing provider
const provider = eks.OidcProviderNative.fromOidcProviderArn(this, 'Provider', 'arn:aws:iam::123456:oidc-provider/oidc.eks.eu-west-1.amazonaws.com/id/AB123456ABC');

// or create a new one using an existing issuer url
declare const issuerUrl: string;
const provider2 = new eks.OidcProviderNative(this, 'Provider', {
  url: issuerUrl,
});

const cluster = eks.Cluster.fromClusterAttributes(this, 'MyCluster', {
  clusterName: 'Cluster',
  openIdConnectProvider: provider,
  kubectlRoleArn: 'arn:aws:iam::123456:role/service-role/k8sservicerole',
});

const serviceAccount = cluster.addServiceAccount('MyServiceAccount');

const bucket = new s3.Bucket(this, 'Bucket');
bucket.grantReadWrite(serviceAccount);
```

Note that adding service accounts requires running `kubectl` commands against the cluster.
This means you must also pass the `kubectlRoleArn` when importing the cluster.
See [Using existing Clusters](https://github.com/aws/aws-cdk/tree/main/packages/aws-cdk-lib/aws-eks#using-existing-clusters).


##### Migrating from eks.OpenIdConnectProvider to eks.OidcProviderNative

`eks.OpenIdConnectProvider` creates an IAM OIDC (OpenId Connect) provider using a custom resource while `eks.OidcProviderNative` uses the CFN L1 (AWS::IAM::OidcProvider) to create the provider. It is recommended for new and existing projects to use `eks.OidcProviderNative`. Migrating from the `eks.OpenIdConnectProvider` is not as trivial as switching out the property since the property controls the creation of a resource whose type is changing. Due to the potential complexlity of the migration and the requirement of a manual step (`cdk import`) we are not deprecating the `eks.OpenIdConnectProvider` construct but encourge you to migrate.

To migrate without temporarily removing the OIDCProvider, follow these steps:

1. Set the `removalPolicy` of `cluster.openIdConnectProvider` to `RETAIN`.

   ```ts
   import * as cdk from 'aws-cdk-lib';
   declare const cluster: eks.Cluster;

   cdk.RemovalPolicies.of(cluster.openIdConnectProvider).apply(cdk.RemovalPolicy.RETAIN);
   ```

2. Run `cdk diff` to verify the changes are expected then `cdk deploy`.

3. Add the following to the `context` field of your `cdk.json` to enable the feature flag that creates the native oidc provider.

   ```json
   "@aws-cdk/aws-eks:useNativeOidcProvider": true,
   ```

4. Run `cdk diff` and ensure the changes are expected. Example of an expected diff:

   ```bash
   Resources
   [-] Custom::AWSCDKOpenIdConnectProvider TestCluster/OpenIdConnectProvider/Resource TestClusterOpenIdConnectProviderE18F0FD0 orphan
   [-] AWS::IAM::Role Custom::AWSCDKOpenIdConnectProviderCustomResourceProvider/Role CustomAWSCDKOpenIdConnectProviderCustomResourceProviderRole517FED65 destroy
   [-] AWS::Lambda::Function Custom::AWSCDKOpenIdConnectProviderCustomResourceProvider/Handler CustomAWSCDKOpenIdConnectProviderCustomResourceProviderHandlerF2C543E0 destroy
   [+] AWS::IAM::OIDCProvider TestCluster/OidcProviderNative TestClusterOidcProviderNative0BE3F155
   ```

5. Run `cdk import --force` and provide the ARN of the existing OpenIdConnectProvider when prompted. You will get a warning about pending changes to existing resources which is expected.

6. Run `cdk deploy` to apply any pending changes. This will apply the destroy/orphan changes in the above example.

If you are creating the OpenIdConnectProvider manually via `new eks.OpenIdConnectProvider`, follow these steps:

1. Set the `removalPolicy` of the existing `OpenIdConnectProvider` to `RemovalPolicy.RETAIN`.

   ```ts
   import * as cdk from 'aws-cdk-lib';
   // Step 1: Add retain policy to existing provider
   const existingProvider = new eks.OpenIdConnectProvider(this, 'Provider', {
     url: 'https://oidc.eks.us-west-2.amazonaws.com/id/EXAMPLE',
     removalPolicy: cdk.RemovalPolicy.RETAIN, // Add this line
   });
   ```

2. Deploy with the retain policy to avoid deletion of the underlying resource.

   ```bash
   cdk deploy
   ```

3. Replace `OpenIdConnectProvider` with `OidcProviderNative` in your code.

   ```ts
   // Step 3: Replace with native provider
   const nativeProvider = new eks.OidcProviderNative(this, 'Provider', {
     url: 'https://oidc.eks.us-west-2.amazonaws.com/id/EXAMPLE',
   });
   ```

4. Run `cdk diff` and verify the changes are expected. Example of an expected diff:

   ```bash
   Resources
   [-] Custom::AWSCDKOpenIdConnectProvider TestCluster/OpenIdConnectProvider/Resource TestClusterOpenIdConnectProviderE18F0FD0 orphan
   [-] AWS::IAM::Role Custom::AWSCDKOpenIdConnectProviderCustomResourceProvider/Role CustomAWSCDKOpenIdConnectProviderCustomResourceProviderRole517FED65 destroy
   [-] AWS::Lambda::Function Custom::AWSCDKOpenIdConnectProviderCustomResourceProvider/Handler CustomAWSCDKOpenIdConnectProviderCustomResourceProviderHandlerF2C543E0 destroy
   [+] AWS::IAM::OIDCProvider TestCluster/OidcProviderNative TestClusterOidcProviderNative0BE3F155
   ```

5. Run `cdk import --force` to import the existing OIDC provider resource by providing the existing ARN.

6. Run `cdk deploy` to apply any pending changes. This will apply the destroy/orphan operations in the example diff above.


### Pod Identities

[Amazon EKS Pod Identities](https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html) is a feature that simplifies how
Kubernetes applications running on Amazon EKS can obtain AWS IAM credentials. It provides a way to associate an IAM role with a
Kubernetes service account, allowing pods to retrieve temporary AWS credentials without the need
to manage IAM roles and policies directly.

By default, `ServiceAccount` creates an `OpenIdConnectProvider` for 
[IRSA(IAM roles for service accounts)](https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html) if
`identityType` is `undefined` or `IdentityType.IRSA`.

You may opt in Amaozn EKS Pod Identities as below:

```ts
declare const cluster: eks.Cluster;

new eks.ServiceAccount(this, 'ServiceAccount', {
  cluster,
  name: 'test-sa',
  namespace: 'default',
  identityType: eks.IdentityType.POD_IDENTITY,
});
```

When you create the ServiceAccount with the `identityType` set to `POD_IDENTITY`,
`ServiceAccount` contruct will perform the following actions behind the scenes:

1. It will create an IAM role with the necessary trust policy to allow the "pods.eks.amazonaws.com" principal to assume the role.
This trust policy grants the EKS service the permission to retrieve temporary AWS credentials on behalf of the pods using this service account.

2. It will enable the "Amazon EKS Pod Identity Agent" add-on on the EKS cluster. This add-on is responsible for managing the temporary
AWS credentials and making them available to the pods.

3. It will create an association between the IAM role and the Kubernetes service account. This association allows the pods using this
service account to obtain the temporary AWS credentials from the associated IAM role.

This simplifies the process of configuring IAM permissions for your Kubernetes applications running on Amazon EKS. It handles the creation of the IAM role,
the installation of the Pod Identity Agent add-on, and the association between the role and the service account, making it easier to manage AWS credentials
for your applications.

## Applying Kubernetes Resources

The library supports several popular resource deployment mechanisms, among which are:

### Kubernetes Manifests

The `KubernetesManifest` construct or `cluster.addManifest` method can be used
to apply Kubernetes resource manifests to this cluster.

> When using `cluster.addManifest`, the manifest construct is defined within the cluster's stack scope. If the manifest contains
> attributes from a different stack which depend on the cluster stack, a circular dependency will be created and you will get a synth time error.
> To avoid this, directly use `new KubernetesManifest` to create the manifest in the scope of the other stack.

The following examples will deploy the [paulbouwer/hello-kubernetes](https://github.com/paulbouwer/hello-kubernetes)
service on the cluster:

```ts
declare const cluster: eks.Cluster;
const appLabel = { app: "hello-kubernetes" };

const deployment = {
  apiVersion: "apps/v1",
  kind: "Deployment",
  metadata: { name: "hello-kubernetes" },
  spec: {
    replicas: 3,
    selector: { matchLabels: appLabel },
    template: {
      metadata: { labels: appLabel },
      spec: {
        containers: [
          {
            name: "hello-kubernetes",
            image: "paulbouwer/hello-kubernetes:1.5",
            ports: [ { containerPort: 8080 } ],
          },
        ],
      },
    },
  },
};

const service = {
  apiVersion: "v1",
  kind: "Service",
  metadata: { name: "hello-kubernetes" },
  spec: {
    type: "LoadBalancer",
    ports: [ { port: 80, targetPort: 8080 } ],
    selector: appLabel,
  }
};

// option 1: use a construct
new eks.KubernetesManifest(this, 'hello-kub', {
  cluster,
  manifest: [ deployment, service ],
});

// or, option2: use `addManifest`
cluster.addManifest('hello-kub', service, deployment);
```

#### ALB Controller Integration

The `KubernetesManifest` construct can detect ingress resources inside your manifest and automatically add the necessary annotations
so they are picked up by the ALB Controller.

> See [Alb Controller](#alb-controller)

To that end, it offers the following properties:

* `ingressAlb` - Signal that the ingress detection should be done.
* `ingressAlbScheme` - Which ALB scheme should be applied. Defaults to `internal`.

#### Adding resources from a URL

The following example will deploy the resource manifest hosting on remote server:

```text
// This example is only available in TypeScript

import * as yaml from 'js-yaml';
import * as request from 'sync-request';

declare const cluster: eks.Cluster;
const manifestUrl = 'https://url/of/manifest.yaml';
const manifest = yaml.safeLoadAll(request('GET', manifestUrl).getBody());
cluster.addManifest('my-resource', manifest);
```

#### Dependencies

There are cases where Kubernetes resources must be deployed in a specific order.
For example, you cannot define a resource in a Kubernetes namespace before the
namespace was created.

You can represent dependencies between `KubernetesManifest`s using
`resource.node.addDependency()`:

```ts
declare const cluster: eks.Cluster;
const namespace = cluster.addManifest('my-namespace', {
  apiVersion: 'v1',
  kind: 'Namespace',
  metadata: { name: 'my-app' },
});

const service = cluster.addManifest('my-service', {
  metadata: {
    name: 'myservice',
    namespace: 'my-app',
  },
  spec: { }, // ...
});

service.node.addDependency(namespace); // will apply `my-namespace` before `my-service`.
```

**NOTE:** when a `KubernetesManifest` includes multiple resources (either directly
or through `cluster.addManifest()`) (e.g. `cluster.addManifest('foo', r1, r2,
r3,...)`), these resources will be applied as a single manifest via `kubectl`
and will be applied sequentially (the standard behavior in `kubectl`).

---

Since Kubernetes manifests are implemented as CloudFormation resources in the
CDK. This means that if the manifest is deleted from your code (or the stack is
deleted), the next `cdk deploy` will issue a `kubectl delete` command and the
Kubernetes resources in that manifest will be deleted.

#### Resource Pruning

When a resource is deleted from a Kubernetes manifest, the EKS module will
automatically delete these resources by injecting a _prune label_ to all
manifest resources. This label is then passed to [`kubectl apply --prune`].

[`kubectl apply --prune`]: https://kubernetes.io/docs/tasks/manage-kubernetes-objects/declarative-config/#alternative-kubectl-apply-f-directory-prune-l-your-label

Pruning is enabled by default but can be disabled through the `prune` option
when a cluster is defined:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

new eks.Cluster(this, 'MyCluster', {
  version: eks.KubernetesVersion.V1_36,
  prune: false,
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

#### Manifests Validation

The `kubectl` CLI supports applying a manifest by skipping the validation.
This can be accomplished by setting the `skipValidation` flag to `true` in the `KubernetesManifest` props.

```ts
declare const cluster: eks.Cluster;
new eks.KubernetesManifest(this, 'HelloAppWithoutValidation', {
  cluster,
  manifest: [{ foo: 'bar' }],
  skipValidation: true,
});
```

### Helm Charts

The `HelmChart` construct or `cluster.addHelmChart` method can be used
to add Kubernetes resources to this cluster using Helm.

> When using `cluster.addHelmChart`, the manifest construct is defined within the cluster's stack scope. If the manifest contains
> attributes from a different stack which depend on the cluster stack, a circular dependency will be created and you will get a synth time error.
> To avoid this, directly use `new HelmChart` to create the chart in the scope of the other stack.

The following example will install the [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/) to your cluster using Helm.

```ts
declare const cluster: eks.Cluster;
// option 1: use a construct
new eks.HelmChart(this, 'NginxIngress', {
  cluster,
  chart: 'nginx-ingress',
  repository: 'https://helm.nginx.com/stable',
  namespace: 'kube-system',
});

// or, option2: use `addHelmChart`
cluster.addHelmChart('NginxIngress', {
  chart: 'nginx-ingress',
  repository: 'https://helm.nginx.com/stable',
  namespace: 'kube-system',
});
```

Helm charts will be installed and updated using `helm upgrade --install`, where a few parameters
are being passed down (such as `repo`, `values`, `version`, `namespace`, `wait`, `timeout`, etc).
This means that if the chart is added to CDK with the same release name, it will try to update
the chart in the cluster.

Additionally, the `chartAsset` property can be an `aws-s3-assets.Asset`. This allows the use of local, private helm charts.

```ts
import * as s3Assets from 'aws-cdk-lib/aws-s3-assets';

declare const cluster: eks.Cluster;
const chartAsset = new s3Assets.Asset(this, 'ChartAsset', {
  path: '/path/to/asset'
});

cluster.addHelmChart('test-chart', {
  chartAsset: chartAsset,
});
```

Nested values passed to the `values` parameter should be provided as a nested dictionary:

```ts
declare const cluster: eks.Cluster;

cluster.addHelmChart('ExternalSecretsOperator', {
  chart: 'external-secrets',
  release: 'external-secrets',
  repository: 'https://charts.external-secrets.io',
  namespace: 'external-secrets',
  values: {
    installCRDs: true,
    webhook: {
      port: 9443
    }
  },
});
```

Helm chart can come with Custom Resource Definitions (CRDs) defined that by default will be installed by helm as well. However in special cases it might be needed to skip the installation of CRDs, for that the property `skipCrds` can be used.

```ts
declare const cluster: eks.Cluster;
// option 1: use a construct
new eks.HelmChart(this, 'NginxIngress', {
  cluster,
  chart: 'nginx-ingress',
  repository: 'https://helm.nginx.com/stable',
  namespace: 'kube-system',
  skipCrds: true,
});
```

### OCI Charts

OCI charts are also supported.
Also replace the `${VARS}` with appropriate values.

```ts
declare const cluster: eks.Cluster;
// option 1: use a construct
new eks.HelmChart(this, 'MyOCIChart', {
  cluster,
  chart: 'some-chart',
  repository: 'oci://${ACCOUNT_ID}.dkr.ecr.${ACCOUNT_REGION}.amazonaws.com/${REPO_NAME}',
  namespace: 'oci',
  version: '0.0.1'
});

```

Helm charts are implemented as CloudFormation resources in CDK.
This means that if the chart is deleted from your code (or the stack is
deleted), the next `cdk deploy` will issue a `helm uninstall` command and the
Helm chart will be deleted.

When there is no `release` defined, a unique ID will be allocated for the release based
on the construct path.

By default, all Helm charts will be installed concurrently. In some cases, this
could cause race conditions where two Helm charts attempt to deploy the same
resource or if Helm charts depend on each other. You can use
`chart.node.addDependency()` in order to declare a dependency order between
charts:

```ts
declare const cluster: eks.Cluster;
const chart1 = cluster.addHelmChart('MyChart', {
  chart: 'foo',
});
const chart2 = cluster.addHelmChart('MyChart', {
  chart: 'bar',
});

chart2.node.addDependency(chart1);
```

### CDK8s Charts

[CDK8s](https://cdk8s.io/) is an open-source library that enables Kubernetes manifest authoring using familiar programming languages. It is founded on the same technologies as the AWS CDK, such as [`constructs`](https://github.com/aws/constructs) and [`jsii`](https://github.com/aws/jsii).

> To learn more about cdk8s, visit the [Getting Started](https://cdk8s.io/docs/latest/getting-started/) tutorials.

The EKS module natively integrates with cdk8s and allows you to apply cdk8s charts on AWS EKS clusters via the `cluster.addCdk8sChart` method.

In addition to `cdk8s`, you can also use [`cdk8s+`](https://cdk8s.io/docs/latest/plus/), which provides higher level abstraction for the core kubernetes api objects.
You can think of it like the `L2` constructs for Kubernetes. Any other `cdk8s` based libraries are also supported, for example [`cdk8s-debore`](https://github.com/toricls/cdk8s-debore).

To get started, add the following dependencies to your `package.json` file:

```json
"dependencies": {
  "cdk8s": "^2.0.0",
  "cdk8s-plus-25": "^2.0.0",
  "constructs": "^10.5.0"
}
```

Note that here we are using `cdk8s-plus-25` as we are targeting Kubernetes version 1.25.0. If you operate a different kubernetes version, you should
use the corresponding `cdk8s-plus-XX` library.
See [Select the appropriate cdk8s+ library](https://cdk8s.io/docs/latest/plus/#i-operate-kubernetes-version-1xx-which-cdk8s-library-should-i-be-using)
for more details.

Similarly to how you would create a stack by extending `aws-cdk-lib.Stack`, we recommend you create a chart of your own that extends `cdk8s.Chart`,
and add your kubernetes resources to it. You can use `aws-cdk` construct attributes and properties inside your `cdk8s` construct freely.

In this example we create a chart that accepts an `s3.Bucket` and passes its name to a kubernetes pod as an environment variable.

`+ my-chart.ts`

```ts nofixture
import { aws_s3 as s3 } from 'aws-cdk-lib';
import * as constructs from 'constructs';
import * as cdk8s from 'cdk8s';
import * as kplus from 'cdk8s-plus-25';

export interface MyChartProps {
  readonly bucket: s3.Bucket;
}

export class MyChart extends cdk8s.Chart {
  constructor(scope: constructs.Construct, id: string, props: MyChartProps) {
    super(scope, id);

    new kplus.Pod(this, 'Pod', {
      containers: [
        {
          image: 'my-image',
          envVariables: {
            BUCKET_NAME: kplus.EnvValue.fromValue(props.bucket.bucketName),
          },
        }
      ],
    });
  }
}
```

Then, in your AWS CDK app:

```ts fixture=cdk8schart
declare const cluster: eks.Cluster;

// some bucket..
const bucket = new s3.Bucket(this, 'Bucket');

// create a cdk8s chart and use `cdk8s.App` as the scope.
const myChart = new MyChart(new cdk8s.App(), 'MyChart', { bucket });

// add the cdk8s chart to the cluster
cluster.addCdk8sChart('my-chart', myChart);
```

#### Custom CDK8s Constructs

You can also compose a few stock `cdk8s+` constructs into your own custom construct. However, since mixing scopes between `aws-cdk` and `cdk8s` is currently not supported, the `Construct` class
you'll need to use is the one from the [`constructs`](https://github.com/aws/constructs) module, and not from `aws-cdk-lib` like you normally would.
This is why we used `new cdk8s.App()` as the scope of the chart above.

```ts nofixture
import * as constructs from 'constructs';
import * as cdk8s from 'cdk8s';
import * as kplus from 'cdk8s-plus-25';

export interface LoadBalancedWebService {
  readonly port: number;
  readonly image: string;
  readonly replicas: number;
}

const app = new cdk8s.App();
const chart = new cdk8s.Chart(app, 'my-chart');

export class LoadBalancedWebService extends constructs.Construct {
  constructor(scope: constructs.Construct, id: string, props: LoadBalancedWebService) {
    super(scope, id);

    const deployment = new kplus.Deployment(chart, 'Deployment', {
      replicas: props.replicas,
      containers: [ new kplus.Container({ image: props.image }) ],
    });

    deployment.exposeViaService({
      ports: [
        { port: props.port },
      ],
      serviceType: kplus.ServiceType.LOAD_BALANCER,
    });
  }
}
```

#### Manually importing k8s specs and CRD's

If you find yourself unable to use `cdk8s+`, or just like to directly use the `k8s` native objects or CRD's, you can do so by manually importing them using the `cdk8s-cli`.

See [Importing kubernetes objects](https://cdk8s.io/docs/latest/cli/import/) for detailed instructions.

## Patching Kubernetes Resources

The `KubernetesPatch` construct can be used to update existing kubernetes
resources. The following example can be used to patch the `hello-kubernetes`
deployment from the example above with 5 replicas.

```ts
declare const cluster: eks.Cluster;
new eks.KubernetesPatch(this, 'hello-kub-deployment-label', {
  cluster,
  resourceName: "deployment/hello-kubernetes",
  applyPatch: { spec: { replicas: 5 } },
  restorePatch: { spec: { replicas: 3 } },
})
```

## Querying Kubernetes Resources

The `KubernetesObjectValue` construct can be used to query for information about kubernetes objects,
and use that as part of your CDK application.

For example, you can fetch the address of a [`LoadBalancer`](https://kubernetes.io/docs/concepts/services-networking/service/#loadbalancer) type service:

```ts
declare const cluster: eks.Cluster;
// query the load balancer address
const myServiceAddress = new eks.KubernetesObjectValue(this, 'LoadBalancerAttribute', {
  cluster: cluster,
  objectType: 'service',
  objectName: 'my-service',
  jsonPath: '.status.loadBalancer.ingress[0].hostname', // https://kubernetes.io/docs/reference/kubectl/jsonpath/
});

// pass the address to a lambda function
const proxyFunction = new lambda.Function(this, 'ProxyFunction', {
  handler: 'index.handler',
  code: lambda.Code.fromInline('my-code'),
  runtime: lambda.Runtime.NODEJS_LATEST,
  environment: {
    myServiceAddress: myServiceAddress.value,
  },
})
```

Specifically, since the above use-case is quite common, there is an easier way to access that information:

```ts
declare const cluster: eks.Cluster;
const loadBalancerAddress = cluster.getServiceLoadBalancerAddress('my-service');
```

## Add-ons

[Add-ons](https://docs.aws.amazon.com/eks/latest/userguide/eks-add-ons.html) is a software that provides supporting operational capabilities to Kubernetes applications. The EKS module supports adding add-ons to your cluster using the `eks.Addon` class.

```ts
declare const cluster: eks.Cluster;

new eks.Addon(this, 'Addon', {
  cluster,
  addonName: 'coredns',
  addonVersion: 'v1.11.4-eksbuild.2',
  // whether to preserve the add-on software on your cluster but Amazon EKS stops managing any settings for the add-on.
  preserveOnDelete: false,
  configurationValues: {
    replicaCount: 2,
  },
});
```

## Using existing clusters

The Amazon EKS library allows defining Kubernetes resources such as [Kubernetes
manifests](#kubernetes-resources) and [Helm charts](#helm-charts) on clusters
that are not defined as part of your CDK app.

First, you'll need to "import" a cluster to your CDK app. To do that, use the
`eks.Cluster.fromClusterAttributes()` static method:

```ts
const cluster = eks.Cluster.fromClusterAttributes(this, 'MyCluster', {
  clusterName: 'my-cluster-name',
  kubectlRoleArn: 'arn:aws:iam::1111111:role/iam-role-that-has-masters-access',
});
```

Then, you can use `addManifest` or `addHelmChart` to define resources inside
your Kubernetes cluster. For example:

```ts
declare const cluster: eks.Cluster;
cluster.addManifest('Test', {
  apiVersion: 'v1',
  kind: 'ConfigMap',
  metadata: {
    name: 'myconfigmap',
  },
  data: {
    Key: 'value',
    Another: '123454',
  },
});
```

At the minimum, when importing clusters for `kubectl` management, you will need
to specify:

* `clusterName` - the name of the cluster.
* `kubectlRoleArn` - the ARN of an IAM role mapped to the `system:masters` RBAC
  role. If the cluster you are importing was created using the AWS CDK, the
  CloudFormation stack has an output that includes an IAM role that can be used.
  Otherwise, you can create an IAM role and map it to `system:masters` manually.
  The trust policy of this role should include the the
  `arn:aws::iam::${accountId}:root` principal in order to allow the execution
  role of the kubectl resource to assume it.

If the cluster is configured with private-only or private and restricted public
Kubernetes [endpoint access](#endpoint-access), you must also specify:

* `kubectlSecurityGroupId` - the ID of an EC2 security group that is allowed
  connections to the cluster's control security group. For example, the EKS managed [cluster security group](#cluster-security-group).
* `kubectlPrivateSubnetIds` - a list of private VPC subnets IDs that will be used
  to access the Kubernetes endpoint.

## Logging

EKS supports cluster logging for 5 different types of events:

* API requests to the cluster.
* Cluster access via the Kubernetes API.
* Authentication requests into the cluster.
* State of cluster controllers.
* Scheduling decisions.

You can enable logging for each one separately using the `clusterLogging`
property. For example:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

const cluster = new eks.Cluster(this, 'Cluster', {
  // ...
  version: eks.KubernetesVersion.V1_36,
  clusterLogging: [
    eks.ClusterLoggingTypes.API,
    eks.ClusterLoggingTypes.AUTHENTICATOR,
    eks.ClusterLoggingTypes.SCHEDULER,
  ],
  kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
});
```

## NodeGroup Repair Config

You can enable Managed Node Group [auto-repair config](https://docs.aws.amazon.com/eks/latest/userguide/node-health.html#node-auto-repair) using `enableNodeAutoRepair`
property. For example:

```ts
declare const cluster: eks.Cluster;

cluster.addNodegroupCapacity('NodeGroup', {
  enableNodeAutoRepair:true,
});
```

## Known Issues and Limitations

* [One cluster per stack](https://github.com/aws/aws-cdk/issues/10073)
* [Service Account dependencies](https://github.com/aws/aws-cdk/issues/9910)
* [Support isolated VPCs](https://github.com/aws/aws-cdk/issues/12171)
