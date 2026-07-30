# Amazon EKS V2 Construct Library

The aws-eks-v2 module is a rewrite of the existing aws-eks module (https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_eks-readme.html). This new iteration leverages native L1 CFN resources, replacing the previous custom resource approach for creating EKS clusters and Fargate Profiles.

Compared to the original EKS module, it has the following major changes:

- Use native L1 AWS::EKS::Cluster resource to replace custom resource Custom::AWSCDK-EKS-Cluster
- Use native L1 AWS::EKS::FargateProfile resource to replace custom resource Custom::AWSCDK-EKS-FargateProfile
- Kubectl Handler will not be created by default. It will only be created if users specify it.
- Remove AwsAuth construct. Permissions to the cluster will be managed by Access Entry.
- Remove the limit of 1 cluster per stack
- Remove nested stacks
- API changes to make them more ergonomic.

## Quick start

Here is the minimal example of defining an AWS EKS cluster

```ts
const cluster = new eks.Cluster(this, 'hello-eks', {
  version: eks.KubernetesVersion.V1_36,
});
```

## Architecture

```text                                             +-----------------+
                                         kubectl    |                 |
                                      +------------>| Kubectl Handler |
                                      |             |   (Optional)    |
                                      |             +-----------------+
+-------------------------------------+-------------------------------------+
|                        EKS Cluster (Auto Mode)                            |
|                          AWS::EKS::Cluster                                |
|                                                                           |
|  +---------------------------------------------------------------------+  |
|  |           Auto Mode Compute (Managed by EKS) (Default)              |  |
|  |                                                                     |  |
|  |  - Automatically provisions EC2 instances                           |  |
|  |  - Auto scaling based on pod requirements                           |  |
|  |  - No manual node group configuration needed                        |  |
|  |                                                                     |  |
|  +---------------------------------------------------------------------+  |
|                                                                           |
+---------------------------------------------------------------------------+
```

In a nutshell:

- **[Auto Mode](#eks-auto-mode)** (Default) – The fully managed capacity mode in EKS.  
  EKS automatically provisions and scales  EC2 capacity based on pod requirements.  
  It manages internal *system* and *general-purpose* NodePools, handles networking and storage setup, and removes the need for user-managed node groups or Auto Scaling Groups.

  ```ts
  const cluster = new eks.Cluster(this, 'AutoModeCluster', {
    version: eks.KubernetesVersion.V1_36,
    // Auto Mode is enabled by default
  });
  ```

- **[Managed Node Groups](#managed-node-groups)** – The semi-managed capacity mode.  
  EKS provisions and manages EC2 nodes on your behalf but you configure the instance types, scaling ranges, and update strategy.  
  AWS handles node health, draining, and rolling updates while you retain control over scaling and cost optimization.

  You can also define *Fargate Profiles* that determine which pods or namespaces run on Fargate infrastructure.

  ```ts
  const cluster = new eks.Cluster(this, 'ManagedNodeCluster', {
    version: eks.KubernetesVersion.V1_36,
    defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
  });
  
  // Add a Fargate Profile for specific workloads (e.g., default namespace)
  cluster.addFargateProfile('FargateProfile', {
    selectors: [
      { namespace: 'default' }, // Run pods in 'default' on Fargate
  ],
  });
  ```

- **[Fargate Mode](#fargate-profiles)** – The Fargate capacity mode.  
  EKS runs your pods directly on AWS Fargate without provisioning EC2 nodes.

  ```ts
  const cluster = new eks.FargateCluster(this, 'FargateCluster', {
    version: eks.KubernetesVersion.V1_36,
  });
  ```

- **[Self-Managed Nodes](#self-managed-capacity)** – The fully manual capacity mode.  
  You create and manage EC2 instances (via an Auto Scaling Group) and connect them to the cluster manually.  
  This provides maximum flexibility for custom AMIs or configurations but also the highest operational overhead.

  ```ts
  const cluster = new eks.Cluster(this, 'SelfManagedCluster', {
    version: eks.KubernetesVersion.V1_36,
  });
  
  // Add self-managed Auto Scaling Group
  cluster.addAutoScalingGroupCapacity('self-managed-asg', {
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM),
    minCapacity: 1,
    maxCapacity: 5,
  });
  ```

- **[Kubectl Handler](#kubectl-support) (Optional)** – A Lambda-backed custom resource created by the AWS CDK to execute `kubectl` commands (like `apply` or `patch`) during deployment.  
  Regardless of the capacity mode, this handler may still be created to apply Kubernetes manifests as part of CDK provisioning.

## Provisioning cluster

Creating a new cluster is done using the `Cluster` constructs. The only required property is the kubernetes version.

```ts
new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
});
```

You can also use `FargateCluster` to provision a cluster that uses only fargate workers.

```ts
new eks.FargateCluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
});
```

**Note: Unlike the previous EKS cluster, `Kubectl Handler` will not
be created by default. It will only be deployed when `kubectlProviderOptions`
property is used.**

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

new eks.Cluster(this, 'hello-eks', {
  version: eks.KubernetesVersion.V1_36,
  kubectlProviderOptions: {
    kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
  }
});
```

### EKS Auto Mode

[Amazon EKS Auto Mode](https://aws.amazon.com/eks/auto-mode/) extends AWS management of Kubernetes clusters beyond the cluster itself, allowing AWS to set up and manage the infrastructure that enables the smooth operation of your workloads.

#### Using Auto Mode

While `aws-eks` uses `DefaultCapacityType.NODEGROUP` by default, `aws-eks-v2` uses `DefaultCapacityType.AUTOMODE` as the default capacity type.

Auto Mode is enabled by default when creating a new cluster without specifying any capacity-related properties:

```ts
// Create EKS cluster with Auto Mode implicitly enabled
const cluster = new eks.Cluster(this, 'EksAutoCluster', {
  version: eks.KubernetesVersion.V1_36,
});
```

You can also explicitly enable Auto Mode using `defaultCapacityType`:

```ts
// Create EKS cluster with Auto Mode explicitly enabled
const cluster = new eks.Cluster(this, 'EksAutoCluster', {
  version: eks.KubernetesVersion.V1_36,
  defaultCapacityType: eks.DefaultCapacityType.AUTOMODE,
});
```

#### Node Pools

When Auto Mode is enabled, the cluster comes with two default node pools:

- `system`: For running system components and add-ons
- `general-purpose`: For running your application workloads

These node pools are managed automatically by EKS. You can configure which node pools to enable through the `compute` property:

```ts
const cluster = new eks.Cluster(this, 'EksAutoCluster', {
  version: eks.KubernetesVersion.V1_36,
  defaultCapacityType: eks.DefaultCapacityType.AUTOMODE,
  compute: {
    nodePools: ['system', 'general-purpose'],
  },
});
```

For more information, see [Create a Node Pool for EKS Auto Mode](https://docs.aws.amazon.com/eks/latest/userguide/create-node-pool.html).

#### Disabling Default Node Pools

You can disable the default node pools entirely by setting an empty array for `nodePools`. This is useful when you want to use Auto Mode features but manage your compute resources separately:

```ts
const cluster = new eks.Cluster(this, 'EksAutoCluster', {
  version: eks.KubernetesVersion.V1_36,
  defaultCapacityType: eks.DefaultCapacityType.AUTOMODE,
  compute: {
    nodePools: [], // Disable default node pools
  },
});
```

When node pools are disabled this way, no IAM role will be created for the node pools, preventing deployment failures that would otherwise occur when a role is created without any node pools.

### Node Groups as the default capacity type

If you prefer to manage your own node groups instead of using Auto Mode, you can use the traditional node group approach by specifying `defaultCapacityType` as `NODEGROUP`:

```ts
// Create EKS cluster with traditional managed node group
const cluster = new eks.Cluster(this, 'EksCluster', {
  version: eks.KubernetesVersion.V1_36,
  defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
  defaultCapacity: 3, // Number of instances
  defaultCapacityInstance: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.LARGE),
});
```

You can also create a cluster with no initial capacity and add node groups later:

```ts
const cluster = new eks.Cluster(this, 'EksCluster', {
  version: eks.KubernetesVersion.V1_36,
  defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
  defaultCapacity: 0,
});

// Add node groups as needed
cluster.addNodegroupCapacity('custom-node-group', {
  minSize: 1,
  maxSize: 3,
  instanceTypes: [ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.LARGE)],
});
```

Read [Managed node groups](#managed-node-groups) for more information on how to add node groups to the cluster.

### Mixed with Auto Mode and Node Groups

You can combine Auto Mode with traditional node groups for specific workload requirements:

```ts
const cluster = new eks.Cluster(this, 'Cluster', {
  version: eks.KubernetesVersion.V1_36,
  defaultCapacityType: eks.DefaultCapacityType.AUTOMODE,
  compute: {
    nodePools: ['system', 'general-purpose'],
  },
});

// Add specialized node group for specific workloads
cluster.addNodegroupCapacity('specialized-workload', {
  minSize: 1,
  maxSize: 3,
  instanceTypes: [ec2.InstanceType.of(ec2.InstanceClass.C5, ec2.InstanceSize.XLARGE)],
  labels: {
    workload: 'specialized',
  },
});
```

### Important Notes

1. Auto Mode and traditional capacity management are mutually exclusive at the default capacity level. You cannot opt in to Auto Mode and specify `defaultCapacity` or `defaultCapacityInstance`.

2. When Auto Mode is enabled:
   - The cluster will automatically manage compute resources
   - Node pools cannot be modified, only enabled or disabled
   - EKS will handle scaling and management of the node pools

3. Auto Mode requires specific IAM permissions. The construct will automatically attach the required managed policies.

### Managed node groups

Amazon EKS managed node groups automate the provisioning and lifecycle management of nodes (Amazon EC2 instances) for Amazon EKS Kubernetes clusters.
With Amazon EKS managed node groups, you don't need to separately provision or register the Amazon EC2 instances that provide compute capacity to run your Kubernetes applications. You can create, update, or terminate nodes for your cluster with a single operation. Nodes run using the latest Amazon EKS optimized AMIs in your AWS account while node updates and terminations gracefully drain nodes to ensure that your applications stay available.

> For more details visit [Amazon EKS Managed Node Groups](https://docs.aws.amazon.com/eks/latest/userguide/managed-node-groups.html).

By default, when using `DefaultCapacityType.NODEGROUP`, this library will allocate a managed node group with 2 *m5.large* instances (this instance type suits most common use-cases, and is good value for money).

```ts
new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
});
```

At cluster instantiation time, you can customize the number of instances and their type:

```ts
new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
  defaultCapacity: 5,
  defaultCapacityInstance: ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.SMALL),
});
```

To access the node group that was created on your behalf, you can use `cluster.defaultNodegroup`.

Additional customizations are available post instantiation. To apply them, set the default capacity to 0, and use the `cluster.addNodegroupCapacity` method:

```ts
const cluster = new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
  defaultCapacity: 0,
});

cluster.addNodegroupCapacity('custom-node-group', {
  instanceTypes: [new ec2.InstanceType('m5.large')],
  minSize: 4,
  diskSize: 100,
});
```

#### Default AMI type (under feature flag)

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
const cluster = new eks.FargateCluster(this, 'MyCluster', {
  version: eks.KubernetesVersion.V1_36,
});
```

`FargateCluster` will create a default `FargateProfile` which can be accessed via the cluster's `defaultProfile` property. The created profile can also be customized by passing options as with `addFargateProfile`.

**NOTE**: Classic Load Balancers and Network Load Balancers are not supported on
pods running on Fargate. For ingress, we recommend that you use the [ALB Ingress
Controller](https://docs.aws.amazon.com/eks/latest/userguide/alb-ingress.html)
on Amazon EKS (minimum version v1.1.4).

### Self-managed capacity

Self-managed capacity gives you the most control over your worker nodes by allowing you to create and manage your own EC2 Auto Scaling Groups. This approach provides maximum flexibility for custom AMIs, instance configurations, and scaling policies, but requires more operational overhead.

You can add self-managed capacity to any cluster using the `addAutoScalingGroupCapacity` method:

```ts
const cluster = new eks.Cluster(this, 'Cluster', {
  version: eks.KubernetesVersion.V1_36,
});

cluster.addAutoScalingGroupCapacity('self-managed-nodes', {
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM),
  minCapacity: 1,
  maxCapacity: 10,
  desiredCapacity: 3,
});
```

You can specify custom subnets for the Auto Scaling Group:

```ts
declare const vpc: ec2.Vpc;
declare const cluster: eks.Cluster;

cluster.addAutoScalingGroupCapacity('custom-subnet-nodes', {
  vpcSubnets: { subnets: vpc.privateSubnets },
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM),
  minCapacity: 2,
});
```

### Endpoint Access

When you create a new cluster, Amazon EKS creates an endpoint for the managed Kubernetes API server that you use to communicate with your cluster (using Kubernetes management tools such as `kubectl`)

You can configure the [cluster endpoint access](https://docs.aws.amazon.com/eks/latest/userguide/cluster-endpoint.html) by using the `endpointAccess` property:

```ts
const cluster = new eks.Cluster(this, 'hello-eks', {
  version: eks.KubernetesVersion.V1_36,
  endpointAccess: eks.EndpointAccess.PRIVATE, // No access outside of your VPC.
});
```

The default value is `eks.EndpointAccess.PUBLIC_AND_PRIVATE`. Which means the cluster endpoint is accessible from outside of your VPC, but worker node traffic and `kubectl` commands issued by this library stay within your VPC.

### Alb Controller

Some Kubernetes resources are commonly implemented on AWS with the help of the [ALB Controller](https://kubernetes-sigs.github.io/aws-load-balancer-controller/latest/).

From the docs:

> AWS Load Balancer Controller is a controller to help manage Elastic Load Balancers for a Kubernetes cluster.
>
> - It satisfies Kubernetes Ingress resources by provisioning Application Load Balancers.
> - It satisfies Kubernetes Service resources by provisioning Network Load Balancers.

To deploy the controller on your EKS cluster, configure the `albController` property:

```ts
new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  albController: {
    version: eks.AlbControllerVersion.V3_2_2,
  },
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
});
```

To overwrite an existing ALB controller service account, use the `overwriteServiceAccount` property:

```ts
new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  albController: {
    version: eks.AlbControllerVersion.V3_2_2,
    overwriteServiceAccount: true,
  },
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

You can specify the VPC of the cluster using the `vpc` and `vpcSubnets` properties:

```ts
declare const vpc: ec2.Vpc;

new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  vpc,
  vpcSubnets: [{ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }],
});
```

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

There is an additional components you might want to provision within the VPC.

The `KubectlHandler` is a Lambda function responsible to issuing `kubectl` and `helm` commands against the cluster when you add resource manifests to the cluster.

The handler association to the VPC is derived from the `endpointAccess` configuration. The rule of thumb is: *If the cluster VPC can be associated, it will be*.

Breaking this down, it means that if the endpoint exposes private access (via `EndpointAccess.PRIVATE` or `EndpointAccess.PUBLIC_AND_PRIVATE`), and the VPC contains **private** subnets, the Lambda function will be provisioned inside the VPC and use the private subnets to interact with the cluster. This is the common use-case.

If the endpoint does not expose private access (via `EndpointAccess.PUBLIC`) **or** the VPC does not contain private subnets, the function will not be provisioned within the VPC.

If your use-case requires control over the IAM role that the KubeCtl Handler assumes, a custom role can be passed through the ClusterProps (as `kubectlLambdaRole`) of the EKS Cluster construct.

### Kubectl Support

You can choose to have CDK create a `Kubectl Handler` - a Python Lambda Function to
apply k8s manifests using `kubectl apply`. This handler will not be created by default.

To create a `Kubectl Handler`, use `kubectlProviderOptions` when creating the cluster.
`kubectlLayer` is the only required property in `kubectlProviderOptions`.

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

new eks.Cluster(this, 'hello-eks', {
  version: eks.KubernetesVersion.V1_36,
  kubectlProviderOptions: {
    kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
  }
});
```

`Kubectl Handler` created along with the cluster will be granted admin permissions to the cluster.

If you want to use an existing kubectl provider function, for example with tight trusted entities on your IAM Roles - you can import the existing provider and then use the imported provider when importing the cluster:

```ts
const handlerRole = iam.Role.fromRoleArn(this, 'HandlerRole', 'arn:aws:iam::123456789012:role/lambda-role');
// get the serviceToken from the custom resource provider
const functionArn = lambda.Function.fromFunctionName(this, 'ProviderOnEventFunc', 'ProviderframeworkonEvent-XXX').functionArn;
const kubectlProvider = eks.KubectlProvider.fromKubectlProviderAttributes(this, 'KubectlProvider', {
  serviceToken: functionArn,
  role: handlerRole,
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
  kubectlProviderOptions: {
    kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
    environment: {
        'http_proxy': 'http://proxy.myproxy.com',
    },
  }
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
  kubectlProviderOptions: {
    kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
  },
});
```

#### Memory

By default, the kubectl provider is configured with 1024MiB of memory. You can use the `memory` option to specify the memory size for the AWS Lambda function:

```ts
import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

new eks.Cluster(this, 'MyCluster', {
  kubectlProviderOptions: {
    kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
    memory: Size.gibibytes(4),
  },
  version: eks.KubernetesVersion.V1_36,
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

When you create a cluster, you can specify a `mastersRole`. The `Cluster` construct will associate this role with `AmazonEKSClusterAdminPolicy` through [Access Entry](https://docs.aws.amazon.com/eks/latest/userguide/access-policy-permissions.html).

```ts
declare const role: iam.Role;
new eks.Cluster(this, 'HelloEKS', {
  version: eks.KubernetesVersion.V1_36,
  mastersRole: role,
});
```

If you do not specify it, you won't have access to the cluster from outside of the CDK application.

### Encryption

When you create an Amazon EKS cluster, envelope encryption of Kubernetes secrets using the AWS Key Management Service (AWS KMS) can be enabled.
The documentation on [creating a cluster](https://docs.aws.amazon.com/eks/latest/userguide/create-cluster.html)
can provide more details about the customer master key (CMK) that can be used for the encryption.

You can use the `secretsEncryptionKey` to configure which key the cluster will use to encrypt Kubernetes secrets. By default, an AWS Managed key will be used.

> This setting can only be specified when the cluster is created and cannot be updated.

```ts
const secretsKey = new kms.Key(this, 'SecretsKey');
const cluster = new eks.Cluster(this, 'MyCluster', {
  secretsEncryptionKey: secretsKey,
  version: eks.KubernetesVersion.V1_36,
});
```

You can also use a similar configuration for running a cluster built using the FargateCluster construct.

```ts
const secretsKey = new kms.Key(this, 'SecretsKey');
const cluster = new eks.FargateCluster(this, 'MyFargateCluster', {
  secretsEncryptionKey: secretsKey,
  version: eks.KubernetesVersion.V1_36,
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

In the new EKS module, `ConfigMap` is deprecated. Clusters created by the new module will use `API` as authentication mode. Access Entry will be the only way for granting permissions to specific IAM users and roles.

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

const cluster = new eks.Cluster(this, 'Cluster', {
  vpc,
  mastersRole: clusterAdminRole,
  version: eks.KubernetesVersion.V1_36,
  kubectlProviderOptions: {
    kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
    memory: Size.gibibytes(4),
  },
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
```

#### Access Entry Types

You can optionally specify an access entry type when granting access. This is particularly useful for EKS Auto Mode clusters with custom node roles, which require the `EC2` type:

```ts
declare const cluster: eks.Cluster;
declare const nodeRole: iam.Role;

// Grant access with EC2 type for Auto Mode node role
cluster.grantAccess('nodeAccess', nodeRole.roleArn, [
  eks.AccessPolicy.fromAccessPolicyName('AmazonEKSAutoNodePolicy', {
    accessScopeType: eks.AccessScopeType.CLUSTER,
  }),
], { accessEntryType: eks.AccessEntryType.EC2 });
```

The following access entry types are supported:

- `STANDARD` - Default type for standard IAM principals (default when not specified)
- `FARGATE_LINUX` - For Fargate profiles
- `EC2_LINUX` - For EC2 Linux worker nodes
- `EC2_WINDOWS` - For EC2 Windows worker nodes
- `EC2` - For EKS Auto Mode node roles
- `HYBRID_LINUX` - For EKS Hybrid Nodes
- `HYPERPOD_LINUX` - For Amazon SageMaker HyperPod

**Note**: Access entries with type `EC2`, `HYBRID_LINUX`, or `HYPERPOD_LINUX` cannot have access policies attached per AWS EKS API constraints. For these types, use the `AccessEntry` construct directly with an empty access policies array.

By default, the cluster creator role will be granted the cluster admin permissions. You can disable it by setting 
`bootstrapClusterCreatorAdminPermissions` to false. 

> **Note** - Switching `bootstrapClusterCreatorAdminPermissions` on an existing cluster would cause cluster replacement and should be avoided in production.


### Service Accounts

With services account you can provide Kubernetes Pods access to AWS resources.

```ts
import * as s3 from 'aws-cdk-lib/aws-s3';
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

You can also add service accounts to existing clusters.
To do so, pass the `openIdConnectProvider` property when you import the cluster into the application.

```ts
import * as s3 from 'aws-cdk-lib/aws-s3';
// you can import an existing provider
const provider = eks.OidcProviderNative.fromOidcProviderArn(this, 'Provider', 'arn:aws:iam::123456:oidc-provider/oidc.eks.eu-west-1.amazonaws.com/id/AB123456ABC');

// or create a new one using an existing issuer url
declare const issuerUrl: string;
const provider2 = new eks.OidcProviderNative(this, 'Provider', {
  url: issuerUrl,
});

import { KubectlV36Layer } from '@aws-cdk/lambda-layer-kubectl-v36';

const cluster = eks.Cluster.fromClusterAttributes(this, 'MyCluster', {
  clusterName: 'Cluster',
  openIdConnectProvider: provider,
  kubectlProviderOptions: {
    kubectlLayer: new KubectlV36Layer(this, 'kubectl'),
  }});
	
const serviceAccount = cluster.addServiceAccount('MyServiceAccount');

const bucket = new s3.Bucket(this, 'Bucket');
bucket.grantReadWrite(serviceAccount);
```

Note that adding service accounts requires running `kubectl` commands against the cluster which requires you to provide `kubectlProviderOptions` in the cluster props to create the `kubectl` provider. See [Kubectl Support](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-eks-v2-readme.html#kubectl-support)


#### Migrating from the deprecated eks.OpenIdConnectProvider to eks.OidcProviderNative

`eks.OpenIdConnectProvider` creates an IAM OIDC (OpenId Connect) provider using a custom resource while `eks.OidcProviderNative` uses the CFN L1 (AWS::IAM::OidcProvider) to create the provider. It is recommended for new and existing projects to use `eks.OidcProviderNative`. 

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


### Cluster Security Group

When you create an Amazon EKS cluster, a [cluster security group](https://docs.aws.amazon.com/eks/latest/userguide/sec-group-reqs.html)
is automatically created as well. This security group is designed to allow all traffic from the control plane and managed node groups to flow freely
between each other.

The ID for that security group can be retrieved after creating the cluster.

```ts
declare const cluster: eks.Cluster;
const clusterSecurityGroupId = cluster.clusterSecurityGroupId;
```

## Applying Kubernetes Resources

To apply kubernetes resource, kubectl provider needs to be created for the cluster. You can use `kubectlProviderOptions` to create the kubectl Provider.

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

- `ingressAlb` - Signal that the ingress detection should be done.
- `ingressAlbScheme` - Which ALB scheme should be applied. Defaults to `internal`.

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
new eks.Cluster(this, 'MyCluster', {
  version: eks.KubernetesVersion.V1_36,
  prune: false,
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

#### Custom CDK8s Constructs

You can also compose a few stock `cdk8s+` constructs into your own custom construct. However, since mixing scopes between `aws-cdk` and `cdk8s` is currently not supported, the `Construct` class
you'll need to use is the one from the [`constructs`](https://github.com/aws/constructs) module, and not from `aws-cdk-lib` like you normally would.
This is why we used `new cdk8s.App()` as the scope of the chart above.

```ts
import * as constructs from 'constructs';
import * as cdk8s from 'cdk8s';
import * as kplus from 'cdk8s-plus-25';

interface LoadBalancedWebService {
  readonly port: number;
  readonly image: string;
  readonly replicas: number;
}

const app = new cdk8s.App();
const chart = new cdk8s.Chart(app, 'my-chart');

class LoadBalancedWebService extends constructs.Construct {
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

The EKS library allows defining Kubernetes resources such as [Kubernetes
manifests](#kubernetes-resources) and [Helm charts](#helm-charts) on clusters
that are not defined as part of your CDK app.

First you will need to import the kubectl provider and cluster created in another stack

```ts
const handlerRole = iam.Role.fromRoleArn(this, 'HandlerRole', 'arn:aws:iam::123456789012:role/lambda-role');

const kubectlProvider = eks.KubectlProvider.fromKubectlProviderAttributes(this, 'KubectlProvider', {
  serviceToken: 'arn:aws:lambda:us-east-2:123456789012:function:my-function:1',
  role: handlerRole,
});

const cluster = eks.Cluster.fromClusterAttributes(this, 'Cluster', {
  clusterName: 'cluster',
  kubectlProvider,
});
```

Then, you can use `addManifest` or `addHelmChart` to define resources inside
your Kubernetes cluster.

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

## Logging

EKS supports cluster logging for 5 different types of events:

- API requests to the cluster.
- Cluster access via the Kubernetes API.
- Authentication requests into the cluster.
- State of cluster controllers.
- Scheduling decisions.

You can enable logging for each one separately using the `clusterLogging`
property. For example:

```ts
const cluster = new eks.Cluster(this, 'Cluster', {
  // ...
  version: eks.KubernetesVersion.V1_36,
  clusterLogging: [
    eks.ClusterLoggingTypes.API,
    eks.ClusterLoggingTypes.AUTHENTICATOR,
    eks.ClusterLoggingTypes.SCHEDULER,
  ],
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
