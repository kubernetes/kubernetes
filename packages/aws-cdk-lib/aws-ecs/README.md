# Amazon ECS Construct Library


This package contains constructs for working with **Amazon Elastic Container
Service** (Amazon ECS).

Amazon Elastic Container Service (Amazon ECS) is a fully managed container orchestration service.

For further information on Amazon ECS,
see the [Amazon ECS documentation](https://docs.aws.amazon.com/ecs)

The following example creates an Amazon ECS cluster, adds capacity to it, and
runs a service on it:

```ts
declare const vpc: ec2.Vpc;

// Create an ECS cluster
const cluster = new ecs.Cluster(this, 'Cluster', { vpc });

// Add capacity to it
cluster.addCapacity('DefaultAutoScalingGroupCapacity', {
  instanceType: new ec2.InstanceType("t2.xlarge"),
});

const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');

taskDefinition.addContainer('DefaultContainer', {
  image: ecs.ContainerImage.fromRegistry("amazon/amazon-ecs-sample"),
  memoryLimitMiB: 512,
});

// Instantiate an Amazon ECS Service
const ecsService = new ecs.Ec2Service(this, 'Service', {
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
  circuitBreaker: {
    enable: true,
  },
});
```

For a set of constructs defining common ECS architectural patterns, see the `aws-cdk-lib/aws-ecs-patterns` package.

## Launch Types: AWS Fargate vs Amazon EC2 vs AWS ECS Anywhere

There are three sets of constructs in this library:

- Use the `Ec2TaskDefinition` and `Ec2Service` constructs to run tasks on Amazon EC2 instances running in your account.
- Use the `FargateTaskDefinition` and `FargateService` constructs to run tasks on
  instances that are managed for you by AWS.
- Use the `ExternalTaskDefinition` and `ExternalService` constructs to run AWS ECS Anywhere tasks on self-managed infrastructure.

Here are the main differences:

- **Amazon EC2**: instances are under your control. Complete control of task to host
  allocation. Required to specify at least a memory reservation or limit for
  every container. Can use Host, Bridge and AwsVpc networking modes. Can attach
  Classic Load Balancer. Can share volumes between container and host.
- **AWS Fargate**: tasks run on AWS-managed instances, AWS manages task to host
  allocation for you. Requires specification of memory and cpu sizes at the
  taskdefinition level. Only supports AwsVpc networking modes and
  Application/Network Load Balancers. Only the AWS log driver is supported.
  Many host features are not supported such as adding kernel capabilities
  and mounting host devices/volumes inside the container.
- **AWS ECS Anywhere**: tasks are run and managed by AWS ECS Anywhere on infrastructure
  owned by the customer. Bridge, Host and None networking modes are supported. Does not
  support autoscaling, load balancing, cloudmap or attachment of volumes.

For more information on Amazon EC2 vs AWS Fargate, networking and ECS Anywhere see the AWS Documentation:
[AWS Fargate](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html),
[Task Networking](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-networking.html),
[ECS Anywhere](https://aws.amazon.com/ecs/anywhere/)

## Clusters

A `Cluster` defines the infrastructure to run your
tasks on. You can run many tasks on a single cluster.

The following code creates a cluster that can run AWS Fargate tasks:

```ts
declare const vpc: ec2.Vpc;

const cluster = new ecs.Cluster(this, 'Cluster', {
  vpc,
});
```

By default, storage is encrypted with AWS-managed key. You can specify customer-managed key using:
```ts
declare const key: kms.Key;

const cluster = new ecs.Cluster(this, 'Cluster', {
  managedStorageConfiguration: {
    fargateEphemeralStorageKmsKey: key,
    kmsKey: key,
  },
});
```

The following code imports an existing cluster using the ARN which can be used to
import an Amazon ECS service either EC2 or Fargate.

```ts
const clusterArn = 'arn:aws:ecs:us-east-1:012345678910:cluster/clusterName';

const cluster = ecs.Cluster.fromClusterArn(this, 'Cluster', clusterArn);
```

To use tasks with Amazon EC2 launch-type, you have to add capacity to
the cluster in order for tasks to be scheduled on your instances.  Typically,
you add an AutoScalingGroup with instances running the latest
Amazon ECS-optimized AMI to the cluster. There is a method to build and add such an
AutoScalingGroup automatically, or you can supply a customized AutoScalingGroup
that you construct yourself. It's possible to add multiple AutoScalingGroups
with various instance types.

The following example creates an Amazon ECS cluster and adds capacity to it:

```ts
declare const vpc: ec2.Vpc;

const cluster = new ecs.Cluster(this, 'Cluster', {
  vpc,
});

// Either add default capacity
cluster.addCapacity('DefaultAutoScalingGroupCapacity', {
  instanceType: new ec2.InstanceType("t2.xlarge"),
});

// Or add customized capacity. Be sure to start the Amazon ECS-optimized AMI.
const autoScalingGroup = new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType: new ec2.InstanceType('t2.xlarge'),
  machineImage: ecs.EcsOptimizedImage.amazonLinux(),
  // Or use Amazon ECS-Optimized Amazon Linux 2 AMI
  // machineImage: EcsOptimizedImage.amazonLinux2(),
  desiredCapacity: 3,
  // ... other options here ...
});

const capacityProvider = new ecs.AsgCapacityProvider(this, 'AsgCapacityProvider', {
  autoScalingGroup,
});
cluster.addAsgCapacityProvider(capacityProvider);
```

If you omit the property `vpc`, the construct will create a new VPC with two AZs.

By default, all machine images will auto-update to the latest version
on each deployment, causing a replacement of the instances in your AutoScalingGroup
if the AMI has been updated since the last deployment.

If task draining is enabled, ECS will transparently reschedule tasks on to the new
instances before terminating your old instances. If you have disabled task draining,
the tasks will be terminated along with the instance. To prevent that, you
can pick a non-updating AMI by passing `cacheInContext: true`, but be sure
to periodically update to the latest AMI manually by using the [CDK CLI
context management commands](https://docs.aws.amazon.com/cdk/latest/guide/context.html):

```ts
declare const vpc: ec2.Vpc;
const autoScalingGroup = new autoscaling.AutoScalingGroup(this, 'ASG', {
  machineImage: ecs.EcsOptimizedImage.amazonLinux({ cachedInContext: true }),
  vpc,
  instanceType: new ec2.InstanceType('t2.micro'),
});
```

To customize the cache key, use the `additionalCacheKey` parameter.
This allows you to have multiple lookups with the same parameters
cache their values separately. This can be useful if you want to
scope the context variable to a construct (ie, using `additionalCacheKey: this.node.path`),
so that if the value in the cache needs to be updated, it does not need to be updated
for all constructs at the same time.

```ts
declare const vpc: ec2.Vpc;
const autoScalingGroup = new autoscaling.AutoScalingGroup(this, 'ASG', {
  machineImage: ecs.EcsOptimizedImage.amazonLinux({ cachedInContext: true, additionalCacheKey: this.node.path }),
  vpc,
  instanceType: new ec2.InstanceType('t2.micro'),
});
```

To use `LaunchTemplate` with `AsgCapacityProvider`, make sure to specify the `userData` in the `LaunchTemplate`:

```ts
declare const vpc: ec2.Vpc;
const launchTemplate = new ec2.LaunchTemplate(this, 'ASG-LaunchTemplate', {
  instanceType: new ec2.InstanceType('t3.medium'),
  machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
  userData: ec2.UserData.forLinux(),
});

const autoScalingGroup = new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  mixedInstancesPolicy: {
    instancesDistribution: {
      onDemandPercentageAboveBaseCapacity: 50,
    },
    launchTemplate: launchTemplate,
  },
});

const cluster = new ecs.Cluster(this, 'Cluster', { vpc });

const capacityProvider = new ecs.AsgCapacityProvider(this, 'AsgCapacityProvider', {
  autoScalingGroup,
  machineImageType: ecs.MachineImageType.AMAZON_LINUX_2,
});

cluster.addAsgCapacityProvider(capacityProvider);
```

The following code retrieve the Amazon Resource Names (ARNs) of tasks that are a part of a specified ECS cluster.
It's useful when you want to grant permissions to a task to access other AWS resources.

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;
const taskARNs = cluster.arnForTasks('*'); // arn:aws:ecs:<region>:<regionId>:task/<clusterName>/*

// Grant the task permission to access other AWS resources
taskDefinition.addToTaskRolePolicy(
  new iam.PolicyStatement({
    actions: ['ecs:UpdateTaskProtection'],
    resources: [taskARNs],
  })
)
```

To manage task protection settings in an ECS cluster, you can use the `grantTaskProtection` method.
This method grants the `ecs:UpdateTaskProtection` permission to a specified IAM entity.

```ts
// Assume 'cluster' is an instance of ecs.Cluster
declare const cluster: ecs.Cluster;
declare const taskRole: iam.Role;

// Grant ECS Task Protection permissions to the role
// Now 'taskRole' has the 'ecs:UpdateTaskProtection' permission on all tasks in the cluster
cluster.grantTaskProtection(taskRole);
```

### Bottlerocket

[Bottlerocket](https://aws.amazon.com/bottlerocket/) is a Linux-based open source operating system that is
purpose-built by AWS for running containers. You can launch Amazon ECS container instances with the Bottlerocket AMI.

The following example will create a capacity with self-managed Amazon EC2 capacity of 2 `c5.large` Linux instances running with `Bottlerocket` AMI.

The following example adds Bottlerocket capacity to the cluster:

```ts
declare const cluster: ecs.Cluster;

cluster.addCapacity('bottlerocket-asg', {
  minCapacity: 2,
  instanceType: new ec2.InstanceType('c5.large'),
  machineImage: new ecs.BottleRocketImage(),
});
```

You can also specify an NVIDIA-compatible AMI such as in this example:

```ts
declare const cluster: ecs.Cluster;

cluster.addCapacity('bottlerocket-asg', {
  instanceType: new ec2.InstanceType('p3.2xlarge'),
  machineImage: new ecs.BottleRocketImage({
      variant: ecs.BottlerocketEcsVariant.AWS_ECS_2_NVIDIA,
  }),
});
```

### ARM64 (Graviton) Instances

To launch instances with ARM64 hardware, you can use the Amazon ECS-optimized
Amazon Linux 2 (arm64) AMI. Based on Amazon Linux 2, this AMI is recommended
for use when launching your EC2 instances that are powered by Arm-based AWS
Graviton Processors.

```ts
declare const cluster: ecs.Cluster;

cluster.addCapacity('graviton-cluster', {
  minCapacity: 2,
  instanceType: new ec2.InstanceType('c6g.large'),
  machineImage: ecs.EcsOptimizedImage.amazonLinux2(ecs.AmiHardwareType.ARM),
});
```

Bottlerocket is also supported:

```ts
declare const cluster: ecs.Cluster;

cluster.addCapacity('graviton-cluster', {
  minCapacity: 2,
  instanceType: new ec2.InstanceType('c6g.large'),
  machineImageType: ecs.MachineImageType.BOTTLEROCKET,
});
```

### Neuron Instances

To launch Amazon EC2 Inf1, Trn1 or Inf2 instances, you can use the Amazon ECS optimized Amazon Linux 2 (Neuron) AMI. It comes pre-configured with AWS Inferentia and AWS Trainium drivers and the AWS Neuron runtime for Docker which makes running machine learning inference workloads easier on Amazon ECS.

```ts
declare const cluster: ecs.Cluster;

cluster.addCapacity('neuron-cluster', {
  minCapacity: 2,
  instanceType: new ec2.InstanceType('inf1.xlarge'),
  machineImage: ecs.EcsOptimizedImage.amazonLinux2(ecs.AmiHardwareType.NEURON),
});
```

### Spot Instances

To add spot instances into the cluster, you must specify the `spotPrice` in the `ecs.AddCapacityOptions` and optionally enable the `spotInstanceDraining` property.

```ts
declare const cluster: ecs.Cluster;

// Add an AutoScalingGroup with spot instances to the existing cluster
cluster.addCapacity('AsgSpot', {
  maxCapacity: 2,
  minCapacity: 2,
  desiredCapacity: 2,
  instanceType: new ec2.InstanceType('c5.xlarge'),
  spotPrice: '0.0735',
  // Enable the Automated Spot Draining support for Amazon ECS
  spotInstanceDraining: true,
});
```

### SNS Topic Encryption

When the `ecs.AddCapacityOptions` that you provide has a non-zero `taskDrainTime` (the default) then an SNS topic and Lambda are created to ensure that the
cluster's instances have been properly drained of tasks before terminating. The SNS Topic is sent the instance-terminating lifecycle event from the AutoScalingGroup,
and the Lambda acts on that event. If you wish to engage [server-side encryption](https://docs.aws.amazon.com/sns/latest/dg/sns-data-encryption.html) for this SNS Topic
then you may do so by providing a KMS key for the `topicEncryptionKey` property of `ecs.AddCapacityOptions`.

```ts
// Given
declare const cluster: ecs.Cluster;
declare const key: kms.Key;
// Then, use that key to encrypt the lifecycle-event SNS Topic.
cluster.addCapacity('ASGEncryptedSNS', {
  instanceType: new ec2.InstanceType("t2.xlarge"),
  desiredCapacity: 3,
  topicEncryptionKey: key,
});
```

### Container Insights

On a cluster, CloudWatch Container Insights can be enabled by setting the `containerInsightsV2` property. [Container Insights](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/cloudwatch-container-insights.html)
can be disabled, enabled, or enhanced.

```ts
const cluster = new ecs.Cluster(this, 'Cluster', {
  containerInsightsV2: ecs.ContainerInsights.ENHANCED
});
```

## Task definitions

A task definition describes what a single copy of a **task** should look like.
A task definition has one or more containers; typically, it has one
main container (the *default container* is the first one that's added
to the task definition, and it is marked *essential*) and optionally
some supporting containers which are used to support the main container,
doings things like upload logs or metrics to monitoring services.

To run a task or service with Amazon EC2 launch type, use the `Ec2TaskDefinition`. For AWS Fargate tasks/services, use the
`FargateTaskDefinition`. For AWS ECS Anywhere use the `ExternalTaskDefinition`. These classes
provide simplified APIs that only contain properties relevant for each specific launch type.

For a `FargateTaskDefinition`, specify the task size (`memoryLimitMiB` and `cpu`):

```ts
const fargateTaskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef', {
  memoryLimitMiB: 512,
  cpu: 256,
});
```

On Fargate Platform Version 1.4.0 or later, you may specify up to 200GiB of
[ephemeral storage](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-task-storage.html#fargate-task-storage-pv14):

```ts
const fargateTaskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef', {
  memoryLimitMiB: 512,
  cpu: 256,
  ephemeralStorageGiB: 100,
});
```

To specify the process namespace to use for the containers in the task, use the `pidMode` property:

```ts
const fargateTaskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef', {
  runtimePlatform: {
    operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
    cpuArchitecture: ecs.CpuArchitecture.ARM64,
  },
  memoryLimitMiB: 512,
  cpu: 256,
  pidMode: ecs.PidMode.TASK,
});
```

**Note:** `pidMode` is only supported for tasks that are hosted on AWS Fargate if the tasks are using platform version 1.4.0
or later (Linux). Only the `task` option is supported for Linux containers. `pidMode` isn't supported for Windows containers on Fargate.
If `pidMode` is specified for a Fargate task, then `runtimePlatform.operatingSystemFamily` must also be specified.

To add containers to a task definition, call `addContainer()`:

```ts
const fargateTaskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef', {
  memoryLimitMiB: 512,
  cpu: 256,
});
const container = fargateTaskDefinition.addContainer("WebContainer", {
  // Use an image from DockerHub
  image: ecs.ContainerImage.fromRegistry("amazon/amazon-ecs-sample"),
  // ... other options here ...
});
```

For an `Ec2TaskDefinition`:

```ts
const ec2TaskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef', {
  networkMode: ecs.NetworkMode.BRIDGE,
});

const container = ec2TaskDefinition.addContainer("WebContainer", {
  // Use an image from DockerHub
  image: ecs.ContainerImage.fromRegistry("amazon/amazon-ecs-sample"),
  memoryLimitMiB: 1024,
  // ... other options here ...
});
```

For an `ExternalTaskDefinition`:

```ts
const externalTaskDefinition = new ecs.ExternalTaskDefinition(this, 'TaskDef');

const container = externalTaskDefinition.addContainer("WebContainer", {
  // Use an image from DockerHub
  image: ecs.ContainerImage.fromRegistry("amazon/amazon-ecs-sample"),
  memoryLimitMiB: 1024,
  // ... other options here ...
});
```

You can specify container properties when you add them to the task definition, or with various methods, e.g.:

To add a port mapping when adding a container to the task definition, specify the `portMappings` option:

```ts
declare const taskDefinition: ecs.TaskDefinition;

taskDefinition.addContainer("WebContainer", {
  image: ecs.ContainerImage.fromRegistry("amazon/amazon-ecs-sample"),
  memoryLimitMiB: 1024,
  portMappings: [{ containerPort: 3000 }],
});
```

To add port mappings directly to a container definition, call `addPortMappings()`:

```ts
declare const container: ecs.ContainerDefinition;

container.addPortMappings({
  containerPort: 3000,
});
```

Sometimes it is useful to be able to configure port ranges for a container, e.g. to run applications such as game servers
and real-time streaming which typically require multiple ports to be opened simultaneously. This feature is supported on
both Linux and Windows operating systems for both the EC2 and AWS Fargate launch types. There is a maximum limit of 100
port ranges per container, and you cannot specify overlapping port ranges.

Docker recommends that you turn off the `docker-proxy` in the Docker daemon config file when you have a large number of ports.
For more information, see [Issue #11185](https://github.com/moby/moby/issues/11185) on the GitHub website.

```ts
declare const container: ecs.ContainerDefinition;

container.addPortMappings({
    containerPort: ecs.ContainerDefinition.CONTAINER_PORT_USE_RANGE,
    containerPortRange: '8080-8081',
});
```

To add data volumes to a task definition, call `addVolume()`:

```ts
const fargateTaskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef', {
  memoryLimitMiB: 512,
  cpu: 256,
});
const volume = {
  // Use an Elastic FileSystem
  name: "mydatavolume",
  efsVolumeConfiguration: {
    fileSystemId: "EFS",
    // ... other options here ...
  },
};

const container = fargateTaskDefinition.addVolume(volume);
```

> Note: ECS Anywhere doesn't support volume attachments in the task definition.

To use a TaskDefinition that can be used with either Amazon EC2 or
AWS Fargate launch types, use the `TaskDefinition` construct.

When creating a task definition you have to specify what kind of
tasks you intend to run: Amazon EC2, AWS Fargate, or both.
The following example uses both:

```ts
const taskDefinition = new ecs.TaskDefinition(this, 'TaskDef', {
  memoryMiB: '512',
  cpu: '256',
  networkMode: ecs.NetworkMode.AWS_VPC,
  compatibility: ecs.Compatibility.EC2_AND_FARGATE,
});
```

To grant a principal permission to run your `TaskDefinition`, you can use the `TaskDefinition.grantRun()` method:

```ts
declare const role: iam.IGrantable;
const taskDef = new ecs.TaskDefinition(this, 'TaskDef', {
  cpu: '512',
  memoryMiB: '512',
  compatibility: ecs.Compatibility.EC2_AND_FARGATE,
});

// Gives role required permissions to run taskDef
taskDef.grantRun(role);
```

To deploy containerized applications that require the allocation of standard input (stdin) or a terminal (tty), use the `interactive` property.

This parameter corresponds to `OpenStdin` in the [Create a container](https://docs.docker.com/engine/api/v1.35/#tag/Container/operation/ContainerCreate) section of the [Docker Remote API](https://docs.docker.com/engine/api/v1.35/)
and the `--interactive` option to [docker run](https://docs.docker.com/engine/reference/run/#security-configuration).

```ts
declare const taskDefinition: ecs.TaskDefinition;

taskDefinition.addContainer("Container", {
  image: ecs.ContainerImage.fromRegistry("amazon/amazon-ecs-sample"),
  interactive: true,
});
```

### Images

Images supply the software that runs inside the container. Images can be
obtained from either DockerHub or from ECR repositories, built directly from a local Dockerfile, or use an existing tarball.

- `ecs.ContainerImage.fromRegistry(imageName)`: use a public image.
- `ecs.ContainerImage.fromRegistry(imageName, { credentials: mySecret })`: use a private image that requires credentials.
- `ecs.ContainerImage.fromEcrRepository(repo, tagOrDigest)`: use the given ECR repository as the image
  to start. If no tag or digest is provided, "latest" is assumed.
- `ecs.ContainerImage.fromAsset('./image')`: build and upload an
  image directly from a `Dockerfile` in your source directory.
- `ecs.ContainerImage.fromDockerImageAsset(asset)`: uses an existing
  `aws-cdk-lib/aws-ecr-assets.DockerImageAsset` as a container image.
- `ecs.ContainerImage.fromTarball(file)`: use an existing tarball.
- `new ecs.TagParameterContainerImage(repository)`: use the given ECR repository as the image
  but a CloudFormation parameter as the tag.

### Environment variables

To pass environment variables to the container, you can use the `environment`, `environmentFiles`, and `secrets` props.

```ts
declare const secret: secretsmanager.Secret;
declare const dbSecret: secretsmanager.Secret;
declare const parameter: ssm.StringParameter;
declare const taskDefinition: ecs.TaskDefinition;
declare const s3Bucket: s3.Bucket;

const newContainer = taskDefinition.addContainer('container', {
  image: ecs.ContainerImage.fromRegistry("amazon/amazon-ecs-sample"),
  memoryLimitMiB: 1024,
  environment: { // clear text, not for sensitive data
    STAGE: 'prod',
  },
  environmentFiles: [ // list of environment files hosted either on local disk or S3
    ecs.EnvironmentFile.fromAsset('./demo-env-file.env'),
    ecs.EnvironmentFile.fromBucket(s3Bucket, 'assets/demo-env-file.env'),
  ],
  secrets: { // Retrieved from AWS Secrets Manager or AWS Systems Manager Parameter Store at container start-up.
    SECRET: ecs.Secret.fromSecretsManager(secret),
    DB_PASSWORD: ecs.Secret.fromSecretsManager(dbSecret, 'password'), // Reference a specific JSON field, (requires platform version 1.4.0 or later for Fargate tasks)
    API_KEY: ecs.Secret.fromSecretsManagerVersion(secret, { versionId: '12345' }, 'apiKey'), // Reference a specific version of the secret by its version id or version stage (requires platform version 1.4.0 or later for Fargate tasks)
    PARAMETER: ecs.Secret.fromSsmParameter(parameter),
  },
});
newContainer.addEnvironment('QUEUE_NAME', 'MyQueue');
newContainer.addSecret('API_KEY', ecs.Secret.fromSecretsManager(secret));
newContainer.addSecret('DB_PASSWORD', ecs.Secret.fromSecretsManager(secret, 'password'));
```

The task execution role is automatically granted read permissions on the secrets/parameters. Further details provided in the AWS documentation
about [specifying environment variables](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/taskdef-envfiles.html).

### Linux parameters

To apply additional linux-specific options related to init process and memory management to the container, use the `linuxParameters` property:

```ts
declare const taskDefinition: ecs.TaskDefinition;

taskDefinition.addContainer('container', {
  image: ecs.ContainerImage.fromRegistry("amazon/amazon-ecs-sample"),
  memoryLimitMiB: 1024,
  linuxParameters: new ecs.LinuxParameters(this, 'LinuxParameters', {
    initProcessEnabled: true,
    sharedMemorySize: 1024,
    maxSwap: Size.mebibytes(5000),
    swappiness: 90,
  }),
});
```

### System controls

To set system controls (kernel parameters) on the container, use the `systemControls` prop:

```ts
declare const taskDefinition: ecs.TaskDefinition;

taskDefinition.addContainer('container', {
  image: ecs.ContainerImage.fromRegistry("amazon/amazon-ecs-sample"),
  memoryLimitMiB: 1024,
  systemControls: [
    {
      namespace: 'net.ipv6.conf.all.default.disable_ipv6',
      value: '1',
    },
  ],
});
```

### Restart policy

To enable a restart policy for the container, set `enableRestartPolicy` to true and also specify
`restartIgnoredExitCodes` and `restartAttemptPeriod` if necessary.

```ts
declare const taskDefinition: ecs.TaskDefinition;

taskDefinition.addContainer('container', {
  image: ecs.ContainerImage.fromRegistry("amazon/amazon-ecs-sample"),
  enableRestartPolicy: true,
  restartIgnoredExitCodes: [0, 127],
  restartAttemptPeriod: Duration.seconds(360),
});
```

### Enable Fault Injection
You can utilize fault injection with Amazon ECS on both Amazon EC2 and Fargate to test how their application responds to certain impairment scenarios. These tests provide information you can use to optimize your application's performance and resiliency.

When fault injection is enabled, the Amazon ECS container agent allows tasks access to new fault injection endpoints.
Fault injection only works with tasks using the `AWS_VPC` or `HOST` network modes.

For more infomation, see [Use fault injection with your Amazon ECS and Fargate workloads](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fault-injection.html).

To enable Fault Injection for the task definiton, set `enableFaultInjection` to true.

```ts
new ecs.Ec2TaskDefinition(this, 'Ec2TaskDefinition', {
  enableFaultInjection: true,
});
```

## Docker labels

You can add labels to the container with the `dockerLabels` property or with the `addDockerLabel` method:

```ts
declare const taskDefinition: ecs.TaskDefinition;

const container = taskDefinition.addContainer('cont', {
  image: ecs.ContainerImage.fromRegistry("amazon/amazon-ecs-sample"),
  memoryLimitMiB: 1024,
  dockerLabels: {
    foo: 'bar',
  },
});

container.addDockerLabel('label', 'value');
```

### Using Windows containers on Fargate

AWS Fargate supports Amazon ECS Windows containers. For more details, please see this [blog post](https://aws.amazon.com/tw/blogs/containers/running-windows-containers-with-amazon-ecs-on-aws-fargate/)

```ts
// Create a Task Definition for the Windows container to start
const taskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef', {
  runtimePlatform: {
    operatingSystemFamily: ecs.OperatingSystemFamily.WINDOWS_SERVER_2019_CORE,
    cpuArchitecture: ecs.CpuArchitecture.X86_64,
  },
  cpu: 1024,
  memoryLimitMiB: 2048,
});

taskDefinition.addContainer('windowsservercore', {
  logging: ecs.LogDriver.awsLogs({ streamPrefix: 'win-iis-on-fargate' }),
  portMappings: [{ containerPort: 80 }],
  image: ecs.ContainerImage.fromRegistry('mcr.microsoft.com/windows/servercore/iis:windowsservercore-ltsc2019'),
});
```

### Using Windows authentication with gMSA

Amazon ECS supports Active Directory authentication for Linux containers through a special kind of service account called a group Managed Service Account (gMSA). For more details, please see the [product documentation on how to implement on Windows containers](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/windows-gmsa.html), or this [blog post on how to implement on  Linux containers](https://aws.amazon.com/blogs/containers/using-windows-authentication-with-gmsa-on-linux-containers-on-amazon-ecs/).

There are two types of CredentialSpecs, domained-join or domainless. Both types support creation from a S3 bucket, a SSM parameter, or by directly specifying a location for the file in the constructor.

A domian-joined gMSA container looks like:

```ts
// Make sure the task definition's execution role has permissions to read from the S3 bucket or SSM parameter where the CredSpec file is stored.
declare const parameter: ssm.IParameter;
declare const taskDefinition: ecs.TaskDefinition;

// Domain-joined gMSA container from a SSM parameter
taskDefinition.addContainer('gmsa-domain-joined-container', {
  image: ecs.ContainerImage.fromRegistry("amazon/amazon-ecs-sample"),
  cpu: 128,
  memoryLimitMiB: 256,
  credentialSpecs: [ecs.DomainJoinedCredentialSpec.fromSsmParameter(parameter)],
});
```

A domianless gMSA container looks like:

```ts
// Make sure the task definition's execution role has permissions to read from the S3 bucket or SSM parameter where the CredSpec file is stored.
declare const bucket: s3.Bucket;
declare const taskDefinition: ecs.TaskDefinition;

// Domainless gMSA container from a S3 bucket object.
taskDefinition.addContainer('gmsa-domainless-container', {
  image: ecs.ContainerImage.fromRegistry("amazon/amazon-ecs-sample"),
  cpu: 128,
  memoryLimitMiB: 256,
  credentialSpecs: [ecs.DomainlessCredentialSpec.fromS3Bucket(bucket, 'credSpec')],
});
```

### Using Graviton2 with Fargate

AWS Graviton2 supports AWS Fargate. For more details, please see this [blog post](https://aws.amazon.com/blogs/aws/announcing-aws-graviton2-support-for-aws-fargate-get-up-to-40-better-price-performance-for-your-serverless-containers/)

```ts
// Create a Task Definition for running container on Graviton Runtime.
const taskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef', {
  runtimePlatform: {
    operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
    cpuArchitecture: ecs.CpuArchitecture.ARM64,
  },
  cpu: 1024,
  memoryLimitMiB: 2048,
});

taskDefinition.addContainer('webarm64', {
  logging: ecs.LogDriver.awsLogs({ streamPrefix: 'graviton2-on-fargate' }),
  portMappings: [{ containerPort: 80 }],
  image: ecs.ContainerImage.fromRegistry('public.ecr.aws/nginx/nginx:latest-arm64v8'),
});
```

## Service

A `Service` instantiates a `TaskDefinition` on a `Cluster` a given number of
times, optionally associating them with a load balancer.
If a task fails,
Amazon ECS automatically restarts the task.

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  desiredCount: 5,
  minHealthyPercent: 100,
  circuitBreaker: {
    enable: true,
  },
});
```

ECS Anywhere service definition looks like:

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

const service = new ecs.ExternalService(this, 'Service', {
  cluster,
  taskDefinition,
  desiredCount: 5,
  minHealthyPercent: 100,
});
```

`Services` by default will create a security group if not provided.
If you'd like to specify which security groups to use you can override the `securityGroups` property.

By default, the service will use the revision of the passed task definition generated when the `TaskDefinition`
is deployed by CloudFormation. However, this may not be desired if the revision is externally managed,
for example through CodeDeploy.

To set a specific revision number or the special `latest` revision, use the `taskDefinitionRevision` parameter:

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

new ecs.ExternalService(this, 'Service', {
  cluster,
  taskDefinition,
  desiredCount: 5,
  minHealthyPercent: 100,
  taskDefinitionRevision: ecs.TaskDefinitionRevision.of(1),
});

new ecs.ExternalService(this, 'Service', {
  cluster,
  taskDefinition,
  desiredCount: 5,
  minHealthyPercent: 100,
  taskDefinitionRevision: ecs.TaskDefinitionRevision.LATEST,
});
```

### Deployment circuit breaker and rollback

Amazon ECS [deployment circuit breaker](https://aws.amazon.com/tw/blogs/containers/announcing-amazon-ecs-deployment-circuit-breaker/)
automatically rolls back unhealthy service deployments, eliminating the need for manual intervention.

Use `circuitBreaker` to enable the deployment circuit breaker which determines whether a service deployment
will fail if the service can't reach a steady state.
You can optionally enable `rollback` for automatic rollback.

See [Using the deployment circuit breaker](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/deployment-type-ecs.html) for more details.

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;
const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
  circuitBreaker: {
    enable: true,
    rollback: true
  },
});
```

> Note: ECS Anywhere doesn't support deployment circuit breakers and rollback.

### Deployment alarms

Amazon ECS [deployment alarms]
(https://aws.amazon.com/blogs/containers/automate-rollbacks-for-amazon-ecs-rolling-deployments-with-cloudwatch-alarms/)
allow monitoring and automatically reacting to changes during a rolling update
by using Amazon CloudWatch metric alarms.

Amazon ECS starts monitoring the configured deployment alarms as soon as one or
more tasks of the updated service are in a running state. The deployment process
continues until the primary deployment is healthy and has reached the desired
count and the active deployment has been scaled down to 0. Then, the deployment
remains in the IN_PROGRESS state for an additional "bake time." The length the
bake time is calculated based on the evaluation periods and period of the alarms.
After the bake time, if none of the alarms have been activated, then Amazon ECS
considers this to be a successful update and deletes the active deployment and
changes the status of the primary deployment to COMPLETED.

```ts
import * as cw from 'aws-cdk-lib/aws-cloudwatch';

declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;
declare const elbAlarm: cw.Alarm;

const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
  deploymentAlarms: {
    alarmNames: [elbAlarm.alarmName],
    behavior: ecs.AlarmBehavior.ROLLBACK_ON_ALARM,
  },
});

// Defining a deployment alarm after the service has been created
const cpuAlarmName =  'MyCpuMetricAlarm';
new cw.Alarm(this, 'CPUAlarm', {
  alarmName: cpuAlarmName,
  metric: service.metricCpuUtilization(),
  evaluationPeriods: 2,
  threshold: 80,
});
service.enableDeploymentAlarms([cpuAlarmName], {
  behavior: ecs.AlarmBehavior.FAIL_ON_ALARM,
});
```

> Note: Deployment alarms are only available when `deploymentController` is set
> to `DeploymentControllerType.ECS`, which is the default.

#### Troubleshooting circular dependencies

I saw this info message during synth time. What do I do?

```text
Deployment alarm ({"Ref":"MyAlarmABC1234"}) enabled on MyEcsService may cause a
circular dependency error when this stack deploys. The alarm name references the
alarm's logical id, or another resource. See the 'Deployment alarms' section in
the module README for more details.
```

If your app deploys successfully with this message, you can disregard it. But it
indicates that you could encounter a circular dependency error when you try to
deploy. If you want to alarm on metrics produced by the service, there will be a
circular dependency between the service and its deployment alarms. In this case,
there are two options to avoid the circular dependency.

1. Define the physical name for the alarm. Use a defined physical name that is
   unique within the deployment environment for the alarm name when creating the
   alarm, and re-use the defined name. This name could be a hardcoded string, a
   string generated based on the environment, or could reference another
   resource that does not depend on the service.
2. Define the physical name for the service. Then, don't use
   `metricCpuUtilization()` or similar methods. Create the metric object
   separately by referencing the service metrics using this name.

Option 1, defining a physical name for the alarm:
```ts
import * as cw from 'aws-cdk-lib/aws-cloudwatch';

declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
});

const cpuAlarmName =  'MyCpuMetricAlarm';
const myAlarm = new cw.Alarm(this, 'CPUAlarm', {
  alarmName: cpuAlarmName,
  metric: service.metricCpuUtilization(),
  evaluationPeriods: 2,
  threshold: 80,
});

// Using `myAlarm.alarmName` here will cause a circular dependency
service.enableDeploymentAlarms([cpuAlarmName], {
  behavior: ecs.AlarmBehavior.FAIL_ON_ALARM,
});
```

Option 2, defining a physical name for the service:

```ts
import * as cw from 'aws-cdk-lib/aws-cloudwatch';

declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;
const serviceName = 'MyFargateService';
const service = new ecs.FargateService(this, 'Service', {
  serviceName,
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
});

const cpuMetric = new cw.Metric({
  metricName: 'CPUUtilization',
  namespace: 'AWS/ECS',
  period: Duration.minutes(5),
  statistic: 'Average',
  dimensionsMap: {
    ClusterName: cluster.clusterName,
    // Using `service.serviceName` here will cause a circular dependency
    ServiceName: serviceName,
  },
});
const myAlarm = new cw.Alarm(this, 'CPUAlarm', {
  alarmName: 'cpuAlarmName',
  metric: cpuMetric,
  evaluationPeriods: 2,
  threshold: 80,
});

service.enableDeploymentAlarms([myAlarm.alarmName], {
  behavior: ecs.AlarmBehavior.FAIL_ON_ALARM,
});
```

This issue only applies if the metrics to alarm on are emitted by the service
itself. If the metrics are emitted by a different resource, that does not depend
on the service, there will be no restrictions on the alarm name.

### Include an application/network load balancer

`Services` are load balancing targets and can be added to a target group, which will be attached to an application/network load balancers:

```ts
declare const vpc: ec2.Vpc;
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;
const service = new ecs.FargateService(this, 'Service', { cluster, taskDefinition, minHealthyPercent: 100 });

const lb = new elbv2.ApplicationLoadBalancer(this, 'LB', { vpc, internetFacing: true });
const listener = lb.addListener('Listener', { port: 80 });
const targetGroup1 = listener.addTargets('ECS1', {
  port: 80,
  targets: [service],
});
const targetGroup2 = listener.addTargets('ECS2', {
  port: 80,
  targets: [service.loadBalancerTarget({
    containerName: 'MyContainer',
    containerPort: 8080
  })],
});
```

> Note: ECS Anywhere doesn't support application/network load balancers.

Note that in the example above, the default `service` only allows you to register the first essential container or the first mapped port on the container as a target and add it to a new target group. To have more control over which container and port to register as targets, you can use `service.loadBalancerTarget()` to return a load balancing target for a specific container and port.

Alternatively, you can also create all load balancer targets to be registered in this service, add them to target groups, and attach target groups to listeners accordingly.

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;
declare const vpc: ec2.Vpc;
const service = new ecs.FargateService(this, 'Service', { cluster, taskDefinition, minHealthyPercent: 100 });

const lb = new elbv2.ApplicationLoadBalancer(this, 'LB', { vpc, internetFacing: true });
const listener = lb.addListener('Listener', { port: 80 });
service.registerLoadBalancerTargets(
  {
    containerName: 'web',
    containerPort: 80,
    newTargetGroupId: 'ECS',
    listener: ecs.ListenerConfig.applicationListener(listener, {
      protocol: elbv2.ApplicationProtocol.HTTPS
    }),
  },
);
```

### Using a Load Balancer from a different Stack

If you want to put your Load Balancer and the Service it is load balancing to in
different stacks, you may not be able to use the convenience methods
`loadBalancer.addListener()` and `listener.addTargets()`.

The reason is that these methods will create resources in the same Stack as the
object they're called on, which may lead to cyclic references between stacks.
Instead, you will have to create an `ApplicationListener` in the service stack,
or an empty `TargetGroup` in the load balancer stack that you attach your
service to.

See the [ecs/cross-stack-load-balancer example](https://github.com/aws-samples/aws-cdk-examples/tree/master/typescript/ecs/cross-stack-load-balancer/)
for the alternatives.

### Include a classic load balancer

`Services` can also be directly attached to a classic load balancer as targets:

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;
declare const vpc: ec2.Vpc;
const service = new ecs.Ec2Service(this, 'Service', { cluster, taskDefinition, minHealthyPercent: 100 });

const lb = new elb.LoadBalancer(this, 'LB', { vpc });
lb.addListener({ externalPort: 80 });
lb.addTarget(service);
```

Similarly, if you want to have more control over load balancer targeting:

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;
declare const vpc: ec2.Vpc;
const service = new ecs.Ec2Service(this, 'Service', { cluster, taskDefinition, minHealthyPercent: 100 });

const lb = new elb.LoadBalancer(this, 'LB', { vpc });
lb.addListener({ externalPort: 80 });
lb.addTarget(service.loadBalancerTarget({
  containerName: 'MyContainer',
  containerPort: 80,
}));
```

There are two higher-level constructs available which include a load balancer for you that can be found in the aws-ecs-patterns module:

- `LoadBalancedFargateService`
- `LoadBalancedEc2Service`

### Import existing services

`Ec2Service` and `FargateService` provide methods to import existing EC2/Fargate services.
The ARN of the existing service has to be specified to import the service.

Since AWS has changed the [ARN format for ECS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-account-settings.html#ecs-resource-ids),
feature flag `@aws-cdk/aws-ecs:arnFormatIncludesClusterName` must be enabled to use the new ARN format.
The feature flag changes behavior for the entire CDK project. Therefore it is not possible to mix the old and the new format in one CDK project.

```tss
declare const cluster: ecs.Cluster;

// Import service from EC2 service attributes
const service = ecs.Ec2Service.fromEc2ServiceAttributes(this, 'EcsService', {
  serviceArn: 'arn:aws:ecs:us-west-2:123456789012:service/my-http-service',
  cluster,
});

// Import service from EC2 service ARN
const service = ecs.Ec2Service.fromEc2ServiceArn(this, 'EcsService', 'arn:aws:ecs:us-west-2:123456789012:service/my-http-service');

// Import service from Fargate service attributes
const service = ecs.FargateService.fromFargateServiceAttributes(this, 'EcsService', {
  serviceArn: 'arn:aws:ecs:us-west-2:123456789012:service/my-http-service',
  cluster,
});

// Import service from Fargate service ARN
const service = ecs.FargateService.fromFargateServiceArn(this, 'EcsService', 'arn:aws:ecs:us-west-2:123456789012:service/my-http-service');
```

### Availability Zone rebalancing

ECS services running in AWS can be launched in multiple VPC subnets that are
each in different Availability Zones (AZs) to achieve high availability. Fargate
services launched this way will automatically try to achieve an even spread of
service tasks across AZs, and EC2 services can be instructed to do the same with
placement strategies. This ensures that the service has equal availability in
each AZ.

```ts
declare const vpc: ec2.Vpc;
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  // Fargate will try to ensure an even spread of newly launched tasks across
  // all AZs corresponding to the public subnets of the VPC.
  vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
});
```

However, those approaches only affect how newly launched tasks are placed.
Service tasks can still become unevenly spread across AZs if there is an
infrastructure event, like an AZ outage or a lack of available compute capacity
in an AZ. During such events, newly launched tasks may be placed in AZs in such
a way that tasks are not evenly spread across all AZs. After the infrastructure
event is over, the service will remain imbalanced until new tasks are launched
for some other reason, such as a service deployment.

Availability Zone rebalancing is a feature whereby ECS actively tries to correct
service AZ imbalances whenever they exist, by moving service tasks from
overbalanced AZs to underbalanced AZs. When an imbalance is detected, ECS will
launch new tasks in underbalanced AZs, then stop existing tasks in overbalanced
AZs, to ensure an even spread.

You can enabled Availability Zone rebalancing when creating your service:

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  availabilityZoneRebalancing: ecs.AvailabilityZoneRebalancing.ENABLED,
});
```

Availability Zone rebalancing works in the following configurations:
- Services that use the Replica strategy.
- Services that specify Availability Zone spread as the first task placement
  strategy, or do not specify a placement strategy.

You can't use Availability Zone rebalancing with services that meet any of the
following criteria:
- Uses the Daemon strategy.
- Uses the EXTERNAL launch type (ECSAnywhere).
- Uses 100% for the maximumPercent value.
- Uses a Classic Load Balancer.
- Uses the `attribute:ecs.availability-zone` as a task placement constraint.

## Task Auto-Scaling

You can configure the task count of a service to match demand. Task auto-scaling is
configured by calling `autoScaleTaskCount()`:

```ts
declare const target: elbv2.ApplicationTargetGroup;
declare const service: ecs.BaseService;
const scaling = service.autoScaleTaskCount({ maxCapacity: 10 });
scaling.scaleOnCpuUtilization('CpuScaling', {
  targetUtilizationPercent: 50,
});

scaling.scaleOnRequestCount('RequestScaling', {
  requestsPerTarget: 10000,
  targetGroup: target,
});
```

Task auto-scaling is powered by *Application Auto-Scaling*.
See that section for details.

## Integration with CloudWatch Events

To start an Amazon ECS task on an Amazon EC2-backed Cluster, instantiate an
`aws-cdk-lib/aws-events-targets.EcsTask` instead of an `Ec2Service`:

```ts
declare const cluster: ecs.Cluster;
// Create a Task Definition for the container to start
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromAsset(path.resolve(__dirname, '..', 'eventhandler-image')),
  memoryLimitMiB: 256,
  logging: new ecs.AwsLogDriver({ streamPrefix: 'EventDemo', mode: ecs.AwsLogDriverMode.NON_BLOCKING }),
});

// An Rule that describes the event trigger (in this case a scheduled run)
const rule = new events.Rule(this, 'Rule', {
  schedule: events.Schedule.expression('rate(1 minute)'),
});

// Pass an environment variable to the container 'TheContainer' in the task
rule.addTarget(new targets.EcsTask({
  cluster,
  taskDefinition,
  taskCount: 1,
  containerOverrides: [{
    containerName: 'TheContainer',
    environment: [{
      name: 'I_WAS_TRIGGERED',
      value: 'From CloudWatch Events'
    }],
  }],
}));
```

## Log Drivers

Currently Supported Log Drivers:

- awslogs
- fluentd
- gelf
- journald
- json-file
- splunk
- syslog
- awsfirelens
- Generic
- none

### awslogs Log Driver

```ts
// Create a Task Definition for the container to start
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  memoryLimitMiB: 256,
  logging: ecs.LogDrivers.awsLogs({
    streamPrefix: 'EventDemo',
    mode: ecs.AwsLogDriverMode.NON_BLOCKING,
    maxBufferSize: Size.mebibytes(25),
  }),
});
```

### fluentd Log Driver

```ts
// Create a Task Definition for the container to start
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  memoryLimitMiB: 256,
  logging: ecs.LogDrivers.fluentd(),
});
```

### gelf Log Driver

```ts
// Create a Task Definition for the container to start
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  memoryLimitMiB: 256,
  logging: ecs.LogDrivers.gelf({ address: 'my-gelf-address' }),
});
```

### journald Log Driver

```ts
// Create a Task Definition for the container to start
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  memoryLimitMiB: 256,
  logging: ecs.LogDrivers.journald(),
});
```

### json-file Log Driver

```ts
// Create a Task Definition for the container to start
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  memoryLimitMiB: 256,
  logging: ecs.LogDrivers.jsonFile(),
});
```

### splunk Log Driver

```ts
declare const secret: ecs.Secret;

// Create a Task Definition for the container to start
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  memoryLimitMiB: 256,
  logging: ecs.LogDrivers.splunk({
    secretToken: secret,
    url: 'my-splunk-url',
  }),
});
```

### syslog Log Driver

```ts
// Create a Task Definition for the container to start
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  memoryLimitMiB: 256,
  logging: ecs.LogDrivers.syslog(),
});
```

### firelens Log Driver

```ts
// Create a Task Definition for the container to start
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  memoryLimitMiB: 256,
  logging: ecs.LogDrivers.firelens({
    options: {
        Name: 'firehose',
        region: 'us-west-2',
        delivery_stream: 'my-stream',
    },
  }),
});
```

To pass secrets to the log configuration, use the `secretOptions` property of the log configuration. The task execution role is automatically granted read permissions on the secrets/parameters.

```ts
declare const secret: secretsmanager.Secret;
declare const parameter: ssm.StringParameter;

const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  memoryLimitMiB: 256,
  logging: ecs.LogDrivers.firelens({
    options: {
      // ... log driver options here ...
    },
    secretOptions: { // Retrieved from AWS Secrets Manager or AWS Systems Manager Parameter Store
      apikey: ecs.Secret.fromSecretsManager(secret),
      host: ecs.Secret.fromSsmParameter(parameter),
    },
  }),
});
```

When forwarding logs to CloudWatch Logs using Fluent Bit, you can set the retention period for the newly created Log Group by specifying the `log_retention_days` parameter.
If a Fluent Bit container has not been added, CDK will automatically add it to the task definition, and the necessary IAM permissions will be added to the task role.
If you are adding the Fluent Bit container manually, ensure to add the `logs:PutRetentionPolicy` policy to the task role.

```ts
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  memoryLimitMiB: 256,
  logging: ecs.LogDrivers.firelens({
    options: {
      Name: 'cloudwatch',
      region: 'us-west-2',
      log_group_name: 'firelens-fluent-bit',
      log_stream_prefix: 'from-fluent-bit',
      auto_create_group: 'true',
      log_retention_days: '1',
    },
  }),
});
```

> Visit [Fluent Bit CloudWatch Configuration Parameters](https://docs.fluentbit.io/manual/pipeline/outputs/cloudwatch#configuration-parameters)
for more details.

### Generic Log Driver

A generic log driver object exists to provide a lower level abstraction of the log driver configuration.

```ts
// Create a Task Definition for the container to start
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  memoryLimitMiB: 256,
  logging: new ecs.GenericLogDriver({
    logDriver: 'fluentd',
    options: {
      tag: 'example-tag',
    },
  }),
});
```

### none Log Driver

The none log driver disables logging for the container (Docker `none` driver).

```ts
// Create a Task Definition for the container to start
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  memoryLimitMiB: 256,
  logging: ecs.LogDrivers.none(),
});
```

## CloudMap Service Discovery

To register your ECS service with a CloudMap Service Registry, you may add the
`cloudMapOptions` property to your service:

```ts
declare const taskDefinition: ecs.TaskDefinition;
declare const cluster: ecs.Cluster;

const service = new ecs.Ec2Service(this, 'Service', {
  cluster,
  taskDefinition,
  cloudMapOptions: {
    // Create A records - useful for AWSVPC network mode.
    dnsRecordType: cloudmap.DnsRecordType.A,
  },
});
```

With `bridge` or `host` network modes, only `SRV` DNS record types are supported.
By default, `SRV` DNS record types will target the default container and default
port. However, you may target a different container and port on the same ECS task:

```ts
declare const taskDefinition: ecs.TaskDefinition;
declare const cluster: ecs.Cluster;

// Add a container to the task definition
const specificContainer = taskDefinition.addContainer('Container', {
  image: ecs.ContainerImage.fromRegistry('/aws/aws-example-app'),
  memoryLimitMiB: 2048,
});

// Add a port mapping
specificContainer.addPortMappings({
  containerPort: 7600,
  protocol: ecs.Protocol.TCP,
});

new ecs.Ec2Service(this, 'Service', {
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
  cloudMapOptions: {
    // Create SRV records - useful for bridge networking
    dnsRecordType: cloudmap.DnsRecordType.SRV,
    // Targets port TCP port 7600 `specificContainer`
    container: specificContainer,
    containerPort: 7600,
  },
});
```

### Associate With a Specific CloudMap Service

You may associate an ECS service with a specific CloudMap service. To do
this, use the service's `associateCloudMapService` method:

```ts
declare const cloudMapService: cloudmap.Service;
declare const ecsService: ecs.FargateService;

ecsService.associateCloudMapService({
  service: cloudMapService,
});
```

### Using an Existing Cloud Map Namespace

You can use an existing Cloud Map namespace as the default namespace for a cluster
instead of creating a new one. This is useful when you want to share a namespace
across multiple clusters or when you want to use a namespace that was created
outside of CDK:

```ts
declare const vpc: ec2.Vpc;

// Create or reference an existing namespace
const existingNamespace = new cloudmap.PrivateDnsNamespace(this, 'Namespace', {
  name: 'example.local',
  vpc,
});

const cluster = new ecs.Cluster(this, 'Cluster', { vpc });

// Use the existing namespace as the default
cluster.addExistingDefaultCloudMapNamespace({
  namespace: existingNamespace,
  useForServiceConnect: true,
});
```

You can also import an existing namespace:

```ts
declare const vpc: ec2.Vpc;

const importedNamespace = cloudmap.PrivateDnsNamespace.fromPrivateDnsNamespaceAttributes(
  this, 'ImportedNamespace', {
    namespaceId: 'ns-xxxxxxxxxxxxx',
    namespaceArn: 'arn:aws:servicediscovery:us-east-1:123456789012:namespace/ns-xxxxxxxxxxxxx',
    namespaceName: 'example.local',
  }
);

const cluster = new ecs.Cluster(this, 'Cluster', { vpc });

cluster.addExistingDefaultCloudMapNamespace({
  namespace: importedNamespace,
});
```

## Capacity Providers

There are two major families of Capacity Providers: [AWS
Fargate](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-capacity-providers.html)
(including Fargate Spot) and EC2 [Auto Scaling
Group](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/asg-capacity-providers.html)
Capacity Providers. Both are supported.

### Fargate Capacity Providers

To enable Fargate capacity providers, you can either set
`enableFargateCapacityProviders` to `true` when creating your cluster, or by
invoking the `enableFargateCapacityProviders()` method after creating your
cluster. This will add both `FARGATE` and `FARGATE_SPOT` as available capacity
providers on your cluster.

```ts
declare const vpc: ec2.Vpc;

const cluster = new ecs.Cluster(this, 'FargateCPCluster', {
  vpc,
  enableFargateCapacityProviders: true,
});

const taskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef');

taskDefinition.addContainer('web', {
  image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
});

new ecs.FargateService(this, 'FargateService', {
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
  capacityProviderStrategies: [
    {
      capacityProvider: 'FARGATE_SPOT',
      weight: 2,
    },
    {
      capacityProvider: 'FARGATE',
      weight: 1,
    },
  ],
});
```

### Auto Scaling Group Capacity Providers

To add an Auto Scaling Group Capacity Provider, first create an EC2 Auto Scaling
Group. Then, create an `AsgCapacityProvider` and pass the Auto Scaling Group to
it in the constructor. Then add the Capacity Provider to the cluster. Finally,
you can refer to the Provider by its name in your service's or task's Capacity
Provider strategy.

> **Note**: Cross-stack capacity provider registration is not supported. The ECS cluster and its capacity providers must be created in the same stack to avoid circular dependency issues.

By default, Auto Scaling Group Capacity Providers will manage the scale-in and
scale-out behavior of the auto scaling group based on the load your tasks put on
the cluster, this is called [Managed Scaling](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/asg-capacity-providers.html#asg-capacity-providers-managed-scaling). If you'd
rather manage scaling behavior yourself set `enableManagedScaling` to `false`.

Additionally [Managed Termination Protection](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/cluster-auto-scaling.html#managed-termination-protection) is enabled by default to
prevent scale-in behavior from terminating instances that have non-daemon tasks
running on them. This is ideal for tasks that can be run to completion. If your
tasks are safe to interrupt then this protection can be disabled by setting
`enableManagedTerminationProtection` to `false`. Managed Scaling must be enabled for
Managed Termination Protection to work.

> Currently there is a known [CloudFormation issue](https://github.com/aws/containers-roadmap/issues/631)
> that prevents CloudFormation from automatically deleting Auto Scaling Groups that
> have Managed Termination Protection enabled. To work around this issue you could set
> `enableManagedTerminationProtection` to `false` on the Auto Scaling Group Capacity
> Provider. If you'd rather not disable Managed Termination Protection, you can [manually
> delete the Auto Scaling Group](https://docs.aws.amazon.com/autoscaling/ec2/userguide/as-process-shutdown.html).
> For other workarounds, see [this GitHub issue](https://github.com/aws/aws-cdk/issues/18179).

Managed instance draining facilitates graceful termination of Amazon ECS instances.
This allows your service workloads to stop safely and be rescheduled to non-terminating instances.
Infrastructure maintenance and updates are preformed without disruptions to workloads.
To use managed instance draining, set enableManagedDraining to true.

```ts
declare const vpc: ec2.Vpc;

const cluster = new ecs.Cluster(this, 'Cluster', {
  vpc,
});

const autoScalingGroup = new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType: new ec2.InstanceType('t2.micro'),
  machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
  minCapacity: 0,
  maxCapacity: 100,
});

const capacityProvider = new ecs.AsgCapacityProvider(this, 'AsgCapacityProvider', {
  autoScalingGroup,
  instanceWarmupPeriod: 300,
});
cluster.addAsgCapacityProvider(capacityProvider);

const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');

taskDefinition.addContainer('web', {
  image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
  memoryReservationMiB: 256,
});

new ecs.Ec2Service(this, 'EC2Service', {
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
  capacityProviderStrategies: [
    {
      capacityProvider: capacityProvider.capacityProviderName,
      weight: 1,
    },
  ],
});
```

### Managed Instances Capacity Providers

Managed Instances Capacity Providers allow you to use AWS-managed EC2 instances for your ECS tasks while providing more control over instance selection than standard Fargate. AWS handles the instance lifecycle, patching, and maintenance while you can specify detailed instance requirements. You can  define detailed instance requirements to control which types of instances are used for your workloads.

Capacity Option Type provides the purchasing option for the EC2 instances used in the capacity provider. Determines whether to use On-Demand or Spot instances. Valid values are `ON_DEMAND` and `SPOT`. Defaults to `ON_DEMAND` when not specified. Changing this value will trigger replacement of the capacity provider. For more information, see [Amazon EC2 billing and purchasing options](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-purchasing-options.html) in the Amazon EC2 User Guide.

See [ECS documentation for Managed Instances Capacity Provider](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/managed-instances-capacity-providers-concept.html) for more documentation.

#### IAM Roles Setup
Managed instances require an infrastructure and an EC2 instance profile. You can either provide your own infrastructure role and/or instance profile, or let the construct create them automatically.

Option 1: Let CDK create the role and instance profile automatically
```ts
declare const vpc: ec2.Vpc;

const cluster = new ecs.Cluster(this, 'Cluster', { vpc });

const securityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', {
  vpc,
  description: 'Security group for managed instances',
});

const miCapacityProvider = new ecs.ManagedInstancesCapacityProvider(this, 'MICapacityProvider', {
  capacityOptionType: ecs.CapacityOptionType.SPOT,
  subnets: vpc.privateSubnets,
  securityGroups: [securityGroup],
  instanceRequirements: {
    vCpuCountMin: 1,
    memoryMin: Size.gibibytes(2),
  },
});

// Optionally configure security group rules using IConnectable interface
miCapacityProvider.connections.allowFrom(ec2.Peer.ipv4(vpc.vpcCidrBlock), ec2.Port.tcp(80));

// Add the capacity provider to the cluster
cluster.addManagedInstancesCapacityProvider(miCapacityProvider);

const taskDefinition = new ecs.TaskDefinition(this, 'TaskDef', {
  memoryMiB: '512',
  cpu: '256',
  networkMode: ecs.NetworkMode.AWS_VPC,
  compatibility: ecs.Compatibility.MANAGED_INSTANCES,
});

taskDefinition.addContainer('web', {
  image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
  memoryReservationMiB: 256,
});

new ecs.FargateService(this, 'FargateService', {
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
  capacityProviderStrategies: [
    {
      capacityProvider: miCapacityProvider.capacityProviderName,
      weight: 1,
    },
  ],
});
```

Option 2: If you don't want to use the `AmazonECSInfrastructureRolePolicyForManagedInstances` managed policy for the ECS infrastructure role, you can create a custom infrastructure role with the required permissions. See [documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/infrastructure_IAM_role.html) for what permissions are needed for the ECS infrastructure role.

You can also choose not to use the automatically created ec2InstanceProfile. See [ECS documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/managed-instances-instance-profile.html) for what permissions are required for the profile's role.

```ts
declare const vpc: ec2.Vpc;

const cluster = new ecs.Cluster(this, 'Cluster', { vpc });

// Add your custom policies to the role.
const customInstanceRole = new iam.Role(this, 'CustomInstanceRole', {
  assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
});

const customInstanceProfile = new iam.InstanceProfile(this, 'CustomInstanceProfile', {
  role: customInstanceRole,
});

// Add your custom policies to the role.
const customInfrastructureRole = new iam.Role(this, 'CustomInfrastructureRole', {
  assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
});

// Add PassRole permission to allow ECS to pass the instance role to EC2.
customInfrastructureRole.addToPolicy(new iam.PolicyStatement({
  effect: iam.Effect.ALLOW,
  actions: ['iam:PassRole'],
  resources: [customInstanceRole.roleArn],
  conditions: {
    StringEquals: {
      'iam:PassedToService': 'ec2.amazonaws.com',
    },
  },
}));

const securityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', {
  vpc,
  description: 'Security group for managed instances',
});

const miCapacityProviderCustom = new ecs.ManagedInstancesCapacityProvider(this, 'MICapacityProviderCustomRoles', {
  infrastructureRole: customInfrastructureRole,
  ec2InstanceProfile: customInstanceProfile,
  subnets: vpc.privateSubnets,
  securityGroups: [securityGroup],
});

// Add the capacity provider to the cluster
cluster.addManagedInstancesCapacityProvider(miCapacityProviderCustom);

const taskDefinition = new ecs.TaskDefinition(this, 'TaskDef', {
  memoryMiB: '512',
  cpu: '256',
  networkMode: ecs.NetworkMode.AWS_VPC,
  compatibility: ecs.Compatibility.MANAGED_INSTANCES,
});

taskDefinition.addContainer('web', {
  image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
  memoryReservationMiB: 256,
});


new ecs.FargateService(this, 'FargateService', {

  cluster,
  taskDefinition,
  minHealthyPercent: 100,
  capacityProviderStrategies: [
    {
      capacityProvider: miCapacityProviderCustom.capacityProviderName,
      weight: 1,
    },
  ],
});
```

You can specify detailed instance requirements to control which types of instances are used:

```ts
declare const vpc: ec2.Vpc;

const securityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', {
  vpc,
  description: 'Security group for managed instances',
});

const miCapacityProvider = new ecs.ManagedInstancesCapacityProvider(this, 'MICapacityProvider', {
  subnets: vpc.privateSubnets,
  securityGroups: [securityGroup],
  instanceRequirements: {
    // Required: CPU and memory constraints
    vCpuCountMin: 2,
    vCpuCountMax: 8,
    memoryMin: Size.gibibytes(4),
    memoryMax: Size.gibibytes(32),

    // CPU preferences
    cpuManufacturers: [ec2.CpuManufacturer.INTEL, ec2.CpuManufacturer.AMD],
    instanceGenerations: [ec2.InstanceGeneration.CURRENT],

    // Instance type filtering
    allowedInstanceTypes: ['m5.*', 'c5.*'],

    // Performance characteristics
    burstablePerformance: ec2.BurstablePerformance.EXCLUDED,
    bareMetal: ec2.BareMetal.EXCLUDED,

    // Accelerator requirements (for ML/AI workloads)
    acceleratorTypes: [ec2.AcceleratorType.GPU],
    acceleratorManufacturers: [ec2.AcceleratorManufacturer.NVIDIA],
    acceleratorNames: [ec2.AcceleratorName.T4, ec2.AcceleratorName.V100],
    acceleratorCountMin: 1,

    // Storage requirements
    localStorage: ec2.LocalStorage.REQUIRED,
    localStorageTypes: [ec2.LocalStorageType.SSD],
    totalLocalStorageGBMin: 100,

    // Network requirements
    networkInterfaceCountMin: 2,
    networkBandwidthGbpsMin: 10,

    // Cost optimization
    onDemandMaxPricePercentageOverLowestPrice: 10,
  },
});

```
#### Note: Service Replacement When Migrating from LaunchType to CapacityProviderStrategy

**Understanding the Limitation**

The ECS [CreateService API](https://docs.aws.amazon.com/AmazonECS/latest/APIReference/API_CreateService.html#ECS-CreateService-request-launchType) does not allow specifying both `launchType` and `capacityProviderStrategies` simultaneously. When you specify `capacityProviderStrategies`, the CDK uses those capacity providers instead of a launch type. This is a limitation of the ECS API and CloudFormation, not a CDK bug.

**Impact on Updates**

Because `launchType` is immutable during updates, switching from `launchType` to `capacityProviderStrategies` requires CloudFormation to replace the service. This means your existing service will be deleted and recreated with the new configuration. This behavior is expected and reflects the underlying API constraints.

**Workaround**

While we work on a long-term solution, you can use the following [escape hatch](https://docs.aws.amazon.com/cdk/v2/guide/cfn-layer.html) to preserve your service during the migration:

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;
declare const miCapacityProvider: ecs.ManagedInstancesCapacityProvider;

const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  capacityProviderStrategies: [
    {
      capacityProvider: miCapacityProvider.capacityProviderName,
      weight: 1,
    },
  ],
});

// Escape hatch: Force launchType at the CloudFormation level to prevent service replacement
const cfnService = service.node.defaultChild as ecs.CfnService;
cfnService.launchType = 'FARGATE'; // or 'FARGATE_SPOT' depending on your capacity provider
```

### Cluster Default Provider Strategy

A capacity provider strategy determines whether ECS tasks are launched on EC2 instances or Fargate/Fargate Spot. It can be specified at the cluster, service, or task level, and consists of one or more capacity providers. You can specify an optional base and weight value for finer control of how tasks are launched. The `base` specifies a minimum number of tasks on one capacity provider, and the `weight`s of each capacity provider determine how tasks are distributed after `base` is satisfied.

You can associate a default capacity provider strategy with an Amazon ECS cluster. After you do this, a default capacity provider strategy is used when creating a service or running a standalone task in the cluster and whenever a custom capacity provider strategy or a launch type isn't specified. We recommend that you define a default capacity provider strategy for each cluster.

For more information visit https://docs.aws.amazon.com/AmazonECS/latest/developerguide/cluster-capacity-providers.html

When the service does not have a capacity provider strategy, the cluster's default capacity provider strategy will be used. Default Capacity Provider Strategy can be added by using the method `addDefaultCapacityProviderStrategy`. A capacity provider strategy cannot contain a mix of EC2 Autoscaling Group capacity providers and Fargate providers.

```ts
declare const capacityProvider: ecs.AsgCapacityProvider;

const cluster = new ecs.Cluster(this, 'EcsCluster', {
  enableFargateCapacityProviders: true,
});
cluster.addAsgCapacityProvider(capacityProvider);

cluster.addDefaultCapacityProviderStrategy([
  { capacityProvider: 'FARGATE', base: 10, weight: 50 },
  { capacityProvider: 'FARGATE_SPOT', weight: 50 },
]);
```

```ts
declare const capacityProvider: ecs.AsgCapacityProvider;

const cluster = new ecs.Cluster(this, 'EcsCluster', {
  enableFargateCapacityProviders: true,
});
cluster.addAsgCapacityProvider(capacityProvider);

cluster.addDefaultCapacityProviderStrategy([
  { capacityProvider: capacityProvider.capacityProviderName },
]);
```

## Elastic Inference Accelerators

Currently, this feature is only supported for services with EC2 launch types.

To add elastic inference accelerators to your EC2 instance, first add
`inferenceAccelerators` field to the Ec2TaskDefinition and set the `deviceName`
and `deviceType` properties.

```ts
const inferenceAccelerators = [{
  deviceName: 'device1',
  deviceType: 'eia2.medium',
}];

const taskDefinition = new ecs.Ec2TaskDefinition(this, 'Ec2TaskDef', {
  inferenceAccelerators,
});
```

To enable using the inference accelerators in the containers, add `inferenceAcceleratorResources`
field and set it to a list of device names used for the inference accelerators. Each value in the
list should match a `DeviceName` for an `InferenceAccelerator` specified in the task definition.

```ts
declare const taskDefinition: ecs.TaskDefinition;
const inferenceAcceleratorResources = ['device1'];

taskDefinition.addContainer('cont', {
  image: ecs.ContainerImage.fromRegistry('test'),
  memoryLimitMiB: 1024,
  inferenceAcceleratorResources,
});
```

## ECS Exec command

Please note, ECS Exec leverages AWS Systems Manager (SSM). So as a prerequisite for the exec command
to work, you need to have the SSM plugin for the AWS CLI installed locally. For more information, see
[Install Session Manager plugin for AWS CLI](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html).

To enable the ECS Exec feature for your containers, set the boolean flag `enableExecuteCommand` to `true` in
your `Ec2Service`, `FargateService` or `ExternalService`.

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

const service = new ecs.Ec2Service(this, 'Service', {
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
  enableExecuteCommand: true,
});
```

### Enabling logging

You can enable sending logs of your execute session commands to a CloudWatch log group or S3 bucket by configuring
the `executeCommandConfiguration` property for your cluster. The default configuration will send the
logs to the CloudWatch Logs using the `awslogs` log driver that is configured in your task definition. Please note,
when using your own `logConfiguration` the log group or S3 Bucket specified must already be created.

To encrypt data using your own KMS Customer Key (CMK), you must create a CMK and provide the key in the `kmsKey` field
of the `executeCommandConfiguration`. To use this key for encrypting CloudWatch log data or S3 bucket, make sure to associate the key
to these resources on creation.

```ts
declare const vpc: ec2.Vpc;
const kmsKey = new kms.Key(this, 'KmsKey');

// Pass the KMS key in the `encryptionKey` field to associate the key to the log group
const logGroup = new logs.LogGroup(this, 'LogGroup', {
  encryptionKey: kmsKey,
});

// Pass the KMS key in the `encryptionKey` field to associate the key to the S3 bucket
const execBucket = new s3.Bucket(this, 'EcsExecBucket', {
  encryptionKey: kmsKey,
});

const cluster = new ecs.Cluster(this, 'Cluster', {
  vpc,
  executeCommandConfiguration: {
    kmsKey,
    logConfiguration: {
      cloudWatchLogGroup: logGroup,
      cloudWatchEncryptionEnabled: true,
      s3Bucket: execBucket,
      s3EncryptionEnabled: true,
      s3KeyPrefix: 'exec-command-output',
    },
    logging: ecs.ExecuteCommandLogging.OVERRIDE,
  },
});
```

## Amazon ECS Service Connect

Service Connect is a managed AWS mesh network offering. It simplifies DNS queries and inter-service communication for
ECS Services by allowing customers to set up simple DNS aliases for their services, which are accessible to all
services that have enabled Service Connect.

To enable Service Connect, you must have created a CloudMap namespace. The CDK can infer your cluster's default CloudMap namespace,
or you can specify a custom namespace. You must also have created a named port mapping on at least one container in your Task Definition.

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;
declare const containerOptions: ecs.ContainerDefinitionOptions;

const container = taskDefinition.addContainer('MyContainer', containerOptions);

container.addPortMappings({
  name: 'api',
  containerPort: 8080,
});

cluster.addDefaultCloudMapNamespace({
  name: 'local',
});

const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
  serviceConnectConfiguration: {
    services: [
      {
        portMappingName: 'api',
        dnsName: 'http-api',
        port: 80,
      },
    ],
  },
});
```

Service Connect-enabled services may now reach this service at `http-api:80`. Traffic to this endpoint will
be routed to the container's port 8080.

To opt a service into using service connect without advertising a port, simply call the 'enableServiceConnect' method on an initialized service.

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
});
service.enableServiceConnect();
```

Service Connect also allows custom logging, Service Discovery name, and configuration of the port where service connect traffic is received.

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

const customService = new ecs.FargateService(this, 'CustomizedService', {
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
  serviceConnectConfiguration: {
    logDriver: ecs.LogDrivers.awsLogs({
      streamPrefix: 'sc-traffic',
    }),
    services: [
      {
        portMappingName: 'api',
        dnsName: 'customized-api',
        port: 80,
        ingressPortOverride: 20040,
        discoveryName: 'custom',
      },
    ],
  },
});
```

To set a timeout for service connect, use `idleTimeout` and `perRequestTimeout`.

**Note**: If `idleTimeout` is set to a time that is less than `perRequestTimeout`, the connection will close when
the `idleTimeout` is reached and not the `perRequestTimeout`.

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
  serviceConnectConfiguration: {
    services: [
      {
        portMappingName: 'api',
        idleTimeout: Duration.minutes(5),
        perRequestTimeout: Duration.minutes(5),
      },
    ],
  },
});
```

> Visit [Amazon ECS support for configurable timeout for services running with Service Connect](https://aws.amazon.com/about-aws/whats-new/2024/01/amazon-ecs-configurable-timeout-service-connect/) for more details.

### Service Connect Access Logs

[Service Connect access logs](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-connect-envoy-access-logs.html) provide detailed telemetry about individual requests processed by the Service Connect proxy, including HTTP methods, paths, response codes, and timing information.
These logs complement application logs by capturing per-request traffic metadata.

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  serviceConnectConfiguration: {
    services: [
      {
        portMappingName: 'api',
      },
    ],
    accessLogConfiguration: {
      format: ecs.ServiceConnectAccessLogFormat.JSON,
      includeQueryParameters: true,
    },
    // When configuring access log,
    // you also need to configure the log driver accordingly.
    logDriver: ecs.LogDrivers.awsLogs({
      streamPrefix: 'prefix',
    }),
  },
});
```

The `format` option determines the format of the access log output:

- `ServiceConnectAccessLogFormat.TEXT` - Human-readable text format
- `ServiceConnectAccessLogFormat.JSON` - Structured JSON format for log analysis tools

The `includeQueryParameters` option specifies whether to include query parameters in the access logs.
When enabled, query parameters from HTTP requests are included in the logs. Consider security and privacy implications, as query parameters may contain sensitive information such as request IDs and tokens.
By default, this parameter is `false`.

## ServiceManagedVolume

Amazon ECS now supports the attachment of Amazon Elastic Block Store (EBS) volumes to ECS tasks,
allowing you to utilize persistent, high-performance block storage with your ECS services.
This feature supports various use cases, such as using EBS volumes as extended ephemeral storage or
loading data from EBS snapshots.
You can also specify `encrypted: true` so that ECS will manage the KMS key. If you want to use your own KMS key, you may do so by providing both `encrypted: true` and `kmsKeyId`.

You can only attach a single volume for each task in the ECS Service.

To add an empty EBS Volume to an ECS Service, call service.addVolume().

```ts
declare const cluster: ecs.Cluster;
const taskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef');

const container = taskDefinition.addContainer('web', {
  image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
  portMappings: [{
    containerPort: 80,
    protocol: ecs.Protocol.TCP,
  }],
});

const volume = new ecs.ServiceManagedVolume(this, 'EBSVolume', {
  name: 'ebs1',
  managedEBSVolume: {
    size: Size.gibibytes(15),
    volumeType: ec2.EbsDeviceVolumeType.GP3,
    fileSystemType: ecs.FileSystemType.XFS,
    tagSpecifications: [{
      tags: {
        purpose: 'production',
      },
      propagateTags: ecs.EbsPropagatedTagSource.SERVICE,
    }],
  },
});

volume.mountIn(container, {
  containerPath: '/var/lib',
  readOnly: false,
});

taskDefinition.addVolume(volume);

const service = new ecs.FargateService(this, 'FargateService', {
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
});

service.addVolume(volume);
```

To create an EBS volume from an existing snapshot by specifying the `snapShotId` while adding a volume to the service.

```ts
declare const container: ecs.ContainerDefinition;
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

const volumeFromSnapshot = new ecs.ServiceManagedVolume(this, 'EBSVolume', {
  name: 'nginx-vol',
  managedEBSVolume: {
    snapShotId: 'snap-066877671789bd71b',
    volumeType: ec2.EbsDeviceVolumeType.GP3,
    fileSystemType: ecs.FileSystemType.XFS,
    // Specifies the Amazon EBS Provisioned Rate for Volume Initialization.
    // Valid range is between 100 and 300 MiB/s.
    volumeInitializationRate: Size.mebibytes(200),
  },
});

volumeFromSnapshot.mountIn(container, {
  containerPath: '/var/lib',
  readOnly: false,
});
taskDefinition.addVolume(volumeFromSnapshot);
const service = new ecs.FargateService(this, 'FargateService', {
  cluster,
  taskDefinition,
  minHealthyPercent: 100,
});

service.addVolume(volumeFromSnapshot);
```

## Enable pseudo-terminal (TTY) allocation

You can allocate a pseudo-terminal (TTY) for a container passing `pseudoTerminal` option while adding the container
to the task definition.
This maps to Tty option in the ["Create a container section"](https://docs.docker.com/engine/api/v1.38/#operation/ContainerCreate)
of the [Docker Remote API](https://docs.docker.com/engine/api/v1.38/) and the --tty option to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/).

```ts
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  pseudoTerminal: true
});
```

## Disable service container image version consistency

You can disable the
[container image "version consistency"](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/deployment-type-ecs.html#deployment-container-image-stability)
feature of ECS service deployments on a per-container basis.

```ts
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  versionConsistency: ecs.VersionConsistency.DISABLED,
});
```

## Specify a container ulimit

You can specify a container `ulimits` by specifying them in the `ulimits` option while adding the container
to the task definition.

```ts
const taskDefinition = new ecs.Ec2TaskDefinition(this, 'TaskDef');
taskDefinition.addContainer('TheContainer', {
  image: ecs.ContainerImage.fromRegistry('example-image'),
  ulimits: [{
    hardLimit: 128,
    name: ecs.UlimitName.RSS,
    softLimit: 128,
  }],
});
```

## Service Connect TLS

Service Connect TLS is a feature that allows you to secure the communication between services using TLS.

You can specify the `tls` option in the `services` array of the `serviceConnectConfiguration` property.

The `tls` property is an object with the following properties:

- `role`: The IAM role that's associated with the Service Connect TLS.
- `awsPcaAuthorityArn`: The ARN of the certificate root authority that secures your service.
- `kmsKey`: The KMS key used for encryption and decryption.

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;
declare const kmsKey: kms.IKey;
declare const role: iam.IRole;

const service = new ecs.FargateService(this, 'FargateService', {
  cluster,
  taskDefinition,
  serviceConnectConfiguration: {
    services: [
      {
        tls: {
          role,
          kmsKey,
          awsPcaAuthorityArn: 'arn:aws:acm-pca:us-east-1:123456789012:certificate-authority/123456789012',
        },
        portMappingName: 'api',
      },
    ],
    namespace: 'sample namespace',
  },
});
```

## ECS Native Blue/Green Deployment

Amazon ECS supports native blue/green deployments that allow you to deploy new versions of your services with zero downtime. This deployment strategy creates a new set of tasks (green) alongside the existing tasks (blue), then shifts traffic from the old version to the new version.

[Amazon ECS blue/green deployments](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/deployment-type-blue-green.html)

```ts
import * as lambda from 'aws-cdk-lib/aws-lambda';

declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;
declare const lambdaHook: lambda.Function;
declare const blueTargetGroup: elbv2.ApplicationTargetGroup;
declare const greenTargetGroup: elbv2.ApplicationTargetGroup;
declare const prodListenerRule: elbv2.ApplicationListenerRule;

const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  deploymentStrategy: ecs.DeploymentStrategy.BLUE_GREEN,
});

service.addLifecycleHook(new ecs.DeploymentLifecycleLambdaTarget(lambdaHook, 'PreScaleHook', {
  lifecycleStages: [ecs.DeploymentLifecycleStage.PRE_SCALE_UP],
}));

const target = service.loadBalancerTarget({
  containerName: 'nginx',
  containerPort: 80,
  protocol: ecs.Protocol.TCP,
  alternateTarget: new ecs.AlternateTarget('AlternateTarget', {
    alternateTargetGroup: greenTargetGroup,
    productionListener: ecs.ListenerRuleConfiguration.applicationListenerRule(prodListenerRule),
  }),
});

target.attachToApplicationTargetGroup(blueTargetGroup);
```

## Buil-in Linear and Canary Deployments

Amazon ECS supports progressive deployment strategies that allow you to validate new service revisions before shifting all production traffic. Both strategies require an Application Load Balancer (ALB) with target groups for traffic routing.

### Linear Deployment

Linear deployment strategy shifts production traffic in equal percentage increments with configurable wait times between each step:

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;
declare const blueTargetGroup: elbv2.ApplicationTargetGroup;
declare const greenTargetGroup: elbv2.ApplicationTargetGroup;
declare const prodListenerRule: elbv2.ApplicationListenerRule;

const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  deploymentStrategy: ecs.DeploymentStrategy.LINEAR,
  linearConfiguration: {
    stepPercent: 10.0,
    stepBakeTime: Duration.minutes(5),
  },
});

const target = service.loadBalancerTarget({
  containerName: 'web',
  containerPort: 80,
  alternateTarget: new ecs.AlternateTarget('AlternateTarget', {
    alternateTargetGroup: greenTargetGroup,
    productionListener: ecs.ListenerRuleConfiguration.applicationListenerRule(prodListenerRule),
  }),
});

target.attachToApplicationTargetGroup(blueTargetGroup);
```

Valid values:
- `stepPercent`: 3.0 to 100.0 (multiples of 0.1). Default: 10.0
- `stepBakeTime`: 0 to 1440 minutes (24 hours). Default: 6 minutes

### Canary Deployment

Canary deployment strategy shifts a fixed percentage of traffic to the new service revision for testing, then shifts the remaining traffic after a bake period:

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;
declare const blueTargetGroup: elbv2.ApplicationTargetGroup;
declare const greenTargetGroup: elbv2.ApplicationTargetGroup;
declare const prodListenerRule: elbv2.ApplicationListenerRule;

const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  deploymentStrategy: ecs.DeploymentStrategy.CANARY,
  canaryConfiguration: {
    stepPercent: 5.0,
    stepBakeTime: Duration.minutes(10),
  },
});

const target = service.loadBalancerTarget({
  containerName: 'web',
  containerPort: 80,
  alternateTarget: new ecs.AlternateTarget('AlternateTarget', {
    alternateTargetGroup: greenTargetGroup,
    productionListener: ecs.ListenerRuleConfiguration.applicationListenerRule(prodListenerRule),
  }),
});

target.attachToApplicationTargetGroup(blueTargetGroup);
```

Valid values:
- `stepPercent`: 0.1 to 100.0 (multiples of 0.1). Default: 5.0
- `stepBakeTime`: 0 to 1440 minutes (24 hours). Default: 10 minutes

## Daemon Scheduling Strategy
You can specify whether service use Daemon scheduling strategy by specifying `daemon` option in Service constructs. See [differences between Daemon and Replica scheduling strategy](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs_services.html)

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

new ecs.Ec2Service(this, 'Ec2Service', {
  cluster,
  taskDefinition,
  daemon: true,
});

new ecs.ExternalService(this, 'ExternalService', {
  cluster,
  taskDefinition,
  daemon: true,
});
```

### Force New Deployment

You can force a new deployment of a service without changing the task definition or desired count.
This is useful when you want ECS to pull the latest container image with the same tag or to trigger a redeployment.

When called without a nonce, a timestamp is generated automatically. This means every `cdk synth`
produces a different template and every `cdk deploy` triggers a new deployment, regardless of
whether any code has changed. To control when deployments happen, provide a stable nonce that only
changes when you intentionally want to redeploy (e.g., an image digest or a version string).

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
});

// Force a new deployment with an auto-generated nonce (deploys on every `cdk deploy`)
service.forceNewDeployment();

// Or provide your own nonce to control when deployments are triggered
service.forceNewDeployment('my-custom-nonce-v2');
```

Alternatively, you can configure `forceNewDeployment` declaratively as a constructor option.
This approach also allows you to explicitly disable the feature with `enabled: false`.

```ts
declare const cluster: ecs.Cluster;
declare const taskDefinition: ecs.TaskDefinition;

// Force a new deployment on every `cdk deploy` by using a time-based nonce
const service = new ecs.FargateService(this, 'Service', {
  cluster,
  taskDefinition,
  forceNewDeployment: {
    enabled: true,
    nonce: Date.now().toString(),
  },
});
```

Calling the `forceNewDeployment()` method takes precedence over the constructor option. The nonce passed
to the method (or the auto-generated one when none is provided) overrides any value configured through the
`forceNewDeployment` property.

## Mixins

ECS provides [mixins](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib-readme.html#mixins) that can be applied to L1 and L2 constructs.

### ClusterSettings

Applies one or more cluster settings to an ECS cluster. If a setting with the same name already exists, its value is replaced:

```ts
new ecs.CfnCluster(this, 'Cluster')
  .with(new ecs.mixins.ClusterSettings([{ name: 'containerInsights', value: 'enhanced' }]));
```
