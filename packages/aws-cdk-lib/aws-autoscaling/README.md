# Amazon EC2 Auto Scaling Construct Library



This module is part of the [AWS Cloud Development Kit](https://github.com/aws/aws-cdk) project.

## Auto Scaling Group

An `AutoScalingGroup` represents a number of instances on which you run your code. You
pick the size of the fleet, the instance type and the OS image:

```ts
declare const vpc: ec2.Vpc;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.MICRO),

  // The latest Amazon Linux 2 image
  machineImage: ec2.MachineImage.latestAmazonLinux2(),
});
```

Creating an `AutoScalingGroup` from a Launch Configuration has been deprecated. All new accounts created after December 31, 2023 will no longer be able to create Launch Configurations. With the `@aws-cdk/aws-autoscaling:generateLaunchTemplateInsteadOfLaunchConfig` feature flag set to true, `AutoScalingGroup` properties used to create a Launch Configuration will now be used to create a `LaunchTemplate` using a [Launch Configuration to `LaunchTemplate` mapping](https://docs.aws.amazon.com/autoscaling/ec2/userguide/migrate-launch-configurations-with-cloudformation.html#launch-configuration-mapping-reference). Specifically, the following `AutoScalingGroup` properties will be used to generate a `LaunchTemplate`:
* machineImage
* keyName (deprecated, prefer keyPair)
* keyPair
* instanceType
* instanceMonitoring
* securityGroup
* role
* userData
* associatePublicIpAddress
* spotPrice
* blockDevices

After the Launch Configuration is replaced with a `LaunchTemplate`, any new instances launched by the `AutoScalingGroup` will use the new `LaunchTemplate`. Existing instances are not affected. To update an existing instance, you can allow the `AutoScalingGroup` to gradually replace existing instances with new instances based on the `terminationPolicies` for the `AutoScalingGroup`. Alternatively, you can terminate them yourself and force the `AutoScalingGroup` to launch new instances to maintain the `desiredCapacity`.

Support for creating an `AutoScalingGroup` from a `LaunchTemplate` was added in CDK version 2.21.0. Users on a CDK version earlier than version 2.21.0 that need to create an `AutoScalingGroup` with an account created after December 31, 2023 must update their CDK version to 2.21.0 or later. Users on CDK versions 2.21.0 up to, but not including 2.86.0, must use a manually created `LaunchTemplate` to create an `AutoScalingGroup` for accounts created after December 31, 2023. CDK version 2.86.0 or later will automatically generate a `LaunchTemplate` using the `AutoScalingGroup` properties mentioned above.

For additional migration information, please see: [Migrating to a `LaunchTemplate` from a Launch Configuration](https://docs.aws.amazon.com/autoscaling/ec2/userguide/migrate-to-launch-templates.html)

NOTE: AutoScalingGroup has a property called `allowAllOutbound` (allowing the instances to contact the
internet) which is set to `true` by default. Be sure to set this to `false`  if you don't want
your instances to be able to start arbitrary connections. Alternatively, you can specify an existing security
group to attach to the instances that are launched, rather than have the group create a new one.

```ts
declare const vpc: ec2.Vpc;

const mySecurityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', { vpc });
new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.MICRO),
  machineImage: ec2.MachineImage.latestAmazonLinux2(),
  securityGroup: mySecurityGroup,
});
```

Alternatively, to enable more advanced features, you can create an `AutoScalingGroup` from a supplied `LaunchTemplate`:

```ts
declare const vpc: ec2.Vpc;
declare const launchTemplate: ec2.LaunchTemplate;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  launchTemplate: launchTemplate
});
```

To launch a mixture of Spot and on-demand instances, and/or with multiple instance types, you can create an `AutoScalingGroup` from a `MixedInstancesPolicy`:

```ts
declare const vpc: ec2.Vpc;
declare const launchTemplate1: ec2.LaunchTemplate;
declare const launchTemplate2: ec2.LaunchTemplate;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  mixedInstancesPolicy: {
    instancesDistribution: {
      onDemandPercentageAboveBaseCapacity: 50, // Mix Spot and On-Demand instances
    },
    launchTemplate: launchTemplate1,
    launchTemplateOverrides: [ // Mix multiple instance types
      { instanceType: new ec2.InstanceType('t3.micro') },
      { instanceType: new ec2.InstanceType('t3a.micro') },
      { instanceType: new ec2.InstanceType('t4g.micro'), launchTemplate: launchTemplate2 },
    ],
  }
});
```

You can specify instances requirements with the `instanceRequirements ` property:

```ts
declare const vpc: ec2.Vpc;
declare const launchTemplate1: ec2.LaunchTemplate;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  mixedInstancesPolicy: {
    launchTemplate: launchTemplate1,
    launchTemplateOverrides: [
      {
        instanceRequirements: {
          vCpuCount: { min: 4, max: 8 },
          memoryMiB: { min: 16384 },
          cpuManufacturers: ['intel'],
        },
      }
    ],
  }
});
```

## Machine Images (AMIs)

AMIs control the OS that gets launched when you start your EC2 instance. The EC2
library contains constructs to select the AMI you want to use.

Depending on the type of AMI, you select it a different way.

The latest version of Amazon Linux and Microsoft Windows images are
selectable by instantiating one of these classes:

[example of creating images](test/example.images.lit.ts)

> NOTE: The Amazon Linux images selected will be cached in your `cdk.json`, so that your
> AutoScalingGroups don't automatically change out from under you when you're making unrelated
> changes. To update to the latest version of Amazon Linux, remove the cache entry from the `context`
> section of your `cdk.json`.
>
> We will add command-line options to make this step easier in the future.

## AutoScaling Instance Counts

AutoScalingGroups make it possible to raise and lower the number of instances in the group,
in response to (or in advance of) changes in workload.

When you create your AutoScalingGroup, you specify a `minCapacity` and a
`maxCapacity`. AutoScaling policies that respond to metrics will never go higher
or lower than the indicated capacity (but scheduled scaling actions might, see
below).

There are three ways to scale your capacity:

* **In response to a metric** (also known as step scaling); for example, you
  might want to scale out if the CPU usage across your cluster starts to rise,
  and scale in when it drops again.
* **By trying to keep a certain metric around a given value** (also known as
  target tracking scaling); you might want to automatically scale out and in to
  keep your CPU usage around 50%.
* **On a schedule**; you might want to organize your scaling around traffic
  flows you expect, by scaling out in the morning and scaling in in the
  evening.

The general pattern of autoscaling will look like this:

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

const autoScalingGroup = new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType,
  machineImage,

  minCapacity: 5,
  maxCapacity: 100
  // ...
});

// Then call one of the scaling methods (explained below)
//
// autoScalingGroup.scaleOnMetric(...);
//
// autoScalingGroup.scaleOnCpuUtilization(...);
// autoScalingGroup.scaleOnIncomingBytes(...);
// autoScalingGroup.scaleOnOutgoingBytes(...);
// autoScalingGroup.scaleOnRequestCount(...);
// autoScalingGroup.scaleToTrackMetric(...);
//
// autoScalingGroup.scaleOnSchedule(...);
```

### Step Scaling

This type of scaling scales in and out in deterministics steps that you
configure, in response to metric values. For example, your scaling strategy to
scale in response to a metric that represents your average worker pool usage
might look like this:

```plaintext
 Scaling        -1          (no change)          +1       +3
            │        │                       │        │        │
            ├────────┼───────────────────────┼────────┼────────┤
            │        │                       │        │        │
Worker use  0%      10%                     50%       70%     100%
```

(Note that this is not necessarily a recommended scaling strategy, but it's
a possible one. You will have to determine what thresholds are right for you).

Note that in order to set up this scaling strategy, you will have to emit a
metric representing your worker utilization from your instances. After that,
you would configure the scaling something like this:

```ts
declare const autoScalingGroup: autoscaling.AutoScalingGroup;

const workerUtilizationMetric = new cloudwatch.Metric({
    namespace: 'MyService',
    metricName: 'WorkerUtilization'
});

autoScalingGroup.scaleOnMetric('ScaleToCPU', {
  metric: workerUtilizationMetric,
  scalingSteps: [
    { upper: 10, change: -1 },
    { lower: 50, change: +1 },
    { lower: 70, change: +3 },
  ],
  evaluationPeriods: 10,
  datapointsToAlarm: 5,

  // Change this to AdjustmentType.PERCENT_CHANGE_IN_CAPACITY to interpret the
  // 'change' numbers before as percentages instead of capacity counts.
  adjustmentType: autoscaling.AdjustmentType.CHANGE_IN_CAPACITY,
});
```

The AutoScaling construct library will create the required CloudWatch alarms and
AutoScaling policies for you.

### Target Tracking Scaling

This type of scaling scales in and out in order to keep a metric around a value
you prefer. There are four types of predefined metrics you can track, or you can
choose to track a custom metric. If you do choose to track a custom metric,
be aware that the metric has to represent instance utilization in some way
(AutoScaling will scale out if the metric is higher than the target, and scale
in if the metric is lower than the target).

If you configure multiple target tracking policies, AutoScaling will use the
one that yields the highest capacity.

The following example scales to keep the CPU usage of your instances around
50% utilization:

```ts
declare const autoScalingGroup: autoscaling.AutoScalingGroup;

autoScalingGroup.scaleOnCpuUtilization('KeepSpareCPU', {
  targetUtilizationPercent: 50
});
```

To scale on average network traffic in and out of your instances:

```ts
declare const autoScalingGroup: autoscaling.AutoScalingGroup;

autoScalingGroup.scaleOnIncomingBytes('LimitIngressPerInstance', {
    targetBytesPerSecond: 10 * 1024 * 1024 // 10 MB/s
});
autoScalingGroup.scaleOnOutgoingBytes('LimitEgressPerInstance', {
    targetBytesPerSecond: 10 * 1024 * 1024 // 10 MB/s
});
```

To scale on the average request count per instance (only works for
AutoScalingGroups that have been attached to Application Load
Balancers):

```ts
declare const autoScalingGroup: autoscaling.AutoScalingGroup;

autoScalingGroup.scaleOnRequestCount('LimitRPS', {
    targetRequestsPerSecond: 1000
});
```

### Scheduled Scaling

This type of scaling is used to change capacities based on time. It works by
changing `minCapacity`, `maxCapacity` and `desiredCapacity` of the
AutoScalingGroup, and so can be used for two purposes:

* Scale in and out on a schedule by setting the `minCapacity` high or
  the `maxCapacity` low.
* Still allow the regular scaling actions to do their job, but restrict
  the range they can scale over (by setting both `minCapacity` and
  `maxCapacity` but changing their range over time).

A schedule is expressed as a cron expression. The `Schedule` class has a `cron` method to help build cron expressions.

The following example scales the fleet out in the morning, going back to natural
scaling (all the way down to 1 instance if necessary) at night:

```ts
declare const autoScalingGroup: autoscaling.AutoScalingGroup;

autoScalingGroup.scaleOnSchedule('PrescaleInTheMorning', {
  schedule: autoscaling.Schedule.cron({ hour: '8', minute: '0' }),
  minCapacity: 20,
});

autoScalingGroup.scaleOnSchedule('AllowDownscalingAtNight', {
  schedule: autoscaling.Schedule.cron({ hour: '20', minute: '0' }),
  minCapacity: 1
});
```

### Health checks for instances

You can configure the health checks for the instances in the Auto Scaling group.

Possible health check types are EC2, EBS, ELB, and VPC_LATTICE. EC2 is the default health check and cannot be disabled.

If you want to configure the EC2 health check, use the `HealthChecks.ec2` method:

```ts
declare const vpc: ec2.Vpc;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.MICRO),
  machineImage: ec2.MachineImage.latestAmazonLinux2(),
  healthChecks: autoscaling.HealthChecks.ec2({
    gracePeriod: Duration.seconds(100),
  }),
});
```

If you also want to configure the additional health checks other than EC2, use the `HealthChecks.withAdditionalChecks` method.
EC2 is implicitly included, so you can specify types other than EC2.

```ts
declare const vpc: ec2.Vpc;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.MICRO),
  machineImage: ec2.MachineImage.latestAmazonLinux2(),
  healthChecks: autoscaling.HealthChecks.withAdditionalChecks({
    gracePeriod: Duration.seconds(100),
    additionalTypes: [
      autoscaling.AdditionalHealthCheckType.EBS,
      autoscaling.AdditionalHealthCheckType.ELB,
      autoscaling.AdditionalHealthCheckType.VPC_LATTICE,
    ],
  }),
});
```

> Visit [Health checks for instances in an Auto Scaling group](https://docs.aws.amazon.com/autoscaling/ec2/userguide/ec2-auto-scaling-health-checks.html) for more details.

### Instance Maintenance Policy

You can configure an instance maintenance policy for your Auto Scaling group to
meet specific capacity requirements during events that cause instances to be replaced,
such as an instance refresh or the health check process.

For example, suppose you have an Auto Scaling group that has a small number of instances.
You want to avoid the potential disruptions from terminating and then replacing an instance
when health checks indicate an impaired instance. With an instance maintenance policy, you
can make sure that Amazon EC2 Auto Scaling first launches a new instance and then waits for
it to be fully ready before terminating the unhealthy instance.

An instance maintenance policy also helps you minimize any potential disruptions in cases where
multiple instances are replaced at the same time. You set the `minHealthyPercentage`
and the `maxHealthyPercentage` for the policy, and your Auto Scaling group can only
increase and decrease capacity within that minimum-maximum range when replacing instances.
A larger range increases the number of instances that can be replaced at the same time.

```ts
declare const vpc: ec2.Vpc;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.MICRO),
  machineImage: ec2.MachineImage.latestAmazonLinux2(),
  maxHealthyPercentage: 200,
  minHealthyPercentage: 100,
});
```

> Visit [Instance maintenance policies](https://docs.aws.amazon.com/autoscaling/ec2/userguide/ec2-auto-scaling-instance-maintenance-policy.html) for more details.

### Block Devices

This type specifies how block devices are exposed to the instance. You can specify virtual devices and EBS volumes.

#### GP3 Volumes

You can only specify the `throughput` on GP3 volumes.

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

const autoScalingGroup = new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType,
  machineImage,
  blockDevices: [
    {
        deviceName: 'gp3-volume',
        volume: autoscaling.BlockDeviceVolume.ebs(15, {
          volumeType: autoscaling.EbsDeviceVolumeType.GP3,
          throughput: 125,
        }),
      },
  ],
  // ...
});
```

## Configuring Instances using CloudFormation Init

It is possible to use the CloudFormation Init mechanism to configure the
instances in the AutoScalingGroup. You can write files to it, run commands,
start services, etc. See the documentation of
[AWS::CloudFormation::Init](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-init.html)
and the documentation of CDK's `aws-ec2` library for more information.

When you specify a CloudFormation Init configuration for an AutoScalingGroup:

* you *must* also specify `signals` to configure how long CloudFormation
  should wait for the instances to successfully configure themselves.
* you *should* also specify an `updatePolicy` to configure how instances
  should be updated when the AutoScalingGroup is updated (for example,
  when the AMI is updated). If you don't specify an update policy, a *rolling
  update* is chosen by default.

Here's an example of using CloudFormation Init to write a file to the
instance hosts on startup:

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType,
  machineImage,

  // ...

  init: ec2.CloudFormationInit.fromElements(
    ec2.InitFile.fromString('/etc/my_instance', 'This got written during instance startup'),
  ),
  signals: autoscaling.Signals.waitForAll({
    timeout: Duration.minutes(10),
  }),
});
```

## Signals

In normal operation, CloudFormation will send a Create or Update command to
an AutoScalingGroup and proceed with the rest of the deployment without waiting
for the *instances in the AutoScalingGroup*.

Configure `signals` to tell CloudFormation to wait for a specific number of
instances in the AutoScalingGroup to have been started (or failed to start)
before moving on. An instance is supposed to execute the
[`cfn-signal`](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-signal.html)
program as part of its startup to indicate whether it was started
successfully or not.

If you use CloudFormation Init support (described in the previous section),
the appropriate call to `cfn-signal` is automatically added to the
AutoScalingGroup's UserData. If you don't use the `signals` directly, you are
responsible for adding such a call yourself.

The following type of `Signals` are available:

* `Signals.waitForAll([options])`: wait for all of `desiredCapacity` amount of instances
  to have started (recommended).
* `Signals.waitForMinCapacity([options])`: wait for a `minCapacity` amount of instances
  to have started (use this if waiting for all instances takes too long and you are happy
  with a minimum count of healthy hosts).
* `Signals.waitForCount(count, [options])`: wait for a specific amount of instances to have
  started.

There are two `options` you can configure:

* `timeout`: maximum time a host startup is allowed to take. If a host does not report
  success within this time, it is considered a failure. Default is 5 minutes.
* `minSuccessPercentage`: percentage of hosts that needs to be healthy in order for the
  update to succeed. If you set this value lower than 100, some percentage of hosts may
  report failure, while still considering the deployment a success. Default is 100%.

## Update Policy

The *update policy* describes what should happen to running instances when the definition
of the AutoScalingGroup is changed. For example, if you add a command to the UserData
of an AutoScalingGroup, do the existing instances get replaced with new instances that
have executed the new UserData? Or do the "old" instances just keep on running?

It is recommended to always use an update policy, otherwise the current state of your
instances also depends the previous state of your instances, rather than just on your
source code. This degrades the reproducibility of your deployments.

The following update policies are available:

* `UpdatePolicy.none()`: leave existing instances alone (not recommended).
* `UpdatePolicy.rollingUpdate([options])`: progressively replace the existing
  instances with new instances, in small batches. At any point in time,
  roughly the same amount of total instances will be running. If the deployment
  needs to be rolled back, the fresh instances will be replaced with the "old"
  configuration again.
* `UpdatePolicy.replacingUpdate([options])`: build a completely fresh copy
  of the new AutoScalingGroup next to the old one. Once the AutoScalingGroup
  has been successfully created (and the instances started, if `signals` is
  configured on the AutoScalingGroup), the old AutoScalingGroup is deleted.
  If the deployment needs to be rolled back, the new AutoScalingGroup is
  deleted and the old one is left unchanged.
* `UpdatePolicy.instanceRefresh([options])`: gradually replace the existing
  instances with new instances by having Amazon EC2 Auto Scaling perform an
  instance refresh, keeping a configurable percentage of the group healthy and
  in service throughout.

To customize the instance refresh, pass options:

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;
declare const alarm: cloudwatch.Alarm;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType,
  machineImage,

  updatePolicy: autoscaling.UpdatePolicy.instanceRefresh({
    strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
    minHealthyPercentage: 90,
    maxHealthyPercentage: 100,
    instanceWarmup: Duration.seconds(300),
    checkpointPercentages: [50, 100],
    checkpointDelay: Duration.minutes(10),
    alarms: [alarm],
  }),
});
```

> **Note:** CloudFormation reads the `UpdatePolicy` from the *rollback target*
> template (the template being rolled back to), not from the template that
> triggered the update. Keep the `updatePolicy` on the Auto Scaling group
> permanently — if you remove it after the triggering update, a subsequent
> rollback will not perform an instance refresh.

If you omit `minHealthyPercentage`/`maxHealthyPercentage`, they fall back to the
Auto Scaling group's instance maintenance policy when one is defined; otherwise
CloudFormation defaults to `100`/`110` for the `Rolling` strategy
(launch-before-terminate with a 10% surge) and `90`/`100` for `ReplaceRootVolume`.

## Allowing Connections

See the documentation of the `aws-cdk-lib/aws-ec2` package for more information
about allowing connections between resources backed by instances.

## Max Instance Lifetime

To enable the max instance lifetime support, specify `maxInstanceLifetime` property
for the `AutoscalingGroup` resource. The value must be between 1 and 365 days(inclusive).
To clear a previously set value, leave this property undefined.

## Instance Monitoring

To disable detailed instance monitoring, specify `instanceMonitoring` property
for the `AutoscalingGroup` resource as `Monitoring.BASIC`. Otherwise detailed monitoring
will be enabled.

## Monitoring Group Metrics

Group metrics are used to monitor group level properties; they describe the group rather than any of its instances (e.g GroupMaxSize, the group maximum size). To enable group metrics monitoring, use the `groupMetrics` property.
All group metrics are reported in a granularity of 1 minute at no additional charge.

See [EC2 docs](https://docs.aws.amazon.com/autoscaling/ec2/userguide/as-instance-monitoring.html#as-group-metrics) for a list of all available group metrics.

To enable group metrics monitoring using the `groupMetrics` property:

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

// Enable monitoring of all group metrics
new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType,
  machineImage,

  // ...

  groupMetrics: [autoscaling.GroupMetrics.all()],
});

// Enable monitoring for a subset of group metrics
new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType,
  machineImage,

  // ...

  groupMetrics: [new autoscaling.GroupMetrics(autoscaling.GroupMetric.MIN_SIZE, autoscaling.GroupMetric.MAX_SIZE)],
});
```

## Termination policies

Auto Scaling uses [termination policies](https://docs.aws.amazon.com/autoscaling/ec2/userguide/as-instance-termination.html)
to determine which instances it terminates first during scale-in events. You
can specify one or more termination policies with the `terminationPolicies`
property:

[Custom termination policy](https://docs.aws.amazon.com/autoscaling/ec2/userguide/lambda-custom-termination-policy.html) with lambda
can be used to determine which instances to terminate based on custom logic.
The custom termination policy can be specified using `TerminationPolicy.CUSTOM_LAMBDA_FUNCTION`. If this is
specified, you must also supply a value of lambda arn in the `terminationPolicyCustomLambdaFunctionArn` property and
attach necessary [permission](https://docs.aws.amazon.com/autoscaling/ec2/userguide/lambda-custom-termination-policy.html#lambda-custom-termination-policy-create-function)
to invoke the lambda function.

If there are multiple termination policies specified,
custom termination policy with lambda `TerminationPolicy.CUSTOM_LAMBDA_FUNCTION`
must be specified first.

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;
declare const arn: string;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType,
  machineImage,

  // ...

  terminationPolicies: [
    autoscaling.TerminationPolicy.CUSTOM_LAMBDA_FUNCTION,
    autoscaling.TerminationPolicy.OLDEST_INSTANCE,
    autoscaling.TerminationPolicy.DEFAULT,
  ],

  //terminationPolicyCustomLambdaFunctionArn property must be specified if the TerminationPolicy.CUSTOM_LAMBDA_FUNCTION is used
  terminationPolicyCustomLambdaFunctionArn: arn,
});
```

## Protecting new instances from being terminated on scale-in

By default, Auto Scaling can terminate an instance at any time after launch when
scaling in an Auto Scaling Group, subject to the group's [termination
policy](https://docs.aws.amazon.com/autoscaling/ec2/userguide/as-instance-termination.html).

However, you may wish to protect newly-launched instances from being scaled in
if they are going to run critical applications that should not be prematurely
terminated. EC2 Capacity Providers for Amazon ECS requires this attribute be
set to `true`.

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType,
  machineImage,

  // ...

  newInstancesProtectedFromScaleIn: true,
});
```

## Configuring Capacity Rebalancing

Indicates whether Capacity Rebalancing is enabled. Otherwise, Capacity Rebalancing is disabled. When you turn on Capacity Rebalancing, Amazon EC2 Auto Scaling attempts to launch a Spot Instance whenever Amazon EC2 notifies that a Spot Instance is at an elevated risk of interruption. After launching a new instance, it then terminates an old instance. For more information, see [Use Capacity Rebalancing to handle Amazon EC2 Spot Interruptions](https://docs.aws.amazon.com/autoscaling/ec2/userguide/ec2-auto-scaling-capacity-rebalancing.html) in the in the Amazon EC2 Auto Scaling User Guide.

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType,
  machineImage,

  // ...

  capacityRebalance: true,
});
```

## Connecting to your instances using SSM Session Manager

SSM Session Manager makes it possible to connect to your instances from the
AWS Console, without preparing SSH keys.

To do so, you need to:

* Use an image with [SSM agent](https://docs.aws.amazon.com/systems-manager/latest/userguide/ssm-agent.html) installed
  and configured. [Many images come with SSM Agent
  preinstalled](https://docs.aws.amazon.com/systems-manager/latest/userguide/ami-preinstalled-agent.html), otherwise you
  may need to manually put instructions to [install SSM
  Agent](https://docs.aws.amazon.com/systems-manager/latest/userguide/sysman-manual-agent-install.html) into your
  instance's UserData or use EC2 Init).
* Create the AutoScalingGroup with `ssmSessionPermissions: true`.

If these conditions are met, you can connect to the instance from the EC2 Console. Example:

```ts
declare const vpc: ec2.Vpc;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO),

  // Amazon Linux 2 comes with SSM Agent by default
  machineImage: ec2.MachineImage.latestAmazonLinux2(),

  // Turn on SSM
  ssmSessionPermissions: true,
});
```

## Configuring Instance Metadata Service (IMDS)

### Toggling IMDSv1

You can configure [EC2 Instance Metadata Service](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html) options to either
allow both IMDSv1 and IMDSv2 or enforce IMDSv2 when interacting with the IMDS.

To do this for a single `AutoScalingGroup`, you can use set the `requireImdsv2` property.
The example below demonstrates IMDSv2 being required on a single `AutoScalingGroup`:

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType,
  machineImage,

  // ...

  requireImdsv2: true,
});
```

You can also use `AutoScalingGroupRequireImdsv2Aspect` to apply the operation to multiple AutoScalingGroups.
The example below demonstrates the `AutoScalingGroupRequireImdsv2Aspect` being used to require IMDSv2 for all AutoScalingGroups in a stack:

```ts
const aspect = new autoscaling.AutoScalingGroupRequireImdsv2Aspect();

Aspects.of(this).add(aspect);
```

## Warm Pool

Auto Scaling offers [a warm pool](https://docs.aws.amazon.com/autoscaling/ec2/userguide/ec2-auto-scaling-warm-pools.html) which gives an ability to decrease latency for applications that have exceptionally long boot times. You can create a warm pool with default parameters as below:

```ts
declare const autoScalingGroup: autoscaling.AutoScalingGroup;

autoScalingGroup.addWarmPool();
```

You can also customize a warm pool by configuring parameters:

```ts
declare const autoScalingGroup: autoscaling.AutoScalingGroup;

autoScalingGroup.addWarmPool({
  minSize: 1,
  reuseOnScaleIn: true,
});
```

### Default Instance Warming

You can use the default instance warmup feature to improve the Amazon CloudWatch metrics used for dynamic scaling.
When default instance warmup is not enabled, each instance starts contributing usage data to the aggregated metrics
as soon as the instance reaches the InService state. However, if you enable default instance warmup, this lets
your instances finish warming up before they contribute the usage data.

To optimize the performance of scaling policies that scale continuously, such as target tracking and step scaling
policies, we strongly recommend that you enable the default instance warmup, even if its value is set to 0 seconds.

To set up Default Instance Warming for an autoscaling group, simply pass it in as a prop

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;


new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType,
  machineImage,

  // ...

  defaultInstanceWarmup: Duration.seconds(5),
});
```

## Configuring KeyPair for instances

You can use a keyPair to build your asg when you decide not to use a ready-made LanchTemplate.

To configure KeyPair for an autoscaling group, pass the `keyPair` as a prop:

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

const myKeyPair = new ec2.KeyPair(this, 'MyKeyPair');

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType,
  machineImage,

  // ...

  keyPair: myKeyPair,
});
```

## Capacity Distribution Strategy

If launches fail in an Availability Zone, the following strategies are available.

* `BALANCED_BEST_EFFORT` (default) - If launches fail in an Availability Zone, Auto Scaling will attempt to launch in another healthy Availability Zone instead.
* `BALANCED_ONLY` - If launches fail in an Availability Zone, Auto Scaling will continue to attempt to launch in the unhealthy zone to preserve a balanced distribution.

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType,
  machineImage,

  // ...

  azCapacityDistributionStrategy: autoscaling.CapacityDistributionStrategy.BALANCED_ONLY,
});
```

## Deletion Protection

You can enable deletion protection to prevent your Auto Scaling group from being accidentally deleted. Deletion protection blocks the DeleteAutoScalingGroup API operation, requiring you to first update the deletion protection setting before you can delete the Auto Scaling group.

```ts
declare const vpc: ec2.Vpc;

new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO),
  machineImage: ec2.MachineImage.latestAmazonLinux2(),
  deletionProtection: autoscaling.DeletionProtection.PREVENT_ALL_DELETION,
});
```

The following deletion protection levels are available:

* `DeletionProtection.NONE` (default) - No deletion protection. The Auto Scaling group can be deleted with or without the force delete option.
* `DeletionProtection.PREVENT_FORCE_DELETION` - Prevents force deletion operations. This allows deletion of empty Auto Scaling groups but blocks force deletion that would terminate all instances.
* `DeletionProtection.PREVENT_ALL_DELETION` - Prevents all deletion operations. This provides the strongest protection and requires explicitly disabling deletion protection before the Auto Scaling group can be deleted.

**Note:** When using `PREVENT_ALL_DELETION`, you must first update the deletion protection setting before deleting the CloudFormation stack containing the Auto Scaling group.

## Instance Lifecycle Policy

You can configure an instance lifecycle policy to control how instances are handled during lifecycle events, particularly when lifecycle hooks are abandoned or fail. This allows fine-grained control over when to preserve instances for manual intervention.

The instance lifecycle policy defines retention triggers that specify when instances should be moved to a Retained state rather than terminated. Retained instances don't count toward desired capacity and remain until you manually terminate them.

**Important:** To use instance lifecycle policies in your Auto Scaling group, you must also configure a termination lifecycle hook. If you configure an instance lifecycle policy but don't have any termination lifecycle hooks, the policy has no effect. Instance lifecycle policies will only apply when termination lifecycle actions are abandoned, not when they complete successfully with the CONTINUE result.

```ts
declare const vpc: ec2.Vpc;
declare const instanceType: ec2.InstanceType;
declare const machineImage: ec2.IMachineImage;

const asg = new autoscaling.AutoScalingGroup(this, 'ASG', {
  vpc,
  instanceType,
  machineImage,

  // Configure instance lifecycle policy
  instanceLifecyclePolicy: {
    retentionTriggers: {
      terminateHookAbandon: autoscaling.TerminateHookAbandonAction.RETAIN,
    },
  },
});

// Add termination lifecycle hook (required for the policy to take effect)
asg.addLifecycleHook('TerminationHook', {
  lifecycleTransition: autoscaling.LifecycleTransition.INSTANCE_TERMINATING,
});
```

The `terminateHookAbandon` trigger specifies the action when a termination lifecycle hook is abandoned due to failure, timeout, or explicit abandonment. You can set it to:

* `RETAIN` - Move instances to a Retained state for manual investigation
* `TERMINATE` - Use default termination behavior (instances are terminated normally)

This feature is particularly useful for debugging failed instances or preserving instances that contain important data during lifecycle hook failures.

## Future work

* [ ] CloudWatch Events (impossible to add currently as the AutoScalingGroup ARN is
  necessary to make this rule and this cannot be accessed from CloudFormation).
