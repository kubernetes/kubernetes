# Amazon DocumentDB Construct Library


## Starting a Clustered Database

To set up a clustered DocumentDB database, define a `DatabaseCluster`. You must
always launch a database in a VPC. Use the `vpcSubnets` attribute to control whether
your instances will be launched privately or publicly:

```ts
declare const vpc: ec2.Vpc;
const cluster = new docdb.DatabaseCluster(this, 'Database', {
  masterUser: {
    username: 'myuser', // NOTE: 'admin' is reserved by DocumentDB
    excludeCharacters: '\"@/:', // optional, defaults to the set "\"@/" and is also used for eventually created rotations
    secretName: '/myapp/mydocdb/masteruser', // optional, if you prefer to specify the secret name
  },
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.MEMORY5, ec2.InstanceSize.LARGE),
  vpcSubnets: {
    subnetType: ec2.SubnetType.PUBLIC,
  },
  vpc,
  copyTagsToSnapshot: true  // whether to save the cluster tags when creating the snapshot.
});
```

By default, the master password will be generated and stored in AWS Secrets Manager with auto-generated description.

Your cluster will be empty by default.

## Serverless Clusters

DocumentDB supports serverless clusters that automatically scale capacity based on your application's needs.
To create a serverless cluster, specify the `serverlessV2ScalingConfiguration` instead of `instanceType`:

```ts
declare const vpc: ec2.Vpc;
const cluster = new docdb.DatabaseCluster(this, 'Database', {
  masterUser: {
    username: 'myuser',
  },
  vpc,
  serverlessV2ScalingConfiguration: {
    minCapacity: 0.5,
    maxCapacity: 2,
  },
  engineVersion: '5.0.0', // Serverless requires engine version 5.0.0 or higher
});
```

**Note**: DocumentDB serverless requires engine version 5.0.0 or higher and is not compatible with all features. See the [AWS documentation](https://docs.aws.amazon.com/documentdb/latest/developerguide/docdb-serverless-limitations.html) for limitations.

## Connecting

To control who can access the cluster, use the `.connections` attribute. DocumentDB databases have a default port, so
you don't need to specify the port:

```ts
declare const cluster: docdb.DatabaseCluster;
cluster.connections.allowDefaultPortFromAnyIpv4('Open to the world');
```

The endpoints to access your database cluster will be available as the `.clusterEndpoint` and `.clusterReadEndpoint`
attributes:

```ts
declare const cluster: docdb.DatabaseCluster;
const writeAddress = cluster.clusterEndpoint.socketAddress;   // "HOSTNAME:PORT"
```

If you have existing security groups you would like to add to the cluster, use the `addSecurityGroups` method. Security
groups added in this way will not be managed by the `Connections` object of the cluster.

```ts
declare const vpc: ec2.Vpc;
declare const cluster: docdb.DatabaseCluster;

const securityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', {
  vpc,
});
cluster.addSecurityGroups(securityGroup);
```

## Deletion protection

Deletion protection can be enabled on an Amazon DocumentDB cluster to prevent accidental deletion of the cluster:

```ts
declare const vpc: ec2.Vpc;
const cluster = new docdb.DatabaseCluster(this, 'Database', {
  masterUser: {
    username: 'myuser',
  },
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.MEMORY5, ec2.InstanceSize.LARGE),
  vpcSubnets: {
    subnetType: ec2.SubnetType.PUBLIC,
  },
  vpc,
  deletionProtection: true, // Enable deletion protection.
});
```

## Rotating credentials

When the master password is generated and stored in AWS Secrets Manager, it can be rotated automatically:

```ts
declare const cluster: docdb.DatabaseCluster;
cluster.addRotationSingleUser(); // Will rotate automatically after 30 days
```

[example of setting up master password rotation for a cluster](test/integ.cluster-rotation.lit.ts)

The multi user rotation scheme is also available:

```ts
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';

declare const myImportedSecret: secretsmanager.Secret;
declare const cluster: docdb.DatabaseCluster;

cluster.addRotationMultiUser('MyUser', {
  secret: myImportedSecret, // This secret must have the `masterarn` key
});
```

It's also possible to create user credentials together with the cluster and add rotation:

```ts
declare const cluster: docdb.DatabaseCluster;
const myUserSecret = new docdb.DatabaseSecret(this, 'MyUserSecret', {
  username: 'myuser',
  masterSecret: cluster.secret,
});
const myUserSecretAttached = myUserSecret.attach(cluster); // Adds DB connections information in the secret

cluster.addRotationMultiUser('MyUser', { // Add rotation using the multi user scheme
  secret: myUserSecretAttached, // This secret must have the `masterarn` key
});
```

**Note**: This user must be created manually in the database using the master credentials.
The rotation will start as soon as this user exists.

See also [aws-cdk-lib/aws-secretsmanager](https://github.com/aws/aws-cdk/blob/main/packages/aws-cdk-lib/aws-secretsmanager/README.md) for credentials rotation of existing clusters.

## Audit and profiler Logs

Sending audit or profiler needs to be configured in two places:

1. Check / create the needed options in your ParameterGroup for [audit](https://docs.aws.amazon.com/documentdb/latest/developerguide/event-auditing.html#event-auditing-enabling-auditing) and
[profiler](https://docs.aws.amazon.com/documentdb/latest/developerguide/profiling.html#profiling.enable-profiling) logs.
2. Enable the corresponding option(s) when creating the `DatabaseCluster`:

```ts
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from'aws-cdk-lib/aws-logs';

declare const myLogsPublishingRole: iam.Role;
declare const vpc: ec2.Vpc;

const cluster = new docdb.DatabaseCluster(this, 'Database', {
  masterUser: {
    username: 'myuser',
  },
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.MEMORY5, ec2.InstanceSize.LARGE),
  vpcSubnets: {
    subnetType: ec2.SubnetType.PUBLIC,
  },
  vpc,
  exportProfilerLogsToCloudWatch: true, // Enable sending profiler logs
  exportAuditLogsToCloudWatch: true, // Enable sending audit logs
  cloudWatchLogsRetention: logs.RetentionDays.THREE_MONTHS, // Optional - default is to never expire logs
  cloudWatchLogsRetentionRole: myLogsPublishingRole, // Optional - a role will be created if not provided
});
```

## Enable Performance Insights

By enabling this feature it will be cascaded and enabled in all instances inside the cluster:

```ts
declare const vpc: ec2.Vpc;

const cluster = new docdb.DatabaseCluster(this, 'Database', {
  masterUser: {
    username: 'myuser',
  },
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.MEMORY5, ec2.InstanceSize.LARGE),
  vpcSubnets: {
    subnetType: ec2.SubnetType.PUBLIC,
  },
  vpc,
  enablePerformanceInsights: true, // Enable Performance Insights in all instances under this cluster
});
```

## Removal Policy

This resource supports the snapshot removal policy.
To specify it use the `removalPolicy` property:

```ts
declare const vpc: ec2.Vpc;

const cluster = new docdb.DatabaseCluster(this, 'Database', {
  masterUser: {
    username: 'myuser',
  },
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.MEMORY5, ec2.InstanceSize.LARGE),
  vpcSubnets: {
    subnetType: ec2.SubnetType.PUBLIC,
  },
  vpc,
  removalPolicy: RemovalPolicy.SNAPSHOT,
});
```

**Note**: A `RemovalPolicy.DESTROY` removal policy will be applied to the
cluster's instances and security group by default as they don't support the snapshot
removal policy.

> Visit [DeletionPolicy](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-attribute-deletionpolicy.html) for more details.

To specify a custom removal policy for the cluster's instances, use the
`instanceRemovalPolicy` property:

```ts
declare const vpc: ec2.Vpc;

const cluster = new docdb.DatabaseCluster(this, 'Database', {
  masterUser: {
    username: 'myuser',
  },
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.MEMORY5, ec2.InstanceSize.LARGE),
  vpcSubnets: {
    subnetType: ec2.SubnetType.PUBLIC,
  },
  vpc,
  removalPolicy: RemovalPolicy.SNAPSHOT,
  instanceRemovalPolicy: RemovalPolicy.RETAIN,
});
```

To specify a custom removal policy for the cluster's security group, use the
`securityGroupRemovalPolicy` property:

```ts
declare const vpc: ec2.Vpc;

const cluster = new docdb.DatabaseCluster(this, 'Database', {
  masterUser: {
    username: 'myuser',
  },
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.MEMORY5, ec2.InstanceSize.LARGE),
  vpcSubnets: {
    subnetType: ec2.SubnetType.PUBLIC,
  },
  vpc,
  removalPolicy: RemovalPolicy.SNAPSHOT,
  securityGroupRemovalPolicy: RemovalPolicy.RETAIN,
});
```

## CA certificate

Use the `caCertificate` property to specify the [CA certificate](https://docs.aws.amazon.com/documentdb/latest/developerguide/ca_cert_rotation.html) to use for all instances inside the cluster:

```ts
declare const vpc: ec2.Vpc;

const cluster = new docdb.DatabaseCluster(this, 'Database', {
  masterUser: {
    username: 'myuser',
  },
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.MEMORY5, ec2.InstanceSize.LARGE),
  vpcSubnets: {
    subnetType: ec2.SubnetType.PUBLIC,
  },
  vpc,
  caCertificate: docdb.CaCertificate.RDS_CA_RSA4096_G1, // CA certificate for all instances under this cluster
});
```

## Storage Type

You can specify [storage type](https://docs.aws.amazon.com/documentdb/latest/developerguide/db-cluster-storage-configs.html) for the cluster.

```ts
declare const vpc: ec2.Vpc;

const cluster = new docdb.DatabaseCluster(this, 'Database', {
  masterUser: {
    username: 'myuser',
  },
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.MEMORY5, ec2.InstanceSize.LARGE),
  vpc,
  storageType: docdb.StorageType.IOPT1, // Default is StorageType.STANDARD
});
```

**Note**: `StorageType.IOPT1` is supported starting with engine version 5.0.0.

**Note**: For serverless clusters, storage type is managed automatically and cannot be specified.

## Maintenance Windows

DocumentDB has two independent maintenance windows: one for cluster-wide events (engine upgrades, etc.) and one per instance (reboots, patches). Use `preferredMaintenanceWindow` to control the cluster window and `instanceMaintenanceWindow` to control the window applied to every auto-created instance.

**Note**: `instanceMaintenanceWindow` only applies to provisioned clusters. It has no effect on serverless clusters because they don't create instances.

```ts
declare const vpc: ec2.Vpc;

const cluster = new docdb.DatabaseCluster(this, 'Database', {
  masterUser: {
    username: 'myuser',
  },
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.MEMORY5, ec2.InstanceSize.LARGE),
  vpc,
  preferredMaintenanceWindow: 'tue:04:17-tue:04:47',  // cluster-wide events
  instanceMaintenanceWindow: 'sat:09:00-sat:09:30',   // applied to every instance
});
```

If you want both the cluster and its instances to share the same window, set both props to the same value. When `instanceMaintenanceWindow` is not provided, a random 30-minute window is picked for each instance, which can cause maintenance events outside the cluster window.
