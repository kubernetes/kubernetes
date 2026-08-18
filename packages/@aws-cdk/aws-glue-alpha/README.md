# AWS Glue Construct Library
<!--BEGIN STABILITY BANNER-->

---

![cdk-constructs: Experimental](https://img.shields.io/badge/cdk--constructs-experimental-important.svg?style=for-the-badge)

> The APIs of higher level constructs in this module are experimental and under active development.
> They are subject to non-backward compatible changes or removal in any future version. These are
> not subject to the [Semantic Versioning](https://semver.org/) model and breaking changes will be
> announced in the release notes. This means that while you may use them, you may need to update
> your source code when upgrading to a newer version of this package.

---

<!--END STABILITY BANNER-->

This module is part of the [AWS Cloud Development Kit](https://github.com/aws/aws-cdk) project.

## README

[AWS Glue](https://aws.amazon.com/glue/) is a serverless data integration
service that makes it easier to discover, prepare, move, and integrate data
from multiple sources for analytics, machine learning (ML), and application
development.

The Glue L2 construct has convenience methods working backwards from common
use cases and sets required parameters to defaults that align with recommended
best practices for each job type. It also provides customers with a balance
between flexibility via optional parameter overrides, and opinionated
interfaces that discouraging anti-patterns, resulting in reduced time to develop
and deploy new resources.

### References

* [Glue Launch Announcement](https://aws.amazon.com/blogs/aws/launch-aws-glue-now-generally-available/)
* [Glue Documentation](https://docs.aws.amazon.com/glue/index.html)
* [Glue L1 (CloudFormation) Constructs](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/AWS_Glue.html)
* Prior version of the [@aws-cdk/aws-glue-alpha module](https://github.com/aws/aws-cdk/blob/v2.51.1/packages/%40aws-cdk/aws-glue/README.md)

## Create a Glue Job

A Job encapsulates a script that connects to data sources, processes
them, and then writes output to a data target. There are four types of Glue
Jobs: Spark (ETL and Streaming), Python Shell, and Flex Jobs. Most
of the required parameters for these jobs are common across all types,
but there are a few differences depending on the languages supported
and features provided by each type. For all job types, the L2 defaults
to AWS best practice recommendations, such as:

* Use of Secrets Manager for Connection JDBC strings
* Glue job autoscaling
* Default parameter values for Glue job creation

This iteration of the L2 construct introduces breaking changes to
the existing glue-alpha-module, but these changes streamline the developer
experience, introduce new constants for defaults, and replacing synth-time
validations with interface contracts for enforcement of the parameter combinations
that Glue supports. As an opinionated construct, the Glue L2 construct does
not allow developers to create resources that use non-current versions
of Glue or deprecated language dependencies (e.g. deprecated versions of Python).
As always, L1s allow you to specify a wider range of parameters if you need
or want to use alternative configurations.

Optional and required parameters for each job are enforced via interface
rather than validation; see [Glue's public documentation](https://docs.aws.amazon.com/glue/latest/dg/aws-glue-api.html)
for more granular details.

### Spark Jobs

#### ETL Jobs

ETL jobs support pySpark and Scala languages, for which there are separate but
similar constructors. ETL jobs default to the G2 worker type, but you can
override this default with other supported worker type values (G1, G2, G4
and G8). ETL jobs defaults to Glue version 4.0, which you can override to 3.0.
The following ETL features are enabled by default:
`—enable-metrics, —enable-continuous-cloudwatch-log.`
The Spark UI (`—enable-spark-ui`) is off by default; enable it by setting the
`sparkUI` prop.
You can find more details about version, worker type and other features in
[Glue's public documentation](https://docs.aws.amazon.com/glue/latest/dg/aws-glue-api-jobs-job.html).

Reference the pyspark-etl-jobs.test.ts and scalaspark-etl-jobs.test.ts unit tests
for examples of required-only and optional job parameters when creating these
types of jobs.

For the sake of brevity, examples are shown using the pySpark job variety.

Example with only required parameters:

```ts
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
declare const stack: cdk.Stack;
declare const role: iam.IRole;
declare const script: glue.Code;
new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
  role,
  script,
  jobName: 'PySparkETLJob',
});
```

Example with optional override parameters:

```ts
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
declare const stack: cdk.Stack;
declare const role: iam.IRole;
declare const script: glue.Code;
new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
  jobName: 'PySparkETLJobCustomName',
  description: 'This is a description',
  role,
  script,
  glueVersion: glue.GlueVersion.V5_1,
  continuousLogging: { enabled: false },
  workerConfiguration: {
    workerType: glue.WorkerType.G_2X,
    numberOfWorkers: 2,
  },
  maxConcurrentRuns: 100,
  timeout: cdk.Duration.hours(2),
  connections: [glue.Connection.fromConnectionName(stack, 'Connection', 'connectionName')],
  securityConfiguration: glue.SecurityConfiguration.fromSecurityConfigurationName(stack, 'SecurityConfig', 'securityConfigName'),
  tags: {
    FirstTagName: 'FirstTagValue',
    SecondTagName: 'SecondTagValue',
    XTagName: 'XTagValue',
  },
  maxRetries: 2,
});
```

#### Streaming Jobs

Streaming jobs are similar to ETL jobs, except that they perform ETL on data
streams using the Apache Spark Structured Streaming framework. Some Spark
job features are not available to Streaming ETL jobs. They support Scala
and pySpark languages. PySpark streaming jobs run on Python 3. It
defaults to the G2 worker type and Glue 4.0, both of which you can override.
The following best practice features are enabled by default:
`—enable-metrics, —enable-continuous-cloudwatch-log`.
The Spark UI (`—enable-spark-ui`) is off by default; enable it by setting the
`sparkUI` prop.

Reference the pyspark-streaming-jobs.test.ts and scalaspark-streaming-jobs.test.ts 
unit tests for examples of required-only and optional job parameters when creating
these types of jobs.

Example with only required parameters:

```ts
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
declare const stack: cdk.Stack;
declare const role: iam.IRole;
declare const script: glue.Code;
new glue.PySparkStreamingJob(stack, 'ImportedJob', { role, script });
```

Example with optional override parameters:

```ts
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
declare const stack: cdk.Stack;
declare const role: iam.IRole;
declare const script: glue.Code;
new glue.PySparkStreamingJob(stack, 'PySparkStreamingJob', {
  jobName: 'PySparkStreamingJobCustomName',
  description: 'This is a description',
  role,
  script,
  glueVersion: glue.GlueVersion.V5_1,
  continuousLogging: { enabled: false },
  workerConfiguration: {
    workerType: glue.WorkerType.G_2X,
    numberOfWorkers: 2,
  },
  maxConcurrentRuns: 100,
  timeout: cdk.Duration.hours(2),
  connections: [glue.Connection.fromConnectionName(stack, 'Connection', 'connectionName')],
  securityConfiguration: glue.SecurityConfiguration.fromSecurityConfigurationName(stack, 'SecurityConfig', 'securityConfigName'),
  tags: {
    FirstTagName: 'FirstTagValue',
    SecondTagName: 'SecondTagValue',
    XTagName: 'XTagValue',
  },
  maxRetries: 2,
});
```

#### Flex Jobs

The flexible execution class is appropriate for non-urgent jobs such as
pre-production jobs, testing, and one-time data loads. Flexible jobs default
to Glue version 5.0 and worker type `G_2X`. The following best practice
features are enabled by default:
`—enable-metrics, —enable-continuous-cloudwatch-log`
The Spark UI (`—enable-spark-ui`) is off by default; enable it by setting the
`sparkUI` prop.

Reference the pyspark-flex-etl-jobs.test.ts and scalaspark-flex-etl-jobs.test.ts 
unit tests for examples of required-only and optional job parameters when creating
these types of jobs.

Example with only required parameters:

```ts
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
declare const stack: cdk.Stack;
declare const role: iam.IRole;
declare const script: glue.Code;
new glue.PySparkFlexEtlJob(stack, 'ImportedJob', { role, script });
```

Example with optional override parameters:

```ts
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
declare const stack: cdk.Stack;
declare const role: iam.IRole;
declare const script: glue.Code;
new glue.PySparkFlexEtlJob(stack, 'pySparkFlexEtlJob', {
  jobName: 'pySparkFlexEtlJob',
  description: 'This is a description',
  role,
  script,
  glueVersion: glue.GlueVersion.V5_1,
  continuousLogging: { enabled: false },
  workerConfiguration: {
    workerType: glue.WorkerType.G_2X,
    numberOfWorkers: 2,
  },
  maxConcurrentRuns: 100,
  timeout: cdk.Duration.hours(2),
  connections: [glue.Connection.fromConnectionName(stack, 'Connection', 'connectionName')],
  securityConfiguration: glue.SecurityConfiguration.fromSecurityConfigurationName(stack, 'SecurityConfig', 'securityConfigName'),
  tags: {
    FirstTagName: 'FirstTagValue',
    SecondTagName: 'SecondTagValue',
    XTagName: 'XTagValue',
  },
  maxRetries: 2,
});
```

### Python Shell Jobs

Python shell jobs support a Python version that depends on the AWS Glue
version you use. These can be used to schedule and run tasks that don't
require an Apache Spark environment. Python shell jobs default to
Python 3.9 and a MaxCapacity of `0.0625`. Python 3.9 supports pre-loaded
analytics libraries using the `library-set=analytics` flag, which is
enabled by default.

Reference the pyspark-shell-job.test.ts unit tests for examples of 
required-only and optional job parameters when creating these types of jobs.

Example with only required parameters:

```ts
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
declare const stack: cdk.Stack;
declare const role: iam.IRole;
declare const script: glue.Code;
new glue.PythonShellJob(stack, 'ImportedJob', { role, script });
```

Example with optional override parameters:

```ts
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
declare const stack: cdk.Stack;
declare const role: iam.IRole;
declare const script: glue.Code;
declare const extraPythonFile: glue.Code;
new glue.PythonShellJob(stack, 'PythonShellJob', {
  jobName: 'PythonShellJobCustomName',
  description: 'This is a description',
  pythonVersion: glue.PythonVersion.THREE_NINE,
  maxCapacity: glue.MaxCapacity.DPU_1,
  role,
  script,
  extraPythonFiles: [extraPythonFile],
  glueVersion: glue.GlueVersion.V3_0,
  continuousLogging: { enabled: false },
  maxConcurrentRuns: 100,
  timeout: cdk.Duration.hours(2),
  connections: [glue.Connection.fromConnectionName(stack, 'Connection', 'connectionName')],
  securityConfiguration: glue.SecurityConfiguration.fromSecurityConfigurationName(stack, 'SecurityConfig', 'securityConfigName'),
  tags: {
    FirstTagName: 'FirstTagValue',
    SecondTagName: 'SecondTagValue',
    XTagName: 'XTagValue',
  },
  maxRetries: 2,
});
```

### Ray Jobs

> **⚠️ DEPRECATED:** AWS Glue for Ray is closed to new customers as of April 30, 2026 and is in maintenance mode.
> Migrate to [Amazon EKS with KubeRay Operator](https://docs.aws.amazon.com/glue/latest/dg/awsglue-ray-jobs-availability-change.html).

The `RayJob` construct, `Runtime.RAY_TWO_FOUR`, and `JobType.RAY` are deprecated and will be removed in a future release.

### Metrics Control

By default, Glue jobs enable CloudWatch metrics (`--enable-metrics`) and observability metrics (`--enable-observability-metrics`) for monitoring and debugging. You can disable these metrics to reduce CloudWatch costs:

```ts
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
declare const stack: cdk.Stack;
declare const role: iam.IRole;
declare const script: glue.Code;

// Disable both metrics for cost optimization
new glue.PySparkEtlJob(stack, 'CostOptimizedJob', {
  role,
  script,
  enableMetrics: false,
  enableObservabilityMetrics: false,
});

// Selective control - keep observability, disable profiling
new glue.PySparkEtlJob(stack, 'SelectiveJob', {
  role,
  script,
  enableMetrics: false,
  // enableObservabilityMetrics defaults to true
});
```

This feature is available for all Spark job types (ETL, Streaming, Flex).

### Enable Job Run Queuing

AWS Glue job queuing monitors your account level quotas and limits. If quotas or limits are insufficient to start a Glue job run, AWS Glue will automatically queue the job and wait for limits to free up. Once limits become available, AWS Glue will retry the job run. Glue jobs will queue for limits like max concurrent job runs per account, max concurrent Data Processing Units (DPU), and resource unavailable due to IP address exhaustion in Amazon Virtual Private Cloud (Amazon VPC).

Enable job run queuing by setting the `jobRunQueuingEnabled` property to `true`.

```ts
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
declare const stack: cdk.Stack;
declare const role: iam.IRole;
declare const script: glue.Code;
new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
  role,
  script,
  jobName: 'PySparkETLJob',
  jobRunQueuingEnabled: true
});
```

### Uploading scripts from the CDK app repository to S3

Similar to other L2 constructs, the Glue L2 automates uploading local
scripts to S3. Use `glue.Code.fromAsset(path)` to point at a script in your
local file structure; it is uploaded to the CDK-managed asset bucket. To
reference a script that already exists in S3, use
`glue.Code.fromBucket(bucket, key)`, which performs no upload. A `script` is
required for every job.

Reference the unit tests for examples of repo and S3 code target examples.

### Workflow Triggers

You can use Glue workflows to create and visualize complex
extract, transform, and load (ETL) activities involving multiple crawlers,
jobs, and triggers. Standalone triggers are an anti-pattern, so you must
create triggers from within a workflow using the L2 construct.

Within a workflow object, there are functions to create different
types of triggers with actions and predicates. You add triggers to the
workflow, and each trigger references the jobs or crawlers it runs as its
actions.

`startOnCreation` applies to scheduled triggers (and, via
`ConditionalTriggerOptions`, conditional triggers) only. It defaults to `false`,
but you can override it if you prefer for your trigger to start on creation.

Reference the workflow-triggers.test.ts unit tests for examples of creating
workflows and triggers.

```ts
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
declare const stack: cdk.Stack;
declare const role: iam.IRole;
declare const script: glue.Code;

// Create a job to run from the workflow
const job = new glue.PySparkEtlJob(stack, 'Job', { role, script });

// Create a workflow and add a trigger that runs the job
const workflow = new glue.Workflow(stack, 'Workflow');
workflow.addOnDemandTrigger('OnDemandTrigger', {
  actions: [{ job }],
});
```

#### **1. On-Demand Triggers**

On-demand triggers can start glue jobs or crawlers. This construct provides
convenience functions to create on-demand crawler or job triggers. The constructor
takes an optional description parameter, but abstracts the requirement of an
actions list using the job or crawler objects using conditional types.

#### **2. Scheduled Triggers**

You can create scheduled triggers using cron expressions. This construct
provides daily and weekly convenience functions,
as well as a custom function that allows you to create your own
custom timing using the [existing event Schedule class](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_events.Schedule.html)
without having to build your own cron expressions. The L2 extracts
the expression that Glue requires from the Schedule object. The constructor
takes an optional description and a list of jobs or crawlers as actions.

#### **3. Notify  Event Triggers**

There are two types of notify event triggers: batching and non-batching.
For batching triggers, you must specify `BatchSize`. For non-batching
triggers, `BatchSize` defaults to 1. For both triggers, `BatchWindow`
defaults to 900 seconds, but you can override the window to align with
your workload's requirements.

#### **4. Conditional Triggers**

Conditional triggers have a predicate and actions associated with them.
The trigger actions are executed when the predicateCondition is true.

### Connection Properties

A `Connection` allows Glue jobs, crawlers and development endpoints to access
certain types of data stores.

* **Secrets Management**
    You must specify JDBC connection credentials in Secrets Manager and
    provide the Secrets Manager Key name as a property to the job connection.

* **Networking - the CDK determines the best fit subnet for Glue connection
configuration**
    The prior version of the glue-alpha-module requires the developer to
    specify the subnet of the Connection when it’s defined. Now, you can still
    specify the specific subnet you want to use, but are no longer required
    to. You are only required to provide a VPC and either a public or private
    subnet selection. Without a specific subnet provided, the L2 leverages the
    existing [EC2 Subnet Selection](https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_ec2/SubnetSelection.html)
    library to make the best choice selection for the subnet.

```ts
declare const securityGroup: ec2.SecurityGroup;
declare const subnet: ec2.Subnet;
new glue.Connection(this, 'MyConnection', {
  type: glue.ConnectionType.NETWORK,
  // The security groups granting AWS Glue inbound access to the data source within the VPC
  securityGroups: [securityGroup],
  // The VPC subnet which contains the data source
  subnet,
});
```

For RDS `Connection` by JDBC, it is recommended to manage credentials using AWS Secrets Manager. To use Secret, specify `SECRET_ID` in `properties` like the following code. Note that in this case, the subnet must have a route to the AWS Secrets Manager VPC endpoint or to the AWS Secrets Manager endpoint through a NAT gateway.

```ts
declare const securityGroup: ec2.SecurityGroup;
declare const subnet: ec2.Subnet;
declare const db: rds.DatabaseCluster;
new glue.Connection(this, "RdsConnection", {
  type: glue.ConnectionType.JDBC,
  securityGroups: [securityGroup],
  subnet,
  properties: {
    JDBC_CONNECTION_URL: `jdbc:mysql://${db.clusterEndpoint.socketAddress}/databasename`,
    JDBC_ENFORCE_SSL: "false",
    SECRET_ID: db.secret!.secretName,
  },
});
```

Connection `properties` are emitted verbatim into the CloudFormation template, so
any credential placed there in plaintext is stored in plaintext in the template,
`cdk.out`, and source control. Reference a Secrets Manager secret through
`SECRET_ID` (as above) instead. If a property key looks like a credential (for
example `PASSWORD`, `SECRET`, or `TOKEN`) and holds a plaintext literal, the
construct emits a synthesis-time warning.

If you need to use a connection type that doesn't exist as a static member on `ConnectionType`, you can instantiate a `ConnectionType` object, e.g: `new glue.ConnectionType('NEW_TYPE')`.

See [Adding a Connection to Your Data Store](https://docs.aws.amazon.com/glue/latest/dg/populate-add-connection.html) and [Connection Structure](https://docs.aws.amazon.com/glue/latest/dg/aws-glue-api-catalog-connections.html#aws-glue-api-catalog-connections-Connection) documentation for more information on the supported data stores and their configurations.

## SecurityConfiguration

A `SecurityConfiguration` is a set of security properties that can be used by AWS Glue to encrypt data at rest.

Each encryption config is built with a factory that pairs the encryption mode
with its key, so illegal combinations (such as an S3-managed encryption carrying
a KMS key) cannot be expressed:

```ts
new glue.SecurityConfiguration(this, 'MySecurityConfiguration', {
  cloudWatchEncryption: glue.CloudWatchEncryption.kms(),
  jobBookmarksEncryption: glue.JobBookmarksEncryption.clientSideKms(),
  s3Encryption: glue.S3Encryption.kms(),
});
```

By default, a shared KMS key is created for use with the encryption configurations that require one. You can also supply your own key to any factory, for example, for CloudWatch encryption:

```ts
declare const key: kms.Key;
new glue.SecurityConfiguration(this, 'MySecurityConfiguration', {
  cloudWatchEncryption: glue.CloudWatchEncryption.kms(key),
});
```

Use `glue.S3Encryption.s3Managed()` for S3-managed (SSE-S3) encryption, which takes no key.

See [documentation](https://docs.aws.amazon.com/glue/latest/dg/encryption-security-configuration.html) for more info for Glue encrypting data written by Crawlers, Jobs, and Development Endpoints.

## Catalog

The Glue Data Catalog is a persistent metadata store for your data assets. Every
account has an implicit, account-wide catalog that always exists, and you can also
create additional catalogs as `AWS::Glue::Catalog` resources (for example, to
federate to another metastore).

A catalog's encryption is fixed when the catalog is created: a catalog either
carries encryption settings or it does not. This keeps its configuration easy to
reason about — there are no mutation methods that change encryption after the fact.

### The account-wide catalog

Use `Catalog.forAccount(scope)` to obtain the implicit account catalog. It is not
a CloudFormation resource — it always exists. Repeated calls within the same stack
return the same instance:

```ts
const catalog = glue.Catalog.forAccount(this);
```

To configure Data Catalog encryption for the account, use
`Catalog.encryptAccount(scope, options)`:

```ts
declare const key: kms.Key;
glue.Catalog.encryptAccount(this, {
  encryptionAtRest: glue.DataCatalogEncryptionAtRest.kms(key),
});
```

Because encryption is fixed at construction, `encryptAccount` must be called
*before* the account catalog is first used in the stack — before any
`Catalog.forAccount(this)` call, and before any `Database` that uses the account
catalog. Calling it after the account catalog has been materialized throws.

The account catalog's encryption is an account- and region-wide setting, managed
through the singleton `PutDataCatalogEncryptionSettings` API. Configure it in
exactly one stack. Configuring it from multiple stacks in the same account and
region makes those stacks overwrite one another at deploy time, and the result is
order-dependent. Unlike duplicate settings within a single stack (which
CloudFormation rejects), this cross-stack conflict is not caught at synthesis
time, because each stack synthesizes to its own template.

### Creating a catalog

To create a new catalog resource, use the `Catalog` constructor. Encryption is
configured through the `encryptionAtRest` and `connectionPasswordEncryption` props:

```ts
new glue.Catalog(this, 'MyCatalog', {
  catalogName: 'my-catalog',
  description: 'my catalog description',
});
```

### Encryption at rest

Configure Data Catalog encryption at rest through the `encryptionAtRest` option
(on `Catalog.encryptAccount` or the `Catalog` constructor).
It accepts a `DataCatalogEncryptionAtRest` describing the mode:

```ts
declare const key: kms.Key;

// SSE-KMS with a customer-managed key
glue.Catalog.encryptAccount(this, {
  encryptionAtRest: glue.DataCatalogEncryptionAtRest.kms(key),
});

// SSE-KMS with an AWS-managed key (omit the key)
new glue.Catalog(this, 'ManagedKeyCatalog', {
  catalogName: 'managed-key-catalog',
  encryptionAtRest: glue.DataCatalogEncryptionAtRest.kms(),
});

// Disable encryption at rest
new glue.Catalog(this, 'PlaintextCatalog', {
  catalogName: 'plaintext-catalog',
  encryptionAtRest: glue.DataCatalogEncryptionAtRest.disabled(),
});
```

When you use `SSE-KMS-WITH-SERVICE-ROLE`, AWS Glue accesses the KMS key through a
service role you provide. If you pass a customer-managed key, the role is
automatically granted the permissions it needs to encrypt and decrypt catalog data:

```ts
import * as iam from 'aws-cdk-lib/aws-iam';
declare const key: kms.Key;
declare const role: iam.IRole;
glue.Catalog.encryptAccount(this, {
  encryptionAtRest: glue.DataCatalogEncryptionAtRest.kmsWithServiceRole(role, key),
});
```

The customer-managed key, when configured, is exposed on the catalog as
`encryptionKey` (and the connection-password key as `connectionPasswordKey`), so
you can reference it to grant additional access. It is undefined when encryption is
disabled or an AWS-managed key is used.

### Connection password encryption

Independently from encryption at rest, the Data Catalog can encrypt the passwords
stored in connection properties. Configure it through the
`connectionPasswordEncryption` option:

```ts
declare const key: kms.Key;
glue.Catalog.encryptAccount(this, {
  connectionPasswordEncryption: {
    kmsKey: key,
    // Whether GetConnection/GetConnections return the password encrypted (default: true)
    returnConnectionPasswordEncrypted: true,
  },
});
```

The two encryption blocks are independent: enabling one does not require the other,
and each may use a different KMS key. The customer-managed key for connection
passwords is exposed as `connectionPasswordKey`.

### Importing a catalog

You can import an existing catalog by ARN or by id. An imported catalog is a pure
identity handle — it emits no resources and does not manage the catalog's
encryption:

```ts
const byId = glue.Catalog.fromCatalogId(this, 'ById', 'my-catalog-id');
const byArn = glue.Catalog.fromCatalogArn(this, 'ByArn', 'arn:aws:glue:us-east-1:123456789012:catalog/my-catalog-id');
```

To manage the Data Catalog encryption of a catalog you did not create in this
stack, add a `CfnDataCatalogEncryptionSettings` resource targeting its id
directly. Do this from exactly one stack: like the account catalog, a catalog has
a single encryption configuration, so two settings resources targeting the same id
race to overwrite one another at deploy time. Within a single stack this is caught
by CloudFormation template validation (E3019, duplicate primary identifiers);
across stacks it is not, since each stack synthesizes to its own template.

```ts
import { CfnDataCatalogEncryptionSettings } from 'aws-cdk-lib/aws-glue';

new CfnDataCatalogEncryptionSettings(this, 'Encryption', {
  catalogId: 'my-catalog-id',
  dataCatalogEncryptionSettings: {
    encryptionAtRest: { catalogEncryptionMode: 'SSE-KMS' },
  },
});
```

## Database

A `Database` is a logical grouping of `Tables` in the Glue Catalog.

```ts
new glue.Database(this, 'MyDatabase', {
  databaseName: 'my_database',
  description: 'my_database_description',
});
```

Because a database is a container for tables and their metadata, it is retained
by default when removed from the stack, to avoid accidental data loss. Set
`removalPolicy` to `RemovalPolicy.DESTROY` to have it deleted instead:

```ts
import { RemovalPolicy } from 'aws-cdk-lib';

new glue.Database(this, 'MyDatabase', {
  databaseName: 'my_database',
  removalPolicy: RemovalPolicy.DESTROY,
});
```

## Table

A Glue table describes a table of data in S3: its structure (column names and types), location of data (S3 objects with a common prefix in a S3 bucket), and format for the files (Json, Avro, Parquet, etc.):

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  database: myDatabase,
  columns: [{
    name: 'col1',
    type: glue.Schema.STRING,
  }, {
    name: 'col2',
    type: glue.Schema.array(glue.Schema.STRING),
    comment: 'col2 is an array of strings' // comment is optional
  }],
  dataFormat: glue.DataFormat.JSON,
});
```

By default, a S3 bucket will be created to store the table's data but you can manually pass the `bucket` and `s3Prefix`:

```ts
declare const myBucket: s3.Bucket;
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  bucket: myBucket,
  s3Prefix: 'my-table/',
  // ...
  database: myDatabase,
  columns: [{
    name: 'col1',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
});
```

Glue tables can be configured to contain user-defined properties, to describe the physical storage of table data, through the `storageParameters` property:

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  storageParameters: [
    glue.StorageParameter.skipHeaderLineCount(1),
    glue.StorageParameter.compressionType(glue.CompressionType.GZIP),
    glue.StorageParameter.custom('separatorChar', ',')
  ],
  // ...
  database: myDatabase,
  columns: [{
    name: 'col1',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
});
```

Glue tables can also be configured to contain user-defined table properties through the [`parameters`](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-glue-table-tableinput.html#cfn-glue-table-tableinput-parameters) property:

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  parameters: {
    key1: 'val1',
    key2: 'val2',
  },
  database: myDatabase,
  columns: [{
    name: 'col1',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
});
```

### Partition Keys

To improve query performance, a table can specify `partitionKeys` on which data is stored and queried separately. For example, you might partition a table by `year` and `month` to optimize queries based on a time window:

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  database: myDatabase,
  columns: [{
    name: 'col1',
    type: glue.Schema.STRING,
  }],
  partitionKeys: [{
    name: 'year',
    type: glue.Schema.SMALL_INT,
  }, {
    name: 'month',
    type: glue.Schema.SMALL_INT,
  }],
  dataFormat: glue.DataFormat.JSON,
});
```

### Partition Indexes

Another way to improve query performance is to specify partition indexes. If no partition indexes are
present on the table, AWS Glue loads all partitions of the table and filters the loaded partitions using
the query expression. The query takes more time to run as the number of partitions increase. With an
index, the query will try to fetch a subset of the partitions instead of loading all partitions of the
table.

The keys of a partition index must be a subset of the partition keys of the table. You can have a
maximum of 3 partition indexes per table. To specify a partition index, you can use the `partitionIndexes`
property:

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  database: myDatabase,
  columns: [{
    name: 'col1',
    type: glue.Schema.STRING,
  }],
  partitionKeys: [{
    name: 'year',
    type: glue.Schema.SMALL_INT,
  }, {
    name: 'month',
    type: glue.Schema.SMALL_INT,
  }],
  partitionIndexes: [{
    indexName: 'my-index', // optional
    keyNames: ['year'],
  }], // supply up to 3 indexes
  dataFormat: glue.DataFormat.JSON,
});
```

Alternatively, you can call the `addPartitionIndex()` function on a table:

```ts
declare const myTable: glue.S3Table;
myTable.addPartitionIndex({
  indexName: 'my-index',
  keyNames: ['year'],
});
```

### Partition Filtering

If you have a table with a large number of partitions that grows over time, consider using AWS Glue partition indexing and filtering.

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  database: myDatabase,
  columns: [{
      name: 'col1',
      type: glue.Schema.STRING,
  }],
  partitionKeys: [{
      name: 'year',
      type: glue.Schema.SMALL_INT,
  }, {
      name: 'month',
      type: glue.Schema.SMALL_INT,
  }],
  dataFormat: glue.DataFormat.JSON,
  enablePartitionFiltering: true,
});
```

### Partition Projection

Partition projection allows Athena to automatically add new partitions as new data arrives, without requiring `ALTER TABLE ADD PARTITION` statements. This improves query performance and reduces management overhead by eliminating the need to manually manage partition metadata.

For more information, see the [AWS documentation on partition projection](https://docs.aws.amazon.com/athena/latest/ug/partition-projection.html).

#### INTEGER Projection

For partition keys with sequential numeric values:

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  database: myDatabase,
  columns: [{
    name: 'data',
    type: glue.Schema.STRING,
  }],
  partitionKeys: [{
    name: 'year',
    type: glue.Schema.INTEGER,
  }],
  dataFormat: glue.DataFormat.JSON,
  partitionProjection: {
    year: glue.PartitionProjectionConfiguration.integer({
      min: 2020,
      max: 2023,
      interval: 1,  // optional, defaults to 1
      digits: 4,    // optional, pads with leading zeros
    }),
  },
});
```

#### DATE Projection

For partition keys with date or timestamp values. Supports both fixed dates and relative dates using `NOW`:

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  database: myDatabase,
  columns: [{
    name: 'data',
    type: glue.Schema.STRING,
  }],
  partitionKeys: [{
    name: 'date',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
  partitionProjection: {
    date: glue.PartitionProjectionConfiguration.date({
      min: '2020-01-01',
      max: '2023-12-31',
      format: 'yyyy-MM-dd',
      interval: 1,  // optional, defaults to 1
      intervalUnit: glue.DateIntervalUnit.DAYS,  // optional: YEARS, MONTHS, WEEKS, DAYS, HOURS, MINUTES, SECONDS
    }),
  },
});
```

You can also use relative dates with `NOW`:

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  database: myDatabase,
  columns: [{
    name: 'data',
    type: glue.Schema.STRING,
  }],
  partitionKeys: [{
    name: 'date',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
  partitionProjection: {
    date: glue.PartitionProjectionConfiguration.date({
      min: 'NOW-3YEARS',
      max: 'NOW',
      format: 'yyyy-MM-dd',
    }),
  },
});
```

#### ENUM Projection

For partition keys with a known set of values:

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  database: myDatabase,
  columns: [{
    name: 'data',
    type: glue.Schema.STRING,
  }],
  partitionKeys: [{
    name: 'region',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
  partitionProjection: {
    region: glue.PartitionProjectionConfiguration.enum({
      values: ['us-east-1', 'us-west-2', 'eu-west-1'],
    }),
  },
});
```

#### INJECTED Projection

For custom partition values injected at query time:

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  database: myDatabase,
  columns: [{
    name: 'data',
    type: glue.Schema.STRING,
  }],
  partitionKeys: [{
    name: 'custom',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
  partitionProjection: {
    custom: glue.PartitionProjectionConfiguration.injected(),
  },
});
```

#### Multiple Partition Projections

You can configure partition projection for multiple partition keys:

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  database: myDatabase,
  columns: [{
    name: 'data',
    type: glue.Schema.STRING,
  }],
  partitionKeys: [
    {
      name: 'year',
      type: glue.Schema.INTEGER,
    },
    {
      name: 'month',
      type: glue.Schema.INTEGER,
    },
    {
      name: 'region',
      type: glue.Schema.STRING,
    },
  ],
  dataFormat: glue.DataFormat.JSON,
  partitionProjection: {
    year: glue.PartitionProjectionConfiguration.integer({
      min: 2020,
      max: 2023,
    }),
    month: glue.PartitionProjectionConfiguration.integer({
      min: 1,
      max: 12,
      digits: 2,
    }),
    region: glue.PartitionProjectionConfiguration.enum({
      values: ['us-east-1', 'us-west-2'],
    }),
  },
});
```

### Glue Connections

Glue connections allow external data connections to third party databases and data warehouses. However, these connections can also be assigned to Glue Tables, allowing you to query external data sources using the Glue Data Catalog.

Whereas `S3Table` will point to (and if needed, create) a bucket to store the tables' data, `ExternalTable` will point to an existing table in a data source. For example, to create a table in Glue that points to a table in Redshift:

```ts
declare const myConnection: glue.Connection;
declare const myDatabase: glue.Database;
new glue.ExternalTable(this, 'MyTable', {
  connection: myConnection,
  externalDataLocation: 'default_db_public_example', // A table in Redshift
  // ...
  database: myDatabase,
  columns: [{
    name: 'col1',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
});
```

## [Encryption](https://docs.aws.amazon.com/athena/latest/ug/encryption.html)

When the table creates its own S3 bucket (i.e. you do not pass an explicit `bucket`), that bucket enforces SSL: a bucket policy denies any request made over plain HTTP. If you provide your own bucket, enabling `enforceSSL` on it is your responsibility.

You can enable encryption on a Table's data:

* [S3Managed](https://docs.aws.amazon.com/AmazonS3/latest/dev/UsingServerSideEncryption.html) - (default) Server side encryption (`SSE-S3`) with an Amazon S3-managed key.

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  encryption: glue.TableEncryption.S3_MANAGED,
  // ...
  database: myDatabase,
  columns: [{
    name: 'col1',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
});
```

* [Kms](https://docs.aws.amazon.com/AmazonS3/latest/dev/UsingKMSEncryption.html) - Server-side encryption (`SSE-KMS`) with an AWS KMS Key managed by the account owner.

```ts
declare const myDatabase: glue.Database;
// KMS key is created automatically
new glue.S3Table(this, 'MyTable', {
  encryption: glue.TableEncryption.KMS,
  // ...
  database: myDatabase,
  columns: [{
    name: 'col1',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
});

// with an explicit KMS key
new glue.S3Table(this, 'MyTable', {
  encryption: glue.TableEncryption.KMS,
  encryptionKey: new kms.Key(this, 'MyKey'),
  // ...
  database: myDatabase,
  columns: [{
    name: 'col1',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
});
```

* [KmsManaged](https://docs.aws.amazon.com/AmazonS3/latest/dev/UsingKMSEncryption.html) - Server-side encryption (`SSE-KMS`), like `Kms`, except with an AWS KMS Key managed by the AWS Key Management Service.

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  encryption: glue.TableEncryption.KMS_MANAGED,
  // ...
  database: myDatabase,
  columns: [{
    name: 'col1',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
});
```

* [ClientSideKms](https://docs.aws.amazon.com/AmazonS3/latest/dev/UsingClientSideEncryption.html#client-side-encryption-kms-managed-master-key-intro) - Client-side encryption (`CSE-KMS`) with an AWS KMS Key managed by the account owner.

```ts
declare const myDatabase: glue.Database;
// KMS key is created automatically
new glue.S3Table(this, 'MyTable', {
  encryption: glue.TableEncryption.CLIENT_SIDE_KMS, 
  // ...
  database: myDatabase,
  columns: [{
    name: 'col1',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
});

// with an explicit KMS key
new glue.S3Table(this, 'MyTable', {
  encryption: glue.TableEncryption.CLIENT_SIDE_KMS,
  encryptionKey: new kms.Key(this, 'MyKey'),
  // ...
  database: myDatabase,
  columns: [{
    name: 'col1',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
});
```

*Note: you cannot provide a `Bucket` when creating the `S3Table` if you wish to use server-side encryption (`KMS`, `KMS_MANAGED` or `S3_MANAGED`)*.

### Marking table data as encrypted

Both `S3Table` and `ExternalTable` set the `has_encrypted_data` table parameter, which
Athena reads when querying client-side (`CSE-KMS`) encrypted datasets. It defaults to `true`.
Set `hasEncryptedData` to `false` when the underlying data is not encrypted:

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  hasEncryptedData: false,
  database: myDatabase,
  columns: [{
    name: 'col1',
    type: glue.Schema.STRING,
  }],
  dataFormat: glue.DataFormat.JSON,
});
```

Do not set `has_encrypted_data` through the free-form `parameters` map as well - a value
there that conflicts with `hasEncryptedData` is rejected at synthesis time.

## Types

A table's schema is a collection of columns, each of which have a `name` and a `type`. Types are recursive structures, consisting of primitive and complex types:

```ts
declare const myDatabase: glue.Database;
new glue.S3Table(this, 'MyTable', {
  columns: [{
    name: 'primitive_column',
    type: glue.Schema.STRING,
  }, {
    name: 'array_column',
    type: glue.Schema.array(glue.Schema.INTEGER),
    comment: 'array<integer>',
  }, {
    name: 'map_column',
    type: glue.Schema.map(
      glue.Schema.STRING,
      glue.Schema.TIMESTAMP),
    comment: 'map<string,timestamp>',
  }, {
    name: 'struct_column',
    type: glue.Schema.struct([{
      name: 'nested_column',
      type: glue.Schema.DATE,
      comment: 'nested comment',
    }]),
    comment: "struct<nested_column:date COMMENT 'nested comment'>",
  }],
  // ...
  database: myDatabase,
  dataFormat: glue.DataFormat.JSON,
});  
```

## Public FAQ

### What are we launching today?

We’re launching new features to an AWS CDK Glue L2 Construct to provide
best-practice defaults and convenience methods to create Glue Jobs, Connections,
Triggers, Workflows, and the underlying permissions and configuration.

### Why should I use this Construct?

Developers should use this Construct to reduce the amount of boilerplate
code and complexity each individual has to navigate, and make it easier to
create best-practice Glue resources.

### What’s not in scope?

Glue Crawlers and other resources that are now managed by the AWS LakeFormation
team are not in scope for this effort. Developers should use existing methods
to create these resources, and the new Glue L2 construct assumes they already
exist as inputs. While best practice is for application and infrastructure code
to be as close as possible for teams using fully-implemented DevOps mechanisms,
in practice these ETL scripts are likely managed by a data science team who
know Python or Scala and don’t necessarily own or manage their own
infrastructure deployments. We want to meet developers where they are, and not
assume that all of the code resides in the same repository, Developers can
automate this themselves via the CDK, however, if they do own both.

Validating Glue version and feature use per AWS region at synth time is also
not in scope. AWS’ intention is for all features to eventually be propagated to
all Global regions, so the complexity involved in creating and updating region-
specific configuration to match shifting feature sets does not out-weigh the
likelihood that a developer will use this construct to deploy resources to a
region without a particular new feature to a region that doesn’t yet support
it without researching or manually attempting to use that feature before
developing it via IaC. The developer will, of course, still get feedback from
the underlying Glue APIs as CloudFormation deploys the resources similar to the
current CDK L1 Glue experience.
