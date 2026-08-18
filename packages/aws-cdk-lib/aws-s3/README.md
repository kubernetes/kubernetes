# Amazon S3 Construct Library



Define an S3 bucket.

```ts
const bucket = new s3.Bucket(this, 'MyFirstBucket');
```

`Bucket` constructs expose the following deploy-time attributes:

- `bucketArn` - the ARN of the bucket (i.e. `arn:aws:s3:::amzn-s3-demo-bucket`)
- `bucketName` - the name of the bucket (i.e. `amzn-s3-demo-bucket`)
- `bucketWebsiteUrl` - the Website URL of the bucket (i.e.
  `http://amzn-s3-demo-bucket.s3-website-us-west-1.amazonaws.com`)
- `bucketDomainName` - the URL of the bucket (i.e. `amzn-s3-demo-bucket.s3.amazonaws.com`)
- `bucketDualStackDomainName` - the dual-stack URL of the bucket (i.e.
  `amzn-s3-demo-bucket.s3.dualstack.eu-west-1.amazonaws.com`)
- `bucketRegionalDomainName` - the regional URL of the bucket (i.e.
  `amzn-s3-demo-bucket.s3.eu-west-1.amazonaws.com`)
- `arnForObjects(pattern)` - the ARN of an object or objects within the bucket (i.e.
  `arn:aws:s3:::amzn-s3-demo-bucket/exampleobject.png` or
  `arn:aws:s3:::amzn-s3-demo-bucket/Development/*`)
- `urlForObject(key)` - the HTTP URL of an object within the bucket (i.e.
  `https://s3.cn-north-1.amazonaws.com.cn/china-bucket/mykey`)
- `virtualHostedUrlForObject(key)` - the virtual-hosted style HTTP URL of an object
  within the bucket (i.e. `https://china-bucket-s3.cn-north-1.amazonaws.com.cn/mykey`)
- `s3UrlForObject(key)` - the S3 URL of an object within the bucket (i.e.
  `s3://bucket/mykey`)

## Encryption

Define a KMS-encrypted bucket:

```ts
const bucket = new s3.Bucket(this, 'MyEncryptedBucket', {
  encryption: s3.BucketEncryption.KMS,
});

// you can access the encryption key:
assert(bucket.encryptionKey instanceof kms.Key);
```

You can also supply your own key:

```ts
const myKmsKey = new kms.Key(this, 'MyKey');

const bucket = new s3.Bucket(this, 'MyEncryptedBucket', {
  encryption: s3.BucketEncryption.KMS,
  encryptionKey: myKmsKey,
});

assert(bucket.encryptionKey === myKmsKey);
```

Enable KMS-SSE encryption via [S3 Bucket Keys](https://docs.aws.amazon.com/AmazonS3/latest/dev/bucket-key.html):

```ts
const bucket = new s3.Bucket(this, 'MyEncryptedBucket', {
  encryption: s3.BucketEncryption.KMS,
  bucketKeyEnabled: true,
});
```

Use `BucketEncryption.ManagedKms` to use the S3 master KMS key:

```ts
const bucket = new s3.Bucket(this, 'Buck', {
  encryption: s3.BucketEncryption.KMS_MANAGED,
});

assert(bucket.encryptionKey == null);
```

Enable DSSE encryption:

```
const bucket = new s3.Bucket(stack, 'MyDSSEBucket', {
  encryption: s3.BucketEncryption.DSSE_MANAGED,
  bucketKeyEnabled: true,
});
```

Explicitly block uploads encrypted with SSE-C:

```ts
const bucket = new s3.Bucket(this, 'MySsecBlockedBucket', {
  blockedEncryptionTypes: [s3.BlockedEncryptionType.SSE_C],
});
```

Allow uploads with all encryption types:

```ts
const bucket = new s3.Bucket(this, 'MyBucket', {
  blockedEncryptionTypes: [s3.BlockedEncryptionType.NONE],
});
```

## Permissions

A bucket policy will be automatically created for the bucket upon the first call to
`addToResourcePolicy(statement)`:

```ts
const bucket = new s3.Bucket(this, 'MyBucket');
const result = bucket.addToResourcePolicy(
  new iam.PolicyStatement({
    actions: ['s3:GetObject'],
    resources: [bucket.arnForObjects('file.txt')],
    principals: [new iam.AccountRootPrincipal()],
  })
);
```

If you try to add a policy statement to an existing bucket, this method will
not do anything:

```ts
const bucket = s3.Bucket.fromBucketName(this, 'existingBucket', 'amzn-s3-demo-bucket');

// No policy statement will be added to the resource
const result = bucket.addToResourcePolicy(
  new iam.PolicyStatement({
    actions: ['s3:GetObject'],
    resources: [bucket.arnForObjects('file.txt')],
    principals: [new iam.AccountRootPrincipal()],
  })
);
```

That's because it's not possible to tell whether the bucket
already has a policy attached, let alone to re-use that policy to add more
statements to it. We recommend that you always check the result of the call:

```ts
const bucket = new s3.Bucket(this, 'MyBucket');
const result = bucket.addToResourcePolicy(
  new iam.PolicyStatement({
    actions: ['s3:GetObject'],
    resources: [bucket.arnForObjects('file.txt')],
    principals: [new iam.AccountRootPrincipal()],
  })
);

if (!result.statementAdded) {
  // Uh-oh! Someone probably made a mistake here.
}
```

The bucket policy can be directly accessed after creation to add statements or
adjust the removal policy.

```ts
const bucket = new s3.Bucket(this, 'MyBucket');
bucket.policy?.applyRemovalPolicy(cdk.RemovalPolicy.RETAIN);
```

Most of the time, you won't have to manipulate the bucket policy directly.
Instead, buckets have "grant" methods called to give prepackaged sets of permissions
to other resources. For example:

```ts
declare const myLambda: lambda.Function;

const bucket = new s3.Bucket(this, 'MyBucket');
bucket.grantReadWrite(myLambda);
```

Will give the Lambda's execution role permissions to read and write
from the bucket.

### Understanding "grant" Methods

The S3 construct library provides several grant methods for `IBucketRef` instances, which can
be accessed via the `BucketGrants` class:

```ts
declare const principal: iam.IPrincipal;
declare const bucket: s3.IBucketRef;

s3.BucketGrants.fromBucket(bucket).delete(principal);
```

If `bucket` is an instance of `CfnBucket`, and the grants process involves adding statements
to the bucket policy, then the `BucketGrants` class will, by default, do the same thing it
would do for an instance of `Bucket`: create a new bucket policy (or reuse an existing one)
and add the necessary statements to it.

But if you want to customize this behavior, you can register an instance of `IResourcePolicyFactory`
for the `AWS::S3::Bucket` CloudFormation type:

```ts nofixture
import { CfnResource } from 'aws-cdk-lib';
import { IResourcePolicyFactory, IResourceWithPolicyV2, PolicyStatement, ResourceWithPolicies } from 'aws-cdk-lib/aws-iam';
import { Construct, IConstruct } from 'constructs';

declare const scope: Construct;
class MyFactory implements IResourcePolicyFactory {
  forResource(resource: CfnResource): IResourceWithPolicyV2 {
    return {
      env: resource.env,
      addToResourcePolicy(statement: PolicyStatement) {
        // custom implementation to add the statement to the resource policy
        return { statementAdded: true, policyDependable: resource };
      }
    }
  }
}

ResourceWithPolicies.register(scope, 'AWS::S3::Bucket', new MyFactory());
```

`IResourcePolicyFactory` is responsible for converting a construct into a `IResourceWithPolicyV2`,
effectively providing an ad-hoc way to extend the behavior of L1s to support grants the same way
as L2s do.

The `BucketGrants` class has many methods, but two of them have a special behavior. These two
accept an `objectsKeyPattern` parameter to restrict granted permissions to specific resources:
- `read`
- `readWrite`

When examining the synthesized policy, you'll notice it includes both your specified object key
patterns and the bucket itself. This is by design. Some permissions (like `s3:ListBucket`) apply
at the bucket level, while others (like `s3:GetObject`) apply to specific objects.

Specifically, the [`s3:ListBucket` action operates on bucket resources](https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazons3.html#amazons3-bucket)
and requires the bucket ARN to work properly. This might be seen as a bug, giving the impression that more permissions were granted than the ones you intended, but the reality is that the policy does not ignore your `objectsKeyPattern` - object-specific actions like `s3:GetObject`
will still be limited to the resources defined in your pattern.

If you need to restrict the `s3:ListBucket` action to specific paths, you can add a `Condition` to your policy that limits the `objectsKeyPattern` to specific folders. For more details and examples, see the [AWS documentation on bucket policies](https://docs.aws.amazon.com/AmazonS3/latest/userguide/example-bucket-policies.html#example-bucket-policies-folders).

## Attribute-Based Access Control (ABAC)

You can enable ABAC (Attribute-Based Access Control) for an S3 general purpose bucket.
When ABAC is enabled for the general purpose bucket, you can use tags to manage access to the general purpose buckets as well as for cost tracking purposes.
When ABAC is disabled for the general purpose buckets, you can only use tags for cost tracking purposes.

To enable ABAC on a bucket:

```ts
const bucket = new s3.Bucket(this, 'Bucket', {
  abacStatus: true,
});
```

By default, if `abacStatus` is not specified, ABAC will not be enabled for the bucket.

For more information about ABAC and how to use it with S3, see the [AWS documentation on ABAC](https://docs.aws.amazon.com/AmazonS3/latest/userguide/buckets-tagging.html).

## AWS Foundational Security Best Practices

### Enforcing SSL

To require all requests use Secure Socket Layer (SSL):

```ts
const bucket = new s3.Bucket(this, 'Bucket', {
  enforceSSL: true,
});
```

To require a minimum TLS version for all requests:

```ts
const bucket = new s3.Bucket(this, 'Bucket', {
  enforceSSL: true,
  minimumTLSVersion: 1.2,
});
```

## Bucket Naming

By default, CloudFormation assigns a unique bucket name. You can also specify a `bucketName` directly, but this creates a globally unique name that could conflict with other accounts.

### Account-Regional Bucket Namespace

Using `bucketNamePrefix` with `bucketNamespace` set to `ACCOUNT_REGIONAL`, the bucket name is scoped to your account and region, reducing the risk of name conflicts. CloudFormation appends `-<accountId>-<region>-an` to the prefix to form the full name.

```ts
new s3.Bucket(this, 'MyBucket', {
  bucketNamePrefix: 'my-app',
  bucketNamespace: s3.BucketNamespace.ACCOUNT_REGIONAL,
});
// Resulting bucket name: my-app-123456789012-us-east-1-an
```

Note that `bucketName` cannot be used together with `bucketNamePrefix` or `bucketNamespace`.

For more information, see the [AWS documentation on bucket namespaces](https://docs.aws.amazon.com/AmazonS3/latest/userguide/gpbucketnamespaces.html).

## Sharing buckets between stacks

To use a bucket in a different stack in the same CDK application, pass the object to the other stack:

```ts
/**
 * Stack that defines the bucket
 */
class Producer extends Stack {
  public readonly myBucket: s3.Bucket;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const bucket = new s3.Bucket(this, 'MyBucket', {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    this.myBucket = bucket;
  }
}

interface ConsumerProps extends cdk.StackProps {
  userBucket: s3.IBucket;
}

/**
 * Stack that consumes the bucket
 */
class Consumer extends Stack {
  constructor(scope: Construct, id: string, props: ConsumerProps) {
    super(scope, id, props);

    const user = new iam.User(this, 'MyUser');
    props.userBucket.grantReadWrite(user);
  }
}

const app = new App();
const producer = new Producer(app, 'ProducerStack');
new Consumer(app, 'ConsumerStack', { userBucket: producer.myBucket });
```

## Importing existing buckets

To import an existing bucket into your CDK application, use the `Bucket.fromBucketAttributes`
factory method. This method accepts `BucketAttributes` which describes the properties of an already
existing bucket:

Note that this method allows importing buckets with legacy names containing uppercase letters (`A-Z`) or underscores (`_`), which were
permitted for buckets created before March 1, 2018. For buckets created after this date, uppercase letters and underscores
are not allowed in the bucket name.

```ts
declare const myLambda: lambda.Function;
const bucket = s3.Bucket.fromBucketAttributes(this, 'ImportedBucket', {
  bucketArn: 'arn:aws:s3:::amzn-s3-demo-bucket',
});

// now you can just call methods on the bucket
const filter: s3.NotificationKeyFilter = { prefix: 'home/myusername/*' };
bucket.addEventNotification(s3.EventType.OBJECT_CREATED, new s3n.LambdaDestination(myLambda), filter);
```

Alternatively, short-hand factories are available as `Bucket.fromBucketName` and
`Bucket.fromBucketArn`, which will derive all bucket attributes from the bucket
name or ARN respectively:

```ts
const byName = s3.Bucket.fromBucketName(this, 'BucketByName', 'amzn-s3-demo-bucket');
const byArn = s3.Bucket.fromBucketArn(this, 'BucketByArn', 'arn:aws:s3:::amzn-s3-demo-bucket');
```

The bucket's region defaults to the current stack's region, but can also be explicitly set in cases where one of the bucket's
regional properties needs to contain the correct values.

```ts
const myCrossRegionBucket = s3.Bucket.fromBucketAttributes(this, 'CrossRegionImport', {
  bucketArn: 'arn:aws:s3:::amzn-s3-demo-bucket',
  region: 'us-east-1',
});
// myCrossRegionBucket.bucketRegionalDomainName === 'amzn-s3-demo-bucket.s3.us-east-1.amazonaws.com'
```

## Bucket Notifications

The Amazon S3 notification feature enables you to receive notifications when
certain events happen in your bucket as described under [S3 Bucket
Notifications] of the S3 Developer Guide.

To subscribe for bucket notifications, use the `bucket.addEventNotification` method. The
`bucket.addObjectCreatedNotification` and `bucket.addObjectRemovedNotification` can also be used for
these common use cases.

The following example will subscribe an SNS topic to be notified of all `s3:ObjectCreated:*` events:

```ts
const bucket = new s3.Bucket(this, 'MyBucket');
const topic = new sns.Topic(this, 'MyTopic');
bucket.addEventNotification(s3.EventType.OBJECT_CREATED, new s3n.SnsDestination(topic));
```

This call will also ensure that the topic policy can accept notifications for
this specific bucket.

Supported S3 notification targets are exposed by the `aws-cdk-lib/aws-s3-notifications` package.

It is also possible to specify S3 object key filters when subscribing. The
following example will notify `myQueue` when objects prefixed with `foo/` and
have the `.jpg` suffix are removed from the bucket.

```ts
declare const myQueue: sqs.Queue;
const bucket = new s3.Bucket(this, 'MyBucket');
bucket.addEventNotification(s3.EventType.OBJECT_REMOVED, new s3n.SqsDestination(myQueue), {
  prefix: 'foo/',
  suffix: '.jpg',
});
```

Adding notifications on existing buckets:

```ts
declare const topic: sns.Topic;
const bucket = s3.Bucket.fromBucketAttributes(this, 'ImportedBucket', {
  bucketArn: 'arn:aws:s3:::amzn-s3-demo-bucket',
});
bucket.addEventNotification(s3.EventType.OBJECT_CREATED, new s3n.SnsDestination(topic));
```

If you do not want for S3 to validate permissions of Amazon SQS, Amazon SNS, and Lambda destinations you can use the `notificationsSkipDestinationValidation` flag:

```ts
declare const myQueue: sqs.Queue;
const bucket = new s3.Bucket(this, 'MyBucket', {
  notificationsSkipDestinationValidation: true,
});
bucket.addEventNotification(s3.EventType.OBJECT_REMOVED, new s3n.SqsDestination(myQueue));
```

When you add an event notification to a bucket, a custom resource is created to
manage the notifications. By default, a new role is created for the Lambda
function that implements this feature. If you want to use your own role instead,
you should provide it in the `Bucket` constructor:

```ts
declare const myRole: iam.IRole;
const bucket = new s3.Bucket(this, 'MyBucket', {
  notificationsHandlerRole: myRole,
});
```

Whatever role you provide, the CDK will try to modify it by adding the
permissions from `AWSLambdaBasicExecutionRole` (an AWS managed policy) as well
as the permissions `s3:PutBucketNotification` and `s3:GetBucketNotification`.
If you’re passing an imported role, and you don’t want this to happen, configure
it to be immutable:

```ts
const importedRole = iam.Role.fromRoleArn(this, 'role', 'arn:aws:iam::123456789012:role/RoleName', {
  mutable: false,
});
```

> If you provide an imported immutable role, make sure that it has at least all
> the permissions mentioned above. Otherwise, the deployment will fail!

[s3 bucket notifications]: https://docs.aws.amazon.com/AmazonS3/latest/dev/NotificationHowTo.html

### EventBridge notifications

Amazon S3 can send events to Amazon EventBridge whenever certain events happen in your bucket.
Unlike other destinations, you don't need to select which event types you want to deliver.

The following example will enable EventBridge notifications:

```ts
const bucket = new s3.Bucket(this, 'MyEventBridgeBucket', {
  eventBridgeEnabled: true,
});
```

[s3 eventbridge notifications]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/EventBridge.html

## Block Public Access

Use `blockPublicAccess` to specify [block public access settings] on the bucket.

Enable all block public access settings:

```ts
const bucket = new s3.Bucket(this, 'MyBlockedBucket', {
  blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
});
```

Block and ignore public ACLs (other options remain unblocked):

```ts
const bucket = new s3.Bucket(this, 'MyBlockedBucket', {
  blockPublicAccess: s3.BlockPublicAccess.BLOCK_ACLS_ONLY,
});
```

Alternatively, specify the settings manually (unspecified options will remain blocked):

```ts
const bucket = new s3.Bucket(this, 'MyBlockedBucket', {
  blockPublicAccess: new s3.BlockPublicAccess({ blockPublicPolicy: false }),
});
```

When `blockPublicPolicy` is set to `true`, `grantPublicRead()` throws an error.

[block public access settings]: https://docs.aws.amazon.com/AmazonS3/latest/dev/access-control-block-public-access.html

## Public Read Access

Use `publicReadAccess` to allow public read access to the bucket.

Note that to enable `publicReadAccess`, make sure both bucket-level and account-level block public access control is disabled.

Bucket-level block public access control can be configured through `blockPublicAccess` property. Account-level block public
access control can be configured on AWS Console -> S3 -> Block Public Access settings for this account (Navigation Panel).
```ts
const bucket = new s3.Bucket(this, 'Bucket', {
  publicReadAccess: true,
  blockPublicAccess: {
    blockPublicPolicy: false,
    blockPublicAcls: false,
    ignorePublicAcls: false,
    restrictPublicBuckets: false,
  },
});
```

## Logging configuration

Use `serverAccessLogsBucket` to describe where server access logs are to be stored.

```ts
const accessLogsBucket = new s3.Bucket(this, 'AccessLogsBucket');

const bucket = new s3.Bucket(this, 'MyBucket', {
  serverAccessLogsBucket: accessLogsBucket,
});
```

It's also possible to specify a prefix for Amazon S3 to assign to all log object keys.

```ts
const accessLogsBucket = new s3.Bucket(this, 'AccessLogsBucket');

const bucket = new s3.Bucket(this, 'MyBucket', {
  serverAccessLogsBucket: accessLogsBucket,
  serverAccessLogsPrefix: 'logs',
});
```

You have two options for the log object key format.
`Non-date-based partitioning` is the default log object key format and appears as follows:

```txt
[DestinationPrefix][YYYY]-[MM]-[DD]-[hh]-[mm]-[ss]-[UniqueString]
```

```ts
const accessLogsBucket = new s3.Bucket(this, 'AccessLogsBucket');

const bucket = new s3.Bucket(this, 'MyBucket', {
  serverAccessLogsBucket: accessLogsBucket,
  serverAccessLogsPrefix: 'logs',
  // You can use a simple prefix with `TargetObjectKeyFormat.simplePrefix()`, but it is the same even if you do not specify `targetObjectKeyFormat` property.
  targetObjectKeyFormat: s3.TargetObjectKeyFormat.simplePrefix(),
});
```

Another option is `Date-based partitioning`.
If you choose this format, you can select either the event time or the delivery time of the log file as the date source used in the log format.
This format appears as follows:

```txt
[DestinationPrefix][SourceAccountId]/[SourceRegion]/[SourceBucket]/[YYYY]/[MM]/[DD]/[YYYY]-[MM]-[DD]-[hh]-[mm]-[ss]-[UniqueString]
```

```ts
const accessLogsBucket = new s3.Bucket(this, 'AccessLogsBucket');

const bucket = new s3.Bucket(this, 'MyBucket', {
  serverAccessLogsBucket: accessLogsBucket,
  serverAccessLogsPrefix: 'logs',
  targetObjectKeyFormat: s3.TargetObjectKeyFormat.partitionedPrefix(s3.PartitionDateSource.EVENT_TIME),
});
```

### Allowing access log delivery using a Bucket Policy (recommended)

When possible, it is recommended to use a bucket policy to grant access instead of
using ACLs. When the `@aws-cdk/aws-s3:serverAccessLogsUseBucketPolicy` feature flag
is enabled, this is done by default for server access logs. If S3 Server Access Logs
are the only logs delivered to your bucket (or if all other services logging to the
bucket support using bucket policy instead of ACLs), you can set object ownership
to [bucket owner enforced](#bucket-owner-enforced-recommended), as is recommended.

```ts
const accessLogsBucket = new s3.Bucket(this, 'AccessLogsBucket', {
  objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
});

const bucket = new s3.Bucket(this, 'MyBucket', {
  serverAccessLogsBucket: accessLogsBucket,
  serverAccessLogsPrefix: 'logs',
});
```

The above code will create a new bucket policy if none exists or update the
existing bucket policy to allow access log delivery.

However, there could be an edge case if the `accessLogsBucket` also defines a bucket
policy resource using the L1 Construct. Although the mixing of L1 and L2 Constructs is not
recommended, there are no mechanisms in place to prevent users from doing this at the moment.

```ts
const bucketName = "amzn-s3-demo-bucket";
const accessLogsBucket = new s3.Bucket(this, 'AccessLogsBucket', {
  objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
  bucketName,
});

// Creating a bucket policy using L1
const bucketPolicy = new s3.CfnBucketPolicy(this, "BucketPolicy", {
  bucket: bucketName,
  policyDocument: {
    Statement: [
      {
        Action: 's3:*',
        Effect: 'Deny',
        Principal: {
          AWS: '*',
        },
        Resource: [
          accessLogsBucket.bucketArn,
          `${accessLogsBucket.bucketArn}/*`
        ],
      },
    ],
    Version: '2012-10-17',
  },
});

// 'serverAccessLogsBucket' will create a new L2 bucket policy
// to allow log delivery and overwrite the L1 bucket policy.
const bucket = new s3.Bucket(this, 'MyBucket', {
  serverAccessLogsBucket: accessLogsBucket,
  serverAccessLogsPrefix: 'logs',
});
```

The above example uses the L2 Bucket Construct with the L1 CfnBucketPolicy Construct. However,
when `serverAccessLogsBucket` is set, a new L2 Bucket Policy resource will be created
which overwrites the permissions defined in the L1 Bucket Policy causing unintended
behaviours.

As noted above, we highly discourage the mixed usage of L1 and L2 Constructs. The recommended
approach would to define the bucket policy using `addToResourcePolicy` method.

```ts
const accessLogsBucket = new s3.Bucket(this, 'AccessLogsBucket', {
  objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
});

accessLogsBucket.addToResourcePolicy(
  new iam.PolicyStatement({
    actions: ['s3:*'],
    resources: [accessLogsBucket.bucketArn, accessLogsBucket.arnForObjects('*')],
    principals: [new iam.AnyPrincipal()],
  })
)

const bucket = new s3.Bucket(this, 'MyBucket', {
  serverAccessLogsBucket: accessLogsBucket,
  serverAccessLogsPrefix: 'logs',
});
```

Alternatively, users can use the L2 Bucket Policy Construct
`BucketPolicy.fromCfnBucketPolicy` to wrap around `CfnBucketPolicy` Construct. This will allow the subsequent bucket policy generated by `serverAccessLogsBucket` usage to append to the existing bucket policy instead of overwriting.

```ts
const bucketName = "amzn-s3-demo-bucket";
const accessLogsBucket = new s3.Bucket(this, 'AccessLogsBucket', {
  objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
  bucketName,
});

const bucketPolicy = new s3.CfnBucketPolicy(this, "BucketPolicy", {
  bucket: bucketName,
  policyDocument: {
    Statement: [
      {
        Action: 's3:*',
        Effect: 'Deny',
        Principal: {
          AWS: '*',
        },
        Resource: [
          accessLogsBucket.bucketArn,
          `${accessLogsBucket.bucketArn}/*`
        ],
      },
    ],
    Version: '2012-10-17',
  },
});

// Wrap L1 Construct with L2 Bucket Policy Construct. Subsequent
// generated bucket policy to allow access log delivery would append
// to the current policy.
s3.BucketPolicy.fromCfnBucketPolicy(bucketPolicy);

const bucket = new s3.Bucket(this, 'MyBucket', {
  serverAccessLogsBucket: accessLogsBucket,
  serverAccessLogsPrefix: 'logs',
});
```

## S3 Inventory

An [inventory](https://docs.aws.amazon.com/AmazonS3/latest/dev/storage-inventory.html) contains a list of the objects in the source bucket and metadata for each object. The inventory lists are stored in the destination bucket as a CSV file compressed with GZIP, as an Apache optimized row columnar (ORC) file compressed with ZLIB, or as an Apache Parquet (Parquet) file compressed with Snappy.

You can configure multiple inventory lists for a bucket. You can configure what object metadata to include in the inventory, whether to list all object versions or only current versions, where to store the inventory list file output, and whether to generate the inventory on a daily or weekly basis.

```ts
const inventoryBucket = new s3.Bucket(this, 'InventoryBucket');

const dataBucket = new s3.Bucket(this, 'DataBucket', {
  inventories: [
    {
      frequency: s3.InventoryFrequency.DAILY,
      includeObjectVersions: s3.InventoryObjectVersion.CURRENT,
      destination: {
        bucket: inventoryBucket,
      },
    },
    {
      frequency: s3.InventoryFrequency.WEEKLY,
      includeObjectVersions: s3.InventoryObjectVersion.ALL,
      destination: {
        bucket: inventoryBucket,
        prefix: 'with-all-versions',
      },
    },
  ],
});
```

If the destination bucket is created as part of the same CDK application, the necessary permissions will be automatically added to the bucket policy.
However, if you use an imported bucket (i.e `Bucket.fromXXX()`), you'll have to make sure it contains the following policy document:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InventoryAndAnalyticsExamplePolicy",
      "Effect": "Allow",
      "Principal": { "Service": "s3.amazonaws.com" },
      "Action": "s3:PutObject",
      "Resource": ["arn:aws:s3:::amzn-s3-demo-destination-bucket/*"]
    }
  ]
}
```

## Website redirection

You can use the two following properties to specify the bucket [redirection policy]. Please note that these methods cannot both be applied to the same bucket.

[redirection policy]: https://docs.aws.amazon.com/AmazonS3/latest/dev/how-to-page-redirect.html#advanced-conditional-redirects

### Static redirection

You can statically redirect a to a given Bucket URL or any other host name with `websiteRedirect`:

```ts
const bucket = new s3.Bucket(this, 'MyRedirectedBucket', {
  websiteRedirect: { hostName: 'www.example.com' },
});
```

### Routing rules

Alternatively, you can also define multiple `websiteRoutingRules`, to define complex, conditional redirections:

```ts
const bucket = new s3.Bucket(this, 'MyRedirectedBucket', {
  websiteRoutingRules: [
    {
      hostName: 'www.example.com',
      httpRedirectCode: '302',
      protocol: s3.RedirectProtocol.HTTPS,
      replaceKey: s3.ReplaceKey.prefixWith('test/'),
      condition: {
        httpErrorCodeReturnedEquals: '200',
        keyPrefixEquals: 'prefix',
      },
    },
  ],
});
```

## Filling the bucket as part of deployment

To put files into a bucket as part of a deployment (for example, to host a
website), see the `aws-cdk-lib/aws-s3-deployment` package, which provides a
resource that can do just that.

## The URL for objects

S3 provides two types of URLs for accessing objects via HTTP(S). Path-Style and
[Virtual Hosted-Style](https://docs.aws.amazon.com/AmazonS3/latest/dev/VirtualHosting.html)
URL. Path-Style is a classic way and will be
[deprecated](https://aws.amazon.com/jp/blogs/aws/amazon-s3-path-deprecation-plan-the-rest-of-the-story).
We recommend to use Virtual Hosted-Style URL for newly made bucket.

You can generate both of them.

```ts
const bucket = new s3.Bucket(this, 'MyBucket');
bucket.urlForObject('objectname'); // Path-Style URL
bucket.virtualHostedUrlForObject('objectname'); // Virtual Hosted-Style URL
bucket.virtualHostedUrlForObject('objectname', { regional: false }); // Virtual Hosted-Style URL but non-regional
```

## Object Ownership

You can use one of following properties to specify the bucket [object Ownership].

[object ownership]: https://docs.aws.amazon.com/AmazonS3/latest/dev/about-object-ownership.html

### Object writer

The Uploading account will own the object.

```ts
new s3.Bucket(this, 'MyBucket', {
  objectOwnership: s3.ObjectOwnership.OBJECT_WRITER,
});
```

### Bucket owner preferred

The bucket owner will own the object if the object is uploaded with the bucket-owner-full-control canned ACL. Without this setting and canned ACL, the object is uploaded and remains owned by the uploading account.

```ts
new s3.Bucket(this, 'MyBucket', {
  objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_PREFERRED,
});
```

### Bucket owner enforced (recommended)

ACLs are disabled, and the bucket owner automatically owns and has full control
over every object in the bucket. ACLs no longer affect permissions to data in the
S3 bucket. The bucket uses policies to define access control.

```ts
new s3.Bucket(this, 'MyBucket', {
  objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
});
```

Some services may not not support log delivery to buckets that have object ownership
set to bucket owner enforced, such as
[S3 buckets using ACLs](#allowing-access-log-delivery-using-a-bucket-policy-recommended)
or [CloudFront Distributions][cloudfront s3 bucket].

[cloudfront s3 bucket]: https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/AccessLogs.html#AccessLogsBucketAndFileOwnership

## Bucket deletion

When a bucket is removed from a stack (or the stack is deleted), the S3
bucket will be removed according to its removal policy (which by default will
simply orphan the bucket and leave it in your AWS account). If the removal
policy is set to `RemovalPolicy.DESTROY`, the bucket will be deleted as long
as it does not contain any objects.

To override this and force all objects to get deleted during bucket deletion,
enable the`autoDeleteObjects` option.

When `autoDeleteObjects` is enabled, `s3:PutBucketPolicy` is added to the bucket policy. This is done to allow the custom resource this feature is built on to add a deny policy for `s3:PutObject` to the bucket policy when a delete stack event occurs. Adding this deny policy prevents new objects from being written to the bucket. Doing this prevents race conditions with external bucket writers during the deletion process.

```ts
const bucket = new s3.Bucket(this, 'MyTempFileBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
  autoDeleteObjects: true,
});
```

**Warning** if you have deployed a bucket with `autoDeleteObjects: true`,
switching this to `false` in a CDK version _before_ `1.126.0` will lead to
all objects in the bucket being deleted. Be sure to update your bucket resources
by deploying with CDK version `1.126.0` or later **before** switching this value to `false`.

## Transfer Acceleration

[Transfer Acceleration](https://docs.aws.amazon.com/AmazonS3/latest/userguide/transfer-acceleration.html) can be configured to enable fast, easy, and secure transfers of files over long distances:

```ts
const bucket = new s3.Bucket(this, 'MyBucket', {
  transferAcceleration: true,
});
```

To access the bucket that is enabled for Transfer Acceleration, you must use a special endpoint. The URL can be generated using method `transferAccelerationUrlForObject`:

```ts
const bucket = new s3.Bucket(this, 'MyBucket', {
  transferAcceleration: true,
});
bucket.transferAccelerationUrlForObject('objectname');
```

## Intelligent Tiering

[Intelligent Tiering](https://docs.aws.amazon.com/AmazonS3/latest/userguide/intelligent-tiering.html) can be configured to automatically move files to glacier:

```ts
new s3.Bucket(this, 'MyBucket', {
  intelligentTieringConfigurations: [
    {
      name: 'foo',
      prefix: 'folder/name',
      archiveAccessTierTime: Duration.days(90),
      deepArchiveAccessTierTime: Duration.days(180),
      tags: [{ key: 'tagname', value: 'tagvalue' }],
    },
  ],
});
```

## Lifecycle Rule

[Managing lifecycle](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html) can be configured transition or expiration actions.

```ts
const bucket = new s3.Bucket(this, 'MyBucket', {
  lifecycleRules: [
    {
      abortIncompleteMultipartUploadAfter: Duration.minutes(30),
      enabled: false,
      expiration: Duration.days(30),
      expirationDate: new Date(),
      expiredObjectDeleteMarker: false,
      id: 'id',
      noncurrentVersionExpiration: Duration.days(30),

      // the properties below are optional
      noncurrentVersionsToRetain: 123,
      noncurrentVersionTransitions: [
        {
          storageClass: s3.StorageClass.GLACIER,
          transitionAfter: Duration.days(30),

          // the properties below are optional
          noncurrentVersionsToRetain: 123,
        },
      ],
      objectSizeGreaterThan: 500,
      prefix: 'prefix',
      objectSizeLessThan: 10000,
      transitions: [
        {
          storageClass: s3.StorageClass.GLACIER,

          // exactly one of transitionAfter or transitionDate must be specified
          transitionAfter: Duration.days(30),
          // transitionDate: new Date(), // cannot specify both
        },
      ],
    },
  ],
});
```

To indicate which default minimum object size behavior is applied to the lifecycle configuration, use the
`transitionDefaultMinimumObjectSize` property.

The default value of the property before September 2024 is `TransitionDefaultMinimumObjectSize.VARIES_BY_STORAGE_CLASS`
that allows objects smaller than 128 KB to be transitioned only to the S3 Glacier and S3 Glacier Deep Archive storage classes,
otherwise `TransitionDefaultMinimumObjectSize.ALL_STORAGE_CLASSES_128_K` that prevents objects smaller than 128 KB from being
transitioned to any storage class.

To customize the minimum object size for any transition you
can add a filter that specifies a custom `objectSizeGreaterThan` or `objectSizeLessThan` for `lifecycleRules`
property. Custom filters always take precedence over the default transition behavior.

```ts
new s3.Bucket(this, 'MyBucket', {
  transitionDefaultMinimumObjectSize: s3.TransitionDefaultMinimumObjectSize.VARIES_BY_STORAGE_CLASS,
  lifecycleRules: [
    {
      transitions: [{
        storageClass: s3.StorageClass.DEEP_ARCHIVE,
        transitionAfter: Duration.days(30),
      }],
    },
    {
      objectSizeLessThan: 300000,
      objectSizeGreaterThan: 200000,
      transitions: [{
        storageClass: s3.StorageClass.ONE_ZONE_INFREQUENT_ACCESS,
        transitionAfter: Duration.days(30),
      }],
    },
  ],
});
```

## Object Lock Configuration

[Object Lock](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock-overview.html)
can be configured to enable a write-once-read-many model for an S3 bucket. Object Lock must be
configured when a bucket is created; if a bucket is created without Object Lock, it cannot be
enabled later via the CDK.

Object Lock can be enabled on an S3 bucket by specifying:

```ts
const bucket = new s3.Bucket(this, 'MyBucket', {
  objectLockEnabled: true,
});
```

Usually, it is desired to not just enable Object Lock for a bucket but to also configure a
[retention mode](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock-overview.html#object-lock-retention-modes)
and a [retention period](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock-overview.html#object-lock-retention-periods).
These can be specified by providing `objectLockDefaultRetention`:

```ts
// Configure for governance mode with a duration of 7 years
new s3.Bucket(this, 'Bucket1', {
  objectLockDefaultRetention: s3.ObjectLockRetention.governance(Duration.days(7 * 365)),
});

// Configure for compliance mode with a duration of 1 year
new s3.Bucket(this, 'Bucket2', {
  objectLockDefaultRetention: s3.ObjectLockRetention.compliance(Duration.days(365)),
});
```

## Replicating Objects

You can use [replicating objects](https://docs.aws.amazon.com/AmazonS3/latest/userguide/replication.html) to enable automatic, asynchronous copying of objects across Amazon S3 buckets.
Buckets that are configured for object replication can be owned by the same AWS account or by different accounts.
You can replicate objects to a single destination bucket or to multiple destination buckets.
The destination buckets can be in different AWS Regions or within the same Region as the source bucket.

To replicate objects to a destination bucket, you can specify the `replicationRules` property:

```ts
declare const destinationBucket1: s3.IBucket;
declare const destinationBucket2: s3.IBucket;
declare const replicationRole: iam.IRole;
declare const encryptionKey: kms.IKey;
declare const destinationEncryptionKey: kms.IKey;

const sourceBucket = new s3.Bucket(this, 'SourceBucket', {
  // Versioning must be enabled on both the source and destination bucket
  versioned: true,
  // Optional. Specify the KMS key to use for encrypts objects in the source bucket.
  encryptionKey,
  // Optional. If not specified, a new role will be created.
  replicationRole,
  replicationRules: [
    {
      // The destination bucket for the replication rule.
      destination: destinationBucket1,
      // The priority of the rule.
      // Amazon S3 will attempt to replicate objects according to all replication rules.
      // However, if there are two or more rules with the same destination bucket, then objects will be replicated according to the rule with the highest priority.
      // The higher the number, the higher the priority.
      // It is essential to specify priority explicitly when the replication configuration has multiple rules.
      priority: 1,
    },
    {
      destination: destinationBucket2,
      priority: 2,
      // Whether to specify S3 Replication Time Control (S3 RTC).
      // S3 RTC replicates most objects that you upload to Amazon S3 in seconds,
      // and 99.99 percent of those objects within specified time.
      replicationTimeControl: s3.ReplicationTimeValue.FIFTEEN_MINUTES,
      // Whether to enable replication metrics about S3 RTC.
      // If set, metrics will be output to indicate whether replication by S3 RTC took longer than the configured time.
      metrics: s3.ReplicationTimeValue.FIFTEEN_MINUTES,
      // The kms key to use for the destination bucket.
      kmsKey: destinationEncryptionKey,
      // The storage class to use for the destination bucket.
      storageClass: s3.StorageClass.INFREQUENT_ACCESS,
      // Whether to replicate objects with SSE-KMS encryption.
      sseKmsEncryptedObjects: false,
      // Whether to replicate modifications on replicas.
      replicaModifications: true,
      // Whether to replicate delete markers.
      // This property cannot be enabled if the replication rule has a tag filter.
      deleteMarkerReplication: false,
      // The ID of the rule.
      id: 'full-settings-rule',
      // The object filter for the rule.
      filter: {
        // The prefix filter for the rule.
        prefix: 'prefix',
        // The tag filter for the rule.
        tags: [
          {
            key: 'tagKey',
            value: 'tagValue',
          },
        ],
      }
    },
  ],
});

// Grant permissions to the replication role.
// This method is not required if you choose to use an auto-generated replication role or manually grant permissions.
sourceBucket.grantReplicationPermission(replicationRole, {
  // Optional. Specify the KMS key to use for decrypting objects in the source bucket.
  sourceDecryptionKey: encryptionKey,
  destinations: [
    { bucket: destinationBucket1 },
    { bucket: destinationBucket2, encryptionKey: destinationEncryptionKey },
  ],
  // The 'encryptionKey' property within the 'destinations' array is optional.
  // If not specified for a destination bucket, this method assumes that
  // given destination bucket is not encrypted.
});
```

### Cross Account Replication

You can also set a destination bucket from a different account as the replication destination.

Cross-account replication requires a bucket policy on the destination bucket. If the destination
bucket is managed by CDK, use `addReplicationPolicy()` to add the required permissions.

```ts
// A bucket defined in a CDK stack in the destination account.
declare const destinationBucket: s3.Bucket;
// The account ID that owns the source bucket.
declare const sourceAccountId: string;

const sourceBucket = new s3.Bucket(this, 'SourceBucket', {
  versioned: true,
  replicationRules: [
    {
      destination: destinationBucket,
      priority: 1,
      // Change replica ownership to the account that owns the destination bucket.
      accessControlTransition: true,
    },
  ],
});

// Add the required permissions to the destination bucket policy.
if (sourceBucket.replicationRoleArn) {
  destinationBucket.addReplicationPolicy(
    sourceBucket.replicationRoleArn,
    true,
    sourceAccountId,
  );
}
```

The account ID passed to `addReplicationPolicy()` is the **source** bucket owner's account ID. CDK
uses the destination bucket construct's account for the replication configuration.

If the destination bucket uses `ObjectOwnership.BUCKET_OWNER_ENFORCED`, the destination bucket owner
already owns all replicated objects and `accessControlTransition` is not required. The ownership
override shown above is useful for ACL-enabled destination buckets.

When importing an existing cross-account destination into the source stack, use
`Bucket.fromBucketAttributes()` and set its `account` attribute to the destination bucket owner's
account ID. `Bucket.fromBucketArn()` and `Bucket.fromBucketName()` assume that the bucket belongs to
the same account as the scope where it is imported. Referenced buckets cannot have their bucket
policy modified by CDK, so configure the
[required bucket policy](https://docs.aws.amazon.com/AmazonS3/latest/userguide/replication-walkthrough-2.html)
separately in the destination account.

In a cross-account scenario, you can use a KMS key to encrypt object replicas. The KMS key owner
must grant the source bucket owner permission to use the key. AWS managed keys do not allow
cross-account use and therefore cannot be used for cross-account replication.

For more information, see
[Changing the replica owner](https://docs.aws.amazon.com/AmazonS3/latest/userguide/replication-change-owner.html).

## Mixins

S3 provides several [mixins](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib-readme.html#mixins) that can be applied to L1 and L2 constructs.

### BucketAutoDeleteObjects

Automatically deletes all objects from a bucket when the bucket is removed from the stack or when the stack is deleted. Requires the bucket's removal policy to be set to `DESTROY`:

```ts
new s3.CfnBucket(this, 'Bucket')
  .with(new s3.mixins.BucketAutoDeleteObjects());
```

### BucketVersioning

Enables or suspends versioning on an S3 bucket:

```ts
new s3.CfnBucket(this, 'Bucket')
  .with(new s3.mixins.BucketVersioning());
```

### BucketBlockPublicAccess

Blocks public access on an S3 bucket. Defaults to blocking all public access:

```ts
new s3.CfnBucket(this, 'Bucket')
  .with(new s3.mixins.BucketBlockPublicAccess());
```

### BucketPolicyStatements

Adds IAM policy statements to a bucket policy:

```ts
new s3.CfnBucketPolicy(this, 'Policy', {
  bucket: new s3.CfnBucket(this, 'Bucket').ref,
  policyDocument: new iam.PolicyDocument(),
}).with(new s3.mixins.BucketPolicyStatements([
  new iam.PolicyStatement({
    actions: ['s3:GetObject'],
    resources: ['*'],
    principals: [new iam.AnyPrincipal()],
  }),
]));
```
