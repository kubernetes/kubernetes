# AWS S3 Deployment Construct Library

This library allows populating an S3 bucket with the contents of .zip files
from other S3 buckets or from local disk.

The following example defines a publicly accessible S3 bucket with web hosting
enabled and populates it from a local directory on disk.

```ts
const websiteBucket = new s3.Bucket(this, 'WebsiteBucket', {
  websiteIndexDocument: 'index.html',
  publicReadAccess: true,
});

new s3deploy.BucketDeployment(this, 'DeployWebsite', {
  sources: [s3deploy.Source.asset('./website-dist')],
  destinationBucket: websiteBucket,
  destinationKeyPrefix: 'web/static', // optional prefix in destination bucket
});
```

This is what happens under the hood:

1. When this stack is deployed (either via `cdk deploy` or via CI/CD), the
   contents of the local `website-dist` directory will be archived and uploaded
   to an intermediary assets bucket (the `StagingBucket` of the CDK bootstrap stack).
   If there is more than one source, they will be individually uploaded.
2. The `BucketDeployment` construct synthesizes a Lambda-backed custom CloudFormation resource
   of type `Custom::CDKBucketDeployment` into the template. The source bucket/key
   is set to point to the assets bucket.
3. The custom resource invokes its associated Lambda function, which downloads the .zip archive,
   extracts it and issues `aws s3 sync --delete` against the destination bucket (in this case
   `websiteBucket`). If there is more than one source, the sources will be
   downloaded and merged pre-deployment at this step.

If you are referencing the filled bucket in another construct that depends on
the files already be there, be sure to use `deployment.deployedBucket`. This
will ensure the bucket deployment has finished before the resource that uses
the bucket is created:

```ts
declare const websiteBucket: s3.Bucket;

const deployment = new s3deploy.BucketDeployment(this, 'DeployWebsite', {
  sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
  destinationBucket: websiteBucket,
});

new ConstructThatReadsFromTheBucket(this, 'Consumer', {
  // Use 'deployment.deployedBucket' instead of 'websiteBucket' here
  bucket: deployment.deployedBucket,
});
```

It is also possible to add additional sources using the `addSource` method.

```ts
declare const websiteBucket: s3.IBucket;

const deployment = new s3deploy.BucketDeployment(this, 'DeployWebsite', {
  sources: [s3deploy.Source.asset('./website-dist')],
  destinationBucket: websiteBucket,
  destinationKeyPrefix: 'web/static', // optional prefix in destination bucket
});

deployment.addSource(s3deploy.Source.asset('./another-asset'));
```

For the Lambda function to download object(s) from the source bucket, besides the obvious
`s3:GetObject*` permissions, the Lambda's execution role needs the `kms:Decrypt` and `kms:DescribeKey`
permissions on the KMS key that is used to encrypt the bucket. By default, when the source bucket is
encrypted with the S3 managed key of the account, these permissions are granted by the key's
resource-based policy, so they do not need to be on the Lambda's execution role policy explicitly.
However, if the encryption key is not the s3 managed one, its resource-based policy is quite likely
to NOT grant such KMS permissions. In this situation, the Lambda execution will fail with an error
message like below:

```txt
download failed: ...
An error occurred (AccessDenied) when calling the GetObject operation:
User: *** is not authorized to perform: kms:Decrypt on the resource associated with this ciphertext
because no identity-based policy allows the kms:Decrypt action
```

When this happens, users can use the public `handlerRole` property of `BucketDeployment` to manually
add the KMS permissions:

```ts
declare const destinationBucket: s3.Bucket;

const deployment = new s3deploy.BucketDeployment(this, 'DeployFiles', {
  sources: [s3deploy.Source.asset(path.join(__dirname, 'source-files'))],
  destinationBucket,
});

deployment.handlerRole.addToPolicy(
  new iam.PolicyStatement({
    actions: ['kms:Decrypt', 'kms:DescribeKey'],
    effect: iam.Effect.ALLOW,
    resources: ['<encryption key ARN>'],
  }),
);
```

The situation above could arise from the following scenarios:

- User created a customer managed KMS key and passed its ID to the `cdk bootstrap` command via
  the `--bootstrap-kms-key-id` CLI option.
  The [default key policy](https://docs.aws.amazon.com/kms/latest/developerguide/key-policy-default.html#key-policy-default-allow-root-enable-iam)
  alone is not sufficient to grant the Lambda KMS permissions.

- A corporation uses its own custom CDK bootstrap process, which encrypts the CDK `StagingBucket`
  by a KMS key from a management account of the corporation's AWS Organization. In this cross-account
  access scenario, the KMS permissions must be explicitly present in the Lambda's execution role policy.

- One of the sources for the `BucketDeployment` comes from the `Source.bucket` static method, which
  points to a bucket whose encryption key is not the S3 managed one, and the resource-based policy
  of the encryption key is not sufficient to grant the Lambda `kms:Decrypt` and `kms:DescribeKey`
  permissions.

## Supported sources

The following source types are supported for bucket deployments:

- Local .zip file: `s3deploy.Source.asset('/path/to/local/file.zip')`
- Local directory: `s3deploy.Source.asset('/path/to/local/directory')`
- Another bucket: `s3deploy.Source.bucket(bucket, zipObjectKey)`
- String data: `s3deploy.Source.data('object-key.txt', 'hello, world!')`
  (supports [deploy-time values](#data-with-deploy-time-values))
- JSON data: `s3deploy.Source.jsonData('object-key.json', { json: 'object' })`
  (supports [deploy-time values](#data-with-deploy-time-values))
- YAML data: `s3deploy.Source.yamlData('object-key.yaml', { yaml: 'object' })`
  (supports [deploy-time values](#data-with-deploy-time-values))

To create a source from a single file, you can pass `AssetOptions` to exclude
all but a single file:

- Single file: `s3deploy.Source.asset('/path/to/local/directory', { exclude: ['**', '!onlyThisFile.txt'] })`

**IMPORTANT** The `aws-s3-deployment` module is only intended to be used with
zip files from trusted sources. Directories bundled by the CDK CLI (by using
`Source.asset()` on a directory) are safe. If you are using `Source.asset()` or
`Source.bucket()` to reference an existing zip file, make sure you trust the
file you are referencing. Zips from untrusted sources might be able to execute
arbitrary code in the Lambda Function used by this module, and use its permissions
to read or write unexpected files in the S3 bucket.

## Retain on Delete

By default, the contents of the destination bucket will **not** be deleted when the
`BucketDeployment` resource is removed from the stack or when the destination is
changed. You can use the option `retainOnDelete: false` to disable this behavior,
in which case the contents will be deleted.

Configuring this has a few implications you should be aware of:

- **Logical ID Changes**

  Changing the logical ID of the `BucketDeployment` construct, without changing the destination
  (for example due to refactoring, or intentional ID change) **will result in the deletion of the objects**.
  This is because CloudFormation will first create the new resource, which will have no affect,
  followed by a deletion of the old resource, which will cause a deletion of the objects,
  since the destination hasn't changed, and `retainOnDelete` is `false`.

- **Destination Changes**

  When the destination bucket or prefix is changed, all files in the previous destination will **first** be
  deleted and then uploaded to the new destination location. This could have availability implications
  on your users.

### General Recommendations

#### Shared Bucket

If the destination bucket **is not** dedicated to the specific `BucketDeployment` construct (i.e shared by other entities),
we recommend to always configure the `destinationKeyPrefix` property. This will prevent the deployment from
accidentally deleting data that wasn't uploaded by it.

#### Dedicated Bucket

If the destination bucket **is** dedicated, it might be reasonable to skip the prefix configuration,
in which case, we recommend to remove `retainOnDelete: false`, and instead, configure the
[`autoDeleteObjects`](https://docs.aws.amazon.com/cdk/api/latest/docs/aws-s3-readme.html#bucket-deletion)
property on the destination bucket. This will avoid the logical ID problem mentioned above.

## Prune

By default, files in the destination bucket that don't exist in the source will be deleted
when the `BucketDeployment` resource is created or updated. You can use the option `prune: false` to disable
this behavior, in which case the files will not be deleted.

```ts
declare const destinationBucket: s3.Bucket;
new s3deploy.BucketDeployment(this, 'DeployMeWithoutDeletingFilesOnDestination', {
  sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
  destinationBucket,
  prune: false,
});
```

This option also enables you to
multiple bucket deployments for the same destination bucket & prefix,
each with its own characteristics. For example, you can set different cache-control headers
based on file extensions:

```ts
declare const destinationBucket: s3.Bucket;
new s3deploy.BucketDeployment(this, 'BucketDeployment', {
  sources: [s3deploy.Source.asset('./website', { exclude: ['index.html'] })],
  destinationBucket,
  cacheControl: [
    s3deploy.CacheControl.maxAge(Duration.days(365)),
    s3deploy.CacheControl.immutable(),
  ],
  prune: false,
});

new s3deploy.BucketDeployment(this, 'HTMLBucketDeployment', {
  sources: [s3deploy.Source.asset('./website', { exclude: ['*', '!index.html'] })],
  destinationBucket,
  cacheControl: [
    s3deploy.CacheControl.maxAge(Duration.seconds(0)),
  ],
  prune: false,
});
```

## Exclude and Include Filters

There are two points at which filters are evaluated in a deployment: asset bundling and the actual deployment. If you simply want to exclude files in the asset bundling process, you should leverage the `exclude` property of `AssetOptions` when defining your source:

```ts
declare const destinationBucket: s3.Bucket;
new s3deploy.BucketDeployment(this, 'HTMLBucketDeployment', {
  sources: [s3deploy.Source.asset('./website', { exclude: ['*', '!index.html'] })],
  destinationBucket,
});
```

If you want to specify filters to be used in the deployment process, you can use the `exclude` and `include` filters on `BucketDeployment`.  If excluded, these files will not be deployed to the destination bucket. In addition, if the file already exists in the destination bucket, it will not be deleted if you are using the `prune` option:

```ts
declare const destinationBucket: s3.Bucket;
new s3deploy.BucketDeployment(this, 'DeployButExcludeSpecificFiles', {
  sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
  destinationBucket,
  exclude: ['*.txt'],
});
```

These filters follow the same format that is used for the AWS CLI.  See the CLI documentation for information on [Using Include and Exclude Filters](https://docs.aws.amazon.com/cli/latest/reference/s3/index.html#use-of-exclude-and-include-filters).

## Objects metadata

You can specify metadata to be set on all the objects in your deployment.
There are 2 types of metadata in S3: system-defined metadata and user-defined metadata.
System-defined metadata have a special purpose, for example cache-control defines how long to keep an object cached.
User-defined metadata are not used by S3 and keys always begin with `x-amz-meta-` (this prefix is added automatically).

System defined metadata keys include the following:

- cache-control (`--cache-control` in `aws s3 sync`)
- content-disposition (`--content-disposition` in `aws s3 sync`)
- content-encoding (`--content-encoding` in `aws s3 sync`)
- content-language (`--content-language` in `aws s3 sync`)
- content-type (`--content-type` in `aws s3 sync`)
- expires (`--expires` in `aws s3 sync`)
- x-amz-storage-class (`--storage-class` in `aws s3 sync`)
- x-amz-website-redirect-location (`--website-redirect` in `aws s3 sync`)
- x-amz-server-side-encryption (`--sse` in `aws s3 sync`)
- x-amz-server-side-encryption-aws-kms-key-id (`--sse-kms-key-id` in `aws s3 sync`)
- x-amz-server-side-encryption-customer-algorithm (`--sse-c-copy-source` in `aws s3 sync`)
- x-amz-acl (`--acl` in `aws s3 sync`)

You can find more information about system defined metadata keys in
[S3 PutObject documentation](https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObject.html)
and [`aws s3 sync` documentation](https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html).

```ts
const websiteBucket = new s3.Bucket(this, 'WebsiteBucket', {
  websiteIndexDocument: 'index.html',
  publicReadAccess: true,
});

new s3deploy.BucketDeployment(this, 'DeployWebsite', {
  sources: [s3deploy.Source.asset('./website-dist')],
  destinationBucket: websiteBucket,
  destinationKeyPrefix: 'web/static', // optional prefix in destination bucket
  metadata: { A: "1", b: "2" }, // user-defined metadata

  // system-defined metadata
  contentType: "text/html",
  contentLanguage: "en",
  storageClass: s3deploy.StorageClass.INTELLIGENT_TIERING,
  serverSideEncryption: s3deploy.ServerSideEncryption.AES_256,
  cacheControl: [
    s3deploy.CacheControl.setPublic(),
    s3deploy.CacheControl.maxAge(Duration.hours(1)),
  ],
  accessControl: s3.BucketAccessControl.BUCKET_OWNER_FULL_CONTROL,
});
```

## CloudFront Invalidation

You can provide a CloudFront distribution and optional paths to invalidate after the bucket deployment finishes.

```ts
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';

const bucket = new s3.Bucket(this, 'Destination');

// Handles buckets whether or not they are configured for website hosting.
const distribution = new cloudfront.Distribution(this, 'Distribution', {
  defaultBehavior: { origin: new origins.S3Origin(bucket) },
});

new s3deploy.BucketDeployment(this, 'DeployWithInvalidation', {
  sources: [s3deploy.Source.asset('./website-dist')],
  destinationBucket: bucket,
  distribution,
  distributionPaths: ['/images/*.png'],
});
```

By default, the deployment will wait for invalidation to succeed to complete. This will poll Cloudfront for a maximum of 13 minutes to check for a successful invalidation. The drawback to this is that the deployment will fail if invalidation fails or if it takes longer than 13 minutes. As a workaround, there is the option `waitForDistributionInvalidation`, which can be set to false to skip waiting for the invalidation, but this can be risky as invalidation errors will not be reported.

```ts
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';

declare const bucket: s3.IBucket;
declare const distribution: cloudfront.IDistribution;

new s3deploy.BucketDeployment(this, 'DeployWithInvalidation', {
  sources: [s3deploy.Source.asset('./website-dist')],
  destinationBucket: bucket,
  distribution,
  distributionPaths: ['/images/*.png'],
  // Invalidate cache but don't wait or verify that invalidation has completed successfully.
  waitForDistributionInvalidation: false
});
```

## Signed Content Payloads

By default, deployment uses streaming uploads which set the `x-amz-content-sha256`
request header to `UNSIGNED-PAYLOAD` (matching default behavior of the AWS CLI tool).
In cases where bucket policy restrictions require signed content payloads, you can enable
generation of a signed `x-amz-content-sha256` request header with `signContent: true`.

```ts
declare const bucket: s3.IBucket;

new s3deploy.BucketDeployment(this, 'DeployWithSignedPayloads', {
  sources: [s3deploy.Source.asset('./website-dist')],
  destinationBucket: bucket,
  signContent: true,
});
```

## Size Limits

The default memory limit for the deployment resource is 1024MiB. If you need to
copy larger files, you can use the `memoryLimit` configuration to increase the
size of the AWS Lambda resource handler.

The default ephemeral storage size for the deployment resource is 512MiB. If you
need to upload larger files, you may hit this limit. You can use the
`ephemeralStorageSize` configuration to increase the storage size of the AWS Lambda
resource handler.

> NOTE: a new AWS Lambda handler will be created in your stack for each combination
> of memory and storage size.

## JSON-Aware Source Processing

When using `Source.jsonData` with CDK Tokens (references to construct properties), you may need to enable the escaping option. This is particularly important when the referenced properties might contain special characters that require proper JSON escaping (like double quotes, line breaks, etc.).

```ts
declare const bucket: s3.Bucket;
declare const param: ssm.StringParameter;

// Example with a secret value that contains double quotes
const deployment = new s3deploy.BucketDeployment(this, 'JsonDeployment', {
  sources: [
    s3deploy.Source.jsonData('config.json', {
      api_endpoint: 'https://api.example.com',
      secretValue: param.stringValue, // value with double quotes
      config: {
        enabled: true,
        features: ['feature1', 'feature2']
      }
    },
    // Enable escaping at deployment time
    { escape: true },
    )
  ],
  destinationBucket: bucket,
});
```

## EFS Support

If your workflow needs more disk space than default (512 MB) disk space, you may attach an EFS storage to underlying
lambda function. To Enable EFS support set `efs` and `vpc` props for BucketDeployment.

Check sample usage below.
Please note that creating VPC inline may cause stack deletion failures. It is shown as below for simplicity.
To avoid such condition, keep your network infra (VPC) in a separate stack and pass as props.

```ts
declare const destinationBucket: s3.Bucket;
declare const vpc: ec2.Vpc;

new s3deploy.BucketDeployment(this, 'DeployMeWithEfsStorage', {
  sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
  destinationBucket,
  destinationKeyPrefix: 'efs/',
  useEfs: true,
  vpc,
  retainOnDelete: false,
});
```

## Data with deploy-time values

The content passed to `Source.data()`, `Source.jsonData()`, or `Source.yamlData()` can include
references that will get resolved only during deployment. Only a subset of CloudFormation functions
are supported however, namely: Ref, Fn::GetAtt, Fn::Join, and Fn::Select (Fn::Split may be nested under Fn::Select).

For example:

```ts
import * as sns from 'aws-cdk-lib/aws-sns';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';

declare const destinationBucket: s3.Bucket;
declare const topic: sns.Topic;
declare const tg: elbv2.ApplicationTargetGroup;

const appConfig = {
  topic_arn: topic.topicArn,
  base_url: 'https://my-endpoint',
  lb_name: tg.firstLoadBalancerFullName,
};

new s3deploy.BucketDeployment(this, 'BucketDeployment', {
  sources: [s3deploy.Source.jsonData('config.json', appConfig)],
  destinationBucket,
});
```

The value in `topic.topicArn` is a deploy-time value. It only gets resolved
during deployment by placing a marker in the generated source file and
substituting it when its deployed to the destination with the actual value.

### Substitutions from Templated Files

The `DeployTimeSubstitutedFile` construct allows you to specify substitutions
to make from placeholders in a local file which will be resolved during deployment. This
is especially useful in situations like creating an API from a spec file, where users might
want to reference other CDK resources they have created.

The syntax for template variables is `{{ variableName }}` in your local file. Then, you would
specify the substitutions in CDK like this:

```ts
import * as lambda from 'aws-cdk-lib/aws-lambda';

declare const myLambdaFunction: lambda.Function;
declare const destinationBucket: s3.Bucket;
//(Optional) if provided, the resulting processed file would be uploaded to the destinationBucket under the destinationKey name.
declare const destinationKey: string;
declare const role: iam.Role;

new s3deploy.DeployTimeSubstitutedFile(this, 'MyFile', {
  source: 'my-file.yaml',
  destinationKey: destinationKey,
  destinationBucket: destinationBucket,
  substitutions: {
    variableName: myLambdaFunction.functionName,
  },
  role: role,
});
```

Nested variables, like `{{ {{ foo }} }}` or `{{ foo {{ bar }} }}`, are not supported by this
construct. In the first case of a single variable being is double nested `{{ {{ foo }} }}`, only
the `{{ foo }}` would be replaced by the substitution, and the extra brackets would remain in the file.
In the second case of two nexted variables `{{ foo {{ bar }} }}`, only the `{{ bar }}` would be replaced
in the file.

## Keep Files Zipped

By default, files are zipped, then extracted into the destination bucket.

You can use the option `extract: false` to disable this behavior, in which case, files will remain in a zip file when deployed to S3. To reference the object keys, or filenames, which will be deployed to the bucket, you can use the `objectKeys` getter on the bucket deployment.

```ts
import * as cdk from 'aws-cdk-lib';

declare const destinationBucket: s3.Bucket;

const myBucketDeployment = new s3deploy.BucketDeployment(this, 'DeployMeWithoutExtractingFilesOnDestination', {
  sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
  destinationBucket,
  extract: false,
});

new cdk.CfnOutput(this, 'ObjectKey', {
  value: cdk.Fn.select(0, myBucketDeployment.objectKeys),
});
```

## Controlling the Output of Source Object Keys

By default, the keys of the source objects copied to the destination bucket are returned in the Data property of the custom resource. However, you can disable this behavior by setting the outputObjectKeys property to false. This is particularly useful when the number of objects is too large and might exceed the size limit of the responseData property.

```ts
import * as cdk from 'aws-cdk-lib';

declare const destinationBucket: s3.Bucket;

const myBucketDeployment = new s3deploy.BucketDeployment(this, 'DeployMeWithoutExtractingFilesOnDestination', {
  sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
  destinationBucket,
  outputObjectKeys: false,
});

new cdk.CfnOutput(this, 'ObjectKey', {
  value: cdk.Fn.select(0, myBucketDeployment.objectKeys),
});
```

## Specifying a Custom VPC, Subnets, and Security Groups in BucketDeployment

By default, the AWS CDK BucketDeployment construct runs in a publicly accessible environment. However, for enhanced security and compliance, you may need to deploy your assets from within a VPC while restricting network access through custom subnets and security groups.

### Using a Custom VPC

To deploy assets within a private network, specify the vpc property in BucketDeploymentProps. This ensures that the deployment Lambda function executes within your specified VPC.

```ts
const vpc = ec2.Vpc.fromLookup(this, 'ExistingVPC', { vpcId: 'vpc-12345678' });
const bucket = new s3.Bucket(this, 'MyBucket');

new s3deploy.BucketDeployment(this, 'DeployToS3', {
    destinationBucket: bucket,
    vpc: vpc, 
    sources: [s3deploy.Source.asset('./website')],
});
```

### Specifying Subnets for Deployment

By default, when you specify a VPC, the BucketDeployment function is deployed in the private subnets of that VPC.
However, you can customize the subnet selection using the vpcSubnets property.

```ts
const vpc = ec2.Vpc.fromLookup(this, 'ExistingVPC', { vpcId: 'vpc-12345678' });
const bucket = new s3.Bucket(this, 'MyBucket');

new s3deploy.BucketDeployment(this, 'DeployToS3', {
    destinationBucket: bucket,
    vpc: vpc,
    vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
    sources: [s3deploy.Source.asset('./website')],
});
```

### Defining Custom Security Groups

For enhanced network security, you can now specify custom security groups in BucketDeploymentProps.
This allows fine-grained control over ingress and egress rules for the deployment Lambda function.

```ts
const vpc = ec2.Vpc.fromLookup(this, 'ExistingVPC', { vpcId: 'vpc-12345678' });
const bucket = new s3.Bucket(this, 'MyBucket');

const securityGroup = new ec2.SecurityGroup(this, 'CustomSG', {
    vpc: vpc,
    description: 'Allow HTTPS outbound access',
    allowAllOutbound: false,
});

securityGroup.addEgressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(443), 'Allow HTTPS traffic');

new s3deploy.BucketDeployment(this, 'DeployWithSecurityGroup', {
    destinationBucket: bucket,
    vpc: vpc,
    securityGroups: [securityGroup],
    sources: [s3deploy.Source.asset('./website')],
});
```

## Notes

- This library uses an AWS CloudFormation custom resource which is about 10MiB in
  size. The code of this resource is bundled with this library.
- AWS Lambda execution time is limited to 15min. This limits the amount of data
  which can be deployed into the bucket by this timeout.
- When the `BucketDeployment` is removed from the stack, the contents are retained
  in the destination bucket ([#952](https://github.com/aws/aws-cdk/issues/952)).
- If you are using `s3deploy.Source.bucket()` to take the file source from
  another bucket: the deployed files will only be updated if the key (file name)
  of the file in the source  bucket changes. Mutating the file in place will not
  be good enough: the custom resource will simply not run if the properties don't
  change.
  - If you use assets (`s3deploy.Source.asset()`) you don't need to worry
    about this: the asset system will make sure that if the files have changed,
    the file name is unique and the deployment will run.

## Development

The custom resource is implemented in Python 3.9 in order to be able to leverage
the AWS CLI for "aws s3 sync".
The code is now in the `@aws-cdk/custom-resource-handlers` package under [`lib/aws-s3-deployment/bucket-deployment-handler`](https://github.com/aws/aws-cdk/tree/main/packages/@aws-cdk/custom-resource-handlers/lib/aws-s3-deployment/bucket-deployment-handler/) and
unit tests are under [`test/aws-s3-deployment/bucket-deployment-handler`](https://github.com/aws/aws-cdk/tree/main/packages/@aws-cdk/custom-resource-handlers/test/aws-s3-deployment/bucket-deployment-handler/).

This package requires Python 3.9 during build time in order to create the custom
resource Lambda bundle and test it. It also relies on a few bash scripts, so
might be tricky to build on Windows.

## Roadmap

- [ ] Support "blue/green" deployments ([#954](https://github.com/aws/aws-cdk/issues/954))
