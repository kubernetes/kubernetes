# AWS Lambda Construct Library


This construct library allows you to define AWS Lambda Functions.

```ts
const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
});
```

When deployed, this construct creates or updates an existing
`AWS::Lambda::Function` resource. When updating, AWS CloudFormation calls the
[UpdateFunctionConfiguration](https://docs.aws.amazon.com/lambda/latest/api/API_UpdateFunctionConfiguration.html)
and [UpdateFunctionCode](https://docs.aws.amazon.com/lambda/latest/api/API_UpdateFunctionCode.html)
Lambda APIs under the hood. Because these calls happen sequentially, and
invocations can happen between these calls, your function may encounter errors
in the time between the calls. For example, if you update an existing Lambda
function by removing an environment variable and the code that references that
environment variable in the same CDK deployment, you may see invocation errors
related to a missing environment variable. To work around this, you can invoke
your function against a version or alias by default, rather than the `$LATEST`
version.

To further mitigate these issues, you can ensure consistency between your function code and infrastructure configuration by defining environment variables as a single source of truth in your CDK stack. You can define them in a separate `env.ts` file and reference them in both your handler and CDK configuration. This approach allows you to catch errors at compile time, benefit from improved IDE support, minimize the risk of mismatched configurations, and enhance maintainability.

## Handler Code

The `lambda.Code` class includes static convenience methods for various types of
runtime code.

 * `lambda.Code.fromBucket(bucket, key, objectVersion)` - specify an S3 object
   that contains the archive of your runtime code.
 * `lambda.Code.fromBucketV2(bucket, key, {objectVersion: version, sourceKMSKey: key})` - specify an S3 object
  that contains the archive of your runtime code.

  ```ts

  import { Key } from 'aws-cdk-lib/aws-kms';
  import * as s3 from 'aws-cdk-lib/aws-s3';

  const bucket = new s3.Bucket(this, 'Bucket');
  declare const key: Key;
  
  const options = {
      sourceKMSKey: key,
  };
  const fnBucket = new lambda.Function(this, 'myFunction2', {
      runtime: lambda.Runtime.NODEJS_LATEST,
      handler: 'index.handler',
      code: lambda.Code.fromBucketV2(bucket, 'python-lambda-handler.zip', options),
  });

  ```
 * `lambda.Code.fromInline(code)` - inline the handle code as a string. This is
   limited to supported runtimes.
 * `lambda.Code.fromAsset(path)` - specify a directory or a .zip file in the local
   filesystem which will be zipped and uploaded to S3 before deployment. See also
   [bundling asset code](#bundling-asset-code).
 * `lambda.Code.fromDockerBuild(path, options)` - use the result of a Docker
   build as code. The runtime code is expected to be located at `/asset` in the
   image and will be zipped and uploaded to S3 as an asset.
 * `lambda.Code.fromCustomCommand(output, command, customCommandOptions)` - 
   supply a command that is invoked during cdk synth. That command is meant to direct
   the generated code to output (a zip file or a directory), which is then used as the 
   code for the created AWS Lambda.

The following example shows how to define a Python function and deploy the code
from the local directory `my-lambda-handler` to it:

[Example of Lambda Code from Local Assets](test/integ.assets.lit.ts)

When deploying a stack that contains this code, the directory will be zip
archived and then uploaded to an S3 bucket, then the exact location of the S3
objects will be passed when the stack is deployed.

During synthesis, the CDK expects to find a directory on disk at the asset
directory specified. Note that we are referencing the asset directory relatively
to our CDK project directory. This is especially important when we want to share
this construct through a library. Different programming languages will have
different techniques for bundling resources into libraries.

## Docker Images

Lambda functions allow specifying their handlers within docker images. The docker
image can be an image from ECR or a local asset that the CDK will package and load
into ECR.

The following `DockerImageFunction` construct uses a local folder with a
Dockerfile as the asset that will be used as the function handler.

```ts
new lambda.DockerImageFunction(this, 'AssetFunction', {
  code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, 'docker-handler')),
});
```

You can also specify an image that already exists in ECR as the function handler.

```ts
import * as ecr from 'aws-cdk-lib/aws-ecr';
const repo = new ecr.Repository(this, 'Repository');

new lambda.DockerImageFunction(this, 'ECRFunction', {
  code: lambda.DockerImageCode.fromEcr(repo),
});
```

The props for these docker image resources allow overriding the image's `CMD`, `ENTRYPOINT`, and `WORKDIR`
configurations as well as choosing a specific tag or digest. See their docs for more information.

To deploy a `DockerImageFunction` on Lambda `arm64` architecture, specify `Architecture.ARM_64` in `architecture`.
This will bundle docker image assets for `arm64` architecture with `--platform linux/arm64` even if build within an `x86_64` host.

With that being said, if you are bundling `DockerImageFunction` for Lambda `amd64` architecture from a `arm64` machine like a Macbook with `arm64` CPU, you would 
need to specify `architecture: lambda.Architecture.X86_64` as well. This ensures the `--platform` argument is passed to the image assets
bundling process so you can bundle up `X86_64` images from the `arm64` machine.

```ts
new lambda.DockerImageFunction(this, 'AssetFunction', {
  code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, 'docker-arm64-handler')),
  architecture: lambda.Architecture.ARM_64,
});
```

## Execution Role

Lambda functions assume an IAM role during execution. In CDK by default, Lambda
functions will use an autogenerated Role if one is not provided.

The autogenerated Role is automatically given permissions to execute the Lambda
function. To reference the autogenerated Role:

```ts
const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
});

const role = fn.role; // the Role
```

You can also provide your own IAM role. Provided IAM roles will not automatically
be given permissions to execute the Lambda function. To provide a role and grant
it appropriate permissions:

```ts
const myRole = new iam.Role(this, 'My Role', {
  assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
});

const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  role: myRole, // user-provided role
});

myRole.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName("service-role/AWSLambdaBasicExecutionRole"));
myRole.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName("service-role/AWSLambdaVPCAccessExecutionRole")); // only required if your function lives in a VPC
```

## Function Timeout

AWS Lambda functions have a default timeout of 3 seconds, but this can be increased
up to 15 minutes. The timeout is available as a property of `Function` so that
you can reference it elsewhere in your stack. For instance, you could use it to create
a CloudWatch alarm to report when your function timed out:

```ts
import * as cdk from 'aws-cdk-lib';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';

const fn = new lambda.Function(this, 'MyFunction', {
   runtime: lambda.Runtime.NODEJS_LATEST,
   handler: 'index.handler',
   code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
   timeout: Duration.minutes(5),
});

if (fn.timeout) {
   new cloudwatch.Alarm(this, `MyAlarm`, {
      metric: fn.metricDuration().with({
         statistic: 'Maximum',
      }),
      evaluationPeriods: 1,
      datapointsToAlarm: 1,
      threshold: fn.timeout.toMilliseconds(),
      treatMissingData: cloudwatch.TreatMissingData.IGNORE,
      alarmName: 'My Lambda Timeout',
   });
}
```

## Advanced Logging

You can have more control over your function logs, by specifying the log format
(Json or plain text), the system log level, the application log level, as well
as choosing the log group:

```ts
import { ILogGroup } from 'aws-cdk-lib/aws-logs';

declare const logGroup: ILogGroup;

new lambda.Function(this, 'Lambda', {
  code: new lambda.InlineCode('foo'),
  handler: 'index.handler',
  runtime: lambda.Runtime.NODEJS_LATEST,
  loggingFormat: lambda.LoggingFormat.JSON,
  systemLogLevelV2: lambda.SystemLogLevel.INFO,
  applicationLogLevelV2: lambda.ApplicationLogLevel.INFO,
  logGroup: logGroup,
});
```

To use `applicationLogLevelV2` and/or `systemLogLevelV2` you must set `loggingFormat` to `LoggingFormat.JSON`.

## Customizing Log Group Creation

### Log Group Creation Methods

| Method | Description | Tag Propagation | Prop Injection | Aspects | Notes |
|--------|-------------|-----------------|----------------|---------|-------|
| **logRetention prop** | Legacy approach using Custom Resource | False | False | False | Does not support TPA |
| **logGroup prop** | Explicitly supplied by user in CDK app | True | True | True | Full support for TPA |
| **Lazy creation** | Lambda service creates logGroup on first invocation | False | False | False | Occurs when both logRetention and logGroup are undefined and USE_CDK_MANAGED_LAMBDA_LOGGROUP is false |
| **USE_CDK_MANAGED_LAMBDA_LOGGROUP** | CDK Lambda function construct creates log group with default props | True | True | True | Feature flag must be enabled |

*TPA: Tag propagation, Prop Injection, Aspects*

#### Order of precedence 
```text
                       Highest Precedence
                             |
             +---------------+---------------+
             |                               |
  +-------------------------+      +------------------------+
  |   logRetention is set   |      |     logGroup is set    |
  +-----------+-------------+      +----------+-------------+
              |                               |
              v                               v
      Create LogGroup via            Use the provided LogGroup
  Custom Resource (retention       instance (CDK-managed, 
  managed, logGroup disallowed)    logRetention disallowed)
              |                               |
              +---------------+---------------+
                              |
                              v
          +-----------------------------------------------+
          |         Feature flag enabled:                 |
          | aws-cdk:aws-lambda:useCdkManagedLogGroup      |
          +------------------ +---------------------------+
                              |
                              v
              Create LogGroup at synth time 
          (CDK-managed, default settings for logGroup)
                              |
                              v
                  +---------------------------+
                  | Default (no config set)   |
                  +------------+--------------+
                              |
                              v
     Lambda service creates log group on first invocation runtime
            (CDK does not manage the log group resource)
```
### Tag Propagation

Refer section `Log Group Creation Methods` to find out which modes support tag propagation. 
As an example, adding the following line in your cdk app will also propagate to the logGroup. 
```
const fn = new lambda.Function(this, 'MyFunctionWithFFTrue', {
  runtime: lambda.Runtime.NODEJS_20_X,
  handler: 'handler.main',
  code: lambda.Code.fromAsset('lambda'),
});
cdk.Tags.of(fn).add('env', 'dev'); // the tag is also added to the log group
```

### Log removal policy

When using the deprecated `logRetention` property for creating a LogGroup, you can configure log removal policy:
```ts
import * as logs from 'aws-cdk-lib/aws-logs';

const fn = new lambda.Function(this, 'MyFunctionWithFFTrue', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'handler.main',
  code: lambda.Code.fromAsset('lambda'),
  logRetention: logs.RetentionDays.INFINITE,
  logRemovalPolicy: RemovalPolicy.RETAIN,
});
```

## Resource-based Policies

AWS Lambda supports resource-based policies for controlling access to Lambda
functions and layers on a per-resource basis. In particular, this allows you to
give permission to AWS services, AWS Organizations, or other AWS accounts to
modify and invoke your functions.

### Grant function access to AWS services

```ts
// Grant permissions to a service
declare const fn: lambda.Function;
const principal = new iam.ServicePrincipal('my-service');

fn.grantInvoke(principal);

// Equivalent to:
fn.addPermission('my-service Invocation', {
  principal: principal,
});
```

You can also restrict permissions given to AWS services by providing
a source account or ARN (representing the account and identifier of the resource
that accesses the function or layer).

**Important**: 
> By default `fn.grantInvoke()` grants permission to the principal to invoke any version of the function, including all past ones. If you only want the principal to be granted permission to invoke the latest version or the unqualified Lambda ARN, use `grantInvokeLatestVersion(grantee)`.

```ts
declare const fn: lambda.Function;
const principal = new iam.ServicePrincipal('my-service');
// Grant invoke only to latest version and unqualified lambda arn
fn.grantInvokeLatestVersion(principal);

```

If you want to grant access for invoking a specific version of Lambda function, you can use `fn.grantInvokeVersion(grantee, version)`

```ts
declare const fn: lambda.Function;
const principal = new iam.ServicePrincipal('my-service');
declare const version: lambda.IVersion;
// Grant invoke only to the specific version
fn.grantInvokeVersion(principal, version);
```

For more information, see
[Granting function access to AWS services](https://docs.aws.amazon.com/lambda/latest/dg/access-control-resource-based.html#permissions-resource-serviceinvoke)
in the AWS Lambda Developer Guide.

### Grant function access to an AWS Organization

```ts
// Grant permissions to an entire AWS organization
declare const fn: lambda.Function;
const org = new iam.OrganizationPrincipal('o-xxxxxxxxxx');

fn.grantInvoke(org);
```

In the above example, the `principal` will be `*` and all users in the
organization `o-xxxxxxxxxx` will get function invocation permissions.

You can restrict permissions given to the organization by specifying an
AWS account or role as the `principal`:

```ts
// Grant permission to an account ONLY IF they are part of the organization
declare const fn: lambda.Function;
const account = new iam.AccountPrincipal('123456789012');

fn.grantInvoke(account.inOrganization('o-xxxxxxxxxx'));
```

For more information, see
[Granting function access to an organization](https://docs.aws.amazon.com/lambda/latest/dg/access-control-resource-based.html#permissions-resource-xorginvoke)
in the AWS Lambda Developer Guide.

### Grant function access to other AWS accounts

```ts
// Grant permission to other AWS account
declare const fn: lambda.Function;
const account = new iam.AccountPrincipal('123456789012');

fn.grantInvoke(account);
```

For more information, see
[Granting function access to other accounts](https://docs.aws.amazon.com/lambda/latest/dg/access-control-resource-based.html#permissions-resource-xaccountinvoke)
in the AWS Lambda Developer Guide.

### Grant function access to unowned principals

Providing an unowned principal (such as account principals, generic ARN
principals, service principals, and principals in other accounts) to a call to
`fn.grantInvoke` will result in a resource-based policy being created. If the
principal in question has conditions limiting the source account or ARN of the
operation (see above), these conditions will be automatically added to the
resource policy.

```ts
declare const fn: lambda.Function;
const servicePrincipal = new iam.ServicePrincipal('my-service');
const sourceArn = 'arn:aws:s3:::amzn-s3-demo-bucket';
const sourceAccount = '111122223333';
const servicePrincipalWithConditions = servicePrincipal.withConditions({
  ArnLike: {
    'aws:SourceArn': sourceArn,
  },
  StringEquals: {
    'aws:SourceAccount': sourceAccount,
  },
});

fn.grantInvoke(servicePrincipalWithConditions);
```

### Grant function access to a CompositePrincipal

To grant invoke permissions to a `CompositePrincipal` use the `grantInvokeCompositePrincipal` method:

```ts
declare const fn: lambda.Function;
const compositePrincipal = new iam.CompositePrincipal(
  new iam.OrganizationPrincipal('o-zzzzzzzzzz'),
  new iam.ServicePrincipal('apigateway.amazonaws.com'),
);

fn.grantInvokeCompositePrincipal(compositePrincipal);
```

## Versions

You can use
[versions](https://docs.aws.amazon.com/lambda/latest/dg/configuration-versions.html)
to manage the deployment of your AWS Lambda functions. For example, you can
publish a new version of a function for beta testing without affecting users of
the stable production version.

The function version includes the following information:

* The function code and all associated dependencies.
* The Lambda runtime that executes the function.
* All of the function settings, including the environment variables.
* A unique Amazon Resource Name (ARN) to identify this version of the function.

You could create a version to your lambda function using the `Version` construct.

```ts
declare const fn: lambda.Function;
const version = new lambda.Version(this, 'MyVersion', {
  lambda: fn,
});
```

The major caveat to know here is that a function version must always point to a
specific 'version' of the function. When the function is modified, the version
will continue to point to the 'then version' of the function.

One way to ensure that the `lambda.Version` always points to the latest version
of your `lambda.Function` is to set an environment variable which changes at
least as often as your code does. This makes sure the function always has the
latest code. For instance -

```ts
const codeVersion = "stringOrMethodToGetCodeVersion";
const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  environment: {
    'CodeVersionString': codeVersion,
  },
});
```

The `fn.latestVersion` property returns a `lambda.IVersion` which represents
the `$LATEST` pseudo-version.

However, most AWS services require a specific AWS Lambda version,
and won't allow you to use `$LATEST`. Therefore, you would normally want
to use `lambda.currentVersion`.

The `fn.currentVersion` property can be used to obtain a `lambda.Version`
resource that represents the AWS Lambda function defined in your application.
Any change to your function's code or configuration will result in the creation
of a new version resource. You can specify options for this version through the
`currentVersionOptions` property.

NOTE: The `currentVersion` property is only supported when your AWS Lambda function
uses either `lambda.Code.fromAsset` or `lambda.Code.fromInline`. Other types
of code providers (such as `lambda.Code.fromBucket`) require that you define a
`lambda.Version` resource directly since the CDK is unable to determine if
their contents had changed.

### `currentVersion`: Updated hashing logic

To produce a new lambda version each time the lambda function is modified, the
`currentVersion` property under the hood, computes a new logical id based on the
properties of the function. This informs CloudFormation that a new
`AWS::Lambda::Version` resource should be created pointing to the updated Lambda
function.

However, a bug was introduced in this calculation that caused the logical id to
change when it was not required (ex: when the Function's `Tags` property, or
when the `DependsOn` clause was modified). This caused the deployment to fail
since the Lambda service does not allow creating duplicate versions.

This has been fixed in the AWS CDK but *existing* users need to opt-in via a
[feature flag]. Users who have run `cdk init` since this fix will be opted in,
by default.

Otherwise, you will need to enable the [feature flag]
`@aws-cdk/aws-lambda:recognizeVersionProps`. Since CloudFormation does not
allow duplicate versions, you will also need to make some modification to
your function so that a new version can be created. To efficiently and trivially
modify all your lambda functions at once, you can attach the
`FunctionVersionUpgrade` aspect to the stack, which slightly alters the
function description. This aspect is intended for one-time use to upgrade the
version of all your functions at the same time, and can safely be removed after
deploying once.

```ts
const stack = new Stack();
Aspects.of(stack).add(new lambda.FunctionVersionUpgrade(LAMBDA_RECOGNIZE_VERSION_PROPS));
```

When the new logic is in effect, you may rarely come across the following error:
`The following properties are not recognized as version properties`. This will
occur, typically when [property overrides] are used, when a new property
introduced in `AWS::Lambda::Function` is used that CDK is still unaware of.

To overcome this error, use the API `Function.classifyVersionProperty()` to
record whether a new version should be generated when this property is changed.
This can be typically determined by checking whether the property can be
modified using the *[UpdateFunctionConfiguration]* API or not.

### `currentVersion`: Updated hashing logic for layer versions

An additional update to the hashing logic fixes two issues surrounding layers.
Prior to this change, updating the lambda layer version would have no effect on
the function version. Also, the order of lambda layers provided to the function
was unnecessarily baked into the hash.

This has been fixed in the AWS CDK starting with version 2.27. If you ran
`cdk init` with an earlier version, you will need to opt-in via a [feature flag].
If you run `cdk init` with v2.27 or later, this fix will be opted in, by default.

Existing users will need to enable the [feature flag]
`@aws-cdk/aws-lambda:recognizeLayerVersion`. Since CloudFormation does not
allow duplicate versions, they will also need to make some modification to
their function so that a new version can be created. To efficiently and trivially
modify all your lambda functions at once, users can attach the
`FunctionVersionUpgrade` aspect to the stack, which slightly alters the
function description. This aspect is intended for one-time use to upgrade the
version of all your functions at the same time, and can safely be removed after
deploying once.

```ts
const stack = new Stack();
Aspects.of(stack).add(new lambda.FunctionVersionUpgrade(LAMBDA_RECOGNIZE_LAYER_VERSION));
```

[feature flag]: https://docs.aws.amazon.com/cdk/latest/guide/featureflags.html
[property overrides]: https://docs.aws.amazon.com/cdk/latest/guide/cfn_layer.html#cfn_layer_raw
[UpdateFunctionConfiguration]: https://docs.aws.amazon.com/lambda/latest/dg/API_UpdateFunctionConfiguration.html

## Aliases

You can define one or more
[aliases](https://docs.aws.amazon.com/lambda/latest/dg/configuration-aliases.html)
for your AWS Lambda function. A Lambda alias is like a pointer to a specific
Lambda function version. Users can access the function version using the alias
ARN.

The `version.addAlias()` method can be used to define an AWS Lambda alias that
points to a specific version.

The following example defines an alias named `live` which will always point to a
version that represents the function as defined in your CDK app. When you change
your lambda code or configuration, a new resource will be created. You can
specify options for the current version through the `currentVersionOptions`
property.

```ts
const fn = new lambda.Function(this, 'MyFunction', {
  currentVersionOptions: {
    removalPolicy: RemovalPolicy.RETAIN, // retain old versions
    retryAttempts: 1,                   // async retry attempts
  },
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
});

fn.addAlias('live');
```

## Function URL

A function URL is a dedicated HTTP(S) endpoint for your Lambda function. When you create a function URL, Lambda automatically generates a unique URL endpoint for you. Function URLs can be created for the latest version Lambda Functions, or Function Aliases (but not for Versions).

Function URLs are dual stack-enabled, supporting IPv4 and IPv6, and cross-origin resource sharing (CORS) configuration. After you configure a function URL for your function, you can invoke your function through its HTTP(S) endpoint via a web browser, curl, Postman, or any HTTP client. To invoke a function using IAM authentication your HTTP client must support SigV4 signing.

See the [Invoking Function URLs](https://docs.aws.amazon.com/lambda/latest/dg/urls-invocation.html) section of the AWS Lambda Developer Guide
for more information on the input and output payloads of Functions invoked in this way.

### IAM-authenticated Function URLs

To create a Function URL which can be called by an IAM identity, call `addFunctionUrl()`, followed by `grantInvokeFunctionUrl()`:

```ts
// Can be a Function or an Alias
declare const fn: lambda.Function;
declare const myRole: iam.Role;

const fnUrl = fn.addFunctionUrl();
fnUrl.grantInvokeUrl(myRole);

new CfnOutput(this, 'TheUrl', {
  // The .url attributes will return the unique Function URL
  value: fnUrl.url,
});
```

Calls to this URL need to be signed with SigV4.

### Anonymous Function URLs

To create a Function URL which can be called anonymously, pass `authType: FunctionUrlAuthType.NONE` to `addFunctionUrl()`:

```ts
// Can be a Function or an Alias
declare const fn: lambda.Function;

const fnUrl = fn.addFunctionUrl({
  authType: lambda.FunctionUrlAuthType.NONE,
});

new CfnOutput(this, 'TheUrl', {
  value: fnUrl.url,
});
```

### Important Function URL Permission Update - Oct 2025
Starting Oct 2025, Function URL invocation will require two permissions
- lambda:InvokeFunctionUrl
- lambda:InvokeFunction (New)

CDK has updated `grantInvokeUrl` and `addFunctionUrl` to add both permission above.

If your existing CDK stack uses `grantInvokeUrl` or `addFunctionUrl`, your next deployment will automatically add the `lambda:InvokeFunction` permission without requiring any code changes. This ensures your Function URLs continue working seamlessly. No additional actions are needed.

### CORS configuration for Function URLs

If you want your Function URLs to be invokable from a web page in browser, you
will need to configure cross-origin resource sharing to allow the call (if you do
not do this, your browser will refuse to make the call):

```ts
declare const fn: lambda.Function;

fn.addFunctionUrl({
  authType: lambda.FunctionUrlAuthType.NONE,
  cors: {
    // Allow this to be called from websites on https://example.com.
    // Can also be ['*'] to allow all domain.
    allowedOrigins: ['https://example.com'],

    // More options are possible here, see the documentation for FunctionUrlCorsOptions
  },
});
```

### Invoke Mode for Function URLs

Invoke mode determines how AWS Lambda invokes your function. You can configure the invoke mode when creating a Function URL using the invokeMode property

```ts
declare const fn: lambda.Function;

fn.addFunctionUrl({
  authType: lambda.FunctionUrlAuthType.NONE,
  invokeMode: lambda.InvokeMode.RESPONSE_STREAM,
});
```

If the invokeMode property is not specified, the default BUFFERED mode will be used.

## Layers

The `lambda.LayerVersion` class can be used to define Lambda layers and manage
granting permissions to other AWS accounts or organizations.

[Example of Lambda Layer usage](test/integ.layer-version.lit.ts)

By default, updating a layer creates a new layer version, and CloudFormation will delete the old version as part of the stack update.

Alternatively, a removal policy can be used to retain the old version:

```ts
new lambda.LayerVersion(this, 'MyLayer', {
  removalPolicy: RemovalPolicy.RETAIN,
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
});
```

## Architecture

Lambda functions, by default, run on compute systems that have the 64 bit x86 architecture.

The AWS Lambda service also runs compute on the ARM architecture, which can reduce cost
for some workloads.

A lambda function can be configured to be run on one of these platforms:

```ts
new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  architecture: lambda.Architecture.ARM_64,
});
```

Similarly, lambda layer versions can also be tagged with architectures it is compatible with.

```ts
new lambda.LayerVersion(this, 'MyLayer', {
  removalPolicy: RemovalPolicy.RETAIN,
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  compatibleArchitectures: [lambda.Architecture.X86_64, lambda.Architecture.ARM_64],
});
```

## Capacity Providers

Lambda capacity providers allow you to run Lambda functions on dedicated compute resources instead of the default serverless execution environment. Capacity providers can have multiple functions and function versions associated with them, but a function can be attached to at most one capacity provider.

### Creating a Capacity Provider

To create a capacity provider, you need to specify the VPC configuration and optionally configure permissions, scaling, and instance requirements:

```ts
import * as ec2 from 'aws-cdk-lib/aws-ec2';

// Create a VPC for the capacity provider
const vpc = new ec2.Vpc(this, 'MyVpc');
const securityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', { vpc });

// Create a basic capacity provider
const capacityProvider = new lambda.CapacityProvider(this, 'MyCapacityProvider', {
  capacityProviderName: 'my-capacity-provider',
  subnets: vpc.privateSubnets,
  securityGroups: [securityGroup],
});
```

By default, permissions are granted to the capacity provider to manage instances using the AWS managed policy AWSLambdaManagedEC2ResourceOperator. You can also supply a custom role via the `operatorRole` parameter.

### Configuring Scaling Policies

You can configure target tracking scaling policies to automatically scale your capacity provider based on CPU utilization:

```ts
import * as ec2 from 'aws-cdk-lib/aws-ec2';

const vpc = new ec2.Vpc(this, 'MyVpc');
const securityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', { vpc });

const capacityProvider = new lambda.CapacityProvider(this, 'MyCapacityProvider', {
  subnets: vpc.privateSubnets,
  securityGroups: [securityGroup],
  scalingOptions: lambda.ScalingOptions.manual([
    lambda.TargetTrackingScalingPolicy.cpuUtilization(70),
  ]),
});
```

### Instance Type Configuration

You can control which EC2 instance types the capacity provider can use:

```ts
import * as ec2 from 'aws-cdk-lib/aws-ec2';

const vpc = new ec2.Vpc(this, 'MyVpc');
const securityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', { vpc });

// Allow only specific instance families
const allowCapacityProvider = new lambda.CapacityProvider(this, 'MyCapacityProviderAllowed', {
  subnets: vpc.privateSubnets,
  securityGroups: [securityGroup],
  instanceTypeFilter: lambda.InstanceTypeFilter.allow([
    ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.LARGE),
    ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.XLARGE),
  ]),
});

// Or exclude specific instance types
const excludeCapacityProvider = new lambda.CapacityProvider(this, 'MyCapacityProviderExcluded', {
  subnets: vpc.privateSubnets,
  securityGroups: [securityGroup],
  instanceTypeFilter: lambda.InstanceTypeFilter.exclude([
    ec2.InstanceType.of(ec2.InstanceClass.T2, ec2.InstanceSize.MICRO),
  ]),
});
```

### System Logging

You can configure system logging to monitor capacity provider scaling activity:

```ts
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as logs from 'aws-cdk-lib/aws-logs';

const vpc = new ec2.Vpc(this, 'MyVpc');
const securityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', { vpc });

const capacityProvider = new lambda.CapacityProvider(this, 'MyCapacityProvider', {
  subnets: vpc.privateSubnets,
  securityGroups: [securityGroup],
  logGroup: new logs.LogGroup(this, 'CpLogs', {
    logGroupName: '/aws/lambda/capacity-provider/my-cp',
  }),
  systemLogLevel: lambda.SystemLogLevel.DEBUG,
});
```

### Using a Capacity Provider with Lambda Functions

Once you have a capacity provider, you can configure Lambda functions to use it:

```ts
declare const capacityProvider: lambda.CapacityProvider;

const fn = new lambda.Function(this, 'MyFunction', {
  // Runtime must be equal to or newer than NODEJS_LATEST
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
});

// Associate the function with the capacity provider
capacityProvider.addFunction(fn, {
  perExecutionEnvironmentMaxConcurrency: 10,
  executionEnvironmentMemoryGiBPerVCpu: 4,
});
```

Note that once you use a capacity provider in a function, it cannot be removed, only changed.

#### CapacityProviderFunctionOptions (addFunction method)

| Name | Type | Required | Description |
|------|------|----------|-------------|
| perExecutionEnvironmentMaxConcurrency | number | No | Maximum concurrent invokes per execution environment. |
| executionEnvironmentMemoryGiBPerVCpu | number | No | Memory per VCPU in GiB. |
| publishToLatestPublished | boolean | No | Whether to automatically publish to $LATEST.PUBLISHED version. |
| latestPublishedScalingConfig | LatestPublishedScalingConfig | No | Scaling configuration for $LATEST.PUBLISHED version. |

#### LatestPublishedScalingConfig

| Name | Type | Required | Description |
|------|------|----------|-------------|
| minExecutionEnvironments | number | No | Minimum execution environments for $LATEST.PUBLISHED version. |
| maxExecutionEnvironments | number | No | Maximum execution environments for $LATEST.PUBLISHED version. |

### Capacity Provider Versions

When publishing Lambda versions that use capacity providers, you can configure scaling settings specific to each version.

To publish a permanent version with specific scaling properties, you can use the [`currentVersion`](#currentversion-updated-hashing-logic) property of the function:

```ts
const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  currentVersionOptions: {
    minExecutionEnvironments: 3,
  }
});

const version = fn.latestVersion
```

You can also specify scaling properties for the special `$LATEST.PUBLISHED` version when attaching the function to the capacity provider.

`$LATEST.PUBLISHED` is a version that is automatically published when you deploy changes to your function.

```ts
declare const capacityProvider: lambda.CapacityProvider;
declare const fn: lambda.Function;

capacityProvider.addFunction(fn, {
  latestPublishedScalingConfig: {
    minExecutionEnvironments: 5,
    maxExecutionEnvironments: 25,
  },
});
```

#### VersionOptions (Capacity Provider Related Fields)

| Name | Type | Required | Description |
|------|------|----------|-------------|
| minExecutionEnvironments | number | No | Minimum execution environments to maintain for this version. |
| maxExecutionEnvironments | number | No | Maximum execution environments allowed for this version. |

### Security and Encryption

Capacity providers support encryption using AWS KMS keys:

```ts
import * as kms from 'aws-cdk-lib/aws-kms';
import * as ec2 from 'aws-cdk-lib/aws-ec2';

const vpc = new ec2.Vpc(this, 'MyVpc');
const securityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', { vpc });
const kmsKey = new kms.Key(this, 'MyKey');

const capacityProvider = new lambda.CapacityProvider(this, 'MyCapacityProvider', {
  subnets: vpc.privateSubnets,
  securityGroups: [securityGroup],
  kmsKey: kmsKey,
});
```

### Capacity Provider Configuration Reference

#### CapacityProviderProps

| Name | Type | Required | Description |
|------|------|----------|-------------|
| capacityProviderName | string | No | Name of the capacity provider. Must be unique within the AWS account and region. |
| securityGroups | ISecurityGroup[] | Yes | Security groups to associate with EC2 instances. Up to 5 allowed. |
| subnets | ISubnet[] | Yes | Subnets where the capacity provider can launch EC2 instances. 1-16 subnets supported. |
| operatorRole | IRole | No | IAM role for Lambda to manage the capacity provider. Uses AWS Managed Policy AWSLambdaManagedEC2ResourceOperator by default. |
| architectures | Architecture[] | No | Instruction set architecture for compute instances. |
| instanceTypeFilter | InstanceTypeFilter | No | Filter for allowed or excluded instance types. |
| maxVCpuCount | number | No | Maximum number of EC2 instances for scaling. |
| scalingOptions | ScalingOptions | No | Scaling configuration including policies. |
| kmsKey | IKey | No | KMS key for encrypting capacity provider data. |
| logGroup | ILogGroup | No | CloudWatch log group for capacity provider system logs. |
| systemLogLevel | SystemLogLevel | No | Level of detail for capacity provider system logs (DEBUG, INFO, WARN). |

## Lambda Insights

Lambda functions can be configured to use CloudWatch [Lambda Insights](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Lambda-Insights.html)
which provides low-level runtime metrics for a Lambda functions.

```ts
new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  insightsVersion: lambda.LambdaInsightsVersion.VERSION_1_0_98_0,
});
```

If the version of insights is not yet available in the CDK, you can also provide the ARN directly as so -

```ts
const layerArn = 'arn:aws:lambda:us-east-1:580247275435:layer:LambdaInsightsExtension:14';
new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  insightsVersion: lambda.LambdaInsightsVersion.fromInsightVersionArn(layerArn),
});
```

If you are deploying an ARM_64 Lambda Function, you must specify a
Lambda Insights Version >= `1_0_119_0`.

```ts
new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  architecture: lambda.Architecture.ARM_64,
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  insightsVersion: lambda.LambdaInsightsVersion.VERSION_1_0_119_0,
});
```

### Parameters and Secrets Extension

Lambda functions can be configured to use the Parameters and Secrets Extension. The Parameters and Secrets Extension can be used to retrieve and cache [secrets](https://docs.aws.amazon.com/secretsmanager/latest/userguide/retrieving-secrets_lambda.html) from Secrets Manager or [parameters](https://docs.aws.amazon.com/systems-manager/latest/userguide/ps-integration-lambda-extensions.html) from Parameter Store in Lambda functions without using an SDK.

```ts
import * as sm from 'aws-cdk-lib/aws-secretsmanager';
import * as ssm from 'aws-cdk-lib/aws-ssm';

const secret = new sm.Secret(this, 'Secret');
const parameter = new ssm.StringParameter(this, 'Parameter', {
  parameterName: 'mySsmParameterName',
  stringValue: 'mySsmParameterValue',
});

const paramsAndSecrets = lambda.ParamsAndSecretsLayerVersion.fromVersion(lambda.ParamsAndSecretsVersions.V1_0_103, {
  cacheSize: 500,
  logLevel: lambda.ParamsAndSecretsLogLevel.DEBUG,
});

const lambdaFunction = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  architecture: lambda.Architecture.ARM_64,
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  paramsAndSecrets,
});

secret.grantRead(lambdaFunction);
parameter.grantRead(lambdaFunction);
```

If the version of Parameters and Secrets Extension is not yet available in the CDK, you can also provide the ARN directly as so:

```ts
import * as sm from 'aws-cdk-lib/aws-secretsmanager';
import * as ssm from 'aws-cdk-lib/aws-ssm';

const secret = new sm.Secret(this, 'Secret');
const parameter = new ssm.StringParameter(this, 'Parameter', {
  parameterName: 'mySsmParameterName',
  stringValue: 'mySsmParameterValue',
});

const layerArn = 'arn:aws:lambda:us-east-1:177933569100:layer:AWS-Parameters-and-Secrets-Lambda-Extension:4';
const paramsAndSecrets = lambda.ParamsAndSecretsLayerVersion.fromVersionArn(layerArn, {
  cacheSize: 500,
});

const lambdaFunction = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  architecture: lambda.Architecture.ARM_64,
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  paramsAndSecrets,
});

secret.grantRead(lambdaFunction);
parameter.grantRead(lambdaFunction);
```

## Event Rule Target

You can use an AWS Lambda function as a target for an Amazon CloudWatch event
rule:

```ts
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';

declare const fn: lambda.Function;
const rule = new events.Rule(this, 'Schedule Rule', {
 schedule: events.Schedule.cron({ minute: '0', hour: '4' }),
});
rule.addTarget(new targets.LambdaFunction(fn));
```

## Event Sources

AWS Lambda supports a [variety of event sources](https://docs.aws.amazon.com/lambda/latest/dg/invoking-lambda-function.html).

In most cases, it is possible to trigger a function as a result of an event by
using one of the `add<Event>Notification` methods on the source construct. For
example, the `s3.Bucket` construct has an `onEvent` method which can be used to
trigger a Lambda when an event, such as PutObject occurs on an S3 bucket.

An alternative way to add event sources to a function is to use `function.addEventSource(source)`.
This method accepts an `IEventSource` object. The module __@aws-cdk/aws-lambda-event-sources__
includes classes for the various event sources supported by AWS Lambda.

For example, the following code adds an SQS queue as an event source for a function:

```ts
import * as eventsources from 'aws-cdk-lib/aws-lambda-event-sources';
import * as sqs from 'aws-cdk-lib/aws-sqs';

declare const fn: lambda.Function;
const queue = new sqs.Queue(this, 'Queue');
fn.addEventSource(new eventsources.SqsEventSource(queue));
```

The following code adds an S3 bucket notification as an event source:

```ts
import * as eventsources from 'aws-cdk-lib/aws-lambda-event-sources';
import * as s3 from 'aws-cdk-lib/aws-s3';

declare const fn: lambda.Function;
const bucket = new s3.Bucket(this, 'Bucket');
fn.addEventSource(new eventsources.S3EventSource(bucket, {
  events: [ s3.EventType.OBJECT_CREATED, s3.EventType.OBJECT_REMOVED ],
  filters: [ { prefix: 'subdir/' } ] // optional
}));
```

The following code adds an DynamoDB notification as an event source filtering insert events:

```ts
import * as eventsources from 'aws-cdk-lib/aws-lambda-event-sources';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';

declare const fn: lambda.Function;
const table = new dynamodb.Table(this, 'Table', {
  partitionKey: {
    name: 'id',
    type: dynamodb.AttributeType.STRING,
  },
  stream: dynamodb.StreamViewType.NEW_IMAGE,
});
fn.addEventSource(new eventsources.DynamoEventSource(table, {
  startingPosition: lambda.StartingPosition.LATEST,
  filters: [lambda.FilterCriteria.filter({ eventName: lambda.FilterRule.isEqual('INSERT') })],
}));
```

By default, Lambda will encrypt Filter Criteria using AWS managed keys. But if you want to use a self managed KMS key to encrypt the filters, You can specify the self managed key using the `filterEncryption` property.

```ts
import * as eventsources from 'aws-cdk-lib/aws-lambda-event-sources';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import { Key } from 'aws-cdk-lib/aws-kms';

declare const fn: lambda.Function;
const table = new dynamodb.Table(this, 'Table', {
  partitionKey: {
    name: 'id',
    type: dynamodb.AttributeType.STRING,
  },
  stream: dynamodb.StreamViewType.NEW_IMAGE,
});
// Your self managed KMS key
const myKey = Key.fromKeyArn(
  this,
  'SourceBucketEncryptionKey',
  'arn:aws:kms:us-east-1:123456789012:key/<key-id>',
);

fn.addEventSource(new eventsources.DynamoEventSource(table, {
  startingPosition: lambda.StartingPosition.LATEST,
  filters: [lambda.FilterCriteria.filter({ eventName: lambda.FilterRule.isEqual('INSERT') })],
  filterEncryption: myKey,
}));
```

> Lambda requires allow `kms:Decrypt` on Lambda principal `lambda.amazonaws.com` to use the key for Filter Criteria Encryption. If you create the KMS key in the stack, CDK will automatically add this permission to the Key when you creates eventSourceMapping. However, if you import the key using function like `Key.fromKeyArn` then you need to add the following permission to the KMS key before using it to encrypt Filter Criteria

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "kms:Decrypt",
            "Resource": "*"
        }
    ]
}
```

### Observability

Customers can now opt-in to get enhanced metrics for their event source mapping that capture each stage of processing using the `MetricsConfig` property.

The following code shows how to opt in for the enhanced metrics. 

```ts
import * as eventsources from 'aws-cdk-lib/aws-lambda-event-sources';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';

declare const fn: lambda.Function;
const table = new dynamodb.Table(this, 'Table', {
  partitionKey: {
    name: 'id',
    type: dynamodb.AttributeType.STRING,
  },
  stream: dynamodb.StreamViewType.NEW_IMAGE,
});

fn.addEventSource(new eventsources.DynamoEventSource(table, {
  startingPosition: lambda.StartingPosition.LATEST,
  metricsConfig: {
    metrics: [lambda.MetricType.EVENT_COUNT],
  }
}));
```

See the documentation for the __@aws-cdk/aws-lambda-event-sources__ module for more details.

## Imported Lambdas

When referencing an imported lambda in the CDK, use `fromFunctionArn()` for most use cases:

```ts
const fn = lambda.Function.fromFunctionArn(
  this,
  'Function',
  'arn:aws:lambda:us-east-1:123456789012:function:MyFn',
);
```

The `fromFunctionAttributes()` API is available for more specific use cases:

```ts
const fn = lambda.Function.fromFunctionAttributes(this, 'Function', {
  functionArn: 'arn:aws:lambda:us-east-1:123456789012:function:MyFn',
  // The following are optional properties for specific use cases and should be used with caution:

  // Use Case: imported function is in the same account as the stack. This tells the CDK that it
  // can modify the function's permissions.
  sameEnvironment: true,

  // Use Case: imported function is in a different account and user commits to ensuring that the
  // imported function has the correct permissions outside the CDK.
  skipPermissions: true,
});
```

`Function.fromFunctionArn()` and `Function.fromFunctionAttributes()` will attempt to parse the Function's Region and Account ID from the ARN. `addPermissions` will only work on the `Function` object if the Region and Account ID are deterministically the same as the scope of the Stack the referenced `Function` object is created in.
If the containing Stack is environment-agnostic or the Function ARN is a Token, this comparison will fail, and calls to `Function.addPermission` will do nothing.
If you know Function permissions can safely be added, you can use `Function.fromFunctionName()` instead, or pass `sameEnvironment: true` to `Function.fromFunctionAttributes()`.

```ts
const fn = lambda.Function.fromFunctionName(this, 'Function', 'MyFn');
```

## Lambda with DLQ

A dead-letter queue can be automatically created for a Lambda function by
setting the `deadLetterQueueEnabled: true` configuration. In such case CDK creates
a `sqs.Queue` as `deadLetterQueue`.

```ts
const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromInline('exports.handler = function(event, ctx, cb) { return cb(null, "hi"); }'),
  deadLetterQueueEnabled: true,
});
```

It is also possible to provide a dead-letter queue instead of getting a new queue created:

```ts
import * as sqs from 'aws-cdk-lib/aws-sqs';

const dlq = new sqs.Queue(this, 'DLQ');
const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromInline('exports.handler = function(event, ctx, cb) { return cb(null, "hi"); }'),
  deadLetterQueue: dlq,
});
```

You can also use a `sns.Topic` instead of an `sqs.Queue` as dead-letter queue:

```ts
import * as sns from 'aws-cdk-lib/aws-sns';

const dlt = new sns.Topic(this, 'DLQ');
const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromInline('// your code here'),
  deadLetterTopic: dlt,
});
```

See [the AWS documentation](https://docs.aws.amazon.com/lambda/latest/dg/dlq.html)
to learn more about AWS Lambdas and DLQs.

## Lambda with X-Ray Tracing

```ts
const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromInline('exports.handler = function(event, ctx, cb) { return cb(null, "hi"); }'),
  tracing: lambda.Tracing.ACTIVE,
});
```

See [the AWS documentation](https://docs.aws.amazon.com/lambda/latest/dg/lambda-x-ray.html)
to learn more about AWS Lambda's X-Ray support.

## Lambda with AWS Distro for OpenTelemetry layer

To have automatic integration with XRay without having to add dependencies or change your code, you can use the
[AWS Distro for OpenTelemetry Lambda (ADOT) layer](https://aws-otel.github.io/docs/getting-started/lambda).
Consuming the latest ADOT layer can be done with the following snippet:

```ts
import {
  AdotLambdaExecWrapper,
  AdotLayerVersion,
  AdotLambdaLayerJavaScriptSdkVersion,
} from 'aws-cdk-lib/aws-lambda';

const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromInline('exports.handler = function(event, ctx, cb) { return cb(null, "hi"); }'),
  adotInstrumentation: {
    layerVersion: AdotLayerVersion.fromJavaScriptSdkLayerVersion(AdotLambdaLayerJavaScriptSdkVersion.LATEST),
    execWrapper: AdotLambdaExecWrapper.REGULAR_HANDLER,
  },
});
```

To use a different layer version, use one of the following helper functions for the `layerVersion` prop:

* `AdotLayerVersion.fromJavaScriptSdkLayerVersion`
* `AdotLayerVersion.fromPythonSdkLayerVersion`
* `AdotLayerVersion.fromJavaSdkLayerVersion`
* `AdotLayerVersion.fromJavaAutoInstrumentationSdkLayerVersion`
* `AdotLayerVersion.fromGenericSdkLayerVersion`

Each helper function expects a version value from a corresponding enum-like class as below:

* `AdotLambdaLayerJavaScriptSdkVersion`
* `AdotLambdaLayerPythonSdkVersion`
* `AdotLambdaLayerJavaSdkVersion`
* `AdotLambdaLayerJavaAutoInstrumentationSdkVersion`
* `AdotLambdaLayerGenericSdkVersion`

For more examples, see our [the integration test](test/integ.lambda-adot.ts).

If you want to retrieve the ARN of the ADOT Lambda layer without enabling ADOT in a Lambda function:

```ts
declare const fn: lambda.Function;
const layerArn = lambda.AdotLambdaLayerJavaSdkVersion.V1_19_0.layerArn(fn.stack, fn.architecture);
```

When using the `AdotLambdaLayerPythonSdkVersion` the `AdotLambdaExecWrapper` needs to be `AdotLambdaExecWrapper.INSTRUMENT_HANDLER` as per [AWS Distro for OpenTelemetry Lambda Support For Python](https://aws-otel.github.io/docs/getting-started/lambda/lambda-python)

## Lambda with Profiling

The following code configures the lambda function with CodeGuru profiling. By default, this creates a new CodeGuru
profiling group -

```ts
const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.PYTHON_3_9,
  handler: 'index.handler',
  code: lambda.Code.fromAsset('lambda-handler'),
  profiling: true,
});
```

The `profilingGroup` property can be used to configure an existing CodeGuru profiler group.

CodeGuru profiling is supported for all Java runtimes and Python3.6+ runtimes.

See [the AWS documentation](https://docs.aws.amazon.com/codeguru/latest/profiler-ug/setting-up-lambda.html)
to learn more about AWS Lambda's Profiling support.

## Lambda with Reserved Concurrent Executions

```ts
const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromInline('exports.handler = function(event, ctx, cb) { return cb(null, "hi"); }'),
  reservedConcurrentExecutions: 100,
});
```

https://docs.aws.amazon.com/lambda/latest/dg/invocation-recursion.html

## Lambda with SnapStart

SnapStart is currently supported on Python 3.12, Python 3.13, .NET 8, and Java 11 and later [Java managed runtimes](https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtimes.html). SnapStart does not support provisioned concurrency, Amazon Elastic File System (Amazon EFS), or ephemeral storage greater than 512 MB. After you enable Lambda SnapStart for a particular Lambda function, publishing a new version of the function will trigger an optimization process.

See [the AWS documentation](https://docs.aws.amazon.com/lambda/latest/dg/snapstart.html) to learn more about AWS Lambda SnapStart

```ts
const fn = new lambda.Function(this, 'MyFunction', {
  code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
  runtime: lambda.Runtime.JAVA_11,
  handler: 'example.Handler::handleRequest',
  snapStart: lambda.SnapStartConf.ON_PUBLISHED_VERSIONS,
  });

const version = fn.currentVersion;
```

## Lambda with Durable Configuration

Lambda functions can be configured with durability to enable long-running, stateful executions.

```ts
const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  durableConfig: { executionTimeout: Duration.hours(1), retentionPeriod: Duration.days(30) }, // 1 hour timeout, 30 days retention
});
```
*Note:* If adding the `durableConfig` property to an existing lambda function, a resource replacement will be triggered.
For the resource replacement to succeed, you cannot explicitly set the `functionName` parameter, or else it will
result in CloudFormation deployment errors. Modifying the parameters within `durableConfig` will not trigger a resource replacement.

*Further Note:* Deletion for durable functions will wait for all running executions to complete. CloudFormation will wait up to 1 hour, 
after which the stack update or deletion will timeout. If you have functions that are stuck deleting, please check if there
are any running executions for the function.

## AutoScaling

You can use Application AutoScaling to automatically configure the provisioned concurrency for your functions. AutoScaling can be set to track utilization or be based on a schedule. To configure AutoScaling on a function alias:

```ts
import * as appscaling from 'aws-cdk-lib/aws-applicationautoscaling';

declare const fn: lambda.Function;
const alias = fn.addAlias('prod');

// Create AutoScaling target
const as = alias.addAutoScaling({ maxCapacity: 50 });

// Configure Target Tracking
as.scaleOnUtilization({
  utilizationTarget: 0.5,
});

// Configure Scheduled Scaling
as.scaleOnSchedule('ScaleUpInTheMorning', {
  schedule: appscaling.Schedule.cron({ hour: '8', minute: '0'}),
  minCapacity: 20,
});
```

[Example of Lambda AutoScaling usage](test/integ.autoscaling.lit.ts)

See [the AWS documentation](https://docs.aws.amazon.com/lambda/latest/dg/invocation-scaling.html) on autoscaling lambda functions.

## Log Group

By default, Lambda functions automatically create a log group with the name `/aws/lambda/<function-name>` upon first execution with
log data set to never expire.
This is convenient, but prevents you from changing any of the properties of this auto-created log group using the AWS CDK.
For example you cannot set log retention or assign a data protection policy.

To fully customize the logging behavior of your Lambda function, create a `logs.LogGroup` ahead of time and use the `logGroup` property to instruct the Lambda function to send logs to it.
This way you can use the full features set supported by Amazon CloudWatch Logs.

```ts
import { LogGroup } from 'aws-cdk-lib/aws-logs';

const myLogGroup = new LogGroup(this, 'MyLogGroupWithLogGroupName', {
  logGroupName: 'customLogGroup',
});

new lambda.Function(this, 'Lambda', {
  code: new lambda.InlineCode('foo'),
  handler: 'index.handler',
  runtime: lambda.Runtime.NODEJS_LATEST,
  logGroup: myLogGroup,
});
```

Providing a user-controlled log group was rolled out to commercial regions on 2023-11-16.
If you are deploying to another type of region, please check regional availability first.

## Lambda with Recursive Loop protection

Recursive loop protection is to stop unintended loops. The customers are opted in by default for Lambda to detect and terminate unintended loops between Lambda and other AWS Services.
The property can be assigned two values here, "Allow" and "Terminate". 

The default value is set to "Terminate", which lets the Lambda to detect and terminate the recursive loops. 

When the value is set to "Allow", the customers opt out of recursive loop detection and Lambda does not terminate recursive loops if any. 

See [the AWS documentation](https://docs.aws.amazon.com/lambda/latest/dg/invocation-recursion.html) to learn more about AWS Lambda Recusrive Loop Detection

```ts
const fn = new lambda.Function(this, 'MyFunction', {
  code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
  runtime: lambda.Runtime.JAVA_11,
  handler: 'example.Handler::handleRequest',
  recursiveLoop: lambda.RecursiveLoop.TERMINATE,
  });
```

## Lambda with Tenant Isolation

Lambda functions can be configured with tenant isolation to ensure that different tenants never share the same execution environment. This is useful for SaaS applications where you need to guarantee compute isolation between untrusted tenants while using a single Lambda function.

```ts
const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  tenancyConfig: lambda.TenancyConfig.PER_TENANT,
});
```

**Important considerations:**

* **Immutable configuration**: Tenant isolation can only be configured during function creation and cannot be modified on existing functions.
* **Incompatible features**: The following features are not compatible with tenant isolation and will result in CloudFormation deployment errors:
  * **Provisioned Concurrency**
  * **Function URLs**
  * **SnapStart**
  * **Event Source Mappings** (except API Gateway):
    * ❌ SQS
    * ❌ DynamoDB
    * ❌ Kinesis
    * ❌ MSK
    * ❌ Self-managed Kafka
    * ✅ API Gateway (supported)

CDK validates these restrictions at synthesis time and provides clear error messages when incompatible features are configured.

### Legacy Log Retention

As an alternative to providing a custom, user controlled log group, the legacy `logRetention` property can be used to set a different expiration period.
This feature uses a Custom Resource to change the log retention of the automatically created log group.

By default, CDK uses the AWS SDK retry options when creating a log group. The `logRetentionRetryOptions` property
allows you to customize the maximum number of retries and base backoff duration.

*Note* that a [CloudFormation custom
resource](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-cfn-customresource.html) is added
to the stack that pre-creates the log group as part of the stack deployment, if it already doesn't exist, and sets the
correct log retention period (never expire, by default). This Custom Resource will also create a log group to log events of the custom resource. The log retention period for this addtional log group is hard-coded to 1 day.

*Further note* that, if the log group already exists and the `logRetention` is not set, the custom resource will reset
the log retention to never expire even if it was configured with a different value.

## FileSystem Access

You can configure a function to mount an Amazon Elastic File System (Amazon EFS) to a
directory in your runtime environment with the `filesystem` property. To access Amazon EFS
from lambda function, the Amazon EFS access point will be required.

The following sample allows the lambda function to mount the Amazon EFS access point to `/mnt/msg` in the runtime environment and access the filesystem with the POSIX identity defined in `posixUser`.

```ts
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as efs from 'aws-cdk-lib/aws-efs';

// create a new VPC
const vpc = new ec2.Vpc(this, 'VPC');

// create a new Amazon EFS filesystem
const fileSystem = new efs.FileSystem(this, 'Efs', { vpc });

// create a new access point from the filesystem
const accessPoint = fileSystem.addAccessPoint('AccessPoint', {
  // set /export/lambda as the root of the access point
  path: '/export/lambda',
  // as /export/lambda does not exist in a new efs filesystem, the efs will create the directory with the following createAcl
  createAcl: {
    ownerUid: '1001',
    ownerGid: '1001',
    permissions: '750',
  },
  // enforce the POSIX identity so lambda function will access with this identity
  posixUser: {
    uid: '1001',
    gid: '1001',
  },
});

const fn = new lambda.Function(this, 'MyLambda', {
  // mount the access point to /mnt/msg in the lambda runtime environment
  filesystem: lambda.FileSystem.fromEfsAccessPoint(accessPoint, '/mnt/msg'),
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  vpc,
});
```

## S3 Files Filesystem

[Amazon S3 Files](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-files.html) provides a fully managed NFS file system backed by an S3 bucket.
Lambda functions can mount an S3 Files file system to read and write files using standard filesystem operations,
which is useful for workloads that need POSIX-compatible file access to S3 data — such as ML inference, media processing, or legacy applications that expect a local mount path.

> **Note:** S3 Files currently only has L1 (CloudFormation) constructs. The setup below
> requires more manual wiring than the equivalent EFS integration. L2 constructs will
> simplify this in a future release.

S3 Files uses the same NFS infrastructure as Amazon EFS. To mount the file system in Lambda, you need:

- **File system** (`CfnFileSystem`) — the NFS file system backed by your S3 bucket. The backing bucket must have [versioning enabled](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html), as S3 Files relies on object versions to maintain consistency between the file system and S3. The file system requires an IAM role with two sets of permissions:
  - **S3 permissions** — read/write access to the bucket and its objects so S3 Files can synchronize data.
  - **EventBridge permissions** — S3 Files creates and manages EventBridge rules (prefixed `DO-NOT-DELETE-S3-Files`) to detect S3 object changes and trigger data synchronization. The role needs permission to manage these rules, plus read-only access to list rules for monitoring.
- **Mount targets** (`CfnMountTarget`) — ENIs placed in your VPC subnets that allow NFS clients (like Lambda) to connect. Each mount target needs a security group that permits inbound NFS traffic (TCP port 2049).
- **Access point** (`CfnAccessPoint`) — defines the POSIX user identity and root directory path that Lambda uses when accessing the file system. This scopes and isolates the function's view of the file system.


```ts
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3files from 'aws-cdk-lib/aws-s3files';

const vpc = new ec2.Vpc(this, 'Vpc');

// Versioning is required — S3 Files relies on object versions for consistency.
const bucket = new s3.Bucket(this, 'Bucket', { versioned: true });

// S3 Files assumes this role to sync data between S3 and the file system.
const role = new iam.Role(this, 'S3FilesRole', {
  assumedBy: new iam.ServicePrincipal('elasticfilesystem.amazonaws.com'),
});

// S3 permissions: read/write access to the bucket and objects
role.addToPolicy(new iam.PolicyStatement({
  actions: ['s3:ListBucket*'],
  resources: [bucket.bucketArn],
}));
role.addToPolicy(new iam.PolicyStatement({
  actions: ['s3:AbortMultipartUpload', 's3:DeleteObject', 's3:GetObject*', 's3:List*', 's3:PutObject*'],
  resources: [bucket.arnForObjects('*')],
}));

// EventBridge permissions: S3 Files creates rules prefixed "DO-NOT-DELETE-S3-Files"
// to detect S3 object changes and trigger data synchronization.
role.addToPolicy(new iam.PolicyStatement({
  actions: [
    'events:DeleteRule', 'events:DisableRule', 'events:EnableRule',
    'events:PutRule', 'events:PutTargets', 'events:RemoveTargets',
  ],
  resources: [`arn:${cdk.Aws.PARTITION}:events:*:*:rule/DO-NOT-DELETE-S3-Files*`],
  conditions: { StringEquals: { 'events:ManagedBy': 'elasticfilesystem.amazonaws.com' } },
}));
role.addToPolicy(new iam.PolicyStatement({
  actions: ['events:DescribeRule', 'events:ListRuleNamesByTarget', 'events:ListRules', 'events:ListTargetsByRule'],
  resources: [`arn:${cdk.Aws.PARTITION}:events:*:*:rule/*`],
}));

const fileSystem = new s3files.CfnFileSystem(this, 'S3FilesFs', {
  bucket: bucket.bucketArn,
  roleArn: role.roleArn,
});

const sg = new ec2.SecurityGroup(this, 'MountTargetSG', { vpc });

// Create a mount target in each private subnet so Lambda can reach the file system via NFS.
vpc.privateSubnets.forEach((subnet, i) =>
  new s3files.CfnMountTarget(this, `MountTarget${i}`, {
    fileSystemId: fileSystem.attrFileSystemId,
    subnetId: subnet.subnetId,
    securityGroups: [sg.securityGroupId],
  }),
);

// The access point defines the POSIX identity and root path Lambda uses on the file system.
const accessPoint = new s3files.CfnAccessPoint(this, 'AccessPoint', {
  fileSystemId: fileSystem.attrFileSystemId,
  rootDirectory: {
    path: '/export/lambda',
    creationPermissions: { ownerGid: '1001', ownerUid: '1001', permissions: '750' },
  },
  posixUser: { gid: '1001', uid: '1001' },
});

const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  vpc,
  filesystem: lambda.FileSystem.fromS3FilesAccessPoint(accessPoint, '/mnt/s3files'),
});
```

## IPv6 support

You can configure IPv6 connectivity for lambda function by setting `Ipv6AllowedForDualStack` to true.
It allows Lambda functions to specify whether the IPv6 traffic should be allowed when using dual-stack VPCs.
To access IPv6 network using Lambda, Dual-stack VPC is required. Using dual-stack VPC a function communicates with subnet over either of IPv4 or IPv6.

```ts
import * as ec2 from 'aws-cdk-lib/aws-ec2';

const natProvider = ec2.NatProvider.gateway();

// create dual-stack VPC
const vpc = new ec2.Vpc(this, 'DualStackVpc', {
  ipProtocol: ec2.IpProtocol.DUAL_STACK,
  subnetConfiguration: [
    {
      name: 'Ipv6Public1',
      subnetType: ec2.SubnetType.PUBLIC,
    },
    {
      name: 'Ipv6Public2',
      subnetType: ec2.SubnetType.PUBLIC,
    },
    {
      name: 'Ipv6Private1',
      subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
    },
  ],
  natGatewayProvider: natProvider,
});

const natGatewayId = natProvider.configuredGateways[0].gatewayId;
(vpc.privateSubnets[0] as ec2.PrivateSubnet).addIpv6Nat64Route(natGatewayId);

const fn = new lambda.Function(this, 'Lambda_with_IPv6_VPC', {
  code: new lambda.InlineCode('def main(event, context): pass'),
  handler: 'index.main',
  runtime: lambda.Runtime.PYTHON_3_9,
  vpc,
  ipv6AllowedForDualStack: true,
});
```

## Outbound traffic
By default, when creating a Lambda function, it would add a security group outbound rule to allow sending all network traffic (except IPv6). This is controlled by `allowAllOutbound` in function properties, which has a default value of `true`.

To allow outbound IPv6 traffic by default, explicitly set `allowAllIpv6Outbound` to `true` in function properties as shown below (the default value for `allowAllIpv6Outbound` is `false`):
```ts
import * as ec2 from 'aws-cdk-lib/aws-ec2';

const vpc = new ec2.Vpc(this, 'Vpc');

const fn = new lambda.Function(this, 'LambdaWithIpv6Outbound', {
  code: new lambda.InlineCode('def main(event, context): pass'),
  handler: 'index.main',
  runtime: lambda.Runtime.PYTHON_3_9,
  vpc: vpc,
  allowAllIpv6Outbound: true,
});
```

Do not specify `allowAllOutbound` or `allowAllIpv6Outbound` property if the `securityGroups` or `securityGroup` property is set. Instead, configure these properties directly on the security group.

## Ephemeral Storage

You can configure ephemeral storage on a function to control the amount of storage it gets for reading
or writing data, allowing you to use AWS Lambda for ETL jobs, ML inference, or other data-intensive workloads.
The ephemeral storage will be accessible in the functions' `/tmp` directory.

```ts
import { Size } from 'aws-cdk-lib';

const fn = new lambda.Function(this, 'MyFunction', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
  ephemeralStorageSize: Size.mebibytes(1024),
});
```

Read more about using this feature in [this AWS blog post](https://aws.amazon.com/blogs/aws/aws-lambda-now-supports-up-to-10-gb-ephemeral-storage/).

## Singleton Function

The `SingletonFunction` construct is a way to guarantee that a lambda function will be guaranteed to be part of the stack,
once and only once, irrespective of how many times the construct is declared to be part of the stack. This is guaranteed
as long as the `uuid` property and the optional `lambdaPurpose` property stay the same whenever they're declared into the
stack.

A typical use case of this function is when a higher level construct needs to declare a Lambda function as part of it but
needs to guarantee that the function is declared once. However, a user of this higher level construct can declare it any
number of times and with different properties. Using `SingletonFunction` here with a fixed `uuid` will guarantee this.

For example, the `AwsCustomResource` construct requires only one single lambda function for all api calls that are made.

## Bundling Asset Code

When using `lambda.Code.fromAsset(path)` it is possible to bundle the code by running a
command in a Docker container. The asset path will be mounted at `/asset-input`. The
Docker container is responsible for putting content at `/asset-output`. The content at
`/asset-output` will be zipped and used as Lambda code.

Example with Python:

```ts
new lambda.Function(this, 'Function', {
  code: lambda.Code.fromAsset(path.join(__dirname, 'my-python-handler'), {
    bundling: {
      image: lambda.Runtime.PYTHON_3_9.bundlingImage,
      command: [
        'bash', '-c',
        'pip install -r requirements.txt -t /asset-output && cp -au . /asset-output'
      ],
    },
  }),
  runtime: lambda.Runtime.PYTHON_3_9,
  handler: 'index.handler',
});
```

Runtimes expose a `bundlingImage` property that points to the [AWS SAM](https://github.com/awslabs/aws-sam-cli) build image.

Use `cdk.DockerImage.fromRegistry(image)` to use an existing image or
`cdk.DockerImage.fromBuild(path)` to build a specific image:

```ts
new lambda.Function(this, 'Function', {
  code: lambda.Code.fromAsset('/path/to/handler', {
    bundling: {
      image: DockerImage.fromBuild('/path/to/dir/with/DockerFile', {
        buildArgs: {
          ARG1: 'value1',
        },
      }),
      command: ['my', 'cool', 'command'],
    },
  }),
  runtime: lambda.Runtime.PYTHON_3_9,
  handler: 'index.handler',
});
```

## Language-specific APIs

Language-specific higher level constructs are provided in separate modules:

* `aws-cdk-lib/aws-lambda-nodejs`: [Github](https://github.com/aws/aws-cdk/tree/main/packages/aws-cdk-lib/aws-lambda-nodejs) & [CDK Docs](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_lambda_nodejs-readme.html)
* `@aws-cdk/aws-lambda-python-alpha`: [Github](https://github.com/aws/aws-cdk/tree/main/packages/%40aws-cdk/aws-lambda-python-alpha) & [CDK Docs](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-lambda-python-alpha-readme.html)

## Code Signing

Code signing for AWS Lambda helps to ensure that only trusted code runs in your Lambda functions.
When enabled, AWS Lambda checks every code deployment and verifies that the code package is signed by a trusted source.
For more information, see [Configuring code signing for AWS Lambda](https://docs.aws.amazon.com/lambda/latest/dg/configuration-codesigning.html).
The following code configures a function with code signing.

Please note the code will not be automatically signed before deployment. To ensure your code is properly signed, you'll need to conduct the code signing process either through the AWS CLI (Command Line Interface) [start-signing-job](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/signer/start-signing-job.html) or by accessing the AWS Signer console.

```ts
import * as signer from 'aws-cdk-lib/aws-signer';

const signingProfile = new signer.SigningProfile(this, 'SigningProfile', {
  platform: signer.Platform.AWS_LAMBDA_SHA384_ECDSA,
});

const codeSigningConfig = new lambda.CodeSigningConfig(this, 'CodeSigningConfig', {
  signingProfiles: [signingProfile],
});

new lambda.Function(this, 'Function', {
  codeSigningConfig,
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
});
```

## Runtime updates

Lambda runtime management controls help reduce the risk of impact to your workloads in the rare event of a runtime version incompatibility.
For more information, see [Runtime management controls](https://docs.aws.amazon.com/lambda/latest/dg/runtimes-update.html#runtime-management-controls)

```ts
new lambda.Function(this, 'Lambda', {
  runtimeManagementMode: lambda.RuntimeManagementMode.AUTO,
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
});
```

If you want to set the "Manual" setting, using the ARN of the runtime version as the argument.

```ts
new lambda.Function(this, 'Lambda', {
  runtimeManagementMode: lambda.RuntimeManagementMode.manual('runtimeVersion-arn'),
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'lambda-handler')),
});
```

## Exclude Patterns for Assets

When using `lambda.Code.fromAsset(path)` an `exclude` property allows you to ignore particular files for assets by providing patterns for file paths to exclude. Note that this has no effect on `Assets` bundled using the `bundling` property.

The `ignoreMode` property can be used with the `exclude` property to specify the file paths to ignore based on the [.gitignore specification](https://git-scm.com/docs/gitignore) or the [.dockerignore specification](https://docs.docker.com/engine/reference/builder/#dockerignore-file). The default behavior is to ignore file paths based on simple glob patterns.

```ts
new lambda.Function(this, 'Function', {
  code: lambda.Code.fromAsset(path.join(__dirname, 'my-python-handler'), {
    exclude: ['*.ignore'],
    ignoreMode: IgnoreMode.DOCKER, // Default is IgnoreMode.GLOB
  }),
  runtime: lambda.Runtime.PYTHON_3_9,
  handler: 'index.handler',
});
```

You can also write to include only certain files by using a negation.

```ts
new lambda.Function(this, 'Function', {
  code: lambda.Code.fromAsset(path.join(__dirname, 'my-python-handler'), {
    exclude: ['*', '!index.py'],
  }),
  runtime: lambda.Runtime.PYTHON_3_9,
  handler: 'index.handler',
});
```
