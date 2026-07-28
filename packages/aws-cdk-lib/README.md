# AWS Cloud Development Kit Library

The AWS CDK construct library provides APIs to define your CDK application and add
CDK constructs to the application.

## Usage

### Upgrade from CDK 1.x

When upgrading from CDK 1.x, remove all dependencies to individual CDK packages
from your dependencies file and follow the rest of the sections.

### Installation

To use this package, you need to declare this package and the `constructs` package as
dependencies.

According to the kind of project you are developing:

For projects that are CDK libraries in NPM, declare them both under the `devDependencies` **and** `peerDependencies` sections.
To make sure your library is compatible with the widest range of CDK versions: pick the minimum `aws-cdk-lib` version
that your library requires; declare a range dependency with a caret on that version in peerDependencies, and declare a
point version dependency on that version in devDependencies.

For example, let's say the minimum version your library needs is `2.38.0`. Your `package.json` should look like this:

```javascript
{
  "peerDependencies": {
    "aws-cdk-lib": "^2.38.0",
    "constructs": "^10.5.0"
  },
  "devDependencies": {
    /* Install the oldest version for testing so we don't accidentally use features from a newer version than we declare */
    "aws-cdk-lib": "2.38.0"
  }
}
```

For CDK apps, declare them under the `dependencies` section. Use a caret so you always get the latest version:

```json
{
  "dependencies": {
    "aws-cdk-lib": "^2.38.0",
    "constructs": "^10.5.0"
  }
}
```

### Use in your code

#### Classic import

You can use a classic import to get access to each service namespaces:

```ts nofixture
import { Stack, App, aws_s3 as s3 } from 'aws-cdk-lib';

const app = new App();
const stack = new Stack(app, 'TestStack');

new s3.Bucket(stack, 'TestBucket');
```

#### Barrel import

Alternatively, you can use "barrel" imports:

```ts nofixture
import { App, Stack } from 'aws-cdk-lib';
import { Bucket } from 'aws-cdk-lib/aws-s3';

const app = new App();
const stack = new Stack(app, 'TestStack');

new Bucket(stack, 'TestBucket');
```

<!--BEGIN CORE DOCUMENTATION-->

## Stacks and Stages

A `Stack` is the smallest physical unit of deployment, and maps directly onto
a CloudFormation Stack. You define a Stack by defining a subclass of `Stack`
-- let's call it `MyStack` -- and instantiating the constructs that make up
your application in `MyStack`'s constructor. You then instantiate this stack
one or more times to define different instances of your application. For example,
you can instantiate it once using few and cheap EC2 instances for testing,
and once again using more and bigger EC2 instances for production.

When your application grows, you may decide that it makes more sense to split it
out across multiple `Stack` classes. This can happen for a number of reasons:

- You could be starting to reach the maximum number of resources allowed in a single
  stack (this is currently 500).
- You could decide you want to separate out stateful resources and stateless resources
  into separate stacks, so that it becomes easy to tear down and recreate the stacks
  that don't have stateful resources.
- There could be a single stack with resources (like a VPC) that are shared
  between multiple instances of other stacks containing your applications.

As soon as your conceptual application starts to encompass multiple stacks,
it is convenient to wrap them in another construct that represents your
logical application. You can then treat that new unit the same way you used
to be able to treat a single stack: by instantiating it multiple times
for different instances of your application.

You can define a custom subclass of `Stage`, holding one or more
`Stack`s, to represent a single logical instance of your application.

As a final note: `Stack`s are not a unit of reuse. They describe physical
deployment layouts, and as such are best left to application builders to
organize their deployments with. If you want to vend a reusable construct,
define it as a subclasses of `Construct`: the consumers of your construct
will decide where to place it in their own stacks.

## Stack Synthesizers

Each Stack has a *synthesizer*, an object that determines how and where
the Stack should be synthesized and deployed. The synthesizer controls
aspects like:

- How does the stack reference assets? (Either through CloudFormation
  parameters the CLI supplies, or because the Stack knows a predefined
  location where assets will be uploaded).
- What roles are used to deploy the stack? These can be bootstrapped
  roles, roles created in some other way, or just the CLI's current
  credentials.

The following synthesizers are available:

- `DefaultStackSynthesizer`: recommended. Uses predefined asset locations and
  roles created by the modern bootstrap template. Access control is done by
  controlling who can assume the deploy role. This is the default stack
  synthesizer in CDKv2.
- `LegacyStackSynthesizer`: Uses CloudFormation parameters to communicate
  asset locations, and the CLI's current permissions to deploy stacks. This
  is the default stack synthesizer in CDKv1.
- `CliCredentialsStackSynthesizer`: Uses predefined asset locations, and the
  CLI's current permissions.

Each of these synthesizers takes configuration arguments. To configure
a stack with a synthesizer, pass it as one of its properties:

```ts
new MyStack(app, 'MyStack', {
  synthesizer: new DefaultStackSynthesizer({
    fileAssetsBucketName: 'amzn-s3-demo-bucket',
  }),
});
```

For more information on bootstrapping accounts and customizing synthesis,
see [Bootstrapping in the CDK Developer Guide](https://docs.aws.amazon.com/cdk/latest/guide/bootstrapping.html).

### STS Role Options

You can configure STS options that instruct the CDK CLI on which configuration should it use when assuming
the various roles that are involved in a deployment operation.

Refer to [the bootstrapping guide](https://docs.aws.amazon.com/cdk/v2/guide/bootstrapping-env.html#bootstrapping-env-roles) for further context.

These options are available via the `DefaultStackSynthesizer` properties:

```ts
class MyStack extends Stack {
  constructor(scope: Construct, id: string, props: StackProps) {
    super(scope, id, {
      ...props,
      synthesizer: new DefaultStackSynthesizer({
        deployRoleExternalId: '',
        deployRoleAdditionalOptions: {
          // https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html#API_AssumeRole_RequestParameters
        },
        fileAssetPublishingExternalId: '',
        fileAssetPublishingRoleAdditionalOptions: {
          // https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html#API_AssumeRole_RequestParameters
        },
        imageAssetPublishingExternalId: '',
        imageAssetPublishingRoleAdditionalOptions: {
          // https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html#API_AssumeRole_RequestParameters
        },
        lookupRoleExternalId: '',
        lookupRoleAdditionalOptions: {
          // https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRole.html#API_AssumeRole_RequestParameters
        },
      })
    });
  }
}
```

> Note that the `*additionalOptions` property does not allow passing `ExternalId` or `RoleArn`, as these options
> have dedicated properties that configure them.

#### Session Tags

STS session tags are used to implement [Attribute-Based Access Control](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction_attribute-based-access-control.html) (ABAC).

See [IAM tutorial: Define permissions to access AWS resources based on tags](https://docs.aws.amazon.com/IAM/latest/UserGuide/tutorial_attribute-based-access-control.html).

You can pass session tags for each [role created during bootstrap](https://docs.aws.amazon.com/cdk/v2/guide/bootstrapping-env.html#bootstrapping-env-roles) via the `*additionalOptions` property:

```ts
class MyStack extends Stack {
  constructor(parent: Construct, id: string, props: StackProps) {
    super(parent, id, {
      ...props,
      synthesizer: new DefaultStackSynthesizer({
        deployRoleAdditionalOptions: {
          Tags: [{ Key: 'Department', Value: 'Engineering' }]
        },
        fileAssetPublishingRoleAdditionalOptions: {
          Tags: [{ Key: 'Department', Value: 'Engineering' }]
        },
        imageAssetPublishingRoleAdditionalOptions: {
          Tags: [{ Key: 'Department', Value: 'Engineering' }]
        },
        lookupRoleAdditionalOptions: {
          Tags: [{ Key: 'Department', Value: 'Engineering' }]
        },
      })
    });
  }
}
```

This will cause the CDK CLI to include session tags when assuming each of these roles during deployment.
Note that the trust policy of the role must contain permissions for the `sts:TagSession` action.

Refer to the [IAM user guide on session tags](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_session-tags.html#id_session-tags_permissions-required).

- If you are using a custom bootstrap template, make sure the template includes these permissions.
- If you are using the default bootstrap template from a CDK version lower than XXXX, you will need to rebootstrap your enviroment (once).

## Nested Stacks

[Nested stacks](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/using-cfn-nested-stacks.html) are stacks created as part of other stacks. You create a nested stack within another stack by using the `NestedStack` construct.

As your infrastructure grows, common patterns can emerge in which you declare the same components in multiple templates. You can separate out these common components and create dedicated templates for them. Then use the resource in your template to reference other templates, creating nested stacks.

For example, assume that you have a load balancer configuration that you use for most of your stacks. Instead of copying and pasting the same configurations into your templates, you can create a dedicated template for the load balancer. Then, you just use the resource to reference that template from within other templates.

The following example will define a single top-level stack that contains two nested stacks: each one with a single Amazon S3 bucket:

```ts
class MyNestedStack extends cfn.NestedStack {
  constructor(scope: Construct, id: string, props?: cfn.NestedStackProps) {
    super(scope, id, props);

    new s3.Bucket(this, 'NestedBucket');
  }
}

class MyParentStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    new MyNestedStack(this, 'Nested1');
    new MyNestedStack(this, 'Nested2');
  }
}
```

Resources references across nested/parent boundaries (even with multiple levels of nesting) will be wired by the AWS CDK
through CloudFormation parameters and outputs. When a resource from a parent stack is referenced by a nested stack,
a CloudFormation parameter will automatically be added to the nested stack and assigned from the parent; when a resource
from a nested stack is referenced by a parent stack, a CloudFormation output will be automatically be added to the
nested stack and referenced using `Fn::GetAtt "Outputs.Xxx"` from the parent.

Nested stacks also support the use of Docker image and file assets.

## Accessing resources in a different stack

You can pass resource references between stacks freely, including across regions and
accounts. The CDK automatically wires the underlying CloudFormation mechanism for you.

```ts
const prod = { account: '123456789012', region: 'us-east-1' };

const stack1 = new StackThatProvidesABucket(app, 'Stack1' , { env: prod });

// stack2 will take a property { bucket: IBucket }
const stack2 = new StackThatExpectsABucket(app, 'Stack2', {
  bucket: stack1.bucket,
  env: prod
});
```

This also works across regions. For example, you can create a CloudFront distribution
in `us-east-2` that references an ACM certificate in `us-east-1`:

```ts
const stack1 = new Stack(app, 'Stack1', {
  env: {
    region: 'us-east-1',
  },
});
const cert = new acm.Certificate(stack1, 'Cert', {
  domainName: '*.example.com',
  validation: acm.CertificateValidation.fromDns(route53.PublicHostedZone.fromHostedZoneId(stack1, 'Zone', 'Z0329774B51CGXTDQV3X')),
});

const stack2 = new Stack(app, 'Stack2', {
  env: {
    region: 'us-east-2',
  },
});
new cloudfront.Distribution(stack2, 'Distribution', {
  defaultBehavior: {
    origin: new origins.HttpOrigin('example.com'),
  },
  domainNames: ['dev.example.com'],
  certificate: cert,
});
```

### Reference strength

Every cross-stack reference has a *strength* that determines the CloudFormation mechanism
used to pass the value and the coupling it creates between stacks. There are three
strengths:

**Strong** (default) — the producing stack cannot be deleted while any consumer exists.
This is enforced by CloudFormation itself.

**Weak** — uses `Fn::GetStackOutput` to read an output directly from the producing stack.
No coupling is created: the producing stack or resource can be deleted independently.
This means consuming stacks may temporarily reference a nonexistent resource until they
are updated as well.

**Both** — a transitional state used when migrating from strong to weak. The producer
keeps the strong-side artifacts while the consumer switches to `Fn::GetStackOutput`.
Once all consumers have been deployed with the weak mechanism, the strong-side artifacts
can be safely removed.

The exact CloudFormation realization depends on the strength and whether the reference
crosses region or account boundaries:

|                            | Strong (default)                                  | Both                                                                                         | Weak                                                            |
| -------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Same account and region    | Generates a `Fn::ImportValue` reference           | Generates a `Fn::GetStackOutput` reference AND an Export, but not the `Fn::ImportValue`      | Generates a `Fn::GetStackOutput` reference                      |
| Same account, cross-region | Generates a pair of `ExportWriter`/`ExportReader` | Generates a `Fn::GetStackOutput` reference AND an `ExportWriter`, but not the `ExportReader` | Generates a `Fn::GetStackOutput` reference                      |
| Cross-account              | Not possible. Falls back to weak.                 | Generates a `Fn::GetStackOutput` reference + cross-account role                              | Generates a `Fn::GetStackOutput` reference + cross-account role |

> [!NOTE]
> Strong cross-region references rely on Custom Resources, which are restricted to a
> CloudFormation response body size of
> [4096 bytes](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/crpg-ref-responses.html).
> To prevent deployment errors, limit the use of nested stacks and minimize stack name length.

### The deadly embrace

Strong references create a *deadly embrace*: a circular dependency between stacks that
prevents any of them from being updated or deleted. This happens because CloudFormation
Exports cannot be removed while any other stack imports them, and the producing stack
cannot be updated to remove the Export as long as the consuming stack still has a
`Fn::ImportValue` referencing it.

In practice, you hit the deadly embrace when you try to remove a resource that is
referenced by another stack. CloudFormation will reject the update with an error like:

```text
Export Stack1:ExportsOutputFnGetAtt-****** cannot be deleted as it is in use by Stack2
```

The solution is to first weaken the reference (switching the consumer away from
`Fn::ImportValue`), deploy, and then remove the resource. The sections below explain
how to do this.

### Controlling reference strength

There are three ways to control the strength of cross-stack references, each operating
at a different scope:

#### App-wide: context key

Set `@aws-cdk/core:defaultCrossStackReferences` in `cdk.json` to change the default for
all references in the app:

```json
{
  "context": {
    "@aws-cdk/core:defaultCrossStackReferences": "weak"
  }
}
```

You can also set this on a specific scope (stack, construct) using
`CrossStackReferences.of(scope).consume(strength)`:

```ts
declare const consumer: Stack;
// All references consumed by this stack will be weak
CrossStackReferences.of(consumer).consume(ReferenceStrength.WEAK);
```

#### Per-resource: `CrossStackReferences.of(resource).produce(strength)`

Override the strength for all references pointing at a specific resource:

```ts
declare const producer: Stack;
declare const consumer: Stack;

const bucket = new s3.Bucket(producer, 'SharedBucket');
CrossStackReferences.of(bucket).produce(ReferenceStrength.WEAK);

// This reference will use Fn::GetStackOutput regardless of the global setting
new CfnOutput(consumer, 'BucketName', { value: bucket.bucketName });
```

Other resources in the same stack continue to use the global default.

#### Per-usage: `Stack.consumeReference()`

Override the strength of a single reference usage without affecting other usages of the
same resource:

```ts
declare const topic: sns.Topic;

const consumer = new Stack(app, 'Consumer', {
  env: { account: '123456789012', region: 'us-east-1' },
});
new sns.Subscription(consumer, 'Subscription', {
  topic: sns.Topic.fromTopicArn(consumer, 'Topic',
    Stack.consumeReference(topic.topicArn, ReferenceStrength.WEAK)),
  endpoint: 'https://example.com/webhook',
  protocol: sns.SubscriptionProtocol.HTTPS,
});
```

The `consumeListReference` method is the equivalent for string list references.

### Resolving the deadly embrace

To remove a resource that has strong cross-stack references, you must weaken the
references before removing the resource. There are two approaches depending on whether
you want to weaken all references to the resource or just a specific one.

#### Weakening all references to a resource

Use `CrossStackReferences.of(resource).produce(...)` to weaken every reference pointing
at the resource. This requires two deployments before you can remove it:

DEPLOYMENT 1: switch to `BOTH` (keeps the strong-side artifacts while consumers switch
to the weak mechanism)

```ts
declare const bucket: s3.Bucket;
CrossStackReferences.of(bucket).produce(ReferenceStrength.BOTH);
```

DEPLOYMENT 2: switch to `WEAK` (removes the strong-side artifacts now that no consumer
depends on them)

```ts
declare const bucket: s3.Bucket;
CrossStackReferences.of(bucket).produce(ReferenceStrength.WEAK);
```

DEPLOYMENT 3: remove the bucket from `stack1` and any references from `stack2`.

#### Weakening a single reference

Use `Stack.consumeReference()` to weaken just one specific usage. This is useful when
a resource is referenced from multiple stacks, and you only want to decouple one of them.
The same two-deployment migration applies:

DEPLOYMENT 1: wrap the reference with `consumeReference` (defaults to `BOTH`)

```ts
declare const bucket: s3.Bucket;
declare const consumer: Stack;

// Previously: bucket.bucketArn was used directly
new CfnOutput(consumer, 'BucketArn', {
  value: Stack.consumeReference(bucket.bucketArn),
});
```

DEPLOYMENT 2: switch to `WEAK`

```ts
declare const bucket: s3.Bucket;
declare const consumer: Stack;

new CfnOutput(consumer, 'BucketArn', {
  value: Stack.consumeReference(bucket.bucketArn, ReferenceStrength.WEAK),
});
```

DEPLOYMENT 3: remove the resource or the reference as needed.

## Durations

To make specifications of time intervals unambiguous, a single class called
`Duration` is used throughout the AWS Construct Library by all constructs
that that take a time interval as a parameter (be it for a timeout, a
rate, or something else).

An instance of Duration is constructed by using one of the static factory
methods on it:

```ts
Duration.seconds(300)   // 5 minutes
Duration.minutes(5)     // 5 minutes
Duration.hours(1)       // 1 hour
Duration.days(7)        // 7 days
Duration.parse('PT5M')  // 5 minutes
```

Durations can be added or subtracted together:

```ts
Duration.minutes(1).plus(Duration.seconds(60)); // 2 minutes
Duration.minutes(5).minus(Duration.seconds(10)); // 290 secondes
```

## Size (Digital Information Quantity)

To make specification of digital storage quantities unambiguous, a class called
`Size` is available.

An instance of `Size` is initialized through one of its static factory methods:

```ts
Size.kibibytes(200) // 200 KiB
Size.mebibytes(5)   // 5 MiB
Size.gibibytes(40)  // 40 GiB
Size.tebibytes(200) // 200 TiB
Size.pebibytes(3)   // 3 PiB
```

Instances of `Size` created with one of the units can be converted into others.
By default, conversion to a higher unit will fail if the conversion does not produce
a whole number. This can be overridden by unsetting `integral` property.

```ts
Size.mebibytes(2).toKibibytes()                                             // yields 2048
Size.kibibytes(2050).toMebibytes({ rounding: SizeRoundingBehavior.FLOOR })  // yields 2
```

## Bitrate

To make specification of bitrate values unambiguous, a class called
`Bitrate` is available.

An instance of `Bitrate` is initialized through one of its static factory methods:

```ts
Bitrate.bps(5000)   // 5,000 bits per second
Bitrate.kbps(500)   // 500 kilobits per second
Bitrate.mbps(10)    // 10 megabits per second
Bitrate.gbps(1)     // 1 gigabit per second
```

Instances of `Bitrate` created with one of the units can be converted into others:

```ts
Bitrate.mbps(10).toBps()    // yields 10000000
Bitrate.mbps(10).toKbps()  // yields 10000
```

## Secrets

To help avoid accidental storage of secrets as plain text, we use the `SecretValue` type to
represent secrets. Any construct that takes a value that should be a secret (such as
a password or an access key) will take a parameter of type `SecretValue`.

The best practice is to store secrets in AWS Secrets Manager and reference them using `SecretValue.secretsManager`:

```ts
const secret = SecretValue.secretsManager('secretId', {
  jsonField: 'password', // optional: key of a JSON field to retrieve (defaults to all content),
  versionId: 'id',       // optional: id of the version (default AWSCURRENT)
  versionStage: 'stage', // optional: version stage name (default AWSCURRENT)
});
```

Using AWS Secrets Manager is the recommended way to reference secrets in a CDK app.
`SecretValue` also supports the following secret sources:

- `SecretValue.unsafePlainText(secret)`: stores the secret as plain text in your app and the resulting template (not recommended).
- `SecretValue.secretsManager(secret)`: refers to a secret stored in Secrets Manager
- `SecretValue.ssmSecure(param, version)`: refers to a secret stored as a SecureString in the SSM
 Parameter Store. If you don't specify the exact version, AWS CloudFormation uses the latest
 version of the parameter.
- `SecretValue.cfnParameter(param)`: refers to a secret passed through a CloudFormation parameter (must have `NoEcho: true`).
- `SecretValue.cfnDynamicReference(dynref)`: refers to a secret described by a CloudFormation dynamic reference (used by `ssmSecure` and `secretsManager`).
- `SecretValue.resourceAttribute(attr)`: refers to a secret returned from a CloudFormation resource creation.

`SecretValue`s should only be passed to constructs that accept properties of type
`SecretValue`. These constructs are written to ensure your secrets will not be
exposed where they shouldn't be. If you try to use a `SecretValue` in a
different location, an error about unsafe secret usage will be thrown at
synthesis time.

If you rotate the secret's value in Secrets Manager, you must also change at
least one property on the resource where you are using the secret, to force
CloudFormation to re-read the secret.

`SecretValue.ssmSecure()` is only supported for a limited set of resources.
[Click here for a list of supported resources and properties](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/dynamic-references.html#template-parameters-dynamic-patterns-resources).

`SecretValue.cfnDynamicReferenceKey` takes the same parameters as `SecretValue.secretsManager` and returns a key which can be used within a [dynamic reference](#dynamic-references) to dynamically load a secret from AWS Secrets Manager.

## ARN manipulation

Sometimes you will need to put together or pick apart Amazon Resource Names
(ARNs). The functions `stack.formatArn()` and `stack.splitArn()` exist for
this purpose.

`formatArn()` can be used to build an ARN from components. It will automatically
use the region and account of the stack you're calling it on:

```ts
declare const stack: Stack;

// Builds "arn:<PARTITION>:lambda:<REGION>:<ACCOUNT>:function:MyFunction"
stack.formatArn({
  service: 'lambda',
  resource: 'function',
  arnFormat: ArnFormat.COLON_RESOURCE_NAME,
  resourceName: 'MyFunction'
});
```

`splitArn()` can be used to get a single component from an ARN. `splitArn()`
will correctly deal with both literal ARNs and deploy-time values (tokens),
but in case of a deploy-time value be aware that the result will be another
deploy-time value which cannot be inspected in the CDK application.

```ts
declare const stack: Stack;

// Extracts the function name out of an AWS Lambda Function ARN
const arnComponents = stack.splitArn(arn, ArnFormat.COLON_RESOURCE_NAME);
const functionName = arnComponents.resourceName;
```

Note that the format of the resource separator depends on the service and
may be any of the values supported by `ArnFormat`. When dealing with these
functions, it is important to know the format of the ARN you are dealing with.

For an exhaustive list of ARN formats used in AWS, see [AWS ARNs and
Namespaces](https://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html)
in the AWS General Reference.

Some L1 constructs also have an auto-generated static `arnFor<ResourceName>()`
method that can be used to generate ARNs for resources of that type. For example,
`sns.Topic.arnForTopic(topic)` can be used to generate an ARN for a given topic.
Note that the parameter to this method is of type `ITopicRef`, which means that
it can be used with both `Topic` (L2) and `CfnTopic` (L1) constructs.

## Dependencies

### Construct Dependencies

Sometimes AWS resources depend on other resources, and the creation of one
resource must be completed before the next one can be started.

In general, CloudFormation will correctly infer the dependency relationship
between resources based on the property values that are used. In the cases where
it doesn't, the AWS Construct Library will add the dependency relationship for
you.

If you need to add an ordering dependency that is not automatically inferred,
you do so by adding a dependency relationship using
`constructA.node.addDependency(constructB)`. This will add a dependency
relationship between all resources in the scope of `constructA` and all
resources in the scope of `constructB`.

If you want a single object to represent a set of constructs that are not
necessarily in the same scope, you can use a `DependencyGroup`. The
following creates a single object that represents a dependency on two
constructs, `constructB` and `constructC`:

```ts
// Declare the dependable object
const bAndC = new DependencyGroup();
bAndC.add(constructB);
bAndC.add(constructC);

// Take the dependency
constructA.node.addDependency(bAndC);
```

### Stack Dependencies

Two different stack instances can have a dependency on one another. This
happens when an resource from one stack is referenced in another stack. In
that case, CDK records the cross-stack referencing of resources,
automatically produces the right CloudFormation primitives, and adds a
dependency between the two stacks. You can also manually add a dependency
between two stacks by using the `stackA.addStackDependency(stackB)` method.

A stack dependency has the following implications:

- Cyclic dependencies are not allowed, so if `stackA` is using resources from
  `stackB`, the reverse is not possible anymore.
- Stacks with dependencies between them are treated specially by the CDK
  toolkit:
  - If `stackA` depends on `stackB`, running `cdk deploy stackA` will also
    automatically deploy `stackB`.
  - `stackB`'s deployment will be performed *before* `stackA`'s deployment.

### CfnResource Dependencies

To make declaring dependencies between `CfnResource` objects easier, you can declare dependencies from one `CfnResource` object on another by using the `cfnResource1.addResourceDependency(cfnResource2)` method. This method will work for resources both within the same stack and across stacks as it detects the relative location of the two resources and adds the dependency either to the resource or between the relevant stacks, as appropriate. If more complex logic is in needed, you can similarly remove, replace, or view dependencies between `CfnResource` objects with the `CfnResource` `removeResourceDependency`, `replaceDependency`, and `obtainDependencies` methods, respectively.

## Custom Resources

Custom Resources are CloudFormation resources that are implemented by arbitrary
user code. They can do arbitrary lookups or modifications during a
CloudFormation deployment.

Custom resources are backed by *custom resource providers*. Commonly, these are
Lambda Functions that are deployed in the same deployment as the one that
defines the custom resource itself, but they can also be backed by Lambda
Functions deployed previously, or code responding to SNS Topic events running on
EC2 instances in a completely different account. For more information on custom
resource providers, see the next section.

Once you have a provider, each definition of a `CustomResource` construct
represents one invocation. A single provider can be used for the implementation
of arbitrarily many custom resource definitions. A single definition looks like
this:

```ts
new CustomResource(this, 'MyMagicalResource', {
  resourceType: 'Custom::MyCustomResource', // must start with 'Custom::'

  // the resource properties
  // properties like serviceToken or serviceTimeout are ported into properties automatically
  // try not to use key names similar to these or there will be a risk of overwriting those values
  properties: {
    Property1: 'foo',
    Property2: 'bar',
  },

  // the ARN of the provider (SNS/Lambda) which handles
  // CREATE, UPDATE or DELETE events for this resource type
  // see next section for details
  serviceToken: 'ARN',

  // the maximum time, in seconds, that can elapse before a custom resource operation times out.
  serviceTimeout: Duration.seconds(60),
});
```

### Custom Resource Providers

Custom resources are backed by a **custom resource provider** which can be
implemented in one of the following ways. The following table compares the
various provider types (ordered from low-level to high-level):

| Provider                                                             | Compute Type | Error Handling | Submit to CloudFormation |   Max Timeout   | Language | Footprint |
| -------------------------------------------------------------------- | :----------: | :------------: | :----------------------: | :-------------: | :------: | :-------: |
| [sns.Topic](#amazon-sns-topic)                                       | Self-managed |     Manual     |          Manual          |    Unlimited    |   Any    |  Depends  |
| [lambda.Function](#aws-lambda-function)                              |  AWS Lambda  |     Manual     |          Manual          |      15min      |   Any    |   Small   |
| [core.CustomResourceProvider](#the-corecustomresourceprovider-class) |  AWS Lambda  |      Auto      |           Auto           |      15min      | Node.js  |   Small   |
| [custom-resources.Provider](#the-custom-resource-provider-framework) |  AWS Lambda  |      Auto      |           Auto           | Unlimited Async |   Any    |   Large   |

Legend:

- **Compute type**: which type of compute can be used to execute the handler.
- **Error Handling**: whether errors thrown by handler code are automatically
  trapped and a FAILED response is submitted to CloudFormation. If this is
  "Manual", developers must take care of trapping errors. Otherwise, events
  could cause stacks to hang.
- **Submit to CloudFormation**: whether the framework takes care of submitting
  SUCCESS/FAILED responses to CloudFormation through the event's response URL.
- **Max Timeout**: maximum allows/possible timeout.
- **Language**: which programming languages can be used to implement handlers.
- **Footprint**: how many resources are used by the provider framework itself.

#### A note about singletons

When defining resources for a custom resource provider, you will likely want to
define them as a *stack singleton* so that only a single instance of the
provider is created in your stack and which is used by all custom resources of
that type.

Here is a basic pattern for defining stack singletons in the CDK. The following
examples ensures that only a single SNS topic is defined:

```ts
function getOrCreate(scope: Construct): sns.Topic {
  const stack = Stack.of(scope);
  const uniqueid = 'GloballyUniqueIdForSingleton'; // For example, a UUID from `uuidgen`
  const existing = stack.node.tryFindChild(uniqueid);
  if (existing) {
    return existing as sns.Topic;
  }
  return new sns.Topic(stack, uniqueid);
}
```

#### Amazon SNS Topic

Every time a resource event occurs (CREATE/UPDATE/DELETE), an SNS notification
is sent to the SNS topic. Users must process these notifications (e.g. through a
fleet of worker hosts) and submit success/failure responses to the
CloudFormation service.

> You only need to use this type of provider if your custom resource cannot run on AWS Lambda, for reasons other than the 15
> minute timeout. If you are considering using this type of provider because you want to write a custom resource provider that may need
> to wait for more than 15 minutes for the API calls to stabilize, have a look at the [`custom-resources`](#the-custom-resource-provider-framework) module first.
>
> Refer to the [CloudFormation Custom Resource documentation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/template-custom-resources.html) for information on the contract your custom resource needs to adhere to.

Set `serviceToken` to `topic.topicArn`  in order to use this provider:

```ts
const topic = new sns.Topic(this, 'MyProvider');

new CustomResource(this, 'MyResource', {
  serviceToken: topic.topicArn
});
```

#### AWS Lambda Function

An AWS lambda function is called *directly* by CloudFormation for all resource
events. The handler must take care of explicitly submitting a success/failure
response to the CloudFormation service and handle various error cases.

> **We do not recommend you use this provider type.** The CDK has wrappers around Lambda Functions that make them easier to work with.
>
> If you do want to use this provider, refer to the [CloudFormation Custom Resource documentation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/template-custom-resources.html) for information on the contract your custom resource needs to adhere to.

Set `serviceToken` to `lambda.functionArn` to use this provider:

```ts
const fn = new lambda.SingletonFunction(this, 'MyProvider', functionProps);

new CustomResource(this, 'MyResource', {
  serviceToken: fn.functionArn,
});
```

#### The `core.CustomResourceProvider` class

The class [`@aws-cdk/core.CustomResourceProvider`] offers a basic low-level
framework designed to implement simple and slim custom resource providers. It
currently only supports Node.js-based user handlers, represents permissions as raw
JSON blobs instead of `iam.PolicyStatement` objects, and it does not have
support for asynchronous waiting (handler cannot exceed the 15min lambda
timeout). The `CustomResourceProviderRuntime` supports runtime `nodejs12.x`,
`nodejs14.x`, `nodejs16.x`, `nodejs18.x`, `nodejs20.x`, `nodejs22.x` and `nodejs24.x`.

[`@aws-cdk/core.CustomResourceProvider`]: https://docs.aws.amazon.com/cdk/api/latest/docs/@aws-cdk_core.CustomResourceProvider.html

> **As an application builder, we do not recommend you use this provider type.** This provider exists purely for custom resources that are part of the AWS Construct Library.
>
> The [`custom-resources`](#the-custom-resource-provider-framework) provider is more convenient to work with and more fully-featured.

The provider has a built-in singleton method which uses the resource type as a
stack-unique identifier and returns the service token:

```ts
const serviceToken = CustomResourceProvider.getOrCreate(this, 'Custom::MyCustomResourceType', {
  codeDirectory: `${__dirname}/my-handler`,
  runtime: CustomResourceProviderRuntime.NODEJS_24_X,
  description: "Lambda function created by the custom resource provider",
});

new CustomResource(this, 'MyResource', {
  resourceType: 'Custom::MyCustomResourceType',
  serviceToken: serviceToken
});
```

The directory (`my-handler` in the above example) must include an `index.js` file. It cannot import
external dependencies or files outside this directory. It must export an async
function named `handler`. This function accepts the CloudFormation resource
event object and returns an object with the following structure:

```js
exports.handler = async function(event) {
  const id = event.PhysicalResourceId; // only for "Update" and "Delete"
  const props = event.ResourceProperties;
  const oldProps = event.OldResourceProperties; // only for "Update"s

  switch (event.RequestType) {
    case "Create":
      // ...

    case "Update":
      // ...

      // if an error is thrown, a FAILED response will be submitted to CFN
      throw new Error('Failed!');

    case "Delete":
      // ...
  }

  return {
    // (optional) the value resolved from `resource.ref`
    // defaults to "event.PhysicalResourceId" or "event.RequestId"
    PhysicalResourceId: "REF",

    // (optional) calling `resource.getAtt("Att1")` on the custom resource in the CDK app
    // will return the value "BAR".
    Data: {
      Att1: "BAR",
      Att2: "BAZ"
    },

    // (optional) user-visible message
    Reason: "User-visible message",

    // (optional) hides values from the console
    NoEcho: true
  };
}
```

Here is an complete example of a custom resource that summarizes two numbers:

`sum-handler/index.js`:

```js
exports.handler = async (e) => {
  return {
    Data: {
      Result: e.ResourceProperties.lhs + e.ResourceProperties.rhs,
    },
  };
};
```

`sum.ts`:

```ts nofixture
import { Construct } from 'constructs';
import {
  CustomResource,
  CustomResourceProvider,
  CustomResourceProviderRuntime,
  Token,
} from 'aws-cdk-lib';

export interface SumProps {
  readonly lhs: number;
  readonly rhs: number;
}

export class Sum extends Construct {
  public readonly result: number;

  constructor(scope: Construct, id: string, props: SumProps) {
    super(scope, id);

    const resourceType = 'Custom::Sum';
    const serviceToken = CustomResourceProvider.getOrCreate(this, resourceType, {
      codeDirectory: `${__dirname}/sum-handler`,
      runtime: CustomResourceProviderRuntime.NODEJS_22_X,
    });

    const resource = new CustomResource(this, 'Resource', {
      resourceType: resourceType,
      serviceToken: serviceToken,
      properties: {
        lhs: props.lhs,
        rhs: props.rhs
      }
    });

    this.result = Token.asNumber(resource.getAtt('Result'));
  }
}
```

Usage will look like this:

```ts fixture=README-custom-resource-provider
const sum = new Sum(this, 'MySum', { lhs: 40, rhs: 2 });
new CfnOutput(this, 'Result', { value: Token.asString(sum.result) });
```

To access the ARN of the provider's AWS Lambda function role, use the `getOrCreateProvider()`
built-in singleton method:

```ts
const provider = CustomResourceProvider.getOrCreateProvider(this, 'Custom::MyCustomResourceType', {
  codeDirectory: `${__dirname}/my-handler`,
  runtime: CustomResourceProviderRuntime.NODEJS_22_X,
});

const roleArn = provider.roleArn;
```

This role ARN can then be used in resource-based IAM policies.

To add IAM policy statements to this role, use `addToRolePolicy()`:

```ts
const provider = CustomResourceProvider.getOrCreateProvider(this, 'Custom::MyCustomResourceType', {
  codeDirectory: `${__dirname}/my-handler`,
  runtime: CustomResourceProviderRuntime.NODEJS_22_X,
});
provider.addToRolePolicy({
  Effect: 'Allow',
  Action: 's3:GetObject',
  Resource: '*',
})
```

Note that `addToRolePolicy()` uses direct IAM JSON policy blobs, *not* a
`iam.PolicyStatement` object like you will see in the rest of the CDK.

#### The Custom Resource Provider Framework

The [`@aws-cdk/custom-resources`] module includes an advanced framework for
implementing custom resource providers.

[`@aws-cdk/custom-resources`]: https://docs.aws.amazon.com/cdk/api/latest/docs/custom-resources-readme.html

Handlers are implemented as AWS Lambda functions, which means that they can be
implemented in any Lambda-supported runtime. Furthermore, this provider has an
asynchronous mode, which means that users can provide an `isComplete` lambda
function which is called periodically until the operation is complete. This
allows implementing providers that can take up to two hours to stabilize.

Set `serviceToken` to `provider.serviceToken` to use this type of provider:

```ts
const provider = new customresources.Provider(this, 'MyProvider', {
  onEventHandler,
  isCompleteHandler, // optional async waiter
});

new CustomResource(this, 'MyResource', {
  serviceToken: provider.serviceToken
});
```

See the [documentation](https://docs.aws.amazon.com/cdk/api/latest/docs/aws-cdk-lib.custom_resources-readme.html) for more details.

## AWS CloudFormation features

A CDK stack synthesizes to an AWS CloudFormation Template. This section
explains how this module allows users to access low-level CloudFormation
features when needed.

### Stack Outputs

CloudFormation [stack outputs][cfn-stack-output] and exports are created using
the `CfnOutput` class:

```ts
new CfnOutput(this, 'OutputName', {
  value: myBucket.bucketName,
  description: 'The name of an S3 bucket', // Optional
  exportName: 'TheAwesomeBucket', // Registers a CloudFormation export named "TheAwesomeBucket"
});
```

You can also use the `exportValue` method to export values as stack outputs:

```ts
declare const stack: Stack;

stack.exportValue(myBucket.bucketName, {
  name: 'TheAwesomeBucket',
  description: 'The name of an S3 bucket',
});
```

[cfn-stack-output]: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/outputs-section-structure.html

### Parameters

CloudFormation templates support the use of [Parameters][cfn-parameters] to
customize a template. They enable CloudFormation users to input custom values to
a template each time a stack is created or updated. While the CDK design
philosophy favors using build-time parameterization, users may need to use
CloudFormation in a number of cases (for example, when migrating an existing
stack to the AWS CDK).

Template parameters can be added to a stack by using the `CfnParameter` class:

```ts
new CfnParameter(this, 'MyParameter', {
  type: 'Number',
  default: 1337,
  // See the API reference for more configuration props
});
```

The value of parameters can then be obtained using one of the `value` methods.
As parameters are only resolved at deployment time, the values obtained are
placeholder tokens for the real value (`Token.isUnresolved()` would return `true`
for those):

```ts
const param = new CfnParameter(this, 'ParameterName', { /* config */ });

// If the parameter is a String
param.valueAsString;

// If the parameter is a Number
param.valueAsNumber;

// If the parameter is a List
param.valueAsList;
```

[cfn-parameters]: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/parameters-section-structure.html

### Pseudo Parameters

CloudFormation supports a number of [pseudo parameters][cfn-pseudo-params],
which resolve to useful values at deployment time. CloudFormation pseudo
parameters can be obtained from static members of the `Aws` class.

It is generally recommended to access pseudo parameters from the scope's `stack`
instead, which guarantees the values produced are qualifying the designated
stack, which is essential in cases where resources are shared cross-stack:

```ts
// "this" is the current construct
const stack = Stack.of(this);

stack.account; // Returns the AWS::AccountId for this stack (or the literal value if known)
stack.region;  // Returns the AWS::Region for this stack (or the literal value if known)
stack.partition; // Returns the AWS::Partition for this stack (or the literal value if known)
```

[cfn-pseudo-params]: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/pseudo-parameter-reference.html

### Resource Options

CloudFormation resources can also specify [resource
attributes][cfn-resource-attributes]. The `CfnResource` class allows
accessing those through the `cfnOptions` property:

```ts
const rawBucket = new s3.CfnBucket(this, 'Bucket', { /* ... */ });
// -or-
const rawBucketAlt = myBucket.node.defaultChild as s3.CfnBucket;

// then
rawBucket.cfnOptions.condition = new CfnCondition(this, 'EnableBucket', { /* ... */ });
rawBucket.cfnOptions.metadata = {
  metadataKey: 'MetadataValue',
};
```

Resource dependencies (the `DependsOn` attribute) is modified using the
`cfnResource.addDependency` method:

```ts
const resourceA = new CfnResource(this, 'ResourceA', resourceProps);
const resourceB = new CfnResource(this, 'ResourceB', resourceProps);

resourceB.addResourceDependency(resourceA);
```

[cfn-resource-attributes]: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-product-attribute-reference.html

#### CreationPolicy

Some resources support a [CreationPolicy][creation-policy] to be specified as a CfnOption.

The creation policy is invoked only when AWS CloudFormation creates the associated resource. Currently, the only AWS CloudFormation resources that support creation policies are `CfnAutoScalingGroup`, `CfnInstance`, `CfnWaitCondition` and `CfnFleet`.

The `CfnFleet` resource from the `aws-appstream` module supports specifying `startFleet` as
a property of the creationPolicy on the resource options. Setting it to true will make AWS CloudFormation wait until the fleet is started before continuing with the creation of
resources that depend on the fleet resource.

```ts
const fleet = new appstream.CfnFleet(this, 'Fleet', {
  instanceType: 'stream.standard.small',
  name: 'Fleet',
  computeCapacity: {
    desiredInstances: 1,
  },
  imageName: 'AppStream-AmazonLinux2-09-21-2022',
});
fleet.cfnOptions.creationPolicy = {
  startFleet: true,
};
```

The properties passed to the level 2 constructs `AutoScalingGroup` and `Instance` from the
`aws-ec2` module abstract what is passed into the `CfnOption` properties `resourceSignal` and
`autoScalingCreationPolicy`, but when using level 1 constructs you can specify these yourself.

The CfnWaitCondition resource from the `aws-cloudformation` module supports the `resourceSignal`.
The format of the timeout is `PT#H#M#S`. In the example below AWS Cloudformation will wait for
3 success signals to occur within 15 minutes before the status of the resource will be set to
`CREATE_COMPLETE`.

```ts
declare const resource: CfnResource;

resource.cfnOptions.creationPolicy = {
  resourceSignal: {
    count: 3,
    timeout: 'PR15M',
  }
};
```

[creation-policy]: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-attribute-creationpolicy.html

### Intrinsic Functions and Condition Expressions

CloudFormation supports [intrinsic functions][cfn-intrinsics]. These functions
can be accessed from the `Fn` class, which provides type-safe methods for each
intrinsic function as well as condition expressions:

```ts
declare const myObjectOrArray: any;
declare const myArray: any;

// To use Fn::Base64
Fn.base64('SGVsbG8gQ0RLIQo=');

// To compose condition expressions:
const environmentParameter = new CfnParameter(this, 'Environment');
Fn.conditionAnd(
  // The "Environment" CloudFormation template parameter evaluates to "Production"
  Fn.conditionEquals('Production', environmentParameter),
  // The AWS::Region pseudo-parameter value is NOT equal to "us-east-1"
  Fn.conditionNot(Fn.conditionEquals('us-east-1', Aws.REGION)),
);

// To use Fn::ToJsonString
Fn.toJsonString(myObjectOrArray);

// To use Fn::Length
Fn.len(Fn.split(',', myArray));
```

When working with deploy-time values (those for which `Token.isUnresolved`
returns `true`), idiomatic conditionals from the programming language cannot be
used (the value will not be known until deployment time). When conditional logic
needs to be expressed with un-resolved values, it is necessary to use
CloudFormation conditions by means of the `CfnCondition` class:

```ts
const environmentParameter = new CfnParameter(this, 'Environment');
const isProd = new CfnCondition(this, 'IsProduction', {
  expression: Fn.conditionEquals('Production', environmentParameter),
});

// Configuration value that is a different string based on IsProduction
const stage = Fn.conditionIf(isProd.logicalId, 'Beta', 'Prod').toString();

// Make Bucket creation condition to IsProduction by accessing
// and overriding the CloudFormation resource
const bucket = new s3.Bucket(this, 'Bucket');
const cfnBucket = myBucket.node.defaultChild as s3.CfnBucket;
cfnBucket.cfnOptions.condition = isProd;
```

[cfn-intrinsics]: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/intrinsic-function-reference.html

### Mappings

CloudFormation [mappings][cfn-mappings] are created and queried using the
`CfnMappings` class:

```ts
const regionTable = new CfnMapping(this, 'RegionTable', {
  mapping: {
    'us-east-1': {
      regionName: 'US East (N. Virginia)',
      // ...
    },
    'us-east-2': {
      regionName: 'US East (Ohio)',
      // ...
    },
    // ...
  }
});

regionTable.findInMap(Aws.REGION, 'regionName')
```

This will yield the following template:

```yaml
Mappings:
  RegionTable:
    us-east-1:
      regionName: US East (N. Virginia)
    us-east-2:
      regionName: US East (Ohio)
```

Mappings can also be synthesized "lazily"; lazy mappings will only render a "Mappings"
section in the synthesized CloudFormation template if some `findInMap` call is unable to
immediately return a concrete value due to one or both of the keys being unresolved tokens
(some value only available at deploy-time).

For example, the following code will not produce anything in the "Mappings" section. The
call to `findInMap` will be able to resolve the value during synthesis and simply return
`'US East (Ohio)'`.

```ts
const regionTable = new CfnMapping(this, 'RegionTable', {
  mapping: {
    'us-east-1': {
      regionName: 'US East (N. Virginia)',
    },
    'us-east-2': {
      regionName: 'US East (Ohio)',
    },
  },
  lazy: true,
});

regionTable.findInMap('us-east-2', 'regionName');
```

On the other hand, the following code will produce the "Mappings" section shown above,
since the top-level key is an unresolved token. The call to `findInMap` will return a token that resolves to
`{ "Fn::FindInMap": [ "RegionTable", { "Ref": "AWS::Region" }, "regionName" ] }`.

```ts
declare const regionTable: CfnMapping;

regionTable.findInMap(Aws.REGION, 'regionName');
```

An optional default value can also be passed to `findInMap`. If either key is not found in the map and the mapping is lazy, `findInMap` will return the default value and not render the mapping.
If the mapping is not lazy or either key is an unresolved token, the call to `findInMap` will return a token that resolves to
`{ "Fn::FindInMap": [ "MapName", "TopLevelKey", "SecondLevelKey", { "DefaultValue": "DefaultValue" } ] }`, and the mapping will be rendered.
Note that the `AWS::LanguageExtentions` transform is added to enable the default value functionality.

For example, the following code will again not produce anything in the "Mappings" section. The
call to `findInMap` will be able to resolve the value during synthesis and simply return
`'Region not found'`.

```ts
const regionTable = new CfnMapping(this, 'RegionTable', {
  mapping: {
    'us-east-1': {
      regionName: 'US East (N. Virginia)',
    },
    'us-east-2': {
      regionName: 'US East (Ohio)',
    },
  },
  lazy: true,
});

regionTable.findInMap('us-west-1', 'regionName', 'Region not found');
```

[cfn-mappings]: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/mappings-section-structure.html

### Dynamic References

CloudFormation supports [dynamically resolving][cfn-dynamic-references] values
for SSM parameters (including secure strings) and Secrets Manager. Encoding such
references is done using the `CfnDynamicReference` class:

```ts
new CfnDynamicReference(
  CfnDynamicReferenceService.SECRETS_MANAGER,
  'secret-id:secret-string:json-key:version-stage:version-id',
);
```

[cfn-dynamic-references]: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/dynamic-references.html

## RemovalPolicies

The `RemovalPolicies` class provides a convenient way to manage removal policies for AWS CDK resources within a construct scope. It allows you to apply removal policies to multiple resources at once, with options to include or exclude specific resource types.

```typescript
declare const scope: Construct;
declare const parent: Construct;
declare const bucket: s3.CfnBucket;

// Apply DESTROY policy to all resources in a scope
RemovalPolicies.of(scope).destroy();

// Apply RETAIN policy to all resources in a scope
RemovalPolicies.of(scope).retain();

// Apply SNAPSHOT policy to all resources in a scope
RemovalPolicies.of(scope).snapshot();

// Apply RETAIN_ON_UPDATE_OR_DELETE policy to all resources in a scope
RemovalPolicies.of(scope).retainOnUpdateOrDelete();

// Apply RETAIN policy only to specific resource types
RemovalPolicies.of(parent).retain({
  applyToResourceTypes: [
    'AWS::DynamoDB::Table',
    bucket.cfnResourceType, // 'AWS::S3::Bucket'
    rds.CfnDBInstance.CFN_RESOURCE_TYPE_NAME, // 'AWS::RDS::DBInstance'
  ],
});

// Apply SNAPSHOT policy excluding specific resource types
RemovalPolicies.of(scope).snapshot({
  excludeResourceTypes: ['AWS::Test::Resource'],
});
```

### RemovalPolicies vs MissingRemovalPolicies

CDK provides two different classes for managing removal policies:

- RemovalPolicies: Always applies the specified removal policy, overriding any existing policies.
- MissingRemovalPolicies: Applies the removal policy only to resources that don't already have a policy set.

```typescript
// Override any existing policies
RemovalPolicies.of(scope).retain();

// Only apply to resources without existing policies
MissingRemovalPolicies.of(scope).retain();
```

### Aspect Priority

Both RemovalPolicies and MissingRemovalPolicies are implemented as [Aspects](#aspects). You can control the order in which they're applied using the priority parameter:

```typescript
declare const stack: Stack;

// Apply in a specific order based on priority
RemovalPolicies.of(stack).retain({ priority: 100 });
RemovalPolicies.of(stack).destroy({ priority: 200 }); // This will override the RETAIN policy
```

For RemovalPolicies, the policies are applied in order of aspect execution, with the last applied policy overriding previous ones. The priority only affects the order in which aspects are applied during synthesis.

#### Note

When using MissingRemovalPolicies with priority, a warning will be issued as this can lead to unexpected behavior. This is because MissingRemovalPolicies only applies to resources without existing policies, making priority less relevant.

### Template Options & Transform

CloudFormation templates support a number of options, including which Macros or
[Transforms][cfn-transform] to use when deploying the stack. Those can be
configured using the `stack.templateOptions` property:

```ts
const stack = new Stack(app, 'StackName');

stack.templateOptions.description = 'This will appear in the AWS console';
stack.templateOptions.transforms = ['AWS::Serverless-2016-10-31'];
stack.templateOptions.metadata = {
  metadataKey: 'MetadataValue',
};
```

[cfn-transform]: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/transform-section-structure.html

### Emitting Raw Resources

The `CfnResource` class allows emitting arbitrary entries in the
[Resources][cfn-resources] section of the CloudFormation template.

```ts
new CfnResource(this, 'ResourceId', {
  type: 'AWS::S3::Bucket',
  properties: {
    BucketName: 'amzn-s3-demo-bucket'
  },
});
```

As for any other resource, the logical ID in the CloudFormation template will be
generated by the AWS CDK, but the type and properties will be copied verbatim in
the synthesized template.

[cfn-resources]: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/resources-section-structure.html

### Including raw CloudFormation template fragments

When migrating a CloudFormation stack to the AWS CDK, it can be useful to
include fragments of an existing template verbatim in the synthesized template.
This can be achieved using the `CfnInclude` class.

```ts
new CfnInclude(this, 'ID', {
  template: {
    Resources: {
      Bucket: {
        Type: 'AWS::S3::Bucket',
        Properties: {
          BucketName: 'amzn-s3-demo-bucket'
        }
      }
    }
  },
});
```

### Termination Protection

You can prevent a stack from being accidentally deleted by enabling termination
protection on the stack. If a user attempts to delete a stack with termination
protection enabled, the deletion fails and the stack--including its status--remains
unchanged. Enabling or disabling termination protection on a stack sets it for any
nested stacks belonging to that stack as well. You can enable termination protection
on a stack by setting the `terminationProtection` prop to `true`.

```ts
const stack = new Stack(app, 'StackName', {
  terminationProtection: true,
});
```

You can also set termination protection with the setter after you've instantiated the stack.

```ts
const stack = new Stack(app, 'StackName', {});
stack.terminationProtection = true;
```

By default, termination protection is disabled.

### Description

You can add a description of the stack in the same way as `StackProps`.

```ts
const stack = new Stack(app, 'StackName', {
  description: 'This is a description.',
});
```

### Receiving CloudFormation Stack Events

You can add one or more SNS Topic ARNs to any Stack:

```ts
const stack = new Stack(app, 'StackName', {
  notificationArns: ['arn:aws:sns:us-east-1:123456789012:Topic'],
});
```

Stack events will be sent to any SNS Topics in this list. These ARNs are added to those specified using
the `--notification-arns` command line option.

Note that in order to do delete notification ARNs entirely, you must pass an empty array ([]) instead of omitting it.
If you omit the property, no action on existing ARNs will take place.

> [!NOTE]
> Adding the `notificationArns` property (or using the `--notification-arns` CLI options) will **override**
> any existing ARNs configured on the stack. If you have an external system managing notification ARNs,
> either migrate to use this mechanism, or avoid specfying notification ARNs with the CDK.

### CfnJson

`CfnJson` allows you to postpone the resolution of a JSON blob from
deployment-time. This is useful in cases where the CloudFormation JSON template
cannot express a certain value.

A common example is to use `CfnJson` in order to render a JSON map which needs
to use intrinsic functions in keys. Since JSON map keys must be strings, it is
impossible to use intrinsics in keys and `CfnJson` can help.

The following example defines an IAM role which can only be assumed by
principals that are tagged with a specific tag.

```ts
const tagParam = new CfnParameter(this, 'TagName');

const stringEquals = new CfnJson(this, 'ConditionJson', {
  value: {
    [`aws:PrincipalTag/${tagParam.valueAsString}`]: true,
  },
});

const principal = new iam.AccountRootPrincipal().withConditions({
  StringEquals: stringEquals,
});

new iam.Role(this, 'MyRole', { assumedBy: principal });
```

**Explanation**: since in this example we pass the tag name through a parameter, it
can only be resolved during deployment. The resolved value can be represented in
the template through a `{ "Ref": "TagName" }`. However, since we want to use
this value inside a [`aws:PrincipalTag/TAG-NAME`](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_condition-keys.html#condition-keys-principaltag)
IAM operator, we need it in the *key* of a `StringEquals` condition. JSON keys
*must be* strings, so to circumvent this limitation, we use `CfnJson`
to "delay" the rendition of this template section to deploy-time. This means
that the value of `StringEquals` in the template will be `{ "Fn::GetAtt": [ "ConditionJson", "Value" ] }`, and will only "expand" to the operator we synthesized during deployment.

### Stack Resource Limit

When deploying to AWS CloudFormation, it needs to keep in check the amount of resources being added inside a Stack. Currently it's possible to check the limits in the [AWS CloudFormation quotas](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cloudformation-limits.html) page.

It's possible to synthesize the project with more Resources than the allowed (or even reduce the number of Resources).

Set the context key `@aws-cdk/core:stackResourceLimit` with the proper value, being 0 for disable the limit of resources.

### Template Indentation

The AWS CloudFormation templates generated by CDK include indentation by default.
Indentation makes the templates more readable, but also increases their size,
and CloudFormation templates cannot exceed 1MB.

It's possible to reduce the size of your templates by suppressing indentation.

To do this for all templates, set the context key `@aws-cdk/core:suppressTemplateIndentation` to `true`.

To do this for a specific stack, add a `suppressTemplateIndentation: true` property to the
stack's `StackProps` parameter. You can also set this property to `false` to override
the context key setting.

Similarly, to do this for a specific nested stack, add a `suppressTemplateIndentation: true` property to its `NestedStackProps` parameter. You can also set this property to `false` to override the context key setting.

## App Context

[Context values](https://docs.aws.amazon.com/cdk/v2/guide/context.html) are key-value pairs that can be associated with an app, stack, or construct.
One common use case for context is to use it for enabling/disabling [feature flags](https://docs.aws.amazon.com/cdk/v2/guide/featureflags.html). There are several places
where context can be specified. They are listed below in the order they are evaluated (items at the
top take precedence over those below).

- The `node.setContext()` method
- The `postCliContext` prop when you create an `App`
- The CLI via the `--context` CLI argument
- The `cdk.json` file via the `context` key:
- The `cdk.context.json` file:
- The `~/.cdk.json` file via the `context` key:
- The `context` prop when you create an `App`

### Examples of setting context

```ts
new App({
  context: {
    '@aws-cdk/core:newStyleStackSynthesis': true,
  },
});
```

```ts
const app = new App();
app.node.setContext('@aws-cdk/core:newStyleStackSynthesis', true);
```

```ts
new App({
  postCliContext: {
    '@aws-cdk/core:newStyleStackSynthesis': true,
  },
});
```

```console
cdk synth --context @aws-cdk/core:newStyleStackSynthesis=true
```

#### `cdk.json`

```json
{
  "context": {
    "@aws-cdk/core:newStyleStackSynthesis": true
  }
}
```

#### `cdk.context.json`

```json
{
  "@aws-cdk/core:newStyleStackSynthesis": true
}
```

#### `~/.cdk.json`

```json
{
  "context": {
    "@aws-cdk/core:newStyleStackSynthesis": true
  }
}
```

## IAM Permissions Boundary

It is possible to apply an [IAM permissions boundary](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_boundaries.html)
to all roles within a specific construct scope. The most common use case would
be to apply a permissions boundary at the `Stage` level.

```ts
const prodStage = new Stage(app, 'ProdStage', {
  permissionsBoundary: PermissionsBoundary.fromName('cdk-${Qualifier}-PermissionsBoundary'),
});
```

Any IAM Roles or Users created within this Stage will have the default
permissions boundary attached.

For more details see the [Permissions Boundary](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_iam-readme.html#permissions-boundaries) section in the IAM guide.

## Template and Policy Validation

To improve iteration speed on your infrastructure and get feedback on your
changes as early as possible, the CloudFormation templates produced by CDK are
validated immediately after synthesis. They will be automatically validated
against a comprehensive set of rules, checking for potential deployment failures
and AWS best practices.

You can extend the set of validations with additional plugins if you want,
like [cdk-nag](https://github.com/cdklabs/cdk-nag) which is focused on
standards compliance, or you can write your own organization-specific validations.

If there are any violations, the synthesis will fail and a report will be
printed to the console and to a file.

### CloudFormationValidatePlugin

By default, a default rule set is added in the form of the
`CloudFormationValidatePlugin`, which is powered by
[@aws/cloudformation-validate](https://github.com/aws-cloudformation/cloudformation-validate).

This rule set checks for misconfigurations that will fail deployments (which
will be reported as errors), and for violations of AWS best practices (reported
as warnings). To suppress a reported warning or error, add the following to
your application:

```ts fixture=validation-plugin
const app = new App();

// You can use any scope here, closer to the violation is safer
Validations.of(app).acknowledge({
  id: 'CloudFormation-Validate::W9999',
  reason: 'This is not recommended but we have a good reason to do it like this',
});
```

The plugin supports loading custom Rego or CloudFormation guard rule sets, which you
can configure if you add instantiate the plugin explicitly to your app:

```ts fixture=validation-plugin
const app = new App();

// Rules text, read from disk perhaps
declare const myRules: string;

Validations.of(app).addPlugins(new CloudFormationValidatePlugin({
  guardRules: [{
    name: 'My rules',
    content: myRules,
  }],
}));
```

### Additional plugins

You can also add custom plugins like [cdk-nag](https://github.com/cdklabs/cdk-nag) and
[CfnGuardValidator](https://github.com/cdklabs/cdk-validator-cfnguard),
or author custom plugins for validation tools such as [OPA](https://www.openpolicyagent.org/).

To use one or more validation plugins in your application, use the
`Validations.of()` API:

```ts fixture=validation-plugin
// globally for the entire app (an app is a stage)
const app = new App();
Validations.of(app).addPlugins(new ThirdPartyPluginX());
Validations.of(app).addPlugins(new ThirdPartyPluginY());

// only apply to a particular stage
const prodStage = new Stage(app, 'ProdStage');
Validations.of(prodStage).addPlugins(new ThirdPartyPluginX());
```

Immediately after synthesis, all plugins registered this way will be invoked to
validate all the templates generated in the scope you defined. In particular, if
you register the templates in the `App` object, all templates will be subject to
validation.

> **Warning**
> Other than modifying the cloud assembly, plugins can do anything that your CDK
> application can. They can read data from the filesystem, access the network
> etc. It's your responsibility as the consumer of a plugin to verify that it is
> secure to use.

### Validation Reporting

By default, the report is output in two ways:

- A JSON file called `policy-validation-report.json` is written to the cloud assembly directory.
- A human-readable format is printed to the standard error output.

To disable either format, explicitly set the corresponding context key to `false`:

```ts fixture=validation-plugin
// Disable pretty-printed console output (JSON file still written)
const app = new App({
  context: { '@aws-cdk/core:validationReportPrettyPrint': false },
});
```

### For plugin authors

The communication protocol between the CDK core module and your policy tool is
defined by the `IPolicyValidationPlugin` interface. To create a new plugin you must
write a class that implements this interface. There are two things you need to
implement: the plugin name (by overriding the `name` property), and the
`validate()` method.

The framework will call `validate()`, passing an `IPolicyValidationContext` object.
The location of the templates to be validated is given by `templatePaths`. The
plugin should return an instance of `PolicyValidationPluginReport`. This object
represents the report that the user will receive at the end of the synthesis.

```ts fixture=validation-plugin
class MyPlugin implements IPolicyValidationPlugin {
  public readonly name = 'MyPlugin';

  public validate(context: IPolicyValidationContext): PolicyValidationPluginReport {
    // First read the templates using context.templatePaths...

    // ...then perform the validation, and then compose and return the report.
    // Using hard-coded values here for better clarity:
    return {
      success: false,
      violations: [{
        ruleName: 'CKV_AWS_117',
        description: 'Ensure that AWS Lambda function is configured inside a VPC',
        fix: 'https://docs.bridgecrew.io/docs/ensure-that-aws-lambda-function-is-configured-inside-a-vpc-1',
        violatingResources: [{
          resourceLogicalId: 'MyFunction3BAA72D1',
          templatePath: '/home/johndoe/myapp/cdk.out/MyService.template.json',
          locations: ['Properties/VpcConfig'],
        }],
      }],
    };
  }
}
```

In addition to the name, plugins may optionally report their version (`version`
property) and a list of IDs of the rules they are going to evaluate (`ruleIds`
property).

Note that plugins are not allowed to modify anything in the cloud assembly. Any
attempt to do so will result in synthesis failure.

If your plugin depends on an external tool, keep in mind that some developers may
not have that tool installed in their workstations yet. To minimize friction, we
highly recommend that you provide some installation script along with your
plugin package, to automate the whole process. Better yet, run that script as
part of the installation of your package. With `npm`, for example, you can run
add it to the `postinstall`
[script](https://docs.npmjs.com/cli/v9/using-npm/scripts) in the `package.json`
file.

## Annotations

Construct authors can add annotations to constructs to report at three different
levels: `ERROR`, `WARN`, `INFO`.

Typically warnings are added for things that are important for the user to be
aware of, but will not cause deployment errors in all cases. Some common
scenarios are (non-exhaustive list):

- Warn when the user needs to take a manual action, e.g. IAM policy should be
  added to an referenced resource.
- Warn if the user configuration might not follow best practices (but is still
  valid)
- Warn if the user is using a deprecated API

*Annotations* feed into the *Validations* mechanism, so any construct-level
*annotation you add will be reported via the validations report.

### Acknowledging Warnings

If you would like to run with `--strict` mode enabled (warnings will throw
errors) it is possible to `acknowledge` warnings to make the warning go away.

For example, if > 10 IAM managed policies are added to an IAM Group, a warning
will be created:

```text
IAM:Group:MaxPoliciesExceeded: You added 11 to IAM Group my-group. The maximum number of managed policies attached to an IAM group is 10.
```

If you have requested a [quota increase](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_iam-quotas.html#reference_iam-quotas-entities)
you may have the ability to add > 10 managed policies which means that this
warning does not apply to you. You can acknowledge this by `acknowledging` the
warning by the `id`.

```ts
Annotations.of(this).acknowledgeWarning('IAM:Group:MaxPoliciesExceeded', 'Account has quota increased to 20');

// Because Annotations ultimately become Validations, you can also acknowledge the Validation
Validations.of(this).acknowledge({
  id: 'Construct-Annotations::IAM:Group:MaxPoliciesExceeded',
  reason: 'Account has quota increased to 20',
});
```

### Acknowledging Infos

Informational messages can also be emitted and acknowledged. Use `addInfoV2()`
to add an info message that can later be suppressed with `acknowledgeInfo()`.
Unlike warnings, info messages are not affected by the `--strict` mode and will never cause synthesis to fail.

```ts
Annotations.of(this).addInfoV2('my-lib:Construct.someInfo', 'Some message explaining the info');
Annotations.of(this).acknowledgeInfo('my-lib:Construct.someInfo', 'This info can be ignored');

// Because Annotations ultimately become Validations, you can also acknowledge the Validation
Validations.of(this).acknowledge({
  id: 'Construct-Annotations::my-lib:Construct.someInfo',
  reason: 'Some message explaining the info',
});
```

## Mixins

CDK Mixins provide a new, advanced way to add functionality through composable abstractions.
Unlike traditional L2 constructs that bundle all features together, Mixins allow you to pick and choose exactly the capabilities you need for constructs.

Mixins are an *addition*, not a replacement for construct properties.
They are applied during or after construct construction using the `.with()` method:

```ts fixture=README-mixins
// Apply mixins fluently with .with()
new s3.CfnBucket(scope, "MyL1Bucket")
  .with(new BucketBlockPublicAccess())
  .with(new BucketAutoDeleteObjects());

// Apply multiple mixins to the same construct
new s3.CfnBucket(scope, "MyL1Bucket")
  .with(new BucketBlockPublicAccess(), new BucketAutoDeleteObjects());

// Mixins work with all types of constructs:
// L1, L2 and even custom constructs
new s3.Bucket(stack, 'MyL2Bucket').with(new BucketBlockPublicAccess());
new CustomBucket(stack, 'MyCustomBucket').with(new BucketBlockPublicAccess());
```

There is an alternative form available that allows additional, advanced configuration of Mixin application: `Mixins.of()`.

```ts fixture=README-mixins
import { ConstructSelector } from "aws-cdk-lib/core";

// Basic: Apply mixins to any construct, calls can be chained
const myBucket = new s3.CfnBucket(scope, "MyBucket");
Mixins.of(myBucket)
  .apply(new BucketBlockPublicAccess())
  .apply(new BucketAutoDeleteObjects());

// Basic: Or multiple Mixins passed to apply
Mixins.of(myBucket)
  .apply(new BucketBlockPublicAccess(), new BucketAutoDeleteObjects());

// Advanced: Apply to constructs matching a selector, e.g. match by ID
Mixins.of(
  scope,
  ConstructSelector.byId("prod/**")
).apply(new CustomProdSecurityConfig());

// Advanced: Require a mixin to be applied to every node in the construct tree
Mixins.of(stack)
  .apply(new CustomProdSecurityConfig())
  .requireAll();
```

### How Mixins are applied

Each construct has a `with()` method and Mixins will be applied to all nodes of the construct.
Sometimes more control is needed.
Especially when authoring construct libraries, it may be desirable to have full control over the Mixin application process.
Think of the L3 pattern again: How can you encode the rules to which Mixins may or may not be applied in your L3?
This is where `Mixins.of()` and the `MixinApplicator` class come in.
They provide more complex ways to select targets, apply Mixins and set expectations.

#### Mixin application on construct trees

When working with construct trees like Stacks (as opposed to single resources),
`Mixins.of()` offers a more comprehensive API to configure how Mixins are applied.
By default, Mixins are applied to all supported constructs in the tree:

```ts fixture=README-mixins
// Apply to all constructs in a scope
Mixins.of(scope).apply(new BucketBlockPublicAccess());
```

Optionally, you may select specific constructs:

```ts fixture=README-mixins
import { ConstructSelector } from "aws-cdk-lib/core";

// Apply to a given L1 resource or L2 resource construct
Mixins.of(
  bucket,
  ConstructSelector.cfnResource() // provided CfnResource or a CfnResource default child
).apply(new BucketBlockPublicAccess());

// Apply to all resources of a specific type
Mixins.of(
  scope,
  ConstructSelector.resourcesOfType(s3.CfnBucket.CFN_RESOURCE_TYPE_NAME)
).apply(new BucketBlockPublicAccess());

// Alternative: select by CloudFormation resource type name
Mixins.of(
  scope,
  ConstructSelector.resourcesOfType("AWS::S3::Bucket")
).apply(new BucketBlockPublicAccess());

// Apply to constructs matching a pattern
Mixins.of(
  scope,
  ConstructSelector.byId("prod/**")
).apply(new CustomProdSecurityConfig());

// The default is to apply to all constructs in the scope
Mixins.of(
  scope,
  ConstructSelector.all() // pass through to IConstruct.findAll()
).apply(new CustomProdSecurityConfig());
```

#### Mixins that must be used

Sometimes you need assertions that a Mixin has been applied to certain set of constructs.
`Mixins.of(...)` keeps track of Mixin applications and this report can be used to create assertions.

It comes with two convenience helpers:
Use `requireAll()` to assert the Mixin will be applied to all selected constructs.
If a construct is in the selection that is not supported by the Mixin, this will throw an error.
The `requireAny()` helper will assert the Mixin was applied to at least one construct from the selection.
If the Mixin wasn't applied to any construct at all, this will throw an error.

Both helpers will only check future calls of `apply()`.
Set them before calling `apply()` to take effect.

```ts fixture=README-mixins
Mixins.of(scope, selector)
  // Assert Mixin was applied to all constructs in the selection
  .requireAll()
  // Or assert Mixin was applied to at least one construct in the selection
  // .requireAny()
  .apply(new BucketBlockPublicAccess());

// Get an application report for manual assertions
const report = Mixins.of(scope).apply(new BucketBlockPublicAccess()).report;
```

### Creating Custom Mixins

Mixins are simple classes that implement the `IMixin` interface (usually by extending the abstract `Mixin` class):

```ts fixture=README-mixins
class EnableVersioning extends Mixin implements IMixin {
  supports(construct: any): construct is s3.CfnBucket {
    return s3.CfnBucket.isCfnBucket(construct);
  }

  applyTo(bucket: IConstruct): void {
    (bucket as s3.CfnBucket).versioningConfiguration = {
      status: "Enabled"
    };
  }
}

// Usage
new s3.CfnBucket(scope, "MyBucket")
  .with(new EnableVersioning());
```

We recommend to implement Mixins at the L1 level and to have them target a specific resource construct.
This way, the same Mixin can be applied to constructs from all levels.

When applied, the `.supports()` method is used to decided if a Mixin can be applied to a given construct.
Depending on the application method (see below), the Mixin is then applied, skipped or an error is thrown.

```ts fixture=README-mixins
bucketAccessLogsMixin.supports(bucket); // returns `true`
bucketAccessLogsMixin.supports(queue); // returns `false`
```

#### Validation with Mixins

Mixins have two distinct phases: Initialization and application.
During initialization only the Mixin's input properties are available, but during application we also have access the target construct.

Mixins should validate their properties and targets as early as possible.
During initialization validate all input properties.
Then during application validate any target dependent pre-conditions or interactions with Mixin properties.

Like with constructs, Mixins should *throw an error* in case of unrecoverable failures and use *annotations* for recoverable ones.
It is best practices to collect errors and throw as a group whenever possible.
Mixins can attach *[lazy validators](https://github.com/aws/aws-cdk/blob/main/docs/DESIGN_GUIDELINES.md#attaching-lazy-validators)* to the target construct.
Use this to ensure a certain property is met at end of an app's execution.

```ts fixture=README-mixins
class MyEncryptionAtRest extends Mixin {
  constructor(props: MyEncryptionAtRestProps = {}) {
    super();
    // Validate Mixin props at construction time
    if (props.bucketKey && props.algorithm === 'aws:kms:dsse') {
      throw new Error("Cannot use S3 Bucket Key and DSSE together");
    }
  }

  supports(construct: any): construct is s3.CfnBucket {
    return s3.CfnBucket.isCfnBucket(construct);
  }

  applyTo(target: s3.CfnBucket): s3.CfnBucket {
    // Validate pre-conditions on the target, throw if error is unrecoverable
    if (!target.bucketEncryption) {
      throw new Error("Bucket encryption not configured");
    }

    // Validate properties are met after app execution
    target.node.addValidation({
      validate: () => isKmsEncrypted(target)
        ? ['This bucket must use aws:kms encryption.']
        : []
    });

    target.bucketEncryption = {
      serverSideEncryptionConfiguration: [{
        bucketKeyEnabled: true,
        serverSideEncryptionByDefault: {
          sseAlgorithm: "aws:kms"
        }
      }]
    };
    return target;
  }
}
```

#### Mixins and Aspects

Mixins and Aspects are similar concepts and both are implementations of the [visitor pattern](https://en.wikipedia.org/wiki/Visitor_pattern).
They crucially differ in their time of application:

- Mixins are always applied *immediately*, they are a tool of *imperative* programming.
- Aspects are applied *after* all other code during the synthesis phase, this makes them *declarative*.

Both Mixins and Aspects have valid use cases and complement each other.
We recommend to use Mixins to *make changes*, and to use Aspects to *validate behaviors*.
Aspects may also be used when changes need to apply to *future additions*, for examples in custom libraries.

## Aspects

[Aspects](https://docs.aws.amazon.com/cdk/v2/guide/aspects.html) is a feature in CDK that allows you to apply operations or transformations across all
constructs in a construct tree. Common use cases include tagging resources, enforcing encryption on S3 Buckets, or applying specific security or
compliance rules to all resources in a stack.

Conceptually, there are two types of Aspects:

- **Read-only aspects** scan the construct tree but do not make changes to the tree. Common use cases of read-only aspects include performing validations
(for example, enforcing that all S3 Buckets have versioning enabled) and logging (for example, collecting information about all deployed resources for
audits or compliance).
- **Mutating aspects** either (1.) add new nodes or (2.) mutate existing nodes of the tree in-place. One commonly used mutating Aspect is adding Tags to
resources. An example of an Aspect that adds a node is one that automatically adds a security group to every EC2 instance in the construct tree if
no default is specified.

Here is a simple example of creating and applying an Aspect on a Stack to enable versioning on all S3 Buckets:

```ts
class EnableBucketVersioning implements IAspect {
  visit(node: IConstruct) {
    if (node instanceof s3.CfnBucket) {
      node.versioningConfiguration = {
        status: 'Enabled'
      };
    }
  }
}

const app = new App();
const stack = new MyStack(app, 'MyStack');

// Apply the aspect to enable versioning on all S3 Buckets
Aspects.of(stack).add(new EnableBucketVersioning());
```

### Aspect Stabilization

The modern behavior is that Aspects automatically run on newly added nodes to the construct tree. This is controlled by the
flag `@aws-cdk/core:aspectStabilization`, which is default for new projects (since version 2.172.0).

The old behavior of Aspects (without stabilization) was that Aspect invocation runs once on the entire construct
tree. This meant that nested Aspects (Aspects that create new Aspects) are not invoked and nodes created by Aspects at a higher level of the construct tree are not visited.

To enable the stabilization behavior for older versions, use this feature by putting the following into your `cdk.context.json`:

```json
{
  "@aws-cdk/core:aspectStabilization": true
}
```

### Aspect Priorities

Users can specify the order in which Aspects are applied on a construct by using the optional priority parameter when applying an Aspect. Priority
values must be non-negative integers, where a higher number means the Aspect will be applied later, and a lower number means it will be applied sooner.

By default, newly created nodes always inherit aspects. Priorities are mainly for ordering between mutating aspects on the construct tree.

CDK provides standard priority values for mutating and readonly aspects to help ensure consistency across different construct libraries.
Note that Aspects that have same priority value are not guaranteed to be executed
in a consistent order.

```ts
/**
 * Default Priority values for Aspects.
 */
class AspectPriority {
  /**
   * Suggested priority for Aspects that mutate the construct tree.
   */
  static readonly MUTATING: number = 200;

  /**
   * Suggested priority for Aspects that only read the construct tree.
   */
  static readonly READONLY: number = 1000;

  /**
   * Default priority for Aspects that are applied without a priority.
   */
  static readonly DEFAULT: number = 500;
}
```

If no priority is provided, the default value will be 500. This ensures that aspects without a specified priority run after mutating aspects but before
any readonly aspects.

Correctly applying Aspects with priority values ensures that mutating aspects (such as adding tags or resources) run before validation aspects. This allows users to avoid misconfigurations and ensure that the final
construct tree is fully validated before being synthesized.

### Applying Aspects with Priority

```ts
class MutatingAspect implements IAspect {
  visit(node: IConstruct) {
    // Modifies a resource in some way
  }
}

class ValidationAspect implements IAspect {
  visit(node: IConstruct) {
    // Perform some readonly validation on the cosntruct tree
  }
}

const stack = new Stack();

Aspects.of(stack).add(new MutatingAspect(), { priority: AspectPriority.MUTATING } );  // Run first (mutating aspects)
Aspects.of(stack).add(new ValidationAspect(), { priority: AspectPriority.READONLY } );  // Run later (readonly aspects)
```

### Inspecting applied aspects and changing priorities

We also give customers the ability to view all of their applied aspects and override the priority on these aspects.
The `AspectApplication` class represents an Aspect that is applied to a node of the construct tree with a priority.

Users can access AspectApplications on a node by calling `applied` from the Aspects class as follows:

```ts
declare const root: Construct;
const app = new App();
const stack = new MyStack(app, 'MyStack');

Aspects.of(stack).add(new MyAspect());

let aspectApplications: AspectApplication[] = Aspects.of(root).applied;

for (const aspectApplication of aspectApplications) {
  // The aspect we are applying
  console.log(aspectApplication.aspect);
  // The construct we are applying the aspect to
  console.log(aspectApplication.construct);
  // The priority it was applied with
  console.log(aspectApplication.priority);

  // Change the priority
  aspectApplication.priority = 700;
}
```

### Converting between Aspects and Mixins

Since Mixins and Aspects are both implementations of the visitor pattern, they can be converted from each other using the `Shims` class:

```ts fixture=README-mixins
// Applies an Aspect immediately as a Mixin
const versioningMixin = Shims.asMixin(new EnableBucketVersioning());
Mixins.of(scope).apply(versioningMixin);

// Delays application of a Mixin to the synthesis phase
const publicAccessAspect = Shims.asAspect(new BucketBlockPublicAccess());
Aspects.of(scope).add(publicAccessAspect);
```

When shimming a Mixin to an Aspect, the Mixin will automatically only be applied to supported constructs (via `supports()`).
Going from an Aspect to a Mixin, the Aspect will be applied to every node.

## Blueprint Property Injection

The goal of Blueprint Property Injection is to provide builders an automatic way to set default property values.

Construct authors can declare that a Construct can have it properties injected by adding `@propertyInjectable`
class decorator and specifying `PROPERTY_INJECTION_ID` readonly property.
All L2 Constructs will support Property Injection so organizations can write injectors to set their Construct Props.

Organizations can set default property values to a Construct by writing Injectors for builders to consume.

Here is a simple example of an Injector for APiKey that sets enabled to false.

```ts fixture=README-Injectable
class ApiKeyPropsInjector implements IPropertyInjector {
  readonly constructUniqueId: string;

  constructor() {
    this.constructUniqueId = api.ApiKey.PROPERTY_INJECTION_ID;
  }

  inject(originalProps: api.ApiKeyProps, context: InjectionContext): api.ApiKeyProps {
    return {
      enabled: false,
      apiKeyName: originalProps.apiKeyName,
      customerId: originalProps.customerId,
      defaultCorsPreflightOptions: originalProps.defaultCorsPreflightOptions,
      defaultIntegration: originalProps.defaultIntegration,
      defaultMethodOptions: originalProps.defaultMethodOptions,
      description: originalProps.description,
      generateDistinctId: originalProps.generateDistinctId,
      resources: originalProps.resources,
      stages: originalProps.stages,
      value: originalProps.value,
    };
  }
}
```

Some notes:

- ApiKey must have a `PROPERTY_INJECTION_ID` property, in addition to having `@propertyInjectable` class decorator.
- We set ApiKeyProps.enabled to false, if it is not provided; otherwise it would use the value that was passed in.
- It is also possible to force ApiKeyProps.enabled to false, and not provide a way for the builders to overwrite it.

Here is an example of how builders can use the injector the org created.

```ts fixture=README-blueprints
const stack = new Stack(app, 'my-stack', {
  propertyInjectors: [new ApiKeyPropsInjector()],
});
new api.ApiKey(stack, 'my-api-key', {});
```

This is equivalent to:

```ts fixture=README-blueprints
const stack = new Stack(app, 'my-stack', {});
new api.ApiKey(stack, 'my-api-key', {
  enabled: false,
});
```

Some notes:

- We attach the injectors to Stack in this example, but we can also attach them to App or Stage.
- All the ApiKey created in the scope of stack will get `enabled: false`.
- Builders can overwrite the default value with `new ApiKey(stack, 'my-api-key', {enabled: true});`

If you specify two or more injectors for the same Constructs, the last one is in effect.  In the example below, `ApiKeyPropsInjector` will never be applied.

```ts fixture=README-blueprints
const stack = new Stack(app, 'my-stack', {
  propertyInjectors: [
    new ApiKeyPropsInjector(),
    new AnotherApiKeyPropsInjector(),
  ],
});
```

For more information, please see the [RFC](https://github.com/aws/aws-cdk-rfcs/blob/main/text/0693-property-injection.md).

<!--END CORE DOCUMENTATION-->
