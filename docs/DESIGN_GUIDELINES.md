# AWS Construct Library Design Guidelines

The AWS Construct Library is a rich class library of CDK constructs which
represent all resources offered by the AWS Cloud and higher-level constructs for
achieving common tasks.

The purpose of this document is to provide guidelines for designing the APIs in
the AWS Construct Library in order to ensure a consistent and integrated
experience across the entire AWS surface area.

- [AWS Construct Library Design Guidelines](#aws-construct-library-design-guidelines)
  - [Preface](#preface)
  - [What's Included](#whats-included)
  - [Mixins, Facades, and Traits](#mixins-facades-and-traits)
    - [Mixins](#mixins)
    - [Facades](#facades)
    - [Traits](#traits)
    - [When to use which](#when-to-use-which)
  - [API Design](#api-design)
    - [Modules](#modules)
    - [Construct Class](#construct-class)
    - [Construct Interface](#construct-interface)
      - [Defining construct interfaces: interface types](#defining-construct-interfaces-interface-types)
      - [Defining construct interfaces: what goes onto the resource interface](#defining-construct-interfaces-what-goes-onto-the-resource-interface)
      - [Consuming construct interfaces: what interface type to choose](#consuming-construct-interfaces-what-interface-type-to-choose)
      - [What if the entire L2 interface is too large, but the reference interface is too small?](#what-if-the-entire-l2-interface-is-too-large-but-the-reference-interface-is-too-small)
      - [Abstract Base](#abstract-base)
    - [Props](#props)
      - [Types](#types)
      - [Defaults](#defaults)
      - [Flat](#flat)
      - [Concise](#concise)
      - [Naming](#naming)
      - [Property Documentation](#property-documentation)
      - [Enums](#enums)
      - [Unions](#unions)
    - [Attributes](#attributes)
    - [Configuration](#configuration)
      - [Prefer Additions](#prefer-additions)
      - [Dropped Mutations](#dropped-mutations)
    - [Factories](#factories)
    - [Referenced Resources](#referenced-resources)
      - [“from” Methods](#from-methods)
      - [From-attributes](#from-attributes)
    - [Roles](#roles)
    - [Resource Policies](#resource-policies)
    - [VPC](#vpc)
    - [Grants](#grants)
      - [Traits](#traits-1)
      - [Auto-generation and manual implementation](#auto-generation-and-manual-implementation)
    - [Metrics](#metrics)
    - [Events](#events)
    - [Connections](#connections)
    - [Integrations](#integrations)
    - [State](#state)
    - [Physical Names - TODO](#physical-names---todo)
    - [Tags](#tags)
    - [Secrets](#secrets)
  - [Project Structure](#project-structure)
    - [Code Organization](#code-organization)
  - [Implementation](#implementation)
    - [General Principles](#general-principles)
    - [Construct IDs](#construct-ids)
    - [Errors](#errors)
      - [Avoid Errors If Possible](#avoid-errors-if-possible)
      - [Error reporting mechanism](#error-reporting-mechanism)
      - [Throwing exceptions](#throwing-exceptions)
      - [Never Catch Exceptions](#never-catch-exceptions)
      - [Attaching (lazy) Validators](#attaching-lazy-validators)
      - [Attaching Errors/Warnings](#attaching-errorswarnings)
      - [Error messages](#error-messages)
    - [Tokens](#tokens)
    - [Lazys](#lazys)
  - [Documentation](#documentation)
    - [Inline Documentation](#inline-documentation)
    - [Readme](#readme)
  - [Testing](#testing)
    - [Unit tests](#unit-tests)
    - [Integration tests](#integration-tests)
    - [Versioning](#versioning)
  - [Naming \& Style](#naming--style)
    - [Naming Conventions](#naming-conventions)
    - [Coding Style](#coding-style)

## Preface

As much as possible, the guidelines in this document are enforced using the
[**awslint** tool](https://www.npmjs.com/package/awslint) which reflects on the
APIs and verifies that the APIs adhere to the guidelines. When a guideline is
backed by a linter rule, the rule name will be referenced like this:
_[awslint:resource-class-is-construct]_.

For the purpose of this document, we will use "Foo" to denote the official name
of the resource as defined in the AWS CloudFormation resource specification
(i.e. "Bucket", "Queue", "Topic", etc). This notation allows deriving names from
the official name. For example, `FooProps` would be `BucketProps`, `TopicProps`,
etc, `IFoo` would be `IBucket`, `ITopic` and so forth.

The guidelines in this document use TypeScript (and npm package names) since
this is the source programming language used to author the library, which is
later packaged and published to all programming languages through
[jsii](https://github.com/awslabs/jsii).

When designing APIs for the AWS Construct Library (and these guidelines), we
follow the tenets of the AWS CDK:

* **Meet developers where they are**: our APIs are based on the mental model of
the user, and not the mental model of the service APIs, which are normally
designed against the constraints of the backend system and the fact that these
APIs are used through network requests. It's okay to enable multiple ways to
achieve the same thing, in order to make it more natural for users who come from
different mental models.
* **Full coverage**: the AWS Construct Library exposes the full surface area of
AWS. It is not opinionated about which parts of the service API should be
used. However, it offers sensible defaults to allow users to get started quickly
with best practices, but allows them to fully customize this behavior. We use a
layered architecture so that users can choose the level of abstraction that fits
their needs.
* **Designed for the CDK**: the AWS Construct Library is primarily optimized for
AWS customers who use the CDK idiomatically and natively.  As much as possible,
the APIs are non-leaky and do not require that users understand how AWS
CloudFormation works. If users wish to “escape” from the abstraction, the APIs
offer explicit ways to do that, so that users won't be blocked by missing
capabilities or issues.
* **Open**: the AWS Construct Library is an open and extensible framework. It is
also open source. It heavily relies on interfaces to allow developers to extend
its behavior and provide their own custom implementations. Anyone should be able
to publish constructs that look & feel exactly like any construct in the AWS
Construct Library.
* **Designed for jsii**: the AWS Construct Library is built with jsii. This
allows the library to be used from all supported programming languages. jsii
poses restrictions on language features that cannot be idiomatically represented
in target languages.

## What's Included

The AWS Construct Library, which is shipped as part of the AWS CDK constructs
representing AWS resources.

The AWS Construct Library has multiple layers of constructs, beginning
with low-level constructs, which we call _CFN Resources_ (short for
CloudFormation resources), or L1 (short for "level 1"). These constructs
directly represent all resources available in AWS CloudFormation. CFN Resources
are periodically generated from the AWS CloudFormation Resource
Specification. They are named **Cfn**_Xyz_, where _Xyz_ is name of the
resource. For example, CfnBucket represents the AWS::S3::Bucket AWS
CloudFormation resource. When you use Cfn resources, you must explicitly
configure all resource properties, which requires a complete understanding of
the details of the underlying AWS CloudFormation resource model.

The next level of constructs, L2, also represent AWS resources, but with a
higher-level, intent-based API. They provide similar functionality, but provide
the defaults, boilerplate, and glue logic you'd be writing yourself with a CFN
Resource construct. L2 constructs offer convenient defaults and reduce the need
to know all the details about the AWS resources they represent, while providing
convenience methods that make it simpler to work with the resource. For example,
the `s3.Bucket` class represents an Amazon S3 bucket with additional properties
and methods, such as `bucket.addLifeCycleRule()`, which adds a lifecycle rule to
the bucket.

An L2 is a composition of several building blocks around a CFN Resource
Construct. These building blocks are Mixins, Facades, and Traits (see
[Mixins, Facades, and Traits](#mixins-facades-and-traits) below). Examples of
behaviors that an L2 commonly include:

- Strongly-typed modeling of the underlying L1 properties
- Mixins that modify the resource's own configuration (e.g., enabling versioning
  on a bucket, auto-deleting objects on removal).
- Facades that provide integrations with other resources (e.g., grants,
  events, metrics).
- Traits that advertise capabilities to other constructs (e.g., encryptable,
  has resource policy).

In addition to the above, some L2s may introduce more complex and
helpful functionality, either part of the original L2 itself, or as part of a
separate construct. The most common form of these L2s are integration constructs
that model interactions between different services (e.g., SNS publishing to SQS,
CodePipeline actions that trigger Lambda functions).

The next level of abstraction present within the CDK are what we designate as
"L2.5s": a step above the L2s in terms of abstraction, but not quite at the
level of complete patterns or applications.  These constructs still largely
focus on a single logical resource -- in contrast to "patterns" which combine
multiple resources -- but are customized for a specific common usage scenario of
an L2. Examples of L2.5s in the CDK are `aws-apigateway.LambdaRestApi`,
`aws-lambda-nodejs.NodeJsFunction`, `aws-rds.ServerlessCluster` and `eks.FargateCluster`.

L2.5 constructs will be considered for inclusion in the CDK if they...

- cover a common usage scenario that can be used by a significant portion of
  the community;
- provide significant ease of use over the base L2 (via usage-specific defaults
  convenience methods or improved strong-typing);
- simplify or enable another L2 within the CDK

The CDK also currently includes some even higher-level constructs, which we call
patterns. These constructs often involve multiple kinds of resources and are
designed to help you complete common tasks in AWS or represent entire
applications. For example, the
`aws-ecs-patterns.ApplicationLoadBalancedFargateService` construct represents an
architecture that includes an AWS Fargate container cluster employing an
Application Load Balancer (ALB). These patterns are typically difficult to
design to be one-size-fits-all and are best suited to be published as separate
libraries, rather than included directly in the CDK.

## Mixins, Facades, and Traits

The AWS Construct Library uses three composable building blocks to provide
functionality around CFN Resource Constructs. Together with the CFN Resource
itself, these building blocks form an L2 construct. Each building block has a
distinct role and can be used independently of an L2.

### Mixins

Mixins are **inward-looking features** that extend a resource's own behavior.
They are composable abstractions applied to constructs via the `.with()` method.

A Mixin is a feature *of* the target resource. The defining question is: "is
this feature about the target resource?" If yes, it is a Mixin — regardless of
whether it sets properties on the L1, creates auxiliary resources (e.g. custom
resource handlers, delivery sources), or both. Mixins may accept other
constructs as props (e.g. a destination log group), but the feature remains
about the target resource.

Mixins usually operate on a single primary resource and can be applied to L1s,
L2s, or custom constructs alike. They are not designed for features that serve
an external consumer rather than the target resource — use a
[Facade](#facades) for that.

Examples: `BucketVersioning`, `BucketAutoDeleteObjects`, `BucketBlockPublicAccess`.

```ts
// Apply mixins to an L1
new s3.CfnBucket(this, 'Bucket')
  .with(new s3.mixins.BucketVersioning())
  .with(new s3.mixins.BucketBlockPublicAccess());

// Apply mixins to an L2 (delegates to the L1 default child)
new s3.Bucket(this, 'Bucket', { removalPolicy: RemovalPolicy.DESTROY })
  .with(new s3.mixins.BucketAutoDeleteObjects());
```

When to use a Mixin:

- The feature is *about* the target resource — it extends the resource's own
  behavior or lifecycle.
- The feature sets properties on the L1 resource (e.g. enabling versioning).
- The feature creates auxiliary resources that serve the primary resource (e.g.
  custom resource handlers, delivery sources, policy resources).
- The same feature should be applicable to both L1 and L2 constructs.
- You want to allow users to compose features independently of the L2
  construct's props.

Mixins are _not_ a replacement for construct properties. They cannot change the
optionality of properties or change defaults. When an L2 property simply passes
a value through to the L1 resource without additional logic, use `CfnPropsMixin`
in the L2 glue code instead of writing a standalone mixin.

For detailed implementation guidelines, see the
[Mixins Design Guidelines](./MIXINS_DESIGN_GUIDELINES.md).

### Facades

Facades are **resource-specific simplified interfaces that provide
integrations** for a resource with external consumers. They are standalone
classes with a static factory method (e.g., `fromBucket()` or `of()`) that
accepts a resource reference interface. Facades are exposed as properties on the
construct interface.

The defining characteristic of a Facade is directionality: a Facade serves an
*external consumer*, not the target resource. For example, `BucketGrants`
exists to serve the grantee (a role that needs access), not the bucket. Compare
this to a Mixin like `BucketAutoDeleteObjects`, which is a feature *of* the
bucket regardless of any external consumer.

Facades are always specific to a particular resource type — that is why it is
`BucketGrants` and not just `Grants`. While Facades for different resources look
similar, each contains resource-specific logic.

Some Facades are auto-generated and available for most resources (e.g.,
`BucketMetrics`, `BucketReflection`). Others are handwritten for resources that
need custom logic (e.g., `BucketGrants`). Because Facades are standalone classes
that only depend on the resource reference interface, third-party packages can
provide their own Facades for any resource without modifying `aws-cdk-lib`.

Examples: `BucketGrants` (handwritten), `BucketMetrics` (generated),
`BucketReflection` (generated).

```ts
// Facades are typically accessed through the construct interface
bucket.grants.read(role);

// Facades can also be used standalone with L1 constructs
const grants = BucketGrants.fromBucket(cfnBucket);
grants.read(role);
```

When to use a Facade:

- The feature serves an external consumer, not the target resource (e.g., IAM
  permissions serve the grantee, CloudWatch metrics serve the operator).
- The feature should work with both L1 and L2 constructs.
- The feature is not *about* the target resource's own behavior.

For detailed implementation guidelines, see the
[Facades and Traits Design Guidelines](./FACADES_AND_TRAITS_DESIGN_GUIDELINES.md).

The [Grants](#grants) section below describes the most common Facade in detail.

### Traits

Traits are **service-agnostic contracts** that describe a capability any
resource can have. They are factory interfaces that wrap L1 resources into
objects exposing higher-level interfaces. Unlike Facades which are specific to
one resource type, Traits are generic — `IResourceWithPolicyV2` can represent
any resource that has a resource policy, whether it is a bucket, a queue, or a
topic.

Traits enable Facades to discover capabilities of resources without requiring a
full L2 implementation. Examples: `IEncryptedResource` (via 
`IEncryptedResourceFactory`), `IResourceWithPolicyV2` (via 
`IResourcePolicyFactory`), `IConnectable`, `IGrantable`.

Traits are primarily an implementation detail used by Facades and the grant
system. They are not typically part of the public-facing API that end users
interact with directly.

For detailed implementation guidelines for both Facades and Traits, see the
[Facades and Traits Design Guidelines](./FACADES_AND_TRAITS_DESIGN_GUIDELINES.md).

### When to use which

| Question                                         | Mixin                           | Facade           | Trait            |
| ------------------------------------------------ | ------------------------------- | ---------------- | ---------------- |
| Is the feature *about* the target resource?      | yes                             | no               |                  |
| Does it serve an external consumer?              | no                              | yes              | yes              |
| Does it advertise a service-agnostic capability? | cross-service Mixins            | no               | yes              |
| Is it specific to one resource type?             | yes                             | yes              | no               |
| Should it work with L1 constructs?               | yes                             | yes              | yes              |
| Is it user-facing?                               | yes                             | yes              | rarely           |
| Primary builder audience                         | construct author & app builders | construct author | construct author |
| Primary user audience                            | app builder                     | app builder      | construct author |

For new features, prefer Mixins and Facades over adding methods or properties
directly to L2 constructs. Existing L2 constructs will continue to work and
may retain their current API for backward compatibility, but new functionality
should be implemented as one of these building blocks first.

## API Design

### Modules

AWS resources are organized into modules based on their AWS service. For
example, the "Bucket" resource, which is offered by the Amazon S3 service will
be available under the **aws-cdk-lib/aws-s3** module. We will use the “aws-” prefix
for all AWS services, regardless of whether their marketing name uses an
“Amazon” prefix (e.g. “Amazon S3”). Non-AWS services supported by AWS
CloudFormation (like the Alexa::ASK namespace) will be **@aws-cdk/alexa-ask**.

The name of the module is based on the AWS namespace of this service, which is
consistent with the AWS SDKs and AWS CloudFormation _[awslint:module-name]_.

All major versions of an AWS namespace will be mastered in the AWS Construct
Library under the root namespace. For example resources of the **ApiGatewayV2**
namespace will be available under the **aws-cdk-lib/aws-apigateway** module (and
not under “v2) _[awslint:module-v2]_.

In some cases, it makes sense to introduce secondary modules for a certain
service (e.g. aws-s3-notifications, aws-lambda-event-sources, etc). The name of
the secondary module will be
**aws-cdk-lib/aws-xxx-\<secondary-module\>**_[awslint:module-secondary]_.

Documentation for how to use secondary modules should be in the main module. The
README file should refer users to the central module
_[awslint:module-secondary-readme-redirect]_.

### Construct Class

Constructs are the basic building block of CDK applications. They represent
abstract cloud components of any complexity. Constructs in the AWS Construct
Library normally represent physical AWS resources (such as an SQS queue) but
they can also represent abstract composition of other constructs (such as
**LoadBalancedFargateService**).

Most of the guidelines in this document apply to all constructs in the AWS
Construct Library, regardless of whether they represent concrete AWS resources
or abstractions. However, you will notice that some sections explicitly call out
guidelines that apply only to AWS resources (and in many cases
enforced/implemented by the **Resource** base class).

AWS services are modeled around the concept of *resources*. Services normally
expose one or more resources through their APIs, which can be provisioned
through the APIs control plane or through AWS CloudFormation.

Every resource available in the AWS platform will have a corresponding resource
construct class to represents it. For example, the **s3.Bucket** construct
represents Amazon S3 Buckets, the **dynamodb.Table** construct represents an
Amazon DynamoDB table. The name of resource constructs must be identical to the
name of the resource in the AWS API, which should be consistent with the
resource name in the AWS CloudFormation spec _[awslint:resource-class]_.

> The _awslint:resource-class_ rule is a **warning** (instead of an error). This
  allows us to gradually expand the coverage of the library.

Classes which represent AWS resources are constructs and they must extend the
**cdk.Resource** class directly or indirectly
_[awslint:resource-class-extends-resource]_.

> Resource constructs are normally implemented using low-level CloudFormation
  (“CFN”) constructs, which are automatically generated from the AWS
  CloudFormation resource specification.

The signature (both argument names and types) of all construct initializers
(constructors) must be as follows _[awslint:construct-ctor]_:

```ts
constructor(scope: cdk.Construct, id: string, props: FooProps)
```

The **props** argument must be of type FooProps
[_awslint:construct-ctor-props-type_].

If all props are optional, the `props` argument must also be optional
_[awslint:construct-ctor-props-optional]_.

```ts
constructor(scope: cdk.Construct, id: string, props: FooProps = {})
```

> Using `= {}` as a default value is preferable to using an optional qualifier
  (`?`) since it will ensure that props will never be `undefined` and therefore
  easier to parse in the method body.

As a rule of thumb, most constructs should directly extend the **Construct** or
**Resource** instead of another construct. Prefer representing polymorphic
behavior through interfaces and not through inheritance.

Construct classes should extend only one of the following classes
[_awslint:construct-inheritance_]:

* The **Resource** class (if it represents an AWS resource)
* The **Construct** class (if it represents an abstract component)
* The **XxxBase** class (which, in turn extends **Resource**)

In the CDK, we can't reliably use `instanceof` for type comparison at runtime,
for a number of reasons: there may be multiple copies of the same package in 
`node_modules`, multiple versions in one tree, jsii cross-language boundaries.
In all these cases, `instanceOf` could return false even for the same shape (for
example, two occurrences of the `Stack` class, in different paths inside 
`node_modules`).

Instead, if you need to compare specific types, define a static type check
method called **isFoo** with the following implementation:

```ts
const IS_FOO = Symbol.for('@aws-cdk/aws-foo.Foo');

export class Foo {
  public static isFoo(x: any): x is Foo {
    return IS_FOO in x;
  }

  constructor(scope: Construct, id: string, props: FooProps) {
    super(scope, id);

    Object.defineProperty(this, IS_FOO, { value: true });
  }
}
```

### Construct Interface

One of the important tenets of the AWS Construct Library is to use strong-types
when referencing resources across the library. This is in contrast to how AWS
backend APIs (and, consequently, AWS CloudFormation) model reference via one of
their *runtime attributes* (such as the resource's ARN). Since the AWS CDK is a
client-side abstraction, we can offer developers a much richer experience by
using *object references* instead of *attribute references*.

Using object references instead of attribute references allows consumers of
these objects to have a richer interaction with the consumed object. They can
reference runtime attributes such as the resource's ARN, but also utilize logic
encapsulated by the target object.

Here's an example: when a user defines an S3 bucket, they can pass in a KMS key
that will be used for bucket encryption:

```ts
new s3.Bucket(this, 'MyBucket', { encryptionKey: key });
```

The **Bucket** class can now use **key.keyArn** to obtain the ARN for the key,
but it can also call the **key.grantEncrypt** method as a result of a call to
**bucket.grantWrite**. Separation of concerns is a basic OO design principle:
the fact that the Bucket class needs the ARN or that it needs to request
encryption permissions are not the user's concern, and the API of the Bucket
class should not “leak” these implementation details. In the future, the Bucket
class can decide to interact differently with the **key** and this won't require
expanding its surface area. It also allows the **Key** class to change its
behavior (i.e. add an IAM action to enable encryption of certain types of keys)
without affecting the API of the consumer.

#### Defining construct interfaces: interface types

In order to model the concept of owned and unowned constructs, and allow
multiple implementations of each, we will almost exclusively be referencing
constructs and resources by their *interfaces*, instead of by their *class*.

For every resource, a **reference interface** is automatically generated (ex.
`IBucketRef`). This interface defines the bare minimum information necessary
to "point to" a resource.

For example:

```ts
/**
 * Indicates that this resource can be referenced as a Bucket.
 */
export interface IBucketRef extends IConstruct, IEnvironmentAware {
  /**
   * A reference to a Bucket resource.
   */
  readonly bucketRef: BucketReference;
}

/**
 * A reference to a Bucket resource.
 */
export interface BucketReference {
  /**
   * The BucketName of the Bucket resource.
   */
  readonly bucketName: string;

  /**
   * The ARN of the Bucket resource.
   */
  readonly bucketArn: string;
}
```

Additionally, constructs typically implement a handwritten **resource
interface** (ex. `IBucket`). This interface extends the reference interface with
additional L2 API convenience functions _[awslint:construct-interface]_,
which give access to additional CloudFormation attributes or Facade classes:

For example:

```ts
interface IBucket extends IResource, IBucketRef {
  /**
   * The URL of the static website.
   * @attribute
   */
  readonly bucketWebsiteUrl: string;

  /**
   * Grant permissions on this bucket
   */
  readonly grant: BucketGrants;

  /**
   * Returns an ARN that represents an object in the bucket
   */
  arnForObjects(keyPattern: string): string;

  // ... more members here ...
}
```

#### Defining construct interfaces: what goes onto the resource interface

An L2 construct should provide additional features to its consumers.
Here is a (non-exhaustive) set:

- CloudFormation attribute getters (ex: `bucketWebsiteUrl`)
- Reflections on the state of the construct (ex: `isEncrypted`, `isVersioned`).
- Functions to calculate something (ex: `arnForObjects`)
- Functions to create new constructs that relate to the given resource
  (ex: `addEventNotification`)
- Functions to grant permissions to perform actions on the given resource (ex: `grantRead` or `grant.read`).
- Functions to build metric objects based on metrics of the given resource (ex: `metricBucketSize` or `metrics.bucketSize`).

Traditionally, we used to put all of these features directly on the L2 resource
interface; that led to huge L2 interfaces and it being impossible to benefit
from these features with alternative class implementations or L1 resource
classes.

We want those features to be used as widely as possible.
Therefore each feature is implemented in a separate class called a **Facade**
(see [Mixins, Facades, and Traits](#mixins-facades-and-traits)).
**They must not be added directly to the resource interface**.
New resource interfaces are designed like this;
old interfaces are being migrated to the new style over time (when we
migrate, the old functions remain in place and forward to the new style
implementation).

```ts
// This shows a hypothetical new design for the IBucket interface
// This is required for new interfaces and still have to migrate the old
// interfaces to the new style (in practice, `IBucket` hasn't been migrated yet)
export interface IBucket extends IResource, IBucketRef {

  // ✅ Attribute getters stay on the resource interface

  /**
   * The URL of the static website.
   * @attribute
   */
  readonly bucketWebsiteUrl: string;

  // 👉 Facade: functions that only need public information move to a separate class (planned)
  readonly helpers: BucketHelpers;

  // 👉 Facade: create new constructs that relate to this resource (planned)
  readonly create: BucketCreateHelpers;

  // 👉 Facade: grant permissions for a Bucket
  // The BucketGrants class can be code-generated (see the "Grants" section below).
  readonly grants: BucketGrants;

  // 👉 Facade: obtain metrics for a Bucket (planned)
  // The BucketMetrics class is automatically generated from an external source and does not need to be handwritten.
  readonly metrics: BucketMetrics;

  // 👉 Facade: derive state from the construct tree instead of storing input values (planned)
  readonly reflections: BucketReflection;
}

// This class is handwritten.
export class BucketHelpers {
  public static of(bucket: IBucketRef) { /* ... */ }

  /**
   * Returns an ARN that represents an object in the bucket
   */
  arnForObjects(keyPattern: string): string { /* ... */ }
}

// This class can be handwritten or generated (planned).
// Generation should be preferred for simple resources.
export class BucketCreateHelpers {
  public static of(bucket: IBucketRef) { /* ... */ }
}

// This class can be handwritten or generated.
// Generation should be preferred for simple resources.
export class BucketGrants {
  public static of(bucket: IBucketRef) { /* ... */ }

  /**
   * Grant read permissions for this bucket and it's contents to an IAM
   * principal (Role/Group/User).
   */
  public read(identity: IGrantable, objectsKeyPattern: any = '*') { /* ... */ }
}

// This class is generated.
export class BucketMetrics {
  public static of(bucket: IBucketRef) { /* ... */ }

  /**
   * The size of this bucket
   */
  public bucketSize(options?: MetricsOptions): IMetric { /* ... */ }
}

// Reflections derive state from the L1 configuration rather than storing input values.
// This class is handwritten using shared reflection helpers.
export class BucketReflection {
  public static of(bucket: IBucketRef) { /* ... */ }

  /**
   * Whether versioning is enabled on this bucket.
   */
  public get isVersioned(): boolean { /* ... */ }

  /**
   * Whether this bucket is encrypted.
   */
  public get isEncrypted(): boolean { /* ... */ }
}
```

#### Consuming construct interfaces: what interface type to choose

When accepting constructs and resources as parameters (both in a function, as
well as in a `Props` *struct* that is a parameter to a function), you should
accept them by an interface, not by their class. This makes it possible to
accept different types of objects that represent a real-world resource, each
with different capabilities. Some examples of different types of objects are:

- An L1
- An AWS-written L2
- A custom L2, for use inside an organization
- Not an **owned resource** (i.e., a resource created and managed by
  your CDK app and the underlying CloudFormation Stack) but a **referenced
  resource** (i a resource that already exist in the account, that we are
  allowed to point to but not change).

When choosing an interface to accept, we want to accept an as wide array as
possible of type of input resource objects.  This means your construct should
pose as little requirements on its parameters as possible.  You should prefer
interfaces with **fewer members** over interfaces with **more members**. "Fewer
members" means "fewer requirements", so easier to implement on a lot of classes!
Unfortunately, it also means fewer guarantees, so pick your interface type
according to what plan to do with the input construct.

Here are your choices, from most preferred to least preferred.

| Name                  | Example      | When to use                                                                                                                                                                                                             |
| --------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reference interface   | `IBucketRef` | A resource of a certain type, that you can only reference. You can get its name or ARN. (**Should be your default choice**)                                                                                             |
| L2 interface          | `IBucket`    | A resource of a certain type with convenience functions and additional attributes. Usually read-only, sometimes mutable. May or may not be "owned" (part of a Stack we are deploying) (**Most likely for legacy code**) |
| L2 Resource construct | `Bucket`     | A resource that is part of a Stack we are deploying. Has convenience functions and additional attributes, can be mutated. (**Only in exceptional cases**)                                                               |

Given the examples above, if you write a construct to accept a Bucket source you
you should prefer to accept `IBucketRef > IBucket > Bucket`, in that order.

There are 2 linter rules that enforce this:

* Prefer an interface over a concrete class _[awslint:ref-via-interface]_.
* Prefer a reference interface over an L2 interface _[awslint:prefer-ref-interface]_.

#### What if the entire L2 interface is too large, but the reference interface is too small?

A reference interface (`IBucketRef`, for example) indicates that an object
represents a Bucket resource, but the only thing it allows you to do is to name it.

How should you write your code if you need additional features, one of the ones
you would normally find on the traditional (and large) `IBucket` interface? That
depends on the type of feature.

**If the feature doesn't need the object's cooperation**: it might
be that the feature is just a convenience method that uses information that's
derivable from the reference interface already.

For example, `bucket.arnForObjects('file.zip')` just takes the bucket's name and
constructs a larger formatted string from it. You can write this as a standalone
helper function, or as a collection of helper functions on a class.

There should be a Facade class with these kinds of functions already for this
resource. If there isn't, write a new one.

If this applies to your use case, you can accept an `IBucketRef` type and
instantiate the Facade yourself.

```ts
var bucket: IBucket;
var bucketRef: IBucketRef;

// ❌ Requires a full implementation of IBucket for a small helper function
bucket.arnForObjects('file.zip')

// ✅ Free helper function, just accessible to you, the construct author
arnForObjects(bucketRef, 'file.zip')
// ✅ Even better, a collection of helper functions to you and library customers
BucketHelpers.for(bucketRef).arnForObjects('file.zip')
```

**If the feature needs the object's cooperation**: if you want to layer on
*limited additional features onto a minimal interface,
you can use [intersection
types](https://www.typescriptlang.org/docs/handbook/unions-and-intersections.html#intersection-types).
This requires an interface type describing just the part of the construct
surface area you need.

This requires an interface that exclusively describes the feature we want, and
nothing else. For example, `IGrantable` represents resources that can be granted
to. Most of the time when a construct takes a Role, we just need the role name
and we want to be able to grant to it.

To represent that, we can use the intersection `IRoleRef & IGrantable`:

```ts
export interface MyConstructProps {
  // ❌ Requires a full IRole while the only thing we actually need is `grantPrincipal`
  readonly role: IRole;

  // ✅ Just says "a Role resource that is also grantable"
  readonly role: IRoleRef & IGrantable;
}
```

**If the feature needs the object's cooperation and it hasn't been
generalized**: accept the resource interface for now (`IBucket`).

#### Abstract Base

It is recommended to implement an abstract base class **FooBase** for each
resource **Foo**. The base class would normally implement the entire
construct interface and leave attributes as abstract properties.

```ts
abstract class FooBase extends Resource implements IFoo {
  public abstract fooName: string;
  public abstract fooArn: string;

  // .. concrete implementation of IFoo (grants, metrics, factories),
  // should only rely on "fooName" and "fooArn" theoretically
}
```

The construct base class can then be used to implement the various
deserialization and import methods by defining an ad-hoc local class which
simply provides an implementation for the attributes (see “Serialization” below
for an example).

The abstract base class should be internal and not exported in the module's API
_[awslint:construct-base-is-private]_. This is only a recommended (linter
warning).

### Props

Constructs are defined by creating a new instance and passing it a set of
**props** to the constructor. Throughout this document, we will refer to these
as “props” (to distinguish them from JavaScript object properties).

 The props argument for the **Foo** construct should be a struct (interface with
 only readonly properties) named **FooProps** _[awslint:props-struct-name]_.

> Even if a construct props simply extends from some other Props struct and does
  not add any new properties, you should still define it, so it will be
  extensible in the future without breaking users in languages like Java where
  the props struct name is explicitly named.

Props are the most important aspect of designing a construct. Props are the
entry point of the construct. They should reflect the entire surface area of the
service through semantics that are intuitive to how developers perceive the
service and its capabilities.

When designing the props of an AWS resource, consult the AWS Console experience
for creating this resource. Service teams spend a lot of energy thinking about
this experience. This is a great resource for learning about the mental model of
the user. Aligning with the console also makes it easier for users to jump back
and forth between the AWS Console (the web frontend of AWS) and the CDK (the
“programmatic frontend” of AWS).

This section describes guidelines for construct props.

#### Types

Use **strong types** (and specifically, construct interfaces) instead of
physical attributes when referencing other resources. For example, instead of
**keyArn**, use **kms.IKey** [_awslint:props-no-arn-refs_].

Do not “leak” the details or types of the CFN layer when defining your construct
API. In almost all cases, a richer object-oriented API can be exposed to
encapsulate the low-level surface [_awslint:props-no-cfn-types_].

Do not use the **Token** type. It provides zero type safety, and is a functional
interface that may not translate cleanly in other JSII runtimes. Therefore, it should
be avoided wherever possible [_awslint:props-no-tokens_].

#### Defaults

A prop should be *required* only if there is no possible sensible default value
that can be provided *or calculated*.

Sensible defaults have a tremendous impact on the developer experience. They
offer a quick way to get started with minimal cognitive load, but do not limit users
from harnessing the full power of the resource, and customizing its behavior.

> A good way to determine what's the right sensible default is to refer to the
  AWS Console resource creation experience. In many cases, there will be
  alignment.

The **@default** documentation tag must be included on all optional properties
of interfaces.

In cases where the default behavior can be described by a value (typically the
case for booleans and enums, sometimes for strings and numbers), the value immediately
follows the **@default** tag and should be a valid JavaScript value (as in:
`@default false`, or `@default "stringValue"`).

In the majority of cases, the default behavior is not a specific value but
rather depends on circumstances/context. The default documentation tag must
begin with a “**-**" and then include a description of the default behavior
_[awslint:props-default-doc]_. This is specially true if the property
is a complex value or a reference to an object: don't write `@default
undefined`, describe the behavior that happens if the property is not
supplied.

Describe the default value or default behavior, even if it's not CDK that
controls the default. For example, if an absent value does not get rendered
into the template and it's ultimately the AWS *service* that determines the
default behavior, we still describe it in our documentation.

Examples:

```ts
// ✅ DO - uses a '-' and describes the behavior

/**
 * External KMS key to use for bucket encryption.
 *
 * @default - if encryption is set to "Kms" and this property is undefined, a
 * new KMS key will be created and associated with this bucket.
 */
encryptionKey?: kms.IEncryptionKey;
```

```ts
/**
 * External KMS key to use for bucket encryption.
 *
 * @default undefined
 *            ❌ DO NOT - that the value is 'undefined' by default is implied. However,
 *                        what will the *behavior* be if the value is left out?
 */
encryptionKey?: kms.IEncryptionKey;
```

```ts
/**
 * Minimum capacity of the AutoScaling resource
 *
 * @default - no minimum capacity
 *             ❌ DO NOT - there most certainly is. It's probably 0 or 1.
 *
 * // OR
 * @default - the minimum capacity is the default minimum capacity
 *             ❌ DO NOT - this is circular and useless to the reader.
 *                         Describe what will actually happen.
 */
minCapacity?: number;
```

#### Flat

Do not introduce artificial nesting for props. It hinders discoverability and
makes it cumbersome to use in some languages (like Java).

You can use a shared prefix for related properties to make them appear next to
each other in documentation and code completion:

For example, instead of:

```ts
new Bucket(this, 'MyBucket', {
  bucketWebSiteConfiguration: {
    errorDocument: '404.html',
    indexDocument: 'index.html',
  }
});
```

Prefer:

```ts
new Bucket(this, 'MyBucket', {
  websiteErrorDocument: '404.html',
  websiteIndexDocument: 'index.html'
});
```

The rule above targets nesting that only *groups* related properties. If grouping
is your goal, you can achieve it with the shared prefix, and avoid making life 
harder for other language users.

But the situation is different when you are dealing with co-dependent properties.
If you have a case in which two or more properties are only valid together —
none of them makes sense on its own — then nesting lets the type system reject 
the incomplete combinations. Here the ergonomic loss is outweighed by making an
illegal state unrepresentable rather than caught by a runtime check.

The rule of thumb is: if flattening the properties would force you to add
construction-time validation that throws when some — but not all — of them are
set, group them into a required-together object instead. If each property is
independently valid, keep them flat. (Properties that are mutually exclusive
rather than required-together are usually better modeled with factory methods or
subtypes than with a nested object — see [Prefer Additions](#prefer-additions)
and the enum-like class pattern.)

For example, a Glue Spark job's `workerType` and `numberOfWorkers` must be set
together. Flattened, the pair is severable, so the construct has to reject the
partial input at synthesis:

```ts
new PySparkEtlJob(this, 'Job', {
  workerType: WorkerType.G_2X, // throws at synth: numberOfWorkers is required with workerType
});
```

Grouping them into a value object whose fields are both required makes "one
without the other" impossible to express in the first place:

```ts
new PySparkEtlJob(this, 'Job', {
  workerConfiguration: {
    workerType: WorkerType.G_2X,
    numberOfWorkers: 2,
  },
});
```

#### Concise

Property names should be short and concise as possible and take into
consideration the ample context in which the property is used. Being concise
doesn't mean inventing new semantics. It just means that you can remove
redundant context from the property names. For example, there is no need to 
repeat the resource type, the property type or indicate that this is a 
"configuration". Prefer “readCapacity” versus “readCapacityUnits”.

#### Naming

We prefer the terminology used by the official AWS service documentation over
new terminology, even if you think it's not ideal. It helps users diagnose
issues and map the mental model of the construct to the service APIs,
documentation and examples. For example, don't be tempted to change SQS's
**dataKeyReusePeriod** with **keyRotation** because it will be hard for people
to diagnose problems. They won't be able to just search for “sqs dataKeyReuse”
and find topics on it.

> We can relax this guideline when this is about generic terms (like
  `httpStatus` instead of `statusCode`). The important semantics to preserve are
  for *service features*: I wouldn't want to rename "lambda layers" to "lambda
  dependencies" just because it makes more sense because then users won't be
  able to bind these terms to the service documentation.

Indicate units of measurement in property names that don't use a strong
type. Use “milli”, “sec”, “min”, “hr”, “Bytes”, “KiB”, “MiB”, “GiB” (KiB=1024
bytes, while KB=1000 bytes).

#### Property Documentation

Every prop must have detailed documentation. It is recommended to **copy** from
the official AWS documentation in English if possible so that language and style
will match the service.

#### Enums

When relevant, use enums to represent multiple choices.

```ts
export enum MyEnum {
  OPTION1 = 'op21',
  OPTION2 = 'opt2',
}
```

A common pattern in AWS is to allow users to select from a predefined set of
common options, but also allow the user to provide their own customized values.

A pattern for an "Enum-like Class" should be used in such cases:

```ts
export interface MyProps {
  readonly option: MyOption;
}

export class MyOption {
  public static COMMON_OPTION_1 = new MyOption('common.option-1');
  public static COMMON_OPTION_2 = new MyOption('common.option-2');

  public constructor(public readonly customValue: string) { }
}
```

Then usage would be:

```ts
new BoomBoom(this, 'Boom', {
  option: MyOption.COMMON_OPTION_1
});
```

Alternatively, you can also force users to go through static factory methods to
get values. This can be useful when there is some transformation you want to do
in the user-provided data:

```ts
export class MyOption {
  public static COMMON_OPTION_1 = new MyOption('common.option-1');
  public static COMMON_OPTION_2 = new MyOption('common.option-2');

  public static custom(value: string) {
    return new MyOption(value);
  }
  
  public static specialCase(value: string) {
    return new MyOption(value + '-some-suffix-or-whatever');
  }

  // 'protected' iso. 'private' so that someone that really wants to can still
  // do subclassing. But maybe might as well be private.
  protected constructor(public readonly value: string) { }
}

// Usage
new BoomBoom(this, 'Boom', {
  option: MyOption.COMMON_OPTION_1
});

new BoomBoom(this, 'Boom', {
  option: MyOption.custom('my-value')
});
```

#### Unions

Do not use TypeScript union types in construct APIs (`string | number`) since
many of the target languages supported by the CDK cannot strongly-model such
types _[awslint:props-no-unions]_.

Instead, use a class with static methods:

```ts
new lambda.Function(this, 'MyFunction', {
  code: lambda.Code.asset('/asset/path'), // or
  code: lambda.Code.bucket(myBucket, 'bundle.zip'), // or
  code: lambda.Code.inline('code')
  // etc
})
```

### Attributes

Every AWS resource has a set of "physical" runtime attributes such as ARN,
physical names, URLs, etc. These attributes are commonly late-bound, which means
they can only be resolved during deployment, when AWS CloudFormation actually
provisions the resource.

All properties that represent resource attributes must include the JSDoc tag
**@attribute** _[awslint:attribute-tag]_.

All attribute names must begin with the type name as a prefix (e.g. 
***bucket*Arn** instead of just **arn**). This implies that if a property begins
with the type name, it must have an **@attribute** tag.

All resource attributes must be represented as readonly properties of the
resource interface _[awslint:attribute-readonly]_.

Resource attributes should use a type that corresponds to the resolved AWS
CloudFormation type (e.g. **string**, **string[]**).

> Resource attributes almost always represent string values (URL, ARN,
  name). Sometimes they might also represent a list of strings. Since attribute
  values can either be late-bound ("a promise to a string") or concrete ("a
  string"), the AWS CDK has a mechanism called "tokens" which allows codifying
  late-bound values into strings or string arrays. This approach was chosen in
  order to dramatically simplify the type-system and ergonomics of CDK code. As
  long as users treat these attributes as opaque values (e.g. not try to parse
  them or manipulate them), they can be used interchangeably.

If needed, you can query whether an object includes unresolved tokens by using
the **Token.isUnresolved(x)** method.

To ensure users are aware that the value returned by attribute properties should
be treated as an opaque token, the JSDoc “@returns” annotation should begin with
“**@returns a $token representing the xxxxx**”.

### Configuration

When an app defines a construct or resource, it specifies its provisioning
configuration upon initialization. For example, when an SQS queue is defined,
its visibility timeout can be configured.

Naturally, when constructs are imported (unowned), the importing app does not
have control over its configuration (e.g. you cannot change the visibility
timeout of an SQS queue that you didn't create). Therefore, construct interfaces
cannot include methods that require access/mutation of configuration.

One of the problems with configuration mutation is that there could be a race
condition between two parts of the app, trying to set contradicting values.

There are a number use cases where you'd want to provide APIs which expose or
mutate the construct's configuration. For example,
**lambda.Function.addEnvironment** is a useful method that adds an environment
variable to the function's runtime environment, and used occasionally to inject
dependencies.

> Note that there are APIs that look like they mutate the construct, but in fact
  they are **factories** (i.e. they define resources on the user's stack). Those
  APIs _should_ be exposed on the construct interface and not on the construct
  class.

To help avoid the common mistake of exposing non-configuration APIs on the
construct class (versus the construct interface), we require that configuration
APIs (methods/properties) defined on the construct class will be annotated with
the **@config** jsdoc tag.

```ts
interface IFoo extends IConstruct {
  bar(): void;
}

class Foo extends Construct implements IFoo {
  public bar() { }

  @config
  public goo() { }

  public mutateMe() { } // ERROR! missing "@config" or missing on IFoo
}
```

#### Prefer Additions

As a rule of thumb, “adding” items to configuration props of type unordered
array is normally considered safe as it will unlikely cause race conditions. If
the prop is a map (like in **addEnvironment**), write defensive code that will
throw if two values are assigned to the same key.

#### Dropped Mutations

Since all references across the library are done through a construct's
interface, methods that are only available on the concrete construct class will
not be accessible by code that uses the interface type. For example, code that
accepts a **lambda.IFunction** will not see the **addEnvironment** method.

In most cases, this is desirable, as it ensures that only the code the owns the
construct (instantiated it), will be able to mutate its configuration.

However, there are certain areas in the library, where, for the sake of
consistency and interoperability, we allow mutating methods to be exposed on the
interface. For example, **grant** methods are exposed on the construct interface
and not on the concrete class. In most cases, when you grant a permission on an
AWS resource, the *principal's* policy needs to be updated, which mutates the
consumer. However, there are certain cases where a *resource policy* must be
updated. However, if the resource is unowned, it doesn't make sense (or even
impossible) to update its policy (there is usually a 1:1 relationship between a
resource and a resource policy). In such cases, we decided that grant methods
will simply skip any changes to resource policies, but will attach a
**permission notice** to the app, which will be printed when the stack is
synthesized by the toolkit.

### Factories

In most AWS services, there are one or more resources which can be referred to as
“primary resources” (normally one), while other resources exposed by the service
can be considered “secondary resources”.

For example, the AWS Lambda service exposes the **Function** resource, which can
be considered the primary resource while **Layer**, **Permission**, **Alias**
are considered secondary. For API Gateway, the primary resource is **RestApi**,
and there are many secondary resources such as **Method**, **Resource**,
**Deployment**, **Authorizer**.

Secondary resources are normally associated with the primary resource (i.e. a
reference to the primary resource must be supplied upon initialization).

Users should be able to define secondary resources either by directly
instantiating their construct class (like any other construct), and passing in a
reference to the primary resource's construct interface *or* it is recommended
to implement convenience methods on the primary resource that will facilitate
defining secondary resources. This improves discoverability and ergonomics.

For example, **lambda.Function.addLayer** can be used to add a layer to the
function, **apigw.RestApi.addResource** can be used to add to an API.

Methods for defining a secondary resource “Bar” associated with a primary
resource “Foo” should have the following signature:

```ts
export interface IFoo {
  addBar(...): Bar;
}
```

Notice that:

* The method has an “add” prefix.
  It implies that users are adding something to their stack.
* The method is implemented on the construct interface
  (to allow adding secondary resources to unowned constructs).
* The method returns a “Bar” instance (owned).

In order to reuse the set of props used to configure the secondary resource,
define a base interface for **FooProps** called **FooOptions** to allow
secondary resource factory methods to reuse props:

```ts
export interface LogStreamOptions {
  logStreamName?: string;
}

export interface LogStreamProps extends LogStreamOptions {
  logGroup: ILogGroup;
}

export interface ILogGroup {
  addLogStream(id: string, options?: LogStreamOptions): LogStream;
}
```

### Referenced resources

> "Referenced resources" were formerly called "imported resources", but that may lead to confusion
> because there is also a feature called "cdk import" that actually brings unowned
> resources under CloudFormation's control. Therefore, the current preferred terminology
> here has changed to "referencing" instead.

Construct classes should expose a set of static factory methods with a
“**from**” prefix that will allow users to reference *unowned* constructs into
their app.

The signature of all “from” methods should adhere to the following rules
_[awslint:from-signature]_:

* First argument must be **scope** of type **Construct**.
* Second argument is a **string**. This string will be used to determine the
  ID of the new construct. If the referencing method uses some value that is
  promised to be unique within the stack scope (such as ARN, export name),
  this value can be reused as the construct ID.
* Returns an object that implements the construct interface (**IFoo**).

#### “from” Methods

Resource constructs should export static “from” methods for referencing unowned
resources given one or more of its physical attributes such as ARN, name, etc. All
constructs should have at least one `fromXxx` method _[awslint:from-method]_:

```ts
static fromFooArn(scope: Construct, id: string, bucketArn: string): IFoo;
static fromFooName(scope: Construct, id: string, bucketName: string): IFoo;
```

> Since AWS constructs usually export all resource attributes, the logic behind
  the various “from\<Attribute\>” methods would normally need to convert one
  attribute to another. For example, given a name, it would need to render the
  ARN of the resource. Therefore, if **from\<Attribute\>** methods expect to be
  able to parse their input, they must verify that the input (e.g. ARN, name)
  doesn't have unresolved tokens (using **Token.isUnresolved**). Preferably, they
  can use **Stack.parseArn** to achieve this purpose.

If a resource has an ARN attribute, it should implement at least a **fromFooArn**
referencing method [_awslint:from-arn_].

To implement **fromAttribute** methods, use the abstract base class construct as
follows:

<!-- markdownlint-disable MD013 -->
```ts
class Foo {
  static fromArn(scope: Construct, fooArn: string): IFoo {
    class _Foo extends FooBase {
      public get fooArn() { return fooArn; }
      public get fooName() { return this.node.stack.parseArn(fooArn).resourceName; }
    }

    return new _Foo(scope, fooArn);
  }
}
```
<!-- markdownlint-enable MD013 -->

#### From-attributes

If a resource has more than a single attribute (“ARN” and “name” are usually
considered a single attribute since it's usually possible to convert one to the
other), then the resource should provide a static **fromAttributes** method to
allow users to explicitly supply values to all resource attributes when they
reference an external (unowned) resource [_awslint:from-attributes_].

```ts
static fromFooAttributes(scope: Construct, id: string, attrs: FooAttributes): IFoo;
```

### Roles

If a CloudFormation resource has a **Role** property, it normally represents the
IAM role that will be used by the resource to perform operations on behalf of
the user.

Constructs that represent such resources should conform to the following
guidelines.

An optional prop called **role** of type **iam.IRoleRef** should be exposed to allow
users to "bring their own role", and use either an owned or unowned role.

If the construct is going to grant permissions to the role, which is usually the case,
the type should include **iam.IGrantable**, in a type intersection as follows:

```ts
interface FooProps {
  /**
   * The role to associate with foo.
   *
   * @default - a role will be automatically created
   */
  role?: iam.IRoleRef & iam.IGrantable;
}
```

The construct interface should expose a **role** property, and extend
**iam.IGrantable**:

```ts
interface IFoo extends iam.IGrantable {
  /**
   * The role associated with foo. If foo is an unowned resource, no role will be available.
   */
  readonly role?: iam.IRoleRef;
}
```

This property will be `undefined` if this is an unowned construct (e.g. was not
defined within the current app).

An **addToRolePolicy** method must be exposed on the construct interface to
allow adding statements to the role's policy:

```ts
interface IFoo {
  addToRolePolicy(statement: iam.Statement): void;
}
```

If the construct is unowned, this method should no-op and issue a **permissions
notice** (TODO) to the user indicating that they should ensure that the role of
this resource should have the specified permission.

Implementing **IGrantable** brings an implementation burden of **grantPrincipal:
IPrincipal**. This property must be set to the **role** if available, or to a
new **iam.ImportedResourcePrincipal** if the resource is referenced and the role
is not available.

### Resource Policies

Resource policies are IAM policies defined on the side of the resource (as
oppose to policies attached to the IAM principal). Different resources expose
different APIs for controlling their resource policy. For example, ECR
repositories have a **RepositoryPolicyText** prop, SQS queues offer a
**QueuePolicy** resource, etc.

Constructs that represents resources with a resource policy should encapsulate
the details of how resource policies are created behind a uniform API as
described in this section.

When a construct represents an AWS resource that supports a resource policy, it
should expose an optional prop that will allow initializing resource with a
specified policy _[awslint:resource-policy-prop]_:

```ts
resourcePolicy?: iam.PolicyStatement[]
```

Furthermore, the construct *interface* should include a method that allows users
to add statements to the resource policy
_[awslint:resource-policy-add-to-policy]_:

```ts
interface IFoo extends iam.IResourceWithPolicy {
  addToResourcePolicy(statement: iam.PolicyStatement): void;
}
```

For some resources, such as ECR repositories, it is impossible (in
CloudFormation) to add a resource policy if the resource is unowned (the policy
is coupled with the resource creation). In such cases, the implementation of
`addToResourcePolicy` should add a **permission** **notice** to the construct
(using `node.addInfo`) indicating to the user that they must ensure that the
resource policy of that specified resource should include the specified
statement.

### VPC

Compute resources such as AWS Lambda functions, Amazon ECS clusters, AWS
CodeBuild projects normally allow users to specify the VPC configuration in
which they will be placed. The underlying CFN resources would normally have a
property or a set of properties that accept the set of subnets in which to place
the resources.

In most cases, the preferred default behavior is to place the resource in all
private subnets available within the VPC.

Props of such constructs should include the following properties
_[awslint:vpc-props]_:

```ts
/**
 * The VPC in which to run your CodeBuild project.
 */
vpc: ec2.IVpc; // usually this is required

/**
 * Which VPC subnets to use for your CodeBuild project.
 *
 * @default - uses all private subnets in your VPC
 */
vpcSubnetSelection?: ec2.SubnetSelection;
```

### Grants

Grants are one of the most powerful concepts in the AWS Construct Library. They
offer a higher level, intent-based, API for managing IAM permissions for AWS
resources. Grants are a type of [Facade](#facades) — they provide integrations
between a resource and IAM principals.

Grants for a given resource class are implemented in an accompanying class. For
example, **sns.Topic** has a corresponding **sns.TopicGrants** class. To get an
instance of the Grants class for a given resource, use the static **from<Resource>()**:

```ts
const role = new iam.Role(/*...*/);

const topic = new sns.Topic(/*...*/);
const grants = TopicGrants.fromTopic(topic);
```

The `fromTopic()` method accepts the resource reference interface (`sns.ITopicRef`),
which allows you to use the same class for both L1 and L2 constructs:

```ts
const topic = new sns.CfnTopic(/*...*/);
const grants = TopicGrants.fromTopic(topic);
```

The instance methods of the Grants classes can be used to grant a grantee
(such as an IAM principal) permission to perform a set of actions on the resource:

```ts
// Allow the role to publish and subscribe to the topic
topicGrants.publish(role);
topicGrants.subscribe(role);
```

For every resource that has an accompanying Grants class, the construct interface
should include a public **grants** property that returns an instance of the 
Grants class:

```ts
export abstract class TopicBase extends Resource implements ITopic, IEncryptedResource {
  ...
  public readonly grants: TopicGrants = TopicGrants.fromTopic(this);
  ...
}
```

#### Traits

To enable grant methods to work with L1 constructs, the CDK uses factory
interfaces called [Traits](#traits) that wrap L1 resources into objects
exposing higher-level interfaces:

- `IResourcePolicyFactory` wraps an L1 into an object implementing 
`IResourceWithPolicyV2`, enabling resource policy manipulation.
- `IEncryptedResourceFactory` wraps an L1 into an object implementing 
`IEncryptedResource`, enabling KMS key grants.

`IResourceWithPolicyV2` and `IEncryptedResource` are collectively called "traits". These are the two
traits currently in use. More may be added if other common patterns in L1 resources can be
abstracted through this mechanism.

The CDK provides default implementations for common L1 resources, but it's also possible to register custom factories
for any CloudFormation resource type:

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

// After this, every time the Grants class encounters a CfnResource of type 'AWS::Some::Type',
// it will be able to use MyFactory to attempt to add statements to its resource policy.
ResourceWithPolicies.register(scope, 'AWS::Some::Type', new MyFactory());
```

#### Auto-generation and manual implementation

The `TopicGrants` class, and many others, are generated automatically from the `grants.json`
file present at the root of each individual module (`packages/aws-sns` for SNS constructs and
so on). The `grants.json` file has the following general structure:

```json
{
  "resources": {
    "Topic": {
      "isEncrypted": true,
      "hasResourcePolicy": true,
      "grants": {
        "publish": {
          "actions": ["sns:Publish"],
          "keyActions": ["kms:Decrypt", "kms:GenerateDataKey*"],
          "docSummary": "Grant topic publishing permissions to the given identity"
        },
        "subscribe": {
          "actions": ["sns:Subscribe"],
          "arnFormat": "${topicArn}/*"
        }
      }
    }
  }
}
```

where:

* `Topic` - the class to generate grants for. This will lead to a class named TopicGrants.
* `isEncrypted` - indicates whether the resource is encrypted with a KMS key. When true, the `actions()` method will
have an `options` parameter of type `EncryptedPermissionOptions` that allows users to specify additional KMS permissions
to be granted on the key. If left undefined, but at least one grant method includes `keyActions`, the CDK will assume
that the resource is encrypted and the same behavior will apply. Note that if `isEncrypted` is explicitly set to false,
it is an error to specify `keyActions` in any of the grants.
* `hasResourcePolicy` - indicates whether the resource supports a resource policy. When true, all auto-generated methods in the Grants class will attempt to add statements to the resource policy when applicable. When false, the methods will only modify the principal's policy.
* `publish` - the name of a grant.
* `actions` - the actions to encompass in the grant.
* `keyActions` - if the resource has an associated KMS key, also grant these permissions on the key. Notice that the resource must implement the `iam.IEncryptedResource` interface for this to work.
* `docSummary` - the public documentation for the method.
* `arnFormat` - In some cases, the policy applies to a specific ARN patterns, rather than just the ARN of the resource.

Code generated from the `grants.json` file will have a very basic logic: it will try to add the given statement to the
principal's policy. If `hasResourcePolicy` is true, it will also attempt to add the statement to the resource policy.
This will only work if the resource implements the `iam.IResourceWithPolicyV2` interface or -- in case of L1s -- if
there is a `IResourcePolicyFactory` registered for its type (see previous section). If `keyActions` are specified in the
JSON file, it will also attempt to grant the specified permissions on the associated KMS key, if the resource implements
the `iam.IEncryptedResource` interface (or, similarly to resource policies, if there is a `IEncryptedResourceFactory`
registered for it).

If your permission use case requires additional logic, such as combining multiple `Grant` instances or handling
additional parameters, you will need to implement the Grants class manually.

Historically, grant methods were implemented directly on the resource construct interface (e.g.
`sns.ITopic.grantPublish(principal)`). For backward compatibility reasons, these methods are still
present on the resource interfaces, but new grant implementations are only allowed through the Grants
classes [_awslint:no-grants_].

### Metrics

Almost all AWS resources emit CloudWatch metrics, which can be used with alarms
and dashboards.

AWS construct interfaces should include a set of “metric” methods which
represent the CloudWatch metrics emitted from this resource.

At a minimum (and enforced by IResource), all resources should have a single
method called **metric**, which returns a **cloudwatch.Metric** object
associated with this instance (usually this method will simply set the right
metrics namespace and dimensions:

```ts
metric(metricName: string, options?: cloudwatch.MetricOptions): cloudwatch.Metric;
```

> **Exclusion**: If a resource does not emit CloudWatch metrics, this rule may
    be excluded

Additional metric methods should be exposed with the official metric name as a
suffix and adhere to the following rules:

* Name should be “metricXxx” where “Xxx” is the official metric name
* Accepts a single “options” argument of type **MetricOptions**
* Returns a **Metric** object

```ts
interface IFunction {
  metricDuration(options?: cloudwatch.MetricOptions): cloudwatch.Metric;
  metricInvocations(options?: cloudwatch.MetricOptions): cloudwatch.Metric;
  metricThrottles(options?: cloudwatch.MetricOptions): cloudwatch.Metric;
}
```

It is sometimes desirable to use a metric that applies to all resources of a
certain type within the account. To facilitate this, resources should expose a
static method called **metricAll**. Additional **metricAll** static methods can
also be exposed.

<!-- markdownlint-disable MD013 -->
```ts
class Function extends Resource implements IFunction {
  public static metricAll(metricName: string, options?: cloudwatch.MetricOptions): cloudwatch.Metric;
  public static metricAllErrors(props?: cloudwatch.MetricOptions): cloudwatch.Metric;
}
```
<!-- markdownlint-enable MD013 -->

### Events

Many AWS resources emit events to the CloudWatch event bus. Such resources should
have a set of “onXxx” methods available on their construct interface
_[awslint:events-in-interface]_.

All “on” methods should have the following signature
[_awslint:events-method-signature_]:

```ts
onXxx(id: string, target: events.IEventRuleTarget, options?: XxxOptions): cloudwatch.EventRule;
```

When a resource emits CloudWatch events, it should at least have a single
generic **onEvent** method to allow users to specify the event name
[_awslint:events-generic_]:

```ts
onEvent(event: string, id: string, target: events.IEventRuleTarget): cloudwatch.EventRule
```

### Connections

AWS resources that use EC2 security groups to manage network security should
implement the **connections API** interface by having the construct interface
extend **ec2.IConnectable** _[awslint:connectable-interface]_.

### Integrations

Many AWS services offer “integrations” with other services. For example, AWS
CodePipeline has actions that can trigger AWS Lambda functions, ECS tasks,
CodeBuild projects and more. AWS Lambda can be triggered by a variety of event
sources, AWS CloudWatch event rules can trigger many types of targets, SNS can
publish to SQS and Lambda, etc, etc.

> See [aws-cdk#1743](https://github.com/awslabs/aws-cdk/issues/1743) for a
  discussion on the various design options.

AWS integrations normally have a single **central** service and a set of
**consumed** services. For example, AWS CodePipeline is the central service and
consumes multiple services that can be used as pipeline actions. AWS Lambda is
the central service and can be triggered by multiple event sources.

Integrations are an abstract concept, not necessarily a specific mechanism. For
example, each AWS Lambda event source is implemented in a different way (SNS,
Bucket notifications, CloudWatch events, etc), but conceptually, *some* users
like to think about AWS Lambda as the “center”. It is also completely legitimate
to have multiple ways to connect two services on AWS. To trigger an AWS Lambda
function from an SNS topic, you could either use the integration or the SNS APIs
directly.

Integrations should be modeled using an **interface** (i.e. **IEventSource**)
exported in the API of the central module (e.g. “aws-lambda”) and implemented by
classes in the integrations module (“aws-lambda-event-sources”)
[_awslint:integrations-interface_].

```ts
// aws-lambda
interface IEventSource {
  bind(fn: IFunction): void;
}
```

A method “addXxx” should be defined on the construct interface and adhere to the
following rules _[awslint:integrations-add-method]:_

* Should accept any object that implements the integrations interface
* Should not return anything (void)
* Implementation should call “bind” on the integration object

```ts
interface IFunction extends IResource {
  public addEventSource(eventSource: IEventSource) {
    eventSource.bind(this);
  }
}
```

An optional array prop should allow declaratively applying integrations (sugar
to calling “addXxx”):

```ts
interface FunctionProps {
  eventSources?: IEventSource[];
}
```

Lastly, to ease discoverability and maintain a sane dependency graphs, all
integrations for a certain service should be mastered in a single secondary
module named aws-*xxx*-*yyy* (where *xxx* is the service name and *yyy* is the
integration name). For example, **aws-s3-notifications**,
**aws-lambda-event-sources**, **aws-codepipeline-actions**. All implementations
of the integration interface should reside in a single module
_[awslint:integrations-in-single-module]_.

```ts
// aws-lambda-event-sources
class DynamoEventSource implements IEventSource {
  constructor(table: dynamodb.ITable, options?: DynamoEventSourceOptions) { ... }

  public bind(fn: IFunction) {
    // ...do your magic
  }
}
```

When integration classes define new constructs in **bind**, they should be aware
that they are adding into a scope they don't fully control. This means they
should find a way to ensure that construct IDs do not conflict. This is a
domain-specific problem.

### State

Persistent resources are AWS resource which hold persistent state, such as
databases, tables, buckets, etc.

To make sure stateful resources can be easily identified, all resource
constructs must include the **@stateful** or **@stateless** JSDoc annotations at
the class level _[awslint:state-annotation]_.

This annotation enables the following linting rules.

```ts
/**
 * @stateful
 */
export class Table { }
```

Persistent resources must have a **removalPolicy** prop, defaults to
**Orphan** _[awslint:state-removal-policy-prop]_:

```ts
import { RemovalPolicy } from '@aws-cdk/cdk';

export interface TableProps {
  /**
   * @default ORPHAN
   */
  readonly removalPolicy?: RemovalPolicy;
}
```

Removal policy is applied at the CFN resource level using the
**RemovalPolicy.apply(resource)**:

```ts
RemovalPolicy.apply(cfnTable, props.removalPolicy);
```

The **IResource** interface requires that all resource constructs implement a
property **stateful** which returns **true** or **false** to allow runtime
checks query whether a resource is persistent
_[awslint:state-stateful-property]_.

### Physical Names - TODO

See <https://github.com/awslabs/aws-cdk/issues/2283>

### Tags

The AWS platform has a powerful tagging system that can be used to tag resources
with key/values. The AWS CDK exposes this capability through the **Tags**
"aspect", which can seamlessly tag all taggable resources within a subtree:

```ts
// add a tag to all taggable resources under "myConstruct"
Tags.of(myConstruct).add("myKey", "myValue");
```

Constructs for AWS resources that can be tagged must have an optional **tags**
hash in their props [_awslint:tags-prop_], wired straight through to the
underlying L1 default child.

Only L1 (`Cfn*`) resources implement the `ITaggable` / `ITaggableV2` interfaces —
those interfaces are auto-generated as part of the CloudFormation spec import.
L2 constructs MUST NOT implement `ITaggable` or `ITaggableV2` and MUST NOT
expose a `TagManager` on themselves; `TagManager.of(l2)` returning `undefined`
is intentional. `Tags.of(scope)` works on any `IConstruct` and walks the tree
to apply tags to every taggable L1 underneath, which is the only well-defined
tagging semantic for an L2 (an L2 typically aggregates multiple L1 resources,
so there is no single sensible target for tags applied "to the L2").

For the prescriptive agent-oriented version of this rule plus a worked
anti-pattern, see [AGENTS_CONSTRUCT_DESIGN.md § Tags](./AGENTS_CONSTRUCT_DESIGN.md#tags).

### Secrets

If you expect a secret in your API (such as passwords, tokens), use the
**cdk.SecretValue** class to signal to users that they should not include
secrets in their CDK code or templates.

If a property is named “password” it must use the **SecretValue** type
[_awslint:secret-password_].  If a property has the word “token” in it, it must
use the SecretValue type [_awslint:secret-token_].

## Project Structure

### Code Organization

* Code should be under `lib/`
* Entry point should be `lib/index.ts` and should only contain “imports”
  for other files.
* No need to put every class in a separate file. Try to think of a
  reader-friendly organization of your source files.

## Implementation

The following guidelines and recommendations apply are related to the
implementation of AWS constructs.

### General Principles

* Do not future proof.
* No fluent APIs.
* Good APIs “speak” in the language of the user. The terminology your API uses
  should be intuitive and represent the mental model your user brings over,
  not one that you made up and you force them to learn.
* Multiple ways of achieving the same thing is legitimate.
* Constantly maintain the invariants.
* The fewer “if statements” the better.

### Construct IDs

Construct IDs (the second argument passed to all constructs when they are
defined) are used to formulate resource logical IDs which must be **stable**
across updates. The logical ID of a resource is calculated based on the **full
path** of its construct in the construct scope hierarchy. This means that any
change to a logical ID in this path will invalidate all the logical IDs within
this scope. This will result in **replacements of all underlying resources**
within the next update, which is extremely undesirable.

As described above, use the ID “**Resource**” for the primary resource of an AWS
construct.

Therefore, when implementing constructs, you should treat the construct
hierarchy and all construct IDs as part of the **external contract** of the
construct. Any change to either should be considered and called out as a
breaking change.

There is no need to concatenate logical IDs. If you find yourself needing to
that to ensure uniqueness, it's an indication that you may be able to create
another abstraction, or even just a **Construct** instance to serve as a
namespace:

```ts
const privateSubnets = new Construct(this, 'PrivateSubnets');
const publicSubnets = new Construct(this, 'PublishSubnets');

for (const az of availabilityZones) {
  new Subnet(privateSubnets, az);
  new Subnet(publicSubnets, az, { public: true });
}
```

### Errors

#### Avoid Errors If Possible

Always prefer to do the right thing for the user instead of raising an
error. Only fail if the user has explicitly specified bad configuration. For
example, VPC has **enableDnsHostnames** and **enableDnsSupport**. DNS hostnames
*require* DNS support, so only fail if the user enabled DNS hostnames but
explicitly disabled DNS support. Otherwise, auto-enable DNS support for them.

#### Error reporting mechanism

There are three mechanism you can use to report errors:

* Eagerly throw an exception (fails synthesis)
* Attach a (lazy) validator to a construct (fails synthesis)
* Attach errors to a construct (succeeds synthesis, fails deployment)

Between these, the first two fail synthesis, while the latter doesn't. Failing synthesis
means that no Cloud Assembly will be produced.

The distinction becomes apparent when you consider multiple stacks in the same Cloud
Assembly:

* If synthesis fails due to an error in *one* stack (either by throwing an exception
  or by failing validation), the other stack can also not be deployed.
* In contrast, if you attach an error to a construct in one stack, that stack cannot
  be deployed but the other one still can.

Choose one of the first two methods if the failure is caused by a misuse of the API,
which the user should be alerted to and fix as quickly as possible. Choose attaching
an error to a construct if the failure is due to environmental factors outside the
direct use of the API surface (for example, lack of context provider lookup values).

#### Throwing exceptions

This should be the preferred error reporting method.

Validate input as early as it is passed into your code (ctor, methods,
etc) and bail out by throwing an `Error`. No need to create subclasses of
Error since all errors in the CDK are unrecoverable.

When validating inputs, don't forget to account for the fact that these
values may be `Token`s and not available for inspection at synthesis time.

Example:

```ts
if (!Token.isUnresolved(props.minCapacity) && props.minCapacity < 1) {
  throw new Error(`'minCapacity' should be at least 1, got '${props.minCapacity}'`);
}
```

#### Never Catch Exceptions

All CDK errors are unrecoverable. If a method wishes to signal a recoverable
error, this should be modeled in a return value and not through exceptions.

#### Attaching (lazy) Validators

In the rare case where the integrity of your construct can only be checked
after the app has completed its initialization, call the
**this.node.addValidation()** method to add a validation object. This will
generally only be necessary if you want to produce an error when a certain
interaction with your construct did *not* happen (for example, a property
that should have been configured over the lifetime of the construct, wasn't):

Always prefer early input validation over post-validation, as the necessity
of these should be rare.

Example:

```ts
this.node.addValidation({
  // 'validate' should return a string[] list of errors
  validate: () => this.rules.length === 0
    ? ['At least one Rule must be added. Call \'addRule()\' to add Rules.']
    : []
  }
});
```

#### Attaching Errors/Warnings

You can also “attach” an error or a warning to a construct via
the **Annotations** class. These methods (e.g., `Annotations.of(construct).addWarning`)
will attach CDK metadata to your construct, which will be displayed to the user
by the toolchain when the stack is deployed.

Errors will not allow deployment and warnings will only be displayed in
highlight (unless `--strict` mode is used).

```ts
if (!Token.isUnresolved(subnetIds) && subnetIds.length < 2) {
  Annotations.of(this).addError(`Need at least 2 subnet ids, got: ${JSON.stringify(subnetIds)}`);
}
```

#### Error messages

Think about error messages from the point of view of the end user of the CDK.
This is not necessarily someone who knows about the internals of your
construct library, so try to phrase the message in a way that would make
sense to them.

For example, if a value the user supplied gets handed off between a number of
functions before finally being validated, phrase the message in terms of the
API the user interacted with, not in terms of the internal APIs.

A good error message should include the following components:

* What went wrong, in a way that makes sense to a top-level user
* An example of the incorrect value provided (if applicable)
* An example of the expected/allowed values (if applicable)
* The message should explain the (most likely) cause and change the user can
  make to rectify the situation

The message should be all lowercase and not end in a period, or contain
information that can be obtained from the stack trace.

```ts
// ✅ DO - show the value you got and be specific about what the user should do
`supply at least one of minCapacity or maxCapacity, got ${JSON.stringify(action)}`

// ❌ DO NOT - this tells the user nothing about what's wrong or what they should do
`required values are missing`

// ❌ DO NOT - this error only makes sense if you know the implementation
`'undefined' is not a number`
```

### Tokens

Do not use `Fn::Sub` in constructs. It forces you to embed logical IDs and
references as literal strings, which couples your code to implementation
details and bypasses CDK's automatic reference and dependency tracking.
Compose strings from token-returning APIs instead (`Stack.of(scope).formatArn()`,
resource attribute properties like `bucket.bucketArn`, `Fn.join`, `Fn.ref`).

### Deferred Values

#### Box API (preferred for new code)

L2 constructs that accumulate state after construction (e.g., alarm actions, policy
statements, security group rules) should use the **Box API** instead of `Lazy`.
Boxes are mutable containers that implement `IResolvable` and capture stack traces
at mutation call sites, enabling accurate property-to-source-code attribution.

**Before (Lazy — legacy, do not use in new code):**

```ts
class MyL2 extends Resource {
  private readonly _actions: string[] = [];

  constructor(scope: Construct, id: string) {
    super(scope, id);
    new CfnResource(this, 'Resource', {
      // Stack trace captured HERE (constructor), not useful
      actions: Lazy.list({ produce: () => this._actions }),
    });
  }

  addAction(action: string) {
    this._actions.push(action); // No stack trace captured
  }
}
```

**After (Box — preferred):**

```ts
@noBoxStackTraces
class MyL2 extends Resource {
  private readonly _actions: IArrayBox<string> = Box.fromArray([]);

  constructor(scope: Construct, id: string) {
    super(scope, id);
    new CfnResource(this, 'Resource', {
      actions: Token.asList(this._actions),
    });
  }

  addAction(action: string) {
    this._actions.push(action); // Stack trace captured HERE — user's code
  }
}
```

Key rules:
- `Box.fromArray([])` for lists, `Box.fromValue(x)` for scalars, `Box.fromMap()` for maps, `Box.fromSet()` for sets
- Pass to L1 via `Token.asList(box)`, `Token.asString(box)`, `Token.asNumber(box)`, or `Token.asAny(box)` for complex/object values
- `Box.fromArray` resolves to `undefined` when empty by default (no manual empty-array → `undefined` mapping needed). Pass `{ omitEmpty: false }` as the second argument to resolve to an empty array instead
- Apply `@noBoxStackTraces` on classes that create or mutate Boxes in their constructor
- Use `box.derive(fn)` for single-source transforms, `Box.combine({ a: boxA, b: boxB }, ({ a, b }) => ...)` for multi-source derived values

See `packages/aws-cdk-lib/core/adr/box-api.md` for the full ADR.

#### Lazy (legacy)

`Lazy` remains available and is not deprecated, but new L2 constructs should prefer
Boxes for better debuggability. If you must use `Lazy`:

- Do not use a `Lazy` to perform a mutation on the construct tree
- `Lazy`s are resolved after the construct tree has been synthesized — mutating it
  at that point has non-obvious consequences
- For `Lazy.any()` wrapping arrays, pass `{ omitEmptyArray: true }` to resolve empty arrays to `undefined`

```ts
// DO NOT do this — bind() mutates the construct tree
this.lazyProperty = Lazy.any({
  produce: () => {
    return props.logging.bind(this, this); // BAD: tree mutation in Lazy
  },
});
```

## Documentation

Documentation style (copy from official AWS docs) No need to Capitalize Resource
Names Like Buckets after they've been defined Stability (@stable, @experimental)
Use the word “define” to describe resources/constructs in your stack (instead of
“~~create~~”, “configure”, “provision”).  Use a summary line and separate the
doc body from the summary line using an empty line.

### Inline Documentation

All public APIs must be documented when first introduced
[_awslint:docs-public-apis_].

Do not add documentation on overrides/implementations. The public reference
documentation will automatically copy the base documentation to the derived
APIs, so it's better to avoid the confusion.

Use the following JSDoc tags: **@param**, **@returns**, **@default**, **@see**,
**@example.**

### Readme

* Header should include maturity level.
* Example for the simple use case should be almost the first thing.
* If there are multiple common use cases, provide an example for each one and
  describe what happens under the hood at a high level
  (e.g. which resources are created).
* Reference docs are not needed.
* Use literate (`.lit.ts`) integration tests into README file.

## Testing

### Unit tests

* Unit test utility functions and object models separately from constructs. If
  you want them to be “package-private”, just put them in a separate file and
  import `../lib/my-util` from your unit test code.
* Failing tests should be prefixed with “fails”

### Integration tests

* Integration tests should be under `test/integ.xxx.ts` and should basically
  just be CDK apps that can be deployed using “cdk deploy” (in the meantime).

### Versioning

* Changing a construct's ID or its position in the scope hierarchy changes the
  derived CloudFormation logical ID, which replaces the underlying resources
  (causing data loss). Treat such changes as breaking changes under semantic
  versioning — they are only permitted in `-alpha` modules.

## Naming & Style

### Naming Conventions

* **Class names**: PascalCase
* **Properties**: camelCase
* **Methods (static and non-static)**: camelCase
* **Interfaces** (“behavioral interface”): IMyInterface
* **Structs** (“data interfaces”): MyDataStruct
* **Enums**: PascalCase, **Members**: SNAKE_UPPER

### Coding Style

* **Indentation**: 2 spaces
* **Line length**: 150
* **String literals**: use single-quotes (`'`) or backticks (```)
* **Semicolons**: at the end of each code statement and declaration
  (incl. properties and imports).
* **Comments**: start with lower-case, end with a period.
