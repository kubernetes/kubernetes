# Construct Design Rules — Mixins, Facades, Traits & API Design

> Linked from [AGENTS.md](../AGENTS.md). Read this when designing a new L2 construct, adding features to an existing one, or implementing mixins/facades/traits. For implementation details on grants, metrics, events, connections, IAM, and other cross-cutting patterns, see [AGENTS_CONSTRUCT_IMPLEMENTATION.md](./AGENTS_CONSTRUCT_IMPLEMENTATION.md).

---

## Feature Placement Decision

Start here. Use this table to pick the right pattern for every new feature:

| Pattern | When to use | When NOT to use | Details |
|---------|------------|-----------------|---------|
| **Mixin** | Feature is *about* the target resource — extends its behavior, lifecycle, or L1 props. Has logic beyond simple prop passthrough. | Feature serves an external consumer (→ Facade). Feature advertises a service-agnostic capability (→ Trait). Feature is a simple L1 prop passthrough with no logic (→ CfnPropsMixin). Do not use to change optionality of construct properties or defaults. | [§ Mixins](#mixins) |
| **CfnPropsMixin** | L2 property simply passes a value through to the L1 resource without additional logic. Used in L2 glue code. | Feature has validation, creates auxiliary resources, or contains any logic (→ standalone Mixin). Feature serves an external consumer (→ Facade). | [§ Mixins — L1 Property Merge Strategy](#l1-property-merge-strategy) |
| **Facade** | Feature serves an external consumer, not the target resource. Resource-specific. Examples: Grants (serves the grantee), Metrics, Events. | Feature is about the target resource's own behavior (→ Mixin). Feature is service-agnostic (→ Trait). Do not use for equal-peer integrations. | [§ Facades](#facades) |
| **Trait** | Feature advertises a service-agnostic capability any resource can have (e.g., "has a resource policy", "is encryptable"). Not specific to one resource type. | Feature is specific to a single resource type (→ Facade or Mixin). Feature is user-facing (Traits are rarely user-facing — they are consumed by Facades and the grant system). | [§ Traits](#traits) |
| **L2 construct method** | None of the above apply. Feature is construct-specific glue that doesn't fit a building block. | Prefer building blocks first — L2 glue code MUST contain only prop mapping, defaults, and wiring (no feature logic). | [§ API Design — Methods & Mutation](#api-design--methods--mutation) |

You MUST implement new features as building blocks first, optionally exposing through L2 props for convenience.

---

## Mixins

### Scope & Placement

- You MUST use a Mixin only when the feature is *about* the target resource — extending its own behavior, lifecycle, or L1 properties
- You MUST NOT use a Mixin for features that serve an external consumer, for equal-peer integrations, or to change the optionality of construct properties or defaults — use a Facade or `CfnPropsMixin` for those cases instead

### Class Structure

- You MUST extend the `Mixin` base class from `aws-cdk-lib`
- You MUST implement `supports(construct)` as a type guard and `applyTo(construct)` — the framework automatically delegates from L2 to the L1 default child via `.with()`
- See [AGENTS_CONSTRUCT_IMPLEMENTATION.md § Mixin Implementation](./AGENTS_CONSTRUCT_IMPLEMENTATION.md#mixin-implementation) for code patterns

### Naming Conventions

- You MUST prefix mixin class names with the target resource name: `BucketVersioning` not `Versioning`

### Import Pattern

- Users access mixins as `{service}.mixins.{MixinName}` (e.g., `s3.mixins.BucketVersioning`)

### Validation

- You MUST validate mixin inputs as early as possible — constructor, `applyTo()`, and deferred via `node.addValidation()`
- You SHOULD collect multiple errors and throw them as a group
- See [AGENTS_CONSTRUCT_IMPLEMENTATION.md § Mixin Validation](./AGENTS_CONSTRUCT_IMPLEMENTATION.md#mixin-validation) for the three-phase pattern

### L1 Property Merge Strategy

- You MUST consider how a mixin interacts with existing L1 configuration (set by user, L2, or other mixins)
- You SHOULD use `CfnPropsMixin` with `PropertyMergeStrategy` instead of modifying properties directly
- You MUST document the merge behavior in the mixin's JSDoc
- See [AGENTS_CONSTRUCT_IMPLEMENTATION.md § Mixin Merge Strategy](./AGENTS_CONSTRUCT_IMPLEMENTATION.md#mixin-merge-strategy) for usage details

### Documentation

- You MUST include a `## Mixins` section in the service module README documenting each mixin with a brief description and usage example

---

## Facades

- You MUST use a Facade when the feature serves an external consumer (not the target resource)
- Each Facade MUST be specific to a single resource type, named `{Resource}{Feature}` (e.g., `BucketGrants` not `Grants`)
- Facades are standalone classes with a `static from{Resource}(resource: IFooRef)` factory method accepting the reference interface
- Facades are exposed as readonly properties on the construct interface
- You SHOULD prefer auto-generation for Metrics, Grants, and Create Helpers Facade classes on simple resources — handwrite only when custom logic is needed
- Reflections MUST derive state from the L1 configuration (construct tree), not from stored input values

Common facade types — see [AGENTS_CONSTRUCT_IMPLEMENTATION.md](./AGENTS_CONSTRUCT_IMPLEMENTATION.md) for implementation rules:
- **Grants** (`{Resource}Grants`) — IAM permission helpers, e.g., `BucketGrants.grantRead(grantee)`
- **Metrics** — CloudWatch metric factories, e.g., `metric(metricName, options?)`
- **Events** — CloudWatch event rule factories, e.g., `onXxx(id, target, options?)`

---

## Traits

- You MUST design Traits as service-agnostic contracts describing a capability any resource can have (e.g., "has a resource policy") — see `core/lib/helpers-internal/traits.ts`
- Traits MUST NOT be specific to a single resource type
- Traits SHOULD rarely be user-facing — they are implementation details consumed by Facades and the grant system
- Traits SHOULD be registered as factories so Facades can discover capabilities on L1 resources without requiring a full L2

---

## API Design — Modules & Naming

### Module Organization

- You MUST organize AWS resources into modules under `aws-cdk-lib/aws-{service}` using the `aws-` prefix regardless of marketing name
- You MUST place all major versions under the root namespace — not version-suffixed modules
- You MUST name secondary modules as `aws-{service}-{secondary}`
- Secondary module documentation MUST redirect to the main module's README

### Alpha Modules

- Alpha modules live in `packages/@aws-cdk/aws-{service}-alpha/` — separate packages from `aws-cdk-lib`
- Breaking changes are allowed — version is `0.0.0`, stability is `experimental`
- You MUST peer-depend on `aws-cdk-lib` and `constructs` — import as `from 'aws-cdk-lib/aws-{service}'`, not relative paths
- You MUST colocate integration tests in `test/` (not in `@aws-cdk-testing/framework-integ`)
- You MUST include a `rosetta/default.ts-fixture` for README code compilation

### Resource & Type Naming

- You MUST name resource construct classes identically to the AWS API/CloudFormation resource name (e.g., `Bucket`, `Table`)
- You MUST derive all related types from this name: `FooProps`, `IFoo`, `IFooRef`
- PascalCase for classes/enums, camelCase for properties/methods, `I`-prefix for behavioral interfaces, no `I`-prefix for data interfaces (structs), SNAKE_UPPER_CASE for enum members

### Property Naming

- You MUST use official AWS service terminology — do not rename service-specific terms
- Keep names concise by removing redundant context (resource type, property type, "configuration") without inventing new semantics
- Include units of measurement in names when not using a strong type: `milli`, `sec`, `min`, `hr`, `Bytes`, `KiB`, `MiB`, `GiB`

### Property Structure (flat vs. nested)

- You MUST keep props flat by default — do not introduce nesting that only *groups* related props. Use a shared name prefix instead (`websiteErrorDocument`/`websiteIndexDocument`, not `websiteConfiguration: { ... }`)
- EXCEPTION: You SHOULD group two or more **co-dependent** props (only valid together, none meaningful alone) into a required-together nested value object when doing so makes the incomplete combinations unrepresentable at compile time. Making an illegal state unrepresentable outranks flatness here
- Litmus test: if flattening would force construction-time validation that throws when *some but not all* of the props are set, nest them; if each prop is independently valid, keep it flat
- Mutually *exclusive* props are NOT this case — model those with factory methods or enum-like classes (a nested object cannot enforce exclusivity)
- Example: a Glue Spark job's `workerType` + `numberOfWorkers` are required-together → `workerConfiguration: { workerType, numberOfWorkers }` instead of two severable top-level props guarded by a synth-time throw

### Default Behavior

- You MUST define the default behavior for every optional prop — what happens when the user omits it is a design decision, not just a documentation task. The `@default` JSDoc tag documents it, but the behavior itself must be intentionally designed.

### Resource Name Props

- Whether a resource accepts a `{resource}Name` prop and whether it's required or optional is a design decision
- You MUST define what happens when the name is omitted (auto-generated? error?) — default behavior is part of the user contract

---

## API Design — Construct Structure

### Base Classes & Inheritance

- You MUST extend only `Resource` (for AWS resources), `Construct` (for abstract components), or `{Foo}Base` (which extends `Resource`)
- Prefer direct extension over deep inheritance hierarchies
- Represent polymorphic behavior through interfaces, not inheritance

### Constructor

- You MUST use the standard signature: `constructor(scope: Construct, id: string, props: FooProps)`
- Default `props = {}` (not `?`) when all properties are optional

### Static Type Check

- You SHOULD define a static `isFoo(x: any): x is Foo` method on every construct class — do not use `instanceof`
- See [AGENTS_CONSTRUCT_IMPLEMENTATION.md § Static Type Check](./AGENTS_CONSTRUCT_IMPLEMENTATION.md#static-type-check) for the `Symbol.for` pattern

---

## API Design — Interfaces & Type Hierarchy

### Construct Interface (`IFoo`)

- You MUST define `IFoo` for every resource
- CloudFormation attribute getters go directly on the interface
- All features (grants, metrics, helpers, reflections, create-helpers) MUST be implemented as separate Facade classes exposed as readonly properties
- You MUST NOT add new feature methods directly to the resource interface

### Reference Interface (`IFooRef`)

- `IFooRef` MUST contain only the bare minimum identifiers needed to point to a resource (typically name and ARN)
- Do not add convenience methods or additional attributes
- `IFooRef` is auto-generated from the CloudFormation spec (e.g., `IBucketRef` in `s3.generated.ts`) — do not manually define it, just import and consume it

### Accepting Resources as Parameters

- You MUST accept constructs by their interface type (not concrete class), preferring the narrowest interface:
  1. `IFooRef` (default) — when you only need identifiers
  2. `IFoo` — when you need convenience functions
  3. `Foo` — only in exceptional cases
- Use intersection types (e.g., `IRoleRef & IGrantable`) when you need limited additional capabilities
- Instantiate Facades yourself when the reference interface is sufficient

### Resource Attributes

- You MUST expose all CloudFormation resource attributes as readonly properties on the resource interface
- Prefix with the type name: `bucketArn` not `arn`, `functionName` not `name`
- Annotate each attribute property with `@attribute` JSDoc tag
- Treat attribute values as opaque tokens — do not parse or manipulate

---

## API Design — Methods & Mutation

### Configuration Mutation

- You MUST NOT include configuration mutation methods on the construct interface (`IFoo`) — they belong on the concrete class only (imported constructs cannot be reconfigured)
- Annotate mutation methods with `@config` JSDoc tag
- Exception: grant methods and factory methods SHOULD be on the interface because they serve external consumers or create new resources

### Method Verb Semantics

- Method verb choice carries semantic meaning: `add` implies the parent owns the child's lifecycle, `create` implies a new standalone resource, `define` implies declarative configuration — choosing the wrong verb is a design error, not just a naming issue

### Factory Methods for Secondary Resources

- You SHOULD implement convenience `add{Bar}(...)` factory methods on the construct interface for creating associated secondary resources
- The method MUST return the secondary resource instance
- You MUST define a `BarOptions` base interface (without the primary resource reference) that `BarProps` extends, so factory methods accept `BarOptions` with the primary resource implicit
- Factory methods SHOULD live on the construct class (not input types), cover the full capability of the underlying API
- You SHOULD prefer extending existing methods with parameters over adding new methods
- Before adding `addXxx()` to large interfaces, consider standalone constructs or Facades

### Import (`from*`) Methods

- You MUST provide at least one static `from{Attribute}` method on every resource construct for importing unowned resources
- Signature: `(scope: Construct, id: string, ...): IFoo`
- Resources with an ARN MUST have `fromFooArn`
- Resources with multiple independent attributes MUST have `fromFooAttributes(scope, id, attrs: FooAttributes)`
- Imported constructs MUST only set properties meaningful for the imported resource — no placeholder objects for unavailable properties
- `from*` methods MUST NOT transform or validate identifiers (they are pass-through)
- When `fromAttributes` receives multiple identifying attributes, prefer the most specific one (ARN) over throwing

### `fromLookup` Methods

- When implementing `fromLookup` methods via context providers, the return path MUST preserve the full resource identity (region, account)
- Prefer `fromResourceAttributes()` over `fromResourceName()` to avoid reconstructing ARNs with the wrong environment

---

## Standard Interface Extensions

These are design decisions about what interfaces and props every L2 construct MUST or SHOULD include. For implementation details, see [AGENTS_CONSTRUCT_IMPLEMENTATION.md](./AGENTS_CONSTRUCT_IMPLEMENTATION.md).

### CloudWatch Metrics

- You MUST expose a generic `metric(metricName, options?)` method, named `metricXxx` methods using official metric names, and a static `metricAll` method for account-wide metrics on all resource constructs that emit CloudWatch metrics

### CloudWatch Events

- You MUST expose `onXxx(id, target, options?)` methods and a generic `onEvent(event, id, target)` method on the construct interface for resources that emit CloudWatch events

### Connections

- You MUST have the construct interface extend `ec2.IConnectable` for resources that use EC2 security groups to manage network security

### VPC Placement

- You SHOULD include `vpc: ec2.IVpc` (usually required) and optional `vpcSubnetSelection?: ec2.SubnetSelection` props on compute constructs that support VPC placement, defaulting to all private subnets

### IAM Role Integration

- You MUST expose an optional `role: iam.IRole` prop and a readonly `role?: iam.IRole` property on the interface (undefined for imported constructs)
- You MUST extend `iam.IGrantable` and provide `addToRolePolicy(statement)` on the interface

### IAM Resource Policy Integration

- You SHOULD expose an optional `resourcePolicy` prop and have the interface extend `iam.IResourceWithPolicy` with `addToResourcePolicy(statement)`

### Integration Pattern

- You MUST define integration interfaces (e.g., `IEventSource`) with a `bind` method in the central module
- Expose an `addXxx` method on the construct interface that accepts the integration interface and calls `bind`
- Include an optional array prop for declarative application

### Stateful/Stateless & Removal Policy

- You MUST annotate every resource construct with `@stateful` or `@stateless` JSDoc
- You MUST implement the `stateful` property on `IResource`
- You MUST include a `removalPolicy?: RemovalPolicy` prop (defaulting to `ORPHAN`) on stateful resources

### Tags

- You MUST include an optional `tags` hash prop on taggable resources, and you MUST plumb it straight through to the L1 default child (e.g. `tags: props.tags`)
- You MUST NOT make an L2 implement `ITaggable` or `ITaggableV2`. Only L1 (`Cfn*`) resources implement those interfaces — they are auto-generated for every taggable CloudFormation resource. `ITaggableV2` is the modern variant for L1s where the auto-generator could not produce a `tags: TagManager` field
- You MUST NOT expose a `tags: TagManager` or `cdkTagManager: TagManager` field on an L2. `TagManager.of(l2)` is expected to return `undefined` — that is the correct, intentional behavior
- The user-facing API for applying tags to any scope is `Tags.of(scope: IConstruct).add(key, value)`. It works on any `IConstruct` (L2s, Stages, Stacks, the App) — it traverses the construct tree via aspects and applies tags to every taggable L1 underneath, regardless of whether the scope itself is taggable
- Tagging an L2 directly has no single well-defined meaning (tag only the primary resource? all aggregated resources? ones added later via mutation methods?). It is deliberately not modeled — the traversal-from-`Tags.of` semantics are the only supported answer

**❌ Anti-pattern — do not do this:**

```ts
// Adding ITaggableV2 to an L2 just to make `TagManager.of(dashboard)` return something.
export class Dashboard extends Resource implements cdk.ITaggableV2 {
  public readonly cdkTagManager: cdk.TagManager;
  // ...
}
```

This is wrong because: (1) an L2 typically aggregates multiple L1 resources, so a single `TagManager` on the L2 has no defined target; (2) `Tags.of(dashboard).add(...)` already does the right thing without it — it traverses and tags every taggable L1 underneath; (3) it introduces an `ITaggableV2`-conforming surface that consumers may start to rely on, locking in semantics that were never designed.

**✅ Correct pattern:**

```ts
// L2: just expose a `tags` prop and pass it to the L1.
export class Dashboard extends Resource {
  constructor(scope: Construct, id: string, props: DashboardProps) {
    super(scope, id);
    new CfnDashboard(this, 'Resource', {
      // ...
      tags: props.tags, // L1 owns ITaggable — that's the only place it lives
    });
  }
}

// User code:
Tags.of(myDashboard).add('Environment', 'prod'); // traverses and tags all L1s underneath
```

### Secrets

- You MUST use `cdk.SecretValue` for any prop that accepts a secret value — any property named `password` or containing the word `token` MUST use `SecretValue`

---

## Type Design Principles

### Polymorphism over Booleans

- You SHOULD model behavioral variation through polymorphism (abstract methods, separate classes, or static factory methods) rather than boolean flags
- Prefer the simplest type that captures the design intent — wrapper classes and class hierarchies are only justified when they provide real type safety or extensibility

### Interface Design

- You MUST ensure all inherited properties in a props interface are valid for the specific construct
- Interface members SHOULD only exist if consumed by external code
- When a class has multiple factory methods with different requirements, use separate interfaces rather than shared interfaces with runtime validation

### Type Reuse

- You SHOULD reuse existing types across modules when semantically equivalent
- Prefix interface names with the service domain to avoid ambiguity (e.g., `IBedrockInvokable`)
- Ensure enum completeness across all supported variants of a platform or engine

### Enums & Enum-Like Classes

- You SHOULD use TypeScript enums for fixed choice sets where the options are fully known and unlikely to change (e.g., `BucketEncryption`)
- You SHOULD use enum-like classes — a class with static members for common options plus a public or protected constructor accepting a raw string — when users need both predefined options and the ability to specify custom values (e.g., `ec2.InstanceType` with `InstanceType.of(...)` and `new InstanceType('c5.xlarge')`)

---

## Backward Compatibility & Deprecation

- You MUST NOT change any public-facing code that breaks backward compatibility (APIs, method signatures, validations, enum members, prop contracts, making required properties optional)
- When fixing defects or extending constructs for new modes, deprecate the old version and introduce a corrected replacement (e.g., `FooV2`) rather than modifying the original
- You MUST NOT include `@deprecated` properties in newly introduced interfaces
- You SHOULD keep implementation-detail types unexported (marked `@internal`) to minimize backward-compatibility obligations and public API surface
- You MUST document jsii compatibility changes in `allowed-breaking-changes.txt` with the correct entry type (`changed-type`, `strengthened`, `weakened`, or `removed`)
- Warning and error IDs passed to `Annotations.addWarningV2()` are part of the public API contract — changing them is a breaking change because consumers may reference them for suppression or filtering
