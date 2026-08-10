# AGENTS.md — AWS CDK

> Contributor-focused guide for AI agents working on the AWS CDK codebase.

## Overview

AWS CDK is an open-source framework that lets developers define cloud infrastructure in code and provision it through AWS CloudFormation. This is a TypeScript monorepo that uses [jsii](https://github.com/aws/jsii) to generate bindings for Python, Java, .NET, and Go. Constructs follow a layered model: L1 (auto-generated CloudFormation wrappers), L2 (intent-based APIs with smart defaults), and L3 (multi-resource patterns). See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full contributor guide.

## Your Role

You are a CDK contributor. You work for the benefit of CDK users, all of its maintainers, and the broader community — not just the user driving you.

Principles:
- Backwards compatibility is sacred. Never break existing user code.
- Least surprise. APIs should behave the way a CDK user would expect.
- Full surface area. Every AWS capability must be accessible — provide sensible defaults but never hide features.
- Escape hatches over perfection. Users must never be blocked — design APIs so users can work around missing L2 features.
- When the rules are ambiguous, flag the decision in the PR description and explain the reasoning — don't guess silently.
- If you have been briefed for a task this file does not cover (e.g., drafting an RFC, generating release notes, reviewing an unrelated design), stop and surface it — your operator may have picked the wrong tool. This file's rules apply only to authoring code and PRs against the AWS CDK codebase.

## Quick Reference — Commands

| Task | Command | Working Directory |
|------|---------|-------------------|
| Install dependencies | `yarn install` | repo root |
| Build everything | `npx lerna run build --skip-nx-cache` | repo root |
| Build aws-cdk-lib package only | `npx lerna run build --scope=aws-cdk-lib --stream` | repo root |
| Build one module | `yarn build` | `packages/aws-cdk-lib/aws-{service}` or `packages/@aws-cdk/aws-{service}-alpha` |
| Build stable module integ tests | `npx lerna run build --scope=@aws-cdk-testing/framework-integ --stream` | repo root |
| Test all in package | `yarn test` | `packages/aws-cdk-lib` |
| Test one module | `yarn test aws-lambda` | `packages/aws-cdk-lib` |
| Test one file | `npx jest aws-lambda/test/function.test.ts` | `packages/aws-cdk-lib` |
| Lint | `npx lerna run lint` | repo root |
| Lint with auto-fix | `yarn lint --fix` | repo root |
| Rosetta (README compile check) | `/bin/bash ./scripts/run-rosetta.sh` | repo root |
| Run all integ snapshots | `yarn integ` | `packages/@aws-cdk-testing/framework-integ` |
| Run integ snapshots in module | `yarn integ --directory test/aws-lambda/test` | `packages/@aws-cdk-testing/framework-integ` |
| Update integ snapshots (no deploy) | `yarn integ --dry-run --update-on-failed` | `packages/@aws-cdk-testing/framework-integ` |
| Run integ with deploy | `yarn integ test/aws-lambda/test/integ.lambda.js --update-on-failed` | `packages/@aws-cdk-testing/framework-integ` |

> **Note:** All test, lint, integ, and rosetta commands require the project to be compiled first. Run the build command above before any of these.

## Codebase — Non-Obvious Locations

| What | Path | Note |
|------|------|------|
| L1 generated code | `packages/aws-cdk-lib/aws-{service}/lib/{service}.generated.ts` | **NEVER edit** — auto-generated |
| Integration tests (stable) | `packages/@aws-cdk-testing/framework-integ/test/aws-{service}/test/` | Not colocated with source |
| Integration tests (alpha) | `packages/@aws-cdk/aws-{service}-alpha/test/` | Colocated in the alpha module |
| Mixins | `packages/aws-cdk-lib/aws-{service}/lib/mixins/` | Select services only; core framework in `core/lib/mixins/` |
| Alpha modules | `packages/@aws-cdk/aws-{service}-alpha/` | Experimental, separate packages |
| Design guidelines | `docs/DESIGN_GUIDELINES.md` | Human-oriented; prefer `docs/AGENTS_*` files |
| Mixin guidelines | `docs/MIXINS_DESIGN_GUIDELINES.md` | Human-oriented; prefer `docs/AGENTS_*` files |
| Facade & Trait guidelines | `docs/FACADES_AND_TRAITS_DESIGN_GUIDELINES.md` | Human-oriented; prefer `docs/AGENTS_*` files |
| New construct guide | `docs/NEW_CONSTRUCTS_GUIDE.md` | Human-oriented; prefer `docs/AGENTS_*` files |

## Architecture — The Layer Model

- **L1 (`Cfn*`)**: Auto-generated from CloudFormation spec. Never manually edit.
- **L2**: Hand-written intent-based API with defaults. Where most work happens.
- **L3 (Patterns)**: Multi-resource compositions. Legacy L3s exist in `aws-ecs-patterns` and `aws-route53-patterns`. New L3s should NOT be added to this repo.

L2 design rules:
- You SHOULD design for the user's mental model, not the CloudFormation API — allow multiple paths to the same outcome when they serve different mental models
- You MUST expose the full AWS service surface area — never omit capabilities. Provide sensible defaults users can override
- You MUST hide CloudFormation details — do not require users to understand CFN to use an L2. Do not leak implementation details (ARNs, IAM actions, internal wiring) through the API
- You MUST provide escape hatches — expose the underlying L1 construct so users are never blocked by missing L2 features
- You SHOULD define resource contracts as interfaces — ensure third-party constructs can look and feel like first-party constructs
- You MUST NOT make L2s taggable themselves. Only L1 (`Cfn*`) resources implement `ITaggable` / `ITaggableV2`. L2s expose an optional `tags` prop wired to the L1 default child; users tag at any scope via `Tags.of(scope).add(...)`, which traverses the tree — see [AGENTS_CONSTRUCT_DESIGN.md § Tags](./docs/AGENTS_CONSTRUCT_DESIGN.md#tags)

### L2 Building Blocks

| Block | Scope | Purpose |
|-------|-------|---------|
| **Mixin** | Inward — about the resource | Extends resource behavior/lifecycle/L1 props |
| **Facade** | Outward — serves consumers | Wraps resource for grants, metrics, events |
| **Trait** | Cross-cutting | Service-agnostic capability contract |
| **CfnPropsMixin** | Simple glue | Thin L1 property passthrough, no logic |

### Feature Placement Decision

1. Modifies resource's own L1 props, no logic → `CfnPropsMixin`, STOP
2. Modifies resource's own L1 props, has logic → standalone `Mixin`, STOP
3. Serves external consumer → `Facade`, STOP
4. Advertises capability other constructs query → `Trait`, STOP
5. Otherwise → L2 construct method

> For full rules on each building block, see [AGENTS_CONSTRUCT_DESIGN.md#feature-placement-decision](./docs/AGENTS_CONSTRUCT_DESIGN.md#feature-placement-decision). For additional human-oriented detail: [DESIGN_GUIDELINES.md#mixins-facades-and-traits](./docs/DESIGN_GUIDELINES.md#mixins-facades-and-traits), [MIXINS_DESIGN_GUIDELINES.md](./docs/MIXINS_DESIGN_GUIDELINES.md), and [FACADES_AND_TRAITS_DESIGN_GUIDELINES.md](./docs/FACADES_AND_TRAITS_DESIGN_GUIDELINES.md).

## Construct Anatomy

Standard constructor: `constructor(scope: Construct, id: string, props: FooProps)`
- Default `props = {}` when all optional (not `?`)
- Use `"Resource"` as ID for the primary CFN resource

### Type Hierarchy

| Type | Purpose | Use as Parameter |
|------|---------|-----------------|
| `IFooRef` | Bare identifiers (ARN, name) — auto-generated | Default — when you only need IDs |
| `IFoo` | Full interface + Facade properties | When you need convenience methods |
| `FooBase` | Abstract base (exported, but treat as internal) | Never |
| `Foo` | Concrete class | Exceptional cases only |

### Static Type Check (never use `instanceof`)

L1 `Cfn*` constructs have it auto-generated (via `spec2cdk`); some core classes, such as `App`,
`Stack` and `Stage`, implement the same pattern by hand.

```ts
public static isFoo(x: any): x is Foo {
  return x !== null && typeof x === 'object' && Symbol.for('@aws-cdk/aws-{service}.{Foo}') in x;
}
```

## Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Classes/Enums | PascalCase | `Bucket`, `EngineVersion` |
| Properties/Methods | camelCase | `bucketArn`, `addToRolePolicy` |
| Behavioral interfaces | `I` prefix | `IBucket`, `IGrantable` |
| Data interfaces (structs) | No prefix | `BucketProps` |
| Enum members | SNAKE_UPPER_CASE | `AURORA_MYSQL` |
| Acronyms in classes | PascalCase | `JsonPattern` not `JSONPattern` |
| Event handlers | Past tense | `onImageBuildSucceeded` |
| Factory methods (import) | `from` prefix | `fromBucketArn()`, `fromFunctionName()` |
| Factory methods (enum-like) | `of()` static method | `EngineVersion.of()` |

- Use official AWS service terminology — don't rename
- Remove redundant context from prop names (resource type, "configuration")
- Include units when no strong type: `timeoutSec`, `memorySizeMiB`

## Props Design

- Name: `FooProps` — always a struct (readonly properties only)
- Flat — no artificial nesting, use shared prefixes for related props
- Every optional prop needs `@default` tag:
  - Simple: `@default true`
  - Context-dependent: `@default - uses the account default encryption`
  - Avoid `@default undefined` — describe the behavior instead
- Use strong CDK types (`Duration`, `Size`) over raw numbers
- Use construct interfaces in props — not ARN strings. Prefer `IFooRef`, then `IFoo` (see Type Hierarchy above)
- No L1 (CFN) types in L2 props
- No TypeScript union types (jsii incompatible) — use enum-like classes, separate props, or factory methods instead
- No `Token` type in props
- `SecretValue` type for any password/secret/token properties

## Security Rules

- SHOULD prefer specific IAM actions over full-service wildcards (`s3:*`), but suffix wildcards (`s3:GetObject*`) are acceptable
- MUST scope resource ARNs to most specific prefix
- SHOULD group related actions by resource scope into single PolicyStatements
- MUST use Grant helper methods (`addToPrincipal`/`addToPrincipalOrResource`) — not hand-rolled PolicyStatements
- MUST include `aws:SourceAccount`/`aws:SourceArn` conditions in trust policies (confused deputy prevention)
- SHOULD include `kms:ViaService` in KMS grants
- MUST emit synthesis-time warnings via `Annotations.of(construct).addWarningV2()` when configuration results in public access

## Implementation Patterns

### Error Handling

- Use `ValidationError` (with scope) or `UnscopedValidationError` (no scope) — never plain `Error`
- Error codes use the `lit` tagged template literal (required, enforces compile-time literal):
  ```ts
  throw new ValidationError(lit`DescriptiveErrorCode`, 'error message', scope);
  throw new UnscopedValidationError(lit`DescriptiveErrorCode`, 'error message');
  ```
  Import `lit` from `../../core/lib/private/literal-string` (adjust relative path based on file depth). Codes are PascalCase. Reuse across packages if cause and resolution are shared.
- Prefer auto-correcting config over errors — only fail on contradictory input
- Error messages: lowercase, no period, include wrong value via `JSON.stringify()`, expected values, what to change
- Three mechanisms: (1) eagerly throw for API misuse, (2) `node.addValidation()` for post-init checks, (3) `Annotations.of(construct).addError()` for environmental issues
- Never catch exceptions — all CDK errors are unrecoverable. Model recoverable errors in return values instead.

### Token Safety

Tokens can encode strings, numbers, and lists. Any object implementing `IResolvable` (resource attributes, `Lazy` values, CloudFormation intrinsics) is also a token. `Token.isUnresolved()` detects all types.

- Check `Token.isUnresolved()` before any validation on tokenized values — strings, numbers, AND lists:
  ```ts
  if (!Token.isUnresolved(props.name) && props.name.length > 64) { ... }   // string
  if (!Token.isUnresolved(props.port) && props.port > 65535) { ... }       // number
  if (!Token.isUnresolved(props.subnets) && props.subnets.length < 2) { ... } // list
  ```
- Tokenized lists always have `.length === 1` (the marker) — never trust `.length`, `.map()`, or iteration without checking first
- Use `!== undefined` (not truthiness) for optional prop checks — token-encoded values can be falsy
- Use `Tokenization.stringifyNumber()` to safely convert a possibly-tokenized number to string
- Don't use resource attributes (Tokens) in hash calculations for physical names

### Deferred Values (Box API)

L2 constructs that accumulate state after construction (e.g., adding actions, policy statements, security groups) MUST use the **Box API** to defer value resolution — not `Lazy`. Boxes implement `IResolvable` and capture stack traces at mutation call sites (under `CDK_DEBUG`), enabling accurate property attribution in synthesized templates.

- Use `Box.fromArray<T>([])` for accumulator lists, `Box.fromValue<T>(initial)` for single values, `Box.fromMap<K,V>()` for maps, `Box.fromSet<A>()` for sets
- Pass to L1 props via `Token.asList(box)`, `Token.asString(box)`, `Token.asNumber(box)`, or `Token.asAny(box)` for complex/object values
- `Box.fromArray` resolves to `undefined` when empty (omitEmpty default) — no manual empty-array check needed. Pass `{ omitEmpty: false }` to resolve to an empty array instead
- Mutate via `box.push(item)` or `box.set(newValue)` — each captures a stack trace at the call site
- Use `box.derive(fn)` for single-source transforms or `Box.combine({ name: box, ... }, ({ name, ... }) => ...)` for multi-source derived values
- Apply `@noBoxStackTraces` decorator on L2 classes that create or mutate Boxes in their constructor (suppresses irrelevant internal traces)
- NEVER mutate construct tree in Lazy or Box callbacks

`Lazy` is legacy — existing code still uses it but new L2 constructs MUST prefer Boxes. See `packages/aws-cdk-lib/core/adr/box-api.md` for full rationale.

**Before (legacy — do not use in new code):**
```ts
alarmActions: Lazy.list({ produce: () => this.alarmActionArns }),
```

**After (preferred):**
```ts
protected readonly _alarmActionArns: IArrayBox<string> = Box.fromArray([]);
// in constructor:
alarmActions: Token.asList(this._alarmActionArns),
// in mutating method:
this._alarmActionArns.push(newArn); // stack trace captured here
```

- Map empty arrays to `undefined` for CFN properties
- Optional nested CFN objects: `undefined` (not `{}`) when no sub-properties set

### ARN Construction

- Use `Stack.of(scope).formatArn()` — never hardcode ARN strings
- No `Fn::Sub` (FnSub) in CDK constructs
- No `Lazy.string` for physical names — use `generatePhysicalName()` + `getResourceNameAttribute()`

## Feature Flags

Required when a change alters observable behavior of existing API.

- Use correct `FlagType`: `BugFix` when old behavior was wrong, `ApiDefault` when old behavior is valid but not recommended. `ApiDefault` requires `compatibilityWithOldBehaviorMd` field.
- New flags: set `introducedIn: { v2: 'V2NEXT' }`, `recommendedValue: true`, `unconfiguredBehavesLike: { v2: false }` — ensures existing apps keep old behavior
- Flags should tighten security (reduce trust/permissions), never loosen it. If you need broader permissions, make it an explicit API option
- Don't use flags when a new construct replaces an old one — deprecate the old construct instead
- Flag variables: `is`/`has` prefix — `const isReducedScope = FeatureFlags.of(this).isEnabled(cxapi.MY_FLAG)`
- Warn about behavior changes via `Annotations.of(this).addWarningV2()` — not custom props. Users suppress via `acknowledgeWarning()`

Consuming a flag:
```ts
import { FeatureFlags } from '../../core';
import * as cxapi from '../../cx-api';
if (FeatureFlags.of(this).isEnabled(cxapi.MY_NEW_FLAG)) { ... }
```

## Documentation

### JSDoc

- Document all public APIs (classes, methods, properties, interfaces) when first introduced
- Summary line, blank line, then body:
  ```ts
  /**
   * The encryption key for this bucket.
   *
   * If specified, objects will be encrypted using this key.
   */
  ```
- Tags: `@param`, `@returns`, `@default`, `@see`, `@example`
- `@attribute` on CloudFormation attribute properties:
  ```ts
  /**
   * The ARN of this bucket.
   * @attribute
   */
  readonly bucketArn: string;
  ```
- Attribute names must begin with the type name: `bucketArn` not `arn`, `functionName` not `name`
- Copy prop documentation from official AWS docs when available
- Don't add docs on overrides — they inherit from the base interface

### Module READMEs

- Each `aws-cdk-lib/aws-{service}` has a README that renders as official API docs
- Must include: maturity level, simple example near top, examples per use case
- README code blocks (` ```ts `) must compile — verified by Rosetta. Use ` ```ts nofixture ` to skip
- All `feat()` PRs must include README updates

## Testing

### Unit Tests

- Use `Template.fromStack(stack).hasResourceProperties()` with `Match.objectLike` — assert specific properties, not entire templates:
  ```ts
  Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
    VersioningConfiguration: { Status: 'Enabled' },
  });
  ```
- Other `Match` helpers: `Match.objectEquals`, `Match.arrayWith`, `Match.stringLikeRegexp`, `Match.absent()`
- Avoid `Match.anyValue()` — it weakens assertions and hides regressions; assert the specific value, using `Match.stringLikeRegexp()` for non-deterministic values (asset hashes, generated IDs)
- Avoid `app.synth()` — `Template.fromStack()` synthesizes the stack internally, so an explicit synth is redundant
- `test.each` for boundary conditions: `test.each([0, -1, 256])('fails for invalid value %d', (val) => { ... })`
- Error tests: assert on specific error message, prefix test name with "fails"
- Test utility functions separately from constructs (e.g. `util.test.ts`)
- Grant methods: test with `new iam.Role()`, `Role.fromRoleArn()`, `new iam.User()`, `new iam.ServicePrincipal()`
- Include backward-compatibility tests when adding new optional props — default behavior must be preserved
- Preserve deprecated API tests with `testDeprecated` (from `@aws-cdk/cdk-build-tools`)
- Avoid `overrideLogicalId` in tests — couples tests to internal naming

### Integration Tests

- Stable modules: `integ.*.ts` under `packages/@aws-cdk-testing/framework-integ/test/{module}/test/`
- Alpha modules: `integ.*.ts` colocated in `packages/@aws-cdk/aws-{service}-alpha/test/`
- Use `IntegTest` construct — do NOT include `app.synth()`:
  ```ts
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'TestStack');
  // ... define resources ...
  const test = new integ.IntegTest(app, 'Test', { testCases: [stack] });
  // Assertions — optional but preferred (required for custom resources)
  test.assertions.awsApiCall('S3', 'getBucketVersioning', { Bucket: bucket.bucketName })
    .expect(integ.ExpectedResult.objectLike({ Status: 'Enabled' }));
  ```
- Assertions are preferred on all new integ tests; REQUIRED for custom resources
- Avoid `ec2.Peer.anyIpv4()`/`anyIpv6()` and set ALB listener `open: false` unless testing open access
- Snapshots: `.js.snapshot` extension
- Separate tests by feature area with descriptive names
- Update ALL affected snapshots (even cross-package) when changing logical IDs

Required for: new CFN resource types, new CFN properties, cross-service integrations, new versions, Custom Resources.

## PR Conventions

### Titles (conventional commit format)

| Type | When | Example |
|------|------|---------|
| `feat(module):` | New feature | `feat(s3): add intelligent tiering support` |
| `fix(module):` | Bug fix | `fix(lambda): correct timeout validation` |
| `docs(module):` | Documentation only | `docs(mixins): expand mixin guidelines` |
| `refactor(module):` | Feature-preserving refactor | `refactor(ec2): simplify subnet selection` |
| `chore(module):` | Build/config/minor | `chore(core): update dependencies` |

- Module scope optional for repo-wide changes: `chore: update dependencies`
- Lowercase, no period at end
- `feat` and `fix` PRs MUST reference an issue: `fixes #<issue>` or `closes #<issue>`
- `feat()` PRs require unit tests, integration snapshots, and README updates
- Breaking changes are only allowed in `-alpha` libraries. Declare with `BREAKING CHANGE:` in the PR body before the `---` line
- One concern per PR — submit cosmetic changes separately

## Anti-Patterns — Things NOT To Do

- **MUST NOT use jsii-incompatible patterns** — mapped types, conditional types, overloaded functions, TypeScript namespaces, `export const` objects (use `public static readonly` on classes). MUST NOT move public types between files — file location is part of the external contract in jsii bindings
- **MUST NOT use fluent API patterns** (method chaining returning `this`) — jsii languages can't chain methods that return `this`, and it hides mutation behind a return value ([DESIGN_GUIDELINES.md#general-principles](./docs/DESIGN_GUIDELINES.md#general-principles))
- **MUST NOT add speculative abstractions** — add what customers need today; unused abstractions become maintenance burden and API surface that can't be removed ([DESIGN_GUIDELINES.md#general-principles](./docs/DESIGN_GUIDELINES.md#general-principles))
- **MUST NOT change construct IDs** — logical IDs derive from the full construct path; any change replaces all resources in scope, causing data loss ([DESIGN_GUIDELINES.md#construct-ids](./docs/DESIGN_GUIDELINES.md#construct-ids))
- **MUST NOT leave commented-out code, dead code, or `eslint-disable` directives** — they rot, confuse future contributors, and mask real lint violations
- **MUST NOT add validation to existing constructs without considering backwards compatibility** — adding validation that rejects previously accepted input is a breaking change. Two cases apply:
  - **Input that previously deployed successfully** → MUST gate behind a feature flag so existing apps continue to synthesize
  - **Input that was accepted by synth but always failed at deploy time** (e.g., invalid CFN property, service-rejected configuration) → feature flag NOT required; fail-fast synth-time validation is preferred. The PR MUST document why the breaking change is justified (e.g., "this input always caused CloudFormation error X")

## Key References

`AGENTS_CONSTRUCT_DESIGN.md` and `AGENTS_CONSTRUCT_IMPLEMENTATION.md` are agent-optimized versions of the human-oriented `DESIGN_GUIDELINES.md`. They contain the same rules but in a structured, prescriptive format. Always prefer the `AGENTS_*` files and only fall back to `DESIGN_GUIDELINES.md` when the agentic files don't cover a topic.

| File | Format | What it covers | When to read it |
|------|--------|---------------|-----------------|
| [`docs/AGENTS_CONSTRUCT_DESIGN.md`](./docs/AGENTS_CONSTRUCT_DESIGN.md) | Agent-optimized | Construct design rules — mixins, facades, traits, grants, metrics, events, connections, API patterns | When designing or extending an L2 construct |
| [`docs/AGENTS_CONSTRUCT_IMPLEMENTATION.md`](./docs/AGENTS_CONSTRUCT_IMPLEMENTATION.md) | Agent-optimized | Implementation patterns — grants, metrics, events, connections, IAM, VPC, removal policy | When implementing cross-cutting L2 patterns |
| [`docs/DESIGN_GUIDELINES.md`](./docs/DESIGN_GUIDELINES.md) | Human reference | Authoritative API design reference | Before designing a new L2 API, adding props, or making architectural decisions |
| [`docs/MIXINS_DESIGN_GUIDELINES.md`](./docs/MIXINS_DESIGN_GUIDELINES.md) | Human reference | Mixin architecture and implementation | When adding a feature that modifies resource behavior or L1 props |
| [`docs/FACADES_AND_TRAITS_DESIGN_GUIDELINES.md`](./docs/FACADES_AND_TRAITS_DESIGN_GUIDELINES.md) | Human reference | Facade and Trait architecture and implementation | When implementing a new Facade or Trait factory |
| [`docs/NEW_CONSTRUCTS_GUIDE.md`](./docs/NEW_CONSTRUCTS_GUIDE.md) | Human reference | Step-by-step new construct walkthrough | When creating a new L2 construct from scratch |
| [`CONTRIBUTING.md`](./CONTRIBUTING.md) | Human reference | Contribution workflow, PR process, setup | First-time setup, PR submission, or understanding review process |
| [`INTEGRATION_TESTS.md`](./INTEGRATION_TESTS.md) | Human reference | Integration test deep-dive | When writing, running, or debugging integration tests |

All code is TypeScript compiled via [jsii](https://github.com/aws/jsii/) to other languages.
Every public API must be jsii-compatible.
