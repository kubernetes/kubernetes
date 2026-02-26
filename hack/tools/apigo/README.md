# apigo: Explicit API Surface Management

`apigo` is a tool for managing the exported API surface of Kubernetes Go modules (such as `client-go` and
`apimachinery`).

Because these modules have a massive ecosystem of downstream consumers, implicit or accidental API breakages have a
severe blast radius. `apigo` transitions our API management from implicit, point-in-time git diffs to explicit,
declarative files checked directly into the repository. This guarantees that we honor the API contracts published in
past releases and forces intentionality when evolving our libraries.

## Go API Compatibility Best Practices

Before modifying an exported symbol, consider if the change can be designed in a backward-compatible way. Kubernetes
tries to adhere to the Go 1 compatibility promise for its exported library surfaces, although is not always possible.

For a deeper understanding of how to evolve Go APIs without breaking downstream users, refer to the Go team's
presentation: [Codebase Refactoring (with help from Go)](https://go.dev/talks/2016/refactor.article).

### 1. Functions and Methods

Changing the signature of an existing exported function (e.g., adding parameters or changing return types) immediately
breaks any consumer calling it.

- **Best Practice:** Leave the existing function intact, mark it with a `// Deprecated:` comment, and introduce a
  new function alongside it. For example, rather than adding a context to `GetPod(name string)`, introduce
  `GetPodWithContext(ctx context.Context, name string)`.

### 2. Structs

Removing, renaming, or changing the type of an exported struct field is a breaking change.

- **Best Practice:** Adding new exported fields to a struct is generally safe, provided the struct is intended to be
  instantiated by pointer and the new field has a safe, logical zero-value.

### 3. Interfaces (The "Hidden" Breakage)

Adding a new method to an exported interface is a breaking change. Any downstream consumer who implemented your
interface in their own code will fail to compile because their type no longer satisfies the expanded interface.

- **Best Practice:** Define a new interface that embeds the old one, or provide a secondary interface that consumers
  can optionally implement (and check via type-assertion at runtime).

## Categorizing Breaking Changes

Sometimes, a breaking change is unavoidable—often to remove deeply flawed, insecure, or long-deprecated APIs. When
`apigo` detects a breakage, CI will fail unless the breakage is explicitly justified in the `except.txt` file.

We categorize allowable breaking changes into two main types:

### Category A: Mitigable Compile-Time Breaks

These are intentional removals of technical debt where the user's code will fail to compile, but the fix is trivial and
mechanical.

- **Requirement:** You must provide explicit instructions or code snippets detailing how the consumer should update
  their code.
- _Example:_ Removing a deprecated constructor in favor of a newer options-based builder pattern.

### Category B: Fundamental Architecture Breaks

These occur when an API is fundamentally unsafe or impossible to support, and there is no direct 1:1 replacement.

- **Requirement:** You must thoroughly explain the rationale so downstream users understand why they are required to
  redesign their business logic or state management.

## Workflow: Adding to the API

If you add a new exported feature, CI will fail because the new symbols are not registered in the baseline. To fix
this:

1. Run `make update-apigo` (or `./hack/update-apigo.sh`).
2. The tool will automatically append your new symbols to the module's `apigo/next.txt` file.
3. Commit the changes to `next.txt` so reviewers can clearly see the newly expanded API surface in the PR diff.

## Workflow: Breaking or Removing an API

If you modify a signature or remove an exported symbol, `apigo` will fail CI. To bypass this, you must explicitly
document the breakage in `except.txt` and obtain approval from a designated reviewer.

1. Open the `apigo/except.txt` file in the module you are modifying.
2. Add the exact signature of the symbol you are breaking or removing.
3. You must add a comment block immediately above the exception using the strict `PR:` and `Migration:`
   format.

**Example `except.txt` entry:**

```text
# PR: [https://github.com/kubernetes/kubernetes/pull/12345](https://github.com/kubernetes/kubernetes/pull/12345)
# Category: A (Mitigable)
# Migration: Downstream consumers should migrate to NewClientWithOptions().
# The old DefaultKubernetesUserAgent() was deprecated in v1.28 and is now removed.
pkg k8s.io/client-go/rest, func DefaultKubernetesUserAgent() string
```
