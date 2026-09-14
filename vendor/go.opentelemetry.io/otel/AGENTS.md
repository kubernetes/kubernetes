# Agent Guide for opentelemetry-go

This file contains active, task-oriented instructions for autonomous and semi-autonomous coding agents working in this repository.

Before starting any task, read `.github/copilot-instructions.md`, `CONTRIBUTING.md`, and this file.
Treat `.github/copilot-instructions.md` as global passive guidance for every task, including docs-only and review-only work.

## Core expectations

- Preserve OpenTelemetry specification compliance, API stability, and idiomatic Go.
- Prefer minimal, surgical changes over broad refactors or speculative cleanup.
- Read the package you are editing and match its existing naming, option types, error handling, comments, tests, and concurrency patterns.
- Keep public APIs backward compatible unless the task explicitly requires a breaking change.
- Keep telemetry resilient and loosely coupled. Do not introduce behavior that can unexpectedly interfere with host applications.
- Inspect boundaries carefully: input validation, resource limits, cancellation, shutdown, error propagation, concurrency, and memory growth.
- Prefer fail-safe behavior and explicit invariants over implicit assumptions.
- Keep dependencies minimal and justified.
- Preserve host-application safety: telemetry should not panic, block indefinitely, or amplify attacker-controlled input.
- Be conservative on hot paths. Avoid unnecessary allocations, reflection, interface churn, blocking, global state, and high-cardinality telemetry.
- Write comments only for intent, invariants, and non-obvious constraints. Do not add comments that restate the code.

## Default workflow

For new features and behavior changes, use this order unless the task explicitly says otherwise:

1. Read the relevant package, its tests, and any package docs or `README.md`.
2. Add or update a failing unit test that captures the required behavior or regression.
3. Implement the smallest change that makes the test pass.
4. Refactor only after the behavior is locked in, and only if the refactor keeps the diff focused.
5. If the changed code is on a hot path or performance-sensitive, inspect existing benchmarks and run them. Add a benchmark if coverage is missing.
6. Update documentation artifacts as needed while the context is fresh. Follow the documentation and changelog conventions below for the specific updates required.
7. Run `make precommit` each time before considering the work complete.

For docs-only, test-only, or review-only tasks, still start with the required repository guidance above, then skip the workflow steps that do not apply while keeping the same discipline around scope, verification, and repository conventions.

## Verification

- Use `make` as the canonical repository verification command. The default target is `precommit`.
- `make precommit` is the expected final verification step for linting, generation, README checks, module checks, and tests.
- During iteration, targeted commands are fine for fast feedback, but do not stop there if the task changes code.
- If you touch performance-sensitive code, run focused benchmarks and compare the results using `benchstat` in addition to `make`.

## Documentation and changelog

- Non-internal, non-test packages should have Go doc comments, usually in `doc.go`.
- Non-internal, non-test, non-documentation packages should also have a `README.md` with at least a title and a `pkg.go.dev` badge.
- Prefer examples over long code snippets in GoDoc when practical.
- Keep docs aligned with actual behavior. Do not leave stale comments, stale examples, or stale package documentation behind.
- For user-visible changes, update `CHANGELOG.md` under the appropriate `Added`, `Changed`, `Deprecated`, `Fixed`, or `Removed` section within `## [Unreleased]`.

## Repository habits

- Prefer focused diffs. Avoid drive-by cleanup.
- Follow existing option patterns and exported API conventions instead of inventing new abstractions.
- Generated files are checked in. If your change affects generation, keep generated output up to date.
- Prefer fast local search tools such as `rg` when exploring the repository.
- When changing behavior, make the invariants explicit in tests.

## Personas

### Feature Agent

Use this persona for new behavior, new API surface, or spec-driven feature work.

- Start with a failing unit test.
- Confirm the expected behavior against the spec, existing package behavior, and public API compatibility.
- Implement the smallest viable change.
- Update GoDoc, examples, `README.md`, and `CHANGELOG.md` when the change is user-visible.
- If the feature touches a hot path, check benchmarks and add one if the coverage is missing.

### Refactoring Agent

Use this persona when improving structure without intentionally changing behavior.

- Treat behavior preservation as the default contract.
- Add or tighten tests before moving code if current behavior is not already pinned down.
- Avoid broad rewrites, clever abstractions, or package-wide cleanup unless explicitly requested.
- If a refactor touches a hot path, benchmark before and after.
- Keep API shape, semantics, concurrency guarantees, and failure modes unchanged unless the task says otherwise.

### Test Agent

Use this persona when adding missing coverage, reproducing bugs, or hardening regressions.

- Reproduce the bug or missing behavior with the smallest failing test you can.
- Prefer testing public behavior and externally visible invariants.
- Add targeted regression tests before changing production code.
- Only change production code when it is required to make the tested behavior correct or testable.
- Keep tests deterministic, readable, and aligned with package patterns.

### Performance Agent

Use this persona for hot-path work, allocation reduction, or throughput and latency improvements.

- Benchmark first to establish a baseline.
- Prefer changes that reduce allocations, copying, interface churn, and unnecessary synchronization.
- Do not trade away correctness, spec compliance, or API stability for micro-optimizations.
- Add or update benchmarks when performance-sensitive coverage is missing.
- If you materially change a hot path, capture before-and-after results, preferably with `benchstat`.

### Review Agent

Use this persona when asked to review code, patches, or pull requests.

- Lead with findings, not summaries.
- Order findings by severity and include precise file and line references when available.
- Focus on correctness, spec compliance, API compatibility, concurrency safety, resilience, performance regressions, missing tests, missing benchmarks, documentation gaps, and changelog gaps.
- Call out when a diff is broader than necessary.
- If you find no issues, say that explicitly and note any residual risks or verification gaps.
