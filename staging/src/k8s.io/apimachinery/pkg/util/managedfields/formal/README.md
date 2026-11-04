# SSA ownership reference model

This is an executable Lean 4 prototype of the **proposed** full/partial
item-manager semantics discussed in
[structured-merge-diff#336](https://github.com/kubernetes-sigs/structured-merge-diff/pull/336).
It specifies the ownership contract; it is not a verification of the current
Go implementation, nor does it change Kubernetes behavior.

The project lives beside `managedfields` because
[`internal/structuredmerge.go`](../internal/structuredmerge.go) connects
Kubernetes objects and managed fields to SMD. Merge/pruning is implemented in
SMD; admission, defaulting, conversion, and persistence involve other layers.

The [property catalogue](PROPERTIES.md) maps existing SSA obligations to KEP-555
and pinned Kubernetes tests. It distinguishes the new ownership rules, general
theorems, concrete execution proofs, and open implementation obligations.

The self-contained [HTML proof explorer](docs/proofs.html) presents exact Lean
statements, proof-source listings, standard-axiom dependencies, and interactive
links between lemmas. Its dependency edges are exported from accepted proof
terms. Short explanations and a YAML walkthrough distinguish general theorems,
guard consequences, concrete executions, and open model gaps.

## Contract

After every accepted transition, each **existing associative-list item** has
at least one individual full manager.

A manager is full for a particular item when it claims the item's membership
and its entire required core. List-map keys are encoded in the item path;
claiming membership asserts those keys. Required object fields are traversed
recursively: the **same** manager must cover all required branches. Optional
descendants are not added to the enclosing item's required core.

An optional nested list item has its own required core and may have a different
full manager. Its full manager does not protect it from deletion of an ancestor.

- Partial omission releases only that manager's claims and otherwise unclaimed
  values.
- Omission by the last full manager deletes the item, including partial
  managers' descendants.
- An item survives if another full manager remains.
- A request retaining an item but leaving no full manager is rejected atomically.
- Key-only items have an empty non-key required core: every membership claimant
  is full, and the item disappears after the last claimant leaves.
- Force transfers changed values **before** checking the resulting ownership.
  Force does not bypass the invariant. Equal values can be co-owned.

The fragmenting request is rejected even though the stored values remain valid:

```yaml
# A first applies this complete item.
items:
- name: database
  host: db.example.com
  port: 5432
---
# B co-claims port without changing its value.
items:
- name: database
  port: 5432
---
# A tries to release port: rejected, since neither manager would remain full.
items:
- name: database
  host: db.example.com
```

The old state, including A's port ownership, is preserved on rejection. A safe
handoff has B acquire the complete required core before A relinquishes it.

## Representation decisions

`Schema` is a finite **instance-expanded** list of item descriptors. The
universe includes the concrete keys that requests may create. It is not an
OpenAPI parser or a schema for an unbounded universe of future keys. Schema
checks reject duplicate descriptors, missing ancestor descriptors, and required
cores that cross into nested item boundaries.

Object values and ownership are separate. Typed paths distinguish ordinary
field names from composite list-map keys. Scalar/atomic values are opaque
strings; named objects, collections, and item membership have explicit markers.
Inputs are normalized assertions, not raw Kubernetes requests.

**Required granular lists need an explicit presence claim**, including when
the applied list is empty. The model preserves that assertion separately from
ownership of the list's elements. This is a design decision to make the
required-list rule executable. Current SMD fieldsets do not retain all those
presence assertions: a Go adapter must define how to represent or recover them.
This model must not be used as evidence that the current PR already does so.

Structural required-core checking is separate from complete value validation.
CEL and admission rules do not infer lifetime ownership. The present model does
not implement arbitrary OpenAPI constraints, CEL, webhooks, schema evolution,
defaulting, API-version conversion, unstructured/deduced types, set elements,
or the full semantics of Update and non-apply Patch. It also does not model
concurrent storage retries: transitions are serialized.

Already-fragmented states are rejected as invalid initial states. A migration
or recovery policy is a separate design obligation. The claims here concern
the fixed schema and normalized states accepted by this model.

## Files and proof obligations

- `SSA/Model.lean`: typed paths, required cores, values, claims, and invariant.
- `SSA/Apply.lean`: merge, conflicts, force transfer, lifetime deletion,
  field release, and atomic acceptance.
- `SSA/Properties.lean`: general proofs of accepted/committed-state validity,
  per-item full-owner witnesses, atomic rejection, same-manager required
  coverage, key-only behavior, direct-deletion authorization, subtree framing,
  ancestor-deletion dominance, preservation of unwritten/unreleased values
  outside deleted subtrees, and monotonicity under added co-ownership.
- `SSA/Contracts.lean`: general proofs of intent and exact actor-footprint
  preservation, exclusive/shared scalar pruning, conflicts, force transfer,
  retained full-successor ownership, finite-history validity, and immediate
  repeated-Apply idempotence, including force.
- `SSA/Examples.lean`: kernel-checked executions of fourteen scenarios, the
  valid-values/invalid-ownership counterexample, a reduced PR-style unsafe
  deletion, and same-apply idempotence.
- `SSA/Audit.lean`: automatically audits every model declaration and its
  transitive axioms, including new/private theorems and definitions. It also
  rejects source modules missing from the audited import graph.
- `SSA/Fixtures.lean` and `Main.lean`: executable JSON fixture protocol.
- `bridge/`, `cmd/ssa-compare/`, and `testdata/bridge.yaml`: the actual
  managedfields comparison, independent YAML expectations, and generated traces.

Invariant preservation alone can be obtained by rejecting every request.
Therefore the executable proofs also establish successful partial releases,
handoffs, key-only sharing, sibling preservation, nested role changes, populated
and empty required lists, and last-full-manager deletion.

The frame theorems concern the subtree-deletion phase, not commutativity of
arbitrary Apply requests. The monotonicity theorems concern adding claims, not
arbitrary force transfers. The unsafe rule is a reduced countermodel of the
reported PR failure, not a mechanically translated Go implementation.

Idempotence is proved for every accepted request, not just a fixture: immediately
repeating that request succeeds and returns exactly the same model state.
Handoff safety requires an already-full successor and excludes ancestor deletion
of its claims; general acquisition/release progress remains open. None of these
theorems supplies Kubernetes defaulting, storage, or Go implementation semantics.

## Build and verification

Install the Lean version pinned in `lean-toolchain` using
[Elan](https://lean-lang.org/install/manual/), or use that release's native
binaries. No Mathlib or other external Lake packages are required.

From this directory:

```sh
lake build
bash verify.sh
lake exe ssa-model --list
lake exe ssa-model --trace reject-fragmented-required-ownership
```

For a cached installation, set `SSA_LAKE=/absolute/path/to/lake` when invoking
`verify.sh`. The script checks the pinned version, cleans and rebuilds with
warnings treated as errors, automatically audits declarations against Lean's
standard logical axioms (`propext`, `Quot.sound`, and `Classical.choice`), and
rechecks the stored proof terms with `leanchecker --fresh Main`. Negative audit
probes must reject an additional axiom, a hidden incomplete proof, native
evaluation, and an unimported source module.
An imported source module outside the SSA namespace is also checked.

It then executes the Lean scenarios, replays and regenerates the JSON snapshot,
runs the Go adapter tests, and compares both engines over the independent corpus
plus 64 generated sequences of 32 requests. Go, at the version required by
apimachinery, and Python 3 must be installed. Ordinary Go tests need no Lean
installation. Finally, verification rejects a stale HTML proof explorer.

To regenerate the explorer after changing Lean source:

```sh
lake build
lake env lean SSA/Audit.lean
python3 docs/generate.py
```

The generator checks that exported source snapshots match the working tree.
The HTML needs no JavaScript packages, network access, or web server to run.

The [CI entrypoint](ci/README.md) can install the pinned Lean release with a
checked-in SHA-256 and saves the complete verification log. A concrete Prow
presubmit configuration is included for review in `kubernetes/test-infra`;
it is not active merely by placing it in this repository. No Lean dependency
is added to Kubernetes' normal Go build.

## JSON protocol and implementation bridge

```sh
lake exe ssa-model --fixtures
lake exe ssa-model --check fixtures/scenarios.json
lake exe ssa-model --check - < fixtures/scenarios.json
lake exe ssa-model --replay fixtures/scenarios.json
```

The checked-in fixture array is generated from the Lean examples. Each fixture
contains a schema, an initial state, and ordered requests with expected errors,
present/absent cells, and full/partial manager assertions. Each request replaces
one manager's complete normalized footprint. Malformed or empty fixture input
fails. JSON reports and traces include complete post-request claims.

The separate, hand-authored [YAML corpus](testdata/bridge.yaml) supplies ordinary
objects and complete expected values/owners. The bridge normalizes its requests
for Lean and runs the original YAML objects through the real
`structuredMergeManager`, with the real manager-identity and metadata-stripping
wrappers. Every transition also round-trips managed fields through FieldsV1.
It compares every value/container/item cell, representable scalar/item ownership,
and acceptance or error, after every request.

The bridge uses one fixed structural schema, one API version, strings, atomic
maps, granular objects, and nested name-keyed associative lists. Required cores
in this adapter contain direct scalar fields. Collection/map presence ownership
is explicitly excluded from the shared projection because SMD does not retain
all of the model's presence claims; the object values themselves are still
compared, including empty containers. Set lists, nulls, conversion, admission,
defaulting, and arbitrary OpenAPI schemas remain outside this bridge.
Associative-list ordering is outside the observation model.

Every intentional proposal difference needs two complete independent expected
results, a reason, and `differenceKind: proposal`. There are currently two:
last-full-manager deletion and rejection of ownership fragmentation. An additional
`model-gap` case records synthetic object-presence ownership retaining an empty
object after a force transfer. It is an unresolved model/normalization defect,
not an intended SSA semantic change. Both engines execute all of these cases;
a new discrepancy or a stale exception fails verification.

Generated traces currently explore the common top-level scalar domain. Failures
report the first failing request prefix, which `ssa-compare -reproducer FILE`
can replay. These differential checks are bounded implementation evidence.
A general Go refinement proof does not exist.

Changes to the ownership contract should update the model, proofs, generated
fixtures, and implementation comparison in the same review. The existing SSA
implementation and the new experimental semantics must be compared explicitly;
current Go SSA is not expected to pass fixtures for rules it does not implement.
