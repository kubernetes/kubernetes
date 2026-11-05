# Confirmed SSA bugs and PR #336 failures

This register records reproduced behavior in the current checkout and the
pinned SMD revisions. Baseline SSA bugs and failures of the PR's proposed
full-manager rule have separate IDs. A conflict or validation rejection is
reported together with the preceding transition that caused it.

## Evidence overview

| ID | Affected behavior | Result | Evidence |
| --- | --- | --- | --- |
| SSA-01 | Explicitly empty associative list after releasing owned items | First Apply removes the field; identical repetition creates `[]`. Required/CEL presence validation can reject every retry. | Lean refutation and source-bound witness; actual managedfields; both pinned SMD revisions; CRD HTTP/etcd. |
| SSA-02 | Explicitly empty granular map after releasing an owned child | First Apply produces `null` while retaining the map claim. Optional non-nullable maps cannot be emptied. Nullable maps change on repetition and can cause a conflict for another manager's identical intent. | Lean execution/intent counterexamples and all finite retry counts; source-bound witness; actual managedfields; both pinned SMD revisions; CRD HTTP/etcd. |
| PR336-01 | Fragmented required ownership followed by partial-manager omission | A partial manager can delete another partial manager's required data. | Lean execution; exact pinned head/base comparison. |
| PR336-02 | Direct required-map membership shortcut | The classifier calls a manager full without ownership of required children. An invalid item can remain after the actual full owner leaves. | General classifier theorem; concrete Lean execution; actual pinned head/base outcomes. |
| PR336-03 | Required granular-list ownership | A supplier owning all child items is classified partial because Apply generated no direct list-presence claim. Another partial manager can delete its data. | General classifier fact; populated/empty Lean counterexamples; exact pinned head/base comparison. |

The synthetic `before-first-apply` Update owner is an additional observable stage
of SSA-01's trace. It is modeled and checked, and is not counted as a separate
independent bug.

## SSA-01: an explicitly empty list disappears

A previously applied `endpoints: [{name: db, host: old}]`, then applies:

```yaml
endpoints: []
```

The first result lacks `endpoints`. The second identical request produces
`endpoints: []`. OpenAPI `required: [endpoints]` or CEL
`has(self.endpoints)` rejects the first result, preserving the old data and
claims, so identical retries cannot reach the second result. Empty-list
creation under the same schema succeeds.

See the [full list trace and default-wrapper initialization](baseline-empty-list.md).
The source-bound proof uses normalized typed/set primitives; actual Go and HTTP
execution independently reproduce this concrete witness.

## SSA-02: an explicitly empty map becomes null

The relevant CRD schema is at
`spec.versions[*].schema.openAPIV3Schema`. No field is required here:

```yaml
type: object
properties:
  settings:
    type: object
    properties:
      left:
        type: string
```

A first applies this resource content:

```yaml
settings:
  left: old
```

A then applies:

```yaml
settings: {}
```

At the SMD boundary the result is:

```yaml
settings: null
```

The map's presence claim **still belongs to A**, encoded as
`{"f:settings":{}}`. This distinguishes SSA-02 from a missing ownership marker.
Removing the last child through `RemoveItems` materializes a retained granular
map slot as null. The pruning restoration phases keep the map claim without
reconstructing the explicitly asserted empty object.

### Non-nullable maps: valid intent cannot be applied

The schema above permits `settings: {}` and omission of `settings`. It rejects
`settings: null`. The real HTTP request therefore fails validation even though
its desired value is valid. GET observes unchanged data, ownership and
`resourceVersion` after each attempt. Repetition always starts from the same
populated map. Creating another resource with `settings: {}` succeeds.

`ContainerCounterexamples.nonnullable_map_retries_never_empty` proves this
failure for every finite retry count under the explicit object-or-absent
validator. It accepts absence, so the failure requires no required-field rule.

### Nullable maps: data changes and identical intent can conflict

Add `nullable: true` to the `settings` schema:

| Step | Request | Observed result |
| --- | --- | --- |
| A creates | `settings: {left: old}` | A owns `settings.left`. |
| A empties | `settings: {}` | Accepted as `settings: null`; A owns `settings`. |
| B shares the same empty intent | `settings: {}` | Conflict against A's stored null; no state is committed. |
| A repeats | `settings: {}` | Accepted as the actual empty map, preserving A's claim. |
| B retries | `settings: {}` | Accepted; A and B now co-own the same empty map. |

The conflict is a downstream effect of the first accepted request changing the
intended value. The conflict detector compares against the stored null. This
trace does not assert that equal **stored** values conflict.

### CEL and successful control cases

A nullable map with root CEL `has(self.settings)` also rejects the unexpected
null and every identical retry. Nullable null is exposed as absent to that CEL
presence check. Three successful controls locate the affected path:

- Creating an empty map succeeds under each tested policy.
- An atomic map replaces the old object with `{}` on the first Apply.
- A non-nullable map with `default: {}` repairs the produced null before
  validation, making the first request succeed.

The actual HTTP test checks all five variants: non-nullable, nullable, nullable
with CEL, non-nullable with an empty-object default, and atomic. It checks
persisted data, claims, rejection atomicity and storage no-ops.

### Proof and reproduction entrypoints

The Lean module is [SSA/ContainerCounterexamples.lean](../SSA/ContainerCounterexamples.lean):

- `empty_map_null_for_every_scalar_payload`, a family quantified over every
  encoded scalar payload for this map/child shape.
- `empty_map_first_apply_creates_null` and `empty_map_second_apply_repairs_data`.
- `applied_empty_map_intent_is_not_preserved`.
- `empty_map_can_conflict_with_identical_previous_intent`.
- `valid_empty_map_intent_is_rejected` and `nonnullable_map_retries_never_empty`.
- `empty_object_default_repairs_this_proposal` for the explicit model mutator.
- `imported_pruning_creates_null`, using all four imported source functions,
  realized callees and every lawful version-map schedule under normalized
  primitives.

The `explicit-empty-map-becomes-null-before-repetition` case is present in both
[baseline](../testdata/bridge.yaml) and [pinned-upstream](../testdata/pr336.yaml)
corpora. `TestSSAEmptyGranularMapPresence` runs through the actual API server.

## PR336-01: a partial manager can authorize deletion

An item is keyed by `name`; `host` and `port` are required:

```yaml
# A creates the required core.
items: [{name: db, host: old, port: p}]
---
# B shares port.
items: [{name: db, port: p}]
---
# A releases port but keeps host.
items: [{name: db, host: old}]
```

The PR accepts all three requests. Both required values remain, but neither
manager is full. B then applies `{}`. The PR deletes the item, including A's
host. The base retains A's remaining contribution.

The proposed invariant rejects A's fragmenting request atomically. Existing
fragmented states still need an explicit rollout/repair policy.

## PR336-02: a map membership claim bypasses required children

For an item with required `conn` and required `conn.endpoint`, A applies
`{name: db, conn: {endpoint: live}}`; B applies `{name: db, conn: {}}`.
The shared live object is valid, but B owns only item membership and `conn`
presence. The PR's direct-member shortcut calls B full without ownership of
`endpoint`.

After A applies `{}`, the item remains as `{name: db, conn: null}`. Both pinned
versions retain this invalid candidate; the PR-specific defect is the
full-manager classifier. A real schema validator can reject such a candidate,
which prevents the intended deletion from completing.

## PR336-03: required granular lists are misclassified

For an item whose `children` list is required, A supplies
`{name: db, children: [{name: child}]}`; B adds
`{name: db, note: injected}`. A owns the nested item, but Apply generated no
direct claim on `children`, so the PR calls A partial. B then applies `{}`.
The PR deletes the whole item and A's child; the base retains them.

The same failure occurs when A explicitly supplied `children: []`. The
FieldsV1 grammar supports a list-presence claim, but ordinary Apply extraction
does not generate it for this known granular list.

Explicit `children: null` is a contrasting control: extraction creates a direct
field member, so the PR retains A's membership when B leaves. The pinned corpus
also checks a direct null member on a required map. FieldSet stores paths
without the model's value/presence tags; the raw classifier therefore checks
direct path membership for both. This comparison exposed and corrected a model
mismatch. Null's API validity remains a separate schema/validation question.

## Verification and limits

From the formal project directory:

```sh
bash ci/verify.sh
bash ci/verify-apiserver.sh
```

The first command runs the kernel checks, independent outcomes and actual
pinned SMD head/base comparisons. The second starts local etcd and
apiextensions-apiserver and executes the list and map HTTP tests.

API-version conversion, arbitrary webhook implementations, all defaulting
programs and the complete Go request pipeline do not have universal refinement
proofs. These limits do not weaken the concrete reproduced counterexamples.
The stronger policy's own intent/idempotence theorems describe its separate
transition, rather than current SSA's disproved unrestricted laws.
