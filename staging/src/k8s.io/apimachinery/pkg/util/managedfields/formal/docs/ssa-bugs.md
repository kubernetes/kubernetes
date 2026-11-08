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
| SSA-03 | Omission after another manager Updates an item child | The item and its foreign-owned child disappear; the list becomes null. A nullable schema accepts this; a non-nullable schema rejects every retry. | Reachable Lean Apply/Update counterexample and source witness; actual managedfields and pinned SMD; nullable/non-nullable CRD HTTP. Existing report: [#134902](https://github.com/kubernetes/kubernetes/issues/134902). |
| SSA-04 | Status Apply includes an additional non-status root field | The status actor acquires the unchanged field even though the REST strategy resets it. A later main-endpoint change conflicts. | Reachable Lean refutation; complete imported Apply with explicit primitive instances; real CRD HTTP/etcd and conventional spec/status controls. |
| [SSA-05](https://github.com/kubernetes/kubernetes/issues/143942) | Mixed int64/float64 comparison near `2^63` | Scalar-only intent is lost as a false no-op. Adding an unrelated field lets B change A's numeric value while both retain ownership, including with force. | Lean intent/conflict/force refutations and imported Apply witnesses; native equality and numeric-key set failure; managedfields; CRD HTTP/etcd/GET with exact-integer conflict and transfer controls. |
| [SSA-06](https://github.com/kubernetes/kubernetes/issues/143943) | FieldsV1 numeric-key decoding | Exact integer key `2^53+1` is read as its neighbor `2^53`. B adding that distinct list item falsely conflicts against A. | Lean canonical-trie membership refutation; exact native byte/key round trips in vendored and both pinned SMD revisions; CRD HTTP/GET and small-integer control. |
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

## SSA-03: an Update-owned child does not retain its item

A applies:

```yaml
endpoints:
- name: db
  host: old
```

B then Updates the object, changing `host` to `new`. B owns
`endpoints[name="db"].host`, but does not acquire the item's membership. A
still owns that membership. A applies an object omitting `endpoints`, without
force. Raw SMD returns:

```yaml
endpoints: null
```

The item and B's host claim are gone. This formally refutes BASE-07's broad
handoff wording without the existing conditional theorems' lifetime premise.
Owning a child through Update is different from acquiring the item through
Apply. This trace concerns baseline SSA and also occurs on both pinned SMD
revisions without extra required metadata.

For an optional **non-nullable** list, actual HTTP rejects the generated null,
preserving data, claims and resourceVersion. Every identical retry fails from
the same state, even though creating an object without that optional list is
valid. With `nullable: true` on the list schema, HTTP accepts the null and the
loss of the old claims. These are the two policies checked by
`TestSSAUpdateOwnedListItemPruning`.

This is the behavior discussed in existing Kubernetes
[#134902](https://github.com/kubernetes/kubernetes/issues/134902), which was
closed by lifecycle automation rather than a linked production fix.
`SSA/HandoffCounterexamples.lean` proves the reachable ownership transition,
the refutation of unrestricted foreign-value survival, rejection for every
finite retry count under the explicit list-or-absent validator, and the imported
four-function pruning witness with all callees realized. The primitive and
single-version proof boundaries remain explicit.

## SSA-04: status acquires claims on fields it cannot change

A structural CRD can have root fields other than `spec` and `status`. Enable
`subresources: {status: {}}` in the CRD version and use this root schema:

```yaml
type: object
properties:
  settings:
    type: object
    properties:
      desired:
        type: string
  status:
    type: object
    properties:
      observed:
        type: string
```

Apply these requests with the same literal field-manager name `A`, without force:

```yaml
# Main endpoint, initial Apply.
apiVersion: example.test/v1
kind: Example
metadata:
  name: demo
settings:
  desired: one
---
# /status endpoint, next Apply. The non-status value is unchanged.
apiVersion: example.test/v1
kind: Example
metadata:
  name: demo
settings:
  desired: one
status:
  observed: ready
---
# Main endpoint, next Apply: unexpectedly conflicts with A's /status identity.
apiVersion: example.test/v1
kind: Example
metadata:
  name: demo
settings:
  desired: two
```

Main and status are separate manager identities even with the same name. After
the status request, the stored status ownership entry contains both
`f:settings.f:desired` and `f:status.f:observed`. The status strategy restores
all old non-status data, but preserves the computed managedFields. Its
`GetResetFields` names only `metadata` and `spec`; `settings` passes through the
ownership filter. No webhook, default, conversion or schema migration is needed.
The main request then returns a conflict; GET confirms unchanged data, claims
and resourceVersion.

`TestSSAStatusResetOwnership` checks the complete HTTP history. Its conventional
`spec` control confirms that status cannot change or claim spec, and main cannot
change or claim status. The independent bridge and both pinned SMD revisions
check the underlying co-ownership/conflict sequence; they do not simulate REST
subresource filtering. `ResetCounterexamples` formalizes the broader false
contract, while `ApplySource` and `ResetFields` retain conditional general
exclusion results for roots actually present in the filter.

This is BASE-12b in the catalogue. Conventional spec/status exclusion (BASE-12)
is a narrower obligation and is not refuted by this example. The mismatch is in
the CRD status strategy's declared reset scope, independently of PR #336.

## SSA-05: rounded numeric comparison loses intent or misses a conflict

Tracked in [Kubernetes #143942](https://github.com/kubernetes/kubernetes/issues/143942).

The CRD's `spec.versions[*].schema.openAPIV3Schema` contains:

```yaml
type: object
properties:
  amount:
    type: number
  note:
    type: string
```

Neither field is required. No default, CEL rule, webhook, conversion or schema
change is involved. On each fresh resource, A first applies:

```yaml
amount: 9223372036854775807
```

The API stores this exact int64. B then applies without force:

```yaml
amount: 9223372036854775808
```

That numeric token is represented as binary64 `2^63`. It has a different exact
magnitude from A's int64, but mixed comparison rounds A's integer to the same
float. Two fresh-resource variants expose different effects:

| B's request | Persisted result |
| --- | --- |
| Only the numeric assertion above | False whole-object no-op: the original int64 remains, while B co-claims `amount`. B's scalar intent is lost. |
| The same numeric assertion plus `note: changed` | The note makes the whole object differ. The new binary64 magnitude persists without conflict, while A and B both retain the `amount` claim. |
| Exact int64 `9223372036854775806` plus the note | HTTP conflict; all data, claims and resourceVersion remain unchanged. |
| The rounded `2^63` change plus the note, with force | Accepted with the new magnitude, but A and B both retain `amount`. The foreign claim is not transferred. |
| Exact int64 `9223372036854775806` plus the note, with force | Accepted; B owns `amount` and A no longer does. |

`TestSSANumericComparison` verifies response types and values, both claims,
and a persisted GET after every accepted request or conflict. JSON encoders
may print binary64 `2^63` using a rounded shortest decimal spelling such as
`9223372036854776000`; the test observes the decoded float's exact native
magnitude. The initial int64 is checked before B's request, so this is not
pre-rounding in a client.

The two forced variants refute unrestricted BASE-06 and provide its positive
control. The source updater instance actually transfers detected conflicts
and drops empty ownership entries. `forced_integer_control_transfers` checks
that behavior through imported Apply; native managedfields and HTTP/GET agree.
Thus `forced_transfer_is_false` is caused by rounded comparison, rather than
an omitted transfer rule in the model.

### Source cause and equality boundary

SMD `value.EqualsUsing` casts the integer to float64 whenever either operand is
a float; integer/integer comparison remains exact. `CompareUsing` uses the
same mixed conversion. Both MaxInt64 and MaxInt64-1 round to binary64 `2^63`,
although they are different integers. Consequently equality and comparator
incomparability are nontransitive. No faithful quotient into ordinary equality
can repair that relation on the full represented domain.

The final updater misses the numeric modification, so it neither conflicts nor
removes A's claim. The complete imported Apply body then uses whole-object
equality to choose its nil-on-no-op return. The unrelated note changes this
second decision, explaining the two outcomes without changing the number.

The failure also reaches field-path set algebra: with numeric keys
`[MaxInt64-1, MaxInt64]` minus `[binary64 2^63]`, real `Difference` retains
MaxInt64, although the excluded set's real `Has` says it contains that key.
Both inputs are individually sorted. The usual algebra proofs remain valid
under their explicit lawful-comparator premises; the current numeric comparator
cannot satisfy those premises globally.

### Kernel and implementation evidence

[`SSA/NumericCounterexamples.lean`](../SSA/NumericCounterexamples.lean) contains:

- `equality_is_not_transitive`, `incomparability_is_not_transitive` and
  `no_faithful_equality_projection`.
- `difference_retains_an_excluded_member`, executing the complete imported
  element-set body rather than only a replacement algorithm.
- `original_is_source_reachable`, `scalar_only_is_false_noop`,
  `independent_note_changes_foreign_number` and `integer_change_conflicts`,
  executing the complete imported Apply body. The fresh-manager prune callback
  is realized by the imported nil-last-set branch.
- `exact_scalar_intent_is_false` and
  `nonforce_foreign_value_safety_is_false`, refuting unrestricted BASE-01/04.
- `forced_change_retains_foreign_claim` and `forced_transfer_is_false`,
  refuting unrestricted BASE-06, with `forced_integer_control_transfers` as
  the successful exact-comparison control.

The numeric specification models finite nonnegative integral witnesses and
nearest-ties-even integer rounding. It does not prove general IEEE conversion
or all typed/updater primitives. The executable source report is independently
compared with native Go equality, element sets and managedfields; the HTTP test
additionally checks actual request decoding, storage and claims. Those concrete
agreements establish these witnesses without inventing a universal primitive
refinement. The ordinary normalized model's positive laws remain separate.

The bridge's corpus/path/default decoder now preserves int64 tokens before
falling back to float64. Its previous untyped float64 decoding could round away
exact integers before both engines saw them. That adapter defect is fixed and
regression-tested, not counted as a second production bug.

## SSA-06: FieldsV1 rounding aliases distinct integer list keys

Tracked in [Kubernetes #143943](https://github.com/kubernetes/kubernetes/issues/143943).

The CRD's `spec.versions[*].schema.openAPIV3Schema` contains:

```yaml
type: object
properties:
  endpoints:
    type: array
    x-kubernetes-list-type: map
    x-kubernetes-list-map-keys: [id]
    items:
      type: object
      required: [id]
      properties:
        id:
          type: integer
          format: int64
        payload:
          type: string
```

Without force, A applies the first document and B applies the second:

```yaml
# A owns one exact integer-keyed item.
endpoints:
- id: 9007199254740993
  payload: old
---
# B adds a different integer-keyed item.
endpoints:
- id: 9007199254740992
  payload: independent
```

A's stored data and raw FieldsV1 both retain the exact first integer. B's
request should add an independent item, but returns three conflicts against A:
item membership, `id` and `payload`. GET confirms atomic rollback, including
claims and resourceVersion. A control with A's key 42 and B's key 41 succeeds,
preserving both exact items.

### Decoding changes the selector

The standalone fieldpath round trip exposes the same cause:

```text
before: {"f:endpoints":{"k:{\"id\":9007199254740993}":{}}}
after:  {"f:endpoints":{"k:{\"id\":9007199254740992}":{}}}
```

The embedded JSON numeric selector is read into float64; nearest-ties-even
rounding moves `2^53+1` to `2^53`. The next Apply therefore treats A as owning
B's distinct item. This is a different defect from SSA-05's comparison of
already represented numbers: this codec loses an exact integer during parsing.
It reproduces in the vendored v6 dependency and both pinned v7 PR revisions.

Ordinary field-set `Has`/`Equals` checks do not reveal the loss: their mixed
comparison also rounds the original integer. The new bridge check inspects
the exact decoded magnitude and complete reserialized bytes independently.
It does not treat harmless formatting differences as lost ownership.

### Formal and API evidence

[`NumericCodecCounterexamples`](../SSA/NumericCodecCounterexamples.lean) proves
that the starting trie is canonical and encodes the exact original key. Under
the concrete numeric-read specification, membership leaves the original path
and appears at the neighbor. `exact_numeric_path_roundtrip_is_false` refutes
BASE-10 even with that canonical-input premise. The small-key control remains
lossless.

This does not refute `FieldsV1.root_roundtrip`: that structural theorem begins
after path labels have been decoded. General JSON/parser refinement is still
open. The counterexample specifies the known numeric-read callback, rather
than claiming a verified JSON interpreter. Complete real Go round trips agree
with its observations in all three selected dependency revisions.

`TestSSANumericListKeyOwnership` checks actual CRD decoding, the exact initial
wire key, the false conflict, persisted rollback and independent small-key
success. No default, CEL, webhook, conversion or topology migration is involved.

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
apiextensions-apiserver and executes the container, handoff, reset and numeric HTTP tests.

API-version conversion, arbitrary webhook implementations, all defaulting
programs and the complete Go request pipeline do not have universal refinement
proofs. These limits do not weaken the concrete reproduced counterexamples.
The stronger policy's own intent/idempotence theorems describe its separate
transition, rather than current SSA's disproved unrestricted laws.
