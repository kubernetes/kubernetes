# Baseline counterexample: applying an empty associative list

Immediate data idempotence is false for current SSA. This is observable through
the CRD HTTP API, independently of PR #336. The pinned PR head and base also
produce the same inner SMD counterexample.

## Requests and observed data

The list is granular, keyed by `name`, with no defaults. A initially applies:

```yaml
endpoints:
- name: db
  host: old
```

The same manager then repeatedly applies this intent:

```yaml
endpoints: []
```

| Apply of the empty-list intent | Stored data at that field | Stored ownership in the full CRD pipeline |
| --- | --- | --- |
| First | Field absent | No managers remain |
| Second | `endpoints: []` | No managers remain |
| Third | `endpoints: []` | `before-first-apply`, operation `Update`, claims the list's presence |
| Fourth | `endpoints: []` | Same Update claim; this request is a storage no-op |

The first, second and third requests each change `resourceVersion`. Only the
first two change application data. The third is a separate ownership change
in the default field-manager wrapper. The SMD-only bridge omits that wrapper
and therefore reaches a fixed point earlier.

## Why the first Apply loses the empty list

Apply's `ToFieldSet` extracts item membership and child claims for a known
associative list, but no direct list-presence claim, including for `[]`.
Pruning expands the previous item claims to the named list field and removes
it. The newly applied empty list has no ownership claim to restore it. Once
the old item claims are gone, the next identical intent creates the empty list.

This refutes unconditional immediate data idempotence and exact preservation
of every explicitly supplied container. It does not refute scalar omission,
scalar conflict detection, or every conditional idempotence statement.

The FieldsV1 grammar can represent list presence. The third HTTP request
actually produces `{"f:endpoints":{}}` through Update initialization. The
problem is how Apply generates and maintains claims, rather than a limitation
of the serialized field-set grammar.

## Required and CEL validation prevent retries from making progress

Either of these root schema constraints accepts a present empty list:

```yaml
# OpenAPI presence constraint
required: [endpoints]
```

```yaml
# Alternative CEL presence constraint
x-kubernetes-validations:
- rule: has(self.endpoints)
```

The first empty-list Apply instead proposes absence, so validation rejects it.
GET then returns the original populated list, ownership and `resourceVersion`.
Repeating the request starts from that same state and is rejected again.
Creating a separate resource with `endpoints: []` succeeds under both policies,
which confirms that the requested empty value itself is valid.

The HTTP regression exercises both policies with no mutating webhooks. A
webhook which enforces the same presence condition would have the same
old/proposed-value distinction, but arbitrary webhook implementations are not
verified here. A mutating webhook or a schema default could alter this trace.

## Lean statements and implementation checks

- `Counterexamples.raw_empty_list_first_apply_loses_presence` and
  `raw_empty_list_second_apply_adds_presence` prove the two raw transitions.
- `Counterexamples.raw_apply_is_not_unconditionally_idempotent` refutes the
  universally quantified immediate-idempotence proposition.
  `raw_apply_is_not_presence_idempotent` refutes the corresponding
  data-observation law, independently of metadata or internal list ordering.
- `Counterexamples.pr_has_same_empty_list_counterexample` proves the same
  witness for PR semantics without additional required metadata.
- `Counterexamples.required_list_retries_never_empty_the_list` proves that
  every finite number of validation-and-commit retries preserves the populated
  state. This uses the explicit presence-validator policy.
- `RawSourceCounterexamples.imported_pruning_loses_explicit_empty_list` and
  `imported_pruning_second_apply_retains_empty_list` connect the witness to the
  imported source AST. All pruning callees are realized, and the statements
  quantify over every lawful version-map iteration schedule. Typed/set
  primitives remain the normalized instance, with their existing proof boundary.
- `testdata/bridge.yaml` checks exact data and sparse ownership against the
  actual managedfields implementation. Its explicit `bootstrap: true` case
  wraps the actual `NewSkipNonAppliedManager`, matching the separate
  `Bootstrap.bootstrapExisting` Update-from-empty model.
  `Counterexamples.default_wrapper_bootstraps_empty_list_owner` proves the
  concrete ownership transition; `default_wrapper_fourth_apply_is_fixed`
  proves its next fixed point. Generated structural Apply histories also
  exercise this wrapper. `testdata/pr336.yaml` checks both pinned SMD
  revisions without this Kubernetes-only wrapper. The traces use `items` as
  the field name.
- `TestSSAEmptyAssociativeListPresence` in the Kubernetes
  `apiextensions-apiserver/test/integration` package checks actual HTTP Apply,
  persisted data and FieldsV1, OpenAPI required validation, CEL evaluation,
  atomic rejection, the default wrapper's Update initialization, and
  `resourceVersion`. It uses `endpoints` to avoid the top-level `items` convention
  that identifies Kubernetes list objects.

The general `Contracts.accepted_apply_idempotent` theorem continues to describe
the stronger proposed item-core transition. This counterexample proves why
that theorem cannot be promoted to unconditional correctness of current SSA.
The synthetic-wrapper ownership transition now has a separate model, concrete
proof, inner-wrapper comparison and actual API regression. A general source
refinement of that wrapper, including runtime metadata and error handling,
remains a separate obligation. The bootstrap comparison accepts Apply-only
histories; probabilistic tracking of non-Apply operations is outside its domain.

## Running the checks

From this formal project directory:

```sh
bash ci/verify.sh
bash ci/verify-apiserver.sh
```

The first command checks the Lean kernel and both SMD revisions. The second
requires a Kubernetes checkout, starts a real API server and etcd, and runs the
HTTP regression. It installs the checkout's normal etcd dependency when absent
and retains the complete logs. The proposed Kubernetes Prow job runs both.
