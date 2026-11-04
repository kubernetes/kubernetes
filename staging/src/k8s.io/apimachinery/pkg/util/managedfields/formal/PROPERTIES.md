# Property catalogue and proof status

This catalogue separates the existing SSA contract from the proposed
full/partial-item-manager extension. It is a specification and traceability
index, not a claim that Kubernetes or SMD has been verified.

## Reading the status

- **General theorem:** kernel-checked for all model inputs satisfying the stated
  assumptions, of arbitrary finite size.
- **Definition/guard theorem:** kernel-checked, but largely follows by unfolding
  the chosen predicate or the acceptance guard.
- **Execution proof:** kernel-checked for named concrete executions only.
- **Open:** not proved; examples or an executable definition do not close it.
- **Outside the model:** the necessary operation or representation is absent.

Internal list ordering is not an API observation. Intended equivalence compares
cells and claims by membership. Storage resource versions, timestamps, defaulting,
and conversion need their own observational model.

## Sources

The normative starting point is [KEP-555: Server-Side Apply][kep]. The KEP's
[proposal][proposal] calls itself a requirements/mission statement and refers
to older design documents. Its [topology section][topology] and
[testing plan][testing] are useful inputs, not a complete mathematical
specification. We have not incorporated the linked historical Google Docs.

The test links below are pinned to the Kubernetes checkout used to create this
prototype, `aedfaadf0d9a70d819ad681bec674f50ccb42975`. Tests provide concrete
observations to generalize. They do not themselves prove the generalized law,
and implementation behavior must not silently become the normative contract.

Full-manager invariants, lifetime deletion by the last full manager, and required
collection-presence claims are **new design choices**, not existing KEP-555
guarantees.

## Existing SSA obligations

| ID | Observable contract | Source | Current proof boundary |
| --- | --- | --- | --- |
| BASE-01 | Every successfully applied value and presence assertion is present in the result. | KEP goals; typed merge/leaf tests linked in its testing plan. | `Contracts.accepted_preserves_intent`; normalized requests and a valid initial state are required. |
| BASE-02 | An omitted exclusively owned scalar disappears. | [Exclusive omission test][exclusive]. | `Contracts.exclusive_omission_removes_value`; applies to non-defaulted scalar cells in the candidate result. It does not claim the final API value stays absent after defaulting. |
| BASE-03 | An omitted shared scalar survives while another owner remains. | [Shared omission test][shared]. | `Contracts.shared_omission_preserves_value`, with explicit no-overwrite and no-ancestor-deletion assumptions. The latter matters for the proposed extension. |
| BASE-04 | A changed foreign-owned scalar conflicts unless force is set. | [Apply/Update/force test][conflict]. | `Contracts.foreign_change_conflicts` for normalized Apply. Update itself is outside the model. |
| BASE-05 | Asserting the same values cannot alone cause an ownership conflict. | Shared-ownership cases; KEP cooperative-management goal. | `Contracts.equal_writes_do_not_conflict`. Other guards may still reject the request. |
| BASE-06 | Force transfers conflicting value claims, while equal-value claims may remain shared. | [Apply/Update/force test][conflict]. | `Contracts.force_transfers_changed_claim` and `Contracts.unchanged_foreign_claim_preserved` concern `updatedClaims`; subsequent lifetime deletion can still remove a subtree, and acceptance requires valid values and complete item ownership. |
| BASE-07 | Once B has acquired ownership, A's omission must not erase B's retained values. | [Controller handoff test][handoff]. | `Contracts.shared_omission_preserves_value` and `Contracts.handoff_preserves_full_successor` prove safety under explicit ownership and lifetime assumptions. General handoff progress is open. The existing controller test uses Update, which is not yet represented. |
| BASE-08 | Repeating the same intent by the same manager without intervening writes is observationally idempotent. | [Semantic no-op test][noop], [defaulted CRD no-op test][default-noop]. | `Contracts.accepted_apply_idempotent` proves immediate repetition succeeds with exactly the same model state, including force. `Contracts.repeated_apply_observationally_equal` exposes the weaker observation-level result. Defaulting, resourceVersion, and persistence are outside the model. |
| BASE-09 | Atomic units cannot be partially overwritten; granular sibling fields can be independently managed. | [KEP topology][topology]. | Atoms are opaque scalar cells. The independent bridge corpus exercises both topologies against managedfields. General topology mapping and independent-request commutativity remain unproved. |
| BASE-10 | Field sets and paths survive serialization round trips. | [Fields round trip][roundtrip], KEP testing plan. | The Go bridge checks the real FieldsV1 round trip after every request, including API version and operation. This is bounded evidence, not a general serialization theorem. |
| BASE-11 | API-version conversion preserves the intended correspondence of values and claims. | [ManagedFields version conversion test][conversion]. | Outside the model; no theorem assumes that all schema conversions are invertible. |
| BASE-12 | Spec/status reset fields do not acquire claims in the other subresource. | [Reset-fields test][reset]. | Outside the model; subresources and reset filters are absent. |

No unrestricted commutativity or confluence is asserted. Conflicting requests,
force transfers, shared lifetime boundaries, and cross-field validation can make
request order observable.

## General transition theorems

The public contract theorems are in [SSA/Contracts.lean](SSA/Contracts.lean).
They quantify over arbitrary finite model states, schemas, and requests, rather
than an enumeration of examples. The important premises and conclusions are:

| Theorem | Premises | Conclusion |
| --- | --- | --- |
| `accepted_preserves_intent` | Apply succeeds; a cell is asserted in the request. | That exact value or presence marker is in the result. Lifetime deletion and pruning cannot silently erase applied intent. |
| `accepted_actor_claims_exact` | Apply succeeds. | The actor's resulting footprint is exactly its requested footprint. Old actor claims do not linger; new assertions are not silently lost. |
| `exclusive_omission_removes_value` | The actor previously owned the scalar; every old claim for it belongs to that actor; the new footprint omits it. | The scalar is absent from the candidate, hence from any accepted result. |
| `shared_omission_preserves_value` | A foreign scalar owner remains; there is no write at that path or lifetime deletion of an ancestor. | The old scalar survives in the candidate. |
| `foreign_change_conflicts` | Valid initial state, normalized intent, changed foreign-owned scalar, force disabled. | The request returns the conflict error. |
| `accepted_nonforce_preserves_foreign_assertion` | Apply succeeds without force and asserts a foreign-owned scalar. | Its value was not changed. This does not forbid the proposed last-full-manager deletion from removing foreign fields elsewhere. |
| `handoff_preserves_full_successor` | Apply succeeds without force; another manager was full; no lifetime-deletion root covers that manager's claims. | That manager remains full. This is preservation after acquisition, not a proof that acquisition always succeeds. |
| `accepted_reapply_candidate_fixed` | A request has succeeded once. | Running its transition phases again yields exactly that accepted state. |
| `accepted_apply_idempotent` | A request has succeeded once. | Repetition also passes every guard and returns the same state, including with force. |
| `finite_history_preserves_validity` | A valid starting state and any finite list of requests. | The committed result remains valid, including when requests are rejected. Every prefix is another finite list, so the statement applies to every committed prefix. |

The fixed-point proof uses facts about the transition, not an idempotence guard:
accepted results contain every asserted cell, have unique cell paths, retain
exactly the actor's desired claims, and put the requested writes and claims into
a stable list form. Thus repetition makes no value changes, releases no claims,
has no lifetime-deletion roots, and changes neither list. The final acceptance
proof then uses validity of that unchanged state.

These are conditional safety laws for this executable transition. In particular,
intent preservation and idempotence alone would be vacuous for an implementation
that rejects every first request. Successful execution proofs supply concrete
progress witnesses; general progress under useful preconditions is still a
separate obligation.

## Proposed full/partial extension

| ID | Contract | Proof status |
| --- | --- | --- |
| FULL-01 | Every surviving item has an individual full manager after every accepted transition. | Guard theorem: `accepted_complete`, `every_surviving_item_has_full`. |
| FULL-02 | All required branches of one item must be covered by the same manager. | Definition theorem: `required_branches_same_manager`; fragmented-object execution proof. |
| FULL-03 | A partial manager cannot directly authorize item deletion. | Definition theorem: `partial_cannot_directly_delete`. Ancestor deletion is an explicit exception. |
| FULL-04 | Another full manager blocks direct lifetime deletion. | Definition theorem: `surviving_full_blocks_direct_deletion`; full ownership must be checked after transfers. |
| FULL-05 | A full child does not block deletion of an ancestor item. | General subtree theorem: `ancestor_deletion_dominates`; nested execution proofs. |
| FULL-06 | Key-only items remain until their last membership claimant leaves. | `key_only_full` is definitional; sharing and final deletion also have execution proofs. |
| FULL-07 | Fragmenting the required claims is rejected atomically, even when all required values remain. | General acceptance/commit guarantees and concrete counterexample proofs. Existing malformed states require a separate migration policy. |
| FULL-08 | Adding claims cannot destroy existing full ownership. | General `owns_monotone`, `covers_monotone`, `full_monotone`; these do not cover arbitrary claim removal or force. |
| FULL-09 | Required granular collections retain a presence claim even when empty. | Representation assumption plus populated/empty-list execution proofs. A current-SMD adapter is missing. |

The extension must be reviewed against the existing contract. In particular,
FULL-03/FULL-05 create a lifetime-based exception to a naive “every shared field
always survives omission” statement. The tests for controller contributions to
list items also depend on whether the contributor used Apply or Update; this
difference must remain explicit.

## What the first 23 statements meant

The original `SSA/Properties.lean` contains seventeen general statements,
including helpers and definition/guard consequences. The original
`SSA/Examples.lean` contributes six statements: one checks the assertions of
fourteen scenarios (49 transitions), and the other five concern specific
counterexamples or a particular idempotent execution.

Those numbers are historical accounting, not a correctness score or a count of
independent safety guarantees. Newly added contract theorems are listed in
`SSA/Contracts.lean` and audited along with their helper dependencies.

In particular:

- `accepted_valid` checks this model's `valid`, not all Kubernetes validation.
- `rejected_is_atomic` proves the behavior of the model's `commit` wrapper,
  not etcd transactionality.
- `partial_cannot_directly_delete` concerns one lifetime-deletion predicate,
  not every reason an item can disappear.
- The frame theorems concern specific phases or explicitly excluded effects.
- The simplified unsafe PR rule is a countermodel, not a verified translation
  of a particular Go revision.

## Remaining proof layers

Intent preservation, exclusive/shared pruning, conflict behavior, exact
idempotence, and arbitrary finite-history validity now have general proofs.
Finite-history validity builds on the one-step guard theorem; it does not prove
availability or successful handoff.

1. Prove general handoff progress: construct accepted acquisition/release
   requests under explicit preconditions. One successful fixture is insufficient.
2. Prove independence/commutativity only under precise disjointness assumptions
   that also exclude shared lifetime boundaries and validation dependencies.
3. Add Update, normalization/defaulting, conversion, and the ownership protocol
   before claiming the corresponding Kubernetes integration guarantees.
4. Extend the bounded Go adapter to broader schema and normalization coverage.
   Differential testing is implementation evidence; a proof of refinement
   would be a further step.

The bounded Go adapter now exists: [bridge](bridge/) and the independent
[YAML corpus](testdata/bridge.yaml) compare every transition, with exact,
separately classified proposal differences. Generated traces add scalar-domain
coverage. Arbitrary schema normalization and a general refinement theorem remain
open.

## Known model gaps requiring semantic review

- `shapeValid` currently allows a scalar/atomic cell and a descendant cell at the
  same time. The existing preservation theorem therefore preserves a weaker
  predicate than a complete object-tree well-formedness condition.
- Normalization currently gives managers explicit object-presence claims for
  nonempty objects. After A applies `settings.left`, B force-transfers it, and B
  omits it, Lean retains `settings: {}` through A's synthetic presence claim,
  while current SSA removes `settings`. The bridge executes and characterizes
  this exact discrepancy as `differenceKind: model-gap`.

Passing the automated jobs does not resolve these semantic review obligations.
In particular, characterizing a known discrepancy does not establish equivalence
on that input. New discrepancies fail; removing an exception also requires
updating its independent expected outcomes.

No “all requests rejected” implementation should satisfy the entire catalogue:
the progress and intent obligations must be kept alongside preservation.

[kep]: https://github.com/kubernetes/enhancements/blob/7e841ee17c39a07b3aeea71cc76995b662efdeb8/keps/sig-api-machinery/555-server-side-apply/README.md
[proposal]: https://github.com/kubernetes/enhancements/blob/7e841ee17c39a07b3aeea71cc76995b662efdeb8/keps/sig-api-machinery/555-server-side-apply/README.md#proposal
[topology]: https://github.com/kubernetes/enhancements/blob/7e841ee17c39a07b3aeea71cc76995b662efdeb8/keps/sig-api-machinery/555-server-side-apply/README.md#api-topology
[testing]: https://github.com/kubernetes/enhancements/blob/7e841ee17c39a07b3aeea71cc76995b662efdeb8/keps/sig-api-machinery/555-server-side-apply/README.md#testing-plan
[exclusive]: https://github.com/kubernetes/kubernetes/blob/aedfaadf0d9a70d819ad681bec674f50ccb42975/test/integration/apiserver/apply/apply_test.go#L2256
[shared]: https://github.com/kubernetes/kubernetes/blob/aedfaadf0d9a70d819ad681bec674f50ccb42975/test/integration/apiserver/apply/apply_test.go#L2363
[conflict]: https://github.com/kubernetes/kubernetes/blob/aedfaadf0d9a70d819ad681bec674f50ccb42975/test/integration/apiserver/apply/apply_test.go#L747
[handoff]: https://github.com/kubernetes/kubernetes/blob/aedfaadf0d9a70d819ad681bec674f50ccb42975/test/integration/apiserver/apply/apply_test.go#L2473
[noop]: https://github.com/kubernetes/kubernetes/blob/aedfaadf0d9a70d819ad681bec674f50ccb42975/test/integration/apiserver/apply/apply_test.go#L554
[default-noop]: https://github.com/kubernetes/kubernetes/blob/aedfaadf0d9a70d819ad681bec674f50ccb42975/test/integration/apiserver/apply/apply_crd_test.go#L713
[roundtrip]: https://github.com/kubernetes/kubernetes/blob/aedfaadf0d9a70d819ad681bec674f50ccb42975/staging/src/k8s.io/apimachinery/pkg/util/managedfields/internal/fields_test.go#L30
[conversion]: https://github.com/kubernetes/kubernetes/blob/aedfaadf0d9a70d819ad681bec674f50ccb42975/test/integration/apiserver/apply/apply_test.go#L1545
[reset]: https://github.com/kubernetes/kubernetes/blob/aedfaadf0d9a70d819ad681bec674f50ccb42975/test/integration/apiserver/apply/reset_fields_test.go#L55
