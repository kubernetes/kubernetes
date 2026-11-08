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

The completion target is a proof or a counterexample for every catalogue entry.
A positive model theorem does not close the corresponding implementation
obligation while its Go primitives or request stages remain unverified. A
counterexample can close an unrestricted claim when its Lean witness and an
independent execution of the implementation agree. A narrower positive theorem
is recorded alongside a refutation, rather than substituted for the original
claim. Definition and guard theorems remain identified as such.

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
| BASE-01 | Owned scalar intent is retained; empty/null containers need phase-specific treatment. | KEP goals; typed merge/leaf tests linked in its testing plan. | **Refuted for unrestricted baseline numeric scalars.** `NumericCounterexamples.exact_scalar_intent_is_false` gives an accepted false no-op that discards a valid numeric assertion while granting its claim; actual managedfields and CRD HTTP/GET reproduce it. `Contracts.accepted_preserves_intent` remains the separate proposed-core theorem. Current/PR empty/null handling also does not inherit that stronger exact-container conclusion. |
| BASE-02 | An omitted exclusively owned scalar disappears. | [Exclusive omission test][exclusive]. | `Contracts.exclusive_omission_removes_value`; applies to non-defaulted scalar cells in the candidate result. It does not claim the final API value stays absent after defaulting. |
| BASE-03 | An omitted shared scalar survives while another owner remains. | [Shared omission test][shared]. | `Contracts.shared_omission_preserves_value`, with explicit no-overwrite and no-ancestor-deletion assumptions. The latter matters for the proposed extension. |
| BASE-04 | A changed foreign-owned scalar conflicts unless force is set. | [Apply/Update/force test][conflict]. | **Refuted for unrestricted baseline numeric scalars.** `NumericCounterexamples.nonforce_foreign_value_safety_is_false` changes A's exact numeric magnitude without force while retaining A's claim; managedfields and CRD HTTP/GET reproduce it. `WireContracts.foreign_change_conflicts` and `accepted_nonforce_foreign_unchanged` remain general normalized-model laws, conditional on faithful typed comparison for application to Go. Update is a separate transition. |
| BASE-05 | Asserting the same values cannot alone cause an ownership conflict. | Shared-ownership cases; KEP cooperative-management goal. | `Contracts.equal_writes_do_not_conflict`. Other guards may still reject the request. |
| BASE-06 | Force transfers conflicting value claims, while equal-value claims may remain shared. | [Apply/Update/force test][conflict]. | Current/PR `WireContracts.force_transfers_changed_claim`, `unchanged_foreign_claim_preserved` and `accepted_foreign_claim_exact` now characterize claims in the final accepted result. `forced_accepts_valid_candidate` proves progress through the conflict gate while retaining structural checks. Typed comparison/sets and the Go updater remain implementation obligations. The separate proposed-core results concern its own ownership phase. |
| BASE-07 | Once B has acquired ownership, A's omission must not erase B's retained values. | [Controller handoff test][handoff]. | **Refuted without an item-lifetime premise.** `HandoffCounterexamples.unrestricted_foreign_omission_safety_is_false` gives a reachable Apply/Update history: B owns a scalar child but not its item, and A's non-forced omission removes both. Nullable CRD HTTP accepts that deletion; non-nullable validation rejects the generated null. The conditional `Contracts.shared_omission_preserves_value` / `handoff_preserves_full_successor` retain their explicit lifetime assumptions. General Go refinement of those positive laws remains open. |
| BASE-08 | Repeating the same intent by the same manager without intervening writes is observationally idempotent. | [Semantic no-op test][noop], [defaulted CRD no-op test][default-noop]. | **Refuted as an unrestricted baseline law.** `Counterexamples.raw_apply_is_not_unconditionally_idempotent` supplies an empty-associative-list witness, reproduced through CRD HTTP and both pinned SMD revisions. `Contracts.accepted_apply_idempotent` proves the stronger proposed item-core transition only. See [the counterexample](docs/baseline-empty-list.md). |
| BASE-09 | Atomic units cannot be partially overwritten; granular sibling fields can be independently managed. | [KEP topology][topology]. | Atoms are opaque scalar cells. The independent bridge corpus exercises both topologies against managedfields. General topology mapping and independent-request commutativity remain unproved. |
| BASE-10 | Field sets and paths survive serialization round trips. | [Fields round trip][roundtrip], KEP testing plan. | **Refuted for unrestricted numeric keys.** `NumericCodecCounterexamples.exact_numeric_path_roundtrip_is_false` moves membership from exact integer key `2^53+1` to its neighbor. The original trie is canonical and encodes the exact key. Complete real FieldsV1 round trips reproduce it in vendored and both pinned SMD versions; CRD HTTP shows a false conflict between distinct integer-keyed items. `FieldsV1.root_roundtrip` / `root_membership_roundtrip` remain arbitrary-depth structural theorems after labels have been decoded. General JSON/parser refinement remains open. |
| BASE-11 | API-version conversion preserves the intended correspondence of values and claims. | [ManagedFields version conversion test][conversion]. | Outside the model; no theorem assumes that all schema conversions are invertible. |
| BASE-12 | Spec/status reset fields do not acquire claims in the other subresource. | [Reset-fields test][reset]. | `ApplySource.imported_apply_excludes_reset_fields` proves exclusion through the complete imported Apply body, conditional on the configured prefix filter and the final updater's explicit actor-non-growth contract. `ResetFields` proves arbitrary-depth prefix exclusion, idempotence and trie membership refinement. HTTP controls confirm both directions for conventional spec/status fields. The Go filter implementation, updater non-growth and complete subresource envelope remain refinement obligations. |
| BASE-12b | Every field restored by a subresource's REST strategy is also excluded from that subresource's new claims. | Generalization of BASE-12 to valid CRDs with additional root fields. | **Refuted.** `ResetCounterexamples.broad_status_reset_contract_is_false` gives a reachable co-ownership/conflict witness. The CRD status strategy restores `settings`, but its configured reset roots exclude only spec and metadata. HTTP reproduces a status claim on unchanged `settings.desired`, followed by a conflict on the original main manager's next change. This does not refute the narrower conventional spec/status contract. |

For the positive BASE-04/06 wire-model theorems, “changed” compares final model
values, not merely request payloads. The statements quantify over every foreign manager and both
Apply/Update ownership flags. Presence of the final target is explicit; a
removed target instead loses its claim by `removed_foreign_claim_dropped`.
These general wire-model proofs do not close the Go implementation obligations
listed in their rows. In particular, defining semantic change by Go's rounded
comparison would hide the numeric BASE-04 refutation. Observable numeric
magnitudes differ in that witness; the existing comparator misses the change.

SSA-05 starts with A owning exact int64 `9223372036854775807`. B asserts
`9223372036854775808`, decoded as binary64 `2^63`, without force. With only
`amount`, Apply treats the result as a no-op, keeps A's old value and adds B's
claim. If B also adds unrelated `note: changed`, the numeric change persists,
with both claims intact. An int64-to-int64 change to `9223372036854775806`
correctly conflicts. No required rule, default, CEL or webhook is involved.
This closes the unrestricted BASE-01/04 claims by counterexample, not the
remaining positive implementation obligations. See [SSA-05](docs/ssa-bugs.md#ssa-05-rounded-numeric-comparison-loses-intent-or-misses-a-conflict).

BASE-07's counterexample starts from an empty object. A applies
`items: [{name: db, host: old}]`; B uses Update to set `host: new`. B acquires
only the host leaf, while A retains item membership. A then applies `{}` without
force. Raw SSA produces `items: null`, with neither the item nor B's host claim.
This is distinct from B acquiring item membership through Apply. The known
[Kubernetes #134902](https://github.com/kubernetes/kubernetes/issues/134902)
documents the resulting non-nullable-list validation failure. The new HTTP
test checks both schema policies and unchanged storage after every rejected
retry; the source witness realizes all four imported pruning functions under
every lawful version-map schedule and normalized primitives.

No unrestricted commutativity or confluence is asserted. Conflicting requests,
force transfers, shared lifetime boundaries, and cross-field validation can make
request order observable.

## General proposed-core transition theorems

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
progress witnesses; `Progress.unchanged_snapshot_acquisition` additionally proves successful acquisition under general fresh-manager and unchanged-snapshot preconditions.

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
| FULL-09 | Required granular collections retain a presence claim even when empty. | Explicit presence is a proposed representation assumption. The current/PR modes retain actual sparse FieldsV1 and prove concrete misclassification counterexamples; the upstream runner confirms them. |

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
- The original simplified unsafe rule remains a countermodel. `Semantics` and
  `Pruning` separately model the pinned PR and are compared to its actual Go code.

## Structural, progress and operation theorems

| Area | General statement | Boundary |
| --- | --- | --- |
| Structure | `Contracts.proper_ancestor_is_container`, `scalar_has_no_descendants` | Arbitrary path depth and finite cell sets; no recursive tree-codec refinement theorem. |
| Locality | `Locality.required_core_local`, `full_manager_is_item_local` | Agreement of claims below the scope; independent scope does not imply arbitrary request commutativity. |
| Acquisition | `Progress.unchanged_snapshot_acquisition` | Fresh actor, normalized sparse footprint, complete snapshot of unchanged data and a valid initial state; succeeds and extends old claims. |
| Object scope | `accepted_objects_have_full_owner`, `object_full_is_local`, `partial_object_cannot_directly_delete` | Optional required-bearing maps and root constraints are instantiated independently. Item and ordinary map lifetimes remain distinct. |
| Update | `update_candidate_replaces_data`, `update_preserves_untouched_claim`, `update_claims_touched_target` | Full replacement, touched-field ownership, operation flags; version/subresource envelopes are not formalized. |
| Boundary | `Boundary.accepted_final_valid`, `accepted_satisfies_validator`, `validation_rejection_is_atomic` | Arbitrary mutator/relational validator functions, final check and atomic commit. No CEL/webhook interpreter correctness theorem. |
| Validation limits | `value_only_validation_cannot_enforce_ownership`, `validator_can_block_any_handoff` | Same data can carry different claims; arbitrary external validation can prevent otherwise safe progress. |
| PR compatibility | `no_required_transition_equivalence` | All finite states/requests and metadata-free item schemas, actual three-pass pruning modes. |
| PR classifier | `explicit_map_shortcuts_required_children`, `granular_list_without_member_is_partial`, `updater_keeps_membership` | Direct classifier facts, distinct from a complete Go refinement proof. |
| PR direct-member projection | `State.ownsMember`, `direct_value_member_covers_containers` | FieldSet has no value/presence tag. Explicit null is a direct member and is counted by the raw classifier. Pinned map/list controls close the former model mismatch; API nullability and the stronger policy need their separate schema mapping. |
| Raw pruning | `remove_items_has_no_selected_descendant` | Selected subtrees are absent; unselected empty container slots may become null. |
| Raw ownership | `WireContracts.merged_foreign_claim`, `candidate_foreign_claim_exact` | Foreign claims participate in pruning before transfer; final survival is characterized exactly. These are current/PR wire-model laws. |
| Raw conflicts/force | `WireContracts.accepted_nonforce_foreign_unchanged`, `candidate_force_independent`, `accepted_also_succeeds_forced` | The final pruned result is compared with original ownership. Force changes the conflict gate, not the candidate. |
| Raw accepted ownership | `WireContracts.foreign_change_conflicts`, `forced_accepts_valid_candidate`, `accepted_foreign_claim_exact`, `force_transfers_changed_claim`, `unchanged_foreign_claim_preserved` | General current/PR results for final accepted data and claims. Typed comparison, field-set operations and Go updater orchestration remain refinement obligations. |
| Raw acquisition | `WireContracts.unchanged_snapshot_acquisition` | Both current/PR modes, arbitrary structurally valid finite states, fresh actor, normalized unchanged snapshot; all original claims survive. No full-manager premise. |
| Source helpers | `SourceContracts.imported_dangling_matches_spec`, `imported_owned_version_matches_spec` | Complete Go ASTs, all error/missing-version branches, returned failure payloads and explicit primitive parameters. |
| Source Apply | `ApplySource.imported_apply_matches_spec`, `imported_apply_excludes_reset_fields` | Complete Go AST, reconciliation/merge/extraction errors, filter choice, old-set snapshot, installation, prune/update ordering and nil-on-no-op return. Reset exclusion requires the stated filter and final updater non-growth contract; the opaque primitives are not thereby verified. |
| Source element algebra | `ElementAlgebra.imported_matches`, `imported_membership`, `imported_preserves_order`, `imported_preserves_uniqueness` | Complete Go ASTs of PathElementSet Difference, Union and Intersection. Shared induction covers arbitrary finite loop lengths, short-circuit bounds, safe indices, equal-key representatives and final suffixes. Membership and canonical outputs require sorted inputs and explicit strict-total-order comparator laws. The existing difference theorem statements remain specializations, including order preservation without comparator laws or a sorted right input. The actual comparator and recursive Set/SetNodeMap operations remain refinement obligations. |
| Source version loops | `OwnedSource.imported_owned_items_matches_spec` | Complete Go AST, arbitrary finite manager/version counts, lexical scope, last-applied version first, every lawful remaining-map permutation and all propagated errors. |
| Source orchestration | `PruneSource.imported_prune_matches_spec` | Complete Go AST of `prune`; nil/empty early exits, initial conversion, both restoration phases, errors and final conversion. Callees are explicit contracts. |
| Pruning composition | `PruningRefinement.single_version_pruning_refinement` | Realizes all three callee contracts from imported helpers and equates source pruning to the three-pass model, for arbitrary states/field sets and any finite nonempty same-version manager map. Typed/set primitive implementations remain obligations. |
| Field-set normalization | `FieldPathSet.paths_membership`, `flat_difference_refines_trie`, `flat_intersection_refines_trie` | Arbitrary finite trie depth. Ordinary membership difference is distinct from subtree removal. Raw selection may retain empty structural nodes, so this is not a JSON codec theorem. |
| Baseline idempotence counterexample | `Counterexamples.raw_apply_is_not_unconditionally_idempotent`, `pr_has_same_empty_list_counterexample` | Explicit accepted raw transitions change field presence on immediate repetition. The universal baseline proposition is therefore false, independently of the PR. |
| Source counterexample | `RawSourceCounterexamples.imported_pruning_loses_explicit_empty_list` | Imported pruning AST with all callees realized, normalized primitives and every lawful iteration schedule; checked against actual Go and API execution. |
| Required-list retry failure | `Counterexamples.required_list_retries_never_empty_the_list` | Every finite retry count under the explicit presence validator; the real OpenAPI and CEL implementations reproduce the rejection through HTTP. |
| Default-wrapper initialization | `bootstrap_preserves_data`, `bootstrap_keeps_nonempty_manager_map`, `bootstrap_creates_update_claims` | General laws of the explicit Update-from-empty model. Concrete ownership-bootstrap/fixed-point proofs, the actual wrapper comparison and HTTP regression agree. Full wrapper source refinement and probabilistic Update tracking remain open. |
| Empty-map intent loss | `ContainerCounterexamples.empty_map_first_apply_creates_null`, `applied_empty_map_intent_is_not_preserved` | An explicit map claim survives while the asserted empty object becomes null. Confirmed on both pinned SMD revisions and through CRD HTTP. |
| Optional-map retry failure | `ContainerCounterexamples.nonnullable_map_retries_never_empty` | Every finite retry count under object-or-absent validation. The actual optional non-nullable map schema rejects the produced null; no required-field rule is needed. |
| Empty-map conflict/default controls | `empty_map_can_conflict_with_identical_previous_intent`, `empty_object_default_repairs_this_proposal` | Concrete stored-null conflict, co-ownership after repair and a specific default mutator. HTTP tests exercise nullable/CEL/defaulted/atomic variants; these are not universal defaulting theorems. |
| Handoff counterexample | `HandoffCounterexamples.unrestricted_foreign_omission_safety_is_false`, `nonnullable_list_omission_retries_fail`, `imported_pruning_removes_update_child` | Reachable Apply/Update history; foreign scalar ownership alone does not protect its item. Lean, actual Go, both pinned SMD revisions and nullable/non-nullable HTTP tests check the trace. Positive safety still needs lifetime premises. |

`SSA/Counterexamples.lean` kernel-checks the raw PR fragmentation, partial-owner
deletion, required-map shortcut, required-list failure, and opposite roles in
`foo.bar` / `foo.abc`. The sibling fixture checks whole-request rejection and
scope-local optional-object deletion. These finite witnesses complement the
general locality and acquisition theorems; they are not universal liveness proofs.

## Implementation checks and boundaries

The baseline adapter now accepts supplied structural schemas, recursive named
types, composite keys, scalar sets, JSON scalars, atomics, unknown map entries,
empty/null containers and Apply/Update. It compares actual sparse ownership,
including map/container presence. A separate harness runs the same independent
PR corpus against pinned v7 head and base, including merge tests and field-set
wire round trips. No arbitrary mismatch is tolerated; there is no `model-gap`
exception category.

The former scalar-ancestor and synthetic nonempty-map presence defects are
closed. General structural theorems cover the former, and the force/unclaim
regression agrees with current managedfields for the latter. The broader
comparison also led to explicit modeling of implicit key leaves, named fields,
three-pass restoration, null promotion and final ownership updates.

The catalogue audit additionally found that a direct member acquired by applying
null remains valid when another manager adds children to that granular map or
list. FieldSet members have no permanent value/presence type. The quantified
null-map control in `ContainerCounterexamples` and explicit Go controls
cover that transition; the empty-map conflict remains a contrasting case.
`wireMergeCell` also models typed merge's distinction between nonempty granular
containers (where null is a merge operand) and empty/null containers (which are
treated as leaves). A separate deterministic nullable-container family exercises
these changes through mixed Apply/Update histories. These corrections close
observed model discrepancies, not the universal typed-merge refinement.

The adapter audit also found an operation/name alias: literal Apply manager
`A@update` collided with an Update by `A`. Operation-tagged escaping now preserves
their distinct identities. Independent Go/pinned cases check both ownership
and later omission, and collision tests include quoted, Unicode and escaped
names. This corrects the comparison boundary; it does not prove the complete
Kubernetes manager-identifier JSON codec or add multi-version/subresource support.

A source-bound refinement covers the four pruning functions, including both
unbounded finite map loops. Same-version composition realizes the callee
contracts and specializes the entire pruning path to the normalized model.
The importer, subset semantics, import/signature checks and primitive contracts
need human review. Unsupported syntax and unmodeled aliasing are rejected.
Source digests and generated ASTs are checked by CI; eight semantic Go mutations
must fail in their expected proof modules. The five updater imports include the
complete Apply body; its source theorem covers orchestration, including reset
filter selection and installation. Its merge, extraction, reconciliation,
pruning and final updater callbacks remain explicit contracts. The pruning
composition realizes the four pruning callees; it does not discharge the
remaining updater or typed/set implementations. Successful callback values
are non-nil and logical managed maps are allocated; arbitrary nil-success
payloads and aliasing side effects are not modeled. Version conversion,
defaulting, schema evolution, complete subresource envelopes, duplicate-list behavior,
key defaults and storage concurrency remain integration obligations. The
adapter rejects unsupported/duplicate shapes rather than silently projecting
them away. Required-empty-list fullness still needs a representation design;
the actual PR modes expose that problem instead of assuming it solved.

The general exact idempotence theorem concerns the proposed item-core `apply`,
not arbitrary admission policies, Update, or the entire added object-scope
operation. The unrestricted current-SSA version is false: an explicitly applied
empty list can disappear on the first request and reappear on the second.
The Kubernetes HTTP regression additionally observes a later synthetic Update
owner from `skipNonAppliedManager`. A separate bootstrap-enabled bridge and
model now cover that ownership transition for Apply-only histories. These are
documented counterexamples, not suppressed comparison discrepancies.
The original history theorem is a core safety induction; `scoped_history_preserves_invariants` also covers mixed scoped Apply/Update histories. The boundary
composition theorems and bounded implementation comparisons are additional,
different forms of evidence.

The three element-set imports are `PathElementSet.Difference`, `Union` and
`Intersection` from the selected dependency's `fieldpath/element.go`. Their
shared source proof covers fresh output allocation, slices, short-circuit
bounds, pure element comparisons and index increments with a linear fuel bound
at arbitrary finite sizes. All usual membership laws, sorted outputs and
uniqueness follow under explicit sorted-input/comparator premises. Nat rank
ordering has a constructive law witness; the old difference-only statements
retain their original assumptions through specialization.

The independent Go check additionally covers all 4,096 subset pairs of a fixed
six-element domain per operation (12,288 cases total), spanning field, key,
value and index selectors, with input-size and membership preservation checks.
A mutation of each method's equal-key guard must fail the unchanged shared
proof. A faithful equality/order realization over the unrestricted mixed Go
numeric domain is **refuted**, rather than left as an assumed future proof.
`NumericCounterexamples.equality_is_not_transitive` and
`no_faithful_equality_projection` show that no ordinary-identity quotient can
faithfully encode even the represented witness domain. Comparator
incomparability is also nontransitive. Sorted numeric-key sets yield an imported
Difference result containing a member that the excluded set's real `Has` finds;
the native/source comparison checks this exact witness.

The rounding specification covers only finite nonnegative integral witnesses,
with native Go and HTTP independently checking the concrete results. General
IEEE conversion, typed comparison, `Has`/binary search, recursive Set/SetNodeMap
operations and complete composition with Apply still need their own treatment.
The conditional element-algebra theorems remain valid; their comparator premises
cannot be universally discharged for the existing mixed numeric implementation.
These element
components do not silently discharge the whole reset filter or the normalized
full-path primitives used by pruning.

The numeric-key round-trip refutation is a distinct codec failure (SSA-06).
The wire ownership key initially contains exact `9007199254740993`; reading
its embedded JSON number produces float64 `9007199254740992`. The native
check inspects exact magnitudes and reserialized bytes: real `Has`/`Equals`
cannot serve as an independent oracle because their mixed comparison also
rounds. The existing per-transition round-trip equality check therefore did
not establish BASE-10 on that domain. The new source-independent comparison
and real API test close that unrestricted claim by counterexample. The Lean
counterexample specifies this concrete numeric read callback; it does not
claim to verify the full JSON/parser implementation.

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
