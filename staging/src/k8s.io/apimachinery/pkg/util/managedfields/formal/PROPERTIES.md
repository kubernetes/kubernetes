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
| BASE-02 | An omitted exclusively owned scalar disappears. | [Exclusive omission test][exclusive]. | **Refuted for unrestricted numeric-keyed items.** `NumericOmissionCounterexamples.unconditional_exclusive_omission_is_false` establishes raw exclusive ownership before decoding, then the unchanged omitted payload survives through phantom sharing with another item's owner. Four imported pruning functions are linked under normalized primitives; native managedfields and real CRD HTTP reproduce the reachable history, with small-key and no-B controls. `Contracts.exclusive_omission_removes_value` remains the proposed-core theorem. General typed/set refinement remains open; no default or mutation is involved. |
| BASE-03 | An omitted shared scalar survives while another owner remains. | [Shared omission test][shared]. | **Refuted without an enclosing-item lifetime premise, including in baseline SSA.** `HandoffCounterexamples.unrestricted_shared_omission_safety_is_false` first establishes actual A/B scalar co-ownership: B Updates the child and A reasserts its unchanged value. A then omits the item and removes both claims and the value. Imported pruning, native managedfields, both pinned SMD revisions and nullable/non-nullable CRD HTTP check the history. An Apply-owned-item control survives. `Contracts.shared_omission_preserves_value` remains the conditional proposed-core law, with no-overwrite and no-ancestor-deletion assumptions. |
| BASE-04 | A changed foreign-owned scalar conflicts unless force is set. | [Apply/Update/force test][conflict]. | **Refuted for unrestricted baseline numeric scalars.** `NumericCounterexamples.nonforce_foreign_value_safety_is_false` changes A's exact numeric magnitude without force while retaining A's claim; managedfields and CRD HTTP/GET reproduce it. `WireContracts.foreign_change_conflicts` and `accepted_nonforce_foreign_unchanged` remain general normalized-model laws, conditional on faithful typed comparison for application to Go. Update is a separate transition. |
| BASE-05 | Asserting the same values cannot alone cause an ownership conflict. | Shared-ownership cases; KEP cooperative-management goal. | `Contracts.equal_writes_do_not_conflict`. Other guards may still reject the request. |
| BASE-06 | Force transfers conflicting value claims, while equal-value claims may remain shared. | [Apply/Update/force test][conflict]. | **Refuted for unrestricted baseline numeric changes.** `NumericCounterexamples.forced_transfer_is_false` changes the exact magnitude with force while A retains its amount claim. Actual managedfields and CRD HTTP/GET reproduce this; an exact integer forced change correctly transfers the claim in Lean, Go and HTTP. Current/PR `WireContracts.force_transfers_changed_claim`, `unchanged_foreign_claim_preserved`, `accepted_foreign_claim_exact` and `forced_accepts_valid_candidate` retain their normalized comparison premises. Typed primitives/updater refinement remains separate; the stronger core concerns its own phase. |
| BASE-07 | Once B has acquired ownership, A's omission must not erase B's retained values. | [Controller handoff test][handoff]. | **Refuted without an item-lifetime premise.** `HandoffCounterexamples.unrestricted_foreign_omission_safety_is_false` gives a reachable Apply/Update history: B owns a scalar child but not its item, and A's non-forced omission removes both. Nullable CRD HTTP accepts that deletion; non-nullable validation rejects the generated null. The conditional `Contracts.shared_omission_preserves_value` / `handoff_preserves_full_successor` retain their explicit lifetime assumptions. General Go refinement of those positive laws remains open. |
| BASE-08 | Repeating the same intent by the same manager without intervening writes is observationally idempotent. | [Semantic no-op test][noop], [defaulted CRD no-op test][default-noop]. | **Refuted as an unrestricted baseline law.** `Counterexamples.raw_apply_is_not_unconditionally_idempotent` supplies an empty-associative-list witness, reproduced through CRD HTTP and both pinned SMD revisions. `Contracts.accepted_apply_idempotent` proves the stronger proposed item-core transition only. See [the counterexample](docs/baseline-empty-list.md). |
| BASE-09 | Atomic units cannot be partially overwritten. | [KEP topology][topology]. | Atoms are opaque scalar cells. `AtomicFieldContracts.imported_atomic_fields` binds complete typed field-set visitors: an atomic scope adds only its whole nonempty path, never visiting children; every other membership is preserved. The empty root is not tracked. `AtomicMergeContracts.imported_atomic_merge` now composes complete map/list dispatch, doLeaf and the real ruleKeepRHS closure: whole right operand if present, otherwise whole left, with a previously selected leaf protected and per-call cleanup. Explicit JSON null differs from a missing Go interface. `AtomicRemoveContracts.imported_atomic_container` additionally proves whole extraction and prior-output preservation on removal from complete structural scalar/map/list ASTs; `imported_atomic_remove_fresh` requires initially empty output. `RemoveFactoryContracts.imported_factory` derives fresh output from the complete caller; `RemoveFactoryBridge` composes selected source handlers. Actual schema selection, source coercion/Unstructured/allocator contracts, Set.Insert, typed method dispatch, general removal/comparison and complete request composition remain open; the entire atomic-overwrite contract is not yet closed. The granular-independence part is BASE-09b. |
| BASE-09b | A fresh manager can add a distinct granular list-map item without conflicting with ownership of an existing item. | Granular-list independence from [KEP topology][topology], previously part of BASE-09. | **Refuted for valid integer-keyed items.** `GranularKeyCounterexamples.unconditional_independent_item_progress_is_false` executes imported Apply after the concrete numeric ownership decoder; distinct unclaimed neighbor intent conflicts. Initial raw ownership is source-reachable, with a realized fresh-manager pruning callback and a successful small-key control. Native managedfields and actual CRD HTTP reproduce it. Atomic integrity and conditional independence remain separate positive obligations. |
| BASE-10 | Field sets and paths survive serialization round trips. | [Fields round trip][roundtrip], KEP testing plan. | **Refuted for unrestricted numeric keys.** `NumericCodecCounterexamples.exact_numeric_path_roundtrip_is_false` moves membership from exact integer key `2^53+1` to its neighbor. The original trie is canonical and encodes the exact key. Complete real FieldsV1 round trips reproduce it in vendored and both pinned SMD versions; CRD HTTP shows a false conflict between distinct integer-keyed items. `FieldsV1.root_roundtrip` / `root_membership_roundtrip` remain arbitrary-depth structural theorems after labels have been decoded. General JSON/parser refinement remains open. |
| BASE-11 | API-version conversion preserves the intended correspondence of values and claims. | [ManagedFields version conversion test][conversion]. | Outside the model; no theorem assumes that all schema conversions are invertible. |
| BASE-12 | Spec/status reset fields do not acquire claims in the other subresource. | [Reset-fields test][reset]. | `UpdaterBridge.source_apply_excludes_reset_fields` composes complete imported Apply with a source-backed internal updater. `UpdaterContracts.imported_update_never_adds_claims` derives the updater frame for every manager from the exact left-subset law for Difference. `ResetFields` proves arbitrary-depth prefix exclusion, idempotence and trie membership refinement. HTTP controls confirm conventional spec/status behavior. Recursive Difference exact-path origin is source-bound through `SetContracts`. `SetFilterContracts.locally_ordered_filter_excludes_root` additionally connects conventional flat-root exclusion to both RecursiveDifference bodies and the actual Filter wrapper, under local direct-label order and Has recognition. `FilterHasBridge.source_filter_excludes_root` now derives that recognition through complete Has/Less and selected standard-library search source, with local comparator-zero faithfulness and Go word bounds. Other recursive set/filter laws, ownership representation, interpreter/runtime refinement and the complete subresource envelope remain obligations; the original theorem with an explicit updater contract remains available. |
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
With force, the same rounded change also retains A's claim, refuting BASE-06.
The updater instance implements forced transfer and empty-entry cleanup: an
exact int64-to-int64 forced control correctly moves the claim to B. This
control is proved in `forced_integer_control_transfers` and independently
checked through managedfields and HTTP, so the force refutation does not rely
on omitting the transfer rule from the model.
This closes the unrestricted BASE-01/04/06 claims by counterexample, not the
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
| FULL-06 | Within a surviving enclosing lifetime, a key-only item remains until its last membership claimant leaves. | `KeyOnly.direct_deletion_iff_last_membership` characterizes the direct rule for arbitrary finite claim sets, including with force. `accepted_last_membership_prunes_subtree` proves complete subtree removal after the last release. `accepted_foreign_membership_preserves_item` proves survival while a foreign member remains, with an explicit no-proper-ancestor-deletion premise. These are proposed item-core laws; `key_only_full` remains the separate definition theorem. |
| FULL-07 | Fragmenting the required claims is rejected atomically, even when all required values remain. | General acceptance/commit guarantees and concrete counterexample proofs. Existing malformed states require a separate migration policy. |
| FULL-08 | Adding claims cannot destroy existing full ownership. | General `owns_monotone`, `covers_monotone`, `full_monotone`; these do not cover arbitrary claim removal or force. |
| FULL-09 | Required granular collections retain a presence claim even when empty. | Explicit presence is a proposed representation assumption. The current/PR modes retain actual sparse FieldsV1 and prove concrete misclassification counterexamples; the upstream runner confirms them. |

The extension must be reviewed against the existing contract. In particular,
FULL-03/FULL-05 create a lifetime-based exception to a naive “every shared field
always survives omission” statement. The tests for controller contributions to
list items also depend on whether the contributor used Apply or Update; this
difference must remain explicit.

FULL-06 is a local lifetime rule, subject to FULL-05. Unconditional survival of
a nested key-only item while any member remains would contradict the already
proved ancestor-deletion rule. `KeyOnly` makes that boundary explicit: the
survival theorem excludes only **proper ancestor** item deletions, and derives
the absence of direct deletion from retained membership. It does not assume
that the item survived. Acceptance supplies schema uniqueness and final
data/claim consistency. The last-release theorem removes all descendants,
including optional foreign-owned fields. Ordinary object lifetimes belong to
the separate scoped operation; these new theorems concern the item-core
transition rather than arbitrary admission or every scoped operation.

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
| Source Apply | `ApplySource.imported_apply_matches_spec`, `imported_apply_excludes_reset_fields` | Complete Go AST, reconciliation/merge/extraction errors, filter choice, old-set snapshot, installation, prune/update ordering and nil-on-no-op return. The original reset theorem retains its updater contract; `UpdaterBridge` now supplies it through imported update execution under Difference left-subset preservation. The configured filter and other primitive/runtime boundaries remain explicit. |
| Source ownership update | `UpdaterContracts.imported_update_never_adds_claims`, `imported_update_preserves_prefix_exclusion`, `UpdaterBridge.source_apply_excludes_reset_fields` | Complete internal update AST, explicit shared comparison heap, finite control-flow invariant, arbitrary finite manager lists/versions/schedules, error and force paths. Requires exact left-subset preservation by Difference. The bridge derives the final-updater frame and composes it with Apply filtering. Native vendored/pinned cache and force controls agree. This is relative to private-phase entry after decoding/actor installation; no whole-request no-growth or universal Go primitive/progress theorem is claimed. |
| Source element algebra | `ElementAlgebra.imported_matches`, `imported_membership`, `imported_preserves_order`, `imported_preserves_uniqueness` | Complete Go ASTs of PathElementSet Difference, Union and Intersection. Shared induction covers arbitrary finite loop lengths, short-circuit bounds, safe indices, equal-key representatives and final suffixes. Membership and canonical outputs require sorted inputs and explicit strict-total-order comparator laws. The existing difference theorem statements remain specializations, including order preservation without comparator laws or a sorted right input. The actual comparator and complete recursive membership laws remain refinement obligations; the weaker recursive Difference exact-left-path-origin law is source-proved separately. |
| Source recursive Difference | `RecursiveSets.difference_never_adds_paths`, `SetContracts.imported_node_matches`, `imported_set_matches`, `imported_difference_never_adds_paths`, `SetBridge.source_updater_with_recursive_difference_does_not_grow_actor` | Both complete Go ASTs, finite mutual recursion, direct members independent from children, all comparators and Empty callbacks. Every output exact path originates in the left tree; no faithful-order premise. An explicit encoded-tree adapter discharges the updater Difference premise. Native vendored/pinned comparisons cover 4,102 tree pairs, exact numeric tokens, empty structure and depth 64. Other set membership laws, actual ownership-codec/heap representation and universal runtime refinement remain open. |
| Source reset-root filter | `SetFilterContracts.imported_flat_node_matches`, `imported_flat_set_matches`, `imported_flat_filter_matches`, `locally_ordered_filter_excludes_root`, `ElementLocalOrder.imported_local_membership` | Both complete RecursiveDifference ASTs plus actual Filter wrapper; theorem covers flat exclusion roots only. Arbitrary finite suffix iteration and unbounded descendant depth. Direct-member order is required only on the visited root-label domain; descendants may contain the known mixed-numeric collisions. Has recognition is derived separately by `FilterHasBridge`; actual comparison/root/heap representation and full Apply/API composition remain open. 518 native/pinned comparisons and an ordinary-Difference substitution mutation support the source/runtime boundary. |
| Source membership lookup | `SearchIndex.unsigned_midpoint_eq_half`, `SearchContracts.imported_search_matches`, `SearchMembership.imported_search_membership`, `HasBridge.imported_has_matches`, `imported_has_membership`, `FilterHasBridge.source_filter_excludes_root` | Complete selected Go BinarySearchFunc plus SMD Has/Less ASTs, E=T label instantiation, safe bounds and progress at every finite representable size. Exact membership requires only the active domain’s order/equality laws. Derives the root filter’s former Has-recognition premise. Actual comparator/field-name and root/heap/ownership realization plus complete request composition remain open. Native 776 lookup queries, 128 Less comparisons and 16 near-limit Go midpoint controls; final-zero-test mutation must fail both proof and native membership oracle. |
| Source atomic field extraction | `AtomicFieldContracts.imported_atomic_fields`, `imported_atomic_claims_scope`, `imported_atomic_never_adds_descendant` | Complete typed toFieldSetWalker doMap/doList ASTs at atomic entry, generic finite paths/prior claims and opaque input/coercion handles. Whole-scope insertion, no child visitor, nil errors and captured deferred cleanup. Empty root insertion is ignored; prior descendant claims are preserved, not declared impossible. Insert is a membership projection; actual insertion/coercion/allocator and full heap/runtime/schema/request realization remain open. Native 320 map/list cases and both pinned revisions agree, including null, empty, width/depth 64 and deferred Free counts. Separate map/list guard mutations fail proof and native scope-claim oracles. |
| Source atomic merge | `AtomicMergeContracts.imported_keep_right`, `imported_leaf`, `imported_atomic_merge`, `imported_atomic_right_is_whole`, `imported_atomic_absent_right_keeps_left` | Complete map/list dispatch, doLeaf and ruleKeepRHS function-literal ASTs. Arbitrary opaque operands and pure coercion-resource outcomes, finite resource lengths, old output, inLeaf state, caller locals and pending frees. Missing interface and present JSON null are distinct; right/left selection copies a whole operand. Fresh function frames derive source callees, preserve caller locals/defers and free right before left. The initializer's conversion is checked to bind the actual function type. Actual value/coercion/allocator heap realization, schema/typed-wrapper dispatch, removal/comparison and full SSA remain open. 2,352 native/current and pinned comparisons check exact JSON tokens, null/missing, old leaf, scope/input immutability and cleanup; three source mutations fail kernel and independent native oracles. |
| Source atomic removal/extraction | `AtomicRemoveContracts.imported_scalar`, `imported_atomic_container`, `imported_atomic_remove_fresh`, `imported_atomic_extract_whole` | Complete structural scalar/map/list ASTs, including non-atomic loops/callbacks; unsupported execution fails explicitly. Arbitrary opaque input and pure shape/resource/Unstructured callbacks, finite lengths and caller state. Extract copies a whole matching value; remove preserves prior output and therefore needs fresh output for absence. Mismatching shapes return unchanged; captured Free behavior and caller-frame preservation are source-derived. 1,152 native/current and pinned comparisons cover both modes, old output, typed nil/empty/wide/deep/numeric data and selection/input immutability. Three mutations fail unchanged proof and native output oracle. Wrapper initialization is source-derived separately; actual schema selection, primitives/heap, general non-atomic removal and complete Apply remain open. |
| Source removal caller | `RemoveFactoryContracts.imported_factory`, `RemoveFactoryBridge.imported_atomic_factory`, `imported_scalar_factory`, `imported_factory_ignores_reported_errors` | Complete removeItemsWithSchema AST derives struct zero initialization, exact constructor fields and wrapped output. Arbitrary normal resolver-error payloads are discarded; unsupported execution is separate. Selected source handlers discharge the fresh-output premise. Actual schema handler selection, allocator/wrapper primitives and heap remain explicit boundaries. Another 1,152 native/current and pinned caller comparisons cover valid schemas, missing named types, invalid atoms, both modes and exact immutable data. Three constructor/call/result mutations fail unchanged proof and native output oracle. |
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
The structural removal importer retains unsupported subtrees; its interpreter fails if they are reached, and the atomic certificate proves early return before them. Other import profiles still reject unsupported syntax. Source digests and generated ASTs are checked by CI; twenty-three semantic Go mutations
must fail in their expected proof modules. The six updater imports include the
complete Apply and internal ownership-update bodies. The Apply theorem covers
orchestration, including reset filter selection and installation. Its merge, extraction, reconciliation,
pruning and final updater callbacks remain explicit contracts in the original theorem.
The pruning composition realizes the four pruning callees. `UpdaterBridge` now
realizes the final update callback and derives non-growth from a source-wide
invariant plus Difference left-subset preservation. `SetSource` additionally
imports both recursive Difference methods. `SetContracts` proves their finite
recursive execution and exact path origin without any comparator-order or Empty
premise. `SetBridge` derives the updater premise for an explicit finite path
representation. Other recursive set/filter membership laws, actual ownership
codec/heap representation and complete interpreter/runtime/request refinement remain open. Successful callback values
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
IEEE conversion, actual typed/PathElement comparison, complete recursive
Set/SetNodeMap membership laws and complete composition with Apply still need
their own treatment. Has and the selected standard-library binary search have
separate source proofs under local order/equality laws and Go word bounds.
The recursive Difference non-growth law is now
source-proved without the false global comparator premises. The flat-root filter additionally uses local-domain order for direct members, leaving descendant comparisons unconstrained.
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

The same SSA-06 history now has a complete imported-Apply witness in
`GranularKeyCounterexamples`. A's exact original integer item is created from
empty; B's different neighbor is absent from the original footprint. Decoding
ownership produces a false conflict on B's new item. If B forces the request,
A's unchanged original payload remains but its old claims disappear. The
fixed integer-key/string-payload primitive instance implements real added and
modified fields plus forced transfer; it does not encode the bad outcome as
an arbitrary callback. Native current/pinned managedfields and real HTTP
independently check both effects and the successful small-key control.
The instance remains a concrete structural-schema witness, not general typed
or codec refinement. BASE-09 was split to retain its still-open atomic part
while recording the granular-progress refutation precisely.

SSA-06 also refutes BASE-02. A creates two adjacent integer-keyed items with
payloads `low` and `high`. B co-applies only the upper item with the same value.
On the wire the lower payload is still exclusively A-owned, and B owns the
exact upper selector. A retains both item keys but omits only the lower payload.
With keys `2^53` and `2^53+1`, B's decoded selector becomes the lower key and
protects the omitted payload. The full HTTP request succeeds and persists it.
`NumericOmissionCounterexamples` explicitly proves raw target existence,
exclusive raw ownership, omission without item deletion, and the false
candidate-level omission law. Its pruning witness links all four complete
imported functions under normalized primitives and every lawful map schedule;
it does not prove the general mixed-comparator or complete Apply refinement.
Native vendored/pinned execution and CRD HTTP agree on the four outcomes:
large shared keys retain the lower payload; large unshared and both small-key
controls remove it. The native/API history independently establishes reachability
of the concrete raw precondition rather than assuming arbitrary initial damage.

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
