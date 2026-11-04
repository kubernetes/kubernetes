/-
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-/
import SSA.Apply
import SSA.Properties

namespace SSA.Contracts

/-- Observation ignores the order of the internal cell and claim lists. -/
def ObservationallyEqual (left right : State) : Prop :=
  (∀ cell, cell ∈ left.cells ↔ cell ∈ right.cells) ∧
  (∀ claim, claim ∈ left.claims ↔ claim ∈ right.claims)

private theorem state_ext (left right : State)
    (cells : left.cells = right.cells) (claims : left.claims = right.claims) : left = right := by
  cases left
  cases right
  cases cells
  cases claims
  rfl

private theorem accepted_candidate (schema : Schema) (before after : State) (r : Request)
    (accepted : apply schema before r = .ok after) :
    candidate schema before r = after := by
  unfold apply at accepted
  split at accepted
  · cases accepted
  · split at accepted
    · cases accepted
    · split at accepted
      · cases accepted
      · dsimp only at accepted
        split at accepted
        · cases accepted
        · split at accepted
          · cases accepted
          · exact Except.ok.inj accepted

private theorem accepted_initial_valid (schema : Schema) (before after : State) (r : Request)
    (accepted : apply schema before r = .ok after) : valid schema before = true := by
  unfold apply at accepted
  split at accepted
  · cases accepted
  · rename_i initial
    simpa using initial

private theorem accepted_normalized (schema : Schema) (before after : State) (r : Request)
    (accepted : apply schema before r = .ok after) : normalized r = true := by
  unfold apply at accepted
  split at accepted
  · cases accepted
  · split at accepted
    · cases accepted
    · rename_i norm
      simpa using norm

private theorem desired_writes (r : Request) (norm : normalized r = true) :
    r.desired = r.writes.map Cell.footprint := by
  simp only [normalized, Bool.and_eq_true, beq_iff_eq] at norm
  exact norm.1.1

private theorem under_iff_prefixes (parent child : Path) :
    under parent child = true ↔ parent ∈ prefixes child := by
  induction child generalizing parent with
  | nil => cases parent <;> simp [under, prefixes]
  | cons head tail ih =>
      cases parent with
      | nil => simp [under, prefixes]
      | cons first rest =>
          constructor
          · intro h
            simp only [under, List.isPrefixOf_cons_cons, Bool.and_eq_true, beq_iff_eq] at h
            rcases h with ⟨heads, tails⟩
            have member := (ih rest).mp tails
            rw [heads]
            exact List.mem_cons_of_mem _ (List.mem_map.mpr ⟨rest, member, rfl⟩)
          · intro h
            simp only [prefixes, List.mem_cons, List.mem_map] at h
            rcases h with impossible | ⟨p, member, equality⟩
            · cases impossible
            · have parts := List.cons.inj equality
              simpa [under, parts.1, parts.2] using (ih p).mpr member

private theorem schema_item_path (schema : Schema) (before : State)
    (initial : valid schema before = true) (item : ItemSpec) (member : item ∈ schema) :
    isItemPath item.path = true := by
  have h := initial
  simp only [valid, shapeValid, schemaValid, Bool.and_eq_true, List.all_eq_true] at h
  exact ((h.1.1.1.1.1.2 item member).1.1)

private theorem written_outside_deletions (schema : Schema) (before : State) (r : Request)
    (initial : valid schema before = true) (norm : normalized r = true)
    (cell : Cell) (written : cell ∈ r.writes) :
    (deletedRoots schema before (merged before r) r).any
      (fun root => under root cell.path) = false := by
  apply Bool.eq_false_iff.mpr
  intro deleted
  obtain ⟨root, rootMember, below⟩ := List.any_eq_true.mp deleted
  obtain ⟨item, kept, rfl⟩ := List.mem_map.mp rootMember
  obtain ⟨inSchema, deleting⟩ := List.mem_filter.mp kept
  have keyPath := schema_item_path schema before initial item inSchema
  have prefixMember := (under_iff_prefixes item.path cell.path).mp below
  have hn := norm
  simp only [normalized, Bool.and_eq_true, List.all_eq_true] at hn
  have membership := hn.2 cell written item.path prefixMember
  have desired : r.desired.contains ⟨item.path, .item⟩ = true := by
    simpa [keyPath] using membership
  simp only [deletesItem, Bool.and_eq_true] at deleting
  have impossible := deleting.1.2
  rw [desired] at impossible
  cases impossible

/-- Every asserted value or presence marker survives a successful apply. -/
theorem accepted_preserves_intent (schema : Schema) (before after : State) (r : Request)
    (accepted : apply schema before r = .ok after)
    (cell : Cell) (written : cell ∈ r.writes) :
    cell ∈ after.cells := by
  have initial := accepted_initial_valid schema before after r accepted
  have norm := accepted_normalized schema before after r accepted
  have intent : cell.footprint ∈ r.desired := by
    rw [desired_writes r norm]
    exact List.mem_map.mpr ⟨cell, written, rfl⟩
  have roots := written_outside_deletions schema before r initial norm cell written
  have mergedMember : cell ∈ (merged before r).cells := by
    simp [merged, written]
  have remaining := (subtree_frame (merged before r)
    (deletedRoots schema before (merged before r) r) cell roots).mpr mergedMember
  rw [← accepted_candidate schema before after r accepted]
  change cell ∈ List.filter _ _
  apply List.mem_filter.mpr
  refine ⟨remaining, ?_⟩
  have asserted : r.desired.contains cell.footprint = true := by simpa using intent
  cases kind : cell.content with
  | item => rfl
  | value text =>
      simp only [Cell.footprint, kind] at asserted
      have member : ⟨cell.path, ClaimKind.value⟩ ∈ r.desired := by simpa using asserted
      simp [released, member]
  | object =>
      simp only [Cell.footprint, kind] at asserted
      have member : ⟨cell.path, ClaimKind.presence⟩ ∈ r.desired := by simpa using asserted
      simp [released, member]
  | collection =>
      simp only [Cell.footprint, kind] at asserted
      have member : ⟨cell.path, ClaimKind.presence⟩ ∈ r.desired := by simpa using asserted
      simp [released, member]

/-- A conflicting change to a foreign-owned value is rejected unless force is set. -/
theorem foreign_change_conflicts (schema : Schema) (before : State) (r : Request)
    (initial : valid schema before = true) (norm : normalized r = true)
    (notForced : r.force = false) (cell : Cell) (written : cell ∈ r.writes)
    (different : changed before cell = true) (claim : Claim)
    (owned : claim ∈ before.claims) (foreign : claim.manager ≠ r.manager)
    (target : claim.target = ⟨cell.path, .value⟩) :
    apply schema before r = .error .conflict := by
  have conflict : conflicts before r = true := by
    apply List.any_eq_true.mpr
    refine ⟨cell, written, ?_⟩
    simp only [different, Bool.true_and]
    apply List.any_eq_true.mpr
    exact ⟨claim, owned, by simp [target, foreign]⟩
  simp [apply, initial, norm, conflict, notForced]

/-- Equal-value assertions cannot cause a conflict, regardless of existing owners. -/
theorem equal_writes_do_not_conflict (before : State) (r : Request)
    (unchanged : ∀ cell ∈ r.writes, changed before cell = false) :
    conflicts before r = false := by
  apply Bool.eq_false_iff.mpr
  intro conflict
  obtain ⟨cell, written, hc⟩ := List.any_eq_true.mp conflict
  simp [unchanged cell written] at hc

/-- Force removes each foreign value claim whose asserted value actually changes. -/
theorem force_transfers_changed_claim (before : State) (r : Request)
    (forced : r.force = true) (claim : Claim) (foreign : claim.manager ≠ r.manager)
    (cell : Cell) (written : cell ∈ r.writes) (different : changed before cell = true)
    (target : claim.target = ⟨cell.path, .value⟩) :
    claim ∉ updatedClaims before r := by
  have changes : r.writes.any
      (fun c => c.path == claim.target.path && changed before c) = true :=
    List.any_eq_true.mpr ⟨cell, written, by simp [target, different]⟩
  have kind : (claim.target.kind == .value) = true := by simp [target]
  have old : claim ∉ before.claims.filter (fun c => c.manager != r.manager &&
      !(r.force && c.target.kind == .value &&
        r.writes.any (fun cell => cell.path == c.target.path && changed before cell))) := by
    simp [forced, kind, changes]
  have new : claim ∉ r.desired.map (fun target => Claim.mk r.manager target) := by
    intro member
    obtain ⟨target', _, equality⟩ := List.mem_map.mp member
    have managers := congrArg Claim.manager equality
    exact foreign managers.symm
  exact List.not_mem_append old new

/-- Equal-value assertions retain foreign co-ownership, including under force. -/
theorem unchanged_foreign_claim_preserved (before : State) (r : Request) (claim : Claim)
    (owned : claim ∈ before.claims) (foreign : claim.manager ≠ r.manager)
    (unchanged : ∀ cell ∈ r.writes,
      cell.path = claim.target.path → changed before cell = false) :
    claim ∈ updatedClaims before r := by
  have noChange : r.writes.any
      (fun c => c.path == claim.target.path && changed before c) = false := by
    apply Bool.eq_false_iff.mpr
    intro h
    obtain ⟨cell, mem, hp⟩ := List.any_eq_true.mp h
    simp only [Bool.and_eq_true, beq_iff_eq] at hp
    simp [unchanged cell mem hp.1] at hp
  simp [updatedClaims, List.mem_filter, owned, foreign, noChange]

/-- A foreign owner is retained when this request does not overwrite its field. -/
theorem unwritten_foreign_claim_preserved (before : State) (r : Request) (claim : Claim)
    (owned : claim ∈ before.claims) (foreign : claim.manager ≠ r.manager)
    (unwritten : r.writes.any (fun c => c.path == claim.target.path) = false) :
    claim ∈ updatedClaims before r := by
  apply unchanged_foreign_claim_preserved before r claim owned foreign
  intro cell member path
  have found : r.writes.any (fun c => c.path == claim.target.path) = true :=
    List.any_eq_true.mpr ⟨cell, member, by simp [path]⟩
  simp [unwritten] at found

/-- Shared omission preserves the value, provided no lifetime deletion removes its subtree. -/
theorem shared_omission_preserves_value (schema : Schema) (before : State) (r : Request)
    (cell : Cell) (text : String) (present : cell ∈ before.cells)
    (kind : cell.content = .value text) (claim : Claim) (owned : claim ∈ before.claims)
    (foreign : claim.manager ≠ r.manager) (target : claim.target = ⟨cell.path, .value⟩)
    (unwritten : r.writes.any (fun c => c.path == cell.path) = false)
    (unrelated : (deletedRoots schema before (merged before r) r).any
      (fun root => under root cell.path) = false) :
    cell ∈ (candidate schema before r).cells := by
  have kept := unwritten_foreign_claim_preserved before r claim owned foreign
    (by simpa [target] using unwritten)
  have remaining := (claims_frame (merged before r)
    (deletedRoots schema before (merged before r) r) claim
      (by simpa [target] using unrelated)).mpr kept
  have isClaimed : claimed
      (dropSubtrees (merged before r) (deletedRoots schema before (merged before r) r))
      cell.path .value = true :=
    List.any_eq_true.mpr ⟨claim, remaining, by simp [target]⟩
  have mergedCell : cell ∈ (merged before r).cells := by
    simp [merged, List.mem_filter, present, unwritten]
  have keptCell := (subtree_frame (merged before r)
    (deletedRoots schema before (merged before r) r) cell unrelated).mpr mergedCell
  change cell ∈ List.filter _ _
  exact List.mem_filter.mpr ⟨keptCell, by simp [kind, isClaimed]⟩

/-- Exclusive omission removes a non-defaulted scalar; no other claim can rescue it. -/
theorem exclusive_omission_removes_value (schema : Schema) (before : State) (r : Request)
    (cell : Cell) (text : String) (kind : cell.content = .value text)
    (owned : before.owns r.manager cell.path .value = true)
    (exclusive : ∀ claim ∈ before.claims,
      claim.target = ⟨cell.path, .value⟩ → claim.manager = r.manager)
    (omitted : ⟨cell.path, ClaimKind.value⟩ ∉ r.desired) :
    cell ∉ (candidate schema before r).cells := by
  let remaining := dropSubtrees (merged before r) (deletedRoots schema before (merged before r) r)
  have noClaim : claimed remaining cell.path .value = false := by
    apply Bool.eq_false_iff.mpr
    intro hc
    obtain ⟨claim, member, target⟩ := List.any_eq_true.mp hc
    have targetEq : claim.target = ⟨cell.path, .value⟩ := by simpa using target
    have updated : claim ∈ updatedClaims before r := (List.mem_filter.mp member).1
    simp only [updatedClaims, List.mem_append, List.mem_filter, List.mem_map] at updated
    rcases updated with ⟨oldMember, kept⟩ | ⟨fp, desired, equality⟩
    · have managerEq := exclusive claim oldMember targetEq
      simp [managerEq] at kept
    · have fpEq : fp = ⟨cell.path, ClaimKind.value⟩ := by
        simpa [← equality] using targetEq
      exact omitted (fpEq ▸ desired)
  have dropping : released before r cell.path .value = true := by
    simp [released, owned, omitted]
  change cell ∉ remaining.cells.filter _
  intro member
  have keep := (List.mem_filter.mp member).2
  simp only [kind] at keep
  change (!(released before r cell.path .value && !claimed remaining cell.path .value)) = true at keep
  simp [dropping, noClaim] at keep

/-- The resulting actor footprint is exactly the normalized intent, not its old footprint. -/
theorem accepted_actor_claims_exact (schema : Schema) (before after : State) (r : Request)
    (accepted : apply schema before r = .ok after) (fp : Footprint) :
    after.owns r.manager fp.path fp.kind = true ↔ fp ∈ r.desired := by
  have norm := accepted_normalized schema before after r accepted
  have initial := accepted_initial_valid schema before after r accepted
  rw [← accepted_candidate schema before after r accepted]
  constructor
  · intro owned
    obtain ⟨claim, member, ownedTarget⟩ := List.any_eq_true.mp owned
    simp only [Bool.and_eq_true, beq_iff_eq] at ownedTarget
    have updated : claim ∈ updatedClaims before r := (List.mem_filter.mp member).1
    simp only [updatedClaims, List.mem_append, List.mem_filter, List.mem_map] at updated
    rcases updated with ⟨_, kept⟩ | ⟨target, desired, equality⟩
    · simp [ownedTarget.1] at kept
    · have targetEq : target = fp := by
        have h := ownedTarget.2
        simpa [← equality] using h
      exact targetEq ▸ desired
  · intro desired
    have mapped : fp ∈ r.writes.map Cell.footprint := by
      simpa only [desired_writes r norm] using desired
    obtain ⟨cell, written, targetEq⟩ := List.mem_map.mp mapped
    have roots := written_outside_deletions schema before r initial norm cell written
    have pathEq : cell.path = fp.path := congrArg Footprint.path targetEq
    have claimMember : Claim.mk r.manager fp ∈ updatedClaims before r :=
      List.mem_append_right _ (List.mem_map.mpr ⟨fp, desired, rfl⟩)
    have kept := (claims_frame (merged before r)
      (deletedRoots schema before (merged before r) r) ⟨r.manager, fp⟩
      (by simpa [pathEq] using roots)).mpr claimMember
    apply List.any_eq_true.mpr
    exact ⟨⟨r.manager, fp⟩, kept, by simp⟩

/-- Successful non-forced requests cannot change an asserted foreign-owned scalar. -/
theorem accepted_nonforce_preserves_foreign_assertion
    (schema : Schema) (before after : State) (r : Request)
    (accepted : apply schema before r = .ok after) (notForced : r.force = false)
    (cell : Cell) (written : cell ∈ r.writes) (claim : Claim)
    (owned : claim ∈ before.claims) (foreign : claim.manager ≠ r.manager)
    (target : claim.target = ⟨cell.path, .value⟩) :
    changed before cell = false := by
  cases different : changed before cell with
  | false => rfl
  | true =>
      have rejection := foreign_change_conflicts schema before r
        (accepted_initial_valid schema before after r accepted)
        (accepted_normalized schema before after r accepted)
        notForced cell written different claim owned foreign target
      rw [rejection] at accepted
      cases accepted

private theorem full_of_retained_claims (before after : State) (manager : String)
    (retained : ∀ claim ∈ before.claims, claim.manager = manager → claim ∈ after.claims)
    (item : ItemSpec) (wasFull : full before manager item = true) :
    full after manager item = true := by
  have valueClaims : ∀ path kind, before.owns manager path kind = true →
      after.owns manager path kind = true := by
    intro path kind owns
    obtain ⟨claim, member, asserted⟩ := List.any_eq_true.mp owns
    have owner : claim.manager = manager := by
      have h := asserted
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      exact h.1
    exact List.any_eq_true.mpr ⟨claim, retained claim member owner, asserted⟩
  have presenceClaims : ∀ path, before.ownsMapPresence manager path = true →
      after.ownsMapPresence manager path = true := by
    intro path owns
    obtain ⟨claim, member, asserted⟩ := List.any_eq_true.mp owns
    have owner : claim.manager = manager := by
      have h := asserted
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      exact h.1
    exact List.any_eq_true.mpr ⟨claim, retained claim member owner, asserted⟩
  have coreClaims : ∀ core, covers before manager core = true →
      covers after manager core = true := by
    intro core
    induction core with
    | none => intro _; rfl
    | value path => exact valueClaims path .value
    | collection path => exact valueClaims path .presence
    | object path children ih =>
        simp only [covers, Bool.and_eq_true]
        exact fun h => ⟨presenceClaims path h.1, ih h.2⟩
    | both left right ihLeft ihRight =>
        simp only [covers, Bool.and_eq_true]
        exact fun h => ⟨ihLeft h.1, ihRight h.2⟩
  simp only [full, Bool.and_eq_true] at wasFull ⊢
  exact ⟨valueClaims item.path .item wasFull.1, coreClaims item.required wasFull.2⟩

/-- A full successor survives another actor's release when no ancestor removes its claims. -/
theorem handoff_preserves_full_successor (schema : Schema) (before after : State) (r : Request)
    (accepted : apply schema before r = .ok after) (notForced : r.force = false)
    (successor : String) (other : successor ≠ r.manager)
    (item : ItemSpec) (wasFull : full before successor item = true)
    (outside : ∀ claim ∈ before.claims, claim.manager = successor →
      (deletedRoots schema before (merged before r) r).any
        (fun root => under root claim.target.path) = false) :
    full after successor item = true := by
  apply full_of_retained_claims before after successor ?_ item wasFull
  intro claim member owner
  have updated : claim ∈ updatedClaims before r := by
    simp [updatedClaims, List.mem_filter, member, notForced, owner, other]
  have kept := (claims_frame (merged before r)
    (deletedRoots schema before (merged before r) r) claim (outside claim member owner)).mpr updated
  rw [← accepted_candidate schema before after r accepted]
  exact kept

/-- Rejected requests are included: every committed prefix of a finite history stays valid. -/
def run (schema : Schema) : State → List Request → State
  | before, [] => before
  | before, request :: rest => run schema (commit before (apply schema before request)) rest

theorem finite_history_preserves_validity (schema : Schema) (before : State)
    (initial : valid schema before = true) (requests : List Request) :
    valid schema (run schema before requests) = true := by
  induction requests generalizing before with
  | nil => exact initial
  | cons request rest ih =>
      exact ih _ (committed_valid schema before request initial)

private theorem eraseDups_length_le {α : Type} [BEq α] (xs : List α) :
    xs.eraseDups.length ≤ xs.length := by
  match xs with
  | [] => simp
  | x :: xs =>
      rw [List.eraseDups_cons]
      simp only [List.length_cons]
      exact Nat.succ_le_succ (Nat.le_trans
        (eraseDups_length_le (xs.filter (fun y => !y == x))) (List.length_filter_le _ _))
termination_by xs.length
decreasing_by
  exact Nat.lt_succ_of_le (List.length_filter_le _ _)

private theorem nodup_of_eraseDups_length {α : Type} [BEq α] [LawfulBEq α]
    (xs : List α) (equalLength : xs.eraseDups.length = xs.length) : xs.Nodup := by
  induction xs with
  | nil => exact List.nodup_nil
  | cons x xs ih =>
      rw [List.eraseDups_cons] at equalLength
      have tailLength := Nat.succ.inj equalLength
      have dedupLe := eraseDups_length_le (xs.filter (fun y => !y == x))
      have filterLe := List.length_filter_le (fun y => !y == x) xs
      have filterLength : (xs.filter (fun y => !y == x)).length = xs.length := by omega
      have allKept := List.length_filter_eq_length_iff.mp filterLength
      have filterSelf := List.filter_eq_self.mpr allKept
      have absent : x ∉ xs := by
        intro member
        have impossible := allKept x member
        simp at impossible
      rw [filterSelf] at tailLength
      exact List.nodup_cons.mpr ⟨absent, ih tailLength⟩

private theorem cells_unique_by_path (cells : List Cell) (unique : (cells.map (·.path)).Nodup)
    (left right : Cell) (hl : left ∈ cells) (hr : right ∈ cells)
    (same : left.path = right.path) : left = right := by
  induction cells with
  | nil => cases hl
  | cons head tail ih =>
      simp only [List.map_cons, List.nodup_cons] at unique
      simp only [List.mem_cons] at hl hr
      rcases hl with rfl | hl <;> rcases hr with rfl | hr
      · rfl
      · exact False.elim (unique.1 (List.mem_map.mpr ⟨right, hr, same.symm⟩))
      · exact False.elim (unique.1 (List.mem_map.mpr ⟨left, hl, same⟩))
      · exact ih unique.2 hl hr

private theorem valid_unique_paths (schema : Schema) (s : State)
    (validState : valid schema s = true) : (s.cells.map (·.path)).Nodup := by
  simp only [valid, shapeValid, Bool.and_eq_true] at validState
  have equalLength := validState.1.1.1.1.2
  apply nodup_of_eraseDups_length
  simpa using equalLength

/-- An accepted result already contains every written cell with its asserted value. -/
theorem accepted_reapply_writes_unchanged (schema : Schema) (before once : State) (r : Request)
    (accepted : apply schema before r = .ok once) :
    ∀ cell ∈ r.writes, changed once cell = false := by
  intro cell written
  apply Bool.eq_false_iff.mpr
  intro changedValue
  obtain ⟨old, present, different⟩ := List.any_eq_true.mp changedValue
  simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne] at different
  have asserted := accepted_preserves_intent schema before once r accepted cell written
  have same := cells_unique_by_path once.cells
    (valid_unique_paths schema once (accepted_valid schema before once r accepted))
    old cell present asserted different.1
  exact different.2 (congrArg Cell.content same)

/-- Once accepted, the request can be repeated without producing an ownership conflict. -/
theorem accepted_reapply_has_no_conflicts (schema : Schema) (before once : State) (r : Request)
    (accepted : apply schema before r = .ok once) : conflicts once r = false :=
  equal_writes_do_not_conflict once r
    (accepted_reapply_writes_unchanged schema before once r accepted)

private theorem accepted_reapply_releases_nothing
    (schema : Schema) (before once : State) (r : Request)
    (accepted : apply schema before r = .ok once) (path : Path) (kind : ClaimKind) :
    released once r path kind = false := by
  unfold released
  cases owned : once.owns r.manager path kind with
  | false => simp
  | true =>
      have desired := (accepted_actor_claims_exact schema before once r accepted ⟨path, kind⟩).mp owned
      simp [desired]

private theorem accepted_reapply_deletes_nothing
    (schema : Schema) (before once : State) (r : Request)
    (accepted : apply schema before r = .ok once) :
    deletedRoots schema once (merged once r) r = [] := by
  unfold deletedRoots
  have noItems : schema.filter (deletesItem once (merged once r) r) = [] := by
    apply List.filter_eq_nil_iff.mpr
    intro item _ deleting
    simp only [deletesItem, full, Bool.and_eq_true] at deleting
    have desired := (accepted_actor_claims_exact schema before once r accepted
      ⟨item.path, .item⟩).mp deleting.1.1.2.1
    simp [desired] at deleting
  rw [noItems]
  rfl

private theorem accepted_reapply_candidate_is_merge
    (schema : Schema) (before once : State) (r : Request)
    (accepted : apply schema before r = .ok once) :
    candidate schema once r = merged once r := by
  have noRoots := accepted_reapply_deletes_nothing schema before once r accepted
  change pruneReleased once r
    (dropSubtrees (merged once r) (deletedRoots schema once (merged once r) r)) = _
  rw [noRoots]
  have drop : dropSubtrees (merged once r) [] = merged once r := by
    apply state_ext <;> exact List.filter_eq_self.mpr (by intro _ _; rfl)
  rw [drop]
  apply state_ext
  · apply List.filter_eq_self.mpr
    intro cell _
    cases cell.content <;>
      simp [accepted_reapply_releases_nothing schema before once r accepted]
  · rfl

private theorem filter_prefix {α : Type} (front tail : List α) (predicate : α → Bool)
    (kept : ∀ x ∈ front, predicate x = true) :
    (front ++ tail).filter predicate = front ++ tail.filter predicate := by
  rw [List.filter_append, List.filter_eq_self.mpr kept]

private theorem accepted_cells_normal_form
    (schema : Schema) (before once : State) (r : Request)
    (accepted : apply schema before r = .ok once) :
    once.cells = r.writes ++ once.cells.filter
      (fun old => !r.writes.any (fun c => c.path == old.path)) := by
  let roots := deletedRoots schema before (merged before r) r
  let remaining := dropSubtrees (merged before r) roots
  let noWrite : Cell → Bool := fun old => !r.writes.any (fun c => c.path == old.path)
  let noRoot : Cell → Bool := fun c => !roots.any (fun root => under root c.path)
  let keep : Cell → Bool := fun c => match c.content with
    | .value _ => !(released before r c.path .value && !claimed remaining c.path .value)
    | .object | .collection => !(released before r c.path .presence &&
        !remaining.claims.any (fun claim => under c.path claim.target.path))
    | .item => true
  have norm := accepted_normalized schema before once r accepted
  have writesNoRoot : ∀ cell ∈ r.writes, noRoot cell = true := by
    intro cell member
    simp only [noRoot, roots, written_outside_deletions schema before r
      (accepted_initial_valid schema before once r accepted) norm cell member, Bool.not_false]
  have writesKept : ∀ cell ∈ r.writes, keep cell = true := by
    intro cell member
    have asserted : cell.footprint ∈ r.desired := by
      rw [desired_writes r norm]
      exact List.mem_map.mpr ⟨cell, member, rfl⟩
    cases kind : cell.content <;> simp_all [keep, released, Cell.footprint]
  let tail := ((before.cells.filter noWrite).filter noRoot).filter keep
  have form : once.cells = r.writes ++ tail := by
    rw [← accepted_candidate schema before once r accepted]
    change ((r.writes ++ before.cells.filter noWrite).filter noRoot).filter keep = _
    rw [filter_prefix _ _ _ writesNoRoot, filter_prefix _ _ _ writesKept]
  have writesExcluded : r.writes.filter noWrite = [] := by
    apply List.filter_eq_nil_iff.mpr
    intro cell member
    have found : r.writes.any (fun c => c.path == cell.path) = true :=
      List.any_eq_true.mpr ⟨cell, member, by simp⟩
    simp [noWrite, found]
  have tailKept : tail.filter noWrite = tail := by
    apply List.filter_eq_self.mpr
    intro cell member
    exact (List.mem_filter.mp (List.mem_filter.mp (List.mem_filter.mp member).1).1).2
  rw [form, List.filter_append, writesExcluded, List.nil_append, tailKept]

private theorem accepted_claims_normal_form
    (schema : Schema) (before once : State) (r : Request)
    (accepted : apply schema before r = .ok once) :
    once.claims = once.claims.filter (fun c => c.manager != r.manager) ++
      r.desired.map (fun target => Claim.mk r.manager target) := by
  let roots := deletedRoots schema before (merged before r) r
  let noRoot : Claim → Bool := fun c => !roots.any (fun root => under root c.target.path)
  let old := before.claims.filter (fun c => c.manager != r.manager &&
    !(r.force && c.target.kind == .value &&
      r.writes.any (fun cell => cell.path == c.target.path && changed before cell)))
  let fresh := r.desired.map (fun target => Claim.mk r.manager target)
  have freshKept : fresh.filter noRoot = fresh := by
    apply List.filter_eq_self.mpr
    intro claim member
    obtain ⟨target, desired, rfl⟩ := List.mem_map.mp member
    rw [desired_writes r (accepted_normalized schema before once r accepted)] at desired
    obtain ⟨cell, written, rfl⟩ := List.mem_map.mp desired
    simp only [noRoot, roots, Cell.footprint]
    cases cell.content <;>
      simp [written_outside_deletions schema before r
        (accepted_initial_valid schema before once r accepted)
        (accepted_normalized schema before once r accepted) cell written]
  have form : once.claims = old.filter noRoot ++ fresh := by
    rw [← accepted_candidate schema before once r accepted]
    change (old ++ fresh).filter noRoot = _
    rw [List.filter_append, freshKept]
  have oldKept : (old.filter noRoot).filter (fun c => c.manager != r.manager) =
      old.filter noRoot := by
    apply List.filter_eq_self.mpr
    intro claim member
    have oldMember := (List.mem_filter.mp member).1
    have kept := (List.mem_filter.mp oldMember).2
    simp only [Bool.and_eq_true] at kept
    exact kept.1
  have freshExcluded : fresh.filter (fun c => c.manager != r.manager) = [] := by
    simp [fresh, List.filter_map]
  rw [form, List.filter_append, oldKept, freshExcluded, List.append_nil]

/-- Every accepted result is an exact fixed point, including the model's internal list order. -/
theorem accepted_reapply_candidate_fixed (schema : Schema) (before once : State) (r : Request)
    (accepted : apply schema before r = .ok once) : candidate schema once r = once := by
  rw [accepted_reapply_candidate_is_merge schema before once r accepted]
  apply state_ext
  · exact (accepted_cells_normal_form schema before once r accepted).symm
  · change updatedClaims once r = once.claims
    have noChange : ∀ path, r.writes.any
        (fun cell => cell.path == path && changed once cell) = false := by
      intro path
      apply Bool.eq_false_iff.mpr
      intro found
      obtain ⟨cell, written, changed⟩ := List.any_eq_true.mp found
      simp [accepted_reapply_writes_unchanged schema before once r accepted cell written] at changed
    have sameFilter : once.claims.filter (fun c => c.manager != r.manager &&
        !(r.force && c.target.kind == .value &&
          r.writes.any (fun cell => cell.path == c.target.path && changed once cell))) =
        once.claims.filter (fun c => c.manager != r.manager) := by
      apply List.filter_congr
      intro claim _
      simp only [noChange, Bool.and_false, Bool.not_false, Bool.and_true]
    unfold updatedClaims
    rw [sameFilter]
    exact (accepted_claims_normal_form schema before once r accepted).symm

/-- Immediate repetition succeeds and returns exactly the same model state, even with force. -/
theorem accepted_apply_idempotent (schema : Schema) (before once : State) (r : Request)
    (accepted : apply schema before r = .ok once) : apply schema once r = .ok once := by
  have good := accepted_valid schema before once r accepted
  have norm := accepted_normalized schema before once r accepted
  have noConflict := accepted_reapply_has_no_conflicts schema before once r accepted
  have fixed := accepted_reapply_candidate_fixed schema before once r accepted
  have parts := good
  simp only [valid, Bool.and_eq_true] at parts
  simp [apply, good, norm, noConflict, fixed, parts.1.1, parts.1.2, parts.2]

/-- The API-level observation follows from the stronger exact model fixed point. -/
theorem repeated_apply_observationally_equal
    (schema : Schema) (before once twice : State) (r : Request)
    (first : apply schema before r = .ok once) (second : apply schema once r = .ok twice) :
    ObservationallyEqual once twice := by
  have same : once = twice := Except.ok.inj ((accepted_apply_idempotent schema before once r first).symm.trans second)
  subst twice
  exact ⟨fun _ => Iff.rfl, fun _ => Iff.rfl⟩

end SSA.Contracts
