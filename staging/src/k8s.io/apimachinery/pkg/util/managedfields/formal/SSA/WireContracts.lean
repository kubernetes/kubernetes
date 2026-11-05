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
import SSA.Semantics
import SSA.Contracts

namespace SSA.WireContracts

/-- These contracts concern the current/PR pruning transition, not the proposed item core. -/
def raw (mode : Semantics) : Prop := mode = .current ∨ mode = .predrag

/-- Pruning sees all foreign claims, even on a forced request. Go transfers them afterwards. -/
theorem merged_foreign_claim (before : State) (r : Request) (claim : Claim)
    (foreign : claim.manager ≠ r.manager) :
    claim ∈ (wireMerged before r).claims ↔ claim ∈ before.claims := by
  simp only [wireMerged, List.mem_append, List.mem_filter, List.mem_map]
  constructor
  · intro member
    rcases member with member | ⟨target, _, equal⟩
    · exact member.1
    · have same := congrArg Claim.manager equal
      exact (foreign same.symm).elim
  · intro member
    exact Or.inl ⟨member, by simp [foreign]⟩

/-- Force changes acceptance, not the object or claims offered to pruning. -/
theorem merged_force_independent (before : State) (r : Request) (force : Bool) :
    wireMerged before { r with force } = wireMerged before r := by
  rfl

theorem pruned_force_independent (mode : Semantics) (schema : Schema)
    (before : State) (r : Request) (force : Bool) :
    wirePruned mode schema before { r with force } = wirePruned mode schema before r := by
  simp only [wirePruned, merged_force_independent, pruneFieldSets]

/-- A current/PR candidate is independent of force; only the conflict gate consults it. -/
theorem candidate_force_independent (mode : Semantics) (schema : Schema)
    (before : State) (r : Request) (force : Bool) (modeRaw : raw mode) :
    candidateWith mode schema before { r with force } = candidateWith mode schema before r := by
  rcases modeRaw with rfl | rfl <;>
    simp only [candidateWith, pruned_force_independent]

/-- Final ownership drops changed foreign targets only after the final object is known. -/
theorem candidate_foreign_claim_exact (mode : Semantics) (schema : Schema)
    (before : State) (r : Request) (claim : Claim) (modeRaw : raw mode)
    (foreign : claim.manager ≠ r.manager) :
    claim ∈ (candidateWith mode schema before r).claims ↔
      claim ∈ before.claims ∧
      (wirePruned mode schema before r).cells.any (fun cell => cell.path == claim.target.path) = true ∧
      (wirePruned mode schema before r).cells.any
        (fun cell => cell.path == claim.target.path &&
          modifiedCell before (wirePruned mode schema before r) cell) = false := by
  have pruningClaims : (wirePruned mode schema before r).claims = (wireMerged before r).claims := by
    simp only [wirePruned, pruneFieldSets, pruneWithFields, addBackOwnedFields, addBackDanglingFields, removeItems]
    split <;> rfl
  rcases modeRaw with rfl | rfl <;>
    simp only [candidateWith, List.mem_filter, pruningClaims,
      merged_foreign_claim before r claim foreign] <;>
    simp [foreign]

/-- An accepted request without force cannot modify a foreign-owned final target. -/
theorem accepted_nonforce_no_conflicts (mode : Semantics) (schema : Schema)
    (before after : State) (r : Request) (modeRaw : raw mode) (nonforce : r.force = false)
    (accepted : applyWith mode schema before r = .ok after) :
    wireConflicts before before (wirePruned mode schema before r) r.manager = false := by
  rcases modeRaw with rfl | rfl <;>
    simp only [applyWith, nonforce, Bool.not_false, Bool.and_true] at accepted <;>
    split at accepted
  all_goals try { cases accepted }
  all_goals split at accepted
  all_goals try { cases accepted }
  all_goals split at accepted
  all_goals first | { cases accepted } | { simpa only [Bool.eq_false_iff] using ‹¬_ = true› }

/-- Conflict detection quantifies over original foreign owners and the final pruned data. -/
theorem accepted_nonforce_foreign_unchanged (mode : Semantics) (schema : Schema)
    (before after : State) (r : Request) (claim : Claim) (cell : Cell)
    (modeRaw : raw mode) (nonforce : r.force = false)
    (accepted : applyWith mode schema before r = .ok after)
    (owned : claim ∈ before.claims) (foreign : claim.manager ≠ r.manager)
    (present : cell ∈ (wirePruned mode schema before r).cells)
    (samePath : cell.path = claim.target.path) :
    modifiedCell before (wirePruned mode schema before r) cell = false := by
  have noConflicts := accepted_nonforce_no_conflicts mode schema before after r modeRaw nonforce accepted
  apply Bool.eq_false_iff.mpr
  intro changedTrue
  have conflict : wireConflicts before before (wirePruned mode schema before r) r.manager = true := by
    unfold wireConflicts
    apply List.any_eq_true.mpr
    refine ⟨claim, owned, Bool.and_eq_true_iff.mpr ⟨bne_iff_ne.mpr foreign, ?_⟩⟩
    exact List.any_eq_true.mpr ⟨cell, present, by simp [samePath, changedTrue]⟩
  simp [noConflicts] at conflict

/-- Forcing an already successful raw Apply changes neither its success nor its result. -/
theorem accepted_also_succeeds_forced (mode : Semantics) (schema : Schema)
    (before after : State) (r : Request) (modeRaw : raw mode)
    (accepted : applyWith mode schema before r = .ok after) :
    applyWith mode schema before { r with force := true } = .ok after := by
  have same := candidate_force_independent mode schema before r true modeRaw
  have normalizedForce : normalized { r with force := true } = normalized r := rfl
  rcases modeRaw with rfl | rfl <;>
    simp only [applyWith] at accepted ⊢
  all_goals split at accepted
  all_goals try { cases accepted }
  all_goals split at accepted
  all_goals try { cases accepted }
  all_goals split at accepted
  all_goals try { cases accepted }
  all_goals simp_all only [Bool.not_true, Bool.and_false, Bool.false_eq_true, ↓reduceIte]

/-- Comparing an existing cell with the same uniquely keyed object reports no modification. -/
theorem self_comparison_unchanged (before : State) (cell : Cell)
    (unique : (before.cells.map (·.path)).Nodup) (present : cell ∈ before.cells) :
    modifiedCell before before cell = false := by
  unfold modifiedCell
  split
  · rfl
  · rename_i old found
    have samePath : old.path = cell.path := eq_of_beq
      (List.find?_some (p := fun old : Cell => old.path == cell.path) found)
    have same := Contracts.cells_unique_by_path before.cells unique old cell
      (List.mem_of_find?_eq_some found) present samePath
    simp [same]

/-- Raw SSA also admits fresh co-ownership; success does not rely on the stronger full guard. -/
theorem unchanged_snapshot_acquisition (mode : Semantics) (schema : Schema)
    (before : State) (r : Request) (modeRaw : raw mode)
    (initial : shapeValid schema before = true) (norm : normalized r = true)
    (fresh : ∀ claim ∈ before.claims, claim.manager ≠ r.manager)
    (snapshot : r.cells = before.cells) :
    ∃ after, applyWith mode schema before r = .ok after ∧ after.cells = before.cells ∧
      after.claims = before.claims ++ r.desired.map (fun target => ⟨r.manager, target, true⟩) := by
  let after : State := {
    cells := before.cells
    claims := before.claims ++ r.desired.map (fun target => ⟨r.manager, target, true⟩) }
  have shape := initial
  simp only [shapeValid, Bool.and_eq_true] at shape
  have unique := Contracts.nodup_of_eraseDups_length (before.cells.map (·.path))
    (by simpa using shape.1.1.2)
  have oldClaims : before.claims.filter (fun claim => claim.manager != r.manager) = before.claims := by
    apply List.filter_eq_self.mpr
    intro claim member
    simpa using fresh claim member
  have noPrevious : before.claims.filter (fun claim => claim.manager == r.manager) = [] := by
    apply List.filter_eq_nil_iff.mpr
    intro claim member
    simp [fresh claim member]
  have oldEmpty : before.cells.filter (fun old => !r.cells.any (fun c => c.path == old.path)) = [] := by
    apply List.filter_eq_nil_iff.mpr
    intro cell member
    have found : r.cells.any (fun c => c.path == cell.path) = true := by
      rw [snapshot]
      exact List.any_eq_true.mpr ⟨cell, member, by simp⟩
    simp [found]
  have sameMerge : wireMerged before r = after := by
    simp only [wireMerged, merged, oldEmpty, List.append_nil, oldClaims]
    rw [snapshot]
  have noLast : ensureNamed r.named (claimFields
      (before.claims.filter (fun claim => claim.manager == r.manager))) = [] := by
    simp only [noPrevious, claimFields, List.flatMap_nil, ensureNamed, List.any_nil, List.nil_append]
    exact List.filter_eq_nil_iff.mpr (by intros; simp)
  have samePruned : wirePruned mode schema before r = after := by
    simp only [wirePruned, pruneFieldSets, pruneWithFields, noLast, List.isEmpty_nil, ↓reduceIte]
    exact sameMerge
  have newTargets : ∀ fp ∈ r.desired, targetExists before fp = true := by
    have normalizedIntent := norm
    simp only [normalized, Bool.and_eq_true, beq_iff_eq] at normalizedIntent
    intro fp desired
    rw [normalizedIntent.1.1.1] at desired
    obtain ⟨cell, written, rfl⟩ := List.mem_map.mp desired
    have stored : cell ∈ before.cells := by
      rw [← snapshot]
      exact List.mem_append_left r.context written
    cases kind : cell.content <;>
      simp only [Cell.footprint, kind, targetExists, State.hasValue, State.hasContainer,
        State.hasItem, Bool.or_eq_true] <;>
      first | exact List.any_eq_true.mpr ⟨cell, stored, by simp [kind]⟩
            | exact Or.inl (List.any_eq_true.mpr ⟨cell, stored, by simp [kind]⟩)
  have shapeAfter : shapeValid schema after = true := by
    simp only [shapeValid, Bool.and_eq_true, List.all_eq_true] at initial ⊢
    refine ⟨⟨⟨initial.1.1.1, initial.1.1.2⟩, initial.1.2⟩, ?_⟩
    intro claim member
    rcases List.mem_append.mp member with old | new
    · exact initial.2 claim old
    · obtain ⟨fp, desired, rfl⟩ := List.mem_map.mp new
      exact newTargets fp desired
  have unchanged : ∀ cell ∈ after.cells, modifiedCell before after cell = false := by
    exact fun cell member => self_comparison_unchanged before cell unique member
  have claimsHavePaths : ∀ claim ∈ after.claims,
      after.cells.any (fun cell => cell.path == claim.target.path) = true := by
    intro claim member
    have existsTarget : targetExists after claim.target = true := by
      have validShape := shapeAfter
      simp only [shapeValid, Bool.and_eq_true, List.all_eq_true] at validShape
      exact validShape.2 claim member
    cases kind : claim.target.kind <;>
      simp only [targetExists, kind, State.hasValue, State.hasItem, State.hasContainer,
        Bool.or_eq_true] at existsTarget
    all_goals first
      | { obtain ⟨cell, present, property⟩ := List.any_eq_true.mp existsTarget
          exact List.any_eq_true.mpr ⟨cell, present, (Bool.and_eq_true_iff.mp property).1⟩ }
      | { rcases existsTarget with existsTarget | existsTarget <;>
            obtain ⟨cell, present, property⟩ := List.any_eq_true.mp existsTarget <;>
            exact List.any_eq_true.mpr ⟨cell, present, (Bool.and_eq_true_iff.mp property).1⟩ }
  have noChange (claim : Claim) : after.cells.any
      (fun cell => cell.path == claim.target.path && modifiedCell before after cell) = false := by
    apply Bool.eq_false_iff.mpr
    intro found
    obtain ⟨cell, present, property⟩ := List.any_eq_true.mp found
    simp [unchanged cell present] at property
  have sameCandidate : candidateWith mode schema before r = after := by
    rcases modeRaw with rfl | rfl <;>
      simp only [candidateWith, samePruned]
    all_goals congr 1
    all_goals apply List.filter_eq_self.mpr
    all_goals intro claim member
    all_goals simp [claimsHavePaths claim member, noChange]
  have noConflict : wireConflicts before before (wirePruned mode schema before r) r.manager = false := by
    simp only [wireConflicts, samePruned]
    apply Bool.eq_false_iff.mpr
    intro conflict
    obtain ⟨claim, _, property⟩ := List.any_eq_true.mp conflict
    simp [noChange] at property
  refine ⟨after, ?_, rfl, rfl⟩
  rcases modeRaw with rfl | rfl <;>
    simp [applyWith, initial, norm, noConflict, sameCandidate, shapeAfter]

end SSA.WireContracts
