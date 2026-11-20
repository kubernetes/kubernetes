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
import SSA.Contracts

namespace SSA.Progress

/-- A fresh manager can acquire a normalized snapshot of existing data without a force write. -/
theorem unchanged_snapshot_acquisition (schema : Schema) (before : State) (r : Request)
    (initial : valid schema before = true) (norm : normalized r = true)
    (fresh : ∀ claim ∈ before.claims, claim.manager ≠ r.manager)
    (snapshot : r.cells = before.cells) :
    ∃ after, apply schema before r = .ok after ∧ after.cells = before.cells ∧ ClaimsExtend before after := by
  let after : State := {
    cells := before.cells
    claims := before.claims ++ r.desired.map (fun target => ⟨r.manager, target, true⟩) }
  have noOwn : ∀ path kind, before.owns r.manager path kind = false := by
    intro path kind
    apply Bool.eq_false_iff.mpr
    intro owned
    obtain ⟨claim, member, same⟩ := List.any_eq_true.mp owned
    have equality : claim.manager = r.manager := (Bool.and_eq_true_iff.mp same).1 |> eq_of_beq
    exact fresh claim member equality
  have noRoots : deletedRoots schema before (merged before r) r = [] := by
    unfold deletedRoots
    rw [List.filter_eq_nil_iff.mpr (by intro item _; simp [deletesItem, full, noOwn])]
    rfl
  have noRelease : releasesAny before r = false := by
    apply Bool.eq_false_iff.mpr
    intro found
    obtain ⟨claim, member, released⟩ := List.any_eq_true.mp found
    exact fresh claim member ((Bool.and_eq_true_iff.mp released).1 |> eq_of_beq)
  have sameMerge : merged before r = after := by
    have oldEmpty : before.cells.filter (fun old => !r.cells.any (fun c => c.path == old.path)) = [] := by
      apply List.filter_eq_nil_iff.mpr
      intro cell member
      have found : r.cells.any (fun c => c.path == cell.path) = true := by
        rw [snapshot]
        exact List.any_eq_true.mpr ⟨cell,member,by simp⟩
      simp [found]
    have oldClaims : before.claims.filter (fun c => c.manager != r.manager &&
        !(r.force && c.target.kind == .value &&
          r.writes.any (fun cell => cell.path == c.target.path && changed before cell))) = before.claims := by
      apply List.filter_eq_self.mpr
      intro claim member
      have unchanged : ∀ cell ∈ r.writes, changed before cell = false := by
        intro cell written
        apply Bool.eq_false_iff.mpr
        intro changedValue
        obtain ⟨old,present,different⟩ := List.any_eq_true.mp changedValue
        simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne] at different
        have cellMember : cell ∈ before.cells := by
          rw [← snapshot]; exact List.mem_append_left r.context written
        have shape := initial
        simp only [valid, shapeValid, Bool.and_eq_true] at shape
        have nodup := Contracts.nodup_of_eraseDups_length (before.cells.map (·.path))
          (by simpa using shape.1.1.1.1.2)
        have equal := Contracts.cells_unique_by_path before.cells nodup old cell present cellMember different.1
        exact different.2 (congrArg Cell.content equal)
      have noChange : r.writes.any (fun cell => cell.path == claim.target.path && changed before cell) = false := by
        apply Bool.eq_false_iff.mpr
        intro found
        obtain ⟨cell,member,different⟩ := List.any_eq_true.mp found
        simp [unchanged cell member] at different
      simp [fresh claim member, noChange]
    unfold merged updatedClaims
    rw [oldEmpty, oldClaims, snapshot]
    simp only [List.append_nil]
    rfl
  have sameCandidate : candidate schema before r = after := by
    change pruneReleased before r (dropSubtrees (merged before r) (deletedRoots schema before (merged before r) r)) = after
    rw [noRoots, sameMerge]
    have dropped : dropSubtrees after [] = after := by
      have cells : after.cells.filter (fun _ => true) = after.cells := List.filter_eq_self.mpr (by intros; rfl)
      have claims : after.claims.filter (fun _ => true) = after.claims := List.filter_eq_self.mpr (by intros; rfl)
      simp only [dropSubtrees, List.any_nil, Bool.not_false, cells, claims]
    rw [dropped]
    have allKept : ∀ cell ∈ after.cells, keepCell before r after cell = true := by
      intro cell _
      cases kind : cell.content <;> simp [keepCell, kind, retainedValue, released, noOwn, noRelease]
    simp [pruneReleased, List.filter_eq_self.mpr allKept]
  have extend : ClaimsExtend before after := fun claim member => List.mem_append_left _ member
  have newTargets : ∀ fp ∈ r.desired, targetExists before fp = true := by
    have hn := norm
    simp only [normalized, Bool.and_eq_true, beq_iff_eq] at hn
    intro fp desired
    rw [hn.1.1.1] at desired
    obtain ⟨cell,written,rfl⟩ := List.mem_map.mp desired
    have stored : cell ∈ before.cells := by
      rw [← snapshot]; exact List.mem_append_left r.context written
    cases kind : cell.content <;>
      simp only [Cell.footprint, kind, targetExists, State.hasValue, State.hasContainer, State.hasItem, Bool.or_eq_true] <;>
      first | exact List.any_eq_true.mpr ⟨cell,stored,by simp [kind]⟩
            | exact Or.inl (List.any_eq_true.mpr ⟨cell,stored,by simp [kind]⟩)
  have shapeBefore := initial
  simp only [valid, Bool.and_eq_true] at shapeBefore
  have shapeAfter : shapeValid schema after = true := by
    have h := shapeBefore.1.1
    simp only [shapeValid, Bool.and_eq_true, List.all_eq_true] at h ⊢
    refine ⟨⟨⟨h.1.1.1,h.1.1.2⟩,h.1.2⟩,?_⟩
    intro claim member
    rcases List.mem_append.mp member with old | new
    · exact h.2 claim old
    · obtain ⟨fp,desired,rfl⟩ := List.mem_map.mp new
      exact newTargets fp desired
  have sameRequired : ∀ core, requiredPresent after core = requiredPresent before core := by
    intro core
    induction core with
    | none => rfl
    | value path => rfl
    | collection path => rfl
    | object path children ih => simp only [requiredPresent, ih]; rfl
    | both left right ihLeft ihRight => simp only [requiredPresent, ihLeft, ihRight]
  have valuesAfter : valuesValid schema after = true := by
    simpa only [valuesValid, State.hasItem, sameRequired] using shapeBefore.1.2
  have completeAfter : complete schema after = true := by
    simp only [complete, List.all_eq_true] at shapeBefore ⊢
    intro item member
    have previous := shapeBefore.2 item member
    change (!before.hasItem item.path || hasFull after item) = true
    cases existsItem : before.hasItem item.path with
    | false => simp only [Bool.not_false, Bool.true_or]
    | true =>
        have previousFull : hasFull before item = true := by simpa [existsItem] using previous
        obtain ⟨claim,owned,wasFull⟩ := List.any_eq_true.mp previousFull
        have nowFull := full_monotone before after extend claim.manager item wasFull
        have nowWitness : hasFull after item = true := List.any_eq_true.mpr ⟨claim,extend claim owned,nowFull⟩
        simp [nowWitness]
  have noConflict : conflicts before r = false := by
    apply Contracts.equal_writes_do_not_conflict
    intro cell written
    apply Bool.eq_false_iff.mpr
    intro changedValue
    obtain ⟨old,stored,different⟩ := List.any_eq_true.mp changedValue
    simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne] at different
    have member : cell ∈ before.cells := by rw [← snapshot]; exact List.mem_append_left _ written
    have unique := Contracts.valid_unique_paths schema before initial
    exact different.2 (congrArg Cell.content (Contracts.cells_unique_by_path before.cells unique old cell stored member different.1))
  exact ⟨after,by simp [apply,initial,norm,noConflict,sameCandidate,shapeAfter,valuesAfter,completeAfter],rfl,extend⟩

end SSA.Progress
