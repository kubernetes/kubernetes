/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.Contracts

namespace SSA.KeyOnly

/-- For an empty required core, any membership claimant supplies a full-manager witness. -/
theorem has_full_iff_membership (s : State) (path : Path) :
    hasFull s ⟨path,.none⟩ = true ↔
      ∃ claim ∈ s.claims, claim.target = ⟨path,.item⟩ := by
  constructor
  · intro complete
    obtain ⟨candidate,_,fullOwner⟩ := List.any_eq_true.mp complete
    have owned : s.owns candidate.manager path .item = true := by
      simpa [full,covers] using fullOwner
    obtain ⟨claim,present,matched⟩ := List.any_eq_true.mp owned
    exact ⟨claim,present,(Bool.and_eq_true_iff.mp matched).2 |> beq_iff_eq.mp⟩
  · rintro ⟨claim,present,target⟩
    apply List.any_eq_true.mpr
    refine ⟨claim,present,?_⟩
    simp only [full,covers,Bool.and_true]
    apply List.any_eq_true.mpr
    exact ⟨claim,present,by simp [target]⟩

/-- Scalar force transfer cannot remove another manager's item membership. -/
theorem foreign_membership_survives_merge (before : State) (r : Request) (path : Path)
    (claim : Claim) (present : claim ∈ before.claims) (target : claim.target = ⟨path,.item⟩)
    (foreign : claim.manager ≠ r.manager) : claim ∈ (merged before r).claims := by
  apply List.mem_append_left
  apply List.mem_filter.mpr
  exact ⟨present,by simp [target,foreign]⟩

/-- After the actor omits membership, exactly the old foreign memberships remain. -/
theorem merged_membership_iff_foreign (before : State) (r : Request) (path : Path)
    (released : r.desired.contains ⟨path,.item⟩ = false) :
    (∃ claim ∈ (merged before r).claims, claim.target = ⟨path,.item⟩) ↔
      ∃ claim ∈ before.claims, claim.target = ⟨path,.item⟩ ∧ claim.manager ≠ r.manager := by
  constructor
  · rintro ⟨claim,present,target⟩
    change claim ∈ _ ++ _ at present
    rcases List.mem_append.mp present with old | fresh
    · obtain ⟨present,kept⟩ := List.mem_filter.mp old
      refine ⟨claim,present,target,?_⟩
      exact bne_iff_ne.mp (Bool.and_eq_true_iff.mp kept).1
    · obtain ⟨footprint,desired,same⟩ := List.mem_map.mp fresh
      subst claim
      have sameTarget : footprint = ⟨path,.item⟩ := target
      subst footprint
      have notDesired : ¬(⟨path,.item⟩ : Footprint) ∈ r.desired := by simpa using released
      exact False.elim (notDesired desired)
  · rintro ⟨claim,present,target,foreign⟩
    exact ⟨claim,foreign_membership_survives_merge before r path claim present target foreign,target⟩

/-- Direct lifetime deletion is exactly last-membership release, at arbitrary finite manager counts.
    Enclosing lifetime deletion is a separate operation. -/
theorem direct_deletion_iff_last_membership (before : State) (r : Request) (path : Path)
    (existsItem : before.hasItem path = true)
    (owned : before.owns r.manager path .item = true)
    (released : r.desired.contains ⟨path,.item⟩ = false) :
    deletesItem before (merged before r) r ⟨path,.none⟩ = true ↔
      ∀ claim ∈ before.claims, claim.target = ⟨path,.item⟩ → claim.manager = r.manager := by
  have fullOwner : full before r.manager ⟨path,.none⟩ = true := by
    simpa [full,covers] using owned
  simp only [deletesItem,existsItem,fullOwner,released,Bool.true_and,Bool.not_false]
  have inversion (b : Bool) : (!b) = true ↔ ¬b = true := by cases b <;> simp
  rw [inversion,has_full_iff_membership,merged_membership_iff_foreign before r path released]
  simp

/-- The last release prunes the whole item, including optional foreign-owned descendants. -/
theorem last_membership_prunes_subtree (schema : Schema) (before : State) (r : Request)
    (path : Path) (itemInSchema : (⟨path,.none⟩ : ItemSpec) ∈ schema)
    (existsItem : before.hasItem path = true)
    (owned : before.owns r.manager path .item = true)
    (released : r.desired.contains ⟨path,.item⟩ = false)
    (last : ∀ claim ∈ before.claims, claim.target = ⟨path,.item⟩ → claim.manager = r.manager)
    (cell : Cell) (descendant : under path cell.path = true) :
    cell ∉ (candidate schema before r).cells := by
  have deleting := (direct_deletion_iff_last_membership before r path existsItem owned released).mpr last
  have selected : path ∈ deletedRoots schema before (merged before r) r :=
    List.mem_map.mpr ⟨⟨path,.none⟩,List.mem_filter.mpr ⟨itemInSchema,deleting⟩,rfl⟩
  have absent := ancestor_deletion_dominates (merged before r)
    (deletedRoots schema before (merged before r) r) path cell selected descendant
  intro present
  exact absent (List.mem_filter.mp present).1

/-- Acceptance cannot reintroduce a subtree removed by the last membership release. -/
theorem accepted_last_membership_prunes_subtree (schema : Schema) (before after : State) (r : Request)
    (path : Path) (itemInSchema : (⟨path,.none⟩ : ItemSpec) ∈ schema)
    (existsItem : before.hasItem path = true)
    (owned : before.owns r.manager path .item = true)
    (released : r.desired.contains ⟨path,.item⟩ = false)
    (last : ∀ claim ∈ before.claims, claim.target = ⟨path,.item⟩ → claim.manager = r.manager)
    (accepted : apply schema before r = .ok after)
    (cell : Cell) (descendant : under path cell.path = true) :
    cell ∉ after.cells := by
  rw [← Contracts.accepted_candidate schema before after r accepted]
  exact last_membership_prunes_subtree schema before r path itemInSchema existsItem owned released last cell descendant

/-- A retained member preserves a key-only item unless a proper ancestor's lifetime ends.
    The accepted transition supplies schema uniqueness and the final data/claim consistency check. -/
theorem accepted_foreign_membership_preserves_item (schema : Schema) (before after : State) (r : Request)
    (path : Path) (itemInSchema : (⟨path,.none⟩ : ItemSpec) ∈ schema)
    (claim : Claim) (present : claim ∈ before.claims) (target : claim.target = ⟨path,.item⟩)
    (foreign : claim.manager ≠ r.manager)
    (noAncestor : (deletedRoots schema before (merged before r) r).any
      (fun root => properUnder root path) = false)
    (accepted : apply schema before r = .ok after) : after.hasItem path = true := by
  have beforeValid := Contracts.accepted_initial_valid schema before after r accepted
  have schemaGood : schemaValid schema = true := by
    simp only [valid,shapeValid,Bool.and_eq_true] at beforeValid
    exact beforeValid.1.1.1.1.1
  have unique := Contracts.schema_unique_paths schema schemaGood
  have inMerged := foreign_membership_survives_merge before r path claim present target foreign
  have fullAfter : hasFull (merged before r) ⟨path,.none⟩ = true :=
    (has_full_iff_membership (merged before r) path).mpr ⟨claim,inMerged,target⟩
  have noDeletion : (deletedRoots schema before (merged before r) r).any
      (fun root => under root path) = false := by
    apply Bool.eq_false_iff.mpr
    intro someRoot
    obtain ⟨root,selected,below⟩ := List.any_eq_true.mp someRoot
    by_cases same : root = path
    · obtain ⟨item,kept,rootPath⟩ := List.mem_map.mp selected
      obtain ⟨member,deleting⟩ := List.mem_filter.mp kept
      have sameItem := Contracts.members_equal_of_nodup_map schema (·.path) unique
        item ⟨path,.none⟩ member itemInSchema (rootPath.trans same)
      subst item
      have blocked := surviving_full_blocks_direct_deletion before (merged before r) r ⟨path,.none⟩ fullAfter
      rw [blocked] at deleting
      cases deleting
    · have proper : properUnder root path = true := by simp [properUnder,below,same]
      have forbidden : (deletedRoots schema before (merged before r) r).any
          (fun candidate => properUnder candidate path) = true :=
        List.any_eq_true.mpr ⟨root,selected,proper⟩
      rw [noAncestor] at forbidden
      cases forbidden
  have kept := (claims_frame (merged before r)
    (deletedRoots schema before (merged before r) r) claim (by simpa [target] using noDeletion)).mpr inMerged
  have finalClaim : claim ∈ after.claims := by
    rw [← Contracts.accepted_candidate schema before after r accepted]
    exact kept
  have afterValid := accepted_valid schema before after r accepted
  simp only [valid,shapeValid,Bool.and_eq_true] at afterValid
  have existsTarget := List.all_eq_true.mp afterValid.1.1.2 claim finalClaim
  simpa [targetExists,target] using existsTarget

end SSA.KeyOnly
