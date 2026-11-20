/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.FieldPathSet
import SSA.GoSubset

namespace SSA.ResetFields

open GoSubset FieldsV1

def excluded (roots : FieldSet) (path : Path) : Bool :=
  roots.any (fun root => under root path)

/-- Reset filters remove direct membership and every descendant, unlike ordinary Difference. -/
def filter (roots fields : FieldSet) : FieldSet :=
  fields.filter (fun path => !excluded roots path)

theorem filter_membership (roots fields : FieldSet) (path : Path) :
    path ∈ filter roots fields ↔ path ∈ fields ∧ excluded roots path = false := by
  simp [filter]

theorem filter_excludes_descendants (roots fields : FieldSet) (root path : Path)
    (reset : root ∈ roots) (descendant : under root path = true) :
    path ∉ filter roots fields := by
  have blocked : excluded roots path = true :=
    List.any_eq_true.mpr ⟨root,reset,descendant⟩
  simp [filter_membership,blocked]

theorem filter_is_idempotent (roots fields : FieldSet) :
    filter roots (filter roots fields) = filter roots fields := by
  simp [filter,List.filter_filter]

theorem filter_preserves_other_top_field (fields : FieldSet) (reset kept : String)
    (different : kept ≠ reset) (suffix : Path) :
    .field kept :: suffix ∈ filter [[.field reset]] fields ↔ .field kept :: suffix ∈ fields := by
  simp [filter_membership,excluded,under,List.isPrefixOf_cons_cons,Ne.symm different]

def filterTrie (roots : FieldSet) (trie : Trie) : Trie :=
  FieldPathSet.select trie (fun path => !excluded roots path)

/-- The normalized prefix filter agrees with recursive trie selection at arbitrary finite depth. -/
theorem flat_filter_refines_trie (roots : FieldSet) (trie : Trie) (path : Path) :
    path ∈ filter roots (FieldPathSet.paths trie) ↔
      memberAt (filterTrie roots trie) path = true := by
  simp [filter_membership,filterTrie,FieldPathSet.select_membership,FieldPathSet.paths_membership]

theorem actor_entry_set (managers : ManagedFields) (actor : String)
    (installed entry : ManagerFields) :
    (actor,entry) ∈ managerSet managers actor installed ↔ entry = installed := by
  simp [managerSet,eq_comm]

/-- Installing the filtered Actor set cannot acquire any reset field or its descendants. -/
theorem installed_actor_excludes_reset (roots fields : FieldSet) (managers : ManagedFields)
    (actor version : String) (entry : ManagerFields)
    (installed : (actor,entry) ∈ managerSet managers actor ⟨filter roots fields,version,true⟩)
    (root path : Path) (reset : root ∈ roots) (descendant : under root path = true) :
    path ∉ entry.fields := by
  have same := (actor_entry_set managers actor _ entry).mp installed
  subst entry
  exact filter_excludes_descendants roots fields root path reset descendant

/-- This is the remaining updater callback obligation, not a premise about the entire Apply result. -/
def ActorDoesNotGrow (before after : ManagedFields) (actor : String) : Prop :=
  ∀ entry, (actor,entry) ∈ after → ∀ path ∈ entry.fields,
    ∃ old, (actor,old) ∈ before ∧ path ∈ old.fields

/-- Reset exclusion composes with a subsequent ownership-update stage that cannot grow the Actor set. -/
theorem update_preserves_reset_exclusion (roots fields : FieldSet) (managers after : ManagedFields)
    (actor version : String)
    (update : ActorDoesNotGrow (managerSet managers actor ⟨filter roots fields,version,true⟩) after actor)
    (entry : ManagerFields) (owned : (actor,entry) ∈ after)
    (root path : Path) (reset : root ∈ roots) (descendant : under root path = true) :
    path ∉ entry.fields := by
  intro member
  obtain ⟨old,installed,present⟩ := update entry owned path member
  exact installed_actor_excludes_reset roots fields managers actor version old
    installed root path reset descendant present

end SSA.ResetFields

