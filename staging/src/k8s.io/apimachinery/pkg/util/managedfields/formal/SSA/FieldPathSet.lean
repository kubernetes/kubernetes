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
import SSA.FieldsV1
import SSA.Pruning

namespace SSA.FieldPathSet

open FieldsV1

/-- Enumerate direct membership as well as descendant membership; the two are independent. -/
def paths : Trie → FieldSet
  | .node member children =>
      (if member then [[]] else []) ++ children.flatMap fun child =>
        (paths child.2).map (child.1 :: ·)
termination_by trie => sizeOf trie
decreasing_by
  have smaller := List.sizeOf_lt_of_mem (show child ∈ children from by assumption)
  cases child
  simp_all <;> omega

/-- This normalization is exact at every finite depth, not only for a list of fixtures. -/
theorem paths_membership (trie : Trie) (path : Path) :
    path ∈ paths trie ↔ memberAt trie path = true := by
  induction path generalizing trie with
  | nil =>
      cases trie with
      | node member children =>
          cases member <;> simp [paths, memberAt, List.mem_flatMap, List.mem_map]
  | cons segment rest ih =>
      cases trie with
      | node member children =>
          have rootAbsent : segment :: rest ∉ (if member then [[]] else ([] : FieldSet)) := by
            cases member <;> simp
          simp only [paths, memberAt, List.mem_append, rootAbsent, false_or,
            List.mem_flatMap, List.mem_map, List.any_eq_true, Bool.and_eq_true, beq_iff_eq]
          constructor
          · rintro ⟨child, present, suffix, member, same⟩
            rcases List.cons.inj same with ⟨sameKey, samePath⟩
            subst suffix
            exact ⟨child, present, sameKey, (ih child.2).mp member⟩
          · rintro ⟨child, present, sameKey, member⟩
            exact ⟨child, present, rest, (ih child.2).mpr member, by simp [sameKey]⟩

/-- Empty structural nodes are retained here; wire normalization is a separate operation. -/
def select : Trie → (Path → Bool) → Trie
  | .node member children, keep => .node (member && keep [])
      (children.map fun child => (child.1, select child.2 (fun path => keep (child.1 :: path))))
termination_by trie _ => sizeOf trie
decreasing_by
  have smaller := List.sizeOf_lt_of_mem (show child ∈ children from by assumption)
  cases child
  simp_all <;> omega

/-- Selecting paths cannot create ancestor membership from a descendant claim. -/
theorem select_membership (trie : Trie) (keep : Path → Bool) (path : Path) :
    memberAt (select trie keep) path = (memberAt trie path && keep path) := by
  induction path generalizing trie keep with
  | nil => cases trie <;> simp [select, memberAt]
  | cons segment rest ih =>
      cases trie with
      | node member children =>
          simp only [select, memberAt, List.any_map, Function.comp_def]
          simp only [ih]
          apply Bool.eq_iff_iff.mpr
          simp only [List.any_eq_true, Bool.and_eq_true, beq_iff_eq]
          constructor
          · rintro ⟨child, present, same, member, selected⟩
            exact ⟨⟨child, present, same, member⟩, by simpa [same] using selected⟩
          · rintro ⟨⟨child, present, same, member⟩, selected⟩
            exact ⟨child, present, same, member, by simpa [same] using selected⟩

def difference (left right : Trie) : Trie := select left (fun path => !memberAt right path)
def intersection (left right : Trie) : Trie := select left (memberAt right)

/-- Ordinary set difference removes only the selected membership, not an entire descendant scope. -/
theorem flat_difference_refines_trie (left right : Trie) (path : Path) :
    path ∈ fieldDifference (paths left) (paths right) ↔ memberAt (difference left right) path = true := by
  simp [fieldDifference, difference, select_membership, paths_membership]

theorem flat_intersection_refines_trie (left right : Trie) (path : Path) :
    path ∈ fieldIntersection (paths left) (paths right) ↔ memberAt (intersection left right) path = true := by
  simp [fieldIntersection, intersection, select_membership, paths_membership]

end SSA.FieldPathSet
