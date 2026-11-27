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
import SSA.Properties

namespace SSA.Locality

/-- All ownership comparisons are restricted to the item being classified. -/
def ClaimsAgreeUnder (left right : State) (root : Path) : Prop :=
  ∀ claim, under root claim.target.path = true → (claim ∈ left.claims ↔ claim ∈ right.claims)

private theorem under_trans (a b c : Path) (ab : under a b = true) (bc : under b c = true) :
    under a c = true := by
  exact List.isPrefixOf_iff_prefix.mpr ((List.isPrefixOf_iff_prefix.mp ab).trans
    (List.isPrefixOf_iff_prefix.mp bc))

private theorem local_owns (left right : State) (root path : Path)
    (agree : ClaimsAgreeUnder left right root) (inside : under root path = true)
    (manager : String) (kind : ClaimKind) : left.owns manager path kind = right.owns manager path kind := by
  apply Bool.eq_iff_iff.mpr
  constructor <;> intro owns
  · obtain ⟨claim,member,matched⟩ := List.any_eq_true.mp owns
    have target : claim.target = ⟨path,kind⟩ := eq_of_beq ((Bool.and_eq_true_iff.mp matched).2)
    exact List.any_eq_true.mpr ⟨claim,(agree claim (by simpa [target] using inside)).mp member,matched⟩
  · obtain ⟨claim,member,matched⟩ := List.any_eq_true.mp owns
    have target : claim.target = ⟨path,kind⟩ := eq_of_beq ((Bool.and_eq_true_iff.mp matched).2)
    exact List.any_eq_true.mpr ⟨claim,(agree claim (by simpa [target] using inside)).mpr member,matched⟩

private theorem local_map_presence (left right : State) (root path : Path)
    (agree : ClaimsAgreeUnder left right root) (inside : under root path = true)
    (manager : String) : left.ownsMapPresence manager path = right.ownsMapPresence manager path := by
  apply Bool.eq_iff_iff.mpr
  constructor <;> intro owns
  · obtain ⟨claim,member,matched⟩ := List.any_eq_true.mp owns
    have descendant := (Bool.and_eq_true_iff.mp matched).2
    exact List.any_eq_true.mpr ⟨claim,(agree claim (under_trans root path claim.target.path inside descendant)).mp member,matched⟩
  · obtain ⟨claim,member,matched⟩ := List.any_eq_true.mp owns
    have descendant := (Bool.and_eq_true_iff.mp matched).2
    exact List.any_eq_true.mpr ⟨claim,(agree claim (under_trans root path claim.target.path inside descendant)).mpr member,matched⟩

/-- Required recursion is local even when required maps are nested to arbitrary finite depth. -/
theorem required_core_local (left right : State) (manager : String) (parent : Path) (core : RequiredCore)
    (agree : ClaimsAgreeUnder left right parent) (within : coreWithin parent core = true) :
    covers left manager core = covers right manager core := by
  induction core generalizing parent with
  | none => rfl
  | value path =>
      exact local_owns left right parent path agree (by simpa [coreWithin,requiredPath] using (Bool.and_eq_true_iff.mp ((Bool.and_eq_true_iff.mp within).1)).1) manager .value
  | collection path =>
      exact local_owns left right parent path agree (by simpa [coreWithin,requiredPath] using (Bool.and_eq_true_iff.mp ((Bool.and_eq_true_iff.mp within).1)).1) manager .presence
  | object path children ih =>
      simp only [coreWithin, requiredPath, Bool.and_eq_true] at within
      have inside := within.1.1.1
      have childAgree : ClaimsAgreeUnder left right path := fun claim underChild =>
        agree claim (under_trans parent path claim.target.path inside underChild)
      simp only [covers, local_map_presence left right parent path agree inside manager, ih path childAgree within.2]
  | both l r ihLeft ihRight =>
      simp only [coreWithin, Bool.and_eq_true] at within
      simp only [covers, ihLeft parent agree within.1, ihRight parent agree within.2]

/-- Changing ownership outside one item cannot change that item's full-manager classification. -/
theorem full_manager_is_item_local (left right : State) (manager : String) (item : ItemSpec)
    (agree : ClaimsAgreeUnder left right item.path) (within : coreWithin item.path item.required = true) :
    full left manager item = full right manager item := by
  simp only [full, local_owns left right item.path item.path agree (by simp [under]) manager .item,
    required_core_local left right manager item.path item.required agree within]

end SSA.Locality
