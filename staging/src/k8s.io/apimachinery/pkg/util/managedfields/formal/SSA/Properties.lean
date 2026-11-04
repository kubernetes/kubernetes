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

namespace SSA

theorem accepted_valid (schema : Schema) (before after : State) (r : Request)
    (h : apply schema before r = .ok after) : valid schema after = true := by
  unfold apply at h
  split at h
  · cases h
  · split at h
    · cases h
    · split at h
      · cases h
      · dsimp only at h
        split at h
        · cases h
        · split at h
          · cases h
          · cases h
            simp_all [valid]

theorem accepted_complete (schema : Schema) (before after : State) (r : Request)
    (h : apply schema before r = .ok after) : complete schema after = true := by
  have hv := accepted_valid schema before after r h
  simp only [valid, Bool.and_eq_true] at hv
  exact hv.2

theorem every_surviving_item_has_full (schema : Schema) (before after : State) (r : Request)
    (h : apply schema before r = .ok after) (item : ItemSpec) (hi : item ∈ schema)
    (existsItem : after.hasItem item.path = true) :
    ∃ manager, full after manager item = true := by
  have hc := accepted_complete schema before after r h
  simp only [complete, List.all_eq_true] at hc
  have hf : hasFull after item = true := by simpa [existsItem] using hc item hi
  obtain ⟨claim, _, hfull⟩ := List.any_eq_true.mp hf
  exact ⟨claim.manager, hfull⟩

theorem rejected_is_atomic (before : State) (error : ApplyError) :
    commit before (.error error) = before := rfl

theorem committed_valid (schema : Schema) (before : State) (r : Request)
    (h : valid schema before = true) :
    valid schema (commit before (apply schema before r)) = true := by
  cases result : apply schema before r with
  | error error => simpa [commit] using h
  | ok after => exact accepted_valid schema before after r result

theorem key_only_full (s : State) (manager : String) (path : Path) :
    full s manager ⟨path, .none⟩ = s.owns manager path .item := by
  simp [full, covers]

/-- One witness must own both branches; a coalition of different owners is insufficient. -/
theorem required_branches_same_manager (s : State) (manager : String)
    (path : Path) (left right : RequiredCore) :
    full s manager ⟨path, .both left right⟩ = true ↔
      s.owns manager path .item = true ∧
      covers s manager left = true ∧ covers s manager right = true := by
  simp [full, covers, Bool.and_eq_true]

theorem partial_cannot_directly_delete (before provisional : State) (r : Request)
    (item : ItemSpec) (partialManager : full before r.manager item = false) :
    deletesItem before provisional r item = false := by
  simp [deletesItem, partialManager]

theorem surviving_full_blocks_direct_deletion (before provisional : State) (r : Request)
    (item : ItemSpec) (remaining : hasFull provisional item = true) :
    deletesItem before provisional r item = false := by
  simp [deletesItem, remaining]

/-- An unrelated subtree keeps its cells and ownership through lifetime pruning. -/
theorem subtree_frame (s : State) (roots : List Path) (cell : Cell)
    (unrelated : roots.any (fun root => under root cell.path) = false) :
    cell ∈ (dropSubtrees s roots).cells ↔ cell ∈ s.cells := by
  simp [dropSubtrees, List.mem_filter, unrelated]

theorem claims_frame (s : State) (roots : List Path) (claim : Claim)
    (unrelated : roots.any (fun root => under root claim.target.path) = false) :
    claim ∈ (dropSubtrees s roots).claims ↔ claim ∈ s.claims := by
  simp [dropSubtrees, List.mem_filter, unrelated]

/-- Pruning cannot drop an old scalar that is neither overwritten, released, nor below a deletion. -/
theorem candidate_preserves_unrelated_value (schema : Schema) (before : State) (r : Request)
    (cell : Cell) (text : String) (present : cell ∈ before.cells)
    (kind : cell.content = .value text)
    (unwritten : r.writes.any (fun c => c.path == cell.path) = false)
    (unrelated : (deletedRoots schema before (merged before r) r).any
      (fun root => under root cell.path) = false)
    (unreleased : released before r cell.path .value = false) :
    cell ∈ (candidate schema before r).cells := by
  have hm : cell ∈ (merged before r).cells := by
    simp [merged, List.mem_filter, present, unwritten]
  have hd := (subtree_frame (merged before r)
    (deletedRoots schema before (merged before r) r) cell unrelated).mpr hm
  change cell ∈ List.filter _
    (dropSubtrees (merged before r) (deletedRoots schema before (merged before r) r)).cells
  apply List.mem_filter.mpr
  exact ⟨hd, by simp [kind, unreleased]⟩

/-- A child's full owner cannot retain a cell beneath a deleted ancestor. -/
theorem ancestor_deletion_dominates (s : State) (roots : List Path) (root : Path)
    (cell : Cell) (deleted : root ∈ roots) (descendant : under root cell.path = true) :
    cell ∉ (dropSubtrees s roots).cells := by
  have found : roots.any (fun p => under p cell.path) = true :=
    List.any_eq_true.mpr ⟨root, deleted, descendant⟩
  simp [dropSubtrees, List.mem_filter, found]

/-- Acquiring co-ownership cannot make an existing complete owner partial. -/
def ClaimsExtend (before after : State) : Prop :=
  ∀ claim ∈ before.claims, claim ∈ after.claims

theorem owns_monotone (before after : State) (h : ClaimsExtend before after)
    (manager : String) (path : Path) (kind : ClaimKind)
    (owned : before.owns manager path kind = true) :
    after.owns manager path kind = true := by
  obtain ⟨claim, mem, matched⟩ := List.any_eq_true.mp owned
  exact List.any_eq_true.mpr ⟨claim, h claim mem, matched⟩

theorem map_presence_monotone (before after : State) (h : ClaimsExtend before after)
    (manager : String) (path : Path)
    (owned : before.ownsMapPresence manager path = true) :
    after.ownsMapPresence manager path = true := by
  obtain ⟨claim, mem, matched⟩ := List.any_eq_true.mp owned
  exact List.any_eq_true.mpr ⟨claim, h claim mem, matched⟩

theorem covers_monotone (before after : State) (h : ClaimsExtend before after)
    (manager : String) (core : RequiredCore)
    (owned : covers before manager core = true) :
    covers after manager core = true := by
  induction core with
  | none => rfl
  | value path => exact owns_monotone before after h manager path .value owned
  | collection path => exact owns_monotone before after h manager path .presence owned
  | object path core ih =>
      simp only [covers, Bool.and_eq_true] at owned ⊢
      exact ⟨map_presence_monotone before after h manager path owned.1, ih owned.2⟩
  | both left right ihLeft ihRight =>
      simp only [covers, Bool.and_eq_true] at owned ⊢
      exact ⟨ihLeft owned.1, ihRight owned.2⟩

theorem full_monotone (before after : State) (h : ClaimsExtend before after)
    (manager : String) (item : ItemSpec) (owned : full before manager item = true) :
    full after manager item = true := by
  simp only [full, Bool.and_eq_true] at owned ⊢
  exact ⟨owns_monotone before after h manager item.path .item owned.1,
    covers_monotone before after h manager item.required owned.2⟩

end SSA
