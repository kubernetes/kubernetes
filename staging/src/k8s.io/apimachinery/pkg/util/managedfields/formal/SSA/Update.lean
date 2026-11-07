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

namespace SSA

/-- An Update owns added/modified fields, not every field present in the replacement object. -/
def updateTouches (before after : State) (cell : Cell) : Bool :=
  !before.cells.any (fun old => old.path == cell.path) || modifiedCell before after cell

/-- Manager identities include operation, as they do in the managedFields API envelope. -/
def updateCandidate (before : State) (r : Request) : State :=
  let data : State := { cells := r.cells }
  let changedTargets := (r.cells.filter (updateTouches before { cells := r.cells })).map Cell.footprint
  { cells := r.cells
    claims := (before.claims.filter (fun (claim : Claim) =>
      targetExists data claim.target && !changedTargets.any (fun target => target.path == claim.target.path)) ++
      changedTargets.map (fun target => Claim.mk r.manager target false)).eraseDups }

/-- An explicit replacement may remove an entire subtree even when other managers own it. -/
def update (mode : Semantics) (schema : Schema) (before : State) (r : Request) : Except ApplyError State :=
  let initial := if mode == .full then valid schema before else shapeValid schema before
  if !initial then .error .invalidInitialState
  else if !normalized r then .error .invalidRequest
  else
    let after := updateCandidate before r
    if !shapeValid schema after then .error .invalidResult
    else if mode == .full && !(valuesValid schema after && complete schema after) then .error .orphanedItem
    else .ok after

def writeWith (operation : Operation) (mode : Semantics) (schema : Schema)
    (before : State) (r : Request) : Except ApplyError State :=
  match operation with
  | .apply => applyWith mode schema before r
  | .update => update mode schema before r

/-- Update's data is the complete replacement, with no implicit lifetime pruning. -/
theorem update_candidate_replaces_data (before : State) (r : Request) :
    (updateCandidate before r).cells = r.cells := rfl

/-- Update transfers only touched targets; unrelated existing claims survive. -/
theorem update_preserves_untouched_claim (before : State) (r : Request) (claim : Claim)
    (member : claim ∈ before.claims)
    (stillExists : targetExists { cells := r.cells } claim.target = true)
    (untouched : ((r.cells.filter (updateTouches before { cells := r.cells })).map Cell.footprint).any
      (fun target => target.path == claim.target.path) = false) :
    claim ∈ (updateCandidate before r).claims := by
  simp only [updateCandidate, List.mem_eraseDups, List.mem_append, List.mem_filter]
  left
  exact ⟨member,by simp [stillExists,untouched]⟩

/-- Every touched footprint becomes Update ownership, so PR #336 must respect the operation flag. -/
theorem update_claims_touched_target (before : State) (r : Request) (cell : Cell)
    (written : cell ∈ r.writes) (touched : updateTouches before { cells := r.cells } cell = true) :
    Claim.mk r.manager cell.footprint false ∈ (updateCandidate before r).claims := by
  simp only [updateCandidate, List.mem_eraseDups, List.mem_append]
  right
  apply List.mem_map.mpr
  refine ⟨cell.footprint,List.mem_map.mpr ⟨cell,List.mem_filter.mpr ⟨List.mem_append_left _ written,touched⟩,rfl⟩,rfl⟩

/-- The full policy also checks the invariant after an explicit replacement. -/
theorem accepted_full_update_valid (schema : Schema) (before after : State) (r : Request)
    (accepted : update .full schema before r = .ok after) : valid schema after = true := by
  simp only [update,beq_self_eq_true,Bool.true_and,↓reduceIte] at accepted
  split at accepted
  · cases accepted
  · split at accepted
    · cases accepted
    · split at accepted
      · cases accepted
      · split at accepted
        · cases accepted
        · cases accepted
          simp_all [valid]

end SSA
