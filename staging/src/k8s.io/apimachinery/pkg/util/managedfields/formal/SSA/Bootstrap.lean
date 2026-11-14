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
import SSA.Update

namespace SSA

/-- Default-manager initialization is an Update identity, distinct from an applying actor. -/
def beforeFirstApply : String := "before-first-apply@update"

def existingSnapshot (before : State) : Request :=
  { manager := beforeFirstApply, operation := .update
    desired := before.cells.map Cell.footprint, writes := before.cells }

/-- The Apply branch of skipNonAppliedManager seeds Update ownership whenever the
    stored manager map is empty, including after an earlier Apply lost its last claim. -/
def bootstrapExisting (before : State) : State :=
  if before.claims.isEmpty then updateCandidate {} (existingSnapshot before)
  else before

/-- Initialization records ownership of existing data without changing that data. -/
theorem bootstrap_preserves_data (before : State) :
    (bootstrapExisting before).cells = before.cells := by
  unfold bootstrapExisting
  split
  · simp [updateCandidate, existingSnapshot, Request.cells]
  · rfl

theorem bootstrap_keeps_nonempty_manager_map (before : State) (nonempty : before.claims ≠ []) :
    bootstrapExisting before = before := by
  cases fields : before.claims with
  | nil => exact (nonempty fields).elim
  | cons head tail => simp [bootstrapExisting, fields]

/-- An Update from an empty object owns every added footprint, including an empty list. -/
theorem bootstrap_from_empty_claims (before : State) (empty : before.claims = []) :
    bootstrapExisting before =
      { before with claims := (before.cells.map (fun cell =>
          Claim.mk beforeFirstApply cell.footprint false)).eraseDups } := by
  have touched : before.cells.filter (updateTouches {} { cells := before.cells }) = before.cells := by
    apply List.filter_eq_self.mpr
    intro cell _
    simp [updateTouches]
  simp [bootstrapExisting, empty, updateCandidate, existingSnapshot, Request.cells, touched, Function.comp_def]

/-- The synthetic owner is never an applying manager; PR #336 exempts this operation class. -/
theorem bootstrap_creates_update_claims (before : State) (empty : before.claims = [])
    (claim : Claim) (member : claim ∈ (bootstrapExisting before).claims) :
    claim.manager = beforeFirstApply ∧ claim.applied = false := by
  rw [bootstrap_from_empty_claims before empty] at member
  obtain ⟨cell, _, rfl⟩ := List.mem_map.mp (List.mem_eraseDups.mp member)
  exact ⟨rfl, rfl⟩

end SSA
