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
import SSA.ObjectScope

namespace SSA.Boundary

/-- These stages describe the API-server boundary without pretending to verify a CEL interpreter. -/
structure Policy where
  mutate : State → State
  validate : State → State → Bool

/-- Mutations precede validation; both operate on a proposed state and commit atomically. -/
def finish (schema : Schema) (policy : Policy) (before : State)
    (proposal : Except ApplyError State) : Except ApplyError State :=
  match proposal with
  | .error error => .error error
  | .ok proposed =>
      let final := policy.mutate proposed
      if valid schema final && policy.validate before final then .ok final
      else .error .invalidResult

/-- The final invariant must be checked after mutation, not only before the webhook runs. -/
theorem accepted_final_valid (schema : Schema) (policy : Policy) (before after : State)
    (proposal : Except ApplyError State)
    (accepted : finish schema policy before proposal = .ok after) : valid schema after = true := by
  cases proposal with
  | error error => cases accepted
  | ok proposed =>
      simp only [finish] at accepted
      split at accepted
      · rename_i good
        cases accepted
        exact (Bool.and_eq_true_iff.mp good).1
      · cases accepted

/-- An arbitrary relational validator can constrain old/new values and multiple fields. -/
theorem accepted_satisfies_validator (schema : Schema) (policy : Policy) (before after : State)
    (proposal : Except ApplyError State)
    (accepted : finish schema policy before proposal = .ok after) :
    policy.validate before after = true := by
  cases proposal with
  | error error => cases accepted
  | ok proposed =>
      simp only [finish] at accepted
      split at accepted
      · rename_i good
        cases accepted
        exact (Bool.and_eq_true_iff.mp good).2
      · cases accepted

/-- Validator rejection rolls back both data and ownership, including all webhook mutations. -/
theorem validation_rejection_is_atomic (schema : Schema) (policy : Policy)
    (before proposed : State) (rejected : policy.validate before (policy.mutate proposed) = false) :
    commit before (finish schema policy before (.ok proposed)) = before := by
  simp [finish, rejected, commit]

/-- Value-only validation cannot distinguish two ownership assignments with identical data. -/
theorem value_only_validation_cannot_enforce_ownership (check : List Cell → Bool)
    (left right : State) (same : left.cells = right.cells) :
    check left.cells = check right.cells := by rw [same]

/-- Even with all ownership guards, an arbitrary admission policy can forbid a handoff. -/
theorem validator_can_block_any_handoff (schema : Schema) (mutate : State → State)
    (before proposed : State) :
    finish schema ⟨mutate, fun _ _ => false⟩ before (.ok proposed) = .error .invalidResult := by
  simp [finish]

/-- A no-op mutation and a permissive validator preserve every already-valid proposal. -/
theorem permissive_boundary_accepts (schema : Schema) (before proposed : State)
    (good : valid schema proposed = true) :
    finish schema ⟨id, fun _ _ => true⟩ before (.ok proposed) = .ok proposed := by
  simp [finish, good]

/-- The complete boundary rechecks all object scopes after the mutating stage. -/
def finishScoped (objects : List ObjectSpec) (schema : Schema) (policy : Policy) (before : State)
    (proposal : Except ApplyError State) : Except ApplyError State :=
  checkObjects objects .full (finish schema policy before proposal)

/-- Successful post-admission results satisfy item, object, and arbitrary relational validation. -/
theorem accepted_scoped_boundary (objects : List ObjectSpec) (schema : Schema) (policy : Policy)
    (before after : State) (proposal : Except ApplyError State)
    (accepted : finishScoped objects schema policy before proposal = .ok after) :
    valid schema after = true ∧ objectsValid objects after = true ∧ policy.validate before after = true := by
  have core := checked_proposal objects (finish schema policy before proposal) after accepted
  exact ⟨accepted_final_valid schema policy before after proposal core,
    check_objects_valid objects _ after accepted,
    accepted_satisfies_validator schema policy before after proposal core⟩

end SSA.Boundary
