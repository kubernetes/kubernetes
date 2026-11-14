/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.UpdaterContracts
import SSA.ApplySource

namespace SSA.UpdaterBridge

open GoSubset (ManagedFields ManagerFields)
open UpdaterSubset

structure FailureMap (Err : Type) where
  runtime : String → Err
  conflict : ManagedFields → Err

/-- Supported source errors are explicit; fuel/type failures remain failures, never invented successes. -/
def decode {Obj Err : Type} (errors : FailureMap Err) : Except String (Result Obj Err) → Except Err ManagedFields
  | .ok (.returned [.managed updated,.reference _,.nil] _) => .ok updated
  | .ok (.returned [.nil,.nil,.error error] _) => .error error
  | .ok (.returned [.nil,.nil,.conflicts owners] _) => .error (errors.conflict owners)
  | .ok (.returned [.nil,.nil,.literalError message] _) => .error (errors.runtime message)
  | .error message => .error (errors.runtime message)
  | _ => .error (errors.runtime "ownership-updater interpreter returned an unsupported result")

def finish {Obj Err : Type} (ops : Ops Obj Err) (errors : FailureMap Err)
    (old new : Obj) (version : String) (owners : ManagedFields) (actor : String) (force : Bool) : Except Err ManagedFields :=
  decode errors (exec ops 256 UpdaterSource.updateProgram (initial old new version owners actor force))

def applyOps {Obj Err : Type} (ops : Ops Obj Err) (errors : FailureMap Err) : GoSubset.Ops Obj Err :=
  {ops.primitives with finishUpdate := finish ops errors}

theorem decoded_success_is_source_success {Obj Err : Type} (errors : FailureMap Err)
    (result : Except String (Result Obj Err)) (updated : ManagedFields) (accepted : decode errors result = .ok updated) :
    ∃ address after, result = .ok (.returned [.managed updated,.reference address,.nil] after) := by
  unfold decode at accepted
  split at accepted <;> try cases accepted
  rename_i address state
  exact ⟨address,state,rfl⟩

/-- Source non-growth supplies the former opaque final-updater contract. -/
theorem source_update_does_not_grow_actor {Obj Err : Type} (ops : Ops Obj Err)
    (errors : FailureMap Err) (shrinks : UpdaterContracts.DifferenceShrinks ops) :
    ApplySource.UpdateDoesNotGrowActor (applyOps ops errors) := by
  intro old new version owners actor force updated accepted
  obtain ⟨address,after,executed⟩ := decoded_success_is_source_success errors _ updated accepted
  have bounded := UpdaterContracts.imported_update_never_adds_claims ops shrinks old new version owners updated
    actor force 256 [.reference address,.nil] after executed
  intro entry present path claimed
  exact bounded actor path ⟨entry,present,claimed⟩

theorem source_apply_excludes_reset_fields {Obj Err : Type} (ops : Ops Obj Err)
    (errors : FailureMap Err) (shrinks : UpdaterContracts.DifferenceShrinks ops)
    (live config : Obj) (version : String) (managers after : ManagedFields) (actor : String)
    (force : Bool) (object : GoSubset.Value Obj Err) (roots : FieldSet)
    (table : List (String × (FieldSet → FieldSet)))
    (modern : ops.primitives.ignoredFields = none) (configured : ops.primitives.ignoreFilters = some table)
    (selected : GoSubset.filterLookup table version = some (ResetFields.filter roots))
    (accepted : GoSubset.exec (applyOps ops errors) 128 GoSource.applyProgram
      (ApplySource.applyEnv live config version managers actor force) =
      .ok (.returned [object,.managed after,.nil]))
    (entry : ManagerFields) (owned : (actor,entry) ∈ after)
    (root path : Path) (reset : root ∈ roots) (descendant : under root path = true) :
    path ∉ entry.fields := by
  exact ApplySource.imported_apply_excludes_reset_fields (applyOps ops errors) live config version managers after actor
    force object roots table modern configured selected (source_update_does_not_grow_actor ops errors shrinks)
    accepted entry owned root path reset descendant

end SSA.UpdaterBridge
