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
import SSA.SourceContracts

namespace SSA.PruneSource

open GoSubset SourceContracts

set_option maxRecDepth 10000
set_option maxHeartbeats 4000000

def lastValue {Obj Err : Type} : Option (FieldSet × String) → Value Obj Err
  | none => .nil
  | some (fields, version) => .versioned fields version

def pruneEnv {Obj Err : Type} (merged : Obj) (managers : ManagedFields) (actor : String)
    (last : Option (FieldSet × String)) : Env Obj Err := [
  ("s", .updater), ("fmt", .formatter), ("merged", .object merged),
  ("managers", .managed managers), ("applyingManager", .text actor), ("lastSet", lastValue last)]

/-- The complete source orchestration, with both restoration components explicit. -/
def pruneSpec {Obj Err : Type} (ops : Ops Obj Err) (merged : Obj) (managers : ManagedFields)
    (actor : String) (last : Option (FieldSet × String)) (actorVersion : String) : List (Value Obj Err) :=
  match last with
  | none => [.object merged, .nil]
  | some (lastFields, version) =>
      if ops.empty lastFields then [.object merged, .nil]
      else match ops.convert merged version with
        | .error error =>
            if ops.missing error then [.object merged, .nil]
            else [.nil, .error (ops.wrap "failed to convert merged object to last applied version: %v" error)]
        | .ok converted =>
            let first := ops.remove converted (ops.ensureNamed converted converted lastFields)
            match ops.ownedItems converted first version managers actor with
            | .error error => [.nil, .error (ops.wrap "failed add back owned items: %v" error)]
            | .ok owned =>
                match ops.danglingItems converted owned lastFields version with
                | .error error => [.nil, .error (ops.wrap "failed add back dangling items: %v" error)]
                | .ok dangling => conversionOutcome ops dangling actorVersion

/-- Source execution short-circuits nil and empty last sets and propagates every phase's errors. -/
theorem imported_prune_matches_spec {Obj Err : Type} (ops : Ops Obj Err) (merged : Obj)
    (managers : ManagedFields) (actor : String) (last : Option (FieldSet × String))
    (entry : ManagerFields) (actorPresent : managerLookup managers actor = some entry) :
    exec ops 64 GoSource.pruneProgram (pruneEnv merged managers actor last) =
      .ok (.returned (pruneSpec ops merged managers actor last entry.version)) := by
  cases last with
  | none =>
      simp [GoSource.pruneProgram, exec, eval, evalAll, lookup, pruneEnv,
        lastValue, isNil, Bind.bind, Except.bind, pruneSpec]
  | some last =>
      rcases last with ⟨lastFields, version⟩
      cases empty : ops.empty lastFields with
      | true =>
          simp [GoSource.pruneProgram, exec, eval, evalAll, call, lookup, pruneEnv,
            lastValue, isNil, Bind.bind, Except.bind, pruneSpec, empty]
      | false =>
          cases conversion : ops.convert merged version with
          | error error =>
              cases missing : ops.missing error <;>
                simp [GoSource.pruneProgram, exec, eval, evalAll, call, bindNames, lookup, pruneEnv,
                  lastValue, isNil, Bind.bind, Except.bind, pruneSpec, conversionOutcome, empty, conversion, missing]
          | ok converted =>
              cases owned : ops.ownedItems converted
                (ops.remove converted (ops.ensureNamed converted converted lastFields)) version managers actor with
              | error error =>
                  simp [GoSource.pruneProgram, exec, eval, evalAll, call, bindNames, lookup, pruneEnv,
                    lastValue, isNil, Bind.bind, Except.bind, pruneSpec, conversionOutcome, outcome,
                    empty, conversion, owned]
              | ok ownedObject =>
                  cases dangling : ops.danglingItems converted ownedObject lastFields version with
                  | error error =>
                      simp [GoSource.pruneProgram, exec, eval, evalAll, call, bindNames, lookup, pruneEnv,
                        lastValue, isNil, Bind.bind, Except.bind, pruneSpec, conversionOutcome, outcome,
                        empty, conversion, owned, dangling]
                  | ok final =>
                      simp [GoSource.pruneProgram, exec, eval, evalAll, call, bindNames, lookup, pruneEnv,
                        lastValue, isNil, Bind.bind, Except.bind, pruneSpec, outcome, conversionOutcome, mapIndex,
                        empty, conversion, owned, dangling, actorPresent]

/-- Every manager's fields are included; source map grouping remains a separate refinement obligation. -/
def restoringFields (managers : ManagedFields) : FieldSet :=
  managers.flatMap (fun entry => entry.2.fields)

def modelOps (named : FieldSet) : Ops State Unit :=
  { normalizedOps named with
    ownedItems := fun merged pruned _ managers _ =>
      .ok (addBackOwnedFields named merged pruned (restoringFields managers))
    danglingItems := fun merged pruned fields _ => .ok (addBackDanglingFields named merged pruned fields) }

/-- The dangling callback is supplied by the already verified imported helper. -/
theorem model_dangling_callback_correct (named : FieldSet) (merged pruned : State)
    (fields : FieldSet) (version : String) :
    exec (modelOps named) 64 GoSource.danglingProgram (initialEnv merged pruned fields version) =
      .ok (.returned [.object (addBackDanglingFields named merged pruned fields), .nil]) := by
  rw [imported_dangling_matches_spec]
  rfl

/-- The owned-items callback's per-version operation is supplied by its imported helper. -/
theorem model_owned_callback_correct (named : FieldSet) (merged pruned : State)
    (managers : ManagedFields) (version : String) :
    exec (modelOps named) 64 GoSource.ownedVersionProgram
      (ownedVersionEnv merged pruned (restoringFields managers) version) =
      .ok (.returned [.object merged,
        .object (addBackOwnedFields named merged pruned (restoringFields managers)), .nil]) := by
  rw [imported_owned_version_matches_spec]
  rfl

/-- The imported complete pruning function computes the model's three passes for arbitrary states. -/
theorem source_prune_refines_three_passes (named : FieldSet) (merged : State)
    (managers : ManagedFields) (actor : String) (fields : FieldSet) (version : String)
    (entry : ManagerFields) (actorPresent : managerLookup managers actor = some entry) :
    exec (modelOps named) 64 GoSource.pruneProgram
      (pruneEnv merged managers actor (some (fields, version))) =
      .ok (.returned [.object (pruneWithFields named merged fields (restoringFields managers)), .nil]) := by
  rw [imported_prune_matches_spec (modelOps named) merged managers actor _ entry actorPresent]
  cases empty : fields.isEmpty <;>
    simp [pruneSpec, modelOps, normalizedOps, conversionOutcome, pruneWithFields,
      ensure_named_empty_iff, empty]

end SSA.PruneSource
