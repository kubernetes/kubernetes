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
import SSA.GoSource

namespace SSA.SourceContracts

open GoSubset

set_option maxRecDepth 10000
set_option maxHeartbeats 4000000

def initialEnv {Obj Err : Type} (merged pruned : Obj) (last : FieldSet)
    (version : String) : Env Obj Err := [
  ("s", .updater), ("fmt", .formatter), ("merged", .object merged),
  ("pruned", .object pruned), ("lastSet", .versioned last version)]

/-- An independent outcome specification includes every Go error and missing-version branch. -/
def danglingSpec {Obj Err : Type} (ops : Ops Obj Err) (merged pruned : Obj)
    (last : FieldSet) (version : String) : Except Err Obj :=
  match ops.convert pruned version with
  | .error error =>
      if ops.missing error then .ok merged
      else .error (ops.wrap "failed to convert pruned object to last applied version: %v" error)
  | .ok converted => do
      let prunedFields ← match ops.toFields converted with
        | .ok fields => .ok fields
        | .error error => .error (ops.wrap "failed to create field set from pruned object in last applied version: %v" error)
      let mergedFields ← match ops.toFields merged with
        | .ok fields => .ok fields
        | .error error => .error (ops.wrap "failed to create field set from merged object in last applied version: %v" error)
      let named := ops.ensureNamed merged merged
      .ok (ops.remove merged (ops.intersection
        (ops.difference (named mergedFields) (named prunedFields)) (named last)))

/-- The theorem executes the imported source AST, with conversion and typed primitives explicit. -/
theorem imported_dangling_matches_spec {Obj Err : Type} (ops : Ops Obj Err)
    (merged pruned : Obj) (last : FieldSet) (version : String) :
    exec ops 64 GoSource.danglingProgram (initialEnv merged pruned last version) =
      .ok (.returned (outcome (danglingSpec ops merged pruned last version))) := by
  cases conversion : ops.convert pruned version with
  | error error =>
      cases missing : ops.missing error <;>
        simp [GoSource.danglingProgram, exec, eval, evalAll, call, bindNames, lookup, initialEnv, Bind.bind, Except.bind,
          outcome, conversionOutcome, isNil, danglingSpec, conversion, missing]
  | ok converted =>
      cases prunedFields : ops.toFields converted with
      | error error =>
          simp [GoSource.danglingProgram, exec, eval, evalAll, call, bindNames, lookup, initialEnv, Bind.bind, Except.bind,
            outcome, conversionOutcome, fieldOutcome, isNil, danglingSpec, conversion, prunedFields]
      | ok prunedFields =>
          cases mergedFields : ops.toFields merged with
          | error error =>
              simp [GoSource.danglingProgram, exec, eval, evalAll, call, bindNames, lookup, initialEnv, Bind.bind, Except.bind,
                outcome, conversionOutcome, fieldOutcome, isNil, danglingSpec, conversion, prunedFields, mergedFields]
          | ok mergedFields =>
              simp [GoSource.danglingProgram, exec, eval, evalAll, call, bindNames, lookup, initialEnv, Bind.bind, Except.bind,
                outcome, conversionOutcome, fieldOutcome, isNil, danglingSpec, conversion, prunedFields, mergedFields]

def ownedVersionEnv {Obj Err : Type} (merged pruned : Obj) (managed : FieldSet)
    (version : String) : Env Obj Err := [
  ("s", .updater), ("fmt", .formatter), ("merged", .object merged),
  ("pruned", .object pruned), ("version", .text version), ("managed", .fields managed)]

/-- Missing-version exits preserve the actual object/error pair returned by conversion. -/
def ownedVersionSpec {Obj Err : Type} (ops : Ops Obj Err) (merged pruned : Obj)
    (managed : FieldSet) (version : String) : Except Err (Option Obj × Option Obj) :=
  match ops.convert merged version with
  | .error error =>
      if ops.missing error then .ok (ops.failedConversion merged version error, some pruned)
      else .error (ops.wrapAt "failed to convert merged object at version %v: %v" version error)
  | .ok convertedMerged =>
      match ops.convert pruned version with
      | .error error =>
          if ops.missing error then .ok (some convertedMerged, ops.failedConversion pruned version error)
          else .error (ops.wrapAt "failed to convert pruned object at version %v: %v" version error)
      | .ok convertedPruned => do
          let mergedFields ← match ops.toFields convertedMerged with
            | .ok fields => .ok fields
            | .error error => .error (ops.wrapAt "failed to create field set from merged object at version %v: %v" version error)
          let prunedFields ← match ops.toFields convertedPruned with
            | .ok fields => .ok fields
            | .error error => .error (ops.wrapAt "failed to create field set from pruned object at version %v: %v" version error)
          let named := ops.ensureNamed convertedMerged convertedMerged
          .ok (some convertedMerged, some (ops.remove convertedMerged (ops.difference
            (named mergedFields) (ops.union (named prunedFields) (named managed)))))

/-- Every source branch of addBackOwnedItemsForVersion satisfies the independent contract. -/
theorem imported_owned_version_matches_spec {Obj Err : Type} (ops : Ops Obj Err)
    (merged pruned : Obj) (managed : FieldSet) (version : String) :
    exec ops 64 GoSource.ownedVersionProgram (ownedVersionEnv merged pruned managed version) =
      .ok (.returned (pairOutcome (ownedVersionSpec ops merged pruned managed version))) := by
  cases first : ops.convert merged version with
  | error error =>
      cases missing : ops.missing error <;>
        simp [GoSource.ownedVersionProgram, exec, eval, evalAll, call, bindNames, lookup,
          ownedVersionEnv, Bind.bind, Except.bind, conversionOutcome, pairOutcome,
          ownedVersionSpec, isNil, first, missing, optionalObject]
  | ok convertedMerged =>
      cases second : ops.convert pruned version with
      | error error =>
          cases missing : ops.missing error <;>
            simp [GoSource.ownedVersionProgram, exec, eval, evalAll, call, bindNames, lookup,
              ownedVersionEnv, Bind.bind, Except.bind, conversionOutcome, pairOutcome,
              ownedVersionSpec, isNil, first, second, missing, optionalObject]
      | ok convertedPruned =>
          cases mergedFields : ops.toFields convertedMerged with
          | error error =>
              simp [GoSource.ownedVersionProgram, exec, eval, evalAll, call, bindNames, lookup,
                ownedVersionEnv, Bind.bind, Except.bind, conversionOutcome, pairOutcome, fieldOutcome,
                ownedVersionSpec, isNil, first, second, mergedFields]
          | ok mergedFields =>
              cases prunedFields : ops.toFields convertedPruned with
              | error error =>
                  simp [GoSource.ownedVersionProgram, exec, eval, evalAll, call, bindNames, lookup,
                    ownedVersionEnv, Bind.bind, Except.bind, conversionOutcome, pairOutcome, fieldOutcome,
                    ownedVersionSpec, isNil, first, second, mergedFields, prunedFields]
              | ok prunedFields =>
                  simp [GoSource.ownedVersionProgram, exec, eval, evalAll, call, bindNames, lookup,
                    ownedVersionEnv, Bind.bind, Except.bind, conversionOutcome, pairOutcome, fieldOutcome,
                    ownedVersionSpec, isNil, first, second, mergedFields, prunedFields, optionalObject]

/-- Single-version model primitives instantiate the imported source without conversion errors. -/
def normalizedOps (named : FieldSet) : Ops State Unit where
  convert object _ := .ok object
  missing _ := false
  toFields object := .ok (dataFields named object)
  ensureNamed _ _ fields := ensureNamed named fields
  remove := removeItems
  wrap _ error := error

/-- Actual imported Go control flow refines the model's final pruning expression for all states. -/
theorem source_dangling_refines_model (named : FieldSet) (merged pruned : State)
    (last : FieldSet) (version : String) :
    exec (normalizedOps named) 64 GoSource.danglingProgram (initialEnv merged pruned last version) =
      .ok (.returned [.object (removeItems merged (fieldIntersection
        (fieldDifference (ensureNamed named (dataFields named merged))
          (ensureNamed named (dataFields named pruned))) (ensureNamed named last))), .nil]) := by
  rw [imported_dangling_matches_spec]
  rfl

/-- The imported owned-items pass computes the normalized restoration expression at any depth. -/
theorem source_owned_version_refines_model (named : FieldSet) (merged pruned : State)
    (managed : FieldSet) (version : String) :
    exec (normalizedOps named) 64 GoSource.ownedVersionProgram (ownedVersionEnv merged pruned managed version) =
      .ok (.returned [.object merged, .object (removeItems merged
        (fieldDifference (ensureNamed named (dataFields named merged))
          (ensureNamed named (dataFields named pruned) ++ ensureNamed named managed))), .nil]) := by
  rw [imported_owned_version_matches_spec]
  rfl

end SSA.SourceContracts
