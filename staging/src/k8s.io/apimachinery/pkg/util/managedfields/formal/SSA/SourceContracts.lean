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
      .ok (ops.remove merged (fieldIntersection
        (fieldDifference (named mergedFields) (named prunedFields)) (named last)))

/-- The theorem executes the imported source AST, with conversion and typed primitives explicit. -/
theorem imported_dangling_matches_spec {Obj Err : Type} (ops : Ops Obj Err)
    (merged pruned : Obj) (last : FieldSet) (version : String) :
    exec ops 64 GoSource.danglingProgram (initialEnv merged pruned last version) =
      .ok (.returned (outcome (danglingSpec ops merged pruned last version))) := by
  cases conversion : ops.convert pruned version with
  | error error =>
      cases missing : ops.missing error <;>
        simp [GoSource.danglingProgram, exec, eval, evalAll, call, bindNames, lookup, initialEnv, Bind.bind, Except.bind,
          outcome, danglingSpec, conversion, missing]
  | ok converted =>
      cases prunedFields : ops.toFields converted with
      | error error =>
          simp [GoSource.danglingProgram, exec, eval, evalAll, call, bindNames, lookup, initialEnv, Bind.bind, Except.bind,
            outcome, fieldOutcome, danglingSpec, conversion, prunedFields]
      | ok prunedFields =>
          cases mergedFields : ops.toFields merged with
          | error error =>
              simp [GoSource.danglingProgram, exec, eval, evalAll, call, bindNames, lookup, initialEnv, Bind.bind, Except.bind,
                outcome, fieldOutcome, danglingSpec, conversion, prunedFields, mergedFields]
          | ok mergedFields =>
              simp [GoSource.danglingProgram, exec, eval, evalAll, call, bindNames, lookup, initialEnv, Bind.bind, Except.bind,
                outcome, fieldOutcome, danglingSpec, conversion, prunedFields, mergedFields]

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

end SSA.SourceContracts
