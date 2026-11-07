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
import SSA.PruneSource

namespace SSA.OwnedSource

open GoSubset

set_option maxRecDepth 10000
set_option maxHeartbeats 4000000

def FrameExcept {Obj Err : Type} (writes : List String) (before after : Env Obj Err) : Prop :=
  ∀ name, name ∉ writes → lookup after name = lookup before name

/-- Bodies are extracted from the imported AST, not copied from the Go source. -/
def groupingBody : List Stmt :=
  match (GoSource.ownedItemsProgram[2]? : Option GoSubset.Stmt) with
  | some (GoSubset.Stmt.rangeMap _ _ _ body) => body
  | _ => []

def restorationBody : List Stmt :=
  match (GoSource.ownedItemsProgram[5]? : Option GoSubset.Stmt) with
  | some (GoSubset.Stmt.rangeMap _ _ _ body) => body
  | _ => []

def priorityBody : List Stmt :=
  match (GoSource.ownedItemsProgram[4]? : Option GoSubset.Stmt) with
  | some (GoSubset.Stmt.branch _ body) => body
  | _ => []

def restorationTail : List Stmt := GoSource.ownedItemsProgram.drop 5

def afterGrouping : List Stmt := GoSource.ownedItemsProgram.drop 3

theorem version_lookup_set (entries : VersionFields) (version : String) (fields : FieldSet) :
    versionLookup (versionSet entries version fields) version = some fields := by
  simp [versionSet, versionLookup]

theorem version_set_overwrite (entries : VersionFields) (version : String) (old fields : FieldSet) :
    versionSet (versionSet entries version old) version fields = versionSet entries version fields := by
  simp [versionSet, versionDelete, List.filter_filter]

theorem iteration_empty (order : VersionIteration) : order.choose [] = [] := by
  apply List.eq_nil_of_length_eq_zero
  simpa using (order.permutation []).length_eq

def groupStep {Obj Err : Type} (ops : Ops Obj Err) (entries : VersionFields) (entry : ManagerFields) : VersionFields :=
  versionSet entries entry.version
    (ops.union ((versionLookup entries entry.version).getD ops.newSet) entry.fields)

/-- One complete imported grouping iteration updates exactly the selected API version. -/
theorem grouping_body_exec {Obj Err : Type} (ops : Ops Obj Err) (env : Env Obj Err)
    (entries : VersionFields) (entry : ManagerFields)
    (map : lookup env "managedAtVersion" = .versions entries)
    (manager : lookup env "managerSet$1" = .versioned entry.fields entry.version)
    (package : lookup env "fieldpath" = .fieldpathPackage) :
    ∃ next, exec ops 61 groupingBody env = .ok (.next next) ∧
      lookup next "managedAtVersion" = .versions (groupStep ops entries entry) ∧
      FrameExcept ["managedAtVersion", "ok$2"] env next := by
  cases found : versionLookup entries entry.version with
  | some fields =>
      let next : Env Obj Err := ("managedAtVersion", .versions
        (versionSet entries entry.version (ops.union fields entry.fields))) :: ("ok$2", .flag true) :: env
      refine ⟨next, ?_, ?_, ?_⟩
      · simp [groupingBody, GoSource.ownedItemsProgram, exec, eval, evalAll, call, bindNames,
          lookup, mapIndex, Bind.bind, Except.bind, next, map, manager, found]
      · simp [next, lookup, groupStep, found]
      · intro name absent
        have different : name ≠ "managedAtVersion" ∧ name ≠ "ok$2" := by simpa using absent
        simp [next, lookup, different.1, different.2]
  | none =>
      let intermediate := versionSet entries entry.version ops.newSet
      let next : Env Obj Err :=
        ("managedAtVersion", .versions (versionSet intermediate entry.version (ops.union ops.newSet entry.fields))) ::
        ("managedAtVersion", .versions intermediate) :: ("ok$2", .flag false) :: env
      refine ⟨next, ?_, ?_, ?_⟩
      · simp [groupingBody, GoSource.ownedItemsProgram, exec, eval, evalAll, call, bindNames,
          lookup, mapIndex, Bind.bind, Except.bind, next, intermediate, map, manager, found,
          package, version_lookup_set]
      · simp [next, lookup, intermediate, groupStep, found, version_set_overwrite]
      · intro name absent
        have different : name ≠ "managedAtVersion" ∧ name ≠ "ok$2" := by simpa using absent
        simp [next, lookup, different.1, different.2]

/-- The source grouping loop terminates for every finite number of managers. -/
theorem grouping_range_exec {Obj Err : Type} (ops : Ops Obj Err) (managers : ManagedFields)
    (env : Env Obj Err) (entries : VersionFields)
    (map : lookup env "managedAtVersion" = .versions entries)
    (package : lookup env "fieldpath" = .fieldpathPackage) :
    ∃ next, execRange ops 61 (managers.map fun row =>
        (Value.text row.1, Value.versioned row.2.fields row.2.version))
        "_" "managerSet$1" groupingBody env = .ok (.next next) ∧
      lookup next "managedAtVersion" =
        .versions (managers.foldl (fun fields row => groupStep ops fields row.2) entries) ∧
      FrameExcept ["managedAtVersion", "ok$2", "managerSet$1"] env next := by
  induction managers generalizing env entries with
  | nil =>
      exact ⟨env, by simp [execRange], map, by intro name _; rfl⟩
  | cons row rest ih =>
      let iteration : Env Obj Err := ("managerSet$1", .versioned row.2.fields row.2.version) :: env
      have step := grouping_body_exec ops iteration entries row.2
        (by simp [iteration, lookup, map]) (by simp [iteration, lookup])
        (by simp [iteration, lookup, package])
      obtain ⟨middle, step, grouped, frame⟩ := step
      have packageMiddle : lookup middle "fieldpath" = .fieldpathPackage := by
        rw [frame "fieldpath" (by simp)]
        simp [iteration, lookup, package]
      obtain ⟨next, remaining, result, remainingFrame⟩ := ih middle (groupStep ops entries row.2) grouped packageMiddle
      have bindings : bindNames ["_", "managerSet$1"]
          [Value.text row.1, Value.versioned row.2.fields row.2.version] env = .ok iteration := by
        simp [bindNames, iteration]
      refine ⟨next, ?_, result, ?_⟩
      · simpa only [List.map_cons, execRange, bindings, Bind.bind, Except.bind, step] using remaining
      · intro name absent
        have different : name ≠ "managedAtVersion" ∧ name ≠ "ok$2" ∧ name ≠ "managerSet$1" := by
          simpa using absent
        rw [remainingFrame name absent, frame name (by simp [different.1, different.2.1])]
        simp [iteration, lookup, different.2.2]

theorem object_option_roundtrip {Obj Err : Type} (object : Option Obj) :
    objectOption (optionalObject (Err := Err) object) = some object := by
  cases object <;> rfl

def RestoringContext {Obj Err : Type} (env : Env Obj Err) (state : Option Obj × Option Obj) : Prop :=
  lookup env "s" = .updater ∧ lookup env "nil" = .nil ∧
    lookup env "merged" = optionalObject state.1 ∧ lookup env "pruned" = optionalObject state.2

def restoreAll {Obj Err : Type} (ops : Ops Obj Err) : VersionFields →
    (Option Obj × Option Obj) → Except Err (Option Obj × Option Obj)
  | [], state => .ok state
  | (version, fields) :: rest, state => do
      let next ← ops.ownedVersionItems state.1 state.2 version fields
      restoreAll ops rest next

/-- One imported restoration iteration handles nullable intermediate values and all errors. -/
theorem restoration_body_exec {Obj Err : Type} (ops : Ops Obj Err) (env : Env Obj Err)
    (state : Option Obj × Option Obj) (version : String) (fields : FieldSet)
    (context : RestoringContext env state)
    (versionValue : lookup env "version$5" = .text version)
    (fieldValue : lookup env "managed$6" = .fields fields) :
    exec ops 58 restorationBody env =
      match ops.ownedVersionItems state.1 state.2 version fields with
      | .error error => .ok (.returned [.nil, .error error])
      | .ok next => .ok (.next (("err", .nil) ::
          ("pruned", optionalObject next.2) :: ("merged", optionalObject next.1) :: env)) := by
  rcases context with ⟨updater, nilValue, merged, pruned⟩
  cases operation : ops.ownedVersionItems state.1 state.2 version fields with
  | error error =>
      simp [restorationBody, GoSource.ownedItemsProgram, exec, eval, evalAll, call, bindNames,
        lookup, Bind.bind, Except.bind, pairOutcome, isNil, object_option_roundtrip,
        updater, nilValue, merged, pruned, versionValue, fieldValue, operation]
  | ok next =>
      rcases next with ⟨nextMerged, nextPruned⟩
      simp [restorationBody, GoSource.ownedItemsProgram, exec, eval, evalAll, call, bindNames,
        lookup, Bind.bind, Except.bind, pairOutcome, isNil, object_option_roundtrip,
        updater, merged, pruned, versionValue, fieldValue, operation]

/-- All finite version schedules execute exactly their recursive restoration contract. -/
theorem restoration_range_exec {Obj Err : Type} (ops : Ops Obj Err) (rows : VersionFields)
    (env : Env Obj Err) (state : Option Obj × Option Obj) (context : RestoringContext env state) :
    match restoreAll ops rows state with
    | .error error => execRange ops 58
        (rows.map fun row => (Value.text row.1, Value.fields row.2))
        "version$5" "managed$6" restorationBody env = .ok (.returned [.nil, .error error])
    | .ok final => ∃ next, execRange ops 58
        (rows.map fun row => (Value.text row.1, Value.fields row.2))
        "version$5" "managed$6" restorationBody env = .ok (.next next) ∧
        RestoringContext next final ∧
        FrameExcept ["merged", "pruned", "err", "version$5", "managed$6"] env next := by
  induction rows generalizing env state with
  | nil =>
      exact ⟨env, by simp [execRange], context, by intro name _; rfl⟩
  | cons row rest ih =>
      let iteration : Env Obj Err := ("managed$6", .fields row.2) :: ("version$5", .text row.1) :: env
      have iterationContext : RestoringContext iteration state := by
        rcases context with ⟨updater, nilValue, merged, pruned⟩
        simp [RestoringContext, iteration, lookup, updater, nilValue, merged, pruned]
      have body := restoration_body_exec ops iteration state row.1 row.2 iterationContext
        (by simp [iteration, lookup]) (by simp [iteration, lookup])
      have bindings : bindNames ["version$5", "managed$6"] [Value.text row.1, Value.fields row.2] env =
          .ok iteration := by simp [bindNames, iteration]
      cases operation : ops.ownedVersionItems state.1 state.2 row.1 row.2 with
      | error error =>
          simp only [operation] at body
          simp only [restoreAll, operation, Bind.bind, Except.bind]
          simp only [List.map_cons, execRange, bindings, Bind.bind, Except.bind, body]
      | ok pair =>
          let middle : Env Obj Err := ("err", .nil) :: ("pruned", optionalObject pair.2) ::
            ("merged", optionalObject pair.1) :: iteration
          have middleContext : RestoringContext middle pair := by
            rcases iterationContext with ⟨updater, nilValue, _, _⟩
            simp [RestoringContext, middle, lookup, updater, nilValue]
          have remainder := ih middle pair middleContext
          cases remaining : restoreAll ops rest pair with
          | error error =>
              simp only [remaining] at remainder
              simp only [restoreAll, operation, Bind.bind, Except.bind, remaining]
              simpa only [List.map_cons, execRange, bindings, Bind.bind, Except.bind,
                body, operation, middle] using remainder
          | ok final =>
              simp only [remaining] at remainder
              obtain ⟨next, result, nextContext, frame⟩ := remainder
              simp only [restoreAll, operation, Bind.bind, Except.bind, remaining]
              refine ⟨next, ?_, nextContext, ?_⟩
              · simpa only [List.map_cons, execRange, bindings, Bind.bind, Except.bind,
                  body, operation, middle] using result
              · intro name absent
                rw [frame name absent]
                have different : name ≠ "merged" ∧ name ≠ "pruned" ∧ name ≠ "err" ∧
                    name ≠ "version$5" ∧ name ≠ "managed$6" := by simpa using absent
                simp [middle, iteration, lookup, different.1, different.2.1, different.2.2.1,
                  different.2.2.2.1, different.2.2.2.2]

def restoredOutcome {Obj Err : Type} : Except Err (Option Obj × Option Obj) → List (Value Obj Err)
  | .error error => [.nil, .error error]
  | .ok state => [optionalObject state.2, .nil]

/-- The imported remaining-version loop and final return execute every lawful map schedule. -/
theorem restoration_tail_exec {Obj Err : Type} (ops : Ops Obj Err) (env : Env Obj Err)
    (entries : VersionFields) (state : Option Obj × Option Obj) (context : RestoringContext env state)
    (map : lookup env "managedAtVersion" = .versions entries) :
    exec ops 59 restorationTail env =
      .ok (.returned (restoredOutcome (restoreAll ops (ops.iteration.choose entries) state))) := by
  have split : exec ops 59 restorationTail env = (do
      let result ← execRange ops 58
        ((ops.iteration.choose entries).map fun row => (Value.text row.1, Value.fields row.2))
        "version$5" "managed$6" restorationBody env
      match result with
      | .returned values => .ok (.returned values)
      | .next next => .ok (.returned [lookup next "pruned", lookup next "nil"])) := by
    simp [restorationTail, restorationBody, GoSource.ownedItemsProgram, exec, eval, evalAll,
      mapEntries, Bind.bind, Except.bind, map] <;> first | rfl | (split <;> rfl)
  have execution := restoration_range_exec ops (ops.iteration.choose entries) env state context
  cases result : restoreAll ops (ops.iteration.choose entries) state with
  | error error =>
      simp only [result] at execution
      rw [split, execution]
      rfl
  | ok final =>
      simp only [result] at execution
      obtain ⟨next, execution, nextContext, _⟩ := execution
      rw [split, execution]
      simp [Bind.bind, Except.bind, restoredOutcome, nextContext.2.1, nextContext.2.2.2]

/-- The pruned version is restored first and removed before the remaining map is traversed. -/
theorem priority_body_exec {Obj Err : Type} (ops : Ops Obj Err) (env : Env Obj Err)
    (entries : VersionFields) (state : Option Obj × Option Obj) (version : String) (fields : FieldSet)
    (context : RestoringContext env state)
    (map : lookup env "managedAtVersion" = .versions entries)
    (versionValue : lookup env "prunedVersion" = .text version)
    (fieldValue : lookup env "managed$3" = .fields fields) :
    exec ops 59 priorityBody env =
      match ops.ownedVersionItems state.1 state.2 version fields with
      | .error error => .ok (.returned [.nil, .error error])
      | .ok next => .ok (.next (("managedAtVersion", .versions (versionDelete entries version)) ::
          ("err", .nil) :: ("pruned", optionalObject next.2) :: ("merged", optionalObject next.1) :: env)) := by
  rcases context with ⟨updater, nilValue, merged, pruned⟩
  cases operation : ops.ownedVersionItems state.1 state.2 version fields with
  | error error =>
      simp [priorityBody, GoSource.ownedItemsProgram, exec, eval, evalAll, call, bindNames,
        lookup, Bind.bind, Except.bind, pairOutcome, isNil, object_option_roundtrip,
        updater, nilValue, merged, pruned, versionValue, fieldValue, operation]
  | ok next =>
      rcases next with ⟨nextMerged, nextPruned⟩
      simp [priorityBody, GoSource.ownedItemsProgram, exec, eval, evalAll, call, bindNames,
        lookup, Bind.bind, Except.bind, pairOutcome, isNil, object_option_roundtrip,
        updater, merged, pruned, versionValue, fieldValue, operation, map]

def restoreGrouped {Obj Err : Type} (ops : Ops Obj Err) (entries : VersionFields)
    (state : Option Obj × Option Obj) (version : String) : Except Err (Option Obj × Option Obj) :=
  match versionLookup entries version with
  | none => restoreAll ops (ops.iteration.choose entries) state
  | some fields => do
      let next ← ops.ownedVersionItems state.1 state.2 version fields
      restoreAll ops (ops.iteration.choose (versionDelete entries version)) next

/-- All source statements after grouping refine priority restoration followed by the remaining fold. -/
theorem after_grouping_exec {Obj Err : Type} (ops : Ops Obj Err) (env : Env Obj Err)
    (entries : VersionFields) (state : Option Obj × Option Obj) (version : String)
    (context : RestoringContext env state)
    (map : lookup env "managedAtVersion" = .versions entries)
    (versionValue : lookup env "prunedVersion" = .text version) :
    exec ops 61 afterGrouping env = .ok (.returned (restoredOutcome (restoreGrouped ops entries state version))) := by
  cases found : versionLookup entries version with
  | none =>
      let next : Env Obj Err := ("ok$4", .flag false) :: ("managed$3", .nil) :: env
      have nextContext : RestoringContext next state := by
        rcases context with ⟨updater, nilValue, merged, pruned⟩
        simp [RestoringContext, next, lookup, updater, nilValue, merged, pruned]
      have tail := restoration_tail_exec ops next entries state nextContext (by simp [next, lookup, map])
      have decomposition : exec ops 61 afterGrouping env = exec ops 59 restorationTail next := by
        unfold afterGrouping
        simp only [GoSource.ownedItemsProgram, List.drop]
        rw [exec_assign ops 60]
        simp [evalAll, eval, mapIndex, map, versionValue, found, bindNames,
          Bind.bind, Except.bind]
        rw [exec_branch ops 59]
        simp [eval, lookup, Bind.bind, Except.bind, next, restorationTail, GoSource.ownedItemsProgram]
      rw [decomposition, tail]
      simp [restoreGrouped, found]
  | some fields =>
      let initial : Env Obj Err := ("ok$4", .flag true) :: ("managed$3", .fields fields) :: env
      have initialContext : RestoringContext initial state := by
        rcases context with ⟨updater, nilValue, merged, pruned⟩
        simp [RestoringContext, initial, lookup, updater, nilValue, merged, pruned]
      have priority := priority_body_exec ops initial entries state version fields initialContext
        (by simp [initial, lookup, map]) (by simp [initial, lookup, versionValue]) (by simp [initial, lookup])
      have decomposition : exec ops 61 afterGrouping env = (do
          match ← exec ops 59 priorityBody initial with
          | .returned values => .ok (.returned values)
          | .next next => exec ops 59 restorationTail next) := by
        unfold afterGrouping
        simp only [GoSource.ownedItemsProgram, List.drop]
        rw [exec_assign ops 60]
        simp [evalAll, eval, mapIndex, map, versionValue, found, bindNames,
          Bind.bind, Except.bind]
        rw [exec_branch ops 59]
        simp [eval, lookup, Bind.bind, Except.bind, initial, restorationTail, priorityBody,
          GoSource.ownedItemsProgram] <;> first | rfl | (split <;> rfl)
      cases operation : ops.ownedVersionItems state.1 state.2 version fields with
      | error error =>
          rw [decomposition, priority]
          simp [operation, restoreGrouped, found, Bind.bind, Except.bind, restoredOutcome]
      | ok pair =>
          let next : Env Obj Err := ("managedAtVersion", .versions (versionDelete entries version)) ::
            ("err", .nil) :: ("pruned", optionalObject pair.2) :: ("merged", optionalObject pair.1) :: initial
          have nextContext : RestoringContext next pair := by
            rcases initialContext with ⟨updater, nilValue, _, _⟩
            simp [RestoringContext, next, lookup, updater, nilValue]
          have tail := restoration_tail_exec ops next (versionDelete entries version) pair nextContext
            (by simp [next, lookup])
          rw [decomposition, priority]
          simpa [operation, restoreGrouped, found, Bind.bind, Except.bind, next] using tail

def ownedEnv {Obj Err : Type} (merged pruned : Obj) (version : String)
    (managers : ManagedFields) (actor : String) : Env Obj Err := [
  ("s", .updater), ("fmt", .formatter), ("fieldpath", .fieldpathPackage),
  ("merged", .object merged), ("pruned", .object pruned), ("prunedVersion", .text version),
  ("managedFields", .managed managers), ("applyingManager", .text actor)]

def groupedFields {Obj Err : Type} (ops : Ops Obj Err) (managers : ManagedFields) : VersionFields :=
  managers.foldl (fun fields row => groupStep ops fields row.2) []

def ownedSpec {Obj Err : Type} (ops : Ops Obj Err) (merged pruned : Obj)
    (version : String) (managers : ManagedFields) : List (Value Obj Err) :=
  restoredOutcome (restoreGrouped ops (groupedFields ops managers) (some merged, some pruned) version)

/-- The entire imported helper refines grouping, priority restoration and any lawful remaining order. -/
theorem imported_owned_items_matches_spec {Obj Err : Type} (ops : Ops Obj Err)
    (merged pruned : Obj) (version : String) (managers : ManagedFields) (actor : String) :
    exec ops 64 GoSource.ownedItemsProgram (ownedEnv merged pruned version managers actor) =
      .ok (.returned (ownedSpec ops merged pruned version managers)) := by
  let setup : Env Obj Err := ("managedAtVersion", .versions []) :: ("err", .nil) ::
    ownedEnv merged pruned version managers actor
  have group := grouping_range_exec ops managers setup []
    (by simp [setup, lookup]) (by simp [setup, ownedEnv, lookup])
  obtain ⟨after, grouping, result, frame⟩ := group
  have afterContext : RestoringContext after (some merged, some pruned) := by
    refine ⟨?_, ?_, ?_, ?_⟩
    all_goals rw [frame _ (by simp)]
    all_goals simp [setup, ownedEnv, lookup, optionalObject]
  have afterVersion : lookup after "prunedVersion" = .text version := by
    rw [frame _ (by simp)]
    simp [setup, ownedEnv, lookup]
  have decomposition :
      exec ops 64 GoSource.ownedItemsProgram (ownedEnv merged pruned version managers actor) = (do
        match ← execRange ops 61 (managers.map fun row =>
            (Value.text row.1, Value.versioned row.2.fields row.2.version))
            "_" "managerSet$1" groupingBody setup with
        | .returned values => .ok (.returned values)
        | .next next => exec ops 61 afterGrouping next) := by
    simp only [GoSource.ownedItemsProgram]
    rw [exec_declare_error ops 63, exec_assign ops 62]
    simp [evalAll, eval, bindNames, Bind.bind, Except.bind]
    rw [exec_range ops 61]
    simp [eval, lookup, ownedEnv, mapEntries, Bind.bind, Except.bind, setup, groupingBody,
      afterGrouping, GoSource.ownedItemsProgram] <;> first | rfl | (split <;> rfl)
  rw [decomposition, grouping]
  simp only [Bind.bind, Except.bind]
  exact after_grouping_exec ops after _ _ version afterContext result afterVersion

end SSA.OwnedSource
