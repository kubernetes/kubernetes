/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.PruneSource
import SSA.ResetFields

namespace SSA.ApplySource

open GoSubset

set_option maxRecDepth 10000
set_option maxHeartbeats 4000000

def applyEnv {Obj Err : Type} (live config : Obj) (version : String)
    (managers : ManagedFields) (actor : String) (force : Bool) : Env Obj Err := [
  ("s",.updater),("fmt",.formatter),("fieldpath",.fieldpathPackage),("value",.valuePackage),
  ("liveObject",.object live),("configObject",.object config),("version",.text version),
  ("managers",.managed managers),("manager",.text actor),("force",.flag force)]

def rejected {Obj Err : Type} (error : Err) : List (Value Obj Err) :=
  [.nil,.managed [],.error error]

/-- Filter selection distinguishes nil maps, missing version entries, and an allocated empty set. -/
def filteredFields {Obj Err : Type} (ops : Ops Obj Err) (version : String)
    (fields : FieldSet) : Except String FieldSet :=
  match ops.ignoredFields with
  | some excluded =>
      match versionLookup excluded version with
      | some excluded => .ok (ops.excludeFields fields excluded)
      | none => .error "Go exclude filter dereferences a nil set"
  | none =>
      match ops.ignoreFilters with
      | none => .ok fields
      | some filters =>
          match filterLookup filters version with
          | none => .ok fields
          | some filter => .ok (filter fields)

def oldVersion (managers : ManagedFields) (actor : String) : Option (FieldSet × String) :=
  (managerLookup managers actor).map (fun entry => (entry.fields,entry.version))

/-- The ownership-update callback runs on the pruned object; no-op detection preserves its returned managers. -/
def afterPruneSpec {Obj Err : Type} (ops : Ops Obj Err) (live pruned : Obj) (version : String)
    (owned : ManagedFields) (actor : String) (force : Bool) : List (Value Obj Err) :=
  match ops.finishUpdate live pruned version owned actor force with
  | .error error => rejected error
  | .ok updated =>
      let object := if !ops.returnInputOnNoop && ops.equalObjects live pruned then .nil else .object pruned
      [object,.managed updated,.nil]

def afterFilterSpec {Obj Err : Type} (ops : Ops Obj Err) (live merged : Obj) (version : String)
    (owned : ManagedFields) (actor : String) (last : Option (FieldSet × String))
    (force : Bool) : List (Value Obj Err) :=
  match ops.pruneObject merged owned actor last with
  | .error error => rejected (ops.wrap "failed to prune fields: %v" error)
  | .ok pruned => afterPruneSpec ops live pruned version owned actor force

def afterFieldsSpec {Obj Err : Type} (ops : Ops Obj Err) (live merged : Obj) (version : String)
    (reconciled : ManagedFields) (actor : String) (fields : FieldSet)
    (force : Bool) : Except String (List (Value Obj Err)) :=
  if ops.ignoredFields.isSome && ops.ignoreFilters.isSome then
    .ok [.nil,.nil,.literalError "IgnoreFilter and IgnoreFilter may not both be set"]
  else do
    let selected ← filteredFields ops version fields
    let owned := managerSet reconciled actor ⟨selected,version,true⟩
    .ok (afterFilterSpec ops live merged version owned actor (oldVersion reconciled actor) force)

/-- Independent outcomes cover reconciliation, merge, extraction, filtering, pruning, update and no-op return. -/
def applySpec {Obj Err : Type} (ops : Ops Obj Err) (live config : Obj) (version : String)
    (managers : ManagedFields) (actor : String) (force : Bool) : Except String (List (Value Obj Err)) :=
  match ops.reconcileFields live managers with
  | .error error => .ok (rejected error)
  | .ok reconciled =>
      match ops.mergeObjects live config with
      | .error error => .ok (rejected (ops.wrap "failed to merge config: %v" error))
      | .ok merged =>
          match ops.toFields config with
          | .error error => .ok (rejected (ops.wrap "failed to get field set: %v" error))
          | .ok fields => afterFieldsSpec ops live merged version reconciled actor fields force

-- Opaque phase boundaries keep simplification from expanding every callback/error combination.
@[irreducible] private def afterFieldsProgram := GoSource.applyProgram.drop 8
@[irreducible] private def afterFilterProgram := GoSource.applyProgram.drop 13

private theorem split_program :
    GoSource.applyProgram = GoSource.applyProgram.take 8 ++ afterFieldsProgram := by
  unfold afterFieldsProgram
  rfl

private theorem split_fields_program :
    afterFieldsProgram = (GoSource.applyProgram.drop 8).take 5 ++ afterFilterProgram := by
  unfold afterFieldsProgram afterFilterProgram
  rfl

private theorem source_after_filter {Obj Err : Type} (ops : Ops Obj Err) (live merged : Obj)
    (version : String) (owned : ManagedFields) (actor : String) (last : Option (FieldSet × String))
    (force : Bool) (env : Env Obj Err)
    (s : lookup env "s" = .updater) (fmt : lookup env "fmt" = .formatter)
    (value : lookup env "value" = .valuePackage)
    (nil : lookup env "nil" = .nil)
    (object : lookup env "newObject" = .object merged)
    (managers : lookup env "managers" = .managed owned)
    (manager : lookup env "manager" = .text actor)
    (lastSet : lookup env "lastSet" = PruneSource.lastValue last)
    (liveObject : lookup env "liveObject" = .object live)
    (versionValue : lookup env "version" = .text version)
    (forceValue : lookup env "force" = .flag force) :
    exec ops 115 afterFilterProgram env =
      .ok (.returned (afterFilterSpec ops live merged version owned actor last force)) := by
  cases pruning : ops.pruneObject merged owned actor last with
  | error error =>
      cases last <;>
        simp_all [afterFilterProgram,GoSource.applyProgram,afterFilterSpec,exec,
          eval,evalAll,call,lookup,bindNames,isNil,outcome,lastFields,
          PruneSource.lastValue,rejected,Bind.bind,Except.bind]
  | ok pruned =>
      cases updating : ops.finishUpdate live pruned version owned actor force with
      | error error =>
          cases last <;>
            simp_all [afterFilterProgram,GoSource.applyProgram,afterFilterSpec,afterPruneSpec,exec,
              eval,evalAll,call,lookup,bindNames,isNil,outcome,updateOutcome,lastFields,
              PruneSource.lastValue,rejected,Bind.bind,Except.bind]
      | ok updated =>
          cases last <;> cases input : ops.returnInputOnNoop <;>
            cases noop : ops.equalObjects live pruned <;>
            simp_all [afterFilterProgram,GoSource.applyProgram,afterFilterSpec,afterPruneSpec,exec,
              eval,evalAll,call,lookup,bindNames,isNil,outcome,updateOutcome,lastFields,
              PruneSource.lastValue,Bind.bind,Except.bind]

private def fieldsEnv {Obj Err : Type} (live config merged : Obj) (version : String)
    (original reconciled : ManagedFields) (actor : String) (fields : FieldSet) (force : Bool) : Env Obj Err :=
  [("err",.nil),("set",.fields fields),
    ("lastSet",PruneSource.lastValue (oldVersion reconciled actor)),
    ("err",.nil),("newObject",.object merged),("err",.nil),("managers",.managed reconciled),
    ("err",.nil)] ++
    applyEnv live config version original actor force

private theorem source_after_fields {Obj Err : Type} (ops : Ops Obj Err) (live config merged : Obj)
    (version : String) (original reconciled : ManagedFields) (actor : String)
    (fields : FieldSet) (force : Bool) :
    exec ops 120 afterFieldsProgram (fieldsEnv live config merged version original reconciled actor fields force) =
      (afterFieldsSpec ops live merged version reconciled actor fields force).map Result.returned := by
  rw [split_fields_program]
  cases ignored : ops.ignoredFields with
  | none =>
      cases filters : ops.ignoreFilters with
      | none =>
          simp [GoSource.applyProgram,exec,eval,evalAll,call,lookup,
            bindNames,isNil,mapIndex,fieldsEnv,applyEnv,ignored,filters,afterFieldsSpec,filteredFields,
            Bind.bind,Except.bind,Option.isSome]
          apply source_after_filter <;> simp [lookup]
      | some table =>
          cases selected : filterLookup table version <;>
            simp [GoSource.applyProgram,exec,eval,evalAll,call,lookup,
              bindNames,isNil,mapIndex,fieldsEnv,applyEnv,ignored,filters,selected,afterFieldsSpec,filteredFields,
              Bind.bind,Except.bind,Option.isSome]
          all_goals apply source_after_filter <;> simp [lookup]
  | some excluded =>
      cases filters : ops.ignoreFilters with
      | some table =>
          simp [GoSource.applyProgram,exec,eval,evalAll,call,lookup,isNil,fieldsEnv,applyEnv,
            ignored,filters,afterFieldsSpec,Bind.bind,Except.bind,Except.map]
      | none =>
          cases selected : versionLookup excluded version <;>
            simp [GoSource.applyProgram,exec,eval,evalAll,call,lookup,
              bindNames,isNil,mapIndex,fieldsEnv,applyEnv,ignored,filters,selected,afterFieldsSpec,filteredFields,
              Bind.bind,Except.bind,Option.isSome,Bool.and_false]
          all_goals first | rfl | apply source_after_filter
          all_goals simp [lookup]

/-- Complete imported Apply control flow agrees with the specification for arbitrary primitive outcomes. -/
theorem imported_apply_matches_spec {Obj Err : Type} (ops : Ops Obj Err) (live config : Obj)
    (version : String) (managers : ManagedFields) (actor : String) (force : Bool) :
    exec ops 128 GoSource.applyProgram (applyEnv live config version managers actor force) =
      (applySpec ops live config version managers actor force).map Result.returned := by
  rw [split_program]
  cases reconciliation : ops.reconcileFields live managers with
  | error error =>
      simp [GoSource.applyProgram,exec,eval,evalAll,call,lookup,bindNames,isNil,applyEnv,
        managedOutcome,applySpec,rejected,reconciliation,Bind.bind,Except.bind,Except.map]
  | ok reconciled =>
      cases merging : ops.mergeObjects live config with
      | error error =>
          simp [GoSource.applyProgram,exec,eval,evalAll,call,lookup,bindNames,isNil,applyEnv,
            managedOutcome,outcome,applySpec,rejected,reconciliation,merging,Bind.bind,Except.bind,Except.map]
      | ok merged =>
          cases last : managerLookup reconciled actor
          all_goals cases extraction : ops.toFields config with
          | error error =>
              simp [GoSource.applyProgram,exec,eval,evalAll,call,lookup,bindNames,isNil,applyEnv,
                managedOutcome,outcome,fieldOutcome,applySpec,rejected,mapIndex,
                reconciliation,merging,extraction,last,Bind.bind,Except.bind,Except.map]
          | ok fields =>
              have tail := source_after_fields ops live config merged version managers reconciled actor fields force
              simpa [GoSource.applyProgram,exec,eval,evalAll,call,lookup,bindNames,isNil,applyEnv,
                  fieldsEnv,managedOutcome,outcome,fieldOutcome,applySpec,rejected,mapIndex,
                  oldVersion,PruneSource.lastValue,last,reconciliation,merging,extraction,Bind.bind,Except.bind,Except.map] using tail

/-- A returned source result is exactly the independent specification result, including nil on no-op. -/
theorem imported_apply_success_iff {Obj Err : Type} (ops : Ops Obj Err) (live config : Obj)
    (version : String) (managers : ManagedFields) (actor : String) (force : Bool)
    (result : List (Value Obj Err)) :
    exec ops 128 GoSource.applyProgram (applyEnv live config version managers actor force) =
      .ok (.returned result) ↔ applySpec ops live config version managers actor force = .ok result := by
  rw [imported_apply_matches_spec]
  cases outcome : applySpec ops live config version managers actor force <;> simp [Except.map]

/-- The isolated update obligation allows deletion/transfer but no new Actor fields. -/
def UpdateDoesNotGrowActor {Obj Err : Type} (ops : Ops Obj Err) : Prop :=
  ∀ live object version before actor force after,
    ops.finishUpdate live object version before actor force = .ok after →
    ResetFields.ActorDoesNotGrow before after actor

/-- For configured exclusion roots, source Apply's final Actor claims remain filtered,
    provided the remaining updater callback satisfies its explicit non-growth contract. -/
theorem imported_apply_excludes_reset_fields {Obj Err : Type} (ops : Ops Obj Err)
    (live config : Obj) (version : String) (managers after : ManagedFields) (actor : String)
    (force : Bool) (object : Value Obj Err) (roots : FieldSet)
    (table : List (String × (FieldSet → FieldSet)))
    (modern : ops.ignoredFields = none) (configured : ops.ignoreFilters = some table)
    (selected : filterLookup table version = some (ResetFields.filter roots))
    (updateFrame : UpdateDoesNotGrowActor ops)
    (accepted : exec ops 128 GoSource.applyProgram (applyEnv live config version managers actor force) =
      .ok (.returned [object,.managed after,.nil]))
    (entry : ManagerFields) (owned : (actor,entry) ∈ after)
    (root path : Path) (reset : root ∈ roots) (descendant : under root path = true) :
    path ∉ entry.fields := by
  have result := (imported_apply_success_iff ops live config version managers actor force _).mp accepted
  cases reconciliation : ops.reconcileFields live managers with
  | error error => simp [applySpec,reconciliation,rejected] at result
  | ok reconciled =>
      cases merging : ops.mergeObjects live config with
      | error error => simp [applySpec,reconciliation,merging,rejected] at result
      | ok merged =>
          cases extraction : ops.toFields config with
          | error error => simp [applySpec,reconciliation,merging,extraction,rejected] at result
          | ok fields =>
              let installed := managerSet reconciled actor ⟨ResetFields.filter roots fields,version,true⟩
              have filtered : afterFilterSpec ops live merged version installed actor
                  (oldVersion reconciled actor) force = [object,.managed after,.nil] := by
                simpa [applySpec,reconciliation,merging,extraction,afterFieldsSpec,filteredFields,
                  modern,configured,selected,installed,Bind.bind,Except.bind] using result
              cases pruning : ops.pruneObject merged installed actor (oldVersion reconciled actor) with
              | error error => simp [afterFilterSpec,pruning,rejected] at filtered
              | ok pruned =>
                  cases updating : ops.finishUpdate live pruned version installed actor force with
                  | error error => simp [afterFilterSpec,pruning,afterPruneSpec,updating,rejected] at filtered
                  | ok updated =>
                      have same : updated = after := by
                        have outputs : [
                            (if !ops.returnInputOnNoop && ops.equalObjects live pruned
                              then .nil else .object pruned),.managed updated,.nil] =
                            [object,.managed after,.nil] := by
                          simpa only [afterFilterSpec,pruning,afterPruneSpec,updating] using filtered
                        exact Value.managed.inj (List.cons.inj (List.cons.inj outputs).2).1
                      subst updated
                      exact ResetFields.update_preserves_reset_exclusion roots fields reconciled after actor version
                        (updateFrame live pruned version installed actor force after updating)
                        entry owned root path reset descendant

end SSA.ApplySource
