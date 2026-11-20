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
import SSA.OwnedSource

namespace SSA.PruningRefinement

open GoSubset SourceContracts OwnedSource

set_option maxRecDepth 10000
set_option maxHeartbeats 4000000

/-- The callback has the outcome proved for the imported per-version helper. -/
def linkedOps (named : FieldSet) (order : VersionIteration) : Ops State Unit :=
  { PruneSource.modelOps named with
    iteration := order
    ownedVersionItems := fun merged pruned version fields =>
      match merged, pruned with
      | some merged, some pruned => ownedVersionSpec (normalizedOps named) merged pruned fields version
      | _, _ => .error () }

theorem linked_version_callback_correct (named : FieldSet) (order : VersionIteration)
    (merged pruned : State) (fields : FieldSet) (version : String) :
    exec (linkedOps named order) 64 GoSource.ownedVersionProgram
        (ownedVersionEnv merged pruned fields version) =
      .ok (.returned (pairOutcome ((linkedOps named order).ownedVersionItems
        (some merged) (some pruned) version fields))) := by
  rw [imported_owned_version_matches_spec]
  rfl

theorem group_same_version_from (named : FieldSet) (order : VersionIteration) (version : String)
    (managers : ManagedFields) (accumulated : FieldSet)
    (same : ∀ row ∈ managers, row.2.version = version) :
    managers.foldl (fun fields row => groupStep (linkedOps named order) fields row.2)
        [(version, accumulated)] =
      [(version, accumulated ++ PruneSource.restoringFields managers)] := by
  induction managers generalizing accumulated with
  | nil => simp [PruneSource.restoringFields]
  | cons row rest ih =>
      have headVersion := same row (by simp)
      have tailVersions : ∀ entry ∈ rest, entry.2.version = version := by
        intro entry member
        exact same entry (by simp [member])
      have step : groupStep (linkedOps named order) [(version, accumulated)] row.2 =
          [(version, accumulated ++ row.2.fields)] := by
        simp [groupStep, linkedOps, PruneSource.modelOps, normalizedOps, headVersion,
          versionLookup, versionSet, versionDelete]
      simp only [List.foldl_cons]
      rw [step, ih (accumulated ++ row.2.fields) tailVersions]
      simp [PruneSource.restoringFields, List.append_assoc]

/-- The source grouping loop collects all fields, even with arbitrarily many empty manager sets. -/
theorem group_single_version (named : FieldSet) (order : VersionIteration) (version : String)
    (managers : ManagedFields) (nonempty : managers ≠ [])
    (same : ∀ row ∈ managers, row.2.version = version) :
    groupedFields (linkedOps named order) managers = [(version, PruneSource.restoringFields managers)] := by
  cases managers with
  | nil => exact (nonempty rfl).elim
  | cons row rest =>
      have headVersion := same row (by simp)
      have tailVersions : ∀ entry ∈ rest, entry.2.version = version := by
        intro entry member
        exact same entry (by simp [member])
      have first : groupStep (linkedOps named order) [] row.2 = [(version, row.2.fields)] := by
        simp [groupStep, linkedOps, PruneSource.modelOps, normalizedOps, headVersion,
          versionLookup, versionSet, versionDelete]
      simp only [groupedFields, List.foldl_cons]
      rw [first, group_same_version_from named order version rest row.2.fields tailVersions]
      rfl

/-- With one API version, the actual imported map wrapper agrees with the model callback. -/
theorem linked_owned_callback_correct (named : FieldSet) (order : VersionIteration)
    (merged pruned : State) (version : String) (managers : ManagedFields) (actor : String)
    (nonempty : managers ≠ []) (same : ∀ row ∈ managers, row.2.version = version) :
    exec (linkedOps named order) 64 GoSource.ownedItemsProgram
        (ownedEnv merged pruned version managers actor) =
      .ok (.returned (outcome ((linkedOps named order).ownedItems merged pruned version managers actor))) := by
  rw [imported_owned_items_matches_spec]
  simp only [ownedSpec]
  rw [group_single_version named order version managers nonempty same]
  simp [restoreGrouped, versionLookup, linkedOps, PruneSource.modelOps, normalizedOps,
    ownedVersionSpec, Bind.bind, Except.bind, versionDelete, iteration_empty,
    restoreAll, restoredOutcome, outcome, optionalObject, addBackOwnedFields]

theorem linked_dangling_callback_correct (named : FieldSet) (order : VersionIteration)
    (merged pruned : State) (fields : FieldSet) (version : String) :
    exec (linkedOps named order) 64 GoSource.danglingProgram (initialEnv merged pruned fields version) =
      .ok (.returned (outcome ((linkedOps named order).danglingItems merged pruned fields version))) := by
  rw [imported_dangling_matches_spec]
  rfl

/-- The complete imported pruning orchestration agrees with its model for arbitrary finite inputs. -/
theorem linked_prune_matches_model (named : FieldSet) (order : VersionIteration) (merged : State)
    (managers : ManagedFields) (actor : String) (fields : FieldSet) (version : String)
    (entry : ManagerFields) (actorPresent : managerLookup managers actor = some entry) :
    exec (linkedOps named order) 64 GoSource.pruneProgram
        (PruneSource.pruneEnv merged managers actor (some (fields, version))) =
      .ok (.returned [.object (pruneWithFields named merged fields (PruneSource.restoringFields managers)), .nil]) := by
  rw [PruneSource.imported_prune_matches_spec (linkedOps named order) merged managers actor _ entry actorPresent]
  cases empty : fields.isEmpty <;>
    simp [PruneSource.pruneSpec, linkedOps, PruneSource.modelOps, normalizedOps, conversionOutcome,
      pruneWithFields, ensure_named_empty_iff, empty]

/-- Identity-conversion executions never reach the nullable-input fallback in the callee contract. -/
theorem linked_restore_preserves_objects (named : FieldSet) (order : VersionIteration)
    (rows : VersionFields) (merged pruned : State) :
    ∃ finalMerged finalPruned, restoreAll (linkedOps named order) rows (some merged, some pruned) =
      .ok (some finalMerged, some finalPruned) := by
  induction rows generalizing merged pruned with
  | nil => exact ⟨merged, pruned, rfl⟩
  | cons row rest ih =>
      let restored := addBackOwnedFields named merged pruned row.2
      obtain ⟨finalMerged, finalPruned, remaining⟩ := ih merged restored
      refine ⟨finalMerged, finalPruned, ?_⟩
      change (do
        let next ← Except.ok (some merged, some restored)
        restoreAll (linkedOps named order) rest next) = _
      exact remaining

/-- These are the three callee obligations used by the imported pruning orchestration. -/
structure CalleesRealized (ops : Ops State Unit) (version : String) (managers : ManagedFields)
    (actor : String) : Prop where
  perVersion : ∀ merged pruned fields,
    exec ops 64 GoSource.ownedVersionProgram (ownedVersionEnv merged pruned fields version) =
      .ok (.returned (pairOutcome (ops.ownedVersionItems (some merged) (some pruned) version fields)))
  owned : ∀ merged pruned,
    exec ops 64 GoSource.ownedItemsProgram (ownedEnv merged pruned version managers actor) =
      .ok (.returned (outcome (ops.ownedItems merged pruned version managers actor)))
  dangling : ∀ merged pruned fields,
    exec ops 64 GoSource.danglingProgram (initialEnv merged pruned fields version) =
      .ok (.returned (outcome (ops.danglingItems merged pruned fields version)))

/-- All four imported functions and all manager counts compose for the single-version model. -/
theorem single_version_pruning_refinement (named : FieldSet) (order : VersionIteration)
    (merged : State) (managers : ManagedFields) (actor : String) (fields : FieldSet)
    (version : String) (entry : ManagerFields)
    (actorPresent : managerLookup managers actor = some entry)
    (same : ∀ row ∈ managers, row.2.version = version) :
    CalleesRealized (linkedOps named order) version managers actor ∧
      exec (linkedOps named order) 64 GoSource.pruneProgram
        (PruneSource.pruneEnv merged managers actor (some (fields, version))) =
      .ok (.returned [.object (pruneWithFields named merged fields (PruneSource.restoringFields managers)), .nil]) := by
  have nonempty : managers ≠ [] := by
    intro empty
    simp [empty, managerLookup] at actorPresent
  refine ⟨⟨?_, ?_, ?_⟩, linked_prune_matches_model named order merged managers actor fields version entry actorPresent⟩
  · exact fun merged pruned fields => linked_version_callback_correct named order merged pruned fields version
  · exact fun merged pruned => linked_owned_callback_correct named order merged pruned version managers actor nonempty same
  · exact fun merged pruned fields => linked_dangling_callback_correct named order merged pruned fields version

end SSA.PruningRefinement
