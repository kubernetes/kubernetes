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
import SSA.Counterexamples
import SSA.PruningRefinement

namespace SSA.RawSourceCounterexamples

open Counterexamples GoSubset PruningRefinement

set_option maxRecDepth 10000
set_option maxHeartbeats 4000000

/-- The new Apply owns no item fields; old item claims are passed separately as lastSet. -/
def actorFields : ManagerFields := ⟨[], "v1", true⟩
def managers : ManagedFields := [("A", actorFields)]
def lastFields : FieldSet := claimFields (populatedList.claims.filter (fun claim => claim.manager == "A"))

/-- The imported source has the counterexample under the normalized primitives,
    with every pruning callee realized, under every lawful map iteration schedule. -/
theorem imported_pruning_loses_explicit_empty_list (order : VersionIteration) :
    CalleesRealized (linkedOps emptyList.named order) "v1" managers "A" ∧
    exec (linkedOps emptyList.named order) 64 GoSource.pruneProgram
      (PruneSource.pruneEnv (wireMerged populatedList emptyList) managers "A"
        (some (lastFields, "v1"))) =
      .ok (.returned [.object firstEmptyList, .nil]) := by
  exact single_version_pruning_refinement emptyList.named order
    (wireMerged populatedList emptyList) managers "A" lastFields "v1" actorFields rfl
    (by intro row member; simp [managers] at member; subst row; rfl)

/-- After the old item footprint is gone, the identical request takes the source's empty-set exit. -/
theorem imported_pruning_second_apply_retains_empty_list (order : VersionIteration) :
    exec (linkedOps emptyList.named order) 64 GoSource.pruneProgram
      (PruneSource.pruneEnv (wireMerged firstEmptyList emptyList) managers "A"
        (some ([], "v1"))) =
      .ok (.returned [.object secondEmptyList, .nil]) := by
  exact (single_version_pruning_refinement emptyList.named order
    (wireMerged firstEmptyList emptyList) managers "A" [] "v1" actorFields rfl
    (by intro row member; simp [managers] at member; subst row; rfl)).2

end SSA.RawSourceCounterexamples
