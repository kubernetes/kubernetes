/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.UpdaterContracts

namespace SSA.UpdaterExamples

open GoSubset (ManagedFields managerLookup)
open UpdaterSubset

set_option maxRecDepth 10000
set_option maxHeartbeats 4000000

def amount : Path := [.field "amount"]

/-- A one-scalar comparison with two identity-converted versions, both excluding amount. -/
def ops : Ops Nat Unit where
  primitives := {
    convert := fun object _ => .ok object
    missing := fun _ => false
    toFields := fun _ => .ok [amount]
    ensureNamed := fun _ _ fields => fields
    remove := fun object _ => object
    wrap := fun _ error => error
    ignoreFilters := some [("v1",fun _ => []),("v2",fun _ => [])] }
  compare := fun old new => .ok {modified := if old == new then [] else [amount]}

def owners : ManagedFields :=
  [("A",⟨[amount],"v2",true⟩),("B",⟨[amount],"v1",false⟩)]

def filteredCacheObserved : Bool :=
  match exec ops 128 UpdaterSource.updateProgram (initial 0 1 "v2" owners "A" false) with
  | .ok (.returned [.managed actual,.reference rootReference,.nil] state) =>
      match read state "compare$3", read state "versions",
          select ops (.reference rootReference) "Modified" state.heap with
      | .reference localReference, .comparisons cache, .ok (.fields modified) =>
          modified.isEmpty &&
          (cache.find? (fun entry => entry.1 == "v1")).map Prod.snd == some localReference &&
          (match heapRead state.heap localReference with
            | .ok comparison => comparison.modified.isEmpty
            | _ => false) &&
          (match managerLookup actual "B" with
            | some entry => entry.fields.contains amount && !entry.applied
            | none => false)
      | _,_,_ => false
  | _ => false

/-- The complete imported function's uncached-version path observes the in-place filter. -/
theorem imported_update_observes_in_place_filter : filteredCacheObserved = true := by
  simp [filteredCacheObserved,UpdaterSource.updateProgram,exec,execRange,initial,
    runAll,runEval,eval,evalAll,UpdaterSubset.lookup,UpdaterSubset.read,write,bindNames,call,select,mapIndex,mapSet,
    allocate,heapWrite,heapRead,transform,isNil,scalarEqual,ops,owners,amount,
    naturalIteration,GoSubset.managerLookup,fieldIntersection,Bind.bind,Except.bind]

structure Observation where
  filteredCacheObserved : Bool
  forcedClaimsStayBounded : Bool
  deriving Lean.ToJson

def extra : Path := [.field "extra"]

def forceOps : Ops Nat Unit := {
  ops with
    primitives := {ops.primitives with ignoreFilters := none}
    compare := fun old new => .ok {
      modified := if old == new then [] else [amount]
      added := if old == new then [] else [extra] } }

def forceOwners : ManagedFields :=
  [("A",⟨[amount,extra],"v1",true⟩),("B",⟨[amount],"v1",true⟩)]

def forcedClaimsStayBounded : Bool :=
  match exec forceOps 128 UpdaterSource.updateProgram (initial 0 1 "v1" forceOwners "A" true) with
  | .ok (.returned [.managed actual,.reference _,.nil] _) =>
      actual.length == 1 &&
      (match managerLookup actual "A" with
        | some entry => entry.fields.contains amount && entry.fields.contains extra && entry.applied
        | none => false) &&
      (match managerLookup actual "B" with | none => true | _ => false)
  | _ => false

theorem imported_forced_update_does_not_give_B_new_fields : forcedClaimsStayBounded = true := by
  simp [forcedClaimsStayBounded,UpdaterSource.updateProgram,exec,execRange,initial,
    runAll,runEval,eval,evalAll,UpdaterSubset.lookup,UpdaterSubset.read,write,bindNames,call,select,mapIndex,mapSet,mapDelete,
    allocate,heapWrite,heapRead,isNil,scalarEqual,forceOps,ops,forceOwners,amount,extra,
    naturalIteration,GoSubset.managerLookup,GoSubset.managerSet,fieldIntersection,fieldDifference,Bind.bind,Except.bind]

def observe : Observation := ⟨filteredCacheObserved,forcedClaimsStayBounded⟩

end SSA.UpdaterExamples
