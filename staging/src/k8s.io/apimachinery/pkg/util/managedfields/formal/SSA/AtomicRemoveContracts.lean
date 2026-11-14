/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.AtomicRemoveSubset

namespace SSA.AtomicRemoveContracts

open AtomicRemoveSubset RawGoSyntax

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

/-- A scalar helper copies the complete value; caller-side path selection governs deletion. -/
theorem imported_scalar (ops : Ops Obj) (state : Machine Obj) :
    invoke ops AtomicRemoveSource.scalarProgram state = .ok {state with output := ops.unstructured state.input} := by
  cases output : ops.unstructured state.input <;>
    simp [invoke,AtomicRemoveSource.scalarProgram,exec,execList,eval,evalAll,
      field,find,atom,nodes,AtomicRemoveSubset.read,select,call,assign,data,output,Bind.bind,Except.bind]

def program : Kind → Node
  | .list => AtomicRemoveSource.listProgram
  | .map => AtomicRemoveSource.mapProgram

def cleanup (kind : Kind) (resource : Option Resource) : List (Option Nat) :=
  match kind,resource with
  | .list,value => [value.map (·.handle)]
  | .map,none => []
  | .map,some value => [some value.handle]

def emptyResource : Option Resource → Bool
  | none => true
  | some value => value.length == 0

def atomicResult (ops : Ops Obj) (kind : Kind) (state : Machine Obj) : Machine Obj :=
  if ops.isKind kind state.input then
    {state with
      output := if state.extract then ops.unstructured state.input else state.output
      freed := state.freed ++ cleanup kind (ops.asContainer kind state.input)}
  else state

/-- Atomic and empty containers return before every child traversal; output is never fragmented. -/
theorem imported_atomic_container (ops : Ops Obj) (kind : Kind) (state : Machine Obj)
    (atomic : state.atomic = true) :
    invoke ops (program kind) state = .ok (atomicResult ops kind state) := by
  cases shape : ops.isKind kind state.input <;>
    cases extracting : state.extract <;>
    cases view : ops.asContainer kind state.input <;>
    cases output : ops.unstructured state.input <;>
    cases empty : emptyResource (ops.asContainer kind state.input) <;>
    cases kind <;>
    cases state <;>
    simp_all (config := {maxSteps := 1000000}) [invoke,program,AtomicRemoveSource.mapProgram,AtomicRemoveSource.listProgram,
      exec,execList,eval,evalAll,field,find,atom,nodes,AtomicRemoveSubset.read,lookup,select,call,assign,
      deferFree,resource,data,equal,isNil,atomicResult,cleanup,emptyResource,Bind.bind,Except.bind]

/-- Fresh output is essential: remove does not clear a previously populated private walker. -/
theorem imported_atomic_remove_fresh (ops : Ops Obj) (kind : Kind) (state after : Machine Obj)
    (atomic : state.atomic = true) (removing : state.extract = false) (fresh : state.output = none)
    (returned : invoke ops (program kind) state = .ok after) : after.output = none := by
  rw [imported_atomic_container ops kind state atomic] at returned
  cases returned
  cases shape : ops.isKind kind state.input <;> simp [atomicResult,removing,fresh,shape]

theorem imported_atomic_extract_whole (ops : Ops Obj) (kind : Kind) (state after : Machine Obj)
    (atomic : state.atomic = true) (extracting : state.extract = true)
    (matching : ops.isKind kind state.input = true)
    (returned : invoke ops (program kind) state = .ok after) :
    after.output = ops.unstructured state.input ∧ after.frame = state.frame ∧ after.input = state.input := by
  rw [imported_atomic_container ops kind state atomic] at returned
  cases returned
  simp [atomicResult,extracting,matching]

end SSA.AtomicRemoveContracts
