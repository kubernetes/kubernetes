/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.AtomicMergeSubset

namespace SSA.AtomicMergeContracts

open AtomicMergeMachine AtomicMergeSubset

def chooseWhole (walker : Walker Obj) : Option Obj :=
  match walker.right with
  | some value => some value
  | none => match walker.left with
    | some value => some value
    | none => walker.output

def sourceRule (ops : Ops Obj) (state : Machine Obj) : Except String (Machine Obj) :=
  invokeVoid ops AtomicMergeSource.keepRightProgram state

/-- The imported closure copies an entire operand, preserving explicit null through the output pointer. -/
theorem imported_keep_right (ops : Ops Obj) (state : Machine Obj) :
    sourceRule ops state = .ok {state with walker := {state.walker with output := chooseWhole state.walker}} := by
  cases left : state.walker.left <;> cases right : state.walker.right <;>
    simp [sourceRule,invokeVoid,invoke,AtomicMergeSource.keepRightProgram,exec,many,one,eval,
      AtomicMergeMachine.read,operand,isNil,call,assign,write,setField,restore,chooseWhole,left,right,
      Bind.bind,Except.bind]
  all_goals cases current : state.walker <;> simp_all

def withSourceRule (ops : Ops Obj) : Ops Obj :=
  {ops with rule := sourceRule ops}

def sourceLeaf (ops : Ops Obj) (state : Machine Obj) : Except String (Machine Obj) :=
  invokeVoid (withSourceRule ops) AtomicMergeSource.leafProgram state

def leafResult (state : Machine Obj) : Machine Obj :=
  if state.walker.inLeaf then state
  else {state with
    walker := {state.walker with inLeaf := true,output := chooseWhole state.walker}
    events := state.events ++ [.rule]}

/-- A previously selected atomic leaf suppresses every subsequent rule invocation. -/
theorem imported_leaf (ops : Ops Obj) (state : Machine Obj) :
    sourceLeaf ops state = .ok (leafResult state) := by
  cases selected : state.walker.inLeaf <;>
    simp [sourceLeaf,invokeVoid,invoke,AtomicMergeSource.leafProgram,exec,many,one,eval,
      AtomicMergeMachine.read,setField,effect,withSourceRule,imported_keep_right,restore,leafResult,
      selected,chooseWhole,Bind.bind,Except.bind]

def linkedOps (ops : Ops Obj) : Ops Obj :=
  {withSourceRule ops with leaf := sourceLeaf ops}

def program : Kind → List GoSubset.Stmt
  | .list => AtomicMergeSource.listProgram
  | .map => AtomicMergeSource.mapProgram

def emptyResource : Option Resource → Bool
  | none => true
  | some value => value.length == 0

def freed (resource : Option Resource) : List Event :=
  resource.toList.map (fun value => .free value.handle)

def atomicResult (ops : Ops Obj) (kind : Kind) (state : Machine Obj) : Machine Obj :=
  let selected := leafResult {state with events := state.events ++ [.leaf]}
  {selected with
    events := selected.events ++ freed (ops.deref kind "rhs: " state.walker.right) ++
      freed (ops.deref kind "lhs: " state.walker.left)}

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

/-- Atomic dispatch executes both coercions and real leaf/rule source, with per-call LIFO cleanup. -/
theorem imported_atomic_merge (ops : Ops Obj) (kind : Kind) (state : Machine Obj)
    (atomic : state.atomic = true) :
    invoke (linkedOps ops) (program kind) false state = .ok ([.nil],atomicResult ops kind state) := by
  cases covered : state.walker.inLeaf <;>
    cases left : ops.deref kind "lhs: " state.walker.left <;>
    cases right : ops.deref kind "rhs: " state.walker.right <;>
    cases leftEmpty : emptyResource (ops.deref kind "lhs: " state.walker.left) <;>
    cases rightEmpty : emptyResource (ops.deref kind "rhs: " state.walker.right) <;>
    cases kind <;>
    simp_all [program,AtomicMergeSource.listProgram,AtomicMergeSource.mapProgram,
      invoke,exec,many,one,eval,AtomicMergeMachine.read,operand_roundtrip,isNil,call,assign,write,
      captureFree,effect,linkedOps,withSourceRule,imported_leaf,restore,atomicResult,
      leafResult,emptyResource,resource,freed,List.append_assoc,Bind.bind,Except.bind]

theorem imported_atomic_right_is_whole (ops : Ops Obj) (kind : Kind) (state after : Machine Obj)
    (atomic : state.atomic = true) (fresh : state.walker.inLeaf = false)
    (value : Obj) (present : state.walker.right = some value)
    (returned : invoke (linkedOps ops) (program kind) false state = .ok ([.nil],after)) :
    after.walker.output = some value ∧ after.walker.inLeaf = true ∧ after.frame = state.frame := by
  rw [imported_atomic_merge ops kind state atomic] at returned
  cases returned
  simp [atomicResult,leafResult,chooseWhole,fresh,present]

theorem imported_atomic_absent_right_keeps_left (ops : Ops Obj) (kind : Kind) (state after : Machine Obj)
    (atomic : state.atomic = true) (fresh : state.walker.inLeaf = false)
    (absent : state.walker.right = none) (value : Obj) (present : state.walker.left = some value)
    (returned : invoke (linkedOps ops) (program kind) false state = .ok ([.nil],after)) :
    after.walker.output = some value := by
  rw [imported_atomic_merge ops kind state atomic] at returned
  cases returned
  simp [atomicResult,leafResult,chooseWhole,fresh,absent,present]

end SSA.AtomicMergeContracts
