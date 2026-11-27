/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.AtomicMergeSource
import SSA.AtomicFieldSubset

namespace SSA.AtomicMergeMachine

abbrev Kind := AtomicFieldSubset.Kind

structure Resource where
  handle : Nat
  length : Nat
  deriving Repr

/-- None denotes a missing Go interface/pointer; Some may itself contain JSON null. -/
structure Walker (Obj : Type) where
  left : Option Obj
  right : Option Obj
  output : Option Obj := none
  inLeaf : Bool := false

structure Frame (Obj : Type) where
  left : Option Resource := none
  right : Option Resource := none
  emptyPromote : Bool := false
  temporary : Option Obj := none
  deferred : List Nat := []

inductive Event where
  | rule | leaf | free (handle : Nat)
  deriving Repr, BEq

structure Machine (Obj : Type) where
  walker : Walker Obj
  frame : Frame Obj := {}
  atomic : Bool := true
  events : List Event := []

structure Ops (Obj : Type) where
  deref : Kind → String → Option Obj → Option Resource
  rule : Machine Obj → Except String (Machine Obj) :=
    fun _ => .error "Merge rule source is not configured"
  leaf : Machine Obj → Except String (Machine Obj) :=
    fun _ => .error "Merge leaf source is not configured"

inductive Value (Obj : Type) where
  | walker | typeInfo | schemaPackage | allocator | nil
  | operand (value : Obj)
  | data (value : Obj)
  | pointer (value : Obj)
  | container (value : Resource)
  | relation (atomic : Bool)
  | flag (value : Bool)
  | number (value : Nat)
  | text (value : String)

def operand : Option Obj → Value Obj
  | none => .nil
  | some value => .operand value

def resource : Option Resource → Value Obj
  | none => .nil
  | some value => .container value

def isNil : Value Obj → Bool
  | .nil => true
  | _ => false

def read (state : Machine Obj) : String → Except String (Value Obj)
  | "w" => .ok .walker
  | "t" => .ok .typeInfo
  | "schema" => .ok .schemaPackage
  | "nil" | "errs" => .ok .nil
  | "lhs" => .ok (resource state.frame.left)
  | "rhs" => .ok (resource state.frame.right)
  | "emptyPromoteToLeaf" => .ok (.flag state.frame.emptyPromote)
  | "v$1" | "v$2" =>
      match state.frame.temporary with
      | some value => .ok (.data value)
      | none => .error "Merge temporary was read before assignment"
  | _ => .error "Unsupported atomic merge variable"

def toOperand : Value Obj → Except String (Option Obj)
  | .nil => .ok none
  | .operand value => .ok (some value)
  | _ => .error "Merge coercion requires an optional operand"

theorem operand_roundtrip (value : Option Obj) : toOperand (operand value) = .ok value := by
  cases value <;> rfl

def call (ops : Ops Obj) (receiver : Value Obj) (method : String)
    (arguments : List (Value Obj)) : Except String (List (Value Obj)) := do
  match receiver,method,arguments with
  | .walker,"derefMap",[.text label,value] =>
      .ok [resource (ops.deref .map label (← toOperand value)),.nil]
  | .walker,"derefList",[.text label,value] =>
      .ok [resource (ops.deref .list label (← toOperand value)),.nil]
  | .container value,"Length",[] => .ok [.number value.length]
  | .container value,"Empty",[] => .ok [.flag (value.length == 0)]
  | .operand value,"Unstructured",[] => .ok [.data value]
  | _,_,_ => .error "Unmodeled atomic merge call"

def write (state : Machine Obj) (name : String) (value : Value Obj) : Except String (Machine Obj) :=
  match name,value with
  | "_",_ => .ok state
  | "lhs",.nil => .ok {state with frame := {state.frame with left := none}}
  | "rhs",.nil => .ok {state with frame := {state.frame with right := none}}
  | "lhs",.container value => .ok {state with frame := {state.frame with left := some value}}
  | "rhs",.container value => .ok {state with frame := {state.frame with right := some value}}
  | "emptyPromoteToLeaf",.flag flag => .ok {state with frame := {state.frame with emptyPromote := flag}}
  | "v$1",.data value | "v$2",.data value => .ok {state with frame := {state.frame with temporary := some value}}
  | "errs",.nil => .ok state
  | _,_ => .error "Unsupported atomic merge assignment"

def assign : List String → List (Value Obj) → Machine Obj → Except String (Machine Obj)
  | [],[],state => .ok state
  | name::names,value::values,state => do
      assign names values (← write state name value)
  | _,_,_ => .error "Atomic merge assignment arity mismatch"

def setField (state : Machine Obj) (name field : String) (value : Value Obj) : Except String (Machine Obj) :=
  match name,field,value with
  | "w","inLeaf",.flag flag => .ok {state with walker := {state.walker with inLeaf := flag}}
  | "w","out",.pointer value => .ok {state with walker := {state.walker with output := some value}}
  | "w","out",.nil => .ok {state with walker := {state.walker with output := none}}
  | _,_,_ => .error "Atomic merge writes require the walker's inLeaf or out field"

/-- Callee locals/defers belong to a fresh frame; cleanup cannot run the caller's pending frees. -/
def restore (before after : Machine Obj) : Machine Obj :=
  {before with walker := after.walker,events := after.events ++ after.frame.deferred.map Event.free}

end SSA.AtomicMergeMachine
