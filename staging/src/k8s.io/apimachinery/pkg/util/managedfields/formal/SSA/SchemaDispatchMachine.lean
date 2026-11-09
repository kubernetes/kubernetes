/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SchemaDispatchSource
import SSA.AtomicRemoveSubset

namespace SSA.SchemaDispatchMachine

structure Atom (M S L : Type) where
  map : Option M := none
  scalar : Option S := none
  list : Option L := none

structure Reference (R : Type) where
  original : R
  named : Option String := none

inductive Kind where
  | floating | integer | string | boolean | list | map
  deriving Repr, BEq

structure Frame (Obj M S L R : Type) where
  atom : Atom M S L := {}
  current : Atom M S L := {}
  reference : Option (Reference R) := none
  value : Option Obj := none
  found : Bool := false
  name : String := ""
  typeName : String := ""

structure Machine (Obj M S L R H : Type) where
  frame : Frame Obj M S L R := {}
  handler : H

/-- Normal validation errors are data; interpreter failures have a separate Except channel. -/
structure Ops (Obj M S L R H Err : Type) where
  resolve : Reference R → Atom M S L × Bool
  isKind : Kind → Obj → Bool
  errorf : String → String → Err
  nilErrors : Err
  map : M → H → Except String (Err × H)
  scalar : S → H → Except String (Err × H)
  list : L → H → Except String (Err × H)
  deduce : Atom M S L → Option Obj → H → Except String (Atom M S L × H) :=
    fun _ _ _ => .error "Source atom deduction is not configured"
  handle : Atom M S L → Reference R → H → Except String (Err × H) :=
    fun _ _ _ => .error "Source atom handler is not configured"

inductive Value (Obj M S L R Err : Type) where
  | schema | handler | schemaPackage | nil
  | object (value : Obj)
  | atom (value : Atom M S L)
  | reference (value : Reference R)
  | map (value : M)
  | scalar (value : S)
  | list (value : L)
  | text (value : String)
  | stringPointer (value : String)
  | flag (value : Bool)
  | errors (value : Err)

def optionalObject : Option Obj → Value Obj M S L R Err
  | none => .nil
  | some value => .object value

def toOptionalObject : Value Obj M S L R Err → Except String (Option Obj)
  | .nil => .ok none
  | .object value => .ok (some value)
  | _ => .error "Atom deduction requires an optional Value interface"

theorem optional_roundtrip (value : Option Obj) : toOptionalObject (optionalObject value : Value Obj M S L R Err) = .ok value := by
  cases value <;> rfl

def read (state : Machine Obj M S L R H) : String → Except String (Value Obj M S L R Err)
  | "s" => .ok .schema
  | "schema" => .ok .schemaPackage
  | "ah" => .ok .handler
  | "nil" => .ok .nil
  | "val" | "v" => .ok (optionalObject state.frame.value)
  | "atom" => .ok (.atom state.frame.atom)
  | "a" => .ok (.atom state.frame.current)
  | "tr" => match state.frame.reference with
      | some value => .ok (.reference value)
      | none => .error "Schema reference was not initialized"
  | "ok" => .ok (.flag state.frame.found)
  | "name" => .ok (.text state.frame.name)
  | "typeName" => .ok (.text state.frame.typeName)
  | _ => .error "Unbound schema dispatch variable"

def select : Value Obj M S L R Err → String → Except String (Value Obj M S L R Err)
  | .atom value,"Map" => .ok (match value.map with | none => .nil | some value => .map value)
  | .atom value,"Scalar" => .ok (match value.scalar with | none => .nil | some value => .scalar value)
  | .atom value,"List" => .ok (match value.list with | none => .nil | some value => .list value)
  | .reference value,"NamedType" => .ok (match value.named with | none => .nil | some value => .stringPointer value)
  | _,_ => .error "Unmodeled schema dispatch selection"

def isNil : Value Obj M S L R Err → Except String Bool
  | .nil => .ok true
  | .object _ | .map _ | .scalar _ | .list _ | .stringPointer _ => .ok false
  | _ => .error "Unmodeled schema dispatch nil comparison"

def write (state : Machine Obj M S L R H) : String → Value Obj M S L R Err → Except String (Machine Obj M S L R H)
  | "a",.atom value => .ok {state with frame := {state.frame with current := value}}
  | "atom",.atom value => .ok {state with frame := {state.frame with atom := value}}
  | "ok",.flag value => .ok {state with frame := {state.frame with found := value}}
  | "name",.text value => .ok {state with frame := {state.frame with name := value}}
  | "typeName",.text value => .ok {state with frame := {state.frame with typeName := value}}
  | _,_ => .error "Unmodeled schema dispatch assignment"

def assign : List String → List (Value Obj M S L R Err) → Machine Obj M S L R H →
    Except String (Machine Obj M S L R H)
  | [],[],state => .ok state
  | name::names,value::values,state => do assign names values (← write state name value)
  | _,_,_ => .error "Schema dispatch assignment arity mismatch"

def setAtomField (value : Atom M S L) : String → Value Obj M S L R Err → Except String (Atom M S L)
  | "Map",.map map => .ok {value with map := some map}
  | "Scalar",.scalar scalar => .ok {value with scalar := some scalar}
  | "List",.list list => .ok {value with list := some list}
  | "Map",.nil => .ok {value with map := none}
  | "Scalar",.nil => .ok {value with scalar := none}
  | "List",.nil => .ok {value with list := none}
  | _,_ => .error "Unmodeled schema atom constructor field"

def method (ops : Ops Obj M S L R H Err) (receiver : Value Obj M S L R Err) (name : String)
    (arguments : List (Value Obj M S L R Err)) (state : Machine Obj M S L R H) :
    Except String (List (Value Obj M S L R Err) × Machine Obj M S L R H) := do
  match receiver,name,arguments with
  | .schema,"Resolve",[.reference reference] =>
      let (atom,found) := ops.resolve reference
      .ok ([.atom atom,.flag found],state)
  | .object value,"IsFloat",[] => .ok ([.flag (ops.isKind .floating value)],state)
  | .object value,"IsInt",[] => .ok ([.flag (ops.isKind .integer value)],state)
  | .object value,"IsString",[] => .ok ([.flag (ops.isKind .string value)],state)
  | .object value,"IsBool",[] => .ok ([.flag (ops.isKind .boolean value)],state)
  | .object value,"IsList",[] => .ok ([.flag (ops.isKind .list value)],state)
  | .object value,"IsMap",[] => .ok ([.flag (ops.isKind .map value)],state)
  | .handler,"doMap",[.map value] =>
      let (errors,handler) ← ops.map value state.handler
      .ok ([.errors errors],{state with handler})
  | .handler,"doScalar",[.scalar value] =>
      let (errors,handler) ← ops.scalar value state.handler
      .ok ([.errors errors],{state with handler})
  | .handler,"doList",[.list value] =>
      let (errors,handler) ← ops.list value state.handler
      .ok ([.errors errors],{state with handler})
  | _,_,_ => .error "Unmodeled schema dispatch method"

def functionCall (ops : Ops Obj M S L R H Err) (name : String)
    (arguments : List (Value Obj M S L R Err)) (state : Machine Obj M S L R H) :
    Except String (List (Value Obj M S L R Err) × Machine Obj M S L R H) := do
  match name,arguments with
  | "errorf",[.text format,.text value] => .ok ([.errors (ops.errorf format value)],state)
  | "deduceAtom",[.atom atom,value] =>
      let value ← toOptionalObject value
      let (after,handler) ← ops.deduce atom value state.handler
      .ok ([.atom after],{state with handler})
  | "handleAtom",[.atom atom,.reference reference,.handler] =>
      let (errors,handler) ← ops.handle atom reference state.handler
      .ok ([.errors errors],{state with handler})
  | _,_ => .error "Unmodeled schema dispatch function"

end SSA.SchemaDispatchMachine
