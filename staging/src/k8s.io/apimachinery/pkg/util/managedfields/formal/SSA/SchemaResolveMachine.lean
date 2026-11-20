/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SchemaLookupContracts

namespace SSA.SchemaResolveMachine

variable {M S L R Rel P : Type}

/-- A container's relationship is separate from its otherwise opaque immutable contents.
    This is a value view of a Go struct; physical pointer identity is a representation obligation. -/
structure Container (P Rel : Type) where
  payload : P
  relationship : Rel

abbrev Atom (M S L Rel : Type) := SchemaDispatchMachine.Atom (Container M Rel) S (Container L Rel)
abbrev Reference (M S L R Rel : Type) := SchemaLookupSubset.Reference (Container M Rel) S (Container L Rel) R Rel

structure State (M S L R Rel : Type) where
  cache : Option (R → Option (Atom M S L Rel)) := none
  held : Bool := false
  events : List Bool := []
  plainCalls : Nat := 0

structure Ops (M S L Rel : Type) where
  findNamed : String → Atom M S L Rel × Bool

structure Copy (P Rel : Type) where
  payload : Option P := none
  relationship : Option Rel := none

def copyOf (value : Container P Rel) : Copy P Rel :=
  {payload := some value.payload,relationship := some value.relationship}

def address (value : Copy P Rel) : Except String (Container P Rel) := do
  match value.payload,value.relationship with
  | some payload,some relationship => .ok {payload,relationship}
  | _,_ => .error "Schema copy address requires initialized contents and relationship"

inductive Value (M S L R Rel : Type) where
  | schema | lock | nil | mapLocal | listLocal
  | reference (value : Reference M S L R Rel)
  | atom (value : Atom M S L Rel)
  | map (value : Container M Rel)
  | list (value : Container L Rel)
  | scalar (value : S)
  | mapCopy (value : Copy M Rel)
  | listCopy (value : Copy L Rel)
  | relationship (value : Rel)
  | relationshipPointer (value : Rel)
  | cache (value : Option (R → Option (Atom M S L Rel)))
  | flag (value : Bool)

structure Deferred (M S L R Rel : Type) where
  receiver : Value M S L R Rel
  method : String
  arguments : List (Value M S L R Rel)

structure Machine (M S L R Rel : Type) where
  reference : Reference M S L R Rel
  state : State M S L R Rel
  result : Atom M S L Rel := {}
  found : Bool := false
  mapCopy : Copy M Rel := {}
  listCopy : Copy L Rel := {}
  mapEscaped : Bool := false
  listEscaped : Bool := false
  deferred : List (Deferred M S L R Rel) := []

def read (frame : Machine M S L R Rel) : String → Except String (Value M S L R Rel)
  | "s" => .ok .schema
  | "tr" => .ok (.reference frame.reference)
  | "result" => .ok (.atom frame.result)
  | "exists" => .ok (.flag frame.found)
  | "mapCopy" => .ok (.mapCopy frame.mapCopy)
  | "listCopy" => .ok (.listCopy frame.listCopy)
  | "nil" => .ok .nil
  | "true" => .ok (.flag true)
  | "false" => .ok (.flag false)
  | _ => .error "Unbound schema resolver variable"

def select (frame : Machine M S L R Rel) : Value M S L R Rel → String → Except String (Value M S L R Rel)
  | .schema,"lock" => .ok .lock
  | .schema,"resolvedTypes" => .ok (.cache frame.state.cache)
  | .reference value,"ElementRelationship" =>
      .ok (match value.relationship with | none => .nil | some rel => .relationshipPointer rel)
  | .atom value,"Map" => .ok (match value.map with | none => .nil | some value => .map value)
  | .atom value,"Scalar" => .ok (match value.scalar with | none => .nil | some value => .scalar value)
  | .atom value,"List" => .ok (match value.list with | none => .nil | some value => .list value)
  | _,_ => .error "Unmodeled schema resolver selection"

def isNil : Value M S L R Rel → Except String Bool
  | .nil | .cache none => .ok true
  | .map _ | .scalar _ | .list _ | .relationshipPointer _ | .cache (some _) => .ok false
  | _ => .error "Unmodeled schema resolver nil comparison"

def plain (ops : Ops M S L Rel) (reference : Reference M S L R Rel) : Except String (Atom M S L Rel × Bool) :=
  SchemaLookupSubset.invoke {findNamed := ops.findNamed} SchemaLookupSource.plainProgram reference

theorem plain_source (ops : Ops M S L Rel) (reference : Reference M S L R Rel) :
    plain ops reference = .ok (SchemaLookupContracts.plainResult {findNamed := ops.findNamed} reference) :=
  SchemaLookupContracts.imported_plain _ _

def method (ops : Ops M S L Rel) (receiver : Value M S L R Rel) (name : String)
    (arguments : List (Value M S L R Rel)) (frame : Machine M S L R Rel) :
    Except String (List (Value M S L R Rel) × Machine M S L R Rel) := do
  match receiver,name,arguments with
  | .schema,"resolveNoOverrides",[.reference reference] =>
      let (value,found) ← plain ops reference
      .ok ([.atom value,.flag found],{frame with state := {frame.state with plainCalls := frame.state.plainCalls+1}})
  | .lock,"Lock",[] =>
      if frame.state.held then .error "Schema resolver would reacquire a held mutex"
      else .ok ([],{frame with state := {frame.state with held := true,events := frame.state.events++[true]}})
  | .lock,"Unlock",[] =>
      if frame.state.held then
        .ok ([],{frame with state := {frame.state with held := false,events := frame.state.events++[false]}})
      else .error "Schema resolver would unlock an unheld mutex"
  | .map value,"CopyInto",[.mapLocal] =>
      if frame.mapEscaped then .error "Schema map copy cannot change after its pointer escapes"
      else .ok ([],{frame with mapCopy := copyOf value})
  | _,_,_ => .error "Unmodeled schema resolver method"

def lookup (cache : Option (R → Option (Atom M S L Rel))) (key : R) : Atom M S L Rel × Bool :=
  match cache.bind (fun values => values key) with
  | some value => (value,true)
  | none => ({},false)

def store [DecidableEq R] (cache : Option (R → Option (Atom M S L Rel))) (key : R) (value : Atom M S L Rel) :
    Except String (Option (R → Option (Atom M S L Rel))) :=
  match cache with
  | none => .error "Schema resolver would write a nil map"
  | some values => .ok (some (fun other => if other = key then some value else values other))

end SSA.SchemaResolveMachine
