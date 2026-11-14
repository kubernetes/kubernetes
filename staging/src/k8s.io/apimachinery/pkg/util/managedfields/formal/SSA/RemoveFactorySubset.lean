/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.AtomicRemoveContracts

namespace SSA.RemoveFactorySubset

open RawGoSyntax (Node field find)
open AtomicRemoveSubset (atom nodes)

structure Request (Obj Sch Fields Ref : Type) where
  input : Obj
  schema : Sch
  fields : Fields
  typeRef : Ref
  extract : Bool

/-- Omitted Go struct fields receive their zero values before the handler sees the walker. -/
structure Walker (Obj Sch Fields : Type) where
  input : Option Obj := none
  schema : Option Sch := none
  fields : Option Fields := none
  allocator : Bool := false
  output : Option Obj := none
  extract : Bool := false

structure Ops (Obj Sch Fields Ref Err : Type) where
  resolve : Sch → Ref → Obj → Walker Obj Sch Fields → Except String (Walker Obj Sch Fields × Err)
  wrap : Option Obj → Obj

inductive Value (Obj Sch Fields Ref : Type) where
  | object (value : Obj)
  | raw (value : Option Obj)
  | schema (value : Sch)
  | fields (value : Fields)
  | typeRef (value : Ref)
  | flag (value : Bool)
  | valuePackage | allocator | walker
  | recordValue (value : Walker Obj Sch Fields)
  | allocated (value : Walker Obj Sch Fields)

structure State (Obj Sch Fields : Type) where
  walker : Option (Walker Obj Sch Fields) := none

def read (request : Request Obj Sch Fields Ref) (state : State Obj Sch Fields) :
    String → Except String (Value Obj Sch Fields Ref)
  | "val" => .ok (.object request.input)
  | "schema" => .ok (.schema request.schema)
  | "toRemove" => .ok (.fields request.fields)
  | "typeRef" => .ok (.typeRef request.typeRef)
  | "shouldExtract" => .ok (.flag request.extract)
  | "value" => .ok .valuePackage
  | "nil" => .ok (.raw none)
  | "w" => if state.walker.isSome then .ok .walker else .error "Removal walker is not allocated"
  | _ => .error "Unbound removal factory variable"

def writeField (state : Walker Obj Sch Fields) : String → Value Obj Sch Fields Ref →
    Except String (Walker Obj Sch Fields)
  | "value",.object value => .ok {state with input := some value}
  | "value",.raw none => .ok {state with input := none}
  | "schema",.schema value => .ok {state with schema := some value}
  | "toRemove",.fields value => .ok {state with fields := some value}
  | "allocator",.allocator => .ok {state with allocator := true}
  | "out",.raw value => .ok {state with output := value}
  | "shouldExtract",.flag value => .ok {state with extract := value}
  | _,_ => .error "Unmodeled removal factory struct field"

mutual
  def eval (ops : Ops Obj Sch Fields Ref Err) (request : Request Obj Sch Fields Ref) :
      Nat → Node → State Obj Sch Fields → Except String (Value Obj Sch Fields Ref)
    | 0,_,_ => .error "Removal factory expression syntax fuel exhausted"
    | fuel+1,expression,state => do
        match expression with
        | .record "Ident" _ => read request state (← atom (field expression "Name"))
        | .record "SelectorExpr" _ =>
            match ← eval ops request fuel (field expression "X") state,
                field (field expression "Sel") "Name",state.walker with
            | .walker,.atom "out",some walker => .ok (.raw walker.output)
            | _,_,_ => .error "Unmodeled removal factory selector"
        | .record "UnaryExpr" _ =>
            match field expression "Op",← eval ops request fuel (field expression "X") state with
            | .atom "&",.recordValue walker => .ok (.allocated walker)
            | .atom "!",.flag value => .ok (.flag (!value))
            | _,_ => .error "Unmodeled removal factory unary operation"
        | .record "CompositeLit" _ =>
            match field expression "Type" with
            | .record "Ident" fields =>
                match find fields "Name" with
                | .atom "removingWalker" =>
                    .ok (.recordValue (← construct ops request fuel (← nodes (field expression "Elts")) state {}))
                | _ => .error "Removal factory requires the concrete removingWalker type"
            | _ => .error "Unmodeled removal factory record type"
        | .record "CallExpr" _ =>
            match field expression "Fun",field expression "Ellipsis" with
            | .record "SelectorExpr" fields,.bit false =>
                match ← eval ops request fuel (find fields "X") state,
                    field (find fields "Sel") "Name",
                    ← evalAll ops request fuel (← nodes (field expression "Args")) state with
                | .valuePackage,.atom "NewFreelistAllocator",[] => .ok .allocator
                | .valuePackage,.atom "NewValueInterface",[.raw value] => .ok (.object (ops.wrap value))
                | _,_,_ => .error "Unmodeled removal factory package call"
            | _,_ => .error "Unmodeled removal factory expression call"
        | _ => .error "Unmodeled removal factory expression syntax"
  termination_by fuel _ _ => (fuel,0)

  def evalAll (ops : Ops Obj Sch Fields Ref Err) (request : Request Obj Sch Fields Ref) :
      Nat → List Node → State Obj Sch Fields → Except String (List (Value Obj Sch Fields Ref))
    | 0,_,_ => .error "Removal factory argument syntax fuel exhausted"
    | _+1,[],_ => .ok []
    | fuel+1,first::rest,state => do
        .ok ((← eval ops request fuel first state)::(← evalAll ops request fuel rest state))
  termination_by fuel _ _ => (fuel,1)

  def construct (ops : Ops Obj Sch Fields Ref Err) (request : Request Obj Sch Fields Ref) :
      Nat → List Node → State Obj Sch Fields → Walker Obj Sch Fields → Except String (Walker Obj Sch Fields)
    | 0,_,_,_ => .error "Removal factory field syntax fuel exhausted"
    | _+1,[],_,walker => .ok walker
    | fuel+1,first::rest,state,walker => do
        match first,field first "Key" with
        | .record "KeyValueExpr" _,.record "Ident" keys =>
            let next ← writeField walker (← atom (find keys "Name"))
              (← eval ops request fuel (field first "Value") state)
            construct ops request fuel rest state next
        | _,_ => .error "Removal factory fields must have explicit names"
  termination_by fuel _ _ _ => (fuel,2)
end

def resolve (ops : Ops Obj Sch Fields Ref Err) (request : Request Obj Sch Fields Ref)
    (expression : Node) (state : State Obj Sch Fields) : Except String (State Obj Sch Fields) := do
  match expression,field expression "Fun",field expression "Ellipsis" with
  | .record "CallExpr" _,.record "Ident" fields,.bit false =>
      match find fields "Name",← evalAll ops request 64 (← nodes (field expression "Args")) state,state.walker with
      | .atom "resolveSchema",[.schema schema,.typeRef typeRef,.object value,.walker],some walker =>
          let (after,_) ← ops.resolve schema typeRef value walker
          .ok {walker := some after}
      | _,_,_ => .error "Unmodeled removal factory schema dispatch"
  | _,_,_ => .error "Unmodeled removal factory effect"

inductive Result (Obj Sch Fields : Type) where
  | next (state : State Obj Sch Fields)
  | returned (value : Obj)

mutual
  def exec (ops : Ops Obj Sch Fields Ref Err) (request : Request Obj Sch Fields Ref) :
      Nat → Node → State Obj Sch Fields → Except String (Result Obj Sch Fields)
    | 0,_,_ => .error "Removal factory statement syntax fuel exhausted"
    | fuel+1,statement,state => do
        match statement with
        | .record "BlockStmt" _ => execList ops request fuel (← nodes (field statement "List")) state
        | .record "AssignStmt" _ =>
            match ← nodes (field statement "Lhs"),← nodes (field statement "Rhs"),field statement "Tok" with
            | [.record "Ident" fields],[rhs],.atom ":=" =>
                match find fields "Name",← eval ops request 64 rhs state,state.walker with
                | .atom "w",.allocated walker,none => .ok (.next {walker := some walker})
                | _,_,_ => .error "Removal factory requires a fresh walker pointer binding"
            | _,_,_ => .error "Unmodeled removal factory assignment"
        | .record "ExprStmt" _ => .ok (.next (← resolve ops request (field statement "X") state))
        | .record "ReturnStmt" _ =>
            match ← evalAll ops request 64 (← nodes (field statement "Results")) state with
            | [.object value] => .ok (.returned value)
            | _ => .error "Removal factory must return a wrapped value"
        | _ => .error "Unmodeled removal factory statement syntax"
  termination_by fuel _ _ => (fuel,0)

  def execList (ops : Ops Obj Sch Fields Ref Err) (request : Request Obj Sch Fields Ref) :
      Nat → List Node → State Obj Sch Fields → Except String (Result Obj Sch Fields)
    | 0,_,_ => .error "Removal factory block syntax fuel exhausted"
    | _+1,[],state => .ok (.next state)
    | fuel+1,first::rest,state => do
        match ← exec ops request fuel first state with
        | .returned value => .ok (.returned value)
        | .next next => execList ops request fuel rest next
  termination_by fuel _ _ => (fuel,1)
end

def invoke (ops : Ops Obj Sch Fields Ref Err) (request : Request Obj Sch Fields Ref) : Except String Obj := do
  match ← exec ops request 32 AtomicRemoveSource.factoryProgram {} with
  | .returned value => .ok value
  | _ => .error "Removal factory fell through without returning a value"

end SSA.RemoveFactorySubset
