/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.AtomicRemoveSource
import SSA.AtomicMergeMachine

namespace SSA.AtomicRemoveSubset

open RawGoSyntax (Node field find)
abbrev Kind := AtomicMergeMachine.Kind
abbrev Resource := AtomicMergeMachine.Resource

inductive Value (Obj : Type) where
  | walker | input | allocator | typeInfo | schemaPackage | nil
  | data (value : Obj)
  | container (value : Resource)
  | flag (value : Bool)
  | number (value : Nat)
  | relation (atomic : Bool)

structure Frame (Obj : Type) where
  locals : List (String × Value Obj) := []
  deferred : List (Option Nat) := []

structure Machine (Obj : Type) where
  input : Obj
  output : Option Obj := none
  extract : Bool
  atomic : Bool := true
  frame : Frame Obj := {}
  freed : List (Option Nat) := []

structure Ops (Obj : Type) where
  isKind : Kind → Obj → Bool
  asContainer : Kind → Obj → Option Resource
  unstructured : Obj → Option Obj

def resource : Option Resource → Value Obj
  | none => .nil
  | some value => .container value

def data : Option Obj → Value Obj
  | none => .nil
  | some value => .data value

def lookup : List (String × Value Obj) → String → Except String (Value Obj)
  | [],_ => .error "Unbound removal variable"
  | (key,value)::rest,name => if key == name then .ok value else lookup rest name

def read (state : Machine Obj) : String → Except String (Value Obj)
  | "w" => .ok .walker
  | "t" => .ok .typeInfo
  | "schema" => .ok .schemaPackage
  | "nil" => .ok .nil
  | name => lookup state.frame.locals name

def select (state : Machine Obj) : Value Obj → String → Except String (Value Obj)
  | .walker,"value" => .ok .input
  | .walker,"allocator" => .ok .allocator
  | .walker,"shouldExtract" => .ok (.flag state.extract)
  | .typeInfo,"ElementRelationship" => .ok (.relation state.atomic)
  | .schemaPackage,"Atomic" => .ok (.relation true)
  | _,_ => .error "Unmodeled removal selector"

def call (ops : Ops Obj) (state : Machine Obj) : Value Obj → String → List (Value Obj) → Except String (Value Obj)
  | .input,"IsList",[] => .ok (.flag (ops.isKind .list state.input))
  | .input,"IsMap",[] => .ok (.flag (ops.isKind .map state.input))
  | .input,"AsListUsing",[.allocator] => .ok (resource (ops.asContainer .list state.input))
  | .input,"AsMapUsing",[.allocator] => .ok (resource (ops.asContainer .map state.input))
  | .input,"Unstructured",[] => .ok (data (ops.unstructured state.input))
  | .container value,"Length",[] => .ok (.number value.length)
  | .container value,"Empty",[] => .ok (.flag (value.length == 0))
  | _,_,_ => .error "Unmodeled removal call"

def isNil : Value Obj → Bool
  | .nil => true
  | _ => false

def equal : Value Obj → Value Obj → Except String (Value Obj)
  | .nil,right => .ok (.flag (isNil right))
  | left,.nil => .ok (.flag (isNil left))
  | .relation left,.relation right => .ok (.flag (left == right))
  | .number left,.number right => .ok (.flag (left == right))
  | _,_ => .error "Unmodeled removal equality"

def atom : Node → Except String String
  | .atom value => .ok value
  | _ => .error "Expected Go AST atom"

def nodes : Node → Except String (List Node)
  | .sequence values => .ok values
  | _ => .error "Expected Go AST sequence"

mutual
  def eval (ops : Ops Obj) : Nat → Node → Machine Obj → Except String (Value Obj)
    | 0,_,_ => .error "Removal expression syntax fuel exhausted"
    | fuel+1,expression,state =>
        match expression with
        | .record "Ident" _ => do read state (← atom (field expression "Name"))
        | .record "SelectorExpr" _ => do
            select state (← eval ops fuel (field expression "X") state)
              (← atom (field (field expression "Sel") "Name"))
        | .record "CallExpr" _ => do
            match field expression "Fun",field expression "Ellipsis" with
            | .record "SelectorExpr" fields,.bit false =>
                call ops state (← eval ops fuel (find fields "X") state)
                  (← atom (field (find fields "Sel") "Name"))
                  (← evalAll ops fuel (← nodes (field expression "Args")) state)
            | _,_ => .error "Unmodeled removal function or variadic call"
        | .record "UnaryExpr" _ => do
            match field expression "Op",← eval ops fuel (field expression "X") state with
            | .atom "!",.flag value => .ok (.flag (!value))
            | _,_ => .error "Unmodeled removal unary operation"
        | .record "BinaryExpr" _ => do
            let left ← eval ops fuel (field expression "X") state
            match field expression "Op" with
            | .atom "||" => match left with
                | .flag value => if value then .ok (.flag true) else do
                    match ← eval ops fuel (field expression "Y") state with
                    | .flag value => .ok (.flag value)
                    | _ => .error "Removal disjunction requires flags"
                | _ => .error "Removal disjunction requires flags"
            | .atom "==" => equal left (← eval ops fuel (field expression "Y") state)
            | .atom "!=" => do
                match ← equal left (← eval ops fuel (field expression "Y") state) with
                | .flag value => .ok (.flag (!value))
                | _ => .error "Removal inequality requires flags"
            | _ => .error "Unmodeled removal binary operation"
        | .record "BasicLit" _ =>
            match field expression "Kind",field expression "Value" with
            | .atom "INT",.atom "0" => .ok (.number 0)
            | _,_ => .error "Unmodeled removal literal"
        | _ => .error "Unmodeled removal expression syntax"
  termination_by fuel _ _ => (fuel,0)

  def evalAll (ops : Ops Obj) : Nat → List Node → Machine Obj → Except String (List (Value Obj))
    | 0,_,_ => .error "Removal arguments syntax fuel exhausted"
    | _+1,[],_ => .ok []
    | fuel+1,first::rest,state => do
        .ok ((← eval ops fuel first state)::(← evalAll ops fuel rest state))
  termination_by fuel _ _ => (fuel,1)
end

def assign (lhs : Node) (token : String) (value : Value Obj) (state : Machine Obj) : Except String (Machine Obj) := do
  match lhs with
  | .record "Ident" _ =>
      let name ← atom (field lhs "Name")
      if (token == ":=" || token == "=") && (name == "l" || name == "m") then
        match value with
        | .nil | .container _ => .ok {state with frame := {state.frame with locals := (name,value)::state.frame.locals}}
        | _ => .error "Removal local requires a container"
      else .error "Unmodeled removal local assignment"
  | .record "SelectorExpr" _ =>
      match field (field lhs "X") "Name",field (field lhs "Sel") "Name",token,value with
      | .atom "w",.atom "out","=",.nil => .ok {state with output := none}
      | .atom "w",.atom "out","=",.data value => .ok {state with output := some value}
      | _,_,_,_ => .error "Unmodeled removal field assignment"
  | _ => .error "Unmodeled removal assignment target"

def deferFree (ops : Ops Obj) (expression : Node) (state : Machine Obj) : Except String (Machine Obj) := do
  match expression with
  | .record "CallExpr" _ =>
      match field expression "Fun",field expression "Ellipsis" with
      | .record "SelectorExpr" fields,.bit false =>
          match ← eval ops 32 (find fields "X") state,field (find fields "Sel") "Name",
              ← evalAll ops 32 (← nodes (field expression "Args")) state with
          | .allocator,.atom "Free",[.container value] =>
              .ok {state with frame := {state.frame with deferred := some value.handle::state.frame.deferred}}
          | .allocator,.atom "Free",[.nil] =>
              .ok {state with frame := {state.frame with deferred := none::state.frame.deferred}}
          | _,_,_ => .error "Unmodeled removal defer target"
      | _,_ => .error "Unmodeled removal defer call"
  | _ => .error "Unmodeled removal defer syntax"

inductive Result (Obj : Type) where
  | next (state : Machine Obj)
  | returned (values : List (Value Obj)) (state : Machine Obj)

mutual
  def exec (ops : Ops Obj) : Nat → Node → Machine Obj → Except String (Result Obj)
    | 0,_,_ => .error "Removal statement syntax fuel exhausted"
    | fuel+1,statement,state => do
        match statement with
        | .absent => .ok (.next state)
        | .record "BlockStmt" _ => execList ops fuel (← nodes (field statement "List")) state
        | .record "AssignStmt" _ =>
            match ← nodes (field statement "Lhs"),← nodes (field statement "Rhs") with
            | [lhs],[rhs] => .ok (.next (← assign lhs (← atom (field statement "Tok")) (← eval ops 64 rhs state) state))
            | _,_ => .error "Unmodeled removal assignment arity"
        | .record "IfStmt" _ =>
            match field statement "Init" with
            | .absent =>
                match ← eval ops 64 (field statement "Cond") state with
                | .flag value => exec ops fuel (field statement (if value then "Body" else "Else")) state
                | _ => .error "Removal conditional requires a flag"
            | _ => .error "Unmodeled removal conditional initializer"
        | .record "DeferStmt" _ => .ok (.next (← deferFree ops (field statement "Call") state))
        | .record "ReturnStmt" _ => .ok (.returned (← evalAll ops 64 (← nodes (field statement "Results")) state) state)
        | _ => .error "Unmodeled removal statement syntax"
  termination_by fuel _ _ => (fuel,0)

  def execList (ops : Ops Obj) : Nat → List Node → Machine Obj → Except String (Result Obj)
    | 0,_,_ => .error "Removal block syntax fuel exhausted"
    | _+1,[],state => .ok (.next state)
    | fuel+1,statement::rest,state => do
        match ← exec ops fuel statement state with
        | .returned values final => .ok (.returned values final)
        | .next next => execList ops fuel rest next
  termination_by fuel _ _ => (fuel,1)
end

def invoke (ops : Ops Obj) (program : Node) (state : Machine Obj) : Except String (Machine Obj) := do
  match ← exec ops 32 program {state with frame := {}} with
  | .returned [.nil] final => .ok {state with output := final.output,freed := final.freed ++ final.frame.deferred}
  | _ => .error "Removal source did not return nil validation errors"

end SSA.AtomicRemoveSubset
