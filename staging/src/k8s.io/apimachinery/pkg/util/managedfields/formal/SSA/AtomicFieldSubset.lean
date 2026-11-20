/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.AtomicFieldSource

namespace SSA.AtomicFieldSubset

open GoSubset (Expr Stmt)

inductive Kind where
  | list | map
  deriving DecidableEq, BEq, Repr

/-- Handles abstract allocator resources; their contents and coercion errors are not inspected here. -/
structure Ops (Obj : Type) where
  view : Kind → Obj → Option Nat

structure Machine (Obj α : Type) where
  input : Obj
  path : List α
  fields : List (List α)
  atomic : Bool := true
  resourceLocal : Option Nat := none
  deferred : List Nat := []
  freed : List Nat := []

inductive Value (α : Type) where
  | walker | typeInfo | schemaPackage | allocator | input
  | coercion (kind : Kind)
  | relation (atomic : Bool)
  | container (handle : Nat)
  | path (path : List α)
  | fieldSet | nil | flag (value : Bool)

def resource : Option Nat → Value α
  | none => .nil
  | some handle => .container handle

def read (state : Machine Obj α) : String → Except String (Value α)
  | "v" => .ok .walker
  | "t" => .ok .typeInfo
  | "schema" => .ok .schemaPackage
  | "listValue" => .ok (.coercion .list)
  | "mapValue" => .ok (.coercion .map)
  | "list" | "m" => .ok (resource state.resourceLocal)
  | "nil" | "errs" => .ok .nil
  | _ => .error "Unsupported typed field-set variable"

def isNil : Value α → Bool
  | .nil => true
  | _ => false

mutual
  def eval (ops : Ops Obj) : Nat → Expr → Machine Obj α → Except String (List (Value α))
    | 0,_,_ => .error "Typed expression fuel exhausted"
    | fuel+1,expression,state =>
        match expression with
        | .var name => do .ok [← read state name]
        | .flag flag => .ok [.flag flag]
        | .select receiver field => do
            match ← one ops fuel receiver state,field with
            | .walker,"allocator" => .ok [.allocator]
            | .walker,"value" => .ok [.input]
            | .walker,"path" => .ok [.path state.path]
            | .walker,"set" => .ok [.fieldSet]
            | .typeInfo,"ElementRelationship" => .ok [.relation state.atomic]
            | .schemaPackage,"Atomic" => .ok [.relation true]
            | _,_ => .error "Unsupported typed field-set selection"
        | .neNil expression => do
            .ok [.flag (!(isNil (← one ops fuel expression state)))]
        | .eqNil expression => do
            .ok [.flag (isNil (← one ops fuel expression state))]
        | .notFlag expression => do
            match ← one ops fuel expression state with
            | .flag flag => .ok [.flag (!flag)]
            | _ => .error "Typed negation requires a boolean"
        | .equalScalars left right => do
            match ← one ops fuel left state, ← one ops fuel right state with
            | .relation a,.relation b => .ok [.flag (a == b)]
            | _,_ => .error "Typed relation equality requires schema relationships"
        | .call receiver method arguments => do
            match ← one ops fuel receiver state,method,← many ops fuel arguments state with
            | .coercion kind,"Invoke",[.allocator,.input] =>
                .ok [resource (ops.view kind state.input),.nil]
            | _,_,_ => .error "Unmodeled typed visitor call"
        | _ => .error "Unsupported typed field-set expression"
  termination_by fuel _ _ => (fuel,0)

  def one (ops : Ops Obj) (fuel : Nat) (expression : Expr) (state : Machine Obj α) : Except String (Value α) := do
    match ← eval ops fuel expression state with
    | [value] => .ok value
    | _ => .error "Typed field-set expression requires exactly one result"
  termination_by (fuel,1)

  def many (ops : Ops Obj) : Nat → List Expr → Machine Obj α → Except String (List (Value α))
    | 0,_,_ => .error "Typed argument fuel exhausted"
    | _+1,[],_ => .ok []
    | fuel+1,head::rest,state => do
        .ok ((← eval ops fuel head state) ++ (← many ops fuel rest state))
  termination_by fuel _ _ => (fuel,2)
end

def assign (names : List String) (values : List (Value α)) (state : Machine Obj α) : Except String (Machine Obj α) :=
  match names,values with
  | ["list","_"],[.nil,_] | ["m","_"],[.nil,_] => .ok {state with resourceLocal := none}
  | ["list","_"],[.container handle,_] | ["m","_"],[.container handle,_] =>
      .ok {state with resourceLocal := some handle}
  | ["errs"],[.nil] => .ok state
  | _,_ => .error "Unsupported typed field-set assignment"

/-- The field-set projection uses membership; repeated insertion need not preserve list multiplicity. -/
def insert (fields : List (List α)) (path : List α) : List (List α) :=
  if path.isEmpty then fields else path::fields

def effect (ops : Ops Obj) (expression : Expr) (state : Machine Obj α) : Except String (Machine Obj α) := do
  match expression with
  | .call receiver "Insert" [argument] =>
      match ← one ops 32 receiver state, ← one ops 32 argument state with
      | .fieldSet,.path path => .ok {state with fields := insert state.fields path}
      | _,_ => .error "Typed Insert requires the walker field set and a path"
  | _ => .error "Unsupported typed field-set effect"

/-- Go evaluates the receiver and arguments when defer is registered, not when it runs. -/
def captureFree (ops : Ops Obj) (expression : Expr) (state : Machine Obj α) : Except String (Machine Obj α) := do
  match expression with
  | .call receiver "Free" [argument] =>
      match ← one ops 32 receiver state, ← one ops 32 argument state with
      | .allocator,.container handle => .ok {state with deferred := handle::state.deferred}
      | _,_ => .error "Typed Free requires an allocator and non-nil captured resource"
  | _ => .error "Unsupported typed deferred call"

inductive Result (Obj α : Type) where
  | next (state : Machine Obj α)
  | returned (values : List (Value α)) (state : Machine Obj α)

def exec (ops : Ops Obj) : Nat → List Stmt → Machine Obj α → Except String (Result Obj α)
  | 0,_,_ => .error "Typed statement fuel exhausted"
  | _+1,[],state => .ok (.next state)
  | fuel+1,statement::rest,state =>
      match statement with
      | .assign names expressions => do
          let next ← assign names (← many ops 64 expressions state) state
          exec ops fuel rest next
      | .branch condition body => do
          match ← one ops 64 condition state with
          | .flag true =>
              match ← exec ops fuel body state with
              | .returned values final => .ok (.returned values final)
              | .next next => exec ops fuel rest next
          | .flag false => exec ops fuel rest state
          | _ => .error "Typed field-set branch requires a boolean"
      | .effect call => do
          let next ← effect ops call state
          exec ops fuel rest next
      | .deferCall call => do
          let next ← captureFree ops call state
          exec ops fuel rest next
      | .ret expressions => do
          .ok (.returned (← many ops 64 expressions state) state)
      | _ => .error "Unsupported typed field-set statement"

def finish (state : Machine Obj α) : Machine Obj α :=
  {state with freed := state.freed ++ state.deferred,deferred := []}

/-- Allocator cleanup is observed as a LIFO trace; general heap/runtime realization remains separate. -/
def run (ops : Ops Obj) (program : List Stmt) (state : Machine Obj α) : Except String (Result Obj α) := do
  match ← exec ops 12 program state with
  | .returned values final => .ok (.returned values (finish final))
  | .next _ => .error "Typed visitor unexpectedly fell through without returning"

end SSA.AtomicFieldSubset
