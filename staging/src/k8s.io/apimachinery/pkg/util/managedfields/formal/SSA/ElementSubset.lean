/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.GoSubset

namespace SSA.ElementSubset

open GoSubset

/-- Element-set methods read two inputs and append to a fresh output; input storage is immutable. -/
structure Machine (α : Type) where
  left : List α
  right : List α
  output : List α := []
  i : Nat := 0
  j : Nat := 0

inductive Value (α : Type) where
  | set (members : List α)
  | members (values : List α)
  | element (value : α)
  | number (value : Nat)
  | flag (value : Bool)

def read (state : Machine α) : String → Except String (Value α)
  | "s" => .ok (.set state.left)
  | "s2" => .ok (.set state.right)
  | "out" => .ok (.set state.output)
  | "i" => .ok (.number state.i)
  | "j" => .ok (.number state.j)
  | _ => .error "Unsupported element-set variable"

def write (state : Machine α) (name : String) (value : Value α) : Except String (Machine α) :=
  match name,value with
  | "out",.set members => .ok {state with output := members}
  | "i",.number i => .ok {state with i}
  | "j",.number j => .ok {state with j}
  | _,_ => .error "Unsupported element-set assignment"

def eval (lt : α → α → Bool) : Nat → Expr → Machine α → Except String (Value α)
  | 0,_,_ => .error "Element-set expression fuel exhausted"
  | fuel+1,expression,state =>
      match expression with
      | .var name => read state name
      | .number number => .ok (.number number)
      | .emptyElementSet => .ok (.set [])
      | .select receiver "members" => do
          match ← eval lt fuel receiver state with
          | .set members => .ok (.members members)
          | _ => .error "Element-set member selection requires a set"
      | .length receiver => do
          match ← eval lt fuel receiver state with
          | .members members => .ok (.number members.length)
          | _ => .error "Element-set length requires a slice"
      | .less left right => do
          match ← eval lt fuel left state, ← eval lt fuel right state with
          | .number left,.number right => .ok (.flag (left < right))
          | _,_ => .error "Element-set index comparison requires numbers"
      | .index receiver index => do
          match ← eval lt fuel receiver state, ← eval lt fuel index state with
          | .members members,.number index =>
              match members[index]? with
              | some element => .ok (.element element)
              | none => .error "Go element-set index out of bounds"
          | _,_ => .error "Element-set indexing requires a slice and number"
      | .sliceFrom receiver index => do
          match ← eval lt fuel receiver state, ← eval lt fuel index state with
          | .members members,.number index =>
              if index ≤ members.length then .ok (.members (members.drop index))
              else .error "Go element-set slice out of bounds"
          | _,_ => .error "Element-set slicing requires a slice and number"
      | .appendOne receiver value => do
          match ← eval lt fuel receiver state, ← eval lt fuel value state with
          | .members members,.element element => .ok (.members (members ++ [element]))
          | _,_ => .error "Element-set append requires a slice and element"
      | .concatenate left right => do
          match ← eval lt fuel left state, ← eval lt fuel right state with
          | .members left,.members right => .ok (.members (left ++ right))
          | _,_ => .error "Element-set variadic append requires two slices"
      | .call receiver "Less" [argument] => do
          match ← eval lt fuel receiver state, ← eval lt fuel argument state with
          | .element left,.element right => .ok (.flag (lt left right))
          | _,_ => .error "PathElement.Less requires two elements"
      | .andAlso left right => do
          match ← eval lt fuel left state with
          | .flag false => .ok (.flag false)
          | .flag true =>
              match ← eval lt fuel right state with
              | .flag value => .ok (.flag value)
              | _ => .error "Element-set conjunction requires booleans"
          | _ => .error "Element-set conjunction requires booleans"
      | .notFlag expression => do
          match ← eval lt fuel expression state with
          | .flag value => .ok (.flag (!value))
          | _ => .error "Element-set negation requires a boolean"
      | _ => .error "Unsupported element-set expression"

def evalAll (lt : α → α → Bool) : List Expr → Machine α → Except String (List (Value α))
  | [],_ => .ok []
  | expression::rest,state => do
      let value ← eval lt 32 expression state
      let values ← evalAll lt rest state
      .ok (value::values)

def assign : List String → List (Value α) → Machine α → Except String (Machine α)
  | [],[],state => .ok state
  | name::names,value::values,state => do
      let next ← write state name value
      assign names values next
  | _,_,_ => .error "Element-set assignment arity mismatch"

inductive Result (α : Type) where
  | next (state : Machine α)
  | returned (values : List (Value α))

/-- Fuel bounds executed loop iterations as well as nesting; the source theorem supplies a size bound. -/
def exec (lt : α → α → Bool) : Nat → List Stmt → Machine α → Except String (Result α)
  | 0,_,_ => .error "Element-set statement fuel exhausted"
  | _+1,[],state => .ok (.next state)
  | fuel+1,statement::rest,state =>
      match statement with
      | .assign names expressions => do
          let values ← evalAll lt expressions state
          let next ← assign names values state
          exec lt fuel rest next
      | .fieldSet "out" "members" expression => do
          match ← eval lt 32 expression state with
          | .members output => exec lt fuel rest {state with output}
          | _ => .error "Element-set field assignment requires members"
      | .increment name => do
          match ← read state name with
          | .number number =>
              let next ← write state name (.number (number+1))
              exec lt fuel rest next
          | _ => .error "Element-set increment requires a number"
      | .branch condition body => do
          match ← eval lt 32 condition state with
          | .flag true =>
              match ← exec lt fuel body state with
              | .returned values => .ok (.returned values)
              | .next next => exec lt fuel rest next
          | .flag false => exec lt fuel rest state
          | _ => .error "Element-set conditional requires a boolean"
      | .branchElse condition body alternative => do
          match ← eval lt 32 condition state with
          | .flag condition =>
              match ← exec lt fuel (if condition then body else alternative) state with
              | .returned values => .ok (.returned values)
              | .next next => exec lt fuel rest next
          | _ => .error "Element-set conditional requires a boolean"
      | .whileLoop condition body => do
          match ← eval lt 32 condition state with
          | .flag true =>
              match ← exec lt fuel body state with
              | .returned values => .ok (.returned values)
              | .next next => exec lt fuel (.whileLoop condition body::rest) next
          | .flag false => exec lt fuel rest state
          | _ => .error "Element-set loop requires a boolean"
      | .ret expressions => do
          let values ← evalAll lt expressions state
          .ok (.returned values)
      | _ => .error "Unsupported element-set statement"

end SSA.ElementSubset
