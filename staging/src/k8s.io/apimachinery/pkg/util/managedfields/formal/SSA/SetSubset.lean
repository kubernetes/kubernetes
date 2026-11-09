/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.RecursiveSets
import SSA.SetSource

namespace SSA.SetSubset

open GoSubset

open RecursiveSets

/-- Only freshly allocated output nodes are mutable; source input aliases are rejected by the importer. -/
structure Machine (α : Type) where
  left : Tree α
  right : Tree α
  nodeMode : Bool := true
  output : List (α × Tree α) := []
  child : Tree α := .node [] []
  i : Nat := 0
  j : Nat := 0

inductive Value (α : Type) where
  | tree (value : Tree α)
  | nodes (values : List (α × Tree α))
  | members (values : List (α × Tree α))
  | elements (values : List α)
  | node (value : α × Tree α)
  | label (value : α)
  | number (value : Nat)
  | flag (value : Bool)

structure Ops (α : Type) where
  less : α → α → Bool
  isEmpty : Tree α → Bool
  difference : Tree α → Tree α → Tree α
  nodesDifference : List (α × Tree α) → List (α × Tree α) → List (α × Tree α)

def read (state : Machine α) : String → Except String (Value α)
  | "s" => .ok (if state.nodeMode then .nodes state.left.children else .tree state.left)
  | "s2" => .ok (.tree state.right)
  | "out" => .ok (.nodes state.output)
  | "diff$1" => .ok (.tree state.child)
  | "i" => .ok (.number state.i)
  | "j" => .ok (.number state.j)
  | _ => .error "Unsupported recursive-set variable"

def write (state : Machine α) (name : String) (value : Value α) : Except String (Machine α) :=
  match name,value with
  | "out",.nodes output => .ok {state with output}
  | "diff$1",.tree child => .ok {state with child}
  | "i",.number i => .ok {state with i}
  | "j",.number j => .ok {state with j}
  | _,_ => .error "Unsupported recursive-set assignment"

/-- Pointers are observational here: supported code mutates only its fresh output and never compares addresses. -/
def eval (ops : Ops α) : Nat → Expr → Machine α → Except String (Value α)
  | 0,_,_ => .error "Recursive-set expression fuel exhausted"
  | fuel+1,expression,state =>
      match expression with
      | .var name => read state name
      | .number number => .ok (.number number)
      | .emptyNodeMap => .ok (.nodes [])
      | .addressOf value | .dereference value => eval ops fuel value state
      | .setRecord members children => do
          match ← eval ops fuel members state, ← eval ops fuel children state with
          | .elements members,.nodes children => .ok (.tree (.node members children))
          | _,_ => .error "Set construction requires members and child nodes"
      | .setNode label child => do
          match ← eval ops fuel label state, ← eval ops fuel child state with
          | .label label,.tree child => .ok (.node (label,child))
          | _,_ => .error "Child construction requires a label and set"
      | .select receiver field => do
          match ← eval ops fuel receiver state,field with
          | .tree tree,"Members" => .ok (.elements tree.members)
          | .tree tree,"Children" => .ok (.nodes tree.children)
          | .nodes nodes,"members" => .ok (.members nodes)
          | .node child,"pathElement" => .ok (.label child.1)
          | .node child,"set" => .ok (.tree child.2)
          | _,_ => .error "Unsupported recursive-set field selection"
      | .length receiver => do
          match ← eval ops fuel receiver state with
          | .members members => .ok (.number members.length)
          | _ => .error "Recursive-set length requires a slice"
      | .less left right => do
          match ← eval ops fuel left state, ← eval ops fuel right state with
          | .number left,.number right => .ok (.flag (left < right))
          | _,_ => .error "Recursive-set index comparison requires numbers"
      | .index receiver index => do
          match ← eval ops fuel receiver state, ← eval ops fuel index state with
          | .members members,.number index =>
              match members[index]? with
              | some child => .ok (.node child)
              | none => .error "Go child-node index out of bounds"
          | _,_ => .error "Recursive-set indexing requires a slice and number"
      | .sliceFrom receiver index => do
          match ← eval ops fuel receiver state, ← eval ops fuel index state with
          | .members members,.number index =>
              if index ≤ members.length then .ok (.members (members.drop index))
              else .error "Go child-node slice out of bounds"
          | _,_ => .error "Recursive-set slicing requires a slice and number"
      | .appendOne receiver value => do
          match ← eval ops fuel receiver state, ← eval ops fuel value state with
          | .members members,.node child => .ok (.members (members ++ [child]))
          | _,_ => .error "Recursive-set append requires a slice and child"
      | .concatenate left right => do
          match ← eval ops fuel left state, ← eval ops fuel right state with
          | .members left,.members right => .ok (.members (left ++ right))
          | _,_ => .error "Recursive-set variadic append requires two slices"
      | .call receiver method arguments => do
          let receiver ← eval ops fuel receiver state
          match receiver,method,arguments with
          | .tree tree,"Empty",[] => .ok (.flag (ops.isEmpty tree))
          | receiver,"Difference",[argument] =>
              match receiver,← eval ops fuel argument state with
              | .elements left,.elements right => .ok (.elements (ElementAlgebra.compute .difference ops.less left right))
              | .tree left,.tree right => .ok (.tree (ops.difference left right))
              | .nodes left,.tree right => .ok (.nodes (ops.nodesDifference left right.children))
              | _,_ => .error "Unsupported recursive-set Difference receivers"
          | .label left,"Less",[argument] =>
              match ← eval ops fuel argument state with
              | .label right => .ok (.flag (ops.less left right))
              | _ => .error "PathElement.Less requires two labels"
          | _,_,_ => .error "Unsupported recursive-set call"
      | .andAlso left right => do
          match ← eval ops fuel left state with
          | .flag false => .ok (.flag false)
          | .flag true =>
              match ← eval ops fuel right state with
              | .flag value => .ok (.flag value)
              | _ => .error "Recursive-set conjunction requires booleans"
          | _ => .error "Recursive-set conjunction requires booleans"
      | .notFlag expression => do
          match ← eval ops fuel expression state with
          | .flag value => .ok (.flag (!value))
          | _ => .error "Recursive-set negation requires a boolean"
      | _ => .error "Unsupported recursive-set expression"

def evalAll (ops : Ops α) : List Expr → Machine α → Except String (List (Value α))
  | [],_ => .ok []
  | expression::rest,state => do
      let value ← eval ops 32 expression state
      let values ← evalAll ops rest state
      .ok (value::values)

def assign : List String → List (Value α) → Machine α → Except String (Machine α)
  | [],[],state => .ok state
  | name::names,value::values,state => do
      let next ← write state name value
      assign names values next
  | _,_,_ => .error "Recursive-set assignment arity mismatch"

inductive Result (α : Type) where
  | next (state : Machine α)
  | returned (values : List (Value α))

/-- Fuel bounds executed loop iterations as well as nesting; the source theorem supplies a size bound. -/
def exec (ops : Ops α) : Nat → List Stmt → Machine α → Except String (Result α)
  | 0,_,_ => .error "Recursive-set statement fuel exhausted"
  | _+1,[],state => .ok (.next state)
  | fuel+1,statement::rest,state =>
      match statement with
      | .assign names expressions => do
          let values ← evalAll ops expressions state
          let next ← assign names values state
          exec ops fuel rest next
      | .fieldSet "out" "members" expression => do
          match ← eval ops 32 expression state with
          | .members output => exec ops fuel rest {state with output}
          | _ => .error "Recursive-set field assignment requires members"
      | .increment name => do
          match ← read state name with
          | .number number =>
              let next ← write state name (.number (number+1))
              exec ops fuel rest next
          | _ => .error "Recursive-set increment requires a number"
      | .branch condition body => do
          match ← eval ops 32 condition state with
          | .flag true =>
              match ← exec ops fuel body state with
              | .returned values => .ok (.returned values)
              | .next next => exec ops fuel rest next
          | .flag false => exec ops fuel rest state
          | _ => .error "Recursive-set conditional requires a boolean"
      | .branchElse condition body alternative => do
          match ← eval ops 32 condition state with
          | .flag condition =>
              match ← exec ops fuel (if condition then body else alternative) state with
              | .returned values => .ok (.returned values)
              | .next next => exec ops fuel rest next
          | _ => .error "Recursive-set conditional requires a boolean"
      | .whileLoop condition body => do
          match ← eval ops 32 condition state with
          | .flag true =>
              match ← exec ops fuel body state with
              | .returned values => .ok (.returned values)
              | .next next => exec ops fuel (.whileLoop condition body::rest) next
          | .flag false => exec ops fuel rest state
          | _ => .error "Recursive-set loop requires a boolean"
      | .ret expressions => do
          let values ← evalAll ops expressions state
          .ok (.returned values)
      | _ => .error "Unsupported recursive-set statement"

end SSA.SetSubset
