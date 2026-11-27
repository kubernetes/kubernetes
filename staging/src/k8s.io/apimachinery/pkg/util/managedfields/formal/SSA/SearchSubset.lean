/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SearchSource
import SSA.HasSource

namespace SSA.SearchSubset

open GoSubset

structure Ops (α : Type) where
  compare : α → α → Int
  wordBits : Nat := 64
  search : List α → α → Except String (Nat × Bool) :=
    fun _ _ => .error "BinarySearchFunc callback is not configured"

structure Machine (α : Type) where
  members : List α
  target : α
  other : α
  n : Nat := 0
  i : Nat := 0
  j : Nat := 0
  h : Nat := 0
  found : Bool := false

inductive Value (α : Type) where
  | set (members : List α)
  | array (members : List α)
  | element (label : α)
  | number (value : Nat)
  | integer (value : Int)
  | flag (value : Bool)
  | slicesPackage
  | comparator

def read (state : Machine α) : String → Except String (Value α)
  | "x" => .ok (.array state.members)
  | "s" => .ok (.set state.members)
  | "target" | "pe" | "e" => .ok (.element state.target)
  | "rhs" => .ok (.element state.other)
  | "n" => .ok (.number state.n)
  | "i" => .ok (.number state.i)
  | "j" => .ok (.number state.j)
  | "h$1" => .ok (.number state.h)
  | "found" => .ok (.flag state.found)
  | "slices" => .ok .slicesPackage
  | "cmp" => .ok .comparator
  | _ => .error "Unsupported search variable"

def write (state : Machine α) (name : String) (value : Value α) : Except String (Machine α) :=
  match name,value with
  | "_",_ => .ok state
  | "n",.number n => .ok {state with n}
  | "i",.number i => .ok {state with i}
  | "j",.number j => .ok {state with j}
  | "h$1",.number h => .ok {state with h}
  | "found",.flag found => .ok {state with found}
  | _,_ => .error "Search assignments require a supported mutable scalar"

/-- Converting an overflowing Go sum to uint preserves its residue before the unsigned shift. -/
def unsignedMidpoint (bits left right : Nat) : Nat := ((left+right) % (2^bits)) / 2

def maxIndex (bits : Nat) : Nat := 2^(bits-1)-1

def scalarEqual : Value α → Value α → Except String Bool
  | .number a,.number b => .ok (a == b)
  | .integer a,.integer b => .ok (a == b)
  | .integer a,.number b => .ok (a == Int.ofNat b)
  | .number a,.integer b => .ok (Int.ofNat a == b)
  | .flag a,.flag b => .ok (a == b)
  | _,_ => .error "Search equality requires two scalar values"

mutual
  def eval (ops : Ops α) : Nat → Expr → Machine α → Except String (List (Value α))
    | 0,_,_ => .error "Search expression fuel exhausted"
    | fuel+1,expression,state =>
        match expression with
        | .var name => do .ok [← read state name]
        | .number value => .ok [.number value]
        | .flag value => .ok [.flag value]
        | .pathComparator => .ok [.comparator]
        | .select receiver "members" => do
            match ← one ops fuel receiver state with
            | .set members => .ok [.array members]
            | _ => .error "Has member selection requires a set"
        | .length receiver => do
            match ← one ops fuel receiver state with
            | .array members => .ok [.number members.length]
            | _ => .error "Search length requires an array"
        | .index receiver index => do
            match ← one ops fuel receiver state, ← one ops fuel index state with
            | .array members,.number index =>
                match members[index]? with
                | some label => .ok [.element label]
                | none => .error "Go search index out of bounds"
            | _,_ => .error "Search indexing requires an array and natural index"
        | .less left right => do
            match ← one ops fuel left state, ← one ops fuel right state with
            | .number a,.number b => .ok [.flag (a < b)]
            | .integer a,.number b => .ok [.flag (a < Int.ofNat b)]
            | .number a,.integer b => .ok [.flag (Int.ofNat a < b)]
            | .integer a,.integer b => .ok [.flag (a < b)]
            | _,_ => .error "Search comparison requires numbers"
        | .addNumbers left right => do
            match ← one ops fuel left state, ← one ops fuel right state with
            | .number a,.number b => .ok [.number (a+b)]
            | _,_ => .error "Search index addition requires natural indices"
        | .unsignedMidpoint left right => do
            match ← one ops fuel left state, ← one ops fuel right state with
            | .number a,.number b => .ok [.number (unsignedMidpoint ops.wordBits a b)]
            | _,_ => .error "Search midpoint requires natural bounds"
        | .equalScalars left right => do
            let same ← scalarEqual (← one ops fuel left state) (← one ops fuel right state)
            .ok [.flag same]
        | .andAlso left right => do
            match ← one ops fuel left state with
            | .flag false => .ok [.flag false]
            | .flag true =>
                match ← one ops fuel right state with
                | .flag result => .ok [.flag result]
                | _ => .error "Search conjunction requires booleans"
            | _ => .error "Search conjunction requires booleans"
        | .notFlag expression => do
            match ← one ops fuel expression state with
            | .flag flag => .ok [.flag (!flag)]
            | _ => .error "Search negation requires a boolean"
        | .call receiver method arguments => do
            match ← one ops fuel receiver state,method,← many ops fuel arguments state with
            | .comparator,"Invoke",[.element a,.element b] => .ok [.integer (ops.compare a b)]
            | .element a,"Compare",[.element b] => .ok [.integer (ops.compare a b)]
            | .slicesPackage,"BinarySearchFunc",[.array members,.element target,.comparator] => do
                let (index,found) ← ops.search members target
                .ok [.number index,.flag found]
            | _,_,_ => .error "Unsupported search call"
        | _ => .error "Unsupported search expression"
  termination_by fuel _ _ => (fuel,0)

  def one (ops : Ops α) (fuel : Nat) (expression : Expr) (state : Machine α) : Except String (Value α) := do
    match ← eval ops fuel expression state with
    | [value] => .ok value
    | _ => .error "Search expression must return exactly one value"
  termination_by (fuel,1)

  def many (ops : Ops α) : Nat → List Expr → Machine α → Except String (List (Value α))
    | 0,_,_ => .error "Search argument fuel exhausted"
    | _+1,[],_ => .ok []
    | fuel+1,expression::rest,state => do
        let values ← eval ops fuel expression state
        let remaining ← many ops fuel rest state
        .ok (values ++ remaining)
  termination_by fuel _ _ => (fuel,2)
end

def assign : List String → List (Value α) → Machine α → Except String (Machine α)
  | [],[],state => .ok state
  | name::names,value::values,state => do
      let next ← write state name value
      assign names values next
  | _,_,_ => .error "Search assignment arity mismatch"

inductive Result (α : Type) where
  | next (state : Machine α)
  | returned (values : List (Value α))

def exec (ops : Ops α) : Nat → List Stmt → Machine α → Except String (Result α)
  | 0,_,_ => .error "Search statement fuel exhausted"
  | _+1,[],state => .ok (.next state)
  | fuel+1,statement::rest,state =>
      match statement with
      | .assign names expressions => do
          let values ← many ops 64 expressions state
          let next ← assign names values state
          exec ops fuel rest next
      | .branch condition body => do
          match ← one ops 64 condition state with
          | .flag true =>
              match ← exec ops fuel body state with
              | .returned values => .ok (.returned values)
              | .next next => exec ops fuel rest next
          | .flag false => exec ops fuel rest state
          | _ => .error "Search conditional requires a boolean"
      | .branchElse condition body alternative => do
          match ← one ops 64 condition state with
          | .flag flag =>
              match ← exec ops fuel (if flag then body else alternative) state with
              | .returned values => .ok (.returned values)
              | .next next => exec ops fuel rest next
          | _ => .error "Search conditional requires a boolean"
      | .whileLoop condition body => do
          match ← one ops 64 condition state with
          | .flag true =>
              match ← exec ops fuel body state with
              | .returned values => .ok (.returned values)
              | .next next => exec ops fuel (.whileLoop condition body::rest) next
          | .flag false => exec ops fuel rest state
          | _ => .error "Search loop requires a boolean"
      | .ret expressions => do
          .ok (.returned (← many ops 64 expressions state))
      | _ => .error "Unsupported search statement"

end SSA.SearchSubset
