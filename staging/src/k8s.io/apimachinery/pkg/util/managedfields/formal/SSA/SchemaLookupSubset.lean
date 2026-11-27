/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SchemaLookupSource
import SSA.SchemaDispatchMachine

namespace SSA.SchemaLookupSubset

open RawGoSyntax (Node field find)
open AtomicRemoveSubset (atom nodes)
abbrev Atom := SchemaDispatchMachine.Atom

structure Reference (M S L R Rel : Type) where
  original : R
  named : Option String := none
  inlined : Atom M S L := {}
  relationship : Option Rel := none

structure Ops (M S L R Rel : Type) where
  findNamed : String → Atom M S L × Bool
  plain : Reference M S L R Rel → Except String (Atom M S L × Bool) :=
    fun _ => .error "Source no-override lookup is not configured"

inductive Value (M S L R Rel : Type) where
  | schema | nil
  | reference (value : Reference M S L R Rel)
  | atom (value : Atom M S L)
  | typeDef (value : Atom M S L)
  | text (value : String)
  | stringPointer (value : String)
  | relationshipPointer (value : Rel)
  | flag (value : Bool)

structure Machine (M S L R Rel : Type) where
  reference : Reference M S L R Rel
  result : Atom M S L := {}
  typeDef : Atom M S L := {}
  found : Bool := false

def read (state : Machine M S L R Rel) : String → Except String (Value M S L R Rel)
  | "s" => .ok .schema
  | "tr" => .ok (.reference state.reference)
  | "result" => .ok (.atom state.result)
  | "t" => .ok (.typeDef state.typeDef)
  | "ok" => .ok (.flag state.found)
  | "nil" => .ok .nil
  | "true" => .ok (.flag true)
  | "false" => .ok (.flag false)
  | _ => .error "Unbound schema lookup variable"

def select : Value M S L R Rel → String → Except String (Value M S L R Rel)
  | .reference value,"NamedType" => .ok (match value.named with | none => .nil | some name => .stringPointer name)
  | .reference value,"Inlined" => .ok (.atom value.inlined)
  | .reference value,"ElementRelationship" => .ok (match value.relationship with | none => .nil | some rel => .relationshipPointer rel)
  | .typeDef value,"Atom" => .ok (.atom value)
  | _,_ => .error "Unmodeled schema lookup selection"

def isNil : Value M S L R Rel → Except String Bool
  | .nil => .ok true
  | .stringPointer _ | .relationshipPointer _ => .ok false
  | _ => .error "Unmodeled schema lookup nil comparison"

def call (ops : Ops M S L R Rel) : Value M S L R Rel → String → List (Value M S L R Rel) →
    Except String (List (Value M S L R Rel))
  | .schema,"FindNamedType",[.text name] =>
      let (result,found) := ops.findNamed name
      .ok [.typeDef result,.flag found]
  | .schema,"resolveNoOverrides",[.reference reference] => do
      let (result,found) ← ops.plain reference
      .ok [.atom result,.flag found]
  | _,_,_ => .error "Unmodeled schema lookup call"

def write (state : Machine M S L R Rel) : String → Value M S L R Rel → Except String (Machine M S L R Rel)
  | "result",.atom value => .ok {state with result := value}
  | "t",.typeDef value => .ok {state with typeDef := value}
  | "ok",.flag value => .ok {state with found := value}
  | _,_ => .error "Unmodeled schema lookup assignment"

def assign : List String → List (Value M S L R Rel) → Machine M S L R Rel → Except String (Machine M S L R Rel)
  | [],[],state => .ok state
  | name::names,value::values,state => do assign names values (← write state name value)
  | _,_,_ => .error "Schema lookup assignment arity mismatch"

mutual
  def eval (ops : Ops M S L R Rel) : Nat → Node → Machine M S L R Rel → Except String (List (Value M S L R Rel))
    | 0,_,_ => .error "Schema lookup expression syntax fuel exhausted"
    | fuel+1,expression,state => do
        match expression with
        | .record "Ident" _ => .ok [← SchemaLookupSubset.read state (← atom (field expression "Name"))]
        | .record "ParenExpr" _ => eval ops fuel (field expression "X") state
        | .record "SelectorExpr" _ =>
            .ok [← select (← one ops fuel (field expression "X") state) (← atom (field (field expression "Sel") "Name"))]
        | .record "StarExpr" _ =>
            match ← one ops fuel (field expression "X") state with
            | .stringPointer name => .ok [.text name]
            | _ => .error "Schema lookup requires a non-nil name pointer"
        | .record "UnaryExpr" _ =>
            match field expression "Op",← one ops fuel (field expression "X") state with
            | .atom "!",.flag value => .ok [.flag (!value)]
            | _,_ => .error "Unmodeled schema lookup unary operation"
        | .record "BinaryExpr" _ =>
            match field expression "Op",← one ops fuel (field expression "X") state,
                ← one ops fuel (field expression "Y") state with
            | .atom "==",value,.nil => .ok [.flag (← isNil value)]
            | .atom "!=",value,.nil => .ok [.flag (!(← isNil value))]
            | _,_,_ => .error "Unmodeled schema lookup binary operation"
        | .record "CompositeLit" _ =>
            match field expression "Type",field expression "Elts" with
            | .record "Ident" fields,.sequence [] =>
                match find fields "Name" with
                | .atom "Atom" => .ok [.atom {}]
                | _ => .error "Unmodeled schema lookup constructor"
            | _,_ => .error "Unmodeled schema lookup constructor fields"
        | .record "CallExpr" _ =>
            match field expression "Fun",field expression "Ellipsis" with
            | .record "SelectorExpr" fields,.bit false =>
                call ops (← one ops fuel (find fields "X") state)
                  (← atom (field (find fields "Sel") "Name")) (← many ops fuel (← nodes (field expression "Args")) state)
            | _,_ => .error "Unmodeled schema lookup call syntax"
        | _ => .error "Unmodeled schema lookup expression syntax"
  termination_by fuel _ _ => (fuel,0)

  def one (ops : Ops M S L R Rel) (fuel : Nat) (expression : Node) (state : Machine M S L R Rel) :
      Except String (Value M S L R Rel) := do
    match ← eval ops fuel expression state with
    | [value] => .ok value
    | _ => .error "Schema lookup expression requires one value"
  termination_by (fuel,1)

  def many (ops : Ops M S L R Rel) : Nat → List Node → Machine M S L R Rel → Except String (List (Value M S L R Rel))
    | 0,_,_ => .error "Schema lookup argument syntax fuel exhausted"
    | _+1,[],_ => .ok []
    | fuel+1,first::rest,state => do .ok ((← eval ops fuel first state) ++ (← many ops fuel rest state))
  termination_by fuel _ _ => (fuel,2)
end

inductive Result (M S L R Rel : Type) where
  | next (state : Machine M S L R Rel)
  | returned (values : List (Value M S L R Rel))

def names : List Node → Except String (List String)
  | [] => .ok []
  | (.record "Ident" fields)::rest => do .ok ((← atom (find fields "Name"))::(← names rest))
  | _ => .error "Schema lookup assignments require local names"

mutual
  def exec (ops : Ops M S L R Rel) : Nat → Node → Machine M S L R Rel → Except String (Result M S L R Rel)
    | 0,_,_ => .error "Schema lookup statement syntax fuel exhausted"
    | fuel+1,statement,state => do
        match statement with
        | .absent => .ok (.next state)
        | .record "BlockStmt" _ => execList ops fuel (← nodes (field statement "List")) state
        | .record "AssignStmt" _ =>
            match field statement "Tok" with
            | .atom ":=" | .atom "=" =>
                .ok (.next (← assign (← names (← nodes (field statement "Lhs")))
                  (← many ops 64 (← nodes (field statement "Rhs")) state) state))
            | _ => .error "Unmodeled schema lookup assignment token"
        | .record "IfStmt" _ =>
            match field statement "Init" with
            | .absent =>
                match ← one ops 64 (field statement "Cond") state with
                | .flag value =>
                    if value then exec ops fuel (field statement "Body") state
                    else exec ops fuel (field statement "Else") state
                | _ => .error "Schema lookup conditional requires a flag"
            | _ => .error "Unmodeled schema lookup conditional initializer"
        | .record "ReturnStmt" _ => .ok (.returned (← many ops 64 (← nodes (field statement "Results")) state))
        | _ => .error "Unmodeled schema lookup statement syntax"
  termination_by fuel _ _ => (fuel,0)

  def execList (ops : Ops M S L R Rel) : Nat → List Node → Machine M S L R Rel → Except String (Result M S L R Rel)
    | 0,_,_ => .error "Schema lookup block syntax fuel exhausted"
    | _+1,[],state => .ok (.next state)
    | fuel+1,first::rest,state => do
        match ← exec ops fuel first state with
        | .returned values => .ok (.returned values)
        | .next next => execList ops fuel rest next
  termination_by fuel _ _ => (fuel,1)
end

def invoke (ops : Ops M S L R Rel) (program : Node) (reference : Reference M S L R Rel) : Except String (Atom M S L × Bool) := do
  match ← exec ops 32 program {reference} with
  | .returned [.atom value,.flag found] => .ok (value,found)
  | _ => .error "Schema lookup source did not return an atom/found pair"

end SSA.SchemaLookupSubset
