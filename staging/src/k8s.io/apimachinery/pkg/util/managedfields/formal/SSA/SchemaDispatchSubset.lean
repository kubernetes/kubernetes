/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SchemaDispatchMachine

namespace SSA.SchemaDispatchSubset

open RawGoSyntax (Node field find)
open AtomicRemoveSubset (atom nodes)
open SchemaDispatchMachine

mutual
  def eval (ops : Ops Obj M S L R H Err) : Nat → Node → Machine Obj M S L R H →
      Except String (List (Value Obj M S L R Err) × Machine Obj M S L R H)
    | 0,_,_ => .error "Schema expression syntax fuel exhausted"
    | fuel+1,expression,state => do
        match expression with
        | .record "Ident" _ => .ok ([← SchemaDispatchMachine.read state (← atom (field expression "Name"))],state)
        | .record "ParenExpr" _ => eval ops fuel (field expression "X") state
        | .record "SelectorExpr" _ =>
            let (receiver,after) ← one ops fuel (field expression "X") state
            .ok ([← select receiver (← atom (field (field expression "Sel") "Name"))],after)
        | .record "BasicLit" _ =>
            match field expression "Kind" with
            | .atom "STRING" => .ok ([.text (← atom (field expression "DecodedString"))],state)
            | _ => .error "Unmodeled schema literal"
        | .record "StarExpr" _ =>
            let (value,after) ← one ops fuel (field expression "X") state
            match value with
            | .stringPointer value => .ok ([.text value],after)
            | _ => .error "Schema dereference requires a non-nil string pointer"
        | .record "UnaryExpr" _ =>
            let (value,after) ← one ops fuel (field expression "X") state
            match field expression "Op",value with
            | .atom "!",.flag value => .ok ([.flag (!value)],after)
            | .atom "*",.stringPointer value => .ok ([.text value],after)
            | _,_ => .error "Unmodeled schema unary operation"
        | .record "BinaryExpr" _ =>
            let (left,afterLeft) ← one ops fuel (field expression "X") state
            let (right,afterRight) ← one ops fuel (field expression "Y") afterLeft
            match field expression "Op",left,right with
            | .atom "==",value,.nil => .ok ([.flag (← isNil value)],afterRight)
            | .atom "!=",value,.nil => .ok ([.flag (!(← isNil value))],afterRight)
            | .atom "+",.text left,.text right => .ok ([.text (left ++ right)],afterRight)
            | _,_,_ => .error "Unmodeled schema binary operation"
        | .record "CompositeLit" _ =>
            match field expression "Type" with
            | .record "SelectorExpr" fields =>
                match find fields "X",field (find fields "Sel") "Name" with
                | .record "Ident" packageFields,.atom "Atom" =>
                    match find packageFields "Name" with
                    | .atom "schema" =>
                        let (value,after) ← construct ops fuel (← nodes (field expression "Elts")) state {}
                        .ok ([.atom value],after)
                    | _ => .error "Schema atom constructor requires the checked package"
                | _,_ => .error "Unmodeled schema composite type"
            | _ => .error "Unmodeled schema composite type"
        | .record "CallExpr" _ =>
            match field expression "Fun",field expression "Ellipsis" with
            | .record "SelectorExpr" fields,.bit false =>
                let (receiver,afterReceiver) ← one ops fuel (find fields "X") state
                let (arguments,afterArguments) ← many ops fuel (← nodes (field expression "Args")) afterReceiver
                method ops receiver (← atom (field (find fields "Sel") "Name")) arguments afterArguments
            | .record "Ident" fields,.bit false =>
                let (arguments,afterArguments) ← many ops fuel (← nodes (field expression "Args")) state
                functionCall ops (← atom (find fields "Name")) arguments afterArguments
            | _,_ => .error "Unmodeled schema dispatch call syntax"
        | _ => .error "Unmodeled schema expression syntax"
  termination_by fuel _ _ => (fuel,0)

  def one (ops : Ops Obj M S L R H Err) (fuel : Nat) (expression : Node) (state : Machine Obj M S L R H) :
      Except String (Value Obj M S L R Err × Machine Obj M S L R H) := do
    let (values,after) ← eval ops fuel expression state
    match values with
    | [value] => .ok (value,after)
    | _ => .error "Schema expression requires one result"
  termination_by (fuel,1)

  def many (ops : Ops Obj M S L R H Err) : Nat → List Node → Machine Obj M S L R H →
      Except String (List (Value Obj M S L R Err) × Machine Obj M S L R H)
    | 0,_,_ => .error "Schema argument syntax fuel exhausted"
    | _+1,[],state => .ok ([],state)
    | fuel+1,first::rest,state => do
        let (values,afterFirst) ← eval ops fuel first state
        let (remaining,afterRest) ← many ops fuel rest afterFirst
        .ok (values ++ remaining,afterRest)
  termination_by fuel _ _ => (fuel,2)

  def construct (ops : Ops Obj M S L R H Err) : Nat → List Node → Machine Obj M S L R H → Atom M S L →
      Except String (Atom M S L × Machine Obj M S L R H)
    | 0,_,_,_ => .error "Schema atom field syntax fuel exhausted"
    | _+1,[],state,value => .ok (value,state)
    | fuel+1,first::rest,state,value => do
        match first,field first "Key" with
        | .record "KeyValueExpr" _,.record "Ident" fields =>
            let (argument,after) ← one ops fuel (field first "Value") state
            let next ← setAtomField value (← atom (find fields "Name")) argument
            construct ops fuel rest after next
        | _,_ => .error "Schema atom fields must have explicit names"
  termination_by fuel _ _ _ => (fuel,3)
end

inductive Result (Obj M S L R H Err : Type) where
  | next (state : Machine Obj M S L R H)
  | returned (values : List (Value Obj M S L R Err)) (state : Machine Obj M S L R H)

/-- Case expressions are evaluated in order and stop as soon as one matches. -/
def anyCondition (ops : Ops Obj M S L R H Err) (fuel : Nat) : List Node → Machine Obj M S L R H →
    Except String (Bool × Machine Obj M S L R H)
  | [],state => .ok (false,state)
  | first::rest,state => do
      let (value,after) ← one ops fuel first state
      match value with
      | .flag value => if value then .ok (true,after) else anyCondition ops fuel rest after
      | _ => .error "Schema switch case requires a boolean"

def names : List Node → Except String (List String)
  | [] => .ok []
  | (.record "Ident" fields)::rest => do .ok ((← atom (find fields "Name"))::(← names rest))
  | _ => .error "Schema assignments require bound local variables"

mutual
  def exec (ops : Ops Obj M S L R H Err) : Nat → Node → Machine Obj M S L R H →
      Except String (Result Obj M S L R H Err)
    | 0,_,_ => .error "Schema statement syntax fuel exhausted"
    | fuel+1,statement,state => do
        match statement with
        | .absent => .ok (.next state)
        | .record "BlockStmt" _ => execList ops fuel (← nodes (field statement "List")) state
        | .record "AssignStmt" _ =>
            match field statement "Tok" with
            | .atom ":=" | .atom "=" =>
                let (values,after) ← many ops 64 (← nodes (field statement "Rhs")) state
                .ok (.next (← assign (← names (← nodes (field statement "Lhs"))) values after))
            | _ => .error "Unmodeled schema assignment token"
        | .record "IfStmt" _ =>
            match field statement "Init" with
            | .absent =>
                let (value,after) ← one ops 64 (field statement "Cond") state
                match value with
                | .flag value =>
                    if value then exec ops fuel (field statement "Body") after
                    else exec ops fuel (field statement "Else") after
                | _ => .error "Schema conditional requires a boolean"
            | _ => .error "Unmodeled schema conditional initializer"
        | .record "SwitchStmt" _ =>
            match field statement "Init",field statement "Tag",field statement "Body" with
            | .absent,.absent,.record "BlockStmt" fields => execCases ops fuel (← nodes (find fields "List")) state
            | _,_,_ => .error "Schema dispatch requires an expressionless switch"
        | .record "ReturnStmt" _ =>
            let (values,after) ← many ops 64 (← nodes (field statement "Results")) state
            .ok (.returned values after)
        | _ => .error "Unmodeled schema statement syntax"
  termination_by fuel _ _ => (fuel,0,0)

  def execList (ops : Ops Obj M S L R H Err) : Nat → List Node → Machine Obj M S L R H →
      Except String (Result Obj M S L R H Err)
    | 0,_,_ => .error "Schema block syntax fuel exhausted"
    | _+1,[],state => .ok (.next state)
    | fuel+1,first::rest,state => do
        match ← exec ops fuel first state with
        | .returned values after => .ok (.returned values after)
        | .next next => execList ops fuel rest next
  termination_by fuel _ _ => (fuel,1,0)

  /-- A selected case exits the switch even when its body does not return. No implicit fallthrough. -/
  def execCases (ops : Ops Obj M S L R H Err) (fuel : Nat) : List Node → Machine Obj M S L R H →
      Except String (Result Obj M S L R H Err)
    | [],state => .ok (.next state)
    | first::rest,state => do
        match first with
        | .record "CaseClause" _ =>
            let conditions ← nodes (field first "List")
            if conditions.isEmpty then throw "Default schema case syntax is not modeled"
            let (matched,after) ← anyCondition ops 64 conditions state
            if matched then execList ops fuel (← nodes (field first "Body")) after
            else execCases ops fuel rest after
        | _ => .error "Unmodeled schema switch clause"
  termination_by clauses _ => (fuel,2,clauses.length)
end

def invokeDeduce (ops : Ops Obj M S L R H Err) (input : Atom M S L) (value : Option Obj) (handler : H) :
    Except String (Atom M S L × H) := do
  let state : Machine Obj M S L R H := {frame := {atom := input,value},handler}
  match ← exec ops 32 SchemaDispatchSource.deduceProgram state with
  | .returned [.atom result] after => .ok (result,after.handler)
  | _ => .error "Atom deduction did not return an atom"

def invokeHandle (ops : Ops Obj M S L R H Err) (input : Atom M S L) (reference : Reference R) (handler : H) :
    Except String (Err × H) := do
  let state : Machine Obj M S L R H := {frame := {current := input,reference := some reference},handler}
  match ← exec ops 32 SchemaDispatchSource.handleProgram state with
  | .returned [.errors errors] after => .ok (errors,after.handler)
  | .returned [.nil] after => .ok (ops.nilErrors,after.handler)
  | _ => .error "Atom handler did not return validation errors"

def invokeResolve (ops : Ops Obj M S L R H Err) (reference : Reference R) (value : Option Obj) (handler : H) :
    Except String (Err × H) := do
  let state : Machine Obj M S L R H := {frame := {reference := some reference,value},handler}
  match ← exec ops 32 SchemaDispatchSource.resolveProgram state with
  | .returned [.errors errors] after => .ok (errors,after.handler)
  | .returned [.nil] after => .ok (ops.nilErrors,after.handler)
  | _ => .error "Schema resolution did not return validation errors"

end SSA.SchemaDispatchSubset
