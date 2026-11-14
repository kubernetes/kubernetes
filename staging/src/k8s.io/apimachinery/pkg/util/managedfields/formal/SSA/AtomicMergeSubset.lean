/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.AtomicMergeMachine

namespace SSA.AtomicMergeSubset

open GoSubset (Expr Stmt)
open AtomicMergeMachine

mutual
  def eval (ops : Ops Obj) : Nat → Expr → Machine Obj → Except String (List (Value Obj))
    | 0,_,_ => .error "Atomic merge expression fuel exhausted"
    | fuel+1,expression,state =>
        match expression with
        | .var name => do .ok [← AtomicMergeMachine.read state name]
        | .number number => .ok [.number number]
        | .flag flag => .ok [.flag flag]
        | .text text => .ok [.text text]
        | .select receiver field => do
            match ← one ops fuel receiver state,field with
            | .walker,"lhs" => .ok [operand state.walker.left]
            | .walker,"rhs" => .ok [operand state.walker.right]
            | .walker,"allocator" => .ok [.allocator]
            | .walker,"inLeaf" => .ok [.flag state.walker.inLeaf]
            | .typeInfo,"ElementRelationship" => .ok [.relation state.atomic]
            | .schemaPackage,"Atomic" => .ok [.relation true]
            | _,_ => .error "Unsupported atomic merge selection"
        | .neNil value => do .ok [.flag (!(isNil (← one ops fuel value state)))]
        | .eqNil value => do .ok [.flag (isNil (← one ops fuel value state))]
        | .notFlag value => do
            match ← one ops fuel value state with
            | .flag flag => .ok [.flag (!flag)]
            | _ => .error "Atomic merge negation requires a flag"
        | .equalScalars left right => do
            match ← one ops fuel left state, ← one ops fuel right state with
            | .relation a,.relation b => .ok [.flag (a == b)]
            | .number a,.number b => .ok [.flag (a == b)]
            | _,_ => .error "Atomic merge equality requires scalar values"
        | .andAlso left right => do
            match ← one ops fuel left state with
            | .flag flag =>
                if flag then
                  match ← one ops fuel right state with
                  | .flag flag => .ok [.flag flag]
                  | _ => .error "Atomic merge conjunction requires flags"
                else .ok [.flag false]
            | _ => .error "Atomic merge conjunction requires flags"
        | .orElse left right => do
            match ← one ops fuel left state with
            | .flag flag =>
                if flag then .ok [.flag true]
                else
                  match ← one ops fuel right state with
                  | .flag flag => .ok [.flag flag]
                  | _ => .error "Atomic merge disjunction requires flags"
            | _ => .error "Atomic merge disjunction requires flags"
        | .addressOf value => do
            match ← one ops fuel value state with
            | .data data => .ok [.pointer data]
            | _ => .error "Atomic merge address requires a local data value"
        | .call receiver method arguments => do
            call ops (← one ops fuel receiver state) method (← many ops fuel arguments state)
        | _ => .error "Unsupported atomic merge expression"
  termination_by fuel _ _ => (fuel,0)

  def one (ops : Ops Obj) (fuel : Nat) (expression : Expr) (state : Machine Obj) : Except String (Value Obj) := do
    match ← eval ops fuel expression state with
    | [value] => .ok value
    | _ => .error "Atomic merge expression requires one result"
  termination_by (fuel,1)

  def many (ops : Ops Obj) : Nat → List Expr → Machine Obj → Except String (List (Value Obj))
    | 0,_,_ => .error "Atomic merge argument fuel exhausted"
    | _+1,[],_ => .ok []
    | fuel+1,head::rest,state => do
        .ok ((← eval ops fuel head state) ++ (← many ops fuel rest state))
  termination_by fuel _ _ => (fuel,2)
end

def effect (ops : Ops Obj) (expression : Expr) (state : Machine Obj) : Except String (Machine Obj) := do
  match expression with
  | .call receiver "rule" [argument] =>
      match ← one ops 32 receiver state, ← one ops 32 argument state with
      | .walker,.walker => ops.rule {state with events := state.events ++ [.rule]}
      | _,_ => .error "Merge rule requires its mutable walker"
  | .call receiver "doLeaf" [] =>
      match ← one ops 32 receiver state with
      | .walker => ops.leaf {state with events := state.events ++ [.leaf]}
      | _ => .error "Merge leaf requires its mutable walker"
  | _ => .error "Unsupported atomic merge effect"

def captureFree (ops : Ops Obj) (expression : Expr) (state : Machine Obj) : Except String (Machine Obj) := do
  match expression with
  | .call receiver "Free" [argument] =>
      match ← one ops 32 receiver state, ← one ops 32 argument state with
      | .allocator,.container value => .ok {state with frame := {state.frame with deferred := value.handle::state.frame.deferred}}
      | _,_ => .error "Atomic merge defer requires a non-nil captured resource"
  | _ => .error "Unsupported atomic merge deferred call"

inductive Result (Obj : Type) where
  | next (state : Machine Obj)
  | returned (values : List (Value Obj)) (state : Machine Obj)

def exec (ops : Ops Obj) : Nat → List Stmt → Machine Obj → Except String (Result Obj)
  | 0,_,_ => .error "Atomic merge statement fuel exhausted"
  | _+1,[],state => .ok (.next state)
  | fuel+1,statement::rest,state =>
      match statement with
      | .assign names expressions => do
          let next ← assign names (← many ops 64 expressions state) state
          exec ops fuel rest next
      | .fieldSet name field expression => do
          let next ← setField state name field (← one ops 64 expression state)
          exec ops fuel rest next
      | .branch condition body => do
          match ← one ops 64 condition state with
          | .flag true =>
              match ← exec ops fuel body state with
              | .returned values final => .ok (.returned values final)
              | .next next => exec ops fuel rest next
          | .flag false => exec ops fuel rest state
          | _ => .error "Atomic merge branch requires a flag"
      | .branchElse condition body alternative => do
          match ← one ops 64 condition state with
          | .flag flag =>
              match ← exec ops fuel (if flag then body else alternative) state with
              | .returned values final => .ok (.returned values final)
              | .next next => exec ops fuel rest next
          | _ => .error "Atomic merge branch requires a flag"
      | .effect call => do
          let next ← effect ops call state
          exec ops fuel rest next
      | .deferCall call => do
          let next ← captureFree ops call state
          exec ops fuel rest next
      | .ret expressions => do .ok (.returned (← many ops 64 expressions state) state)
      | _ => .error "Unsupported atomic merge statement"

def invoke (ops : Ops Obj) (program : List Stmt) (void : Bool) (state : Machine Obj) :
    Except String (List (Value Obj) × Machine Obj) := do
  match ← exec ops 24 program {state with frame := {}} with
  | .returned values final =>
      if void then match values with
        | [] => .ok ([],restore state final)
        | _ => .error "Void merge source returned values"
      else .ok (values,restore state final)
  | .next final =>
      if void then .ok ([],restore state final)
      else .error "Value-returning merge source fell through"

def invokeVoid (ops : Ops Obj) (program : List Stmt) (state : Machine Obj) : Except String (Machine Obj) := do
  let (_,final) ← invoke ops program true state
  .ok final

end SSA.AtomicMergeSubset
