/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.UpdaterSource

namespace SSA.UpdaterSubset

open GoSubset (Expr Stmt ManagerFields ManagedFields managerLookup managerSet versionLookup)

structure Comparison where
  removed : FieldSet := []
  modified : FieldSet := []
  added : FieldSet := []
  deriving Repr, BEq, DecidableEq

/-- Each lexical range has an independent lawful Go map schedule. -/
structure ManagerIteration where
  choose : String → ManagedFields → ManagedFields
  permutation : ∀ loop entries, (choose loop entries).Perm entries

def naturalIteration : ManagerIteration where
  choose _ := id
  permutation _ _ := List.Perm.refl _

structure Ops (Obj Err : Type) where
  primitives : GoSubset.Ops Obj Err
  compare : Obj → Obj → Except Err Comparison
  iteration : ManagerIteration := naturalIteration

inductive Value (Obj Err : Type) where
  | object (object : Obj)
  | error (error : Err)
  | literalError (message : String)
  | conflicts (owners : ManagedFields)
  | nil
  | flag (flag : Bool)
  | text (text : String)
  | number (number : Nat)
  | fields (fields : FieldSet)
  | manager (entry : ManagerFields)
  | managed (owners : ManagedFields)
  | reference (address : Nat)
  | comparisons (entries : List (String × Nat))
  | fieldVersions (entries : List (String × FieldSet))
  | filters (entries : List (String × (FieldSet → FieldSet)))
  | filter (transform : FieldSet → FieldSet)
  | updater
  | converter
  | formatter
  | fieldpathPackage

abbrev Env (Obj Err : Type) := List (String × Value Obj Err)

/-- Comparison methods mutate shared objects; locals and the version cache hold addresses. -/
structure Heap where
  contents : List (Nat × Comparison) := []
  nextAddress : Nat := 0

structure Machine (Obj Err : Type) where
  env : Env Obj Err := []
  heap : Heap := {}

def lookup {Obj Err : Type} (env : Env Obj Err) (name : String) : Value Obj Err :=
  ((env.find? (fun entry => entry.1 == name)).map Prod.snd).getD .nil

def read {Obj Err : Type} (state : Machine Obj Err) (name : String) : Value Obj Err :=
  lookup state.env name

def write {Obj Err : Type} (state : Machine Obj Err) (name : String) (value : Value Obj Err) : Machine Obj Err :=
  {state with env := (name,value) :: state.env}

def heapRead (heap : Heap) (address : Nat) : Except String Comparison :=
  match heap.contents.find? (fun entry => entry.1 == address) with
  | some entry => .ok entry.2
  | none => .error "Go comparison reference is not allocated"

def heapWrite (heap : Heap) (address : Nat) (value : Comparison) : Heap :=
  {heap with contents := (address,value) :: heap.contents.filter (fun entry => entry.1 != address)}

def allocate (heap : Heap) (value : Comparison) : Nat × Heap :=
  (heap.nextAddress, {heapWrite heap heap.nextAddress value with nextAddress := heap.nextAddress+1})

def bindNames {Obj Err : Type} : List String → List (Value Obj Err) → Machine Obj Err → Except String (Machine Obj Err)
  | [],[],state => .ok state
  | "_"::names,_::values,state => bindNames names values state
  | name::names,value::values,state => bindNames names values (write state name value)
  | _,_,_ => .error "Go updater assignment arity mismatch"

def mapIndex {Obj Err : Type} (map : Value Obj Err) (key : String) : Except String (Value Obj Err × Bool) :=
  match map with
  | .nil => .ok (.nil,false)
  | .managed owners =>
      match managerLookup owners key with
      | none => .ok (.nil,false)
      | some entry => .ok (.manager entry,true)
  | .comparisons entries =>
      match entries.find? (fun entry => entry.1 == key) with
      | none => .ok (.nil,false)
      | some entry => .ok (.reference entry.2,true)
  | .fieldVersions entries =>
      match versionLookup entries key with
      | none => .ok (.nil,false)
      | some fields => .ok (.fields fields,true)
  | .filters entries =>
      match entries.find? (fun entry => entry.1 == key) with
      | none => .ok (.nil,false)
      | some entry => .ok (.filter entry.2,true)
  | _ => .error "Go updater indexing requires a supported map"

def transform (filter : FieldSet → FieldSet) (comparison : Comparison) : Comparison :=
  ⟨filter comparison.removed,filter comparison.modified,filter comparison.added⟩

def call {Obj Err : Type} (ops : Ops Obj Err) (receiver : Value Obj Err) (method : String)
    (args : List (Value Obj Err)) (state : Heap) : Except String (List (Value Obj Err) × Heap) := do
  match receiver,method,args with
  | .object old,"Compare",[.object new] =>
      match ops.compare old new with
      | .error error => .ok ([.nil,.error error],state)
      | .ok comparison =>
          let (address,next) := allocate state comparison
          .ok ([.reference address,.nil],next)
  | .converter,"Convert",[.object object,.text version] =>
      match ops.primitives.convert object version with
      | .ok object => .ok ([.object object,.nil],state)
      | .error error => .ok ([.nil,.error error],state)
  | .converter,"IsMissingVersionError",[.error error] => .ok ([.flag (ops.primitives.missing error)],state)
  | .manager entry,"Set",[] => .ok ([.fields entry.fields],state)
  | .manager entry,"APIVersion",[] => .ok ([.text entry.version],state)
  | .manager entry,"Applied",[] => .ok ([.flag entry.applied],state)
  | .fields left,"Difference",[.fields right] => .ok ([.fields (ops.primitives.difference left right)],state)
  | .fields left,"Union",[.fields right] => .ok ([.fields (ops.primitives.union left right)],state)
  | .fields left,"Intersection",[.fields right] => .ok ([.fields (ops.primitives.intersection left right)],state)
  | .fields fields,"Empty",[] => .ok ([.flag (ops.primitives.empty fields)],state)
  | .fieldpathPackage,"NewVersionedSet",[.fields fields,.text version,.flag applied] =>
      .ok ([.manager ⟨fields,version,applied⟩],state)
  | .reference address,"ExcludeFields",[.nil] =>
      let _ ← heapRead state address
      .ok ([.reference address],state)
  | .reference address,"ExcludeFields",[.fields excluded] =>
      let comparison ← heapRead state address
      let next := if ops.primitives.empty excluded then state
        else heapWrite state address (transform (fun fields => ops.primitives.excludeFields fields excluded) comparison)
      .ok ([.reference address],next)
  | .reference address,"FilterFields",[.nil] =>
      let _ ← heapRead state address
      .ok ([.reference address],state)
  | .reference address,"FilterFields",[.filter filter] =>
      let comparison ← heapRead state address
      .ok ([.reference address],heapWrite state address (transform filter comparison))
  | .formatter,"Errorf",[.text message,.error error] => .ok ([.error (ops.primitives.wrap message error)],state)
  | .formatter,"Errorf",[.text message] => .ok ([.literalError message],state)
  | _,_,_ => .error "Unsupported Go updater primitive"

def select {Obj Err : Type} (ops : Ops Obj Err) (receiver : Value Obj Err) (field : String)
    (state : Heap) : Except String (Value Obj Err) :=
  match receiver with
  | .updater =>
      if field == "Converter" then .ok .converter
      else if field == "IgnoredFields" then .ok (match ops.primitives.ignoredFields with
        | none => .nil
        | some fields => .fieldVersions fields)
      else if field == "IgnoreFilter" then .ok (match ops.primitives.ignoreFilters with
        | none => .nil
        | some filters => .filters filters)
      else .error "Unsupported Go updater field"
  | .reference address => do
      let comparison ← heapRead state address
      if field == "Removed" then .ok (.fields comparison.removed)
      else if field == "Modified" then .ok (.fields comparison.modified)
      else if field == "Added" then .ok (.fields comparison.added)
      else .error "Unsupported Go comparison field"
  | _ => .error "Go updater selector has an invalid receiver"

def scalarEqual {Obj Err : Type} : Value Obj Err → Value Obj Err → Except String Bool
  | .text left,.text right => .ok (left == right)
  | .number left,.number right => .ok (left == right)
  | .flag left,.flag right => .ok (left == right)
  | _,_ => .error "Go updater comparison requires scalars of the same type"

def isNil {Obj Err : Type} : Value Obj Err → Bool
  | .nil => true
  | _ => false

mutual
def eval {Obj Err : Type} (ops : Ops Obj Err) (env : Env Obj Err) : Nat → Expr → Heap →
    Except String (List (Value Obj Err) × Heap)
  | 0,_,_ => .error "Go updater expression fuel exhausted"
  | fuel+1,expression,state => do
      match expression with
      | .var name => .ok ([lookup env name],state)
      | .text text => .ok ([.text text],state)
      | .number number => .ok ([.number number],state)
      | .flag flag => .ok ([.flag flag],state)
      | .emptyManagedMap => .ok ([.managed []],state)
      | .select receiver field =>
          let (values,next) ← eval ops env fuel receiver state
          match values with
          | [receiver] => .ok ([← select ops receiver field next],next)
          | _ => .error "Go updater selector requires one receiver"
      | .call receiver method args =>
          let (values,next) ← eval ops env fuel receiver state
          match values with
          | [receiver] =>
              let (args,next) ← evalAll ops env fuel args next
              call ops receiver method args next
          | _ => .error "Go updater call requires one receiver"
      | .neNil value =>
          let (values,next) ← eval ops env fuel value state
          match values with
          | [value] => .ok ([.flag (!isNil value)],next)
          | _ => .error "Go updater nil comparison requires one value"
      | .eqNil value =>
          let (values,next) ← eval ops env fuel value state
          match values with
          | [value] => .ok ([.flag (isNil value)],next)
          | _ => .error "Go updater nil comparison requires one value"
      | .orElse left right =>
          let (values,next) ← eval ops env fuel left state
          match values with
          | [.flag true] => .ok ([.flag true],next)
          | [.flag false] =>
              let (values,next) ← eval ops env fuel right next
              match values with
              | [.flag value] => .ok ([.flag value],next)
              | _ => .error "Go updater disjunction requires booleans"
          | _ => .error "Go updater disjunction requires booleans"
      | .andAlso left right =>
          let (values,next) ← eval ops env fuel left state
          match values with
          | [.flag false] => .ok ([.flag false],next)
          | [.flag true] =>
              let (values,next) ← eval ops env fuel right next
              match values with
              | [.flag value] => .ok ([.flag value],next)
              | _ => .error "Go updater conjunction requires booleans"
          | _ => .error "Go updater conjunction requires booleans"
      | .notFlag value =>
          let (values,next) ← eval ops env fuel value state
          match values with
          | [.flag flag] => .ok ([.flag (!flag)],next)
          | _ => .error "Go updater negation requires a boolean"
      | .index receiver key =>
          let (maps,next) ← eval ops env fuel receiver state
          let (keys,next) ← eval ops env fuel key next
          match maps,keys with
          | [map],[.text key] =>
              let (value,_) ← mapIndex map key
              .ok ([value],next)
          | _,_ => .error "Go updater indexing requires a map and text key"
      | .indexTwo receiver key =>
          let (maps,next) ← eval ops env fuel receiver state
          let (keys,next) ← eval ops env fuel key next
          match maps,keys with
          | [map],[.text key] =>
              let (value,present) ← mapIndex map key
              .ok ([value,.flag present],next)
          | _,_ => .error "Go updater indexing requires a map and text key"
      | .equalScalars left right =>
          let (left,next) ← eval ops env fuel left state
          let (right,next) ← eval ops env fuel right next
          match left,right with
          | [left],[right] => .ok ([.flag (← scalarEqual left right)],next)
          | _,_ => .error "Go updater scalar equality requires single operands"
      | .length expression =>
          let (values,next) ← eval ops env fuel expression state
          match values with
          | [.managed entries] => .ok ([.number entries.length],next)
          | _ => .error "Go updater len requires a managed-field map"
      | .singletonComparisonMap key value =>
          let (keys,next) ← eval ops env fuel key state
          let (values,next) ← eval ops env fuel value next
          match keys,values with
          | [.text key],[.reference address] => .ok ([.comparisons [(key,address)]],next)
          | _,_ => .error "Go comparison cache requires a text key and reference"
      | .conflictsFromManagers managers =>
          let (values,next) ← eval ops env fuel managers state
          match values with
          | [.managed entries] => .ok ([.conflicts entries],next)
          | _ => .error "Go conflicts require a managed-field map"
      | _ => .error "Unsupported Go updater expression"

def evalAll {Obj Err : Type} (ops : Ops Obj Err) (env : Env Obj Err) : Nat → List Expr → Heap →
    Except String (List (Value Obj Err) × Heap)
  | 0,_,_ => .error "Go updater argument fuel exhausted"
  | _+1,[],state => .ok ([],state)
  | fuel+1,expression::rest,state => do
      let (values,next) ← eval ops env fuel expression state
      let (remaining,next) ← evalAll ops env fuel rest next
      .ok (values ++ remaining,next)
end

/-- Expressions can mutate the comparison heap, but cannot assign local variables or managed maps. -/
def runEval {Obj Err : Type} (ops : Ops Obj Err) (fuel : Nat) (expression : Expr)
    (state : Machine Obj Err) : Except String (List (Value Obj Err) × Machine Obj Err) := do
  let (values,heap) ← eval ops state.env fuel expression state.heap
  .ok (values,{state with heap})

def runAll {Obj Err : Type} (ops : Ops Obj Err) (fuel : Nat) (expressions : List Expr)
    (state : Machine Obj Err) : Except String (List (Value Obj Err) × Machine Obj Err) := do
  let (values,heap) ← evalAll ops state.env fuel expressions state.heap
  .ok (values,{state with heap})

inductive Result (Obj Err : Type) where
  | next (state : Machine Obj Err)
  | returned (values : List (Value Obj Err)) (state : Machine Obj Err)
  | continued (state : Machine Obj Err)

def mapSet {Obj Err : Type} (map key value : Value Obj Err) : Except String (Value Obj Err) :=
  match map,key,value with
  | .managed entries,.text key,.manager entry => .ok (.managed (managerSet entries key entry))
  | .comparisons entries,.text key,.reference address =>
      .ok (.comparisons ((key,address) :: entries.filter (fun entry => entry.1 != key)))
  | _,_,_ => .error "Go updater assignment requires an allocated writable map"

def mapDelete {Obj Err : Type} (map key : Value Obj Err) : Except String (Value Obj Err) :=
  match map,key with
  | .managed entries,.text key => .ok (.managed (entries.filter (fun entry => entry.1 != key)))
  | _,_ => .error "Go updater deletion requires a managed-field map"

mutual
/-- Fuel bounds syntax depth; finite map size is handled by a separate recursive measure. -/
def exec {Obj Err : Type} (ops : Ops Obj Err) : Nat → List Stmt → Machine Obj Err → Except String (Result Obj Err)
  | 0,_,_ => .error "Go updater statement fuel exhausted"
  | _+1,[],state => .ok (.next state)
  | fuel+1,statement::rest,state => do
      match statement with
      | .assign names expressions =>
          let (values,next) ← runAll ops 128 expressions state
          let next ← bindNames names values next
          exec ops fuel rest next
      | .declareNil name | .declareError name => exec ops fuel rest (write state name .nil)
      | .ret expressions =>
          let (values,next) ← runAll ops 128 expressions state
          .ok (.returned values next)
      | .continueLoop => .ok (.continued state)
      | .mapSet name key value =>
          let (keys,next) ← runEval ops 128 key state
          let (values,next) ← runEval ops 128 value next
          match keys,values with
          | [key],[value] =>
              let map ← mapSet (read next name) key value
              exec ops fuel rest (write next name map)
          | _,_ => .error "Go updater map assignment requires single operands"
      | .mapDelete name key =>
          let (keys,next) ← runEval ops 128 key state
          match keys with
          | [key] =>
              let map ← mapDelete (read next name) key
              exec ops fuel rest (write next name map)
          | _ => .error "Go updater deletion requires a single key"
      | .branch condition body =>
          let (values,next) ← runEval ops 128 condition state
          match values with
          | [.flag false] => exec ops fuel rest next
          | [.flag true] =>
              match ← exec ops fuel body next with
              | .returned values final => .ok (.returned values final)
              | .continued final => .ok (.continued final)
              | .next final => exec ops fuel rest final
          | _ => .error "Go updater conditional requires a boolean"
      | .branchElse condition body alternative =>
          let (values,next) ← runEval ops 128 condition state
          match values with
          | [.flag condition] =>
              match ← exec ops fuel (if condition then body else alternative) next with
              | .returned values final => .ok (.returned values final)
              | .continued final => .ok (.continued final)
              | .next final => exec ops fuel rest final
          | _ => .error "Go updater conditional requires a boolean"
      | .rangeMap key value collection body =>
          let (values,next) ← runEval ops 128 collection state
          match values with
          | [.managed entries] =>
              match ← execRange ops fuel (ops.iteration.choose key entries) key value body next with
              | .returned values final => .ok (.returned values final)
              | .continued _ => .error "Go updater range leaked continue"
              | .next final => exec ops fuel rest final
          | _ => .error "Go updater range requires a managed-field map"
      | _ => .error "Unsupported Go updater statement"
termination_by fuel _ _ => (fuel,0,0)

def execRange {Obj Err : Type} (ops : Ops Obj Err) : Nat → ManagedFields → String → String → List Stmt →
    Machine Obj Err → Except String (Result Obj Err)
  | _,[],_,_,_,state => .ok (.next state)
  | fuel,entry::rest,key,value,body,state => do
      let next ← bindNames [key,value] [.text entry.1,.manager entry.2] state
      match ← exec ops fuel body next with
      | .returned values final => .ok (.returned values final)
      | .continued final | .next final => execRange ops fuel rest key value body final
termination_by fuel entries _ _ _ _ => (fuel,1,entries.length)
end

def initial {Obj Err : Type} (old new : Obj) (version : String) (owners : ManagedFields)
    (actor : String) (force : Bool) : Machine Obj Err := {env := [
  ("s",.updater),("fmt",.formatter),("fieldpath",.fieldpathPackage),
  ("oldObject",.object old),("newObject",.object new),("version",.text version),
  ("managers",.managed owners),("workflow",.text actor),("force",.flag force)]}

end SSA.UpdaterSubset
