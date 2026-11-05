/-
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-/
import SSA.Pruning

namespace SSA.GoSubset

/-- Unsupported Go syntax is rejected by the source importer, not approximated. -/
inductive Expr where
  | var (name : String)
  | text (value : String)
  | select (receiver : Expr) (field : String)
  | call (receiver : Expr) (method : String) (args : List Expr)
  | neNil (value : Expr)
  | eqNil (value : Expr)
  | orElse (left right : Expr)
  | index (receiver key : Expr)
  | indexTwo (receiver key : Expr)
  | emptyVersionMap
  | notFlag (value : Expr)
  deriving Repr

inductive Stmt where
  | assign (names : List String) (values : List Expr)
  | declareError (name : String)
  | branch (condition : Expr) (body : List Stmt)
  | ret (values : List Expr)
  | mapSet (name : String) (key value : Expr)
  | mapDelete (name : String) (key : Expr)
  | rangeMap (key value : String) (collection : Expr) (body : List Stmt)
  deriving Repr

structure ManagerFields where
  fields : FieldSet
  version : String
  applied : Bool := true

abbrev ManagedFields := List (String × ManagerFields)

def managerLookup : ManagedFields → String → Option ManagerFields
  | [], _ => none
  | (key, value) :: rest, name => if name == key then some value else managerLookup rest name

abbrev VersionFields := List (String × FieldSet)

def versionLookup : VersionFields → String → Option FieldSet
  | [], _ => none
  | (key, fields) :: rest, version => if version == key then some fields else versionLookup rest version

def versionDelete (entries : VersionFields) (version : String) : VersionFields :=
  entries.filter (fun entry => entry.1 != version)

def versionSet (entries : VersionFields) (version : String) (fields : FieldSet) : VersionFields :=
  (version, fields) :: versionDelete entries version

/-- Every permutation is a permitted Go map schedule; no fixed iteration order is assumed. -/
structure VersionIteration where
  choose : VersionFields → VersionFields
  permutation : ∀ entries, (choose entries).Perm entries

def naturalIteration : VersionIteration where
  choose := id
  permutation _ := List.Perm.refl _

/-- Primitives remain explicit proof parameters until their implementation is verified. -/
structure Ops (Obj Err : Type) where
  convert : Obj → String → Except Err Obj
  missing : Err → Bool
  toFields : Obj → Except Err FieldSet
  ensureNamed : Obj → Obj → FieldSet → FieldSet
  remove : Obj → FieldSet → Obj
  wrap : String → Err → Err
  failedConversion : Obj → String → Err → Option Obj := fun _ _ _ => none
  wrapAt : String → String → Err → Err := fun message _ error => wrap message error
  difference : FieldSet → FieldSet → FieldSet := fieldDifference
  intersection : FieldSet → FieldSet → FieldSet := fieldIntersection
  union : FieldSet → FieldSet → FieldSet := List.append
  empty : FieldSet → Bool := List.isEmpty
  ownedItems : Obj → Obj → String → ManagedFields → String → Except Err Obj :=
    fun _ pruned _ _ _ => .ok pruned
  danglingItems : Obj → Obj → FieldSet → String → Except Err Obj :=
    fun _ pruned _ _ => .ok pruned
  ownedVersionItems : Option Obj → Option Obj → String → FieldSet →
      Except Err (Option Obj × Option Obj) := fun merged pruned _ _ => .ok (merged, pruned)
  newSet : FieldSet := []
  iteration : VersionIteration := naturalIteration

inductive Value (Obj Err : Type) where
  | object (value : Obj)
  | error (value : Err)
  | nil
  | flag (value : Bool)
  | text (value : String)
  | fields (value : FieldSet)
  | schema (value : Obj)
  | typeRef (value : Obj)
  | versioned (fields : FieldSet) (version : String)
  | managed (managers : ManagedFields)
  | versions (entries : VersionFields)
  | updater
  | converter
  | formatter
  | fieldpathPackage

abbrev Env (Obj Err : Type) := List (String × Value Obj Err)

def lookup {Obj Err : Type} : Env Obj Err → String → Value Obj Err
  | [], _ => .nil
  | (key, value) :: rest, name => if name == key then value else lookup rest name

def bindNames {Obj Err : Type} : List String → List (Value Obj Err) → Env Obj Err →
    Except String (Env Obj Err)
  | [], [], env => .ok env
  | "_" :: names, _ :: values, env => bindNames names values env
  | name :: names, value :: values, env =>
      bindNames names values ((name, value) :: env)
  | _, _, _ => .error "Go tuple-assignment arity mismatch"

def outcome {Obj Err : Type} : Except Err Obj → List (Value Obj Err)
  | .ok object => [.object object, .nil]
  | .error error => [.nil, .error error]

def fieldOutcome {Obj Err : Type} : Except Err FieldSet → List (Value Obj Err)
  | .ok fields => [.fields fields, .nil]
  | .error error => [.nil, .error error]

/-- Go conversion can return an object together with an error; callers may return that object. -/
def optionalObject {Obj Err : Type} : Option Obj → Value Obj Err
  | some object => .object object
  | none => .nil

def conversionOutcome {Obj Err : Type} (ops : Ops Obj Err) (object : Obj) (version : String) :
    List (Value Obj Err) :=
  match ops.convert object version with
  | .ok converted => [.object converted, .nil]
  | .error error => [optionalObject (ops.failedConversion object version error), .error error]

def objectOption {Obj Err : Type} : Value Obj Err → Option (Option Obj)
  | .object object => some (some object)
  | .nil => some none
  | _ => none

def pairOutcome {Obj Err : Type} : Except Err (Option Obj × Option Obj) → List (Value Obj Err)
  | .ok (merged, pruned) => [optionalObject merged, optionalObject pruned, .nil]
  | .error error => [.nil, .nil, .error error]

def isNil {Obj Err : Type} : Value Obj Err → Option Bool
  | .nil => some true
  | .object _ | .error _ | .fields _ | .versioned _ _ | .managed _ | .versions _ | .updater | .converter => some false
  | _ => none

/-- Method dispatch keeps JSON/schema/conversion behavior outside the source-control proof. -/
def call {Obj Err : Type} (ops : Ops Obj Err) : Value Obj Err → String →
    List (Value Obj Err) → Except String (List (Value Obj Err))
  | .converter, "Convert", [.object object, .text version] => .ok (conversionOutcome ops object version)
  | .converter, "IsMissingVersionError", [.error error] => .ok [.flag (ops.missing error)]
  | .object object, "ToFieldSet", [] => .ok (fieldOutcome (ops.toFields object))
  | .object object, "Schema", [] => .ok [.schema object]
  | .object object, "TypeRef", [] => .ok [.typeRef object]
  | .versioned _ version, "APIVersion", [] => .ok [.text version]
  | .versioned fields _, "Set", [] => .ok [.fields fields]
  | .fields fields, "EnsureNamedFieldsAreMembers", [.schema schema, .typeRef typeRef] =>
      .ok [.fields (ops.ensureNamed schema typeRef fields)]
  | .fields left, "Difference", [.fields right] => .ok [.fields (ops.difference left right)]
  | .fields left, "Intersection", [.fields right] => .ok [.fields (ops.intersection left right)]
  | .fields left, "Union", [.fields right] => .ok [.fields (ops.union left right)]
  | .fields fields, "Empty", [] => .ok [.flag (ops.empty fields)]
  | .updater, "addBackOwnedItems", [.object merged, .object pruned, .text version, .managed managers, .text actor] =>
      .ok (outcome (ops.ownedItems merged pruned version managers actor))
  | .updater, "addBackDanglingItems", [.object merged, .object pruned, .versioned fields version] =>
      .ok (outcome (ops.danglingItems merged pruned fields version))
  | .updater, "addBackOwnedItemsForVersion", [merged, pruned, .text version, .fields fields] =>
      match objectOption merged, objectOption pruned with
      | some merged, some pruned => .ok (pairOutcome (ops.ownedVersionItems merged pruned version fields))
      | _, _ => .error "Go owned-items arguments are not typed objects"
  | .fieldpathPackage, "NewSet", [] => .ok [.fields ops.newSet]
  | .object object, "RemoveItems", [.fields fields] => .ok [.object (ops.remove object fields)]
  | .formatter, "Errorf", [.text message, .error error] => .ok [.error (ops.wrap message error)]
  | .formatter, "Errorf", [.text message, .text version, .error error] =>
      .ok [.error (ops.wrapAt message version error)]
  | _, _, _ => .error "Unsupported Go primitive invocation"

def mapIndex {Obj Err : Type} (receiver : Value Obj Err) (key : String) :
    Except String (Value Obj Err × Bool) :=
  match receiver with
  | .managed managers =>
      match managerLookup managers key with
      | some entry => .ok (.versioned entry.fields entry.version, true)
      | none => .ok (.nil, false)
  | .versions entries =>
      match versionLookup entries key with
      | some fields => .ok (.fields fields, true)
      | none => .ok (.nil, false)
  | _ => .error "Go map indexing requires a supported map"

def mapEntries {Obj Err : Type} (ops : Ops Obj Err) : Value Obj Err →
    Except String (List (Value Obj Err × Value Obj Err))
  | .managed managers => .ok (managers.map fun entry =>
      (.text entry.1, .versioned entry.2.fields entry.2.version))
  | .versions entries => .ok ((ops.iteration.choose entries).map fun entry => (.text entry.1, .fields entry.2))
  | _ => .error "Go range requires a supported map"

mutual
def eval {Obj Err : Type} (ops : Ops Obj Err) : Nat → Expr → Env Obj Err →
    Except String (List (Value Obj Err))
  | 0, _, _ => .error "Go expression evaluation exhausted its fuel"
  | fuel + 1, expression, env =>
      match expression with
      | .var name => .ok [lookup env name]
      | .text value => .ok [.text value]
      | .emptyVersionMap => .ok [.versions []]
      | .select receiver field => do
          match ← eval ops fuel receiver env with
          | [.updater] =>
              if field == "Converter" then .ok [.converter]
              else .error "Unsupported Go receiver field"
          | _ => .error "Unsupported Go selector receiver"
      | .call receiver method args => do
          match ← eval ops fuel receiver env with
          | [receiver] =>
              let values ← evalAll ops fuel args env
              call ops receiver method values
          | _ => .error "Go method receiver is not a single value"
      | .neNil expression => do
          match ← eval ops fuel expression env with
          | [value] =>
              match isNil value with
              | some nil => .ok [.flag (!nil)]
              | none => .error "Unsupported Go nil comparison"
          | _ => .error "Go comparison is not a single value"
      | .eqNil expression => do
          match ← eval ops fuel expression env with
          | [value] =>
              match isNil value with
              | some nil => .ok [.flag nil]
              | none => .error "Unsupported Go nil comparison"
          | _ => .error "Go comparison is not a single value"
      | .orElse left right => do
          match ← eval ops fuel left env with
          | [.flag true] => .ok [.flag true]
          | [.flag false] =>
              match ← eval ops fuel right env with
              | [.flag value] => .ok [.flag value]
              | _ => .error "Go disjunction is not boolean"
          | _ => .error "Go disjunction is not boolean"
      | .index receiver key => do
          let receiver ← eval ops fuel receiver env
          let key ← eval ops fuel key env
          match receiver, key with
          | [receiver], [.text key] =>
              let (value, _) ← mapIndex receiver key
              .ok [value]
          | _, _ => .error "Unsupported Go map indexing"
      | .indexTwo receiver key => do
          let receiver ← eval ops fuel receiver env
          let key ← eval ops fuel key env
          match receiver, key with
          | [receiver], [.text key] =>
              let (value, present) ← mapIndex receiver key
              .ok [value, .flag present]
          | _, _ => .error "Unsupported Go map indexing"
      | .notFlag expression => do
          match ← eval ops fuel expression env with
          | [.flag value] => .ok [.flag (!value)]
          | _ => .error "Go negation requires a boolean"

def evalAll {Obj Err : Type} (ops : Ops Obj Err) : Nat → List Expr → Env Obj Err →
    Except String (List (Value Obj Err))
  | 0, _, _ => .error "Go argument evaluation exhausted its fuel"
  | _ + 1, [], _ => .ok []
  | fuel + 1, expression :: rest, env => do
      let first ← eval ops fuel expression env
      let remaining ← evalAll ops fuel rest env
      .ok (first ++ remaining)
end

inductive Result (Obj Err : Type) where
  | returned (values : List (Value Obj Err))
  | next (env : Env Obj Err)

mutual
/-- Imported local variables have distinct lexical identities; assignments retain those identities. -/
def exec {Obj Err : Type} (ops : Ops Obj Err) : Nat → List Stmt → Env Obj Err →
    Except String (Result Obj Err)
  | 0, _, _ => .error "Go statement evaluation exhausted its fuel"
  | _ + 1, [], env => .ok (.next env)
  | fuel + 1, statement :: rest, env =>
      match statement with
      | .ret expressions => do
          let values ← evalAll ops fuel expressions env
          .ok (.returned values)
      | .assign names expressions => do
          let values ← evalAll ops fuel expressions env
          let next ← bindNames names values env
          exec ops fuel rest next
      | .declareError name => exec ops fuel rest ((name, .nil) :: env)
      | .mapSet name key value => do
          let key ← eval ops fuel key env
          let value ← eval ops fuel value env
          match lookup env name, key, value with
          | .versions entries, [.text key], [.fields fields] =>
              exec ops fuel rest ((name, .versions (versionSet entries key fields)) :: env)
          | _, _, _ => .error "Go assignment requires a writable version map"
      | .mapDelete name key => do
          let key ← eval ops fuel key env
          match lookup env name, key with
          | .versions entries, [.text key] =>
              exec ops fuel rest ((name, .versions (versionDelete entries key)) :: env)
          | _, _ => .error "Go deletion requires a writable version map"
      | .rangeMap key value collection body => do
          match ← eval ops fuel collection env with
          | [collection] =>
              let entries ← mapEntries ops collection
              match ← execRange ops fuel entries key value body env with
              | .returned values => .ok (.returned values)
              | .next next => exec ops fuel rest next
          | _ => .error "Go range collection is not a single value"
      | .branch condition body => do
          match ← eval ops fuel condition env with
          | [.flag true] =>
              match ← exec ops fuel body env with
              | .returned values => .ok (.returned values)
              | .next next => exec ops fuel rest next
          | [.flag false] => exec ops fuel rest env
          | _ => .error "Go conditional is not boolean"
termination_by fuel _ _ => (fuel, 0, 0)

/-- Statement fuel bounds syntactic nesting, not the number of managers in a finite map. -/
def execRange {Obj Err : Type} (ops : Ops Obj Err) : Nat → List (Value Obj Err × Value Obj Err) →
    String → String → List Stmt → Env Obj Err → Except String (Result Obj Err)
  | _, [], _, _, _, env => .ok (.next env)
  | fuel, entry :: rest, key, value, body, env => do
      let iteration ← bindNames [key, value] [entry.1, entry.2] env
      match ← exec ops fuel body iteration with
      | .returned values => .ok (.returned values)
      | .next next => execRange ops fuel rest key value body next
termination_by fuel entries _ _ _ _ => (fuel, 1, entries.length)
end

theorem exec_declare_error {Obj Err : Type} (ops : Ops Obj Err) (fuel : Nat) (name : String)
    (rest : List Stmt) (env : Env Obj Err) :
    exec ops (fuel + 1) (.declareError name :: rest) env = exec ops fuel rest ((name, .nil) :: env) := by
  rw [exec]

theorem exec_assign {Obj Err : Type} (ops : Ops Obj Err) (fuel : Nat) (names : List String)
    (expressions : List Expr) (rest : List Stmt) (env : Env Obj Err) :
    exec ops (fuel + 1) (.assign names expressions :: rest) env = (do
      let values ← evalAll ops fuel expressions env
      let next ← bindNames names values env
      exec ops fuel rest next) := by rw [exec]

theorem exec_branch {Obj Err : Type} (ops : Ops Obj Err) (fuel : Nat) (condition : Expr)
    (body rest : List Stmt) (env : Env Obj Err) :
    exec ops (fuel + 1) (.branch condition body :: rest) env = (do
      match ← eval ops fuel condition env with
      | [.flag true] =>
          match ← exec ops fuel body env with
          | .returned values => .ok (.returned values)
          | .next next => exec ops fuel rest next
      | [.flag false] => exec ops fuel rest env
      | _ => .error "Go conditional is not boolean") := by rw [exec]

theorem exec_range {Obj Err : Type} (ops : Ops Obj Err) (fuel : Nat) (key value : String)
    (collection : Expr) (body rest : List Stmt) (env : Env Obj Err) :
    exec ops (fuel + 1) (.rangeMap key value collection body :: rest) env = (do
      match ← eval ops fuel collection env with
      | [collection] =>
          let entries ← mapEntries ops collection
          match ← execRange ops fuel entries key value body env with
          | .returned values => .ok (.returned values)
          | .next next => exec ops fuel rest next
      | _ => .error "Go range collection is not a single value") := by rw [exec]

end SSA.GoSubset
