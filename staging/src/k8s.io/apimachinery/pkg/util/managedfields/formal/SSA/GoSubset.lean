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
  deriving Repr

inductive Stmt where
  | assign (names : List String) (values : List Expr)
  | branch (condition : Expr) (body : List Stmt)
  | ret (values : List Expr)
  deriving Repr

/-- Primitives remain explicit proof parameters until their implementation is verified. -/
structure Ops (Obj Err : Type) where
  convert : Obj → String → Except Err Obj
  missing : Err → Bool
  toFields : Obj → Except Err FieldSet
  ensureNamed : Obj → Obj → FieldSet → FieldSet
  remove : Obj → FieldSet → Obj
  wrap : String → Err → Err

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
  | updater
  | converter
  | formatter

abbrev Env (Obj Err : Type) := List (String × Value Obj Err)

def lookup {Obj Err : Type} : Env Obj Err → String → Value Obj Err
  | [], _ => .nil
  | (key, value) :: rest, name => if name == key then value else lookup rest name

def bindNames {Obj Err : Type} : List String → List (Value Obj Err) → Env Obj Err →
    Except String (Env Obj Err)
  | [], [], env => .ok env
  | name :: names, value :: values, env =>
      bindNames names values ((name, value) :: env)
  | _, _, _ => .error "Go tuple-assignment arity mismatch"

def outcome {Obj Err : Type} : Except Err Obj → List (Value Obj Err)
  | .ok object => [.object object, .nil]
  | .error error => [.nil, .error error]

def fieldOutcome {Obj Err : Type} : Except Err FieldSet → List (Value Obj Err)
  | .ok fields => [.fields fields, .nil]
  | .error error => [.nil, .error error]

/-- Method dispatch keeps JSON/schema/conversion behavior outside the source-control proof. -/
def call {Obj Err : Type} (ops : Ops Obj Err) : Value Obj Err → String →
    List (Value Obj Err) → Except String (List (Value Obj Err))
  | .converter, "Convert", [.object object, .text version] => .ok (outcome (ops.convert object version))
  | .converter, "IsMissingVersionError", [.error error] => .ok [.flag (ops.missing error)]
  | .object object, "ToFieldSet", [] => .ok (fieldOutcome (ops.toFields object))
  | .object object, "Schema", [] => .ok [.schema object]
  | .object object, "TypeRef", [] => .ok [.typeRef object]
  | .versioned _ version, "APIVersion", [] => .ok [.text version]
  | .versioned fields _, "Set", [] => .ok [.fields fields]
  | .fields fields, "EnsureNamedFieldsAreMembers", [.schema schema, .typeRef typeRef] =>
      .ok [.fields (ops.ensureNamed schema typeRef fields)]
  | .fields left, "Difference", [.fields right] => .ok [.fields (fieldDifference left right)]
  | .fields left, "Intersection", [.fields right] => .ok [.fields (fieldIntersection left right)]
  | .object object, "RemoveItems", [.fields fields] => .ok [.object (ops.remove object fields)]
  | .formatter, "Errorf", [.text message, .error error] => .ok [.error (ops.wrap message error)]
  | _, _, _ => .error "Unsupported Go primitive invocation"

mutual
def eval {Obj Err : Type} (ops : Ops Obj Err) : Nat → Expr → Env Obj Err →
    Except String (List (Value Obj Err))
  | 0, _, _ => .error "Go expression evaluation exhausted its fuel"
  | fuel + 1, expression, env =>
      match expression with
      | .var name => .ok [lookup env name]
      | .text value => .ok [.text value]
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
          | [.nil] => .ok [.flag false]
          | [.error _] => .ok [.flag true]
          | _ => .error "Unsupported Go nil comparison"

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

/-- Imported branches contain no declarations or non-returning assignments, so no scope escapes. -/
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
      | .branch condition body => do
          match ← eval ops fuel condition env with
          | [.flag true] =>
              match ← exec ops fuel body env with
              | .returned values => .ok (.returned values)
              | .next next => exec ops fuel rest next
          | [.flag false] => exec ops fuel rest env
          | _ => .error "Go conditional is not boolean"

end SSA.GoSubset
