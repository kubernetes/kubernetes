/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SchemaDispatchSubset

namespace SSA.SchemaDispatchContracts

open SchemaDispatchMachine SchemaDispatchSubset

def scalarKind (ops : Ops Obj M S L R H Err) (value : Obj) : Bool :=
  ops.isKind .floating value || ops.isKind .integer value ||
    ops.isKind .string value || ops.isKind .boolean value

/-- A matched value category stops the switch even when that category is not allowed by the atom. -/
def deduced (ops : Ops Obj M S L R H Err) (input : Atom M S L) : Option Obj → Atom M S L
  | none => input
  | some value =>
      if scalarKind ops value then match input.scalar with
        | some scalar => {scalar := some scalar}
        | none => input
      else if ops.isKind .list value then match input.list with
        | some list => {list := some list}
        | none => input
      else if ops.isKind .map value then match input.map with
        | some map => {map := some map}
        | none => input
      else input

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

macro "schema_deduce_simp" : tactic =>
  `(tactic| simp_all (config := {maxSteps := 1000000}) [invokeDeduce,SchemaDispatchSource.deduceProgram,exec,execList,execCases,
      anyCondition,eval,one,many,construct,RawGoSyntax.field,RawGoSyntax.find,
      AtomicRemoveSubset.atom,AtomicRemoveSubset.nodes,SchemaDispatchMachine.read,
      optionalObject,select,isNil,method,setAtomField,deduced,scalarKind,
      Bind.bind,Except.bind])

/-- The complete source deduction preserves type pointers and never changes handler state. -/
theorem imported_deduce (ops : Ops Obj M S L R H Err) (input : Atom M S L) (value : Option Obj) (handler : H) :
    invokeDeduce ops input value handler = .ok (deduced ops input value,handler) := by
  cases value with
  | none => schema_deduce_simp
  | some value =>
      cases floatingKind : ops.isKind .floating value with
      | true => cases pointer : input.scalar <;> schema_deduce_simp
      | false =>
          cases integerKind : ops.isKind .integer value with
          | true => cases pointer : input.scalar <;> schema_deduce_simp
          | false =>
              cases stringKind : ops.isKind .string value with
              | true => cases pointer : input.scalar <;> schema_deduce_simp
              | false =>
                  cases booleanKind : ops.isKind .boolean value with
                  | true => cases pointer : input.scalar <;> schema_deduce_simp
                  | false =>
                      cases listKind : ops.isKind .list value with
                      | true => cases pointer : input.list <;> schema_deduce_simp
                      | false =>
                          cases mapKind : ops.isKind .map value with
                          | true => cases pointer : input.map <;> schema_deduce_simp
                          | false => schema_deduce_simp

theorem deduced_map_only (ops : Ops Obj M S L R H Err) (pointer : M) (value : Option Obj) :
    deduced ops {map := some pointer} value = {map := some pointer} := by
  cases value <;> simp [deduced]

theorem deduced_scalar_only (ops : Ops Obj M S L R H Err) (pointer : S) (value : Option Obj) :
    deduced ops {scalar := some pointer} value = {scalar := some pointer} := by
  cases value <;> simp [deduced]

theorem deduced_list_only (ops : Ops Obj M S L R H Err) (pointer : L) (value : Option Obj) :
    deduced ops {list := some pointer} value = {list := some pointer} := by
  cases value <;> simp [deduced]

end SSA.SchemaDispatchContracts
