/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SchemaDispatchSubset

namespace SSA.SchemaDispatchHandler

open SchemaDispatchMachine SchemaDispatchSubset

def errorName (reference : Reference R) : String :=
  match reference.named with
  | none => "inlined"
  | some name => "named type: " ++ name

def handled (ops : Ops Obj M S L R H Err) (input : Atom M S L) (reference : Reference R) (handler : H) :
    Except String (Err × H) :=
  match input.map with
  | some value => ops.map value handler
  | none => match input.scalar with
    | some value => ops.scalar value handler
    | none => match input.list with
      | some value => ops.list value handler
      | none => .ok (ops.errorf "schema error: invalid atom: %v" (errorName reference),handler)

set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

macro "schema_handle_simp" : tactic =>
  `(tactic| simp_all (config := {maxSteps := 1000000}) [invokeHandle,SchemaDispatchSource.handleProgram,
      exec,execList,execCases,anyCondition,eval,one,many,
      RawGoSyntax.field,RawGoSyntax.find,AtomicRemoveSubset.atom,AtomicRemoveSubset.nodes,
      SchemaDispatchMachine.read,select,isNil,method,functionCall,assign,write,names,
      handled,errorName,Bind.bind,Except.bind])

/-- Exact map/scalar/list priority, unchanged type pointers and propagated normal errors/post-state. -/
theorem imported_handle (ops : Ops Obj M S L R H Err) (input : Atom M S L)
    (reference : Reference R) (handler : H) :
    invokeHandle ops input reference handler = handled ops input reference handler := by
  cases map : input.map with
  | some pointer => cases outcome : ops.map pointer handler <;> schema_handle_simp
  | none =>
      cases scalar : input.scalar with
      | some pointer => cases outcome : ops.scalar pointer handler <;> schema_handle_simp
      | none =>
          cases list : input.list with
          | some pointer => cases outcome : ops.list pointer handler <;> schema_handle_simp
          | none => cases named : reference.named <;> schema_handle_simp

end SSA.SchemaDispatchHandler
