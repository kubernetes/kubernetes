/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SchemaDispatchContracts
import SSA.SchemaDispatchHandler

namespace SSA.SchemaDispatchResolver

open SchemaDispatchMachine SchemaDispatchSubset

def missingName (reference : Reference R) : String := reference.named.getD "inlined type"

def resolved (ops : Ops Obj M S L R H Err) (reference : Reference R) (value : Option Obj) (handler : H) :
    Except String (Err × H) := do
  let (input,found) := ops.resolve reference
  if found then
    let (after,next) ← ops.deduce input value handler
    ops.handle after reference next
  else .ok (ops.errorf "schema error: no type found matching: %v" (missingName reference),handler)

set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

macro "schema_resolve_simp" : tactic =>
  `(tactic| simp_all (config := {maxSteps := 1000000}) [invokeResolve,SchemaDispatchSource.resolveProgram,
      exec,execList,eval,one,many,RawGoSyntax.field,RawGoSyntax.find,
      AtomicRemoveSubset.atom,AtomicRemoveSubset.nodes,SchemaDispatchMachine.read,
      select,isNil,method,functionCall,optional_roundtrip,assign,write,names,
      resolved,missingName,Bind.bind,Except.bind])

/-- The complete resolver preserves call order, exact arguments, normal errors and handler state. -/
theorem imported_resolve (ops : Ops Obj M S L R H Err) (reference : Reference R) (value : Option Obj) (handler : H) :
    invokeResolve ops reference value handler = resolved ops reference value handler := by
  cases lookup : ops.resolve reference with
  | mk input found =>
      cases found with
      | false => cases named : reference.named <;> schema_resolve_simp
      | true =>
          cases deduction : ops.deduce input value handler with
          | error message => schema_resolve_simp
          | ok result =>
              cases result with
              | mk after next =>
                  cases handling : ops.handle after reference next <;> schema_resolve_simp

/-- Both call sites are supplied by their complete imported source bodies. -/
def linkedOps (ops : Ops Obj M S L R H Err) : Ops Obj M S L R H Err :=
  {ops with deduce := invokeDeduce ops,handle := invokeHandle ops}

def fullResult (ops : Ops Obj M S L R H Err) (reference : Reference R) (value : Option Obj) (handler : H) :
    Except String (Err × H) :=
  let (input,found) := ops.resolve reference
  if found then SchemaDispatchHandler.handled ops (SchemaDispatchContracts.deduced ops input value) reference handler
  else .ok (ops.errorf "schema error: no type found matching: %v" (missingName reference),handler)

/-- Lookup remains a primitive; deduction and map/scalar/list handler selection are source-derived. -/
theorem imported_full_dispatch (ops : Ops Obj M S L R H Err) (reference : Reference R) (value : Option Obj) (handler : H) :
    invokeResolve (linkedOps ops) reference value handler = fullResult ops reference value handler := by
  rw [imported_resolve]
  simp [resolved,linkedOps,fullResult,SchemaDispatchContracts.imported_deduce,
    SchemaDispatchHandler.imported_handle,Bind.bind,Except.bind]

end SSA.SchemaDispatchResolver
