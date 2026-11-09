/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.RemoveFactorySubset

namespace SSA.RemoveFactoryContracts

open RemoveFactorySubset RawGoSyntax
open AtomicRemoveSubset (atom nodes)

def fresh (request : RemoveFactorySubset.Request Obj Sch Fields Ref) : Walker Obj Sch Fields := {
  input := some request.input
  schema := some request.schema
  fields := some request.fields
  allocator := true
  extract := request.extract}

set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

/-- The complete caller derives zero output before dispatch and wraps the resulting raw value. -/
theorem imported_factory (ops : Ops Obj Sch Fields Ref Err) (request : RemoveFactorySubset.Request Obj Sch Fields Ref) :
    invoke ops request = (do
      let (after,_) ← ops.resolve request.schema request.typeRef request.input (fresh request)
      .ok (ops.wrap after.output)) := by
  cases resolved : ops.resolve request.schema request.typeRef request.input (fresh request) <;>
    simp only [fresh] at resolved <;>
    simp [invoke,AtomicRemoveSource.factoryProgram,exec,execList,eval,evalAll,construct,
      field,find,atom,nodes,RemoveFactorySubset.read,writeField,resolve,resolved,
      Bind.bind,Except.bind]

end SSA.RemoveFactoryContracts
