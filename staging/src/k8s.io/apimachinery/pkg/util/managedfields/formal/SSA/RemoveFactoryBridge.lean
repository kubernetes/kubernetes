/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.RemoveFactoryContracts

namespace SSA.RemoveFactoryBridge

open RemoveFactorySubset
open RemoveFactoryContracts (fresh imported_factory)

def requireInput : Option Obj → Except String Obj
  | some value => .ok value
  | none => .error "Removal handler requires a non-nil input Value"

/-- The selected handler is source-backed; proving schema resolution selects it remains separate. -/
def dispatch (ops : AtomicRemoveSubset.Ops Obj) (kind : Option AtomicRemoveSubset.Kind)
    (walker : Walker Obj Sch Fields) : Except String (Walker Obj Sch Fields × Unit) := do
  let input ← requireInput walker.input
  let state : AtomicRemoveSubset.Machine Obj := {input,output := walker.output,extract := walker.extract}
  let program := match kind with
    | none => AtomicRemoveSource.scalarProgram
    | some kind => AtomicRemoveContracts.program kind
  let after ← AtomicRemoveSubset.invoke ops program state
  .ok ({walker with output := after.output},())

def linkedOps (ops : AtomicRemoveSubset.Ops Obj) (kind : Option AtomicRemoveSubset.Kind)
    (wrap : Option Obj → Obj) : Ops Obj Sch Fields Ref Unit := {
  resolve := fun _ _ _ walker => dispatch ops kind walker
  wrap}

/-- Fresh output comes from the complete caller, rather than an assumption on its input state. -/
theorem imported_atomic_factory (ops : AtomicRemoveSubset.Ops Obj) (kind : AtomicRemoveSubset.Kind)
    (wrap : Option Obj → Obj) (request : RemoveFactorySubset.Request Obj Sch Fields Ref) :
    invoke (linkedOps ops (some kind) wrap) request =
      .ok (wrap (if ops.isKind kind request.input && request.extract then ops.unstructured request.input else none)) := by
  rw [imported_factory]
  simp only [linkedOps,dispatch,fresh,requireInput,Bind.bind,Except.bind]
  rw [AtomicRemoveContracts.imported_atomic_container ops kind _ rfl]
  cases shape : ops.isKind kind request.input <;> cases extracting : request.extract <;>
    simp_all [AtomicRemoveContracts.atomicResult]

theorem imported_scalar_factory (ops : AtomicRemoveSubset.Ops Obj) (wrap : Option Obj → Obj)
    (request : RemoveFactorySubset.Request Obj Sch Fields Ref) :
    invoke (linkedOps ops none wrap) request = .ok (wrap (ops.unstructured request.input)) := by
  rw [imported_factory]
  simp only [linkedOps,dispatch,fresh,requireInput,Bind.bind,Except.bind]
  rw [AtomicRemoveContracts.imported_scalar]

/-- Any normal resolver error payload is ignored by this private caller. -/
theorem imported_factory_ignores_reported_errors (wrap : Option Obj → Obj) (reported : Err)
    (request : RemoveFactorySubset.Request Obj Sch Fields Ref) :
    invoke {resolve := fun _ _ _ walker => .ok (walker,reported),wrap} request = .ok (wrap none) := by
  rw [imported_factory]
  rfl

end SSA.RemoveFactoryBridge
