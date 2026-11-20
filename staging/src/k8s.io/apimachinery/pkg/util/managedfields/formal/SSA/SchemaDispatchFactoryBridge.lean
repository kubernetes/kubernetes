/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SchemaDispatchResolver
import SSA.RemoveFactoryBridge

namespace SSA.SchemaDispatchFactoryBridge

abbrev Walker := RemoveFactorySubset.Walker
abbrev SourceOps := SchemaDispatchMachine.Ops
abbrev SourceAtom := SchemaDispatchMachine.Atom
abbrev Reference := SchemaDispatchMachine.Reference

/-- Container queries share the removal primitive; scalar queries remain explicit value contracts. -/
def kindQuery (typed : AtomicRemoveSubset.Ops Obj) (scalar : SchemaDispatchMachine.Kind → Obj → Bool) :
    SchemaDispatchMachine.Kind → Obj → Bool
  | .map,value => typed.isKind .map value
  | .list,value => typed.isKind .list value
  | kind,value => scalar kind value

def removeHandler (typed : AtomicRemoveSubset.Ops Obj) (kind : Option AtomicRemoveSubset.Kind)
    (nilErrors : Err) (walker : Walker Obj Sch Fields) : Except String (Err × Walker Obj Sch Fields) := do
  let (after,_) ← RemoveFactoryBridge.dispatch typed kind walker
  .ok (nilErrors,after)

def sourceOps (typed : AtomicRemoveSubset.Ops Obj)
    (lookup : Reference R → SourceAtom Bool Unit Bool × Bool)
    (scalar : SchemaDispatchMachine.Kind → Obj → Bool) (errorf : String → String → Err) (nilErrors : Err) :
    SourceOps Obj Bool Unit Bool R (Walker Obj Sch Fields) Err := {
  resolve := lookup
  isKind := kindQuery typed scalar
  errorf
  nilErrors
  map := fun atomic walker =>
    if atomic then removeHandler typed (some .map) nilErrors walker
    else .error "Non-atomic map traversal is outside this bridge"
  scalar := fun _ walker => removeHandler typed none nilErrors walker
  list := fun atomic walker =>
    if atomic then removeHandler typed (some .list) nilErrors walker
    else .error "Non-atomic list traversal is outside this bridge"}

/-- The constructor calls the real resolver AST with source-derived deduction, selection and handlers. -/
def linkedFactoryOps (typed : AtomicRemoveSubset.Ops Obj)
    (lookup : Sch → Reference R → SourceAtom Bool Unit Bool × Bool)
    (scalar : SchemaDispatchMachine.Kind → Obj → Bool) (errorf : String → String → Err) (nilErrors : Err)
    (wrap : Option Obj → Obj) : RemoveFactorySubset.Ops Obj Sch Fields (Reference R) Err := {
  resolve := fun schema reference value walker => do
    let (errors,after) ← SchemaDispatchSubset.invokeResolve
      (SchemaDispatchResolver.linkedOps (sourceOps typed (lookup schema) scalar errorf nilErrors))
      reference (some value) walker
    .ok (after,errors)
  wrap}

def result (typed : AtomicRemoveSubset.Ops Obj)
    (lookup : Sch → Reference R → SourceAtom Bool Unit Bool × Bool)
    (scalar : SchemaDispatchMachine.Kind → Obj → Bool) (errorf : String → String → Err) (nilErrors : Err)
    (wrap : Option Obj → Obj) (request : RemoveFactorySubset.Request Obj Sch Fields (Reference R)) : Except String Obj := do
  let (_,after) ← SchemaDispatchResolver.fullResult (sourceOps typed (lookup request.schema) scalar errorf nilErrors)
    request.typeRef (some request.input) (RemoveFactoryContracts.fresh request)
  .ok (wrap after.output)

theorem imported_factory_dispatch (typed : AtomicRemoveSubset.Ops Obj)
    (lookup : Sch → Reference R → SourceAtom Bool Unit Bool × Bool)
    (scalar : SchemaDispatchMachine.Kind → Obj → Bool) (errorf : String → String → Err) (nilErrors : Err)
    (wrap : Option Obj → Obj) (request : RemoveFactorySubset.Request Obj Sch Fields (Reference R)) :
    RemoveFactorySubset.invoke (linkedFactoryOps typed lookup scalar errorf nilErrors wrap) request =
      result typed lookup scalar errorf nilErrors wrap request := by
  rw [RemoveFactoryContracts.imported_factory]
  simp only [linkedFactoryOps]
  rw [SchemaDispatchResolver.imported_full_dispatch]
  cases outcome : SchemaDispatchResolver.fullResult (sourceOps typed (lookup request.schema) scalar errorf nilErrors)
      request.typeRef (some request.input) (RemoveFactoryContracts.fresh request) <;>
    simp [result,outcome,Bind.bind,Except.bind]

/-- An atomic-map lookup drives the entire caller through source selection to the whole-value result. -/
theorem imported_atomic_map (typed : AtomicRemoveSubset.Ops Obj)
    (lookup : Sch → Reference R → SourceAtom Bool Unit Bool × Bool)
    (scalar : SchemaDispatchMachine.Kind → Obj → Bool) (errorf : String → String → Err) (nilErrors : Err)
    (wrap : Option Obj → Obj) (request : RemoveFactorySubset.Request Obj Sch Fields (Reference R))
    (found : lookup request.schema request.typeRef = ({map := some true},true)) :
    RemoveFactorySubset.invoke (linkedFactoryOps typed lookup scalar errorf nilErrors wrap) request =
      .ok (wrap (if typed.isKind .map request.input && request.extract then typed.unstructured request.input else none)) := by
  rw [imported_factory_dispatch]
  simp only [result,SchemaDispatchResolver.fullResult,sourceOps,found]
  rw [SchemaDispatchContracts.deduced_map_only]
  simp only [SchemaDispatchHandler.handled,removeHandler,RemoveFactoryBridge.dispatch,
    RemoveFactoryBridge.requireInput,RemoveFactoryContracts.fresh,Bind.bind,Except.bind]
  rw [AtomicRemoveContracts.imported_atomic_container typed .map _ rfl]
  cases shape : typed.isKind .map request.input <;> cases extracting : request.extract <;>
    simp_all [AtomicRemoveContracts.atomicResult]

end SSA.SchemaDispatchFactoryBridge
