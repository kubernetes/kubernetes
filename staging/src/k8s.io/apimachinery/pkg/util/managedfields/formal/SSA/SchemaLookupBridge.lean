/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SchemaResolveCache
import SSA.SchemaDispatchResolver

namespace SSA.SchemaLookupBridge

open SchemaResolveMachine SchemaResolveContracts
variable {Obj M S L R Rel H Err : Type} [DecidableEq R]

abbrev TypedOps := SchemaDispatchMachine.Ops

def typedReference (reference : Reference M S L R Rel) : SchemaDispatchMachine.Reference R :=
  {original := reference.original,named := reference.named}

/-- The schema method runs first; its actual result realizes the typed resolver's lookup call.
    Kind queries, error formatting and handler primitives retain their existing contracts. -/
def invoke (schema : Ops M S L Rel)
    (typed : TypedOps Obj (Container M Rel) S (Container L Rel) R H Err)
    (reference : Reference M S L R Rel) (value : Option Obj) (handler : H) (state : Runtime M S L R Rel) :
    Except String ((Err × H) × Runtime M S L R Rel) := do
  let (lookup,after) ← SchemaResolveSubset.invoke schema reference state
  let output ← SchemaDispatchSubset.invokeResolve
    (SchemaDispatchResolver.linkedOps {typed with resolve := fun _ => lookup})
    (typedReference reference) value handler
  .ok (output,after)

theorem imported_schema_and_dispatch (schema : Ops M S L Rel)
    (typed : TypedOps Obj (Container M Rel) S (Container L Rel) R H Err)
    (references : R → Reference M S L R Rel) (reference : Reference M S L R Rel)
    (value : Option Obj) (handler : H) (state : Runtime M S L R Rel)
    (stable : references reference.original = reference)
    (valid : SchemaResolveCache.Valid schema references state) (unlocked : state.held = false) :
    invoke schema typed reference value handler state = (do
      let output ← SchemaDispatchResolver.fullResult {typed with resolve := fun _ => wanted schema reference}
        (typedReference reference) value handler
      .ok (output,(computed schema reference state).2)) := by
  unfold invoke
  rw [SchemaResolveCache.imported_correct schema references reference state stable valid unlocked]
  simp only [Bind.bind,Except.bind]
  rw [SchemaDispatchResolver.imported_full_dispatch]

end SSA.SchemaLookupBridge
