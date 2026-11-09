/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SchemaLookupSubset

namespace SSA.SchemaLookupContracts

open SchemaLookupSubset RawGoSyntax
open AtomicRemoveSubset (atom nodes)

def plainResult (ops : Ops M S L R Rel) (reference : Reference M S L R Rel) : Atom M S L × Bool :=
  match reference.named with
  | none => (reference.inlined,true)
  | some name =>
      let (value,found) := ops.findNamed name
      if found then (value,true) else ({},false)

set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

theorem imported_plain (ops : Ops M S L R Rel) (reference : Reference M S L R Rel) :
    invoke ops SchemaLookupSource.plainProgram reference = .ok (plainResult ops reference) := by
  cases named : reference.named with
  | none =>
      simp [invoke,SchemaLookupSource.plainProgram,exec,execList,eval,one,many,field,find,atom,nodes,
        SchemaLookupSubset.read,select,isNil,assign,write,names,plainResult,named,Bind.bind,Except.bind]
  | some name =>
      cases lookup : ops.findNamed name with
      | mk value found =>
          cases found <;>
            simp_all [invoke,SchemaLookupSource.plainProgram,exec,execList,eval,one,many,field,find,atom,nodes,
              SchemaLookupSubset.read,select,isNil,call,assign,write,names,plainResult,Bind.bind,Except.bind]

/-- The full Resolve AST takes its source no-override branch before touching the mutable override cache. -/
theorem imported_fast_path (ops : Ops M S L R Rel) (reference : Reference M S L R Rel)
    (plain : reference.relationship = none) :
    invoke ops SchemaLookupSource.resolveProgram reference = ops.plain reference := by
  cases outcome : ops.plain reference <;>
    simp [invoke,SchemaLookupSource.resolveProgram,exec,execList,eval,one,many,field,find,atom,nodes,
      SchemaLookupSubset.read,select,isNil,call,plain,outcome,Bind.bind,Except.bind]

def linkedOps (ops : Ops M S L R Rel) : Ops M S L R Rel :=
  {ops with plain := invoke ops SchemaLookupSource.plainProgram}

theorem imported_no_override_lookup (ops : Ops M S L R Rel) (reference : Reference M S L R Rel)
    (plain : reference.relationship = none) :
    invoke (linkedOps ops) SchemaLookupSource.resolveProgram reference = .ok (plainResult ops reference) := by
  rw [imported_fast_path _ reference plain]
  exact imported_plain ops reference

end SSA.SchemaLookupContracts
