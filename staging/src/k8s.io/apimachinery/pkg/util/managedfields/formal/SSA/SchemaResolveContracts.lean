/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SchemaResolveSubset

namespace SSA.SchemaResolveContracts

open SchemaResolveMachine SchemaResolveSubset
variable {M S L R Rel : Type} [DecidableEq R]

abbrev Runtime := SchemaResolveMachine.State

def baseResult (ops : Ops M S L Rel) (reference : Reference M S L R Rel) : Atom M S L Rel × Bool :=
  SchemaLookupContracts.plainResult {findNamed := ops.findNamed} reference

/-- Overrides change the first available container, with map taking priority over list and scalar. -/
def overridden (relationship : Rel) (value : Atom M S L Rel) : Atom M S L Rel × Bool :=
  match value.map with
  | some map => ({value with map := some {map with relationship}},true)
  | none => match value.list with
    | some list => ({value with list := some {list with relationship}},true)
    | none => ({},false)

def wanted (ops : Ops M S L Rel) (reference : Reference M S L R Rel) : Atom M S L Rel × Bool :=
  match reference.relationship with
  | none => baseResult ops reference
  | some relationship =>
      let (value,found) := baseResult ops reference
      if found then overridden relationship value else ({},false)

def allocated (state : Runtime M S L R Rel) : Runtime M S L R Rel :=
  {state with cache := some (state.cache.getD (fun _ => none)),held := false,events := state.events++[true,false]}

def cacheInsert (state : Runtime M S L R Rel) (key : R) (value : Atom M S L Rel) : Runtime M S L R Rel :=
  {state with cache := some (fun other => if other = key then some value else state.cache.bind (fun values => values other))}

def computed (ops : Ops M S L Rel) (reference : Reference M S L R Rel) (state : Runtime M S L R Rel) :
    (Atom M S L Rel × Bool) × Runtime M S L R Rel :=
  match reference.relationship with
  | none => (baseResult ops reference,{state with plainCalls := state.plainCalls+1})
  | some _ =>
      let after := allocated state
      match state.cache.bind (fun values => values reference.original) with
      | some value => ((value,true),after)
      | none =>
          let outcome := wanted ops reference
          let after := {after with plainCalls := after.plainCalls+1}
          (outcome,if outcome.2 then cacheInsert after reference.original outcome.1 else after)

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

macro "schema_resolve_simp" : tactic =>
  `(tactic| simp_all (config := {maxSteps := 1000000}) [SchemaResolveSubset.invoke,
      SchemaLookupSource.resolveProgram,exec,execList,casesList,guards,eval,one,many,
      write,assign,declare,deferCall,unwind,RawGoSyntax.field,RawGoSyntax.find,
      AtomicRemoveSubset.atom,AtomicRemoveSubset.nodes,SchemaResolveMachine.read,
      select,isNil,method,plain_source,copyOf,address,lookup,store,
      computed,allocated,cacheInsert,wanted,overridden,baseResult,List.append_assoc,Bind.bind,Except.bind])

/-- Complete imported Resolve execution, including hits, misses, failure exits and captured unlock.
    The mutex is an explicit serial primitive; this theorem does not verify Go's scheduler or heap. -/
theorem imported_resolve (ops : Ops M S L Rel) (reference : Reference M S L R Rel)
    (state : Runtime M S L R Rel) (unlocked : state.held = false) :
    SchemaResolveSubset.invoke ops reference state = .ok (computed ops reference state) := by
  cases relationship : reference.relationship with
  | none => schema_resolve_simp
  | some relationship =>
      cases cache : state.cache with
      | none =>
          cases base : baseResult ops reference with
          | mk value found =>
              cases found with
              | false => schema_resolve_simp
              | true =>
                  cases map : value.map with
                  | some map => schema_resolve_simp
                  | none =>
                      cases list : value.list with
                      | some list => schema_resolve_simp
                      | none => cases scalar : value.scalar <;> schema_resolve_simp
      | some values =>
          cases hit : values reference.original with
          | some value => schema_resolve_simp
          | none =>
              cases base : baseResult ops reference with
              | mk value found =>
                  cases found with
                  | false => schema_resolve_simp
                  | true =>
                      cases map : value.map with
                      | some map => schema_resolve_simp
                      | none =>
                          cases list : value.list with
                          | some list => schema_resolve_simp
                          | none => cases scalar : value.scalar <;> schema_resolve_simp

end SSA.SchemaResolveContracts
