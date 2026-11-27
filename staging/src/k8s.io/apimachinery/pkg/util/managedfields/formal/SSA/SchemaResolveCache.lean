/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SchemaResolveContracts

namespace SSA.SchemaResolveCache

open SchemaResolveMachine SchemaResolveContracts
variable {M S L R Rel : Type} [DecidableEq R]

/-- A key denotes the same immutable reference and named-type environment throughout a history. -/
def Valid (ops : Ops M S L Rel) (references : R → Reference M S L R Rel) (state : Runtime M S L R Rel) : Prop :=
  ∀ key value, state.cache.bind (fun values => values key) = some value →
    wanted ops (references key) = (value,true)

omit [DecidableEq R] in
theorem empty_valid (ops : Ops M S L Rel) (references : R → Reference M S L R Rel) :
    Valid ops references ({} : Runtime M S L R Rel) := by
  simp [Valid]

omit [DecidableEq R] in
theorem allocated_lookup (state : Runtime M S L R Rel) (key : R) :
    (allocated state).cache.bind (fun values => values key) = state.cache.bind (fun values => values key) := by
  cases cache : state.cache <;> simp [allocated,cache]

omit [DecidableEq R] in
theorem allocated_valid (ops : Ops M S L Rel) (references : R → Reference M S L R Rel)
    (state : Runtime M S L R Rel) (valid : Valid ops references state) :
    Valid ops references (allocated state) := by
  intro key value present
  exact valid key value (by simpa only [allocated_lookup] using present)

theorem insert_valid (ops : Ops M S L Rel) (references : R → Reference M S L R Rel)
    (state : Runtime M S L R Rel) (valid : Valid ops references state)
    (key : R) (value : Atom M S L Rel) (correct : wanted ops (references key) = (value,true)) :
    Valid ops references (cacheInsert state key value) := by
  intro other stored present
  by_cases same : other = key
  · simp [cacheInsert,same] at present
    simpa [same,present] using correct
  · simp [cacheInsert,same] at present
    exact valid other stored present

theorem computed_result (ops : Ops M S L Rel) (references : R → Reference M S L R Rel)
    (reference : Reference M S L R Rel) (state : Runtime M S L R Rel)
    (stable : references reference.original = reference) (valid : Valid ops references state) :
    (computed ops reference state).1 = wanted ops reference := by
  cases relationship : reference.relationship with
  | none => simp [computed,wanted,relationship]
  | some rel =>
      cases hit : state.cache.bind (fun values => values reference.original) with
      | none => simp [computed,relationship,hit]
      | some value =>
          have correct := valid reference.original value hit
          simpa [computed,relationship,hit,stable] using correct.symm

theorem computed_valid (ops : Ops M S L Rel) (references : R → Reference M S L R Rel)
    (reference : Reference M S L R Rel) (state : Runtime M S L R Rel)
    (stable : references reference.original = reference) (valid : Valid ops references state) :
    Valid ops references (computed ops reference state).2 := by
  cases relationship : reference.relationship with
  | none => simpa [computed,relationship,Valid] using valid
  | some rel =>
      cases hit : state.cache.bind (fun values => values reference.original) with
      | some value => simpa [computed,relationship,hit] using allocated_valid ops references state valid
      | none =>
          have allocatedValid := allocated_valid ops references state valid
          cases outcome : wanted ops reference with
          | mk value found =>
              cases found with
              | false => simpa [computed,relationship,hit,outcome,Valid] using allocatedValid
              | true =>
                  have initialized : Valid ops references {allocated state with plainCalls := (allocated state).plainCalls+1} :=
                    allocatedValid
                  have correct : wanted ops (references reference.original) = (value,true) := by simpa [stable] using outcome
                  simpa [computed,relationship,hit,outcome] using
                    insert_valid ops references _ initialized reference.original value correct

theorem computed_unlocked (ops : Ops M S L Rel) (reference : Reference M S L R Rel)
    (state : Runtime M S L R Rel) (unlocked : state.held = false) :
    (computed ops reference state).2.held = false := by
  cases relationship : reference.relationship with
  | none => simpa [computed,relationship] using unlocked
  | some rel =>
      cases hit : state.cache.bind (fun values => values reference.original) with
      | some value => simp [computed,relationship,hit,allocated]
      | none => cases found : (wanted ops reference).2 <;> simp [computed,relationship,hit,found,allocated,cacheInsert]

/-- The no-cache specification is realized by the full source from every valid unlocked cache. -/
theorem imported_correct (ops : Ops M S L Rel) (references : R → Reference M S L R Rel)
    (reference : Reference M S L R Rel) (state : Runtime M S L R Rel)
    (stable : references reference.original = reference) (valid : Valid ops references state)
    (unlocked : state.held = false) :
    SchemaResolveSubset.invoke ops reference state =
      .ok (wanted ops reference,(computed ops reference state).2) := by
  rw [imported_resolve ops reference state unlocked]
  congr 1
  exact Prod.ext (computed_result ops references reference state stable valid) rfl

def run (ops : Ops M S L Rel) (references : R → Reference M S L R Rel) :
    List R → Runtime M S L R Rel → Except String (List (Atom M S L Rel × Bool) × Runtime M S L R Rel)
  | [],state => .ok ([],state)
  | key::keys,state => do
      let (outcome,state) ← SchemaResolveSubset.invoke ops (references key) state
      let (outcomes,state) ← run ops references keys state
      .ok (outcome::outcomes,state)

/-- Arbitrary finite repeated/mixed-key histories preserve cache soundness and release the mutex.
    Immutable references and the named lookup are explicit premises, not facts about arbitrary Go mutation. -/
theorem imported_history (ops : Ops M S L Rel) (references : R → Reference M S L R Rel)
    (identity : ∀ key, (references key).original = key) (keys : List R)
    (state : Runtime M S L R Rel) (valid : Valid ops references state) (unlocked : state.held = false) :
    ∃ after, run ops references keys state = .ok (keys.map (fun key => wanted ops (references key)),after) ∧
      Valid ops references after ∧ after.held = false := by
  induction keys generalizing state with
  | nil => exact ⟨state,rfl,valid,unlocked⟩
  | cons key keys ih =>
      have stable : references (references key).original = references key := by rw [identity key]
      have step := imported_correct ops references (references key) state stable valid unlocked
      have validAfter := computed_valid ops references (references key) state stable valid
      have unlockedAfter := computed_unlocked ops (references key) state unlocked
      obtain ⟨after,tail,validFinal,unlockedFinal⟩ := ih _ validAfter unlockedAfter
      refine ⟨after,?_,validFinal,unlockedFinal⟩
      simp [run,step,tail,Bind.bind,Except.bind]

end SSA.SchemaResolveCache
