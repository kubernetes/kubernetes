/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.GranularKeyCounterexamples
import SSA.PruningRefinement

namespace SSA.NumericOmissionCounterexamples

open GoSubset PruningRefinement
open NumericCounterexamples (roundInteger)
open NumericCodecCounterexamples (neighbor itemPath)
open GranularKeyCounterexamples (itemFields payloadPath)

set_option maxRecDepth 10000
set_option maxHeartbeats 4000000

def listPath : Path := [.field "endpoints"]
def listCell : Cell := ⟨listPath,.collection⟩
def itemCell (key : Nat) : Cell := ⟨itemPath key,.item⟩
def payloadCell (key : Nat) (encoded : String) : Cell := ⟨payloadPath key,.value encoded⟩
def named (low : Nat) : FieldSet :=
  [listPath,payloadPath low,payloadPath (low+1)] ++ keyLeaves (itemPath low) ++ keyLeaves (itemPath (low+1))

def document (low : Nat) : State :=
  ⟨[listCell,itemCell low,payloadCell low "\"low\"",itemCell (low+1),payloadCell (low+1) "\"high\""],[]⟩

def intent (low : Nat) : State :=
  ⟨[listCell,itemCell low,itemCell (low+1),payloadCell (low+1) "\"high\""],[]⟩

/-- A's raw upper-item claim has already disappeared in the large shared history.
    Native API execution checks this reachable precondition independently. -/
def actorHasUpper (low : Nat) (shared : Bool) : Bool :=
  !shared || roundInteger (low+1) != low

def oldFields (low : Nat) (shared decoded : Bool) : FieldSet :=
  let read := fun key => if decoded then roundInteger key else key
  itemFields (read low) ++ if actorHasUpper low shared then itemFields (read (low+1)) else []

def priorOwners (low : Nat) (shared decoded : Bool) : ManagedFields :=
  [("A",⟨oldFields low shared decoded,"v1",true⟩)] ++
    if shared then [("B",⟨itemFields (if decoded then roundInteger (low+1) else low+1),"v1",true⟩)] else []

def nextOwners (low : Nat) (shared : Bool) : ManagedFields :=
  managerSet (priorOwners low shared true) "A" ⟨dataFields (named low) (intent low),"v1",true⟩

/-- This is a claim about the stored exact wire meaning, before selector decoding. -/
def RawExclusive (low : Nat) (shared : Bool) : Prop :=
  ∃ actor, ("A",actor) ∈ priorOwners low shared false ∧ payloadPath low ∈ actor.fields ∧
    ∀ owner ∈ priorOwners low shared false, payloadPath low ∈ owner.2.fields → owner.1 = "A"

def rawTargetsExist (low : Nat) (shared : Bool) : Bool :=
  (priorOwners low shared false).all fun owner =>
    owner.2.fields.all (dataFields (named low) (document low)).contains

def pruned (low : Nat) (shared : Bool) : State :=
  pruneWithFields (named low) (document low) (oldFields low shared true)
    (PruneSource.restoringFields (nextOwners low shared))

/-- The four complete imported pruning bodies realize this trace under the
    existing normalized typed/set primitives, for every lawful map schedule. -/
theorem imported_pruning_matches (low : Nat) (shared : Bool) (order : VersionIteration) :
    CalleesRealized (linkedOps (named low) order) "v1" (nextOwners low shared) "A" ∧
    exec (linkedOps (named low) order) 64 GoSource.pruneProgram
      (PruneSource.pruneEnv (document low) (nextOwners low shared) "A"
        (some (oldFields low shared true,"v1"))) =
      .ok (.returned [.object (pruned low shared),.nil]) := by
  apply single_version_pruning_refinement (named low) order (document low)
    (nextOwners low shared) "A" (oldFields low shared true) "v1"
    ⟨dataFields (named low) (intent low),"v1",true⟩
  · rfl
  · intro row member
    simp only [nextOwners,managerSet,List.mem_cons] at member
    rcases member with rfl | member
    · rfl
    · have prior := (List.mem_filter.mp member).1
      cases shared with
      | false =>
          simp [priorOwners] at prior
          subst row
          rfl
      | true =>
          simp [priorOwners] at prior
          rcases prior with rfl | rfl <;> rfl

theorem lower_payload_is_raw_exclusive : RawExclusive neighbor true := by
  refine ⟨⟨oldFields neighbor true false,"v1",true⟩,by simp [priorOwners],by decide,?_⟩
  intro owner present claimed
  simp [priorOwners] at present
  rcases present with rfl | rfl
  · rfl
  · have absent : payloadPath neighbor ∉ itemFields (neighbor+1) := by decide
    exact False.elim (absent claimed)

theorem prior_wire_targets_exist : rawTargetsExist neighbor true = true := by decide

theorem omission_keeps_both_items_and_does_not_assert_lower_payload :
    itemCell neighbor ∈ (intent neighbor).cells ∧
    itemCell (neighbor+1) ∈ (intent neighbor).cells ∧
    payloadPath neighbor ∉ dataFields (named neighbor) (intent neighbor) := by decide

theorem foreign_upper_selector_becomes_lower_claim :
    payloadPath neighbor ∉ itemFields (neighbor+1) ∧
    payloadPath neighbor ∈ itemFields (roundInteger (neighbor+1)) := by decide

theorem omitted_exclusive_payload_survives :
    payloadCell neighbor "\"low\"" ∈ (pruned neighbor true).cells ∧
    (pruned neighbor true).cells = (document neighbor).cells := by decide

/-- No defaults, mutation or API admission stage occurs in this candidate-level law. -/
def EveryExclusiveOmissionDisappears : Prop :=
  ∀ low shared, low+1 < 2^63 → rawTargetsExist low shared = true → RawExclusive low shared →
    payloadPath low ∉ dataFields (named low) (intent low) →
    (pruned low shared).hasValue (payloadPath low) = false

theorem unconditional_exclusive_omission_is_false : ¬EveryExclusiveOmissionDisappears := by
  intro removes
  have absent := removes neighbor true (by decide) prior_wire_targets_exist lower_payload_is_raw_exclusive
    omission_keeps_both_items_and_does_not_assert_lower_payload.2.2
  have present : (pruned neighbor true).hasValue (payloadPath neighbor) = true := by decide
  rw [present] at absent
  contradiction

theorem omission_controls_remove_lower_payload :
    (pruned 41 true).hasValue (payloadPath 41) = false ∧
    (pruned 41 false).hasValue (payloadPath 41) = false ∧
    (pruned neighbor false).hasValue (payloadPath neighbor) = false := by decide

structure Observation where
  largeSharedRetainsLower : Bool
  largeUnsharedRemovesLower : Bool
  smallSharedRemovesLower : Bool
  smallUnsharedRemovesLower : Bool
  deriving Lean.ToJson

def observe : Observation := {
  largeSharedRetainsLower := (pruned neighbor true).hasValue (payloadPath neighbor)
  largeUnsharedRemovesLower := !(pruned neighbor false).hasValue (payloadPath neighbor)
  smallSharedRemovesLower := !(pruned 41 true).hasValue (payloadPath 41)
  smallUnsharedRemovesLower := !(pruned 41 false).hasValue (payloadPath 41)
}

end SSA.NumericOmissionCounterexamples
