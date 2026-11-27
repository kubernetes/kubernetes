/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SearchContracts
import SSA.ElementLocalOrder

namespace SSA.SearchMembership

open SearchSubset SearchAlgorithm SearchContracts

def negative (compare : α → α → Int) (a b : α) : Bool := compare a b < 0

/-- Only labels in the searched array and the query need the comparator laws. -/
theorem before_is_prefix (compare : α → α → Int) (domain : α → Prop)
    (laws : ElementLocalOrder.OrderOn domain (negative compare)) (members : List α)
    (valid : ∀ x ∈ members, domain x) (target : α) (targetDomain : domain target)
    (sorted : members.Pairwise (fun a b => negative compare a b = true)) :
    Prefix (beforeAt compare members target) members.length := by
  intro a b earlier bInside selected
  have aInside : a < members.length := by omega
  have before := (List.pairwise_iff_getElem.mp sorted) a b aInside bInside earlier
  have later : negative compare members[b] target = true := by
    simpa [negative,beforeAt,bInside] using selected
  have atEarlier := laws.trans members[a] members[b] target
    (valid _ (List.getElem_mem aInside)) (valid _ (List.getElem_mem bInside)) targetDomain before later
  simpa [negative,beforeAt,aInside] using atEarlier

/-- Binary search has exact membership only when equality and ordering are faithful on this domain. -/
theorem found_iff_member (compare : α → α → Int) (domain : α → Prop)
    (laws : ElementLocalOrder.OrderOn domain (negative compare))
    (zeroFaithful : ∀ a b, domain a → domain b → (compare a b = 0 ↔ a = b))
    (members : List α) (valid : ∀ x ∈ members, domain x)
    (target : α) (targetDomain : domain target)
    (sorted : members.Pairwise (fun a b => negative compare a b = true)) :
    let answer := lowerBound (beforeAt compare members target) 0 members.length
    foundAt compare members target answer = true ↔ target ∈ members := by
  let answer := lowerBound (beforeAt compare members target) 0 members.length
  have bounds := result_bounds (beforeAt compare members target) 0 members.length (by omega)
  have split := boundary (beforeAt compare members target) members.length 0 members.length
    (by omega) (by omega) (before_is_prefix compare domain laws members valid target targetDomain sorted)
  change foundAt compare members target answer = true ↔ target ∈ members
  constructor
  · intro found
    by_cases inside : answer < members.length
    · have same : compare members[answer] target = 0 := by simpa [foundAt,inside] using found
      have identical := (zeroFaithful members[answer] target (valid _ (List.getElem_mem inside)) targetDomain).mp same
      exact identical ▸ List.getElem_mem inside
    · have absent : members[answer]? = none := List.getElem?_eq_none (by omega)
      simp [foundAt,absent] at found
  · intro present
    obtain ⟨position,inside,same⟩ := List.mem_iff_getElem.mp present
    have sameZero : compare target target = 0 := (zeroFaithful target target targetDomain targetDomain).mpr rfl
    by_cases earlier : position < answer
    · have before := split.1 position (by omega) earlier
      have selfFalse : beforeAt compare members target position = false := by simp [beforeAt,inside,same,sameZero]
      simp [selfFalse] at before
    · by_cases samePosition : position = answer
      · subst position
        simp [foundAt,inside,same,sameZero]
      · have answerInside : answer < members.length := by omega
        have before := (List.pairwise_iff_getElem.mp sorted) answer position answerInside inside (by omega)
        have atAnswer : beforeAt compare members target answer = true := by
          simpa [beforeAt,answerInside,negative,same] using before
        have after := split.2 answer (by omega) answerInside
        simp [atAnswer] at after

theorem imported_search_membership (ops : Ops α) (positive : 0 < ops.wordBits)
    (domain : α → Prop) (laws : ElementLocalOrder.OrderOn domain (negative ops.compare))
    (zeroFaithful : ∀ a b, domain a → domain b → (ops.compare a b = 0 ↔ a = b))
    (members : List α) (valid : ∀ x ∈ members, domain x)
    (target : α) (targetDomain : domain target)
    (sorted : members.Pairwise (fun a b => negative ops.compare a b = true))
    (representable : members.length ≤ maxIndex ops.wordBits) (index : Nat) (found : Bool)
    (returned : exec ops (members.length+19) SearchSource.binarySearchProgram {members,target,other := target} =
      .ok (.returned [.number index,.flag found])) : found = true ↔ target ∈ members := by
  rw [imported_search_matches ops positive members target representable] at returned
  cases returned
  exact found_iff_member ops.compare domain laws zeroFaithful members valid target targetDomain sorted

end SSA.SearchMembership
