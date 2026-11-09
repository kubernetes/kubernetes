/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SearchIndex

namespace SSA.SearchAlgorithm

/-- The independent interval algorithm does not assume an ordered comparator for termination. -/
def lowerBound (before : Nat → Bool) (left right : Nat) : Nat :=
  if left < right then
    let middle := (left+right)/2
    if before middle then lowerBound before (middle+1) right
    else lowerBound before left middle
  else left
termination_by right-left
decreasing_by all_goals omega

theorem result_bounds (before : Nat → Bool) (left right : Nat) (ordered : left ≤ right) :
    left ≤ lowerBound before left right ∧ lowerBound before left right ≤ right := by
  rw [lowerBound]
  by_cases active : left < right
  · rw [ite_eq_left active]
    have middleBounds : left ≤ (left+right)/2 ∧ (left+right)/2 < right := by omega
    dsimp only
    by_cases selected : before ((left+right)/2) = true
    · simp only [ite_eq_left selected]
      have inner := result_bounds before ((left+right)/2+1) right (by omega)
      omega
    · simp only [ite_eq_right selected]
      have inner := result_bounds before left ((left+right)/2) (by omega)
      omega
  · rw [ite_eq_right active]
    omega
termination_by right-left

/-- Truth is a prefix only within the actual array; behavior at unvisited indices is irrelevant. -/
def Prefix (before : Nat → Bool) (size : Nat) : Prop :=
  ∀ a b, a < b → b < size → before b = true → before a = true

theorem boundary (before : Nat → Bool) (size left right : Nat)
    (ordered : left ≤ right) (within : right ≤ size) (monotone : Prefix before size) :
    let result := lowerBound before left right
    (∀ k, left ≤ k → k < result → before k = true) ∧
    (∀ k, result ≤ k → k < right → before k = false) := by
  rw [lowerBound]
  by_cases active : left < right
  · rw [ite_eq_left active]
    have middleBounds : left ≤ (left+right)/2 ∧ (left+right)/2 < right := by omega
    dsimp only
    by_cases selected : before ((left+right)/2) = true
    · rw [ite_eq_left selected]
      have inner := boundary before size ((left+right)/2+1) right (by omega) within monotone
      constructor
      · intro k start stop
        by_cases skipped : k ≤ (left+right)/2
        · by_cases same : k = (left+right)/2
          · simpa [same] using selected
          · exact monotone k ((left+right)/2) (by omega) (by omega) selected
        · exact inner.1 k (by omega) stop
      · intro k start stop
        exact inner.2 k start stop
    · rw [ite_eq_right selected]
      have selected : before ((left+right)/2) = false := Bool.eq_false_iff.mpr selected
      have inner := boundary before size left ((left+right)/2) middleBounds.1 (by omega) monotone
      constructor
      · intro k start stop
        exact inner.1 k start stop
      · intro k start stop
        by_cases earlier : k < (left+right)/2
        · exact inner.2 k start earlier
        · apply Bool.eq_false_iff.mpr
          intro contrary
          by_cases same : k = (left+right)/2
          · simp [same,selected] at contrary
          · have atMiddle := monotone ((left+right)/2) k (by omega) (by omega) contrary
            simp [selected] at atMiddle
  · rw [ite_eq_right active]
    constructor <;> intros <;> omega
termination_by right-left

end SSA.SearchAlgorithm
