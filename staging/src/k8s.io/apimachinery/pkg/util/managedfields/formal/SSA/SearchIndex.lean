/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SearchSubset

namespace SSA.SearchIndex

open SearchSubset

/-- Valid Go array bounds make the uint sum exact, even if the intermediate signed sum overflows. -/
theorem unsigned_midpoint_eq_half (bits : Nat) (positive : 0 < bits) (left right : Nat)
    (leftBound : left ≤ maxIndex bits) (rightBound : right ≤ maxIndex bits) :
    unsignedMidpoint bits left right = (left+right)/2 := by
  cases bits with
  | zero => omega
  | succ bits =>
      have powerPositive : 0 < 2^bits := Nat.pow_pos (by decide)
      have sumSmall : left+right < 2^(bits+1) := by
        simp only [maxIndex,Nat.add_sub_cancel] at leftBound rightBound
        rw [Nat.pow_succ]
        omega
      simp [unsignedMidpoint,Nat.mod_eq_of_lt sumSmall]

theorem unsigned_midpoint_inside (bits : Nat) (positive : 0 < bits) (left right : Nat)
    (active : left < right) (rightBound : right ≤ maxIndex bits) :
    left ≤ unsignedMidpoint bits left right ∧ unsignedMidpoint bits left right < right := by
  rw [unsigned_midpoint_eq_half bits positive left right (by omega) rightBound]
  omega

/-- Each branch strictly shortens the interval and keeps both the index and its increment representable. -/
theorem unsigned_midpoint_progress (bits : Nat) (positive : 0 < bits) (left right : Nat)
    (active : left < right) (rightBound : right ≤ maxIndex bits) :
    let middle := unsignedMidpoint bits left right
    middle < right ∧ middle+1 ≤ right ∧ middle+1 ≤ maxIndex bits ∧
      right-(middle+1) < right-left ∧ middle-left < right-left := by
  have inside := unsigned_midpoint_inside bits positive left right active rightBound
  dsimp
  omega

end SSA.SearchIndex
