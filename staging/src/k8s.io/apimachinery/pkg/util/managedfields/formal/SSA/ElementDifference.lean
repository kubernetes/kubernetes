/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.ElementAlgebra

namespace SSA.ElementDifference

open ElementSubset

variable {α : Type} {lt : α → α → Bool}

abbrev StrictOrder (lt : α → α → Bool) := ElementOrder.StrictOrder lt
abbrev difference (lt : α → α → Bool) := ElementAlgebra.compute .difference lt

/-- The existing difference contract is a specialization of the shared algebra proof. -/
theorem difference_sublist (lt : α → α → Bool) (left right : List α) :
    (difference lt left right).Sublist left := by
  exact ElementAlgebra.compute_left_sublist .difference lt left right (by decide)

/-- Arbitrary finite sorted inputs yield exactly ordinary membership difference. -/
theorem difference_membership (laws : StrictOrder lt) (left right : List α)
    (sortedLeft : left.Pairwise (fun a b => lt a b = true))
    (sortedRight : right.Pairwise (fun a b => lt a b = true)) (x : α) :
    x ∈ difference lt left right ↔ x ∈ left ∧ x ∉ right := by
  exact ElementAlgebra.compute_membership .difference laws left right sortedLeft sortedRight x

/-- The complete imported method equals two-pointer difference for every finite input, even unsorted. -/
theorem imported_difference_matches (lt : α → α → Bool) (left right : List α) :
    exec lt (left.length+right.length+19) ElementSource.differenceProgram {left := left,right := right} =
      .ok (.returned [.set (difference lt left right)]) := by
  exact ElementAlgebra.imported_matches .difference lt left right

/-- The imported Go method implements membership difference on arbitrary finite ordered input sets. -/
theorem imported_difference_membership (laws : StrictOrder lt) (left right : List α)
    (sortedLeft : left.Pairwise (fun a b => lt a b = true))
    (sortedRight : right.Pairwise (fun a b => lt a b = true)) (result : List α)
    (returned : exec lt (left.length+right.length+19) ElementSource.differenceProgram {left := left,right := right} =
      .ok (.returned [.set result])) (x : α) :
    x ∈ result ↔ x ∈ left ∧ x ∉ right := by
  exact ElementAlgebra.imported_membership .difference laws left right sortedLeft sortedRight result returned x

/-- This stronger difference-only frame needs no comparator laws or sorted right input. -/
theorem imported_difference_preserves_order (lt : α → α → Bool) (left right : List α)
    (sorted : left.Pairwise (fun a b => lt a b = true)) (result : List α)
    (returned : exec lt (left.length+right.length+19) ElementSource.differenceProgram {left := left,right := right} =
      .ok (.returned [.set result])) : result.Pairwise (fun a b => lt a b = true) := by
  rw [imported_difference_matches] at returned
  cases returned
  exact List.Pairwise.sublist (difference_sublist lt left right) sorted

end SSA.ElementDifference
