/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.ElementSubset

namespace SSA.ElementOrder

variable {α : Type} {lt : α → α → Bool}

/-- The actual PathElement comparator must realize these laws on its represented domain. -/
structure StrictOrder (lt : α → α → Bool) : Prop where
  irrefl : ∀ a, lt a a = false
  trans : ∀ a b c, lt a b = true → lt b c = true → lt a c = true
  total : ∀ a b, lt a b = false → lt b a = false → a = b

theorem head_absent (laws : StrictOrder lt) (a : α) (rest : List α)
    (sorted : (a::rest).Pairwise (fun a b => lt a b = true)) : a ∉ rest := by
  intro present
  have before := (List.pairwise_cons.mp sorted).1 a present
  simp [laws.irrefl] at before

theorem before_all (laws : StrictOrder lt) (a b : α) (rest : List α)
    (sorted : (b::rest).Pairwise (fun a b => lt a b = true))
    (before : lt a b = true) : ∀ x ∈ b::rest, lt a x = true := by
  intro x member
  rcases List.mem_cons.mp member with same | present
  · simpa [same] using before
  · exact laws.trans a b x before ((List.pairwise_cons.mp sorted).1 x present)

theorem before_absent (laws : StrictOrder lt) (a b : α) (rest : List α)
    (sorted : (b::rest).Pairwise (fun a b => lt a b = true))
    (before : lt a b = true) : a ∉ b::rest := by
  intro present
  have cycle := before_all laws a b rest sorted before a present
  simp [laws.irrefl] at cycle

/-- The rank comparator used by the independent executable comparison satisfies the premises. -/
theorem nat_strict_order : StrictOrder (fun (a b : Nat) => a < b) where
  irrefl := by intro a; simp
  trans := by intro a b c ab bc; simp_all; omega
  total := by intro a b ab ba; simp_all; omega

theorem sorted_unique (laws : StrictOrder lt) (values : List α)
    (sorted : values.Pairwise (fun a b => lt a b = true)) : values.Nodup := by
  apply List.Pairwise.imp (R := fun a b => lt a b = true) (S := fun a b => a ≠ b) _ sorted
  intro a b before same
  subst b
  simp [laws.irrefl] at before

end SSA.ElementOrder
