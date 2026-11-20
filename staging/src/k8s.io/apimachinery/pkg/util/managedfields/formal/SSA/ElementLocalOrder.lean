/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.ElementAlgebra

namespace SSA.ElementLocalOrder

open ElementAlgebra

variable {α β : Type}

/-- Ordering is needed only on labels visited by this particular element operation. -/
structure OrderOn (domain : α → Prop) (lt : α → α → Bool) : Prop where
  irrefl : ∀ a, domain a → lt a a = false
  trans : ∀ a b c, domain a → domain b → domain c → lt a b = true → lt b c = true → lt a c = true
  total : ∀ a b, domain a → domain b → lt a b = false → lt b a = false → a = b

def restrict (domain : α → Prop) : (values : List α) → (∀ x ∈ values, domain x) → List {x // domain x}
  | [],_ => []
  | head::rest,valid => ⟨head,valid head (by simp)⟩ ::
      restrict domain rest (fun x present => valid x (List.mem_cons_of_mem head present))

theorem map_restrict (domain : α → Prop) (values : List α) (valid : ∀ x ∈ values, domain x) :
    (restrict domain values valid).map Subtype.val = values := by
  induction values with
  | nil => rfl
  | cons head rest induction => simp [restrict,induction]

theorem mem_restrict (domain : α → Prop) (values : List α) (valid : ∀ x ∈ values, domain x)
    (x : α) (inDomain : domain x) : (⟨x,inDomain⟩ : {x // domain x}) ∈ restrict domain values valid ↔ x ∈ values := by
  induction values with
  | nil => simp [restrict]
  | cons head rest induction => simp [restrict,induction]

/-- Projection commutes with the algorithms even when the comparator is not an order. -/
theorem compute_map (operation : ElementAlgebra.Operation) (lt : α → α → Bool) (project : β → α) (left right : List β) :
    (compute operation (fun a b => lt (project a) (project b)) left right).map project =
      compute operation lt (left.map project) (right.map project) := by
  match left,right with
  | [],_ => cases operation <;> simp [compute,finish]
  | a::left,[] => cases operation <;> simp [compute,finish]
  | a::left,b::right =>
      by_cases before : lt (project a) (project b) = true
      · rw [compute,List.map_cons,List.map_cons,compute,ite_eq_left before,ite_eq_left before]
        have induction := compute_map operation lt project left (b::right)
        cases operation <;> simp [emit,induction]
      · by_cases after : lt (project b) (project a) = true
        · rw [compute,List.map_cons,List.map_cons,compute,ite_eq_right before,ite_eq_right before,
            ite_eq_left after,ite_eq_left after]
          have induction := compute_map operation lt project (a::left) right
          cases operation <;> simp [emit,induction]
        · rw [compute,List.map_cons,List.map_cons,compute,ite_eq_right before,ite_eq_right before,
            ite_eq_right after,ite_eq_right after]
          have induction := compute_map operation lt project left right
          cases operation <;> simp [emit,induction]
termination_by left.length+right.length

theorem restricted_order (domain : α → Prop) (lt : α → α → Bool) (laws : OrderOn domain lt) :
    ElementOrder.StrictOrder (fun (a b : {x // domain x}) => lt a.val b.val) where
  irrefl a := laws.irrefl a.val a.property
  trans a b c := laws.trans a.val b.val c.val a.property b.property c.property
  total a b before after := Subtype.ext (laws.total a.val b.val a.property b.property before after)

theorem restrict_preserves_order (domain : α → Prop) (lt : α → α → Bool) (values : List α)
    (valid : ∀ x ∈ values, domain x) (sorted : values.Pairwise (fun a b => lt a b = true)) :
    (restrict domain values valid).Pairwise (fun a b => lt a.val b.val = true) := by
  have projected : ((restrict domain values valid).map Subtype.val).Pairwise (fun a b => lt a b = true) := by
    simpa only [map_restrict] using sorted
  simpa only [List.pairwise_map] using projected

/-- A lawful field-name domain suffices; unrelated numeric labels may violate all global order laws. -/
theorem compute_local_membership (operation : ElementAlgebra.Operation) (domain : α → Prop) (lt : α → α → Bool)
    (laws : OrderOn domain lt) (left right : List α)
    (leftDomain : ∀ x ∈ left, domain x) (rightDomain : ∀ x ∈ right, domain x)
    (leftSorted : left.Pairwise (fun a b => lt a b = true))
    (rightSorted : right.Pairwise (fun a b => lt a b = true))
    (x : α) (inDomain : domain x) :
    x ∈ compute operation lt left right ↔ membership operation left right x := by
  let a := restrict domain left leftDomain
  let b := restrict domain right rightDomain
  let selected : {x // domain x} := ⟨x,inDomain⟩
  have projected := compute_map operation lt (Subtype.val : {x // domain x} → α) a b
  simp only [a,b,map_restrict] at projected
  have memberLaw := compute_membership operation (restricted_order domain lt laws) a b
    (restrict_preserves_order domain lt left leftDomain leftSorted)
    (restrict_preserves_order domain lt right rightDomain rightSorted) selected
  have projectedMembership :
      x ∈ (compute operation (fun (a b : {x // domain x}) => lt a.val b.val) a b).map Subtype.val ↔
      selected ∈ compute operation (fun (a b : {x // domain x}) => lt a.val b.val) a b := by
    constructor
    · intro present
      obtain ⟨element,owned,same⟩ := List.mem_map.mp present
      have identical : element = selected := Subtype.ext same
      exact identical ▸ owned
    · intro present
      exact List.mem_map.mpr ⟨selected,present,rfl⟩
  rw [← projected,projectedMembership,memberLaw]
  cases operation <;> simp [membership,a,b,selected,mem_restrict]

theorem imported_local_membership (operation : ElementAlgebra.Operation) (domain : α → Prop)
    (lt : α → α → Bool) (laws : OrderOn domain lt) (left right : List α)
    (leftDomain : ∀ x ∈ left, domain x) (rightDomain : ∀ x ∈ right, domain x)
    (leftSorted : left.Pairwise (fun a b => lt a b = true))
    (rightSorted : right.Pairwise (fun a b => lt a b = true)) (result : List α)
    (returned : ElementSubset.exec lt (left.length+right.length+19) (program operation) {left,right} =
      .ok (.returned [.set result])) (x : α) (inDomain : domain x) :
    x ∈ result ↔ membership operation left right x := by
  rw [ElementAlgebra.imported_matches] at returned
  cases returned
  exact compute_local_membership operation domain lt laws left right leftDomain rightDomain leftSorted rightSorted x inDomain

end SSA.ElementLocalOrder
