/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.HasBridge
import SSA.SetFilterContracts

namespace SSA.FilterHasBridge

open RecursiveSets

/-- Both field-set primitives now have their own complete imported method correspondences. -/
def setOps (search : SearchSubset.Ops α) (isEmpty : Tree α → Bool) : SetSubset.Ops α :=
  {SetContracts.recursiveOps (SearchMembership.negative search.compare) isEmpty with has := HasBridge.has search}

theorem set_has_is_source (search : SearchSubset.Ops α) (isEmpty : Tree α → Bool)
    (positive : 0 < search.wordBits) (members : List α)
    (representable : members.length ≤ SearchSubset.maxIndex search.wordBits) (target : α) :
    SearchSubset.exec (HasBridge.withSourceSearch search) 3 HasSource.hasProgram {members,target,other := target} =
      .ok (.returned [.flag ((setOps search isEmpty).has members target)]) :=
  HasBridge.imported_has_matches search positive members target representable

theorem set_less_is_source (search : SearchSubset.Ops α) (isEmpty : Tree α → Bool) (left right : α) :
    SearchSubset.exec search 2 HasSource.lessProgram {members := [],target := left,other := right} =
      .ok (.returned [.flag ((setOps search isEmpty).less left right)]) :=
  HasBridge.imported_less_matches search left right

/-- The former Has-recognition premise follows from Has and selected-library source execution. -/
theorem source_filter_excludes_root (search : SearchSubset.Ops α) (isEmpty : Tree α → Bool)
    (positive : 0 < search.wordBits) (left : Tree α) (roots : List α)
    (domain : α → Prop) (laws : ElementLocalOrder.OrderOn domain (SearchMembership.negative search.compare))
    (zeroFaithful : ∀ a b, domain a → domain b → (search.compare a b = 0 ↔ a = b))
    (leftDomain : ∀ x ∈ left.members, domain x) (rootDomain : ∀ x ∈ roots, domain x)
    (leftSorted : left.members.Pairwise (fun a b => SearchMembership.negative search.compare a b = true))
    (rootSorted : roots.Pairwise (fun a b => SearchMembership.negative search.compare a b = true))
    (representable : roots.length ≤ SearchSubset.maxIndex search.wordBits)
    (root : α) (reset : root ∈ roots) (suffix : List α) (result : Tree α)
    (returned : SetSubset.exec (SetFilterContracts.flatOps (setOps search isEmpty)) 2
      SetSource.excludeFilterProgram {left,right := .node roots [],nodeMode := false} =
        .ok (.returned [.tree result])) : root::suffix ∉ paths result := by
  have sourceLookup := set_has_is_source search isEmpty positive roots representable root
  have lookupMembership := HasBridge.imported_has_membership search positive domain laws zeroFaithful
    roots rootDomain root (rootDomain root reset) rootSorted representable
    ((setOps search isEmpty).has roots root) sourceLookup
  have recognized : (setOps search isEmpty).has roots root = true := lookupMembership.mpr reset
  exact SetFilterContracts.locally_ordered_filter_excludes_root (setOps search isEmpty) left roots
    domain laws leftDomain rootDomain leftSorted rootSorted root reset recognized suffix result returned

end SSA.FilterHasBridge
