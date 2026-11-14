/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SearchMembership

namespace SSA.HasBridge

open SearchSubset SearchAlgorithm SearchContracts SearchMembership

def answer (ops : Ops α) (members : List α) (target : α) : Nat :=
  lowerBound (beforeAt ops.compare members target) 0 members.length

def has (ops : Ops α) (members : List α) (target : α) : Bool :=
  foundAt ops.compare members target (answer ops members target)

def decode : Except String (Result α) → Except String (Nat × Bool)
  | .ok (.returned [.number index,.flag found]) => .ok (index,found)
  | .error message => .error message
  | _ => .error "Standard search source returned an unsupported result"

/-- Rejected shapes cannot denote a Go array; supported calls execute the complete imported library body. -/
def sourceSearch (ops : Ops α) (members : List α) (target : α) : Except String (Nat × Bool) :=
  if ops.wordBits = 0 then .error "Go search requires a positive word width"
  else if members.length ≤ maxIndex ops.wordBits then
    decode (exec ops (members.length+19) SearchSource.binarySearchProgram {members,target,other := target})
  else .error "Search array length exceeds the Go int bound"

def withSourceSearch (ops : Ops α) : Ops α :=
  {ops with search := sourceSearch ops}

theorem source_search_matches (ops : Ops α) (positive : 0 < ops.wordBits)
    (members : List α) (target : α) (representable : members.length ≤ maxIndex ops.wordBits) :
    sourceSearch ops members target = .ok (answer ops members target,has ops members target) := by
  have nonzero : ops.wordBits ≠ 0 := by omega
  simp [sourceSearch,nonzero,representable,imported_search_matches ops positive members target representable,
    decode,answer,has]

/-- Has delegates to the verified selected-library execution, with no assumed membership oracle. -/
theorem imported_has_matches (ops : Ops α) (positive : 0 < ops.wordBits)
    (members : List α) (target : α) (representable : members.length ≤ maxIndex ops.wordBits) :
    exec (withSourceSearch ops) 3 HasSource.hasProgram {members,target,other := target} =
      .ok (.returned [.flag (has ops members target)]) := by
  simp [HasSource.hasProgram,exec,many,one,eval,assign,SearchSubset.read,SearchSubset.write,
    withSourceSearch,source_search_matches ops positive members target representable,Bind.bind,Except.bind]

theorem imported_less_matches (ops : Ops α) (left right : α) :
    exec ops 2 HasSource.lessProgram {members := [],target := left,other := right} =
      .ok (.returned [.flag (negative ops.compare left right)]) := by
  simp [HasSource.lessProgram,exec,many,one,eval,SearchSubset.read,negative,Bind.bind,Except.bind]

theorem has_iff_member (ops : Ops α) (domain : α → Prop)
    (laws : ElementLocalOrder.OrderOn domain (negative ops.compare))
    (zeroFaithful : ∀ a b, domain a → domain b → (ops.compare a b = 0 ↔ a = b))
    (members : List α) (valid : ∀ x ∈ members, domain x)
    (target : α) (targetDomain : domain target)
    (sorted : members.Pairwise (fun a b => negative ops.compare a b = true)) :
    has ops members target = true ↔ target ∈ members :=
  found_iff_member ops.compare domain laws zeroFaithful members valid target targetDomain sorted

theorem imported_has_membership (ops : Ops α) (positive : 0 < ops.wordBits)
    (domain : α → Prop) (laws : ElementLocalOrder.OrderOn domain (negative ops.compare))
    (zeroFaithful : ∀ a b, domain a → domain b → (ops.compare a b = 0 ↔ a = b))
    (members : List α) (valid : ∀ x ∈ members, domain x)
    (target : α) (targetDomain : domain target)
    (sorted : members.Pairwise (fun a b => negative ops.compare a b = true))
    (representable : members.length ≤ maxIndex ops.wordBits) (found : Bool)
    (returned : exec (withSourceSearch ops) 3 HasSource.hasProgram {members,target,other := target} =
      .ok (.returned [.flag found])) : found = true ↔ target ∈ members := by
  rw [imported_has_matches ops positive members target representable] at returned
  cases returned
  exact has_iff_member ops domain laws zeroFaithful members valid target targetDomain sorted

end SSA.HasBridge
