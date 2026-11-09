/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.ElementAlgebra

namespace SSA.RecursiveSets

/-- Direct membership and descendant nodes are independent, as in Go's Set. -/
inductive Tree (α : Type) where
  | node (members : List α) (children : List (α × Tree α))
  deriving Repr, BEq

def Tree.members : Tree α → List α
  | .node members _ => members

def Tree.children : Tree α → List (α × Tree α)
  | .node _ children => children

def paths : Tree α → List (List α)
  | .node members children => members.map (fun member => [member]) ++
      children.flatMap (fun child => (paths child.2).map (child.1 :: ·))
termination_by tree => sizeOf tree
decreasing_by
  have smaller := List.sizeOf_lt_of_mem (show child ∈ children from by assumption)
  cases child
  simp_all <;> omega

def nodePaths (children : List (α × Tree α)) : List (List α) :=
  children.flatMap (fun child => (paths child.2).map (child.1 :: ·))

def empty (tree : Tree α) : Bool := (paths tree).isEmpty

def PathsFrom (original result : Tree α) : Prop :=
  ∀ path ∈ paths result, path ∈ paths original

/-- The independent two-pointer specification keeps the left label on a matched child. -/
def nodesCompute (lt : α → α → Bool) (isEmpty : Tree α → Bool) (difference : Tree α → Tree α → Tree α) :
    List (α × Tree α) → List (α × Tree α) → List (α × Tree α)
  | [],_ => []
  | left,[] => left
  | a::left,b::right =>
      if lt a.1 b.1 then a :: nodesCompute lt isEmpty difference left (b::right)
      else if lt b.1 a.1 then nodesCompute lt isEmpty difference (a::left) right
      else
        let child := difference a.2 b.2
        (if isEmpty child then [] else [(a.1,child)]) ++ nodesCompute lt isEmpty difference left right
termination_by left right => left.length+right.length

theorem node_paths_cons (head : α × Tree α) (rest : List (α × Tree α)) :
    nodePaths (head::rest) = (paths head.2).map (head.1 :: ·) ++ nodePaths rest := rfl

theorem nodes_compute_never_adds_paths (lt : α → α → Bool) (isEmpty : Tree α → Bool) (difference : Tree α → Tree α → Tree α)
    (left right : List (α × Tree α))
    (childBound : ∀ child ∈ left, ∀ other, PathsFrom child.2 (difference child.2 other)) (path : List α)
    (present : path ∈ nodePaths (nodesCompute lt isEmpty difference left right)) : path ∈ nodePaths left := by
  match left,right with
  | [],_ => simp [nodesCompute,nodePaths] at present
  | a::left,[] => simpa [nodesCompute] using present
  | a::left,b::right =>
      by_cases ab : lt a.1 b.1 = true
      · rw [nodesCompute,ite_eq_left ab,node_paths_cons] at present
        rw [node_paths_cons]
        rcases List.mem_append.mp present with head | tail
        · exact List.mem_append_left _ head
        · exact List.mem_append_right _ (nodes_compute_never_adds_paths lt isEmpty difference left (b::right) (fun child present => childBound child (List.mem_cons_of_mem a present)) path tail)
      · by_cases ba : lt b.1 a.1 = true
        · rw [nodesCompute,ite_eq_right ab,ite_eq_left ba] at present
          exact nodes_compute_never_adds_paths lt isEmpty difference (a::left) right childBound path present
        · rw [nodesCompute,ite_eq_right ab,ite_eq_right ba] at present
          let child := difference a.2 b.2
          change path ∈ nodePaths ((if isEmpty child then [] else [(a.1,child)]) ++ nodesCompute lt isEmpty difference left right) at present
          simp only [nodePaths,List.flatMap_append,List.mem_append] at present
          rw [node_paths_cons]
          rcases present with head | tail
          · cases emptyResult : isEmpty child
            · simp only [emptyResult,Bool.false_eq_true,ite_false,List.flatMap_cons,List.flatMap_nil,List.append_nil,List.mem_map] at head
              obtain ⟨suffix,owned,rfl⟩ := head
              exact List.mem_append_left _ (List.mem_map.mpr ⟨suffix,childBound a (by simp) b.2 suffix owned,rfl⟩)
            · simp [emptyResult] at head
          · exact List.mem_append_right _ (nodes_compute_never_adds_paths lt isEmpty difference left right (fun child present => childBound child (List.mem_cons_of_mem a present)) path tail)
termination_by left.length+right.length

mutual
  /-- Finite structural recursion mirrors the two mutually recursive Go methods. -/
  def difference (lt : α → α → Bool) (isEmpty : Tree α → Bool) (left right : Tree α) : Tree α :=
    .node (ElementAlgebra.compute .difference lt left.members right.members)
      (nodesDifference lt isEmpty left.children right.children)
  termination_by (sizeOf left,sizeOf right)
  decreasing_by
    apply Prod.Lex.left
    cases left
    simp [Tree.children]
    omega

  def nodesDifference (lt : α → α → Bool) (isEmpty : Tree α → Bool) : List (α × Tree α) → List (α × Tree α) → List (α × Tree α)
    | [],_ => []
    | left,[] => left
    | a::left,b::right =>
        if lt a.1 b.1 then a :: nodesDifference lt isEmpty left (b::right)
        else if lt b.1 a.1 then nodesDifference lt isEmpty (a::left) right
        else
          let child := difference lt isEmpty a.2 b.2
          (if isEmpty child then [] else [(a.1,child)]) ++ nodesDifference lt isEmpty left right
  termination_by left right => (sizeOf left,sizeOf right)
  decreasing_by
    all_goals first
      | (apply Prod.Lex.left; cases a <;> simp_all <;> omega)
      | (apply Prod.Lex.right; simp_all <;> omega)
end

theorem nodes_difference_eq_compute (lt : α → α → Bool) (isEmpty : Tree α → Bool) (left right : List (α × Tree α)) :
    nodesDifference lt isEmpty left right = nodesCompute lt isEmpty (difference lt isEmpty) left right := by
  match left,right with
  | [],_ => simp [nodesDifference,nodesCompute]
  | a::left,[] => simp [nodesDifference,nodesCompute]
  | a::left,b::right =>
      rw [nodesDifference,nodesCompute]
      split
      · rw [nodes_difference_eq_compute lt isEmpty left (b::right)]
      · split
        · rw [nodes_difference_eq_compute lt isEmpty (a::left) right]
        · rw [nodes_difference_eq_compute lt isEmpty left right]
termination_by left.length+right.length

/-- Exact path non-growth requires no sorting, strict order, or faithful numeric comparison. -/
theorem difference_never_adds_paths (lt : α → α → Bool) (isEmpty : Tree α → Bool) (left right : Tree α) :
    PathsFrom left (difference lt isEmpty left right) := by
  cases left with
  | node members children =>
      intro path present
      rw [paths]
      change path ∈ members.map (fun member => [member]) ++ nodePaths children
      rw [difference,paths] at present
      rcases List.mem_append.mp present with direct | descendant
      · obtain ⟨member,owned,rfl⟩ := List.mem_map.mp direct
        exact List.mem_append_left _ (List.mem_map.mpr
          ⟨member,(ElementAlgebra.compute_left_sublist .difference lt members right.members (by decide)).subset owned,rfl⟩)
      · rw [nodes_difference_eq_compute] at descendant
        exact List.mem_append_right _ (nodes_compute_never_adds_paths lt isEmpty (difference lt isEmpty) children right.children
          (fun child _ other => difference_never_adds_paths lt isEmpty child.2 other) path descendant)
termination_by sizeOf left
decreasing_by
  have smaller := List.sizeOf_lt_of_mem (show child ∈ children from by assumption)
  cases child
  simp_all <;> omega

end SSA.RecursiveSets
