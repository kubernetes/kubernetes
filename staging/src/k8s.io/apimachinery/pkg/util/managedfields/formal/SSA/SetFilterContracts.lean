/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SetContracts
import SSA.ElementLocalOrder

namespace SSA.SetFilterContracts

open GoSubset (Stmt)
open SetSubset RecursiveSets

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

private def rangeBody : List Stmt :=
  match (SetSource.recursiveNodeProgram[3]? : Option Stmt) with
  | some (.branch _ [.rangeSlice _ _ body]) => body
  | _ => []

def scanStep (ops : Ops α) (child : α × Tree α) (state : Machine α) : Machine α :=
  {state with
    cursorName := "c$2",
    cursor := some child,
    output := if ops.has state.right.members child.1 then state.output else state.output ++ [child]}

def scan (ops : Ops α) : List (α × Tree α) → Machine α → Machine α
  | [],state => state
  | child::rest,state => scan ops rest (scanStep ops child state)

private theorem range_step (ops : Ops α) (fuel : Nat) (child : α × Tree α) (state : Machine α) :
    exec ops (fuel+4) rangeBody {state with cursorName := "c$2",cursor := some child} =
      .ok (.next (scanStep ops child state)) := by
  cases selected : ops.has state.right.members child.1 <;>
    simp [rangeBody,SetSource.recursiveNodeProgram,exec,eval,SetSubset.read,scanStep,selected,Bind.bind,Except.bind]

theorem imported_range_matches_scan (ops : Ops α) (fuel : Nat) (children : List (α × Tree α))
    (state : Machine α) (enough : children.length ≤ fuel) :
    runSlice ops (fuel+4) "c$2" children rangeBody state = .ok (.next (scan ops children state)) := by
  induction fuel generalizing children state with
  | zero =>
      have empty : children = [] := by cases children <;> simp_all
      subst children
      simp [runSlice,scan]
  | succ fuel induction =>
      cases children with
      | nil => simp [runSlice,scan]
      | cons child rest =>
          change runSlice ops (fuel+5) "c$2" (child::rest) rangeBody state = _
          rw [runSlice,range_step]
          simp only [Bind.bind,Except.bind]
          rw [induction rest (scanStep ops child state) (by simpa using Nat.le_of_succ_le_succ enough)]
          rfl

theorem scan_output (ops : Ops α) (children : List (α × Tree α)) (state : Machine α) :
    (scan ops children state).output = state.output ++ children.filter (fun child => !ops.has state.right.members child.1) := by
  induction children generalizing state with
  | nil => simp [scan]
  | cons child rest induction =>
      rw [scan,induction]
      cases selected : ops.has state.right.members child.1 <;>
        simp [scanStep,selected,List.append_assoc]

@[irreducible] private def loopProgram := SetSource.recursiveNodeProgram.drop 2

private def loopCondition : GoSubset.Expr :=
  match (SetSource.recursiveNodeProgram[2]? : Option Stmt) with
  | some (.whileLoop condition _) => condition
  | _ => .flag true

private def loopBody : List Stmt :=
  match (SetSource.recursiveNodeProgram[2]? : Option Stmt) with
  | some (.whileLoop _ body) => body
  | _ => []

private theorem loop_shape : loopProgram =
    .whileLoop loopCondition loopBody :: SetSource.recursiveNodeProgram.drop 3 := by
  unfold loopProgram loopCondition loopBody
  rfl

private theorem children_of_node (members : List α) (children : List (α × Tree α)) :
    (Tree.node members children).children = children := rfl

private theorem members_of_node (members : List α) (children : List (α × Tree α)) :
    (Tree.node members children).members = members := rfl

private theorem flat_loop_condition (ops : Ops α) (left : Tree α) (roots : List α) :
    eval ops 32 loopCondition {left,right := .node roots []} = .ok (.flag false) := by
  by_cases active : 0 < left.children.length <;>
    simp [loopCondition,SetSource.recursiveNodeProgram,eval,SetSubset.read,children_of_node,active,Bind.bind,Except.bind]

private theorem flat_loop_matches (ops : Ops α) (left : Tree α) (roots : List α) :
    exec ops (left.children.length+12) loopProgram {left,right := .node roots []} =
      .ok (.returned [.nodes (left.children.filter (fun child => !ops.has roots child.1))]) := by
  rw [loop_shape,exec,flat_loop_condition]
  simp only [Bind.bind,Except.bind]
  change exec ops (left.children.length+11) (SetSource.recursiveNodeProgram.drop 3) _ = _
  by_cases active : 0 < left.children.length
  · simp [SetSource.recursiveNodeProgram,exec,eval,evalAll,SetSubset.read,
      active,Bind.bind,Except.bind]
    have run := imported_range_matches_scan ops (left.children.length+5) left.children
      ({left,right := .node roots []} : Machine α) (by omega)
    simp [rangeBody,SetSource.recursiveNodeProgram,Nat.add_assoc] at run
    rw [run]
    simp [scan_output,members_of_node]
  · have noChildren : left.children = [] := by
      cases found : left.children <;> simp_all
    simp [SetSource.recursiveNodeProgram,exec,eval,evalAll,SetSubset.read,
      noChildren,Bind.bind,Except.bind]

/-- Flat reset roots never compare or recurse into the excluded object's descendants. -/
theorem imported_flat_node_matches (ops : Ops α) (left : Tree α) (roots : List α) :
    exec ops (left.children.length+14) SetSource.recursiveNodeProgram {left,right := .node roots []} =
      .ok (.returned [.nodes (left.children.filter (fun child => !ops.has roots child.1))]) := by
  have loop := flat_loop_matches ops left roots
  have split : SetSource.recursiveNodeProgram = SetSource.recursiveNodeProgram.take 2 ++ loopProgram := by
    unfold loopProgram
    rfl
  rw [split]
  simpa [SetSource.recursiveNodeProgram,exec,evalAll,eval,assign,SetSubset.write,Nat.add_assoc,Bind.bind,Except.bind] using loop

def flatNodes (ops : Ops α) (children : List (α × Tree α)) (right : Tree α) : List (α × Tree α) :=
  children.filter (fun child => !ops.has right.members child.1)

def flatSet (ops : Ops α) (left right : Tree α) : Tree α :=
  .node (ElementAlgebra.compute .difference ops.less left.members right.members)
    (flatNodes ops left.children right)

/-- These callbacks are justified below for an exclusion set with no child nodes. -/
def flatOps (ops : Ops α) : Ops α :=
  {ops with
    recursiveNodes := fun children right =>
      if right.children.isEmpty then .ok (flatNodes ops children right)
      else .error "Root reset callback requires a flat exclusion set",
    recursiveDifference := fun left right =>
      if right.children.isEmpty then .ok (flatSet ops left right)
      else .error "Root reset callback requires a flat exclusion set"}

theorem flat_node_callback_is_source (ops : Ops α) (left : Tree α) (roots : List α) :
    exec (flatOps ops) (left.children.length+14) SetSource.recursiveNodeProgram {left,right := .node roots []} =
      .ok (.returned [.nodes (flatNodes ops left.children (.node roots []))]) := by
  rw [imported_flat_node_matches]
  rfl

theorem imported_flat_set_matches (ops : Ops α) (left : Tree α) (roots : List α) :
    exec (flatOps ops) 2 SetSource.recursiveSetProgram {left,right := .node roots [],nodeMode := false} =
      .ok (.returned [.tree (flatSet ops left (.node roots []))]) := by
  simp [SetSource.recursiveSetProgram,exec,evalAll,eval,SetSubset.read,flatOps,flatSet,children_of_node,
    Bind.bind,Except.bind,Pure.pure,Except.pure]

theorem imported_flat_filter_matches (ops : Ops α) (left : Tree α) (roots : List α) :
    exec (flatOps ops) 2 SetSource.excludeFilterProgram {left,right := .node roots [],nodeMode := false} =
      .ok (.returned [.tree (flatSet ops left (.node roots []))]) := by
  simp [SetSource.excludeFilterProgram,exec,evalAll,eval,SetSubset.read,flatOps,children_of_node,
    Bind.bind,Except.bind,Pure.pure,Except.pure]

/-- Only membership in a reset root is queried; descendant comparators are irrelevant. -/
theorem flat_set_excludes_root (ops : Ops α) (left : Tree α) (roots : List α) (root : α)
    (directAbsent : root ∉ ElementAlgebra.compute .difference ops.less left.members roots)
    (hasRoot : ops.has roots root = true) (suffix : List α) :
    root::suffix ∉ paths (flatSet ops left (.node roots [])) := by
  intro present
  rw [flatSet,paths] at present
  rcases List.mem_append.mp present with direct | descendant
  · obtain ⟨member,owned,same⟩ := List.mem_map.mp direct
    have key : member = root := (List.cons.inj same).1
    exact directAbsent (key ▸ owned)
  · simp only [flatNodes,members_of_node,List.mem_flatMap,List.mem_map] at descendant
    obtain ⟨child,retained,path,owned,same⟩ := descendant
    have key : child.1 = root := (List.cons.inj same).1
    have allowed := (List.mem_filter.mp retained).2
    simp [key,hasRoot] at allowed

theorem imported_filter_excludes_root (ops : Ops α) (left : Tree α) (roots : List α) (root : α)
    (directAbsent : root ∉ ElementAlgebra.compute .difference ops.less left.members roots)
    (hasRoot : ops.has roots root = true) (suffix : List α) (result : Tree α)
    (returned : exec (flatOps ops) 2 SetSource.excludeFilterProgram {left,right := .node roots [],nodeMode := false} =
      .ok (.returned [.tree result])) : root::suffix ∉ paths result := by
  rw [imported_flat_filter_matches] at returned
  cases returned
  exact flat_set_excludes_root ops left roots root directAbsent hasRoot suffix

/-- The ordering premise applies to direct root labels; descendants need not belong to that domain. -/
theorem locally_ordered_filter_excludes_root (ops : Ops α) (left : Tree α) (roots : List α)
    (domain : α → Prop) (laws : ElementLocalOrder.OrderOn domain ops.less)
    (leftDomain : ∀ x ∈ left.members, domain x) (rootDomain : ∀ x ∈ roots, domain x)
    (leftSorted : left.members.Pairwise (fun a b => ops.less a b = true))
    (rootSorted : roots.Pairwise (fun a b => ops.less a b = true))
    (root : α) (reset : root ∈ roots) (hasRoot : ops.has roots root = true)
    (suffix : List α) (result : Tree α)
    (returned : exec (flatOps ops) 2 SetSource.excludeFilterProgram {left,right := .node roots [],nodeMode := false} =
      .ok (.returned [.tree result])) : root::suffix ∉ paths result := by
  have directAbsent : root ∉ ElementAlgebra.compute .difference ops.less left.members roots := by
    intro owned
    have memberLaw := ElementLocalOrder.compute_local_membership .difference domain ops.less laws
      left.members roots leftDomain rootDomain leftSorted rootSorted root (rootDomain root reset)
    have excluded := (memberLaw.mp owned).2
    exact excluded reset
  exact imported_filter_excludes_root ops left roots root directAbsent hasRoot suffix result returned

end SSA.SetFilterContracts
