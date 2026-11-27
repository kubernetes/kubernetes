/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SetSubset

namespace SSA.SetContracts

open SetSubset RecursiveSets
open ElementAlgebra (Step)

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

def emit (ops : Ops α) (step : Step) (left right : α × Tree α) : List (α × Tree α) :=
  match step with
  | .left => [left]
  | .right => []
  | .both => let child := ops.difference left.2 right.2
             if ops.isEmpty child then [] else [(left.1,child)]

@[irreducible] private def loopProgram := SetSource.nodeDifferenceProgram.drop 2

private def loopCondition : GoSubset.Expr :=
  match (SetSource.nodeDifferenceProgram[2]? : Option GoSubset.Stmt) with
  | some (.whileLoop condition _) => condition
  | _ => .flag false

private def loopBody : List GoSubset.Stmt :=
  match (SetSource.nodeDifferenceProgram[2]? : Option GoSubset.Stmt) with
  | some (.whileLoop _ body) => body
  | _ => []

private theorem loop_shape : loopProgram =
    .whileLoop loopCondition loopBody :: SetSource.nodeDifferenceProgram.drop 3 := by
  unfold loopProgram loopCondition loopBody
  rfl

private theorem loop_step (ops : Ops α) (fuel : Nat) (state next : Machine α)
    (condition : eval ops 32 loopCondition state = .ok (.flag true))
    (body : exec ops fuel loopBody state = .ok (.next next)) :
    exec ops (fuel+1) loopProgram state = exec ops fuel loopProgram next := by
  rw [loop_shape,exec,condition,body]
  rfl

private theorem condition_value (ops : Ops α) (state : Machine α) (nodeMode : state.nodeMode = true) :
    eval ops 32 loopCondition state =
      .ok (.flag (state.i < state.left.children.length && state.j < state.right.children.length)) := by
  by_cases left : state.i < state.left.children.length <;>
    simp [loopCondition,SetSource.nodeDifferenceProgram,eval,SetSubset.read,nodeMode,left,Bind.bind,Except.bind]

private theorem step_left (ops : Ops α) (fuel : Nat) (state : Machine α)
    (nodeMode : state.nodeMode = true)
    (left : state.i < state.left.children.length) (right : state.j < state.right.children.length)
    (before : ops.less state.left.children[state.i].1 state.right.children[state.j].1 = true) :
    exec ops (fuel+18) loopProgram state =
      exec ops (fuel+17) loopProgram
        {state with i := state.i+1,output := state.output ++ emit ops .left state.left.children[state.i] state.right.children[state.j]} := by
  apply loop_step
  · simp [condition_value ops state nodeMode,left,right]
  · simp [loopBody,SetSource.nodeDifferenceProgram,exec,eval,SetSubset.read,SetSubset.write,nodeMode,left,right,before,emit,Bind.bind,Except.bind]

private theorem step_right (ops : Ops α) (fuel : Nat) (state : Machine α)
    (nodeMode : state.nodeMode = true)
    (left : state.i < state.left.children.length) (right : state.j < state.right.children.length)
    (before : ops.less state.left.children[state.i].1 state.right.children[state.j].1 = false)
    (after : ops.less state.right.children[state.j].1 state.left.children[state.i].1 = true) :
    exec ops (fuel+18) loopProgram state =
      exec ops (fuel+17) loopProgram
        {state with j := state.j+1,output := state.output ++ emit ops .right state.left.children[state.i] state.right.children[state.j]} := by
  apply loop_step
  · simp [condition_value ops state nodeMode,left,right]
  · simp [loopBody,SetSource.nodeDifferenceProgram,exec,eval,SetSubset.read,SetSubset.write,nodeMode,left,right,before,after,emit,Bind.bind,Except.bind]

private theorem step_equal (ops : Ops α) (fuel : Nat) (state : Machine α)
    (nodeMode : state.nodeMode = true)
    (left : state.i < state.left.children.length) (right : state.j < state.right.children.length)
    (before : ops.less state.left.children[state.i].1 state.right.children[state.j].1 = false)
    (after : ops.less state.right.children[state.j].1 state.left.children[state.i].1 = false) :
    exec ops (fuel+18) loopProgram state =
      exec ops (fuel+17) loopProgram
        {state with
          i := state.i+1,
          j := state.j+1,
          child := ops.difference state.left.children[state.i].2 state.right.children[state.j].2,
          output := state.output ++ emit ops .both state.left.children[state.i] state.right.children[state.j]} := by
  apply loop_step
  · simp [condition_value ops state nodeMode,left,right]
  · cases isEmpty : ops.isEmpty (ops.difference state.left.children[state.i].2 state.right.children[state.j].2) <;>
      simp [loopBody,SetSource.nodeDifferenceProgram,exec,eval,evalAll,assign,SetSubset.read,SetSubset.write,nodeMode,left,right,before,after,emit,isEmpty,Bind.bind,Except.bind]

private theorem loop_exit (ops : Ops α) (fuel : Nat) (state : Machine α)
    (nodeMode : state.nodeMode = true)
    (safeLeft : state.i ≤ state.left.children.length) (_safeRight : state.j ≤ state.right.children.length)
    (stop : ¬(state.i < state.left.children.length ∧ state.j < state.right.children.length)) :
    exec ops (fuel+17) loopProgram state =
      .ok (.returned [.nodes (state.output ++ state.left.children.drop state.i)]) := by
  have stopped : (decide (state.i < state.left.children.length) && decide (state.j < state.right.children.length)) = false := by
    by_cases left : state.i < state.left.children.length <;>
      by_cases right : state.j < state.right.children.length <;> simp_all
  rw [loop_shape,exec,condition_value ops state nodeMode,stopped]
  change exec ops (fuel+16) (SetSource.nodeDifferenceProgram.drop 3) state = _
  by_cases left : state.i < state.left.children.length
  · simp [SetSource.nodeDifferenceProgram,exec,eval,evalAll,SetSubset.read,nodeMode,left,safeLeft,Bind.bind,Except.bind]
  · have leftEnd : state.i = state.left.children.length := by omega
    simp [SetSource.nodeDifferenceProgram,exec,eval,evalAll,SetSubset.read,nodeMode,leftEnd,Bind.bind,Except.bind]

private theorem nodes_compute_nil_right (ops : Ops α) (left : List (α × Tree α)) :
    nodesCompute ops.less ops.isEmpty ops.difference left [] = left := by
  cases left <;> simp [nodesCompute]

/-- The remaining input lengths bound every imported child-node loop. -/
private theorem loop_matches (ops : Ops α) (fuel : Nat) (state : Machine α)
    (nodeMode : state.nodeMode = true)
    (safeLeft : state.i ≤ state.left.children.length) (safeRight : state.j ≤ state.right.children.length)
    (enough : state.left.children.length-state.i + (state.right.children.length-state.j) ≤ fuel) :
    exec ops (fuel+17) loopProgram state =
      .ok (.returned [.nodes (state.output ++
        nodesCompute ops.less ops.isEmpty ops.difference (state.left.children.drop state.i) (state.right.children.drop state.j))]) := by
  induction fuel generalizing state with
  | zero =>
      have leftEnd : state.i = state.left.children.length := by omega
      have rightEnd : state.j = state.right.children.length := by omega
      rw [loop_exit ops 0 state nodeMode safeLeft safeRight (by omega)]
      simp [leftEnd,rightEnd,nodesCompute]
  | succ fuel induction =>
      by_cases left : state.i < state.left.children.length
      · by_cases right : state.j < state.right.children.length
        · have leftDrop := List.drop_eq_getElem_cons left
          have rightDrop := List.drop_eq_getElem_cons right
          by_cases before : ops.less state.left.children[state.i].1 state.right.children[state.j].1 = true
          · change exec ops (fuel+18) loopProgram state = _
            rw [step_left ops fuel state nodeMode left right before]
            rw [induction _ (by simpa using nodeMode) (by simp; omega) (by simpa using safeRight) (by simp; omega)]
            have step : nodesCompute ops.less ops.isEmpty ops.difference (state.left.children.drop state.i) (state.right.children.drop state.j) =
                emit ops .left state.left.children[state.i] state.right.children[state.j] ++
                  nodesCompute ops.less ops.isEmpty ops.difference (state.left.children.drop (state.i+1)) (state.right.children.drop state.j) := by
              conv => lhs; rw [leftDrop,rightDrop,nodesCompute,ite_eq_left before]
              rw [rightDrop]
              rfl
            dsimp only
            rw [step,List.append_assoc]
          · have before : ops.less state.left.children[state.i].1 state.right.children[state.j].1 = false := Bool.eq_false_iff.mpr before
            by_cases after : ops.less state.right.children[state.j].1 state.left.children[state.i].1 = true
            · change exec ops (fuel+18) loopProgram state = _
              rw [step_right ops fuel state nodeMode left right before after]
              rw [induction _ (by simpa using nodeMode) (by simpa using safeLeft) (by simp; omega) (by simp; omega)]
              have step : nodesCompute ops.less ops.isEmpty ops.difference (state.left.children.drop state.i) (state.right.children.drop state.j) =
                  emit ops .right state.left.children[state.i] state.right.children[state.j] ++
                    nodesCompute ops.less ops.isEmpty ops.difference (state.left.children.drop state.i) (state.right.children.drop (state.j+1)) := by
                conv => lhs; rw [leftDrop,rightDrop,nodesCompute,ite_eq_right (by simp [before]),ite_eq_left after]
                rw [leftDrop]
                simp [emit]
              dsimp only
              rw [step,List.append_assoc]
            · have after : ops.less state.right.children[state.j].1 state.left.children[state.i].1 = false := Bool.eq_false_iff.mpr after
              change exec ops (fuel+18) loopProgram state = _
              rw [step_equal ops fuel state nodeMode left right before after]
              rw [induction _ (by simpa using nodeMode) (by simp; omega) (by simp; omega) (by simp; omega)]
              have step : nodesCompute ops.less ops.isEmpty ops.difference (state.left.children.drop state.i) (state.right.children.drop state.j) =
                  emit ops .both state.left.children[state.i] state.right.children[state.j] ++
                    nodesCompute ops.less ops.isEmpty ops.difference (state.left.children.drop (state.i+1)) (state.right.children.drop (state.j+1)) := by
                conv => lhs; rw [leftDrop,rightDrop,nodesCompute,ite_eq_right (by simp [before]),ite_eq_right (by simp [after])]
                rfl
              dsimp only
              rw [step,List.append_assoc]
        · have rightEnd : state.j = state.right.children.length := by omega
          rw [loop_exit ops (fuel+1) state nodeMode safeLeft safeRight (by omega)]
          simp [rightEnd,nodes_compute_nil_right]
      · have leftEnd : state.i = state.left.children.length := by omega
        rw [loop_exit ops (fuel+1) state nodeMode safeLeft safeRight (by omega)]
        simp [leftEnd,nodesCompute]

/-- The complete imported child method matches its recursive algorithm at every finite size. -/
theorem imported_node_matches (ops : Ops α) (left right : Tree α) :
    exec ops (left.children.length+right.children.length+19) SetSource.nodeDifferenceProgram {left,right} =
      .ok (.returned [.nodes (nodesCompute ops.less ops.isEmpty ops.difference left.children right.children)]) := by
  have loop := loop_matches ops (left.children.length+right.children.length) ({left,right} : Machine α)
    rfl (by simp) (by simp) (by simp)
  have split : SetSource.nodeDifferenceProgram = SetSource.nodeDifferenceProgram.take 2 ++ loopProgram := by
    unfold loopProgram
    rfl
  rw [split]
  simpa [SetSource.nodeDifferenceProgram,exec,evalAll,eval,assign,SetSubset.write,Nat.add_assoc,Bind.bind,Except.bind] using loop

/-- Direct member subtraction is the already source-verified element algorithm. -/
theorem imported_members_match (ops : Ops α) (left right : List α) :
    ElementSubset.exec ops.less (left.length+right.length+19) ElementSource.differenceProgram {left,right} =
      .ok (.returned [.set (ElementAlgebra.compute .difference ops.less left right)]) :=
  ElementAlgebra.imported_matches .difference ops.less left right

/-- Both recursive callbacks are now fixed to the finite structural implementation. -/
def recursiveOps (lt : α → α → Bool) (isEmpty : Tree α → Bool) : Ops α :=
  {less := lt,isEmpty,difference := difference lt isEmpty,nodesDifference := nodesDifference lt isEmpty,
    has := fun _ _ => false}

theorem imported_set_matches (lt : α → α → Bool) (isEmpty : Tree α → Bool) (left right : Tree α) :
    exec (recursiveOps lt isEmpty) 2 SetSource.setDifferenceProgram {left,right,nodeMode := false} =
      .ok (.returned [.tree (difference lt isEmpty left right)]) := by
  rw [difference]
  simp [SetSource.setDifferenceProgram,exec,evalAll,eval,SetSubset.read,recursiveOps,Bind.bind,Except.bind]

theorem recursive_node_call_is_source (lt : α → α → Bool) (isEmpty : Tree α → Bool) (left right : Tree α) :
    exec (recursiveOps lt isEmpty) (left.children.length+right.children.length+19)
      SetSource.nodeDifferenceProgram {left,right} =
      .ok (.returned [.nodes (nodesDifference lt isEmpty left.children right.children)]) := by
  rw [imported_node_matches]
  simp only [recursiveOps,← nodes_difference_eq_compute]

/-- The imported top-level result cannot contain any exact path absent from the left input. -/
theorem imported_difference_never_adds_paths (lt : α → α → Bool) (isEmpty : Tree α → Bool) (left right result : Tree α)
    (returned : exec (recursiveOps lt isEmpty) 2 SetSource.setDifferenceProgram {left,right,nodeMode := false} =
      .ok (.returned [.tree result])) : PathsFrom left result := by
  rw [imported_set_matches] at returned
  cases returned
  exact difference_never_adds_paths lt isEmpty left right

end SSA.SetContracts
