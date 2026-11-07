/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.ElementSource
import SSA.ElementSubset

namespace SSA.ElementDifference

open ElementSubset

variable {α : Type} {lt : α → α → Bool}

/-- The actual PathElement comparator must realize these laws on the represented domain. -/
structure StrictOrder (lt : α → α → Bool) : Prop where
  irrefl : ∀ a, lt a a = false
  trans : ∀ a b c, lt a b = true → lt b c = true → lt a c = true
  total : ∀ a b, lt a b = false → lt b a = false → a = b

/-- Functional factorization of the two-pointer Go loop, not the membership specification. -/
def difference (lt : α → α → Bool) : List α → List α → List α
  | [],_ => []
  | left,[] => left
  | a::left,b::right =>
      if lt a b then a::difference lt left (b::right)
      else if lt b a then difference lt (a::left) right
      else difference lt left right
termination_by left right => left.length+right.length

private theorem difference_nil_right (lt : α → α → Bool) (left : List α) :
    difference lt left [] = left := by
  cases left <;> simp [difference]

/-- The algorithm only removes left-hand elements; their order and uniqueness are preserved. -/
theorem difference_sublist (lt : α → α → Bool) (left right : List α) :
    (difference lt left right).Sublist left := by
  match left,right with
  | [],_ => simp [difference]
  | a::left,[] => simp [difference]
  | a::left,b::right =>
      by_cases before : lt a b = true
      · rw [difference,ite_eq_left before]
        exact (difference_sublist lt left (b::right)).cons_cons a
      · have before : lt a b = false := Bool.eq_false_iff.mpr before
        rw [difference,ite_eq_right (by simp [before])]
        by_cases after : lt b a = true
        · rw [ite_eq_left after]
          exact difference_sublist lt (a::left) right
        · rw [ite_eq_right after]
          exact (difference_sublist lt left right).cons a
termination_by left.length+right.length

private theorem head_absent (laws : StrictOrder lt) (a : α) (rest : List α)
    (sorted : (a::rest).Pairwise (fun a b => lt a b = true)) : a ∉ rest := by
  intro present
  have before := (List.pairwise_cons.mp sorted).1 a present
  simp [laws.irrefl] at before

private theorem before_absent (laws : StrictOrder lt) (a b : α) (rest : List α)
    (sorted : (b::rest).Pairwise (fun a b => lt a b = true))
    (before : lt a b = true) : a ∉ b::rest := by
  intro present
  rcases List.mem_cons.mp present with same | present
  · subst b
    simp [laws.irrefl] at before
  · have next := (List.pairwise_cons.mp sorted).1 a present
    have cycle := laws.trans a b a before next
    simp [laws.irrefl] at cycle

/-- Arbitrary finite sorted inputs yield exactly ordinary membership difference. -/
theorem difference_membership (laws : StrictOrder lt) (left right : List α)
    (sortedLeft : left.Pairwise (fun a b => lt a b = true))
    (sortedRight : right.Pairwise (fun a b => lt a b = true)) (x : α) :
    x ∈ difference lt left right ↔ x ∈ left ∧ x ∉ right := by
  match left,right with
  | [],_ => simp [difference]
  | a::left,[] => simp [difference]
  | a::left,b::right =>
      by_cases ab : lt a b = true
      · have absent := before_absent laws a b right sortedRight ab
        have induction := difference_membership laws left (b::right)
          (List.pairwise_cons.mp sortedLeft).2 sortedRight x
        simp only [difference,ab,ite_true,List.mem_cons,induction]
        grind
      · have ab : lt a b = false := Bool.eq_false_iff.mpr ab
        by_cases ba : lt b a = true
        · have absent := before_absent laws b a left sortedLeft ba
          have induction := difference_membership laws (a::left) right
            sortedLeft (List.pairwise_cons.mp sortedRight).2 x
          simp only [difference,ab,ba,Bool.false_eq_true,ite_false,ite_true,induction,List.mem_cons]
          grind
        · have ba : lt b a = false := Bool.eq_false_iff.mpr ba
          have same := laws.total a b ab ba
          subst b
          have absent := head_absent laws a left sortedLeft
          have induction := difference_membership laws left right
            (List.pairwise_cons.mp sortedLeft).2 (List.pairwise_cons.mp sortedRight).2 x
          simp only [difference,ab,Bool.false_eq_true,ite_false,induction,List.mem_cons]
          grind
termination_by left.length+right.length

@[irreducible] private def loopProgram := ElementSource.differenceProgram.drop 2

private def loopCondition : GoSubset.Expr :=
  match (ElementSource.differenceProgram[2]? : Option GoSubset.Stmt) with
  | some (.whileLoop condition _) => condition
  | _ => .flag false

private def loopBody : List GoSubset.Stmt :=
  match (ElementSource.differenceProgram[2]? : Option GoSubset.Stmt) with
  | some (.whileLoop _ body) => body
  | _ => []

private theorem loop_shape : loopProgram =
    .whileLoop loopCondition loopBody :: ElementSource.differenceProgram.drop 3 := by
  unfold loopProgram loopCondition loopBody
  rfl

private theorem loop_step (lt : α → α → Bool) (fuel : Nat) (state next : Machine α)
    (condition : eval lt 32 loopCondition state = .ok (.flag true))
    (body : exec lt fuel loopBody state = .ok (.next next)) :
    exec lt (fuel+1) loopProgram state = exec lt fuel loopProgram next := by
  rw [loop_shape]
  rw [exec,condition,body]
  rfl

private theorem condition_value (lt : α → α → Bool) (state : Machine α) :
    eval lt 32 loopCondition state =
      .ok (.flag (state.i < state.left.length && state.j < state.right.length)) := by
  by_cases left : state.i < state.left.length <;>
    simp [loopCondition,ElementSource.differenceProgram,eval,ElementSubset.read,left,Bind.bind,Except.bind]

private theorem step_left (lt : α → α → Bool) (fuel : Nat) (state : Machine α)
    (left : state.i < state.left.length) (right : state.j < state.right.length)
    (before : lt state.left[state.i] state.right[state.j] = true) :
    exec lt (fuel+18) loopProgram state =
      exec lt (fuel+17) loopProgram
        {state with i := state.i+1,output := state.output ++ [state.left[state.i]]} := by
  apply loop_step
  · simp [condition_value,left,right]
  · simp [loopBody,ElementSource.differenceProgram,exec,eval,ElementSubset.read,ElementSubset.write,left,right,before,
      Bind.bind,Except.bind]

private theorem step_right (lt : α → α → Bool) (fuel : Nat) (state : Machine α)
    (left : state.i < state.left.length) (right : state.j < state.right.length)
    (before : lt state.left[state.i] state.right[state.j] = false)
    (after : lt state.right[state.j] state.left[state.i] = true) :
    exec lt (fuel+18) loopProgram state =
      exec lt (fuel+17) loopProgram {state with j := state.j+1} := by
  apply loop_step
  · simp [condition_value,left,right]
  · simp [loopBody,ElementSource.differenceProgram,exec,eval,ElementSubset.read,ElementSubset.write,left,right,before,after,
      Bind.bind,Except.bind]

private theorem step_equal (lt : α → α → Bool) (fuel : Nat) (state : Machine α)
    (left : state.i < state.left.length) (right : state.j < state.right.length)
    (before : lt state.left[state.i] state.right[state.j] = false)
    (after : lt state.right[state.j] state.left[state.i] = false) :
    exec lt (fuel+18) loopProgram state =
      exec lt (fuel+17) loopProgram {state with i := state.i+1,j := state.j+1} := by
  apply loop_step
  · simp [condition_value,left,right]
  · simp [loopBody,ElementSource.differenceProgram,exec,eval,ElementSubset.read,ElementSubset.write,left,right,before,after,
      Bind.bind,Except.bind]

private theorem loop_exit (lt : α → α → Bool) (fuel : Nat) (state : Machine α)
    (safe : state.i ≤ state.left.length)
    (stop : ¬(state.i < state.left.length ∧ state.j < state.right.length)) :
    exec lt (fuel+17) loopProgram state =
      .ok (.returned [.set (state.output ++ state.left.drop state.i)]) := by
  have stopped : (decide (state.i < state.left.length) && decide (state.j < state.right.length)) = false := by
    by_cases left : state.i < state.left.length <;>
      by_cases right : state.j < state.right.length <;> simp_all
  rw [loop_shape,exec,condition_value,stopped]
  change exec lt (fuel+16) (ElementSource.differenceProgram.drop 3) state = _
  by_cases remaining : state.i < state.left.length
  · simp [ElementSource.differenceProgram,exec,eval,evalAll,ElementSubset.read,remaining,safe,
      Bind.bind,Except.bind]
  · have endIndex : state.i = state.left.length := by omega
    simp [ElementSource.differenceProgram,exec,eval,evalAll,ElementSubset.read,endIndex,
      Bind.bind,Except.bind]

/-- The loop has a linear fuel bound, never indexes past an input, and returns the exact algorithm result. -/
private theorem loop_matches (lt : α → α → Bool) (fuel : Nat) (state : Machine α)
    (safeLeft : state.i ≤ state.left.length) (safeRight : state.j ≤ state.right.length)
    (enough : state.left.length-state.i + (state.right.length-state.j) ≤ fuel) :
    exec lt (fuel+17) loopProgram state =
      .ok (.returned [.set (state.output ++
        difference lt (state.left.drop state.i) (state.right.drop state.j))]) := by
  induction fuel generalizing state with
  | zero =>
      have leftEnd : state.i = state.left.length := by omega
      have rightEnd : state.j = state.right.length := by omega
      rw [loop_exit lt 0 state safeLeft (by omega)]
      simp [leftEnd,rightEnd,difference]
  | succ fuel induction =>
      by_cases left : state.i < state.left.length
      · by_cases right : state.j < state.right.length
        · have leftDrop := List.drop_eq_getElem_cons left
          have rightDrop := List.drop_eq_getElem_cons right
          by_cases before : lt state.left[state.i] state.right[state.j] = true
          · change exec lt (fuel+18) loopProgram state = _
            rw [step_left lt fuel state left right before]
            rw [induction _ (by simp; omega) (by simpa using safeRight) (by simp; omega)]
            have step : difference lt (state.left.drop state.i) (state.right.drop state.j) =
                state.left[state.i] :: difference lt (state.left.drop (state.i+1)) (state.right.drop state.j) := by
              conv => lhs; rw [leftDrop,rightDrop,difference,ite_eq_left before]
              rw [rightDrop]
            dsimp only
            rw [step]
            simp [List.append_assoc]
          · have before : lt state.left[state.i] state.right[state.j] = false := Bool.eq_false_iff.mpr before
            by_cases after : lt state.right[state.j] state.left[state.i] = true
            · change exec lt (fuel+18) loopProgram state = _
              rw [step_right lt fuel state left right before after]
              rw [induction _ (by simpa using safeLeft) (by simp; omega) (by simp; omega)]
              have step : difference lt (state.left.drop state.i) (state.right.drop state.j) =
                  difference lt (state.left.drop state.i) (state.right.drop (state.j+1)) := by
                conv => lhs; rw [leftDrop,rightDrop,difference,ite_eq_right (by simp [before]),ite_eq_left after]
                rw [leftDrop]
              dsimp only
              rw [step]
            · have after : lt state.right[state.j] state.left[state.i] = false := Bool.eq_false_iff.mpr after
              change exec lt (fuel+18) loopProgram state = _
              rw [step_equal lt fuel state left right before after]
              rw [induction _ (by simp; omega) (by simp; omega) (by simp; omega)]
              have step : difference lt (state.left.drop state.i) (state.right.drop state.j) =
                  difference lt (state.left.drop (state.i+1)) (state.right.drop (state.j+1)) := by
                conv => lhs; rw [leftDrop,rightDrop,difference,ite_eq_right (by simp [before]),ite_eq_right (by simp [after])]
              dsimp only
              rw [step]
        · have rightEnd : state.j = state.right.length := by omega
          rw [loop_exit lt (fuel+1) state safeLeft (by omega)]
          simp [rightEnd,difference_nil_right]
      · have leftEnd : state.i = state.left.length := by omega
        rw [loop_exit lt (fuel+1) state safeLeft (by omega)]
        simp [leftEnd,difference]

/-- The complete imported method equals two-pointer difference for every finite input, even unsorted. -/
theorem imported_difference_matches (lt : α → α → Bool) (left right : List α) :
    exec lt (left.length+right.length+19) ElementSource.differenceProgram {left := left,right := right} =
      .ok (.returned [.set (difference lt left right)]) := by
  have loop := loop_matches lt (left.length+right.length) ({left := left,right := right} : Machine α)
    (by simp) (by simp) (by simp)
  have split : ElementSource.differenceProgram = ElementSource.differenceProgram.take 2 ++ loopProgram := by
    unfold loopProgram
    rfl
  rw [split]
  simpa [ElementSource.differenceProgram,exec,evalAll,eval,assign,ElementSubset.write,
    Nat.add_assoc,Bind.bind,Except.bind] using loop

/-- The imported Go method implements membership difference on arbitrary finite ordered input sets. -/
theorem imported_difference_membership (laws : StrictOrder lt) (left right : List α)
    (sortedLeft : left.Pairwise (fun a b => lt a b = true))
    (sortedRight : right.Pairwise (fun a b => lt a b = true)) (result : List α)
    (returned : exec lt (left.length+right.length+19) ElementSource.differenceProgram {left := left,right := right} =
      .ok (.returned [.set result])) (x : α) :
    x ∈ result ↔ x ∈ left ∧ x ∉ right := by
  rw [imported_difference_matches] at returned
  cases returned
  exact difference_membership laws left right sortedLeft sortedRight x

/-- The returned Go representation remains sorted whenever the left input was sorted. -/
theorem imported_difference_preserves_order (lt : α → α → Bool) (left right : List α)
    (sorted : left.Pairwise (fun a b => lt a b = true)) (result : List α)
    (returned : exec lt (left.length+right.length+19) ElementSource.differenceProgram {left := left,right := right} =
      .ok (.returned [.set result])) : result.Pairwise (fun a b => lt a b = true) := by
  rw [imported_difference_matches] at returned
  cases returned
  exact List.Pairwise.sublist (difference_sublist lt left right) sorted

end SSA.ElementDifference
