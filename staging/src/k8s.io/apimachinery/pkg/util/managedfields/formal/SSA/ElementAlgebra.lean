/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.ElementOrder
import SSA.ElementSource

namespace SSA.ElementAlgebra

open Lean ElementSubset ElementOrder

variable {α : Type} {lt : α → α → Bool}

inductive Operation where
  | difference
  | union
  | intersection
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

inductive Step where
  | left
  | right
  | both

/-- Union emits the right representative on equal keys, exactly as the Go body does. -/
def emit (operation : Operation) (step : Step) (a b : α) : List α :=
  match operation,step with
  | .difference,.left | .union,.left => [a]
  | .union,.right | .union,.both => [b]
  | .intersection,.both => [a]
  | _,_ => []

def finish (operation : Operation) (left right : List α) : List α :=
  match operation with
  | .difference => left
  | .union => left ++ right
  | .intersection => []

/-- Common two-pointer algorithm, factored independently of the set-membership specification. -/
def compute (operation : Operation) (lt : α → α → Bool) : List α → List α → List α
  | [],right => finish operation [] right
  | left,[] => finish operation left []
  | a::left,b::right =>
      if lt a b then emit operation .left a b ++ compute operation lt left (b::right)
      else if lt b a then emit operation .right a b ++ compute operation lt (a::left) right
      else emit operation .both a b ++ compute operation lt left right
termination_by left right => left.length+right.length

def membership (operation : Operation) (left right : List α) (x : α) : Prop :=
  match operation with
  | .difference => x ∈ left ∧ x ∉ right
  | .union => x ∈ left ∨ x ∈ right
  | .intersection => x ∈ left ∧ x ∈ right

private theorem compute_nil_right (operation : Operation) (lt : α → α → Bool) (left : List α) :
    compute operation lt left [] = finish operation left [] := by
  cases left <;> simp [compute,finish]

/-- The three algorithms compute their usual membership laws at arbitrary finite sizes. -/
theorem compute_membership (operation : Operation) (laws : StrictOrder lt) (left right : List α)
    (sortedLeft : left.Pairwise (fun a b => lt a b = true))
    (sortedRight : right.Pairwise (fun a b => lt a b = true)) (x : α) :
    x ∈ compute operation lt left right ↔ membership operation left right x := by
  match left,right with
  | [],_ => cases operation <;> simp [compute,finish,membership]
  | a::left,[] => cases operation <;> simp [compute,finish,membership]
  | a::left,b::right =>
      by_cases ab : lt a b = true
      · have absent := before_absent laws a b right sortedRight ab
        have induction := compute_membership operation laws left (b::right)
          (List.pairwise_cons.mp sortedLeft).2 sortedRight x
        rw [compute,ite_eq_left ab]
        cases operation <;> simp only [emit,List.singleton_append,List.nil_append,List.mem_cons,membership] at * <;> grind
      · have ab : lt a b = false := Bool.eq_false_iff.mpr ab
        by_cases ba : lt b a = true
        · have absent := before_absent laws b a left sortedLeft ba
          have induction := compute_membership operation laws (a::left) right
            sortedLeft (List.pairwise_cons.mp sortedRight).2 x
          rw [compute,ite_eq_right (by simp [ab]),ite_eq_left ba]
          cases operation <;> simp only [emit,List.singleton_append,List.nil_append,List.mem_cons,membership] at * <;> grind
        · have ba : lt b a = false := Bool.eq_false_iff.mpr ba
          have same := laws.total a b ab ba
          subst b
          have absent := head_absent laws a left sortedLeft
          have induction := compute_membership operation laws left right
            (List.pairwise_cons.mp sortedLeft).2 (List.pairwise_cons.mp sortedRight).2 x
          rw [compute,ite_eq_right (by simp [ab]),ite_eq_right (by simp [ba])]
          cases operation <;> simp only [emit,List.singleton_append,List.nil_append,List.mem_cons,membership] at * <;> grind
termination_by left.length+right.length

/-- Difference and intersection remove elements without reordering or duplicating the left input. -/
theorem compute_left_sublist (operation : Operation) (lt : α → α → Bool) (left right : List α)
    (notUnion : operation ≠ .union) : (compute operation lt left right).Sublist left := by
  match left,right with
  | [],_ => cases operation <;> simp_all [compute,finish]
  | a::left,[] => cases operation <;> simp_all [compute,finish]
  | a::left,b::right =>
      by_cases ab : lt a b = true
      · rw [compute,ite_eq_left ab]
        have induction := compute_left_sublist operation lt left (b::right) notUnion
        cases operation with
        | union => exact False.elim (notUnion rfl)
        | difference => simpa [emit] using induction.cons_cons a
        | intersection => simpa [emit] using induction.cons a
      · have ab : lt a b = false := Bool.eq_false_iff.mpr ab
        by_cases ba : lt b a = true
        · rw [compute,ite_eq_right (by simp [ab]),ite_eq_left ba]
          have induction := compute_left_sublist operation lt (a::left) right notUnion
          cases operation <;> simp_all [emit]
        · rw [compute,ite_eq_right (by simp [ab]),ite_eq_right ba]
          have induction := compute_left_sublist operation lt left right notUnion
          cases operation with
          | union => exact False.elim (notUnion rfl)
          | difference => simpa [emit] using induction.cons a
          | intersection => simpa [emit] using induction.cons_cons a
termination_by left.length+right.length

/-- Union retains the strict ordering required by the next set operation. -/
theorem union_pairwise (laws : StrictOrder lt) (left right : List α)
    (sortedLeft : left.Pairwise (fun a b => lt a b = true))
    (sortedRight : right.Pairwise (fun a b => lt a b = true)) :
    (compute .union lt left right).Pairwise (fun a b => lt a b = true) := by
  match left,right with
  | [],_ => simpa [compute,finish] using sortedRight
  | a::left,[] => simpa [compute,finish] using sortedLeft
  | a::left,b::right =>
      by_cases ab : lt a b = true
      · rw [compute,ite_eq_left ab]
        simp only [emit,List.singleton_append,List.pairwise_cons]
        constructor
        · intro x present
          have member := (compute_membership .union laws left (b::right)
            (List.pairwise_cons.mp sortedLeft).2 sortedRight x).mp present
          rcases member with member | member
          · exact (List.pairwise_cons.mp sortedLeft).1 x member
          · exact before_all laws a b right sortedRight ab x member
        · exact union_pairwise laws left (b::right) (List.pairwise_cons.mp sortedLeft).2 sortedRight
      · have ab : lt a b = false := Bool.eq_false_iff.mpr ab
        by_cases ba : lt b a = true
        · rw [compute,ite_eq_right (by simp [ab]),ite_eq_left ba]
          simp only [emit,List.singleton_append,List.pairwise_cons]
          constructor
          · intro x present
            have member := (compute_membership .union laws (a::left) right
              sortedLeft (List.pairwise_cons.mp sortedRight).2 x).mp present
            rcases member with member | member
            · exact before_all laws b a left sortedLeft ba x member
            · exact (List.pairwise_cons.mp sortedRight).1 x member
          · exact union_pairwise laws (a::left) right sortedLeft (List.pairwise_cons.mp sortedRight).2
        · have ba : lt b a = false := Bool.eq_false_iff.mpr ba
          have same := laws.total a b ab ba
          subst b
          rw [compute,ite_eq_right (by simp [ab]),ite_eq_right (by simp [ba])]
          simp only [emit,List.singleton_append,List.pairwise_cons]
          constructor
          · intro x present
            have member := (compute_membership .union laws left right
              (List.pairwise_cons.mp sortedLeft).2 (List.pairwise_cons.mp sortedRight).2 x).mp present
            rcases member with member | member
            · exact (List.pairwise_cons.mp sortedLeft).1 x member
            · exact (List.pairwise_cons.mp sortedRight).1 x member
          · exact union_pairwise laws left right
              (List.pairwise_cons.mp sortedLeft).2 (List.pairwise_cons.mp sortedRight).2
termination_by left.length+right.length

def program : Operation → List GoSubset.Stmt
  | .difference => ElementSource.differenceProgram
  | .union => ElementSource.unionProgram
  | .intersection => ElementSource.intersectionProgram

@[irreducible] private def loopProgram (operation : Operation) := (program operation).drop 2

private def loopCondition : GoSubset.Expr :=
  match (ElementSource.differenceProgram[2]? : Option GoSubset.Stmt) with
  | some (.whileLoop condition _) => condition
  | _ => .flag false

private def loopBody (operation : Operation) : List GoSubset.Stmt :=
  match ((program operation)[2]? : Option GoSubset.Stmt) with
  | some (.whileLoop _ body) => body
  | _ => []

private theorem loop_shape (operation : Operation) : loopProgram operation =
    .whileLoop loopCondition (loopBody operation) :: (program operation).drop 3 := by
  cases operation <;> unfold loopProgram loopCondition loopBody program <;> rfl

private theorem loop_step (operation : Operation) (lt : α → α → Bool) (fuel : Nat) (state next : Machine α)
    (condition : eval lt 32 loopCondition state = .ok (.flag true))
    (body : exec lt fuel (loopBody operation) state = .ok (.next next)) :
    exec lt (fuel+1) (loopProgram operation) state = exec lt fuel (loopProgram operation) next := by
  rw [loop_shape,exec,condition,body]
  rfl

private theorem condition_value (lt : α → α → Bool) (state : Machine α) :
    eval lt 32 loopCondition state =
      .ok (.flag (state.i < state.left.length && state.j < state.right.length)) := by
  by_cases left : state.i < state.left.length <;>
    simp [loopCondition,ElementSource.differenceProgram,eval,ElementSubset.read,left,Bind.bind,Except.bind]

private theorem step_left (operation : Operation) (lt : α → α → Bool) (fuel : Nat) (state : Machine α)
    (left : state.i < state.left.length) (right : state.j < state.right.length)
    (before : lt state.left[state.i] state.right[state.j] = true) :
    exec lt (fuel+18) (loopProgram operation) state =
      exec lt (fuel+17) (loopProgram operation)
        {state with i := state.i+1,output := state.output ++ emit operation .left state.left[state.i] state.right[state.j]} := by
  apply loop_step
  · simp [condition_value,left,right]
  · cases operation <;>
      simp [loopBody,program,ElementSource.differenceProgram,ElementSource.unionProgram,
        ElementSource.intersectionProgram,exec,eval,ElementSubset.read,ElementSubset.write,
        left,right,before,emit,Bind.bind,Except.bind]

private theorem step_right (operation : Operation) (lt : α → α → Bool) (fuel : Nat) (state : Machine α)
    (left : state.i < state.left.length) (right : state.j < state.right.length)
    (before : lt state.left[state.i] state.right[state.j] = false)
    (after : lt state.right[state.j] state.left[state.i] = true) :
    exec lt (fuel+18) (loopProgram operation) state =
      exec lt (fuel+17) (loopProgram operation)
        {state with j := state.j+1,output := state.output ++ emit operation .right state.left[state.i] state.right[state.j]} := by
  apply loop_step
  · simp [condition_value,left,right]
  · cases operation <;>
      simp [loopBody,program,ElementSource.differenceProgram,ElementSource.unionProgram,
        ElementSource.intersectionProgram,exec,eval,ElementSubset.read,ElementSubset.write,
        left,right,before,after,emit,Bind.bind,Except.bind]

private theorem step_equal (operation : Operation) (lt : α → α → Bool) (fuel : Nat) (state : Machine α)
    (left : state.i < state.left.length) (right : state.j < state.right.length)
    (before : lt state.left[state.i] state.right[state.j] = false)
    (after : lt state.right[state.j] state.left[state.i] = false) :
    exec lt (fuel+18) (loopProgram operation) state =
      exec lt (fuel+17) (loopProgram operation)
        {state with
          i := state.i+1,
          j := state.j+1,
          output := state.output ++ emit operation .both state.left[state.i] state.right[state.j]} := by
  apply loop_step
  · simp [condition_value,left,right]
  · cases operation <;>
      simp [loopBody,program,ElementSource.differenceProgram,ElementSource.unionProgram,
        ElementSource.intersectionProgram,exec,eval,ElementSubset.read,ElementSubset.write,
        left,right,before,after,emit,Bind.bind,Except.bind]

private theorem loop_exit (operation : Operation) (lt : α → α → Bool) (fuel : Nat) (state : Machine α)
    (safeLeft : state.i ≤ state.left.length) (safeRight : state.j ≤ state.right.length)
    (stop : ¬(state.i < state.left.length ∧ state.j < state.right.length)) :
    exec lt (fuel+17) (loopProgram operation) state =
      .ok (.returned [.set (state.output ++
        finish operation (state.left.drop state.i) (state.right.drop state.j))]) := by
  have stopped : (decide (state.i < state.left.length) && decide (state.j < state.right.length)) = false := by
    by_cases left : state.i < state.left.length <;>
      by_cases right : state.j < state.right.length <;> simp_all
  rw [loop_shape,exec,condition_value,stopped]
  change exec lt (fuel+16) ((program operation).drop 3) state = _
  by_cases left : state.i < state.left.length
  · have rightEnd : state.j = state.right.length := by omega
    cases operation <;>
      simp [program,ElementSource.differenceProgram,ElementSource.unionProgram,
        ElementSource.intersectionProgram,exec,eval,evalAll,ElementSubset.read,
        left,rightEnd,safeLeft,finish,Bind.bind,Except.bind]
  · have leftEnd : state.i = state.left.length := by omega
    by_cases right : state.j < state.right.length
    · cases operation <;>
        simp [program,ElementSource.differenceProgram,ElementSource.unionProgram,
          ElementSource.intersectionProgram,exec,eval,evalAll,ElementSubset.read,
          leftEnd,right,safeRight,finish,Bind.bind,Except.bind]
    · have rightEnd : state.j = state.right.length := by omega
      cases operation <;>
        simp [program,ElementSource.differenceProgram,ElementSource.unionProgram,
          ElementSource.intersectionProgram,exec,eval,evalAll,ElementSubset.read,
          leftEnd,rightEnd,finish,Bind.bind,Except.bind]

/-- One common induction covers every loop length and all three operation-specific output policies. -/
private theorem loop_matches (operation : Operation) (lt : α → α → Bool) (fuel : Nat) (state : Machine α)
    (safeLeft : state.i ≤ state.left.length) (safeRight : state.j ≤ state.right.length)
    (enough : state.left.length-state.i + (state.right.length-state.j) ≤ fuel) :
    exec lt (fuel+17) (loopProgram operation) state =
      .ok (.returned [.set (state.output ++
        compute operation lt (state.left.drop state.i) (state.right.drop state.j))]) := by
  induction fuel generalizing state with
  | zero =>
      have leftEnd : state.i = state.left.length := by omega
      have rightEnd : state.j = state.right.length := by omega
      rw [loop_exit operation lt 0 state safeLeft safeRight (by omega)]
      simp [leftEnd,rightEnd,compute]
  | succ fuel induction =>
      by_cases left : state.i < state.left.length
      · by_cases right : state.j < state.right.length
        · have leftDrop := List.drop_eq_getElem_cons left
          have rightDrop := List.drop_eq_getElem_cons right
          by_cases before : lt state.left[state.i] state.right[state.j] = true
          · change exec lt (fuel+18) (loopProgram operation) state = _
            rw [step_left operation lt fuel state left right before]
            rw [induction _ (by simp; omega) (by simpa using safeRight) (by simp; omega)]
            have step : compute operation lt (state.left.drop state.i) (state.right.drop state.j) =
                emit operation .left state.left[state.i] state.right[state.j] ++
                  compute operation lt (state.left.drop (state.i+1)) (state.right.drop state.j) := by
              conv => lhs; rw [leftDrop,rightDrop,compute,ite_eq_left before]
              rw [rightDrop]
            dsimp only
            rw [step,List.append_assoc]
          · have before : lt state.left[state.i] state.right[state.j] = false := Bool.eq_false_iff.mpr before
            by_cases after : lt state.right[state.j] state.left[state.i] = true
            · change exec lt (fuel+18) (loopProgram operation) state = _
              rw [step_right operation lt fuel state left right before after]
              rw [induction _ (by simpa using safeLeft) (by simp; omega) (by simp; omega)]
              have step : compute operation lt (state.left.drop state.i) (state.right.drop state.j) =
                  emit operation .right state.left[state.i] state.right[state.j] ++
                    compute operation lt (state.left.drop state.i) (state.right.drop (state.j+1)) := by
                conv => lhs; rw [leftDrop,rightDrop,compute,ite_eq_right (by simp [before]),ite_eq_left after]
                rw [leftDrop]
              dsimp only
              rw [step,List.append_assoc]
            · have after : lt state.right[state.j] state.left[state.i] = false := Bool.eq_false_iff.mpr after
              change exec lt (fuel+18) (loopProgram operation) state = _
              rw [step_equal operation lt fuel state left right before after]
              rw [induction _ (by simp; omega) (by simp; omega) (by simp; omega)]
              have step : compute operation lt (state.left.drop state.i) (state.right.drop state.j) =
                  emit operation .both state.left[state.i] state.right[state.j] ++
                    compute operation lt (state.left.drop (state.i+1)) (state.right.drop (state.j+1)) := by
                conv => lhs; rw [leftDrop,rightDrop,compute,ite_eq_right (by simp [before]),ite_eq_right (by simp [after])]
              dsimp only
              rw [step,List.append_assoc]
        · have rightEnd : state.j = state.right.length := by omega
          rw [loop_exit operation lt (fuel+1) state safeLeft safeRight (by omega)]
          simp [rightEnd,compute_nil_right]
      · have leftEnd : state.i = state.left.length := by omega
        rw [loop_exit operation lt (fuel+1) state safeLeft safeRight (by omega)]
        simp [leftEnd,compute]

/-- All complete imported methods equal the common algorithm for arbitrary finite inputs and comparators. -/
theorem imported_matches (operation : Operation) (lt : α → α → Bool) (left right : List α) :
    exec lt (left.length+right.length+19) (program operation) {left := left,right := right} =
      .ok (.returned [.set (compute operation lt left right)]) := by
  have loop := loop_matches operation lt (left.length+right.length) ({left := left,right := right} : Machine α)
    (by simp) (by simp) (by simp)
  have split : program operation = (program operation).take 2 ++ loopProgram operation := by
    unfold loopProgram
    cases operation <;> rfl
  rw [split]
  cases operation <;>
    simpa [program,ElementSource.differenceProgram,ElementSource.unionProgram,ElementSource.intersectionProgram,
      exec,evalAll,eval,assign,ElementSubset.write,Nat.add_assoc,Bind.bind,Except.bind] using loop

/-- Under the explicit ordering premises, each returned source result has its exact set-membership law. -/
theorem imported_membership (operation : Operation) (laws : StrictOrder lt) (left right : List α)
    (sortedLeft : left.Pairwise (fun a b => lt a b = true))
    (sortedRight : right.Pairwise (fun a b => lt a b = true)) (result : List α)
    (returned : exec lt (left.length+right.length+19) (program operation) {left := left,right := right} =
      .ok (.returned [.set result])) (x : α) : x ∈ result ↔ membership operation left right x := by
  rw [imported_matches] at returned
  cases returned
  exact compute_membership operation laws left right sortedLeft sortedRight x

/-- Every imported operation preserves the sorted representation needed for subsequent calls. -/
theorem imported_preserves_order (operation : Operation) (laws : StrictOrder lt) (left right : List α)
    (sortedLeft : left.Pairwise (fun a b => lt a b = true))
    (sortedRight : right.Pairwise (fun a b => lt a b = true)) (result : List α)
    (returned : exec lt (left.length+right.length+19) (program operation) {left := left,right := right} =
      .ok (.returned [.set result])) : result.Pairwise (fun a b => lt a b = true) := by
  rw [imported_matches] at returned
  cases returned
  cases operation with
  | union => exact union_pairwise laws left right sortedLeft sortedRight
  | difference => exact List.Pairwise.sublist (compute_left_sublist .difference lt left right (by decide)) sortedLeft
  | intersection => exact List.Pairwise.sublist (compute_left_sublist .intersection lt left right (by decide)) sortedLeft

/-- Sorted canonical inputs cannot produce duplicate element keys in any imported operation. -/
theorem imported_preserves_uniqueness (operation : Operation) (laws : StrictOrder lt) (left right : List α)
    (sortedLeft : left.Pairwise (fun a b => lt a b = true))
    (sortedRight : right.Pairwise (fun a b => lt a b = true)) (result : List α)
    (returned : exec lt (left.length+right.length+19) (program operation) {left := left,right := right} =
      .ok (.returned [.set result])) : result.Nodup := by
  exact sorted_unique laws result
    (imported_preserves_order operation laws left right sortedLeft sortedRight result returned)

end SSA.ElementAlgebra
