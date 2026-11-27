/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SearchAlgorithm

namespace SSA.SearchContracts

open SearchSubset SearchAlgorithm

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

def beforeAt (compare : α → α → Int) (members : List α) (target : α) (index : Nat) : Bool :=
  (members[index]?).any (fun label => compare label target < 0)

def foundAt (compare : α → α → Int) (members : List α) (target : α) (index : Nat) : Bool :=
  (members[index]?).any (fun label => compare label target == 0)

@[irreducible] private def loopProgram := SearchSource.binarySearchProgram.drop 2

private def loopCondition : GoSubset.Expr :=
  match (SearchSource.binarySearchProgram[2]? : Option GoSubset.Stmt) with
  | some (.whileLoop condition _) => condition
  | _ => .flag false

private def loopBody : List GoSubset.Stmt :=
  match (SearchSource.binarySearchProgram[2]? : Option GoSubset.Stmt) with
  | some (.whileLoop _ body) => body
  | _ => []

private theorem loop_shape : loopProgram =
    .whileLoop loopCondition loopBody :: SearchSource.binarySearchProgram.drop 3 := by
  unfold loopProgram loopCondition loopBody
  rfl

private theorem condition_value (ops : Ops α) (state : Machine α) :
    one ops 64 loopCondition state = .ok (.flag (state.i < state.j)) := by
  simp [loopCondition,SearchSource.binarySearchProgram,one,eval,SearchSubset.read,Bind.bind,Except.bind]

private theorem loop_step (ops : Ops α) (fuel : Nat) (state next : Machine α)
    (condition : one ops 64 loopCondition state = .ok (.flag true))
    (body : exec ops fuel loopBody state = .ok (.next next)) :
    exec ops (fuel+1) loopProgram state = exec ops fuel loopProgram next := by
  rw [loop_shape,exec,condition,body]
  rfl

private theorem step_left (ops : Ops α) (fuel : Nat) (state : Machine α)
    (active : state.i < state.j)
    (safe : unsignedMidpoint ops.wordBits state.i state.j < state.members.length)
    (before : ops.compare state.members[unsignedMidpoint ops.wordBits state.i state.j] state.target < 0) :
    exec ops (fuel+18) loopProgram state =
      exec ops (fuel+17) loopProgram {state with
        i := unsignedMidpoint ops.wordBits state.i state.j+1,
        h := unsignedMidpoint ops.wordBits state.i state.j} := by
  apply loop_step
  · simp [condition_value,active]
  · simp [loopBody,SearchSource.binarySearchProgram,exec,many,one,eval,assign,SearchSubset.read,SearchSubset.write,
      safe,before,Bind.bind,Except.bind]

private theorem step_right (ops : Ops α) (fuel : Nat) (state : Machine α)
    (active : state.i < state.j)
    (safe : unsignedMidpoint ops.wordBits state.i state.j < state.members.length)
    (before : ¬ops.compare state.members[unsignedMidpoint ops.wordBits state.i state.j] state.target < 0) :
    exec ops (fuel+18) loopProgram state =
      exec ops (fuel+17) loopProgram {state with
        j := unsignedMidpoint ops.wordBits state.i state.j,
        h := unsignedMidpoint ops.wordBits state.i state.j} := by
  apply loop_step
  · simp [condition_value,active]
  · simp [loopBody,SearchSource.binarySearchProgram,exec,many,one,eval,assign,SearchSubset.read,SearchSubset.write,
      safe,before,Bind.bind,Except.bind]

private theorem loop_exit (ops : Ops α) (fuel : Nat) (state : Machine α)
    (stop : ¬state.i < state.j) (safe : state.i ≤ state.members.length) (cached : state.n = state.members.length) :
    exec ops (fuel+17) loopProgram state =
      .ok (.returned [.number state.i,.flag (foundAt ops.compare state.members state.target state.i)]) := by
  rw [loop_shape,exec,condition_value]
  simp only [stop,decide_false,Bind.bind,Except.bind]
  change exec ops (fuel+16) (SearchSource.binarySearchProgram.drop 3) state = _
  by_cases inside : state.i < state.members.length
  · simp [SearchSource.binarySearchProgram,exec,many,one,eval,SearchSubset.read,scalarEqual,foundAt,
      cached,inside,Bind.bind,Except.bind]
  · have atEnd : state.i = state.members.length := by omega
    simp [SearchSource.binarySearchProgram,exec,many,one,eval,SearchSubset.read,foundAt,
      cached,atEnd,Bind.bind,Except.bind]

/-- The size-derived fuel covers every valid interval, independently of comparator ordering. -/
private theorem loop_matches (ops : Ops α) (fuel : Nat) (state : Machine α)
    (positive : 0 < ops.wordBits) (ordered : state.i ≤ state.j)
    (within : state.j ≤ state.members.length) (representable : state.j ≤ maxIndex ops.wordBits)
    (cached : state.n = state.members.length) (enough : state.j-state.i ≤ fuel) :
    let answer := lowerBound (beforeAt ops.compare state.members state.target) state.i state.j
    exec ops (fuel+17) loopProgram state =
      .ok (.returned [.number answer,.flag (foundAt ops.compare state.members state.target answer)]) := by
  dsimp only
  induction fuel generalizing state with
  | zero =>
      have done : state.i = state.j := by omega
      rw [loop_exit ops 0 state (by omega) (by omega) cached]
      simp [done,lowerBound]
  | succ fuel induction =>
      by_cases active : state.i < state.j
      · have inside := SearchIndex.unsigned_midpoint_inside ops.wordBits positive state.i state.j active representable
        have safe : unsignedMidpoint ops.wordBits state.i state.j < state.members.length := by omega
        have progress := SearchIndex.unsigned_midpoint_progress ops.wordBits positive state.i state.j active representable
        have half := SearchIndex.unsigned_midpoint_eq_half ops.wordBits positive state.i state.j (by omega) representable
        by_cases before : ops.compare state.members[unsignedMidpoint ops.wordBits state.i state.j] state.target < 0
        · have selected : beforeAt ops.compare state.members state.target (unsignedMidpoint ops.wordBits state.i state.j) = true := by
            simp [beforeAt,safe,before]
          change exec ops (fuel+18) loopProgram state = _
          rw [step_left ops fuel state active safe before]
          rw [induction _ (by simp; omega) (by simpa using within) (by simpa using representable)
            (by simpa using cached) (by simp; omega)]
          have step : lowerBound (beforeAt ops.compare state.members state.target) state.i state.j =
              lowerBound (beforeAt ops.compare state.members state.target)
                (unsignedMidpoint ops.wordBits state.i state.j+1) state.j := by
            conv => lhs; rw [lowerBound,ite_eq_left active]; dsimp only; rw [← half,selected]
            rfl
          dsimp only
          rw [step]
        · have selected : beforeAt ops.compare state.members state.target (unsignedMidpoint ops.wordBits state.i state.j) = false := by
            simp [beforeAt,safe,before]
          change exec ops (fuel+18) loopProgram state = _
          rw [step_right ops fuel state active safe before]
          rw [induction _ (by simp; omega) (by simp; omega) (by simp; omega)
            (by simpa using cached) (by simp; omega)]
          have step : lowerBound (beforeAt ops.compare state.members state.target) state.i state.j =
              lowerBound (beforeAt ops.compare state.members state.target) state.i
                (unsignedMidpoint ops.wordBits state.i state.j) := by
            conv => lhs; rw [lowerBound,ite_eq_left active]; dsimp only; rw [← half,selected]
            rfl
          dsimp only
          rw [step]
      · have done : state.i = state.j := by omega
        rw [loop_exit ops (fuel+1) state active (by omega) cached]
        simp [done,lowerBound]

/-- Complete selected-library execution is total below Go's actual word-sized array bound. -/
theorem imported_search_matches (ops : Ops α) (positive : 0 < ops.wordBits) (members : List α) (target : α)
    (representable : members.length ≤ maxIndex ops.wordBits) :
    let answer := lowerBound (beforeAt ops.compare members target) 0 members.length
    exec ops (members.length+19) SearchSource.binarySearchProgram {members,target,other := target} =
      .ok (.returned [.number answer,.flag (foundAt ops.compare members target answer)]) := by
  have loop := loop_matches ops members.length
    ({members,target,other := target,n := members.length,j := members.length} : Machine α)
    positive (by simp) (by simp) representable rfl (by simp)
  have split : SearchSource.binarySearchProgram = SearchSource.binarySearchProgram.take 2 ++ loopProgram := by
    unfold loopProgram
    rfl
  rw [split]
  simpa [SearchSource.binarySearchProgram,exec,many,one,eval,assign,SearchSubset.read,SearchSubset.write,
    Nat.add_assoc,Bind.bind,Except.bind] using loop

end SSA.SearchContracts
