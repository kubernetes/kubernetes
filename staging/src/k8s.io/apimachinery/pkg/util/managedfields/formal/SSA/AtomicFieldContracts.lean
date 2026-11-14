/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.AtomicFieldSubset

namespace SSA.AtomicFieldContracts

open AtomicFieldSubset

def program : Kind → List GoSubset.Stmt
  | .list => AtomicFieldSource.listProgram
  | .map => AtomicFieldSource.mapProgram

/-- Neither the value's size/depth nor its children's topology affects atomic field extraction. -/
theorem imported_atomic_fields (ops : Ops Obj) (kind : Kind) (input : Obj)
    (fields : List (List α)) (path : List α) :
    run ops (program kind) {input,fields,path} =
      .ok (.returned [.nil] {
        input := input, path := path, fields := (AtomicFieldSubset.insert fields path),
        resourceLocal := (ops.view kind input), freed := (ops.view kind input).toList}) := by
  cases kind <;> cases resourceValue : ops.view _ input <;>
    simp [program,AtomicFieldSource.listProgram,AtomicFieldSource.mapProgram,
      run,exec,many,one,eval,AtomicFieldSubset.read,assign,effect,captureFree,resource,isNil,finish,
      resourceValue,Bind.bind,Except.bind]

theorem insert_membership (fields : List (List α)) (path query : List α) :
    query ∈ AtomicFieldSubset.insert fields path ↔ query ∈ fields ∨ (query = path ∧ path ≠ []) := by
  cases path <;> simp [AtomicFieldSubset.insert,or_comm,and_comm]

theorem imported_atomic_claims_scope (ops : Ops Obj) (kind : Kind) (input : Obj)
    (fields : List (List α)) (path : List α) (nonempty : path ≠ [])
    (final : Machine Obj α)
    (returned : run ops (program kind) {input,fields,path} = .ok (.returned [.nil] final)) :
    path ∈ final.fields := by
  rw [imported_atomic_fields] at returned
  cases returned
  exact (insert_membership fields path path).mpr (Or.inr ⟨rfl,nonempty⟩)

/-- No newly acquired claim can designate a strict descendant of the atomic scope. -/
theorem imported_atomic_never_adds_descendant (ops : Ops Obj) (kind : Kind) (input : Obj)
    (fields : List (List α)) (path query : List α) (different : query ≠ path)
    (final : Machine Obj α)
    (returned : run ops (program kind) {input,fields,path} = .ok (.returned [.nil] final)) :
    query ∈ final.fields ↔ query ∈ fields := by
  rw [imported_atomic_fields] at returned
  cases returned
  simp [insert_membership,different]

end SSA.AtomicFieldContracts
