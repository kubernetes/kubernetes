/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SetContracts
import SSA.UpdaterBridge

namespace SSA.SetBridge

open RecursiveSets

def onePath : List α → Tree α
  | [] => .node [] []
  | head::rest => if rest.isEmpty then .node [head] [] else .node [] [(head,onePath rest)]

theorem paths_one_path (path : List α) : paths (onePath path) = if path.isEmpty then [] else [path] := by
  induction path with
  | nil => simp [onePath,paths]
  | cons head rest induction =>
      cases rest with
      | nil => simp [onePath,paths]
      | cons next rest =>
          change paths (.node [] [(head,onePath (next::rest))]) = [head::next::rest]
          rw [paths]
          simp only [List.map_nil,List.nil_append,List.flatMap_cons,List.flatMap_nil,List.append_nil]
          rw [induction]
          rfl

def combine (left right : Tree α) : Tree α :=
  .node (left.members ++ right.members) (left.children ++ right.children)

theorem combine_membership (left right : Tree α) (path : List α) :
    path ∈ paths (combine left right) ↔ path ∈ paths left ∨ path ∈ paths right := by
  cases left
  cases right
  simp [combine,paths,Tree.members,Tree.children,List.map_append,List.flatMap_append,or_assoc,or_left_comm]

/-- An explicit finite representation; the root-empty path has no Go Set membership. -/
def encode : List (List α) → Tree α
  | [] => .node [] []
  | path::rest => combine (onePath path) (encode rest)

theorem encode_membership (fields : List (List α)) (path : List α) :
    path ∈ paths (encode fields) ↔ path ∈ fields ∧ path ≠ [] := by
  induction fields with
  | nil => simp [encode,paths]
  | cons head rest induction =>
      rw [encode,combine_membership,paths_one_path,induction]
      cases head <;> simp_all <;> grind

def fieldDifference (lt : Segment → Segment → Bool) (isEmpty : Tree Segment → Bool) (left right : FieldSet) : FieldSet :=
  paths (difference lt isEmpty (encode left) (encode right))

theorem field_difference_shrinks (lt : Segment → Segment → Bool) (isEmpty : Tree Segment → Bool) (left right : FieldSet)
    (path : Path) (present : path ∈ fieldDifference lt isEmpty left right) : path ∈ left :=
  ((encode_membership left path).mp (difference_never_adds_paths lt isEmpty (encode left) (encode right) path present)).1

/-- Decoding/normalization remains outside this adapter; subtraction itself has a source certificate. -/
def withRecursiveDifference {Obj Err : Type} (ops : UpdaterSubset.Ops Obj Err)
    (lt : Segment → Segment → Bool) (isEmpty : Tree Segment → Bool) : UpdaterSubset.Ops Obj Err :=
  {ops with primitives := {ops.primitives with difference := fieldDifference lt isEmpty}}

theorem recursive_difference_discharges_updater_premise {Obj Err : Type} (ops : UpdaterSubset.Ops Obj Err)
    (lt : Segment → Segment → Bool) (isEmpty : Tree Segment → Bool) : UpdaterContracts.DifferenceShrinks (withRecursiveDifference ops lt isEmpty) :=
  field_difference_shrinks lt isEmpty

theorem source_updater_with_recursive_difference_does_not_grow_actor {Obj Err : Type}
    (ops : UpdaterSubset.Ops Obj Err) (errors : UpdaterBridge.FailureMap Err)
    (lt : Segment → Segment → Bool) (isEmpty : Tree Segment → Bool) :
    ApplySource.UpdateDoesNotGrowActor (UpdaterBridge.applyOps (withRecursiveDifference ops lt isEmpty) errors) :=
  UpdaterBridge.source_update_does_not_grow_actor _ errors (recursive_difference_discharges_updater_premise ops lt isEmpty)

end SSA.SetBridge
