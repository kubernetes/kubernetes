/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.MapCopySubset

namespace SSA.MapCopyContracts

open MapCopySubset
variable {F E U Rel C : Type}

def copied (source : MapData F E U Rel C) (destination : MapData F E U Rel C) : MapData F E U Rel C :=
  {source with
    cache := source.cache.or destination.cache
    consumed := if source.cache.isSome then true else destination.consumed}

set_option maxRecDepth 30000
set_option maxHeartbeats 1000000

macro "map_copy_simp" : tactic =>
  `(tactic| simp_all (config := {maxSteps := 1000000}) [invoke,SchemaLookupSource.copyProgram,
      exec,execList,eval,one,many,write,primitive,select,isNil,target,MapCopySubset.read,
      RawGoSyntax.field,RawGoSyntax.find,AtomicRemoveSubset.atom,AtomicRemoveSubset.nodes,
      copied,Bind.bind,Except.bind])

/-- Complete source copy, including nil destination, private-cache transfer and the full Once callback. -/
theorem imported_copy (source : MapData F E U Rel C) (destination : Option (MapData F E U Rel C)) :
    invoke source destination = .ok (destination.map (copied source)) := by
  cases destination with
  | none => map_copy_simp
  | some destination => cases cache : source.cache <;> map_copy_simp

/-- A fresh cache/Once destination copies both exported contents and the available field index. -/
theorem imported_fresh (source destination : MapData F E U Rel C)
    (empty : destination.cache = none) (unused : destination.consumed = false) :
    invoke source (some destination) = .ok (some {source with consumed := source.cache.isSome}) := by
  rw [imported_copy]
  cases cache : source.cache <;> simp [copied,cache,empty,unused]

/-- Exactly the opaque immutable-contents copy assumed by Schema.Resolve, now derived from source. -/
theorem imported_value_view (source destination : MapData F E U Rel C) :
    (invoke source (some destination)).map (fun result => result.map view) = .ok (some (view source)) := by
  rw [imported_copy]
  rfl

/-- The source body supplies the resolver interpreter's exact fresh-copy value contract. -/
theorem imported_resolver_copy {S L R : Type} (source destination : MapData F E U Rel C)
    (ops : SchemaResolveMachine.Ops (F × E × U) S L Rel)
    (frame : SchemaResolveMachine.Machine (F × E × U) S L R Rel) (unescaped : frame.mapEscaped = false) :
    (invoke source (some destination)).map
        (fun result => result.map (fun value => SchemaResolveMachine.copyOf (view value))) =
      (SchemaResolveMachine.method ops (.map (view source)) "CopyInto" [.mapLocal] frame).map
        (fun result => some result.2.mapCopy) := by
  rw [imported_copy]
  simp [SchemaResolveMachine.method,unescaped,SchemaResolveMachine.copyOf,view,copied,Except.map]

def ValidIndex (index : F → C) (value : MapData F E U Rel C) : Prop :=
  if value.consumed then value.cache = some (index value.fields) else value.cache = none

/-- An initialized source index or uninitialized source stays coherent in the fresh destination. -/
theorem imported_fresh_valid_index (index : F → C) (source destination : MapData F E U Rel C)
    (valid : ValidIndex index source) (empty : destination.cache = none) (unused : destination.consumed = false)
    (result : MapData F E U Rel C) (returned : invoke source (some destination) = .ok (some result)) :
    ValidIndex index result := by
  rw [imported_fresh source destination empty unused] at returned
  cases returned
  cases consumed : source.consumed <;> simp_all [ValidIndex]

/-- The fresh-destination premise matters: copying an unindexed source does not clear an old index. -/
theorem imported_uninitialized_retains_old_cache (source destination : MapData F E U Rel C)
    (uninitialized : source.cache = none) :
    invoke source (some destination) = .ok (some {source with cache := destination.cache,consumed := destination.consumed}) := by
  rw [imported_copy]
  simp [copied,uninitialized]

def staleSource : MapData Nat Unit Unit Unit Nat := {fields := 1,elementType := (),unions := (),relationship := ()}
def staleDestination : MapData Nat Unit Unit Unit Nat := {
  fields := 2,elementType := (),unions := (),relationship := (),cache := some 2,consumed := true}

/-- An initialized reused destination can have new fields but its previous field index. -/
theorem initialized_destination_can_break_index :
    ∃ result, invoke staleSource (some staleDestination) = .ok (some result) ∧ ¬ ValidIndex id result := by
  refine ⟨{staleSource with cache := some 2,consumed := true},?_,?_⟩
  · rw [imported_uninitialized_retains_old_cache _ _ rfl]
    rfl
  · simp [ValidIndex,staleSource]

end SSA.MapCopyContracts
