/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.NumericCodecCounterexamples

namespace SSA.GranularKeyCounterexamples

open GoSubset
open NumericCodecCounterexamples (original neighbor itemPath)
open NumericCounterexamples (roundInteger)

/-- The fixed structural schema has exact integer keys and one string payload per item.
    Empty/null containers, omitted payloads and general typed value semantics remain separate. -/
abbrev Object := List (Nat × String)

def originalEntry : Nat × String := (original,"old")
def neighborEntry : Nat × String := (neighbor,"independent")
def payloadPath (number : Nat) : Path := itemPath number ++ [.field "payload"]
def itemFields (number : Nat) : FieldSet :=
  [itemPath number,itemPath number ++ [.field "id"],payloadPath number]

def fields (object : Object) : FieldSet := object.flatMap (fun entry => itemFields entry.1)

def mergeKeys (live config : Object) : Object :=
  live.map (fun prior => (config.find? (fun entry => entry.1 == prior.1)).getD prior) ++
  config.filter (fun entry => !live.any (fun prior => prior.1 == entry.1))

/-- Stored numeric selectors are floats; foreign comparisons round the exact integer change keys. -/
def changesForDecodedOwners (live object : Object) : FieldSet :=
  fields ((object.filter (fun entry => !live.any (fun prior => prior.1 == entry.1))).map
    (fun entry => (roundInteger entry.1,entry.2))) ++
  object.filterMap (fun entry =>
    match live.find? (fun prior => prior.1 == entry.1) with
    | some prior => if prior.2 == entry.2 then none else some (payloadPath (roundInteger entry.1))
    | none => none)

/-- Named primitive instance for the fixed integer-key/string-payload history and decoded ownership.
    The complete imported Apply body is replayed; universal typed/update refinement remains separate. -/
def sourceOps : Ops Object Unit where
  convert object _ := .ok object
  missing _ := false
  toFields object := .ok (fields object)
  ensureNamed _ _ selected := selected
  remove object _ := object
  wrap _ error := error
  mergeObjects live config := .ok (mergeKeys live config)
  finishUpdate live object _ owners actor force :=
    let changed := changesForDecodedOwners live object
    let conflict := fun owner : String × ManagerFields =>
      owner.1 != actor && owner.2.fields.any changed.contains
    if !force && owners.any conflict then .error ()
    else .ok (owners.filterMap fun owner =>
      let retained := if owner.1 != actor then owner.2.fields.filter (fun path => !changed.contains path) else owner.2.fields
      if retained.isEmpty then none else some (owner.1,{owner.2 with fields := retained}))
  equalObjects live object := live == object

def ownersFor (key : Nat) : ManagedFields := [("A",⟨fields [(key,"old")],"v1",true⟩)]

/-- The serialized numeric selector is an exact integer token in this fixed fieldset shape.
    The structural JSON grammar and arbitrary selector parsing remain separate obligations. -/
structure WireOwnership where
  keyLiteral : Nat

def inputOwnership : WireOwnership := ⟨original⟩
def rawMeaning (wire : WireOwnership) : ManagedFields := ownersFor wire.keyLiteral
def decodeOwnership (wire : WireOwnership) : ManagedFields := ownersFor (roundInteger wire.keyLiteral)
def rawOwnersA : ManagedFields := rawMeaning inputOwnership
def decodedOwnersA : ManagedFields := decodeOwnership inputOwnership

def RawOwned (wire : WireOwnership) (path : Path) : Prop :=
  ∃ entry, ("A",entry) ∈ rawMeaning wire ∧ path ∈ entry.fields

def applyWire (config : Object) (force : Bool) :=
  exec sourceOps 128 GoSource.applyProgram
    (ApplySource.applyEnv [originalEntry] config "v1" (decodeOwnership inputOwnership) "B" force)

theorem initial_payload_is_raw_owned : RawOwned inputOwnership (payloadPath original) := by
  exact ⟨⟨fields [(original,"old")],"v1",true⟩,by simp [rawMeaning,inputOwnership,ownersFor],by simp [fields,itemFields]⟩

theorem initial_raw_owner_is_reachable :
    exec sourceOps 128 GoSource.applyProgram
      (ApplySource.applyEnv [] [originalEntry] "v1" [] "A" false) =
      .ok (.returned [.object [originalEntry],.managed rawOwnersA,.nil]) := by
  rw [ApplySource.imported_apply_matches_spec]
  rfl

theorem original_does_not_own_neighbor : itemPath neighbor ∉ fields [originalEntry] := by decide

theorem independent_key_conflicts_after_codec :
    exec sourceOps 128 GoSource.applyProgram
      (ApplySource.applyEnv [originalEntry] [neighborEntry] "v1" decodedOwnersA "B" false) =
      .ok (.returned [.nil,.managed [],.error ()]) := by
  rw [ApplySource.imported_apply_matches_spec]
  rfl

theorem small_independent_key_is_accepted :
    exec sourceOps 128 GoSource.applyProgram
      (ApplySource.applyEnv [(42,"old")] [(41,"independent")] "v1" (ownersFor (roundInteger 42)) "B" false) =
      .ok (.returned [.object [(42,"old"),(41,"independent")],
        .managed (managerSet (ownersFor 42) "B" ⟨fields [(41,"independent")],"v1",true⟩),.nil]) := by
  rw [ApplySource.imported_apply_matches_spec]
  rfl

/-- The fresh B request's prune callback is realized by the imported nil-last-set branch. -/
theorem fresh_prune_callback_is_realized :
    exec sourceOps 64 GoSource.pruneProgram
      (PruneSource.pruneEnv (mergeKeys [originalEntry] [neighborEntry])
        (managerSet decodedOwnersA "B" ⟨fields [neighborEntry],"v1",true⟩) "B" none) =
      .ok (.returned [.object (mergeKeys [originalEntry] [neighborEntry]),.nil]) := by
  apply PruneSource.imported_prune_matches_spec sourceOps (mergeKeys [originalEntry] [neighborEntry])
    (managerSet decodedOwnersA "B" ⟨fields [neighborEntry],"v1",true⟩) "B" none
    ⟨fields [neighborEntry],"v1",true⟩
  rfl

def EveryDistinctKeyCanBeAdded : Prop :=
  ∀ key, key < 2^63 → key ≠ original →
    itemPath key ∉ fields [originalEntry] →
    ∃ object owners,
      applyWire [(key,"independent")] false =
        .ok (.returned [.object object,.managed owners,.nil])

theorem unconditional_independent_item_progress_is_false : ¬EveryDistinctKeyCanBeAdded := by
  intro progresses
  obtain ⟨object,owners,accepted⟩ := progresses neighbor (by decide) (by decide) original_does_not_own_neighbor
  change exec sourceOps 128 GoSource.applyProgram
    (ApplySource.applyEnv [originalEntry] [(neighbor,"independent")] "v1" decodedOwnersA "B" false) = _ at accepted
  have rejected := independent_key_conflicts_after_codec
  simp only [neighborEntry] at rejected
  rw [rejected] at accepted
  cases accepted

/-- Force bypasses the false conflict and removes A's decoded claims, without changing A's payload. -/
theorem forced_addition_drops_unchanged_foreign_owner :
    exec sourceOps 128 GoSource.applyProgram
      (ApplySource.applyEnv [originalEntry] [neighborEntry] "v1" decodedOwnersA "B" true) =
      .ok (.returned [.object [originalEntry,neighborEntry],
        .managed (managerSet [] "B" ⟨fields [neighborEntry],"v1",true⟩),.nil]) := by
  rw [ApplySource.imported_apply_matches_spec]
  rfl

def originalPayload (object : Object) : Option String :=
  (object.find? (fun entry => entry.1 == original)).map Prod.snd

def EveryUnchangedPayloadKeepsWireOwner : Prop :=
  ∀ object owners,
    RawOwned inputOwnership (payloadPath original) →
    applyWire [neighborEntry] true =
      .ok (.returned [.object object,.managed owners,.nil]) →
    originalPayload object = some "old" →
    ∃ entry, ("A",entry) ∈ owners ∧ payloadPath original ∈ entry.fields

theorem unchanged_foreign_payload_ownership_is_false : ¬EveryUnchangedPayloadKeepsWireOwner := by
  intro preserves
  have oldPayload : originalPayload [originalEntry,neighborEntry] = some "old" := by decide
  obtain ⟨entry,present,_⟩ := preserves [originalEntry,neighborEntry]
    (managerSet [] "B" ⟨fields [neighborEntry],"v1",true⟩) initial_payload_is_raw_owned
    forced_addition_drops_unchanged_foreign_owner oldPayload
  simp [managerSet] at present

structure Observation where
  initialDataAccepted : Bool
  rawOwnerDoesNotClaimNeighbor : Bool
  falseConflict : Bool
  smallIndependentAccepted : Bool
  forceKeepsPayloadButDropsOwner : Bool
  deriving Lean.ToJson

def ownsItem (owners : ManagedFields) (actor : String) (key : Nat) : Bool :=
  owners.any fun owner => owner.1 == actor && owner.2.fields.contains (itemPath key)

def observe : Observation := {
  initialDataAccepted := match exec sourceOps 128 GoSource.applyProgram
      (ApplySource.applyEnv [] [originalEntry] "v1" [] "A" false) with
    | .ok (.returned [.object object,.managed _,.nil]) => object == [originalEntry]
    | _ => false
  rawOwnerDoesNotClaimNeighbor := !(fields [originalEntry]).contains (itemPath neighbor)
  falseConflict := match exec sourceOps 128 GoSource.applyProgram
      (ApplySource.applyEnv [originalEntry] [neighborEntry] "v1" decodedOwnersA "B" false) with
    | .ok (.returned [.nil,.managed [],.error _]) => true
    | _ => false
  smallIndependentAccepted := match exec sourceOps 128 GoSource.applyProgram
      (ApplySource.applyEnv [(42,"old")] [(41,"independent")] "v1" (ownersFor (roundInteger 42)) "B" false) with
    | .ok (.returned [.object object,.managed owners,.nil]) =>
        object == [(42,"old"),(41,"independent")] && ownsItem owners "A" 42 && ownsItem owners "B" 41
    | _ => false
  forceKeepsPayloadButDropsOwner := match exec sourceOps 128 GoSource.applyProgram
      (ApplySource.applyEnv [originalEntry] [neighborEntry] "v1" decodedOwnersA "B" true) with
    | .ok (.returned [.object object,.managed owners,.nil]) =>
        originalPayload object == some "old" && !(owners.any fun owner => owner.1 == "A")
    | _ => false
}

end SSA.GranularKeyCounterexamples
