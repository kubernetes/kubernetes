/-
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-/
import SSA.Fixtures
import SSA.Boundary
import SSA.PruningRefinement

namespace SSA.ContainerCounterexamples

open GoSubset PruningRefinement

set_option maxRecDepth 10000
set_option maxHeartbeats 4000000

def settings : Path := [.field "settings"]
def left : Path := settings ++ [.field "left"]
def mapCell : Cell := ⟨settings,.object⟩
def nullCell : Cell := ⟨settings,.value "null"⟩
def leftCell : Cell := ⟨left,.value "\"old\""⟩
def named : FieldSet := [settings,left]

def createMap : Request :=
  { manager := "A",writes := [leftCell],context := [mapCell]
    desired := [leftCell.footprint],named }

def emptyMap (manager : String) : Request :=
  { manager,writes := [mapCell],desired := [mapCell.footprint],named }

def populated := commit {} (applyWith .current [] {} createMap)
def first := commit populated (applyWith .current [] populated (emptyMap "A"))
def second := commit first (applyWith .current [] first (emptyMap "A"))

/-- The raw failure is independent of the removed scalar's encoded value. -/
theorem empty_map_null_for_every_scalar_payload (encodedValue : String) :
    applyWith .current []
      { cells := [mapCell,⟨left,.value encodedValue⟩]
        claims := [⟨"A",⟨left,.value⟩,true⟩] }
      (emptyMap "A") = .ok first := by
  let before : State := {
    cells := [mapCell,⟨left,.value encodedValue⟩]
    claims := [⟨"A",⟨left,.value⟩,true⟩] }
  let after : State := { cells := [nullCell],claims := [⟨"A",mapCell.footprint,true⟩] }
  change applyWith .current [] before (emptyMap "A") = .ok after
  have initial : shapeValid [] before = true := rfl
  have intent : normalized (emptyMap "A") = true := rfl
  have pruned : wirePruned .current [] before (emptyMap "A") = after := by
    let merged : State := {
      cells := [mapCell,⟨left,.value encodedValue⟩]
      claims := [⟨"A",mapCell.footprint,true⟩] }
    let removed : State := { claims := merged.claims }
    have mergedEq : wireMerged before (emptyMap "A") = merged := rfl
    have fields : dataFields named merged = [left] := rfl
    have firstPass : removeItems merged (ensureNamed named [left]) = removed := rfl
    have ownedPass : addBackOwnedFields named merged removed [settings] = after := by
      unfold addBackOwnedFields
      rw [fields]
      change removeItems merged [left] = after
      rfl
    have danglingPass : addBackDanglingFields named merged after [left] = after := by
      unfold addBackDanglingFields
      rw [fields]
      change removeItems merged [left] = after
      rfl
    simp only [wirePruned,mergedEq]
    change pruneWithFields named merged [left] [settings] = after
    unfold pruneWithFields
    change addBackDanglingFields named merged
      (addBackOwnedFields named merged (removeItems merged (ensureNamed named [left])) [settings])
      [left] = after
    rw [firstPass,ownedPass,danglingPass]
  have noConflict : wireConflicts before before after (emptyMap "A").manager = false := rfl
  have result : candidateWith .current [] before (emptyMap "A") = after := by
    unfold candidateWith
    rw [pruned]
    rfl
  have finalShape : shapeValid [] after = true := rfl
  simp [applyWith,initial,intent,pruned,noConflict,result,finalShape]

/-- The explicit map claim survives, yet pruning changes the requested object into null. -/
theorem empty_map_first_apply_creates_null :
    applyWith .current [] populated (emptyMap "A") = .ok first ∧
    first.cells = [nullCell] ∧ first.claims = [⟨"A",mapCell.footprint,true⟩] := by
  exact ⟨rfl,by decide⟩

/-- Repetition repairs the data while keeping exactly the same presence ownership. -/
theorem empty_map_second_apply_repairs_data :
    applyWith .current [] first (emptyMap "A") = .ok second ∧
    second.cells = [mapCell] ∧ first.claims = second.claims := by
  exact ⟨rfl,by decide⟩

/-- FieldSet members do not retain the data type from the request that acquired them. -/
theorem null_member_survives_a_new_map_child (encodedValue : String) :
    applyWith .current []
      { cells := [nullCell],claims := [⟨"A",⟨settings,.value⟩,true⟩] }
      { manager := "B",writes := [⟨left,.value encodedValue⟩],context := [mapCell],
        desired := [⟨left,.value⟩],named } =
      .ok {
        cells := [⟨left,.value encodedValue⟩,mapCell]
        claims := [⟨"A",⟨settings,.value⟩,true⟩,⟨"B",⟨left,.value⟩,true⟩] } := by
  rfl

/-- This loss of intent is independent of whether a direct presence claim exists. -/
theorem applied_empty_map_intent_is_not_preserved :
    mapCell ∈ (emptyMap "A").writes ∧ mapCell ∉ first.cells := by decide

/-- B's identical empty-map intent conflicts only while A's accepted result is null. -/
theorem empty_map_can_conflict_with_identical_previous_intent :
    applyWith .current [] first (emptyMap "B") = .error .conflict ∧
    ∃ shared, applyWith .current [] second (emptyMap "B") = .ok shared ∧
      shared.cells = second.cells ∧
      shared.owns "A" settings .presence = true ∧ shared.owns "B" settings .presence = true := by
  let shared := commit second (applyWith .current [] second (emptyMap "B"))
  exact ⟨rfl,shared,rfl,by decide⟩

/-- Optional non-nullable object validation permits absence or an object, but never null. -/
def objectOrAbsent (state : State) : Bool :=
  !state.cells.any (fun cell => cell.path == settings) || state.cells.contains mapCell

def nonnullable : Boundary.Policy := ⟨id,fun _ state => objectOrAbsent state⟩

def retry : Nat → State
  | 0 => populated
  | attempts + 1 =>
      let before := retry attempts
      commit before (Boundary.finish [] nonnullable before
        (applyWith .current [] before (emptyMap "A")))

/-- Even optional non-nullable maps cannot be emptied by any finite number of identical retries. -/
theorem nonnullable_map_retries_never_empty (attempts : Nat) : retry attempts = populated := by
  induction attempts with
  | zero => rfl
  | succ attempts ih =>
      simp only [retry,ih]
      rfl

/-- Both the old value and the desired empty object satisfy that optional-map policy. -/
theorem valid_empty_map_intent_is_rejected :
    objectOrAbsent populated = true ∧ objectOrAbsent ⟨[mapCell],[]⟩ = true ∧
    objectOrAbsent {} = true ∧
    Boundary.finish [] nonnullable populated
      (applyWith .current [] populated (emptyMap "A")) = .error .invalidResult := by
  exact ⟨rfl,rfl,rfl,rfl⟩

def defaultEmptyObject (state : State) : State :=
  { state with cells := state.cells.map (fun cell => if cell == nullCell then mapCell else cell) }

/-- A matching non-nullable object default repairs this trace before validation. -/
theorem empty_object_default_repairs_this_proposal :
    Boundary.finish [] ⟨defaultEmptyObject,fun _ state => objectOrAbsent state⟩ populated
      (applyWith .current [] populated (emptyMap "A")) = .ok second := rfl

def actorFields : ManagerFields := ⟨[settings],"v1",true⟩
def managers : ManagedFields := [("A",actorFields)]

/-- All four imported pruning functions produce the null result under normalized
    primitives, with realized callees and every lawful map-iteration schedule. -/
theorem imported_pruning_creates_null (order : VersionIteration) :
    CalleesRealized (linkedOps named order) "v1" managers "A" ∧
    exec (linkedOps named order) 64 GoSource.pruneProgram
      (PruneSource.pruneEnv (wireMerged populated (emptyMap "A")) managers "A"
        (some ([left],"v1"))) = .ok (.returned [.object first,.nil]) := by
  exact single_version_pruning_refinement named order (wireMerged populated (emptyMap "A"))
    managers "A" [left] "v1" actorFields rfl
    (by intro row member; simp [managers] at member; subst row; rfl)

def fixtures : List Fixture := [
  { name := "empty-map-null-conflict-and-repair",schema := [],semantics := .current
    steps := [⟨createMap,{present := [mapCell,leftCell]}⟩,
      ⟨emptyMap "A",{present := [nullCell],absent := [left]}⟩,
      ⟨emptyMap "B",{error := some .conflict,present := [nullCell]}⟩,
      ⟨emptyMap "A",{present := [mapCell]}⟩,
      ⟨emptyMap "B",{present := [mapCell]}⟩] }]

theorem fixtures_pass : fixtures.all passes = true := by decide

end SSA.ContainerCounterexamples
