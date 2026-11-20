/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.Fixtures
import SSA.ApplySource

namespace SSA.ResetCounterexamples

open GoSubset

set_option maxRecDepth 10000
set_option maxHeartbeats 4000000

def settings : Path := [.field "settings"]
def desired : Path := settings ++ [.field "desired"]
def status : Path := [.field "status"]
def observed : Path := status ++ [.field "observed"]
def settingsCell : Cell := ⟨settings,.object⟩
def statusCell : Cell := ⟨status,.object⟩
def one : Cell := ⟨desired,.value "\"one\""⟩
def two : Cell := ⟨desired,.value "\"two\""⟩
def ready : Cell := ⟨observed,.value "\"ready\""⟩
def named : FieldSet := [settings,desired,status,observed]

-- These are distinct logical identities for the same API manager in two subresources.
def mainActor := "Main"
def statusActor := "Status"
def statusResetRoots : FieldSet := [[.field "metadata"],[.field "spec"]]

def request (actor : String) (writes context : List Cell) : Request :=
  { manager := actor,writes,context,desired := writes.map Cell.footprint,named }

def create := request mainActor [one] [settingsCell]
def writeStatus := request statusActor [one,ready] [settingsCell,statusCell]
def changeSettings := request mainActor [two] [settingsCell]
def created := commit {} (applyWith .current [] {} create)
def proposed := commit created (applyWith .current [] created writeStatus)

/-- Status PrepareForUpdate restores all non-status data but keeps computed managedFields. -/
def restoreStatus (before proposed : State) : State :=
  { cells := before.cells.filter (fun cell => !under status cell.path) ++
      proposed.cells.filter (fun cell => under status cell.path)
    claims := proposed.claims }

def stored := restoreStatus created proposed

/-- Restoration protects every non-status data cell, independently of the proposed payload. -/
theorem restoration_preserves_nonstatus_data (before proposed : State) (cell : Cell)
    (outside : under status cell.path = false) :
    cell ∈ (restoreStatus before proposed).cells ↔ cell ∈ before.cells := by
  simp [restoreStatus,outside]

/-- The configured filter excludes spec and metadata, but settings is a valid separate root field. -/
theorem configured_filter_keeps_settings :
    ResetFields.filter statusResetRoots [desired,observed] = [desired,observed] := rfl

/-- A fresh status manager co-claims an unchanged non-status field before REST restoration. -/
theorem status_acquires_a_reset_field :
    applyWith .current [] created writeStatus = .ok proposed ∧
    stored.owns statusActor desired .value = true ∧
    one ∈ stored.cells ∧ ready ∈ stored.cells := by
  exact ⟨rfl,by decide⟩

/-- The inappropriate status claim blocks the next regular Apply; rejection is atomic. -/
theorem status_claim_blocks_regular_apply :
    applyWith .current [] stored changeSettings = .error .conflict ∧
    commit stored (applyWith .current [] stored changeSettings) = stored := by
  exact ⟨rfl,rfl⟩

/-- This broader reset contract includes every field restored by the status strategy. -/
def EveryStatusResetFieldIsUnclaimed : Prop :=
  ∀ (before proposal : State) (r : Request),
    shapeValid [] before = true →
    (∀ claim ∈ before.claims, claim.manager ≠ r.manager) →
    ResetFields.filter statusResetRoots (r.writes.map (·.path)) = r.desired.map (·.path) →
    applyWith .current [] before r = .ok proposal →
    ∀ claim ∈ (restoreStatus before proposal).claims,
      claim.manager = r.manager → under status claim.target.path = true

/-- Spec/status-only filtering is insufficient for the API's broader status reset behavior. -/
theorem broad_status_reset_contract_is_false : ¬EveryStatusResetFieldIsUnclaimed := by
  intro safe
  have retained := safe created proposed writeStatus rfl (by decide) rfl rfl
    ⟨statusActor,⟨desired,.value⟩,true⟩ (by decide) rfl
  cases retained

def sourceManagers : ManagedFields := [(mainActor,⟨[desired],"v1",true⟩)]
def sourceSelected : FieldSet := ResetFields.filter statusResetRoots [desired,observed]
def sourceAfter : ManagedFields :=
  managerSet sourceManagers statusActor ⟨sourceSelected,"v1",true⟩

/-- A concrete source instance isolates filter/installation control flow.
    Data merge and the final updater remain explicit primitives, checked independently through HTTP. -/
def sourceOps : Ops State Unit where
  convert object _ := .ok object
  missing _ := false
  toFields _ := .ok [desired,observed]
  ensureNamed _ _ fields := fields
  remove object _ := object
  wrap _ error := error
  mergeObjects _ _ := .ok proposed
  ignoreFilters := some [("v1",ResetFields.filter statusResetRoots)]

/-- The complete imported Apply body installs the unprotected settings claim in this source instance. -/
theorem imported_apply_installs_nonstatus_claim :
    exec sourceOps 128 GoSource.applyProgram
      (ApplySource.applyEnv created proposed "v1" sourceManagers statusActor false) =
      .ok (.returned [.object proposed,.managed sourceAfter,.nil]) := by
  rw [ApplySource.imported_apply_matches_spec]
  rfl

/-- The fresh status Actor's pruning callback is realized by the imported nil-last-set branch. -/
theorem imported_prune_callback_is_realized :
    exec sourceOps 64 GoSource.pruneProgram
      (PruneSource.pruneEnv proposed sourceAfter statusActor none) =
      .ok (.returned [.object proposed,.nil]) := by
  exact PruneSource.imported_prune_matches_spec sourceOps proposed sourceAfter statusActor none
    ⟨sourceSelected,"v1",true⟩ rfl

def fixtures : List Fixture := [
  { name := "raw-status-claim-blocks-nonstatus-root-change",schema := [],semantics := .current
    steps := [⟨create,{present := [one]}⟩,⟨writeStatus,{present := [one,ready]}⟩,
      ⟨changeSettings,{error := some .conflict,present := [one,ready]}⟩] }]

theorem fixtures_pass : fixtures.all passes = true := by decide

end SSA.ResetCounterexamples

