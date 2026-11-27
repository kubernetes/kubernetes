/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.Fixtures
import SSA.Boundary
import SSA.PruningRefinement

namespace SSA.HandoffCounterexamples

open GoSubset PruningRefinement

set_option maxRecDepth 10000
set_option maxHeartbeats 4000000

def items : Path := [.field "items"]
def item : Path := items ++ [.item [("name","\"db\"")]]
def host : Path := item ++ [.field "host"]
def listCell : Cell := ⟨items,.collection⟩
def itemCell : Cell := ⟨item,.item⟩
def oldHost : Cell := ⟨host,.value "\"old\""⟩
def newHost : Cell := ⟨host,.value "\"new\""⟩
def named : FieldSet := [items,host]
def schema : Schema := [⟨item,.none⟩]

def request (manager : String) (writes context : List Cell) : Request :=
  { manager,writes,context,desired := writes.map Cell.footprint,named }

def create := request "A" [itemCell,oldHost] [listCell]
def replace := { request "B@update" [itemCell,newHost] [listCell] with operation := .update }
def leave := request "A" [] []
def created := commit {} (applyWith .current schema {} create)
def updated := commit created (update .current schema created replace)
def removed := commit updated (applyWith .current schema updated leave)

def share := request "A" [itemCell,newHost] [listCell]
def shared := commit updated (applyWith .current schema updated share)
def sharedRemoved := commit shared (applyWith .current schema shared leave)

/-- Reasserting B's unchanged value creates genuine scalar co-ownership before omission. -/
theorem reassertion_shares_update_owned_child :
    applyWith .current schema updated share = .ok shared ∧
    shared.owns "A" host .value = true ∧ shared.owns "B@update" host .value = true ∧
    shared.owns "A" item .item = true ∧ shared.owns "B@update" item .item = false := by
  exact ⟨rfl,by decide⟩

theorem omission_removes_shared_update_child :
    applyWith .current schema shared leave = .ok sharedRemoved ∧
    sharedRemoved.cells = [⟨items,.value "null"⟩] ∧ sharedRemoved.claims = [] := by
  exact ⟨rfl,by decide⟩

/-- This unrestricted BASE-03 reading omits the enclosing item's lifetime condition. -/
def UnrestrictedSharedOmissionSafety : Prop :=
  ∀ (sc : Schema) (before after : State) (r : Request) (path : Path) (other : String),
    shapeValid sc before = true →
    before.owns r.manager path .value = true → before.owns other path .value = true →
    other ≠ r.manager → r.force = false →
    r.cells.any (fun cell => cell.path == path) = false →
    applyWith .current sc before r = .ok after → after.hasValue path = true

theorem unrestricted_shared_omission_safety_is_false : ¬UnrestrictedSharedOmissionSafety := by
  intro retains
  have kept := retains schema shared sharedRemoved leave host "B@update"
    rfl rfl rfl (by decide) rfl rfl rfl
  cases kept

/-- Update transfers the changed child but acquires no list-item membership. -/
theorem update_acquires_child_without_item :
    update .current schema created replace = .ok updated ∧
    updated.owns "B@update" host .value = true ∧
    updated.owns "B@update" item .item = false ∧
    updated.owns "A" item .item = true := by
  exact ⟨rfl,by decide⟩

/-- An accepted omission can remove a foreign Update-owned value inside a released item. -/
theorem omission_removes_foreign_update_child :
    applyWith .current schema updated leave = .ok removed ∧
    removed.cells = [⟨items,.value "null"⟩] ∧ removed.claims = [] ∧
    removed.hasItem item = false ∧ removed.hasValue host = false := by
  exact ⟨rfl,by decide⟩

/-- This intentionally broad reading of BASE-07 has no item-lifetime premise. -/
def UnrestrictedForeignOmissionSafety : Prop :=
  ∀ (sc : Schema) (before after : State) (r : Request) (claim : Claim),
    shapeValid sc before = true →
    applyWith .current sc before r = .ok after →
    r.force = false →
    claim ∈ before.claims →
    claim.manager ≠ r.manager →
    claim.target.kind = .value →
    before.hasValue claim.target.path = true →
    r.cells.any (fun cell => cell.path == claim.target.path) = false →
    after.cells.any (fun cell => cell.path == claim.target.path) = true

/-- The counterexample is reachable from empty through an actual Apply/Update history. -/
theorem unrestricted_foreign_omission_safety_is_false :
    ¬UnrestrictedForeignOmissionSafety := by
  intro safe
  have retained := safe schema updated removed leave ⟨"B@update",⟨host,.value⟩,false⟩
    rfl rfl rfl (by decide) (by decide) rfl rfl rfl
  cases retained

/-- Non-nullable list validation accepts absence, but rejects the produced null. -/
def listOrAbsent (state : State) : Bool :=
  !state.cells.any (fun cell => cell.path == items) || state.cells.contains listCell

def policy : Boundary.Policy := ⟨id,fun _ state => listOrAbsent state⟩
def retry : Nat → State
  | 0 => updated
  | attempts + 1 =>
      let before := retry attempts
      commit before (Boundary.finish schema policy before
        (applyWith .current schema before leave))

/-- Validation leaves B's old value in storage, but every identical omission fails again. -/
theorem nonnullable_list_omission_retries_fail (attempts : Nat) :
    retry attempts = updated := by
  induction attempts with
  | zero => rfl
  | succ attempts ih =>
      simp only [retry,ih]
      rfl

def actorFields : ManagerFields := ⟨[],"v1",true⟩
def managers : ManagedFields :=
  [("A",actorFields),("B@update",⟨[host],"v1",false⟩)]
def lastFields : FieldSet :=
  claimFields (updated.claims.filter (fun claim => claim.manager == "A"))

def lastSharedFields : FieldSet :=
  claimFields (shared.claims.filter (fun claim => claim.manager == "A"))

theorem imported_pruning_removes_shared_child (order : VersionIteration) :
    CalleesRealized (linkedOps named order) "v1" managers "A" ∧
    exec (linkedOps named order) 64 GoSource.pruneProgram
      (PruneSource.pruneEnv (wireMerged shared leave) managers "A"
        (some (lastSharedFields,"v1"))) =
      .ok (.returned [.object (wirePruned .current schema shared leave),.nil]) := by
  exact single_version_pruning_refinement named order (wireMerged shared leave)
    managers "A" lastSharedFields "v1" actorFields rfl
    (by intro row member; simp [managers] at member; rcases member with rfl | rfl <;> rfl)

/-- Imported pruning drops the child under every lawful manager-map schedule. -/
theorem imported_pruning_removes_update_child (order : VersionIteration) :
    CalleesRealized (linkedOps named order) "v1" managers "A" ∧
    exec (linkedOps named order) 64 GoSource.pruneProgram
      (PruneSource.pruneEnv (wireMerged updated leave) managers "A"
        (some (lastFields,"v1"))) =
      .ok (.returned [.object (wirePruned .current schema updated leave),.nil]) := by
  exact single_version_pruning_refinement named order (wireMerged updated leave)
    managers "A" lastFields "v1" actorFields rfl
    (by intro row member; simp [managers] at member; rcases member with rfl | rfl <;> rfl)

def fixtures : List Fixture := [
  { name := "raw-update-child-does-not-retain-item",schema,semantics := .current
    steps := [⟨create,{present := [listCell,itemCell,oldHost]}⟩,
      ⟨replace,{present := [listCell,itemCell,newHost]}⟩,
      ⟨leave,{present := [⟨items,.value "null"⟩],absent := [item,host]}⟩] },
  { name := "raw-shared-update-child-does-not-retain-item",schema,semantics := .current
    steps := [⟨create,{present := [listCell,itemCell,oldHost]}⟩,
      ⟨replace,{present := [listCell,itemCell,newHost]}⟩,
      ⟨share,{present := [listCell,itemCell,newHost]}⟩,
      ⟨leave,{present := [⟨items,.value "null"⟩],absent := [item,host]}⟩] }]

theorem fixtures_pass : fixtures.all passes = true := by decide

end SSA.HandoffCounterexamples

