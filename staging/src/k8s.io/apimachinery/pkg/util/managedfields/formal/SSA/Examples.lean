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

namespace SSA.Examples

set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

def f (name : String) : Segment := .field name
def k (name : String) : Segment := .item [("name", name)]
def items : Path := [f "items"]
def database : Path := items ++ [k "database"]
def host : Path := database ++ [f "host"]
def port : Path := database ++ [f "port"]
def note : Path := database ++ [f "note"]

def listCell : Cell := ⟨items, .collection⟩
def itemCell : Cell := ⟨database, .item⟩
def hostCell : Cell := ⟨host, .value "db.example.com"⟩
def portCell : Cell := ⟨port, .value "5432"⟩
def noteCell : Cell := ⟨note, .value "injected"⟩

def request (manager : String) (cells : List Cell) (force := false) : Request :=
  ⟨manager, cells.map Cell.footprint, cells, force⟩

def endpoint : ItemSpec := ⟨database, .both (.value host) (.value port)⟩
def schema : Schema := [endpoint]
def create := request "A" [listCell, itemCell, hostCell, portCell]
def inject := request "B" [listCell, itemCell, noteCell]
def leaveA := request "A" [listCell]
def leaveB := request "B" [listCell]
def coPort := request "B" [listCell, itemCell, portCell]
def releasePort := request "A" [listCell, itemCell, hostCell]

def fragment : Fixture :=
  { name := "reject-fragmented-required-ownership", schema
    steps := [
      ⟨create, { fullManagers := [("A", database)] }⟩,
      ⟨coPort, { fullManagers := [("A", database)], partialManagers := [("B", database)] }⟩,
      ⟨releasePort,
        { error := some .orphanedItem
          present := [hostCell, portCell]
          fullManagers := [("A", database)]
          partialManagers := [("B", database)] }⟩,
      ⟨leaveB, { present := [hostCell, portCell], fullManagers := [("A", database)] }⟩] }

def partialRelease : Fixture :=
  { name := "partial-release-preserves-full-owner", schema
    steps := [⟨create, {}⟩, ⟨inject, {}⟩,
      ⟨leaveB,
        { present := [hostCell, portCell]
          absent := [note]
          fullManagers := [("A", database)] }⟩] }

def fullDeletion : Fixture :=
  { name := "last-full-deletes-partial-fields", schema
    steps := [⟨create, {}⟩, ⟨inject, {}⟩,
      ⟨leaveA, { absent := [database, host, port, note] }⟩] }

def handoff : Fixture :=
  { name := "acquire-full-before-release", schema
    steps := [⟨create, {}⟩,
      ⟨request "B" [listCell, itemCell, hostCell, portCell],
        { fullManagers := [("A", database), ("B", database)] }⟩,
      ⟨leaveA, { present := [hostCell, portCell], fullManagers := [("B", database)] }⟩,
      ⟨leaveB, { absent := [database, host, port] }⟩] }

def keyOnly : Fixture :=
  { name := "key-only-delete-after-last-claim", schema := [⟨database, .none⟩]
    steps := [
      ⟨request "A" [listCell, itemCell, hostCell], { fullManagers := [("A", database)] }⟩,
      ⟨inject, { fullManagers := [("A", database), ("B", database)] }⟩,
      ⟨leaveA,
        { present := [itemCell, noteCell]
          absent := [host]
          fullManagers := [("B", database)] }⟩,
      ⟨leaveB, { absent := [database, note] }⟩] }

def connection : Path := database ++ [f "connection"]
def settings : Path := database ++ [f "settings"]
def nestedHost : Cell := ⟨connection ++ [f "host"], .value "db.example.com"⟩
def nestedPort : Cell := ⟨settings ++ [f "port"], .value "5432"⟩
def objectSchema : Schema :=
  [⟨database, .both (.object connection (.value nestedHost.path))
    (.object settings (.value nestedPort.path))⟩]

def objectFragment : Fixture :=
  { name := "required-object-branches-need-same-manager", schema := objectSchema
    steps := [
      ⟨request "A" [listCell, itemCell, ⟨connection, .object⟩, nestedHost,
        ⟨settings, .object⟩, nestedPort], { fullManagers := [("A", database)] }⟩,
      ⟨request "B" [listCell, itemCell, ⟨settings, .object⟩, nestedPort],
        { partialManagers := [("B", database)] }⟩,
      ⟨request "A" [listCell, itemCell, ⟨connection, .object⟩, nestedHost],
        { error := some .orphanedItem
          present := [nestedHost, nestedPort]
          fullManagers := [("A", database)] }⟩] }

def endpoints : Path := database ++ [f "endpoints"]
def api : Path := endpoints ++ [k "api"]
def url : Cell := ⟨api ++ [f "url"], .value "https://api.example.com"⟩
def mode : Cell := ⟨database ++ [f "mode"], .value "active"⟩
def childNote : Cell := ⟨api ++ [f "note"], .value "injected"⟩
def nestedSchema : Schema :=
  [⟨database, .value mode.path⟩, ⟨api, .value url.path⟩]
def outerA := request "A" [listCell, itemCell, mode]
def innerB := request "B" [listCell, itemCell, ⟨endpoints, .collection⟩, ⟨api, .item⟩, url]
def annotateA := request "A"
  [listCell, itemCell, mode, ⟨endpoints, .collection⟩, ⟨api, .item⟩, childNote]

def nestedRoles : Fixture :=
  { name := "roles-switch-between-outer-and-inner-items", schema := nestedSchema
    steps := [
      ⟨outerA, {}⟩,
      ⟨innerB,
        { fullManagers := [("A", database), ("B", api)]
          partialManagers := [("B", database)] }⟩,
      ⟨annotateA,
        { fullManagers := [("A", database), ("B", api)]
          partialManagers := [("B", database), ("A", api)] }⟩,
      ⟨outerA,
        { present := [mode, url]
          absent := [childNote.path]
          fullManagers := [("A", database), ("B", api)] }⟩] }

def parentDeletion : Fixture :=
  { name := "parent-deletion-dominates-full-child", schema := nestedSchema
    steps := [⟨outerA, {}⟩, ⟨innerB, {}⟩, ⟨annotateA, {}⟩,
      ⟨leaveA, { absent := [database, api, url.path, childNote.path] }⟩] }

def childDeletion : Fixture :=
  { name := "last-full-child-deletes-only-child", schema := nestedSchema
    steps := [⟨outerA, {}⟩, ⟨innerB, {}⟩, ⟨annotateA, {}⟩,
      ⟨leaveB,
        { present := [itemCell, mode]
          absent := [api, url.path, childNote.path]
          fullManagers := [("A", database)] }⟩] }

/-- Explicit collection-presence claims handle both populated and empty required lists. -/
def requiredList (populated : Bool) : Fixture :=
  let cells := [listCell, itemCell, ⟨endpoints, .collection⟩] ++
    if populated then [⟨api, .item⟩, url] else []
  { name := if populated then "required-list-presence" else "empty-required-list-presence"
    schema := [⟨database, .collection endpoints⟩, ⟨api, .value url.path⟩]
    steps := [
      ⟨request "A" cells, { fullManagers := [("A", database)] }⟩,
      ⟨inject, { partialManagers := [("B", database)] }⟩,
      ⟨leaveB,
        { present := [itemCell, ⟨endpoints, .collection⟩]
          absent := [note]
          fullManagers := [("A", database)] }⟩] }

def conflictsAndForce : Fixture :=
  let differentPort : Cell := ⟨port, .value "6432"⟩
  { name := "conflict-and-force-use-final-ownership", schema
    steps := [
      ⟨create, {}⟩,
      ⟨request "B" [listCell, itemCell, differentPort],
        { error := some .conflict, present := [portCell], fullManagers := [("A", database)] }⟩,
      ⟨request "B" [listCell, itemCell, differentPort] true,
        { error := some .orphanedItem, present := [portCell], fullManagers := [("A", database)] }⟩,
      ⟨request "B" [listCell, itemCell, hostCell, differentPort] true,
        { present := [differentPort]
          fullManagers := [("B", database)]
          partialManagers := [("A", database)] }⟩] }

def sibling : Path := items ++ [k "cache"]
def siblingValue : Cell := ⟨sibling ++ [f "url"], .value "https://cache.example.com"⟩

def independentItems : Fixture :=
  { name := "deleting-one-item-preserves-its-sibling"
    schema := [⟨database, .value host⟩, ⟨sibling, .value siblingValue.path⟩]
    steps := [
      ⟨request "A" [listCell, itemCell, hostCell], {}⟩,
      ⟨request "B" [listCell, ⟨sibling, .item⟩, siblingValue], {}⟩,
      ⟨leaveA,
        { present := [⟨sibling, .item⟩, siblingValue]
          absent := [database, host]
          fullManagers := [("B", sibling)] }⟩] }

def malformedRequest : Fixture :=
  { name := "reject-claims-without-a-matching-apply-value", schema
    steps := [
      ⟨create, {}⟩,
      ⟨{ manager := "B", desired := [⟨port, .value⟩], writes := [] },
        { error := some .invalidRequest, fullManagers := [("A", database)] }⟩,
      ⟨request "B" [portCell],
        { error := some .invalidRequest, fullManagers := [("A", database)] }⟩] }

def fixtures : List Fixture :=
  [fragment, partialRelease, fullDeletion, handoff, keyOnly, objectFragment,
    nestedRoles, parentDeletion, childDeletion, requiredList true, requiredList false,
    conflictsAndForce, independentItems, malformedRequest]

/-- These are kernel-checked finite execution proofs, not unchecked native evaluation. -/
theorem all_fixtures_pass : fixtures.all passes = true := by decide

def afterCreate : State := candidate schema {} create
def afterCoPort : State := candidate schema afterCreate coPort
def unsafeFragment : State := candidate schema afterCoPort releasePort

theorem fragment_values_are_valid : valuesValid schema unsafeFragment = true := by decide
theorem fragment_has_no_full_manager : complete schema unsafeFragment = false := by decide

/-- A reduced PR-style rule lacks the departing-full precondition and the invariant check. -/
def prStyleCandidate (before : State) (r : Request) : State :=
  let provisional := merged before r
  let roots := (schema.filter fun item =>
    before.owns r.manager item.path .item &&
    !r.desired.contains ⟨item.path, .item⟩ && !hasFull provisional item).map (·.path)
  pruneReleased before r (dropSubtrees provisional roots)

theorem unchecked_partial_omission_deletes_foreign_host :
    (prStyleCandidate unsafeFragment leaveB).hasValue host = false := by decide

theorem corrected_apply_rejects_fragment :
    apply schema afterCoPort releasePort = .error .orphanedItem := by rfl

theorem repeating_same_apply_is_idempotent :
    commit afterCreate (apply schema afterCreate create) = afterCreate := by rfl

end SSA.Examples
