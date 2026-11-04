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

namespace SSA.Counterexamples

set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

def items : Path := [.field "items"]
def item : Path := items ++ [.item [("name", "db")]]
def host : Path := item ++ [.field "host"]
def port : Path := item ++ [.field "port"]
def note : Path := item ++ [.field "note"]
def conn : Path := item ++ [.field "conn"]
def endpoint : Path := conn ++ [.field "endpoint"]
def children : Path := item ++ [.field "children"]
def child : Path := children ++ [.item [("name","child")]]
def listCell : Cell := ⟨items,.collection⟩
def itemCell : Cell := ⟨item,.item⟩
def hostCell : Cell := ⟨host,.value "old"⟩
def portCell : Cell := ⟨port,.value "p"⟩
def noteCell : Cell := ⟨note,.value "injected"⟩

def request (manager : String) (writes context : List Cell) : Request :=
  { manager,writes,context,desired := writes.map Cell.footprint
    named := [items,host,port,note,conn,endpoint,children] }

def schema : Schema := [⟨item,.both (.value host) (.value port)⟩]
def create := request "A" [itemCell,hostCell,portCell] [listCell]
def share := request "B" [itemCell,portCell] [listCell]
def fragment := request "A" [itemCell,hostCell] [listCell]
def leave (manager : String) := request manager [] []
def run (mode : Semantics) (schema : Schema) : State → List Request → State
  | before,[] => before
  | before,r::rest => run mode schema (commit before (applyWith mode schema before r)) rest

def beforeFragment := run .predrag schema {} [create,share]
def fragmented := run .predrag schema beforeFragment [fragment]

/-- The raw PR accepts the exact transition which the proposed invariant must reject. -/
theorem pr_accepts_fragment_without_full_owner :
    applyWith .predrag schema beforeFragment fragment = .ok fragmented ∧
    valuesValid schema fragmented = true ∧ complete schema fragmented = false := by
  constructor
  · rfl
  · decide

/-- B never owned the whole required core, yet its omission deletes A's host. -/
theorem pr_partial_can_delete_foreign_required_branch :
    full fragmented "B" ⟨item,.both (.value host) (.value port)⟩ = false ∧
    fragmented.owns "A" host .value = true ∧
    (run .predrag schema fragmented [leave "B"]).hasItem item = false := by decide

/-- The proposed guard rejects the same request atomically, preserving the prior full owner. -/
theorem full_contract_rejects_same_fragment :
    applyWith .full schema beforeFragment fragment = .error .orphanedItem ∧
    full beforeFragment "A" ⟨item,.both (.value host) (.value port)⟩ = true := by
  constructor
  · rfl
  · decide

def mapSchema : Schema := [⟨item,.object conn (.value endpoint)⟩]
def mapCreate := request "A" [itemCell,⟨endpoint,.value "live"⟩] [listCell,⟨conn,.object⟩]
def mapShare := request "B" [itemCell,⟨conn,.object⟩] [listCell]
def beforeMapRelease := run .predrag mapSchema {} [mapCreate,mapShare]
def afterMapRelease := run .predrag mapSchema beforeMapRelease [leave "A"]

/-- The map-member shortcut keeps an item while its required nested value disappears. -/
theorem pr_map_shortcut_retains_invalid_item :
    prCovers beforeMapRelease "B" (.object conn (.value endpoint)) = true ∧
    covers beforeMapRelease "B" (.object conn (.value endpoint)) = false ∧
    afterMapRelease.hasItem item = true ∧ valuesValid mapSchema afterMapRelease = false := by decide

def listSchema : Schema := [⟨item,.collection children⟩,⟨child,.none⟩]
def listCreate := request "A" [itemCell,⟨child,.item⟩] [listCell,⟨children,.collection⟩]
def listShare := request "B" [itemCell,noteCell] [listCell]
def beforeListRelease := run .predrag listSchema {} [listCreate,listShare]

/-- A supplies the required list and all its elements, but the PR still calls A partial. -/
theorem pr_required_list_owner_is_misclassified :
    requiredPresent beforeListRelease (.collection children) = true ∧
    beforeListRelease.owns "A" child .item = true ∧
    prCovers beforeListRelease "A" (.collection children) = false ∧
    (run .predrag listSchema beforeListRelease [leave "B"]).hasItem item = false := by decide

/-- FieldsV1 cannot remember whether A explicitly supplied an empty granular list. -/
theorem empty_granular_list_has_no_ownership_marker :
    (request "A" [itemCell] [listCell,⟨children,.collection⟩]).desired =
    (request "A" [itemCell] [listCell]).desired := rfl

def foo : Path := [.field "foo"]
def bar : Path := foo ++ [.field "bar"]
def abc : Path := foo ++ [.field "abc"]
def left (parent : Path) : Path := parent ++ [.field "left"]
def right (parent : Path) : Path := parent ++ [.field "right"]
def branch (parent : Path) : ObjectSpec := ⟨parent,.both (.value (left parent)) (.value (right parent))⟩
def objectScopes := [branch bar,branch abc]
def valueAt (path : Path) : Cell := ⟨path,.value "v"⟩
def objectRequest (manager : String) (values containers : List Path) :=
  request manager (values.map valueAt) (containers.map (fun p => ⟨p,.object⟩))
def createBar := objectRequest "A" [left bar,right bar] [foo,bar]
def createAbc := objectRequest "B" [right bar,left abc,right abc] [foo,bar,abc]
def shareAbc := objectRequest "A" [left bar,right bar,left abc] [foo,bar,abc]
def fragmentBar := objectRequest "A" [left bar,left abc] [foo,bar,abc]
def leaveBar := objectRequest "A" [left abc] [foo,abc]
def objectStep (s : State) (r : Request) := commit s (writeScoped objectScopes .full [] s r)
def roleState := objectStep (objectStep (objectStep {} createBar) createAbc) shareAbc

/-- Full and partial roles can be opposite for the same managers in sibling objects. -/
theorem sibling_objects_have_independent_roles :
    objectFull roleState "A" (branch bar) = true ∧ objectFull roleState "B" (branch bar) = false ∧
    objectFull roleState "B" (branch abc) = true ∧ objectFull roleState "A" (branch abc) = false := by decide

/-- One request cannot fragment bar just because abc still has a full owner. -/
theorem sibling_fragmentation_rejects_entire_request :
    writeScoped objectScopes .full [] roleState fragmentBar = .error .orphanedItem := rfl

/-- A full manager may remove its optional object while remaining partial in the sibling. -/
theorem sibling_lifetime_deletion_preserves_other_full_owner :
    (objectStep roleState leaveBar).hasValue (right bar) = false ∧
    (objectStep roleState leaveBar).hasValue (right abc) = true ∧
    objectFull (objectStep roleState leaveBar) "B" (branch abc) = true := by decide

def rootField : Path := [.field "required"]
def rootScopes : List ObjectSpec := [⟨[],.value rootField⟩]
def createRoot := request "A" [valueAt rootField] []
def createdRoot := commit {} (writeScoped rootScopes .full [] {} createRoot)

/-- The empty initial state denotes a not-yet-created resource, so root requirements do not block creation. -/
theorem required_root_creation_and_rejection_of_omission :
    objectsValid rootScopes createdRoot = true ∧
    (objectFull createdRoot "A" ⟨[],.value rootField⟩) = true := by decide

theorem required_root_cannot_be_omitted :
    writeScoped rootScopes .full [] createdRoot (leave "A") = .error .orphanedItem := rfl

def fixtures : List Fixture := [
  { name := "required-root-creation-and-protected-omission",schema := [],objects := rootScopes
    steps := [⟨createRoot,{present := [valueAt rootField]}⟩,
      ⟨leave "A",{error := some .orphanedItem,present := [valueAt rootField]}⟩] },
  { name := "scoped-sibling-roles-and-atomic-fragment-rejection",schema := [],objects := objectScopes
    steps := [⟨createBar,{}⟩,⟨createAbc,{}⟩,⟨shareAbc,{}⟩,
      ⟨fragmentBar,{error := some .orphanedItem,present := [valueAt (right bar),valueAt (right abc)]}⟩,
      ⟨leaveBar,{absent := [bar,left bar,right bar],present := [valueAt (right abc)]}⟩] },
  { name := "raw-pr-fragmentation-and-partial-deletion",schema,semantics := .predrag
    steps := [⟨create,{}⟩,⟨share,{}⟩,⟨fragment,{present := [hostCell,portCell],partialManagers := [("A",item),("B",item)]}⟩,
      ⟨leave "B",{absent := [item,host,port]}⟩] },
  { name := "raw-pr-map-member-shortcut",schema := mapSchema,semantics := .predrag
    steps := [⟨mapCreate,{}⟩,⟨mapShare,{}⟩,⟨leave "A",{present := [itemCell],absent := [endpoint]}⟩] },
  { name := "raw-pr-required-list-classifier",schema := listSchema,semantics := .predrag
    steps := [⟨listCreate,{}⟩,⟨listShare,{}⟩,⟨leave "B",{absent := [item,child]}⟩] }]

theorem counterexample_fixtures_pass : fixtures.all passes = true := by decide

end SSA.Counterexamples
