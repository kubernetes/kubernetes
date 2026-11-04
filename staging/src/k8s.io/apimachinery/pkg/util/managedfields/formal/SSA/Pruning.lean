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
import SSA.Apply

namespace SSA

/-- A finite instance of named schema fields; map entries are deliberately absent. -/
abbrev FieldSet := List Path

def ensureNamed (named : FieldSet) (fields : FieldSet) : FieldSet :=
  fields ++ named.filter (fun parent => fields.any (fun child => properUnder parent child))

def keyLeaves (path : Path) : FieldSet :=
  match path.getLast? with
  | some (.item keys) => keys.map (fun key => path ++ [.field key.1])
  | _ => []

def claimFields (claims : List Claim) : FieldSet :=
  claims.flatMap (fun claim => claim.target.path ::
    (if claim.target.kind == .item then keyLeaves claim.target.path else []))

/-- List keys carry key leaves implicitly; known granular lists have no direct field member. -/
def dataFields (named : FieldSet) (s : State) : FieldSet :=
  (s.cells.filter fun cell => match cell.content with
    | .value _ | .item => true
    | .object => !s.cells.any (fun child => properUnder cell.path child.path) || !named.contains cell.path
    | .collection => !named.contains cell.path).flatMap (fun cell =>
      cell.path :: (if cell.content == .item then keyLeaves cell.path else []))

/-- RemoveItems preserves an unselected container slot as null when its last child is pruned. -/
def removeItems (s : State) (fields : FieldSet) : State :=
  let kept := s.cells.filter (fun cell => !fields.any (fun field => under field cell.path))
  { s with cells := kept.map (fun cell =>
      if (cell.content == .object || cell.content == .collection) &&
          fields.any (fun field => properUnder cell.path field) &&
          !kept.any (fun child => properUnder cell.path child.path)
      then { cell with content := .value "null" }
      else cell) }

def fieldDifference (left right : FieldSet) : FieldSet := left.filter (fun p => !right.contains p)
def fieldIntersection (left right : FieldSet) : FieldSet := left.filter right.contains

/-- This is the three-pass SMD pruning algorithm, including dangling-field restoration. -/
def pruneFieldSets (named : FieldSet) (before provisional : State) (r : Request)
    (restoring : FieldSet) : State :=
  let last := ensureNamed named (claimFields (before.claims.filter (fun c => c.manager == r.manager)))
  if last.isEmpty then provisional
  else
    let fields := ensureNamed named (dataFields named provisional)
    let first := removeItems provisional last
    let owned := ensureNamed named restoring
    let addedBack := removeItems provisional (fieldDifference fields (ensureNamed named (dataFields named first) ++ owned))
    let dangling := fieldIntersection (fieldDifference fields (ensureNamed named (dataFields named addedBack))) last
    removeItems provisional dangling

/-- Removing data never fabricates a path or retains a selected subtree. -/
theorem remove_items_has_no_selected_descendant (s : State) (fields : FieldSet)
    (cell : Cell) (member : cell ∈ (removeItems s fields).cells) :
    fields.any (fun field => under field cell.path) = false := by
  obtain ⟨original,present,equality⟩ := List.mem_map.mp member
  have kept := (List.mem_filter.mp present).2
  have samePath : original.path = cell.path := by
    split at equality <;> simpa only using congrArg Cell.path equality
  simpa [samePath] using kept

end SSA
