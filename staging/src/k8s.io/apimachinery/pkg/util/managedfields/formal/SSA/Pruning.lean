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

/-- Recompute from the merged object, restoring both retained data and ownership. -/
def addBackOwnedFields (named : FieldSet) (merged pruned : State) (managed : FieldSet) : State :=
  removeItems merged (fieldDifference (ensureNamed named (dataFields named merged))
    (ensureNamed named (dataFields named pruned) ++ ensureNamed named managed))

/-- Only last-applied fields may remain removed after ownership restoration. -/
def addBackDanglingFields (named : FieldSet) (merged pruned : State) (last : FieldSet) : State :=
  removeItems merged (fieldIntersection
    (fieldDifference (ensureNamed named (dataFields named merged))
      (ensureNamed named (dataFields named pruned))) (ensureNamed named last))

theorem ensure_named_empty_iff (named fields : FieldSet) :
    (ensureNamed named fields).isEmpty = fields.isEmpty := by
  cases fields with
  | nil =>
      have absent : named.filter (fun _ => false) = [] :=
        List.filter_eq_nil_iff.mpr (by intros; simp)
      simp [ensureNamed, absent]
  | cons field fields => rfl

/-- This is the three-pass SMD pruning algorithm, including dangling-field restoration. -/
def pruneWithFields (named : FieldSet) (provisional : State) (rawLast restoring : FieldSet) : State :=
  let last := ensureNamed named rawLast
  if last.isEmpty then provisional
  else
    let first := removeItems provisional last
    addBackDanglingFields named provisional (addBackOwnedFields named provisional first restoring) rawLast

def pruneFieldSets (named : FieldSet) (before provisional : State) (r : Request)
    (restoring : FieldSet) : State :=
  pruneWithFields named provisional (claimFields (before.claims.filter (fun c => c.manager == r.manager))) restoring

/-- Factoring the phases preserves the original normalized three-pass formula exactly. -/
theorem pruning_pipeline_formula (named : FieldSet) (provisional : State) (rawLast restoring : FieldSet) :
    pruneWithFields named provisional rawLast restoring =
      (let last := ensureNamed named rawLast
       if last.isEmpty then provisional
       else
         let fields := ensureNamed named (dataFields named provisional)
         let first := removeItems provisional last
         let owned := ensureNamed named restoring
         let addedBack := removeItems provisional
           (fieldDifference fields (ensureNamed named (dataFields named first) ++ owned))
         let dangling := fieldIntersection
           (fieldDifference fields (ensureNamed named (dataFields named addedBack))) last
         removeItems provisional dangling) := rfl

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
