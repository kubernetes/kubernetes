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
import SSA.Model

namespace SSA

open Lean

inductive Operation where
  | apply
  | update
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

deriving instance ReflBEq, LawfulBEq for Operation

/-- A normalized apply replaces one manager's complete footprint, as SSA does. -/
structure Request where
  manager : String
  desired : List Footprint
  writes : List Cell
  force : Bool := false
  context : List Cell := []
  operation : Operation := .apply
  named : List Path := []
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

/-- Structural nodes are data, but nonempty granular containers do not invent ownership. -/
def Request.cells (r : Request) : List Cell := r.writes ++ r.context

inductive ApplyError where
  | invalidInitialState
  | invalidRequest
  | conflict
  | invalidResult
  | orphanedItem
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

/-- The normalized intent contains exactly the values and presence assertions being applied. -/
def normalized (r : Request) : Bool :=
  r.desired == r.writes.map Cell.footprint &&
    (r.cells.map (·.path)).eraseDups.length == r.cells.length &&
    r.cells.all (fun cell => (prefixes cell.path).all (fun p =>
      !isItemPath p || r.desired.contains ⟨p, .item⟩)) &&
    r.context.all (fun cell => cell.content == .object || cell.content == .collection)

def changed (s : State) (cell : Cell) : Bool :=
  s.cells.any fun old => old.path == cell.path && old.content != cell.content

def conflicts (s : State) (r : Request) : Bool :=
  r.writes.any fun cell => changed s cell &&
    s.claims.any (fun c =>
      c.manager != r.manager && c.target.path == cell.path && c.target.kind == .value)

/-- Force transfers changed values; applying equal values only adds co-ownership. -/
def updatedClaims (s : State) (r : Request) : List Claim :=
  s.claims.filter (fun c => c.manager != r.manager &&
    !(r.force && c.target.kind == .value &&
      r.writes.any (fun cell => cell.path == c.target.path && changed s cell))) ++
  r.desired.map (fun target => ⟨r.manager, target, true⟩)

def merged (s : State) (r : Request) : State :=
  { cells := r.cells ++ s.cells.filter (fun old => !r.cells.any (fun c => c.path == old.path))
    claims := updatedClaims s r }

/-- Granular empty/null containers compare as leaves only while both sides are empty. -/
def modifiedCell (before after : State) (cell : Cell) : Bool :=
  match before.cells.find? (fun old => old.path == cell.path) with
  | none => false
  | some old =>
      let emptyBoth := !before.cells.any (fun child => under cell.path child.path && cell.path != child.path) &&
        !after.cells.any (fun child => under cell.path child.path && cell.path != child.path)
      let leaf := match old.content,cell.content with
        | .value _,.value _ => true
        | _,_ => emptyBoth
      leaf && old.content != cell.content

/-- A direct lifetime deletion requires the departing manager to have been full. -/
def deletesItem (before provisional : State) (r : Request) (item : ItemSpec) : Bool :=
  before.hasItem item.path && full before r.manager item &&
  !r.desired.contains ⟨item.path, .item⟩ && !hasFull provisional item

def deletedRoots (schema : Schema) (before provisional : State) (r : Request) : List Path :=
  (schema.filter (deletesItem before provisional r)).map (·.path)

/-- Parent deletion dominates descendants, even when a child has its own full manager. -/
def dropSubtrees (s : State) (roots : List Path) : State :=
  { cells := s.cells.filter (fun c => !roots.any (fun root => under root c.path))
    claims := s.claims.filter (fun c => !roots.any (fun root => under root c.target.path)) }

def released (before : State) (r : Request) (path : Path) (kind : ClaimKind) : Bool :=
  before.owns r.manager path kind && !r.desired.contains ⟨path, kind⟩

def claimed (s : State) (path : Path) (kind : ClaimKind) : Bool :=
  s.claims.any fun c => c.target == ⟨path, kind⟩

def properUnder (parent child : Path) : Bool := under parent child && parent != child

def releasesAny (before : State) (r : Request) : Bool :=
  before.claims.any (fun claim => claim.manager == r.manager && !r.desired.contains claim.target)

def retainedValue (before : State) (r : Request) (s : State) (cell : Cell) : Bool :=
  !(released before r cell.path .value && !claimed s cell.path .value)

/-- An empty unowned collection is real data; populated containers are inferred from children. -/
def supportsContainer (before : State) (r : Request) (s : State) (cell : Cell) : Bool :=
  match cell.content with
  | .value _ => retainedValue before r s cell
  | .item => true
  | .object | .collection =>
      r.cells.contains cell || claimed s cell.path .presence ||
      (!before.cells.any (fun child => properUnder cell.path child.path) &&
        !released before r cell.path .presence)

def keepCell (before : State) (r : Request) (s : State) (cell : Cell) : Bool :=
  match cell.content with
  | .value _ => retainedValue before r s cell
  | .item => true
  | .object | .collection => !releasesAny before r || s.cells.any (fun descendant =>
      under cell.path descendant.path && supportsContainer before r s descendant)

/-- Prune released values and all unsupported implicit container ancestors together. -/
def pruneReleased (before : State) (r : Request) (s : State) : State :=
  { s with cells := s.cells.filter (keepCell before r s) }

def candidate (schema : Schema) (before : State) (r : Request) : State :=
  let provisional := merged before r
  let remaining := dropSubtrees provisional (deletedRoots schema before provisional r)
  pruneReleased before r remaining

/-- Both values and ownership must be checked before an atomic commit. -/
def apply (schema : Schema) (before : State) (r : Request) : Except ApplyError State :=
  if !valid schema before then .error .invalidInitialState
  else if !normalized r then .error .invalidRequest
  else if conflicts before r && !r.force then .error .conflict
  else
    let after := candidate schema before r
    if !(shapeValid schema after && valuesValid schema after) then .error .invalidResult
    else if !complete schema after then .error .orphanedItem
    else .ok after

/-- Rejections leave the object and managed fields untouched. -/
def commit (before : State) : Except ApplyError State → State
  | .ok after => after
  | .error _ => before

end SSA
