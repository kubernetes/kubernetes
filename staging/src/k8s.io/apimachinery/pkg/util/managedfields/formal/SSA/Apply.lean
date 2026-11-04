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

/-- A normalized apply replaces one manager's complete footprint, as SSA does. -/
structure Request where
  manager : String
  desired : List Footprint
  writes : List Cell
  force : Bool := false
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

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
    (r.writes.map (·.path)).eraseDups.length == r.writes.length &&
    r.writes.all (fun cell => (prefixes cell.path).all (fun p =>
      !isItemPath p || r.desired.contains ⟨p, .item⟩))

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
  r.desired.map (fun target => ⟨r.manager, target⟩)

def merged (s : State) (r : Request) : State :=
  { cells := r.writes ++ s.cells.filter (fun old => !r.writes.any (fun c => c.path == old.path))
    claims := updatedClaims s r }

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

/-- Optional unowned values disappear; remaining descendants keep container markers present. -/
def pruneReleased (before : State) (r : Request) (s : State) : State :=
  { s with cells := s.cells.filter fun c =>
      match c.content with
      | .value _ => !(released before r c.path .value && !claimed s c.path .value)
      | .object | .collection =>
          !(released before r c.path .presence &&
            !s.claims.any (fun claim => under c.path claim.target.path))
      | .item => true }

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
