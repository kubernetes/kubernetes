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
import Lean

namespace SSA

open Lean

/-- Keys carry their values, so different list items have different paths. -/
inductive Segment where
  | field (name : String)
  | item (keys : List (String × String))
  | element (value : String)
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

abbrev Path := List Segment

/-- Paths are already normalized; parsing and API-version conversion are outside this model. -/
def under (parent child : Path) : Bool :=
  parent.isPrefixOf child

/-- An instantiated required core follows required object branches, not optional descendants. -/
inductive RequiredCore where
  | none
  | value (path : Path)
  | object (path : Path) (children : RequiredCore)
  | collection (path : Path)
  | both (left right : RequiredCore)
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

/-- Item descriptors are a finite, instance-expanded schema, including items not yet present. -/
structure ItemSpec where
  path : Path
  required : RequiredCore
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

abbrev Schema := List ItemSpec

/-- Container presence is independent of the values or list items below it. -/
inductive Content where
  | value (text : String)
  | object
  | collection
  | item
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

structure Cell where
  path : Path
  content : Content
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

inductive ClaimKind where
  | value
  | presence
  | item
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

structure Footprint where
  path : Path
  kind : ClaimKind
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

structure Claim where
  manager : String
  target : Footprint
  applied : Bool := true
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

structure State where
  cells : List Cell := []
  claims : List Claim := []
  deriving Repr, BEq, DecidableEq, FromJson, ToJson

deriving instance ReflBEq, LawfulBEq for Segment, RequiredCore, ItemSpec, Content, Cell,
  ClaimKind, Footprint, Claim, State

def Cell.footprint (cell : Cell) : Footprint :=
  ⟨cell.path, match cell.content with
    | .value _ => .value
    | .item => .item
    | .object | .collection => .presence⟩

def State.hasItem (s : State) (path : Path) : Bool :=
  s.cells.any fun c => c.path == path && c.content == .item

def State.hasValue (s : State) (path : Path) : Bool :=
  s.cells.any fun c => c.path == path && match c.content with
    | .value _ => true
    | _ => false

def State.hasContainer (s : State) (path : Path) : Bool :=
  s.cells.any fun c => c.path == path && (c.content == .object || c.content == .collection)

def State.owns (s : State) (manager : String) (path : Path) (kind : ClaimKind) : Bool :=
  s.claims.any fun c => c.manager == manager && c.target == ⟨path, kind⟩

/-- Descendant claims establish a map's presence, but cannot cover its required children. -/
def State.ownsMapPresence (s : State) (manager : String) (path : Path) : Bool :=
  s.claims.any fun c => c.manager == manager && under path c.target.path

def covers (s : State) (manager : String) : RequiredCore → Bool
  | .none => true
  | .value path => s.owns manager path .value
  | .object path children => s.ownsMapPresence manager path && covers s manager children
  | .collection path => s.owns manager path .presence
  | .both left right => covers s manager left && covers s manager right

/-- Required collection existence is explicit, including an applied empty granular list. -/
def full (s : State) (manager : String) (item : ItemSpec) : Bool :=
  s.owns manager item.path .item && covers s manager item.required

def hasFull (s : State) (item : ItemSpec) : Bool :=
  s.claims.any fun c => full s c.manager item

def complete (schema : Schema) (s : State) : Bool :=
  schema.all fun item => !s.hasItem item.path || hasFull s item

def requiredPresent (s : State) : RequiredCore → Bool
  | .none => true
  | .value path => s.hasValue path
  | .object path children =>
      s.cells.any (fun c => c.path == path && c.content == .object) &&
      requiredPresent s children
  | .collection path => s.cells.any (fun c => c.path == path && c.content == .collection)
  | .both left right => requiredPresent s left && requiredPresent s right

def valuesValid (schema : Schema) (s : State) : Bool :=
  schema.all fun item => !s.hasItem item.path || requiredPresent s item.required

def prefixes : Path → List Path
  | [] => [[]]
  | step :: rest => [] :: (prefixes rest).map (step :: ·)

def isItemPath (path : Path) : Bool :=
  match path.getLast? with
  | some (.item _) => true
  | _ => false

/-- Required object branches cannot cross into an independently owned nested list item. -/
def requiredPath (parent child : Path) : Bool :=
  under parent child && parent.length < child.length &&
    (child.drop parent.length).all (fun step => match step with
      | .field _ => true
      | .item _ | .element _ => false)

def coreWithin (parent : Path) : RequiredCore → Bool
  | .none => true
  | .value path | .collection path => requiredPath parent path
  | .object path children => requiredPath parent path && coreWithin path children
  | .both left right => coreWithin parent left && coreWithin parent right

def schemaValid (schema : Schema) : Bool :=
  (schema.map (·.path)).eraseDups.length == schema.length &&
  schema.all (fun item =>
    isItemPath item.path && coreWithin item.path item.required &&
    (prefixes item.path).all (fun p =>
      !isItemPath p || schema.any (fun ancestor => ancestor.path == p)))

def targetExists (s : State) (target : Footprint) : Bool :=
  match target.kind with
  | .value => s.hasValue target.path
  | .presence => s.hasContainer target.path ||
      s.cells.any (fun cell => cell.path == target.path && cell.content == .value "null")
  | .item => s.hasItem target.path

/-- Strict key-name order gives composite selectors one canonical representation. -/
def canonicalKeys : List (String × String) → Bool
  | [] | [_] => true
  | left :: right :: rest => decide (left.1 < right.1) && canonicalKeys (right :: rest)

/-- Each proper prefix is a real container of the kind selected by the next segment. -/
def compatiblePath (s : State) (path : Path) : Bool :=
  !path.isEmpty && (prefixes path).all (fun parent =>
    if parent.isEmpty then
      match path.head? with
      | some (.field _) => true
      | _ => false
    else if parent == path then true
    else match (path.drop parent.length).head? with
      | some (.field _) => s.cells.any (fun cell => cell.path == parent &&
          (cell.content == .object || cell.content == .item))
      | some (.item keys) => !keys.isEmpty &&
          (keys.map Prod.fst).eraseDups.length == keys.length &&
          canonicalKeys keys &&
          s.cells.any (fun cell => cell.path == parent && cell.content == .collection)
      | some (.element _) =>
          s.cells.any (fun cell => cell.path == parent && cell.content == .collection)
      | none => false)

/-- Reject dangling claims, duplicate values, and descendants of absent item instances. -/
def shapeValid (schema : Schema) (s : State) : Bool :=
  schemaValid schema &&
  (s.cells.map (·.path)).eraseDups.length == s.cells.length &&
  s.cells.all (fun c =>
    (c.content != .item || schema.any (fun item => item.path == c.path)) &&
    (prefixes c.path).all (fun p => !isItemPath p || s.hasItem p) &&
    compatiblePath s c.path &&
    (isItemPath c.path == (c.content == .item))) &&
  s.claims.all (fun c => targetExists s c.target)

def valid (schema : Schema) (s : State) : Bool :=
  shapeValid schema s && valuesValid schema s && complete schema s

end SSA
