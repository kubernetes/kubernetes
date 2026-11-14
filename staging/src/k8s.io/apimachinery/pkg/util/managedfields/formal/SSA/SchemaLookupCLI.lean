/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import Lean
import SSA.SchemaResolveCache

namespace SSA.SchemaLookupCLI

open Lean SchemaResolveMachine

private structure ContainerView where
  payload : String
  relationship : String
  deriving FromJson, ToJson

private structure AtomView where
  map : Option ContainerView
  scalar : Option String
  list : Option ContainerView
  deriving FromJson, ToJson

private def AtomView.decode (value : AtomView) : Atom String String String String :=
  {map := value.map.map (fun c => {payload := c.payload,relationship := c.relationship})
   scalar := value.scalar
   list := value.list.map (fun c => {payload := c.payload,relationship := c.relationship})}

private def AtomView.encode (value : Atom String String String String) : AtomView :=
  {map := value.map.map (fun c => {payload := c.payload,relationship := c.relationship})
   scalar := value.scalar
   list := value.list.map (fun c => {payload := c.payload,relationship := c.relationship})}

private structure Entry where
  key : Nat
  atom : AtomView
  deriving FromJson, ToJson

private structure Request where
  operation : String
  atom : AtomView
  named : Option String
  relationship : Option String
  found : Bool
  key : Nat
  allocated : Bool
  cached : List Entry
  observe : List Nat
  deriving FromJson

private structure Report where
  atom : AtomView
  found : Bool
  allocated : Bool
  cached : List Entry
  held : Bool
  deriving ToJson

def evaluate (input : String) : Except String String := do
  let requests ← Json.parse input >>= fromJson? (α := List Request)
  if requests.isEmpty then throw "Schema lookup protocol requires nonempty cases"
  let reports ← requests.mapM fun request => do
    if !request.allocated && !request.cached.isEmpty then throw "A nil schema cache cannot contain entries"
    let atom := request.atom.decode
    let reference : Reference String String String Nat String := {
      original := request.key,named := request.named,inlined := atom,relationship := request.relationship}
    let ops : Ops String String String String := {findNamed := fun _ => (atom,request.found)}
    let state : SchemaResolveMachine.State String String String Nat String := {
      cache := if request.allocated then some (fun key =>
        (request.cached.reverse.find? (fun entry => entry.key == key)).map (fun entry => entry.atom.decode)) else none}
    let ((atom,found),after) ← match request.operation with
      | "plain" => do
          let outcome ← plain ops reference
          pure (outcome,state)
      | "resolve" => SchemaResolveSubset.invoke ops reference state
      | _ => throw "Schema lookup operation must be plain or resolve"
    let cached := request.observe.filterMap fun key =>
      (after.cache.bind (fun values => values key)).map (fun value => ({key,atom := AtomView.encode value} : Entry))
    pure ({atom := AtomView.encode atom,found,allocated := after.cache.isSome,cached,held := after.held} : Report)
  pure (toJson reports).compress

end SSA.SchemaLookupCLI
