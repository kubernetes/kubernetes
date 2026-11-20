/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
namespace SSA.RawGoSyntax

/-- Full structural Go syntax. Unknown operations fail if reached, rather than being erased. -/
inductive Node where
  | absent
  | atom (text : String)
  | bit (present : Bool)
  | sequence (nodes : List Node)
  | record (tag : String) (fields : List (String × Node))
  deriving Repr

def find : List (String × Node) → String → Node
  | [],_ => .absent
  | (name,value)::rest,key => if name == key then value else find rest key

def field : Node → String → Node
  | .record _ fields,key => find fields key
  | _,_ => .absent

end SSA.RawGoSyntax
