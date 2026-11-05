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

namespace SSA.FieldsV1

/-- The field-path trie has independent membership and descendant ownership. -/
inductive Trie where
  | node (member : Bool) (children : List (Segment × Trie))
  deriving Repr

def nonempty : Trie → Prop
  | .node member children => member = true ∨ children ≠ []

/-- Real fieldpath sets have unique edge keys and no empty child sets. -/
inductive Canonical : Trie → Prop where
  | node (member : Bool) (children : List (Segment × Trie))
      (unique : (children.map Prod.fst).Nodup)
      (inhabited : ∀ child ∈ children, nonempty child.2)
      (subtrees : ∀ child ∈ children, Canonical child.2) :
      Canonical (.node member children)

/-- This is the FieldsV1 object grammar after JSON and path-element decoding.
    `dot` denotes the literal "." key; an empty object implicitly denotes a member. -/
inductive Object where
  | fields (dot : Bool) (children : List (Segment × Object))
  deriving Repr

/-- emitContentsV1 writes "." only when membership and descendants coexist. -/
def encode : Trie → Object
  | .node member children =>
      .fields (member && !children.isEmpty) (children.map fun child => (child.1, encode child.2))
termination_by trie => sizeOf trie
decreasing_by
  have memberSize := List.sizeOf_lt_of_mem (show child ∈ children from by assumption)
  cases child
  simp_all <;> omega

/-- readIterV1 treats a leaf object as membership even when no "." key was emitted. -/
def decode : Object → Trie
  | .fields dot children =>
      .node (dot || children.isEmpty) (children.map fun child => (child.1, decode child.2))
termination_by object => sizeOf object
decreasing_by
  have memberSize := List.sizeOf_lt_of_mem (show child ∈ children from by assumption)
  cases child
  simp_all <;> omega

private theorem mapped_isEmpty {α β : Type} (f : α → β) (xs : List α) :
    (xs.map f).isEmpty = xs.isEmpty := by cases xs <;> rfl

private theorem map_fixed {α : Type} (xs : List α) (f : α → α)
    (fixed : ∀ x ∈ xs, f x = x) : xs.map f = xs := by
  induction xs with
  | nil => rfl
  | cons head tail ih =>
      simp only [List.map_cons]
      rw [fixed head (by simp), ih (by intro x member; exact fixed x (by simp [member]))]

/-- Every nonempty canonical subtree survives the compressed member representation. -/
theorem subtree_roundtrip (trie : Trie) (canonical : Canonical trie)
    (present : nonempty trie) : decode (encode trie) = trie := by
  induction canonical with
  | node member children unique inhabited subtrees ih =>
      have childrenSame : children.map (fun child => (child.1, decode (encode child.2))) = children := by
        apply map_fixed
        intro child belongs
        rcases child with ⟨key, child⟩
        simp only [ih (key, child) belongs (inhabited (key, child) belongs)]
      simp only [encode, decode, List.map_map, mapped_isEmpty]
      change Trie.node ((member && !children.isEmpty) || children.isEmpty)
        (children.map (fun child => (child.1, decode (encode child.2)))) = _
      rw [childrenSame]
      cases children with
      | nil =>
          have isMember : member = true := by simpa only [nonempty, ne_eq, not_true_eq_false, or_false] using present
          simp [isMember]
      | cons child rest => simp

/-- Root objects do not represent ownership of the resource's empty path. -/
def decodeRoot : Object → Trie
  | .fields _ children => .node false (children.map fun child => (child.1, decode child.2))

/-- Empty field sets and all canonical nonempty field sets round trip at the root. -/
theorem root_roundtrip (children : List (Segment × Trie))
    (canonical : Canonical (.node false children)) :
    decodeRoot (encode (.node false children)) = .node false children := by
  cases canonical with
  | node _ _ unique inhabited subtrees =>
      simp only [encode, decodeRoot, List.map_map]
      congr 1
      apply map_fixed
      intro child belongs
      rcases child with ⟨key, child⟩
      simp only [Function.comp_apply, subtree_roundtrip child
        (subtrees (key, child) belongs) (inhabited (key, child) belongs)]

def memberAt : Trie → Path → Bool
  | .node member _, [] => member
  | .node _ children, segment :: rest =>
      children.any fun child => child.1 == segment && memberAt child.2 rest
termination_by _ path => path.length

/-- The round trip preserves every claim, including item membership with nested claims. -/
theorem root_membership_roundtrip (children : List (Segment × Trie))
    (canonical : Canonical (.node false children)) (path : Path) :
    memberAt (decodeRoot (encode (.node false children))) path =
      memberAt (.node false children) path := by
  rw [root_roundtrip children canonical]

/-- Owning a descendant never implies owning the container's membership marker. -/
theorem descendant_does_not_imply_membership (key : Segment) (child : Trie) :
    memberAt (.node false [(key, child)]) [] = false := by simp [memberAt]

/-- The empty subtree cannot be serialized as a child without fabricating membership. -/
theorem empty_child_is_not_lossless :
    decode (encode (.node false [])) = .node true [] ∧
      decode (encode (.node false [])) ≠ Trie.node false [] := by
  constructor
  · simp [encode, decode]
  · simp [encode, decode]

end SSA.FieldsV1
