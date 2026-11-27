/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.NumericCounterexamples
import SSA.FieldsV1

namespace SSA.NumericCodecCounterexamples

open NumericCounterexamples

def original : Nat := 2^53+1
def neighbor : Nat := 2^53

/-- Canonical protocol labels contain JSON scalar text, not a float-rounded path identity. -/
def key (number : Nat) : Segment := .item [("id",toString number)]
def itemPath (number : Nat) : Path := [.field "endpoints",key number]

/-- The concrete wire literal is an integer; the numeric-key reader constructs a float.
    This specifies the observed read callback, not a general JSON parser. -/
def readKey (integerToken : Nat) : Segment := key (roundInteger integerToken)

def encoded (selector : Segment) : FieldsV1.Object :=
  .fields false [(.field "endpoints",.fields false [(selector,.fields false [])])]

/-- The structural FieldsV1 grammar receives the already-decoded selector.
    Native Go separately checks both the complete byte round trip and that selector. -/
def roundtrip (integerToken : Nat) : FieldsV1.Trie :=
  FieldsV1.decodeRoot (encoded (readKey integerToken))

def originalTrie (integerToken : Nat) : FieldsV1.Trie :=
  FieldsV1.decodeRoot (encoded (key integerToken))

private theorem singleton_canonical (selector : Segment) (child : FieldsV1.Trie)
    (canonical : FieldsV1.Canonical child) (present : FieldsV1.nonempty child) :
    FieldsV1.Canonical (.node false [(selector,child)]) := by
  refine .node false [(selector,child)] (by simp) ?_ ?_
  · intro candidate member
    have same := List.mem_singleton.mp member
    subst candidate
    exact present
  · intro candidate member
    have same := List.mem_singleton.mp member
    subst candidate
    exact canonical

theorem original_trie_is_canonical : FieldsV1.Canonical (originalTrie original) := by
  have leaf : FieldsV1.Canonical (.node true []) := by
    refine .node true [] (by simp) (by simp) (by simp)
  have item := singleton_canonical (key original) (.node true []) leaf
    (by simp [FieldsV1.nonempty])
  have root := singleton_canonical (.field "endpoints") (.node false [(key original,.node true [])]) item
    (by simp [FieldsV1.nonempty])
  simpa [originalTrie,encoded,FieldsV1.decodeRoot,FieldsV1.decode] using root

theorem original_encodes_exact_key :
    FieldsV1.encode (originalTrie original) = encoded (key original) := by
  simp [originalTrie,encoded,FieldsV1.decodeRoot,FieldsV1.decode,FieldsV1.encode]

theorem exact_integer_keys_are_distinct : key original ≠ key neighbor := by decide

theorem reader_collapses_distinct_integer_keys :
    readKey original = readKey neighbor ∧ original ≠ neighbor := by decide

theorem numeric_key_roundtrip_moves_membership :
    FieldsV1.memberAt (originalTrie original) (itemPath original) = true ∧
    FieldsV1.memberAt (roundtrip original) (itemPath original) = false ∧
    FieldsV1.memberAt (roundtrip original) (itemPath neighbor) = true := by
  have rounded : roundInteger original = neighbor := by decide
  simp only [roundtrip,readKey,rounded]
  simp [originalTrie,itemPath,encoded,
    FieldsV1.decodeRoot,FieldsV1.decode,FieldsV1.memberAt,key,original,neighbor]

/-- This refutes the unrestricted numeric path round-trip contract, not FieldsV1's
    existing structural theorem whose labels have already been decoded. -/
theorem exact_numeric_path_roundtrip_is_false :
    ¬ ∀ integerToken, integerToken < 2^63 →
      FieldsV1.Canonical (originalTrie integerToken) →
      ∀ path, FieldsV1.memberAt (roundtrip integerToken) path =
        FieldsV1.memberAt (originalTrie integerToken) path := by
  intro preserves
  have witness := numeric_key_roundtrip_moves_membership
  have same := preserves original (by decide) original_trie_is_canonical (itemPath original)
  rw [witness.1,witness.2.1] at same
  contradiction

theorem small_key_roundtrip_control :
    roundtrip 42 = originalTrie 42 ∧ readKey 42 ≠ readKey 41 := by
  constructor
  · have rounded : roundInteger 42 = 42 := by decide
    simp [roundtrip,originalTrie,readKey,rounded]
  · decide

structure Observation where
  originalKey : String
  decodedKey : String
  retainedOriginal : Bool
  claimsNeighbor : Bool
  distinctIntegers : Bool
  smallKeyPreserved : Bool
  deriving Lean.ToJson

def observe : Observation := {
  originalKey := toString original
  decodedKey := toString (roundInteger original)
  retainedOriginal := FieldsV1.memberAt (roundtrip original) (itemPath original)
  claimsNeighbor := FieldsV1.memberAt (roundtrip original) (itemPath neighbor)
  distinctIntegers := original != neighbor
  smallKeyPreserved := roundInteger 42 == 42
}

end SSA.NumericCodecCounterexamples
