/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.ApplySource
import SSA.ElementAlgebra

namespace SSA.NumericCounterexamples

open GoSubset

/-- Nearest, ties-to-even quantization at a binary64 spacing. -/
def roundQuantum (spacing number : Nat) : Nat :=
  let whole := number/spacing
  let remainder := number%spacing
  if 2*remainder < spacing then whole*spacing
  else if spacing < 2*remainder then (whole+1)*spacing
  else if whole%2 = 0 then whole*spacing else (whole+1)*spacing

/-- Integer-to-binary64 rounding specification for the finite nonnegative int64 witness domain. -/
def roundInteger (number : Nat) : Nat :=
  roundQuantum (2^((Nat.log2 number+1)-53)) number

/-- Floating witnesses are already represented finite, nonnegative integral binary64 values.
    Nonintegral floats, negative numbers, NaN and infinity are outside this counterexample model. -/
inductive Number where
  | integer (value : Nat)
  | floating (value : Nat)
  deriving Repr, BEq, DecidableEq

def magnitude : Number → Nat
  | .integer value | .floating value => value

def represented : Number → Bool
  | .integer value => value < 2^63
  | .floating value => value ≤ 2^63 && roundInteger value == value

def coerced : Number → Nat
  | .integer value => roundInteger value
  | .floating value => value

/-- Go compares integers exactly to integers, but rounds them when either operand is a float. -/
def numericEqual (left right : Number) : Bool :=
  match left,right with
  | .integer left,.integer right => left == right
  | _,_ => coerced left == coerced right

def numericLess (left right : Number) : Bool :=
  match left,right with
  | .integer left,.integer right => left < right
  | _,_ => coerced left < coerced right

def original : Number := .integer (2^63-1)
def control : Number := .integer (2^63-2)
def desired : Number := .floating (2^63)

theorem witnesses_are_represented :
    represented original = true ∧ represented control = true ∧ represented desired = true := by decide

theorem equality_is_not_transitive :
    numericEqual original desired = true ∧ numericEqual desired control = true ∧
    numericEqual original control = false := by decide

/-- Comparator incomparability fails transitivity too; sorted inputs do not repair it. -/
theorem incomparability_is_not_transitive :
    (numericLess original desired = false ∧ numericLess desired original = false) ∧
    (numericLess desired control = false ∧ numericLess control desired = false) ∧
    numericLess control original = true := by decide

theorem rounded_equality_changes_magnitude :
    numericEqual original desired = true ∧ magnitude original ≠ magnitude desired := by decide

/-- Even on the represented domain, no quotient into identity can faithfully encode this equality. -/
theorem no_faithful_equality_projection :
    ¬ ∃ (α : Type) (encode : Number → α), ∀ left right,
      represented left = true → represented right = true →
      (numericEqual left right = true ↔ encode left = encode right) := by
  rintro ⟨α,encode,faithful⟩
  have domain := witnesses_are_represented
  have values := equality_is_not_transitive
  have first := (faithful original desired domain.1 domain.2.2).mp values.1
  have second := (faithful desired control domain.2.2 domain.2.1).mp values.2.1
  have collapsed := (faithful original control domain.1 domain.2.1).mpr (first.trans second)
  simp [values.2.2] at collapsed

def numericMember (number : Number) (numbers : List Number) : Bool :=
  numbers.any (numericEqual number)

/-- Sorted inputs alone are insufficient: the missing global comparator premise matters. -/
theorem difference_retains_an_excluded_member :
    ([control,original] : List Number).Pairwise (fun a b => numericLess a b = true) ∧
    ([desired] : List Number).Pairwise (fun a b => numericLess a b = true) ∧
    ElementSubset.exec numericLess 22 ElementSource.differenceProgram
      {left := [control,original],right := [desired]} =
        .ok (.returned [.set [original]]) ∧
    numericMember original [desired] = true := by
  refine ⟨by decide,by decide,?_,by decide⟩
  have cd : numericLess control desired = false := by decide
  have dc : numericLess desired control = false := by decide
  have computed : ElementAlgebra.compute .difference numericLess [control,original] [desired] = [original] := by
    simp [ElementAlgebra.compute,ElementAlgebra.emit,ElementAlgebra.finish,cd,dc]
  simpa [ElementAlgebra.program,computed] using
    ElementAlgebra.imported_matches .difference numericLess [control,original] [desired]

structure Object where
  amount : Option Number := none
  note : Bool := false
  deriving Repr

def amountPath : Path := [.field "amount"]
def notePath : Path := [.field "note"]

def fields (object : Object) : FieldSet :=
  (if object.amount.isSome then [amountPath] else []) ++ (if object.note then [notePath] else [])

def amountEqual : Option Number → Option Number → Bool
  | none,none => true
  | some left,some right => numericEqual left right
  | _,_ => false

def merged (live config : Object) : Object :=
  {amount := match config.amount with | none => live.amount | some value => some value
   note := live.note || config.note}

/-- A primitive instance for the fresh-manager scalar history.
    General Go typed merge/comparison, float conversion and updater refinement remain separate. -/
def sourceOps : Ops Object Unit where
  convert object _ := .ok object
  missing _ := false
  toFields object := .ok (fields object)
  ensureNamed _ _ selected := selected
  remove object _ := object
  wrap _ error := error
  mergeObjects live config := .ok (merged live config)
  finishUpdate live object _ owners actor force :=
    let conflicts := fun owner : String × ManagerFields => owner.1 != actor &&
      owner.2.fields.contains amountPath && !amountEqual live.amount object.amount
    if !force && owners.any conflicts then .error ()
    else .ok (owners.filterMap fun owner =>
      let retained := if conflicts owner then owner.2.fields.filter (· != amountPath) else owner.2.fields
      if retained.isEmpty then none else some (owner.1,{owner.2 with fields := retained}))
  equalObjects live object := amountEqual live.amount object.amount && (live.note == object.note)

def initial : Object := {amount := some original}
def intent (note : Bool) : Object := {amount := some desired,note}
def ownersA : ManagedFields := [("A",⟨[amountPath],"v1",true⟩)]
def ownersAfter (note : Bool) : ManagedFields :=
  managerSet ownersA "B" ⟨fields (intent note),"v1",true⟩

/-- A nil no-op result retains live data while the wrapper installs returned ownership. -/
def committedObject (before : Object) : Value Object Unit → Object
  | .object object => object
  | _ => before

def Accepted (config : Object) (object : Value Object Unit) (owners : ManagedFields) : Prop :=
  exec sourceOps 128 GoSource.applyProgram
    (ApplySource.applyEnv initial config "v1" ownersA "B" false) =
      .ok (.returned [object,.managed owners,.nil])

/-- The A-owned starting state is reachable from an empty object through the imported body. -/
theorem original_is_source_reachable :
    exec sourceOps 128 GoSource.applyProgram
      (ApplySource.applyEnv {} initial "v1" [] "A" false) =
        .ok (.returned [.object initial,.managed ownersA,.nil]) := by
  rw [ApplySource.imported_apply_matches_spec]
  rfl

theorem scalar_only_is_false_noop : Accepted (intent false) .nil (ownersAfter false) := by
  rw [Accepted,ApplySource.imported_apply_matches_spec]
  rfl

theorem independent_note_changes_foreign_number :
    Accepted (intent true) (.object (intent true)) (ownersAfter true) ∧
    ("A",⟨[amountPath],"v1",true⟩) ∈ ownersAfter true ∧
    (intent true).amount.map magnitude ≠ initial.amount.map magnitude := by
  refine ⟨?_,by simp [ownersAfter,managerSet,ownersA],by decide⟩
  rw [Accepted,ApplySource.imported_apply_matches_spec]
  rfl

def ForcedAccepted (config : Object) (object : Value Object Unit) (owners : ManagedFields) : Prop :=
  exec sourceOps 128 GoSource.applyProgram
    (ApplySource.applyEnv initial config "v1" ownersA "B" true) =
      .ok (.returned [object,.managed owners,.nil])

/-- Force does not transfer a change that the rounded comparison fails to detect. -/
theorem forced_change_retains_foreign_claim :
    ForcedAccepted (intent true) (.object (intent true)) (ownersAfter true) ∧
    ("A",⟨[amountPath],"v1",true⟩) ∈ ownersAfter true ∧
    (intent true).amount.map magnitude ≠ initial.amount.map magnitude := by
  refine ⟨?_,by simp [ownersAfter,managerSet,ownersA],by decide⟩
  rw [ForcedAccepted,ApplySource.imported_apply_matches_spec]
  rfl

/-- The same forced updater transfers an exactly detected int-to-int modification. -/
theorem forced_integer_control_transfers :
    ForcedAccepted {amount := some control,note := true} (.object {amount := some control,note := true})
      (managerSet [] "B" ⟨[amountPath,notePath],"v1",true⟩) := by
  rw [ForcedAccepted,ApplySource.imported_apply_matches_spec]
  rfl

/-- An exact int-to-int change still reaches the conflict error path. -/
theorem integer_change_conflicts :
    exec sourceOps 128 GoSource.applyProgram
      (ApplySource.applyEnv initial {amount := some control,note := true} "v1" ownersA "B" false) =
        .ok (.returned [.nil,.managed [],.error ()]) := by
  rw [ApplySource.imported_apply_matches_spec]
  rfl

/-- The fresh B request's pruning callback is realized by the imported nil-last-set branch. -/
theorem prune_callback_is_realized (note : Bool) :
    exec sourceOps 64 GoSource.pruneProgram
      (PruneSource.pruneEnv (intent note) (ownersAfter note) "B" none) =
        .ok (.returned [.object (intent note),.nil]) := by
  apply PruneSource.imported_prune_matches_spec sourceOps (intent note) (ownersAfter note) "B" none
    ⟨fields (intent note),"v1",true⟩
  cases note <;> rfl

def EveryAssertedScalarSurvives : Prop :=
  ∀ config wanted object owners, config.amount = some wanted → represented wanted = true →
    Accepted config object owners →
    (committedObject initial object).amount.map magnitude = some (magnitude wanted)

theorem exact_scalar_intent_is_false : ¬EveryAssertedScalarSurvives := by
  intro preserves
  have wrong := preserves (intent false) desired .nil (ownersAfter false) rfl
    witnesses_are_represented.2.2 scalar_only_is_false_noop
  have different : initial.amount.map magnitude ≠ some (magnitude desired) := by decide
  exact different wrong

def EveryNonforceForeignScalarSurvives : Prop :=
  ∀ config object owners, Accepted config object owners →
    ("A",⟨[amountPath],"v1",true⟩) ∈ owners →
    (committedObject initial object).amount.map magnitude = initial.amount.map magnitude

theorem nonforce_foreign_value_safety_is_false : ¬EveryNonforceForeignScalarSurvives := by
  intro preserves
  have witness := independent_note_changes_foreign_number
  have wrong := preserves (intent true) (.object (intent true)) (ownersAfter true) witness.1 witness.2.1
  exact witness.2.2 wrong

def EveryForcedChangeTransfersForeign : Prop :=
  ∀ config object owners, ForcedAccepted config object owners →
    (committedObject initial object).amount.map magnitude ≠ initial.amount.map magnitude →
    ¬ ∃ entry, ("A",entry) ∈ owners ∧ amountPath ∈ entry.fields

theorem forced_transfer_is_false : ¬EveryForcedChangeTransfersForeign := by
  intro transfers
  have witness := forced_change_retains_foreign_claim
  have noClaim := transfers (intent true) (.object (intent true)) (ownersAfter true) witness.1 witness.2.2
  exact noClaim ⟨⟨[amountPath],"v1",true⟩,witness.2.1,by simp⟩

structure Observation where
  roundedOriginal : String
  roundedControl : String
  originalEqualsDesired : Bool
  desiredEqualsControl : Bool
  originalEqualsControl : Bool
  differenceRemainder : String
  excludedContainsRemainder : Bool
  scalarOnlyAmount : String
  withNoteAmount : String
  integerConflict : Bool
  bothCoOwn : Bool
  forcedChangeKeepsForeign : Bool
  forcedExactChangeTransfers : Bool
  deriving Lean.ToJson

def observedAmount (object : Object) : String :=
  match object.amount with
  | none => "absent"
  | some number => toString (magnitude number)

def sourceObservation (config : Object) (force : Bool := false) : Except String (Object × ManagedFields) :=
  match exec sourceOps 128 GoSource.applyProgram
      (ApplySource.applyEnv initial config "v1" ownersA "B" force) with
  | .ok (.returned [object,.managed owners,.nil]) => .ok (committedObject initial object,owners)
  | _ => .error "Numeric source execution did not return an accepted object"

def coOwns (owners : ManagedFields) : Bool :=
  ["A","B"].all (fun actor =>
    owners.any (fun owner => owner.1 == actor && owner.2.fields.contains amountPath))

/-- Executable source replay is compared with native Go primitives and managedfields.
    The HTTP test independently realizes the same fresh-manager history and stored observations. -/
def observe : Except String Observation := do
  let scalar ← sourceObservation (intent false)
  let note ← sourceObservation (intent true)
  let forced ← sourceObservation (intent true) true
  let forcedControl ← sourceObservation {amount := some control,note := true} true
  let remainder ← match ElementSubset.exec numericLess 22 ElementSource.differenceProgram
      {left := [control,original],right := [desired]} with
    | .ok (.returned [.set [number]]) => .ok (toString (magnitude number))
    | _ => .error "Numeric element difference did not return the single witness"
  let conflict := match exec sourceOps 128 GoSource.applyProgram
      (ApplySource.applyEnv initial {amount := some control,note := true} "v1" ownersA "B" false) with
    | .ok (.returned [.nil,.managed [],.error _]) => true
    | _ => false
  return {
    roundedOriginal := toString (coerced original)
    roundedControl := toString (coerced control)
    originalEqualsDesired := numericEqual original desired
    desiredEqualsControl := numericEqual desired control
    originalEqualsControl := numericEqual original control
    differenceRemainder := remainder
    excludedContainsRemainder := numericMember original [desired]
    scalarOnlyAmount := observedAmount scalar.1
    withNoteAmount := observedAmount note.1
    integerConflict := conflict
    bothCoOwn := coOwns scalar.2 && coOwns note.2
    forcedChangeKeepsForeign := coOwns forced.2 &&
      forced.1.amount.map magnitude != initial.amount.map magnitude
    forcedExactChangeTransfers := forcedControl.1.amount.map magnitude == some (magnitude control) &&
      !(forcedControl.2.any fun entry => entry.1 == "A" && entry.2.fields.contains amountPath) &&
      forcedControl.2.any (fun entry => entry.1 == "B" && entry.2.fields.contains amountPath)
  }

end SSA.NumericCounterexamples

