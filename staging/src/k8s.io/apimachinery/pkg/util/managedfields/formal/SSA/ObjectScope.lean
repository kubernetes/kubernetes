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
import SSA.Update
import SSA.Locality

namespace SSA

/-- Required fields also constrain optional maps and the root, not just list-item cores. -/
structure ObjectSpec where
  path : Path
  required : RequiredCore
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

deriving instance ReflBEq, LawfulBEq for ObjectSpec

def objectExists (s : State) (spec : ObjectSpec) : Bool :=
  spec.path.isEmpty || s.cells.any (fun cell => cell.path == spec.path && cell.content == .object)

def objectFull (s : State) (manager : String) (spec : ObjectSpec) : Bool :=
  s.ownsMapPresence manager spec.path && covers s manager spec.required

/-- With no required core an ordinary map needs no additional ownership guard; item keys still do. -/
def objectsValid (objects : List ObjectSpec) (s : State) : Bool :=
  objects.all (fun spec => !objectExists s spec ||
    (requiredPresent s spec.required && (spec.required == .none ||
      s.claims.any (fun claim => objectFull s claim.manager spec))))

def deletesObject (before provisional : State) (r : Request) (spec : ObjectSpec) : Bool :=
  !spec.path.isEmpty && spec.required != .none && objectExists before spec &&
  objectFull before r.manager spec &&
  !r.desired.any (fun fp => under spec.path fp.path) &&
  !provisional.claims.any (fun claim => objectFull provisional claim.manager spec)

def scopedCandidate (objects : List ObjectSpec) (schema : Schema) (before : State) (r : Request) : State :=
  let proposed := candidate schema before r
  dropSubtrees proposed ((objects.filter (deletesObject before proposed r)).map (·.path))

/-- Item and object deletions are computed independently, then guarded as one atomic request. -/
def scopedApply (objects : List ObjectSpec) (schema : Schema) (before : State)
    (r : Request) : Except ApplyError State :=
  if !valid schema before then .error .invalidInitialState
  else if !normalized r then .error .invalidRequest
  else if conflicts before r && !r.force then .error .conflict
  else
    let after := scopedCandidate objects schema before r
    if !(shapeValid schema after && valuesValid schema after) then .error .invalidResult
    else if !complete schema after then .error .orphanedItem
    else .ok after

def checkObjects (objects : List ObjectSpec) (mode : Semantics) : Except ApplyError State → Except ApplyError State
  | .error error => .error error
  | .ok after => if mode == .full && !objectsValid objects after then .error .orphanedItem else .ok after

/-- The proposed policy guards all instantiated object scopes, including optional branches. -/
def writeScoped (objects : List ObjectSpec) (mode : Semantics) (schema : Schema)
    (before : State) (r : Request) : Except ApplyError State :=
  if mode == .full && !objectsValid objects before &&
      !(before.cells.isEmpty && before.claims.isEmpty) then .error .invalidInitialState
  else checkObjects objects mode (
    if mode == .full && r.operation == .apply then scopedApply objects schema before r
    else writeWith r.operation mode schema before r)

theorem check_objects_valid (objects : List ObjectSpec) (proposal : Except ApplyError State)
    (after : State) (accepted : checkObjects objects .full proposal = .ok after) :
    objectsValid objects after = true := by
  cases proposal with
  | error error => cases accepted
  | ok proposed =>
      cases good : objectsValid objects proposed with
      | false => simp [checkObjects,good] at accepted
      | true =>
          have result : proposed = after := by simpa [checkObjects,good] using accepted
          exact result ▸ good

/-- A partial object manager's omission cannot directly delete that object. -/
theorem partial_object_cannot_directly_delete (before provisional : State) (r : Request)
    (spec : ObjectSpec) (partialOwner : objectFull before r.manager spec = false) :
    deletesObject before provisional r spec = false := by simp [deletesObject,partialOwner]

/-- A successful full-policy write leaves every required-bearing object with one whole owner. -/
theorem accepted_objects_have_full_owner (objects : List ObjectSpec) (schema : Schema)
    (before after : State) (r : Request) (spec : ObjectSpec) (member : spec ∈ objects)
    (accepted : writeScoped objects .full schema before r = .ok after)
    (existsObject : objectExists after spec = true) (required : spec.required ≠ .none) :
    ∃ manager, objectFull after manager spec = true := by
  have validObjects : objectsValid objects after = true := by
    unfold writeScoped at accepted
    split at accepted
    · cases accepted
    · exact check_objects_valid objects _ after accepted
  have validSpec := List.all_eq_true.mp validObjects spec member
  simp only [existsObject,Bool.not_true,Bool.false_or,Bool.and_eq_true,
    Bool.or_eq_true,beq_iff_eq,required,false_or] at validSpec
  obtain ⟨claim,_,whole⟩ := List.any_eq_true.mp validSpec.2
  exact ⟨claim.manager,whole⟩

/-- Nested object scopes classify their owners independently; parent cores still recurse through required branches. -/
theorem object_full_is_local (left right : State) (manager : String) (spec : ObjectSpec)
    (agree : Locality.ClaimsAgreeUnder left right spec.path)
    (presence : left.ownsMapPresence manager spec.path = right.ownsMapPresence manager spec.path)
    (within : coreWithin spec.path spec.required = true) :
    objectFull left manager spec = objectFull right manager spec := by
  simp only [objectFull,presence,Locality.required_core_local left right manager spec.path spec.required agree within]

theorem checked_proposal (objects : List ObjectSpec) (proposal : Except ApplyError State)
    (after : State) (accepted : checkObjects objects .full proposal = .ok after) : proposal = .ok after := by
  cases proposal with
  | error error => cases accepted
  | ok proposed =>
      simp only [checkObjects] at accepted
      split at accepted
      · cases accepted
      · exact accepted

private theorem scoped_apply_valid (objects : List ObjectSpec) (schema : Schema)
    (before after : State) (r : Request)
    (accepted : scopedApply objects schema before r = .ok after) : valid schema after = true := by
  unfold scopedApply at accepted
  split at accepted
  · cases accepted
  · split at accepted
    · cases accepted
    · split at accepted
      · cases accepted
      · dsimp only at accepted
        split at accepted
        · cases accepted
        · split at accepted
          · cases accepted
          · cases accepted
            simp_all [valid]

/-- One scoped write preserves both the item and ordinary-object invariants. -/
theorem accepted_scoped_valid (objects : List ObjectSpec) (schema : Schema)
    (before after : State) (r : Request)
    (accepted : writeScoped objects .full schema before r = .ok after) :
    valid schema after = true ∧ objectsValid objects after = true := by
  unfold writeScoped at accepted
  split at accepted
  · cases accepted
  · have objectGood := check_objects_valid objects _ after accepted
    have result := checked_proposal objects _ after accepted
    constructor
    · cases operation : r.operation with
      | apply =>
          apply scoped_apply_valid objects schema before after r
          simpa [operation] using result
      | update =>
          apply accepted_full_update_valid schema before after r
          simpa [operation,writeWith] using result
    · exact objectGood

def runScoped (objects : List ObjectSpec) (schema : Schema) : State → List Request → State
  | before,[] => before
  | before,r::rest => runScoped objects schema (commit before (writeScoped objects .full schema before r)) rest

/-- Every committed prefix is safe, including mixed Apply/Update and whole-request rejection. -/
theorem scoped_history_preserves_invariants (objects : List ObjectSpec) (schema : Schema)
    (before : State) (initial : valid schema before = true ∧ objectsValid objects before = true)
    (requests : List Request) :
    valid schema (runScoped objects schema before requests) = true ∧
    objectsValid objects (runScoped objects schema before requests) = true := by
  induction requests generalizing before with
  | nil => exact initial
  | cons r rest ih =>
      apply ih
      cases result : writeScoped objects .full schema before r with
      | error error => exact initial
      | ok after => exact accepted_scoped_valid objects schema before after r result

end SSA
