/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.UpdaterSubset

namespace SSA.UpdaterContracts

open GoSubset (Expr Stmt ManagedFields)
open UpdaterSubset

set_option maxRecDepth 10000
set_option maxHeartbeats 4000000

/-- This is exact path membership, independent of Go's mixed numeric comparator. -/
def Owns (owners : ManagedFields) (actor : String) (path : Path) : Prop :=
  ∃ entry, (actor,entry) ∈ owners ∧ path ∈ entry.fields

def ClaimsBounded (original current : ManagedFields) : Prop :=
  ∀ actor path, Owns current actor path → Owns original actor path

theorem claims_bounded_refl (owners : ManagedFields) : ClaimsBounded owners owners :=
  fun _ _ owned => owned

theorem claims_bounded_trans (first second third : ManagedFields)
    (left : ClaimsBounded first second) (right : ClaimsBounded second third) :
    ClaimsBounded first third := fun actor path owned => left actor path (right actor path owned)

theorem deleting_claims_is_bounded (original current : ManagedFields) (actor : String)
    (bounded : ClaimsBounded original current) :
    ClaimsBounded original (current.filter (fun row => row.1 != actor)) := by
  intro owner path owned
  obtain ⟨entry,present,claimed⟩ := owned
  exact bounded owner path ⟨entry,(List.mem_filter.mp present).1,claimed⟩

theorem replacing_with_old_claims_is_bounded (original current : ManagedFields)
    (actor : String) (entry : GoSubset.ManagerFields) (bounded : ClaimsBounded original current)
    (oldClaims : ∀ path ∈ entry.fields, Owns current actor path) :
    ClaimsBounded original (GoSubset.managerSet current actor entry) := by
  intro owner path owned
  obtain ⟨selected,present,claimed⟩ := owned
  simp only [GoSubset.managerSet,List.mem_cons] at present
  rcases present with same | prior
  · obtain ⟨rfl,rfl⟩ := Prod.mk.inj same
    exact bounded _ path (oldClaims path claimed)
  · exact bounded owner path ⟨selected,(List.mem_filter.mp prior).1,claimed⟩

def protectedName (tracked : String) (name : String) : Bool :=
  [tracked,"nil","fieldpath"].contains name

/-- The only permitted write to the tracked ownership map subtracts from its
    current entry. The comparison and filter computations are otherwise unrestricted. -/
def subtractsCurrent (tracked keyName : String) : Expr → Bool
  | .call (.var "fieldpath") "NewVersionedSet"
      [.call (.call (.index (.var source) (.var sourceKey)) "Set" []) "Difference" [_],_,_] =>
      source == tracked && sourceKey == keyName
  | _ => false

def mapAssignmentSafe (tracked name : String) (key value : Expr) : Bool :=
  if name == tracked then match key with
    | .var keyName => subtractsCurrent tracked keyName value
    | _ => false
  else !protectedName tracked name

def safeProgram (tracked : String) (program : List Stmt) : Bool :=
  match program with
  | [] => true
  | statement::rest =>
      (match statement with
      | .assign names _ => !names.any (protectedName tracked)
      | .declareNil name | .declareError name => !protectedName tracked name
      | .ret ((.var name)::_) => name == tracked || name == "nil"
      | .mapSet name key value => mapAssignmentSafe tracked name key value
      | .mapDelete name _ => name == tracked || !protectedName tracked name
      | .branch _ body => safeProgram tracked body
      | .branchElse _ body alternative => safeProgram tracked body && safeProgram tracked alternative
      | .rangeMap key value _ body =>
          !protectedName tracked key && !protectedName tracked value && safeProgram tracked body
      | .continueLoop => true
      | _ => false) && safeProgram tracked rest
termination_by sizeOf program

/-- Source-shape premise discharged for the general semantic claim-bound invariant below. -/
theorem imported_update_has_only_subtractive_claim_writes :
    safeProgram "managers" UpdaterSource.updateProgram = true := by
  simp [safeProgram,UpdaterSource.updateProgram,protectedName,subtractsCurrent,mapAssignmentSafe]

theorem heap_write_reads_back (heap : Heap) (address : Nat) (comparison : Comparison) :
    heapRead (heapWrite heap address comparison) address = .ok comparison := by
  simp [heapRead,heapWrite]

/-- Both cached and local comparison references observe the same mutated object. -/
theorem equal_references_observe_same_filter (heap : Heap) (localAddress cached : Nat)
    (same : localAddress = cached) (comparison : Comparison) :
    heapRead (heapWrite heap cached comparison) localAddress = .ok comparison := by
  subst localAddress
  exact heap_write_reads_back heap cached comparison

theorem filter_method_mutates_shared_reference {Obj Err : Type} (ops : Ops Obj Err)
    (heap : Heap) (address : Nat) (comparison : Comparison) (filter : FieldSet → FieldSet)
    (existing : heapRead heap address = .ok comparison) :
    call ops (.reference address) "FilterFields" [.filter filter] heap =
      .ok ([.reference address],heapWrite heap address (transform filter comparison)) := by
  simp [call,existing,Bind.bind,Except.bind]

theorem filtered_reference_exposes_modified_fields {Obj Err : Type} (ops : Ops Obj Err)
    (heap : Heap) (address : Nat) (comparison : Comparison) (filter : FieldSet → FieldSet) :
    select ops (.reference address) "Modified" (heapWrite heap address (transform filter comparison)) =
      .ok (.fields (filter comparison.modified)) := by
  simp [select,heap_write_reads_back,transform,Bind.bind,Except.bind]

theorem expression_preserves_environment {Obj Err : Type} (ops : Ops Obj Err)
    (fuel : Nat) (expression : Expr) (before after : Machine Obj Err) (values : List (Value Obj Err))
    (accepted : runEval ops fuel expression before = .ok (values,after)) :
    after.env = before.env := by
  unfold runEval at accepted
  cases evaluated : eval ops before.env fuel expression before.heap with
  | error message => simp [evaluated,Bind.bind,Except.bind] at accepted
  | ok result =>
      rcases result with ⟨actual,heap⟩
      simp [evaluated,Bind.bind,Except.bind] at accepted
      obtain ⟨_,rfl⟩ := accepted
      rfl

theorem arguments_preserve_environment {Obj Err : Type} (ops : Ops Obj Err)
    (fuel : Nat) (expressions : List Expr) (before after : Machine Obj Err) (values : List (Value Obj Err))
    (accepted : runAll ops fuel expressions before = .ok (values,after)) :
    after.env = before.env := by
  unfold runAll at accepted
  cases evaluated : evalAll ops before.env fuel expressions before.heap with
  | error message => simp [evaluated,Bind.bind,Except.bind] at accepted
  | ok result =>
      rcases result with ⟨actual,heap⟩
      simp [evaluated,Bind.bind,Except.bind] at accepted
      obtain ⟨_,rfl⟩ := accepted
      rfl

def DifferenceShrinks {Obj Err : Type} (ops : Ops Obj Err) : Prop :=
  ∀ left right path, path ∈ ops.primitives.difference left right → path ∈ left

def Invariant {Obj Err : Type} (original : ManagedFields) (tracked : String) (state : Machine Obj Err) : Prop :=
  read state "nil" = .nil ∧ read state "fieldpath" = .fieldpathPackage ∧
    ∃ current, read state tracked = .managed current ∧ ClaimsBounded original current

def ResultBound {Obj Err : Type} (original : ManagedFields) (tracked : String) : Result Obj Err → Prop
  | .next state | .continued state => Invariant original tracked state
  | .returned values state => Invariant original tracked state ∧
      ∀ owners, values.head? = some (.managed owners) → ClaimsBounded original owners

theorem read_write_same {Obj Err : Type} (state : Machine Obj Err) (name : String) (value : Value Obj Err) :
    read (write state name value) name = value := by
  simp [UpdaterSubset.read,write,UpdaterSubset.lookup]

theorem read_write_other {Obj Err : Type} (state : Machine Obj Err) (name query : String)
    (value : Value Obj Err) (different : name ≠ query) :
    read (write state name value) query = read state query := by
  simp [UpdaterSubset.read,write,UpdaterSubset.lookup,different]

theorem invariant_of_same_environment {Obj Err : Type} (original : ManagedFields) (tracked : String)
    (before after : Machine Obj Err) (same : after.env = before.env)
    (valid : Invariant original tracked before) : Invariant original tracked after := by
  simpa only [Invariant,UpdaterSubset.read,same] using valid

theorem eval_variable {Obj Err : Type} (ops : Ops Obj Err) (env : Env Obj Err)
    (fuel : Nat) (name : String) (heap : Heap) :
    eval ops env (fuel+1) (.var name) heap = .ok ([UpdaterSubset.lookup env name],heap) := by
  rw [eval]

theorem eval_call {Obj Err : Type} (ops : Ops Obj Err) (env : Env Obj Err)
    (fuel : Nat) (receiver : Expr) (method : String) (args : List Expr) (heap : Heap) :
    eval ops env (fuel+1) (.call receiver method args) heap = (do
      let (values,next) ← eval ops env fuel receiver heap
      match values with
      | [receiver] =>
          let (arguments,next) ← evalAll ops env fuel args next
          call ops receiver method arguments next
      | _ => .error "Go updater call requires one receiver") := by
  rw [eval]
  simp only [Bind.bind,Except.bind]
  repeat' first | rfl | split

theorem eval_index {Obj Err : Type} (ops : Ops Obj Err) (env : Env Obj Err)
    (fuel : Nat) (receiver key : Expr) (heap : Heap) :
    eval ops env (fuel+1) (.index receiver key) heap = (do
      let (maps,next) ← eval ops env fuel receiver heap
      let (keys,next) ← eval ops env fuel key next
      match maps,keys with
      | [map],[.text key] =>
          let (value,_) ← mapIndex map key
          .ok ([value],next)
      | _,_ => .error "Go updater indexing requires a map and text key") := by
  rw [eval]
  simp only [Bind.bind,Except.bind]
  repeat' first | rfl | split

theorem eval_all_empty {Obj Err : Type} (ops : Ops Obj Err) (env : Env Obj Err)
    (fuel : Nat) (heap : Heap) :
    evalAll ops env (fuel+1) [] heap = .ok ([],heap) := by rw [evalAll]

theorem eval_all_cons {Obj Err : Type} (ops : Ops Obj Err) (env : Env Obj Err)
    (fuel : Nat) (expression : Expr) (rest : List Expr) (heap : Heap) :
    evalAll ops env (fuel+1) (expression::rest) heap = (do
      let (values,next) ← eval ops env fuel expression heap
      let (remaining,next) ← evalAll ops env fuel rest next
      .ok (values ++ remaining,next)) := by rw [evalAll]

def currentFields (tracked keyName : String) : Expr :=
  .call (.index (.var tracked) (.var keyName)) "Set" []

theorem eval_current_fields {Obj Err : Type} (ops : Ops Obj Err) (env : Env Obj Err)
    (fuel : Nat) (tracked keyName actor : String) (owners : ManagedFields) (entry : GoSubset.ManagerFields)
    (heap : Heap) (mapValue : UpdaterSubset.lookup env tracked = .managed owners)
    (keyValue : UpdaterSubset.lookup env keyName = .text actor)
    (selected : GoSubset.managerLookup owners actor = some entry) :
    eval ops env (fuel+6) (currentFields tracked keyName) heap = .ok ([.fields entry.fields],heap) := by
  simp [currentFields,eval_call,eval_index,eval_variable,eval_all_empty,mapValue,keyValue,
    selected,mapIndex,call,Bind.bind,Except.bind]

theorem difference_call_has_single_field_result {Obj Err : Type} (ops : Ops Obj Err)
    (left : FieldSet) (arguments values : List (Value Obj Err)) (before after : Heap)
    (accepted : call ops (.fields left) "Difference" arguments before = .ok (values,after)) :
    ∃ right, arguments = [.fields right] ∧
      values = [.fields (ops.primitives.difference left right)] ∧ after = before := by
  cases arguments with
  | nil => simp [call] at accepted
  | cons first rest =>
      cases rest with
      | cons second rest => simp [call] at accepted
      | nil =>
          cases first <;> simp [call] at accepted
          case fields right =>
            obtain ⟨rfl,rfl⟩ := accepted
            exact ⟨right,rfl,rfl,rfl⟩

theorem eval_difference_of_current_fields {Obj Err : Type} (ops : Ops Obj Err)
    (shrinks : DifferenceShrinks ops) (env : Env Obj Err) (fuel : Nat)
    (tracked keyName actor : String) (owners : ManagedFields) (entry : GoSubset.ManagerFields)
    (excluded : Expr) (before after : Heap) (values : List (Value Obj Err))
    (mapValue : UpdaterSubset.lookup env tracked = .managed owners)
    (keyValue : UpdaterSubset.lookup env keyName = .text actor)
    (selected : GoSubset.managerLookup owners actor = some entry)
    (accepted : eval ops env (fuel+10) (.call (currentFields tracked keyName) "Difference" [excluded]) before =
      .ok (values,after)) :
    ∃ fields, values = [.fields fields] ∧ ∀ path ∈ fields, path ∈ entry.fields := by
  have getter : eval ops env (fuel+9) (currentFields tracked keyName) before =
      .ok ([.fields entry.fields],before) := by
    simpa only [Nat.add_assoc] using
      eval_current_fields ops env (fuel+3) tracked keyName actor owners entry before mapValue keyValue selected
  rw [eval_call,getter] at accepted
  simp only [Bind.bind,Except.bind] at accepted
  cases evaluatedArguments : evalAll ops env (fuel+9) [excluded] before with
  | error message => simp [evaluatedArguments] at accepted
  | ok result =>
      rcases result with ⟨actualArguments,middle⟩
      have primitive : call ops (.fields entry.fields) "Difference" actualArguments middle = .ok (values,after) := by
        simpa only [evaluatedArguments] using accepted
      obtain ⟨right,_,same,_⟩ := difference_call_has_single_field_result ops entry.fields actualArguments values middle after primitive
      exact ⟨ops.primitives.difference entry.fields right,same,fun path present => shrinks _ _ path present⟩

theorem new_versioned_set_keeps_first_fields {Obj Err : Type} (ops : Ops Obj Err)
    (fields : FieldSet) (rest values : List (Value Obj Err)) (before after : Heap)
    (accepted : call ops .fieldpathPackage "NewVersionedSet" (.fields fields::rest) before = .ok (values,after)) :
    ∃ version applied, values = [.manager ⟨fields,version,applied⟩] ∧ after = before := by
  cases rest with
  | nil => simp [call] at accepted
  | cons second rest =>
      cases rest with
      | nil => simp [call] at accepted
      | cons third rest =>
          cases rest with
          | cons fourth rest => simp [call] at accepted
          | nil =>
              cases second <;> simp [call] at accepted
              case text version =>
                cases third <;> simp at accepted
                case flag applied =>
                  obtain ⟨rfl,rfl⟩ := accepted
                  exact ⟨version,applied,rfl,rfl⟩

def subtractExpression (tracked keyName : String) (excluded version applied : Expr) : Expr :=
  .call (.var "fieldpath") "NewVersionedSet"
    [.call (currentFields tracked keyName) "Difference" [excluded],version,applied]

/-- The actual source write expression can only construct claims from its current entry. -/
theorem eval_subtractive_write_cannot_add {Obj Err : Type} (ops : Ops Obj Err)
    (shrinks : DifferenceShrinks ops) (env : Env Obj Err) (fuel : Nat)
    (tracked keyName actor : String) (owners : ManagedFields) (entry : GoSubset.ManagerFields)
    (excluded version applied : Expr) (before after : Heap) (values : List (Value Obj Err))
    (packageValue : UpdaterSubset.lookup env "fieldpath" = .fieldpathPackage)
    (mapValue : UpdaterSubset.lookup env tracked = .managed owners)
    (keyValue : UpdaterSubset.lookup env keyName = .text actor)
    (selected : GoSubset.managerLookup owners actor = some entry)
    (accepted : eval ops env (fuel+14) (subtractExpression tracked keyName excluded version applied) before =
      .ok (values,after)) :
    ∃ newer, values = [.manager newer] ∧ ∀ path ∈ newer.fields, path ∈ entry.fields := by
  unfold subtractExpression at accepted
  rw [eval_call,eval_variable] at accepted
  simp only [packageValue,Bind.bind,Except.bind] at accepted
  rw [eval_all_cons] at accepted
  cases evaluatedFields : eval ops env (fuel+12)
      (.call (currentFields tracked keyName) "Difference" [excluded]) before with
  | error message => simp [evaluatedFields,Bind.bind,Except.bind] at accepted
  | ok result =>
      rcases result with ⟨fieldValues,middle⟩
      have differenceAccepted : eval ops env ((fuel+2)+10)
          (.call (currentFields tracked keyName) "Difference" [excluded]) before = .ok (fieldValues,middle) := by
        simpa only [Nat.add_assoc] using evaluatedFields
      obtain ⟨fields,fieldValuesEq,fieldsBound⟩ := eval_difference_of_current_fields ops shrinks env (fuel+2)
        tracked keyName actor owners entry excluded before middle fieldValues mapValue keyValue selected differenceAccepted
      subst fieldValues
      cases evaluatedRest : evalAll ops env (fuel+12) [version,applied] middle with
      | error message => simp [evaluatedFields,evaluatedRest,Bind.bind,Except.bind] at accepted
      | ok result =>
          rcases result with ⟨remaining,atCall⟩
          have constructed : call ops .fieldpathPackage "NewVersionedSet" (.fields fields::remaining) atCall =
              .ok (values,after) := by
            simpa only [evaluatedFields,evaluatedRest,Bind.bind,Except.bind,List.singleton_append] using accepted
          obtain ⟨newVersion,newApplied,same,_⟩ := new_versioned_set_keeps_first_fields ops fields remaining values atCall after constructed
          exact ⟨⟨fields,newVersion,newApplied⟩,same,fieldsBound⟩

theorem manager_lookup_is_member (owners : ManagedFields) (actor : String) (entry : GoSubset.ManagerFields)
    (selected : GoSubset.managerLookup owners actor = some entry) : (actor,entry) ∈ owners := by
  induction owners with
  | nil => simp [GoSubset.managerLookup] at selected
  | cons row rest ih =>
      rcases row with ⟨name,prior⟩
      by_cases same : actor = name
      · subst name
        have equal : prior = entry := by simpa [GoSubset.managerLookup] using selected
        subst prior
        exact List.mem_cons_self
      · have tail : GoSubset.managerLookup rest actor = some entry := by
          simpa [GoSubset.managerLookup,same] using selected
        exact List.mem_cons_of_mem _ (ih tail)

theorem subtractive_shape (tracked keyName : String) (expression : Expr)
    (checked : subtractsCurrent tracked keyName expression = true) :
    ∃ excluded version applied, expression = subtractExpression tracked keyName excluded version applied := by
  unfold subtractsCurrent at checked
  split at checked
  · simp only [Bool.and_eq_true,beq_iff_eq] at checked
    obtain ⟨rfl,rfl⟩ := checked
    exact ⟨_,_,_,rfl⟩
  · cases checked

theorem invariant_write_unprotected {Obj Err : Type} (original : ManagedFields) (tracked : String)
    (state : Machine Obj Err) (name : String) (value : Value Obj Err)
    (unprotected : protectedName tracked name = false) (valid : Invariant original tracked state) :
    Invariant original tracked (write state name value) := by
  have notTracked : name ≠ tracked := by
    intro same
    subst name
    simp [protectedName] at unprotected
  have notNil : name ≠ "nil" := by
    intro same
    subst name
    simp [protectedName] at unprotected
  have notPackage : name ≠ "fieldpath" := by
    intro same
    subst name
    simp [protectedName] at unprotected
  rcases valid with ⟨nilValue,packageValue,current,mapValue,bounded⟩
  refine ⟨?_,?_,current,?_,bounded⟩
  · exact (read_write_other state name "nil" value notNil).trans nilValue
  · exact (read_write_other state name "fieldpath" value notPackage).trans packageValue
  · exact (read_write_other state name tracked value notTracked).trans mapValue

theorem invariant_bind_names {Obj Err : Type} (original : ManagedFields) (tracked : String)
    (names : List String) (values : List (Value Obj Err)) (before after : Machine Obj Err)
    (unprotected : names.any (protectedName tracked) = false) (valid : Invariant original tracked before)
    (accepted : bindNames names values before = .ok after) : Invariant original tracked after := by
  induction names generalizing values before with
  | nil =>
      cases values with
      | nil =>
          have same : before = after := Except.ok.inj accepted
          subst after
          exact valid
      | cons value rest => simp [bindNames] at accepted
  | cons name names ih =>
      have checks : protectedName tracked name = false ∧ names.any (protectedName tracked) = false := by
        cases head : protectedName tracked name <;> cases tail : names.any (protectedName tracked) <;> simp_all
      cases values with
      | nil => simp [bindNames] at accepted
      | cons value values =>
          by_cases ignored : name = "_"
          · subst name
            apply ih values before checks.2 valid
            simpa only [bindNames] using accepted
          · have tail : bindNames names values (write before name value) = .ok after := by
              simpa [bindNames,ignored] using accepted
            exact ih values (write before name value) checks.2
              (invariant_write_unprotected original tracked before name value checks.1 valid) tail

theorem successful_subtractive_write_has_current_entry {Obj Err : Type} (ops : Ops Obj Err)
    (env : Env Obj Err) (fuel : Nat) (tracked keyName actor : String) (owners : ManagedFields)
    (excluded version applied : Expr) (before after : Heap) (values : List (Value Obj Err))
    (packageValue : UpdaterSubset.lookup env "fieldpath" = .fieldpathPackage)
    (mapValue : UpdaterSubset.lookup env tracked = .managed owners)
    (keyValue : UpdaterSubset.lookup env keyName = .text actor)
    (accepted : eval ops env (fuel+14) (subtractExpression tracked keyName excluded version applied) before =
      .ok (values,after)) : ∃ entry, GoSubset.managerLookup owners actor = some entry := by
  cases selected : GoSubset.managerLookup owners actor with
  | some entry => exact ⟨entry,rfl⟩
  | none =>
      simp [subtractExpression,currentFields,eval_call,eval_index,eval_variable,eval_all_cons,eval_all_empty,
        packageValue,mapValue,keyValue,selected,mapIndex,call,Bind.bind,Except.bind] at accepted

theorem run_eval_has_raw_result {Obj Err : Type} (ops : Ops Obj Err) (fuel : Nat)
    (expression : Expr) (before after : Machine Obj Err) (values : List (Value Obj Err))
    (accepted : runEval ops fuel expression before = .ok (values,after)) :
    eval ops before.env fuel expression before.heap = .ok (values,after.heap) := by
  unfold runEval at accepted
  cases evaluated : eval ops before.env fuel expression before.heap with
  | error message => simp [evaluated,Bind.bind,Except.bind] at accepted
  | ok result =>
      rcases result with ⟨actual,heap⟩
      simp [evaluated,Bind.bind,Except.bind] at accepted
      obtain ⟨rfl,rfl⟩ := accepted
      rfl

theorem invariant_tracked_not_nil {Obj Err : Type} (original : ManagedFields) (tracked : String)
    (state : Machine Obj Err) (valid : Invariant original tracked state) : tracked ≠ "nil" := by
  intro same
  subst tracked
  obtain ⟨nilValue,_,current,mapValue,_⟩ := valid
  have impossible := nilValue.symm.trans mapValue
  cases impossible

theorem invariant_tracked_not_package {Obj Err : Type} (original : ManagedFields) (tracked : String)
    (state : Machine Obj Err) (valid : Invariant original tracked state) : tracked ≠ "fieldpath" := by
  intro same
  subst tracked
  obtain ⟨_,packageValue,current,mapValue,_⟩ := valid
  have impossible := packageValue.symm.trans mapValue
  cases impossible

theorem invariant_store_subtraction {Obj Err : Type} (ops : Ops Obj Err) (shrinks : DifferenceShrinks ops)
    (original : ManagedFields) (tracked keyName actor : String) (expression : Expr)
    (before middle : Machine Obj Err) (newer : GoSubset.ManagerFields) (updated : Value Obj Err)
    (valid : Invariant original tracked before) (shape : subtractsCurrent tracked keyName expression = true)
    (keyValue : read before keyName = .text actor)
    (evaluated : runEval ops 128 expression before = .ok ([.manager newer],middle))
    (stored : mapSet (read middle tracked) (.text actor) (.manager newer) = .ok updated) :
    Invariant original tracked (write middle tracked updated) := by
  have notNil := invariant_tracked_not_nil original tracked before valid
  have notPackage := invariant_tracked_not_package original tracked before valid
  have unchanged := expression_preserves_environment ops 128 expression before middle [.manager newer] evaluated
  obtain ⟨excluded,version,applied,rfl⟩ := subtractive_shape tracked keyName expression shape
  rcases valid with ⟨nilValue,packageValue,current,mapValue,bounded⟩
  have raw := run_eval_has_raw_result ops 128
    (subtractExpression tracked keyName excluded version applied) before middle [.manager newer] evaluated
  obtain ⟨prior,selected⟩ := successful_subtractive_write_has_current_entry ops before.env 114
    tracked keyName actor current excluded version applied before.heap middle.heap [.manager newer]
    packageValue mapValue keyValue raw
  obtain ⟨returned,returnedEq,fieldsBound⟩ := eval_subtractive_write_cannot_add ops shrinks before.env 114
    tracked keyName actor current prior excluded version applied before.heap middle.heap [.manager newer]
    packageValue mapValue keyValue selected raw
  have sameEntry : newer = returned := by simpa using returnedEq
  subst returned
  have mapAfter : read middle tracked = .managed current := by
    simpa only [UpdaterSubset.read,unchanged] using mapValue
  have storedEq : .managed (GoSubset.managerSet current actor newer) = updated := by
    simpa only [mapAfter,mapSet,Except.ok.injEq] using stored
  subst updated
  refine ⟨?_,?_,GoSubset.managerSet current actor newer,read_write_same _ _ _,?_⟩
  · rw [read_write_other middle tracked "nil" _ notNil]
    simpa only [UpdaterSubset.read,unchanged] using nilValue
  · rw [read_write_other middle tracked "fieldpath" _ notPackage]
    simpa only [UpdaterSubset.read,unchanged] using packageValue
  · apply replacing_with_old_claims_is_bounded original current actor newer bounded
    intro path present
    exact ⟨prior,manager_lookup_is_member current actor prior selected,fieldsBound path present⟩

theorem invariant_store_bounded_map {Obj Err : Type} (original owners : ManagedFields) (tracked : String)
    (state : Machine Obj Err) (valid : Invariant original tracked state) (bounded : ClaimsBounded original owners) :
    Invariant original tracked (write state tracked (.managed owners)) := by
  have notNil := invariant_tracked_not_nil original tracked state valid
  have notPackage := invariant_tracked_not_package original tracked state valid
  rcases valid with ⟨nilValue,packageValue,_⟩
  refine ⟨?_,?_,owners,read_write_same _ _ _,bounded⟩
  · exact (read_write_other state tracked "nil" (.managed owners) notNil).trans nilValue
  · exact (read_write_other state tracked "fieldpath" (.managed owners) notPackage).trans packageValue

theorem invariant_store_deletion {Obj Err : Type} (original : ManagedFields) (tracked : String)
    (state : Machine Obj Err) (key updated : Value Obj Err) (valid : Invariant original tracked state)
    (deleted : mapDelete (read state tracked) key = .ok updated) :
    Invariant original tracked (write state tracked updated) := by
  have saved := valid
  rcases valid with ⟨_,_,current,mapValue,bounded⟩
  cases key <;> simp [mapValue,mapDelete] at deleted
  case text actor =>
    subst updated
    exact invariant_store_bounded_map original _ tracked state saved
      (deleting_claims_is_bounded original current actor bounded)

theorem run_eval_variable {Obj Err : Type} (ops : Ops Obj Err) (fuel : Nat) (name : String)
    (state : Machine Obj Err) : runEval ops (fuel+1) (.var name) state = .ok ([read state name],state) := by
  simp [runEval,eval_variable,UpdaterSubset.read,Bind.bind,Except.bind]

theorem managed_map_assignment_shape {Obj Err : Type} (owners : ManagedFields)
    (key value updated : Value Obj Err) (accepted : mapSet (.managed owners) key value = .ok updated) :
    ∃ actor entry, key = .text actor ∧ value = .manager entry ∧
      updated = .managed (GoSubset.managerSet owners actor entry) := by
  cases key <;> simp [mapSet] at accepted
  case text actor =>
    cases value <;> simp at accepted
    case manager entry =>
      subst updated
      exact ⟨actor,entry,rfl,rfl,rfl⟩

theorem run_all_has_raw_result {Obj Err : Type} (ops : Ops Obj Err) (fuel : Nat)
    (expressions : List Expr) (before after : Machine Obj Err) (values : List (Value Obj Err))
    (accepted : runAll ops fuel expressions before = .ok (values,after)) :
    evalAll ops before.env fuel expressions before.heap = .ok (values,after.heap) := by
  unfold runAll at accepted
  cases evaluated : evalAll ops before.env fuel expressions before.heap with
  | error message => simp [evaluated,Bind.bind,Except.bind] at accepted
  | ok result =>
      rcases result with ⟨actual,heap⟩
      simp [evaluated,Bind.bind,Except.bind] at accepted
      obtain ⟨rfl,rfl⟩ := accepted
      rfl

theorem evaluated_first_variable_is_head {Obj Err : Type} (ops : Ops Obj Err) (env : Env Obj Err)
    (fuel : Nat) (name : String) (rest : List Expr) (before after : Heap) (values : List (Value Obj Err))
    (accepted : evalAll ops env (fuel+3) (.var name::rest) before = .ok (values,after)) :
    values.head? = some (UpdaterSubset.lookup env name) := by
  rw [eval_all_cons,eval_variable] at accepted
  simp only [Bind.bind,Except.bind] at accepted
  cases evaluated : evalAll ops env (fuel+2) rest before with
  | error message => simp [evaluated] at accepted
  | ok result =>
      rcases result with ⟨remaining,heap⟩
      simp [evaluated] at accepted
      obtain ⟨rfl,rfl⟩ := accepted
      rfl

theorem safe_return_is_bounded {Obj Err : Type} (ops : Ops Obj Err) (original : ManagedFields)
    (tracked name : String) (rest : List Expr) (before after : Machine Obj Err) (values : List (Value Obj Err))
    (valid : Invariant original tracked before) (allowed : (name == tracked || name == "nil") = true)
    (accepted : runAll ops 128 (.var name::rest) before = .ok (values,after)) :
    ResultBound original tracked (.returned values after) := by
  have raw := run_all_has_raw_result ops 128 (.var name::rest) before after values accepted
  have head := evaluated_first_variable_is_head ops before.env 125 name rest before.heap after.heap values raw
  have unchanged := arguments_preserve_environment ops 128 (.var name::rest) before after values accepted
  refine ⟨invariant_of_same_environment original tracked before after unchanged valid,?_⟩
  intro owners returned
  have value : read before name = .managed owners := Option.some.inj (head.symm.trans returned)
  have choices : name = tracked ∨ name = "nil" := by
    simpa only [Bool.or_eq_true,beq_iff_eq] using allowed
  rcases choices with rfl | rfl
  · rcases valid with ⟨_,_,current,mapValue,bounded⟩
    have same : current = owners := Value.managed.inj (mapValue.symm.trans value)
    subst owners
    exact bounded
  · have impossible := valid.1.symm.trans value
    cases impossible

theorem range_preserves_invariant {Obj Err : Type} (ops : Ops Obj Err) (original : ManagedFields)
    (tracked : String) (fuel : Nat) (entries : ManagedFields) (key value : String) (body : List Stmt)
    (before : Machine Obj Err) (result : Result Obj Err)
    (keySafe : protectedName tracked key = false) (valueSafe : protectedName tracked value = false)
    (bodySound : ∀ state outcome, Invariant original tracked state →
      exec ops fuel body state = .ok outcome → ResultBound original tracked outcome)
    (valid : Invariant original tracked before)
    (accepted : execRange ops fuel entries key value body before = .ok result) :
    ResultBound original tracked result := by
  induction entries generalizing before with
  | nil =>
      have same : .next before = result := by
        simpa only [execRange,Except.ok.injEq] using accepted
      subst result
      exact valid
  | cons entry rest ih =>
      rw [execRange] at accepted
      cases binding : bindNames [key,value] [.text entry.1,.manager entry.2] before with
      | error message => simp [binding,Bind.bind,Except.bind] at accepted
      | ok iteration =>
          have namesSafe : [key,value].any (protectedName tracked) = false := by simp [keySafe,valueSafe]
          have iterationValid := invariant_bind_names original tracked [key,value] [.text entry.1,.manager entry.2]
            before iteration namesSafe valid binding
          cases executing : exec ops fuel body iteration with
          | error message => simp [binding,executing,Bind.bind,Except.bind] at accepted
          | ok step =>
              have stepBound := bodySound iteration step iterationValid executing
              cases step with
              | returned values final =>
                  have same : .returned values final = result := by
                    simpa only [binding,executing,Bind.bind,Except.bind,Except.ok.injEq] using accepted
                  subst result
                  exact stepBound
              | next final =>
                  apply ih final stepBound
                  simpa only [binding,executing,Bind.bind,Except.bind] using accepted
              | continued final =>
                  apply ih final stepBound
                  simpa only [binding,executing,Bind.bind,Except.bind] using accepted

theorem map_assignment_preserves_invariant {Obj Err : Type} (ops : Ops Obj Err)
    (shrinks : DifferenceShrinks ops) (original : ManagedFields) (tracked name : String)
    (key expression : Expr) (rest : List Stmt) (fuel : Nat) (before : Machine Obj Err) (result : Result Obj Err)
    (checked : safeProgram tracked [.mapSet name key expression] = true)
    (continuation : ∀ state, Invariant original tracked state →
      exec ops fuel rest state = .ok result → ResultBound original tracked result)
    (valid : Invariant original tracked before)
    (accepted : exec ops (fuel+1) (.mapSet name key expression::rest) before = .ok result) :
    ResultBound original tracked result := by
  have nodeSafe : mapAssignmentSafe tracked name key expression = true := by
    simpa [safeProgram] using checked
  rw [exec] at accepted
  cases computedKey : runEval ops 128 key before with
  | error message => simp [computedKey,Bind.bind,Except.bind] at accepted
  | ok keyResult =>
      rcases keyResult with ⟨keys,keyState⟩
      have keyValid := invariant_of_same_environment original tracked before keyState
        (expression_preserves_environment ops 128 key before keyState keys computedKey) valid
      cases computedValue : runEval ops 128 expression keyState with
      | error message => simp [computedKey,computedValue,Bind.bind,Except.bind] at accepted
      | ok valueResult =>
          rcases valueResult with ⟨values,valueState⟩
          have valueValid := invariant_of_same_environment original tracked keyState valueState
            (expression_preserves_environment ops 128 expression keyState valueState values computedValue) keyValid
          cases keys with
          | nil => simp [computedKey,computedValue,Bind.bind,Except.bind] at accepted
          | cons returnedKey remainingKeys =>
              cases remainingKeys with
              | cons second remainingKeys => simp [computedKey,computedValue,Bind.bind,Except.bind] at accepted
              | nil =>
                  cases values with
                  | nil => simp [computedKey,computedValue,Bind.bind,Except.bind] at accepted
                  | cons returnedValue remainingValues =>
                      cases remainingValues with
                      | cons second remainingValues => simp [computedKey,computedValue,Bind.bind,Except.bind] at accepted
                      | nil =>
                          cases stored : mapSet (read valueState name) returnedKey returnedValue with
                          | error message => simp [computedKey,computedValue,stored,Bind.bind,Except.bind] at accepted
                          | ok updated =>
                              have storeValid : Invariant original tracked (write valueState name updated) := by
                                by_cases target : name = tracked
                                · subst name
                                  cases key <;> simp [mapAssignmentSafe] at nodeSafe
                                  case var keyName =>
                                    have keyShape : ([read before keyName],before) = ([returnedKey],keyState) := by
                                      simpa only [run_eval_variable,Except.ok.injEq] using computedKey
                                    obtain ⟨keyValues,sameState⟩ := Prod.mk.inj keyShape
                                    subst keyState
                                    obtain ⟨_,_,current,mapValue,_⟩ := valueValid
                                    have typedStore : mapSet (.managed current) returnedKey returnedValue = .ok updated := by
                                      simpa only [mapValue] using stored
                                    obtain ⟨actor,newer,keyEq,valueEq,_⟩ := managed_map_assignment_shape current
                                      returnedKey returnedValue updated typedStore
                                    subst returnedKey
                                    subst returnedValue
                                    have keyRead : read before keyName = .text actor := by simpa using keyValues
                                    exact invariant_store_subtraction ops shrinks original tracked keyName actor expression
                                      before valueState newer updated valid nodeSafe keyRead computedValue stored
                                · have nameSafe : protectedName tracked name = false := by simpa [mapAssignmentSafe,target] using nodeSafe
                                  exact invariant_write_unprotected original tracked valueState name updated nameSafe valueValid
                              apply continuation (write valueState name updated) storeValid
                              simpa only [computedKey,computedValue,stored,Bind.bind,Except.bind] using accepted
theorem map_deletion_preserves_invariant {Obj Err : Type} (ops : Ops Obj Err)
    (original : ManagedFields) (tracked name : String) (key : Expr) (rest : List Stmt)
    (fuel : Nat) (before : Machine Obj Err) (result : Result Obj Err)
    (checked : (name == tracked || !protectedName tracked name) = true)
    (continuation : ∀ state, Invariant original tracked state →
      exec ops fuel rest state = .ok result → ResultBound original tracked result)
    (valid : Invariant original tracked before)
    (accepted : exec ops (fuel+1) (.mapDelete name key::rest) before = .ok result) :
    ResultBound original tracked result := by
  rw [exec] at accepted
  cases computed : runEval ops 128 key before with
  | error message => simp [computed,Bind.bind,Except.bind] at accepted
  | ok valueResult =>
      rcases valueResult with ⟨values,middle⟩
      have middleValid := invariant_of_same_environment original tracked before middle
        (expression_preserves_environment ops 128 key before middle values computed) valid
      cases values with
      | nil => simp [computed,Bind.bind,Except.bind] at accepted
      | cons value remaining =>
          cases remaining with
          | cons second remaining => simp [computed,Bind.bind,Except.bind] at accepted
          | nil =>
              cases deleted : mapDelete (read middle name) value with
              | error message => simp [computed,deleted,Bind.bind,Except.bind] at accepted
              | ok updated =>
                  have storeValid : Invariant original tracked (write middle name updated) := by
                    by_cases target : name = tracked
                    · subst name
                      exact invariant_store_deletion original tracked middle value updated middleValid deleted
                    · have nameSafe : protectedName tracked name = false := by simpa [target] using checked
                      exact invariant_write_unprotected original tracked middle name updated nameSafe middleValid
                  apply continuation (write middle name updated) storeValid
                  simpa only [computed,deleted,Bind.bind,Except.bind] using accepted

theorem safe_program_preserves_claim_bound {Obj Err : Type} (ops : Ops Obj Err)
    (shrinks : DifferenceShrinks ops) (original : ManagedFields) (tracked : String)
    (fuel : Nat) (program : List Stmt) (before : Machine Obj Err) (result : Result Obj Err)
    (checked : safeProgram tracked program = true) (valid : Invariant original tracked before)
    (accepted : exec ops fuel program before = .ok result) : ResultBound original tracked result := by
  induction fuel generalizing program before result with
  | zero => simp [exec] at accepted
  | succ fuel ih =>
      cases program with
      | nil =>
          have same : .next before = result := by simpa only [exec,Except.ok.injEq] using accepted
          subst result
          exact valid
      | cons statement rest =>
          rw [safeProgram.eq_def] at checked
          simp only [Bool.and_eq_true] at checked
          have nodeSafe := checked.1
          have restSafe := checked.2
          have restSound : ∀ state, Invariant original tracked state →
              exec ops fuel rest state = .ok result → ResultBound original tracked result := by
            intro state stateValid executed
            exact ih rest state result restSafe stateValid executed
          cases statement with
          | assign names expressions =>
              have namesSafe : names.any (protectedName tracked) = false := by simpa using nodeSafe
              rw [exec] at accepted
              cases computed : runAll ops 128 expressions before with
              | error message => simp [computed,Bind.bind,Except.bind] at accepted
              | ok evaluation =>
                  rcases evaluation with ⟨values,middle⟩
                  have middleValid := invariant_of_same_environment original tracked before middle
                    (arguments_preserve_environment ops 128 expressions before middle values computed) valid
                  cases assigned : bindNames names values middle with
                  | error message => simp [computed,assigned,Bind.bind,Except.bind] at accepted
                  | ok final =>
                      apply restSound final (invariant_bind_names original tracked names values middle final namesSafe middleValid assigned)
                      simpa only [computed,assigned,Bind.bind,Except.bind] using accepted
          | declareNil name =>
              have nameSafe : protectedName tracked name = false := by simpa using nodeSafe
              rw [exec] at accepted
              exact restSound _ (invariant_write_unprotected original tracked before name .nil nameSafe valid) accepted
          | declareError name =>
              have nameSafe : protectedName tracked name = false := by simpa using nodeSafe
              rw [exec] at accepted
              exact restSound _ (invariant_write_unprotected original tracked before name .nil nameSafe valid) accepted
          | ret expressions =>
              cases expressions with
              | nil => simp at nodeSafe
              | cons first remaining =>
                  cases first <;> simp at nodeSafe
                  case var name =>
                    rw [exec] at accepted
                    cases computed : runAll ops 128 (.var name::remaining) before with
                    | error message => simp [computed,Bind.bind,Except.bind] at accepted
                    | ok evaluation =>
                        rcases evaluation with ⟨values,final⟩
                        have same : .returned values final = result := by
                          simpa only [computed,Bind.bind,Except.bind,Except.ok.injEq] using accepted
                        subst result
                        have allowed : (name == tracked || name == "nil") = true := by simpa using nodeSafe
                        exact safe_return_is_bounded ops original tracked name remaining before final values valid allowed computed
          | continueLoop =>
              have same : .continued before = result := by simpa only [exec,Except.ok.injEq] using accepted
              subst result
              exact valid
          | mapSet name key expression =>
              have single : safeProgram tracked [.mapSet name key expression] = true := by simpa [safeProgram] using nodeSafe
              exact map_assignment_preserves_invariant ops shrinks original tracked name key expression rest fuel
                before result single restSound valid accepted
          | mapDelete name key =>
              exact map_deletion_preserves_invariant ops original tracked name key rest fuel
                before result nodeSafe restSound valid accepted
          | branch condition body =>
              rw [exec] at accepted
              cases computed : runEval ops 128 condition before with
              | error message => simp [computed,Bind.bind,Except.bind] at accepted
              | ok evaluation =>
                  rcases evaluation with ⟨values,middle⟩
                  have middleValid := invariant_of_same_environment original tracked before middle
                    (expression_preserves_environment ops 128 condition before middle values computed) valid
                  cases values with
                  | nil => simp [computed,Bind.bind,Except.bind] at accepted
                  | cons value remaining =>
                      cases remaining with
                      | cons second remaining => simp [computed,Bind.bind,Except.bind] at accepted
                      | nil =>
                          cases value <;> try (simp [computed,Bind.bind,Except.bind] at accepted)
                          case flag flag =>
                            cases flag with
                            | false =>
                                apply restSound middle middleValid
                                simpa only [computed,Bind.bind,Except.bind] using accepted
                            | true =>
                                cases executed : exec ops fuel body middle with
                                | error message => simp [executed] at accepted
                                | ok outcome =>
                                    have bodyBound := ih body middle outcome nodeSafe middleValid executed
                                    cases outcome with
                                    | returned values final =>
                                        have same : .returned values final = result := by
                                          simpa only [computed,executed,Bind.bind,Except.bind,Except.ok.injEq] using accepted
                                        subst result
                                        exact bodyBound
                                    | continued final =>
                                        have same : .continued final = result := by
                                          simpa only [computed,executed,Bind.bind,Except.bind,Except.ok.injEq] using accepted
                                        subst result
                                        exact bodyBound
                                    | next final =>
                                        apply restSound final bodyBound
                                        simpa only [computed,executed,Bind.bind,Except.bind] using accepted
          | branchElse condition body alternative =>
              have branches : safeProgram tracked body = true ∧ safeProgram tracked alternative = true := by
                simpa only [Bool.and_eq_true] using nodeSafe
              rw [exec] at accepted
              cases computed : runEval ops 128 condition before with
              | error message => simp [computed,Bind.bind,Except.bind] at accepted
              | ok evaluation =>
                  rcases evaluation with ⟨values,middle⟩
                  have middleValid := invariant_of_same_environment original tracked before middle
                    (expression_preserves_environment ops 128 condition before middle values computed) valid
                  cases values with
                  | nil => simp [computed,Bind.bind,Except.bind] at accepted
                  | cons value remaining =>
                      cases remaining with
                      | cons second remaining => simp [computed,Bind.bind,Except.bind] at accepted
                      | nil =>
                          cases value <;> try (simp [computed,Bind.bind,Except.bind] at accepted)
                          case flag flag =>
                            have branchSafe : safeProgram tracked (if flag then body else alternative) = true := by
                              cases flag <;> simp [branches.1,branches.2]
                            cases executed : exec ops fuel (if flag then body else alternative) middle with
                            | error message => simp [executed] at accepted
                            | ok outcome =>
                                have bodyBound := ih (if flag then body else alternative) middle outcome branchSafe middleValid executed
                                cases outcome with
                                | returned values final =>
                                    have same : .returned values final = result := by
                                      simpa only [computed,executed,Bind.bind,Except.bind,Except.ok.injEq] using accepted
                                    subst result
                                    exact bodyBound
                                | continued final =>
                                    have same : .continued final = result := by
                                      simpa only [computed,executed,Bind.bind,Except.bind,Except.ok.injEq] using accepted
                                    subst result
                                    exact bodyBound
                                | next final =>
                                    apply restSound final bodyBound
                                    simpa only [computed,executed,Bind.bind,Except.bind] using accepted
          | rangeMap key value collection body =>
              have checks : (protectedName tracked key = false ∧ protectedName tracked value = false) ∧ safeProgram tracked body = true := by
                cases keyFlag : protectedName tracked key <;> cases valueFlag : protectedName tracked value <;> simp_all
              rw [exec] at accepted
              cases computed : runEval ops 128 collection before with
              | error message => simp [computed,Bind.bind,Except.bind] at accepted
              | ok evaluation =>
                  rcases evaluation with ⟨values,middle⟩
                  have middleValid := invariant_of_same_environment original tracked before middle
                    (expression_preserves_environment ops 128 collection before middle values computed) valid
                  cases values with
                  | nil => simp [computed,Bind.bind,Except.bind] at accepted
                  | cons value remaining =>
                      cases remaining with
                      | cons second remaining => simp [computed,Bind.bind,Except.bind] at accepted
                      | nil =>
                          cases value <;> try (simp [computed,Bind.bind,Except.bind] at accepted)
                          case managed entries =>
                            cases executed : execRange ops fuel (ops.iteration.choose key entries) key value body middle with
                            | error message => simp [executed] at accepted
                            | ok outcome =>
                                have bodySound : ∀ state answer, Invariant original tracked state →
                                    exec ops fuel body state = .ok answer → ResultBound original tracked answer := by
                                  intro state answer stateValid execution
                                  exact ih body state answer checks.2 stateValid execution
                                have rangeBound := range_preserves_invariant ops original tracked fuel
                                  (ops.iteration.choose key entries) key value body middle outcome checks.1.1 checks.1.2 bodySound middleValid executed
                                cases outcome with
                                | returned values final =>
                                    have same : .returned values final = result := by
                                      simpa only [computed,executed,Bind.bind,Except.bind,Except.ok.injEq] using accepted
                                    subst result
                                    exact rangeBound
                                | continued final => simp [executed] at accepted
                                | next final =>
                                    apply restSound final rangeBound
                                    simpa only [computed,executed,Bind.bind,Except.bind] using accepted
          | rangeSlice _ _ _ => simp at nodeSafe
          | whileLoop _ _ => simp at nodeSafe
          | increment _ => simp at nodeSafe
          | fieldSet _ _ _ => simp at nodeSafe
          | effect _ => simp at nodeSafe
          | deferCall _ => simp at nodeSafe

theorem initial_ownership_is_bounded {Obj Err : Type} (old new : Obj) (version : String)
    (owners : ManagedFields) (actor : String) (force : Bool) :
    Invariant owners "managers" (initial (Err := Err) old new version owners actor force) := by
  exact ⟨rfl,rfl,owners,rfl,claims_bounded_refl owners⟩

/-- Every successful imported ownership update only retains original exact paths.
    Typed comparison, conversion, filtering and map schedules may be arbitrary;
    the remaining field-set premise is that Difference selects paths from its left input. -/
theorem imported_update_never_adds_claims {Obj Err : Type} (ops : Ops Obj Err)
    (shrinks : DifferenceShrinks ops) (old new : Obj) (version : String) (owners updated : ManagedFields)
    (actor : String) (force : Bool) (fuel : Nat) (tail : List (Value Obj Err)) (after : Machine Obj Err)
    (accepted : exec ops fuel UpdaterSource.updateProgram (initial old new version owners actor force) =
      .ok (.returned (.managed updated::tail) after)) : ClaimsBounded owners updated := by
  have bounded := safe_program_preserves_claim_bound ops shrinks owners "managers" fuel UpdaterSource.updateProgram
    (initial old new version owners actor force) (.returned (.managed updated::tail) after)
    imported_update_has_only_subtractive_claim_writes
    (initial_ownership_is_bounded old new version owners actor force) accepted
  exact bounded.2 updated rfl

/-- New actor claims were filtered before update; source execution cannot reintroduce an excluded prefix. -/
theorem imported_update_preserves_prefix_exclusion {Obj Err : Type} (ops : Ops Obj Err)
    (shrinks : DifferenceShrinks ops) (old new : Obj) (version : String) (owners updated : ManagedFields)
    (actor excludedActor : String) (force : Bool) (fuel : Nat) (tail : List (Value Obj Err))
    (after : Machine Obj Err) (root : Path)
    (excluded : ∀ path, under root path = true → ¬Owns owners excludedActor path)
    (accepted : exec ops fuel UpdaterSource.updateProgram (initial old new version owners actor force) =
      .ok (.returned (.managed updated::tail) after)) :
    ∀ path, under root path = true → ¬Owns updated excludedActor path := by
  intro path below owned
  exact excluded path below
    (imported_update_never_adds_claims ops shrinks old new version owners updated actor force fuel tail after accepted
      excludedActor path owned)

end SSA.UpdaterContracts
