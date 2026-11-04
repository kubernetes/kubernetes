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
import SSA.Contracts
import SSA.Pruning

namespace SSA

/-- Three different contracts: released membership today, PR #336, and the proposed invariant. -/
inductive Semantics where
  | current
  | predrag
  | full
  deriving Repr, BEq, DecidableEq, Lean.FromJson, Lean.ToJson

deriving instance ReflBEq, LawfulBEq for Semantics

/-- The PR accepts an explicit map member before checking that map's required children. -/
def prCovers (s : State) (manager : String) : RequiredCore → Bool
  | .none => true
  | .value path => s.owns manager path .value
  | .object path children => s.owns manager path .presence ||
      (s.ownsMapPresence manager path && prCovers s manager children)
  | .collection path => s.owns manager path .presence
  | .both left right => prCovers s manager left && prCovers s manager right

/-- Key filtering affects only other applying managers, and only while restoring pruned data. -/
def retainsKey (mode : Semantics) (provisional : State) (actor : String)
    (item : ItemSpec) : Bool :=
  provisional.claims.any fun claim => claim.target == ⟨item.path, .item⟩ &&
    match mode with
    | .current => true
    | .predrag => claim.manager == actor || !claim.applied ||
        prCovers provisional claim.manager item.required
    | .full => full provisional claim.manager item

/-- The raw PR does not require the departing manager to have been full. -/
def deletesWith (mode : Semantics) (before provisional : State)
    (r : Request) (item : ItemSpec) : Bool :=
  match mode with
  | .full => deletesItem before provisional r item
  | .current | .predrag => before.hasItem item.path && before.owns r.manager item.path .item &&
      !r.desired.contains ⟨item.path, .item⟩ && !retainsKey mode provisional r.manager item

def rootsWith (mode : Semantics) (schema : Schema) (before provisional : State)
    (r : Request) : List Path :=
  (schema.filter (deletesWith mode before provisional r)).map (·.path)

def restoringClaims (mode : Semantics) (schema : Schema) (s : State) (actor : String) : FieldSet :=
  (s.claims.filter fun claim =>
    if claim.target.kind != .item then true
    else match mode with
      | .current | .full => true
      | .predrag => claim.manager == actor || !claim.applied ||
          schema.all (fun item => item.path != claim.target.path || prCovers s claim.manager item.required)
  ).map (·.target.path) ++ s.claims.flatMap (fun claim =>
    if claim.target.kind == .item then keyLeaves claim.target.path else [])

def wireConflicts (owners before after : State) (actor : String) : Bool :=
  owners.claims.any (fun claim => claim.manager != actor && after.cells.any
    (fun cell => cell.path == claim.target.path && modifiedCell before after cell))

def wireMerged (before : State) (r : Request) : State :=
  let data := merged before r
  { data with claims := before.claims.filter (fun claim => claim.manager != r.manager &&
      !(r.force && data.cells.any (fun cell => cell.path == claim.target.path && modifiedCell before data cell))) ++
      r.desired.map (fun target => ⟨r.manager,target,true⟩) }

def wirePruned (mode : Semantics) (schema : Schema) (before : State) (r : Request) : State :=
  let provisional := wireMerged before r
  pruneFieldSets r.named before provisional r (restoringClaims mode schema provisional r.manager)

def candidateWith (mode : Semantics) (schema : Schema) (before : State) (r : Request) : State :=
  match mode with
  | .full => candidate schema before r
  | .current | .predrag =>
      let pruned := wirePruned mode schema before r
      { pruned with claims := pruned.claims.filter (fun claim =>
          pruned.cells.any (fun cell => cell.path == claim.target.path) &&
          !(claim.manager != r.manager && pruned.cells.any
            (fun cell => cell.path == claim.target.path && modifiedCell before pruned cell))) }

/-- SMD does not run OpenAPI required validation, CEL, admission, or the full-owner guard. -/
def applyWith (mode : Semantics) (schema : Schema) (before : State)
    (r : Request) : Except ApplyError State :=
  match mode with
  | .full => apply schema before r
  | .current | .predrag =>
      if !shapeValid schema before then .error .invalidInitialState
      else if !normalized r then .error .invalidRequest
      else if (wireConflicts before before (wireMerged before r) r.manager ||
          wireConflicts (wireMerged before r) before (wirePruned mode schema before r) r.manager) && !r.force
        then .error .conflict
      else
        let after := candidateWith mode schema before r
        if !shapeValid schema after then .error .invalidResult
        else .ok after

/-- Required metadata is opt-in: absent metadata cannot change any retained membership. -/
theorem no_required_retains_same_key (s : State) (actor : String) (path : Path) :
    retainsKey .predrag s actor ⟨path, .none⟩ = retainsKey .current s actor ⟨path, .none⟩ := by
  simp [retainsKey, prCovers]

/-- This equality is quantified over all states and requests, not one key-only example. -/
theorem no_required_transition_equivalence (schema : Schema) (s : State) (r : Request)
    (metadata : ∀ item ∈ schema, item.required = .none) :
    applyWith .predrag schema s r = applyWith .current schema s r := by
  have same : restoringClaims .predrag schema (wireMerged s r) r.manager =
      restoringClaims .current schema (wireMerged s r) r.manager := by
    unfold restoringClaims
    congr 1
    congr 1
    apply List.filter_congr
    intro claim _
    have all : schema.all (fun item => item.path != claim.target.path ||
        prCovers (wireMerged s r) claim.manager item.required) = true := by
      apply List.all_eq_true.mpr
      intro item member
      simp [metadata item member,prCovers]
    simp [all]
  simp [applyWith, candidateWith, wirePruned, same]

/-- Empty-map ownership is a general classifier counterexample whenever required children are missing. -/
theorem explicit_map_shortcuts_required_children (s : State) (manager : String)
    (path : Path) (children : RequiredCore)
    (member : s.owns manager path .presence = true)
    (missing : covers s manager children = false) :
    prCovers s manager (.object path children) = true ∧
      covers s manager (.object path children) = false := by
  simp [prCovers, covers, member, missing]

/-- Required collection coverage asks for a marker which ordinary granular-list normalization omits. -/
theorem granular_list_without_member_is_partial (s : State) (manager : String) (path : Path)
    (noMember : s.owns manager path .presence = false) :
    prCovers s manager (.collection path) = false := by
  simp [prCovers, noMember]

/-- Update ownership is deliberately exempt from the PR's partial-manager filtering. -/
theorem updater_keeps_membership (s : State) (actor : String) (item : ItemSpec) (claim : Claim)
    (member : claim ∈ s.claims) (target : claim.target = ⟨item.path, .item⟩)
    (updater : claim.applied = false) : retainsKey .predrag s actor item = true := by
  exact List.any_eq_true.mpr ⟨claim, member, by simp [target, updater]⟩

/-- Full/partial is classified separately for each item, rather than once per request. -/
theorem current_key_only_last_claimant (before provisional : State) (r : Request) (path : Path) :
    deletesWith .predrag before provisional r ⟨path, .none⟩ =
      deletesWith .current before provisional r ⟨path, .none⟩ := by
  simp [deletesWith, no_required_retains_same_key]

end SSA
