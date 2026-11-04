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
import SSA.Apply

namespace SSA

open Lean

structure Expected where
  error : Option ApplyError := none
  present : List Cell := []
  absent : List Path := []
  fullManagers : List (String × Path) := []
  partialManagers : List (String × Path) := []
  deriving Repr, FromJson, ToJson

structure FixtureStep where
  request : Request
  expected : Expected := {}
  deriving Repr, FromJson, ToJson

structure Fixture where
  name : String
  schema : Schema
  initial : State := {}
  steps : List FixtureStep
  deriving Repr, FromJson, ToJson

structure Observation where
  error : Option ApplyError
  state : State
  matched : Bool
  deriving Repr, FromJson, ToJson

def isFullAt (schema : Schema) (s : State) (manager : String) (path : Path) : Bool :=
  schema.any fun item => item.path == path && full s manager item

def observe (schema : Schema) (before : State) (step : FixtureStep) : Observation :=
  let result := apply schema before step.request
  let after := commit before result
  let error := match result with
    | .ok _ => none
    | .error e => some e
  let expected := step.expected
  { error, state := after
    matched := error == expected.error &&
      expected.present.all after.cells.contains &&
      expected.absent.all (fun p => !after.cells.any (fun c => c.path == p)) &&
      expected.fullManagers.all (fun (m, p) => isFullAt schema after m p) &&
      expected.partialManagers.all (fun (m, p) =>
        after.owns m p .item && !isFullAt schema after m p) }

def runSteps (schema : Schema) : State → List FixtureStep → List Observation
  | _, [] => []
  | before, step :: steps =>
      let observation := observe schema before step
      observation :: runSteps schema observation.state steps

def runFixture (fixture : Fixture) : List Observation :=
  runSteps fixture.schema fixture.initial fixture.steps

def passes (fixture : Fixture) : Bool :=
  (runFixture fixture).all (·.matched)

end SSA
