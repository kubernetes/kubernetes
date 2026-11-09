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
import SSA

open Lean SSA

private def allFixtures := SSA.Examples.fixtures ++ SSA.Counterexamples.fixtures ++
  SSA.ContainerCounterexamples.fixtures ++ SSA.HandoffCounterexamples.fixtures ++
  SSA.ResetCounterexamples.fixtures

private def report (fixture : Fixture) : Json :=
  Json.mkObj [
    ("name", toJson fixture.name),
    ("passed", toJson (passes fixture)),
    ("observations", toJson (runFixture fixture))]

private def checkFixtures (fixtures : List Fixture) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  for fixture in fixtures do
    let passed := passes fixture
    stdout.putStrLn (Json.mkObj [
      ("name", toJson fixture.name), ("passed", toJson passed)]).compress
    if !passed then
      stderr.putStrLn (report fixture).pretty
  return if fixtures.all passes then 0 else 1

private def usage : String :=
  "ssa-model --self-test | --list | --fixtures | --trace NAME | --check FILE | --replay FILE\n" ++
  "--check and --replay accept normalized fixture arrays; --element-algebra accepts operation/rank-list pairs. " ++
  "Use - to read stdin. --numeric-counterexamples, --numeric-codec-counterexamples, " ++
  "--granular-key-counterexamples and --numeric-omission-counterexamples report numeric witnesses. " ++
  "--updater-observations reports source ownership-update observations; --set-difference accepts recursive rank trees."

private structure ElementCase where
  operation : Option ElementAlgebra.Operation := none
  left : List Nat
  right : List Nat
  deriving FromJson

private def readInput (path : String) : IO String := do
  if path == "-" then (← IO.getStdin).readToEnd else IO.FS.readFile path

private def runElements (path : String) (override : Option ElementAlgebra.Operation) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let text ← readInput path
  match Json.parse text >>= fromJson? (α := List ElementCase) with
  | .error message =>
      stderr.putStrLn s!"Invalid element-set JSON: {message}"
      return 2
  | .ok cases =>
      if cases.isEmpty then
        stderr.putStrLn "Element-set input is empty; supply at least one pair."
        return 2
      let mut reports : List (List Nat) := []
      for input in cases do
        let operation := override.getD (input.operation.getD .difference)
        match ElementSubset.exec (fun a b => a < b) (input.left.length+input.right.length+19)
            (ElementAlgebra.program operation) {left := input.left,right := input.right} with
        | .ok (.returned [.set result]) => reports := result::reports
        | .error message =>
            stderr.putStrLn s!"Imported element-set execution failed: {message}"
            return 1
        | _ =>
            stderr.putStrLn "Imported element-set method did not return a set."
            return 1
      stdout.putStrLn (toJson reports.reverse).compress
      return 0
private def parseSetTree : Nat → Json → Except String (RecursiveSets.Tree Nat)
  | 0,_ => .error "Recursive set JSON exceeds the test protocol depth limit"
  | fuel+1,json => do
      let members ← json.getObjValAs? (List Nat) "members"
      let children ← json.getObjValAs? (List (Nat × Json)) "children"
      let parsed ← children.mapM fun (label,child) => do
        let tree ← parseSetTree fuel child
        pure (label,tree)
      pure (.node members parsed)

private structure SetCases where
  less : List (List Bool)
  pairs : List (Json × Json)
  deriving FromJson

private def runSets (path : String) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let input ← readInput path
  let computation : Except String (List (List (List Nat))) := do
    let cases ← Json.parse input >>= fromJson? (α := SetCases)
    if cases.pairs.isEmpty then throw "Recursive-set input is empty; supply at least one pair"
    cases.pairs.mapM fun (leftJson,rightJson) => do
      let left ← parseSetTree 128 leftJson
      let right ← parseSetTree 128 rightJson
      let ops := SetContracts.recursiveOps (fun a b : Nat => ((cases.less[a]?).bind (·[b]?)).getD false) RecursiveSets.empty
      match ← SetSubset.exec ops 2 SetSource.setDifferenceProgram {left,right,nodeMode := false} with
      | .returned [.tree result] =>
          match ← SetSubset.exec ops (left.children.length+right.children.length+19)
              SetSource.nodeDifferenceProgram {left,right} with
          | .returned [.nodes nodes] =>
              if nodes == result.children then pure (RecursiveSets.paths result)
              else throw "Recursive-set source callbacks disagree"
          | _ => throw "Child-node source did not return a node map"
      | _ => throw "Recursive-set source did not return a set"
  match computation with
  | .ok reports => stdout.putStrLn (toJson reports).compress; return 0
  | .error message => stderr.putStrLn message; return 1

def main (args : List String) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  match args with
  | [] | ["--self-test"] => checkFixtures allFixtures
  | ["--list"] =>
      for fixture in allFixtures do
        stdout.putStrLn fixture.name
      return 0
  | ["--fixtures"] =>
      stdout.putStrLn ("[\n" ++
        String.intercalate ",\n" (allFixtures.map (fun f => (toJson f).compress)) ++
        "\n]")
      return 0
  | ["--trace", name] =>
      match allFixtures.find? (fun fixture => fixture.name == name) with
      | some fixture =>
          stdout.putStrLn (report fixture).pretty
          return if passes fixture then 0 else 1
      | none =>
          stderr.putStrLn s!"Unknown scenario '{name}'; run --list to see available scenarios."
          return 2
  | ["--set-difference",path] => runSets path
  | ["--element-difference",path] => runElements path (some .difference)
  | ["--element-algebra",path] => runElements path none
  | ["--numeric-counterexamples"] =>
      match NumericCounterexamples.observe with
      | .ok observation =>
          stdout.putStrLn (toJson observation).compress
          return 0
      | .error message =>
          stderr.putStrLn message
          return 1
  | ["--numeric-codec-counterexamples"] =>
      stdout.putStrLn (toJson NumericCodecCounterexamples.observe).compress
      return 0
  | ["--granular-key-counterexamples"] =>
      stdout.putStrLn (toJson GranularKeyCounterexamples.observe).compress
      return 0
  | ["--numeric-omission-counterexamples"] =>
      stdout.putStrLn (toJson NumericOmissionCounterexamples.observe).compress
      return 0
  | ["--updater-observations"] =>
      stdout.putStrLn (toJson UpdaterExamples.observe).compress
      return 0
  | [mode, path] =>
      if mode != "--check" && mode != "--replay" then
        stderr.putStrLn usage
        return 2
      let text ← readInput path
      match Json.parse text >>= fromJson? (α := List Fixture) with
      | .ok fixtures =>
          if fixtures.isEmpty then
            stderr.putStrLn "Fixture input is empty; supply at least one scenario."
            return 2
          if mode == "--replay" then
            stdout.putStrLn (toJson (fixtures.map report)).compress
            return 0
          else
            checkFixtures fixtures
      | .error message =>
          stderr.putStrLn s!"Invalid fixture JSON: {message}"
          return 2
  | _ =>
      stderr.putStrLn usage
      return 2
