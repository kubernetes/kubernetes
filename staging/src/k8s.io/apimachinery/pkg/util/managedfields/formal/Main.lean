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
  SSA.ContainerCounterexamples.fixtures ++ SSA.HandoffCounterexamples.fixtures

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
  "--check and --replay accept normalized fixture arrays; use - to read stdin."

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
  | [mode, path] =>
      if mode != "--check" && mode != "--replay" then
        stderr.putStrLn usage
        return 2
      let text ← if path == "-" then (← IO.getStdin).readToEnd else IO.FS.readFile path
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
