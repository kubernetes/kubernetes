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
import Main

open Lean Elab Command

private partial def sourceModules (directory : System.FilePath) (parent : Name) : IO (Array Name) := do
  let mut modules := #[]
  for entry in ← directory.readDir do
    if entry.fileName == ".lake" then continue
    if ← entry.path.isDir then
      modules := modules ++ (← sourceModules entry.path (parent.str entry.fileName))
    else if entry.path.extension == some "lean" then
      modules := modules.push (parent.str (entry.fileName.dropEnd 5).toString)
  return modules

/-- Audit definitions as well as theorems, so an unused axiom or a new theorem cannot escape. -/
elab "audit_ssa" : command => do
  let env ← getEnv
  let modules ← liftIO (sourceModules "." .anonymous)
  for moduleName in modules do
    unless moduleName == `SSA.Audit || env.header.moduleNames.contains moduleName do
      throwError "Project module {moduleName} is not imported by SSA.lean; its proofs are not audited."
  let modelModules := modules.filter (· != `SSA.Audit)
  let mut declarations : Nat := 0
  let mut theorems : Nat := 0
  for (name, info) in env.constants.toList do
    let included := match env.getModuleIdxFor? name with
      | some index => modelModules.contains env.header.moduleNames[index.toNat]!
      | none => env.mainModule != `SSA.Audit
    if included then
      for axiomName in ← collectAxioms name do
        unless #[`propext, `Quot.sound, `Classical.choice].contains axiomName do
          throwError "{name} depends on forbidden axiom {axiomName}."
      declarations := declarations + 1
      if info matches .thmInfo _ then theorems := theorems + 1
  if declarations == 0 || theorems == 0 then
    throwError "The SSA audit found no project declarations or theorems."
  logInfo m!"Axiom audit passed: {declarations} declarations, including {theorems} theorems."

audit_ssa

/-- Documentation reads the accepted environment; the proof graph is not guessed from tactic text. -/
elab "export_ssa_proofs" : command => do
  let env ← getEnv
  let modules ← liftIO (sourceModules "." .anonymous)
  let modelModules := modules.filter (· != `SSA.Audit)
  let mut declarations : Array Json := #[]
  for (name, info) in env.constants.toList do
    let some index := env.getModuleIdxFor? name | continue
    let moduleName := env.header.moduleNames[index.toNat]!
    unless modelModules.contains moduleName do continue
    let (kind, body) := match info with
      | .thmInfo value => ("theorem", some value.value)
      | .defnInfo value => ("definition", some value.value)
      | .opaqueInfo value => ("definition", some value.value)
      | .inductInfo _ => ("type", none)
      | _ => ("generated", none)
    let statement ← liftTermElabM do
      return toString (← Meta.ppExpr info.type)
    let used := info.type.getUsedConstants ++ (body.map Expr.getUsedConstants).getD #[]
    let mut dependencies : NameSet := {}
    for dependency in used do
      if let some dependencyIndex := env.getModuleIdxFor? dependency then
        if modelModules.contains env.header.moduleNames[dependencyIndex.toNat]! then
          dependencies := dependencies.insert (privateToUserName dependency)
    declarations := declarations.push (Json.mkObj [
      ("name", toJson (privateToUserName name).toString),
      ("module", toJson moduleName.toString),
      ("kind", toJson kind),
      ("private", toJson (isPrivateName name)),
      ("statement", toJson statement),
      ("dependencies", toJson ((dependencies.toArray.qsort Name.lt).map Name.toString)),
      ("axioms", toJson ((← collectAxioms name).map Name.toString))])
  let mut sources : Array Json := #[]
  for moduleName in modules do
    let path := moduleName.toString.replace "." "/" ++ ".lean"
    sources := sources.push (Json.mkObj [
      ("path", toJson path), ("source", toJson (← IO.FS.readFile path))])
  IO.FS.createDirAll ".lake"
  IO.FS.writeFile ".lake/proof-data.json" (Json.mkObj [
    ("toolchain", toJson (← IO.FS.readFile "lean-toolchain").trimAscii.toString),
    ("declarations", toJson declarations), ("sources", toJson sources)]).pretty
  logInfo "Exported accepted Lean declarations to .lake/proof-data.json."

export_ssa_proofs
