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
  "--updater-observations reports source ownership-update observations; --set-difference and --root-reset-filter accept recursive rank trees; --has-search and --search-midpoints check selected Go search source; --atomic-fields, --atomic-merge and --atomic-remove check typed atomic scope claims, whole replacement and removal/extraction; --atomic-remove-factory additionally runs the source caller; --schema-dispatch checks source atom deduction and handler selection; --atomic-remove-resolved composes both through the source caller."

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

private def runSets (path : String) (reset : Bool := false) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let input ← readInput path
  let computation : Except String (List (List (List Nat))) := do
    let cases ← Json.parse input >>= fromJson? (α := SetCases)
    if cases.pairs.isEmpty then throw "Recursive-set input is empty; supply at least one pair"
    cases.pairs.mapM fun (leftJson,rightJson) => do
      let left ← parseSetTree 128 leftJson
      let right ← parseSetTree 128 rightJson
      let less := fun a b : Nat => ((cases.less[a]?).bind (·[b]?)).getD false
      let base := SetContracts.recursiveOps less RecursiveSets.empty
      let search : SearchSubset.Ops Nat := {compare := fun a b => if less a b then -1 else if less b a then 1 else 0}
      if reset && !right.children.isEmpty then throw "Root reset protocol requires a flat exclusion set"
      if reset && right.members.length > SearchSubset.maxIndex search.wordBits then throw "Reset roots exceed the Go int bound"
      let ops := if reset then SetFilterContracts.flatOps (FilterHasBridge.setOps search RecursiveSets.empty) else base
      let program := if reset then SetSource.excludeFilterProgram else SetSource.setDifferenceProgram
      let nodeProgram := if reset then SetSource.recursiveNodeProgram else SetSource.nodeDifferenceProgram
      let nodeFuel := if reset then left.children.length+14 else left.children.length+right.children.length+19
      match ← SetSubset.exec ops 2 program {left,right,nodeMode := false} with
      | .returned [.tree result] =>
          match ← SetSubset.exec ops nodeFuel nodeProgram {left,right} with
          | .returned [.nodes nodes] =>
              if nodes == result.children then pure (RecursiveSets.paths result)
              else throw "Recursive-set source callbacks disagree"
          | _ => throw "Child-node source did not return a node map"
      | _ => throw "Recursive-set source did not return a set"
  match computation with
  | .ok reports => stdout.putStrLn (toJson reports).compress; return 0
  | .error message => stderr.putStrLn message; return 1

private structure SearchCase where
  members : List Nat
  target : Nat
  deriving FromJson

private structure SearchCases where
  compare : List (List Int)
  wordBits : Nat := 64
  cases : List SearchCase
  deriving FromJson

private structure SearchReport where
  index : Nat
  found : Bool
  has : Bool
  deriving ToJson

private structure SearchOutput where
  results : List SearchReport
  less : List (List Bool)
  deriving ToJson

private def runSearch (path : String) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let input ← readInput path
  let computation : Except String SearchOutput := do
    let cases ← Json.parse input >>= fromJson? (α := SearchCases)
    if cases.cases.isEmpty || cases.compare.isEmpty then throw "Search input requires a comparator matrix and nonempty cases"
    if cases.wordBits != 32 && cases.wordBits != 64 then throw "Search protocol supports Go word widths 32 and 64"
    let size := cases.compare.length
    if cases.compare.any (fun row => row.length != size) then throw "Search comparator matrix must be square"
    let compare := fun a b : Nat => ((cases.compare[a]?).bind (·[b]?)).getD 0
    let ops : SearchSubset.Ops Nat := {compare,wordBits := cases.wordBits}
    let results ← cases.cases.mapM fun request => do
      if request.target ≥ size || request.members.any (fun label => label ≥ size) then throw "Search label has no comparator matrix entry"
      let (index,found) ← HasBridge.sourceSearch ops request.members request.target
      match ← SearchSubset.exec (HasBridge.withSourceSearch ops) 3 HasSource.hasProgram
          {members := request.members,target := request.target,other := request.target} with
      | .returned [.flag has] =>
          if has != found then throw "Has source and standard search source disagree"
          pure ({index,found,has} : SearchReport)
      | _ => throw "Has source did not return a boolean"
    let less ← (List.range size).mapM fun a => (List.range size).mapM fun b => do
      match ← SearchSubset.exec ops 2 HasSource.lessProgram {members := [],target := a,other := b} with
      | .returned [.flag flag] => pure flag
      | _ => throw "Less source did not return a boolean"
    pure {results,less}
  match computation with
  | .ok reports => stdout.putStrLn (toJson reports).compress; return 0
  | .error message => stderr.putStrLn message; return 1

private structure MidpointCase where
  bits : Nat
  left : Nat
  right : Nat
  deriving FromJson

private structure AtomicFieldCase where
  kind : String
  resource : Option Nat
  path : List Nat
  fields : List (List Nat)
  deriving FromJson

private structure AtomicFieldReport where
  fields : List (List Nat)
  freed : List Nat
  deriving ToJson

private structure AtomicMergeCase where
  kind : String
  lhs : Option String
  rhs : Option String
  output : Option String := none
  inLeaf : Bool := false
  deriving FromJson

private structure AtomicMergeReport where
  output : Option String
  inLeaf : Bool
  ruleCalls : Nat
  freed : List Nat
  deriving ToJson

private def mergeResource (kind : AtomicMergeMachine.Kind) (_ : String) (raw : Option String) : Option AtomicMergeMachine.Resource := do
  let source ← raw
  let parsed ← (Json.parse source).toOption
  let length ← match parsed with
    | .arr values => if kind == .list then some values.size else none
    | .obj fields => if kind == .map then some fields.size else none
    | _ => none
  some {handle := length+1,length}

private def runAtomicMerge (path : String) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let input ← readInput path
  let computation : Except String (List AtomicMergeReport) := do
    let cases ← Json.parse input >>= fromJson? (α := List AtomicMergeCase)
    if cases.isEmpty then throw "Atomic merge protocol requires nonempty cases"
    cases.mapM fun request => do
      let kind ← match request.kind with
        | "list" => .ok AtomicFieldSubset.Kind.list
        | "map" => .ok AtomicFieldSubset.Kind.map
        | _ => .error "Atomic merge kind must be list or map"
      for encoded in [request.lhs,request.rhs,request.output] do
        match encoded with
        | none => pure ()
        | some raw => discard (Json.parse raw)
      let ops : AtomicMergeMachine.Ops String := {deref := mergeResource}
      let initial : AtomicMergeMachine.Machine String := {
        walker := {left := request.lhs,right := request.rhs,output := request.output,inLeaf := request.inLeaf}}
      match ← AtomicMergeSubset.invoke (AtomicMergeContracts.linkedOps ops)
          (AtomicMergeContracts.program kind) false initial with
      | ([.nil],final) =>
          pure {
            output := final.walker.output
            inLeaf := final.walker.inLeaf
            ruleCalls := (final.events.filter (fun event => event == .rule)).length
            freed := final.events.filterMap (fun event => match event with | .free handle => some handle | _ => none)}
      | _ => throw "Atomic merge source returned non-nil validation errors"
  match computation with
  | .ok reports => stdout.putStrLn (toJson reports).compress; return 0
  | .error message => stderr.putStrLn message; return 1

private structure AtomicRemoveCase where
  kind : String
  value : String
  output : Option String := none
  extract : Bool
  shape : Bool
  size : Option Nat
  rawNil : Bool
  deriving FromJson

private structure AtomicRemoveReport where
  output : Option String
  freed : List (Option Nat)
  deriving ToJson

private def runAtomicRemove (path : String) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let input ← readInput path
  let computation : Except String (List AtomicRemoveReport) := do
    let cases ← Json.parse input >>= fromJson? (α := List AtomicRemoveCase)
    if cases.isEmpty then throw "Atomic removal protocol requires nonempty cases"
    cases.mapM fun request => do
      discard (Json.parse request.value)
      match request.output with
      | none => pure ()
      | some raw => discard (Json.parse raw)
      let ops : AtomicRemoveSubset.Ops String := {
        isKind := fun _ _ => request.shape
        asContainer := fun _ _ => request.size.map (fun length => {handle := length+1,length})
        unstructured := fun value => if request.rawNil then none else some value}
      let initial : AtomicRemoveSubset.Machine String := {
        input := request.value,output := request.output,extract := request.extract}
      let program ← match request.kind with
        | "scalar" => .ok AtomicRemoveSource.scalarProgram
        | "map" => .ok AtomicRemoveSource.mapProgram
        | "list" => .ok AtomicRemoveSource.listProgram
        | _ => .error "Atomic removal kind must be scalar, map or list"
      let final ← AtomicRemoveSubset.invoke ops program initial
      pure {output := final.output,freed := final.freed}
  match computation with
  | .ok reports => stdout.putStrLn (toJson reports).compress; return 0
  | .error message => stderr.putStrLn message; return 1

private structure RemoveFactoryCase where
  request : AtomicRemoveCase
  resolverError : Bool

private instance : FromJson RemoveFactoryCase where
  fromJson? json := do
    pure {request := ← fromJson? json,resolverError := ← json.getObjValAs? Bool "resolverError"}

private def runAtomicRemoveFactory (path : String) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let input ← readInput path
  let computation : Except String (List String) := do
    let cases ← Json.parse input >>= fromJson? (α := List RemoveFactoryCase)
    if cases.isEmpty then throw "Removal factory protocol requires nonempty cases"
    cases.mapM fun test => do
      let request := test.request
      discard (Json.parse request.value)
      let kind ← match request.kind with
        | "scalar" => .ok none
        | "map" => .ok (some AtomicFieldSubset.Kind.map)
        | "list" => .ok (some AtomicFieldSubset.Kind.list)
        | _ => .error "Removal factory kind must be scalar, map or list"
      let typedOps : AtomicRemoveSubset.Ops String := {
        isKind := fun _ _ => request.shape
        asContainer := fun _ _ => request.size.map (fun length => {handle := length+1,length})
        unstructured := fun value => if request.rawNil then none else some value}
      let sourceRequest : RemoveFactorySubset.Request String Unit Unit Unit := {
        input := request.value,schema := (),fields := (),typeRef := (),extract := request.extract}
      let wrap := fun value : Option String => value.getD "null"
      if test.resolverError then
        RemoveFactorySubset.invoke {resolve := fun _ _ _ walker => .ok (walker,true),wrap} sourceRequest
      else RemoveFactorySubset.invoke (RemoveFactoryBridge.linkedOps typedOps kind wrap) sourceRequest
  match computation with
  | .ok reports => stdout.putStrLn (toJson reports).compress; return 0
  | .error message => stderr.putStrLn message; return 1

private structure DispatchAtom where
  map : Option String
  scalar : Option String
  list : Option String
  deriving FromJson, ToJson

private structure DispatchCase where
  operation : String
  atom : DispatchAtom
  value : Option String
  kinds : Array Bool
  named : Option String
  found : Bool
  callbackError : Bool
  initial : List String
  deriving FromJson

private structure DispatchReport where
  atom : Option DispatchAtom
  error : String
  calls : List String
  deriving ToJson

private def dispatchKindIndex : SchemaDispatchMachine.Kind → Nat
  | .floating => 0
  | .integer => 1
  | .string => 2
  | .boolean => 3
  | .list => 4
  | .map => 5

private def runSchemaDispatch (path : String) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let input ← readInput path
  let computation : Except String (List DispatchReport) := do
    let cases ← Json.parse input >>= fromJson? (α := List DispatchCase)
    if cases.isEmpty then throw "Schema dispatch protocol requires nonempty cases"
    cases.mapM fun request => do
      if request.kinds.size != 6 then throw "Schema dispatch requires six value-kind flags"
      match request.value with
      | none => pure ()
      | some raw => discard (Json.parse raw)
      let input : SchemaDispatchMachine.Atom String String String := {
        map := request.atom.map,scalar := request.atom.scalar,list := request.atom.list}
      let reference : SchemaDispatchMachine.Reference Unit := {original := (),named := request.named}
      let callback := fun (kind pointer : String) (calls : List String) =>
        let entry := kind ++ "=" ++ pointer
        Except.ok ((if request.callbackError then entry ++ ": callback" else ""),calls ++ [entry])
      let ops : SchemaDispatchMachine.Ops String String String String Unit (List String) String := {
        resolve := fun _ => (input,request.found)
        isKind := fun kind _ => request.kinds[dispatchKindIndex kind]!
        errorf := fun format name => format.replace "%v" name
        nilErrors := ""
        map := callback "map"
        scalar := callback "scalar"
        list := callback "list"}
      match request.operation with
      | "deduce" =>
          let (after,calls) ← SchemaDispatchSubset.invokeDeduce ops input request.value request.initial
          pure {atom := some {map := after.map,scalar := after.scalar,list := after.list},error := "",calls}
      | "handle" =>
          let (error,calls) ← SchemaDispatchSubset.invokeHandle ops input reference request.initial
          pure {atom := none,error,calls}
      | "resolve" =>
          let (error,calls) ← SchemaDispatchSubset.invokeResolve (SchemaDispatchResolver.linkedOps ops)
            reference request.value request.initial
          pure {atom := none,error,calls}
      | _ => throw "Schema operation must be deduce, handle or resolve"
  match computation with
  | .ok reports => stdout.putStrLn (toJson reports).compress; return 0
  | .error message => stderr.putStrLn message; return 1

private structure ResolvedFactoryCase where
  request : AtomicRemoveCase
  mode : String
  kinds : Array Bool

private instance : FromJson ResolvedFactoryCase where
  fromJson? json := do
    pure {
      request := ← fromJson? json
      mode := ← json.getObjValAs? String "resolverMode"
      kinds := ← json.getObjValAs? (Array Bool) "kinds"}

private def runResolvedFactory (path : String) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let input ← readInput path
  let computation : Except String (List String) := do
    let cases ← Json.parse input >>= fromJson? (α := List ResolvedFactoryCase)
    if cases.isEmpty then throw "Resolved factory protocol requires nonempty cases"
    cases.mapM fun test => do
      let request := test.request
      discard (Json.parse request.value)
      if test.kinds.size != 6 then throw "Resolved factory requires six value-kind flags"
      let typed : AtomicRemoveSubset.Ops String := {
        isKind := fun kind _ => test.kinds[if kind == .map then 5 else 4]!
        asContainer := fun _ _ => request.size.map (fun length => {handle := length+1,length})
        unstructured := fun value => if request.rawNil then none else some value}
      let atom : SchemaDispatchMachine.Atom Bool Unit Bool ← match request.kind with
        | "map" => .ok {map := some true}
        | "list" => .ok {list := some true}
        | "scalar" => .ok {scalar := some ()}
        | _ => .error "Resolved factory kind must be scalar, map or list"
      let lookup ← match test.mode with
        | "valid" => .ok (atom,true)
        | "missing-type" => .ok ({},false)
        | "invalid-atom" => .ok ({},true)
        | _ => .error "Resolved factory mode must be valid, missing-type or invalid-atom"
      let reference : SchemaDispatchMachine.Reference Unit := {
        original := (),named := some (if test.mode == "missing-type" then "missing" else "unit")}
      let sourceRequest : RemoveFactorySubset.Request String Unit Unit (SchemaDispatchMachine.Reference Unit) := {
        input := request.value,schema := (),fields := (),typeRef := reference,extract := request.extract}
      let ops := SchemaDispatchFactoryBridge.linkedFactoryOps typed (fun _ _ => lookup)
        (fun kind _ => test.kinds[dispatchKindIndex kind]!)
        (fun format name => format.replace "%v" name) "" (fun value => value.getD "null")
      RemoveFactorySubset.invoke ops sourceRequest
  match computation with
  | .ok reports => stdout.putStrLn (toJson reports).compress; return 0
  | .error message => stderr.putStrLn message; return 1

private def runAtomicFields (path : String) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let input ← readInput path
  let computation : Except String (List AtomicFieldReport) := do
    let cases ← Json.parse input >>= fromJson? (α := List AtomicFieldCase)
    if cases.isEmpty then throw "Atomic field-set protocol requires nonempty cases"
    cases.mapM fun request => do
      let kind ← match request.kind with
        | "list" => .ok AtomicFieldSubset.Kind.list
        | "map" => .ok AtomicFieldSubset.Kind.map
        | _ => .error "Atomic field-set kind must be list or map"
      let ops : AtomicFieldSubset.Ops Unit := {view := fun _ _ => request.resource}
      match ← AtomicFieldSubset.run ops (AtomicFieldContracts.program kind)
          {input := (),path := request.path,fields := request.fields} with
      | .returned [.nil] final => pure {fields := final.fields,freed := final.freed}
      | _ => throw "Atomic field-set source did not return nil validation errors"
  match computation with
  | .ok reports => stdout.putStrLn (toJson reports).compress; return 0
  | .error message => stderr.putStrLn message; return 1

private def runMidpoints (path : String) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let input ← readInput path
  let computation : Except String (List Nat) := do
    let cases ← Json.parse input >>= fromJson? (α := List MidpointCase)
    if cases.isEmpty then throw "Midpoint input is empty"
    cases.mapM fun request => do
      if request.bits != 32 && request.bits != 64 then throw "Midpoint protocol supports Go word widths 32 and 64"
      if request.left > SearchSubset.maxIndex request.bits || request.right > SearchSubset.maxIndex request.bits then
        throw "Midpoint bounds exceed the Go int range"
      pure (SearchSubset.unsignedMidpoint request.bits request.left request.right)
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
  | ["--has-search",path] => runSearch path
  | ["--atomic-fields",path] => runAtomicFields path
  | ["--atomic-merge",path] => runAtomicMerge path
  | ["--atomic-remove",path] => runAtomicRemove path
  | ["--atomic-remove-factory",path] => runAtomicRemoveFactory path
  | ["--schema-dispatch",path] => runSchemaDispatch path
  | ["--atomic-remove-resolved",path] => runResolvedFactory path
  | ["--search-midpoints",path] => runMidpoints path
  | ["--root-reset-filter",path] => runSets path true
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
