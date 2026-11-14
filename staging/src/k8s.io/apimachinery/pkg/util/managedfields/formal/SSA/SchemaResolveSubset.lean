/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SchemaResolveMachine

namespace SSA.SchemaResolveSubset

variable {M S L R Rel P : Type}

open SchemaResolveMachine RawGoSyntax
open AtomicRemoveSubset (atom nodes)

mutual
  def eval (ops : Ops M S L Rel) : Nat → Node → Machine M S L R Rel →
      Except String (List (Value M S L R Rel) × Machine M S L R Rel)
    | 0,_,_ => .error "Schema resolver expression syntax fuel exhausted"
    | fuel+1,expression,frame => do
        match expression with
        | .record "Ident" _ => .ok ([← SchemaResolveMachine.read frame (← atom (field expression "Name"))],frame)
        | .record "ParenExpr" _ => eval ops fuel (field expression "X") frame
        | .record "SelectorExpr" _ =>
            let (value,frame) ← one ops fuel (field expression "X") frame
            .ok ([← select frame value (← atom (field (field expression "Sel") "Name"))],frame)
        | .record "StarExpr" _ =>
            let (value,frame) ← one ops fuel (field expression "X") frame
            match value with
            | .list value => .ok ([.listCopy (copyOf value)],frame)
            | .relationshipPointer value => .ok ([.relationship value],frame)
            | _ => .error "Schema resolver dereference requires a non-nil list or relationship pointer"
        | .record "UnaryExpr" _ =>
            match field expression "Op" with
            | .atom "&" =>
                match field expression "X" with
                | .record "Ident" fields =>
                    match find fields "Name" with
                    | .atom "mapCopy" => .ok ([.mapLocal],frame)
                    | .atom "listCopy" => .ok ([.listLocal],frame)
                    | _ => .error "Schema resolver only takes addresses of its local copies"
                | _ => .error "Unmodeled schema resolver address expression"
            | .atom "!" =>
                let (value,frame) ← one ops fuel (field expression "X") frame
                match value with
                | .flag flag => .ok ([.flag (!flag)],frame)
                | _ => .error "Schema resolver negation requires a flag"
            | _ => .error "Unmodeled schema resolver unary operation"
        | .record "BinaryExpr" _ =>
            let (left,frame) ← one ops fuel (field expression "X") frame
            let (right,frame) ← one ops fuel (field expression "Y") frame
            match field expression "Op",right with
            | .atom "==",.nil => .ok ([.flag (← isNil left)],frame)
            | .atom "!=",.nil => .ok ([.flag (!(← isNil left))],frame)
            | _,_ => .error "Unmodeled schema resolver binary operation"
        | .record "CompositeLit" _ =>
            match field expression "Type",field expression "Elts" with
            | .record "Ident" fields,.sequence [] =>
                match find fields "Name" with
                | .atom "Atom" => .ok ([.atom {}],frame)
                | .atom "Map" => .ok ([.mapCopy {}],frame)
                | _ => .error "Unmodeled schema resolver constructor"
            | _,_ => .error "Unmodeled schema resolver constructor fields"
        | .record "IndexExpr" _ =>
            let (value,frame) ← one ops fuel (field expression "X") frame
            let (key,frame) ← one ops fuel (field expression "Index") frame
            match value,key with
            | .cache values,.reference reference =>
                let (value,found) := lookup values reference.original
                .ok ([.atom value,.flag found],frame)
            | _,_ => .error "Schema resolver index requires a cache and TypeRef"
        | .record "CallExpr" _ =>
            match field expression "Fun",field expression "Ellipsis" with
            | .record "SelectorExpr" fields,.bit false =>
                let (receiver,frame) ← one ops fuel (find fields "X") frame
                let (arguments,frame) ← many ops fuel (← nodes (field expression "Args")) frame
                method ops receiver (← atom (field (find fields "Sel") "Name")) arguments frame
            | .record "Ident" fields,.bit false =>
                match find fields "Name",field expression "Args" with
                | .atom "make",.sequence [.record "MapType" fields] =>
                    match find fields "Key",find fields "Value" with
                    | .record "Ident" key,.record "Ident" value =>
                        match find key "Name",find value "Name" with
                        | .atom "TypeRef",.atom "Atom" => .ok ([.cache (some (fun _ => none))],frame)
                        | _,_ => .error "Unmodeled schema resolver map type"
                    | _,_ => .error "Unmodeled schema resolver map key/value"
                | _,_ => .error "Unmodeled schema resolver builtin"
            | _,_ => .error "Unmodeled schema resolver call syntax"
        | _ => .error "Unmodeled schema resolver expression syntax"
  termination_by fuel _ _ => (fuel,0)

  def one (ops : Ops M S L Rel) (fuel : Nat) (expression : Node) (frame : Machine M S L R Rel) :
      Except String (Value M S L R Rel × Machine M S L R Rel) := do
    let (values,frame) ← eval ops fuel expression frame
    match values with
    | [value] => .ok (value,frame)
    | _ => .error "Schema resolver expression requires one value"
  termination_by (fuel,1)

  def many (ops : Ops M S L Rel) : Nat → List Node → Machine M S L R Rel →
      Except String (List (Value M S L R Rel) × Machine M S L R Rel)
    | 0,_,_ => .error "Schema resolver argument syntax fuel exhausted"
    | _+1,[],frame => .ok ([],frame)
    | fuel+1,first::rest,frame => do
        let (values,frame) ← eval ops fuel first frame
        let (remaining,frame) ← many ops fuel rest frame
        .ok (values++remaining,frame)
  termination_by fuel _ _ => (fuel,2)
end

def write [DecidableEq R] (target : Node) (value : Value M S L R Rel)
    (frame : Machine M S L R Rel) : Except String (Machine M S L R Rel) := do
  match target with
  | .record "Ident" fields =>
      match find fields "Name",value with
      | .atom "result",.atom value => .ok {frame with result := value}
      | .atom "exists",.flag value => .ok {frame with found := value}
      | .atom "mapCopy",.mapCopy value =>
          if frame.mapEscaped then .error "Schema map copy cannot change after its pointer escapes"
          else .ok {frame with mapCopy := value}
      | .atom "listCopy",.listCopy value =>
          if frame.listEscaped then .error "Schema list copy cannot change after its pointer escapes"
          else .ok {frame with listCopy := value}
      | _,_ => .error "Unmodeled schema resolver local assignment"
  | .record "SelectorExpr" fields =>
      match find fields "X" with
      | .record "Ident" receiver =>
          match find receiver "Name",field (find fields "Sel") "Name",value with
          | .atom "s",.atom "resolvedTypes",.cache cache => .ok {frame with state := {frame.state with cache}}
          | .atom "mapCopy",.atom "ElementRelationship",.relationship relationship =>
              if frame.mapEscaped then .error "Schema map copy cannot change after its pointer escapes"
              else .ok {frame with mapCopy := {frame.mapCopy with relationship := some relationship}}
          | .atom "listCopy",.atom "ElementRelationship",.relationship relationship =>
              if frame.listEscaped then .error "Schema list copy cannot change after its pointer escapes"
              else .ok {frame with listCopy := {frame.listCopy with relationship := some relationship}}
          | .atom "result",.atom "Map",.mapLocal =>
              .ok {frame with result := {frame.result with map := some (← address frame.mapCopy)},mapEscaped := true}
          | .atom "result",.atom "List",.listLocal =>
              .ok {frame with result := {frame.result with list := some (← address frame.listCopy)},listEscaped := true}
          | _,_,_ => .error "Unmodeled schema resolver field assignment"
      | _ => .error "Schema resolver field assignment requires a direct frame slot"
  | .record "IndexExpr" fields =>
      match find fields "X",find fields "Index",value with
      | .record "SelectorExpr" receiver,.record "Ident" key,.atom value =>
          match find receiver "X",field (find receiver "Sel") "Name",find key "Name" with
          | .record "Ident" base,.atom "resolvedTypes",.atom "tr" =>
              match find base "Name" with
              | .atom "s" => .ok {frame with state := {frame.state with
                  cache := ← store frame.state.cache frame.reference.original value}}
              | _ => .error "Schema cache assignment requires the current schema"
          | _,_,_ => .error "Schema cache assignment requires the current TypeRef"
      | _,_,_ => .error "Unmodeled schema resolver cache assignment target"
  | _ => .error "Unmodeled schema resolver assignment target"

def assign [DecidableEq R] : List Node → List (Value M S L R Rel) →
    Machine M S L R Rel → Except String (Machine M S L R Rel)
  | [],[],frame => .ok frame
  | target::targets,value::values,frame => do assign targets values (← write target value frame)
  | _,_,_ => .error "Schema resolver assignment arity mismatch"

inductive Result (M S L R Rel : Type) where
  | next (frame : Machine M S L R Rel)
  | returned (values : List (Value M S L R Rel)) (frame : Machine M S L R Rel)

def declare (statement : Node) (frame : Machine M S L R Rel) : Except String (Machine M S L R Rel) := do
  let declaration := field statement "Decl"
  match field declaration "Tok",field declaration "Specs" with
  | .atom "var",.sequence [.record "ValueSpec" fields] =>
      match find fields "Names",find fields "Type",find fields "Values" with
      | .sequence [.record "Ident" name],.record "Ident" type,.sequence [] =>
          match find name "Name",find type "Name" with
          | .atom "result",.atom "Atom" => .ok {frame with result := {}}
          | .atom "exists",.atom "bool" => .ok {frame with found := false}
          | _,_ => .error "Unmodeled schema resolver zero declaration"
      | _,_,_ => .error "Unmodeled schema resolver variable declaration"
  | _,_ => .error "Unmodeled schema resolver declaration"

def deferCall (ops : Ops M S L Rel) (expression : Node) (frame : Machine M S L R Rel) :
    Except String (Machine M S L R Rel) := do
  match expression,field expression "Fun",field expression "Ellipsis" with
  | .record "CallExpr" _,.record "SelectorExpr" fields,.bit false =>
      let (receiver,frame) ← one ops 64 (find fields "X") frame
      let (arguments,frame) ← many ops 64 (← nodes (field expression "Args")) frame
      .ok {frame with deferred := {receiver,method := ← atom (field (find fields "Sel") "Name"),arguments}::frame.deferred}
  | _,_,_ => .error "Unmodeled schema resolver deferred call"

def guards (ops : Ops M S L Rel) : List Node → Machine M S L R Rel → Except String (Bool × Machine M S L R Rel)
  | [],frame => .ok (false,frame)
  | expression::rest,frame => do
      let (value,frame) ← one ops 64 expression frame
      match value with
      | .flag true => .ok (true,frame)
      | .flag false => guards ops rest frame
      | _ => .error "Schema resolver switch guard requires a flag"

mutual
  def exec [DecidableEq R] (ops : Ops M S L Rel) : Nat → Node → Machine M S L R Rel →
      Except String (Result M S L R Rel)
    | 0,_,_ => .error "Schema resolver statement syntax fuel exhausted"
    | fuel+1,statement,frame => do
        match statement with
        | .absent => .ok (.next frame)
        | .record "BlockStmt" _ => execList ops fuel (← nodes (field statement "List")) frame
        | .record "AssignStmt" _ =>
            match field statement "Tok" with
            | .atom ":=" | .atom "=" =>
                let (values,frame) ← many ops 64 (← nodes (field statement "Rhs")) frame
                .ok (.next (← assign (← nodes (field statement "Lhs")) values frame))
            | _ => .error "Unmodeled schema resolver assignment token"
        | .record "DeclStmt" _ => .ok (.next (← declare statement frame))
        | .record "ExprStmt" _ =>
            let (_,frame) ← eval ops 64 (field statement "X") frame
            .ok (.next frame)
        | .record "DeferStmt" _ => .ok (.next (← deferCall ops (field statement "Call") frame))
        | .record "IfStmt" _ =>
            match ← exec ops fuel (field statement "Init") frame with
            | .returned _ _ => .error "Schema resolver conditional initializer returned"
            | .next frame =>
                let (value,frame) ← one ops 64 (field statement "Cond") frame
                match value with
                | .flag value =>
                    if value then exec ops fuel (field statement "Body") frame
                    else exec ops fuel (field statement "Else") frame
                | _ => .error "Schema resolver conditional requires a flag"
        | .record "SwitchStmt" _ =>
            match field statement "Init",field statement "Tag" with
            | .absent,.absent => casesList ops fuel (← nodes (field (field statement "Body") "List")) none frame
            | _,_ => .error "Unmodeled schema resolver switch initializer or tag"
        | .record "ReturnStmt" _ =>
            let (values,frame) ← many ops 64 (← nodes (field statement "Results")) frame
            .ok (.returned values frame)
        | _ => .error "Unmodeled schema resolver statement syntax"
  termination_by fuel _ _ => (fuel,0)

  def execList [DecidableEq R] (ops : Ops M S L Rel) : Nat → List Node → Machine M S L R Rel →
      Except String (Result M S L R Rel)
    | 0,_,_ => .error "Schema resolver block syntax fuel exhausted"
    | _+1,[],frame => .ok (.next frame)
    | fuel+1,first::rest,frame => do
        match ← exec ops fuel first frame with
        | .returned values frame => .ok (.returned values frame)
        | .next frame => execList ops fuel rest frame
  termination_by fuel _ _ => (fuel,1)

  def casesList [DecidableEq R] (ops : Ops M S L Rel) : Nat → List Node → Option (List Node) →
      Machine M S L R Rel → Except String (Result M S L R Rel)
    | 0,_,_,_ => .error "Schema resolver switch syntax fuel exhausted"
    | fuel+1,[],fallback,frame =>
        match fallback with
        | none => .ok (.next frame)
        | some body => execList ops fuel body frame
    | fuel+1,first::rest,fallback,frame => do
        match first with
        | .record "CaseClause" _ =>
            let expressions ← nodes (field first "List")
            let body ← nodes (field first "Body")
            if expressions.isEmpty then
              match fallback with
              | some _ => .error "Schema resolver switch has multiple defaults"
              | none => casesList ops fuel rest (some body) frame
            else
              let (accepted,frame) ← guards ops expressions frame
              if accepted then execList ops fuel body frame
              else casesList ops fuel rest fallback frame
        | _ => .error "Schema resolver switch requires case clauses"
  termination_by fuel _ _ _ => (fuel,2)
end

def unwind (ops : Ops M S L Rel) : List (Deferred M S L R Rel) → Machine M S L R Rel →
    Except String (Machine M S L R Rel)
  | [],frame => .ok frame
  | first::rest,frame => do
      let (_,frame) ← method ops first.receiver first.method first.arguments frame
      unwind ops rest frame

def invoke [DecidableEq R] (ops : Ops M S L Rel) (reference : Reference M S L R Rel)
    (state : SchemaResolveMachine.State M S L R Rel) : Except String ((Atom M S L Rel × Bool) × SchemaResolveMachine.State M S L R Rel) := do
  match ← exec ops 64 SchemaLookupSource.resolveProgram {reference,state} with
  | .returned [.atom value,.flag found] frame =>
      let after ← unwind ops frame.deferred {frame with deferred := []}
      .ok ((value,found),after.state)
  | _ => .error "Schema resolver source did not return an atom/found pair"

end SSA.SchemaResolveSubset
