/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import SSA.SchemaLookupSource
import SSA.SchemaResolveMachine

namespace SSA.MapCopySubset

open RawGoSyntax
open AtomicRemoveSubset (atom nodes)
variable {F E U Rel C : Type}

/-- Exported immutable contents and a serial view of the private field-index cache. -/
structure MapData (F E U Rel C : Type) where
  fields : F
  elementType : E
  unions : U
  relationship : Rel
  cache : Option C := none
  consumed : Bool := false

structure Frame (F E U Rel C : Type) where
  source : MapData F E U Rel C
  target : Option (MapData F E U Rel C)
  loaded : Option C := none

inductive Value (F E U Rel C : Type) where
  | source | target | syncPackage | sourceCache | targetCache | targetOnce | zeroOnce | nil
  | fields (value : F)
  | elementType (value : E)
  | unions (value : U)
  | relationship (value : Rel)
  | cache (value : Option C)
  | flag (value : Bool)
  | closure (body : Node)

inductive Result (F E U Rel C : Type) where
  | next (frame : Frame F E U Rel C)
  | returned (frame : Frame F E U Rel C)

def read (frame : Frame F E U Rel C) : String → Except String (Value F E U Rel C)
  | "m" => .ok .source
  | "dst" => .ok (if frame.target.isSome then .target else .nil)
  | "mm" => .ok (.cache frame.loaded)
  | "sync" => .ok .syncPackage
  | "nil" => .ok .nil
  | _ => .error "Unbound map-copy variable"

def target (frame : Frame F E U Rel C) : Except String (MapData F E U Rel C) :=
  match frame.target with
  | some value => .ok value
  | none => .error "Map copy would dereference a nil destination"

def select (frame : Frame F E U Rel C) : Value F E U Rel C → String → Except String (Value F E U Rel C)
  | .source,"Fields" => .ok (.fields frame.source.fields)
  | .source,"ElementType" => .ok (.elementType frame.source.elementType)
  | .source,"Unions" => .ok (.unions frame.source.unions)
  | .source,"ElementRelationship" => .ok (.relationship frame.source.relationship)
  | .source,"m" => .ok .sourceCache
  | .target,"m" => .ok .targetCache
  | .target,"once" => .ok .targetOnce
  | _,_ => .error "Unmodeled map-copy selection"

def isNil : Value F E U Rel C → Except String Bool
  | .nil | .cache none => .ok true
  | .target | .cache (some _) => .ok false
  | _ => .error "Unmodeled map-copy nil comparison"

def primitive (receiver : Value F E U Rel C) (name : String) (arguments : List (Value F E U Rel C))
    (frame : Frame F E U Rel C) : Except String (List (Value F E U Rel C) × Frame F E U Rel C) := do
  match receiver,name,arguments with
  | .sourceCache,"Load",[] => .ok ([.cache frame.source.cache],frame)
  | .targetCache,"Store",[.cache cache] =>
      .ok ([],{frame with target := some {← target frame with cache}})
  | _,_,_ => .error "Unmodeled map-copy atomic operation"

def write (expression : Node) (value : Value F E U Rel C) (frame : Frame F E U Rel C) : Except String (Frame F E U Rel C) := do
  match expression with
  | .record "Ident" fields =>
      match find fields "Name",value with
      | .atom "mm",.cache cache => .ok {frame with loaded := cache}
      | _,_ => .error "Unmodeled map-copy local assignment"
  | .record "SelectorExpr" fields =>
      match find fields "X" with
      | .record "Ident" receiver =>
          match find receiver "Name" with
          | .atom "dst" =>
              let destination ← target frame
              let after ← match field (find fields "Sel") "Name",value with
                | .atom "Fields",.fields fields => .ok {destination with fields}
                | .atom "ElementType",.elementType elementType => .ok {destination with elementType}
                | .atom "Unions",.unions unions => .ok {destination with unions}
                | .atom "ElementRelationship",.relationship relationship => .ok {destination with relationship}
                | .atom "once",.zeroOnce => .ok {destination with consumed := false}
                | _,_ => .error "Unmodeled map-copy destination field assignment"
              .ok {frame with target := some after}
          | _ => .error "Map copy only assigns destination fields"
      | _ => .error "Unmodeled map-copy assignment receiver"
  | _ => .error "Unmodeled map-copy assignment target"

mutual
  def eval : Nat → Node → Frame F E U Rel C → Except String (List (Value F E U Rel C) × Frame F E U Rel C)
    | 0,_,_ => .error "Map-copy expression syntax fuel exhausted"
    | fuel+1,expression,frame => do
        match expression with
        | .record "Ident" _ => .ok ([← MapCopySubset.read frame (← atom (field expression "Name"))],frame)
        | .record "ParenExpr" _ => eval fuel (field expression "X") frame
        | .record "SelectorExpr" _ =>
            let (receiver,frame) ← one fuel (field expression "X") frame
            .ok ([← select frame receiver (← atom (field (field expression "Sel") "Name"))],frame)
        | .record "BinaryExpr" _ =>
            let (left,frame) ← one fuel (field expression "X") frame
            let (right,frame) ← one fuel (field expression "Y") frame
            match field expression "Op",right with
            | .atom "==",.nil => .ok ([.flag (← isNil left)],frame)
            | .atom "!=",.nil => .ok ([.flag (!(← isNil left))],frame)
            | _,_ => .error "Unmodeled map-copy binary operation"
        | .record "CompositeLit" _ =>
            match field expression "Type",field expression "Elts" with
            | .record "SelectorExpr" fields,.sequence [] =>
                match find fields "X",field (find fields "Sel") "Name" with
                | .record "Ident" receiver,.atom "Once" =>
                    match find receiver "Name" with
                    | .atom "sync" => .ok ([.zeroOnce],frame)
                    | _ => .error "Map-copy Once constructor requires sync binding"
                | _,_ => .error "Unmodeled map-copy composite type"
            | _,_ => .error "Unmodeled map-copy constructor"
        | .record "FuncLit" _ =>
            let type := field expression "Type"
            match field (field type "Params") "List",field type "Results",field type "TypeParams" with
            | .sequence [],.absent,.absent => .ok ([.closure (field expression "Body")],frame)
            | _,_,_ => .error "Map-copy callback must take and return no values"
        | .record "CallExpr" _ =>
            match field expression "Fun",field expression "Ellipsis" with
            | .record "SelectorExpr" fields,.bit false =>
                let (receiver,frame) ← one fuel (find fields "X") frame
                let name ← atom (field (find fields "Sel") "Name")
                let (arguments,frame) ← many fuel (← nodes (field expression "Args")) frame
                match receiver,name,arguments with
                | .targetOnce,"Do",[.closure body] =>
                    let destination ← target frame
                    if destination.consumed then .ok ([],frame)
                    else
                      let after ← match ← exec fuel body frame with
                        | .next after | .returned after => .ok after
                      .ok ([],{after with target := some {← target after with consumed := true}})
                | _,_,_ => primitive receiver name arguments frame
            | _,_ => .error "Unmodeled map-copy call syntax"
        | _ => .error "Unmodeled map-copy expression syntax"
  termination_by fuel _ _ => (fuel,0)

  def one (fuel : Nat) (expression : Node) (frame : Frame F E U Rel C) :
      Except String (Value F E U Rel C × Frame F E U Rel C) := do
    let (values,frame) ← eval fuel expression frame
    match values with
    | [value] => .ok (value,frame)
    | _ => .error "Map-copy expression requires one value"
  termination_by (fuel,1)

  def many : Nat → List Node → Frame F E U Rel C → Except String (List (Value F E U Rel C) × Frame F E U Rel C)
    | 0,_,_ => .error "Map-copy argument syntax fuel exhausted"
    | _+1,[],frame => .ok ([],frame)
    | fuel+1,first::rest,frame => do
        let (values,frame) ← eval fuel first frame
        let (remaining,frame) ← many fuel rest frame
        .ok (values++remaining,frame)
  termination_by fuel _ _ => (fuel,2)

  def exec : Nat → Node → Frame F E U Rel C → Except String (Result F E U Rel C)
    | 0,_,_ => .error "Map-copy statement syntax fuel exhausted"
    | fuel+1,statement,frame => do
        match statement with
        | .absent => .ok (.next frame)
        | .record "BlockStmt" _ => execList fuel (← nodes (field statement "List")) frame
        | .record "AssignStmt" _ =>
            match field statement "Tok",field statement "Lhs",field statement "Rhs" with
            | .atom ":=",.sequence [left],.sequence [right]
            | .atom "=",.sequence [left],.sequence [right] =>
                let (value,frame) ← one fuel right frame
                .ok (.next (← write left value frame))
            | _,_,_ => .error "Unmodeled map-copy assignment syntax"
        | .record "ExprStmt" _ =>
            let (_,frame) ← eval fuel (field statement "X") frame
            .ok (.next frame)
        | .record "IfStmt" _ =>
            match field statement "Init" with
            | .absent =>
                let (value,frame) ← one fuel (field statement "Cond") frame
                match value with
                | .flag value =>
                    if value then exec fuel (field statement "Body") frame
                    else exec fuel (field statement "Else") frame
                | _ => .error "Map-copy conditional requires a flag"
            | _ => .error "Unmodeled map-copy conditional initializer"
        | .record "ReturnStmt" _ =>
            match field statement "Results" with
            | .sequence [] => .ok (.returned frame)
            | _ => .error "Map-copy return must have no values"
        | _ => .error "Unmodeled map-copy statement syntax"
  termination_by fuel _ _ => (fuel,3)

  def execList : Nat → List Node → Frame F E U Rel C → Except String (Result F E U Rel C)
    | 0,_,_ => .error "Map-copy block syntax fuel exhausted"
    | _+1,[],frame => .ok (.next frame)
    | fuel+1,first::rest,frame => do
        match ← exec fuel first frame with
        | .returned after => .ok (.returned after)
        | .next after => execList fuel rest after
  termination_by fuel _ _ => (fuel,4)
end

/-- The destination is disjoint from the immutable source. sync.Once and atomic pointers use serial semantics. -/
def invoke (source : MapData F E U Rel C) (destination : Option (MapData F E U Rel C)) :
    Except String (Option (MapData F E U Rel C)) := do
  match ← exec 128 SchemaLookupSource.copyProgram {source,target := destination} with
  | .next after | .returned after => .ok after.target

def view (value : MapData F E U Rel C) : SchemaResolveMachine.Container (F × E × U) Rel :=
  {payload := (value.fields,value.elementType,value.unions),relationship := value.relationship}

end SSA.MapCopySubset
