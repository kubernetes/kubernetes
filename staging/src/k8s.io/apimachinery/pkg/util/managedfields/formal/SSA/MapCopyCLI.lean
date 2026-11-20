/-
Copyright 2026 The Kubernetes Authors.
SPDX-License-Identifier: Apache-2.0
-/
import Lean
import SSA.MapCopyContracts

namespace SSA.MapCopyCLI

open Lean MapCopySubset

private structure Data where
  fields : String
  elementType : String
  unions : String
  relationship : String
  cache : Option String
  consumed : Bool
  deriving FromJson, ToJson

private def Data.decode (value : Data) : MapData String String String String String :=
  {fields := value.fields,elementType := value.elementType,unions := value.unions,
   relationship := value.relationship,cache := value.cache,consumed := value.consumed}

private def Data.encode (value : MapData String String String String String) : Data :=
  {fields := value.fields,elementType := value.elementType,unions := value.unions,
   relationship := value.relationship,cache := value.cache,consumed := value.consumed}

private structure Request where
  source : Data
  target : Option Data
  deriving FromJson

def evaluate (input : String) : Except String String := do
  let requests ← Json.parse input >>= fromJson? (α := List Request)
  if requests.isEmpty then throw "Map-copy protocol requires nonempty cases"
  let reports ← requests.mapM fun request => do
    let result ← MapCopySubset.invoke request.source.decode (request.target.map Data.decode)
    pure (result.map Data.encode)
  pure (toJson reports).compress

end SSA.MapCopyCLI
