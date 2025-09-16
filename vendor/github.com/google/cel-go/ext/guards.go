// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ext

import (
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// function invocation guards for common call signatures within extension functions.

func intOrError(i int64, err error) ref.Val {
	if err != nil {
		return types.NewErrFromString(err.Error())
	}
	return types.Int(i)
}

func bytesOrError(bytes []byte, err error) ref.Val {
	if err != nil {
		return types.NewErrFromString(err.Error())
	}
	return types.Bytes(bytes)
}

func stringOrError(str string, err error) ref.Val {
	if err != nil {
		return types.NewErrFromString(err.Error())
	}
	return types.String(str)
}

func listStringOrError(strs []string, err error) ref.Val {
	if err != nil {
		return types.NewErrFromString(err.Error())
	}
	return types.DefaultTypeAdapter.NativeToValue(strs)
}

func extractIdent(target ast.Expr) (string, bool) {
	switch target.Kind() {
	case ast.IdentKind:
		return target.AsIdent(), true
	default:
		return "", false
	}
}

func macroTargetMatchesNamespace(ns string, target ast.Expr) bool {
	if id, found := extractIdent(target); found {
		return id == ns
	}
	return false
}
