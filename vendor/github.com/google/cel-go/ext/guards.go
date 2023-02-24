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
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// function invocation guards for common call signatures within extension functions.

func intOrError(i int64, err error) ref.Val {
	if err != nil {
		return types.NewErr(err.Error())
	}
	return types.Int(i)
}

func bytesOrError(bytes []byte, err error) ref.Val {
	if err != nil {
		return types.NewErr(err.Error())
	}
	return types.Bytes(bytes)
}

func stringOrError(str string, err error) ref.Val {
	if err != nil {
		return types.NewErr(err.Error())
	}
	return types.String(str)
}

func listStringOrError(strs []string, err error) ref.Val {
	if err != nil {
		return types.NewErr(err.Error())
	}
	return types.DefaultTypeAdapter.NativeToValue(strs)
}

func macroTargetMatchesNamespace(ns string, target *exprpb.Expr) bool {
	switch target.GetExprKind().(type) {
	case *exprpb.Expr_IdentExpr:
		if target.GetIdentExpr().GetName() != ns {
			return false
		}
		return true
	default:
		return false
	}
}
