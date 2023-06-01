// Copyright 2023 Google LLC
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
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// Bindings returns a cel.EnvOption to configure support for local variable
// bindings in expressions.
//
// # Cel.Bind
//
// Binds a simple identifier to an initialization expression which may be used
// in a subsequenct result expression. Bindings may also be nested within each
// other.
//
//	cel.bind(<varName>, <initExpr>, <resultExpr>)
//
// Examples:
//
//	cel.bind(a, 'hello',
//	cel.bind(b, 'world', a + b + b + a)) // "helloworldworldhello"
//
//	// Avoid a list allocation within the exists comprehension.
//	cel.bind(valid_values, [a, b, c],
//	[d, e, f].exists(elem, elem in valid_values))
//
// Local bindings are not guaranteed to be evaluated before use.
func Bindings() cel.EnvOption {
	return cel.Lib(celBindings{})
}

const (
	celNamespace  = "cel"
	bindMacro     = "bind"
	unusedIterVar = "#unused"
)

type celBindings struct{}

func (celBindings) LibraryName() string {
	return "cel.lib.ext.cel.bindings"
}

func (celBindings) CompileOptions() []cel.EnvOption {
	return []cel.EnvOption{
		cel.Macros(
			// cel.bind(var, <init>, <expr>)
			cel.NewReceiverMacro(bindMacro, 3, celBind),
		),
	}
}

func (celBindings) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

func celBind(meh cel.MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	if !macroTargetMatchesNamespace(celNamespace, target) {
		return nil, nil
	}
	varIdent := args[0]
	varName := ""
	switch varIdent.GetExprKind().(type) {
	case *exprpb.Expr_IdentExpr:
		varName = varIdent.GetIdentExpr().GetName()
	default:
		return nil, &common.Error{
			Message:  "cel.bind() variable names must be simple identifers",
			Location: meh.OffsetLocation(varIdent.GetId()),
		}
	}
	varInit := args[1]
	resultExpr := args[2]
	return meh.Fold(
		unusedIterVar,
		meh.NewList(),
		varName,
		varInit,
		meh.LiteralBool(false),
		meh.Ident(varName),
		resultExpr,
	), nil
}
