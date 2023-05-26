// Copyright 2022 Google LLC
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

// Protos returns a cel.EnvOption to configure extended macros and functions for
// proto manipulation.
//
// Note, all macros use the 'proto' namespace; however, at the time of macro
// expansion the namespace looks just like any other identifier. If you are
// currently using a variable named 'proto', the macro will likely work just as
// intended; however, there is some chance for collision.
//
// # Protos.GetExt
//
// Macro which generates a select expression that retrieves an extension field
// from the input proto2 syntax message. If the field is not set, the default
// value forthe extension field is returned according to safe-traversal semantics.
//
//	proto.getExt(<msg>, <fully.qualified.extension.name>) -> <field-type>
//
// Examples:
//
//	proto.getExt(msg, google.expr.proto2.test.int32_ext) // returns int value
//
// # Protos.HasExt
//
// Macro which generates a test-only select expression that determines whether
// an extension field is set on a proto2 syntax message.
//
//	proto.hasExt(<msg>, <fully.qualified.extension.name>) -> <bool>
//
// Examples:
//
//	proto.hasExt(msg, google.expr.proto2.test.int32_ext) // returns true || false
func Protos() cel.EnvOption {
	return cel.Lib(protoLib{})
}

var (
	protoNamespace = "proto"
	hasExtension   = "hasExt"
	getExtension   = "getExt"
)

type protoLib struct{}

// LibraryName implements the SingletonLibrary interface method.
func (protoLib) LibraryName() string {
	return "cel.lib.ext.protos"
}

// CompileOptions implements the Library interface method.
func (protoLib) CompileOptions() []cel.EnvOption {
	return []cel.EnvOption{
		cel.Macros(
			// proto.getExt(msg, select_expression)
			cel.NewReceiverMacro(getExtension, 2, getProtoExt),
			// proto.hasExt(msg, select_expression)
			cel.NewReceiverMacro(hasExtension, 2, hasProtoExt),
		),
	}
}

// ProgramOptions implements the Library interface method.
func (protoLib) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

// hasProtoExt generates a test-only select expression for a fully-qualified extension name on a protobuf message.
func hasProtoExt(meh cel.MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	if !macroTargetMatchesNamespace(protoNamespace, target) {
		return nil, nil
	}
	extensionField, err := getExtFieldName(meh, args[1])
	if err != nil {
		return nil, err
	}
	return meh.PresenceTest(args[0], extensionField), nil
}

// getProtoExt generates a select expression for a fully-qualified extension name on a protobuf message.
func getProtoExt(meh cel.MacroExprHelper, target *exprpb.Expr, args []*exprpb.Expr) (*exprpb.Expr, *common.Error) {
	if !macroTargetMatchesNamespace(protoNamespace, target) {
		return nil, nil
	}
	extFieldName, err := getExtFieldName(meh, args[1])
	if err != nil {
		return nil, err
	}
	return meh.Select(args[0], extFieldName), nil
}

func getExtFieldName(meh cel.MacroExprHelper, expr *exprpb.Expr) (string, *common.Error) {
	isValid := false
	extensionField := ""
	switch expr.GetExprKind().(type) {
	case *exprpb.Expr_SelectExpr:
		extensionField, isValid = validateIdentifier(expr)
	}
	if !isValid {
		return "", &common.Error{
			Message:  "invalid extension field",
			Location: meh.OffsetLocation(expr.GetId()),
		}
	}
	return extensionField, nil
}

func validateIdentifier(expr *exprpb.Expr) (string, bool) {
	switch expr.GetExprKind().(type) {
	case *exprpb.Expr_IdentExpr:
		return expr.GetIdentExpr().GetName(), true
	case *exprpb.Expr_SelectExpr:
		sel := expr.GetSelectExpr()
		if sel.GetTestOnly() {
			return "", false
		}
		opStr, isIdent := validateIdentifier(sel.GetOperand())
		if !isIdent {
			return "", false
		}
		return opStr + "." + sel.GetField(), true
	default:
		return "", false
	}
}
