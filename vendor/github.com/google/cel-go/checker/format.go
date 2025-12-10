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

package checker

import (
	"fmt"
	"strings"

	chkdecls "github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common/types"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

const (
	kindUnknown = iota + 1
	kindError
	kindFunction
	kindDyn
	kindPrimitive
	kindWellKnown
	kindWrapper
	kindNull
	kindAbstract
	kindType
	kindList
	kindMap
	kindObject
	kindTypeParam
)

// FormatCheckedType converts a type message into a string representation.
func FormatCheckedType(t *exprpb.Type) string {
	switch kindOf(t) {
	case kindDyn:
		return "dyn"
	case kindFunction:
		return formatFunctionExprType(t.GetFunction().GetResultType(),
			t.GetFunction().GetArgTypes(),
			false)
	case kindList:
		return fmt.Sprintf("list(%s)", FormatCheckedType(t.GetListType().GetElemType()))
	case kindObject:
		return t.GetMessageType()
	case kindMap:
		return fmt.Sprintf("map(%s, %s)",
			FormatCheckedType(t.GetMapType().GetKeyType()),
			FormatCheckedType(t.GetMapType().GetValueType()))
	case kindNull:
		return "null"
	case kindPrimitive:
		switch t.GetPrimitive() {
		case exprpb.Type_UINT64:
			return "uint"
		case exprpb.Type_INT64:
			return "int"
		}
		return strings.Trim(strings.ToLower(t.GetPrimitive().String()), " ")
	case kindType:
		if t.GetType() == nil || t.GetType().GetTypeKind() == nil {
			return "type"
		}
		return fmt.Sprintf("type(%s)", FormatCheckedType(t.GetType()))
	case kindWellKnown:
		switch t.GetWellKnown() {
		case exprpb.Type_ANY:
			return "any"
		case exprpb.Type_DURATION:
			return "duration"
		case exprpb.Type_TIMESTAMP:
			return "timestamp"
		}
	case kindWrapper:
		return fmt.Sprintf("wrapper(%s)",
			FormatCheckedType(chkdecls.NewPrimitiveType(t.GetWrapper())))
	case kindError:
		return "!error!"
	case kindTypeParam:
		return t.GetTypeParam()
	case kindAbstract:
		at := t.GetAbstractType()
		params := at.GetParameterTypes()
		paramStrs := make([]string, len(params))
		for i, p := range params {
			paramStrs[i] = FormatCheckedType(p)
		}
		return fmt.Sprintf("%s(%s)", at.GetName(), strings.Join(paramStrs, ", "))
	}
	return t.String()
}

type formatter func(any) string

// FormatCELType formats a types.Type value to a string representation.
//
// The type formatting is identical to FormatCheckedType.
func FormatCELType(t any) string {
	dt := t.(*types.Type)
	switch dt.Kind() {
	case types.AnyKind:
		return "any"
	case types.DurationKind:
		return "duration"
	case types.ErrorKind:
		return "!error!"
	case types.NullTypeKind:
		return "null"
	case types.TimestampKind:
		return "timestamp"
	case types.TypeParamKind:
		return dt.TypeName()
	case types.OpaqueKind:
		if dt.TypeName() == "function" {
			// There is no explicit function type in the new types representation, so information like
			// whether the function is a member function is absent.
			return formatFunctionDeclType(dt.Parameters()[0], dt.Parameters()[1:], false)
		}
	case types.UnspecifiedKind:
		return ""
	}
	if len(dt.Parameters()) == 0 {
		return dt.DeclaredTypeName()
	}
	paramTypeNames := make([]string, 0, len(dt.Parameters()))
	for _, p := range dt.Parameters() {
		paramTypeNames = append(paramTypeNames, FormatCELType(p))
	}
	return fmt.Sprintf("%s(%s)", dt.TypeName(), strings.Join(paramTypeNames, ", "))
}

func formatExprType(t any) string {
	if t == nil {
		return ""
	}
	return FormatCheckedType(t.(*exprpb.Type))
}

func formatFunctionExprType(resultType *exprpb.Type, argTypes []*exprpb.Type, isInstance bool) string {
	return formatFunctionInternal[*exprpb.Type](resultType, argTypes, isInstance, formatExprType)
}

func formatFunctionDeclType(resultType *types.Type, argTypes []*types.Type, isInstance bool) string {
	return formatFunctionInternal[*types.Type](resultType, argTypes, isInstance, FormatCELType)
}

func formatFunctionInternal[T any](resultType T, argTypes []T, isInstance bool, format formatter) string {
	result := ""
	if isInstance {
		target := argTypes[0]
		argTypes = argTypes[1:]
		result += format(target)
		result += "."
	}
	result += "("
	for i, arg := range argTypes {
		if i > 0 {
			result += ", "
		}
		result += format(arg)
	}
	result += ")"
	rt := format(resultType)
	if rt != "" {
		result += " -> "
		result += rt
	}
	return result
}

// kindOf returns the kind of the type as defined in the checked.proto.
func kindOf(t *exprpb.Type) int {
	if t == nil || t.TypeKind == nil {
		return kindUnknown
	}
	switch t.GetTypeKind().(type) {
	case *exprpb.Type_Error:
		return kindError
	case *exprpb.Type_Function:
		return kindFunction
	case *exprpb.Type_Dyn:
		return kindDyn
	case *exprpb.Type_Primitive:
		return kindPrimitive
	case *exprpb.Type_WellKnown:
		return kindWellKnown
	case *exprpb.Type_Wrapper:
		return kindWrapper
	case *exprpb.Type_Null:
		return kindNull
	case *exprpb.Type_Type:
		return kindType
	case *exprpb.Type_ListType_:
		return kindList
	case *exprpb.Type_MapType_:
		return kindMap
	case *exprpb.Type_MessageType:
		return kindObject
	case *exprpb.Type_TypeParam:
		return kindTypeParam
	case *exprpb.Type_AbstractType_:
		return kindAbstract
	}
	return kindUnknown
}
