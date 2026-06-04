// Copyright 2019 Google LLC
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

package cel

import (
	"errors"
	"fmt"
	"reflect"

	"google.golang.org/protobuf/proto"

	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/parser"

	celpb "cel.dev/expr"
	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	anypb "google.golang.org/protobuf/types/known/anypb"
)

// CheckedExprToAst converts a checked expression proto message to an Ast.
func CheckedExprToAst(checkedExpr *exprpb.CheckedExpr) *Ast {
	checked, _ := CheckedExprToAstWithSource(checkedExpr, nil)
	return checked
}

// CheckedExprToAstWithSource converts a checked expression proto message to an Ast,
// using the provided Source as the textual contents.
//
// In general the source is not necessary unless the AST has been modified between the
// `Parse` and `Check` calls as an `Ast` created from the `Parse` step will carry the source
// through future calls.
//
// Prefer CheckedExprToAst if loading expressions from storage.
func CheckedExprToAstWithSource(checkedExpr *exprpb.CheckedExpr, src Source) (*Ast, error) {
	checked, err := ast.ToAST(checkedExpr)
	if err != nil {
		return nil, err
	}
	return &Ast{source: src, impl: checked}, nil
}

// AstToCheckedExpr converts an Ast to an protobuf CheckedExpr value.
//
// If the Ast.IsChecked() returns false, this conversion method will return an error.
func AstToCheckedExpr(a *Ast) (*exprpb.CheckedExpr, error) {
	if !a.IsChecked() {
		return nil, fmt.Errorf("cannot convert unchecked ast")
	}
	return ast.ToProto(a.NativeRep())
}

// ParsedExprToAst converts a parsed expression proto message to an Ast.
func ParsedExprToAst(parsedExpr *exprpb.ParsedExpr) *Ast {
	return ParsedExprToAstWithSource(parsedExpr, nil)
}

// ParsedExprToAstWithSource converts a parsed expression proto message to an Ast,
// using the provided Source as the textual contents.
//
// In general you only need this if you need to recheck a previously checked
// expression, or if you need to separately check a subset of an expression.
//
// Prefer ParsedExprToAst if loading expressions from storage.
func ParsedExprToAstWithSource(parsedExpr *exprpb.ParsedExpr, src Source) *Ast {
	info, _ := ast.ProtoToSourceInfo(parsedExpr.GetSourceInfo())
	if src == nil {
		src = common.NewInfoSource(parsedExpr.GetSourceInfo())
	}
	e, _ := ast.ProtoToExpr(parsedExpr.GetExpr())
	return &Ast{source: src, impl: ast.NewAST(e, info)}
}

// AstToParsedExpr converts an Ast to an protobuf ParsedExpr value.
func AstToParsedExpr(a *Ast) (*exprpb.ParsedExpr, error) {
	return &exprpb.ParsedExpr{
		Expr:       a.Expr(),
		SourceInfo: a.SourceInfo(),
	}, nil
}

// AstToString converts an Ast back to a string if possible.
//
// Note, the conversion may not be an exact replica of the original expression, but will produce
// a string that is semantically equivalent and whose textual representation is stable.
func AstToString(a *Ast) (string, error) {
	return ExprToString(a.NativeRep().Expr(), a.NativeRep().SourceInfo())
}

// ExprToString converts an AST Expr node back to a string using macro call tracking metadata from
// source info if any macros are encountered within the expression.
func ExprToString(e ast.Expr, info *ast.SourceInfo) (string, error) {
	return parser.Unparse(e, info)
}

// RefValueToValue converts between ref.Val and google.api.expr.v1alpha1.Value.
// The result Value is the serialized proto form. The ref.Val must not be error or unknown.
func RefValueToValue(res ref.Val) (*exprpb.Value, error) {
	return ValueAsAlphaProto(res)
}

// ValueAsAlphaProto converts between ref.Val and google.api.expr.v1alpha1.Value.
// The result Value is the serialized proto form. The ref.Val must not be error or unknown.
func ValueAsAlphaProto(res ref.Val) (*exprpb.Value, error) {
	canonical, err := ValueAsProto(res)
	if err != nil {
		return nil, err
	}
	alpha := &exprpb.Value{}
	err = convertProto(canonical, alpha)
	return alpha, err
}

// RefValToExprValue converts between ref.Val and google.api.expr.v1alpha1.ExprValue.
// The result ExprValue is the serialized proto form.
func RefValToExprValue(res ref.Val) (*exprpb.ExprValue, error) {
	return ExprValueAsAlphaProto(res)
}

// ExprValueAsAlphaProto converts between ref.Val and google.api.expr.v1alpha1.ExprValue.
// The result ExprValue is the serialized proto form.
func ExprValueAsAlphaProto(res ref.Val) (*exprpb.ExprValue, error) {
	canonical, err := ExprValueAsProto(res)
	if err != nil {
		return nil, err
	}
	alpha := &exprpb.ExprValue{}
	err = convertProto(canonical, alpha)
	return alpha, err
}

// ExprValueAsProto converts between ref.Val and cel.expr.ExprValue.
// The result ExprValue is the serialized proto form.
func ExprValueAsProto(res ref.Val) (*celpb.ExprValue, error) {
	switch res := res.(type) {
	case *types.Unknown:
		return &celpb.ExprValue{
			Kind: &celpb.ExprValue_Unknown{
				Unknown: &celpb.UnknownSet{
					Exprs: res.IDs(),
				},
			}}, nil
	case *types.Err:
		return &celpb.ExprValue{
			Kind: &celpb.ExprValue_Error{
				Error: &celpb.ErrorSet{
					// Keeping the error code as UNKNOWN since there's no error codes associated with
					// Cel-Go runtime errors.
					Errors: []*celpb.Status{{Code: 2, Message: res.Error()}},
				},
			},
		}, nil
	default:
		val, err := ValueAsProto(res)
		if err != nil {
			return nil, err
		}
		return &celpb.ExprValue{
			Kind: &celpb.ExprValue_Value{Value: val}}, nil
	}
}

// ValueAsProto converts between ref.Val and cel.expr.Value.
// The result Value is the serialized proto form. The ref.Val must not be error or unknown.
func ValueAsProto(res ref.Val) (*celpb.Value, error) {
	switch res.Type() {
	case types.BoolType:
		return &celpb.Value{
			Kind: &celpb.Value_BoolValue{BoolValue: res.Value().(bool)}}, nil
	case types.BytesType:
		return &celpb.Value{
			Kind: &celpb.Value_BytesValue{BytesValue: res.Value().([]byte)}}, nil
	case types.DoubleType:
		return &celpb.Value{
			Kind: &celpb.Value_DoubleValue{DoubleValue: res.Value().(float64)}}, nil
	case types.IntType:
		return &celpb.Value{
			Kind: &celpb.Value_Int64Value{Int64Value: res.Value().(int64)}}, nil
	case types.ListType:
		l := res.(traits.Lister)
		sz := l.Size().(types.Int)
		elts := make([]*celpb.Value, 0, int64(sz))
		for i := types.Int(0); i < sz; i++ {
			v, err := ValueAsProto(l.Get(i))
			if err != nil {
				return nil, err
			}
			elts = append(elts, v)
		}
		return &celpb.Value{
			Kind: &celpb.Value_ListValue{
				ListValue: &celpb.ListValue{Values: elts}}}, nil
	case types.MapType:
		mapper := res.(traits.Mapper)
		sz := mapper.Size().(types.Int)
		entries := make([]*celpb.MapValue_Entry, 0, int64(sz))
		for it := mapper.Iterator(); it.HasNext().(types.Bool); {
			k := it.Next()
			v := mapper.Get(k)
			kv, err := ValueAsProto(k)
			if err != nil {
				return nil, err
			}
			vv, err := ValueAsProto(v)
			if err != nil {
				return nil, err
			}
			entries = append(entries, &celpb.MapValue_Entry{Key: kv, Value: vv})
		}
		return &celpb.Value{
			Kind: &celpb.Value_MapValue{
				MapValue: &celpb.MapValue{Entries: entries}}}, nil
	case types.NullType:
		return &celpb.Value{
			Kind: &celpb.Value_NullValue{}}, nil
	case types.StringType:
		return &celpb.Value{
			Kind: &celpb.Value_StringValue{StringValue: res.Value().(string)}}, nil
	case types.TypeType:
		typeName := res.(ref.Type).TypeName()
		return &celpb.Value{Kind: &celpb.Value_TypeValue{TypeValue: typeName}}, nil
	case types.UintType:
		return &celpb.Value{
			Kind: &celpb.Value_Uint64Value{Uint64Value: res.Value().(uint64)}}, nil
	default:
		any, err := res.ConvertToNative(anyPbType)
		if err != nil {
			return nil, err
		}
		return &celpb.Value{
			Kind: &celpb.Value_ObjectValue{ObjectValue: any.(*anypb.Any)}}, nil
	}
}

var (
	typeNameToTypeValue = map[string]ref.Val{
		"bool":      types.BoolType,
		"bytes":     types.BytesType,
		"double":    types.DoubleType,
		"null_type": types.NullType,
		"int":       types.IntType,
		"list":      types.ListType,
		"map":       types.MapType,
		"string":    types.StringType,
		"type":      types.TypeType,
		"uint":      types.UintType,
	}

	anyPbType = reflect.TypeOf(&anypb.Any{})
)

// ValueToRefValue converts between google.api.expr.v1alpha1.Value and ref.Val.
func ValueToRefValue(adapter types.Adapter, v *exprpb.Value) (ref.Val, error) {
	return AlphaProtoAsValue(adapter, v)
}

// AlphaProtoAsValue converts between google.api.expr.v1alpha1.Value and ref.Val.
func AlphaProtoAsValue(adapter types.Adapter, v *exprpb.Value) (ref.Val, error) {
	canonical := &celpb.Value{}
	if err := convertProto(v, canonical); err != nil {
		return nil, err
	}
	return ProtoAsValue(adapter, canonical)
}

// ProtoAsValue converts between cel.expr.Value and ref.Val.
func ProtoAsValue(adapter types.Adapter, v *celpb.Value) (ref.Val, error) {
	switch v.Kind.(type) {
	case *celpb.Value_NullValue:
		return types.NullValue, nil
	case *celpb.Value_BoolValue:
		return types.Bool(v.GetBoolValue()), nil
	case *celpb.Value_Int64Value:
		return types.Int(v.GetInt64Value()), nil
	case *celpb.Value_Uint64Value:
		return types.Uint(v.GetUint64Value()), nil
	case *celpb.Value_DoubleValue:
		return types.Double(v.GetDoubleValue()), nil
	case *celpb.Value_StringValue:
		return types.String(v.GetStringValue()), nil
	case *celpb.Value_BytesValue:
		return types.Bytes(v.GetBytesValue()), nil
	case *celpb.Value_ObjectValue:
		any := v.GetObjectValue()
		msg, err := anypb.UnmarshalNew(any, proto.UnmarshalOptions{DiscardUnknown: true})
		if err != nil {
			return nil, err
		}
		return adapter.NativeToValue(msg), nil
	case *celpb.Value_MapValue:
		m := v.GetMapValue()
		entries := make(map[ref.Val]ref.Val)
		for _, entry := range m.Entries {
			key, err := ProtoAsValue(adapter, entry.Key)
			if err != nil {
				return nil, err
			}
			pb, err := ProtoAsValue(adapter, entry.Value)
			if err != nil {
				return nil, err
			}
			entries[key] = pb
		}
		return adapter.NativeToValue(entries), nil
	case *celpb.Value_ListValue:
		l := v.GetListValue()
		elts := make([]ref.Val, len(l.Values))
		for i, e := range l.Values {
			rv, err := ProtoAsValue(adapter, e)
			if err != nil {
				return nil, err
			}
			elts[i] = rv
		}
		return adapter.NativeToValue(elts), nil
	case *celpb.Value_TypeValue:
		typeName := v.GetTypeValue()
		tv, ok := typeNameToTypeValue[typeName]
		if ok {
			return tv, nil
		}
		return types.NewObjectTypeValue(typeName), nil
	}
	return nil, errors.New("unknown value")
}

func convertProto(src, dst proto.Message) error {
	pb, err := proto.Marshal(src)
	if err != nil {
		return err
	}
	err = proto.Unmarshal(pb, dst)
	return err
}
