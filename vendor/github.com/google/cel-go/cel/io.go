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
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/parser"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	anypb "google.golang.org/protobuf/types/known/anypb"
)

// CheckedExprToAst converts a checked expression proto message to an Ast.
func CheckedExprToAst(checkedExpr *exprpb.CheckedExpr) *Ast {
	return CheckedExprToAstWithSource(checkedExpr, nil)
}

// CheckedExprToAstWithSource converts a checked expression proto message to an Ast,
// using the provided Source as the textual contents.
//
// In general the source is not necessary unless the AST has been modified between the
// `Parse` and `Check` calls as an `Ast` created from the `Parse` step will carry the source
// through future calls.
//
// Prefer CheckedExprToAst if loading expressions from storage.
func CheckedExprToAstWithSource(checkedExpr *exprpb.CheckedExpr, src Source) *Ast {
	refMap := checkedExpr.GetReferenceMap()
	if refMap == nil {
		refMap = map[int64]*exprpb.Reference{}
	}
	typeMap := checkedExpr.GetTypeMap()
	if typeMap == nil {
		typeMap = map[int64]*exprpb.Type{}
	}
	si := checkedExpr.GetSourceInfo()
	if si == nil {
		si = &exprpb.SourceInfo{}
	}
	if src == nil {
		src = common.NewInfoSource(si)
	}
	return &Ast{
		expr:    checkedExpr.GetExpr(),
		info:    si,
		source:  src,
		refMap:  refMap,
		typeMap: typeMap,
	}
}

// AstToCheckedExpr converts an Ast to an protobuf CheckedExpr value.
//
// If the Ast.IsChecked() returns false, this conversion method will return an error.
func AstToCheckedExpr(a *Ast) (*exprpb.CheckedExpr, error) {
	if !a.IsChecked() {
		return nil, fmt.Errorf("cannot convert unchecked ast")
	}
	return &exprpb.CheckedExpr{
		Expr:         a.Expr(),
		SourceInfo:   a.SourceInfo(),
		ReferenceMap: a.refMap,
		TypeMap:      a.typeMap,
	}, nil
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
	si := parsedExpr.GetSourceInfo()
	if si == nil {
		si = &exprpb.SourceInfo{}
	}
	if src == nil {
		src = common.NewInfoSource(si)
	}
	return &Ast{
		expr:   parsedExpr.GetExpr(),
		info:   si,
		source: src,
	}
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
	expr := a.Expr()
	info := a.SourceInfo()
	return parser.Unparse(expr, info)
}

// RefValueToValue converts between ref.Val and api.expr.Value.
// The result Value is the serialized proto form. The ref.Val must not be error or unknown.
func RefValueToValue(res ref.Val) (*exprpb.Value, error) {
	switch res.Type() {
	case types.BoolType:
		return &exprpb.Value{
			Kind: &exprpb.Value_BoolValue{BoolValue: res.Value().(bool)}}, nil
	case types.BytesType:
		return &exprpb.Value{
			Kind: &exprpb.Value_BytesValue{BytesValue: res.Value().([]byte)}}, nil
	case types.DoubleType:
		return &exprpb.Value{
			Kind: &exprpb.Value_DoubleValue{DoubleValue: res.Value().(float64)}}, nil
	case types.IntType:
		return &exprpb.Value{
			Kind: &exprpb.Value_Int64Value{Int64Value: res.Value().(int64)}}, nil
	case types.ListType:
		l := res.(traits.Lister)
		sz := l.Size().(types.Int)
		elts := make([]*exprpb.Value, 0, int64(sz))
		for i := types.Int(0); i < sz; i++ {
			v, err := RefValueToValue(l.Get(i))
			if err != nil {
				return nil, err
			}
			elts = append(elts, v)
		}
		return &exprpb.Value{
			Kind: &exprpb.Value_ListValue{
				ListValue: &exprpb.ListValue{Values: elts}}}, nil
	case types.MapType:
		mapper := res.(traits.Mapper)
		sz := mapper.Size().(types.Int)
		entries := make([]*exprpb.MapValue_Entry, 0, int64(sz))
		for it := mapper.Iterator(); it.HasNext().(types.Bool); {
			k := it.Next()
			v := mapper.Get(k)
			kv, err := RefValueToValue(k)
			if err != nil {
				return nil, err
			}
			vv, err := RefValueToValue(v)
			if err != nil {
				return nil, err
			}
			entries = append(entries, &exprpb.MapValue_Entry{Key: kv, Value: vv})
		}
		return &exprpb.Value{
			Kind: &exprpb.Value_MapValue{
				MapValue: &exprpb.MapValue{Entries: entries}}}, nil
	case types.NullType:
		return &exprpb.Value{
			Kind: &exprpb.Value_NullValue{}}, nil
	case types.StringType:
		return &exprpb.Value{
			Kind: &exprpb.Value_StringValue{StringValue: res.Value().(string)}}, nil
	case types.TypeType:
		typeName := res.(ref.Type).TypeName()
		return &exprpb.Value{Kind: &exprpb.Value_TypeValue{TypeValue: typeName}}, nil
	case types.UintType:
		return &exprpb.Value{
			Kind: &exprpb.Value_Uint64Value{Uint64Value: res.Value().(uint64)}}, nil
	default:
		any, err := res.ConvertToNative(anyPbType)
		if err != nil {
			return nil, err
		}
		return &exprpb.Value{
			Kind: &exprpb.Value_ObjectValue{ObjectValue: any.(*anypb.Any)}}, nil
	}
}

var (
	typeNameToTypeValue = map[string]*types.TypeValue{
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

// ValueToRefValue converts between exprpb.Value and ref.Val.
func ValueToRefValue(adapter ref.TypeAdapter, v *exprpb.Value) (ref.Val, error) {
	switch v.Kind.(type) {
	case *exprpb.Value_NullValue:
		return types.NullValue, nil
	case *exprpb.Value_BoolValue:
		return types.Bool(v.GetBoolValue()), nil
	case *exprpb.Value_Int64Value:
		return types.Int(v.GetInt64Value()), nil
	case *exprpb.Value_Uint64Value:
		return types.Uint(v.GetUint64Value()), nil
	case *exprpb.Value_DoubleValue:
		return types.Double(v.GetDoubleValue()), nil
	case *exprpb.Value_StringValue:
		return types.String(v.GetStringValue()), nil
	case *exprpb.Value_BytesValue:
		return types.Bytes(v.GetBytesValue()), nil
	case *exprpb.Value_ObjectValue:
		any := v.GetObjectValue()
		msg, err := anypb.UnmarshalNew(any, proto.UnmarshalOptions{DiscardUnknown: true})
		if err != nil {
			return nil, err
		}
		return adapter.NativeToValue(msg), nil
	case *exprpb.Value_MapValue:
		m := v.GetMapValue()
		entries := make(map[ref.Val]ref.Val)
		for _, entry := range m.Entries {
			key, err := ValueToRefValue(adapter, entry.Key)
			if err != nil {
				return nil, err
			}
			pb, err := ValueToRefValue(adapter, entry.Value)
			if err != nil {
				return nil, err
			}
			entries[key] = pb
		}
		return adapter.NativeToValue(entries), nil
	case *exprpb.Value_ListValue:
		l := v.GetListValue()
		elts := make([]ref.Val, len(l.Values))
		for i, e := range l.Values {
			rv, err := ValueToRefValue(adapter, e)
			if err != nil {
				return nil, err
			}
			elts[i] = rv
		}
		return adapter.NativeToValue(elts), nil
	case *exprpb.Value_TypeValue:
		typeName := v.GetTypeValue()
		tv, ok := typeNameToTypeValue[typeName]
		if ok {
			return tv, nil
		}
		return types.NewObjectTypeValue(typeName), nil
	}
	return nil, errors.New("unknown value")
}
