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

// Package ast declares data structures useful for parsed and checked abstract syntax trees
package ast

import (
	"fmt"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"

	structpb "google.golang.org/protobuf/types/known/structpb"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// CheckedAST contains a protobuf expression and source info along with CEL-native type and reference information.
type CheckedAST struct {
	Expr         *exprpb.Expr
	SourceInfo   *exprpb.SourceInfo
	TypeMap      map[int64]*types.Type
	ReferenceMap map[int64]*ReferenceInfo
}

// CheckedASTToCheckedExpr converts a CheckedAST to a CheckedExpr protobouf.
func CheckedASTToCheckedExpr(ast *CheckedAST) (*exprpb.CheckedExpr, error) {
	refMap := make(map[int64]*exprpb.Reference, len(ast.ReferenceMap))
	for id, ref := range ast.ReferenceMap {
		r, err := ReferenceInfoToReferenceExpr(ref)
		if err != nil {
			return nil, err
		}
		refMap[id] = r
	}
	typeMap := make(map[int64]*exprpb.Type, len(ast.TypeMap))
	for id, typ := range ast.TypeMap {
		t, err := types.TypeToExprType(typ)
		if err != nil {
			return nil, err
		}
		typeMap[id] = t
	}
	return &exprpb.CheckedExpr{
		Expr:         ast.Expr,
		SourceInfo:   ast.SourceInfo,
		ReferenceMap: refMap,
		TypeMap:      typeMap,
	}, nil
}

// CheckedExprToCheckedAST converts a CheckedExpr protobuf to a CheckedAST instance.
func CheckedExprToCheckedAST(checked *exprpb.CheckedExpr) (*CheckedAST, error) {
	refMap := make(map[int64]*ReferenceInfo, len(checked.GetReferenceMap()))
	for id, ref := range checked.GetReferenceMap() {
		r, err := ReferenceExprToReferenceInfo(ref)
		if err != nil {
			return nil, err
		}
		refMap[id] = r
	}
	typeMap := make(map[int64]*types.Type, len(checked.GetTypeMap()))
	for id, typ := range checked.GetTypeMap() {
		t, err := types.ExprTypeToType(typ)
		if err != nil {
			return nil, err
		}
		typeMap[id] = t
	}
	return &CheckedAST{
		Expr:         checked.GetExpr(),
		SourceInfo:   checked.GetSourceInfo(),
		ReferenceMap: refMap,
		TypeMap:      typeMap,
	}, nil
}

// ReferenceInfo contains a CEL native representation of an identifier reference which may refer to
// either a qualified identifier name, a set of overload ids, or a constant value from an enum.
type ReferenceInfo struct {
	Name        string
	OverloadIDs []string
	Value       ref.Val
}

// NewIdentReference creates a ReferenceInfo instance for an identifier with an optional constant value.
func NewIdentReference(name string, value ref.Val) *ReferenceInfo {
	return &ReferenceInfo{Name: name, Value: value}
}

// NewFunctionReference creates a ReferenceInfo instance for a set of function overloads.
func NewFunctionReference(overloads ...string) *ReferenceInfo {
	info := &ReferenceInfo{}
	for _, id := range overloads {
		info.AddOverload(id)
	}
	return info
}

// AddOverload appends a function overload ID to the ReferenceInfo.
func (r *ReferenceInfo) AddOverload(overloadID string) {
	for _, id := range r.OverloadIDs {
		if id == overloadID {
			return
		}
	}
	r.OverloadIDs = append(r.OverloadIDs, overloadID)
}

// Equals returns whether two references are identical to each other.
func (r *ReferenceInfo) Equals(other *ReferenceInfo) bool {
	if r.Name != other.Name {
		return false
	}
	if len(r.OverloadIDs) != len(other.OverloadIDs) {
		return false
	}
	if len(r.OverloadIDs) != 0 {
		overloadMap := make(map[string]struct{}, len(r.OverloadIDs))
		for _, id := range r.OverloadIDs {
			overloadMap[id] = struct{}{}
		}
		for _, id := range other.OverloadIDs {
			_, found := overloadMap[id]
			if !found {
				return false
			}
		}
	}
	if r.Value == nil && other.Value == nil {
		return true
	}
	if r.Value == nil && other.Value != nil ||
		r.Value != nil && other.Value == nil ||
		r.Value.Equal(other.Value) != types.True {
		return false
	}
	return true
}

// ReferenceInfoToReferenceExpr converts a ReferenceInfo instance to a protobuf Reference suitable for serialization.
func ReferenceInfoToReferenceExpr(info *ReferenceInfo) (*exprpb.Reference, error) {
	c, err := ValToConstant(info.Value)
	if err != nil {
		return nil, err
	}
	return &exprpb.Reference{
		Name:       info.Name,
		OverloadId: info.OverloadIDs,
		Value:      c,
	}, nil
}

// ReferenceExprToReferenceInfo converts a protobuf Reference into a CEL-native ReferenceInfo instance.
func ReferenceExprToReferenceInfo(ref *exprpb.Reference) (*ReferenceInfo, error) {
	v, err := ConstantToVal(ref.GetValue())
	if err != nil {
		return nil, err
	}
	return &ReferenceInfo{
		Name:        ref.GetName(),
		OverloadIDs: ref.GetOverloadId(),
		Value:       v,
	}, nil
}

// ValToConstant converts a CEL-native ref.Val to a protobuf Constant.
//
// Only simple scalar types are supported by this method.
func ValToConstant(v ref.Val) (*exprpb.Constant, error) {
	if v == nil {
		return nil, nil
	}
	switch v.Type() {
	case types.BoolType:
		return &exprpb.Constant{ConstantKind: &exprpb.Constant_BoolValue{BoolValue: v.Value().(bool)}}, nil
	case types.BytesType:
		return &exprpb.Constant{ConstantKind: &exprpb.Constant_BytesValue{BytesValue: v.Value().([]byte)}}, nil
	case types.DoubleType:
		return &exprpb.Constant{ConstantKind: &exprpb.Constant_DoubleValue{DoubleValue: v.Value().(float64)}}, nil
	case types.IntType:
		return &exprpb.Constant{ConstantKind: &exprpb.Constant_Int64Value{Int64Value: v.Value().(int64)}}, nil
	case types.NullType:
		return &exprpb.Constant{ConstantKind: &exprpb.Constant_NullValue{NullValue: structpb.NullValue_NULL_VALUE}}, nil
	case types.StringType:
		return &exprpb.Constant{ConstantKind: &exprpb.Constant_StringValue{StringValue: v.Value().(string)}}, nil
	case types.UintType:
		return &exprpb.Constant{ConstantKind: &exprpb.Constant_Uint64Value{Uint64Value: v.Value().(uint64)}}, nil
	}
	return nil, fmt.Errorf("unsupported constant kind: %v", v.Type())
}

// ConstantToVal converts a protobuf Constant to a CEL-native ref.Val.
func ConstantToVal(c *exprpb.Constant) (ref.Val, error) {
	if c == nil {
		return nil, nil
	}
	switch c.GetConstantKind().(type) {
	case *exprpb.Constant_BoolValue:
		return types.Bool(c.GetBoolValue()), nil
	case *exprpb.Constant_BytesValue:
		return types.Bytes(c.GetBytesValue()), nil
	case *exprpb.Constant_DoubleValue:
		return types.Double(c.GetDoubleValue()), nil
	case *exprpb.Constant_Int64Value:
		return types.Int(c.GetInt64Value()), nil
	case *exprpb.Constant_NullValue:
		return types.NullValue, nil
	case *exprpb.Constant_StringValue:
		return types.String(c.GetStringValue()), nil
	case *exprpb.Constant_Uint64Value:
		return types.Uint(c.GetUint64Value()), nil
	}
	return nil, fmt.Errorf("unsupported constant kind: %v", c.GetConstantKind())
}
