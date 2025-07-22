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

package ast

import (
	"fmt"

	"google.golang.org/protobuf/proto"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"

	celpb "cel.dev/expr"
	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	structpb "google.golang.org/protobuf/types/known/structpb"
)

// ToProto converts an AST to a CheckedExpr protobouf.
func ToProto(ast *AST) (*exprpb.CheckedExpr, error) {
	refMap := make(map[int64]*exprpb.Reference, len(ast.ReferenceMap()))
	for id, ref := range ast.ReferenceMap() {
		r, err := ReferenceInfoToProto(ref)
		if err != nil {
			return nil, err
		}
		refMap[id] = r
	}
	typeMap := make(map[int64]*exprpb.Type, len(ast.TypeMap()))
	for id, typ := range ast.TypeMap() {
		t, err := types.TypeToExprType(typ)
		if err != nil {
			return nil, err
		}
		typeMap[id] = t
	}
	e, err := ExprToProto(ast.Expr())
	if err != nil {
		return nil, err
	}
	info, err := SourceInfoToProto(ast.SourceInfo())
	if err != nil {
		return nil, err
	}
	return &exprpb.CheckedExpr{
		Expr:         e,
		SourceInfo:   info,
		ReferenceMap: refMap,
		TypeMap:      typeMap,
	}, nil
}

// ToAST converts a CheckedExpr protobuf to an AST instance.
func ToAST(checked *exprpb.CheckedExpr) (*AST, error) {
	refMap := make(map[int64]*ReferenceInfo, len(checked.GetReferenceMap()))
	for id, ref := range checked.GetReferenceMap() {
		r, err := ProtoToReferenceInfo(ref)
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
	info, err := ProtoToSourceInfo(checked.GetSourceInfo())
	if err != nil {
		return nil, err
	}
	root, err := ProtoToExpr(checked.GetExpr())
	if err != nil {
		return nil, err
	}
	ast := NewCheckedAST(NewAST(root, info), typeMap, refMap)
	return ast, nil
}

// ProtoToExpr converts a protobuf Expr value to an ast.Expr value.
func ProtoToExpr(e *exprpb.Expr) (Expr, error) {
	factory := NewExprFactory()
	return exprInternal(factory, e)
}

// ProtoToEntryExpr converts a protobuf struct/map entry to an ast.EntryExpr
func ProtoToEntryExpr(e *exprpb.Expr_CreateStruct_Entry) (EntryExpr, error) {
	factory := NewExprFactory()
	switch e.GetKeyKind().(type) {
	case *exprpb.Expr_CreateStruct_Entry_FieldKey:
		return exprStructField(factory, e.GetId(), e)
	case *exprpb.Expr_CreateStruct_Entry_MapKey:
		return exprMapEntry(factory, e.GetId(), e)
	}
	return nil, fmt.Errorf("unsupported expr entry kind: %v", e)
}

func exprInternal(factory ExprFactory, e *exprpb.Expr) (Expr, error) {
	id := e.GetId()
	switch e.GetExprKind().(type) {
	case *exprpb.Expr_CallExpr:
		return exprCall(factory, id, e.GetCallExpr())
	case *exprpb.Expr_ComprehensionExpr:
		return exprComprehension(factory, id, e.GetComprehensionExpr())
	case *exprpb.Expr_ConstExpr:
		return exprLiteral(factory, id, e.GetConstExpr())
	case *exprpb.Expr_IdentExpr:
		return exprIdent(factory, id, e.GetIdentExpr())
	case *exprpb.Expr_ListExpr:
		return exprList(factory, id, e.GetListExpr())
	case *exprpb.Expr_SelectExpr:
		return exprSelect(factory, id, e.GetSelectExpr())
	case *exprpb.Expr_StructExpr:
		s := e.GetStructExpr()
		if s.GetMessageName() != "" {
			return exprStruct(factory, id, s)
		}
		return exprMap(factory, id, s)
	}
	return factory.NewUnspecifiedExpr(id), nil
}

func exprCall(factory ExprFactory, id int64, call *exprpb.Expr_Call) (Expr, error) {
	var err error
	args := make([]Expr, len(call.GetArgs()))
	for i, a := range call.GetArgs() {
		args[i], err = exprInternal(factory, a)
		if err != nil {
			return nil, err
		}
	}
	if call.GetTarget() == nil {
		return factory.NewCall(id, call.GetFunction(), args...), nil
	}

	target, err := exprInternal(factory, call.GetTarget())
	if err != nil {
		return nil, err
	}
	return factory.NewMemberCall(id, call.GetFunction(), target, args...), nil
}

func exprComprehension(factory ExprFactory, id int64, comp *exprpb.Expr_Comprehension) (Expr, error) {
	iterRange, err := exprInternal(factory, comp.GetIterRange())
	if err != nil {
		return nil, err
	}
	accuInit, err := exprInternal(factory, comp.GetAccuInit())
	if err != nil {
		return nil, err
	}
	loopCond, err := exprInternal(factory, comp.GetLoopCondition())
	if err != nil {
		return nil, err
	}
	loopStep, err := exprInternal(factory, comp.GetLoopStep())
	if err != nil {
		return nil, err
	}
	result, err := exprInternal(factory, comp.GetResult())
	if err != nil {
		return nil, err
	}
	return factory.NewComprehensionTwoVar(id,
		iterRange,
		comp.GetIterVar(),
		comp.GetIterVar2(),
		comp.GetAccuVar(),
		accuInit,
		loopCond,
		loopStep,
		result), nil
}

func exprLiteral(factory ExprFactory, id int64, c *exprpb.Constant) (Expr, error) {
	val, err := ConstantToVal(c)
	if err != nil {
		return nil, err
	}
	return factory.NewLiteral(id, val), nil
}

func exprIdent(factory ExprFactory, id int64, i *exprpb.Expr_Ident) (Expr, error) {
	return factory.NewIdent(id, i.GetName()), nil
}

func exprList(factory ExprFactory, id int64, l *exprpb.Expr_CreateList) (Expr, error) {
	elems := make([]Expr, len(l.GetElements()))
	for i, e := range l.GetElements() {
		elem, err := exprInternal(factory, e)
		if err != nil {
			return nil, err
		}
		elems[i] = elem
	}
	return factory.NewList(id, elems, l.GetOptionalIndices()), nil
}

func exprMap(factory ExprFactory, id int64, s *exprpb.Expr_CreateStruct) (Expr, error) {
	entries := make([]EntryExpr, len(s.GetEntries()))
	var err error
	for i, entry := range s.GetEntries() {
		entries[i], err = exprMapEntry(factory, entry.GetId(), entry)
		if err != nil {
			return nil, err
		}
	}
	return factory.NewMap(id, entries), nil
}

func exprMapEntry(factory ExprFactory, id int64, e *exprpb.Expr_CreateStruct_Entry) (EntryExpr, error) {
	k, err := exprInternal(factory, e.GetMapKey())
	if err != nil {
		return nil, err
	}
	v, err := exprInternal(factory, e.GetValue())
	if err != nil {
		return nil, err
	}
	return factory.NewMapEntry(id, k, v, e.GetOptionalEntry()), nil
}

func exprSelect(factory ExprFactory, id int64, s *exprpb.Expr_Select) (Expr, error) {
	op, err := exprInternal(factory, s.GetOperand())
	if err != nil {
		return nil, err
	}
	if s.GetTestOnly() {
		return factory.NewPresenceTest(id, op, s.GetField()), nil
	}
	return factory.NewSelect(id, op, s.GetField()), nil
}

func exprStruct(factory ExprFactory, id int64, s *exprpb.Expr_CreateStruct) (Expr, error) {
	fields := make([]EntryExpr, len(s.GetEntries()))
	var err error
	for i, field := range s.GetEntries() {
		fields[i], err = exprStructField(factory, field.GetId(), field)
		if err != nil {
			return nil, err
		}
	}
	return factory.NewStruct(id, s.GetMessageName(), fields), nil
}

func exprStructField(factory ExprFactory, id int64, f *exprpb.Expr_CreateStruct_Entry) (EntryExpr, error) {
	v, err := exprInternal(factory, f.GetValue())
	if err != nil {
		return nil, err
	}
	return factory.NewStructField(id, f.GetFieldKey(), v, f.GetOptionalEntry()), nil
}

// ExprToProto serializes an ast.Expr value to a protobuf Expr representation.
func ExprToProto(e Expr) (*exprpb.Expr, error) {
	if e == nil {
		return &exprpb.Expr{}, nil
	}
	switch e.Kind() {
	case CallKind:
		return protoCall(e.ID(), e.AsCall())
	case ComprehensionKind:
		return protoComprehension(e.ID(), e.AsComprehension())
	case IdentKind:
		return protoIdent(e.ID(), e.AsIdent())
	case ListKind:
		return protoList(e.ID(), e.AsList())
	case LiteralKind:
		return protoLiteral(e.ID(), e.AsLiteral())
	case MapKind:
		return protoMap(e.ID(), e.AsMap())
	case SelectKind:
		return protoSelect(e.ID(), e.AsSelect())
	case StructKind:
		return protoStruct(e.ID(), e.AsStruct())
	case UnspecifiedExprKind:
		// Handle the case where a macro reference may be getting translated.
		// A nested macro 'pointer' is a non-zero expression id with no kind set.
		if e.ID() != 0 {
			return &exprpb.Expr{Id: e.ID()}, nil
		}
		return &exprpb.Expr{}, nil
	}
	return nil, fmt.Errorf("unsupported expr kind: %v", e)
}

// EntryExprToProto converts an ast.EntryExpr to a protobuf CreateStruct entry
func EntryExprToProto(e EntryExpr) (*exprpb.Expr_CreateStruct_Entry, error) {
	switch e.Kind() {
	case MapEntryKind:
		return protoMapEntry(e.ID(), e.AsMapEntry())
	case StructFieldKind:
		return protoStructField(e.ID(), e.AsStructField())
	case UnspecifiedEntryExprKind:
		return &exprpb.Expr_CreateStruct_Entry{}, nil
	}
	return nil, fmt.Errorf("unsupported expr entry kind: %v", e)
}

func protoCall(id int64, call CallExpr) (*exprpb.Expr, error) {
	var err error
	var target *exprpb.Expr
	if call.IsMemberFunction() {
		target, err = ExprToProto(call.Target())
		if err != nil {
			return nil, err
		}
	}
	callArgs := call.Args()
	args := make([]*exprpb.Expr, len(callArgs))
	for i, a := range callArgs {
		args[i], err = ExprToProto(a)
		if err != nil {
			return nil, err
		}
	}
	return &exprpb.Expr{
		Id: id,
		ExprKind: &exprpb.Expr_CallExpr{
			CallExpr: &exprpb.Expr_Call{
				Function: call.FunctionName(),
				Target:   target,
				Args:     args,
			},
		},
	}, nil
}

func protoComprehension(id int64, comp ComprehensionExpr) (*exprpb.Expr, error) {
	iterRange, err := ExprToProto(comp.IterRange())
	if err != nil {
		return nil, err
	}
	accuInit, err := ExprToProto(comp.AccuInit())
	if err != nil {
		return nil, err
	}
	loopCond, err := ExprToProto(comp.LoopCondition())
	if err != nil {
		return nil, err
	}
	loopStep, err := ExprToProto(comp.LoopStep())
	if err != nil {
		return nil, err
	}
	result, err := ExprToProto(comp.Result())
	if err != nil {
		return nil, err
	}
	return &exprpb.Expr{
		Id: id,
		ExprKind: &exprpb.Expr_ComprehensionExpr{
			ComprehensionExpr: &exprpb.Expr_Comprehension{
				IterVar:       comp.IterVar(),
				IterVar2:      comp.IterVar2(),
				IterRange:     iterRange,
				AccuVar:       comp.AccuVar(),
				AccuInit:      accuInit,
				LoopCondition: loopCond,
				LoopStep:      loopStep,
				Result:        result,
			},
		},
	}, nil
}

func protoIdent(id int64, name string) (*exprpb.Expr, error) {
	return &exprpb.Expr{
		Id: id,
		ExprKind: &exprpb.Expr_IdentExpr{
			IdentExpr: &exprpb.Expr_Ident{
				Name: name,
			},
		},
	}, nil
}

func protoList(id int64, list ListExpr) (*exprpb.Expr, error) {
	var err error
	elems := make([]*exprpb.Expr, list.Size())
	for i, e := range list.Elements() {
		elems[i], err = ExprToProto(e)
		if err != nil {
			return nil, err
		}
	}
	return &exprpb.Expr{
		Id: id,
		ExprKind: &exprpb.Expr_ListExpr{
			ListExpr: &exprpb.Expr_CreateList{
				Elements:        elems,
				OptionalIndices: list.OptionalIndices(),
			},
		},
	}, nil
}

func protoLiteral(id int64, val ref.Val) (*exprpb.Expr, error) {
	c, err := ValToConstant(val)
	if err != nil {
		return nil, err
	}
	return &exprpb.Expr{
		Id: id,
		ExprKind: &exprpb.Expr_ConstExpr{
			ConstExpr: c,
		},
	}, nil
}

func protoMap(id int64, m MapExpr) (*exprpb.Expr, error) {
	entries := make([]*exprpb.Expr_CreateStruct_Entry, len(m.Entries()))
	var err error
	for i, e := range m.Entries() {
		entries[i], err = EntryExprToProto(e)
		if err != nil {
			return nil, err
		}
	}
	return &exprpb.Expr{
		Id: id,
		ExprKind: &exprpb.Expr_StructExpr{
			StructExpr: &exprpb.Expr_CreateStruct{
				Entries: entries,
			},
		},
	}, nil
}

func protoMapEntry(id int64, e MapEntry) (*exprpb.Expr_CreateStruct_Entry, error) {
	k, err := ExprToProto(e.Key())
	if err != nil {
		return nil, err
	}
	v, err := ExprToProto(e.Value())
	if err != nil {
		return nil, err
	}
	return &exprpb.Expr_CreateStruct_Entry{
		Id: id,
		KeyKind: &exprpb.Expr_CreateStruct_Entry_MapKey{
			MapKey: k,
		},
		Value:         v,
		OptionalEntry: e.IsOptional(),
	}, nil
}

func protoSelect(id int64, s SelectExpr) (*exprpb.Expr, error) {
	op, err := ExprToProto(s.Operand())
	if err != nil {
		return nil, err
	}
	return &exprpb.Expr{
		Id: id,
		ExprKind: &exprpb.Expr_SelectExpr{
			SelectExpr: &exprpb.Expr_Select{
				Operand:  op,
				Field:    s.FieldName(),
				TestOnly: s.IsTestOnly(),
			},
		},
	}, nil
}

func protoStruct(id int64, s StructExpr) (*exprpb.Expr, error) {
	entries := make([]*exprpb.Expr_CreateStruct_Entry, len(s.Fields()))
	var err error
	for i, e := range s.Fields() {
		entries[i], err = EntryExprToProto(e)
		if err != nil {
			return nil, err
		}
	}
	return &exprpb.Expr{
		Id: id,
		ExprKind: &exprpb.Expr_StructExpr{
			StructExpr: &exprpb.Expr_CreateStruct{
				MessageName: s.TypeName(),
				Entries:     entries,
			},
		},
	}, nil
}

func protoStructField(id int64, f StructField) (*exprpb.Expr_CreateStruct_Entry, error) {
	v, err := ExprToProto(f.Value())
	if err != nil {
		return nil, err
	}
	return &exprpb.Expr_CreateStruct_Entry{
		Id: id,
		KeyKind: &exprpb.Expr_CreateStruct_Entry_FieldKey{
			FieldKey: f.Name(),
		},
		Value:         v,
		OptionalEntry: f.IsOptional(),
	}, nil
}

// SourceInfoToProto serializes an ast.SourceInfo value to a protobuf SourceInfo object.
func SourceInfoToProto(info *SourceInfo) (*exprpb.SourceInfo, error) {
	if info == nil {
		return &exprpb.SourceInfo{}, nil
	}
	sourceInfo := &exprpb.SourceInfo{
		SyntaxVersion: info.SyntaxVersion(),
		Location:      info.Description(),
		LineOffsets:   info.LineOffsets(),
		Positions:     make(map[int64]int32, len(info.OffsetRanges())),
		MacroCalls:    make(map[int64]*exprpb.Expr, len(info.MacroCalls())),
	}
	for id, offset := range info.OffsetRanges() {
		sourceInfo.Positions[id] = offset.Start
	}
	for id, e := range info.MacroCalls() {
		call, err := ExprToProto(e)
		if err != nil {
			return nil, err
		}
		sourceInfo.MacroCalls[id] = call
	}
	return sourceInfo, nil
}

// ProtoToSourceInfo deserializes the protobuf into a native SourceInfo value.
func ProtoToSourceInfo(info *exprpb.SourceInfo) (*SourceInfo, error) {
	sourceInfo := &SourceInfo{
		syntax:       info.GetSyntaxVersion(),
		desc:         info.GetLocation(),
		lines:        info.GetLineOffsets(),
		offsetRanges: make(map[int64]OffsetRange, len(info.GetPositions())),
		macroCalls:   make(map[int64]Expr, len(info.GetMacroCalls())),
	}
	for id, offset := range info.GetPositions() {
		sourceInfo.SetOffsetRange(id, OffsetRange{Start: offset, Stop: offset})
	}
	for id, e := range info.GetMacroCalls() {
		call, err := ProtoToExpr(e)
		if err != nil {
			return nil, err
		}
		sourceInfo.SetMacroCall(id, call)
	}
	return sourceInfo, nil
}

// ReferenceInfoToProto converts a ReferenceInfo instance to a protobuf Reference suitable for serialization.
func ReferenceInfoToProto(info *ReferenceInfo) (*exprpb.Reference, error) {
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

// ProtoToReferenceInfo converts a protobuf Reference into a CEL-native ReferenceInfo instance.
func ProtoToReferenceInfo(ref *exprpb.Reference) (*ReferenceInfo, error) {
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
	return AlphaProtoConstantAsVal(c)
}

// AlphaProtoConstantAsVal converts a v1alpha1.Constant protobuf to a CEL-native ref.Val.
func AlphaProtoConstantAsVal(c *exprpb.Constant) (ref.Val, error) {
	if c == nil {
		return nil, nil
	}
	canonical := &celpb.Constant{}
	if err := convertProto(c, canonical); err != nil {
		return nil, err
	}
	return ProtoConstantAsVal(canonical)
}

// ProtoConstantAsVal converts a canonical celpb.Constant protobuf to a CEL-native ref.Val.
func ProtoConstantAsVal(c *celpb.Constant) (ref.Val, error) {
	switch c.GetConstantKind().(type) {
	case *celpb.Constant_BoolValue:
		return types.Bool(c.GetBoolValue()), nil
	case *celpb.Constant_BytesValue:
		return types.Bytes(c.GetBytesValue()), nil
	case *celpb.Constant_DoubleValue:
		return types.Double(c.GetDoubleValue()), nil
	case *celpb.Constant_Int64Value:
		return types.Int(c.GetInt64Value()), nil
	case *celpb.Constant_NullValue:
		return types.NullValue, nil
	case *celpb.Constant_StringValue:
		return types.String(c.GetStringValue()), nil
	case *celpb.Constant_Uint64Value:
		return types.Uint(c.GetUint64Value()), nil
	}
	return nil, fmt.Errorf("unsupported constant kind: %v", c.GetConstantKind())
}

func convertProto(src, dst proto.Message) error {
	pb, err := proto.Marshal(src)
	if err != nil {
		return err
	}
	err = proto.Unmarshal(pb, dst)
	return err
}
