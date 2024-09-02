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

import "github.com/google/cel-go/common/types/ref"

// ExprFactory interfaces defines a set of methods necessary for building native expression values.
type ExprFactory interface {
	// CopyExpr creates a deep copy of the input Expr value.
	CopyExpr(Expr) Expr

	// CopyEntryExpr creates a deep copy of the input EntryExpr value.
	CopyEntryExpr(EntryExpr) EntryExpr

	// NewCall creates an Expr value representing a global function call.
	NewCall(id int64, function string, args ...Expr) Expr

	// NewComprehension creates an Expr value representing a comprehension over a value range.
	NewComprehension(id int64, iterRange Expr, iterVar, accuVar string, accuInit, loopCondition, loopStep, result Expr) Expr

	// NewMemberCall creates an Expr value representing a member function call.
	NewMemberCall(id int64, function string, receiver Expr, args ...Expr) Expr

	// NewIdent creates an Expr value representing an identifier.
	NewIdent(id int64, name string) Expr

	// NewAccuIdent creates an Expr value representing an accumulator identifier within a
	//comprehension.
	NewAccuIdent(id int64) Expr

	// NewLiteral creates an Expr value representing a literal value, such as a string or integer.
	NewLiteral(id int64, value ref.Val) Expr

	// NewList creates an Expr value representing a list literal expression with optional indices.
	//
	// Optional indicies will typically be empty unless the CEL optional types are enabled.
	NewList(id int64, elems []Expr, optIndices []int32) Expr

	// NewMap creates an Expr value representing a map literal expression
	NewMap(id int64, entries []EntryExpr) Expr

	// NewMapEntry creates a MapEntry with a given key, value, and a flag indicating whether
	// the key is optionally set.
	NewMapEntry(id int64, key, value Expr, isOptional bool) EntryExpr

	// NewPresenceTest creates an Expr representing a field presence test on an operand expression.
	NewPresenceTest(id int64, operand Expr, field string) Expr

	// NewSelect creates an Expr representing a field selection on an operand expression.
	NewSelect(id int64, operand Expr, field string) Expr

	// NewStruct creates an Expr value representing a struct literal with a given type name and a
	// set of field initializers.
	NewStruct(id int64, typeName string, fields []EntryExpr) Expr

	// NewStructField creates a StructField with a given field name, value, and a flag indicating
	// whether the field is optionally set.
	NewStructField(id int64, field string, value Expr, isOptional bool) EntryExpr

	// NewUnspecifiedExpr creates an empty expression node.
	NewUnspecifiedExpr(id int64) Expr

	isExprFactory()
}

type baseExprFactory struct{}

// NewExprFactory creates an ExprFactory instance.
func NewExprFactory() ExprFactory {
	return &baseExprFactory{}
}

func (fac *baseExprFactory) NewCall(id int64, function string, args ...Expr) Expr {
	if len(args) == 0 {
		args = []Expr{}
	}
	return fac.newExpr(
		id,
		&baseCallExpr{
			function: function,
			target:   nilExpr,
			args:     args,
			isMember: false,
		})
}

func (fac *baseExprFactory) NewMemberCall(id int64, function string, target Expr, args ...Expr) Expr {
	if len(args) == 0 {
		args = []Expr{}
	}
	return fac.newExpr(
		id,
		&baseCallExpr{
			function: function,
			target:   target,
			args:     args,
			isMember: true,
		})
}

func (fac *baseExprFactory) NewComprehension(id int64, iterRange Expr, iterVar, accuVar string, accuInit, loopCond, loopStep, result Expr) Expr {
	return fac.newExpr(
		id,
		&baseComprehensionExpr{
			iterRange: iterRange,
			iterVar:   iterVar,
			accuVar:   accuVar,
			accuInit:  accuInit,
			loopCond:  loopCond,
			loopStep:  loopStep,
			result:    result,
		})
}

func (fac *baseExprFactory) NewIdent(id int64, name string) Expr {
	return fac.newExpr(id, baseIdentExpr(name))
}

func (fac *baseExprFactory) NewAccuIdent(id int64) Expr {
	return fac.NewIdent(id, "__result__")
}

func (fac *baseExprFactory) NewLiteral(id int64, value ref.Val) Expr {
	return fac.newExpr(id, &baseLiteral{Val: value})
}

func (fac *baseExprFactory) NewList(id int64, elems []Expr, optIndices []int32) Expr {
	optIndexMap := make(map[int32]struct{}, len(optIndices))
	for _, idx := range optIndices {
		optIndexMap[idx] = struct{}{}
	}
	return fac.newExpr(id,
		&baseListExpr{
			elements:    elems,
			optIndices:  optIndices,
			optIndexMap: optIndexMap,
		})
}

func (fac *baseExprFactory) NewMap(id int64, entries []EntryExpr) Expr {
	return fac.newExpr(id, &baseMapExpr{entries: entries})
}

func (fac *baseExprFactory) NewMapEntry(id int64, key, value Expr, isOptional bool) EntryExpr {
	return fac.newEntryExpr(
		id,
		&baseMapEntry{
			key:        key,
			value:      value,
			isOptional: isOptional,
		})
}

func (fac *baseExprFactory) NewPresenceTest(id int64, operand Expr, field string) Expr {
	return fac.newExpr(
		id,
		&baseSelectExpr{
			operand:  operand,
			field:    field,
			testOnly: true,
		})
}

func (fac *baseExprFactory) NewSelect(id int64, operand Expr, field string) Expr {
	return fac.newExpr(
		id,
		&baseSelectExpr{
			operand: operand,
			field:   field,
		})
}

func (fac *baseExprFactory) NewStruct(id int64, typeName string, fields []EntryExpr) Expr {
	return fac.newExpr(
		id,
		&baseStructExpr{
			typeName: typeName,
			fields:   fields,
		})
}

func (fac *baseExprFactory) NewStructField(id int64, field string, value Expr, isOptional bool) EntryExpr {
	return fac.newEntryExpr(
		id,
		&baseStructField{
			field:      field,
			value:      value,
			isOptional: isOptional,
		})
}

func (fac *baseExprFactory) NewUnspecifiedExpr(id int64) Expr {
	return fac.newExpr(id, nil)
}

func (fac *baseExprFactory) CopyExpr(e Expr) Expr {
	// unwrap navigable expressions to avoid unnecessary allocations during copying.
	if nav, ok := e.(*navigableExprImpl); ok {
		e = nav.Expr
	}
	switch e.Kind() {
	case CallKind:
		c := e.AsCall()
		argsCopy := make([]Expr, len(c.Args()))
		for i, arg := range c.Args() {
			argsCopy[i] = fac.CopyExpr(arg)
		}
		if !c.IsMemberFunction() {
			return fac.NewCall(e.ID(), c.FunctionName(), argsCopy...)
		}
		return fac.NewMemberCall(e.ID(), c.FunctionName(), fac.CopyExpr(c.Target()), argsCopy...)
	case ComprehensionKind:
		compre := e.AsComprehension()
		return fac.NewComprehension(e.ID(),
			fac.CopyExpr(compre.IterRange()),
			compre.IterVar(),
			compre.AccuVar(),
			fac.CopyExpr(compre.AccuInit()),
			fac.CopyExpr(compre.LoopCondition()),
			fac.CopyExpr(compre.LoopStep()),
			fac.CopyExpr(compre.Result()))
	case IdentKind:
		return fac.NewIdent(e.ID(), e.AsIdent())
	case ListKind:
		l := e.AsList()
		elemsCopy := make([]Expr, l.Size())
		for i, elem := range l.Elements() {
			elemsCopy[i] = fac.CopyExpr(elem)
		}
		return fac.NewList(e.ID(), elemsCopy, l.OptionalIndices())
	case LiteralKind:
		return fac.NewLiteral(e.ID(), e.AsLiteral())
	case MapKind:
		m := e.AsMap()
		entriesCopy := make([]EntryExpr, m.Size())
		for i, entry := range m.Entries() {
			entriesCopy[i] = fac.CopyEntryExpr(entry)
		}
		return fac.NewMap(e.ID(), entriesCopy)
	case SelectKind:
		s := e.AsSelect()
		if s.IsTestOnly() {
			return fac.NewPresenceTest(e.ID(), fac.CopyExpr(s.Operand()), s.FieldName())
		}
		return fac.NewSelect(e.ID(), fac.CopyExpr(s.Operand()), s.FieldName())
	case StructKind:
		s := e.AsStruct()
		fieldsCopy := make([]EntryExpr, len(s.Fields()))
		for i, field := range s.Fields() {
			fieldsCopy[i] = fac.CopyEntryExpr(field)
		}
		return fac.NewStruct(e.ID(), s.TypeName(), fieldsCopy)
	default:
		return fac.NewUnspecifiedExpr(e.ID())
	}
}

func (fac *baseExprFactory) CopyEntryExpr(e EntryExpr) EntryExpr {
	switch e.Kind() {
	case MapEntryKind:
		entry := e.AsMapEntry()
		return fac.NewMapEntry(e.ID(),
			fac.CopyExpr(entry.Key()), fac.CopyExpr(entry.Value()), entry.IsOptional())
	case StructFieldKind:
		field := e.AsStructField()
		return fac.NewStructField(e.ID(),
			field.Name(), fac.CopyExpr(field.Value()), field.IsOptional())
	default:
		return fac.newEntryExpr(e.ID(), nil)
	}
}

func (*baseExprFactory) isExprFactory() {}

func (fac *baseExprFactory) newExpr(id int64, e exprKindCase) Expr {
	return &expr{
		id:           id,
		exprKindCase: e,
	}
}

func (fac *baseExprFactory) newEntryExpr(id int64, e entryExprKindCase) EntryExpr {
	return &entryExpr{
		id:                id,
		entryExprKindCase: e,
	}
}

var (
	defaultFactory = &baseExprFactory{}
)
