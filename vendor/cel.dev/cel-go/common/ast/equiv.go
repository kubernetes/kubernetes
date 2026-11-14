// Copyright 2024 Google LLC
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
	"math"
	"slices"

	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
)

// EquivOption configures the structural equivalence comparison between ASTs or expressions.
type EquivOption func(*equivOptions)

// EquivTypes configures the type maps to use for validating type equivalence between two expressions or ASTs.
func EquivTypes(aTypes, bTypes map[int64]*types.Type) EquivOption {
	return func(opts *equivOptions) {
		opts.aTypes = aTypes
		opts.bTypes = bTypes
		opts.checkTypes = aTypes != nil || bTypes != nil
	}
}

// EquivReferences configures the reference maps to use for validating reference equivalence between two expressions or ASTs.
func EquivReferences(aRefs, bRefs map[int64]*ReferenceInfo) EquivOption {
	return func(opts *equivOptions) {
		opts.aRefs = aRefs
		opts.bRefs = bRefs
		opts.checkRefs = aRefs != nil || bRefs != nil
	}
}

// EquivIgnoreIdentifiers configures whether comprehension variable names (accuVar, iterVar, iterVar2)
// should be ignored during equivalence comparison.
func EquivIgnoreIdentifiers(enabled ...bool) EquivOption {
	return func(opts *equivOptions) {
		if len(enabled) == 0 {
			opts.ignoreIdentifiers = true
			return
		}
		opts.ignoreIdentifiers = enabled[0]
	}
}

// EquivMacroCalls configures whether to compare expressions by their macro call metadata
// if present, rather than their expanded AST representation.
func EquivMacroCalls(enabled ...bool) EquivOption {
	return func(opts *equivOptions) {
		if len(enabled) == 0 {
			opts.checkMacros = true
			return
		}
		opts.checkMacros = enabled[0]
	}
}

// EquivMacros configures the macro call maps to use for validating macro call equivalence between two expressions or ASTs.
func EquivMacros(aMacros, bMacros map[int64]Expr) EquivOption {
	return func(opts *equivOptions) {
		opts.aMacros = aMacros
		opts.bMacros = bMacros
		opts.checkMacros = aMacros != nil || bMacros != nil
	}
}

type equivOptions struct {
	aTypes            map[int64]*types.Type
	bTypes            map[int64]*types.Type
	aRefs             map[int64]*ReferenceInfo
	bRefs             map[int64]*ReferenceInfo
	aMacros           map[int64]Expr
	bMacros           map[int64]Expr
	checkTypes        bool
	checkRefs         bool
	checkMacros       bool
	ignoreIdentifiers bool
}

// EquivAST determines whether two ASTs are structurally equivalent.
//
// The comparison ignores numeric IDs except for resolving references and types.
// If both inputs are *AST instances, type and reference metadata are automatically
// compared if present.
//
// Note: expressions may be functionally equivalent, but not structurally equivalent.
// Proof of functional equivalence may be checked using the cel-java formal verification
// package (https://github.com/google/cel-java/blob/develop/optimizer/src/main/java/dev/cel/optimizer/verifier/README.md).
func EquivAST(a, b *AST, opts ...EquivOption) bool {
	if a == nil || b == nil {
		return a == b
	}
	optState := &equivOptions{
		aTypes:     a.TypeMap(),
		bTypes:     b.TypeMap(),
		aRefs:      a.ReferenceMap(),
		bRefs:      b.ReferenceMap(),
		aMacros:    a.SourceInfo().MacroCalls(),
		bMacros:    b.SourceInfo().MacroCalls(),
		checkTypes: a.IsChecked() || b.IsChecked(),
		checkRefs:  len(a.ReferenceMap()) > 0 || len(b.ReferenceMap()) > 0,
	}

	for _, opt := range opts {
		opt(optState)
	}

	state := &equivState{equivOptions: optState}
	return state.exprEquiv(a.Expr(), b.Expr())
}

// EquivExpr determines whether two expressions are structurally equivalent.
//
// The comparison ignores numeric IDs except for resolving references and types
// if options are provided.
func EquivExpr(a, b Expr, opts ...EquivOption) bool {
	optState := &equivOptions{}
	for _, opt := range opts {
		opt(optState)
	}
	state := &equivState{equivOptions: optState}
	return state.exprEquiv(a, b)
}

// Equiv returns true if two ASTs are structurally equivalent.
//
// If the ASTs are type-checked, type and reference information will be compared by default.
// Options can be supplied to customize type or reference comparison.
func (a *AST) Equiv(other *AST, opts ...EquivOption) bool {
	return EquivAST(a, other, opts...)
}

type equivState struct {
	*equivOptions
	ignored1 []string
	ignored2 []string
}

func (s *equivState) pushIgnored(comp1, comp2 ComprehensionExpr) int {
	prev := len(s.ignored1)
	if comp1.IterVar() != "" {
		s.ignored1 = append(s.ignored1, comp1.IterVar())
	}
	if comp1.HasIterVar2() {
		s.ignored1 = append(s.ignored1, comp1.IterVar2())
	}
	if comp1.AccuVar() != "" {
		s.ignored1 = append(s.ignored1, comp1.AccuVar())
	}

	if comp2.IterVar() != "" {
		s.ignored2 = append(s.ignored2, comp2.IterVar())
	}
	if comp2.HasIterVar2() {
		s.ignored2 = append(s.ignored2, comp2.IterVar2())
	}
	if comp2.AccuVar() != "" {
		s.ignored2 = append(s.ignored2, comp2.AccuVar())
	}
	return prev
}

func (s *equivState) popIgnored(prev int) {
	s.ignored1 = s.ignored1[:prev]
	s.ignored2 = s.ignored2[:prev]
}

func (s *equivState) isIgnored(name1, name2 string) bool {
	idx1 := lastIgnored(s.ignored1, name1)
	idx2 := lastIgnored(s.ignored2, name2)
	if idx1 >= 0 || idx2 >= 0 {
		return idx1 == idx2
	}
	return name1 == name2
}

func lastIgnored(stack []string, name string) int {
	for i, s := range slices.Backward(stack) {
		if s == name {
			return i
		}
	}
	return -1
}

func (s *equivState) exprEquiv(e1, e2 Expr) bool {
	if e1 == nil || e2 == nil {
		return e1 == e2
	}
	if !s.typeAndRefEquiv(e1.ID(), e2.ID()) {
		return false
	}
	if s.checkMacros {
		m1, found1 := s.aMacros[e1.ID()]
		m2, found2 := s.bMacros[e2.ID()]
		if found1 != found2 {
			return false
		}
		if found1 {
			return s.exprEquiv(m1, m2)
		}
	}
	if e1.Kind() != e2.Kind() {
		return false
	}
	switch e1.Kind() {
	case UnspecifiedExprKind:
		return true
	case IdentKind:
		name1 := e1.AsIdent()
		name2 := e2.AsIdent()
		if s.ignoreIdentifiers {
			return s.isIgnored(name1, name2)
		}
		return name1 == name2
	case LiteralKind:
		return equivLiteral(e1.AsLiteral(), e2.AsLiteral())
	case SelectKind:
		s1 := e1.AsSelect()
		s2 := e2.AsSelect()
		if s1.IsTestOnly() != s2.IsTestOnly() {
			return false
		}
		if s1.FieldName() != s2.FieldName() {
			return false
		}
		return s.exprEquiv(s1.Operand(), s2.Operand())
	case CallKind:
		c1 := e1.AsCall()
		c2 := e2.AsCall()
		if c1.FunctionName() != c2.FunctionName() || c1.IsMemberFunction() != c2.IsMemberFunction() {
			return false
		}
		args1 := c1.Args()
		args2 := c2.Args()
		if len(args1) != len(args2) {
			return false
		}
		if c1.IsMemberFunction() {
			if !s.exprEquiv(c1.Target(), c2.Target()) {
				return false
			}
		}

		for i := range args1 {
			if !s.exprEquiv(args1[i], args2[i]) {
				return false
			}
		}
		return true
	case ListKind:
		l1 := e1.AsList()
		l2 := e2.AsList()
		if l1.Size() != l2.Size() {
			return false
		}
		if len(l1.OptionalIndices()) != len(l2.OptionalIndices()) {
			return false
		}
		elems1 := l1.Elements()
		elems2 := l2.Elements()
		for i := range elems1 {
			if l1.IsOptional(int32(i)) != l2.IsOptional(int32(i)) {
				return false
			}
			if !s.exprEquiv(elems1[i], elems2[i]) {
				return false
			}
		}
		return true
	case MapKind:
		m1 := e1.AsMap()
		m2 := e2.AsMap()
		if m1.Size() != m2.Size() {
			return false
		}
		entries1 := m1.Entries()
		entries2 := m2.Entries()
		for i := range entries1 {
			if !s.entryExprEquiv(entries1[i], entries2[i]) {
				return false
			}
		}
		return true
	case StructKind:
		st1 := e1.AsStruct()
		st2 := e2.AsStruct()
		if st1.TypeName() != st2.TypeName() {
			return false
		}
		fields1 := st1.Fields()
		fields2 := st2.Fields()
		if len(fields1) != len(fields2) {
			return false
		}
		for i := range fields1 {
			if !s.entryExprEquiv(fields1[i], fields2[i]) {
				return false
			}
		}
		return true
	case ComprehensionKind:
		comp1 := e1.AsComprehension()
		comp2 := e2.AsComprehension()
		if comp1.HasIterVar2() != comp2.HasIterVar2() {
			return false
		}
		if (comp1.IterVar() == "") != (comp2.IterVar() == "") ||
			(comp1.AccuVar() == "") != (comp2.AccuVar() == "") {
			return false
		}
		if !s.ignoreIdentifiers {
			if comp1.IterVar() != comp2.IterVar() ||
				comp1.IterVar2() != comp2.IterVar2() ||
				comp1.AccuVar() != comp2.AccuVar() {
				return false
			}
		}
		if !s.exprEquiv(comp1.IterRange(), comp2.IterRange()) ||
			!s.exprEquiv(comp1.AccuInit(), comp2.AccuInit()) {
			return false
		}
		var prev int
		if s.ignoreIdentifiers {
			prev = s.pushIgnored(comp1, comp2)
		}
		equiv := s.exprEquiv(comp1.LoopCondition(), comp2.LoopCondition()) &&
			s.exprEquiv(comp1.LoopStep(), comp2.LoopStep()) &&
			s.exprEquiv(comp1.Result(), comp2.Result())
		if s.ignoreIdentifiers {
			s.popIgnored(prev)
		}
		return equiv
	default:
		return false
	}
}

func (s *equivState) entryExprEquiv(e1, e2 EntryExpr) bool {
	if e1 == nil || e2 == nil {
		return e1 == e2
	}
	if e1.Kind() != e2.Kind() {
		return false
	}
	if !s.typeAndRefEquiv(e1.ID(), e2.ID()) {
		return false
	}
	switch e1.Kind() {
	case MapEntryKind:
		me1 := e1.AsMapEntry()
		me2 := e2.AsMapEntry()
		if me1.IsOptional() != me2.IsOptional() {
			return false
		}
		return s.exprEquiv(me1.Key(), me2.Key()) && s.exprEquiv(me1.Value(), me2.Value())
	case StructFieldKind:
		sf1 := e1.AsStructField()
		sf2 := e2.AsStructField()
		if sf1.IsOptional() != sf2.IsOptional() {
			return false
		}
		if sf1.Name() != sf2.Name() {
			return false
		}
		return s.exprEquiv(sf1.Value(), sf2.Value())
	default:
		return true
	}
}

func (s *equivState) typeAndRefEquiv(id1, id2 int64) bool {
	if s.checkTypes {
		t1, found1 := s.aTypes[id1]
		t2, found2 := s.bTypes[id2]
		if found1 != found2 {
			return false
		}
		if found1 {
			if t1 == nil || t2 == nil {
				return t1 == t2
			}
			if !t1.IsEquivalentType(t2) {
				return false
			}
		}
	}
	if s.checkRefs {
		r1, found1 := s.aRefs[id1]
		r2, found2 := s.bRefs[id2]
		if found1 != found2 {
			return false
		}
		if found1 && !s.refEquiv(r1, r2) {
			return false
		}
	}
	return true
}

func (s *equivState) refEquiv(r1, r2 *ReferenceInfo) bool {
	if r1 == nil || r2 == nil {
		return r1 == r2
	}
	if s.ignoreIdentifiers {
		if !s.isIgnored(r1.Name, r2.Name) {
			return false
		}
	} else if r1.Name != r2.Name {
		return false
	}
	if len(r1.OverloadIDs) != len(r2.OverloadIDs) {
		return false
	}
	if len(r1.OverloadIDs) != 0 {
		overloadMap := make(map[string]struct{}, len(r1.OverloadIDs))
		for _, id := range r1.OverloadIDs {
			overloadMap[id] = struct{}{}
		}
		for _, id := range r2.OverloadIDs {
			if _, found := overloadMap[id]; !found {
				return false
			}
		}
	}
	if r1.Value == nil || r2.Value == nil {
		return r1.Value == r2.Value
	}
	return r1.Value.Equal(r2.Value) == types.True
}

func equivLiteral(l1, l2 ref.Val) bool {
	if l1 == nil || l2 == nil {
		return l1 == l2
	}
	if l1.Type().TypeName() != l2.Type().TypeName() {
		return false
	}
	if d1, ok := l1.(types.Double); ok {
		if d2, ok := l2.(types.Double); ok {
			if math.IsNaN(float64(d1)) && math.IsNaN(float64(d2)) {
				return true
			}
		}
	}
	return l1.Equal(l2) == types.True
}
