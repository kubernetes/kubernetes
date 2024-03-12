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
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// NavigableExpr represents the base navigable expression value with methods to inspect the
// parent and child expressions.
type NavigableExpr interface {
	Expr

	// Type of the expression.
	//
	// If the expression is type-checked, the type check metadata is returned. If the expression
	// has not been type-checked, the types.DynType value is returned.
	Type() *types.Type

	// Parent returns the parent expression node, if one exists.
	Parent() (NavigableExpr, bool)

	// Children returns a list of child expression nodes.
	Children() []NavigableExpr

	// Depth indicates the depth in the expression tree.
	//
	// The root expression has depth 0.
	Depth() int
}

// NavigateAST converts an AST to a NavigableExpr
func NavigateAST(ast *AST) NavigableExpr {
	return NavigateExpr(ast, ast.Expr())
}

// NavigateExpr creates a NavigableExpr whose type information is backed by the input AST.
//
// If the expression is already a NavigableExpr, the parent and depth information will be
// propagated on the new NavigableExpr value; otherwise, the expr value will be treated
// as though it is the root of the expression graph with a depth of 0.
func NavigateExpr(ast *AST, expr Expr) NavigableExpr {
	depth := 0
	var parent NavigableExpr = nil
	if nav, ok := expr.(NavigableExpr); ok {
		depth = nav.Depth()
		parent, _ = nav.Parent()
	}
	return newNavigableExpr(ast, parent, expr, depth)
}

// ExprMatcher takes a NavigableExpr in and indicates whether the value is a match.
//
// This function type should be use with the `Match` and `MatchList` calls.
type ExprMatcher func(NavigableExpr) bool

// ConstantValueMatcher returns an ExprMatcher which will return true if the input NavigableExpr
// is comprised of all constant values, such as a simple literal or even list and map literal.
func ConstantValueMatcher() ExprMatcher {
	return matchIsConstantValue
}

// KindMatcher returns an ExprMatcher which will return true if the input NavigableExpr.Kind() matches
// the specified `kind`.
func KindMatcher(kind ExprKind) ExprMatcher {
	return func(e NavigableExpr) bool {
		return e.Kind() == kind
	}
}

// FunctionMatcher returns an ExprMatcher which will match NavigableExpr nodes of CallKind type whose
// function name is equal to `funcName`.
func FunctionMatcher(funcName string) ExprMatcher {
	return func(e NavigableExpr) bool {
		if e.Kind() != CallKind {
			return false
		}
		return e.AsCall().FunctionName() == funcName
	}
}

// AllMatcher returns true for all descendants of a NavigableExpr, effectively flattening them into a list.
//
// Such a result would work well with subsequent MatchList calls.
func AllMatcher() ExprMatcher {
	return func(NavigableExpr) bool {
		return true
	}
}

// MatchDescendants takes a NavigableExpr and ExprMatcher and produces a list of NavigableExpr values
// matching the input criteria in post-order (bottom up).
func MatchDescendants(expr NavigableExpr, matcher ExprMatcher) []NavigableExpr {
	matches := []NavigableExpr{}
	navVisitor := &baseVisitor{
		visitExpr: func(e Expr) {
			nav := e.(NavigableExpr)
			if matcher(nav) {
				matches = append(matches, nav)
			}
		},
	}
	visit(expr, navVisitor, postOrder, 0, 0)
	return matches
}

// MatchSubset applies an ExprMatcher to a list of NavigableExpr values and their descendants, producing a
// subset of NavigableExpr values which match.
func MatchSubset(exprs []NavigableExpr, matcher ExprMatcher) []NavigableExpr {
	matches := []NavigableExpr{}
	navVisitor := &baseVisitor{
		visitExpr: func(e Expr) {
			nav := e.(NavigableExpr)
			if matcher(nav) {
				matches = append(matches, nav)
			}
		},
	}
	for _, expr := range exprs {
		visit(expr, navVisitor, postOrder, 0, 1)
	}
	return matches
}

// Visitor defines an object for visiting Expr and EntryExpr nodes within an expression graph.
type Visitor interface {
	// VisitExpr visits the input expression.
	VisitExpr(Expr)

	// VisitEntryExpr visits the input entry expression, i.e. a struct field or map entry.
	VisitEntryExpr(EntryExpr)
}

type baseVisitor struct {
	visitExpr      func(Expr)
	visitEntryExpr func(EntryExpr)
}

// VisitExpr visits the Expr if the internal expr visitor has been configured.
func (v *baseVisitor) VisitExpr(e Expr) {
	if v.visitExpr != nil {
		v.visitExpr(e)
	}
}

// VisitEntryExpr visits the entry if the internal expr entry visitor has been configured.
func (v *baseVisitor) VisitEntryExpr(e EntryExpr) {
	if v.visitEntryExpr != nil {
		v.visitEntryExpr(e)
	}
}

// NewExprVisitor creates a visitor which only visits expression nodes.
func NewExprVisitor(v func(Expr)) Visitor {
	return &baseVisitor{
		visitExpr:      v,
		visitEntryExpr: nil,
	}
}

// PostOrderVisit walks the expression graph and calls the visitor in post-order (bottom-up).
func PostOrderVisit(expr Expr, visitor Visitor) {
	visit(expr, visitor, postOrder, 0, 0)
}

// PreOrderVisit walks the expression graph and calls the visitor in pre-order (top-down).
func PreOrderVisit(expr Expr, visitor Visitor) {
	visit(expr, visitor, preOrder, 0, 0)
}

type visitOrder int

const (
	preOrder = iota + 1
	postOrder
)

// TODO: consider exposing a way to configure a limit for the max visit depth.
// It's possible that we could want to configure this on the NewExprVisitor()
// and through MatchDescendents() / MaxID().
func visit(expr Expr, visitor Visitor, order visitOrder, depth, maxDepth int) {
	if maxDepth > 0 && depth == maxDepth {
		return
	}
	if order == preOrder {
		visitor.VisitExpr(expr)
	}
	switch expr.Kind() {
	case CallKind:
		c := expr.AsCall()
		if c.IsMemberFunction() {
			visit(c.Target(), visitor, order, depth+1, maxDepth)
		}
		for _, arg := range c.Args() {
			visit(arg, visitor, order, depth+1, maxDepth)
		}
	case ComprehensionKind:
		c := expr.AsComprehension()
		visit(c.IterRange(), visitor, order, depth+1, maxDepth)
		visit(c.AccuInit(), visitor, order, depth+1, maxDepth)
		visit(c.LoopCondition(), visitor, order, depth+1, maxDepth)
		visit(c.LoopStep(), visitor, order, depth+1, maxDepth)
		visit(c.Result(), visitor, order, depth+1, maxDepth)
	case ListKind:
		l := expr.AsList()
		for _, elem := range l.Elements() {
			visit(elem, visitor, order, depth+1, maxDepth)
		}
	case MapKind:
		m := expr.AsMap()
		for _, e := range m.Entries() {
			if order == preOrder {
				visitor.VisitEntryExpr(e)
			}
			entry := e.AsMapEntry()
			visit(entry.Key(), visitor, order, depth+1, maxDepth)
			visit(entry.Value(), visitor, order, depth+1, maxDepth)
			if order == postOrder {
				visitor.VisitEntryExpr(e)
			}
		}
	case SelectKind:
		visit(expr.AsSelect().Operand(), visitor, order, depth+1, maxDepth)
	case StructKind:
		s := expr.AsStruct()
		for _, f := range s.Fields() {
			visitor.VisitEntryExpr(f)
			visit(f.AsStructField().Value(), visitor, order, depth+1, maxDepth)
		}
	}
	if order == postOrder {
		visitor.VisitExpr(expr)
	}
}

func matchIsConstantValue(e NavigableExpr) bool {
	if e.Kind() == LiteralKind {
		return true
	}
	if e.Kind() == StructKind || e.Kind() == MapKind || e.Kind() == ListKind {
		for _, child := range e.Children() {
			if !matchIsConstantValue(child) {
				return false
			}
		}
		return true
	}
	return false
}

func newNavigableExpr(ast *AST, parent NavigableExpr, expr Expr, depth int) NavigableExpr {
	// Reduce navigable expression nesting by unwrapping the embedded Expr value.
	if nav, ok := expr.(*navigableExprImpl); ok {
		expr = nav.Expr
	}
	nav := &navigableExprImpl{
		Expr:           expr,
		depth:          depth,
		ast:            ast,
		parent:         parent,
		createChildren: getChildFactory(expr),
	}
	return nav
}

type navigableExprImpl struct {
	Expr
	depth          int
	ast            *AST
	parent         NavigableExpr
	createChildren childFactory
}

func (nav *navigableExprImpl) Parent() (NavigableExpr, bool) {
	if nav.parent != nil {
		return nav.parent, true
	}
	return nil, false
}

func (nav *navigableExprImpl) ID() int64 {
	return nav.Expr.ID()
}

func (nav *navigableExprImpl) Kind() ExprKind {
	return nav.Expr.Kind()
}

func (nav *navigableExprImpl) Type() *types.Type {
	return nav.ast.GetType(nav.ID())
}

func (nav *navigableExprImpl) Children() []NavigableExpr {
	return nav.createChildren(nav)
}

func (nav *navigableExprImpl) Depth() int {
	return nav.depth
}

func (nav *navigableExprImpl) AsCall() CallExpr {
	return navigableCallImpl{navigableExprImpl: nav}
}

func (nav *navigableExprImpl) AsComprehension() ComprehensionExpr {
	return navigableComprehensionImpl{navigableExprImpl: nav}
}

func (nav *navigableExprImpl) AsIdent() string {
	return nav.Expr.AsIdent()
}

func (nav *navigableExprImpl) AsList() ListExpr {
	return navigableListImpl{navigableExprImpl: nav}
}

func (nav *navigableExprImpl) AsLiteral() ref.Val {
	return nav.Expr.AsLiteral()
}

func (nav *navigableExprImpl) AsMap() MapExpr {
	return navigableMapImpl{navigableExprImpl: nav}
}

func (nav *navigableExprImpl) AsSelect() SelectExpr {
	return navigableSelectImpl{navigableExprImpl: nav}
}

func (nav *navigableExprImpl) AsStruct() StructExpr {
	return navigableStructImpl{navigableExprImpl: nav}
}

func (nav *navigableExprImpl) createChild(e Expr) NavigableExpr {
	return newNavigableExpr(nav.ast, nav, e, nav.depth+1)
}

func (nav *navigableExprImpl) isExpr() {}

type navigableCallImpl struct {
	*navigableExprImpl
}

func (call navigableCallImpl) FunctionName() string {
	return call.Expr.AsCall().FunctionName()
}

func (call navigableCallImpl) IsMemberFunction() bool {
	return call.Expr.AsCall().IsMemberFunction()
}

func (call navigableCallImpl) Target() Expr {
	t := call.Expr.AsCall().Target()
	if t != nil {
		return call.createChild(t)
	}
	return nil
}

func (call navigableCallImpl) Args() []Expr {
	args := call.Expr.AsCall().Args()
	navArgs := make([]Expr, len(args))
	for i, a := range args {
		navArgs[i] = call.createChild(a)
	}
	return navArgs
}

type navigableComprehensionImpl struct {
	*navigableExprImpl
}

func (comp navigableComprehensionImpl) IterRange() Expr {
	return comp.createChild(comp.Expr.AsComprehension().IterRange())
}

func (comp navigableComprehensionImpl) IterVar() string {
	return comp.Expr.AsComprehension().IterVar()
}

func (comp navigableComprehensionImpl) AccuVar() string {
	return comp.Expr.AsComprehension().AccuVar()
}

func (comp navigableComprehensionImpl) AccuInit() Expr {
	return comp.createChild(comp.Expr.AsComprehension().AccuInit())
}

func (comp navigableComprehensionImpl) LoopCondition() Expr {
	return comp.createChild(comp.Expr.AsComprehension().LoopCondition())
}

func (comp navigableComprehensionImpl) LoopStep() Expr {
	return comp.createChild(comp.Expr.AsComprehension().LoopStep())
}

func (comp navigableComprehensionImpl) Result() Expr {
	return comp.createChild(comp.Expr.AsComprehension().Result())
}

type navigableListImpl struct {
	*navigableExprImpl
}

func (l navigableListImpl) Elements() []Expr {
	pbElems := l.Expr.AsList().Elements()
	elems := make([]Expr, len(pbElems))
	for i := 0; i < len(pbElems); i++ {
		elems[i] = l.createChild(pbElems[i])
	}
	return elems
}

func (l navigableListImpl) IsOptional(index int32) bool {
	return l.Expr.AsList().IsOptional(index)
}

func (l navigableListImpl) OptionalIndices() []int32 {
	return l.Expr.AsList().OptionalIndices()
}

func (l navigableListImpl) Size() int {
	return l.Expr.AsList().Size()
}

type navigableMapImpl struct {
	*navigableExprImpl
}

func (m navigableMapImpl) Entries() []EntryExpr {
	mapExpr := m.Expr.AsMap()
	entries := make([]EntryExpr, len(mapExpr.Entries()))
	for i, e := range mapExpr.Entries() {
		entry := e.AsMapEntry()
		entries[i] = &entryExpr{
			id: e.ID(),
			entryExprKindCase: navigableEntryImpl{
				key:   m.createChild(entry.Key()),
				val:   m.createChild(entry.Value()),
				isOpt: entry.IsOptional(),
			},
		}
	}
	return entries
}

func (m navigableMapImpl) Size() int {
	return m.Expr.AsMap().Size()
}

type navigableEntryImpl struct {
	key   NavigableExpr
	val   NavigableExpr
	isOpt bool
}

func (e navigableEntryImpl) Kind() EntryExprKind {
	return MapEntryKind
}

func (e navigableEntryImpl) Key() Expr {
	return e.key
}

func (e navigableEntryImpl) Value() Expr {
	return e.val
}

func (e navigableEntryImpl) IsOptional() bool {
	return e.isOpt
}

func (e navigableEntryImpl) renumberIDs(IDGenerator) {}

func (e navigableEntryImpl) isEntryExpr() {}

type navigableSelectImpl struct {
	*navigableExprImpl
}

func (sel navigableSelectImpl) FieldName() string {
	return sel.Expr.AsSelect().FieldName()
}

func (sel navigableSelectImpl) IsTestOnly() bool {
	return sel.Expr.AsSelect().IsTestOnly()
}

func (sel navigableSelectImpl) Operand() Expr {
	return sel.createChild(sel.Expr.AsSelect().Operand())
}

type navigableStructImpl struct {
	*navigableExprImpl
}

func (s navigableStructImpl) TypeName() string {
	return s.Expr.AsStruct().TypeName()
}

func (s navigableStructImpl) Fields() []EntryExpr {
	fieldInits := s.Expr.AsStruct().Fields()
	fields := make([]EntryExpr, len(fieldInits))
	for i, f := range fieldInits {
		field := f.AsStructField()
		fields[i] = &entryExpr{
			id: f.ID(),
			entryExprKindCase: navigableFieldImpl{
				name:  field.Name(),
				val:   s.createChild(field.Value()),
				isOpt: field.IsOptional(),
			},
		}
	}
	return fields
}

type navigableFieldImpl struct {
	name  string
	val   NavigableExpr
	isOpt bool
}

func (f navigableFieldImpl) Kind() EntryExprKind {
	return StructFieldKind
}

func (f navigableFieldImpl) Name() string {
	return f.name
}

func (f navigableFieldImpl) Value() Expr {
	return f.val
}

func (f navigableFieldImpl) IsOptional() bool {
	return f.isOpt
}

func (f navigableFieldImpl) renumberIDs(IDGenerator) {}

func (f navigableFieldImpl) isEntryExpr() {}

func getChildFactory(expr Expr) childFactory {
	if expr == nil {
		return noopFactory
	}
	switch expr.Kind() {
	case LiteralKind:
		return noopFactory
	case IdentKind:
		return noopFactory
	case SelectKind:
		return selectFactory
	case CallKind:
		return callArgFactory
	case ListKind:
		return listElemFactory
	case MapKind:
		return mapEntryFactory
	case StructKind:
		return structEntryFactory
	case ComprehensionKind:
		return comprehensionFactory
	default:
		return noopFactory
	}
}

type childFactory func(*navigableExprImpl) []NavigableExpr

func noopFactory(*navigableExprImpl) []NavigableExpr {
	return nil
}

func selectFactory(nav *navigableExprImpl) []NavigableExpr {
	return []NavigableExpr{nav.createChild(nav.AsSelect().Operand())}
}

func callArgFactory(nav *navigableExprImpl) []NavigableExpr {
	call := nav.Expr.AsCall()
	argCount := len(call.Args())
	if call.IsMemberFunction() {
		argCount++
	}
	navExprs := make([]NavigableExpr, argCount)
	i := 0
	if call.IsMemberFunction() {
		navExprs[i] = nav.createChild(call.Target())
		i++
	}
	for _, arg := range call.Args() {
		navExprs[i] = nav.createChild(arg)
		i++
	}
	return navExprs
}

func listElemFactory(nav *navigableExprImpl) []NavigableExpr {
	l := nav.Expr.AsList()
	navExprs := make([]NavigableExpr, len(l.Elements()))
	for i, e := range l.Elements() {
		navExprs[i] = nav.createChild(e)
	}
	return navExprs
}

func structEntryFactory(nav *navigableExprImpl) []NavigableExpr {
	s := nav.Expr.AsStruct()
	entries := make([]NavigableExpr, len(s.Fields()))
	for i, e := range s.Fields() {
		f := e.AsStructField()
		entries[i] = nav.createChild(f.Value())
	}
	return entries
}

func mapEntryFactory(nav *navigableExprImpl) []NavigableExpr {
	m := nav.Expr.AsMap()
	entries := make([]NavigableExpr, len(m.Entries())*2)
	j := 0
	for _, e := range m.Entries() {
		mapEntry := e.AsMapEntry()
		entries[j] = nav.createChild(mapEntry.Key())
		entries[j+1] = nav.createChild(mapEntry.Value())
		j += 2
	}
	return entries
}

func comprehensionFactory(nav *navigableExprImpl) []NavigableExpr {
	compre := nav.Expr.AsComprehension()
	return []NavigableExpr{
		nav.createChild(compre.IterRange()),
		nav.createChild(compre.AccuInit()),
		nav.createChild(compre.LoopCondition()),
		nav.createChild(compre.LoopStep()),
		nav.createChild(compre.Result()),
	}
}
