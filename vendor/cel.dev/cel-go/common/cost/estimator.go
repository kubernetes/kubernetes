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

package cost

import (
	"fmt"
	"math"
	"strconv"

	"cel.dev/cel-go/common/ast"
	"cel.dev/cel-go/common/operators"
	"cel.dev/cel-go/common/overloads"
	"cel.dev/cel-go/common/types"
)

// Estimator provides CallCost and Size estimates for cost calculation.
type Estimator interface {

	// EstimateCallCost returns the CallEstimate for a function, overload and target and arguments.
	//
	// If nil is returned, the cost of the function is calculated by the OverloadCostEstimate if
	// provided, or standard CEL function cost estimation.
	EstimateCallCost(function, overloadID string, target *AstNode, args []AstNode) *CallEstimate

	// EstimateSize returns the SizeEstimate of an AstNode.
	//
	// If nil is returned, the size of the node is calculated by the SizingStrategy.
	EstimateSize(element AstNode) *SizeEstimate
}

// AstNode represents an AST node for estimating cost.
type AstNode interface {
	// Path returns a field path through the provided type declarations to the type of the AstNode, or nil if the AstNode does not
	// represent type directly reachable from the provided type declarations.
	// The first path element is a variable. All subsequent path elements are one of: field name, '@items', '@keys', '@values'.
	Path() []string

	// Type returns the deduced type of the AstNode.
	Type() *types.Type

	// Expr returns the expression of the AstNode.
	Expr() ast.Expr

	// ComputedSize returns a size estimate of the AstNode derived from information available in the CEL expression.
	// For constants and inline list and map declarations, the exact size is returned. For concatenated list, strings
	// and bytes, the size is derived from the size estimates of the operands. nil is returned if there is no
	// computed size available.
	ComputedSize() *SizeEstimate
}

// astNode implements the AstNode interface.
type astNode struct {
	path        []string
	t           *types.Type
	expr        ast.Expr
	derivedSize *SizeEstimate
}

// Path returns the field path to the node, if known.
func (e astNode) Path() []string {
	return e.path
}

// Type returns the type of the node.
func (e astNode) Type() *types.Type {
	return e.t
}

// Expr returns the underlying CEL expression.
func (e astNode) Expr() ast.Expr {
	return e.expr
}

// ComputedSize returns the precomputed or derived size estimate of the node.
func (e astNode) ComputedSize() *SizeEstimate {
	return e.derivedSize
}

// NewAstNode returns an AstNode with the given expression, path, type, and derived size.
func NewAstNode(expr ast.Expr, path []string, t *types.Type, derivedSize *SizeEstimate) AstNode {
	return astNode{
		path:        path,
		t:           t,
		expr:        expr,
		derivedSize: derivedSize,
	}
}

// Option configures flags which affect cost computations.
type Option interface {
	applyCoster(*coster) error
}

type costerOption func(*coster) error

func (opt costerOption) applyCoster(c *coster) error {
	return opt(c)
}

// CostOption is an alias for Option.
//
// Deprecated: use Option.
type CostOption = Option

// PresenceTestHasCost determines whether presence testing has a cost of one or zero.
//
// Defaults to presence test has a cost of one.
func PresenceTestHasCost(hasCost bool) Option {
	return costerOption(func(c *coster) error {
		if hasCost {
			c.presenceTestCost = selectAndIdentCost
			return nil
		}
		c.presenceTestCost = FixedCostEstimate(0)
		return nil
	})
}

// FunctionEstimator provides a CallEstimate given the target and arguments for a specific function, overload pair.
type FunctionEstimator func(estimator Estimator, target *AstNode, args []AstNode) *CallEstimate

// OverloadCostEstimate binds a FunctionEstimator to a specific function overload ID.
//
// When an OverloadCostEstimate is provided, it overrides the cost calculation of the Estimator
// provided to the Cost() call.
func OverloadCostEstimate(overloadID string, functionCoster FunctionEstimator) Option {
	return costerOption(func(c *coster) error {
		if functionCoster == nil {
			return fmt.Errorf("OverloadCostEstimate(%q) requires a FunctionEstimator", overloadID)
		}
		c.overloadEstimators.mutable()[overloadID] = functionCoster
		return nil
	})
}

// Cost estimates the cost of the parsed and type checked CEL expression.
func Cost(checked *ast.AST, estimator Estimator, opts ...Option) (CostEstimate, error) {
	c, err := newCoster(checked, estimator, opts...)
	if err != nil {
		return CostEstimate{}, err
	}
	return c.cost(checked.Expr()), nil
}

// newCoster returns a coster configured by the options. Its configuration is fixed once it is
// returned; only the per-expression state gathered while costing changes.
func newCoster(checked *ast.AST, estimator Estimator, opts ...Option) (*coster, error) {
	c := &coster{
		checkedAST:       checked,
		estimator:        estimator,
		modelRules:       defaultModelRules(),
		exprPaths:        map[int64][]string{},
		localVars:        make(scopes),
		computedSizes:    map[int64]SizeEstimate{},
		presenceTestCost: FixedCostEstimate(1),
	}
	for _, opt := range opts {
		if mo, ok := opt.(ModelOption); ok {
			c.modelRules.override(mo)
		}
	}
	for _, opt := range opts {
		if err := opt.applyCoster(c); err != nil {
			return nil, err
		}
	}
	// The package-level estimators are compiled under the default rules.
	c.standardOverloadEstimators = stdOverloadEstimators
	if !c.modelRules.isDefault() {
		c.standardOverloadEstimators = StandardOverloadEstimatorsWithOptions(c.modelRules.modelOptions()...)
	}
	c.est = versionedEstimator{
		coster:   c,
		delegate: estimator,
		version:  c.modelVersion,
		strategy: c.sizingStrategy,
	}
	c.evalCtx = estimatorEvalContext{
		coster:    c,
		estimator: estimator,
		strategy:  c.sizingStrategy,
		version:   c.modelVersion,
	}
	return c, nil
}

// coster performs recursive cost and size calculation on checked AST expressions.
type coster struct {
	// exprPaths maps from Expr Id to field path.
	exprPaths map[int64][]string
	// localVars tracks the local and iteration variables assigned during evaluation.
	localVars scopes
	// computedSizes tracks the computed sizes of call results.
	computedSizes map[int64]SizeEstimate

	checkedAST *ast.AST
	estimator  Estimator
	modelRules
	overloadEstimators         overloadTable[FunctionEstimator]
	standardOverloadEstimators map[string]FunctionEstimator
	// presenceTestCost will either be a zero or one based on whether has() macros count against cost computations.
	presenceTestCost CostEstimate
	// est and evalCtx are owned by the coster and handed out by pointer, so that passing them to
	// estimators and sizing strategies does not allocate.
	est     versionedEstimator
	evalCtx estimatorEvalContext
}

// localVar captures the local variable size estimates if they exist for variables
type localVar struct {
	exprID int64
	path   []string
	size   *SizeEstimate
}

// scopes is a stack of variable name to integer id stack to handle scopes created by cel.bind() like macros
type scopes map[string][]*localVar

// push adds a variable name to the scope stack with its path and size estimates.
func (s scopes) push(varName string, expr ast.Expr, path []string, size *SizeEstimate) {
	s[varName] = append(s[varName], &localVar{
		exprID: expr.ID(),
		path:   path,
		size:   size,
	})
}

// pop removes the most recent variable from the scope stack for the given name.
func (s scopes) pop(varName string) {
	varStack := s[varName]
	s[varName] = varStack[:len(varStack)-1]
}

// peek returns the top of the stack for the given variable name.
func (s scopes) peek(varName string) (*localVar, bool) {
	varStack := s[varName]
	if len(varStack) > 0 {
		return varStack[len(varStack)-1], true
	}
	return nil, false
}

// pushIterKey pushes the iteration key or index variable for a comprehension onto the scope stack.
func (c *coster) pushIterKey(varName string, rangeExpr ast.Expr) {
	rangeSize := c.sizeOrUnknown(rangeExpr)
	container := c.getType(rangeExpr).Kind()
	var size *SizeEstimate
	subpath := "@keys"
	if container == types.ListKind {
		s := FixedSizeEstimate(1)
		size = &s
		subpath = "@indices"
	} else if rangeSize.Key != nil {
		size = rangeSize.Key
	}
	var varPath []string
	if path := c.getPath(rangeExpr); len(path) > 0 || c.modelVersion < 1 {
		varPath = append(path, subpath)
	}
	c.localVars.push(varName, rangeExpr, varPath, size)
}

// pushIterValue pushes the iteration value variable for a comprehension onto the scope stack.
func (c *coster) pushIterValue(varName string, rangeExpr ast.Expr) {
	rangeSize := c.sizeOrUnknown(rangeExpr)
	var size *SizeEstimate
	if rangeSize.Elem != nil {
		size = rangeSize.Elem
	}
	container := c.getType(rangeExpr).Kind()
	subpath := "@values"
	if container == types.ListKind {
		subpath = "@items"
	}
	var varPath []string
	if path := c.getPath(rangeExpr); len(path) > 0 || c.modelVersion < 1 {
		varPath = append(path, subpath)
	}
	c.localVars.push(varName, rangeExpr, varPath, size)
}

// pushIterSingle pushes a single iteration variable (items for list, keys for map) onto the scope stack.
func (c *coster) pushIterSingle(varName string, rangeExpr ast.Expr) {
	rangeSize := c.sizeOrUnknown(rangeExpr)
	var size *SizeEstimate
	subpath := "@keys"
	container := c.getType(rangeExpr).Kind()
	if container == types.ListKind || (container == types.DynKind && rangeSize.Key == nil && rangeSize.Elem != nil) {
		size = rangeSize.Elem
		subpath = "@items"
	} else {
		size = rangeSize.Key
	}
	var varPath []string
	if path := c.getPath(rangeExpr); len(path) > 0 || c.modelVersion < 1 {
		varPath = append(path, subpath)
	}
	c.localVars.push(varName, rangeExpr, varPath, size)
}

// pushLocalVar records a local variable binding with its path and size estimates.
func (c *coster) pushLocalVar(varName string, e ast.Expr) {
	path := c.getPath(e)
	c.localVars.push(varName, e, path, ownedSize(c.computeSize(e)))
}

// peekLocalVar looks up the top of the scope stack for a local variable name.
func (c *coster) peekLocalVar(varName string) (*localVar, bool) {
	return c.localVars.peek(varName)
}

// popLocalVar pops the scope stack for a local variable name.
func (c *coster) popLocalVar(varName string) {
	c.localVars.pop(varName)
}

// cost recursively calculates the cost estimate for an expression node.
func (c *coster) cost(e ast.Expr) CostEstimate {
	if e == nil {
		return CostEstimate{}
	}
	switch e.Kind() {
	case ast.LiteralKind:
		return constCost
	case ast.IdentKind:
		return c.costIdent(e)
	case ast.SelectKind:
		return c.costSelect(e)
	case ast.CallKind:
		return c.costCall(e)
	case ast.ListKind:
		return c.costCreateList(e)
	case ast.MapKind:
		return c.costCreateMap(e)
	case ast.StructKind:
		return c.costCreateStruct(e)
	case ast.ComprehensionKind:
		if c.isBind(e) {
			return c.costBind(e)
		}
		return c.costComprehension(e)
	default:
		return CostEstimate{}
	}
}

// costIdent estimates the cost of evaluating an identifier and tracks its path.
func (c *coster) costIdent(e ast.Expr) CostEstimate {
	ident := e.AsIdent()
	if v, ok := c.peekLocalVar(ident); ok {
		c.addPath(e, v.path)
	} else {
		c.addPath(e, []string{ident})
	}
	return selectAndIdentCost
}

// costSelect estimates the cost of a field selection or map key access.
func (c *coster) costSelect(e ast.Expr) CostEstimate {
	sel := e.AsSelect()
	var sum CostEstimate
	if sel.IsTestOnly() {
		sum = sum.Add(c.presenceTestCost)
		sum = sum.Add(branchQualifierCost(sel.Operand()))
	} else {
		sum = sum.Add(selectAndIdentCost)
	}
	sum = sum.Add(c.cost(sel.Operand()))
	sum = sum.Add(c.relativeAttributeCost(sel.Operand()))
	targetPath := c.getPath(sel.Operand())
	if len(targetPath) > 0 {
		c.addPath(e, append(targetPath, sel.FieldName()))
	}
	return sum
}

// branchQualifierCost is the cost of the qualification a presence test applies to the branch of a
// conditional at the root of its operand's attribute chain. The interpreter charges it when the
// selected branch is computed. When the branch is an attribute, the estimate already accounts for
// it with the cost of the branch's root, which the interpreter resolves without charging.
func branchQualifierCost(operand ast.Expr) CostEstimate {
	root := attributeChainRoot(operand)
	if !isConditionalCall(root) {
		return CostEstimate{}
	}
	args := root.AsCall().Args()
	if len(args) != 3 {
		return CostEstimate{}
	}
	branch := func(e ast.Expr) CostEstimate {
		if !isAttributeChain(e) || (e.Kind() == ast.SelectKind && e.AsSelect().IsTestOnly()) {
			return FixedCostEstimate(SelectAndIdentCost)
		}
		return branchQualifierCost(e)
	}
	return branch(args[1]).Union(branch(args[2]))
}

// attributeChainRoot returns the expression an attribute chain starts from, past the selections
// and index operations which extend it.
func attributeChainRoot(e ast.Expr) ast.Expr {
	for {
		switch e.Kind() {
		case ast.SelectKind:
			if e.AsSelect().IsTestOnly() {
				return e
			}
			e = e.AsSelect().Operand()
		case ast.CallKind:
			call := e.AsCall()
			switch call.FunctionName() {
			case operators.Index, operators.OptIndex, operators.OptSelect:
				if len(call.Args()) > 0 {
					e = call.Args()[0]
					continue
				}
			}
			return e
		default:
			return e
		}
	}
}

// relativeAttributeCost is the cost of qualifying a value which was computed rather than named.
func (c *coster) relativeAttributeCost(operand ast.Expr) CostEstimate {
	if c.modelVersion < 1 || isAttributeChain(operand) {
		return CostEstimate{}
	}
	return FixedCostEstimate(SelectAndIdentCost)
}

// isAttributeChain reports whether an expression is resolved as part of a single attribute
// during evaluation. A chain begins at an identifier, or at a ternary which selects between
// attributes, and is extended by field selections and index operations, including their
// optional variants.
//
// This predicate must mirror the planner's decision to wrap an operand in a relative
// attribute: any expression which plans to an interpretable attribute extends the chain and
// does not incur an additional attribute resolution cost.
func isAttributeChain(e ast.Expr) bool {
	switch e.Kind() {
	case ast.IdentKind, ast.SelectKind:
		return true
	case ast.CallKind:
		switch e.AsCall().FunctionName() {
		case operators.Index, operators.OptIndex, operators.OptSelect, operators.Conditional:
			return true
		}
	}
	return false
}

// costCall estimates the cost of evaluating a function call expression.
func (c *coster) costCall(e ast.Expr) CostEstimate {
	if e.AsCall().FunctionName() == "cel.@block" {
		return c.costBlock(e)
	}
	// Dyn is just a way to disable type-checking, so return the cost of 1 with the cost of the argument
	if dynEstimate := c.maybeUnwrapDynCall(e); dynEstimate != nil {
		return *dynEstimate
	}

	// Continue estimating the cost of all other calls.
	call := e.AsCall()
	args := call.Args()
	var sum CostEstimate

	// Index-like operators qualify their first argument, which requires a relative attribute
	// when the operand is a computed value rather than a named one.
	switch call.FunctionName() {
	case operators.Index, operators.OptIndex, operators.OptSelect:
		if len(args) > 0 {
			sum = sum.Add(c.relativeAttributeCost(args[0]))
		}
	}

	var argCostBuf [4]CostEstimate
	var argCosts []CostEstimate
	if len(args) <= len(argCostBuf) {
		argCosts = argCostBuf[:len(args)]
	} else {
		argCosts = make([]CostEstimate, len(args))
	}
	for i, arg := range args {
		argCosts[i] = c.cost(arg)
	}
	if call.FunctionName() == operators.Conditional && len(args) == 3 && c.modelVersion >= 1 {
		argCosts[1] = conditionalBranchCost(args[1], argCosts[1])
		argCosts[2] = conditionalBranchCost(args[2], argCosts[2])
	}

	overloadIDs := c.checkedAST.GetOverloadIDs(e.ID())
	if len(overloadIDs) == 0 {
		var argCostSum CostEstimate
		for _, a := range argCosts {
			argCostSum = argCostSum.Add(a)
		}
		return FixedCostEstimate(1).Add(sum).Add(argCostSum)
	}
	if call.IsMemberFunction() {
		sum = sum.Add(c.cost(call.Target()))
	}
	var argTypeBuf [4]AstNode
	var argTypes []AstNode
	var targetNode AstNode
	var targetType *AstNode
	if c.needsCallNodes(overloadIDs) {
		if len(args) <= len(argTypeBuf) {
			argTypes = argTypeBuf[:len(args)]
		} else {
			argTypes = make([]AstNode, len(args))
		}
		for i, arg := range args {
			argTypes[i] = c.newAstNode(arg)
		}
		if call.IsMemberFunction() {
			targetNode = c.newAstNode(call.Target())
			targetType = &targetNode
		}
	}
	// Pick a cost estimate range that covers all the overload cost estimation ranges
	fnCost := CostEstimate{Min: uint64(math.MaxUint64), Max: 0}
	var resultSize *SizeEstimate
	for _, overload := range overloadIDs {
		overloadCost := c.functionCost(e, call.FunctionName(), overload, targetType, argTypes, argCosts)
		fnCost = fnCost.Union(overloadCost.CostEstimate)
		overloadSize := overloadCost.ResultSize
		optIndex := c.modelVersion >= 1 && isOptionalIndex(overload)
		// build and track the field path for index operations
		if field, ok := c.indexFieldPath(overload); ok && len(args) > 0 {
			if targetPath := c.getPath(args[0]); len(targetPath) > 0 {
				c.addPath(e, append(targetPath, field))
			} else if containerSize := c.indexedContainerSize(overload, args[0]); containerSize != nil && containerSize.Elem != nil {
				elemSize := *containerSize.Elem
				overloadSize = &elemSize
			}
		}
		if overloadSize == nil {
			if optIndex {
				overloadSize = c.optionalValueSize(e)
			} else {
				overloadSize = c.computeSize(e)
			}
		}
		if optIndex && overloadSize != nil {
			wrapped := optionalSizeOf(*overloadSize)
			overloadSize = &wrapped
		}
		resultSize = mergeSizeEstimatePtr(resultSize, overloadSize)
	}
	c.setSize(e, resultSize)
	return sum.Add(fnCost)
}

// isOptionalIndex reports whether an overload is an optional index operation, l[?i] or m[?k], which
// returns an optional holding the element it selects.
func isOptionalIndex(overloadID string) bool {
	switch overloadID {
	case overloads.OptIndexList, overloads.OptIndexMap, overloads.OptIndexOptionalList, overloads.OptIndexOptionalMap:
		return true
	}
	return false
}

// indexedContainerSize returns the size of the container an index operation selects from. From
// version 1 the operand of an optional index on an optional receiver is the optional, whose element
// is the container.
func (c *coster) indexedContainerSize(overloadID string, operand ast.Expr) *SizeEstimate {
	size := c.computeSize(operand)
	if size == nil || c.modelVersion < 1 {
		return size
	}
	switch overloadID {
	case overloads.OptIndexOptionalList, overloads.OptIndexOptionalMap:
		return size.Elem
	}
	return size
}

// optionalValueSize estimates the value an optional-typed expression holds from the size hints for
// its path and the value's type.
func (c *coster) optionalValueSize(e ast.Expr) *SizeEstimate {
	t := c.getType(e)
	if t.Kind() != types.OpaqueKind || len(t.Parameters()) != 1 {
		return nil
	}
	node := astNode{path: c.getPath(e), t: t.Parameters()[0]}
	if size, ok := c.sizingStrategy.EstimateSize(c.newEstimateContext(), node); ok {
		return &size
	}
	return nil
}

func (c *coster) needsCallNodes(overloadIDs []string) bool {
	if c.estimator != nil {
		return true
	}
	std := c.standardOverloadEstimators
	for _, id := range overloadIDs {
		if _, ok := c.overloadEstimators.fns[id]; ok {
			return true
		}
		if _, ok := std[id]; ok {
			return true
		}
	}
	return false
}

// indexFieldPath reports the field path segment an index overload qualifies its operand with, so
// that the size hint recorded for a container's elements is found when one of them is read.
//
// The optional forms count only from version 1. Before the revision `l[?0]` recorded no path,
// so the `@items` hint that `l[0]` finds was never consulted and every estimate below an optional
// index was unbounded. A pinned caller keeps that.
func (c *coster) indexFieldPath(overloadID string) (string, bool) {
	switch overloadID {
	case overloads.IndexList:
		return "@items", true
	case overloads.IndexMap:
		return "@values", true
	case overloads.OptIndexList, overloads.OptIndexOptionalList:
		return "@items", c.modelVersion >= 1
	case overloads.OptIndexMap, overloads.OptIndexOptionalMap:
		return "@values", c.modelVersion >= 1
	}
	return "", false
}

// maybeUnwrapDynCall handles the 'dyn' call wrapper, returning an estimate if matched.
func (c *coster) maybeUnwrapDynCall(e ast.Expr) *CostEstimate {
	call := e.AsCall()
	if call.FunctionName() != "dyn" {
		return nil
	}
	arg := call.Args()[0]
	argCost := c.cost(arg)
	c.copySizeEstimates(e, arg)
	callCost := FixedCostEstimate(1).Add(argCost)
	return &callCost
}

// costBlock estimates the cost of a cel.@block([slot0, slot1, ...], resultExpr) call,
// binding each slot expression to @index0, @index1, ... while evaluating the block result.
func (c *coster) costBlock(e ast.Expr) CostEstimate {
	call := e.AsCall()
	args := call.Args()
	if len(args) != 2 || args[0].Kind() != ast.ListKind {
		return FixedCostEstimate(1)
	}
	slots := args[0].AsList().Elements()
	var sum CostEstimate
	for i, slot := range slots {
		sum = sum.Add(c.cost(slot))
		c.pushLocalVar("@index"+strconv.Itoa(i), slot)
	}
	sum = sum.Add(c.cost(args[1]))
	if p := c.getPath(args[1]); len(p) > 0 {
		c.addPath(e, p)
	}
	c.copySizeEstimates(e, args[1])
	for i := len(slots) - 1; i >= 0; i-- {
		c.popLocalVar("@index" + strconv.Itoa(i))
	}
	return sum
}

// costCreateList estimates the cost of constructing a list literal.
func (c *coster) costCreateList(e ast.Expr) CostEstimate {
	create := e.AsList()
	var sum CostEstimate
	for _, elem := range create.Elements() {
		sum = sum.Add(c.cost(elem))
	}
	return sum.Add(createListBaseCost)
}

// costCreateMap estimates the cost of constructing a map literal.
func (c *coster) costCreateMap(e ast.Expr) CostEstimate {
	mapVal := e.AsMap()
	var sum CostEstimate
	for _, ent := range mapVal.Entries() {
		entry := ent.AsMapEntry()
		sum = sum.Add(c.cost(entry.Key()))
		sum = sum.Add(c.cost(entry.Value()))
	}
	return sum.Add(createMapBaseCost)
}

// costCreateStruct estimates the cost of constructing a struct or message literal.
func (c *coster) costCreateStruct(e ast.Expr) CostEstimate {
	msgVal := e.AsStruct()
	var sum CostEstimate
	for _, ent := range msgVal.Fields() {
		field := ent.AsStructField()
		sum = sum.Add(c.cost(field.Value()))
	}
	return sum.Add(createMessageBaseCost)
}

// costComprehension estimates the cost of evaluating a comprehension loop.
func (c *coster) costComprehension(e ast.Expr) CostEstimate {
	comp := e.AsComprehension()
	var sum CostEstimate
	sum = sum.Add(c.cost(comp.IterRange()))
	sum = sum.Add(c.cost(comp.AccuInit()))
	c.pushLocalVar(comp.AccuVar(), comp.AccuInit())

	// Track the iterRange of each IterVar and AccuVar for field path construction
	if comp.HasIterVar2() {
		c.pushIterKey(comp.IterVar(), comp.IterRange())
		c.pushIterValue(comp.IterVar2(), comp.IterRange())
	} else {
		c.pushIterSingle(comp.IterVar(), comp.IterRange())
	}

	// Determine the cost for each element in the loop
	loopCost := c.cost(comp.LoopCondition())
	stepCost := c.cost(comp.LoopStep())
	stepSize := c.computeSize(comp.LoopStep())

	// Clear the intermediate variable tracking.
	c.popLocalVar(comp.IterVar())
	if comp.HasIterVar2() {
		c.popLocalVar(comp.IterVar2())
	}

	// Determine the result cost.
	sum = sum.Add(c.cost(comp.Result()))
	c.localVars.pop(comp.AccuVar())

	// Estimate the cost of the loop.
	rangeSize := c.sizeOrUnknown(comp.IterRange())
	// The loop runs once per element or entry of the range, which the sizing strategy may
	// size by content rather than by length.
	rangeCnt := countBounds(c.sizingStrategy, rangeSize)
	rangeCnt.Key, rangeCnt.Elem = rangeSize.Key, rangeSize.Elem
	rangeCost := rangeCnt.MultiplyByCost(stepCost.Add(loopCost))
	sum = sum.Add(rangeCost)

	switch k := comp.AccuInit().Kind(); k {
	case ast.LiteralKind:
		c.setSize(e, c.computeSize(comp.AccuInit()))
	case ast.ListKind, ast.MapKind:
		accuSize := rangeCnt
		if k == ast.ListKind {
			accuSize.Key = nil
		}
		if stepSize != nil {
			if stepSize.Elem != nil {
				accuSize.Elem = stepSize.Elem
			}
			if k == ast.MapKind && stepSize.Key != nil {
				accuSize.Key = stepSize.Key
			}
		}
		// A conditional step, as in filter, may skip an iteration, so the container may hold
		// no entries at all.
		if isConditionalCall(comp.LoopStep()) {
			accuSize.Min = 0
		}
		// The accumulated container holds at most one entry per iteration; report the size the
		// sizing strategy assigns to it.
		sized := containerSizeFor(c.sizingStrategy, accuSize, accumulatedEntrySize(accuSize))
		sized.Key, sized.Elem = accuSize.Key, accuSize.Elem
		accuSize = sized
		c.setSize(e, &accuSize)
	}
	return sum
}

// accumulatedEntrySize returns the size of one entry of an accumulated container: its element, or
// its key and value. An unknown part is unbounded.
func accumulatedEntrySize(sz SizeEstimate) SizeEstimate {
	entry := UnknownSizeEstimate()
	if sz.Elem != nil {
		entry = RangedSizeEstimate(sz.Elem.Min, sz.Elem.Max)
	}
	if sz.Key != nil {
		withKey := entry.Add(RangedSizeEstimate(sz.Key.Min, sz.Key.Max))
		entry = RangedSizeEstimate(withKey.Min, withKey.Max)
	}
	return entry
}

// conditionalBranchCost is the cost of a conditional's branch from cost model version 1.
//
// A branch which is an attribute chain rooted at an identifier, such as x, m.a or x[0], is resolved
// as part of the conditional's attribute: the interpreter charges the condition and the branch's
// qualifiers, but not the resolution of the branch's root identifier, which the estimate of the
// branch includes. A branch rooted at a nested conditional is priced by that conditional, and a
// computed branch is charged in full.
func conditionalBranchCost(branch ast.Expr, cost CostEstimate) CostEstimate {
	if !isAttributeChain(branch) || attributeChainRoot(branch).Kind() != ast.IdentKind {
		return cost
	}
	if branch.Kind() == ast.SelectKind && branch.AsSelect().IsTestOnly() {
		return cost
	}
	return CostEstimate{
		Min: SafeSubtract(cost.Min, SelectAndIdentCost),
		Max: SafeSubtract(cost.Max, SelectAndIdentCost),
	}
}

// isConditionalCall reports whether e is a call to the conditional operator.
func isConditionalCall(e ast.Expr) bool {
	return e.Kind() == ast.CallKind && e.AsCall().FunctionName() == operators.Conditional
}

// isBind returns true if the given expression represents a cel.bind() macro structure.
func (c *coster) isBind(e ast.Expr) bool {
	comp := e.AsComprehension()
	iterRange := comp.IterRange()
	loopCond := comp.LoopCondition()
	return iterRange.Kind() == ast.ListKind && iterRange.AsList().Size() == 0 &&
		loopCond.Kind() == ast.LiteralKind && loopCond.AsLiteral() == types.False &&
		!isAccumulatorVar(comp.AccuVar())
}

// costBind estimates the cost of a cel.bind() variable declaration macro.
func (c *coster) costBind(e ast.Expr) CostEstimate {
	comp := e.AsComprehension()
	var sum CostEstimate
	// Binds are lazily initialized, so we retain the cost of an empty iteration range.
	sum = sum.Add(c.cost(comp.IterRange()))
	initCost := c.cost(comp.AccuInit())
	if c.modelVersion >= 1 && !mustReadVar(comp.Result(), comp.AccuVar()) {
		// The bound value is computed only if the result reads it, so unless every evaluation of
		// the result reads it, its cost contributes to the upper bound alone.
		initCost = CostEstimate{Min: 0, Max: initCost.Max}
	}
	sum = sum.Add(initCost)

	c.pushLocalVar(comp.AccuVar(), comp.AccuInit())
	sum = sum.Add(c.cost(comp.Result()))
	if p := c.getPath(comp.Result()); len(p) > 0 {
		c.addPath(e, p)
	}
	// Associate the bind output size with the result size before popping the bound variable.
	c.copySizeEstimates(e, comp.Result())
	c.popLocalVar(comp.AccuVar())

	return sum
}

// mustReadVar reports whether every evaluation of e reads the variable name. It is conservative:
// operands which may be skipped, such as the right side of a logical operator, the branches of a
// conditional, the alternative of an optional, and the body of a comprehension, do not count,
// and neither does the bound value of a nested bind, which is itself lazy.
func mustReadVar(e ast.Expr, name string) bool {
	switch e.Kind() {
	case ast.IdentKind:
		return e.AsIdent() == name
	case ast.SelectKind:
		return mustReadVar(e.AsSelect().Operand(), name)
	case ast.CallKind:
		call := e.AsCall()
		args := call.Args()
		switch call.FunctionName() {
		case operators.LogicalAnd, operators.LogicalOr, operators.Conditional:
			return len(args) > 0 && mustReadVar(args[0], name)
		case "or", "orValue":
			if call.IsMemberFunction() {
				return mustReadVar(call.Target(), name)
			}
		}
		if call.IsMemberFunction() && mustReadVar(call.Target(), name) {
			return true
		}
		for _, arg := range args {
			if mustReadVar(arg, name) {
				return true
			}
		}
		return false
	case ast.ListKind:
		for _, elem := range e.AsList().Elements() {
			if mustReadVar(elem, name) {
				return true
			}
		}
		return false
	case ast.MapKind:
		for _, entry := range e.AsMap().Entries() {
			me := entry.AsMapEntry()
			if mustReadVar(me.Key(), name) || mustReadVar(me.Value(), name) {
				return true
			}
		}
		return false
	case ast.StructKind:
		for _, field := range e.AsStruct().Fields() {
			if mustReadVar(field.AsStructField().Value(), name) {
				return true
			}
		}
		return false
	case ast.ComprehensionKind:
		comp := e.AsComprehension()
		if mustReadVar(comp.IterRange(), name) {
			return true
		}
		// A bind's value is lazy: it is computed only when its result reads it, and its result,
		// unless the bind shadows name, may read name directly. A loop's initial accumulator is
		// always computed.
		isBind := comp.IterRange().Kind() == ast.ListKind && comp.IterRange().AsList().Size() == 0 &&
			!isAccumulatorVar(comp.AccuVar())
		if isBind {
			if comp.AccuVar() != name && mustReadVar(comp.Result(), name) {
				return true
			}
			return mustReadVar(comp.Result(), comp.AccuVar()) && mustReadVar(comp.AccuInit(), name)
		}
		return mustReadVar(comp.AccuInit(), name)
	default:
		return false
	}
}

// calculateArgCost combines the costs of a call's arguments at the given cost model version.
func calculateArgCost(overloadID string, argCosts []CostEstimate, version uint32) CostEstimate {
	switch overloadID {
	case overloads.LogicalOr, overloads.LogicalAnd:
		if len(argCosts) == 2 {
			return CostEstimate{Min: argCosts[0].Min, Max: argCosts[0].Add(argCosts[1]).Max}
		}
	case overloads.Conditional:
		if len(argCosts) == 3 {
			return argCosts[0].Add(argCosts[1].Union(argCosts[2]))
		}
	case overloads.OptionalOrOptional, overloads.OptionalOrValueValue:
		// The alternative is evaluated only when the receiver is empty, so it contributes to the
		// upper bound alone. These are member overloads, which means the receiver is costed by
		// the caller and argCosts holds only the alternative. Before version 1 the alternative
		// was summed like any other argument.
		if len(argCosts) == 1 && version >= 1 {
			return CostEstimate{Min: 0, Max: argCosts[0].Max}
		}
	}
	var sum CostEstimate
	for _, a := range argCosts {
		sum = sum.Add(a)
	}
	return sum
}

func (c *coster) functionCost(e ast.Expr, function, overloadID string, target *AstNode, args []AstNode, argCosts []CostEstimate) CallEstimate {
	argCost := calculateArgCost(overloadID, argCosts, c.modelVersion)
	c.est.callID = e.ID()
	estCtx := c.getEstimator()
	if estimator, found := c.overloadEstimators.fns[overloadID]; found {
		if est := estimator(estCtx, target, args); est != nil {
			return CallEstimate{CostEstimate: est.Add(argCost), ResultSize: est.ResultSize}
		}
	}
	if c.estimator != nil {
		if est := c.estimator.EstimateCallCost(function, overloadID, target, args); est != nil {
			return CallEstimate{CostEstimate: est.Add(argCost), ResultSize: est.ResultSize}
		}
	}
	if estimator, found := c.standardOverloadEstimators[overloadID]; found {
		if est := estimator(estCtx, target, args); est != nil {
			return CallEstimate{CostEstimate: est.Add(argCost), ResultSize: est.ResultSize}
		}
	}
	// O(1) functions
	// See CostTracker.costCall for more details about O(1) cost calculations

	// Benchmarks suggest that most of the other operations take +/- 50% of a base cost unit
	// which on an Intel xeon 2.20GHz CPU is 50ns.
	return CallEstimate{CostEstimate: FixedCostEstimate(1).Add(argCost)}
}

// getType returns the deduced type of an expression ID from the checked AST.
func (c *coster) getType(e ast.Expr) *types.Type {
	return c.checkedAST.GetType(e.ID())
}

// getPath returns the tracked field path for an expression node, resolving through local variables if needed.
func (c *coster) getPath(e ast.Expr) []string {
	if e == nil {
		return nil
	}
	if e.Kind() == ast.IdentKind {
		if v, found := c.peekLocalVar(e.AsIdent()); found {
			return v.path[:]
		}
	}
	if path, ok := c.exprPaths[e.ID()]; ok {
		return path
	}
	return nil
}

// addPath associates an expression ID with its path.
func (c *coster) addPath(e ast.Expr, path []string) {
	c.exprPaths[e.ID()] = path
}

var (
	stdOverloadEstimators = StandardOverloadEstimators()
)

func isAccumulatorVar(name string) bool {
	return name == accumulatorName || name == hiddenAccumulatorName
}

func (c *coster) newAstNode(e ast.Expr) AstNode {
	return astNode{
		path:        c.getPath(e),
		t:           c.getType(e),
		expr:        e,
		derivedSize: ownedSize(c.computeSize(e))}
}

// ownedSize returns a size estimate pointer which the caller may retain or modify.
//
// computeSize returns the shared unitSizeEstimate and zeroSizeEstimate for scalars and empty or
// single-unit literals so that the common case does not allocate. Those values back every such
// estimate, so a pointer which escapes into a retained structure, or to a FunctionEstimator through
// AstNode.ComputedSize, is copied first; a write through it must never reach the shared values.
func ownedSize(sz *SizeEstimate) *SizeEstimate {
	if sz == &unitSizeEstimate || sz == &zeroSizeEstimate {
		owned := *sz
		return &owned
	}
	return sz
}

// setSize stores a computed size estimate for an expression ID.
func (c *coster) setSize(e ast.Expr, size *SizeEstimate) {
	if size == nil {
		return
	}
	// Store the computed size with the expression
	c.computedSizes[e.ID()] = *size
}

func (c *coster) sizeOrUnknown(node ast.Expr) SizeEstimate {
	if sz := c.computeSize(node); sz != nil {
		return *sz
	}
	return UnknownSizeEstimate()
}

// copySizeEstimates copies computed sizes and entry sizes from a source expression to a destination expression.
func (c *coster) copySizeEstimates(dst, src ast.Expr) {
	c.setSize(dst, c.computeSize(src))
}

type versionedEstimator struct {
	coster   *coster
	delegate Estimator
	version  uint32
	strategy SizingStrategy
	// callID is the expression ID of the call being estimated, if any.
	callID int64
}

func (v *versionedEstimator) EstimateSize(element AstNode) *SizeEstimate {
	if v.delegate == nil {
		return nil
	}
	return v.delegate.EstimateSize(element)
}

func (v *versionedEstimator) EstimateCallCost(function, overloadID string, target *AstNode, args []AstNode) *CallEstimate {
	if v.delegate == nil {
		return nil
	}
	return v.delegate.EstimateCallCost(function, overloadID, target, args)
}

func (v *versionedEstimator) modelVersion() uint32 {
	return v.version
}

func (v *versionedEstimator) sizingStrategy() SizingStrategy {
	return v.strategy
}

func (v *versionedEstimator) resultType() (*types.Type, bool) {
	return v.coster.checkedAST.GetType(v.callID), true
}

func (c *coster) getEstimator() Estimator {
	return &c.est
}

func (c *coster) newEstimateContext() *estimatorEvalContext {
	return &c.evalCtx
}

// computeSize resolves the size estimate for an expression, caching the result when found.
func (c *coster) computeSize(e ast.Expr) *SizeEstimate {
	if size, ok := c.computedSizes[e.ID()]; ok {
		if size == unitSizeEstimate {
			return &unitSizeEstimate
		}
		if size == zeroSizeEstimate {
			return &zeroSizeEstimate
		}
		return &size
	}
	if e.Kind() == ast.IdentKind {
		varName := e.AsIdent()
		if v, ok := c.peekLocalVar(varName); ok && v.size != nil {
			if c.modelVersion < 1 && c.estimator != nil {
				node := astNode{expr: e, path: c.getPath(e), t: c.getType(e)}
				if size := c.estimator.EstimateSize(node); size != nil {
					c.computedSizes[e.ID()] = *size
					return size
				}
			}
			return v.size
		}
	}
	strategy := c.sizingStrategy
	if strategy == defaultSizing {
		if size := computeExprSize(e); size != nil {
			c.computedSizes[e.ID()] = *size
			return size
		}
		// Without an Estimator to supply hints, only a container's size needs the strategy.
		if t := c.getType(e); c.estimator == nil && !isContainerKind(t.Kind()) {
			if size := computeTypeSize(t, c.modelVersion); size != nil {
				c.computedSizes[e.ID()] = *size
				return size
			}
			return nil
		}
	}
	node := astNode{expr: e, path: c.getPath(e), t: c.getType(e)}
	ctx := c.newEstimateContext()
	if size, ok := strategy.EstimateSize(ctx, node); ok {
		c.computedSizes[e.ID()] = size
		if size == unitSizeEstimate {
			return &unitSizeEstimate
		}
		if size == zeroSizeEstimate {
			return &zeroSizeEstimate
		}
		return &size
	}
	if size := computeExprSize(e); size != nil {
		return size
	}
	if size := computeTypeSize(c.getType(e), c.modelVersion); size != nil {
		return size
	}
	return nil
}

// computeExprSize computes the size estimate from literal and declared container expressions.
func computeExprSize(expr ast.Expr) *SizeEstimate {
	var v uint64
	switch expr.Kind() {
	case ast.LiteralKind:
		switch ck := expr.AsLiteral().(type) {
		case types.String:
			// converting to runes here is an O(n) operation, but
			// this is consistent with how size is computed at runtime,
			// and how the language definition defines string size
			v = uint64(len([]rune(ck)))
		case types.Bytes:
			v = uint64(len(ck))
		case types.Bool, types.Double, types.Duration,
			types.Int, types.Timestamp, types.Uint,
			types.Null:
			return &unitSizeEstimate
		default:
			return nil
		}
	default:
		return nil
	}
	if v == 1 {
		return &unitSizeEstimate
	}
	if v == 0 {
		return &zeroSizeEstimate
	}
	size := FixedSizeEstimate(v)
	return &size
}

// computeTypeSize returns a fixed unit size estimate if the type is a constant-size scalar.
// computeTypeSize returns the size every value of a type has at the given cost model version, or nil
// if values of the type vary in size.
//
// A scalar has size 1. An optional of a scalar has size 1 before version 1, and from version 1 is 1
// plus the scalar, which is its element; see Optional.
func computeTypeSize(t *types.Type, version uint32) *SizeEstimate {
	if isScalar(t) {
		return &unitSizeEstimate
	}
	if isOptionalType(t) && isScalar(t.Parameters()[0]) {
		if version < 1 {
			return &unitSizeEstimate
		}
		size := optionalSizeOf(unitSizeEstimate)
		return &size
	}
	return nil
}

// isOptionalType reports whether t is an optional type with a value type.
func isOptionalType(t *types.Type) bool {
	return t.Kind() == types.OpaqueKind && t.TypeName() == "optional_type" && len(t.Parameters()) == 1
}

// optionalSizeOf returns the size of an optional which holds a value of size v, from cost model
// version 1: 1 plus the value, whose size is the optional's element. See Optional.
func optionalSizeOf(v SizeEstimate) SizeEstimate {
	size := RangedSizeEstimate(SafeAdd(1, v.Min), SafeAdd(1, v.Max))
	size.Elem = &v
	return size
}

// isScalar returns true if the given type is known to be of a constant size at
// compile time. isScalar will return false for strings (they are variable-width)
// in addition to protobuf.Any and protobuf.Value (their size is not knowable at compile time).
func isScalar(t *types.Type) bool {
	switch t.Kind() {
	case types.BoolKind, types.DoubleKind, types.DurationKind, types.IntKind, types.TimestampKind, types.UintKind, types.TypeKind:
		return true
	}
	return false
}

var (
	unknownSizeEstimate = SizeEstimate{Min: 0, Max: math.MaxUint64}
	unknownCostEstimate = unknownSizeEstimate.MultiplyByCostFactor(1)
	unitSizeEstimate    = FixedSizeEstimate(1)
	zeroSizeEstimate    = FixedSizeEstimate(0)

	selectAndIdentCost = FixedCostEstimate(SelectAndIdentCost)
	constCost          = FixedCostEstimate(ConstCost)

	createListBaseCost    = FixedCostEstimate(ListCreateBaseCost)
	createMapBaseCost     = FixedCostEstimate(MapCreateBaseCost)
	createMessageBaseCost = FixedCostEstimate(StructCreateBaseCost)

	accumulatorName       = "__result__"
	hiddenAccumulatorName = "@result"
)
