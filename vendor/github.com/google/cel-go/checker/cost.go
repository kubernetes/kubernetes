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

package checker

import (
	"math"

	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/parser"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// WARNING: Any changes to cost calculations in this file require a corresponding change in interpreter/runtimecost.go

// CostEstimator estimates the sizes of variable length input data and the costs of functions.
type CostEstimator interface {
	// EstimateSize returns a SizeEstimate for the given AstNode, or nil if
	// the estimator has no estimate to provide. The size is equivalent to the result of the CEL `size()` function:
	// length of strings and bytes, number of map entries or number of list items.
	// EstimateSize is only called for AstNodes where
	// CEL does not know the size; EstimateSize is not called for values defined inline in CEL where the size
	// is already obvious to CEL.
	EstimateSize(element AstNode) *SizeEstimate
	// EstimateCallCost returns the estimated cost of an invocation, or nil if
	// the estimator has no estimate to provide.
	EstimateCallCost(function, overloadID string, target *AstNode, args []AstNode) *CallEstimate
}

// CallEstimate includes a CostEstimate for the call, and an optional estimate of the result object size.
// The ResultSize should only be provided if the call results in a map, list, string or bytes.
type CallEstimate struct {
	CostEstimate
	ResultSize *SizeEstimate
}

// AstNode represents an AST node for the purpose of cost estimations.
type AstNode interface {
	// Path returns a field path through the provided type declarations to the type of the AstNode, or nil if the AstNode does not
	// represent type directly reachable from the provided type declarations.
	// The first path element is a variable. All subsequent path elements are one of: field name, '@items', '@keys', '@values'.
	Path() []string
	// Type returns the deduced type of the AstNode.
	Type() *exprpb.Type
	// Expr returns the expression of the AstNode.
	Expr() *exprpb.Expr
	// ComputedSize returns a size estimate of the AstNode derived from information available in the CEL expression.
	// For constants and inline list and map declarations, the exact size is returned. For concatenated list, strings
	// and bytes, the size is derived from the size estimates of the operands. nil is returned if there is no
	// computed size available.
	ComputedSize() *SizeEstimate
}

type astNode struct {
	path        []string
	t           *exprpb.Type
	expr        *exprpb.Expr
	derivedSize *SizeEstimate
}

func (e astNode) Path() []string {
	return e.path
}

func (e astNode) Type() *exprpb.Type {
	return e.t
}

func (e astNode) Expr() *exprpb.Expr {
	return e.expr
}

func (e astNode) ComputedSize() *SizeEstimate {
	if e.derivedSize != nil {
		return e.derivedSize
	}
	var v uint64
	switch ek := e.expr.GetExprKind().(type) {
	case *exprpb.Expr_ConstExpr:
		switch ck := ek.ConstExpr.GetConstantKind().(type) {
		case *exprpb.Constant_StringValue:
			v = uint64(len(ck.StringValue))
		case *exprpb.Constant_BytesValue:
			v = uint64(len(ck.BytesValue))
		case *exprpb.Constant_BoolValue, *exprpb.Constant_DoubleValue, *exprpb.Constant_DurationValue,
			*exprpb.Constant_Int64Value, *exprpb.Constant_TimestampValue, *exprpb.Constant_Uint64Value,
			*exprpb.Constant_NullValue:
			v = uint64(1)
		default:
			return nil
		}
	case *exprpb.Expr_ListExpr:
		v = uint64(len(ek.ListExpr.GetElements()))
	case *exprpb.Expr_StructExpr:
		if ek.StructExpr.GetMessageName() == "" {
			v = uint64(len(ek.StructExpr.GetEntries()))
		}
	default:
		return nil
	}

	return &SizeEstimate{Min: v, Max: v}
}

// SizeEstimate represents an estimated size of a variable length string, bytes, map or list.
type SizeEstimate struct {
	Min, Max uint64
}

// Add adds to another SizeEstimate and returns the sum.
// If add would result in an uint64 overflow, the result is math.MaxUint64.
func (se SizeEstimate) Add(sizeEstimate SizeEstimate) SizeEstimate {
	return SizeEstimate{
		addUint64NoOverflow(se.Min, sizeEstimate.Min),
		addUint64NoOverflow(se.Max, sizeEstimate.Max),
	}
}

// Multiply multiplies by another SizeEstimate and returns the product.
// If multiply would result in an uint64 overflow, the result is math.MaxUint64.
func (se SizeEstimate) Multiply(sizeEstimate SizeEstimate) SizeEstimate {
	return SizeEstimate{
		multiplyUint64NoOverflow(se.Min, sizeEstimate.Min),
		multiplyUint64NoOverflow(se.Max, sizeEstimate.Max),
	}
}

// MultiplyByCostFactor multiplies a SizeEstimate by a cost factor and returns the CostEstimate with the
// nearest integer of the result, rounded up.
func (se SizeEstimate) MultiplyByCostFactor(costPerUnit float64) CostEstimate {
	return CostEstimate{
		multiplyByCostFactor(se.Min, costPerUnit),
		multiplyByCostFactor(se.Max, costPerUnit),
	}
}

// MultiplyByCost multiplies by the cost and returns the product.
// If multiply would result in an uint64 overflow, the result is math.MaxUint64.
func (se SizeEstimate) MultiplyByCost(cost CostEstimate) CostEstimate {
	return CostEstimate{
		multiplyUint64NoOverflow(se.Min, cost.Min),
		multiplyUint64NoOverflow(se.Max, cost.Max),
	}
}

// Union returns a SizeEstimate that encompasses both input the SizeEstimate.
func (se SizeEstimate) Union(size SizeEstimate) SizeEstimate {
	result := se
	if size.Min < result.Min {
		result.Min = size.Min
	}
	if size.Max > result.Max {
		result.Max = size.Max
	}
	return result
}

// CostEstimate represents an estimated cost range and provides add and multiply operations
// that do not overflow.
type CostEstimate struct {
	Min, Max uint64
}

// Add adds the costs and returns the sum.
// If add would result in an uint64 overflow for the min or max, the value is set to math.MaxUint64.
func (ce CostEstimate) Add(cost CostEstimate) CostEstimate {
	return CostEstimate{
		addUint64NoOverflow(ce.Min, cost.Min),
		addUint64NoOverflow(ce.Max, cost.Max),
	}
}

// Multiply multiplies by the cost and returns the product.
// If multiply would result in an uint64 overflow, the result is math.MaxUint64.
func (ce CostEstimate) Multiply(cost CostEstimate) CostEstimate {
	return CostEstimate{
		multiplyUint64NoOverflow(ce.Min, cost.Min),
		multiplyUint64NoOverflow(ce.Max, cost.Max),
	}
}

// MultiplyByCostFactor multiplies a CostEstimate by a cost factor and returns the CostEstimate with the
// nearest integer of the result, rounded up.
func (ce CostEstimate) MultiplyByCostFactor(costPerUnit float64) CostEstimate {
	return CostEstimate{
		multiplyByCostFactor(ce.Min, costPerUnit),
		multiplyByCostFactor(ce.Max, costPerUnit),
	}
}

// Union returns a CostEstimate that encompasses both input the CostEstimates.
func (ce CostEstimate) Union(size CostEstimate) CostEstimate {
	result := ce
	if size.Min < result.Min {
		result.Min = size.Min
	}
	if size.Max > result.Max {
		result.Max = size.Max
	}
	return result
}

// addUint64NoOverflow adds non-negative ints. If the result is exceeds math.MaxUint64, math.MaxUint64
// is returned.
func addUint64NoOverflow(x, y uint64) uint64 {
	if y > 0 && x > math.MaxUint64-y {
		return math.MaxUint64
	}
	return x + y
}

// multiplyUint64NoOverflow multiplies non-negative ints. If the result is exceeds math.MaxUint64, math.MaxUint64
// is returned.
func multiplyUint64NoOverflow(x, y uint64) uint64 {
	if x > 0 && y > 0 && x > math.MaxUint64/y {
		return math.MaxUint64
	}
	return x * y
}

// multiplyByFactor multiplies an integer by a cost factor float and returns the nearest integer value, rounded up.
func multiplyByCostFactor(x uint64, y float64) uint64 {
	xFloat := float64(x)
	if xFloat > 0 && y > 0 && xFloat > math.MaxUint64/y {
		return math.MaxUint64
	}
	return uint64(math.Ceil(xFloat * y))
}

var (
	selectAndIdentCost = CostEstimate{Min: common.SelectAndIdentCost, Max: common.SelectAndIdentCost}
	constCost          = CostEstimate{Min: common.ConstCost, Max: common.ConstCost}

	createListBaseCost    = CostEstimate{Min: common.ListCreateBaseCost, Max: common.ListCreateBaseCost}
	createMapBaseCost     = CostEstimate{Min: common.MapCreateBaseCost, Max: common.MapCreateBaseCost}
	createMessageBaseCost = CostEstimate{Min: common.StructCreateBaseCost, Max: common.StructCreateBaseCost}
)

type coster struct {
	// exprPath maps from Expr Id to field path.
	exprPath map[int64][]string
	// iterRanges tracks the iterRange of each iterVar.
	iterRanges iterRangeScopes
	// computedSizes tracks the computed sizes of call results.
	computedSizes map[int64]SizeEstimate
	checkedExpr   *exprpb.CheckedExpr
	estimator     CostEstimator
}

// Use a stack of iterVar -> iterRange Expr Ids to handle shadowed variable names.
type iterRangeScopes map[string][]int64

func (vs iterRangeScopes) push(varName string, expr *exprpb.Expr) {
	vs[varName] = append(vs[varName], expr.GetId())
}

func (vs iterRangeScopes) pop(varName string) {
	varStack := vs[varName]
	vs[varName] = varStack[:len(varStack)-1]
}

func (vs iterRangeScopes) peek(varName string) (int64, bool) {
	varStack := vs[varName]
	if len(varStack) > 0 {
		return varStack[len(varStack)-1], true
	}
	return 0, false
}

// Cost estimates the cost of the parsed and type checked CEL expression.
func Cost(checker *exprpb.CheckedExpr, estimator CostEstimator) CostEstimate {
	c := coster{
		checkedExpr:   checker,
		estimator:     estimator,
		exprPath:      map[int64][]string{},
		iterRanges:    map[string][]int64{},
		computedSizes: map[int64]SizeEstimate{},
	}
	return c.cost(checker.GetExpr())
}

func (c *coster) cost(e *exprpb.Expr) CostEstimate {
	if e == nil {
		return CostEstimate{}
	}
	var cost CostEstimate
	switch e.GetExprKind().(type) {
	case *exprpb.Expr_ConstExpr:
		cost = constCost
	case *exprpb.Expr_IdentExpr:
		cost = c.costIdent(e)
	case *exprpb.Expr_SelectExpr:
		cost = c.costSelect(e)
	case *exprpb.Expr_CallExpr:
		cost = c.costCall(e)
	case *exprpb.Expr_ListExpr:
		cost = c.costCreateList(e)
	case *exprpb.Expr_StructExpr:
		cost = c.costCreateStruct(e)
	case *exprpb.Expr_ComprehensionExpr:
		cost = c.costComprehension(e)
	default:
		return CostEstimate{}
	}
	return cost
}

func (c *coster) costIdent(e *exprpb.Expr) CostEstimate {
	identExpr := e.GetIdentExpr()

	// build and track the field path
	if iterRange, ok := c.iterRanges.peek(identExpr.GetName()); ok {
		switch c.checkedExpr.TypeMap[iterRange].GetTypeKind().(type) {
		case *exprpb.Type_ListType_:
			c.addPath(e, append(c.exprPath[iterRange], "@items"))
		case *exprpb.Type_MapType_:
			c.addPath(e, append(c.exprPath[iterRange], "@keys"))
		}
	} else {
		c.addPath(e, []string{identExpr.GetName()})
	}

	return selectAndIdentCost
}

func (c *coster) costSelect(e *exprpb.Expr) CostEstimate {
	sel := e.GetSelectExpr()
	var sum CostEstimate
	if sel.GetTestOnly() {
		return sum
	}
	sum = sum.Add(c.cost(sel.GetOperand()))
	targetType := c.getType(sel.GetOperand())
	switch kindOf(targetType) {
	case kindMap, kindObject, kindTypeParam:
		sum = sum.Add(selectAndIdentCost)
	}

	// build and track the field path
	c.addPath(e, append(c.getPath(sel.GetOperand()), sel.GetField()))

	return sum
}

func (c *coster) costCall(e *exprpb.Expr) CostEstimate {
	call := e.GetCallExpr()
	target := call.GetTarget()
	args := call.GetArgs()

	var sum CostEstimate

	argTypes := make([]AstNode, len(args))
	argCosts := make([]CostEstimate, len(args))
	for i, arg := range args {
		argCosts[i] = c.cost(arg)
		argTypes[i] = c.newAstNode(arg)
	}

	ref := c.checkedExpr.ReferenceMap[e.GetId()]
	if ref == nil || len(ref.GetOverloadId()) == 0 {
		return CostEstimate{}
	}
	var targetType AstNode
	if target != nil {
		if call.Target != nil {
			sum = sum.Add(c.cost(call.GetTarget()))
			targetType = c.newAstNode(call.GetTarget())
		}
	}
	// Pick a cost estimate range that covers all the overload cost estimation ranges
	fnCost := CostEstimate{Min: uint64(math.MaxUint64), Max: 0}
	var resultSize *SizeEstimate
	for _, overload := range ref.GetOverloadId() {
		overloadCost := c.functionCost(call.GetFunction(), overload, &targetType, argTypes, argCosts)
		fnCost = fnCost.Union(overloadCost.CostEstimate)
		if overloadCost.ResultSize != nil {
			if resultSize == nil {
				resultSize = overloadCost.ResultSize
			} else {
				size := resultSize.Union(*overloadCost.ResultSize)
				resultSize = &size
			}
		}
		// build and track the field path for index operations
		switch overload {
		case overloads.IndexList:
			if len(args) > 0 {
				c.addPath(e, append(c.getPath(args[0]), "@items"))
			}
		case overloads.IndexMap:
			if len(args) > 0 {
				c.addPath(e, append(c.getPath(args[0]), "@values"))
			}
		}
	}
	if resultSize != nil {
		c.computedSizes[e.GetId()] = *resultSize
	}
	return sum.Add(fnCost)
}

func (c *coster) costCreateList(e *exprpb.Expr) CostEstimate {
	create := e.GetListExpr()
	var sum CostEstimate
	for _, e := range create.GetElements() {
		sum = sum.Add(c.cost(e))
	}
	return sum.Add(createListBaseCost)
}

func (c *coster) costCreateStruct(e *exprpb.Expr) CostEstimate {
	str := e.GetStructExpr()
	if str.MessageName != "" {
		return c.costCreateMessage(e)
	}
	return c.costCreateMap(e)
}

func (c *coster) costCreateMap(e *exprpb.Expr) CostEstimate {
	mapVal := e.GetStructExpr()
	var sum CostEstimate
	for _, ent := range mapVal.GetEntries() {
		key := ent.GetMapKey()
		sum = sum.Add(c.cost(key))

		sum = sum.Add(c.cost(ent.GetValue()))
	}
	return sum.Add(createMapBaseCost)
}

func (c *coster) costCreateMessage(e *exprpb.Expr) CostEstimate {
	msgVal := e.GetStructExpr()
	var sum CostEstimate
	for _, ent := range msgVal.GetEntries() {
		sum = sum.Add(c.cost(ent.GetValue()))
	}
	return sum.Add(createMessageBaseCost)
}

func (c *coster) costComprehension(e *exprpb.Expr) CostEstimate {
	comp := e.GetComprehensionExpr()
	var sum CostEstimate
	sum = sum.Add(c.cost(comp.GetIterRange()))
	sum = sum.Add(c.cost(comp.GetAccuInit()))

	// Track the iterRange of each IterVar for field path construction
	c.iterRanges.push(comp.GetIterVar(), comp.GetIterRange())
	loopCost := c.cost(comp.GetLoopCondition())
	stepCost := c.cost(comp.GetLoopStep())
	c.iterRanges.pop(comp.GetIterVar())
	sum = sum.Add(c.cost(comp.Result))
	rangeCnt := c.sizeEstimate(c.newAstNode(comp.GetIterRange()))
	rangeCost := rangeCnt.MultiplyByCost(stepCost.Add(loopCost))
	sum = sum.Add(rangeCost)

	return sum
}

func (c *coster) sizeEstimate(t AstNode) SizeEstimate {
	if l := t.ComputedSize(); l != nil {
		return *l
	}
	if l := c.estimator.EstimateSize(t); l != nil {
		return *l
	}
	// return an estimate of 1 for return types of set
	// lengths, since strings/bytes/more complex objects could be of
	// variable length
	if isScalar(t.Type()) {
		// TODO: since the logic for size estimation is split between
		// ComputedSize and isScalar, changing one will likely require changing
		// the other, so they should be merged in the future if possible
		return SizeEstimate{Min: 1, Max: 1}
	}
	return SizeEstimate{Min: 0, Max: math.MaxUint64}
}

func (c *coster) functionCost(function, overloadID string, target *AstNode, args []AstNode, argCosts []CostEstimate) CallEstimate {
	argCostSum := func() CostEstimate {
		var sum CostEstimate
		for _, a := range argCosts {
			sum = sum.Add(a)
		}
		return sum
	}

	if est := c.estimator.EstimateCallCost(function, overloadID, target, args); est != nil {
		callEst := *est
		return CallEstimate{CostEstimate: callEst.Add(argCostSum()), ResultSize: est.ResultSize}
	}
	switch overloadID {
	// O(n) functions
	case overloads.StringToBytes:
		if len(args) == 1 {
			sz := c.sizeEstimate(args[0])
			// ResultSize max is when each char converts to 4 bytes.
			return CallEstimate{CostEstimate: sz.MultiplyByCostFactor(common.StringTraversalCostFactor).Add(argCostSum()), ResultSize: &SizeEstimate{Min: sz.Min, Max: sz.Max * 4}}
		}
	case overloads.BytesToString:
		if len(args) == 1 {
			sz := c.sizeEstimate(args[0])
			// ResultSize min is when 4 bytes convert to 1 char.
			return CallEstimate{CostEstimate: sz.MultiplyByCostFactor(common.StringTraversalCostFactor).Add(argCostSum()), ResultSize: &SizeEstimate{Min: sz.Min / 4, Max: sz.Max}}
		}
	case overloads.StartsWithString, overloads.EndsWithString:
		if len(args) == 1 {
			return CallEstimate{CostEstimate: c.sizeEstimate(args[0]).MultiplyByCostFactor(common.StringTraversalCostFactor).Add(argCostSum())}
		}
	case overloads.InList:
		// If a list is composed entirely of constant values this is O(1), but we don't account for that here.
		// We just assume all list containment checks are O(n).
		if len(args) == 2 {
			return CallEstimate{CostEstimate: c.sizeEstimate(args[1]).MultiplyByCostFactor(1).Add(argCostSum())}
		}
	// O(nm) functions
	case overloads.MatchesString:
		// https://swtch.com/~rsc/regexp/regexp1.html applies to RE2 implementation supported by CEL
		if target != nil && len(args) == 1 {
			// Add one to string length for purposes of cost calculation to prevent product of string and regex to be 0
			// in case where string is empty but regex is still expensive.
			strCost := c.sizeEstimate(*target).Add(SizeEstimate{Min: 1, Max: 1}).MultiplyByCostFactor(common.StringTraversalCostFactor)
			// We don't know how many expressions are in the regex, just the string length (a huge
			// improvement here would be to somehow get a count the number of expressions in the regex or
			// how many states are in the regex state machine and use that to measure regex cost).
			// For now, we're making a guess that each expression in a regex is typically at least 4 chars
			// in length.
			regexCost := c.sizeEstimate(args[0]).MultiplyByCostFactor(common.RegexStringLengthCostFactor)
			return CallEstimate{CostEstimate: strCost.Multiply(regexCost).Add(argCostSum())}
		}
	case overloads.ContainsString:
		if target != nil && len(args) == 1 {
			strCost := c.sizeEstimate(*target).MultiplyByCostFactor(common.StringTraversalCostFactor)
			substrCost := c.sizeEstimate(args[0]).MultiplyByCostFactor(common.StringTraversalCostFactor)
			return CallEstimate{CostEstimate: strCost.Multiply(substrCost).Add(argCostSum())}
		}
	case overloads.LogicalOr, overloads.LogicalAnd:
		lhs := argCosts[0]
		rhs := argCosts[1]
		// min cost is min of LHS for short circuited && or ||
		argCost := CostEstimate{Min: lhs.Min, Max: lhs.Add(rhs).Max}
		return CallEstimate{CostEstimate: argCost}
	case overloads.Conditional:
		size := c.sizeEstimate(args[1]).Union(c.sizeEstimate(args[2]))
		conditionalCost := argCosts[0]
		ifTrueCost := argCosts[1]
		ifFalseCost := argCosts[2]
		argCost := conditionalCost.Add(ifTrueCost.Union(ifFalseCost))
		return CallEstimate{CostEstimate: argCost, ResultSize: &size}
	case overloads.AddString, overloads.AddBytes, overloads.AddList:
		if len(args) == 2 {
			lhsSize := c.sizeEstimate(args[0])
			rhsSize := c.sizeEstimate(args[1])
			resultSize := lhsSize.Add(rhsSize)
			switch overloadID {
			case overloads.AddList:
				// list concatenation is O(1), but we handle it here to track size
				return CallEstimate{CostEstimate: CostEstimate{Min: 1, Max: 1}.Add(argCostSum()), ResultSize: &resultSize}
			default:
				return CallEstimate{CostEstimate: resultSize.MultiplyByCostFactor(common.StringTraversalCostFactor).Add(argCostSum()), ResultSize: &resultSize}
			}
		}
	case overloads.LessString, overloads.GreaterString, overloads.LessEqualsString, overloads.GreaterEqualsString,
		overloads.LessBytes, overloads.GreaterBytes, overloads.LessEqualsBytes, overloads.GreaterEqualsBytes,
		overloads.Equals, overloads.NotEquals:
		lhsCost := c.sizeEstimate(args[0])
		rhsCost := c.sizeEstimate(args[1])
		min := uint64(0)
		smallestMax := lhsCost.Max
		if rhsCost.Max < smallestMax {
			smallestMax = rhsCost.Max
		}
		if smallestMax > 0 {
			min = 1
		}
		// equality of 2 scalar values results in a cost of 1
		return CallEstimate{CostEstimate: CostEstimate{Min: min, Max: smallestMax}.MultiplyByCostFactor(common.StringTraversalCostFactor).Add(argCostSum())}
	}
	// O(1) functions
	// See CostTracker.costCall for more details about O(1) cost calculations

	// Benchmarks suggest that most of the other operations take +/- 50% of a base cost unit
	// which on an Intel xeon 2.20GHz CPU is 50ns.
	return CallEstimate{CostEstimate: CostEstimate{Min: 1, Max: 1}.Add(argCostSum())}
}

func (c *coster) getType(e *exprpb.Expr) *exprpb.Type {
	return c.checkedExpr.TypeMap[e.GetId()]
}

func (c *coster) getPath(e *exprpb.Expr) []string {
	return c.exprPath[e.GetId()]
}

func (c *coster) addPath(e *exprpb.Expr, path []string) {
	c.exprPath[e.GetId()] = path
}

func (c *coster) newAstNode(e *exprpb.Expr) *astNode {
	path := c.getPath(e)
	if len(path) > 0 && path[0] == parser.AccumulatorName {
		// only provide paths to root vars; omit accumulator vars
		path = nil
	}
	var derivedSize *SizeEstimate
	if size, ok := c.computedSizes[e.GetId()]; ok {
		derivedSize = &size
	}
	return &astNode{path: path, t: c.getType(e), expr: e, derivedSize: derivedSize}
}

// isScalar returns true if the given type is known to be of a constant size at
// compile time. isScalar will return false for strings (they are variable-width)
// in addition to protobuf.Any and protobuf.Value (their size is not knowable at compile time).
func isScalar(t *exprpb.Type) bool {
	switch kindOf(t) {
	case kindPrimitive:
		if t.GetPrimitive() != exprpb.Type_STRING && t.GetPrimitive() != exprpb.Type_BYTES {
			return true
		}
	case kindWellKnown:
		if t.GetWellKnown() == exprpb.Type_DURATION || t.GetWellKnown() == exprpb.Type_TIMESTAMP {
			return true
		}
	}
	return false
}
