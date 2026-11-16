// Copyright 2026 Google LLC
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

// Package cost provides cost estimation, cost tracking, and saturating arithmetic.
//
// Costs and sizes are unsigned 64-bit values where math.MaxUint64 doubles as the representation
// of an unbounded, or unknown, quantity. Every operation in this package saturates at
// math.MaxUint64 rather than wrapping so that an unbounded input remains unbounded through any
// sequence of operations.
package cost

import (
	"math"

	"cel.dev/cel-go/common/ast"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
	"cel.dev/cel-go/common/types/traits"
)

const (
	// SelectAndIdentCost is the cost of an operation that accesses an identifier or performs a select.
	SelectAndIdentCost = 1

	// ConstCost is the cost of an operation that accesses a constant.
	ConstCost = 0

	// ListCreateBaseCost is the base cost of any operation that creates a new list.
	ListCreateBaseCost = 10

	// MapCreateBaseCost is the base cost of any operation that creates a new map.
	MapCreateBaseCost = 30

	// StructCreateBaseCost is the base cost of any operation that creates a new struct.
	StructCreateBaseCost = 40

	// StringTraversalCostFactor is multiplied to a length of a string when computing the cost of traversing the entire
	// string once.
	StringTraversalCostFactor = 0.1

	// RegexStringLengthCostFactor is multiplied to the length of a regex string pattern when computing the cost of
	// applying the regex to a string of unit cost.
	RegexStringLengthCostFactor = 0.25
)

var (
	// CallCostEstimate is the base cost estimate for an O(1) function call.
	CallCostEstimate = FixedCostEstimate(1)

	// CallCost is the base cost for an O(1) function call.
	CallCost = uint64(1)

	// ListAllocCost is the base cost estimate for allocating a list.
	ListAllocCost = FixedCostEstimate(ListCreateBaseCost)

	// StringCostFactor is the cost factor for traversing a string once.
	StringCostFactor = StringTraversalCostFactor
)

// maxUint64AsFloat is the smallest float64 value greater than math.MaxUint64.
//
// Conversion of a float64 to a uint64 is undefined when the value is out of range, so float
// results are compared against this bound before conversion.
var maxUint64AsFloat = math.Ldexp(1.0, 64)

// SafeAdd returns the sum of the input values, saturating at math.MaxUint64.
func SafeAdd(x, y uint64, rest ...uint64) uint64 {
	sum := x
	if y > 0 && sum > math.MaxUint64-y {
		return math.MaxUint64
	}
	sum += y
	for _, r := range rest {
		if r > 0 && sum > math.MaxUint64-r {
			return math.MaxUint64
		}
		sum += r
	}
	return sum
}

// SafeSubtract returns the difference of x - y, saturating at zero.
// If x is saturated (math.MaxUint64), it remains saturated.
func SafeSubtract(x, y uint64) uint64 {
	if x == math.MaxUint64 {
		return math.MaxUint64
	}
	if x > y {
		return x - y
	}
	return 0
}

// SafeMultiply returns the product of the input values, saturating at math.MaxUint64.
func SafeMultiply(x, y uint64) uint64 {
	if y != 0 && x > math.MaxUint64/y {
		return math.MaxUint64
	}
	return x * y
}

// SafeMultiplyByFactor multiplies a value by a cost factor and returns the nearest integer
// value, rounded up, saturating at math.MaxUint64.
func SafeMultiplyByFactor(x uint64, factor float64) uint64 {
	xFloat := float64(x)
	if xFloat > 0 && factor > 0 && xFloat > math.MaxUint64/factor {
		return math.MaxUint64
	}
	return SafeCeil(xFloat * factor)
}

// SafeCeil returns the smallest integer value greater than or equal to the input, saturating at
// math.MaxUint64 and flooring at zero.
//
// Negative and NaN inputs return zero.
func SafeCeil(x float64) uint64 {
	if math.IsNaN(x) || x <= 0 {
		return 0
	}
	ceil := math.Ceil(x)
	if ceil >= maxUint64AsFloat {
		return math.MaxUint64
	}
	return uint64(ceil)
}

// SafeTrunc returns the integer part of the input, saturating at math.MaxUint64 and flooring at
// zero.
//
// # Negative inputs return zero, and NaN saturates to max uint64
//
// Cost trackers which report a truncated cost should prefer this helper over a direct conversion,
// as conversion of an out-of-range float64 to a uint64 is platform-defined.
func SafeTrunc(x float64) uint64 {
	if x <= 0 {
		return 0
	}
	if math.IsNaN(x) || x >= maxUint64AsFloat {
		return math.MaxUint64
	}
	return uint64(x)
}

// SafeMultiplyByFactorTrunc multiplies a value by a cost factor and returns the integer part of
// the result, truncated, saturating at math.MaxUint64.
//
// This is the truncating counterpart to SafeMultiplyByFactor, which rounds up.
func SafeMultiplyByFactorTrunc(x uint64, factor float64) uint64 {
	xFloat := float64(x)
	if xFloat > 0 && factor > 0 && xFloat > math.MaxUint64/factor {
		return math.MaxUint64
	}
	return SafeTrunc(xFloat * factor)
}

// SizeEstimate represents an estimated size of a variable length string, bytes, map or list.
type SizeEstimate struct {
	// Min is the minimum estimated size (inclusive).
	Min uint64

	// Max is the maximum estimated size (inclusive).
	Max uint64

	// Key indicates the estimated size of a key, set if the size estimate is for a map.
	Key *SizeEstimate

	// Elem indicates the estimated size of a value, set if size estimate is for a map or list.
	Elem *SizeEstimate
}

// UnknownSizeEstimate returns a size between 0 and max uint.
func UnknownSizeEstimate() SizeEstimate {
	return unknownSizeEstimate
}

// FixedSizeEstimate returns a size estimate with a fixed min and max range.
func FixedSizeEstimate(size uint64) SizeEstimate {
	return SizeEstimate{Min: size, Max: size}
}

// RangedSizeEstimate returns a size estimate bounded by min and max.
func RangedSizeEstimate(min, max uint64) SizeEstimate {
	return SizeEstimate{Min: min, Max: max}
}

// AtLeastOneSize returns a size estimate with min and max guaranteed to be at least 1.
func AtLeastOneSize(size SizeEstimate) SizeEstimate {
	if size.Min == 0 {
		size.Min = 1
	}
	if size.Max == 0 {
		size.Max = 1
	}
	return size
}

// ListSizeEstimate returns a SizeEstimate for a list with the given list length and element size.
func ListSizeEstimate(listSize SizeEstimate, elemSize SizeEstimate) SizeEstimate {
	listSize.Key = nil
	listSize.Elem = &elemSize
	return listSize
}

// MapSizeEstimate returns a SizeEstimate for a map with the given map size, key size, and value size.
func MapSizeEstimate(mapSize SizeEstimate, keySize SizeEstimate, valSize SizeEstimate) SizeEstimate {
	mapSize.Key = &keySize
	mapSize.Elem = &valSize
	return mapSize
}

// Add adds to another SizeEstimate and returns the sum.
// If add would result in an uint64 overflow, the result is math.MaxUint64.
func (se SizeEstimate) Add(sizeEstimate SizeEstimate) SizeEstimate {
	res := SizeEstimate{
		Min: SafeAdd(se.Min, sizeEstimate.Min),
		Max: SafeAdd(se.Max, sizeEstimate.Max),
	}
	res.Key = mergeSizeEstimatePtr(se.Key, sizeEstimate.Key)
	res.Elem = mergeSizeEstimatePtr(se.Elem, sizeEstimate.Elem)
	return res
}

// Subtract subtracts another SizeEstimate and returns the difference, saturating at zero.
func (se SizeEstimate) Subtract(sizeEstimate SizeEstimate) SizeEstimate {
	res := SizeEstimate{
		Min: SafeSubtract(se.Min, sizeEstimate.Max),
		Max: SafeSubtract(se.Max, sizeEstimate.Min),
	}
	res.Key = mergeSizeEstimatePtr(se.Key, sizeEstimate.Key)
	res.Elem = mergeSizeEstimatePtr(se.Elem, sizeEstimate.Elem)
	return res
}

// Multiply multiplies by another SizeEstimate and returns the product.
// If multiply would result in an uint64 overflow, the result is math.MaxUint64.
//
// The product carries the union of the operands' key and element sizes, as Add, Subtract and
// Union do: a length scaled by a factor describes the same kind of element it started with. An
// estimator whose result holds something other than its operands' elements — expanded strings, for
// instance — must state that with List or Map rather than let the operands' sizes be inherited.
func (se SizeEstimate) Multiply(sizeEstimate SizeEstimate) SizeEstimate {
	res := SizeEstimate{
		Min: SafeMultiply(se.Min, sizeEstimate.Min),
		Max: SafeMultiply(se.Max, sizeEstimate.Max),
	}
	res.Key = mergeSizeEstimatePtr(se.Key, sizeEstimate.Key)
	res.Elem = mergeSizeEstimatePtr(se.Elem, sizeEstimate.Elem)
	return res
}

// MultiplyByCostFactor multiplies a SizeEstimate by a cost factor and returns the CostEstimate with the
// nearest integer of the result, rounded up.
func (se SizeEstimate) MultiplyByCostFactor(costPerUnit float64) CostEstimate {
	return CostEstimate{
		Min: SafeMultiplyByFactor(se.Min, costPerUnit),
		Max: SafeMultiplyByFactor(se.Max, costPerUnit),
	}
}

// MultiplyByCost multiplies by the cost and returns the product.
// If multiply would result in an uint64 overflow, the result is math.MaxUint64.
func (se SizeEstimate) MultiplyByCost(estimate CostEstimate) CostEstimate {
	return CostEstimate{
		SafeMultiply(se.Min, estimate.Min),
		SafeMultiply(se.Max, estimate.Max),
	}
}

// Union returns a SizeEstimate that encompasses both input SizeEstimate values.
func (se SizeEstimate) Union(size SizeEstimate) SizeEstimate {
	result := se
	if size.Min < result.Min {
		result.Min = size.Min
	}
	if size.Max > result.Max {
		result.Max = size.Max
	}
	result.Key = mergeSizeEstimatePtr(se.Key, size.Key)
	result.Elem = mergeSizeEstimatePtr(se.Elem, size.Elem)
	return result
}

// mergeSizeEstimatePtr merges two optional SizeEstimate pointers using union.
func mergeSizeEstimatePtr(a, b *SizeEstimate) *SizeEstimate {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	u := a.Union(*b)
	return &u
}

// AsCost converts a size estimate to an equivalent cost estimate.
func (se SizeEstimate) AsCost() CostEstimate {
	return se.MultiplyByCostFactor(1)
}

// IsUnknown returns true if the size estimate represents an unknown size (0 to math.MaxUint64 with no key or element sizes).
func (se SizeEstimate) IsUnknown() bool {
	return se == unknownSizeEstimate
}

// Estimate represents an estimated cost range and provides add and multiply operations
// that do not overflow.
type Estimate struct {
	Min, Max uint64
}

// CostEstimate is an alias for Estimate.
//
// Deprecated: use Estimate.
type CostEstimate = Estimate

// UnknownCostEstimate returns a cost with an unknown impact.
func UnknownCostEstimate() CostEstimate {
	return unknownCostEstimate
}

// IsUnknown returns true if the cost estimate represents an unknown cost (0 to math.MaxUint64).
func (ce CostEstimate) IsUnknown() bool {
	return ce == unknownCostEstimate
}

// FixedCostEstimate returns a cost with a fixed min and max range.
func FixedCostEstimate(fixedCost uint64) CostEstimate {
	return CostEstimate{Min: fixedCost, Max: fixedCost}
}

// RangedCostEstimate returns a cost estimate bounded by min and max.
func RangedCostEstimate(min, max uint64) CostEstimate {
	return CostEstimate{Min: min, Max: max}
}

// Add adds the costs and returns the sum.
// If add would result in an uint64 overflow for the min or max, the value is set to math.MaxUint64.
func (ce CostEstimate) Add(estimate CostEstimate) CostEstimate {
	return CostEstimate{
		Min: SafeAdd(ce.Min, estimate.Min),
		Max: SafeAdd(ce.Max, estimate.Max),
	}
}

// Multiply multiplies by the cost and returns the product.
// If multiply would result in an uint64 overflow, the result is math.MaxUint64.
func (ce CostEstimate) Multiply(estimate CostEstimate) CostEstimate {
	return CostEstimate{
		Min: SafeMultiply(ce.Min, estimate.Min),
		Max: SafeMultiply(ce.Max, estimate.Max),
	}
}

// MultiplyByCostFactor multiplies a CostEstimate by a cost factor and returns the CostEstimate with the
// nearest integer of the result, rounded up.
func (ce CostEstimate) MultiplyByCostFactor(costPerUnit float64) CostEstimate {
	return CostEstimate{
		Min: SafeMultiplyByFactor(ce.Min, costPerUnit),
		Max: SafeMultiplyByFactor(ce.Max, costPerUnit),
	}
}

// Union returns a CostEstimate that encompasses both input CostEstimates.
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

// CallEstimate includes a CostEstimate for the call, and an optional estimate of the result object size.
// The ResultSize should only be provided if the call results in a map, list, string or bytes.
type CallEstimate struct {
	CostEstimate

	ResultSize *SizeEstimate
}

// NewCallEstimate creates a new CallEstimate with the given cost and optional result size.
func NewCallEstimate(cost CostEstimate, sz *SizeEstimate) *CallEstimate {
	return &CallEstimate{CostEstimate: cost, ResultSize: sz}
}

// NewListCallEstimate creates a CallEstimate for a call that allocates a list, adding CallCost and
// ListCreateBaseCost to work and setting ResultSize to resultSize.
func NewListCallEstimate(work CostEstimate, resultSize SizeEstimate) *CallEstimate {
	return NewCallEstimate(
		work.Add(FixedCostEstimate(CallCost+ListCreateBaseCost)),
		&resultSize,
	)
}

// ActualSize returns the size of the value for all traits.Sizer values,
// and returns a size of 1 for all other value types.
func ActualSize(value ref.Val) uint64 {
	if sz, ok := value.(traits.Sizer); ok {
		return uint64(sz.Size().(types.Int))
	}
	return 1
}

// EstimateSize returns a SizeEstimate for the given node from its computed size, estimator, or unknown.
func EstimateSize(estimator Estimator, node AstNode) SizeEstimate {
	if node == nil {
		return UnknownSizeEstimate()
	}
	if l := node.ComputedSize(); l != nil {
		return *l
	}
	if estimator != nil {
		if l := estimator.EstimateSize(node); l != nil {
			return *l
		}
	}
	return UnknownSizeEstimate()
}

// EstimateTraversal computes cost as a function of the size of the target object and whether the call allocates memory.
func EstimateTraversal(nodeSize SizeEstimate, costFactor float64, allocationCost *CostEstimate) (CostEstimate, *SizeEstimate) {
	cost := nodeSize.MultiplyByCostFactor(costFactor)
	if allocationCost != nil {
		cost = cost.Add(*allocationCost)
	}
	return cost, &nodeSize
}

// EstimateStringScan estimates cost for scanning a string.
func EstimateStringScan(sz SizeEstimate) (CostEstimate, *SizeEstimate) {
	return EstimateTraversal(sz, StringCostFactor, nil)
}

// EstimateListAlloc estimates cost for allocating a list.
func EstimateListAlloc(sz SizeEstimate, costFactor float64) (CostEstimate, *SizeEstimate) {
	return EstimateTraversal(sz, costFactor, &ListAllocCost)
}

// UintValue returns the value of a non-negative literal int or uint node as a uint64, or false if
// the node is not a non-negative integer literal.
func UintValue(node AstNode) (uint64, bool) {
	if node == nil || node.Expr() == nil || node.Expr().Kind() != ast.LiteralKind {
		return 0, false
	}
	switch v := node.Expr().AsLiteral().(type) {
	case types.Int:
		if v < types.IntZero {
			return 0, false
		}
		return uint64(v), true
	case types.Uint:
		return uint64(v), true
	default:
		return 0, false
	}
}

// NodeAsUintValue returns the value of a literal int node as a uint64, or the default value if the
// node is not a non-negative int literal.
func NodeAsUintValue(node AstNode, defaultVal uint64) uint64 {
	if val, ok := UintValue(node); ok {
		return val
	}
	if node != nil && node.Expr() != nil && node.Expr().Kind() == ast.LiteralKind {
		if v, ok := node.Expr().AsLiteral().(types.Int); ok && v < types.IntZero {
			return 0
		}
	}
	return defaultVal
}
