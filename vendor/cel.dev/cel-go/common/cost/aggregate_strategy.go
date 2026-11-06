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

package cost

import (
	"math"

	"cel.dev/cel-go/common/ast"
	"cel.dev/cel-go/common/operators"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
)

// AggregateSizingStrategy returns a SizingStrategy that computes recursive size estimates
// by following paths during cost estimation, and calculates actual runtime size using
// AggregateSize during cost tracking.
//
// The strategy pins stringUnitLength to 1, overriding the types package default, so that
// raw character and byte lengths reach the cost model unscaled. This is deliberate: sizing
// strategies report raw dimensions, and cost expressions own the conversion to cost units
// by applying factors such as StringTraversalCostFactor.
//
// Callers may supply additional SizeCalculatorOption values, which are applied after the
// pinned default.
//
// Warning: passing types.SizeCalculatorStringUnitLength with a value other than 1 overrides
// the pinned default and causes string costs to be discounted twice, once by the calculator
// and again by the cost factor in the cost expression. A unit length of 10 combined with
// StringTraversalCostFactor yields an effective 100x discount rather than 10x.
func AggregateSizingStrategy(opts ...types.SizeCalculatorOption) SizingStrategy {
	if len(opts) == 0 {
		return defaultAggregateSizing
	}
	defaultOpts := []types.SizeCalculatorOption{types.SizeCalculatorStringUnitLength(1)}
	return &aggregateSizingStrategy{calc: types.NewSizeCalculator(append(defaultOpts, opts...)...)}
}

type aggregateSizingStrategy struct {
	calc *types.SizeCalculator
}

// TrackSize calculates the actual runtime aggregate size of a value during cost tracking.
//
// An optional is 1 plus the aggregate size of the value it holds, and an empty optional is 1, as
// types.Optional reports it; see Optional.
func (a *aggregateSizingStrategy) TrackSize(ctx TrackContext, value ref.Val) (uint64, bool) {
	if value == nil {
		return 0, false
	}
	return uint64(a.calc.AggregateSize(value)), true
}

// EstimateSize computes recursive aggregate size estimates for an AST node during cost estimation.
func (a *aggregateSizingStrategy) EstimateSize(ctx EstimateContext, node AstNode) (SizeEstimate, bool) {
	if node == nil {
		return SizeEstimate{}, false
	}
	if sz := node.ComputedSize(); sz != nil {
		return *sz, true
	}
	if node.Expr() != nil && node.Expr().Kind() == ast.LiteralKind {
		return FixedSizeEstimate(uint64(a.calc.AggregateSize(node.Expr().AsLiteral()))), true
	}
	if node.Type() == nil {
		return estimateScalarOrFallback(ctx, node)
	}
	switch node.Type().Kind() {
	case types.ListKind:
		return estimateAggregateListSize(ctx, node)
	case types.MapKind:
		return estimateAggregateMapSize(ctx, node)
	case types.DynKind, types.AnyKind, types.TypeParamKind:
		return estimateAggregateDynSize(ctx, node)
	default:
		return estimateScalarOrFallback(ctx, node)
	}
}

// estimateAggregateDynSize calculates the aggregate size estimate for a dynamically typed AST node,
// folding nested container hints ("@items", "@keys", "@values") into the aggregate size up to the
// hinted depth.
func estimateAggregateDynSize(ctx EstimateContext, node AstNode) (SizeEstimate, bool) {
	if node.Expr() != nil {
		switch node.Expr().Kind() {
		case ast.ListKind:
			return estimateAggregateListSize(ctx, node)
		case ast.MapKind:
			return estimateAggregateMapSize(ctx, node)
		case ast.CallKind:
			if call := node.Expr().AsCall(); call.FunctionName() == operators.Conditional {
				if u, ok := estimateConditionalSize(ctx, node.Type(), call.Args()); ok {
					return u, true
				}
			}
		}
	}
	var baseSz *SizeEstimate
	if sz, ok := estimateScalarOrFallback(ctx, node); ok {
		if sz.Elem != nil || sz.Key != nil {
			return sz, true
		}
		baseSz = &sz
	}
	if len(node.Path()) > 0 {
		if elemSz := resolveDynamicChild(ctx, node.Path(), "@items"); elemSz != nil {
			if baseSz == nil {
				return combineListSize(nil, elemSz)
			}
			return SizeEstimate{
				Min:  min(baseSz.Min, SafeAdd(1, SafeMultiply(baseSz.Min, max(elemSz.Min, 1)))),
				Max:  max(baseSz.Max, SafeAdd(1, SafeMultiply(baseSz.Max, max(elemSz.Max, 1)))),
				Elem: elemSz,
			}, true
		}
		keySz := resolveDynamicChild(ctx, node.Path(), "@keys")
		valSz := resolveDynamicChild(ctx, node.Path(), "@values")
		if keySz != nil || valSz != nil {
			if baseSz == nil {
				return combineMapSize(nil, keySz, valSz)
			}
			minKey, maxKey, keySz := resolveElemBounds(keySz, baseSz.Max > 0)
			minVal, maxVal, valSz := resolveElemBounds(valSz, baseSz.Max > 0)
			return SizeEstimate{
				Min:  min(baseSz.Min, SafeAdd(1, SafeMultiply(baseSz.Min, SafeAdd(minKey, minVal)))),
				Max:  max(baseSz.Max, SafeAdd(1, SafeMultiply(baseSz.Max, SafeAdd(max(maxKey, 1), max(maxVal, 1))))),
				Key:  keySz,
				Elem: valSz,
			}, true
		}
	}
	if baseSz != nil {
		return *baseSz, true
	}
	return SizeEstimate{}, false
}

// estimateAggregateListSize computes list aggregate size: 1 (header) + listSize * elemSize.
//
// Calls (e.g. `+` or `? :`) already hold aggregate sub-expression sizes. For `+`, 1 is
// deducted to deduplicate headers: (1+L) + (1+R) - 1 = 1+L+R.
// Unhinted variable-width elements default to [1, math.MaxUint64].
func estimateAggregateListSize(ctx EstimateContext, node AstNode) (SizeEstimate, bool) {
	elemType := listElemType(node.Type())
	listSize, elemSize := estimateListExpr(ctx, node, elemType)

	// When node is a call expression, estimateListExpr already evaluated aggregate sizes of sub-expressions.
	if node.Expr() != nil && node.Expr().Kind() == ast.CallKind && listSize != nil {
		call := node.Expr().AsCall()
		if call.FunctionName() == operators.Add && len(call.Args()) == 2 {
			// Concatenation of two lists: (1 + left) + (1 + right) = 2 + left + right.
			// Subtract 1 to correct for the single resulting list container header.
			minVal := max(SafeSubtract(listSize.Min, 1), 1)
			maxVal := max(SafeSubtract(listSize.Max, 1), 1)
			res := RangedSizeEstimate(minVal, maxVal)
			res.Elem = elemSize
			return res, true
		}
		// For other calls (e.g. conditional branches), listSize is already the branch aggregate size.
		listSize.Key = nil
		return *listSize, true
	}

	listSize, elemSize = resolveListSizes(ctx, node, elemType, listSize, elemSize)
	// The list length may be unknown (listSize == nil) while elemSize is known—for example,
	// when the list has fixed-width primitive elements (e.g. list<int> where elemSize is [1, 1])
	// or when a size hint was provided on the "@items" subpath without a hint on the list itself.
	// combineListSize returns an unknown container size [0, MaxUint64] while preserving Elem so
	// downstream operations (such as indexing or comprehensions) can still use the known element size.
	if listSize == nil {
		return combineListSize(nil, elemSize)
	}
	minElem, maxElem, elemSize := resolveElemBounds(elemSize, listSize.Max > 0)
	aggMin := SafeAdd(1, SafeMultiply(listSize.Min, minElem))
	aggMax := SafeAdd(1, SafeMultiply(listSize.Max, maxElem))
	return SizeEstimate{
		Min:  aggMin,
		Max:  aggMax,
		Elem: elemSize,
	}, true
}

// estimateAggregateMapSize computes map aggregate size: 1 (header) + mapSize * (keySize + valSize).
//
// Calls (e.g. `? :`) already hold aggregate sub-expression sizes.
// Unhinted variable-width keys/values default to [1, math.MaxUint64].
func estimateAggregateMapSize(ctx EstimateContext, node AstNode) (SizeEstimate, bool) {
	keyType, valType := mapKeyValueTypes(node.Type())
	mapSize, keySize, valSize := estimateMapExpr(ctx, node, keyType, valType)

	// For call expressions (e.g. conditional branches), mapSize is already the branch aggregate size.
	if node.Expr() != nil && node.Expr().Kind() == ast.CallKind && mapSize != nil {
		return *mapSize, true
	}

	mapSize, keySize, valSize = resolveMapSizes(ctx, node, keyType, valType, mapSize, keySize, valSize)
	// The map entry count may be unknown (mapSize == nil) while keySize or valSize is known
	// (e.g. fixed-width key/value types like int, or hints provided only on "@keys"/"@values").
	// combineMapSize returns an unknown container size while preserving Key and Elem estimates.
	if mapSize == nil {
		return combineMapSize(nil, keySize, valSize)
	}
	minKey, maxKey, keySize := resolveElemBounds(keySize, mapSize.Max > 0)
	minVal, maxVal, valSize := resolveElemBounds(valSize, mapSize.Max > 0)
	entryMin := SafeAdd(minKey, minVal)
	entryMax := SafeAdd(maxKey, maxVal)
	aggMin := SafeAdd(1, SafeMultiply(mapSize.Min, entryMin))
	aggMax := SafeAdd(1, SafeMultiply(mapSize.Max, entryMax))
	return SizeEstimate{
		Min:  aggMin,
		Max:  aggMax,
		Key:  keySize,
		Elem: valSize,
	}, true
}

// resolveElemBounds returns the [min, max] bounds for a child element/key/value in aggregate size
// calculation. Every value has aggregate size at least 1, so each bound is at least 1 and a
// container's size is at least 1 plus its length. For variable-width elements with no hint
// (sz == nil), the upper bound is math.MaxUint64; if the parent container can be non-empty, sz is
// populated with UnknownSizeEstimate().
func resolveElemBounds(sz *SizeEstimate, nonEmptyContainer bool) (uint64, uint64, *SizeEstimate) {
	if sz != nil {
		return max(sz.Min, 1), max(sz.Max, 1), sz
	}
	if nonEmptyContainer {
		u := UnknownSizeEstimate()
		sz = &u
	}
	return 1, math.MaxUint64, sz
}

var defaultAggregateSizing SizingStrategy = &aggregateSizingStrategy{calc: types.NewSizeCalculator(types.SizeCalculatorStringUnitLength(1))}
