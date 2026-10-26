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
	"slices"

	"cel.dev/cel-go/common/ast"
	"cel.dev/cel-go/common/operators"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
)

// DefaultSizingStrategy returns the default SizingStrategy implementation.
func DefaultSizingStrategy() SizingStrategy {
	return defaultSizing
}

// defaultSizingStrategy is the default implementation of SizingStrategy.
type defaultSizingStrategy struct{}

// EstimateSize computes the size estimate for an AST node during cost estimation.
func (defaultSizingStrategy) EstimateSize(ctx EstimateContext, node AstNode) (SizeEstimate, bool) {
	if node == nil {
		return SizeEstimate{}, false
	}
	if sz := node.ComputedSize(); sz != nil {
		return *sz, true
	}
	if node.Expr() != nil {
		if sz := computeExprSize(node.Expr()); sz != nil {
			return *sz, true
		}
	}
	if node.Type() == nil {
		return estimateScalarOrFallback(ctx, node)
	}
	switch node.Type().Kind() {
	case types.ListKind:
		return estimateDefaultListSize(ctx, node)
	case types.MapKind:
		return estimateDefaultMapSize(ctx, node)
	case types.DynKind, types.AnyKind, types.TypeParamKind:
		return estimateDefaultDynSize(ctx, node)
	default:
		return estimateScalarOrFallback(ctx, node)
	}
}

// maxHintDepth bounds how many nested container hint levels ("@items", "@keys", "@values")
// dynamic type size estimation inspects.
const maxHintDepth = 32

// estimateDefaultListSize calculates the size and element size of a list AST node.
func estimateDefaultListSize(ctx EstimateContext, node AstNode) (SizeEstimate, bool) {
	elemType := listElemType(node.Type())
	listSize, elemSize := estimateListExpr(ctx, node, elemType)
	listSize, elemSize = resolveListSizes(ctx, node, elemType, listSize, elemSize)
	return combineListSize(listSize, elemSize)
}

// estimateDefaultMapSize calculates the size, key size, and value size of a map AST node.
func estimateDefaultMapSize(ctx EstimateContext, node AstNode) (SizeEstimate, bool) {
	keyType, valType := mapKeyValueTypes(node.Type())
	mapSize, keySize, valSize := estimateMapExpr(ctx, node, keyType, valType)
	mapSize, keySize, valSize = resolveMapSizes(ctx, node, keyType, valType, mapSize, keySize, valSize)
	return combineMapSize(mapSize, keySize, valSize)
}

// estimateDefaultDynSize calculates the size estimate for a dynamically typed AST node.
// For variable/path nodes, the nesting of hints ("@items", "@keys", "@values") determines
// the possible container depth; any level beyond the hinted depth remains nil (unknown).
func estimateDefaultDynSize(ctx EstimateContext, node AstNode) (SizeEstimate, bool) {
	if node.Expr() != nil {
		switch node.Expr().Kind() {
		case ast.ListKind:
			return estimateDefaultListSize(ctx, node)
		case ast.MapKind:
			return estimateDefaultMapSize(ctx, node)
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
			return combineListSize(baseSz, elemSz)
		}
		keySz := resolveDynamicChild(ctx, node.Path(), "@keys")
		valSz := resolveDynamicChild(ctx, node.Path(), "@values")
		if keySz != nil || valSz != nil {
			return combineMapSize(baseSz, keySz, valSz)
		}
	}
	if baseSz != nil {
		return *baseSz, true
	}
	return SizeEstimate{}, false
}

// resolveDynamicChild probes a child hint subpath ("@items", "@keys", "@values") under a
// dynamic node, recursing only when the estimator provides an explicit hint at that subpath.
func resolveDynamicChild(ctx EstimateContext, parentPath []string, subPath string) *SizeEstimate {
	if len(parentPath) == 0 || len(parentPath) >= maxHintDepth || ctx == nil || ctx.Estimator() == nil {
		return nil
	}
	childPath := append(slices.Clone(parentPath), subPath)
	childNode := NewAstNode(nil, childPath, types.DynType, nil)
	if ctx.Estimator().EstimateSize(childNode) == nil {
		return nil
	}
	sz := ctx.Size(childNode)
	if sz.IsUnknown() {
		return nil
	}
	return &sz
}

// resolveListSizes resolves a list node's container size and element size using the estimator,
// subpath hints ("@items"), and type fallbacks when not already determined from the expression.
func resolveListSizes(ctx EstimateContext, node AstNode, elemType *types.Type, listSize, elemSize *SizeEstimate) (*SizeEstimate, *SizeEstimate) {
	if listSize == nil && ctx != nil && ctx.Estimator() != nil {
		listSize = ctx.Estimator().EstimateSize(node)
	}
	if elemSize == nil && listSize != nil {
		elemSize = cloneSizeEstimate(listSize.Elem)
	}
	if listSize == nil || listSize.Max > 0 || (ctx != nil && versionOf(ctx) < 1) {
		elemSize = resolveChildSize(ctx, node.Path(), "@items", elemType, elemSize)
	}
	// If neither the list length nor element size is known for an AST expression node
	// (e.g. an unhinted list<string> variable), default listSize to unknown so the
	// strategy computes the worst-case bound.
	if listSize == nil && elemSize == nil && node.Expr() != nil {
		u := UnknownSizeEstimate()
		listSize = &u
	}
	return listSize, elemSize
}

// resolveMapSizes resolves a map node's container size, key size, and value size using the
// estimator, subpath hints ("@keys", "@values"), and type fallbacks.
func resolveMapSizes(ctx EstimateContext, node AstNode, keyType, valType *types.Type, mapSize, keySize, valSize *SizeEstimate) (*SizeEstimate, *SizeEstimate, *SizeEstimate) {
	if mapSize == nil && ctx != nil && ctx.Estimator() != nil {
		mapSize = ctx.Estimator().EstimateSize(node)
	}
	if mapSize != nil {
		if keySize == nil {
			keySize = cloneSizeEstimate(mapSize.Key)
		}
		if valSize == nil {
			valSize = cloneSizeEstimate(mapSize.Elem)
		}
	}
	// Query subpaths @keys and @values if hints were not provided on the parent map node.
	if mapSize == nil || mapSize.Max > 0 || (ctx != nil && versionOf(ctx) < 1) {
		keySize = resolveChildSize(ctx, node.Path(), "@keys", keyType, keySize)
		valSize = resolveChildSize(ctx, node.Path(), "@values", valType, valSize)
	}
	// If no map, key, or value sizes are known for an AST expression node (e.g. an unhinted
	// map<string, string>), default mapSize to unknown to compute the worst-case bound.
	if mapSize == nil && keySize == nil && valSize == nil && node.Expr() != nil {
		u := UnknownSizeEstimate()
		mapSize = &u
	}
	return mapSize, keySize, valSize
}

// resolveChildSize looks up child size estimates on the given subPath (e.g. "@items", "@keys", "@values")
// when the current estimate is missing or lacks nested container details, falling back to primitive/type hints.
func resolveChildSize(ctx EstimateContext, parentPath []string, subPath string, childType *types.Type, current *SizeEstimate) *SizeEstimate {
	if current == nil || (current.Elem == nil && (isContainerKind(childType.Kind()) || isDynamicKind(childType.Kind()))) {
		if len(parentPath) > 0 && ctx != nil {
			childPath := append(slices.Clone(parentPath), subPath)
			sz := ctx.Size(NewAstNode(nil, childPath, childType, nil))
			if !sz.IsUnknown() {
				if current == nil {
					current = &sz
				} else {
					current.Elem = sz.Elem
					current.Key = sz.Key
				}
			}
		}
	}
	if current == nil {
		current = fallbackElemSize(ctx, childType)
	}
	return current
}

// isContainerKind reports whether kind is a list or map container type.
func isContainerKind(kind types.Kind) bool {
	return kind == types.ListKind || kind == types.MapKind
}

// isDynamicKind reports whether kind represents a dynamically typed value.
func isDynamicKind(kind types.Kind) bool {
	return kind == types.DynKind || kind == types.AnyKind || kind == types.TypeParamKind
}

// fallbackElemSize resolves fixed-width primitive type sizes via computeTypeSize,
// or queries the estimator with an untyped/pathless node for type-level hints.
func fallbackElemSize(ctx EstimateContext, elemType *types.Type) *SizeEstimate {
	if sz := computeTypeSize(elemType, versionOf(ctx)); sz != nil {
		return ownedSize(sz)
	}
	if ctx != nil && ctx.Estimator() != nil {
		elemNode := NewAstNode(nil, nil, elemType, nil)
		return ctx.Estimator().EstimateSize(elemNode)
	}
	return nil
}

// cloneSizeEstimate makes a copy of a SizeEstimate to prevent in-place mutation of cached/shared estimates.
func cloneSizeEstimate(se *SizeEstimate) *SizeEstimate {
	if se == nil {
		return nil
	}
	clone := *se
	return &clone
}

// TrackSize computes the actual runtime size of a value.
func (defaultSizingStrategy) TrackSize(ctx TrackContext, value ref.Val) (uint64, bool) {
	if value == nil {
		return 0, false
	}
	if opt, isOpt := value.(*types.Optional); isOpt {
		return optionalLength(trackVersionOf(ctx), opt), true
	}
	return ActualSize(value), true
}

// optionalLength reports the size of an optional under the default sizing strategy, at the given
// cost model version.
//
// From version 1 an optional is 1 plus the length of the value it holds, so an empty optional is 1,
// and the optional's element is that value; see Optional. Version 0 sizes every optional as 1, as
// cel-go v0.32.0 charged it through ActualSize, for which an optional is not a traits.Sizer.
func optionalLength(version uint32, opt *types.Optional) uint64 {
	if version < 1 || !opt.HasValue() {
		return 1
	}
	if inner, isOpt := opt.GetValue().(*types.Optional); isOpt {
		return SafeAdd(1, optionalLength(version, inner))
	}
	return SafeAdd(1, ActualSize(opt.GetValue()))
}

// trackVersionContext is implemented by TrackContext values which know the cost model version they
// track under. It is unexported because TrackContext is an interface third parties may implement.
type trackVersionContext interface {
	modelVersion() uint32
}

// trackVersionOf reports the cost model version of a TrackContext: the latest version for a context
// which does not report one.
func trackVersionOf(ctx TrackContext) uint32 {
	if v, ok := ctx.(trackVersionContext); ok {
		return v.modelVersion()
	}
	return defaultModelVersion
}

// optionalValueOf reports whether value is an optional, and if so returns the value it holds, or
// nil when the optional is empty.
func optionalValueOf(value ref.Val) (ref.Val, bool) {
	opt, isOpt := value.(*types.Optional)
	if !isOpt {
		return nil, false
	}
	if !opt.HasValue() {
		return nil, true
	}
	return opt.GetValue(), true
}

// listElemType extracts the element type of a list, defaulting to DynType if not parameterized.
func listElemType(t *types.Type) *types.Type {
	if len(t.Parameters()) > 0 && t.Parameters()[0] != nil {
		return t.Parameters()[0]
	}
	return types.DynType
}

// mapKeyValueTypes extracts key and value types of a map, defaulting to DynType if not parameterized.
func mapKeyValueTypes(t *types.Type) (keyType, valType *types.Type) {
	keyType = types.DynType
	valType = types.DynType
	if len(t.Parameters()) >= 2 {
		keyType = t.Parameters()[0]
		valType = t.Parameters()[1]
	}
	return keyType, valType
}

// unionAccumulator merges a new size estimate into an existing accumulator pointer.
func unionAccumulator(acc *SizeEstimate, next SizeEstimate) *SizeEstimate {
	if acc == nil {
		return &next
	}
	u := acc.Union(next)
	return &u
}

// estimateConditionalSize computes the union SizeEstimate of the branches in a conditional call.
func estimateConditionalSize(ctx EstimateContext, nodeType *types.Type, args []ast.Expr) (SizeEstimate, bool) {
	if len(args) == 3 {
		tVal := ctx.Size(NewAstNode(args[1], nil, nodeType, nil))
		fVal := ctx.Size(NewAstNode(args[2], nil, nodeType, nil))
		return tVal.Union(fVal), true
	}
	return SizeEstimate{}, false
}

// estimateListExpr computes the list and element size estimates for literal list expressions
// or container call operations (e.g. Add, Conditional).
func estimateListExpr(ctx EstimateContext, node AstNode, elemType *types.Type) (listSize *SizeEstimate, elemSize *SizeEstimate) {
	if node.Expr() == nil {
		return nil, nil
	}
	switch node.Expr().Kind() {
	case ast.ListKind:
		list := node.Expr().AsList()
		elements := list.Elements()
		l := FixedSizeEstimate(uint64(len(elements)))
		// An optional element is omitted when it has no value, so only the required
		// elements count toward the minimum length.
		l.Min = uint64(len(elements) - len(list.OptionalIndices()))
		optional := make(map[int32]bool, len(list.OptionalIndices()))
		for _, i := range list.OptionalIndices() {
			optional[i] = true
		}
		for i, elem := range elements {
			if optional[int32(i)] {
				// The list holds the optional's value, if any, rather than the optional.
				var present bool
				if elem, present = optionalLiteralValue(elem); !present {
					l.Max = SafeSubtract(l.Max, 1)
					continue
				}
			}
			elemNode := NewAstNode(elem, nil, elemType, nil)
			elemSize = unionAccumulator(elemSize, ctx.Size(elemNode))
		}
		listSize = &l
	case ast.CallKind:
		call := node.Expr().AsCall()
		args := call.Args()
		switch call.FunctionName() {
		case operators.Add:
			if len(args) == 2 {
				lhs := ctx.Size(NewAstNode(args[0], nil, node.Type(), nil))
				rhs := ctx.Size(NewAstNode(args[1], nil, node.Type(), nil))
				added := lhs.Add(rhs)
				listSize = &added
				elemSize = added.Elem
			}
		case operators.Conditional:
			if u, ok := estimateConditionalSize(ctx, node.Type(), args); ok {
				listSize = &u
				elemSize = u.Elem
			}
		}
	}
	return listSize, elemSize
}

// estimateMapExpr computes map, key, and value size estimates for literal map expressions
// or container call operations (e.g. Conditional).
func estimateMapExpr(ctx EstimateContext, node AstNode, keyType, valType *types.Type) (mapSize *SizeEstimate, keySize *SizeEstimate, valSize *SizeEstimate) {
	if node.Expr() == nil {
		return nil, nil, nil
	}
	switch node.Expr().Kind() {
	case ast.MapKind:
		entries := node.Expr().AsMap().Entries()
		m := FixedSizeEstimate(uint64(len(entries)))
		mapSize = &m
		for _, entry := range entries {
			mapEntry := entry.AsMapEntry()
			if mapEntry.IsOptional() {
				// An optional entry is omitted when its value is absent.
				m.Min = SafeSubtract(m.Min, 1)
			}
			kNode := NewAstNode(mapEntry.Key(), nil, keyType, nil)
			vNode := NewAstNode(mapEntry.Value(), nil, valType, nil)
			keySize = unionAccumulator(keySize, ctx.Size(kNode))
			valSize = unionAccumulator(valSize, ctx.Size(vNode))
		}
	case ast.CallKind:
		call := node.Expr().AsCall()
		switch call.FunctionName() {
		case "cel.@mapInsert":
			args := call.Args()
			if len(args) == 3 {
				lhs := ctx.Size(NewAstNode(args[0], nil, node.Type(), nil))
				kSz := ctx.Size(NewAstNode(args[1], nil, keyType, nil))
				vSz := ctx.Size(NewAstNode(args[2], nil, valType, nil))
				keySize = mergeSizeEstimatePtr(lhs.Key, &kSz)
				valSize = mergeSizeEstimatePtr(lhs.Elem, &vSz)
				added := lhs.Add(FixedSizeEstimate(1))
				added.Key = keySize
				added.Elem = valSize
				mapSize = &added
			} else if len(args) == 2 {
				lhs := ctx.Size(NewAstNode(args[0], nil, node.Type(), nil))
				rhs := ctx.Size(NewAstNode(args[1], nil, node.Type(), nil))
				added := lhs.Add(rhs)
				mapSize = &added
				keySize = added.Key
				valSize = added.Elem
			}
		case operators.Conditional:
			if u, ok := estimateConditionalSize(ctx, node.Type(), call.Args()); ok {
				mapSize = &u
				keySize = u.Key
				valSize = u.Elem
			}
		}
	}
	return mapSize, keySize, valSize
}

// estimateScalarOrFallback returns size estimates using custom estimator hints or primitive type sizes.
func estimateScalarOrFallback(ctx EstimateContext, node AstNode) (SizeEstimate, bool) {
	if node != nil && node.Expr() != nil {
		if sz := computeExprSize(node.Expr()); sz != nil {
			return *sz, true
		}
	}
	if ctx != nil && ctx.Estimator() != nil {
		if sz := ctx.Estimator().EstimateSize(node); sz != nil {
			return *sz, true
		}
	}
	if node.Type() != nil {
		if sz := computeTypeSize(node.Type(), versionOf(ctx)); sz != nil {
			return *sz, true
		}
	}
	return SizeEstimate{}, false
}

// combineListSize constructs a list SizeEstimate combining container length and element size.
//
// A list that can be non-empty (Max > 0) whose element size is unknown carries an explicit
// UnknownSizeEstimate in Elem rather than nil, so combining it with a literal list (e.g.
// ['a'] + unhinted) widens the element size to unknown instead of dropping it to the literal's.
// Empty list literals (Max == 0) leave Elem nil so they contribute no element bound when joined.
func combineListSize(listSize, elemSize *SizeEstimate) (SizeEstimate, bool) {
	if listSize != nil {
		if elemSize == nil && listSize.Max > 0 {
			u := UnknownSizeEstimate()
			elemSize = &u
		}
		listSize.Key = nil
		listSize.Elem = elemSize
		return *listSize, true
	}
	if elemSize != nil {
		res := UnknownSizeEstimate()
		res.Elem = elemSize
		return res, true
	}
	return SizeEstimate{}, false
}

// combineMapSize constructs a map SizeEstimate combining map size, key size, and value size.
//
// As with combineListSize, a map that can be non-empty (Max > 0) records an explicit
// UnknownSizeEstimate for an unhinted Key or Elem so merging with a literal map widens the bound
// rather than dropping the unknown side.
func combineMapSize(mapSize, keySize, valSize *SizeEstimate) (SizeEstimate, bool) {
	if mapSize != nil {
		if mapSize.Max > 0 {
			if keySize == nil {
				u := UnknownSizeEstimate()
				keySize = &u
			}
			if valSize == nil {
				u := UnknownSizeEstimate()
				valSize = &u
			}
		}
		mapSize.Key = keySize
		mapSize.Elem = valSize
		return *mapSize, true
	}
	if keySize != nil || valSize != nil {
		if keySize == nil {
			u := UnknownSizeEstimate()
			keySize = &u
		}
		if valSize == nil {
			u := UnknownSizeEstimate()
			valSize = &u
		}
		res := UnknownSizeEstimate()
		res.Key = keySize
		res.Elem = valSize
		return res, true
	}
	return SizeEstimate{}, false
}

var defaultSizing SizingStrategy = defaultSizingStrategy{}

// optionalLiteralValue returns the value an optional list element contributes: nothing for
// optional.none(), the argument of optional.of(x) or optional.ofNonZeroValue(x), and otherwise
// the element itself, whose value is not known statically.
func optionalLiteralValue(e ast.Expr) (ast.Expr, bool) {
	if e.Kind() != ast.CallKind {
		return e, true
	}
	call := e.AsCall()
	if call.IsMemberFunction() {
		return e, true
	}
	switch call.FunctionName() {
	case "optional.none":
		if len(call.Args()) == 0 {
			return nil, false
		}
	case "optional.of", "optional.ofNonZeroValue":
		if len(call.Args()) == 1 {
			return call.Args()[0], true
		}
	}
	return e, true
}
