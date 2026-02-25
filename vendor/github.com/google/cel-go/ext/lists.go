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

package ext

import (
	"fmt"
	"math"
	"sort"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/decls"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/interpreter"
	"github.com/google/cel-go/parser"
)

var comparableTypes = []*cel.Type{
	cel.IntType,
	cel.UintType,
	cel.DoubleType,
	cel.BoolType,
	cel.DurationType,
	cel.TimestampType,
	cel.StringType,
	cel.BytesType,
}

// Lists returns a cel.EnvOption to configure extended functions for list manipulation.
// As a general note, all indices are zero-based.
//
// # Distinct
//
// Introduced in version: 2 (cost support in version 3)
//
// Returns the distinct elements of a list.
//
//	<list(T)>.distinct() -> <list(T)>
//
// Examples:
//
//	[1, 2, 2, 3, 3, 3].distinct() // return [1, 2, 3]
//	["b", "b", "c", "a", "c"].distinct() // return ["b", "c", "a"]
//	[1, "b", 2, "b"].distinct() // return [1, "b", 2]
//
// # Range
//
// Introduced in version: 2 (cost support in version 3)
//
// Returns a list of integers from 0 to n-1.
//
//	lists.range(<int>) -> <list(int)>
//
// Examples:
//
//	lists.range(5) -> [0, 1, 2, 3, 4]
//
// # Reverse
//
// Introduced in version: 2 (cost support in version 3)
//
// Returns the elements of a list in reverse order.
//
//	<list(T)>.reverse() -> <list(T)>
//
// Examples:
//
//	[5, 3, 1, 2].reverse() // return [2, 1, 3, 5]
//
// # Slice
//
// Introduced in version: 0 (cost support in version 3)
//
// Returns a new sub-list using the indexes provided.
//
//	<list>.slice(<int>, <int>) -> <list>
//
// Examples:
//
//	[1,2,3,4].slice(1, 3) // return [2, 3]
//	[1,2,3,4].slice(2, 4) // return [3 ,4]
//
// # Flatten
//
// Introduced in version: 1 (cost support in version 3)
//
// Flattens a list recursively.
// If an optional depth is provided, the list is flattened to a the specified level.
// A negative depth value will result in an error.
//
//	<list>.flatten() -> <list>
//	<list>.flatten(<int>) -> <list>
//
// Examples:
//
// [1,[2,3],[4]].flatten() // return [1, 2, 3, 4]
// [1,[2,[3,4]]].flatten() // return [1, 2, [3, 4]]
// [1,2,[],[],[3,4]].flatten() // return [1, 2, 3, 4]
// [1,[2,[3,[4]]]].flatten(2) // return [1, 2, 3, [4]]
// [1,[2,[3,[4]]]].flatten(-1) // error
//
// # Sort
//
// Introduced in version: 2 (cost support in version 3)
//
// Sorts a list with comparable elements. If the element type is not comparable
// or the element types are not the same, the function will produce an error.
//
//	<list(T)>.sort() -> <list(T)>
//	T in {int, uint, double, bool, duration, timestamp, string, bytes}
//
// Examples:
//
//	[3, 2, 1].sort() // return [1, 2, 3]
//	["b", "c", "a"].sort() // return ["a", "b", "c"]
//	[1, "b"].sort() // error
//	[[1, 2, 3]].sort() // error
//
// # SortBy
//
// Introduced in version: 2 (cost support in version 3)
//
// Sorts a list by a key value, i.e., the order is determined by the result of
// an expression applied to each element of the list.
// The output of the key expression must be a comparable type, otherwise the
// function will return an error.
//
//	<list(T)>.sortBy(<bindingName>, <keyExpr>) -> <list(T)>
//	keyExpr returns a value in {int, uint, double, bool, duration, timestamp, string, bytes}
//
// Examples:
//
//	[
//	  Player { name: "foo", score: 0 },
//	  Player { name: "bar", score: -10 },
//	  Player { name: "baz", score: 1000 },
//	].sortBy(e, e.score).map(e, e.name)
//	== ["bar", "foo", "baz"]
func Lists(options ...ListsOption) cel.EnvOption {
	l := &listsLib{version: math.MaxUint32}
	for _, o := range options {
		l = o(l)
	}
	return cel.Lib(l)
}

type listsLib struct {
	version uint32
}

// LibraryName implements the SingletonLibrary interface method.
func (listsLib) LibraryName() string {
	return "cel.lib.ext.lists"
}

// ListsOption is a functional interface for configuring the strings library.
type ListsOption func(*listsLib) *listsLib

// ListsVersion configures the version of the string library.
//
// The version limits which functions are available. Only functions introduced
// below or equal to the given version included in the library. If this option
// is not set, all functions are available.
//
// See the library documentation to determine which version a function was introduced.
// If the documentation does not state which version a function was introduced, it can
// be assumed to be introduced at version 0, when the library was first created.
func ListsVersion(version uint32) ListsOption {
	return func(lib *listsLib) *listsLib {
		lib.version = version
		return lib
	}
}

// CompileOptions implements the Library interface method.
func (lib listsLib) CompileOptions() []cel.EnvOption {
	listType := cel.ListType(cel.TypeParamType("T"))
	listListType := cel.ListType(listType)
	listDyn := cel.ListType(cel.DynType)
	opts := []cel.EnvOption{
		cel.Function("slice",
			cel.MemberOverload("list_slice",
				[]*cel.Type{listType, cel.IntType, cel.IntType}, listType,
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					list := args[0].(traits.Lister)
					start := args[1].(types.Int)
					end := args[2].(types.Int)
					result, err := slice(list, start, end)
					if err != nil {
						return types.WrapErr(err)
					}
					return result
				}),
			),
		),
	}
	if lib.version >= 1 {
		opts = append(opts,
			cel.Function("flatten",
				cel.MemberOverload("list_flatten",
					[]*cel.Type{listListType}, listType,
					cel.UnaryBinding(func(arg ref.Val) ref.Val {
						// double-check as type-guards disabled
						list, ok := arg.(traits.Lister)
						if !ok {
							return types.ValOrErr(arg, "no such overload: %v.flatten()", arg.Type())
						}
						flatList, err := flatten(list, 1)
						if err != nil {
							return types.WrapErr(err)
						}

						return types.DefaultTypeAdapter.NativeToValue(flatList)
					}),
				),
				cel.MemberOverload("list_flatten_int",
					[]*cel.Type{listDyn, types.IntType}, listDyn,
					cel.BinaryBinding(func(arg1, arg2 ref.Val) ref.Val {
						// double-check as type-guards disabled
						list, ok := arg1.(traits.Lister)
						if !ok {
							return types.ValOrErr(arg1, "no such overload: %v.flatten(%v)", arg1.Type(), arg2.Type())
						}
						depth, ok := arg2.(types.Int)
						if !ok {
							return types.ValOrErr(arg1, "no such overload: %v.flatten(%v)", arg1.Type(), arg2.Type())
						}
						flatList, err := flatten(list, int64(depth))
						if err != nil {
							return types.WrapErr(err)
						}

						return types.DefaultTypeAdapter.NativeToValue(flatList)
					}),
				),
				// To handle the case where a variable of just `list(T)` is provided at runtime
				// with a graceful failure more, disable the type guards since the implementation
				// can handle lists which are already flat.
				decls.DisableTypeGuards(true),
			),
		)
	}
	if lib.version >= 2 {
		sortDecl := cel.Function("sort",
			append(
				templatedOverloads(comparableTypes, func(t *cel.Type) cel.FunctionOpt {
					return cel.MemberOverload(
						fmt.Sprintf("list_%s_sort", t.TypeName()),
						[]*cel.Type{cel.ListType(t)}, cel.ListType(t),
					)
				}),
				cel.SingletonUnaryBinding(
					func(arg ref.Val) ref.Val {
						// validated by type-guards
						list := arg.(traits.Lister)
						sorted, err := sortList(list)
						if err != nil {
							return types.WrapErr(err)
						}

						return sorted
					},
					// List traits
					traits.ListerType,
				),
			)...,
		)
		opts = append(opts, sortDecl)
		opts = append(opts, cel.Macros(cel.ReceiverMacro("sortBy", 2, sortByMacro)))
		opts = append(opts, cel.Function("@sortByAssociatedKeys",
			append(
				templatedOverloads(comparableTypes, func(u *cel.Type) cel.FunctionOpt {
					return cel.MemberOverload(
						fmt.Sprintf("list_%s_sortByAssociatedKeys", u.TypeName()),
						[]*cel.Type{listType, cel.ListType(u)}, listType,
					)
				}),
				cel.SingletonBinaryBinding(
					func(arg1, arg2 ref.Val) ref.Val {
						// validated by type-guards
						list := arg1.(traits.Lister)
						keys := arg2.(traits.Lister)
						sorted, err := sortListByAssociatedKeys(list, keys)
						if err != nil {
							return types.WrapErr(err)
						}

						return sorted
					},
					// List traits
					traits.ListerType,
				),
			)...,
		))

		opts = append(opts, cel.Function("lists.range",
			cel.Overload("lists_range",
				[]*cel.Type{cel.IntType}, cel.ListType(cel.IntType),
				cel.UnaryBinding(func(n ref.Val) ref.Val {
					result, err := genRange(n.(types.Int))
					if err != nil {
						return types.WrapErr(err)
					}
					return result
				}),
			),
		))
		opts = append(opts, cel.Function("reverse",
			cel.MemberOverload("list_reverse",
				[]*cel.Type{listType}, listType,
				cel.UnaryBinding(func(list ref.Val) ref.Val {
					result, err := reverseList(list.(traits.Lister))
					if err != nil {
						return types.WrapErr(err)
					}
					return result
				}),
			),
		))
		opts = append(opts, cel.Function("distinct",
			cel.MemberOverload("list_distinct",
				[]*cel.Type{listType}, listType,
				cel.UnaryBinding(func(list ref.Val) ref.Val {
					result, err := distinctList(list.(traits.Lister))
					if err != nil {
						return types.WrapErr(err)
					}
					return result
				}),
			),
		))
	}
	if lib.version >= 3 {
		estimators := []checker.CostOption{
			checker.OverloadCostEstimate("list_slice", estimateListSlice),
			checker.OverloadCostEstimate("list_flatten", estimateListFlatten),
			checker.OverloadCostEstimate("list_flatten_int", estimateListFlatten),
			checker.OverloadCostEstimate("lists_range", estimateListsRange),
			checker.OverloadCostEstimate("list_reverse", estimateListReverse),
			checker.OverloadCostEstimate("list_distinct", estimateListDistinct),
		}
		for _, t := range comparableTypes {
			estimators = append(estimators,
				checker.OverloadCostEstimate(
					fmt.Sprintf("list_%s_sort", t.TypeName()),
					estimateListSort(t),
				),
				checker.OverloadCostEstimate(
					fmt.Sprintf("list_%s_sortByAssociatedKeys", t.TypeName()),
					estimateListSortBy(t),
				),
			)
		}
		opts = append(opts, cel.CostEstimatorOptions(estimators...))
	}

	return opts
}

// ProgramOptions implements the Library interface method.
func (lib *listsLib) ProgramOptions() []cel.ProgramOption {
	var opts []cel.ProgramOption
	if lib.version >= 3 {
		// TODO: Add cost trackers for list operations
		trackers := []interpreter.CostTrackerOption{
			interpreter.OverloadCostTracker("list_slice", trackListOutputSize),
			interpreter.OverloadCostTracker("list_flatten", trackListFlatten),
			interpreter.OverloadCostTracker("list_flatten_int", trackListFlatten),
			interpreter.OverloadCostTracker("lists_range", trackListOutputSize),
			interpreter.OverloadCostTracker("list_reverse", trackListOutputSize),
			interpreter.OverloadCostTracker("list_distinct", trackListDistinct),
		}
		for _, t := range comparableTypes {
			trackers = append(trackers,
				interpreter.OverloadCostTracker(
					fmt.Sprintf("list_%s_sort", t.TypeName()),
					trackListSort,
				),
				interpreter.OverloadCostTracker(
					fmt.Sprintf("list_%s_sortByAssociatedKeys", t.TypeName()),
					trackListSortBy,
				),
			)
		}
		opts = append(opts, cel.CostTrackerOptions(trackers...))
	}
	return opts
}

func genRange(n types.Int) (ref.Val, error) {
	var newList []ref.Val
	for i := types.Int(0); i < n; i++ {
		newList = append(newList, i)
	}
	return types.DefaultTypeAdapter.NativeToValue(newList), nil
}

func reverseList(list traits.Lister) (ref.Val, error) {
	var newList []ref.Val
	listLength := list.Size().(types.Int)
	for i := types.Int(0); i < listLength; i++ {
		val := list.Get(listLength - i - 1)
		newList = append(newList, val)
	}
	return types.DefaultTypeAdapter.NativeToValue(newList), nil
}

func slice(list traits.Lister, start, end types.Int) (ref.Val, error) {
	listLength := list.Size().(types.Int)
	if start < 0 || end < 0 {
		return nil, fmt.Errorf("cannot slice(%d, %d), negative indexes not supported", start, end)
	}
	if start > end {
		return nil, fmt.Errorf("cannot slice(%d, %d), start index must be less than or equal to end index", start, end)
	}
	if listLength < end {
		return nil, fmt.Errorf("cannot slice(%d, %d), list is length %d", start, end, listLength)
	}

	var newList []ref.Val
	for i := types.Int(start); i < end; i++ {
		val := list.Get(i)
		newList = append(newList, val)
	}
	return types.DefaultTypeAdapter.NativeToValue(newList), nil
}

func flatten(list traits.Lister, depth int64) ([]ref.Val, error) {
	if depth < 0 {
		return nil, fmt.Errorf("level must be non-negative")
	}

	var newList []ref.Val
	iter := list.Iterator()

	for iter.HasNext() == types.True {
		val := iter.Next()
		nestedList, isList := val.(traits.Lister)

		if !isList || depth == 0 {
			newList = append(newList, val)
			continue
		} else {
			flattenedList, err := flatten(nestedList, depth-1)
			if err != nil {
				return nil, err
			}

			newList = append(newList, flattenedList...)
		}
	}

	return newList, nil
}

func sortList(list traits.Lister) (ref.Val, error) {
	return sortListByAssociatedKeys(list, list)
}

// Internal function used for the implementation of sort() and sortBy().
//
// Sorts a list of arbitrary elements, according to the order produced by sorting
// another list of comparable elements. If the element type of the keys is not
// comparable or the element types are not the same, the function will produce an error.
//
//	<list(T)>.@sortByAssociatedKeys(<list(U)>) -> <list(T)>
//	U in {int, uint, double, bool, duration, timestamp, string, bytes}
//
// Example:
//
//	["foo", "bar", "baz"].@sortByAssociatedKeys([3, 1, 2]) // return ["bar", "baz", "foo"]
func sortListByAssociatedKeys(list, keys traits.Lister) (ref.Val, error) {
	listLength := list.Size().(types.Int)
	keysLength := keys.Size().(types.Int)
	if listLength != keysLength {
		return nil, fmt.Errorf(
			"@sortByAssociatedKeys() expected a list of the same size as the associated keys list, but got %d and %d elements respectively",
			listLength,
			keysLength,
		)
	}
	if listLength == 0 {
		return list, nil
	}
	elem := keys.Get(types.IntZero)
	if _, ok := elem.(traits.Comparer); !ok {
		return nil, fmt.Errorf("list elements must be comparable")
	}

	sortedIndices := make([]ref.Val, 0, listLength)
	for i := types.IntZero; i < listLength; i++ {
		sortedIndices = append(sortedIndices, i)
	}

	var err error
	sort.Slice(sortedIndices, func(i, j int) bool {
		iKey := keys.Get(sortedIndices[i])
		jKey := keys.Get(sortedIndices[j])
		if iKey.Type() != elem.Type() || jKey.Type() != elem.Type() {
			err = fmt.Errorf("list elements must have the same type")
			return false
		}
		return iKey.(traits.Comparer).Compare(jKey) == types.IntNegOne
	})
	if err != nil {
		return nil, err
	}

	sorted := make([]ref.Val, 0, listLength)
	for _, sortedIdx := range sortedIndices {
		sorted = append(sorted, list.Get(sortedIdx))
	}
	return types.DefaultTypeAdapter.NativeToValue(sorted), nil
}

// sortByMacro transforms an expression like:
//
//	mylistExpr.sortBy(e, -math.abs(e))
//
// into something equivalent to:
//
//	cel.bind(
//	   __sortBy_input__,
//	   myListExpr,
//	   __sortBy_input__.@sortByAssociatedKeys(__sortBy_input__.map(e, -math.abs(e))
//	)
func sortByMacro(meh cel.MacroExprFactory, target ast.Expr, args []ast.Expr) (ast.Expr, *cel.Error) {
	varIdent := meh.NewIdent("@__sortBy_input__")
	varName := varIdent.AsIdent()

	targetKind := target.Kind()
	if targetKind != ast.ListKind &&
		targetKind != ast.SelectKind &&
		targetKind != ast.IdentKind &&
		targetKind != ast.ComprehensionKind &&
		targetKind != ast.CallKind {
		return nil, meh.NewError(target.ID(), "sortBy can only be applied to a list, identifier, comprehension, call or select expression")
	}

	mapCompr, err := parser.MakeMap(meh, meh.Copy(varIdent), args)
	if err != nil {
		return nil, err
	}
	callExpr := meh.NewMemberCall("@sortByAssociatedKeys",
		meh.Copy(varIdent),
		mapCompr,
	)

	bindExpr := meh.NewComprehension(
		meh.NewList(),
		"#unused",
		varName,
		target,
		meh.NewLiteral(types.False),
		varIdent,
		callExpr,
	)

	return bindExpr, nil
}

func distinctList(list traits.Lister) (ref.Val, error) {
	listLength := list.Size().(types.Int)
	if listLength == 0 {
		return list, nil
	}
	uniqueList := make([]ref.Val, 0, listLength)
	for i := types.IntZero; i < listLength; i++ {
		val := list.Get(i)
		seen := false
		for j := types.IntZero; j < types.Int(len(uniqueList)); j++ {
			if i == j {
				continue
			}
			other := uniqueList[j]
			if val.Equal(other) == types.True {
				seen = true
				break
			}
		}
		if !seen {
			uniqueList = append(uniqueList, val)
		}
	}

	return types.DefaultTypeAdapter.NativeToValue(uniqueList), nil
}

func templatedOverloads(types []*cel.Type, template func(t *cel.Type) cel.FunctionOpt) []cel.FunctionOpt {
	overloads := make([]cel.FunctionOpt, len(types))
	for i, t := range types {
		overloads[i] = template(t)
	}
	return overloads
}

// estimateListSlice computes an O(n) slice operation with a cost factor of 1.
func estimateListSlice(estimator checker.CostEstimator, target *checker.AstNode, args []checker.AstNode) *checker.CallEstimate {
	if target == nil || len(args) != 2 {
		return nil
	}
	sz := estimateSize(estimator, *target)
	start := nodeAsIntValue(args[0], 0)
	end := nodeAsIntValue(args[1], sz.Max)
	return estimateAllocatingListCall(1, checker.FixedSizeEstimate(end-start))
}

// estimateListsRange computes an O(n) range operation with a cost factor of 1.
func estimateListsRange(estimator checker.CostEstimator, target *checker.AstNode, args []checker.AstNode) *checker.CallEstimate {
	if target != nil || len(args) != 1 {
		return nil
	}
	return estimateAllocatingListCall(1, checker.FixedSizeEstimate(nodeAsIntValue(args[0], math.MaxUint)))
}

// estimateListReverse computes an O(n) reverse operation with a cost factor of 1.
func estimateListReverse(estimator checker.CostEstimator, target *checker.AstNode, args []checker.AstNode) *checker.CallEstimate {
	if target == nil || len(args) != 0 {
		return nil
	}
	return estimateAllocatingListCall(1, estimateSize(estimator, *target))
}

// estimateListFlatten computes an O(n) flatten operation with a cost factor proportional to the flatten depth.
func estimateListFlatten(estimator checker.CostEstimator, target *checker.AstNode, args []checker.AstNode) *checker.CallEstimate {
	if target == nil || len(args) > 1 {
		return nil
	}
	depth := uint64(1)
	if len(args) == 1 {
		depth = nodeAsIntValue(args[0], math.MaxUint)
	}
	return estimateAllocatingListCall(float64(depth), estimateSize(estimator, *target))
}

// Compute an O(n^2) with a cost factor of 2, equivalent to sets.contains with a result list
// which can vary in size from 1 element to the original list size.
func estimateListDistinct(estimator checker.CostEstimator, target *checker.AstNode, args []checker.AstNode) *checker.CallEstimate {
	if target == nil || len(args) != 0 {
		return nil
	}
	sz := estimateSize(estimator, *target)
	costFactor := 2.0
	return estimateAllocatingListCall(costFactor, sz.Multiply(sz))
}

// estimateListSort computes an O(n^2) sort operation with a cost factor of 2 for the equality
// operations against the elements in the list against themselves which occur during the sort computation.
func estimateListSort(t *types.Type) checker.FunctionEstimator {
	return func(estimator checker.CostEstimator, target *checker.AstNode, args []checker.AstNode) *checker.CallEstimate {
		if target == nil || len(args) != 0 {
			return nil
		}
		return estimateListSortCost(estimator, *target, t)
	}
}

// estimateListSortBy computes an O(n^2) sort operation with a cost factor of 2 for the equality
// operations against the sort index list which occur during the sort computation.
func estimateListSortBy(u *types.Type) checker.FunctionEstimator {
	return func(estimator checker.CostEstimator, target *checker.AstNode, args []checker.AstNode) *checker.CallEstimate {
		if target == nil || len(args) != 1 {
			return nil
		}
		// Estimate the size of the list used as the sort index
		return estimateListSortCost(estimator, args[0], u)
	}
}

// estimateListSortCost estimates an O(n^2) sort operation with a cost factor of 2 for the equality
// operations which occur during the sort computation.
func estimateListSortCost(estimator checker.CostEstimator, node checker.AstNode, elemType *types.Type) *checker.CallEstimate {
	sz := estimateSize(estimator, node)
	costFactor := 2.0
	switch elemType {
	case types.StringType, types.BytesType:
		costFactor += common.StringTraversalCostFactor
	}
	return estimateAllocatingListCall(costFactor, sz.Multiply(sz))
}

// estimateAllocatingListCall computes cost as a function of the size of the result list with a
// baseline cost for the call dispatch and the associated list allocation.
func estimateAllocatingListCall(costFactor float64, listSize checker.SizeEstimate) *checker.CallEstimate {
	return estimateListCall(costFactor, listSize, true)
}

// estimateListCall computes cost as a function of the size of the target list and whether the
// call allocates memory.
func estimateListCall(costFactor float64, listSize checker.SizeEstimate, allocates bool) *checker.CallEstimate {
	cost := listSize.MultiplyByCostFactor(costFactor).Add(callCostEstimate)
	if allocates {
		cost = cost.Add(checker.FixedCostEstimate(common.ListCreateBaseCost))
	}
	return &checker.CallEstimate{CostEstimate: cost, ResultSize: &listSize}
}

// trackListOutputSize computes cost as a function of the size of the result list.
func trackListOutputSize(_ []ref.Val, result ref.Val) *uint64 {
	return trackAllocatingListCall(1, actualSize(result))
}

// trackListFlatten computes cost as a function of the size of the result list and the depth of
// the flatten operation.
func trackListFlatten(args []ref.Val, _ ref.Val) *uint64 {
	depth := 1.0
	if len(args) == 2 {
		depth = float64(args[1].(types.Int))
	}
	inputSize := actualSize(args[0])
	return trackAllocatingListCall(depth, inputSize)
}

// trackListDistinct computes costs as a worst-case O(n^2) operation over the input list.
func trackListDistinct(args []ref.Val, _ ref.Val) *uint64 {
	return trackListSelfCompare(args[0].(traits.Lister))
}

// trackListSort computes costs as a worst-case O(n^2) operation over the input list.
func trackListSort(args []ref.Val, result ref.Val) *uint64 {
	return trackListSelfCompare(args[0].(traits.Lister))
}

// trackListSortBy computes costs as a worst-case O(n^2) operation over the sort index list.
func trackListSortBy(args []ref.Val, result ref.Val) *uint64 {
	return trackListSelfCompare(args[1].(traits.Lister))
}

// trackListSelfCompare computes costs as a worst-case O(n^2) operation over the input list.
func trackListSelfCompare(l traits.Lister) *uint64 {
	sz := actualSize(l)
	costFactor := 2.0
	if sz == 0 {
		return trackAllocatingListCall(costFactor, 0)
	}
	elem := l.Get(types.IntZero)
	if elem.Type() == types.StringType || elem.Type() == types.BytesType {
		costFactor += common.StringTraversalCostFactor
	}
	return trackAllocatingListCall(costFactor, sz*sz)
}

// trackAllocatingListCall computes costs as a function of the size of the result list with a baseline cost
// for the call dispatch and the associated list allocation.
func trackAllocatingListCall(costFactor float64, size uint64) *uint64 {
	cost := uint64(float64(size)*costFactor) + callCost + common.ListCreateBaseCost
	return &cost
}

func nodeAsIntValue(node checker.AstNode, defaultVal uint64) uint64 {
	if node.Expr().Kind() != ast.LiteralKind {
		return defaultVal
	}
	lit := node.Expr().AsLiteral()
	if lit.Type() != types.IntType {
		return defaultVal
	}
	val := lit.(types.Int)
	if val < types.IntZero {
		return 0
	}
	return uint64(lit.(types.Int))
}
