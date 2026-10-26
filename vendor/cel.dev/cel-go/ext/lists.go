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
	"slices"
	"sort"

	"cel.dev/cel-go/cel"
	"cel.dev/cel-go/common/ast"
	"cel.dev/cel-go/common/cost"
	"cel.dev/cel-go/common/decls"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
	"cel.dev/cel-go/common/types/traits"
	"cel.dev/cel-go/parser"
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
//
// # HasOnly
//
// Introduced in version: 5
//
// Returns whether every element of the target list is contained within the
// argument list, i.e. whether the target is a subset of the argument. Standard
// CEL equality is used to determine whether two elements are equal, and neither
// list is required to contain unique elements. An empty target list always
// returns true.
//
//	<list(T)>.hasOnly(<list(T)>) -> <bool>
//
// Examples:
//
//	[].hasOnly([1, 2]) // return true
//	[1, 1, 2].hasOnly([1, 2, 3]) // return true
//	[1, 2.0, 3u].hasOnly([1.0, 2u, 3]) // return true
//	[1, 4].hasOnly([1, 2, 3]) // return false
//	[1].hasOnly([]) // return false
//
// # HasAny
//
// Introduced in version: 5
//
// Returns whether the target list has at least one element which is equal to an
// element of the argument list. If either list is empty, the result is false.
//
//	<list(T)>.hasAny(<list(T)>) -> <bool>
//
// Examples:
//
//	[1, 2, 3].hasAny([3, 4]) // return true
//	[[1], [2, 3]].hasAny([[1, 2], [2, 3.0]]) // return true
//	[1, 2].hasAny([3, 4]) // return false
//	[1].hasAny([]) // return false
//
// # HasAll
//
// Introduced in version: 5
//
// Returns whether every element of the argument list is contained within the
// target list, i.e. whether the target is a superset of the argument. An empty
// argument list always returns true.
//
//	<list(T)>.hasAll(<list(T)>) -> <bool>
//
// Examples:
//
//	[1, 2, 3].hasAll([]) // return true
//	[1, 2, 3, 4].hasAll([2, 3]) // return true
//	[1, 2.0, 3u].hasAll([1.0, 2u, 3]) // return true
//	[1, 2, 3].hasAll([2, 4]) // return false
//
// # HasExactly
//
// Introduced in version: 5
//
// Returns whether the target and argument lists are set equivalent, i.e. every
// element of the target list is equal to an element of the argument list and
// vice versa. The lists may differ in size since neither is guaranteed to hold
// unique elements, so size does not factor into the computation.
//
//	<list(T)>.hasExactly(<list(T)>) -> <bool>
//
// Examples:
//
//	[].hasExactly([]) // return true
//	[1].hasExactly([1, 1]) // return true
//	[1].hasExactly([1u, 1.0]) // return true
//	[1, 2, 3].hasExactly([3u, 2.0, 1]) // return true
//	[1, 2].hasExactly([1, 2, 3]) // return false
func Lists(options ...ListsOption) cel.EnvOption {
	l := &listsLib{version: math.MaxUint32, maxRangeSize: defaultMaxRangeSize}
	for _, o := range options {
		l = o(l)
	}
	return cel.Lib(l)
}

const defaultMaxRangeSize = 1_000_000

type listsLib struct {
	version      uint32
	maxRangeSize int64
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

// ListsMaxRangeSize sets the maximum number of elements lists.range() will
// allocate. If not set, the default is 1,000,000. Setting this to zero
// disables the limit (not recommended).
func ListsMaxRangeSize(size int64) ListsOption {
	return func(lib *listsLib) *listsLib {
		lib.maxRangeSize = size
		return lib
	}
}

// CompileOptions implements the Library interface method.
func (lib listsLib) CompileOptions() []cel.EnvOption {
	listType := cel.ListType(cel.TypeParamType("T"))
	listListType := cel.ListType(listType)
	listDyn := cel.ListType(cel.DynType)
	var adapt func() types.Adapter
	captureAdapt := func(e *cel.Env) (*cel.Env, error) {
		adapt = e.CELTypeAdapter
		return e, nil
	}
	opts := []cel.EnvOption{
		captureAdapt,
		cel.Function("slice",
			cel.MemberOverload("list_slice",
				[]*cel.Type{listType, cel.IntType, cel.IntType}, listType,
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					list := args[0].(traits.Lister)
					start := args[1].(types.Int)
					end := args[2].(types.Int)
					result, err := slice(adapt(), list, start, end)
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

						return adapt().NativeToValue(flatList)
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

						return adapt().NativeToValue(flatList)
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
						sorted, err := sortList(adapt(), list)
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
						sorted, err := sortListByAssociatedKeys(adapt(), list, keys)
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

		maxRange := lib.maxRangeSize
		opts = append(opts, cel.Function("lists.range",
			cel.Overload("lists_range",
				[]*cel.Type{cel.IntType}, cel.ListType(cel.IntType),
				cel.UnaryBinding(func(n ref.Val) ref.Val {
					result, err := genRange(adapt(), n.(types.Int), maxRange)
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
					result, err := reverseList(adapt(), list.(traits.Lister))
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
					result, err := distinctList(adapt(), list.(traits.Lister))
					if err != nil {
						return types.WrapErr(err)
					}
					return result
				}),
			),
		))
	}
	if lib.version >= 3 {
		legacy := lib.version == 3
		opts = append(opts,
			cel.CostModel(
				listSliceCostModel("list_slice"),
				listsRangeCostModel("lists_range"),
				listReverseCostModel("list_reverse"),
				listDistinctCostModel("list_distinct", legacy),
				listFlattenCostModel("list_flatten", legacy),
				listFlattenDepthCostModel("list_flatten_int", legacy),
			),
			cel.CostModel(listSortCostModels(legacy)...),
		)
	}
	if lib.version >= 5 {
		opts = append(opts,
			cel.Function("hasOnly",
				cel.MemberOverload("list_hasOnly_list",
					[]*cel.Type{listType, listType}, cel.BoolType,
					cel.BinaryBinding(listHasOnly),
				),
			),
			cel.Function("hasAny",
				cel.MemberOverload("list_hasAny_list",
					[]*cel.Type{listType, listType}, cel.BoolType,
					cel.BinaryBinding(listHasAny),
				),
			),
			cel.Function("hasAll",
				cel.MemberOverload("list_hasAll_list",
					[]*cel.Type{listType, listType}, cel.BoolType,
					cel.BinaryBinding(listHasAll),
				),
			),
			cel.Function("hasExactly",
				cel.MemberOverload("list_hasExactly_list",
					[]*cel.Type{listType, listType}, cel.BoolType,
					cel.BinaryBinding(listHasExactly),
				),
			),
			cel.CostModel(
				listSetOpCostModel("list_hasOnly_list", 1),
				listSetOpCostModel("list_hasAny_list", 1),
				listSetOpCostModel("list_hasAll_list", 1),
				listSetOpCostModel("list_hasExactly_list", 2),
			),
		)
	}

	return opts
}

// ProgramOptions implements the Library interface method.
func (lib *listsLib) ProgramOptions() []cel.ProgramOption {
	return nil
}

func genRange(adapt types.Adapter, n types.Int, maxSize int64) (ref.Val, error) {
	if n < 0 {
		return nil, fmt.Errorf("lists.range: size must be non-negative, got %d", n)
	}
	if maxSize > 0 && int64(n) > maxSize {
		return nil, fmt.Errorf("lists.range: size %d exceeds maximum allowed (%d)", n, maxSize)
	}
	newList := make([]int, 0, n)
	for i := range int(n) {
		newList = append(newList, i)
	}
	return adapt.NativeToValue(newList), nil
}

func reverseList(adapt types.Adapter, list traits.Lister) (ref.Val, error) {
	if reversed, ok := types.MaybeReverseList(adapt, list); ok {
		return reversed, nil
	}
	listLength := list.Size().(types.Int)
	newList := make([]ref.Val, 0, listLength)
	for i := range listLength {
		val := list.Get(listLength - i - 1)
		newList = append(newList, val)
	}
	return adapt.NativeToValue(newList), nil
}

func slice(adapt types.Adapter, list traits.Lister, start, end types.Int) (ref.Val, error) {
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

	if sliced, ok := types.MaybeSliceList(adapt, list, int(start), int(end)); ok {
		return sliced, nil
	}

	newList := make([]ref.Val, 0, end-start)
	for i := types.Int(start); i < end; i++ {
		val := list.Get(i)
		newList = append(newList, val)
	}
	return adapt.NativeToValue(newList), nil
}

func flatten(list traits.Lister, depth int64) ([]ref.Val, error) {
	if depth < 0 {
		return nil, fmt.Errorf("level must be non-negative")
	}

	listLength := list.Size().(types.Int)
	newList := make([]ref.Val, 0, listLength)
	return flattenInto(newList, list, depth)
}

func flattenInto(dst []ref.Val, list traits.Lister, depth int64) ([]ref.Val, error) {
	listLength := list.Size().(types.Int)
	for i := range listLength {
		val := list.Get(i)
		if depth == 0 {
			dst = append(dst, val)
			continue
		}
		if nestedList, isList := val.(traits.Lister); isList {
			var err error
			dst, err = flattenInto(dst, nestedList, depth-1)
			if err != nil {
				return nil, err
			}
		} else {
			dst = append(dst, val)
		}
	}
	return dst, nil
}

func sortList(adapt types.Adapter, list traits.Lister) (ref.Val, error) {
	return sortListByAssociatedKeys(adapt, list, list)
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
func sortListByAssociatedKeys(adapt types.Adapter, list, keys traits.Lister) (ref.Val, error) {
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
	return adapt.NativeToValue(sorted), nil
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

func distinctList(adapt types.Adapter, list traits.Lister) (ref.Val, error) {
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

	return adapt.NativeToValue(uniqueList), nil
}

// listHasOnly returns true if every element of the target list is contained within the argument list.
func listHasOnly(list, other ref.Val) ref.Val {
	return setsContains(other, list)
}

// listHasAny returns true if the target list and the argument list share at least one element.
func listHasAny(list, other ref.Val) ref.Val {
	return setsIntersects(list, other)
}

// listHasAll returns true if every element of the argument list is contained within the target list.
func listHasAll(list, other ref.Val) ref.Val {
	return setsContains(list, other)
}

// listHasExactly returns true if the target list and the argument list are set equivalent.
func listHasExactly(list, other ref.Val) ref.Val {
	return setsEquivalent(list, other)
}

func templatedOverloads(types []*cel.Type, template func(t *cel.Type) cel.FunctionOpt) []cel.FunctionOpt {
	overloads := make([]cel.FunctionOpt, len(types))
	for i, t := range types {
		overloads[i] = template(t)
	}
	return overloads
}

// estimateListSlice computes an O(n) slice operation with a cost factor of 1.
func estimateListSlice(estimator cost.Estimator, target *cost.AstNode, args []cost.AstNode) *cost.CallEstimate {
	if target == nil || len(args) != 2 {
		return nil
	}
	sz := cost.EstimateSize(estimator, *target)
	start := cost.NodeAsUintValue(args[0], 0)
	end := cost.NodeAsUintValue(args[1], sz.Max)
	resSize := cost.FixedSizeEstimate(end - start)
	resSize.Elem = sz.Elem
	return cost.NewListCallEstimate(resSize.AsCost(), resSize)
}

func flattenCallArgs(target *cost.AstNode, args []cost.AstNode) (depth uint64, anyDepth, ok bool) {
	if target == nil || len(args) > 1 {
		return 0, false, false
	}
	if len(args) == 0 {
		return 1, false, true
	}
	if d, isLit := cost.UintValue(args[0]); isLit {
		return d, false, true
	}
	return math.MaxUint, true, true
}

// estimateListFlattenDepth estimates l.flatten(depth) under cost model version 1+.
func estimateListFlattenDepth(legacy bool) cost.FunctionEstimator {
	return func(estimator cost.Estimator, target *cost.AstNode, args []cost.AstNode) *cost.CallEstimate {
		depth, anyDepth, ok := flattenCallArgs(target, args)
		if !ok || len(args) != 1 {
			return nil
		}
		factor := 1.0
		if legacy {
			factor = float64(depth)
		}
		if est, ok := estimateFlattenByContainerSizing(estimator, *target, depth, factor, anyDepth); ok {
			return est
		}
		resSize := estimateFlattenedList(estimator, *target, depth, anyDepth)
		if !legacy {
			return cost.NewListCallEstimate(resSize.AsCost(), resSize)
		}
		sz := cost.EstimateSize(estimator, *target)
		work := sz.MultiplyByCostFactor(factor)
		if anyDepth {
			work.Min = 0
		}
		return cost.NewListCallEstimate(work, resSize)
	}
}

// estimateListFlatten estimates l.flatten() and l.flatten(depth) under cost model version 0.
func estimateListFlatten(legacy bool) cost.FunctionEstimator {
	return func(estimator cost.Estimator, target *cost.AstNode, args []cost.AstNode) *cost.CallEstimate {
		depth, anyDepth, ok := flattenCallArgs(target, args)
		if !ok {
			return nil
		}
		factor := 1.0
		if legacy {
			factor = float64(depth)
		}
		if est, ok := estimateFlattenByContainerSizing(estimator, *target, depth, factor, anyDepth); ok {
			return est
		}
		sz := cost.EstimateSize(estimator, *target)
		if legacy {
			return cost.NewListCallEstimate(sz.MultiplyByCostFactor(factor), sz)
		}
		var resSize cost.SizeEstimate
		if litSize, ok := literalFlattenLength((*target).Expr(), depth); ok {
			resSize = cost.FixedSizeEstimate(litSize.Max)
			resSize.Elem = flattenedElemSize(sz, depth)
		} else {
			resSize = estimateFlattenSize(sz, (*target).Type(), depth)
		}
		return cost.NewListCallEstimate(resSize.AsCost(), resSize)
	}
}

// estimateFlattenByContainerSizing estimates flatten when container sizes are not lengths.
//
// flatten(depth) cannot be described by an OverloadModel, whose quantities cannot express a
// variable number of levels, so its estimator follows the sizing strategy itself. The estimators
// above derive the flattened length from the lengths at each level, which is right only when sizes
// are lengths. Otherwise:
//
//   - When a container's size is its content (the aggregate strategy), flattening removes inner
//     list headers and keeps every other value, so the result is no larger than the receiver. The
//     receiver's length, and the flattened length the trackers charge, are each at most the
//     receiver's size. costFactor scales the charged length: 1 for the flattened list, or the
//     depth for the legacy tracker, which charges depth times the receiver's length.
//   - When the relationship is not known, neither the size nor the length is bounded.
//
// It reports false when sizes are lengths, leaving the estimate to the caller.
func estimateFlattenByContainerSizing(estimator cost.Estimator, target cost.AstNode, depth uint64, costFactor float64, anyDepth bool) (*cost.CallEstimate, bool) {
	switch cost.ContainerSizingOf(estimator) {
	case cost.SizedByContent:
		sz := cost.EstimateSize(estimator, target)
		length := cost.RangedSizeEstimate(0, sz.Max)
		resSize := cost.RangedSizeEstimate(min(1, sz.Max), sz.Max)
		root := flattenLevel{t: target.Type(), sz: &sz}
		resSize.Elem = flattenedElems(root, depth, anyDepth)
		return cost.NewListCallEstimate(length.MultiplyByCostFactor(costFactor), resSize), true
	case cost.SizedOpaque:
		return cost.NewListCallEstimate(cost.UnknownCostEstimate(), cost.UnknownSizeEstimate()), true
	default:
		return nil, false
	}
}

func estimateFlattenSize(sz cost.SizeEstimate, tType *types.Type, depth uint64) cost.SizeEstimate {
	res := cost.RangedSizeEstimate(sz.Min, sz.Max)
	for i, cur := uint64(0), &sz; i < depth && tType.Kind() == types.ListKind && len(tType.Parameters()) > 0; i++ {
		tType = tType.Parameters()[0]
		cur = cur.Elem
		if cur == nil {
			res = res.Multiply(cost.UnknownSizeEstimate())
			break
		}
		res = res.Multiply(*cur)
	}
	res.Key = nil
	res.Elem = flattenedElemSize(sz, depth)
	return res
}

func flattenedElemSize(sz cost.SizeEstimate, depth uint64) *cost.SizeEstimate {
	cur := &sz
	for i := uint64(0); i < depth && cur != nil; i++ {
		cur = cur.Elem
	}
	if cur == nil {
		return nil
	}
	return cur.Elem
}

// The estimates below apply from cost model version 1, when sizes are lengths. They follow
// flattenInto, which descends into every value that is a list, whatever its static type, and keeps
// every other value as a single entry. estimateFlattenByContainerSizing covers the other sizing
// strategies, using flattenedElems for the result's element size.

// maxFlattenLevels bounds how many levels of flattening the estimates follow: flatten(depth) is
// estimated for a depth of at most maxFlattenLevels, which reads the values up to one level
// further down. Flattening which may go deeper, by a larger depth or a non-literal one over a more
// deeply nested list, is estimated as unknown.
const maxFlattenLevels = 16

// flattenLevel describes the values at one level of a list being flattened: their static type and
// their size estimate, if known. Level 0 is the receiver.
type flattenLevel struct {
	t  *types.Type
	sz *cost.SizeEstimate
}

// next describes the values one level down: the elements of the lists at this level.
func (l flattenLevel) next() flattenLevel {
	elemType := types.DynType
	if l.t.Kind() == types.ListKind && len(l.t.Parameters()) > 0 {
		elemType = l.t.Parameters()[0]
	}
	var sz *cost.SizeEstimate
	if l.sz != nil {
		sz = l.sz.Elem
	}
	return flattenLevel{t: elemType, sz: sz}
}

// mayBeList reports whether a value at this level may be a list, which flattening expands.
func (l flattenLevel) mayBeList() bool {
	return isDynamicFlattenType(l.t) || l.t.Kind() == types.ListKind
}

// mayBeOther reports whether a value at this level may be something other than a list, which
// flattening keeps as a single entry.
func (l flattenLevel) mayBeOther() bool {
	return isDynamicFlattenType(l.t) || l.t.Kind() != types.ListKind
}

func isDynamicFlattenType(t *types.Type) bool {
	switch t.Kind() {
	case types.DynKind, types.AnyKind, types.TypeParamKind:
		return true
	}
	return false
}

// estimateFlattenedList bounds the list target.flatten(depth) produces when sizes are lengths.
// anyDepth reports a non-literal depth, which may take any value, so the bounds cover every depth.
func estimateFlattenedList(estimator cost.Estimator, target cost.AstNode, depth uint64, anyDepth bool) cost.SizeEstimate {
	sz := cost.EstimateSize(estimator, target)
	root := flattenLevel{t: target.Type(), sz: &sz}
	expr := target.Expr()
	var res cost.SizeEstimate
	switch {
	case anyDepth:
		if litRes, ok := literalFlattenLengthAnyDepth(expr); ok {
			res = litRes
		} else {
			res = flattenedLengthAnyDepth(root)
		}
	default:
		if litRes, ok := literalFlattenLengthWithin(expr, depth); ok {
			res = litRes
		} else {
			res = flattenedLength(root, depth, 0)
		}
	}
	res.Key = nil
	res.Elem = flattenedElems(root, depth, anyDepth)
	return res
}

// flattenedLength bounds the number of entries flattening the list at level l yields, descending
// into depth more levels. A list contributes its flattened length, and any other value 1.
func flattenedLength(l flattenLevel, depth uint64, levels int) cost.SizeEstimate {
	if l.sz == nil || levels > maxFlattenLevels {
		return cost.UnknownSizeEstimate()
	}
	if depth == 0 {
		return cost.RangedSizeEstimate(l.sz.Min, l.sz.Max)
	}
	elems := l.next()
	var per *cost.SizeEstimate
	if elems.mayBeOther() {
		one := cost.FixedSizeEstimate(1)
		per = &one
	}
	if elems.mayBeList() {
		inner := flattenedLength(elems, depth-1, levels+1)
		per = unionBounds(per, inner)
	}
	return cost.RangedSizeEstimate(
		cost.SafeMultiply(l.sz.Min, per.Min),
		cost.SafeMultiply(l.sz.Max, per.Max))
}

// flattenedLengthAnyDepth bounds the flattened length over every depth. Flattening below the last
// level which may hold a list changes nothing, so the depths up to that level suffice.
func flattenedLengthAnyDepth(root flattenLevel) cost.SizeEstimate {
	res := flattenedLength(root, 0, 0)
	level := root.next()
	for d := uint64(1); level.mayBeList(); d++ {
		if d > maxFlattenLevels {
			return cost.UnknownSizeEstimate()
		}
		res = *unionBounds(&res, flattenedLength(root, d, 0))
		if res.IsUnknown() {
			return res
		}
		level = level.next()
	}
	return res
}

// flattenedElems bounds the elements of the flattened list: the values at levels 1 to depth which
// are not lists, and every value at level depth+1. When anyDepth is set, a value at any level may
// be kept. A level whose values may be kept, but whose size is not known, makes the elements
// unknown.
func flattenedElems(root flattenLevel, depth uint64, anyDepth bool) *cost.SizeEstimate {
	var res *cost.SizeEstimate
	level := root
	for k := uint64(1); ; k++ {
		// flatten(maxFlattenLevels) keeps the values one level below its deepest expansion.
		if k > maxFlattenLevels+1 {
			return unknownSize()
		}
		level = level.next()
		last := !anyDepth && k == depth+1
		if anyDepth || last || level.mayBeOther() {
			if level.sz == nil {
				return unknownSize()
			}
			res = unionBounds(res, *level.sz)
		}
		if last || !level.mayBeList() {
			return res
		}
	}
}

// literalFlattenLength bounds the flattened length of a list literal when all expanded elements up
// to depth are literals or list literals. An optional element may be absent, so it counts toward
// the upper bound only.
func literalFlattenLength(expr ast.Expr, depth uint64) (cost.SizeEstimate, bool) {
	if expr == nil {
		return cost.SizeEstimate{}, false
	}
	if expr.Kind() != ast.ListKind {
		if expr.Kind() == ast.LiteralKind {
			return cost.FixedSizeEstimate(1), true
		}
		return cost.SizeEstimate{}, false
	}
	list := expr.AsList()
	optional := make(map[int32]bool, len(list.OptionalIndices()))
	for _, i := range list.OptionalIndices() {
		optional[i] = true
	}
	var lo, hi uint64
	for i, el := range list.Elements() {
		elSize := cost.FixedSizeEstimate(1)
		if depth > 0 {
			var ok bool
			if elSize, ok = literalFlattenLength(el, depth-1); !ok {
				return cost.SizeEstimate{}, false
			}
		}
		if optional[int32(i)] {
			elSize.Min = 0
		}
		lo, hi = cost.SafeAdd(lo, elSize.Min), cost.SafeAdd(hi, elSize.Max)
	}
	return cost.RangedSizeEstimate(lo, hi), true
}

// listLiteralLevels reports how many levels of list literals expr nests: 0 when expr is not a list
// literal. Flattening a literal of n levels by more than n-1 changes nothing.
func listLiteralLevels(expr ast.Expr) uint64 {
	if expr == nil || expr.Kind() != ast.ListKind {
		return 0
	}
	var deepest uint64
	for _, el := range expr.AsList().Elements() {
		deepest = max(deepest, listLiteralLevels(el))
	}
	return deepest + 1
}

// literalFlattenLengthWithin is literalFlattenLength for flattening which expands at most
// maxFlattenLevels levels of the literal.
func literalFlattenLengthWithin(expr ast.Expr, depth uint64) (cost.SizeEstimate, bool) {
	if levels := listLiteralLevels(expr); levels == 0 || min(depth, levels-1) > maxFlattenLevels {
		return cost.SizeEstimate{}, false
	}
	return literalFlattenLength(expr, depth)
}

// literalFlattenLengthAnyDepth bounds the flattened length of a fully literal list over every
// depth: the depths which expand a level of the literal, of which there may be at most
// maxFlattenLevels.
func literalFlattenLengthAnyDepth(expr ast.Expr) (cost.SizeEstimate, bool) {
	levels := listLiteralLevels(expr)
	if levels == 0 || levels-1 > maxFlattenLevels {
		return cost.SizeEstimate{}, false
	}
	var res *cost.SizeEstimate
	for d := uint64(0); d < levels; d++ {
		sz, ok := literalFlattenLength(expr, d)
		if !ok {
			return cost.SizeEstimate{}, false
		}
		res = unionBounds(res, sz)
	}
	return *res, true
}

// unionBounds returns an estimate covering both a, which may be nil, and b, along with the
// union of their metadata.
func unionBounds(a *cost.SizeEstimate, b cost.SizeEstimate) *cost.SizeEstimate {
	if a == nil {
		return &b
	}
	u := a.Union(b)
	return &u
}

func unknownSize() *cost.SizeEstimate {
	u := cost.UnknownSizeEstimate()
	return &u
}

// estimateListDistinct estimates l.distinct() under ModelVersion0.
func estimateListDistinct(legacy bool) cost.FunctionEstimator {
	return func(estimator cost.Estimator, target *cost.AstNode, args []cost.AstNode) *cost.CallEstimate {
		if target == nil || len(args) != 0 {
			return nil
		}
		elemType := types.DynType
		if t := (*target).Type(); t.Kind() == types.ListKind && len(t.Parameters()) > 0 {
			elemType = t.Parameters()[0]
		}
		return estimateSelfCompareCost(estimator, *target, *target, elemType, legacy, !legacy)
	}
}

// estimateListSort estimates l.sort() under ModelVersion0.
func estimateListSort(t *types.Type, legacy bool) cost.FunctionEstimator {
	return func(estimator cost.Estimator, target *cost.AstNode, args []cost.AstNode) *cost.CallEstimate {
		if target == nil || len(args) != 0 {
			return nil
		}
		return estimateSelfCompareCost(estimator, *target, *target, t, legacy, false)
	}
}

// estimateListSortBy estimates l.sortBy() under ModelVersion0.
func estimateListSortBy(u *types.Type, legacy bool) cost.FunctionEstimator {
	return func(estimator cost.Estimator, target *cost.AstNode, args []cost.AstNode) *cost.CallEstimate {
		if target == nil || len(args) != 1 {
			return nil
		}
		return estimateSelfCompareCost(estimator, args[0], *target, u, legacy, false)
	}
}

func estimateSelfCompareCost(estimator cost.Estimator, keysNode, targetNode cost.AstNode, elemType *types.Type, legacy, clampMin bool) *cost.CallEstimate {
	sz := cost.EstimateSize(estimator, keysNode)
	targetSize := cost.EstimateSize(estimator, targetNode)
	costSize := sz.Multiply(sz)
	if legacy {
		costSize.Elem = targetSize.Elem
		return cost.NewListCallEstimate(costSize.MultiplyByCostFactor(selfCompareFactor(elemType)), costSize)
	}
	elemCost := estimateElementEqualityCost(elemType, estimateItemSize(estimator, targetNode))
	c := costSize.MultiplyByCost(elemCost).MultiplyByCostFactor(2.0)
	resSize := cost.RangedSizeEstimate(sz.Min, sz.Max)
	if clampMin && sz.Min > 0 {
		resSize.Min = 1
	}
	resSize.Elem = targetSize.Elem
	return cost.NewListCallEstimate(c, resSize)
}

func estimateItemSize(estimator cost.Estimator, node cost.AstNode) cost.SizeEstimate {
	if len(node.Path()) == 0 {
		return cost.UnknownSizeEstimate()
	}
	elemType := types.DynType
	if t := node.Type(); t.Kind() == types.ListKind && len(t.Parameters()) > 0 {
		elemType = t.Parameters()[0]
	}
	if l := estimator.EstimateSize(cost.NewAstNode(nil, append(slices.Clone(node.Path()), "@items"), elemType, nil)); l != nil {
		return *l
	}
	return cost.UnknownSizeEstimate()
}

func estimateElementEqualityCost(elemType *types.Type, itemSize cost.SizeEstimate) cost.CostEstimate {
	switch elemType.Kind() {
	case types.StringKind, types.BytesKind:
		return itemSize.MultiplyByCostFactor(cost.StringTraversalCostFactor)
	case types.ListKind, types.MapKind, types.StructKind:
		return cost.UnknownCostEstimate()
	default:
		return cost.FixedCostEstimate(1)
	}
}

// trackListSelfCompare computes the ModelVersion0 O(n^2) runtime charge over args[argIdx].
func trackListSelfCompare(argIdx int) cost.FunctionTracker {
	return func(args []ref.Val, _ ref.Val) *uint64 {
		l := args[argIdx].(traits.Lister)
		sz := cost.ActualSize(l)
		costFactor := 2.0
		if sz > 0 {
			if elem := l.Get(types.IntZero); elem.Type() == types.StringType || elem.Type() == types.BytesType {
				costFactor += cost.StringTraversalCostFactor
			}
		}
		// The scaled size is truncated, matching the historical cost for in-range values, but
		// saturates rather than relying on a platform-defined out-of-range float conversion.
		total := cost.SafeAdd(cost.SafeMultiplyByFactorTrunc(cost.SafeMultiply(sz, sz), costFactor), cost.CallCost, cost.ListCreateBaseCost)
		return &total
	}
}

// listSliceCostModel models l.slice(start, end): it copies end - start elements of the receiver,
// and its size is also at most the receiver's.
func listSliceCostModel(id string) cost.OverloadModel {
	sliceList := cost.List(cost.Sub(cost.IntArg(1, 0), cost.IntArg(0, 0)), cost.TargetElem())
	return cost.MemberOverload(id,
		cost.EvalCost(cost.AllocList(cost.Length(cost.Result()))),
		cost.ResultSize(cost.Ranged(sliceList, cost.Min(sliceList, cost.Target()))),
		cost.BeforeVersion(1, cost.ManualEstimator(estimateListSlice)),
	)
}

// listsRangeCostModel models lists.range(n): it builds n ints, each of size 1 under both built-in
// strategies.
func listsRangeCostModel(id string) cost.OverloadModel {
	return cost.Overload(id,
		cost.EvalCost(cost.AllocList(cost.Length(cost.Result()))),
		cost.ResultSize(cost.List(cost.IntArg(0, math.MaxUint), cost.Const(1))),
	)
}

func listReverseCostModel(id string) cost.OverloadModel {
	return cost.MemberOverload(id,
		cost.EvalCost(cost.AllocList(cost.Length(cost.Result()))),
		cost.ResultSize(cost.Target()),
	)
}

// listDistinctCostModel models l.distinct(): it compares every pair of elements. Its tracker
// raises the factor for string and bytes elements, which a model of the single list(dyn) overload
// cannot see, so the model charges the larger factor. The result holds some of the receiver's
// elements, and at least one when the receiver has any.
func listDistinctCostModel(id string, legacy bool) cost.OverloadModel {
	return cost.MemberOverload(id,
		cost.EvalCost(selfCompareCost(cost.Length(cost.Target()), 2+cost.StringTraversalCostFactor)),
		cost.ResultSize(cost.Ranged(cost.Min(cost.Target(), cost.Const(1)), cost.Target())),
		cost.BeforeVersion(1, cost.ManualCost(estimateListDistinct(legacy), trackListSelfCompare(0))),
	)
}

// flattenedSize bounds the size of l.flatten(), sound under both built-in strategies. Its upper
// bound is the smaller of two bounds:
//
//   - Under the default strategy each element contributes at most max(its size, 1) to the
//     flattened length: a list its length, any other value 1. The first bound is the receiver's
//     length times that.
//   - Under the aggregate strategy the flattened list holds the receiver's contents less the inner
//     list headers, so its size is at most the receiver's size plus its content size, the second
//     bound. The second bound also bounds the default strategy's length, as the receiver's length
//     plus the summed element sizes.
//
// Its lower bound is 0: no bound in terms of the receiver's size and element size holds under
// both strategies, since the inner lists may be empty.
func flattenedSize() cost.QuantityExpr {
	return cost.AtMost(cost.Min(
		cost.List(
			cost.Mul(cost.Target(), cost.AtLeastOneQuantity(cost.TargetElem())),
			cost.ElemOf(cost.TargetElem()),
		),
		cost.Sum(cost.Target(), cost.ElemTotal(cost.Target())),
	))
}

func listFlattenCostModel(id string, legacy bool) cost.OverloadModel {
	work := cost.Length(cost.Result())
	if legacy {
		work = cost.Length(cost.Target())
	}
	return cost.MemberOverload(id,
		cost.EvalCost(cost.AllocList(work)),
		cost.ResultSize(flattenedSize()),
		cost.BeforeVersion(1, cost.ManualEstimator(estimateListFlatten(legacy))),
	)
}

func listFlattenDepthCostModel(id string, legacy bool) cost.OverloadModel {
	work := cost.Length(cost.Result())
	if legacy {
		work = cost.Mul(cost.Length(cost.Target()), cost.IntArg(0, 1))
	}
	return cost.MemberOverload(id,
		cost.EvalCost(cost.AllocList(work)),
		cost.ManualEstimator(estimateListFlattenDepth(legacy)),
		cost.BeforeVersion(1, cost.ManualEstimator(estimateListFlatten(legacy))),
	)
}

// selfCompareCost is the cost of comparing each element of a list of the given length against
// every other, the charge of the sort and distinct trackers.
func selfCompareCost(length cost.QuantityExpr, factor float64) cost.QuantityExpr {
	return cost.AllocList(cost.Scale(cost.Mul(length, length), factor))
}

// selfCompareFactor is the comparison factor of an element type: string and bytes comparisons
// also traverse the values.
func selfCompareFactor(t *types.Type) float64 {
	switch t.Kind() {
	case types.StringKind, types.BytesKind:
		return 2 + cost.StringTraversalCostFactor
	default:
		return 2
	}
}

func listSortCostModels(legacy bool) []cost.OverloadModel {
	models := make([]cost.OverloadModel, 0, 2*len(comparableTypes))
	for _, t := range comparableTypes {
		f := selfCompareFactor(t)
		models = append(models,
			cost.MemberOverload(fmt.Sprintf("list_%s_sort", t.TypeName()),
				cost.EvalCost(selfCompareCost(cost.Length(cost.Target()), f)),
				cost.ResultSize(cost.Target()),
				cost.BeforeVersion(1, cost.ManualCost(estimateListSort(t, legacy), trackListSelfCompare(0))),
			),
			// sortBy compares the keys, and reorders the receiver.
			cost.MemberOverload(fmt.Sprintf("list_%s_sortByAssociatedKeys", t.TypeName()),
				cost.EvalCost(selfCompareCost(cost.Length(cost.Arg(0)), f)),
				cost.ResultSize(cost.Target()),
				cost.BeforeVersion(1, cost.ManualCost(estimateListSortBy(t, legacy), trackListSelfCompare(1))),
			),
		)
	}
	return models
}

func listSetOpCostModel(id string, costFactor float64) cost.OverloadModel {
	return cost.MemberOverload(id,
		cost.EvalCost(cost.Sum(cost.Const(cost.CallCost),
			cost.Scale(cost.Mul(cost.Length(cost.Target()), cost.Length(cost.Arg(0))), costFactor))),
	)
}
