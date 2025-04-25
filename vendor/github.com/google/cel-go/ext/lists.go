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
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/decls"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
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
// Introduced in version: 2
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
// Introduced in version: 2
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
// Introduced in version: 2
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
// Flattens a list recursively.
// If an optional depth is provided, the list is flattened to a the specificied level.
// A negative depth value will result in an error.
//
//	<list>.flatten(<list>) -> <list>
//	<list>.flatten(<list>, <int>) -> <list>
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
// Introduced in version: 2
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
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					n := args[0].(types.Int)
					result, err := genRange(n)
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
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					list := args[0].(traits.Lister)
					result, err := reverseList(list)
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

	return opts
}

// ProgramOptions implements the Library interface method.
func (listsLib) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
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
		if keys.Get(i).Type() != elem.Type() {
			return nil, fmt.Errorf("list elements must have the same type")
		}
		sortedIndices = append(sortedIndices, i)
	}

	sort.Slice(sortedIndices, func(i, j int) bool {
		iKey := keys.Get(sortedIndices[i])
		jKey := keys.Get(sortedIndices[j])
		return iKey.(traits.Comparer).Compare(jKey) == types.IntNegOne
	})

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
