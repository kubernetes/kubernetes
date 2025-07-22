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

package ext

import (
	"fmt"
	"math"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/parser"
)

const (
	mapInsert                 = "cel.@mapInsert"
	mapInsertOverloadMap      = "@mapInsert_map_map"
	mapInsertOverloadKeyValue = "@mapInsert_map_key_value"
)

// TwoVarComprehensions introduces support for two-variable comprehensions.
//
// The two-variable form of comprehensions looks similar to the one-variable counterparts.
// Where possible, the same macro names were used and additional macro signatures added.
// The notable distinction for two-variable comprehensions is the introduction of
// `transformList`, `transformMap`, and `transformMapEntry` support for list and map types
// rather than the more traditional `map` and `filter` macros.
//
// # All
//
// Comprehension which tests whether all elements in the list or map satisfy a given
// predicate. The `all` macro evaluates in a manner consistent with logical AND and will
// short-circuit when encountering a `false` value.
//
//	<list>.all(indexVar, valueVar, <predicate>) -> bool
//	<map>.all(keyVar, valueVar, <predicate>) -> bool
//
// Examples:
//
//	[1, 2, 3].all(i, j, i < j) // returns true
//	{'hello': 'world', 'taco': 'taco'}.all(k, v, k != v) // returns false
//
//	// Combines two-variable comprehension with single variable
//	{'h': ['hello', 'hi'], 'j': ['joke', 'jog']}
//	    .all(k, vals, vals.all(v, v.startsWith(k))) // returns true
//
// # Exists
//
// Comprehension which tests whether any element in a list or map exists which satisfies
// a given predicate. The `exists` macro evaluates in a manner consistent with logical OR
// and will short-circuit when encountering a `true` value.
//
//	<list>.exists(indexVar, valueVar, <predicate>) -> bool
//	<map>.exists(keyVar, valueVar, <predicate>) -> bool
//
// Examples:
//
//	{'greeting': 'hello', 'farewell': 'goodbye'}
//	    .exists(k, v, k.startsWith('good') || v.endsWith('bye')) // returns true
//	[1, 2, 4, 8, 16].exists(i, v, v == 1024 && i == 10) // returns false
//
// # ExistsOne
//
// Comprehension which tests whether exactly one element in a list or map exists which
// satisfies a given predicate expression. This comprehension does not short-circuit in
// keeping with the one-variable exists one macro semantics.
//
//	<list>.existsOne(indexVar, valueVar, <predicate>)
//	<map>.existsOne(keyVar, valueVar, <predicate>)
//
// This macro may also be used with the `exists_one` function name, for compatibility
// with the one-variable macro of the same name.
//
// Examples:
//
//	[1, 2, 1, 3, 1, 4].existsOne(i, v, i == 1 || v == 1) // returns false
//	[1, 1, 2, 2, 3, 3].existsOne(i, v, i == 2 && v == 2) // returns true
//	{'i': 0, 'j': 1, 'k': 2}.existsOne(i, v, i == 'l' || v == 1) // returns true
//
// # TransformList
//
// Comprehension which converts a map or a list into a list value. The output expression
// of the comprehension determines the contents of the output list. Elements in the list
// may optionally be filtered according to a predicate expression, where elements that
// satisfy the predicate are transformed.
//
//	<list>.transformList(indexVar, valueVar, <transform>)
//	<list>.transformList(indexVar, valueVar, <filter>, <transform>)
//	<map>.transformList(keyVar, valueVar, <transform>)
//	<map>.transformList(keyVar, valueVar, <filter>, <transform>)
//
// Examples:
//
//	[1, 2, 3].transformList(indexVar, valueVar,
//	  (indexVar * valueVar) + valueVar) // returns [1, 4, 9]
//	[1, 2, 3].transformList(indexVar, valueVar, indexVar % 2 == 0
//	  (indexVar * valueVar) + valueVar) // returns [1, 9]
//	{'greeting': 'hello', 'farewell': 'goodbye'}
//	  .transformList(k, _, k) // returns ['greeting', 'farewell']
//	{'greeting': 'hello', 'farewell': 'goodbye'}
//	  .transformList(_, v, v) // returns ['hello', 'goodbye']
//
// # TransformMap
//
// Comprehension which converts a map or a list into a map value. The output expression
// of the comprehension determines the value of the output map entry; however, the key
// remains fixed. Elements in the map may optionally be filtered according to a predicate
// expression, where elements that satisfy the predicate are transformed.
//
//	<list>.transformMap(indexVar, valueVar, <transform>)
//	<list>.transformMap(indexVar, valueVar, <filter>, <transform>)
//	<map>.transformMap(keyVar, valueVar, <transform>)
//	<map>.transformMap(keyVar, valueVar, <filter>, <transform>)
//
// Examples:
//
//	[1, 2, 3].transformMap(indexVar, valueVar,
//	  (indexVar * valueVar) + valueVar) // returns {0: 1, 1: 4, 2: 9}
//	[1, 2, 3].transformMap(indexVar, valueVar, indexVar % 2 == 0
//	  (indexVar * valueVar) + valueVar) // returns {0: 1, 2: 9}
//	{'greeting': 'hello'}.transformMap(k, v, v + '!') // returns {'greeting': 'hello!'}
//
// # TransformMapEntry
//
// Comprehension which converts a map or a list into a map value; however, this transform
// expects the entry expression be a map literal. If the tranform produces an entry which
// duplicates a key in the target map, the comprehension will error.  Note, that key
// equality is determined using CEL equality which asserts that numeric values which are
// equal, even if they don't have the same type will cause a key collision.
//
// Elements in the map may optionally be filtered according to a predicate expression, where
// elements that satisfy the predicate are transformed.
//
//	<list>.transformMap(indexVar, valueVar, <transform>)
//	<list>.transformMap(indexVar, valueVar, <filter>, <transform>)
//	<map>.transformMap(keyVar, valueVar, <transform>)
//	<map>.transformMap(keyVar, valueVar, <filter>, <transform>)
//
// Examples:
//
//	// returns {'hello': 'greeting'}
//	{'greeting': 'hello'}.transformMapEntry(keyVar, valueVar, {valueVar: keyVar})
//	// reverse lookup, require all values in list be unique
//	[1, 2, 3].transformMapEntry(indexVar, valueVar, {valueVar: indexVar})
//
//	{'greeting': 'aloha', 'farewell': 'aloha'}
//	  .transformMapEntry(keyVar, valueVar, {valueVar: keyVar}) // error, duplicate key
func TwoVarComprehensions(options ...TwoVarComprehensionsOption) cel.EnvOption {
	l := &compreV2Lib{version: math.MaxUint32}
	for _, o := range options {
		l = o(l)
	}
	return cel.Lib(l)
}

// TwoVarComprehensionsOption declares a functional operator for configuring two-variable comprehensions.
type TwoVarComprehensionsOption func(*compreV2Lib) *compreV2Lib

// TwoVarComprehensionsVersion sets the library version for two-variable comprehensions.
func TwoVarComprehensionsVersion(version uint32) TwoVarComprehensionsOption {
	return func(lib *compreV2Lib) *compreV2Lib {
		lib.version = version
		return lib
	}
}

type compreV2Lib struct {
	version uint32
}

// LibraryName implements that SingletonLibrary interface method.
func (*compreV2Lib) LibraryName() string {
	return "cel.lib.ext.comprev2"
}

// CompileOptions implements the cel.Library interface method.
func (*compreV2Lib) CompileOptions() []cel.EnvOption {
	kType := cel.TypeParamType("K")
	vType := cel.TypeParamType("V")
	mapKVType := cel.MapType(kType, vType)
	opts := []cel.EnvOption{
		cel.Macros(
			cel.ReceiverMacro("all", 3, quantifierAll),
			cel.ReceiverMacro("exists", 3, quantifierExists),
			cel.ReceiverMacro("existsOne", 3, quantifierExistsOne),
			cel.ReceiverMacro("exists_one", 3, quantifierExistsOne),
			cel.ReceiverMacro("transformList", 3, transformList),
			cel.ReceiverMacro("transformList", 4, transformList),
			cel.ReceiverMacro("transformMap", 3, transformMap),
			cel.ReceiverMacro("transformMap", 4, transformMap),
			cel.ReceiverMacro("transformMapEntry", 3, transformMapEntry),
			cel.ReceiverMacro("transformMapEntry", 4, transformMapEntry),
		),
		cel.Function(mapInsert,
			cel.Overload(mapInsertOverloadKeyValue, []*cel.Type{mapKVType, kType, vType}, mapKVType,
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					m := args[0].(traits.Mapper)
					k := args[1]
					v := args[2]
					return types.InsertMapKeyValue(m, k, v)
				})),
			cel.Overload(mapInsertOverloadMap, []*cel.Type{mapKVType, mapKVType}, mapKVType,
				cel.BinaryBinding(func(targetMap, updateMap ref.Val) ref.Val {
					tm := targetMap.(traits.Mapper)
					um := updateMap.(traits.Mapper)
					umIt := um.Iterator()
					for umIt.HasNext() == types.True {
						k := umIt.Next()
						updateOrErr := types.InsertMapKeyValue(tm, k, um.Get(k))
						if types.IsError(updateOrErr) {
							return updateOrErr
						}
						tm = updateOrErr.(traits.Mapper)
					}
					return tm
				})),
		),
	}
	return opts
}

// ProgramOptions implements the cel.Library interface method
func (*compreV2Lib) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

func quantifierAll(mef cel.MacroExprFactory, target ast.Expr, args []ast.Expr) (ast.Expr, *cel.Error) {
	iterVar1, iterVar2, err := extractIterVars(mef, args[0], args[1])
	if err != nil {
		return nil, err
	}

	return mef.NewComprehensionTwoVar(
		target,
		iterVar1,
		iterVar2,
		mef.AccuIdentName(),
		/*accuInit=*/ mef.NewLiteral(types.True),
		/*condition=*/ mef.NewCall(operators.NotStrictlyFalse, mef.NewAccuIdent()),
		/*step=*/ mef.NewCall(operators.LogicalAnd, mef.NewAccuIdent(), args[2]),
		/*result=*/ mef.NewAccuIdent(),
	), nil
}

func quantifierExists(mef cel.MacroExprFactory, target ast.Expr, args []ast.Expr) (ast.Expr, *cel.Error) {
	iterVar1, iterVar2, err := extractIterVars(mef, args[0], args[1])
	if err != nil {
		return nil, err
	}

	return mef.NewComprehensionTwoVar(
		target,
		iterVar1,
		iterVar2,
		mef.AccuIdentName(),
		/*accuInit=*/ mef.NewLiteral(types.False),
		/*condition=*/ mef.NewCall(operators.NotStrictlyFalse, mef.NewCall(operators.LogicalNot, mef.NewAccuIdent())),
		/*step=*/ mef.NewCall(operators.LogicalOr, mef.NewAccuIdent(), args[2]),
		/*result=*/ mef.NewAccuIdent(),
	), nil
}

func quantifierExistsOne(mef cel.MacroExprFactory, target ast.Expr, args []ast.Expr) (ast.Expr, *cel.Error) {
	iterVar1, iterVar2, err := extractIterVars(mef, args[0], args[1])
	if err != nil {
		return nil, err
	}

	return mef.NewComprehensionTwoVar(
		target,
		iterVar1,
		iterVar2,
		mef.AccuIdentName(),
		/*accuInit=*/ mef.NewLiteral(types.Int(0)),
		/*condition=*/ mef.NewLiteral(types.True),
		/*step=*/ mef.NewCall(operators.Conditional, args[2],
			mef.NewCall(operators.Add, mef.NewAccuIdent(), mef.NewLiteral(types.Int(1))),
			mef.NewAccuIdent()),
		/*result=*/ mef.NewCall(operators.Equals, mef.NewAccuIdent(), mef.NewLiteral(types.Int(1))),
	), nil
}

func transformList(mef cel.MacroExprFactory, target ast.Expr, args []ast.Expr) (ast.Expr, *cel.Error) {
	iterVar1, iterVar2, err := extractIterVars(mef, args[0], args[1])
	if err != nil {
		return nil, err
	}

	var transform ast.Expr
	var filter ast.Expr
	if len(args) == 4 {
		filter = args[2]
		transform = args[3]
	} else {
		filter = nil
		transform = args[2]
	}

	//  accumulator = accumulator + [transform]
	step := mef.NewCall(operators.Add, mef.NewAccuIdent(), mef.NewList(transform))
	if filter != nil {
		//  accumulator = (filter) ? accumulator + [transform] : accumulator
		step = mef.NewCall(operators.Conditional, filter, step, mef.NewAccuIdent())
	}

	return mef.NewComprehensionTwoVar(
		target,
		iterVar1,
		iterVar2,
		mef.AccuIdentName(),
		/*accuInit=*/ mef.NewList(),
		/*condition=*/ mef.NewLiteral(types.True),
		step,
		/*result=*/ mef.NewAccuIdent(),
	), nil
}

func transformMap(mef cel.MacroExprFactory, target ast.Expr, args []ast.Expr) (ast.Expr, *cel.Error) {
	iterVar1, iterVar2, err := extractIterVars(mef, args[0], args[1])
	if err != nil {
		return nil, err
	}

	var transform ast.Expr
	var filter ast.Expr
	if len(args) == 4 {
		filter = args[2]
		transform = args[3]
	} else {
		filter = nil
		transform = args[2]
	}

	// accumulator = cel.@mapInsert(accumulator, iterVar1, transform)
	step := mef.NewCall(mapInsert, mef.NewAccuIdent(), mef.NewIdent(iterVar1), transform)
	if filter != nil {
		// accumulator = (filter) ? cel.@mapInsert(accumulator, iterVar1, transform) : accumulator
		step = mef.NewCall(operators.Conditional, filter, step, mef.NewAccuIdent())
	}
	return mef.NewComprehensionTwoVar(
		target,
		iterVar1,
		iterVar2,
		mef.AccuIdentName(),
		/*accuInit=*/ mef.NewMap(),
		/*condition=*/ mef.NewLiteral(types.True),
		step,
		/*result=*/ mef.NewAccuIdent(),
	), nil
}

func transformMapEntry(mef cel.MacroExprFactory, target ast.Expr, args []ast.Expr) (ast.Expr, *cel.Error) {
	iterVar1, iterVar2, err := extractIterVars(mef, args[0], args[1])
	if err != nil {
		return nil, err
	}

	var transform ast.Expr
	var filter ast.Expr
	if len(args) == 4 {
		filter = args[2]
		transform = args[3]
	} else {
		filter = nil
		transform = args[2]
	}

	// accumulator = cel.@mapInsert(accumulator, transform)
	step := mef.NewCall(mapInsert, mef.NewAccuIdent(), transform)
	if filter != nil {
		// accumulator = (filter) ? cel.@mapInsert(accumulator, transform) : accumulator
		step = mef.NewCall(operators.Conditional, filter, step, mef.NewAccuIdent())
	}
	return mef.NewComprehensionTwoVar(
		target,
		iterVar1,
		iterVar2,
		mef.AccuIdentName(),
		/*accuInit=*/ mef.NewMap(),
		/*condition=*/ mef.NewLiteral(types.True),
		step,
		/*result=*/ mef.NewAccuIdent(),
	), nil
}

func extractIterVars(mef cel.MacroExprFactory, arg0, arg1 ast.Expr) (string, string, *cel.Error) {
	iterVar1, err := extractIterVar(mef, arg0)
	if err != nil {
		return "", "", err
	}
	iterVar2, err := extractIterVar(mef, arg1)
	if err != nil {
		return "", "", err
	}
	if iterVar1 == iterVar2 {
		return "", "", mef.NewError(arg1.ID(), fmt.Sprintf("duplicate variable name: %s", iterVar1))
	}
	if iterVar1 == mef.AccuIdentName() || iterVar1 == parser.AccumulatorName {
		return "", "", mef.NewError(arg0.ID(), "iteration variable overwrites accumulator variable")
	}
	if iterVar2 == mef.AccuIdentName() || iterVar2 == parser.AccumulatorName {
		return "", "", mef.NewError(arg1.ID(), "iteration variable overwrites accumulator variable")
	}
	return iterVar1, iterVar2, nil
}

func extractIterVar(mef cel.MacroExprFactory, target ast.Expr) (string, *cel.Error) {
	iterVar, found := extractIdent(target)
	if !found {
		return "", mef.NewError(target.ID(), "argument must be a simple name")
	}
	return iterVar, nil
}
