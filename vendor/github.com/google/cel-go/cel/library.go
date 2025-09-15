// Copyright 2020 Google LLC
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

package cel

import (
	"fmt"
	"math"

	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/decls"
	"github.com/google/cel-go/common/env"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/stdlib"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/interpreter"
	"github.com/google/cel-go/parser"
)

const (
	optMapMacro                = "optMap"
	optFlatMapMacro            = "optFlatMap"
	hasValueFunc               = "hasValue"
	unwrapOptFunc              = "unwrapOpt"
	optionalNoneFunc           = "optional.none"
	optionalOfFunc             = "optional.of"
	optionalOfNonZeroValueFunc = "optional.ofNonZeroValue"
	optionalUnwrapFunc         = "optional.unwrap"
	valueFunc                  = "value"
	unusedIterVar              = "#unused"
)

// Library provides a collection of EnvOption and ProgramOption values used to configure a CEL
// environment for a particular use case or with a related set of functionality.
//
// Note, the ProgramOption values provided by a library are expected to be static and not vary
// between calls to Env.Program(). If there is a need for such dynamic configuration, prefer to
// configure these options outside the Library and within the Env.Program() call directly.
type Library interface {
	// CompileOptions returns a collection of functional options for configuring the Parse / Check
	// environment.
	CompileOptions() []EnvOption

	// ProgramOptions returns a collection of functional options which should be included in every
	// Program generated from the Env.Program() call.
	ProgramOptions() []ProgramOption
}

// SingletonLibrary refines the Library interface to ensure that libraries in this format are only
// configured once within the environment.
type SingletonLibrary interface {
	Library

	// LibraryName provides a namespaced name which is used to check whether the library has already
	// been configured in the environment.
	LibraryName() string
}

// LibraryAliaser generates a simple named alias for the library, for use during environment serialization.
type LibraryAliaser interface {
	LibraryAlias() string
}

// LibrarySubsetter provides the subset description associated with the library, nil if not subset.
type LibrarySubsetter interface {
	LibrarySubset() *env.LibrarySubset
}

// LibraryVersioner provides a version number for the library.
//
// If not implemented, the library version will be flagged as 'latest' during environment serialization.
type LibraryVersioner interface {
	LibraryVersion() uint32
}

// Lib creates an EnvOption out of a Library, allowing libraries to be provided as functional args,
// and to be linked to each other.
func Lib(l Library) EnvOption {
	singleton, isSingleton := l.(SingletonLibrary)
	return func(e *Env) (*Env, error) {
		if isSingleton {
			if e.HasLibrary(singleton.LibraryName()) {
				return e, nil
			}
			e.libraries[singleton.LibraryName()] = singleton
		}
		var err error
		for _, opt := range l.CompileOptions() {
			e, err = opt(e)
			if err != nil {
				return nil, err
			}
		}
		e.progOpts = append(e.progOpts, l.ProgramOptions()...)
		return e, nil
	}
}

// StdLibOption specifies a functional option for configuring the standard CEL library.
type StdLibOption func(*stdLibrary) *stdLibrary

// StdLibSubset configures the standard library to use a subset of its functions and macros.
//
// Since the StdLib is a singleton library, only the first instance of the StdLib() environment options
// will be configured on the environment which means only the StdLibSubset() initially configured with
// the library will be used.
func StdLibSubset(subset *env.LibrarySubset) StdLibOption {
	return func(lib *stdLibrary) *stdLibrary {
		lib.subset = subset
		return lib
	}
}

// StdLib returns an EnvOption for the standard library of CEL functions and macros.
func StdLib(opts ...StdLibOption) EnvOption {
	lib := &stdLibrary{}
	for _, o := range opts {
		lib = o(lib)
	}
	return Lib(lib)
}

// stdLibrary implements the Library interface and provides functional options for the core CEL
// features documented in the specification.
type stdLibrary struct {
	subset *env.LibrarySubset
}

// LibraryName implements the SingletonLibrary interface method.
func (*stdLibrary) LibraryName() string {
	return "cel.lib.std"
}

// LibraryAlias returns the simple name of the library.
func (*stdLibrary) LibraryAlias() string {
	return "stdlib"
}

// LibrarySubset returns the env.LibrarySubset definition associated with the CEL Library.
func (lib *stdLibrary) LibrarySubset() *env.LibrarySubset {
	return lib.subset
}

// CompileOptions returns options for the standard CEL function declarations and macros.
func (lib *stdLibrary) CompileOptions() []EnvOption {
	funcs := stdlib.Functions()
	macros := StandardMacros
	if lib.subset != nil {
		subMacros := []Macro{}
		for _, m := range macros {
			if lib.subset.SubsetMacro(m.Function()) {
				subMacros = append(subMacros, m)
			}
		}
		macros = subMacros
		subFuncs := []*decls.FunctionDecl{}
		for _, fn := range funcs {
			if f, include := lib.subset.SubsetFunction(fn); include {
				subFuncs = append(subFuncs, f)
			}
		}
		funcs = subFuncs
	}
	return []EnvOption{
		func(e *Env) (*Env, error) {
			var err error
			if err = lib.subset.Validate(); err != nil {
				return nil, err
			}
			e.variables = append(e.variables, stdlib.Types()...)
			for _, fn := range funcs {
				existing, found := e.functions[fn.Name()]
				if found {
					fn, err = existing.Merge(fn)
					if err != nil {
						return nil, err
					}
				}
				e.functions[fn.Name()] = fn
			}
			return e, nil
		},
		Macros(macros...),
	}
}

// ProgramOptions returns function implementations for the standard CEL functions.
func (*stdLibrary) ProgramOptions() []ProgramOption {
	return []ProgramOption{}
}

// OptionalTypes enable support for optional syntax and types in CEL.
//
// The optional value type makes it possible to express whether variables have
// been provided, whether a result has been computed, and in the future whether
// an object field path, map key value, or list index has a value.
//
// # Syntax Changes
//
// OptionalTypes are unlike other CEL extensions because they modify the CEL
// syntax itself, notably through the use of a `?` preceding a field name or
// index value.
//
// ## Field Selection
//
// The optional syntax in field selection is denoted as `obj.?field`. In other
// words, if a field is set, return `optional.of(obj.field)“, else
// `optional.none()`. The optional field selection is viral in the sense that
// after the first optional selection all subsequent selections or indices
// are treated as optional, i.e. the following expressions are equivalent:
//
//	obj.?field.subfield
//	obj.?field.?subfield
//
// ## Indexing
//
// Similar to field selection, the optional syntax can be used in index
// expressions on maps and lists:
//
//	list[?0]
//	map[?key]
//
// ## Optional Field Setting
//
// When creating map or message literals, if a field may be optionally set
// based on its presence, then placing a `?` before the field name or key
// will ensure the type on the right-hand side must be optional(T) where T
// is the type of the field or key-value.
//
// The following returns a map with the key expression set only if the
// subfield is present, otherwise an empty map is created:
//
//	{?key: obj.?field.subfield}
//
// ## Optional Element Setting
//
// When creating list literals, an element in the list may be optionally added
// when the element expression is preceded by a `?`:
//
//	[a, ?b, ?c] // return a list with either [a], [a, b], [a, b, c], or [a, c]
//
// # Optional.Of
//
// Create an optional(T) value of a given value with type T.
//
//	optional.of(10)
//
// # Optional.OfNonZeroValue
//
// Create an optional(T) value of a given value with type T if it is not a
// zero-value. A zero-value the default empty value for any given CEL type,
// including empty protobuf message types. If the value is empty, the result
// of this call will be optional.none().
//
//	optional.ofNonZeroValue([1, 2, 3]) // optional(list(int))
//	optional.ofNonZeroValue([]) // optional.none()
//	optional.ofNonZeroValue(0)  // optional.none()
//	optional.ofNonZeroValue("") // optional.none()
//
// # Optional.None
//
// Create an empty optional value.
//
// # HasValue
//
// Determine whether the optional contains a value.
//
//	optional.of(b'hello').hasValue() // true
//	optional.ofNonZeroValue({}).hasValue() // false
//
// # Value
//
// Get the value contained by the optional. If the optional does not have a
// value, the result will be a CEL error.
//
//	optional.of(b'hello').value() // b'hello'
//	optional.ofNonZeroValue({}).value() // error
//
// # Or
//
// If the value on the left-hand side is optional.none(), the optional value
// on the right hand side is returned. If the value on the left-hand set is
// valued, then it is returned. This operation is short-circuiting and will
// only evaluate as many links in the `or` chain as are needed to return a
// non-empty optional value.
//
//	obj.?field.or(m[?key])
//	l[?index].or(obj.?field.subfield).or(obj.?other)
//
// # OrValue
//
// Either return the value contained within the optional on the left-hand side
// or return the alternative value on the right hand side.
//
//	m[?key].orValue("none")
//
// # OptMap
//
// Apply a transformation to the optional's underlying value if it is not empty
// and return an optional typed result based on the transformation. The
// transformation expression type must return a type T which is wrapped into
// an optional.
//
//	msg.?elements.optMap(e, e.size()).orValue(0)
//
// # OptFlatMap
//
// Introduced in version: 1
//
// Apply a transformation to the optional's underlying value if it is not empty
// and return the result. The transform expression must return an optional(T)
// rather than type T. This can be useful when dealing with zero values and
// conditionally generating an empty or non-empty result in ways which cannot
// be expressed with `optMap`.
//
//	msg.?elements.optFlatMap(e, e[?0]) // return the first element if present.
//
// # First
//
// Introduced in version: 2
//
// Returns an optional with the first value from the right hand list, or
// optional.None.
//
// [1, 2, 3].first().value() == 1
//
// # Last
//
// Introduced in version: 2
//
// Returns an optional with the last value from the right hand list, or
// optional.None.
//
// [1, 2, 3].last().value() == 3
//
// This is syntactic sugar for msg.elements[msg.elements.size()-1].
//
// # Unwrap / UnwrapOpt
//
// Introduced in version: 2
//
// Returns a list of all the values that are not none in the input list of optional values.
// Can be used as optional.unwrap(List[T]) or with postfix notation: List[T].unwrapOpt()
//
// optional.unwrap([optional.of(42), optional.none()]) == [42]
// [optional.of(42), optional.none()].unwrapOpt() == [42]
func OptionalTypes(opts ...OptionalTypesOption) EnvOption {
	lib := &optionalLib{version: math.MaxUint32}
	for _, opt := range opts {
		lib = opt(lib)
	}
	return Lib(lib)
}

type optionalLib struct {
	version uint32
}

// OptionalTypesOption is a functional interface for configuring the strings library.
type OptionalTypesOption func(*optionalLib) *optionalLib

// OptionalTypesVersion configures the version of the optional type library.
//
// The version limits which functions are available. Only functions introduced
// below or equal to the given version included in the library. If this option
// is not set, all functions are available.
//
// See the library documentation to determine which version a function was introduced.
// If the documentation does not state which version a function was introduced, it can
// be assumed to be introduced at version 0, when the library was first created.
func OptionalTypesVersion(version uint32) OptionalTypesOption {
	return func(lib *optionalLib) *optionalLib {
		lib.version = version
		return lib
	}
}

// LibraryName implements the SingletonLibrary interface method.
func (*optionalLib) LibraryName() string {
	return "cel.lib.optional"
}

// LibraryAlias returns the simple name of the library.
func (*optionalLib) LibraryAlias() string {
	return "optional"
}

// LibraryVersion returns the version of the library.
func (lib *optionalLib) LibraryVersion() uint32 {
	return lib.version
}

// CompileOptions implements the Library interface method.
func (lib *optionalLib) CompileOptions() []EnvOption {
	paramTypeK := TypeParamType("K")
	paramTypeV := TypeParamType("V")
	optionalTypeV := OptionalType(paramTypeV)
	listTypeV := ListType(paramTypeV)
	mapTypeKV := MapType(paramTypeK, paramTypeV)
	listOptionalTypeV := ListType(optionalTypeV)

	opts := []EnvOption{
		// Enable the optional syntax in the parser.
		enableOptionalSyntax(),

		// Introduce the optional type.
		Types(types.OptionalType),

		// Configure the optMap and optFlatMap macros.
		Macros(ReceiverMacro(optMapMacro, 2, optMap,
			MacroDocs(`perform computation on the value if present and return the result as an optional`),
			MacroExamples(
				common.MultilineDescription(
					`// sub with the prefix 'dev.cel' or optional.none()`,
					`request.auth.tokens.?sub.optMap(id, 'dev.cel.' + id)`),
				`optional.none().optMap(i, i * 2) // optional.none()`))),

		// Global and member functions for working with optional values.
		Function(optionalOfFunc,
			FunctionDocs(`create a new optional_type(T) with a value where any value is considered valid`),
			Overload("optional_of", []*Type{paramTypeV}, optionalTypeV,
				OverloadExamples(`optional.of(1) // optional(1)`),
				UnaryBinding(func(value ref.Val) ref.Val {
					return types.OptionalOf(value)
				}))),
		Function(optionalOfNonZeroValueFunc,
			FunctionDocs(`create a new optional_type(T) with a value, if the value is not a zero or empty value`),
			Overload("optional_ofNonZeroValue", []*Type{paramTypeV}, optionalTypeV,
				OverloadExamples(
					`optional.ofNonZeroValue(null) // optional.none()`,
					`optional.ofNonZeroValue("") // optional.none()`,
					`optional.ofNonZeroValue("hello") // optional.of('hello')`),
				UnaryBinding(func(value ref.Val) ref.Val {
					v, isZeroer := value.(traits.Zeroer)
					if !isZeroer || !v.IsZeroValue() {
						return types.OptionalOf(value)
					}
					return types.OptionalNone
				}))),
		Function(optionalNoneFunc,
			FunctionDocs(`singleton value representing an optional without a value`),
			Overload("optional_none", []*Type{}, optionalTypeV,
				OverloadExamples(`optional.none()`),
				FunctionBinding(func(values ...ref.Val) ref.Val {
					return types.OptionalNone
				}))),
		Function(valueFunc,
			FunctionDocs(`obtain the value contained by the optional, error if optional.none()`),
			MemberOverload("optional_value", []*Type{optionalTypeV}, paramTypeV,
				OverloadExamples(
					`optional.of(1).value() // 1`,
					`optional.none().value() // error`),
				UnaryBinding(func(value ref.Val) ref.Val {
					opt := value.(*types.Optional)
					return opt.GetValue()
				}))),
		Function(hasValueFunc,
			FunctionDocs(`determine whether the optional contains a value`),
			MemberOverload("optional_hasValue", []*Type{optionalTypeV}, BoolType,
				OverloadExamples(`optional.of({1: 2}).hasValue() // true`),
				UnaryBinding(func(value ref.Val) ref.Val {
					opt := value.(*types.Optional)
					return types.Bool(opt.HasValue())
				}))),

		// Implementation of 'or' and 'orValue' are special-cased to support short-circuiting in the
		// evaluation chain.
		Function("or",
			FunctionDocs(`chain optional expressions together, picking the first valued optional expression`),
			MemberOverload("optional_or_optional", []*Type{optionalTypeV, optionalTypeV}, optionalTypeV,
				OverloadExamples(
					`optional.none().or(optional.of(1)) // optional.of(1)`,
					common.MultilineDescription(
						`// either a value from the first list, a value from the second, or optional.none()`,
						`[1, 2, 3][?x].or([3, 4, 5][?y])`)))),
		Function("orValue",
			FunctionDocs(`chain optional expressions together picking the first valued optional or the default value`),
			MemberOverload("optional_orValue_value", []*Type{optionalTypeV, paramTypeV}, paramTypeV,
				OverloadExamples(
					common.MultilineDescription(
						`// pick the value for the given key if the key exists, otherwise return 'you'`,
						`{'hello': 'world', 'goodbye': 'cruel world'}[?greeting].orValue('you')`)))),

		// OptSelect is handled specially by the type-checker, so the receiver's field type is used to determine the
		// optput type.
		Function(operators.OptSelect,
			FunctionDocs(`if the field is present create an optional of the field value, otherwise return optional.none()`),
			Overload("select_optional_field", []*Type{DynType, StringType}, optionalTypeV,
				OverloadExamples(
					`msg.?field // optional.of(field) if non-empty, otherwise optional.none()`,
					`msg.?field.?nested_field // optional.of(nested_field) if both field and nested_field are non-empty.`))),

		// OptIndex is handled mostly like any other indexing operation on a list or map, so the type-checker can use
		// these signatures to determine type-agreement without any special handling.
		Function(operators.OptIndex,
			FunctionDocs(`if the index is present create an optional of the field value, otherwise return optional.none()`),
			Overload("list_optindex_optional_int", []*Type{listTypeV, IntType}, optionalTypeV,
				OverloadExamples(`[1, 2, 3][?x] // element value if x is in the list size, else optional.none()`)),
			Overload("optional_list_optindex_optional_int", []*Type{OptionalType(listTypeV), IntType}, optionalTypeV),
			Overload("map_optindex_optional_value", []*Type{mapTypeKV, paramTypeK}, optionalTypeV,
				OverloadExamples(
					`map_value[?key] // value at the key if present, else optional.none()`,
					common.MultilineDescription(
						`// map key-value if index is a valid map key, else optional.none()`,
						`{0: 2, 2: 4, 6: 8}[?index]`))),
			Overload("optional_map_optindex_optional_value", []*Type{OptionalType(mapTypeKV), paramTypeK}, optionalTypeV)),

		// Index overloads to accommodate using an optional value as the operand.
		Function(operators.Index,
			Overload("optional_list_index_int", []*Type{OptionalType(listTypeV), IntType}, optionalTypeV),
			Overload("optional_map_index_value", []*Type{OptionalType(mapTypeKV), paramTypeK}, optionalTypeV)),
	}
	if lib.version >= 1 {
		opts = append(opts, Macros(ReceiverMacro(optFlatMapMacro, 2, optFlatMap,
			MacroDocs(`perform computation on the value if present and produce an optional value within the computation`),
			MacroExamples(
				common.MultilineDescription(
					`// m = {'key': {}}`,
					`m.?key.optFlatMap(k, k.?subkey) // optional.none()`),
				common.MultilineDescription(
					`// m = {'key': {'subkey': 'value'}}`,
					`m.?key.optFlatMap(k, k.?subkey) // optional.of('value')`),
			))))
	}

	if lib.version >= 2 {
		opts = append(opts, Function("last",
			FunctionDocs(`return the last value in a list if present, otherwise optional.none()`),
			MemberOverload("list_last", []*Type{listTypeV}, optionalTypeV,
				OverloadExamples(
					`[].last() // optional.none()`,
					`[1, 2, 3].last() ? optional.of(3)`),
				UnaryBinding(func(v ref.Val) ref.Val {
					list := v.(traits.Lister)
					sz := list.Size().(types.Int)
					if sz == types.IntZero {
						return types.OptionalNone
					}
					return types.OptionalOf(list.Get(types.Int(sz - 1)))
				}),
			),
		))

		opts = append(opts, Function("first",
			FunctionDocs(`return the first value in a list if present, otherwise optional.none()`),
			MemberOverload("list_first", []*Type{listTypeV}, optionalTypeV,
				OverloadExamples(
					`[].first() // optional.none()`,
					`[1, 2, 3].first() ? optional.of(1)`),
				UnaryBinding(func(v ref.Val) ref.Val {
					list := v.(traits.Lister)
					sz := list.Size().(types.Int)
					if sz == types.IntZero {
						return types.OptionalNone
					}
					return types.OptionalOf(list.Get(types.Int(0)))
				}),
			),
		))

		opts = append(opts, Function(optionalUnwrapFunc,
			FunctionDocs(`convert a list of optional values to a list containing only value which are not optional.none()`),
			Overload("optional_unwrap", []*Type{listOptionalTypeV}, listTypeV,
				OverloadExamples(`optional.unwrap([optional.of(1), optional.none()]) // [1]`),
				UnaryBinding(optUnwrap))))
		opts = append(opts, Function(unwrapOptFunc,
			FunctionDocs(`convert a list of optional values to a list containing only value which are not optional.none()`),
			MemberOverload("optional_unwrapOpt", []*Type{listOptionalTypeV}, listTypeV,
				OverloadExamples(`[optional.of(1), optional.none()].unwrapOpt() // [1]`),
				UnaryBinding(optUnwrap))))
	}

	return opts
}

// ProgramOptions implements the Library interface method.
func (lib *optionalLib) ProgramOptions() []ProgramOption {
	return []ProgramOption{
		CustomDecorator(decorateOptionalOr),
	}
}

// Version returns the current version of the library.
func (lib *optionalLib) Version() uint32 {
	return lib.version
}

func optMap(meh MacroExprFactory, target ast.Expr, args []ast.Expr) (ast.Expr, *Error) {
	varIdent := args[0]
	varName := ""
	switch varIdent.Kind() {
	case ast.IdentKind:
		varName = varIdent.AsIdent()
	default:
		return nil, meh.NewError(varIdent.ID(), "optMap() variable name must be a simple identifier")
	}
	mapExpr := args[1]
	return meh.NewCall(
		operators.Conditional,
		meh.NewMemberCall(hasValueFunc, target),
		meh.NewCall(optionalOfFunc,
			meh.NewComprehension(
				meh.NewList(),
				unusedIterVar,
				varName,
				meh.NewMemberCall(valueFunc, meh.Copy(target)),
				meh.NewLiteral(types.False),
				meh.NewIdent(varName),
				mapExpr,
			),
		),
		meh.NewCall(optionalNoneFunc),
	), nil
}

func optFlatMap(meh MacroExprFactory, target ast.Expr, args []ast.Expr) (ast.Expr, *Error) {
	varIdent := args[0]
	varName := ""
	switch varIdent.Kind() {
	case ast.IdentKind:
		varName = varIdent.AsIdent()
	default:
		return nil, meh.NewError(varIdent.ID(), "optFlatMap() variable name must be a simple identifier")
	}
	mapExpr := args[1]
	return meh.NewCall(
		operators.Conditional,
		meh.NewMemberCall(hasValueFunc, target),
		meh.NewComprehension(
			meh.NewList(),
			unusedIterVar,
			varName,
			meh.NewMemberCall(valueFunc, meh.Copy(target)),
			meh.NewLiteral(types.False),
			meh.NewIdent(varName),
			mapExpr,
		),
		meh.NewCall(optionalNoneFunc),
	), nil
}

func optUnwrap(value ref.Val) ref.Val {
	list := value.(traits.Lister)
	var unwrappedList []ref.Val
	iter := list.Iterator()
	for iter.HasNext() == types.True {
		val := iter.Next()
		opt, isOpt := val.(*types.Optional)
		if !isOpt {
			return types.WrapErr(fmt.Errorf("value %v is not optional", val))
		}
		if opt.HasValue() {
			unwrappedList = append(unwrappedList, opt.GetValue())
		}
	}
	return types.DefaultTypeAdapter.NativeToValue(unwrappedList)
}

func enableOptionalSyntax() EnvOption {
	return func(e *Env) (*Env, error) {
		e.prsrOpts = append(e.prsrOpts, parser.EnableOptionalSyntax(true))
		return e, nil
	}
}

// EnableErrorOnBadPresenceTest enables error generation when a presence test or optional field
// selection is performed on a primitive type.
func EnableErrorOnBadPresenceTest(value bool) EnvOption {
	return features(featureEnableErrorOnBadPresenceTest, value)
}

func decorateOptionalOr(i interpreter.Interpretable) (interpreter.Interpretable, error) {
	call, ok := i.(interpreter.InterpretableCall)
	if !ok {
		return i, nil
	}
	args := call.Args()
	if len(args) != 2 {
		return i, nil
	}
	switch call.Function() {
	case "or":
		if call.OverloadID() != "" && call.OverloadID() != "optional_or_optional" {
			return i, nil
		}
		return &evalOptionalOr{
			id:  call.ID(),
			lhs: args[0],
			rhs: args[1],
		}, nil
	case "orValue":
		if call.OverloadID() != "" && call.OverloadID() != "optional_orValue_value" {
			return i, nil
		}
		return &evalOptionalOrValue{
			id:  call.ID(),
			lhs: args[0],
			rhs: args[1],
		}, nil
	default:
		return i, nil
	}
}

// evalOptionalOr selects between two optional values, either the first if it has a value, or
// the second optional expression is evaluated and returned.
type evalOptionalOr struct {
	id  int64
	lhs interpreter.Interpretable
	rhs interpreter.Interpretable
}

// ID implements the Interpretable interface method.
func (opt *evalOptionalOr) ID() int64 {
	return opt.id
}

// Eval evaluates the left-hand side optional to determine whether it contains a value, else
// proceeds with the right-hand side evaluation.
func (opt *evalOptionalOr) Eval(ctx interpreter.Activation) ref.Val {
	// short-circuit lhs.
	optLHS := opt.lhs.Eval(ctx)
	optVal, ok := optLHS.(*types.Optional)
	if !ok {
		return optLHS
	}
	if optVal.HasValue() {
		return optVal
	}
	return opt.rhs.Eval(ctx)
}

// evalOptionalOrValue selects between an optional or a concrete value. If the optional has a value,
// its value is returned, otherwise the alternative value expression is evaluated and returned.
type evalOptionalOrValue struct {
	id  int64
	lhs interpreter.Interpretable
	rhs interpreter.Interpretable
}

// ID implements the Interpretable interface method.
func (opt *evalOptionalOrValue) ID() int64 {
	return opt.id
}

// Eval evaluates the left-hand side optional to determine whether it contains a value, else
// proceeds with the right-hand side evaluation.
func (opt *evalOptionalOrValue) Eval(ctx interpreter.Activation) ref.Val {
	// short-circuit lhs.
	optLHS := opt.lhs.Eval(ctx)
	optVal, ok := optLHS.(*types.Optional)
	if !ok {
		return optLHS
	}
	if optVal.HasValue() {
		return optVal.GetValue()
	}
	return opt.rhs.Eval(ctx)
}

type timeLegacyLibrary struct{}

func (timeLegacyLibrary) CompileOptions() []EnvOption {
	return timeOverloadDeclarations
}

func (timeLegacyLibrary) ProgramOptions() []ProgramOption {
	return []ProgramOption{}
}

// Declarations and functions which enable using UTC on time.Time inputs when the timezone is unspecified
// in the CEL expression.
var (
	timeOverloadDeclarations = []EnvOption{
		Function(overloads.TimeGetFullYear,
			MemberOverload(overloads.TimestampToYear, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					t := ts.(types.Timestamp)
					return t.Receive(overloads.TimeGetFullYear, overloads.TimestampToYear, []ref.Val{})
				}),
			),
		),
		Function(overloads.TimeGetMonth,
			MemberOverload(overloads.TimestampToMonth, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					t := ts.(types.Timestamp)
					return t.Receive(overloads.TimeGetMonth, overloads.TimestampToMonth, []ref.Val{})
				}),
			),
		),
		Function(overloads.TimeGetDayOfYear,
			MemberOverload(overloads.TimestampToDayOfYear, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					t := ts.(types.Timestamp)
					return t.Receive(overloads.TimeGetDayOfYear, overloads.TimestampToDayOfYear, []ref.Val{})
				}),
			),
		),
		Function(overloads.TimeGetDayOfMonth,
			MemberOverload(overloads.TimestampToDayOfMonthZeroBased, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					t := ts.(types.Timestamp)
					return t.Receive(overloads.TimeGetDayOfMonth, overloads.TimestampToDayOfMonthZeroBased, []ref.Val{})
				}),
			),
		),
		Function(overloads.TimeGetDate,
			MemberOverload(overloads.TimestampToDayOfMonthOneBased, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					t := ts.(types.Timestamp)
					return t.Receive(overloads.TimeGetDate, overloads.TimestampToDayOfMonthOneBased, []ref.Val{})
				}),
			),
		),
		Function(overloads.TimeGetDayOfWeek,
			MemberOverload(overloads.TimestampToDayOfWeek, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					t := ts.(types.Timestamp)
					return t.Receive(overloads.TimeGetDayOfWeek, overloads.TimestampToDayOfWeek, []ref.Val{})
				}),
			),
		),
		Function(overloads.TimeGetHours,
			MemberOverload(overloads.TimestampToHours, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					t := ts.(types.Timestamp)
					return t.Receive(overloads.TimeGetHours, overloads.TimestampToHours, []ref.Val{})
				}),
			),
		),
		Function(overloads.TimeGetMinutes,
			MemberOverload(overloads.TimestampToMinutes, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					t := ts.(types.Timestamp)
					return t.Receive(overloads.TimeGetMinutes, overloads.TimestampToMinutes, []ref.Val{})
				}),
			),
		),
		Function(overloads.TimeGetSeconds,
			MemberOverload(overloads.TimestampToSeconds, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					t := ts.(types.Timestamp)
					return t.Receive(overloads.TimeGetSeconds, overloads.TimestampToSeconds, []ref.Val{})
				}),
			),
		),
		Function(overloads.TimeGetMilliseconds,
			MemberOverload(overloads.TimestampToMilliseconds, []*Type{TimestampType}, IntType,
				UnaryBinding(func(ts ref.Val) ref.Val {
					t := ts.(types.Timestamp)
					return t.Receive(overloads.TimeGetMilliseconds, overloads.TimestampToMilliseconds, []ref.Val{})
				}),
			),
		),
	}
)
