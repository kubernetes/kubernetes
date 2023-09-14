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

package cel

import (
	"fmt"
	"strings"

	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/interpreter/functions"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// Kind indicates a CEL type's kind which is used to differentiate quickly between simple and complex types.
type Kind uint

const (
	// DynKind represents a dynamic type. This kind only exists at type-check time.
	DynKind Kind = iota

	// AnyKind represents a google.protobuf.Any type. This kind only exists at type-check time.
	AnyKind

	// BoolKind represents a boolean type.
	BoolKind

	// BytesKind represents a bytes type.
	BytesKind

	// DoubleKind represents a double type.
	DoubleKind

	// DurationKind represents a CEL duration type.
	DurationKind

	// IntKind represents an integer type.
	IntKind

	// ListKind represents a list type.
	ListKind

	// MapKind represents a map type.
	MapKind

	// NullTypeKind represents a null type.
	NullTypeKind

	// OpaqueKind represents an abstract type which has no accessible fields.
	OpaqueKind

	// StringKind represents a string type.
	StringKind

	// StructKind represents a structured object with typed fields.
	StructKind

	// TimestampKind represents a a CEL time type.
	TimestampKind

	// TypeKind represents the CEL type.
	TypeKind

	// TypeParamKind represents a parameterized type whose type name will be resolved at type-check time, if possible.
	TypeParamKind

	// UintKind represents a uint type.
	UintKind
)

var (
	// AnyType represents the google.protobuf.Any type.
	AnyType = &Type{
		kind:        AnyKind,
		runtimeType: types.NewTypeValue("google.protobuf.Any"),
	}
	// BoolType represents the bool type.
	BoolType = &Type{
		kind:        BoolKind,
		runtimeType: types.BoolType,
	}
	// BytesType represents the bytes type.
	BytesType = &Type{
		kind:        BytesKind,
		runtimeType: types.BytesType,
	}
	// DoubleType represents the double type.
	DoubleType = &Type{
		kind:        DoubleKind,
		runtimeType: types.DoubleType,
	}
	// DurationType represents the CEL duration type.
	DurationType = &Type{
		kind:        DurationKind,
		runtimeType: types.DurationType,
	}
	// DynType represents a dynamic CEL type whose type will be determined at runtime from context.
	DynType = &Type{
		kind:        DynKind,
		runtimeType: types.NewTypeValue("dyn"),
	}
	// IntType represents the int type.
	IntType = &Type{
		kind:        IntKind,
		runtimeType: types.IntType,
	}
	// NullType represents the type of a null value.
	NullType = &Type{
		kind:        NullTypeKind,
		runtimeType: types.NullType,
	}
	// StringType represents the string type.
	StringType = &Type{
		kind:        StringKind,
		runtimeType: types.StringType,
	}
	// TimestampType represents the time type.
	TimestampType = &Type{
		kind:        TimestampKind,
		runtimeType: types.TimestampType,
	}
	// TypeType represents a CEL type
	TypeType = &Type{
		kind:        TypeKind,
		runtimeType: types.TypeType,
	}
	// UintType represents a uint type.
	UintType = &Type{
		kind:        UintKind,
		runtimeType: types.UintType,
	}
)

// Type holds a reference to a runtime type with an optional type-checked set of type parameters.
type Type struct {
	// kind indicates general category of the type.
	kind Kind

	// runtimeType is the runtime type of the declaration.
	runtimeType ref.Type

	// parameters holds the optional type-checked set of type parameters that are used during static analysis.
	parameters []*Type

	// isAssignableType function determines whether one type is assignable to this type.
	// A nil value for the isAssignableType function falls back to equality of kind, runtimeType, and parameters.
	isAssignableType func(other *Type) bool

	// isAssignableRuntimeType function determines whether the runtime type (with erasure) is assignable to this type.
	// A nil value for the isAssignableRuntimeType function falls back to the equality of the type or type name.
	isAssignableRuntimeType func(other ref.Val) bool
}

// IsAssignableType determines whether the current type is type-check assignable from the input fromType.
func (t *Type) IsAssignableType(fromType *Type) bool {
	if t.isAssignableType != nil {
		return t.isAssignableType(fromType)
	}
	return t.defaultIsAssignableType(fromType)
}

// IsAssignableRuntimeType determines whether the current type is runtime assignable from the input runtimeType.
//
// At runtime, parameterized types are erased and so a function which type-checks to support a map(string, string)
// will have a runtime assignable type of a map.
func (t *Type) IsAssignableRuntimeType(val ref.Val) bool {
	if t.isAssignableRuntimeType != nil {
		return t.isAssignableRuntimeType(val)
	}
	return t.defaultIsAssignableRuntimeType(val)
}

// String returns a human-readable definition of the type name.
func (t *Type) String() string {
	if len(t.parameters) == 0 {
		return t.runtimeType.TypeName()
	}
	params := make([]string, len(t.parameters))
	for i, p := range t.parameters {
		params[i] = p.String()
	}
	return fmt.Sprintf("%s(%s)", t.runtimeType.TypeName(), strings.Join(params, ", "))
}

// isDyn indicates whether the type is dynamic in any way.
func (t *Type) isDyn() bool {
	return t.kind == DynKind || t.kind == AnyKind || t.kind == TypeParamKind
}

// equals indicates whether two types have the same kind, type name, and parameters.
func (t *Type) equals(other *Type) bool {
	if t.kind != other.kind ||
		t.runtimeType.TypeName() != other.runtimeType.TypeName() ||
		len(t.parameters) != len(other.parameters) {
		return false
	}
	for i, p := range t.parameters {
		if !p.equals(other.parameters[i]) {
			return false
		}
	}
	return true
}

// defaultIsAssignableType provides the standard definition of what it means for one type to be assignable to another
// where any of the following may return a true result:
// - The from types are the same instance
// - The target type is dynamic
// - The fromType has the same kind and type name as the target type, and all parameters of the target type
//
//	are IsAssignableType() from the parameters of the fromType.
func (t *Type) defaultIsAssignableType(fromType *Type) bool {
	if t == fromType || t.isDyn() {
		return true
	}
	if t.kind != fromType.kind ||
		t.runtimeType.TypeName() != fromType.runtimeType.TypeName() ||
		len(t.parameters) != len(fromType.parameters) {
		return false
	}
	for i, tp := range t.parameters {
		fp := fromType.parameters[i]
		if !tp.IsAssignableType(fp) {
			return false
		}
	}
	return true
}

// defaultIsAssignableRuntimeType inspects the type and in the case of list and map elements, the key and element types
// to determine whether a ref.Val is assignable to the declared type for a function signature.
func (t *Type) defaultIsAssignableRuntimeType(val ref.Val) bool {
	valType := val.Type()
	if !(t.runtimeType == valType || t.isDyn() || t.runtimeType.TypeName() == valType.TypeName()) {
		return false
	}
	switch t.runtimeType {
	case types.ListType:
		elemType := t.parameters[0]
		l := val.(traits.Lister)
		if l.Size() == types.IntZero {
			return true
		}
		it := l.Iterator()
		for it.HasNext() == types.True {
			elemVal := it.Next()
			return elemType.IsAssignableRuntimeType(elemVal)
		}
	case types.MapType:
		keyType := t.parameters[0]
		elemType := t.parameters[1]
		m := val.(traits.Mapper)
		if m.Size() == types.IntZero {
			return true
		}
		it := m.Iterator()
		for it.HasNext() == types.True {
			keyVal := it.Next()
			elemVal := m.Get(keyVal)
			return keyType.IsAssignableRuntimeType(keyVal) && elemType.IsAssignableRuntimeType(elemVal)
		}
	}
	return true
}

// ListType creates an instances of a list type value with the provided element type.
func ListType(elemType *Type) *Type {
	return &Type{
		kind:        ListKind,
		runtimeType: types.ListType,
		parameters:  []*Type{elemType},
	}
}

// MapType creates an instance of a map type value with the provided key and value types.
func MapType(keyType, valueType *Type) *Type {
	return &Type{
		kind:        MapKind,
		runtimeType: types.MapType,
		parameters:  []*Type{keyType, valueType},
	}
}

// NullableType creates an instance of a nullable type with the provided wrapped type.
//
// Note: only primitive types are supported as wrapped types.
func NullableType(wrapped *Type) *Type {
	return &Type{
		kind:        wrapped.kind,
		runtimeType: wrapped.runtimeType,
		parameters:  wrapped.parameters,
		isAssignableType: func(other *Type) bool {
			return NullType.IsAssignableType(other) || wrapped.IsAssignableType(other)
		},
		isAssignableRuntimeType: func(other ref.Val) bool {
			return NullType.IsAssignableRuntimeType(other) || wrapped.IsAssignableRuntimeType(other)
		},
	}
}

// OptionalType creates an abstract parameterized type instance corresponding to CEL's notion of optional.
func OptionalType(param *Type) *Type {
	return OpaqueType("optional", param)
}

// OpaqueType creates an abstract parameterized type with a given name.
func OpaqueType(name string, params ...*Type) *Type {
	return &Type{
		kind:        OpaqueKind,
		runtimeType: types.NewTypeValue(name),
		parameters:  params,
	}
}

// ObjectType creates a type references to an externally defined type, e.g. a protobuf message type.
func ObjectType(typeName string) *Type {
	return &Type{
		kind:        StructKind,
		runtimeType: types.NewObjectTypeValue(typeName),
	}
}

// TypeParamType creates a parameterized type instance.
func TypeParamType(paramName string) *Type {
	return &Type{
		kind:        TypeParamKind,
		runtimeType: types.NewTypeValue(paramName),
	}
}

// Variable creates an instance of a variable declaration with a variable name and type.
func Variable(name string, t *Type) EnvOption {
	return func(e *Env) (*Env, error) {
		et, err := TypeToExprType(t)
		if err != nil {
			return nil, err
		}
		e.declarations = append(e.declarations, decls.NewVar(name, et))
		return e, nil
	}
}

// Function defines a function and overloads with optional singleton or per-overload bindings.
//
// Using Function is roughly equivalent to calling Declarations() to declare the function signatures
// and Functions() to define the function bindings, if they have been defined. Specifying the
// same function name more than once will result in the aggregation of the function overloads. If any
// signatures conflict between the existing and new function definition an error will be raised.
// However, if the signatures are identical and the overload ids are the same, the redefinition will
// be considered a no-op.
//
// One key difference with using Function() is that each FunctionDecl provided will handle dynamic
// dispatch based on the type-signatures of the overloads provided which means overload resolution at
// runtime is handled out of the box rather than via a custom binding for overload resolution via
// Functions():
//
// - Overloads are searched in the order they are declared
// - Dynamic dispatch for lists and maps is limited by inspection of the list and map contents
//
//	at runtime. Empty lists and maps will result in a 'default dispatch'
//
// - In the event that a default dispatch occurs, the first overload provided is the one invoked
//
// If you intend to use overloads which differentiate based on the key or element type of a list or
// map, consider using a generic function instead: e.g. func(list(T)) or func(map(K, V)) as this
// will allow your implementation to determine how best to handle dispatch and the default behavior
// for empty lists and maps whose contents cannot be inspected.
//
// For functions which use parameterized opaque types (abstract types), consider using a singleton
// function which is capable of inspecting the contents of the type and resolving the appropriate
// overload as CEL can only make inferences by type-name regarding such types.
func Function(name string, opts ...FunctionOpt) EnvOption {
	return func(e *Env) (*Env, error) {
		fn := &functionDecl{
			name:      name,
			overloads: []*overloadDecl{},
			options:   opts,
		}
		err := fn.init()
		if err != nil {
			return nil, err
		}
		_, err = functionDeclToExprDecl(fn)
		if err != nil {
			return nil, err
		}
		if existing, found := e.functions[fn.name]; found {
			fn, err = existing.merge(fn)
			if err != nil {
				return nil, err
			}
		}
		e.functions[name] = fn
		return e, nil
	}
}

// FunctionOpt defines a functional  option for configuring a function declaration.
type FunctionOpt func(*functionDecl) (*functionDecl, error)

// SingletonUnaryBinding creates a singleton function definition to be used for all function overloads.
//
// Note, this approach works well if operand is expected to have a specific trait which it implements,
// e.g. traits.ContainerType. Otherwise, prefer per-overload function bindings.
func SingletonUnaryBinding(fn functions.UnaryOp, traits ...int) FunctionOpt {
	trait := 0
	for _, t := range traits {
		trait = trait | t
	}
	return func(f *functionDecl) (*functionDecl, error) {
		if f.singleton != nil {
			return nil, fmt.Errorf("function already has a singleton binding: %s", f.name)
		}
		f.singleton = &functions.Overload{
			Operator:     f.name,
			Unary:        fn,
			OperandTrait: trait,
		}
		return f, nil
	}
}

// SingletonBinaryImpl creates a singleton function definition to be used with all function overloads.
//
// Note, this approach works well if operand is expected to have a specific trait which it implements,
// e.g. traits.ContainerType. Otherwise, prefer per-overload function bindings.
//
// Deprecated: use SingletonBinaryBinding
func SingletonBinaryImpl(fn functions.BinaryOp, traits ...int) FunctionOpt {
	return SingletonBinaryBinding(fn, traits...)
}

// SingletonBinaryBinding creates a singleton function definition to be used with all function overloads.
//
// Note, this approach works well if operand is expected to have a specific trait which it implements,
// e.g. traits.ContainerType. Otherwise, prefer per-overload function bindings.
func SingletonBinaryBinding(fn functions.BinaryOp, traits ...int) FunctionOpt {
	trait := 0
	for _, t := range traits {
		trait = trait | t
	}
	return func(f *functionDecl) (*functionDecl, error) {
		if f.singleton != nil {
			return nil, fmt.Errorf("function already has a singleton binding: %s", f.name)
		}
		f.singleton = &functions.Overload{
			Operator:     f.name,
			Binary:       fn,
			OperandTrait: trait,
		}
		return f, nil
	}
}

// SingletonFunctionImpl creates a singleton function definition to be used with all function overloads.
//
// Note, this approach works well if operand is expected to have a specific trait which it implements,
// e.g. traits.ContainerType. Otherwise, prefer per-overload function bindings.
//
// Deprecated: use SingletonFunctionBinding
func SingletonFunctionImpl(fn functions.FunctionOp, traits ...int) FunctionOpt {
	return SingletonFunctionBinding(fn, traits...)
}

// SingletonFunctionBinding creates a singleton function definition to be used with all function overloads.
//
// Note, this approach works well if operand is expected to have a specific trait which it implements,
// e.g. traits.ContainerType. Otherwise, prefer per-overload function bindings.
func SingletonFunctionBinding(fn functions.FunctionOp, traits ...int) FunctionOpt {
	trait := 0
	for _, t := range traits {
		trait = trait | t
	}
	return func(f *functionDecl) (*functionDecl, error) {
		if f.singleton != nil {
			return nil, fmt.Errorf("function already has a singleton binding: %s", f.name)
		}
		f.singleton = &functions.Overload{
			Operator:     f.name,
			Function:     fn,
			OperandTrait: trait,
		}
		return f, nil
	}
}

// Overload defines a new global overload with an overload id, argument types, and result type. Through the
// use of OverloadOpt options, the overload may also be configured with a binding, an operand trait, and to
// be non-strict.
//
// Note: function bindings should be commonly configured with Overload instances whereas operand traits and
// strict-ness should be rare occurrences.
func Overload(overloadID string, args []*Type, resultType *Type, opts ...OverloadOpt) FunctionOpt {
	return newOverload(overloadID, false, args, resultType, opts...)
}

// MemberOverload defines a new receiver-style overload (or member function) with an overload id, argument types,
// and result type. Through the use of OverloadOpt options, the overload may also be configured with a binding,
// an operand trait, and to be non-strict.
//
// Note: function bindings should be commonly configured with Overload instances whereas operand traits and
// strict-ness should be rare occurrences.
func MemberOverload(overloadID string, args []*Type, resultType *Type, opts ...OverloadOpt) FunctionOpt {
	return newOverload(overloadID, true, args, resultType, opts...)
}

// OverloadOpt is a functional option for configuring a function overload.
type OverloadOpt func(*overloadDecl) (*overloadDecl, error)

// UnaryBinding provides the implementation of a unary overload. The provided function is protected by a runtime
// type-guard which ensures runtime type agreement between the overload signature and runtime argument types.
func UnaryBinding(binding functions.UnaryOp) OverloadOpt {
	return func(o *overloadDecl) (*overloadDecl, error) {
		if o.hasBinding() {
			return nil, fmt.Errorf("overload already has a binding: %s", o.id)
		}
		if len(o.argTypes) != 1 {
			return nil, fmt.Errorf("unary function bound to non-unary overload: %s", o.id)
		}
		o.unaryOp = binding
		return o, nil
	}
}

// BinaryBinding provides the implementation of a binary overload. The provided function is protected by a runtime
// type-guard which ensures runtime type agreement between the overload signature and runtime argument types.
func BinaryBinding(binding functions.BinaryOp) OverloadOpt {
	return func(o *overloadDecl) (*overloadDecl, error) {
		if o.hasBinding() {
			return nil, fmt.Errorf("overload already has a binding: %s", o.id)
		}
		if len(o.argTypes) != 2 {
			return nil, fmt.Errorf("binary function bound to non-binary overload: %s", o.id)
		}
		o.binaryOp = binding
		return o, nil
	}
}

// FunctionBinding provides the implementation of a variadic overload. The provided function is protected by a runtime
// type-guard which ensures runtime type agreement between the overload signature and runtime argument types.
func FunctionBinding(binding functions.FunctionOp) OverloadOpt {
	return func(o *overloadDecl) (*overloadDecl, error) {
		if o.hasBinding() {
			return nil, fmt.Errorf("overload already has a binding: %s", o.id)
		}
		o.functionOp = binding
		return o, nil
	}
}

// OverloadIsNonStrict enables the function to be called with error and unknown argument values.
//
// Note: do not use this option unless absoluately necessary as it should be an uncommon feature.
func OverloadIsNonStrict() OverloadOpt {
	return func(o *overloadDecl) (*overloadDecl, error) {
		o.nonStrict = true
		return o, nil
	}
}

// OverloadOperandTrait configures a set of traits which the first argument to the overload must implement in order to be
// successfully invoked.
func OverloadOperandTrait(trait int) OverloadOpt {
	return func(o *overloadDecl) (*overloadDecl, error) {
		o.operandTrait = trait
		return o, nil
	}
}

type functionDecl struct {
	name        string
	overloads   []*overloadDecl
	options     []FunctionOpt
	singleton   *functions.Overload
	initialized bool
}

// init ensures that a function's options have been applied.
//
// This function is used in both the environment configuration and internally for function merges.
func (f *functionDecl) init() error {
	if f.initialized {
		return nil
	}
	f.initialized = true

	var err error
	for _, opt := range f.options {
		f, err = opt(f)
		if err != nil {
			return err
		}
	}
	if len(f.overloads) == 0 {
		return fmt.Errorf("function %s must have at least one overload", f.name)
	}
	return nil
}

// bindings produces a set of function bindings, if any are defined.
func (f *functionDecl) bindings() ([]*functions.Overload, error) {
	overloads := []*functions.Overload{}
	nonStrict := false
	for _, o := range f.overloads {
		if o.hasBinding() {
			overload := &functions.Overload{
				Operator:     o.id,
				Unary:        o.guardedUnaryOp(f.name),
				Binary:       o.guardedBinaryOp(f.name),
				Function:     o.guardedFunctionOp(f.name),
				OperandTrait: o.operandTrait,
				NonStrict:    o.nonStrict,
			}
			overloads = append(overloads, overload)
			nonStrict = nonStrict || o.nonStrict
		}
	}
	if f.singleton != nil {
		if len(overloads) != 0 {
			return nil, fmt.Errorf("singleton function incompatible with specialized overloads: %s", f.name)
		}
		return []*functions.Overload{
			{
				Operator:     f.name,
				Unary:        f.singleton.Unary,
				Binary:       f.singleton.Binary,
				Function:     f.singleton.Function,
				OperandTrait: f.singleton.OperandTrait,
			},
		}, nil
	}
	if len(overloads) == 0 {
		return overloads, nil
	}
	// Single overload. Replicate an entry for it using the function name as well.
	if len(overloads) == 1 {
		if overloads[0].Operator == f.name {
			return overloads, nil
		}
		return append(overloads, &functions.Overload{
			Operator:     f.name,
			Unary:        overloads[0].Unary,
			Binary:       overloads[0].Binary,
			Function:     overloads[0].Function,
			NonStrict:    overloads[0].NonStrict,
			OperandTrait: overloads[0].OperandTrait,
		}), nil
	}
	// All of the defined overloads are wrapped into a top-level function which
	// performs dynamic dispatch to the proper overload based on the argument types.
	bindings := append([]*functions.Overload{}, overloads...)
	funcDispatch := func(args ...ref.Val) ref.Val {
		for _, o := range f.overloads {
			if !o.matchesRuntimeSignature(args...) {
				continue
			}
			switch len(args) {
			case 1:
				if o.unaryOp != nil {
					return o.unaryOp(args[0])
				}
			case 2:
				if o.binaryOp != nil {
					return o.binaryOp(args[0], args[1])
				}
			}
			if o.functionOp != nil {
				return o.functionOp(args...)
			}
			// eventually this will fall through to the noSuchOverload below.
		}
		return noSuchOverload(f.name, args...)
	}
	function := &functions.Overload{
		Operator:  f.name,
		Function:  funcDispatch,
		NonStrict: nonStrict,
	}
	return append(bindings, function), nil
}

// merge one function declaration with another.
//
// If a function is extended, by say adding new overloads to an existing function, then it is merged with the
// prior definition of the function at which point its overloads must not collide with pre-existing overloads
// and its bindings (singleton, or per-overload) must not conflict with previous definitions either.
func (f *functionDecl) merge(other *functionDecl) (*functionDecl, error) {
	if f.name != other.name {
		return nil, fmt.Errorf("cannot merge unrelated functions. %s and %s", f.name, other.name)
	}
	err := f.init()
	if err != nil {
		return nil, err
	}
	err = other.init()
	if err != nil {
		return nil, err
	}
	merged := &functionDecl{
		name:        f.name,
		overloads:   make([]*overloadDecl, len(f.overloads)),
		options:     []FunctionOpt{},
		initialized: true,
		singleton:   f.singleton,
	}
	copy(merged.overloads, f.overloads)
	for _, o := range other.overloads {
		err := merged.addOverload(o)
		if err != nil {
			return nil, fmt.Errorf("function declaration merge failed: %v", err)
		}
	}
	if other.singleton != nil {
		if merged.singleton != nil {
			return nil, fmt.Errorf("function already has a binding: %s", f.name)
		}
		merged.singleton = other.singleton
	}
	return merged, nil
}

// addOverload ensures that the new overload does not collide with an existing overload signature;
// however, if the function signatures are identical, the implementation may be rewritten as its
// difficult to compare functions by object identity.
func (f *functionDecl) addOverload(overload *overloadDecl) error {
	for index, o := range f.overloads {
		if o.id != overload.id && o.signatureOverlaps(overload) {
			return fmt.Errorf("overload signature collision in function %s: %s collides with %s", f.name, o.id, overload.id)
		}
		if o.id == overload.id {
			if o.signatureEquals(overload) && o.nonStrict == overload.nonStrict {
				// Allow redefinition of an overload implementation so long as the signatures match.
				f.overloads[index] = overload
				return nil
			}
			return fmt.Errorf("overload redefinition in function. %s: %s has multiple definitions", f.name, o.id)
		}
	}
	f.overloads = append(f.overloads, overload)
	return nil
}

func noSuchOverload(funcName string, args ...ref.Val) ref.Val {
	argTypes := make([]string, len(args))
	for i, arg := range args {
		argTypes[i] = arg.Type().TypeName()
	}
	signature := strings.Join(argTypes, ", ")
	return types.NewErr("no such overload: %s(%s)", funcName, signature)
}

// overloadDecl contains all of the relevant information regarding a specific function overload.
type overloadDecl struct {
	id             string
	argTypes       []*Type
	resultType     *Type
	memberFunction bool

	// binding options, optional but encouraged.
	unaryOp    functions.UnaryOp
	binaryOp   functions.BinaryOp
	functionOp functions.FunctionOp

	// behavioral options, uncommon
	nonStrict    bool
	operandTrait int
}

func (o *overloadDecl) hasBinding() bool {
	return o.unaryOp != nil || o.binaryOp != nil || o.functionOp != nil
}

// guardedUnaryOp creates an invocation guard around the provided unary operator, if one is defined.
func (o *overloadDecl) guardedUnaryOp(funcName string) functions.UnaryOp {
	if o.unaryOp == nil {
		return nil
	}
	return func(arg ref.Val) ref.Val {
		if !o.matchesRuntimeUnarySignature(arg) {
			return noSuchOverload(funcName, arg)
		}
		return o.unaryOp(arg)
	}
}

// guardedBinaryOp creates an invocation guard around the provided binary operator, if one is defined.
func (o *overloadDecl) guardedBinaryOp(funcName string) functions.BinaryOp {
	if o.binaryOp == nil {
		return nil
	}
	return func(arg1, arg2 ref.Val) ref.Val {
		if !o.matchesRuntimeBinarySignature(arg1, arg2) {
			return noSuchOverload(funcName, arg1, arg2)
		}
		return o.binaryOp(arg1, arg2)
	}
}

// guardedFunctionOp creates an invocation guard around the provided variadic function binding, if one is provided.
func (o *overloadDecl) guardedFunctionOp(funcName string) functions.FunctionOp {
	if o.functionOp == nil {
		return nil
	}
	return func(args ...ref.Val) ref.Val {
		if !o.matchesRuntimeSignature(args...) {
			return noSuchOverload(funcName, args...)
		}
		return o.functionOp(args...)
	}
}

// matchesRuntimeUnarySignature indicates whether the argument type is runtime assiganble to the overload's expected argument.
func (o *overloadDecl) matchesRuntimeUnarySignature(arg ref.Val) bool {
	if o.nonStrict && types.IsUnknownOrError(arg) {
		return true
	}
	return o.argTypes[0].IsAssignableRuntimeType(arg) && (o.operandTrait == 0 || arg.Type().HasTrait(o.operandTrait))
}

// matchesRuntimeBinarySignature indicates whether the argument types are runtime assiganble to the overload's expected arguments.
func (o *overloadDecl) matchesRuntimeBinarySignature(arg1, arg2 ref.Val) bool {
	if o.nonStrict {
		if types.IsUnknownOrError(arg1) {
			return types.IsUnknownOrError(arg2) || o.argTypes[1].IsAssignableRuntimeType(arg2)
		}
	} else if !o.argTypes[1].IsAssignableRuntimeType(arg2) {
		return false
	}
	return o.argTypes[0].IsAssignableRuntimeType(arg1) && (o.operandTrait == 0 || arg1.Type().HasTrait(o.operandTrait))
}

// matchesRuntimeSignature indicates whether the argument types are runtime assiganble to the overload's expected arguments.
func (o *overloadDecl) matchesRuntimeSignature(args ...ref.Val) bool {
	if len(args) != len(o.argTypes) {
		return false
	}
	if len(args) == 0 {
		return true
	}
	allArgsMatch := true
	for i, arg := range args {
		if o.nonStrict && types.IsUnknownOrError(arg) {
			continue
		}
		allArgsMatch = allArgsMatch && o.argTypes[i].IsAssignableRuntimeType(arg)
	}

	arg := args[0]
	return allArgsMatch && (o.operandTrait == 0 || (o.nonStrict && types.IsUnknownOrError(arg)) || arg.Type().HasTrait(o.operandTrait))
}

// signatureEquals indicates whether one overload has an identical signature to another overload.
//
// Providing a duplicate signature is not an issue, but an overloapping signature is problematic.
func (o *overloadDecl) signatureEquals(other *overloadDecl) bool {
	if o.id != other.id || o.memberFunction != other.memberFunction || len(o.argTypes) != len(other.argTypes) {
		return false
	}
	for i, at := range o.argTypes {
		oat := other.argTypes[i]
		if !at.equals(oat) {
			return false
		}
	}
	return o.resultType.equals(other.resultType)
}

// signatureOverlaps indicates whether one overload has an overlapping signature with another overload.
//
// The 'other' overload must first be checked for equality before determining whether it overlaps in order to be completely accurate.
func (o *overloadDecl) signatureOverlaps(other *overloadDecl) bool {
	if o.memberFunction != other.memberFunction || len(o.argTypes) != len(other.argTypes) {
		return false
	}
	argsOverlap := true
	for i, argType := range o.argTypes {
		otherArgType := other.argTypes[i]
		argsOverlap = argsOverlap &&
			(argType.IsAssignableType(otherArgType) ||
				otherArgType.IsAssignableType(argType))
	}
	return argsOverlap
}

func newOverload(overloadID string, memberFunction bool, args []*Type, resultType *Type, opts ...OverloadOpt) FunctionOpt {
	return func(f *functionDecl) (*functionDecl, error) {
		overload := &overloadDecl{
			id:             overloadID,
			argTypes:       args,
			resultType:     resultType,
			memberFunction: memberFunction,
		}
		var err error
		for _, opt := range opts {
			overload, err = opt(overload)
			if err != nil {
				return nil, err
			}
		}
		err = f.addOverload(overload)
		if err != nil {
			return nil, err
		}
		return f, nil
	}
}

func maybeWrapper(t *Type, pbType *exprpb.Type) *exprpb.Type {
	if t.IsAssignableType(NullType) {
		return decls.NewWrapperType(pbType)
	}
	return pbType
}

// TypeToExprType converts a CEL-native type representation to a protobuf CEL Type representation.
func TypeToExprType(t *Type) (*exprpb.Type, error) {
	switch t.kind {
	case AnyKind:
		return decls.Any, nil
	case BoolKind:
		return maybeWrapper(t, decls.Bool), nil
	case BytesKind:
		return maybeWrapper(t, decls.Bytes), nil
	case DoubleKind:
		return maybeWrapper(t, decls.Double), nil
	case DurationKind:
		return decls.Duration, nil
	case DynKind:
		return decls.Dyn, nil
	case IntKind:
		return maybeWrapper(t, decls.Int), nil
	case ListKind:
		et, err := TypeToExprType(t.parameters[0])
		if err != nil {
			return nil, err
		}
		return decls.NewListType(et), nil
	case MapKind:
		kt, err := TypeToExprType(t.parameters[0])
		if err != nil {
			return nil, err
		}
		vt, err := TypeToExprType(t.parameters[1])
		if err != nil {
			return nil, err
		}
		return decls.NewMapType(kt, vt), nil
	case NullTypeKind:
		return decls.Null, nil
	case OpaqueKind:
		params := make([]*exprpb.Type, len(t.parameters))
		for i, p := range t.parameters {
			pt, err := TypeToExprType(p)
			if err != nil {
				return nil, err
			}
			params[i] = pt
		}
		return decls.NewAbstractType(t.runtimeType.TypeName(), params...), nil
	case StringKind:
		return maybeWrapper(t, decls.String), nil
	case StructKind:
		switch t.runtimeType.TypeName() {
		case "google.protobuf.Any":
			return decls.Any, nil
		case "google.protobuf.Duration":
			return decls.Duration, nil
		case "google.protobuf.Timestamp":
			return decls.Timestamp, nil
		case "google.protobuf.Value":
			return decls.Dyn, nil
		case "google.protobuf.ListValue":
			return decls.NewListType(decls.Dyn), nil
		case "google.protobuf.Struct":
			return decls.NewMapType(decls.String, decls.Dyn), nil
		case "google.protobuf.BoolValue":
			return decls.NewWrapperType(decls.Bool), nil
		case "google.protobuf.BytesValue":
			return decls.NewWrapperType(decls.Bytes), nil
		case "google.protobuf.DoubleValue", "google.protobuf.FloatValue":
			return decls.NewWrapperType(decls.Double), nil
		case "google.protobuf.Int32Value", "google.protobuf.Int64Value":
			return decls.NewWrapperType(decls.Int), nil
		case "google.protobuf.StringValue":
			return decls.NewWrapperType(decls.String), nil
		case "google.protobuf.UInt32Value", "google.protobuf.UInt64Value":
			return decls.NewWrapperType(decls.Uint), nil
		default:
			return decls.NewObjectType(t.runtimeType.TypeName()), nil
		}
	case TimestampKind:
		return decls.Timestamp, nil
	case TypeParamKind:
		return decls.NewTypeParamType(t.runtimeType.TypeName()), nil
	case TypeKind:
		return decls.NewTypeType(decls.Dyn), nil
	case UintKind:
		return maybeWrapper(t, decls.Uint), nil
	}
	return nil, fmt.Errorf("missing type conversion to proto: %v", t)
}

// ExprTypeToType converts a protobuf CEL type representation to a CEL-native type representation.
func ExprTypeToType(t *exprpb.Type) (*Type, error) {
	switch t.GetTypeKind().(type) {
	case *exprpb.Type_Dyn:
		return DynType, nil
	case *exprpb.Type_AbstractType_:
		paramTypes := make([]*Type, len(t.GetAbstractType().GetParameterTypes()))
		for i, p := range t.GetAbstractType().GetParameterTypes() {
			pt, err := ExprTypeToType(p)
			if err != nil {
				return nil, err
			}
			paramTypes[i] = pt
		}
		return OpaqueType(t.GetAbstractType().GetName(), paramTypes...), nil
	case *exprpb.Type_ListType_:
		et, err := ExprTypeToType(t.GetListType().GetElemType())
		if err != nil {
			return nil, err
		}
		return ListType(et), nil
	case *exprpb.Type_MapType_:
		kt, err := ExprTypeToType(t.GetMapType().GetKeyType())
		if err != nil {
			return nil, err
		}
		vt, err := ExprTypeToType(t.GetMapType().GetValueType())
		if err != nil {
			return nil, err
		}
		return MapType(kt, vt), nil
	case *exprpb.Type_MessageType:
		switch t.GetMessageType() {
		case "google.protobuf.Any":
			return AnyType, nil
		case "google.protobuf.Duration":
			return DurationType, nil
		case "google.protobuf.Timestamp":
			return TimestampType, nil
		case "google.protobuf.Value":
			return DynType, nil
		case "google.protobuf.ListValue":
			return ListType(DynType), nil
		case "google.protobuf.Struct":
			return MapType(StringType, DynType), nil
		case "google.protobuf.BoolValue":
			return NullableType(BoolType), nil
		case "google.protobuf.BytesValue":
			return NullableType(BytesType), nil
		case "google.protobuf.DoubleValue", "google.protobuf.FloatValue":
			return NullableType(DoubleType), nil
		case "google.protobuf.Int32Value", "google.protobuf.Int64Value":
			return NullableType(IntType), nil
		case "google.protobuf.StringValue":
			return NullableType(StringType), nil
		case "google.protobuf.UInt32Value", "google.protobuf.UInt64Value":
			return NullableType(UintType), nil
		default:
			return ObjectType(t.GetMessageType()), nil
		}
	case *exprpb.Type_Null:
		return NullType, nil
	case *exprpb.Type_Primitive:
		switch t.GetPrimitive() {
		case exprpb.Type_BOOL:
			return BoolType, nil
		case exprpb.Type_BYTES:
			return BytesType, nil
		case exprpb.Type_DOUBLE:
			return DoubleType, nil
		case exprpb.Type_INT64:
			return IntType, nil
		case exprpb.Type_STRING:
			return StringType, nil
		case exprpb.Type_UINT64:
			return UintType, nil
		default:
			return nil, fmt.Errorf("unsupported primitive type: %v", t)
		}
	case *exprpb.Type_TypeParam:
		return TypeParamType(t.GetTypeParam()), nil
	case *exprpb.Type_Type:
		return TypeType, nil
	case *exprpb.Type_WellKnown:
		switch t.GetWellKnown() {
		case exprpb.Type_ANY:
			return AnyType, nil
		case exprpb.Type_DURATION:
			return DurationType, nil
		case exprpb.Type_TIMESTAMP:
			return TimestampType, nil
		default:
			return nil, fmt.Errorf("unsupported well-known type: %v", t)
		}
	case *exprpb.Type_Wrapper:
		t, err := ExprTypeToType(&exprpb.Type{TypeKind: &exprpb.Type_Primitive{Primitive: t.GetWrapper()}})
		if err != nil {
			return nil, err
		}
		return NullableType(t), nil
	default:
		return nil, fmt.Errorf("unsupported type: %v", t)
	}
}

// ExprDeclToDeclaration converts a protobuf CEL declaration to a CEL-native declaration, either a Variable or Function.
func ExprDeclToDeclaration(d *exprpb.Decl) (EnvOption, error) {
	switch d.GetDeclKind().(type) {
	case *exprpb.Decl_Function:
		overloads := d.GetFunction().GetOverloads()
		opts := make([]FunctionOpt, len(overloads))
		for i, o := range overloads {
			args := make([]*Type, len(o.GetParams()))
			for j, p := range o.GetParams() {
				a, err := ExprTypeToType(p)
				if err != nil {
					return nil, err
				}
				args[j] = a
			}
			res, err := ExprTypeToType(o.GetResultType())
			if err != nil {
				return nil, err
			}
			opts[i] = Overload(o.GetOverloadId(), args, res)
		}
		return Function(d.GetName(), opts...), nil
	case *exprpb.Decl_Ident:
		t, err := ExprTypeToType(d.GetIdent().GetType())
		if err != nil {
			return nil, err
		}
		return Variable(d.GetName(), t), nil
	default:
		return nil, fmt.Errorf("unsupported decl: %v", d)
	}

}

func functionDeclToExprDecl(f *functionDecl) (*exprpb.Decl, error) {
	overloads := make([]*exprpb.Decl_FunctionDecl_Overload, len(f.overloads))
	i := 0
	for _, o := range f.overloads {
		paramNames := map[string]struct{}{}
		argTypes := make([]*exprpb.Type, len(o.argTypes))
		for j, a := range o.argTypes {
			collectParamNames(paramNames, a)
			at, err := TypeToExprType(a)
			if err != nil {
				return nil, err
			}
			argTypes[j] = at
		}
		collectParamNames(paramNames, o.resultType)
		resultType, err := TypeToExprType(o.resultType)
		if err != nil {
			return nil, err
		}
		if len(paramNames) == 0 {
			if o.memberFunction {
				overloads[i] = decls.NewInstanceOverload(o.id, argTypes, resultType)
			} else {
				overloads[i] = decls.NewOverload(o.id, argTypes, resultType)
			}
		} else {
			params := []string{}
			for pn := range paramNames {
				params = append(params, pn)
			}
			if o.memberFunction {
				overloads[i] = decls.NewParameterizedInstanceOverload(o.id, argTypes, resultType, params)
			} else {
				overloads[i] = decls.NewParameterizedOverload(o.id, argTypes, resultType, params)
			}
		}
		i++
	}
	return decls.NewFunction(f.name, overloads...), nil
}

func collectParamNames(paramNames map[string]struct{}, arg *Type) {
	if arg.kind == TypeParamKind {
		paramNames[arg.runtimeType.TypeName()] = struct{}{}
	}
	for _, param := range arg.parameters {
		collectParamNames(paramNames, param)
	}
}

func typeValueToKind(tv *types.TypeValue) (Kind, error) {
	switch tv {
	case types.BoolType:
		return BoolKind, nil
	case types.DoubleType:
		return DoubleKind, nil
	case types.IntType:
		return IntKind, nil
	case types.UintType:
		return UintKind, nil
	case types.ListType:
		return ListKind, nil
	case types.MapType:
		return MapKind, nil
	case types.StringType:
		return StringKind, nil
	case types.BytesType:
		return BytesKind, nil
	case types.DurationType:
		return DurationKind, nil
	case types.TimestampType:
		return TimestampKind, nil
	case types.NullType:
		return NullTypeKind, nil
	case types.TypeType:
		return TypeKind, nil
	default:
		switch tv.TypeName() {
		case "dyn":
			return DynKind, nil
		case "google.protobuf.Any":
			return AnyKind, nil
		case "optional":
			return OpaqueKind, nil
		default:
			return 0, fmt.Errorf("no known conversion for type of %s", tv.TypeName())
		}
	}
}
