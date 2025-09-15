// Copyright 2019 Google LLC
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

package interpreter

import (
	"fmt"
	"sync"

	"github.com/google/cel-go/common/functions"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

// Interpretable can accept a given Activation and produce a value along with
// an accompanying EvalState which can be used to inspect whether additional
// data might be necessary to complete the evaluation.
type Interpretable interface {
	// ID value corresponding to the expression node.
	ID() int64

	// Eval an Activation to produce an output.
	Eval(activation Activation) ref.Val
}

// InterpretableConst interface for tracking whether the Interpretable is a constant value.
type InterpretableConst interface {
	Interpretable

	// Value returns the constant value of the instruction.
	Value() ref.Val
}

// InterpretableAttribute interface for tracking whether the Interpretable is an attribute.
type InterpretableAttribute interface {
	Interpretable

	// Attr returns the Attribute value.
	Attr() Attribute

	// Adapter returns the type adapter to be used for adapting resolved Attribute values.
	Adapter() types.Adapter

	// AddQualifier proxies the Attribute.AddQualifier method.
	//
	// Note, this method may mutate the current attribute state. If the desire is to clone the
	// Attribute, the Attribute should first be copied before adding the qualifier. Attributes
	// are not copyable by default, so this is a capable that would need to be added to the
	// AttributeFactory or specifically to the underlying Attribute implementation.
	AddQualifier(Qualifier) (Attribute, error)

	// Qualify replicates the Attribute.Qualify method to permit extension and interception
	// of object qualification.
	Qualify(vars Activation, obj any) (any, error)

	// QualifyIfPresent qualifies the object if the qualifier is declared or defined on the object.
	// The 'presenceOnly' flag indicates that the value is not necessary, just a boolean status as
	// to whether the qualifier is present.
	QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error)

	// IsOptional indicates whether the resulting value is an optional type.
	IsOptional() bool

	// Resolve returns the value of the Attribute given the current Activation.
	Resolve(Activation) (any, error)
}

// InterpretableCall interface for inspecting Interpretable instructions related to function calls.
type InterpretableCall interface {
	Interpretable

	// Function returns the function name as it appears in text or mangled operator name as it
	// appears in the operators.go file.
	Function() string

	// OverloadID returns the overload id associated with the function specialization.
	// Overload ids are stable across language boundaries and can be treated as synonymous with a
	// unique function signature.
	OverloadID() string

	// Args returns the normalized arguments to the function overload.
	// For receiver-style functions, the receiver target is arg 0.
	Args() []Interpretable
}

// InterpretableConstructor interface for inspecting Interpretable instructions that initialize a list, map
// or struct.
type InterpretableConstructor interface {
	Interpretable

	// InitVals returns all the list elements, map key and values or struct field values.
	InitVals() []Interpretable

	// Type returns the type constructed.
	Type() ref.Type
}

// ObservableInterpretable is an Interpretable which supports stateful observation, such as tracing
// or cost-tracking.
type ObservableInterpretable struct {
	Interpretable
	observers []StatefulObserver
}

// ID implements the Interpretable method to get the expression id associated with the step.
func (oi *ObservableInterpretable) ID() int64 {
	return oi.Interpretable.ID()
}

// Eval proxies to the ObserveEval method while invoking a no-op callback to report the observations.
func (oi *ObservableInterpretable) Eval(vars Activation) ref.Val {
	return oi.ObserveEval(vars, func(any) {})
}

// ObserveEval evaluates an interpretable and performs per-evaluation state-tracking.
//
// This method is concurrency safe and the expectation is that the observer function will use
// a switch statement to determine the type of the state which has been reported back from the call.
func (oi *ObservableInterpretable) ObserveEval(vars Activation, observer func(any)) ref.Val {
	var err error
	// Initialize the state needed for the observers to function.
	for _, obs := range oi.observers {
		vars, err = obs.InitState(vars)
		if err != nil {
			return types.WrapErr(err)
		}
		// Provide an initial reference to the state to ensure state is available
		// even in cases of interrupting errors generated during evaluation.
		observer(obs.GetState(vars))
	}
	result := oi.Interpretable.Eval(vars)
	// Get the state which needs to be reported back as having been observed.
	for _, obs := range oi.observers {
		observer(obs.GetState(vars))
	}
	return result
}

// Core Interpretable implementations used during the program planning phase.

type evalTestOnly struct {
	id int64
	InterpretableAttribute
}

// ID implements the Interpretable interface method.
func (test *evalTestOnly) ID() int64 {
	return test.id
}

// Eval implements the Interpretable interface method.
func (test *evalTestOnly) Eval(ctx Activation) ref.Val {
	val, err := test.Resolve(ctx)
	// Return an error if the resolve step fails
	if err != nil {
		return types.LabelErrNode(test.id, types.WrapErr(err))
	}
	if optVal, isOpt := val.(*types.Optional); isOpt {
		return types.Bool(optVal.HasValue())
	}
	return test.Adapter().NativeToValue(val)
}

// AddQualifier appends a qualifier that will always and only perform a presence test.
func (test *evalTestOnly) AddQualifier(q Qualifier) (Attribute, error) {
	cq, ok := q.(ConstantQualifier)
	if !ok {
		return nil, fmt.Errorf("test only expressions must have constant qualifiers: %v", q)
	}
	return test.InterpretableAttribute.AddQualifier(&testOnlyQualifier{ConstantQualifier: cq})
}

type testOnlyQualifier struct {
	ConstantQualifier
}

// Qualify determines whether the test-only qualifier is present on the input object.
func (q *testOnlyQualifier) Qualify(vars Activation, obj any) (any, error) {
	out, present, err := q.ConstantQualifier.QualifyIfPresent(vars, obj, true)
	if err != nil {
		return nil, err
	}
	if unk, isUnk := out.(types.Unknown); isUnk {
		return unk, nil
	}
	return present, nil
}

// QualifyIfPresent returns whether the target field in the test-only expression is present.
func (q *testOnlyQualifier) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	// Only ever test for presence.
	return q.ConstantQualifier.QualifyIfPresent(vars, obj, true)
}

// QualifierValueEquals determines whether the test-only constant qualifier equals the input value.
func (q *testOnlyQualifier) QualifierValueEquals(value any) bool {
	// The input qualifier will always be of type string
	return q.ConstantQualifier.Value().Value() == value
}

// NewConstValue creates a new constant valued Interpretable.
func NewConstValue(id int64, val ref.Val) InterpretableConst {
	return &evalConst{
		id:  id,
		val: val,
	}
}

type evalConst struct {
	id  int64
	val ref.Val
}

// ID implements the Interpretable interface method.
func (cons *evalConst) ID() int64 {
	return cons.id
}

// Eval implements the Interpretable interface method.
func (cons *evalConst) Eval(ctx Activation) ref.Val {
	return cons.val
}

// Value implements the InterpretableConst interface method.
func (cons *evalConst) Value() ref.Val {
	return cons.val
}

type evalOr struct {
	id    int64
	terms []Interpretable
}

// ID implements the Interpretable interface method.
func (or *evalOr) ID() int64 {
	return or.id
}

// Eval implements the Interpretable interface method.
func (or *evalOr) Eval(ctx Activation) ref.Val {
	var err ref.Val = nil
	var unk *types.Unknown
	for _, term := range or.terms {
		val := term.Eval(ctx)
		boolVal, ok := val.(types.Bool)
		// short-circuit on true.
		if ok && boolVal == types.True {
			return types.True
		}
		if !ok {
			isUnk := false
			unk, isUnk = types.MaybeMergeUnknowns(val, unk)
			if !isUnk && err == nil {
				if types.IsError(val) {
					err = val
				} else {
					err = types.MaybeNoSuchOverloadErr(val)
				}
				err = types.LabelErrNode(or.id, err)
			}
		}
	}
	if unk != nil {
		return unk
	}
	if err != nil {
		return err
	}
	return types.False
}

type evalAnd struct {
	id    int64
	terms []Interpretable
}

// ID implements the Interpretable interface method.
func (and *evalAnd) ID() int64 {
	return and.id
}

// Eval implements the Interpretable interface method.
func (and *evalAnd) Eval(ctx Activation) ref.Val {
	var err ref.Val = nil
	var unk *types.Unknown
	for _, term := range and.terms {
		val := term.Eval(ctx)
		boolVal, ok := val.(types.Bool)
		// short-circuit on false.
		if ok && boolVal == types.False {
			return types.False
		}
		if !ok {
			isUnk := false
			unk, isUnk = types.MaybeMergeUnknowns(val, unk)
			if !isUnk && err == nil {
				if types.IsError(val) {
					err = val
				} else {
					err = types.MaybeNoSuchOverloadErr(val)
				}
				err = types.LabelErrNode(and.id, err)
			}
		}
	}
	if unk != nil {
		return unk
	}
	if err != nil {
		return err
	}
	return types.True
}

type evalEq struct {
	id  int64
	lhs Interpretable
	rhs Interpretable
}

// ID implements the Interpretable interface method.
func (eq *evalEq) ID() int64 {
	return eq.id
}

// Eval implements the Interpretable interface method.
func (eq *evalEq) Eval(ctx Activation) ref.Val {
	lVal := eq.lhs.Eval(ctx)
	rVal := eq.rhs.Eval(ctx)
	if types.IsUnknownOrError(lVal) {
		return lVal
	}
	if types.IsUnknownOrError(rVal) {
		return rVal
	}
	return types.Equal(lVal, rVal)
}

// Function implements the InterpretableCall interface method.
func (*evalEq) Function() string {
	return operators.Equals
}

// OverloadID implements the InterpretableCall interface method.
func (*evalEq) OverloadID() string {
	return overloads.Equals
}

// Args implements the InterpretableCall interface method.
func (eq *evalEq) Args() []Interpretable {
	return []Interpretable{eq.lhs, eq.rhs}
}

type evalNe struct {
	id  int64
	lhs Interpretable
	rhs Interpretable
}

// ID implements the Interpretable interface method.
func (ne *evalNe) ID() int64 {
	return ne.id
}

// Eval implements the Interpretable interface method.
func (ne *evalNe) Eval(ctx Activation) ref.Val {
	lVal := ne.lhs.Eval(ctx)
	rVal := ne.rhs.Eval(ctx)
	if types.IsUnknownOrError(lVal) {
		return lVal
	}
	if types.IsUnknownOrError(rVal) {
		return rVal
	}
	return types.Bool(types.Equal(lVal, rVal) != types.True)
}

// Function implements the InterpretableCall interface method.
func (*evalNe) Function() string {
	return operators.NotEquals
}

// OverloadID implements the InterpretableCall interface method.
func (*evalNe) OverloadID() string {
	return overloads.NotEquals
}

// Args implements the InterpretableCall interface method.
func (ne *evalNe) Args() []Interpretable {
	return []Interpretable{ne.lhs, ne.rhs}
}

type evalZeroArity struct {
	id       int64
	function string
	overload string
	impl     functions.FunctionOp
}

// ID implements the Interpretable interface method.
func (zero *evalZeroArity) ID() int64 {
	return zero.id
}

// Eval implements the Interpretable interface method.
func (zero *evalZeroArity) Eval(ctx Activation) ref.Val {
	return types.LabelErrNode(zero.id, zero.impl())
}

// Function implements the InterpretableCall interface method.
func (zero *evalZeroArity) Function() string {
	return zero.function
}

// OverloadID implements the InterpretableCall interface method.
func (zero *evalZeroArity) OverloadID() string {
	return zero.overload
}

// Args returns the argument to the unary function.
func (zero *evalZeroArity) Args() []Interpretable {
	return []Interpretable{}
}

type evalUnary struct {
	id        int64
	function  string
	overload  string
	arg       Interpretable
	trait     int
	impl      functions.UnaryOp
	nonStrict bool
}

// ID implements the Interpretable interface method.
func (un *evalUnary) ID() int64 {
	return un.id
}

// Eval implements the Interpretable interface method.
func (un *evalUnary) Eval(ctx Activation) ref.Val {
	argVal := un.arg.Eval(ctx)
	// Early return if the argument to the function is unknown or error.
	strict := !un.nonStrict
	if strict && types.IsUnknownOrError(argVal) {
		return argVal
	}
	// If the implementation is bound and the argument value has the right traits required to
	// invoke it, then call the implementation.
	if un.impl != nil && (un.trait == 0 || (!strict && types.IsUnknownOrError(argVal)) || argVal.Type().HasTrait(un.trait)) {
		return types.LabelErrNode(un.id, un.impl(argVal))
	}
	// Otherwise, if the argument is a ReceiverType attempt to invoke the receiver method on the
	// operand (arg0).
	if argVal.Type().HasTrait(traits.ReceiverType) {
		return types.LabelErrNode(un.id, argVal.(traits.Receiver).Receive(un.function, un.overload, []ref.Val{}))
	}
	return types.NewErrWithNodeID(un.id, "no such overload: %s", un.function)
}

// Function implements the InterpretableCall interface method.
func (un *evalUnary) Function() string {
	return un.function
}

// OverloadID implements the InterpretableCall interface method.
func (un *evalUnary) OverloadID() string {
	return un.overload
}

// Args returns the argument to the unary function.
func (un *evalUnary) Args() []Interpretable {
	return []Interpretable{un.arg}
}

type evalBinary struct {
	id        int64
	function  string
	overload  string
	lhs       Interpretable
	rhs       Interpretable
	trait     int
	impl      functions.BinaryOp
	nonStrict bool
}

// ID implements the Interpretable interface method.
func (bin *evalBinary) ID() int64 {
	return bin.id
}

// Eval implements the Interpretable interface method.
func (bin *evalBinary) Eval(ctx Activation) ref.Val {
	lVal := bin.lhs.Eval(ctx)
	rVal := bin.rhs.Eval(ctx)
	// Early return if any argument to the function is unknown or error.
	strict := !bin.nonStrict
	if strict {
		if types.IsUnknownOrError(lVal) {
			return lVal
		}
		if types.IsUnknownOrError(rVal) {
			return rVal
		}
	}
	// If the implementation is bound and the argument value has the right traits required to
	// invoke it, then call the implementation.
	if bin.impl != nil && (bin.trait == 0 || (!strict && types.IsUnknownOrError(lVal)) || lVal.Type().HasTrait(bin.trait)) {
		return types.LabelErrNode(bin.id, bin.impl(lVal, rVal))
	}
	// Otherwise, if the argument is a ReceiverType attempt to invoke the receiver method on the
	// operand (arg0).
	if lVal.Type().HasTrait(traits.ReceiverType) {
		return types.LabelErrNode(bin.id, lVal.(traits.Receiver).Receive(bin.function, bin.overload, []ref.Val{rVal}))
	}
	return types.NewErrWithNodeID(bin.id, "no such overload: %s", bin.function)
}

// Function implements the InterpretableCall interface method.
func (bin *evalBinary) Function() string {
	return bin.function
}

// OverloadID implements the InterpretableCall interface method.
func (bin *evalBinary) OverloadID() string {
	return bin.overload
}

// Args returns the argument to the unary function.
func (bin *evalBinary) Args() []Interpretable {
	return []Interpretable{bin.lhs, bin.rhs}
}

type evalVarArgs struct {
	id        int64
	function  string
	overload  string
	args      []Interpretable
	trait     int
	impl      functions.FunctionOp
	nonStrict bool
}

// NewCall creates a new call Interpretable.
func NewCall(id int64, function, overload string, args []Interpretable, impl functions.FunctionOp) InterpretableCall {
	return &evalVarArgs{
		id:       id,
		function: function,
		overload: overload,
		args:     args,
		impl:     impl,
	}
}

// ID implements the Interpretable interface method.
func (fn *evalVarArgs) ID() int64 {
	return fn.id
}

// Eval implements the Interpretable interface method.
func (fn *evalVarArgs) Eval(ctx Activation) ref.Val {
	argVals := make([]ref.Val, len(fn.args))
	// Early return if any argument to the function is unknown or error.
	strict := !fn.nonStrict
	for i, arg := range fn.args {
		argVals[i] = arg.Eval(ctx)
		if strict && types.IsUnknownOrError(argVals[i]) {
			return argVals[i]
		}
	}
	// If the implementation is bound and the argument value has the right traits required to
	// invoke it, then call the implementation.
	arg0 := argVals[0]
	if fn.impl != nil && (fn.trait == 0 || (!strict && types.IsUnknownOrError(arg0)) || arg0.Type().HasTrait(fn.trait)) {
		return types.LabelErrNode(fn.id, fn.impl(argVals...))
	}
	// Otherwise, if the argument is a ReceiverType attempt to invoke the receiver method on the
	// operand (arg0).
	if arg0.Type().HasTrait(traits.ReceiverType) {
		return types.LabelErrNode(fn.id, arg0.(traits.Receiver).Receive(fn.function, fn.overload, argVals[1:]))
	}
	return types.NewErrWithNodeID(fn.id, "no such overload: %s %d", fn.function, fn.id)
}

// Function implements the InterpretableCall interface method.
func (fn *evalVarArgs) Function() string {
	return fn.function
}

// OverloadID implements the InterpretableCall interface method.
func (fn *evalVarArgs) OverloadID() string {
	return fn.overload
}

// Args returns the argument to the unary function.
func (fn *evalVarArgs) Args() []Interpretable {
	return fn.args
}

type evalList struct {
	id           int64
	elems        []Interpretable
	optionals    []bool
	hasOptionals bool
	adapter      types.Adapter
}

// ID implements the Interpretable interface method.
func (l *evalList) ID() int64 {
	return l.id
}

// Eval implements the Interpretable interface method.
func (l *evalList) Eval(ctx Activation) ref.Val {
	elemVals := make([]ref.Val, 0, len(l.elems))
	// If any argument is unknown or error early terminate.
	for i, elem := range l.elems {
		elemVal := elem.Eval(ctx)
		if types.IsUnknownOrError(elemVal) {
			return elemVal
		}
		if l.hasOptionals && l.optionals[i] {
			optVal, ok := elemVal.(*types.Optional)
			if !ok {
				return types.LabelErrNode(l.id, invalidOptionalElementInit(elemVal))
			}
			if !optVal.HasValue() {
				continue
			}
			elemVal = optVal.GetValue()
		}
		elemVals = append(elemVals, elemVal)
	}
	return l.adapter.NativeToValue(elemVals)
}

func (l *evalList) InitVals() []Interpretable {
	return l.elems
}

func (l *evalList) Type() ref.Type {
	return types.ListType
}

type evalMap struct {
	id           int64
	keys         []Interpretable
	vals         []Interpretable
	optionals    []bool
	hasOptionals bool
	adapter      types.Adapter
}

// ID implements the Interpretable interface method.
func (m *evalMap) ID() int64 {
	return m.id
}

// Eval implements the Interpretable interface method.
func (m *evalMap) Eval(ctx Activation) ref.Val {
	entries := make(map[ref.Val]ref.Val)
	// If any argument is unknown or error early terminate.
	for i, key := range m.keys {
		keyVal := key.Eval(ctx)
		if types.IsUnknownOrError(keyVal) {
			return keyVal
		}
		valVal := m.vals[i].Eval(ctx)
		if types.IsUnknownOrError(valVal) {
			return valVal
		}
		if m.hasOptionals && m.optionals[i] {
			optVal, ok := valVal.(*types.Optional)
			if !ok {
				return types.LabelErrNode(m.id, invalidOptionalEntryInit(keyVal, valVal))
			}
			if !optVal.HasValue() {
				delete(entries, keyVal)
				continue
			}
			valVal = optVal.GetValue()
		}
		entries[keyVal] = valVal
	}
	return m.adapter.NativeToValue(entries)
}

func (m *evalMap) InitVals() []Interpretable {
	if len(m.keys) != len(m.vals) {
		return nil
	}
	result := make([]Interpretable, len(m.keys)+len(m.vals))
	idx := 0
	for i, k := range m.keys {
		v := m.vals[i]
		result[idx] = k
		idx++
		result[idx] = v
		idx++
	}
	return result
}

func (m *evalMap) Type() ref.Type {
	return types.MapType
}

type evalObj struct {
	id           int64
	typeName     string
	fields       []string
	vals         []Interpretable
	optionals    []bool
	hasOptionals bool
	provider     types.Provider
}

// ID implements the Interpretable interface method.
func (o *evalObj) ID() int64 {
	return o.id
}

// Eval implements the Interpretable interface method.
func (o *evalObj) Eval(ctx Activation) ref.Val {
	fieldVals := make(map[string]ref.Val)
	// If any argument is unknown or error early terminate.
	for i, field := range o.fields {
		val := o.vals[i].Eval(ctx)
		if types.IsUnknownOrError(val) {
			return val
		}
		if o.hasOptionals && o.optionals[i] {
			optVal, ok := val.(*types.Optional)
			if !ok {
				return types.LabelErrNode(o.id, invalidOptionalEntryInit(field, val))
			}
			if !optVal.HasValue() {
				delete(fieldVals, field)
				continue
			}
			val = optVal.GetValue()
		}
		fieldVals[field] = val
	}
	return types.LabelErrNode(o.id, o.provider.NewValue(o.typeName, fieldVals))
}

// InitVals implements the InterpretableConstructor interface method.
func (o *evalObj) InitVals() []Interpretable {
	return o.vals
}

// Type implements the InterpretableConstructor interface method.
func (o *evalObj) Type() ref.Type {
	return types.NewObjectType(o.typeName)
}

type evalFold struct {
	id        int64
	accuVar   string
	iterVar   string
	iterVar2  string
	iterRange Interpretable
	accu      Interpretable
	cond      Interpretable
	step      Interpretable
	result    Interpretable
	adapter   types.Adapter

	// note an exhaustive fold will ensure that all branches are evaluated
	// when using mutable values, these branches will mutate the final result
	// rather than make a throw-away computation.
	exhaustive    bool
	interruptable bool
}

// ID implements the Interpretable interface method.
func (fold *evalFold) ID() int64 {
	return fold.id
}

// Eval implements the Interpretable interface method.
func (fold *evalFold) Eval(ctx Activation) ref.Val {
	// Initialize the folder interface
	f := newFolder(fold, ctx)
	defer releaseFolder(f)

	foldRange := fold.iterRange.Eval(ctx)
	if types.IsUnknownOrError(foldRange) {
		return foldRange
	}
	if fold.iterVar2 != "" {
		var foldable traits.Foldable
		switch r := foldRange.(type) {
		case traits.Mapper:
			foldable = types.ToFoldableMap(r)
		case traits.Lister:
			foldable = types.ToFoldableList(r)
		default:
			return types.NewErrWithNodeID(fold.ID(), "unsupported comprehension range type: %T", foldRange)
		}
		foldable.Fold(f)
		return f.evalResult()
	}

	if !foldRange.Type().HasTrait(traits.IterableType) {
		return types.ValOrErr(foldRange, "got '%T', expected iterable type", foldRange)
	}
	iterable := foldRange.(traits.Iterable)
	return f.foldIterable(iterable)
}

// Optional Interpretable implementations that specialize, subsume, or extend the core evaluation
// plan via decorators.

// evalSetMembership is an Interpretable implementation which tests whether an input value
// exists within the set of map keys used to model a set.
type evalSetMembership struct {
	inst     Interpretable
	arg      Interpretable
	valueSet map[ref.Val]ref.Val
}

// ID implements the Interpretable interface method.
func (e *evalSetMembership) ID() int64 {
	return e.inst.ID()
}

// Eval implements the Interpretable interface method.
func (e *evalSetMembership) Eval(ctx Activation) ref.Val {
	val := e.arg.Eval(ctx)
	if types.IsUnknownOrError(val) {
		return val
	}
	if ret, found := e.valueSet[val]; found {
		return ret
	}
	return types.False
}

// evalWatch is an Interpretable implementation that wraps the execution of a given
// expression so that it may observe the computed value and send it to an observer.
type evalWatch struct {
	Interpretable
	observer EvalObserver
}

// Eval implements the Interpretable interface method.
func (e *evalWatch) Eval(vars Activation) ref.Val {
	val := e.Interpretable.Eval(vars)
	e.observer(vars, e.ID(), e.Interpretable, val)
	return val
}

// evalWatchAttr describes a watcher of an InterpretableAttribute Interpretable.
//
// Since the watcher may be selected against at a later stage in program planning, the watcher
// must implement the InterpretableAttribute interface by proxy.
type evalWatchAttr struct {
	InterpretableAttribute
	observer EvalObserver
}

// AddQualifier creates a wrapper over the incoming qualifier which observes the qualification
// result.
func (e *evalWatchAttr) AddQualifier(q Qualifier) (Attribute, error) {
	switch qual := q.(type) {
	// By default, the qualifier is either a constant or an attribute
	// There may be some custom cases where the attribute is neither.
	case ConstantQualifier:
		// Expose a method to test whether the qualifier matches the input pattern.
		q = &evalWatchConstQual{
			ConstantQualifier: qual,
			observer:          e.observer,
			adapter:           e.Adapter(),
		}
	case *evalWatchAttr:
		// Unwrap the evalWatchAttr since the observation will be applied during Qualify or
		// QualifyIfPresent rather than Eval.
		q = &evalWatchAttrQual{
			Attribute: qual.InterpretableAttribute,
			observer:  e.observer,
			adapter:   e.Adapter(),
		}
	case Attribute:
		// Expose methods which intercept the qualification prior to being applied as a qualifier.
		// Using this interface ensures that the qualifier is converted to a constant value one
		// time during attribute pattern matching as the method embeds the Attribute interface
		// needed to trip the conversion to a constant.
		q = &evalWatchAttrQual{
			Attribute: qual,
			observer:  e.observer,
			adapter:   e.Adapter(),
		}
	default:
		// This is likely a custom qualifier type.
		q = &evalWatchQual{
			Qualifier: qual,
			observer:  e.observer,
			adapter:   e.Adapter(),
		}
	}
	_, err := e.InterpretableAttribute.AddQualifier(q)
	return e, err
}

// Eval implements the Interpretable interface method.
func (e *evalWatchAttr) Eval(vars Activation) ref.Val {
	val := e.InterpretableAttribute.Eval(vars)
	e.observer(vars, e.ID(), e.InterpretableAttribute, val)
	return val
}

// evalWatchConstQual observes the qualification of an object using a constant boolean, int,
// string, or uint.
type evalWatchConstQual struct {
	ConstantQualifier
	observer EvalObserver
	adapter  types.Adapter
}

// Qualify observes the qualification of a object via a constant boolean, int, string, or uint.
func (e *evalWatchConstQual) Qualify(vars Activation, obj any) (any, error) {
	out, err := e.ConstantQualifier.Qualify(vars, obj)
	var val ref.Val
	if err != nil {
		val = types.LabelErrNode(e.ID(), types.WrapErr(err))
	} else {
		val = e.adapter.NativeToValue(out)
	}
	e.observer(vars, e.ID(), e.ConstantQualifier, val)
	return out, err
}

// QualifyIfPresent conditionally qualifies the variable and only records a value if one is present.
func (e *evalWatchConstQual) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	out, present, err := e.ConstantQualifier.QualifyIfPresent(vars, obj, presenceOnly)
	var val ref.Val
	if err != nil {
		val = types.LabelErrNode(e.ID(), types.WrapErr(err))
	} else if out != nil {
		val = e.adapter.NativeToValue(out)
	} else if presenceOnly {
		val = types.Bool(present)
	}
	if present || presenceOnly {
		e.observer(vars, e.ID(), e.ConstantQualifier, val)
	}
	return out, present, err
}

// QualifierValueEquals tests whether the incoming value is equal to the qualifying constant.
func (e *evalWatchConstQual) QualifierValueEquals(value any) bool {
	qve, ok := e.ConstantQualifier.(qualifierValueEquator)
	return ok && qve.QualifierValueEquals(value)
}

// evalWatchAttrQual observes the qualification of an object by a value computed at runtime.
type evalWatchAttrQual struct {
	Attribute
	observer EvalObserver
	adapter  ref.TypeAdapter
}

// Qualify observes the qualification of a object via a value computed at runtime.
func (e *evalWatchAttrQual) Qualify(vars Activation, obj any) (any, error) {
	out, err := e.Attribute.Qualify(vars, obj)
	var val ref.Val
	if err != nil {
		val = types.LabelErrNode(e.ID(), types.WrapErr(err))
	} else {
		val = e.adapter.NativeToValue(out)
	}
	e.observer(vars, e.ID(), e.Attribute, val)
	return out, err
}

// QualifyIfPresent conditionally qualifies the variable and only records a value if one is present.
func (e *evalWatchAttrQual) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	out, present, err := e.Attribute.QualifyIfPresent(vars, obj, presenceOnly)
	var val ref.Val
	if err != nil {
		val = types.LabelErrNode(e.ID(), types.WrapErr(err))
	} else if out != nil {
		val = e.adapter.NativeToValue(out)
	} else if presenceOnly {
		val = types.Bool(present)
	}
	if present || presenceOnly {
		e.observer(vars, e.ID(), e.Attribute, val)
	}
	return out, present, err
}

// evalWatchQual observes the qualification of an object by a value computed at runtime.
type evalWatchQual struct {
	Qualifier
	observer EvalObserver
	adapter  types.Adapter
}

// Qualify observes the qualification of a object via a value computed at runtime.
func (e *evalWatchQual) Qualify(vars Activation, obj any) (any, error) {
	out, err := e.Qualifier.Qualify(vars, obj)
	var val ref.Val
	if err != nil {
		val = types.LabelErrNode(e.ID(), types.WrapErr(err))
	} else {
		val = e.adapter.NativeToValue(out)
	}
	e.observer(vars, e.ID(), e.Qualifier, val)
	return out, err
}

// QualifyIfPresent conditionally qualifies the variable and only records a value if one is present.
func (e *evalWatchQual) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	out, present, err := e.Qualifier.QualifyIfPresent(vars, obj, presenceOnly)
	var val ref.Val
	if err != nil {
		val = types.LabelErrNode(e.ID(), types.WrapErr(err))
	} else if out != nil {
		val = e.adapter.NativeToValue(out)
	} else if presenceOnly {
		val = types.Bool(present)
	}
	if present || presenceOnly {
		e.observer(vars, e.ID(), e.Qualifier, val)
	}
	return out, present, err
}

// evalWatchConst describes a watcher of an instConst Interpretable.
type evalWatchConst struct {
	InterpretableConst
	observer EvalObserver
}

// Eval implements the Interpretable interface method.
func (e *evalWatchConst) Eval(vars Activation) ref.Val {
	val := e.Value()
	e.observer(vars, e.ID(), e.InterpretableConst, val)
	return val
}

// evalExhaustiveOr is just like evalOr, but does not short-circuit argument evaluation.
type evalExhaustiveOr struct {
	id    int64
	terms []Interpretable
}

// ID implements the Interpretable interface method.
func (or *evalExhaustiveOr) ID() int64 {
	return or.id
}

// Eval implements the Interpretable interface method.
func (or *evalExhaustiveOr) Eval(ctx Activation) ref.Val {
	var err ref.Val = nil
	var unk *types.Unknown
	isTrue := false
	for _, term := range or.terms {
		val := term.Eval(ctx)
		boolVal, ok := val.(types.Bool)
		// flag the result as true
		if ok && boolVal == types.True {
			isTrue = true
		}
		if !ok && !isTrue {
			isUnk := false
			unk, isUnk = types.MaybeMergeUnknowns(val, unk)
			if !isUnk && err == nil {
				if types.IsError(val) {
					err = val
				} else {
					err = types.MaybeNoSuchOverloadErr(val)
				}
			}
		}
	}
	if isTrue {
		return types.True
	}
	if unk != nil {
		return unk
	}
	if err != nil {
		return err
	}
	return types.False
}

// evalExhaustiveAnd is just like evalAnd, but does not short-circuit argument evaluation.
type evalExhaustiveAnd struct {
	id    int64
	terms []Interpretable
}

// ID implements the Interpretable interface method.
func (and *evalExhaustiveAnd) ID() int64 {
	return and.id
}

// Eval implements the Interpretable interface method.
func (and *evalExhaustiveAnd) Eval(ctx Activation) ref.Val {
	var err ref.Val = nil
	var unk *types.Unknown
	isFalse := false
	for _, term := range and.terms {
		val := term.Eval(ctx)
		boolVal, ok := val.(types.Bool)
		// short-circuit on false.
		if ok && boolVal == types.False {
			isFalse = true
		}
		if !ok && !isFalse {
			isUnk := false
			unk, isUnk = types.MaybeMergeUnknowns(val, unk)
			if !isUnk && err == nil {
				if types.IsError(val) {
					err = val
				} else {
					err = types.MaybeNoSuchOverloadErr(val)
				}
			}
		}
	}
	if isFalse {
		return types.False
	}
	if unk != nil {
		return unk
	}
	if err != nil {
		return err
	}
	return types.True
}

// evalExhaustiveConditional is like evalConditional, but does not short-circuit argument
// evaluation.
type evalExhaustiveConditional struct {
	id      int64
	adapter types.Adapter
	attr    *conditionalAttribute
}

// ID implements the Interpretable interface method.
func (cond *evalExhaustiveConditional) ID() int64 {
	return cond.id
}

// Eval implements the Interpretable interface method.
func (cond *evalExhaustiveConditional) Eval(ctx Activation) ref.Val {
	cVal := cond.attr.expr.Eval(ctx)
	tVal, tErr := cond.attr.truthy.Resolve(ctx)
	fVal, fErr := cond.attr.falsy.Resolve(ctx)
	cBool, ok := cVal.(types.Bool)
	if !ok {
		return types.ValOrErr(cVal, "no such overload")
	}
	if cBool {
		if tErr != nil {
			return types.LabelErrNode(cond.id, types.WrapErr(tErr))
		}
		return cond.adapter.NativeToValue(tVal)
	}
	if fErr != nil {
		return types.LabelErrNode(cond.id, types.WrapErr(fErr))
	}
	return cond.adapter.NativeToValue(fVal)
}

// evalAttr evaluates an Attribute value.
type evalAttr struct {
	adapter  types.Adapter
	attr     Attribute
	optional bool
}

var _ InterpretableAttribute = &evalAttr{}

// ID of the attribute instruction.
func (a *evalAttr) ID() int64 {
	return a.attr.ID()
}

// AddQualifier implements the InterpretableAttribute interface method.
func (a *evalAttr) AddQualifier(qual Qualifier) (Attribute, error) {
	attr, err := a.attr.AddQualifier(qual)
	a.attr = attr
	return attr, err
}

// Attr implements the InterpretableAttribute interface method.
func (a *evalAttr) Attr() Attribute {
	return a.attr
}

// Adapter implements the InterpretableAttribute interface method.
func (a *evalAttr) Adapter() types.Adapter {
	return a.adapter
}

// Eval implements the Interpretable interface method.
func (a *evalAttr) Eval(ctx Activation) ref.Val {
	v, err := a.attr.Resolve(ctx)
	if err != nil {
		return types.LabelErrNode(a.ID(), types.WrapErr(err))
	}
	return a.adapter.NativeToValue(v)
}

// Qualify proxies to the Attribute's Qualify method.
func (a *evalAttr) Qualify(vars Activation, obj any) (any, error) {
	return a.attr.Qualify(vars, obj)
}

// QualifyIfPresent proxies to the Attribute's QualifyIfPresent method.
func (a *evalAttr) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	return a.attr.QualifyIfPresent(vars, obj, presenceOnly)
}

func (a *evalAttr) IsOptional() bool {
	return a.optional
}

// Resolve proxies to the Attribute's Resolve method.
func (a *evalAttr) Resolve(ctx Activation) (any, error) {
	return a.attr.Resolve(ctx)
}

type evalWatchConstructor struct {
	constructor InterpretableConstructor
	observer    EvalObserver
}

// InitVals implements the InterpretableConstructor InitVals function.
func (c *evalWatchConstructor) InitVals() []Interpretable {
	return c.constructor.InitVals()
}

// Type implements the InterpretableConstructor Type function.
func (c *evalWatchConstructor) Type() ref.Type {
	return c.constructor.Type()
}

// ID implements the Interpretable ID function.
func (c *evalWatchConstructor) ID() int64 {
	return c.constructor.ID()
}

// Eval implements the Interpretable Eval function.
func (c *evalWatchConstructor) Eval(vars Activation) ref.Val {
	val := c.constructor.Eval(vars)
	c.observer(vars, c.ID(), c.constructor, val)
	return val
}

func invalidOptionalEntryInit(field any, value ref.Val) ref.Val {
	return types.NewErr("cannot initialize optional entry '%v' from non-optional value %v", field, value)
}

func invalidOptionalElementInit(value ref.Val) ref.Val {
	return types.NewErr("cannot initialize optional list element from non-optional value %v", value)
}

// newFolder creates or initializes a pooled folder instance.
func newFolder(eval *evalFold, ctx Activation) *folder {
	f := folderPool.Get().(*folder)
	f.evalFold = eval
	f.activation = ctx
	return f
}

// releaseFolder resets and releases a pooled folder instance.
func releaseFolder(f *folder) {
	f.reset()
	folderPool.Put(f)
}

// folder tracks the state associated with folding a list or map with a comprehension v2 style macro.
//
// The folder embeds an interpreter.Activation and Interpretable evalFold value as well as implements
// the traits.Folder interface methods.
//
// Instances of a folder are intended to be pooled to minimize allocation overhead with this temporary
// bookkeeping object which supports lazy evaluation of the accumulator init expression which is useful
// in preserving evaluation order semantics which might otherwise be disrupted through the use of
// cel.bind or cel.@block.
type folder struct {
	*evalFold
	activation Activation

	// fold state objects.
	accuVal     ref.Val
	iterVar1Val any
	iterVar2Val any

	// bookkeeping flags to modify Activation and fold behaviors.
	initialized   bool
	mutableValue  bool
	interrupted   bool
	computeResult bool
}

func (f *folder) foldIterable(iterable traits.Iterable) ref.Val {
	it := iterable.Iterator()
	for it.HasNext() == types.True {
		f.iterVar1Val = it.Next()

		cond := f.cond.Eval(f)
		condBool, ok := cond.(types.Bool)
		if f.interrupted || (!f.exhaustive && ok && condBool != types.True) {
			return f.evalResult()
		}

		// Update the accumulation value and check for eval interuption.
		f.accuVal = f.step.Eval(f)
		f.initialized = true
		if f.interruptable && checkInterrupt(f.activation) {
			f.interrupted = true
			return f.evalResult()
		}
	}
	return f.evalResult()
}

// FoldEntry will either fold comprehension v1 style macros if iterVar2 is unset, or comprehension v2 style
// macros if both the iterVar and iterVar2 are set to non-empty strings.
func (f *folder) FoldEntry(key, val any) bool {
	// Default to referencing both values.
	f.iterVar1Val = key
	f.iterVar2Val = val

	// Terminate evaluation if evaluation is interrupted or the condition is not true and exhaustive
	// eval is not enabled.
	cond := f.cond.Eval(f)
	condBool, ok := cond.(types.Bool)
	if f.interrupted || (!f.exhaustive && ok && condBool != types.True) {
		return false
	}

	// Update the accumulation value and check for eval interuption.
	f.accuVal = f.step.Eval(f)
	f.initialized = true
	if f.interruptable && checkInterrupt(f.activation) {
		f.interrupted = true
		return false
	}
	return true
}

// ResolveName overrides the default Activation lookup to perform lazy initialization of the accumulator
// and specialized lookups of iteration values with consideration for whether the final result is being
// computed and the iteration variables should be ignored.
func (f *folder) ResolveName(name string) (any, bool) {
	if name == f.accuVar {
		if !f.initialized {
			f.initialized = true
			initVal := f.accu.Eval(f.activation)
			if !f.exhaustive {
				if l, isList := initVal.(traits.Lister); isList && l.Size() == types.IntZero {
					initVal = types.NewMutableList(f.adapter)
					f.mutableValue = true
				}
				if m, isMap := initVal.(traits.Mapper); isMap && m.Size() == types.IntZero {
					initVal = types.NewMutableMap(f.adapter, map[ref.Val]ref.Val{})
					f.mutableValue = true
				}
			}
			f.accuVal = initVal
		}
		return f.accuVal, true
	}
	if !f.computeResult {
		if name == f.iterVar {
			f.iterVar1Val = f.adapter.NativeToValue(f.iterVar1Val)
			return f.iterVar1Val, true
		}
		if name == f.iterVar2 {
			f.iterVar2Val = f.adapter.NativeToValue(f.iterVar2Val)
			return f.iterVar2Val, true
		}
	}
	return f.activation.ResolveName(name)
}

// Parent returns the activation embedded into the folder.
func (f *folder) Parent() Activation {
	return f.activation
}

// UnknownAttributePatterns implements the PartialActivation interface returning the unknown patterns
// if they were provided to the input activation, or an empty set if the proxied activation is not partial.
func (f *folder) UnknownAttributePatterns() []*AttributePattern {
	if pv, ok := f.activation.(partialActivationConverter); ok {
		if partial, isPartial := pv.AsPartialActivation(); isPartial {
			return partial.UnknownAttributePatterns()
		}
	}
	return []*AttributePattern{}
}

func (f *folder) AsPartialActivation() (PartialActivation, bool) {
	if pv, ok := f.activation.(partialActivationConverter); ok {
		if _, isPartial := pv.AsPartialActivation(); isPartial {
			return f, true
		}
	}
	return nil, false
}

// evalResult computes the final result of the fold after all entries have been folded and accumulated.
func (f *folder) evalResult() ref.Val {
	f.computeResult = true
	if f.interrupted {
		return types.NewErr("operation interrupted")
	}
	res := f.result.Eval(f)
	// Convert a mutable list or map to an immutable one if the comprehension has generated a list or
	// map as a result.
	if !types.IsUnknownOrError(res) && f.mutableValue {
		if _, ok := res.(traits.MutableLister); ok {
			res = res.(traits.MutableLister).ToImmutableList()
		}
		if _, ok := res.(traits.MutableMapper); ok {
			res = res.(traits.MutableMapper).ToImmutableMap()
		}
	}
	return res
}

// reset clears any state associated with folder evaluation.
func (f *folder) reset() {
	f.evalFold = nil
	f.activation = nil
	f.accuVal = nil
	f.iterVar1Val = nil
	f.iterVar2Val = nil

	f.initialized = false
	f.mutableValue = false
	f.interrupted = false
	f.computeResult = false
}

func checkInterrupt(a Activation) bool {
	stop, found := a.ResolveName("#interrupted")
	return found && stop == true
}

var (
	// pool of var folders to reduce allocations during folds.
	folderPool = &sync.Pool{
		New: func() any {
			return &folder{}
		},
	}
)
