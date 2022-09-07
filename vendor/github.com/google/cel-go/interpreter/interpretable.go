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
	"math"

	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/interpreter/functions"
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
	Adapter() ref.TypeAdapter

	// AddQualifier proxies the Attribute.AddQualifier method.
	//
	// Note, this method may mutate the current attribute state. If the desire is to clone the
	// Attribute, the Attribute should first be copied before adding the qualifier. Attributes
	// are not copyable by default, so this is a capable that would need to be added to the
	// AttributeFactory or specifically to the underlying Attribute implementation.
	AddQualifier(Qualifier) (Attribute, error)

	// Qualify replicates the Attribute.Qualify method to permit extension and interception
	// of object qualification.
	Qualify(vars Activation, obj interface{}) (interface{}, error)

	// Resolve returns the value of the Attribute given the current Activation.
	Resolve(Activation) (interface{}, error)
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

// InterpretableConstructor interface for inspecting  Interpretable instructions that initialize a list, map
// or struct.
type InterpretableConstructor interface {
	Interpretable

	// InitVals returns all the list elements, map key and values or struct field values.
	InitVals() []Interpretable

	// Type returns the type constructed.
	Type() ref.Type
}

// Core Interpretable implementations used during the program planning phase.

type evalTestOnly struct {
	id        int64
	op        Interpretable
	field     types.String
	fieldType *ref.FieldType
}

// ID implements the Interpretable interface method.
func (test *evalTestOnly) ID() int64 {
	return test.id
}

// Eval implements the Interpretable interface method.
func (test *evalTestOnly) Eval(ctx Activation) ref.Val {
	// Handle field selection on a proto in the most efficient way possible.
	if test.fieldType != nil {
		opAttr, ok := test.op.(InterpretableAttribute)
		if ok {
			opVal, err := opAttr.Resolve(ctx)
			if err != nil {
				return types.NewErr(err.Error())
			}
			refVal, ok := opVal.(ref.Val)
			if ok {
				opVal = refVal.Value()
			}
			if test.fieldType.IsSet(opVal) {
				return types.True
			}
			return types.False
		}
	}

	obj := test.op.Eval(ctx)
	tester, ok := obj.(traits.FieldTester)
	if ok {
		return tester.IsSet(test.field)
	}
	container, ok := obj.(traits.Container)
	if ok {
		return container.Contains(test.field)
	}
	return types.ValOrErr(obj, "invalid type for field selection.")
}

// Cost provides the heuristic cost of a `has(field)` macro. The cost has at least 1 for determining
// if the field exists, apart from the cost of accessing the field.
func (test *evalTestOnly) Cost() (min, max int64) {
	min, max = estimateCost(test.op)
	min++
	max++
	return
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

// Cost returns zero for a constant valued Interpretable.
func (cons *evalConst) Cost() (min, max int64) {
	return 0, 0
}

// Value implements the InterpretableConst interface method.
func (cons *evalConst) Value() ref.Val {
	return cons.val
}

type evalOr struct {
	id  int64
	lhs Interpretable
	rhs Interpretable
}

// ID implements the Interpretable interface method.
func (or *evalOr) ID() int64 {
	return or.id
}

// Eval implements the Interpretable interface method.
func (or *evalOr) Eval(ctx Activation) ref.Val {
	// short-circuit lhs.
	lVal := or.lhs.Eval(ctx)
	lBool, lok := lVal.(types.Bool)
	if lok && lBool == types.True {
		return types.True
	}
	// short-circuit on rhs.
	rVal := or.rhs.Eval(ctx)
	rBool, rok := rVal.(types.Bool)
	if rok && rBool == types.True {
		return types.True
	}
	// return if both sides are bool false.
	if lok && rok {
		return types.False
	}
	// TODO: return both values as a set if both are unknown or error.
	// prefer left unknown to right unknown.
	if types.IsUnknown(lVal) {
		return lVal
	}
	if types.IsUnknown(rVal) {
		return rVal
	}
	// If the left-hand side is non-boolean return it as the error.
	if types.IsError(lVal) {
		return lVal
	}
	return types.ValOrErr(rVal, "no such overload")
}

// Cost implements the Coster interface method. The minimum possible cost incurs when the left-hand
// side expr is sufficient in determining the evaluation result.
func (or *evalOr) Cost() (min, max int64) {
	return calShortCircuitBinaryOpsCost(or.lhs, or.rhs)
}

type evalAnd struct {
	id  int64
	lhs Interpretable
	rhs Interpretable
}

// ID implements the Interpretable interface method.
func (and *evalAnd) ID() int64 {
	return and.id
}

// Eval implements the Interpretable interface method.
func (and *evalAnd) Eval(ctx Activation) ref.Val {
	// short-circuit lhs.
	lVal := and.lhs.Eval(ctx)
	lBool, lok := lVal.(types.Bool)
	if lok && lBool == types.False {
		return types.False
	}
	// short-circuit on rhs.
	rVal := and.rhs.Eval(ctx)
	rBool, rok := rVal.(types.Bool)
	if rok && rBool == types.False {
		return types.False
	}
	// return if both sides are bool true.
	if lok && rok {
		return types.True
	}
	// TODO: return both values as a set if both are unknown or error.
	// prefer left unknown to right unknown.
	if types.IsUnknown(lVal) {
		return lVal
	}
	if types.IsUnknown(rVal) {
		return rVal
	}
	// If the left-hand side is non-boolean return it as the error.
	if types.IsError(lVal) {
		return lVal
	}
	return types.ValOrErr(rVal, "no such overload")
}

// Cost implements the Coster interface method. The minimum possible cost incurs when the left-hand
// side expr is sufficient in determining the evaluation result.
func (and *evalAnd) Cost() (min, max int64) {
	return calShortCircuitBinaryOpsCost(and.lhs, and.rhs)
}

func calShortCircuitBinaryOpsCost(lhs, rhs Interpretable) (min, max int64) {
	lMin, lMax := estimateCost(lhs)
	_, rMax := estimateCost(rhs)
	return lMin, lMax + rMax + 1
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

// Cost implements the Coster interface method.
func (eq *evalEq) Cost() (min, max int64) {
	return calExhaustiveBinaryOpsCost(eq.lhs, eq.rhs)
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

// Cost implements the Coster interface method.
func (ne *evalNe) Cost() (min, max int64) {
	return calExhaustiveBinaryOpsCost(ne.lhs, ne.rhs)
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
	return zero.impl()
}

// Cost returns 1 representing the heuristic cost of the function.
func (zero *evalZeroArity) Cost() (min, max int64) {
	return 1, 1
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
		return un.impl(argVal)
	}
	// Otherwise, if the argument is a ReceiverType attempt to invoke the receiver method on the
	// operand (arg0).
	if argVal.Type().HasTrait(traits.ReceiverType) {
		return argVal.(traits.Receiver).Receive(un.function, un.overload, []ref.Val{})
	}
	return types.NewErr("no such overload: %s", un.function)
}

// Cost implements the Coster interface method.
func (un *evalUnary) Cost() (min, max int64) {
	min, max = estimateCost(un.arg)
	min++ // add cost for function
	max++
	return
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
		return bin.impl(lVal, rVal)
	}
	// Otherwise, if the argument is a ReceiverType attempt to invoke the receiver method on the
	// operand (arg0).
	if lVal.Type().HasTrait(traits.ReceiverType) {
		return lVal.(traits.Receiver).Receive(bin.function, bin.overload, []ref.Val{rVal})
	}
	return types.NewErr("no such overload: %s", bin.function)
}

// Cost implements the Coster interface method.
func (bin *evalBinary) Cost() (min, max int64) {
	return calExhaustiveBinaryOpsCost(bin.lhs, bin.rhs)
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
		return fn.impl(argVals...)
	}
	// Otherwise, if the argument is a ReceiverType attempt to invoke the receiver method on the
	// operand (arg0).
	if arg0.Type().HasTrait(traits.ReceiverType) {
		return arg0.(traits.Receiver).Receive(fn.function, fn.overload, argVals[1:])
	}
	return types.NewErr("no such overload: %s", fn.function)
}

// Cost implements the Coster interface method.
func (fn *evalVarArgs) Cost() (min, max int64) {
	min, max = sumOfCost(fn.args)
	min++ // add cost for function
	max++
	return
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
	id      int64
	elems   []Interpretable
	adapter ref.TypeAdapter
}

// ID implements the Interpretable interface method.
func (l *evalList) ID() int64 {
	return l.id
}

// Eval implements the Interpretable interface method.
func (l *evalList) Eval(ctx Activation) ref.Val {
	elemVals := make([]ref.Val, len(l.elems))
	// If any argument is unknown or error early terminate.
	for i, elem := range l.elems {
		elemVal := elem.Eval(ctx)
		if types.IsUnknownOrError(elemVal) {
			return elemVal
		}
		elemVals[i] = elemVal
	}
	return l.adapter.NativeToValue(elemVals)
}

func (l *evalList) InitVals() []Interpretable {
	return l.elems
}

func (l *evalList) Type() ref.Type {
	return types.ListType
}

// Cost implements the Coster interface method.
func (l *evalList) Cost() (min, max int64) {
	return sumOfCost(l.elems)
}

type evalMap struct {
	id      int64
	keys    []Interpretable
	vals    []Interpretable
	adapter ref.TypeAdapter
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

// Cost implements the Coster interface method.
func (m *evalMap) Cost() (min, max int64) {
	kMin, kMax := sumOfCost(m.keys)
	vMin, vMax := sumOfCost(m.vals)
	return kMin + vMin, kMax + vMax
}

type evalObj struct {
	id       int64
	typeName string
	fields   []string
	vals     []Interpretable
	provider ref.TypeProvider
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
		fieldVals[field] = val
	}
	return o.provider.NewValue(o.typeName, fieldVals)
}

func (o *evalObj) InitVals() []Interpretable {
	return o.vals
}

func (o *evalObj) Type() ref.Type {
	return types.NewObjectTypeValue(o.typeName)
}

// Cost implements the Coster interface method.
func (o *evalObj) Cost() (min, max int64) {
	return sumOfCost(o.vals)
}

func sumOfCost(interps []Interpretable) (min, max int64) {
	min, max = 0, 0
	for _, in := range interps {
		minT, maxT := estimateCost(in)
		min += minT
		max += maxT
	}
	return
}

type evalFold struct {
	id            int64
	accuVar       string
	iterVar       string
	iterRange     Interpretable
	accu          Interpretable
	cond          Interpretable
	step          Interpretable
	result        Interpretable
	adapter       ref.TypeAdapter
	exhaustive    bool
	interruptable bool
}

// ID implements the Interpretable interface method.
func (fold *evalFold) ID() int64 {
	return fold.id
}

// Eval implements the Interpretable interface method.
func (fold *evalFold) Eval(ctx Activation) ref.Val {
	foldRange := fold.iterRange.Eval(ctx)
	if !foldRange.Type().HasTrait(traits.IterableType) {
		return types.ValOrErr(foldRange, "got '%T', expected iterable type", foldRange)
	}
	// Configure the fold activation with the accumulator initial value.
	accuCtx := varActivationPool.Get().(*varActivation)
	accuCtx.parent = ctx
	accuCtx.name = fold.accuVar
	accuCtx.val = fold.accu.Eval(ctx)
	// If the accumulator starts as an empty list, then the comprehension will build a list
	// so create a mutable list to optimize the cost of the inner loop.
	l, ok := accuCtx.val.(traits.Lister)
	buildingList := false
	if !fold.exhaustive && ok && l.Size() == types.IntZero {
		buildingList = true
		accuCtx.val = types.NewMutableList(fold.adapter)
	}
	iterCtx := varActivationPool.Get().(*varActivation)
	iterCtx.parent = accuCtx
	iterCtx.name = fold.iterVar

	interrupted := false
	it := foldRange.(traits.Iterable).Iterator()
	for it.HasNext() == types.True {
		// Modify the iter var in the fold activation.
		iterCtx.val = it.Next()

		// Evaluate the condition, terminate the loop if false.
		cond := fold.cond.Eval(iterCtx)
		condBool, ok := cond.(types.Bool)
		if !fold.exhaustive && ok && condBool != types.True {
			break
		}
		// Evaluate the evaluation step into accu var.
		accuCtx.val = fold.step.Eval(iterCtx)
		if fold.interruptable {
			if stop, found := ctx.ResolveName("#interrupted"); found && stop == true {
				interrupted = true
				break
			}
		}
	}
	varActivationPool.Put(iterCtx)
	if interrupted {
		varActivationPool.Put(accuCtx)
		return types.NewErr("operation interrupted")
	}

	// Compute the result.
	res := fold.result.Eval(accuCtx)
	varActivationPool.Put(accuCtx)
	// Convert a mutable list to an immutable one, if the comprehension has generated a list as a result.
	if !types.IsUnknownOrError(res) && buildingList {
		if _, ok := res.(traits.MutableLister); ok {
			res = res.(traits.MutableLister).ToImmutableList()
		}
	}
	return res
}

// Cost implements the Coster interface method.
func (fold *evalFold) Cost() (min, max int64) {
	// Compute the cost for evaluating iterRange.
	iMin, iMax := estimateCost(fold.iterRange)

	// Compute the size of iterRange. If the size depends on the input, return the maximum possible
	// cost range.
	foldRange := fold.iterRange.Eval(EmptyActivation())
	if !foldRange.Type().HasTrait(traits.IterableType) {
		return 0, math.MaxInt64
	}
	var rangeCnt int64
	it := foldRange.(traits.Iterable).Iterator()
	for it.HasNext() == types.True {
		it.Next()
		rangeCnt++
	}
	aMin, aMax := estimateCost(fold.accu)
	cMin, cMax := estimateCost(fold.cond)
	sMin, sMax := estimateCost(fold.step)
	rMin, rMax := estimateCost(fold.result)
	if fold.exhaustive {
		cMin = cMin * rangeCnt
		sMin = sMin * rangeCnt
	}

	// The cond and step costs are multiplied by size(iterRange). The minimum possible cost incurs
	// when the evaluation result can be determined by the first iteration.
	return iMin + aMin + cMin + sMin + rMin,
		iMax + aMax + cMax*rangeCnt + sMax*rangeCnt + rMax
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
	if ret, found := e.valueSet[val]; found {
		return ret
	}
	return types.False
}

// Cost implements the Coster interface method.
func (e *evalSetMembership) Cost() (min, max int64) {
	return estimateCost(e.arg)
}

// evalWatch is an Interpretable implementation that wraps the execution of a given
// expression so that it may observe the computed value and send it to an observer.
type evalWatch struct {
	Interpretable
	observer EvalObserver
}

// Eval implements the Interpretable interface method.
func (e *evalWatch) Eval(ctx Activation) ref.Val {
	val := e.Interpretable.Eval(ctx)
	e.observer(e.ID(), e.Interpretable, val)
	return val
}

// Cost implements the Coster interface method.
func (e *evalWatch) Cost() (min, max int64) {
	return estimateCost(e.Interpretable)
}

// evalWatchAttr describes a watcher of an instAttr Interpretable.
//
// Since the watcher may be selected against at a later stage in program planning, the watcher
// must implement the instAttr interface by proxy.
type evalWatchAttr struct {
	InterpretableAttribute
	observer EvalObserver
}

// AddQualifier creates a wrapper over the incoming qualifier which observes the qualification
// result.
func (e *evalWatchAttr) AddQualifier(q Qualifier) (Attribute, error) {
	cq, isConst := q.(ConstantQualifier)
	if isConst {
		q = &evalWatchConstQual{
			ConstantQualifier: cq,
			observer:          e.observer,
			adapter:           e.InterpretableAttribute.Adapter(),
		}
	} else {
		q = &evalWatchQual{
			Qualifier: q,
			observer:  e.observer,
			adapter:   e.InterpretableAttribute.Adapter(),
		}
	}
	_, err := e.InterpretableAttribute.AddQualifier(q)
	return e, err
}

// Cost implements the Coster interface method.
func (e *evalWatchAttr) Cost() (min, max int64) {
	return estimateCost(e.InterpretableAttribute)
}

// Eval implements the Interpretable interface method.
func (e *evalWatchAttr) Eval(vars Activation) ref.Val {
	val := e.InterpretableAttribute.Eval(vars)
	e.observer(e.ID(), e.InterpretableAttribute, val)
	return val
}

// evalWatchConstQual observes the qualification of an object using a constant boolean, int,
// string, or uint.
type evalWatchConstQual struct {
	ConstantQualifier
	observer EvalObserver
	adapter  ref.TypeAdapter
}

// Cost implements the Coster interface method.
func (e *evalWatchConstQual) Cost() (min, max int64) {
	return estimateCost(e.ConstantQualifier)
}

// Qualify observes the qualification of a object via a constant boolean, int, string, or uint.
func (e *evalWatchConstQual) Qualify(vars Activation, obj interface{}) (interface{}, error) {
	out, err := e.ConstantQualifier.Qualify(vars, obj)
	var val ref.Val
	if err != nil {
		val = types.NewErr(err.Error())
	} else {
		val = e.adapter.NativeToValue(out)
	}
	e.observer(e.ID(), e.ConstantQualifier, val)
	return out, err
}

// QualifierValueEquals tests whether the incoming value is equal to the qualifying constant.
func (e *evalWatchConstQual) QualifierValueEquals(value interface{}) bool {
	qve, ok := e.ConstantQualifier.(qualifierValueEquator)
	return ok && qve.QualifierValueEquals(value)
}

// evalWatchQual observes the qualification of an object by a value computed at runtime.
type evalWatchQual struct {
	Qualifier
	observer EvalObserver
	adapter  ref.TypeAdapter
}

// Cost implements the Coster interface method.
func (e *evalWatchQual) Cost() (min, max int64) {
	return estimateCost(e.Qualifier)
}

// Qualify observes the qualification of a object via a value computed at runtime.
func (e *evalWatchQual) Qualify(vars Activation, obj interface{}) (interface{}, error) {
	out, err := e.Qualifier.Qualify(vars, obj)
	var val ref.Val
	if err != nil {
		val = types.NewErr(err.Error())
	} else {
		val = e.adapter.NativeToValue(out)
	}
	e.observer(e.ID(), e.Qualifier, val)
	return out, err
}

// evalWatchConst describes a watcher of an instConst Interpretable.
type evalWatchConst struct {
	InterpretableConst
	observer EvalObserver
}

// Eval implements the Interpretable interface method.
func (e *evalWatchConst) Eval(vars Activation) ref.Val {
	val := e.Value()
	e.observer(e.ID(), e.InterpretableConst, val)
	return val
}

// Cost implements the Coster interface method.
func (e *evalWatchConst) Cost() (min, max int64) {
	return estimateCost(e.InterpretableConst)
}

// evalExhaustiveOr is just like evalOr, but does not short-circuit argument evaluation.
type evalExhaustiveOr struct {
	id  int64
	lhs Interpretable
	rhs Interpretable
}

// ID implements the Interpretable interface method.
func (or *evalExhaustiveOr) ID() int64 {
	return or.id
}

// Eval implements the Interpretable interface method.
func (or *evalExhaustiveOr) Eval(ctx Activation) ref.Val {
	lVal := or.lhs.Eval(ctx)
	rVal := or.rhs.Eval(ctx)
	lBool, lok := lVal.(types.Bool)
	if lok && lBool == types.True {
		return types.True
	}
	rBool, rok := rVal.(types.Bool)
	if rok && rBool == types.True {
		return types.True
	}
	if lok && rok {
		return types.False
	}
	if types.IsUnknown(lVal) {
		return lVal
	}
	if types.IsUnknown(rVal) {
		return rVal
	}
	// TODO: Combine the errors into a set in the future.
	// If the left-hand side is non-boolean return it as the error.
	if types.IsError(lVal) {
		return lVal
	}
	return types.ValOrErr(rVal, "no such overload")
}

// Cost implements the Coster interface method.
func (or *evalExhaustiveOr) Cost() (min, max int64) {
	return calExhaustiveBinaryOpsCost(or.lhs, or.rhs)
}

// evalExhaustiveAnd is just like evalAnd, but does not short-circuit argument evaluation.
type evalExhaustiveAnd struct {
	id  int64
	lhs Interpretable
	rhs Interpretable
}

// ID implements the Interpretable interface method.
func (and *evalExhaustiveAnd) ID() int64 {
	return and.id
}

// Eval implements the Interpretable interface method.
func (and *evalExhaustiveAnd) Eval(ctx Activation) ref.Val {
	lVal := and.lhs.Eval(ctx)
	rVal := and.rhs.Eval(ctx)
	lBool, lok := lVal.(types.Bool)
	if lok && lBool == types.False {
		return types.False
	}
	rBool, rok := rVal.(types.Bool)
	if rok && rBool == types.False {
		return types.False
	}
	if lok && rok {
		return types.True
	}
	if types.IsUnknown(lVal) {
		return lVal
	}
	if types.IsUnknown(rVal) {
		return rVal
	}
	// TODO: Combine the errors into a set in the future.
	// If the left-hand side is non-boolean return it as the error.
	if types.IsError(lVal) {
		return lVal
	}
	return types.ValOrErr(rVal, "no such overload")
}

// Cost implements the Coster interface method.
func (and *evalExhaustiveAnd) Cost() (min, max int64) {
	return calExhaustiveBinaryOpsCost(and.lhs, and.rhs)
}

func calExhaustiveBinaryOpsCost(lhs, rhs Interpretable) (min, max int64) {
	lMin, lMax := estimateCost(lhs)
	rMin, rMax := estimateCost(rhs)
	return lMin + rMin + 1, lMax + rMax + 1
}

// evalExhaustiveConditional is like evalConditional, but does not short-circuit argument
// evaluation.
type evalExhaustiveConditional struct {
	id      int64
	adapter ref.TypeAdapter
	attr    *conditionalAttribute
}

// ID implements the Interpretable interface method.
func (cond *evalExhaustiveConditional) ID() int64 {
	return cond.id
}

// Eval implements the Interpretable interface method.
func (cond *evalExhaustiveConditional) Eval(ctx Activation) ref.Val {
	cVal := cond.attr.expr.Eval(ctx)
	tVal, err := cond.attr.truthy.Resolve(ctx)
	if err != nil {
		return types.NewErr(err.Error())
	}
	fVal, err := cond.attr.falsy.Resolve(ctx)
	if err != nil {
		return types.NewErr(err.Error())
	}
	cBool, ok := cVal.(types.Bool)
	if !ok {
		return types.ValOrErr(cVal, "no such overload")
	}
	if cBool {
		return cond.adapter.NativeToValue(tVal)
	}
	return cond.adapter.NativeToValue(fVal)
}

// Cost implements the Coster interface method.
func (cond *evalExhaustiveConditional) Cost() (min, max int64) {
	return cond.attr.Cost()
}

// evalAttr evaluates an Attribute value.
type evalAttr struct {
	adapter ref.TypeAdapter
	attr    Attribute
}

// ID of the attribute instruction.
func (a *evalAttr) ID() int64 {
	return a.attr.ID()
}

// AddQualifier implements the instAttr interface method.
func (a *evalAttr) AddQualifier(qual Qualifier) (Attribute, error) {
	attr, err := a.attr.AddQualifier(qual)
	a.attr = attr
	return attr, err
}

// Attr implements the instAttr interface method.
func (a *evalAttr) Attr() Attribute {
	return a.attr
}

// Adapter implements the instAttr interface method.
func (a *evalAttr) Adapter() ref.TypeAdapter {
	return a.adapter
}

// Cost implements the Coster interface method.
func (a *evalAttr) Cost() (min, max int64) {
	return estimateCost(a.attr)
}

// Eval implements the Interpretable interface method.
func (a *evalAttr) Eval(ctx Activation) ref.Val {
	v, err := a.attr.Resolve(ctx)
	if err != nil {
		return types.NewErr(err.Error())
	}
	return a.adapter.NativeToValue(v)
}

// Qualify proxies to the Attribute's Qualify method.
func (a *evalAttr) Qualify(ctx Activation, obj interface{}) (interface{}, error) {
	return a.attr.Qualify(ctx, obj)
}

// Resolve proxies to the Attribute's Resolve method.
func (a *evalAttr) Resolve(ctx Activation) (interface{}, error) {
	return a.attr.Resolve(ctx)
}
