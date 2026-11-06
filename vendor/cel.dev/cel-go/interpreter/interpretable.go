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

	"cel.dev/cel-go/common/functions"
	"cel.dev/cel-go/common/operators"
	"cel.dev/cel-go/common/overloads"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
	"cel.dev/cel-go/common/types/traits"
)

// Interpretable evaluates an Activation and produces a value.
type Interpretable interface {
	// ID value corresponding to the expression node.
	ID() int64

	// Eval evaluates an Activation and produces an output.
	Eval(activation Activation) ref.Val
}

// InterpretableV2 evaluates an ExecutionFrame and produces a value.
//
// The ExecutionFrame should not be stored and should always be passed as the first
// argument to any function as it behaves like Golang's context.Context.
type InterpretableV2 interface {
	Interpretable

	// Exec evaluates the expression within the given ExecutionFrame.
	Exec(frame *ExecutionFrame) ref.Val
}

// adaptToV2 adapts a V1 Interpretable implementation to the V2 interface.
//
// This adapter is used to bridge the legacy Interpretable interface to the
// modern InterpretableV2 interface, providing a shim that allows the use of
// both interfaces in the same system.
func adaptToV2(i Interpretable) InterpretableV2 {
	switch v := i.(type) {
	case InterpretableV2:
		return v
	default:
		return &v1Adapter{Interpretable: v}
	}
}

// v1Adapter handles bridging a V1 Interpretable implementation to the V2 interface.
type v1Adapter struct {
	Interpretable
}

// Exec implements the InterpretableV2 interface method.
func (a *v1Adapter) Exec(f *ExecutionFrame) ref.Val {
	return a.Eval(f)
}

// InterpretableConst interface for tracking whether the Interpretable is a constant value.
type InterpretableConst interface {
	InterpretableV2

	// Value returns the constant value of the instruction.
	Value() ref.Val
}

// InterpretableAttribute interface for tracking whether the Interpretable is an attribute.
type InterpretableAttribute interface {
	InterpretableV2

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
	InterpretableV2

	// Function returns the function name as it appears in text or mangled operator name as it
	// appears in the operators.go file.
	Function() string

	// OverloadID returns the overload id associated with the function specialization.
	// Overload ids are stable across language boundaries and can be treated as synonymous with a
	// unique function signature.
	OverloadID() string

	// Args returns the normalized arguments to the function overload.
	// For receiver-style functions, the receiver target is arg 0.
	Args() []InterpretableV2
}

// InterpretableConstructor interface for inspecting Interpretable instructions that initialize a list, map
// or struct.
type InterpretableConstructor interface {
	InterpretableV2

	// InitVals returns all the list elements, map key and values or struct field values.
	InitVals() []InterpretableV2

	// Type returns the type constructed.
	Type() ref.Type
}

// ObservableInterpretable is an Interpretable which supports stateful observation, such as tracing
// or cost-tracking.
type ObservableInterpretable struct {
	InterpretableV2
	observers []StatefulObserver
}

// ID implements the Interpretable method to get the expression id associated with the step.
func (oi *ObservableInterpretable) ID() int64 {
	return oi.InterpretableV2.ID()
}

// Exec implements the InterpretableV2 interface method.
func (oi *ObservableInterpretable) Exec(frame *ExecutionFrame) ref.Val {
	return oi.ObserveExec(frame, func(any) {})
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
	frame, ok := vars.(*ExecutionFrame)
	if !ok {
		var err error
		frame, err = NewExecutionFrame(vars)
		if err != nil {
			return types.NewErr("invalid activation: %v", err)
		}
		defer frame.Close()
	}
	return oi.ObserveExec(frame, observer)
}

// ObserveExec evaluates an interpretable and performs per-evaluation state-tracking.
//
// This method is concurrency safe and the expectation is that the observer function will use
// a switch statement to determine the type of the state which has been reported back from the call.
func (oi *ObservableInterpretable) ObserveExec(frame *ExecutionFrame, observer func(any)) ref.Val {
	// Initialize the state needed for the observers to function.
	for _, obs := range oi.observers {
		state, err := obs.InitState(frame)
		if err != nil {
			return types.WrapErr(err)
		}
		// Provide an initial reference to the state to ensure state is available
		// even in cases of interrupting errors generated during evaluation.
		observer(state)
	}
	result := oi.InterpretableV2.Exec(frame)
	// Get the state which needs to be reported back as having been observed.
	for _, obs := range oi.observers {
		observer(obs.GetState(frame))
	}
	return result
}

// EvalActivation runs an InterpretableV2 Exec method safely with any Activation,
// wrapping it in a pooled ExecutionFrame if needed and releasing it after execution.
func EvalActivation(ctx Activation, exec func(*ExecutionFrame) ref.Val) ref.Val {
	if f, ok := ctx.(*ExecutionFrame); ok {
		return exec(f)
	}
	f, err := NewExecutionFrame(ctx)
	if err != nil {
		return types.NewErr("invalid activation: %v", err)
	}
	res := exec(f)
	f.Close()
	return res
}

// findFrame walks the activation hierarchy via Unwrap and Parent to locate an
// existing ExecutionFrame, if one exists.
func findFrame(a Activation) *ExecutionFrame {
	if wrapper, ok := a.(activationWrapper); ok {
		unwrapped := wrapper.Unwrap()
		if f, ok := unwrapped.(*ExecutionFrame); ok {
			return f
		}
		return findFrame(unwrapped)
	}
	if p := a.Parent(); p != nil {
		if f, ok := p.(*ExecutionFrame); ok {
			return f
		}
		return findFrame(p)
	}
	return nil
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

// Exec implements the InterpretableV2 interface method.
func (test *evalTestOnly) Exec(frame *ExecutionFrame) ref.Val {
	val, err := test.Resolve(frame)
	var res ref.Val
	// Return an error if the resolve step fails
	if err != nil {
		res = types.LabelErrNode(test.id, types.WrapErr(err))
	} else if optVal, isOpt := val.(*types.Optional); isOpt {
		res = types.Bool(optVal.HasValue())
	} else {
		res = test.Adapter().NativeToValue(val)
	}
	if costs := frame.CostTracker(); costs != nil {
		costs.EvalAttribute(test.id, true, res)
	}
	return res
}

// Eval implements the Interpretable interface method.
func (test *evalTestOnly) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, test.Exec)
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
	if unk, isUnk := out.(*types.Unknown); isUnk {
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

// Exec implements the InterpretableV2 interface method.
func (cons *evalConst) Exec(frame *ExecutionFrame) ref.Val {
	return cons.val
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
	terms []InterpretableV2
}

// ID implements the Interpretable interface method.
func (or *evalOr) ID() int64 {
	return or.id
}

// Exec implements the InterpretableV2 interface method.
func (or *evalOr) Exec(frame *ExecutionFrame) ref.Val {
	var err ref.Val = nil
	var unk *types.Unknown
	for _, term := range or.terms {
		val := term.Exec(frame)
		boolVal, ok := val.(types.Bool)
		// short-circuit on true.
		if ok && boolVal == types.True {
			return types.True
		}
		if !ok {
			if unkVal, isUnk := val.(*types.Unknown); isUnk {
				if unk == nil {
					unk = unkVal
				} else {
					unk = types.MergeUnknowns(unk, unkVal)
				}
			} else if err == nil {
				if isError(val) {
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

// Eval implements the Interpretable interface method.
func (or *evalOr) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, or.Exec)
}

type evalAnd struct {
	id    int64
	terms []InterpretableV2
}

// ID implements the Interpretable interface method.
func (and *evalAnd) ID() int64 {
	return and.id
}

// Exec implements the InterpretableV2 interface method.
func (and *evalAnd) Exec(frame *ExecutionFrame) ref.Val {
	var err ref.Val = nil
	var unk *types.Unknown
	for _, term := range and.terms {
		val := term.Exec(frame)
		boolVal, ok := val.(types.Bool)
		// short-circuit on false.
		if ok && boolVal == types.False {
			return types.False
		}
		if !ok {
			if unkVal, isUnk := val.(*types.Unknown); isUnk {
				if unk == nil {
					unk = unkVal
				} else {
					unk = types.MergeUnknowns(unk, unkVal)
				}
			} else if err == nil {
				if isError(val) {
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

// Eval implements the Interpretable interface method.
func (and *evalAnd) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, and.Exec)
}

type evalEq struct {
	id  int64
	lhs InterpretableV2
	rhs InterpretableV2
}

// ID implements the Interpretable interface method.
func (eq *evalEq) ID() int64 {
	return eq.id
}

// Exec implements the InterpretableV2 interface method.
func (eq *evalEq) Exec(frame *ExecutionFrame) ref.Val {
	lVal := eq.lhs.Exec(frame)
	if isError(lVal) {
		// To preserve legacy cost tracking behavior for ==
		// track this cost, but it will be removed in the future.
		if costs := frame.CostTracker(); costs != nil {
			costs.EvalBinary(frame, eq.id, eq, lVal, types.UnknownType, lVal)
		}
		return lVal
	}
	rVal := eq.rhs.Exec(frame)
	if isError(rVal) {
		// To preserve legacy cost tracking behavior for ==,
		// track this cost, but it will be removed in the future.
		if costs := frame.CostTracker(); costs != nil {
			costs.EvalBinary(frame, eq.id, eq, lVal, rVal, rVal)
		}
		return rVal
	}
	if isUnknown(lVal) || isUnknown(rVal) {
		unk := mergeBinaryUnknowns(lVal, rVal)
		if costs := frame.CostTracker(); costs != nil {
			costs.EvalBinary(frame, eq.id, eq, lVal, rVal, unk)
		}
		return unk
	}
	res := types.Equal(lVal, rVal)
	if costs := frame.CostTracker(); costs != nil {
		costs.EvalBinary(frame, eq.id, eq, lVal, rVal, res)
	}
	return res
}

// Eval implements the Interpretable interface method.
func (eq *evalEq) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, eq.Exec)
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
func (eq *evalEq) Args() []InterpretableV2 {
	return []InterpretableV2{eq.lhs, eq.rhs}
}

type evalNe struct {
	id  int64
	lhs InterpretableV2
	rhs InterpretableV2
}

// ID implements the Interpretable interface method.
func (ne *evalNe) ID() int64 {
	return ne.id
}

// Exec implements the InterpretableV2 interface method.
func (ne *evalNe) Exec(frame *ExecutionFrame) ref.Val {
	lVal := ne.lhs.Exec(frame)
	if isError(lVal) {
		// To preserve legacy cost tracking behavior for !=,
		// track this cost, but it will be removed in the future.
		if costs := frame.CostTracker(); costs != nil {
			costs.EvalBinary(frame, ne.id, ne, lVal, types.UnknownType, lVal)
		}
		return lVal
	}
	rVal := ne.rhs.Exec(frame)
	if isError(rVal) {
		// To preserve legacy cost tracking behavior for !=,
		// track this cost, but it will be removed in the future.
		if costs := frame.CostTracker(); costs != nil {
			costs.EvalBinary(frame, ne.id, ne, lVal, rVal, rVal)
		}
		return rVal
	}
	if isUnknown(lVal) || isUnknown(rVal) {
		unk := mergeBinaryUnknowns(lVal, rVal)
		if costs := frame.CostTracker(); costs != nil {
			costs.EvalBinary(frame, ne.id, ne, lVal, rVal, unk)
		}
		return unk
	}
	res := types.Bool(types.Equal(lVal, rVal) != types.True)
	if costs := frame.CostTracker(); costs != nil {
		costs.EvalBinary(frame, ne.id, ne, lVal, rVal, res)
	}
	return res
}

// Eval implements the Interpretable interface method.
func (ne *evalNe) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, ne.Exec)
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
func (ne *evalNe) Args() []InterpretableV2 {
	return []InterpretableV2{ne.lhs, ne.rhs}
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

// Exec implements the InterpretableV2 interface method.
func (zero *evalZeroArity) Exec(frame *ExecutionFrame) ref.Val {
	res := labelErrNode(zero.id, zero.impl())
	if costs := frame.CostTracker(); costs != nil {
		costs.EvalZeroArity(frame, zero.id, zero, res)
	}
	return res
}

// Eval implements the Interpretable interface method.
func (zero *evalZeroArity) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, zero.Exec)
}

// Function implements the InterpretableCall interface method.
func (zero *evalZeroArity) Function() string {
	return zero.function
}

// OverloadID implements the InterpretableCall interface method.
func (zero *evalZeroArity) OverloadID() string {
	return zero.overload
}

// Args implements the InterpretableCall interface method.
func (zero *evalZeroArity) Args() []InterpretableV2 {
	return []InterpretableV2{}
}

type evalUnary struct {
	id        int64
	function  string
	overload  string
	arg       InterpretableV2
	trait     int
	impl      functions.UnaryOp
	nonStrict bool
}

// ID implements the Interpretable interface method.
func (un *evalUnary) ID() int64 {
	return un.id
}

// Exec implements the InterpretableV2 interface method.
func (un *evalUnary) Exec(frame *ExecutionFrame) ref.Val {
	argVal := un.arg.Exec(frame)
	// Early return if the argument to the function is error or unknown in strict mode.
	strict := !un.nonStrict
	if strict {
		switch v := argVal.(type) {
		case *types.Err:
			// To preserve legacy cost tracking behavior for logical not,
			// this cost is tracked, but will be removed in the future.
			if un.function == operators.LogicalNot {
				if costs := frame.CostTracker(); costs != nil {
					costs.EvalUnary(frame, un.id, un, argVal, argVal)
				}
			}
			return v
		case *types.Unknown:
			if costs := frame.CostTracker(); costs != nil {
				costs.EvalUnary(frame, un.id, un, argVal, argVal)
			}
			return v
		}
	}
	var res ref.Val
	// If the implementation is bound and the argument value has the right traits required to
	// invoke it, then call the implementation.
	if un.impl != nil && (un.trait == 0 || matchOperandTrait(un.trait, strict, argVal)) {
		res = labelErrNode(un.id, un.impl(argVal))
	} else if argVal.Type().HasTrait(traits.ReceiverType) {
		// Otherwise, if the argument is a ReceiverType attempt to invoke the receiver method on the
		// operand (arg0).
		res = labelErrNode(un.id, argVal.(traits.Receiver).Receive(un.function, un.overload, []ref.Val{}))
	} else {
		res = types.NewErrWithNodeID(un.id, "no such overload: %s", un.function)
	}
	if costs := frame.CostTracker(); costs != nil {
		costs.EvalUnary(frame, un.id, un, argVal, res)
	}
	return res
}

// Eval implements the Interpretable interface method.
func (un *evalUnary) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, un.Exec)
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
func (un *evalUnary) Args() []InterpretableV2 {
	return []InterpretableV2{un.arg}
}

type evalBinary struct {
	id        int64
	function  string
	overload  string
	lhs       InterpretableV2
	rhs       InterpretableV2
	trait     int
	impl      functions.BinaryOp
	nonStrict bool
}

// ID implements the Interpretable interface method.
func (bin *evalBinary) ID() int64 {
	return bin.id
}

// Exec implements the InterpretableV2 interface method.
func (bin *evalBinary) Exec(frame *ExecutionFrame) ref.Val {
	lVal := bin.lhs.Exec(frame)
	strict := !bin.nonStrict
	if strict {
		switch lVal.(type) {
		case *types.Err:
			return lVal
		case *types.Unknown:
			rVal := bin.rhs.Exec(frame)
			if unk, ok := rVal.(*types.Unknown); ok {
				res := types.MergeUnknowns(lVal.(*types.Unknown), unk)
				if costs := frame.CostTracker(); costs != nil {
					costs.EvalBinary(frame, bin.id, bin, lVal, rVal, res)
				}
				return res
			}
			if costs := frame.CostTracker(); costs != nil {
				costs.EvalBinary(frame, bin.id, bin, lVal, rVal, lVal)
			}
			return lVal
		}
	}
	rVal := bin.rhs.Exec(frame)
	if strict {
		switch rVal.(type) {
		case *types.Err:
			return rVal
		case *types.Unknown:
			if costs := frame.CostTracker(); costs != nil {
				costs.EvalBinary(frame, bin.id, bin, lVal, rVal, rVal)
			}
			return rVal
		}
	}
	var res ref.Val
	// If the implementation is bound and the argument value has the right traits required to
	// invoke it, then call the implementation.
	if bin.impl != nil && (bin.trait == 0 || matchOperandTrait(bin.trait, strict, lVal)) {
		res = labelErrNode(bin.id, bin.impl(lVal, rVal))
	} else if lVal.Type().HasTrait(traits.ReceiverType) {
		// Otherwise, if the argument is a ReceiverType attempt to invoke the receiver method on the
		// operand (arg0).
		res = labelErrNode(bin.id, lVal.(traits.Receiver).Receive(bin.function, bin.overload, []ref.Val{rVal}))
	} else {
		res = types.NewErrWithNodeID(bin.id, "no such overload: %s", bin.function)
	}
	if costs := frame.CostTracker(); costs != nil {
		costs.EvalBinary(frame, bin.id, bin, lVal, rVal, res)
	}
	return res
}

// Eval implements the Interpretable interface method.
func (bin *evalBinary) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, bin.Exec)
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
func (bin *evalBinary) Args() []InterpretableV2 {
	return []InterpretableV2{bin.lhs, bin.rhs}
}

type evalVarArgs struct {
	id        int64
	function  string
	overload  string
	args      []InterpretableV2
	trait     int
	impl      functions.FunctionOp
	nonStrict bool
}

// NewCall creates a new call Interpretable.
func NewCall(id int64, function, overload string, args []InterpretableV2, impl functions.FunctionOp) InterpretableCall {
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

// Exec implements the InterpretableV2 interface method.
func (fn *evalVarArgs) Exec(frame *ExecutionFrame) ref.Val {
	argVals := make([]ref.Val, len(fn.args))
	strict := !fn.nonStrict
	var unk *types.Unknown
	for i, arg := range fn.args {
		argVal := arg.Exec(frame)
		if strict {
			switch v := argVal.(type) {
			case *types.Err:
				return v
			case *types.Unknown:
				if unk == nil {
					unk = v
				} else {
					unk = types.MergeUnknowns(unk, v)
				}
			}
		}
		argVals[i] = argVal
	}
	if strict && unk != nil {
		if costs := frame.CostTracker(); costs != nil {
			costs.EvalVarArgs(frame, fn.id, fn, argVals, unk)
		}
		return unk
	}
	if len(argVals) == 0 {
		var res ref.Val
		if fn.impl != nil {
			res = labelErrNode(fn.id, fn.impl())
		} else {
			res = types.NewErrWithNodeID(fn.id, "no such overload: %s %d", fn.function, fn.id)
		}
		if costs := frame.CostTracker(); costs != nil {
			costs.EvalZeroArity(frame, fn.id, fn, res)
		}
		return res
	}
	var res ref.Val
	// If the implementation is bound and the argument value has the right traits required to
	// invoke it, then call the implementation.
	arg0 := argVals[0]
	if fn.impl != nil && matchOperandTrait(fn.trait, strict, arg0) {
		res = labelErrNode(fn.id, fn.impl(argVals...))
	} else if arg0.Type().HasTrait(traits.ReceiverType) {
		// Otherwise, if the argument is a ReceiverType attempt to invoke the receiver method on the
		// operand (arg0).
		res = labelErrNode(fn.id, arg0.(traits.Receiver).Receive(fn.function, fn.overload, argVals[1:]))
	} else {
		res = types.NewErrWithNodeID(fn.id, "no such overload: %s %d", fn.function, fn.id)
	}
	if costs := frame.CostTracker(); costs != nil {
		costs.EvalVarArgs(frame, fn.id, fn, argVals, res)
	}
	return res
}

// Eval implements the Interpretable interface method.
func (fn *evalVarArgs) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, fn.Exec)
}

// Function implements the InterpretableCall interface method.
func (fn *evalVarArgs) Function() string {
	return fn.function
}

// OverloadID implements the InterpretableCall interface method.
func (fn *evalVarArgs) OverloadID() string {
	return fn.overload
}

// Args returns the normalized arguments to the function overload.
func (fn *evalVarArgs) Args() []InterpretableV2 {
	return fn.args
}

// evalLateBoundFunc represents a function call where the function implementation is resolved from
// the Activation at evaluation time.
type evalLateBoundFunc struct {
	id        int64
	function  string
	overload  string
	args      []InterpretableV2
	trait     int
	nonStrict bool
	member    bool
	dispatch  functions.LateBoundDispatcher
}

// ID implements the Interpretable interface method.
func (fn *evalLateBoundFunc) ID() int64 {
	return fn.id
}

// Function implements the InterpretableCall interface method.
func (fn *evalLateBoundFunc) Function() string {
	return fn.function
}

// OverloadID implements the InterpretableCall interface method.
func (fn *evalLateBoundFunc) OverloadID() string {
	return fn.overload
}

// Args returns the argument Interpretables for the function call.
func (fn *evalLateBoundFunc) Args() []InterpretableV2 {
	return fn.args
}

// Eval implements the Interpretable interface method.
func (fn *evalLateBoundFunc) Eval(vars Activation) ref.Val {
	return EvalActivation(vars, fn.Exec)
}

// Exec implements the InterpretableV2 interface method.
func (fn *evalLateBoundFunc) Exec(frame *ExecutionFrame) ref.Val {
	argVals := make([]ref.Val, len(fn.args))
	strict := !fn.nonStrict
	var unk *types.Unknown
	for i, arg := range fn.args {
		argVals[i] = arg.Exec(frame)
		if strict {
			if types.IsError(argVals[i]) {
				return argVals[i]
			}
			unk, _ = types.MaybeMergeUnknowns(argVals[i], unk)
		}
	}
	if strict && unk != nil {
		if costs := frame.CostTracker(); costs != nil {
			costs.EvalVarArgs(frame, fn.id, fn, argVals, unk)
		}
		return unk
	}
	var res ref.Val
	// A late-bound function is only invoked when the runtime arguments agree with a declared
	// overload and the activation supplies a compatible implementation. Unlike other call
	// types, there is no fallback to receiver-style dispatch on the operand: the function
	// declaration promised a binding supplied at evaluation time, so a missing or mismatched
	// binding is unambiguously an error.
	if lateFn, found := fn.resolve(frame, argVals, strict); found {
		res = types.LabelErrNode(fn.id, lateFn(argVals...))
	} else {
		res = types.NewErrWithNodeID(fn.id, "no such overload: %s", fn.function)
	}
	switch len(argVals) {
	case 0:
		if costs := frame.CostTracker(); costs != nil {
			costs.EvalZeroArity(frame, fn.id, fn, res)
		}
	case 1:
		if costs := frame.CostTracker(); costs != nil {
			costs.EvalUnary(frame, fn.id, fn, argVals[0], res)
		}
	case 2:
		if costs := frame.CostTracker(); costs != nil {
			costs.EvalBinary(frame, fn.id, fn, argVals[0], argVals[1], res)
		}
	default:
		if costs := frame.CostTracker(); costs != nil {
			costs.EvalVarArgs(frame, fn.id, fn, argVals, res)
		}
	}
	return res
}

// resolve determines which declared overload the runtime arguments satisfy and resolves its
// implementation from the activation.
func (fn *evalLateBoundFunc) resolve(frame *ExecutionFrame, args []ref.Val, strict bool) (functions.FunctionOp, bool) {
	overloadID, matched := fn.matchOverload(args, strict)
	if !matched {
		return nil, false
	}
	impl, found := frame.ResolveFunction(fn.function)
	if !found {
		return nil, false
	}
	return func(args ...ref.Val) ref.Val { return impl(overloadID, args...) }, true
}

// matchOverload reports the id of the declared overload which matches the runtime arguments.
//
// Plans which were not created from a function declaration have no signature to check against,
// in which case the call falls back to the operand trait recorded for the call site.
func (fn *evalLateBoundFunc) matchOverload(args []ref.Val, strict bool) (string, bool) {
	if fn.dispatch != nil {
		return fn.dispatch(fn.member, args...)
	}
	if len(args) == 0 || matchOperandTrait(fn.trait, strict, args[0]) {
		return fn.overload, true
	}
	return "", false
}

// matchOperandTrait returns true if the operand satisfies the trait required by the function
// implementation, or if the implementation does not require a trait.
//
// Non-strict functions are given the chance to handle unknown and error operands themselves.
func matchOperandTrait(trait int, strict bool, operand ref.Val) bool {
	return trait == 0 ||
		(!strict && types.IsUnknownOrError(operand)) ||
		operand.Type().HasTrait(trait)
}

// evalList evaluates a list construction expression.
type evalList struct {
	id           int64
	elems        []InterpretableV2
	optionals    []bool
	hasOptionals bool
	adapter      types.Adapter
}

// ID implements the Interpretable interface method.
func (l *evalList) ID() int64 {
	return l.id
}

// Exec implements the InterpretableV2 interface method.
func (l *evalList) Exec(frame *ExecutionFrame) ref.Val {
	var unk *types.Unknown
	if !l.hasOptionals {
		elemVals := make([]ref.Val, len(l.elems))
		for i, elem := range l.elems {
			elemVal := elem.Exec(frame)
			switch v := elemVal.(type) {
			case *types.Err:
				return v
			case *types.Unknown:
				if unk == nil {
					unk = v
				} else {
					unk = types.MergeUnknowns(unk, v)
				}
			}
			elemVals[i] = elemVal
		}
		if unk != nil {
			trackCostCreateList(frame, l.id, unk)
			return unk
		}
		res := types.NewRefValList(l.adapter, elemVals)
		trackCostCreateList(frame, l.id, res)
		return res
	}

	elemVals := make([]ref.Val, 0, len(l.elems))
	for i, elem := range l.elems {
		elemVal := elem.Exec(frame)
		switch v := elemVal.(type) {
		case *types.Err:
			return v
		case *types.Unknown:
			if unk == nil {
				unk = v
			} else {
				unk = types.MergeUnknowns(unk, v)
			}
		case *types.Optional:
			if l.optionals[i] {
				if !v.HasValue() {
					continue
				}
				elemVal = v.GetValue()
			}
		default:
			if l.optionals[i] {
				return types.LabelErrNode(l.id, invalidOptionalElementInit(elemVal))
			}
		}
		elemVals = append(elemVals, elemVal)
	}
	if unk != nil {
		if costs := frame.CostTracker(); costs != nil {
			costs.CreateList(l.id, unk)
		}
		return unk
	}
	res := types.NewRefValList(l.adapter, elemVals)
	if costs := frame.CostTracker(); costs != nil {
		costs.CreateList(l.id, res)
	}
	return res
}

// Eval implements the Interpretable interface method.
func (l *evalList) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, l.Exec)
}

func (l *evalList) InitVals() []InterpretableV2 {
	return l.elems
}

func (l *evalList) Type() ref.Type {
	return types.ListType
}

type evalMap struct {
	id           int64
	keys         []InterpretableV2
	vals         []InterpretableV2
	optionals    []bool
	hasOptionals bool
	adapter      types.Adapter
}

// ID implements the Interpretable interface method.
func (m *evalMap) ID() int64 {
	return m.id
}

// Exec implements the InterpretableV2 interface method.
func (m *evalMap) Exec(frame *ExecutionFrame) ref.Val {
	entries := make(map[ref.Val]ref.Val, len(m.keys))
	var unk *types.Unknown
	if !m.hasOptionals {
		for i, key := range m.keys {
			keyVal := key.Exec(frame)
			switch v := keyVal.(type) {
			case *types.Err:
				return v
			case *types.Unknown:
				if unk == nil {
					unk = v
				} else {
					unk = types.MergeUnknowns(unk, v)
				}
			}

			valVal := m.vals[i].Exec(frame)
			switch v := valVal.(type) {
			case *types.Err:
				return v
			case *types.Unknown:
				if unk == nil {
					unk = v
				} else {
					unk = types.MergeUnknowns(unk, v)
				}
			}
			entries[keyVal] = valVal
		}
		if unk != nil {
			trackCostCreateMap(frame, m.id, unk)
			return unk
		}
		res := types.NewRefValMap(m.adapter, entries)
		trackCostCreateMap(frame, m.id, res)
		return res
	}

	for i, key := range m.keys {
		keyVal := key.Exec(frame)
		switch v := keyVal.(type) {
		case *types.Err:
			return v
		case *types.Unknown:
			if unk == nil {
				unk = v
			} else {
				unk = types.MergeUnknowns(unk, v)
			}
		}

		valVal := m.vals[i].Exec(frame)
		switch v := valVal.(type) {
		case *types.Err:
			return v
		case *types.Unknown:
			if unk == nil {
				unk = v
			} else {
				unk = types.MergeUnknowns(unk, v)
			}
		case *types.Optional:
			if m.optionals[i] {
				if !v.HasValue() {
					delete(entries, keyVal)
					continue
				}
				valVal = v.GetValue()
			}
		default:
			if m.optionals[i] {
				return types.LabelErrNode(m.id, invalidOptionalEntryInit(keyVal, valVal))
			}
		}
		entries[keyVal] = valVal
	}
	if unk != nil {
		if costs := frame.CostTracker(); costs != nil {
			costs.CreateMap(m.id, unk)
		}
		return unk
	}
	res := types.NewRefValMap(m.adapter, entries)
	if costs := frame.CostTracker(); costs != nil {
		costs.CreateMap(m.id, res)
	}
	return res
}

// Eval implements the Interpretable interface method.
func (m *evalMap) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, m.Exec)
}

func (m *evalMap) InitVals() []InterpretableV2 {
	if len(m.keys) != len(m.vals) {
		return nil
	}
	result := make([]InterpretableV2, len(m.keys)+len(m.vals))
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
	vals         []InterpretableV2
	optionals    []bool
	hasOptionals bool
	provider     types.Provider
}

// ID implements the Interpretable interface method.
func (o *evalObj) ID() int64 {
	return o.id
}

// Exec implements the InterpretableV2 interface method.
func (o *evalObj) Exec(frame *ExecutionFrame) ref.Val {
	fieldVals := make(map[string]ref.Val, len(o.fields))
	var unk *types.Unknown
	if !o.hasOptionals {
		for i, field := range o.fields {
			val := o.vals[i].Exec(frame)
			switch v := val.(type) {
			case *types.Err:
				return v
			case *types.Unknown:
				if unk == nil {
					unk = v
				} else {
					unk = types.MergeUnknowns(unk, v)
				}
			}
			fieldVals[field] = val
		}
		if unk != nil {
			trackCostCreateStruct(frame, o.id, unk)
			return unk
		}
		res := labelErrNode(o.id, o.provider.NewValue(o.typeName, fieldVals))
		trackCostCreateStruct(frame, o.id, res)
		return res
	}

	for i, field := range o.fields {
		val := o.vals[i].Exec(frame)
		switch v := val.(type) {
		case *types.Err:
			return v
		case *types.Unknown:
			if unk == nil {
				unk = v
			} else {
				unk = types.MergeUnknowns(unk, v)
			}
		case *types.Optional:
			if o.optionals[i] {
				if !v.HasValue() {
					delete(fieldVals, field)
					continue
				}
				val = v.GetValue()
			}
		default:
			if o.optionals[i] {
				return types.LabelErrNode(o.id, invalidOptionalEntryInit(field, val))
			}
		}
		fieldVals[field] = val
	}
	if unk != nil {
		if costs := frame.CostTracker(); costs != nil {
			costs.CreateStruct(o.id, unk)
		}
		return unk
	}
	res := labelErrNode(o.id, o.provider.NewValue(o.typeName, fieldVals))
	if costs := frame.CostTracker(); costs != nil {
		costs.CreateStruct(o.id, res)
	}
	return res
}

// Eval implements the Interpretable interface method.
func (o *evalObj) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, o.Exec)
}

// InitVals implements the InterpretableConstructor interface method.
func (o *evalObj) InitVals() []InterpretableV2 {
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
	iterRange InterpretableV2
	accu      InterpretableV2
	cond      InterpretableV2
	step      InterpretableV2
	result    InterpretableV2
	adapter   types.Adapter

	// mutableAccu indicates that static analysis of the comprehension has determined that the
	// accumulator is only ever observed by in-place accumulation steps and the final result, and
	// may therefore be backed by a mutable list or map without the mutation being observable.
	mutableAccu bool

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

// Exec implements the InterpretableV2 interface method.
func (fold *evalFold) Exec(frame *ExecutionFrame) ref.Val {
	f := newFolder(fold, frame)
	foldRange := fold.iterRange.Exec(frame)

	// Early return if the folder is an error or unknown.
	if types.IsUnknownOrError(foldRange) {
		releaseFolder(f)
		return foldRange
	}

	// Short-circuit the iteration if the range is empty.
	if sizer, ok := foldRange.(traits.Sizer); ok && sizer.Size() == types.IntZero {
		res := f.evalResult()
		releaseFolder(f)
		return res
	}

	// Otherwise, attempt a two variable fold.
	var foldable traits.Foldable
	switch r := foldRange.(type) {
	case traits.Foldable:
		foldable = r
		if fold.iterVar2 == "" {
			_, f.listV1Fold = foldRange.(traits.Lister)
		}
	case traits.Mapper:
		foldable = types.ToFoldableMap(r)
	case traits.Lister:
		if fold.iterVar2 == "" {
			f.listV1Fold = true
		}
		foldable = types.ToFoldableList(r)
	default:
		if fold.iterVar2 != "" {
			return types.NewErrWithNodeID(fold.ID(), "unsupported comprehension range type: %T", foldRange)
		}
	}
	if foldable != nil {
		foldable.Fold(f)
		res := f.evalResult()
		releaseFolder(f)
		return res
	}

	// If the value is not foldable or just a single variable, fallback to an iterable fold.
	if !foldRange.Type().HasTrait(traits.IterableType) {
		releaseFolder(f)
		return types.ValOrErr(foldRange, "got '%T', expected iterable type", foldRange)
	}
	iterable := foldRange.(traits.Iterable)
	res := f.foldIterable(iterable)
	releaseFolder(f)
	return res
}

// Eval implements the Interpretable interface method.
func (fold *evalFold) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, fold.Exec)
}

// Optional Interpretable implementations that specialize, subsume, or extend the core evaluation
// plan via decorators.

// evalSetMembership is an Interpretable implementation which tests whether an input value
// exists within the set of map keys used to model a set.
type evalSetMembership struct {
	inst     InterpretableV2
	arg      InterpretableV2
	valueSet map[ref.Val]ref.Val
}

// ID implements the Interpretable interface method.
func (e *evalSetMembership) ID() int64 {
	return e.inst.ID()
}

// Exec implements the InterpretableV2 interface method.
func (e *evalSetMembership) Exec(frame *ExecutionFrame) ref.Val {
	val := e.arg.Exec(frame)
	if types.IsUnknownOrError(val) {
		return val
	}
	if ret, found := e.valueSet[val]; found {
		return ret
	}
	return types.False
}

// Eval implements the Interpretable interface method.
func (e *evalSetMembership) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, e.Exec)
}

// evalWatch is an Interpretable implementation that wraps the execution of a given
// expression so that it may observe the computed value and send it to an observer.
type evalWatch struct {
	InterpretableV2
	observer EvalObserver
}

// Exec implements the InterpretableV2 interface method.
func (e *evalWatch) Exec(frame *ExecutionFrame) ref.Val {
	val := e.InterpretableV2.Exec(frame)
	e.observer(frame, e.ID(), e.InterpretableV2, val)
	return val
}

// Eval implements the Interpretable interface method.
func (e *evalWatch) Eval(vars Activation) ref.Val {
	return EvalActivation(vars, e.Exec)
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

// Exec implements the InterpretableV2 interface method.
func (e *evalWatchAttr) Exec(frame *ExecutionFrame) ref.Val {
	val := e.InterpretableAttribute.Exec(frame)
	e.observer(frame, e.ID(), e.InterpretableAttribute, val)
	return val
}

// Eval implements the Interpretable interface method.
func (e *evalWatchAttr) Eval(vars Activation) ref.Val {
	return EvalActivation(vars, e.Exec)
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

// Exec implements the InterpretableV2 interface method.
func (e *evalWatchConst) Exec(frame *ExecutionFrame) ref.Val {
	val := e.Value()
	e.observer(frame, e.ID(), e.InterpretableConst, val)
	return val
}

// Eval implements the Interpretable interface method.
func (e *evalWatchConst) Eval(vars Activation) ref.Val {
	return EvalActivation(vars, e.Exec)
}

// evalExhaustiveOr is just like evalOr, but does not short-circuit argument evaluation.
type evalExhaustiveOr struct {
	id    int64
	terms []InterpretableV2
}

// ID implements the Interpretable interface method.
func (or *evalExhaustiveOr) ID() int64 {
	return or.id
}

// Exec implements the InterpretableV2 interface method.
func (or *evalExhaustiveOr) Exec(frame *ExecutionFrame) ref.Val {
	var err ref.Val = nil
	var unk *types.Unknown
	isTrue := false
	for _, term := range or.terms {
		val := term.Exec(frame)
		boolVal, ok := val.(types.Bool)
		// flag the result as true
		if ok && boolVal == types.True {
			isTrue = true
		}
		if !ok && !isTrue {
			if unkVal, isUnk := val.(*types.Unknown); isUnk {
				if unk == nil {
					unk = unkVal
				} else {
					unk = types.MergeUnknowns(unk, unkVal)
				}
			} else if err == nil {
				if isError(val) {
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

// Eval implements the Interpretable interface method.
func (or *evalExhaustiveOr) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, or.Exec)
}

// evalExhaustiveAnd is just like evalAnd, but does not short-circuit argument evaluation.
type evalExhaustiveAnd struct {
	id    int64
	terms []InterpretableV2
}

// ID implements the Interpretable interface method.
func (and *evalExhaustiveAnd) ID() int64 {
	return and.id
}

// Exec implements the InterpretableV2 interface method.
func (and *evalExhaustiveAnd) Exec(frame *ExecutionFrame) ref.Val {
	var err ref.Val = nil
	var unk *types.Unknown
	isFalse := false
	for _, term := range and.terms {
		val := term.Exec(frame)
		boolVal, ok := val.(types.Bool)
		// short-circuit on false.
		if ok && boolVal == types.False {
			isFalse = true
		}
		if !ok && !isFalse {
			if unkVal, isUnk := val.(*types.Unknown); isUnk {
				if unk == nil {
					unk = unkVal
				} else {
					unk = types.MergeUnknowns(unk, unkVal)
				}
			} else if err == nil {
				if isError(val) {
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

// Eval implements the Interpretable interface method.
func (and *evalExhaustiveAnd) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, and.Exec)
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

// Exec implements the InterpretableV2 interface method.
func (cond *evalExhaustiveConditional) Exec(frame *ExecutionFrame) ref.Val {
	cVal := cond.attr.expr.Exec(frame)
	tVal, tErr := cond.attr.truthy.Resolve(frame)
	fVal, fErr := cond.attr.falsy.Resolve(frame)
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

// Eval implements the Interpretable interface method.
func (cond *evalExhaustiveConditional) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, cond.Exec)
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

// Exec implements the InterpretableV2 interface method.
func (a *evalAttr) Exec(frame *ExecutionFrame) ref.Val {
	v, err := a.attr.Resolve(frame)
	var res ref.Val
	if err != nil {
		res = types.LabelErrNode(a.ID(), types.WrapErr(err))
	} else {
		res = a.adapter.NativeToValue(v)
	}
	if _, isCond := a.attr.(*conditionalAttribute); !isCond {
		if costs := frame.CostTracker(); costs != nil {
			costs.EvalAttribute(a.ID(), false, res)
		}
	}
	return res
}

// Eval implements the Interpretable interface method.
func (a *evalAttr) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, a.Exec)
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
func (c *evalWatchConstructor) InitVals() []InterpretableV2 {
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

// Exec implements the InterpretableV2 interface method.
func (c *evalWatchConstructor) Exec(frame *ExecutionFrame) ref.Val {
	val := c.constructor.Exec(frame)
	c.observer(frame, c.ID(), c.constructor, val)
	return val
}

// Eval implements the Interpretable Eval function.
func (c *evalWatchConstructor) Eval(vars Activation) ref.Val {
	return EvalActivation(vars, c.Exec)
}

func invalidOptionalEntryInit(field any, value ref.Val) ref.Val {
	return types.NewErr("cannot initialize optional entry '%v' from non-optional value %v", field, value)
}

func invalidOptionalElementInit(value ref.Val) ref.Val {
	return types.NewErr("cannot initialize optional list element from non-optional value %v", value)
}

// newFolder creates or initializes a pooled folder instance.
func newFolder(eval *evalFold, frame *ExecutionFrame) *folder {
	f := folderPool.Get().(*folder)
	f.evalFold = eval
	f.frame = frame.Push(f)
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
	frame *ExecutionFrame

	// fold state objects.
	accuVal     ref.Val
	iterVar1Val any
	iterVar2Val any

	// bookkeeping flags to modify Activation and fold behaviors.
	initialized   bool
	mutableValue  bool
	interrupted   bool
	computeResult bool
	listV1Fold    bool
}

func (f *folder) foldIterable(iterable traits.Iterable) ref.Val {
	it := iterable.Iterator()
	for it.HasNext() == types.True {
		f.iterVar1Val = it.Next()

		cond := f.cond.Exec(f.frame)
		if f.interrupted {
			return f.evalResult()
		}
		if !f.exhaustive && cond != types.True {
			if isError(cond) {
				f.interrupted = true
			}
			return f.evalResult()
		}

		// Update the accumulation value and check for eval interruption.
		f.accuVal = f.step.Exec(f.frame)
		f.initialized = true
		if f.interruptable && f.frame.CheckInterrupt() {
			f.interrupted = true
			return f.evalResult()
		}
	}
	return f.evalResult()
}

// FoldKeyOnly reports whether the folder only consumes the entry key (e.g. 1-variable map
// comprehensions where iterVar2 is empty and listV1Fold is false).
func (f *folder) FoldKeyOnly() bool {
	return f.iterVar2 == "" && !f.listV1Fold
}

// FoldEntry will either fold comprehension v1 style macros if iterVar2 is unset, or comprehension v2 style
// macros if both the iterVar and iterVar2 are set to non-empty strings.
func (f *folder) FoldEntry(key, val any) bool {
	if f.listV1Fold {
		f.iterVar1Val = val
	} else {
		f.iterVar1Val = key
		f.iterVar2Val = val
	}

	// Terminate evaluation if evaluation is interrupted or the condition is not true and exhaustive
	// eval is not enabled.
	cond := f.cond.Exec(f.frame)
	if f.interrupted {
		return false
	}
	if !f.exhaustive && cond != types.True {
		if isError(cond) {
			f.interrupted = true
		}
		return false
	}

	// Update the accumulation value and check for eval interruption.
	f.accuVal = f.step.Exec(f.frame)
	f.initialized = true
	if f.interruptable && f.frame.CheckInterrupt() {
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
			initVal := f.accu.Exec(f.frame.parent)
			// Only substitute a mutable accumulator when the planner has proven that the
			// accumulator cannot be aliased, and when evaluation is not exhaustive since
			// exhaustive evaluation would apply in-place updates from untaken branches.
			if f.mutableAccu && !f.exhaustive {
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
			return f.iterVar1Val, true
		}
		if name == f.iterVar2 {
			return f.iterVar2Val, true
		}
	}
	return f.frame.parent.ResolveName(name)
}

// Parent returns the activation embedded into the folder.
func (f *folder) Parent() Activation {
	return f.frame.parent
}

// Unwrap returns the parent activation, thus omitting access to local state
func (f *folder) Unwrap() Activation {
	return f.frame.parent
}

// IsLocalVariable reports whether the variable name is locally bound by the folder scope.
func (f *folder) IsLocalVariable(name string) bool {
	if name == f.accuVar {
		return true
	}
	if !f.computeResult && (name == f.iterVar || name == f.iterVar2) {
		return true
	}
	if varHolder, ok := f.Parent().(localVariableHolder); ok {
		if varHolder.IsLocalVariable(name) {
			return true
		}
	}
	return false
}

// UnknownAttributePatterns implements the PartialActivation interface returning the unknown patterns
// if they were provided to the input activation, or an empty set if the proxied activation is not partial.
func (f *folder) UnknownAttributePatterns() []*AttributePattern {
	if f.frame.parent != nil {
		if partial, isPartial := f.frame.parent.AsPartialActivation(); isPartial {
			return partial.UnknownAttributePatterns()
		}
	}
	return []*AttributePattern{}
}

func (f *folder) AsPartialActivation() (PartialActivation, bool) {
	if f.frame.parent != nil {
		if _, isPartial := f.frame.parent.AsPartialActivation(); isPartial {
			return f, true
		}
	}
	return nil, false
}

// evalResult computes the final result of the fold after all entries have been folded and accumulated.
func (f *folder) evalResult() ref.Val {
	f.computeResult = true
	if f.interrupted {
		return types.WrapErr(InterruptError{})
	}
	res := f.result.Exec(f.frame)
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
	f.frame.Pop()
	f.frame = nil
	f.accuVal = nil
	f.iterVar1Val = nil
	f.iterVar2Val = nil

	f.initialized = false
	f.mutableValue = false
	f.interrupted = false
	f.computeResult = false
	f.listV1Fold = false
}

// InterruptError is a specialized error type used to signal that program evaluation should check
// whether a context cancellation is responsible for the error.
type InterruptError struct{}

// Error returns operation interrupted.
func (InterruptError) Error() string {
	return "operation interrupted"
}

// Is returns whether two errors are interrupt errors.
func (ie InterruptError) Is(target error) bool {
	return target.Error() == ie.Error()
}

var (
	// pool of var folders to reduce allocations during folds.
	folderPool = &sync.Pool{
		New: func() any {
			return &folder{}
		},
	}
)

func trackCostCreateList(frame *ExecutionFrame, id int64, res ref.Val) {
	if costs := frame.CostTracker(); costs != nil {
		costs.CreateList(id, res)
	}
}

func trackCostCreateMap(frame *ExecutionFrame, id int64, res ref.Val) {
	if costs := frame.CostTracker(); costs != nil {
		costs.CreateMap(id, res)
	}
}

func trackCostCreateStruct(frame *ExecutionFrame, id int64, res ref.Val) {
	if costs := frame.CostTracker(); costs != nil {
		costs.CreateStruct(id, res)
	}
}

func trackCostQualify(frame *ExecutionFrame, id int64) {
	if costs := frame.CostTracker(); costs != nil {
		costs.Qualify(id)
	}
}

func isError(val ref.Val) bool {
	_, ok := val.(*types.Err)
	return ok
}

func isUnknown(val ref.Val) bool {
	_, ok := val.(*types.Unknown)
	return ok
}

func mergeBinaryUnknowns(lVal, rVal ref.Val) *types.Unknown {
	lUnk, lIsUnk := lVal.(*types.Unknown)
	rUnk, rIsUnk := rVal.(*types.Unknown)
	if !lIsUnk {
		if !rIsUnk {
			return nil
		}
		return rUnk
	}
	if !rIsUnk {
		return lUnk
	}
	return types.MergeUnknowns(lUnk, rUnk)
}

func mergeUnknown(dst *types.Unknown, src ref.Val) *types.Unknown {
	if unk, ok := src.(*types.Unknown); ok {
		if dst == nil {
			return unk
		}
		return types.MergeUnknowns(dst, unk)
	}
	return dst
}
func labelErrNode(id int64, val ref.Val) ref.Val {
	if err, ok := val.(*types.Err); ok && err.NodeID() == 0 {
		return types.LabelErrNode(id, err)
	}
	return val
}
