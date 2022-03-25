// Copyright 2018 Google LLC
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

// Package interpreter provides functions to evaluate parsed expressions with
// the option to augment the evaluation with inputs and functions supplied at
// evaluation time.
package interpreter

import (
	"github.com/google/cel-go/common/containers"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter/functions"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// Interpreter generates a new Interpretable from a checked or unchecked expression.
type Interpreter interface {
	// NewInterpretable creates an Interpretable from a checked expression and an
	// optional list of InterpretableDecorator values.
	NewInterpretable(checked *exprpb.CheckedExpr,
		decorators ...InterpretableDecorator) (Interpretable, error)

	// NewUncheckedInterpretable returns an Interpretable from a parsed expression
	// and an optional list of InterpretableDecorator values.
	NewUncheckedInterpretable(expr *exprpb.Expr,
		decorators ...InterpretableDecorator) (Interpretable, error)
}

// EvalObserver is a functional interface that accepts an expression id and an observed value.
// The id identifies the expression that was evaluated, the programStep is the Interpretable or Qualifier that
// was evaluated and value is the result of the evaluation.
type EvalObserver func(id int64, programStep interface{}, value ref.Val)

// Observe constructs a decorator that calls all the provided observers in order after evaluating each Interpretable
// or Qualifier during program evaluation.
func Observe(observers ...EvalObserver) InterpretableDecorator {
	if len(observers) == 1 {
		return decObserveEval(observers[0])
	}
	observeFn := func(id int64, programStep interface{}, val ref.Val) {
		for _, observer := range observers {
			observer(id, programStep, val)
		}
	}
	return decObserveEval(observeFn)
}

// EvalCancelledError represents a cancelled program evaluation operation.
type EvalCancelledError struct {
	Message string
	// Type identifies the cause of the cancellation.
	Cause CancellationCause
}

func (e EvalCancelledError) Error() string {
	return e.Message
}

// CancellationCause enumerates the ways a program evaluation operation can be cancelled.
type CancellationCause int

const (
	// ContextCancelled indicates that the operation was cancelled in response to a Golang context cancellation.
	ContextCancelled CancellationCause = iota

	// CostLimitExceeded indicates that the operation was cancelled in response to the actual cost limit being
	// exceeded.
	CostLimitExceeded
)

// TODO: Replace all usages of TrackState with EvalStateObserver

// TrackState decorates each expression node with an observer which records the value
// associated with the given expression id. EvalState must be provided to the decorator.
// This decorator is not thread-safe, and the EvalState must be reset between Eval()
// calls.
// DEPRECATED: Please use EvalStateObserver instead. It composes gracefully with additional observers.
func TrackState(state EvalState) InterpretableDecorator {
	return Observe(EvalStateObserver(state))
}

// EvalStateObserver provides an observer which records the value
// associated with the given expression id. EvalState must be provided to the observer.
// This decorator is not thread-safe, and the EvalState must be reset between Eval()
// calls.
func EvalStateObserver(state EvalState) EvalObserver {
	return func(id int64, programStep interface{}, val ref.Val) {
		state.SetValue(id, val)
	}
}

// ExhaustiveEval replaces operations that short-circuit with versions that evaluate
// expressions and couples this behavior with the TrackState() decorator to provide
// insight into the evaluation state of the entire expression. EvalState must be
// provided to the decorator. This decorator is not thread-safe, and the EvalState
// must be reset between Eval() calls.
func ExhaustiveEval() InterpretableDecorator {
	ex := decDisableShortcircuits()
	return func(i Interpretable) (Interpretable, error) {
		return ex(i)
	}
}

// InterruptableEval annotates comprehension loops with information that indicates they
// should check the `#interrupted` state within a custom Activation.
//
// The custom activation is currently managed higher up in the stack within the 'cel' package
// and should not require any custom support on behalf of callers.
func InterruptableEval() InterpretableDecorator {
	return decInterruptFolds()
}

// Optimize will pre-compute operations such as list and map construction and optimize
// call arguments to set membership tests. The set of optimizations will increase over time.
func Optimize() InterpretableDecorator {
	return decOptimize()
}

// RegexOptimization provides a way to replace an InterpretableCall for a regex function when the
// RegexIndex argument is a string constant. Typically, the Factory would compile the regex pattern at
// RegexIndex and report any errors (at program creation time) and then use the compiled regex for
// all regex function invocations.
type RegexOptimization struct {
	// Function is the name of the function to optimize.
	Function string
	// OverloadID is the ID of the overload to optimize.
	OverloadID string
	// RegexIndex is the index position of the regex pattern argument. Only calls to the function where this argument is
	// a string constant will be delegated to this optimizer.
	RegexIndex int
	// Factory constructs a replacement InterpretableCall node that optimizes the regex function call. Factory is
	// provided with the unoptimized regex call and the string constant at the RegexIndex argument.
	// The Factory may compile the regex for use across all invocations of the call, return any errors and
	// return an interpreter.NewCall with the desired regex optimized function impl.
	Factory func(call InterpretableCall, regexPattern string) (InterpretableCall, error)
}

// CompileRegexConstants compiles regex pattern string constants at program creation time and reports any regex pattern
// compile errors.
func CompileRegexConstants(regexOptimizations ...*RegexOptimization) InterpretableDecorator {
	return decRegexOptimizer(regexOptimizations...)
}

type exprInterpreter struct {
	dispatcher  Dispatcher
	container   *containers.Container
	provider    ref.TypeProvider
	adapter     ref.TypeAdapter
	attrFactory AttributeFactory
}

// NewInterpreter builds an Interpreter from a Dispatcher and TypeProvider which will be used
// throughout the Eval of all Interpretable instances generated from it.
func NewInterpreter(dispatcher Dispatcher,
	container *containers.Container,
	provider ref.TypeProvider,
	adapter ref.TypeAdapter,
	attrFactory AttributeFactory) Interpreter {
	return &exprInterpreter{
		dispatcher:  dispatcher,
		container:   container,
		provider:    provider,
		adapter:     adapter,
		attrFactory: attrFactory}
}

// NewStandardInterpreter builds a Dispatcher and TypeProvider with support for all of the CEL
// builtins defined in the language definition.
func NewStandardInterpreter(container *containers.Container,
	provider ref.TypeProvider,
	adapter ref.TypeAdapter,
	resolver AttributeFactory) Interpreter {
	dispatcher := NewDispatcher()
	dispatcher.Add(functions.StandardOverloads()...)
	return NewInterpreter(dispatcher, container, provider, adapter, resolver)
}

// NewIntepretable implements the Interpreter interface method.
func (i *exprInterpreter) NewInterpretable(
	checked *exprpb.CheckedExpr,
	decorators ...InterpretableDecorator) (Interpretable, error) {
	p := newPlanner(
		i.dispatcher,
		i.provider,
		i.adapter,
		i.attrFactory,
		i.container,
		checked,
		decorators...)
	return p.Plan(checked.GetExpr())
}

// NewUncheckedIntepretable implements the Interpreter interface method.
func (i *exprInterpreter) NewUncheckedInterpretable(
	expr *exprpb.Expr,
	decorators ...InterpretableDecorator) (Interpretable, error) {
	p := newUncheckedPlanner(
		i.dispatcher,
		i.provider,
		i.adapter,
		i.attrFactory,
		i.container,
		decorators...)
	return p.Plan(expr)
}
