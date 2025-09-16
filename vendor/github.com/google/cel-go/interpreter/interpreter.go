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
	"errors"

	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/containers"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// PlannerOption configures the program plan options during interpretable setup.
type PlannerOption func(*planner) (*planner, error)

// Interpreter generates a new Interpretable from a checked or unchecked expression.
type Interpreter interface {
	// NewInterpretable creates an Interpretable from a checked expression and an
	// optional list of PlannerOption values.
	NewInterpretable(exprAST *ast.AST, opts ...PlannerOption) (Interpretable, error)
}

// EvalObserver is a functional interface that accepts an expression id and an observed value.
// The id identifies the expression that was evaluated, the programStep is the Interpretable or Qualifier that
// was evaluated and value is the result of the evaluation.
type EvalObserver func(vars Activation, id int64, programStep any, value ref.Val)

// StatefulObserver observes evaluation while tracking or utilizing stateful behavior.
type StatefulObserver interface {
	// InitState configures stateful metadata on the activation.
	InitState(Activation) (Activation, error)

	// GetState retrieves the stateful metadata from the activation.
	GetState(Activation) any

	// Observe passes the activation and relevant evaluation metadata to the observer.
	// The observe method is expected to do the equivalent of GetState(vars) in order
	// to find the metadata that needs to be updated upon invocation.
	Observe(vars Activation, id int64, programStep any, value ref.Val)
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

// evalStateOption configures the evalStateFactory behavior.
type evalStateOption func(*evalStateFactory) *evalStateFactory

// EvalStateFactory configures the EvalState generator to be used by the EvalStateObserver.
func EvalStateFactory(factory func() EvalState) evalStateOption {
	return func(fac *evalStateFactory) *evalStateFactory {
		fac.factory = factory
		return fac
	}
}

// EvalStateObserver provides an observer which records the value associated with the given expression id.
// EvalState must be provided to the observer.
func EvalStateObserver(opts ...evalStateOption) PlannerOption {
	et := &evalStateFactory{factory: NewEvalState}
	for _, o := range opts {
		et = o(et)
	}
	return func(p *planner) (*planner, error) {
		if et.factory == nil {
			return nil, errors.New("eval state factory not configured")
		}
		p.observers = append(p.observers, et)
		p.decorators = append(p.decorators, decObserveEval(et.Observe))
		return p, nil
	}
}

// evalStateConverter identifies an object which is convertible to an EvalState instance.
type evalStateConverter interface {
	asEvalState() EvalState
}

// evalStateActivation hides state in the Activation in a manner not accessible to expressions.
type evalStateActivation struct {
	vars  Activation
	state EvalState
}

// ResolveName proxies variable lookups to the backing activation.
func (esa evalStateActivation) ResolveName(name string) (any, bool) {
	return esa.vars.ResolveName(name)
}

// Parent proxies parent lookups to the backing activation.
func (esa evalStateActivation) Parent() Activation {
	return esa.vars
}

// AsPartialActivation supports conversion to a partial activation in order to detect unknown attributes.
func (esa evalStateActivation) AsPartialActivation() (PartialActivation, bool) {
	return AsPartialActivation(esa.vars)
}

// asEvalState implements the evalStateConverter method.
func (esa evalStateActivation) asEvalState() EvalState {
	return esa.state
}

// asEvalState walks the Activation hierarchy and returns the first EvalState found, if present.
func asEvalState(vars Activation) (EvalState, bool) {
	if conv, ok := vars.(evalStateConverter); ok {
		return conv.asEvalState(), true
	}
	if vars.Parent() != nil {
		return asEvalState(vars.Parent())
	}
	return nil, false
}

// evalStateFactory holds a reference to a factory function that produces an EvalState instance.
type evalStateFactory struct {
	factory func() EvalState
}

// InitState produces an EvalState instance and bundles it into the Activation in a way which is
// not visible to expression evaluation.
func (et *evalStateFactory) InitState(vars Activation) (Activation, error) {
	state := et.factory()
	return evalStateActivation{vars: vars, state: state}, nil
}

// GetState extracts the EvalState from the Activation.
func (et *evalStateFactory) GetState(vars Activation) any {
	if state, found := asEvalState(vars); found {
		return state
	}
	return nil
}

// Observe records the evaluation state for a given expression node and program step.
func (et *evalStateFactory) Observe(vars Activation, id int64, programStep any, val ref.Val) {
	state, found := asEvalState(vars)
	if !found {
		return
	}
	state.SetValue(id, val)
}

// CustomDecorator configures a custom interpretable decorator for the program.
func CustomDecorator(dec InterpretableDecorator) PlannerOption {
	return func(p *planner) (*planner, error) {
		p.decorators = append(p.decorators, dec)
		return p, nil
	}
}

// ExhaustiveEval replaces operations that short-circuit with versions that evaluate
// expressions and couples this behavior with the TrackState() decorator to provide
// insight into the evaluation state of the entire expression. EvalState must be
// provided to the decorator. This decorator is not thread-safe, and the EvalState
// must be reset between Eval() calls.
func ExhaustiveEval() PlannerOption {
	return CustomDecorator(decDisableShortcircuits())
}

// InterruptableEval annotates comprehension loops with information that indicates they
// should check the `#interrupted` state within a custom Activation.
//
// The custom activation is currently managed higher up in the stack within the 'cel' package
// and should not require any custom support on behalf of callers.
func InterruptableEval() PlannerOption {
	return CustomDecorator(decInterruptFolds())
}

// Optimize will pre-compute operations such as list and map construction and optimize
// call arguments to set membership tests. The set of optimizations will increase over time.
func Optimize() PlannerOption {
	return CustomDecorator(decOptimize())
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
func CompileRegexConstants(regexOptimizations ...*RegexOptimization) PlannerOption {
	return CustomDecorator(decRegexOptimizer(regexOptimizations...))
}

type exprInterpreter struct {
	dispatcher  Dispatcher
	container   *containers.Container
	provider    types.Provider
	adapter     types.Adapter
	attrFactory AttributeFactory
}

// NewInterpreter builds an Interpreter from a Dispatcher and TypeProvider which will be used
// throughout the Eval of all Interpretable instances generated from it.
func NewInterpreter(dispatcher Dispatcher,
	container *containers.Container,
	provider types.Provider,
	adapter types.Adapter,
	attrFactory AttributeFactory) Interpreter {
	return &exprInterpreter{
		dispatcher:  dispatcher,
		container:   container,
		provider:    provider,
		adapter:     adapter,
		attrFactory: attrFactory}
}

// NewIntepretable implements the Interpreter interface method.
func (i *exprInterpreter) NewInterpretable(
	checked *ast.AST,
	opts ...PlannerOption) (Interpretable, error) {
	p := newPlanner(i.dispatcher, i.provider, i.adapter, i.attrFactory, i.container, checked)
	var err error
	for _, o := range opts {
		p, err = o(p)
		if err != nil {
			return nil, err
		}
	}
	return p.Plan(checked.Expr())
}
