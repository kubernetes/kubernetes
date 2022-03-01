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

// TrackState decorates each expression node with an observer which records the value
// associated with the given expression id. EvalState must be provided to the decorator.
// This decorator is not thread-safe, and the EvalState must be reset between Eval()
// calls.
func TrackState(state EvalState) InterpretableDecorator {
	observer := func(id int64, val ref.Val) {
		state.SetValue(id, val)
	}
	return decObserveEval(observer)
}

// ExhaustiveEval replaces operations that short-circuit with versions that evaluate
// expressions and couples this behavior with the TrackState() decorator to provide
// insight into the evaluation state of the entire expression. EvalState must be
// provided to the decorator. This decorator is not thread-safe, and the EvalState
// must be reset between Eval() calls.
func ExhaustiveEval(state EvalState) InterpretableDecorator {
	ex := decDisableShortcircuits()
	obs := TrackState(state)
	return func(i Interpretable) (Interpretable, error) {
		var err error
		i, err = ex(i)
		if err != nil {
			return nil, err
		}
		return obs(i)
	}
}

// Optimize will pre-compute operations such as list and map construction and optimize
// call arguments to set membership tests. The set of optimizations will increase over time.
func Optimize() InterpretableDecorator {
	return decOptimize()
}

type exprInterpreter struct {
	dispatcher  Dispatcher
	container   *containers.Container
	provider    ref.TypeProvider
	adapter     ref.TypeAdapter
	attrFactory AttributeFactory
}

// NewInterpreter builds an Interpreter from a Dispatcher and TypeProvider which will be used
// throughout the Eval of all Interpretable instances gerenated from it.
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
