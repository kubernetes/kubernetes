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

package cel

import (
	"context"
	"fmt"
	"math"
	"sync"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"
)

// Program is an evaluable view of an Ast.
type Program interface {
	// Eval returns the result of an evaluation of the Ast and environment against the input vars.
	//
	// The vars value may either be an `interpreter.Activation` or a `map[string]interface{}`.
	//
	// If the `OptTrackState`, `OptTrackCost` or `OptExhaustiveEval` flags are used, the `details` response will
	// be non-nil. Given this caveat on `details`, the return state from evaluation will be:
	//
	// *  `val`, `details`, `nil` - Successful evaluation of a non-error result.
	// *  `val`, `details`, `err` - Successful evaluation to an error result.
	// *  `nil`, `details`, `err` - Unsuccessful evaluation.
	//
	// An unsuccessful evaluation is typically the result of a series of incompatible `EnvOption`
	// or `ProgramOption` values used in the creation of the evaluation environment or executable
	// program.
	Eval(interface{}) (ref.Val, *EvalDetails, error)

	// ContextEval evaluates the program with a set of input variables and a context object in order
	// to support cancellation and timeouts. This method must be used in conjunction with the
	// InterruptCheckFrequency() option for cancellation interrupts to be impact evaluation.
	//
	// The vars value may either be an `interpreter.Activation` or `map[string]interface{}`.
	//
	// The output contract for `ContextEval` is otherwise identical to the `Eval` method.
	ContextEval(context.Context, interface{}) (ref.Val, *EvalDetails, error)
}

// NoVars returns an empty Activation.
func NoVars() interpreter.Activation {
	return interpreter.EmptyActivation()
}

// PartialVars returns a PartialActivation which contains variables and a set of AttributePattern
// values that indicate variables or parts of variables whose value are not yet known.
//
// The `vars` value may either be an interpreter.Activation or any valid input to the
// interpreter.NewActivation call.
func PartialVars(vars interface{},
	unknowns ...*interpreter.AttributePattern) (interpreter.PartialActivation, error) {
	return interpreter.NewPartialActivation(vars, unknowns...)
}

// AttributePattern returns an AttributePattern that matches a top-level variable. The pattern is
// mutable, and its methods support the specification of one or more qualifier patterns.
//
// For example, the AttributePattern(`a`).QualString(`b`) represents a variable access `a` with a
// string field or index qualification `b`. This pattern will match Attributes `a`, and `a.b`,
// but not `a.c`.
//
// When using a CEL expression within a container, e.g. a package or namespace, the variable name
// in the pattern must match the qualified name produced during the variable namespace resolution.
// For example, when variable `a` is declared within an expression whose container is `ns.app`, the
// fully qualified variable name may be `ns.app.a`, `ns.a`, or `a` per the CEL namespace resolution
// rules. Pick the fully qualified variable name that makes sense within the container as the
// AttributePattern `varName` argument.
//
// See the interpreter.AttributePattern and interpreter.AttributeQualifierPattern for more info
// about how to create and manipulate AttributePattern values.
func AttributePattern(varName string) *interpreter.AttributePattern {
	return interpreter.NewAttributePattern(varName)
}

// EvalDetails holds additional information observed during the Eval() call.
type EvalDetails struct {
	state       interpreter.EvalState
	costTracker *interpreter.CostTracker
}

// State of the evaluation, non-nil if the OptTrackState or OptExhaustiveEval is specified
// within EvalOptions.
func (ed *EvalDetails) State() interpreter.EvalState {
	return ed.state
}

// ActualCost returns the tracked cost through the course of execution when `CostTracking` is enabled.
// Otherwise, returns nil if the cost was not enabled.
func (ed *EvalDetails) ActualCost() *uint64 {
	if ed.costTracker == nil {
		return nil
	}
	cost := ed.costTracker.ActualCost()
	return &cost
}

// prog is the internal implementation of the Program interface.
type prog struct {
	*Env
	evalOpts                EvalOption
	defaultVars             interpreter.Activation
	dispatcher              interpreter.Dispatcher
	interpreter             interpreter.Interpreter
	interruptCheckFrequency uint

	// Intermediate state used to configure the InterpretableDecorator set provided
	// to the initInterpretable call.
	decorators         []interpreter.InterpretableDecorator
	regexOptimizations []*interpreter.RegexOptimization

	// Interpretable configured from an Ast and aggregate decorator set based on program options.
	interpretable     interpreter.Interpretable
	callCostEstimator interpreter.ActualCostEstimator
	costLimit         *uint64
}

func (p *prog) clone() *prog {
	return &prog{
		Env:                     p.Env,
		evalOpts:                p.evalOpts,
		defaultVars:             p.defaultVars,
		dispatcher:              p.dispatcher,
		interpreter:             p.interpreter,
		interruptCheckFrequency: p.interruptCheckFrequency,
	}
}

// newProgram creates a program instance with an environment, an ast, and an optional list of
// ProgramOption values.
//
// If the program cannot be configured the prog will be nil, with a non-nil error response.
func newProgram(e *Env, ast *Ast, opts []ProgramOption) (Program, error) {
	// Build the dispatcher, interpreter, and default program value.
	disp := interpreter.NewDispatcher()

	// Ensure the default attribute factory is set after the adapter and provider are
	// configured.
	p := &prog{
		Env:        e,
		decorators: []interpreter.InterpretableDecorator{},
		dispatcher: disp,
	}

	// Configure the program via the ProgramOption values.
	var err error
	for _, opt := range opts {
		p, err = opt(p)
		if err != nil {
			return nil, err
		}
	}

	// Add the function bindings created via Function() options.
	for _, fn := range e.functions {
		bindings, err := fn.bindings()
		if err != nil {
			return nil, err
		}
		err = disp.Add(bindings...)
		if err != nil {
			return nil, err
		}
	}

	// Set the attribute factory after the options have been set.
	var attrFactory interpreter.AttributeFactory
	if p.evalOpts&OptPartialEval == OptPartialEval {
		attrFactory = interpreter.NewPartialAttributeFactory(e.Container, e.adapter, e.provider)
	} else {
		attrFactory = interpreter.NewAttributeFactory(e.Container, e.adapter, e.provider)
	}
	interp := interpreter.NewInterpreter(disp, e.Container, e.provider, e.adapter, attrFactory)
	p.interpreter = interp

	// Translate the EvalOption flags into InterpretableDecorator instances.
	decorators := make([]interpreter.InterpretableDecorator, len(p.decorators))
	copy(decorators, p.decorators)

	// Enable interrupt checking if there's a non-zero check frequency
	if p.interruptCheckFrequency > 0 {
		decorators = append(decorators, interpreter.InterruptableEval())
	}
	// Enable constant folding first.
	if p.evalOpts&OptOptimize == OptOptimize {
		decorators = append(decorators, interpreter.Optimize())
		p.regexOptimizations = append(p.regexOptimizations, interpreter.MatchesRegexOptimization)
	}
	// Enable regex compilation of constants immediately after folding constants.
	if len(p.regexOptimizations) > 0 {
		decorators = append(decorators, interpreter.CompileRegexConstants(p.regexOptimizations...))
	}

	// Enable exhaustive eval, state tracking and cost tracking last since they require a factory.
	if p.evalOpts&(OptExhaustiveEval|OptTrackState|OptTrackCost) != 0 {
		factory := func(state interpreter.EvalState, costTracker *interpreter.CostTracker) (Program, error) {
			costTracker.Estimator = p.callCostEstimator
			costTracker.Limit = p.costLimit
			// Limit capacity to guarantee a reallocation when calling 'append(decs, ...)' below. This
			// prevents the underlying memory from being shared between factory function calls causing
			// undesired mutations.
			decs := decorators[:len(decorators):len(decorators)]
			var observers []interpreter.EvalObserver

			if p.evalOpts&(OptExhaustiveEval|OptTrackState) != 0 {
				// EvalStateObserver is required for OptExhaustiveEval.
				observers = append(observers, interpreter.EvalStateObserver(state))
			}
			if p.evalOpts&OptTrackCost == OptTrackCost {
				observers = append(observers, interpreter.CostObserver(costTracker))
			}

			// Enable exhaustive eval over a basic observer since it offers a superset of features.
			if p.evalOpts&OptExhaustiveEval == OptExhaustiveEval {
				decs = append(decs, interpreter.ExhaustiveEval(), interpreter.Observe(observers...))
			} else if len(observers) > 0 {
				decs = append(decs, interpreter.Observe(observers...))
			}

			return p.clone().initInterpretable(ast, decs)
		}
		return newProgGen(factory)
	}
	return p.initInterpretable(ast, decorators)
}

func (p *prog) initInterpretable(ast *Ast, decs []interpreter.InterpretableDecorator) (*prog, error) {
	// Unchecked programs do not contain type and reference information and may be slower to execute.
	if !ast.IsChecked() {
		interpretable, err :=
			p.interpreter.NewUncheckedInterpretable(ast.Expr(), decs...)
		if err != nil {
			return nil, err
		}
		p.interpretable = interpretable
		return p, nil
	}

	// When the AST has been checked it contains metadata that can be used to speed up program execution.
	var checked *exprpb.CheckedExpr
	checked, err := AstToCheckedExpr(ast)
	if err != nil {
		return nil, err
	}
	interpretable, err := p.interpreter.NewInterpretable(checked, decs...)
	if err != nil {
		return nil, err
	}
	p.interpretable = interpretable
	return p, nil
}

// Eval implements the Program interface method.
func (p *prog) Eval(input interface{}) (v ref.Val, det *EvalDetails, err error) {
	// Configure error recovery for unexpected panics during evaluation. Note, the use of named
	// return values makes it possible to modify the error response during the recovery
	// function.
	defer func() {
		if r := recover(); r != nil {
			switch t := r.(type) {
			case interpreter.EvalCancelledError:
				err = t
			default:
				err = fmt.Errorf("internal error: %v", r)
			}
		}
	}()
	// Build a hierarchical activation if there are default vars set.
	var vars interpreter.Activation
	switch v := input.(type) {
	case interpreter.Activation:
		vars = v
	case map[string]interface{}:
		vars = activationPool.Setup(v)
		defer activationPool.Put(vars)
	default:
		return nil, nil, fmt.Errorf("invalid input, wanted Activation or map[string]interface{}, got: (%T)%v", input, input)
	}
	if p.defaultVars != nil {
		vars = interpreter.NewHierarchicalActivation(p.defaultVars, vars)
	}
	v = p.interpretable.Eval(vars)
	// The output of an internal Eval may have a value (`v`) that is a types.Err. This step
	// translates the CEL value to a Go error response. This interface does not quite match the
	// RPC signature which allows for multiple errors to be returned, but should be sufficient.
	if types.IsError(v) {
		err = v.(*types.Err)
	}
	return
}

// ContextEval implements the Program interface.
func (p *prog) ContextEval(ctx context.Context, input interface{}) (ref.Val, *EvalDetails, error) {
	if ctx == nil {
		return nil, nil, fmt.Errorf("context can not be nil")
	}
	// Configure the input, making sure to wrap Activation inputs in the special ctxActivation which
	// exposes the #interrupted variable and manages rate-limited checks of the ctx.Done() state.
	var vars interpreter.Activation
	switch v := input.(type) {
	case interpreter.Activation:
		vars = ctxActivationPool.Setup(v, ctx.Done(), p.interruptCheckFrequency)
		defer ctxActivationPool.Put(vars)
	case map[string]interface{}:
		rawVars := activationPool.Setup(v)
		defer activationPool.Put(rawVars)
		vars = ctxActivationPool.Setup(rawVars, ctx.Done(), p.interruptCheckFrequency)
		defer ctxActivationPool.Put(vars)
	default:
		return nil, nil, fmt.Errorf("invalid input, wanted Activation or map[string]interface{}, got: (%T)%v", input, input)
	}
	return p.Eval(vars)
}

// Cost implements the Coster interface method.
func (p *prog) Cost() (min, max int64) {
	return estimateCost(p.interpretable)
}

// progFactory is a helper alias for marking a program creation factory function.
type progFactory func(interpreter.EvalState, *interpreter.CostTracker) (Program, error)

// progGen holds a reference to a progFactory instance and implements the Program interface.
type progGen struct {
	factory progFactory
}

// newProgGen tests the factory object by calling it once and returns a factory-based Program if
// the test is successful.
func newProgGen(factory progFactory) (Program, error) {
	// Test the factory to make sure that configuration errors are spotted at config
	_, err := factory(interpreter.NewEvalState(), &interpreter.CostTracker{})
	if err != nil {
		return nil, err
	}
	return &progGen{factory: factory}, nil
}

// Eval implements the Program interface method.
func (gen *progGen) Eval(input interface{}) (ref.Val, *EvalDetails, error) {
	// The factory based Eval() differs from the standard evaluation model in that it generates a
	// new EvalState instance for each call to ensure that unique evaluations yield unique stateful
	// results.
	state := interpreter.NewEvalState()
	costTracker := &interpreter.CostTracker{}
	det := &EvalDetails{state: state, costTracker: costTracker}

	// Generate a new instance of the interpretable using the factory configured during the call to
	// newProgram(). It is incredibly unlikely that the factory call will generate an error given
	// the factory test performed within the Program() call.
	p, err := gen.factory(state, costTracker)
	if err != nil {
		return nil, det, err
	}

	// Evaluate the input, returning the result and the 'state' within EvalDetails.
	v, _, err := p.Eval(input)
	if err != nil {
		return v, det, err
	}
	return v, det, nil
}

// ContextEval implements the Program interface method.
func (gen *progGen) ContextEval(ctx context.Context, input interface{}) (ref.Val, *EvalDetails, error) {
	if ctx == nil {
		return nil, nil, fmt.Errorf("context can not be nil")
	}
	// The factory based Eval() differs from the standard evaluation model in that it generates a
	// new EvalState instance for each call to ensure that unique evaluations yield unique stateful
	// results.
	state := interpreter.NewEvalState()
	costTracker := &interpreter.CostTracker{}
	det := &EvalDetails{state: state, costTracker: costTracker}

	// Generate a new instance of the interpretable using the factory configured during the call to
	// newProgram(). It is incredibly unlikely that the factory call will generate an error given
	// the factory test performed within the Program() call.
	p, err := gen.factory(state, costTracker)
	if err != nil {
		return nil, det, err
	}

	// Evaluate the input, returning the result and the 'state' within EvalDetails.
	v, _, err := p.ContextEval(ctx, input)
	if err != nil {
		return v, det, err
	}
	return v, det, nil
}

// Cost implements the Coster interface method.
func (gen *progGen) Cost() (min, max int64) {
	// Use an empty state value since no evaluation is performed.
	p, err := gen.factory(emptyEvalState, nil)
	if err != nil {
		return 0, math.MaxInt64
	}
	return estimateCost(p)
}

// EstimateCost returns the heuristic cost interval for the program.
func EstimateCost(p Program) (min, max int64) {
	return estimateCost(p)
}

func estimateCost(i interface{}) (min, max int64) {
	c, ok := i.(interpreter.Coster)
	if !ok {
		return 0, math.MaxInt64
	}
	return c.Cost()
}

type ctxEvalActivation struct {
	parent                  interpreter.Activation
	interrupt               <-chan struct{}
	interruptCheckCount     uint
	interruptCheckFrequency uint
}

// ResolveName implements the Activation interface method, but adds a special #interrupted variable
// which is capable of testing whether a 'done' signal is provided from a context.Context channel.
func (a *ctxEvalActivation) ResolveName(name string) (interface{}, bool) {
	if name == "#interrupted" {
		a.interruptCheckCount++
		if a.interruptCheckCount%a.interruptCheckFrequency == 0 {
			select {
			case <-a.interrupt:
				return true, true
			default:
				return nil, false
			}
		}
		return nil, false
	}
	return a.parent.ResolveName(name)
}

func (a *ctxEvalActivation) Parent() interpreter.Activation {
	return a.parent
}

func newCtxEvalActivationPool() *ctxEvalActivationPool {
	return &ctxEvalActivationPool{
		Pool: sync.Pool{
			New: func() interface{} {
				return &ctxEvalActivation{}
			},
		},
	}
}

type ctxEvalActivationPool struct {
	sync.Pool
}

// Setup initializes a pooled Activation with the ability check for context.Context cancellation
func (p *ctxEvalActivationPool) Setup(vars interpreter.Activation, done <-chan struct{}, interruptCheckRate uint) *ctxEvalActivation {
	a := p.Pool.Get().(*ctxEvalActivation)
	a.parent = vars
	a.interrupt = done
	a.interruptCheckCount = 0
	a.interruptCheckFrequency = interruptCheckRate
	return a
}

type evalActivation struct {
	vars     map[string]interface{}
	lazyVars map[string]interface{}
}

// ResolveName looks up the value of the input variable name, if found.
//
// Lazy bindings may be supplied within the map-based input in either of the following forms:
// - func() interface{}
// - func() ref.Val
//
// The lazy binding will only be invoked once per evaluation.
//
// Values which are not represented as ref.Val types on input may be adapted to a ref.Val using
// the ref.TypeAdapter configured in the environment.
func (a *evalActivation) ResolveName(name string) (interface{}, bool) {
	v, found := a.vars[name]
	if !found {
		return nil, false
	}
	switch obj := v.(type) {
	case func() ref.Val:
		if resolved, found := a.lazyVars[name]; found {
			return resolved, true
		}
		lazy := obj()
		a.lazyVars[name] = lazy
		return lazy, true
	case func() interface{}:
		if resolved, found := a.lazyVars[name]; found {
			return resolved, true
		}
		lazy := obj()
		a.lazyVars[name] = lazy
		return lazy, true
	default:
		return obj, true
	}
}

// Parent implements the interpreter.Activation interface
func (a *evalActivation) Parent() interpreter.Activation {
	return nil
}

func newEvalActivationPool() *evalActivationPool {
	return &evalActivationPool{
		Pool: sync.Pool{
			New: func() interface{} {
				return &evalActivation{lazyVars: make(map[string]interface{})}
			},
		},
	}
}

type evalActivationPool struct {
	sync.Pool
}

// Setup initializes a pooled Activation object with the map input.
func (p *evalActivationPool) Setup(vars map[string]interface{}) *evalActivation {
	a := p.Pool.Get().(*evalActivation)
	a.vars = vars
	return a
}

func (p *evalActivationPool) Put(value interface{}) {
	a := value.(*evalActivation)
	for k := range a.lazyVars {
		delete(a.lazyVars, k)
	}
	p.Pool.Put(a)
}

var (
	emptyEvalState = interpreter.NewEvalState()

	// activationPool is an internally managed pool of Activation values that wrap map[string]interface{} inputs
	activationPool = newEvalActivationPool()

	// ctxActivationPool is an internally managed pool of Activation values that expose a special #interrupted variable
	ctxActivationPool = newCtxEvalActivationPool()
)
