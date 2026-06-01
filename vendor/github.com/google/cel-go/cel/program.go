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
	"sync"

	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/functions"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"
)

// Program is an evaluable view of an Ast.
type Program interface {
	// Eval returns the result of an evaluation of the Ast and environment against the input vars.
	//
	// The vars value may either be an `Activation` or a `map[string]any`.
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
	Eval(any) (ref.Val, *EvalDetails, error)

	// ContextEval evaluates the program with a set of input variables and a context object in order
	// to support cancellation and timeouts. This method must be used in conjunction with the
	// InterruptCheckFrequency() option for cancellation interrupts to be impact evaluation.
	//
	// The vars value may either be an `Activation` or `map[string]any`.
	//
	// The output contract for `ContextEval` is otherwise identical to the `Eval` method.
	ContextEval(context.Context, any) (ref.Val, *EvalDetails, error)
}

// Activation used to resolve identifiers by name and references by id.
//
// An Activation is the primary mechanism by which a caller supplies input into a CEL program.
type Activation = interpreter.Activation

// NewActivation returns an activation based on a map-based binding where the map keys are
// expected to be qualified names used with ResolveName calls.
//
// The input `bindings` may either be of type `Activation` or `map[string]any`.
//
// Lazy bindings may be supplied within the map-based input in either of the following forms:
// - func() any
// - func() ref.Val
//
// The output of the lazy binding will overwrite the variable reference in the internal map.
//
// Values which are not represented as ref.Val types on input may be adapted to a ref.Val using
// the types.Adapter configured in the environment.
func NewActivation(bindings any) (Activation, error) {
	return interpreter.NewActivation(bindings)
}

// PartialActivation extends the Activation interface with a set of unknown AttributePatterns.
type PartialActivation = interpreter.PartialActivation

// NoVars returns an empty Activation.
func NoVars() Activation {
	return interpreter.EmptyActivation()
}

// PartialVars returns a PartialActivation which contains variables and a set of AttributePattern
// values that indicate variables or parts of variables whose value are not yet known.
//
// This method relies on manually configured sets of missing attribute patterns. For a method which
// infers the missing variables from the input and the configured environment, use Env.PartialVars().
//
// The `vars` value may either be an Activation or any valid input to the NewActivation call.
func PartialVars(vars any,
	unknowns ...*AttributePatternType) (PartialActivation, error) {
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
func AttributePattern(varName string) *AttributePatternType {
	return interpreter.NewAttributePattern(varName)
}

// AttributePatternType represents a top-level variable with an optional set of qualifier patterns.
//
// See the interpreter.AttributePattern and interpreter.AttributeQualifierPattern for more info
// about how to create and manipulate AttributePattern values.
type AttributePatternType = interpreter.AttributePattern

// EvalDetails holds additional information observed during the Eval() call.
type EvalDetails struct {
	state       interpreter.EvalState
	costTracker *interpreter.CostTracker
}

// State of the evaluation, non-nil if the OptTrackState or OptExhaustiveEval is specified
// within EvalOptions.
func (ed *EvalDetails) State() interpreter.EvalState {
	if ed == nil {
		return interpreter.NewEvalState()
	}
	return ed.state
}

// ActualCost returns the tracked cost through the course of execution when `CostTracking` is enabled.
// Otherwise, returns nil if the cost was not enabled.
func (ed *EvalDetails) ActualCost() *uint64 {
	if ed == nil || ed.costTracker == nil {
		return nil
	}
	cost := ed.costTracker.ActualCost()
	return &cost
}

// prog is the internal implementation of the Program interface.
type prog struct {
	*Env
	evalOpts                EvalOption
	defaultVars             Activation
	dispatcher              interpreter.Dispatcher
	interpreter             interpreter.Interpreter
	interruptCheckFrequency uint

	// Intermediate state used to configure the InterpretableDecorator set provided
	// to the initInterpretable call.
	plannerOptions     []interpreter.PlannerOption
	regexOptimizations []*interpreter.RegexOptimization

	// Interpretable configured from an Ast and aggregate decorator set based on program options.
	interpretable     interpreter.Interpretable
	observable        *interpreter.ObservableInterpretable
	callCostEstimator interpreter.ActualCostEstimator
	costOptions       []interpreter.CostTrackerOption
	costLimit         *uint64
}

// newProgram creates a program instance with an environment, an ast, and an optional list of
// ProgramOption values.
//
// If the program cannot be configured the prog will be nil, with a non-nil error response.
func newProgram(e *Env, a *ast.AST, opts []ProgramOption) (Program, error) {
	// Build the dispatcher, interpreter, and default program value.
	disp := interpreter.NewDispatcher()

	// Ensure the default attribute factory is set after the adapter and provider are
	// configured.
	p := &prog{
		Env:            e,
		plannerOptions: []interpreter.PlannerOption{},
		dispatcher:     disp,
		costOptions:    []interpreter.CostTrackerOption{},
	}

	// Configure the program via the ProgramOption values.
	var err error
	for _, opt := range opts {
		p, err = opt(p)
		if err != nil {
			return nil, err
		}
	}

	e.funcBindOnce.Do(func() {
		var bindings []*functions.Overload
		e.functionBindings = []*functions.Overload{}
		for _, fn := range e.functions {
			bindings, err = fn.Bindings()
			if err != nil {
				return
			}
			e.functionBindings = append(e.functionBindings, bindings...)
		}
	})
	if err != nil {
		return nil, err
	}

	// Add the function bindings created via Function() options.
	err = disp.Add(e.functionBindings...)
	if err != nil {
		return nil, err
	}

	// Set the attribute factory after the options have been set.
	var attrFactory interpreter.AttributeFactory
	attrFactorOpts := []interpreter.AttrFactoryOption{
		interpreter.EnableErrorOnBadPresenceTest(p.HasFeature(featureEnableErrorOnBadPresenceTest)),
	}
	if p.evalOpts&OptPartialEval == OptPartialEval {
		attrFactory = interpreter.NewPartialAttributeFactory(e.Container, e.adapter, e.provider, attrFactorOpts...)
	} else {
		attrFactory = interpreter.NewAttributeFactory(e.Container, e.adapter, e.provider, attrFactorOpts...)
	}
	interp := interpreter.NewInterpreter(disp, e.Container, e.provider, e.adapter, attrFactory)
	p.interpreter = interp

	// Translate the EvalOption flags into InterpretableDecorator instances.
	plannerOptions := make([]interpreter.PlannerOption, len(p.plannerOptions))
	copy(plannerOptions, p.plannerOptions)

	// Enable interrupt checking if there's a non-zero check frequency
	if p.interruptCheckFrequency > 0 {
		plannerOptions = append(plannerOptions, interpreter.InterruptableEval())
	}
	// Enable constant folding first.
	if p.evalOpts&OptOptimize == OptOptimize {
		plannerOptions = append(plannerOptions, interpreter.Optimize())
		p.regexOptimizations = append(p.regexOptimizations, interpreter.MatchesRegexOptimization)
	}
	// Enable regex compilation of constants immediately after folding constants.
	if len(p.regexOptimizations) > 0 {
		plannerOptions = append(plannerOptions, interpreter.CompileRegexConstants(p.regexOptimizations...))
	}

	// Enable exhaustive eval, state tracking and cost tracking last since they require a factory.
	if p.evalOpts&(OptExhaustiveEval|OptTrackState|OptTrackCost) != 0 {
		costOptCount := len(p.costOptions)
		if p.costLimit != nil {
			costOptCount++
		}
		costOpts := make([]interpreter.CostTrackerOption, 0, costOptCount)
		costOpts = append(costOpts, p.costOptions...)
		if p.costLimit != nil {
			costOpts = append(costOpts, interpreter.CostTrackerLimit(*p.costLimit))
		}
		trackerFactory := func() (*interpreter.CostTracker, error) {
			return interpreter.NewCostTracker(p.callCostEstimator, costOpts...)
		}
		var observers []interpreter.PlannerOption
		if p.evalOpts&(OptExhaustiveEval|OptTrackState) != 0 {
			// EvalStateObserver is required for OptExhaustiveEval.
			observers = append(observers, interpreter.EvalStateObserver())
		}
		if p.evalOpts&OptTrackCost == OptTrackCost {
			observers = append(observers, interpreter.CostObserver(interpreter.CostTrackerFactory(trackerFactory)))
		}
		// Enable exhaustive eval over a basic observer since it offers a superset of features.
		if p.evalOpts&OptExhaustiveEval == OptExhaustiveEval {
			plannerOptions = append(plannerOptions,
				append([]interpreter.PlannerOption{interpreter.ExhaustiveEval()}, observers...)...)
		} else if len(observers) > 0 {
			plannerOptions = append(plannerOptions, observers...)
		}
	}
	return p.initInterpretable(a, plannerOptions)
}

func (p *prog) initInterpretable(a *ast.AST, plannerOptions []interpreter.PlannerOption) (*prog, error) {
	// When the AST has been exprAST it contains metadata that can be used to speed up program execution.
	interpretable, err := p.interpreter.NewInterpretable(a, plannerOptions...)
	if err != nil {
		return nil, err
	}
	p.interpretable = interpretable
	if oi, ok := interpretable.(*interpreter.ObservableInterpretable); ok {
		p.observable = oi
	}
	return p, nil
}

// Eval implements the Program interface method.
func (p *prog) Eval(input any) (out ref.Val, det *EvalDetails, err error) {
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
	var vars Activation
	switch v := input.(type) {
	case Activation:
		vars = v
	case map[string]any:
		vars = activationPool.Setup(v)
		defer activationPool.Put(vars)
	default:
		return nil, nil, fmt.Errorf("invalid input, wanted Activation or map[string]any, got: (%T)%v", input, input)
	}
	if p.defaultVars != nil {
		vars = interpreter.NewHierarchicalActivation(p.defaultVars, vars)
	}
	if p.observable != nil {
		det = &EvalDetails{}
		out = p.observable.ObserveEval(vars, func(observed any) {
			switch o := observed.(type) {
			case interpreter.EvalState:
				det.state = o
			case *interpreter.CostTracker:
				det.costTracker = o
			}
		})
	} else {
		out = p.interpretable.Eval(vars)
	}
	// The output of an internal Eval may have a value (`v`) that is a types.Err. This step
	// translates the CEL value to a Go error response. This interface does not quite match the
	// RPC signature which allows for multiple errors to be returned, but should be sufficient.
	if types.IsError(out) {
		err = out.(*types.Err)
	}
	return
}

// ContextEval implements the Program interface.
func (p *prog) ContextEval(ctx context.Context, input any) (ref.Val, *EvalDetails, error) {
	if ctx == nil {
		return nil, nil, fmt.Errorf("context can not be nil")
	}
	// Configure the input, making sure to wrap Activation inputs in the special ctxActivation which
	// exposes the #interrupted variable and manages rate-limited checks of the ctx.Done() state.
	var vars Activation
	switch v := input.(type) {
	case Activation:
		vars = ctxActivationPool.Setup(v, ctx.Done(), p.interruptCheckFrequency)
		defer ctxActivationPool.Put(vars)
	case map[string]any:
		rawVars := activationPool.Setup(v)
		defer activationPool.Put(rawVars)
		vars = ctxActivationPool.Setup(rawVars, ctx.Done(), p.interruptCheckFrequency)
		defer ctxActivationPool.Put(vars)
	default:
		return nil, nil, fmt.Errorf("invalid input, wanted Activation or map[string]any, got: (%T)%v", input, input)
	}
	return p.Eval(vars)
}

type ctxEvalActivation struct {
	parent                  Activation
	interrupt               <-chan struct{}
	interruptCheckCount     uint
	interruptCheckFrequency uint
}

// ResolveName implements the Activation interface method, but adds a special #interrupted variable
// which is capable of testing whether a 'done' signal is provided from a context.Context channel.
func (a *ctxEvalActivation) ResolveName(name string) (any, bool) {
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

func (a *ctxEvalActivation) Parent() Activation {
	return a.parent
}

func (a *ctxEvalActivation) AsPartialActivation() (interpreter.PartialActivation, bool) {
	pa, ok := a.parent.(interpreter.PartialActivation)
	return pa, ok
}

func newCtxEvalActivationPool() *ctxEvalActivationPool {
	return &ctxEvalActivationPool{
		Pool: sync.Pool{
			New: func() any {
				return &ctxEvalActivation{}
			},
		},
	}
}

type ctxEvalActivationPool struct {
	sync.Pool
}

// Setup initializes a pooled Activation with the ability check for context.Context cancellation
func (p *ctxEvalActivationPool) Setup(vars Activation, done <-chan struct{}, interruptCheckRate uint) *ctxEvalActivation {
	a := p.Pool.Get().(*ctxEvalActivation)
	a.parent = vars
	a.interrupt = done
	a.interruptCheckCount = 0
	a.interruptCheckFrequency = interruptCheckRate
	return a
}

type evalActivation struct {
	vars     map[string]any
	lazyVars map[string]any
}

// ResolveName looks up the value of the input variable name, if found.
//
// Lazy bindings may be supplied within the map-based input in either of the following forms:
// - func() any
// - func() ref.Val
//
// The lazy binding will only be invoked once per evaluation.
//
// Values which are not represented as ref.Val types on input may be adapted to a ref.Val using
// the types.Adapter configured in the environment.
func (a *evalActivation) ResolveName(name string) (any, bool) {
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
	case func() any:
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

// Parent implements the Activation interface
func (a *evalActivation) Parent() Activation {
	return nil
}

func newEvalActivationPool() *evalActivationPool {
	return &evalActivationPool{
		Pool: sync.Pool{
			New: func() any {
				return &evalActivation{lazyVars: make(map[string]any)}
			},
		},
	}
}

type evalActivationPool struct {
	sync.Pool
}

// Setup initializes a pooled Activation object with the map input.
func (p *evalActivationPool) Setup(vars map[string]any) *evalActivation {
	a := p.Pool.Get().(*evalActivation)
	a.vars = vars
	return a
}

func (p *evalActivationPool) Put(value any) {
	a := value.(*evalActivation)
	for k := range a.lazyVars {
		delete(a.lazyVars, k)
	}
	p.Pool.Put(a)
}

var (
	// activationPool is an internally managed pool of Activation values that wrap map[string]any inputs
	activationPool = newEvalActivationPool()

	// ctxActivationPool is an internally managed pool of Activation values that expose a special #interrupted variable
	ctxActivationPool = newCtxEvalActivationPool()
)
