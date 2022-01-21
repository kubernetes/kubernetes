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
	"fmt"
	"math"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// Program is an evaluable view of an Ast.
type Program interface {
	// Eval returns the result of an evaluation of the Ast and environment against the input vars.
	//
	// The vars value may either be an `interpreter.Activation` or a `map[string]interface{}`.
	//
	// If the `OptTrackState` or `OptExhaustiveEval` flags are used, the `details` response will
	// be non-nil. Given this caveat on `details`, the return state from evaluation will be:
	//
	// *  `val`, `details`, `nil` - Successful evaluation of a non-error result.
	// *  `val`, `details`, `err` - Successful evaluation to an error result.
	// *  `nil`, `details`, `err` - Unsuccessful evaluation.
	//
	// An unsuccessful evaluation is typically the result of a series of incompatible `EnvOption`
	// or `ProgramOption` values used in the creation of the evaluation environment or executable
	// program.
	Eval(vars interface{}) (ref.Val, *EvalDetails, error)
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
	state interpreter.EvalState
}

// State of the evaluation, non-nil if the OptTrackState or OptExhaustiveEval is specified
// within EvalOptions.
func (ed *EvalDetails) State() interpreter.EvalState {
	return ed.state
}

// prog is the internal implementation of the Program interface.
type prog struct {
	*Env
	evalOpts      EvalOption
	decorators    []interpreter.InterpretableDecorator
	defaultVars   interpreter.Activation
	dispatcher    interpreter.Dispatcher
	interpreter   interpreter.Interpreter
	interpretable interpreter.Interpretable
	attrFactory   interpreter.AttributeFactory
}

// progFactory is a helper alias for marking a program creation factory function.
type progFactory func(interpreter.EvalState) (Program, error)

// progGen holds a reference to a progFactory instance and implements the Program interface.
type progGen struct {
	factory progFactory
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
		if opt == nil {
			return nil, fmt.Errorf("program options should be non-nil")
		}
		p, err = opt(p)
		if err != nil {
			return nil, err
		}
	}

	// Set the attribute factory after the options have been set.
	if p.evalOpts&OptPartialEval == OptPartialEval {
		p.attrFactory = interpreter.NewPartialAttributeFactory(e.Container, e.adapter, e.provider)
	} else {
		p.attrFactory = interpreter.NewAttributeFactory(e.Container, e.adapter, e.provider)
	}

	interp := interpreter.NewInterpreter(disp, e.Container, e.provider, e.adapter, p.attrFactory)
	p.interpreter = interp

	// Translate the EvalOption flags into InterpretableDecorator instances.
	decorators := make([]interpreter.InterpretableDecorator, len(p.decorators))
	copy(decorators, p.decorators)

	// Enable constant folding first.
	if p.evalOpts&OptOptimize == OptOptimize {
		decorators = append(decorators, interpreter.Optimize())
	}
	// Enable exhaustive eval over state tracking since it offers a superset of features.
	if p.evalOpts&OptExhaustiveEval == OptExhaustiveEval {
		// State tracking requires that each Eval() call operate on an isolated EvalState
		// object; hence, the presence of the factory.
		factory := func(state interpreter.EvalState) (Program, error) {
			decs := append(decorators, interpreter.ExhaustiveEval(state))
			clone := &prog{
				evalOpts:    p.evalOpts,
				defaultVars: p.defaultVars,
				Env:         e,
				dispatcher:  disp,
				interpreter: interp}
			return initInterpretable(clone, ast, decs)
		}
		return initProgGen(factory)
	}
	// Enable state tracking last since it too requires the factory approach but is less
	// featured than the ExhaustiveEval decorator.
	if p.evalOpts&OptTrackState == OptTrackState {
		factory := func(state interpreter.EvalState) (Program, error) {
			decs := append(decorators, interpreter.TrackState(state))
			clone := &prog{
				evalOpts:    p.evalOpts,
				defaultVars: p.defaultVars,
				Env:         e,
				dispatcher:  disp,
				interpreter: interp}
			return initInterpretable(clone, ast, decs)
		}
		return initProgGen(factory)
	}
	return initInterpretable(p, ast, decorators)
}

// initProgGen tests the factory object by calling it once and returns a factory-based Program if
// the test is successful.
func initProgGen(factory progFactory) (Program, error) {
	// Test the factory to make sure that configuration errors are spotted at config
	_, err := factory(interpreter.NewEvalState())
	if err != nil {
		return nil, err
	}
	return &progGen{factory: factory}, nil
}

// initIterpretable creates a checked or unchecked interpretable depending on whether the Ast
// has been run through the type-checker.
func initInterpretable(
	p *prog,
	ast *Ast,
	decorators []interpreter.InterpretableDecorator) (Program, error) {
	var err error
	// Unchecked programs do not contain type and reference information and may be
	// slower to execute than their checked counterparts.
	if !ast.IsChecked() {
		p.interpretable, err =
			p.interpreter.NewUncheckedInterpretable(ast.Expr(), decorators...)
		if err != nil {
			return nil, err
		}
		return p, nil
	}
	// When the AST has been checked it contains metadata that can be used to speed up program
	// execution.
	var checked *exprpb.CheckedExpr
	checked, err = AstToCheckedExpr(ast)
	if err != nil {
		return nil, err
	}
	p.interpretable, err = p.interpreter.NewInterpretable(checked, decorators...)
	if err != nil {
		return nil, err
	}

	return p, nil
}

// Eval implements the Program interface method.
func (p *prog) Eval(input interface{}) (v ref.Val, det *EvalDetails, err error) {
	// Configure error recovery for unexpected panics during evaluation. Note, the use of named
	// return values makes it possible to modify the error response during the recovery
	// function.
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("internal error: %v", r)
		}
	}()
	// Build a hierarchical activation if there are default vars set.
	vars, err := interpreter.NewActivation(input)
	if err != nil {
		return
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

// Cost implements the Coster interface method.
func (p *prog) Cost() (min, max int64) {
	return estimateCost(p.interpretable)
}

// Eval implements the Program interface method.
func (gen *progGen) Eval(input interface{}) (ref.Val, *EvalDetails, error) {
	// The factory based Eval() differs from the standard evaluation model in that it generates a
	// new EvalState instance for each call to ensure that unique evaluations yield unique stateful
	// results.
	state := interpreter.NewEvalState()
	det := &EvalDetails{state: state}

	// Generate a new instance of the interpretable using the factory configured during the call to
	// newProgram(). It is incredibly unlikely that the factory call will generate an error given
	// the factory test performed within the Program() call.
	p, err := gen.factory(state)
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

// Cost implements the Coster interface method.
func (gen *progGen) Cost() (min, max int64) {
	// Use an empty state value since no evaluation is performed.
	p, err := gen.factory(emptyEvalState)
	if err != nil {
		return 0, math.MaxInt64
	}
	return estimateCost(p)
}

var (
	emptyEvalState = interpreter.NewEvalState()
)

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
