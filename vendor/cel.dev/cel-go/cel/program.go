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
	"errors"
	"fmt"
	"time"

	"cel.dev/cel-go/cel/async"
	"cel.dev/cel-go/common/ast"
	"cel.dev/cel-go/common/cost"
	"cel.dev/cel-go/common/functions"
	"cel.dev/cel-go/common/operators"
	"cel.dev/cel-go/common/overloads"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
	"cel.dev/cel-go/interpreter"
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

	// ConcurrentEval evaluates the program concurrently, returning a channel that will receive
	// the final EvalResult when all asynchronous operations complete, or the context expires.
	//
	// The vars value may either be an `Activation` or `map[string]any`.
	//
	// Liveness: ConcurrentEval relies on context cancellation to terminate. If an async function
	// never returns and does not honor its context, and the supplied context has no deadline, the
	// call will block indefinitely. Always pass a context with a deadline or cancellation.
	//
	// Error handling is fail-fast: as soon as a re-evaluation pass yields an error, that error is
	// returned and any still in-flight async calls are cancelled (their contexts are done) and
	// their results discarded. Async functions should therefore be free of unwanted side effects
	// on partial evaluation, or guard them with idempotency/cancellation handling.
	ConcurrentEval(context.Context, any) <-chan EvalResult
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

// FunctionActivation extends the Activation interface with implementations for the functions
// declared with a late binding.
type FunctionActivation = interpreter.FunctionActivation

// LateBoundFunction is the signature required of all late-bound function implementations.
//
// The overload id indicates which of the declared overloads matched the call arguments, which
// permits a single implementation to serve all overloads of the function.
//
// An implementation which depends on the state of the evaluation should capture that state in a
// closure, as the activation is not an input to the call.
type LateBoundFunction = functions.LateBoundOp

// FunctionVars returns a FunctionActivation combining the variable bindings from `vars` with
// the late-bound function implementations from `funcs`. FunctionVars support late-bound function
// definitions where the function behavior depends on the context of the evaluation. This can only
// be accomplished by closing over state that is not otherwise exposed to the CEL author or CEL
// runtime.
//
// - Function bindings occupy a namespace separate from variables, allowing a function and a
// variable to share the same qualified name.
// - The `vars` argument may be an Activation or any valid input to NewActivation (e.g. map[string]any).
// - FunctionVars may not be nested, and only one instance of FunctionVars may be supplied to an
// evaluation. FunctionVars returns an error if `vars` already contains function bindings.
//
// Function implementations can capture evaluation-scoped state via closures without exposing that
// state directly as expression variables:
//
//	roles := map[string]string{"tristan": "admin"}
//	act, err := cel.FunctionVars(map[string]any{"user": "tristan"}, map[string]cel.LateBoundFunction{
//	    "role": func(overloadID string, args ...ref.Val) ref.Val {
//	        role, found := roles[string(args[0].(types.String))]
//	        if !found {
//	            return types.NewErr("no role for user: %s", args[0])
//	        }
//	        return types.String(role)
//	    },
//	})
//
// CEL expects that late-bound functions, like regular functions, yield the same output given the same
// input. For example, a function comparing a validity window `valid_until(t)` could be implemented as
// a late-bound function closed over a fixed timestamp value.
func FunctionVars(vars any, funcs map[string]LateBoundFunction) (FunctionActivation, error) {
	return interpreter.NewFunctionActivation(vars, funcs)
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
	costTracker *cost.Tracker
	memTracker  *types.MemoryTracker
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

// PeakMemory returns the peak memory watermark observed through the course of execution when
// `MemoryTracking` is enabled. Otherwise, returns nil if memory tracking was not enabled.
func (ed *EvalDetails) PeakMemory() *uint32 {
	if ed == nil || ed.memTracker == nil {
		return nil
	}
	peak := ed.memTracker.Peak()
	return &peak
}

// EvalResult encapsulates the response from a ConcurrentEval call.
type EvalResult struct {
	Val         ref.Val
	EvalDetails *EvalDetails
	Err         error
}

// captureCostTracker records the frame's cost tracker within the evaluation details, allocating
// the details when cost tracking is enabled and no details have been created yet.
//
// The details must be fully populated before they are handed off to the caller, as any mutation
// after the hand-off would race with the caller's reads.
func captureCostTracker(det *EvalDetails, frame *interpreter.ExecutionFrame) *EvalDetails {
	tracker := frame.CostTracker()
	if tracker == nil {
		return det
	}
	if det == nil {
		det = &EvalDetails{}
	}
	det.costTracker = tracker
	return det
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
	interpretable     interpreter.InterpretableV2
	observable        *interpreter.ObservableInterpretable
	callCostEstimator cost.ActualCostEstimator
	costOptions       []cost.TrackerOption
	costLimit         *uint64
	memoryOptions     []types.MemoryTrackerOption
	memoryLimit       *uint32

	// hasAsync indicates the planned expression contains an asynchronous function call, which can
	// only be resolved by ConcurrentEval.
	hasAsync bool

	// Async evaluation configuration used by ConcurrentEval.
	drainStrategy             async.DrainStrategy
	asyncObserver             async.Observer
	asyncCompletionBufferSize int
	asyncMaxConcurrency       int
}

// scanOptTargets walks the AST once and reports whether the Optimize()
// decorator (needOpt) and the regex-constant compiler (needRegex) have any
// target node present. The conditions are a superset of what each decorator
// acts on, so a false negative — skipping a decorator that would have
// optimized something — is impossible.
func scanOptTargets(root ast.Expr) (needOpt, needRegex bool) {
	ast.PostOrderVisit(root, ast.NewExprVisitor(func(e ast.Expr) {
		switch e.Kind() {
		case ast.ListKind, ast.MapKind:
			needOpt = true // maybeBuildListLiteral / maybeBuildMapLiteral
		case ast.CallKind:
			switch fn := e.AsCall().FunctionName(); {
			case fn == overloads.Matches:
				needRegex = true
			case fn == operators.In || fn == operators.OldIn:
				needOpt = true // maybeOptimizeSetMembership
			case overloads.IsTypeConversionFunction(fn):
				needOpt = true // maybeOptimizeConstUnary
			}
		}
	}))
	return
}

// newProgram creates a program instance with an environment, an ast, and an optional list of
// ProgramOption values.
//
// If the program cannot be configured the prog will be nil, with a non-nil error response.
func newProgram(e *Env, a *ast.AST, opts []ProgramOption) (Program, error) {
	// Build the env's function bindings and shared dispatcher once (pure functions of the
	// env). The dispatcher holding the env's function bindings is identical across every
	// Program() built from it and read-only during planning — so assemble it once per env
	// and layer a thin child over it here for per-program Functions() isolation, rather than
	// re-indexing overloads on every Program() call.
	sharedDisp, hasAsync, err := e.initDispatcher()
	if err != nil {
		return nil, err
	}
	disp := interpreter.ExtendDispatcher(sharedDisp)

	// Ensure the default attribute factory is set after the adapter and provider are
	// configured.
	p := &prog{
		Env:            e,
		plannerOptions: []interpreter.PlannerOption{},
		dispatcher:     disp,
		costOptions:    []cost.TrackerOption{},
		drainStrategy:  async.DrainReady(100 * time.Microsecond),
		hasAsync:       hasAsync,
	}

	// Configure the program via the ProgramOption values.
	for _, opt := range opts {
		p, err = opt(p)
		if err != nil {
			return nil, err
		}
	}

	// Set the attribute factory after the options have been set.
	var attrFactory interpreter.AttributeFactory
	attrFactorOpts := []interpreter.AttrFactoryOption{
		interpreter.EnableErrorOnBadPresenceTest(p.HasFeature(featureEnableErrorOnBadPresenceTest)),
	}
	if a.SourceInfo().HasExtension("json_name", ast.NewExtensionVersion(1, 1)) {
		if !e.HasFeature(featureJSONFieldNames) {
			return nil, errors.New("the AST extension 'json_name' requires the option cel.JSONFieldNames(true)")
		}
	}
	// Configure the type provider, considering whether the AST indicates whether it supports JSON field names
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
		// The Optimize() decorator (set-membership, constant list/map literals,
		// const type conversions) and the regex-constant compiler each walk
		// every planned node. When the AST provably contains no node they can
		// act on, adding them is pure overhead — so gate each on a single AST
		// scan. The scan condition is a superset of what the decorators touch,
		// so a decorator is only skipped when its target node is definitely
		// absent and evaluation is never affected (an ungated regex would
		// recompile per eval, etc.).
		addOptimize, addRegex := scanOptTargets(a.Expr())
		if addOptimize {
			plannerOptions = append(plannerOptions, interpreter.Optimize())
		}
		if addRegex {
			p.regexOptimizations = append(p.regexOptimizations, interpreter.MatchesRegexOptimization)
		}
	}
	// Enable regex compilation of constants immediately after folding constants.
	if len(p.regexOptimizations) > 0 {
		plannerOptions = append(plannerOptions, interpreter.CompileRegexConstants(p.regexOptimizations...))
	}
	if limit := p.limits[limitRegexProgramSize]; limit > 0 {
		plannerOptions = append(plannerOptions, interpreter.RegexProgramSizeLimit(limit))
	}

	// Enable exhaustive eval, state tracking, cost tracking, and memory tracking last since they
	// require a factory.
	if p.evalOpts&(OptExhaustiveEval|OptTrackState|OptTrackCost|OptTrackMemory) != 0 {
		var observers []interpreter.PlannerOption
		if p.evalOpts&(OptExhaustiveEval|OptTrackState) != 0 {
			// EvalStateObserver is required for OptExhaustiveEval.
			observers = append(observers, interpreter.EvalStateObserver())
		}
		if p.evalOpts&OptTrackCost == OptTrackCost {
			trackerOpts := p.costModel.tracker
			costOptCount := len(trackerOpts) + len(p.costOptions) + 1
			if p.costLimit != nil {
				costOptCount++
			}
			costOpts := make([]cost.TrackerOption, 0, costOptCount)
			costOpts = append(costOpts, cost.TrackerLimitExceededHandler(func() {
				panic(interpreter.EvalCancelledError{
					Cause:   interpreter.CostLimitExceeded,
					Message: "operation cancelled: actual cost limit exceeded",
				})
			}))
			costOpts = append(costOpts, trackerOpts...)
			costOpts = append(costOpts, p.costOptions...)
			if p.costLimit != nil {
				costOpts = append(costOpts, cost.TrackerLimit(*p.costLimit))
			}
			// Creating a new cost tracker for each evaluation causes significant work that
			// needs to be repeated for each evaluation even though the cost tracker is
			// mostly read-only once constructed. Therefore it gets constructed
			// once now and later a cheap clone is used for each evaluation.
			tracker, err := cost.NewTracker(p.callCostEstimator, costOpts...)
			if err != nil {
				return nil, fmt.Errorf("construct cost tracker: %w", err)
			}
			trackerFactory := func() (*cost.Tracker, error) {
				return tracker.Clone()
			}
			plannerOptions = append(plannerOptions, interpreter.CostObserver(interpreter.CostTrackerFactory(trackerFactory)))
		}
		if p.evalOpts&OptTrackMemory == OptTrackMemory {
			memOptCount := len(p.memoryOptions)
			if p.memoryLimit != nil {
				memOptCount++
			}
			memOpts := make([]types.MemoryTrackerOption, 0, memOptCount)
			memOpts = append(memOpts, p.memoryOptions...)
			if p.memoryLimit != nil {
				memOpts = append(memOpts, types.MemoryTrackerLimit(*p.memoryLimit))
			}
			memTrackerFactory := func() (*types.MemoryTracker, error) {
				return types.NewMemoryTracker(memOpts...), nil
			}
			observers = append(observers, interpreter.MemoryObserver(interpreter.MemoryTrackerFactory(memTrackerFactory)))
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
	if p == nil {
		return nil, nil, errors.New("program is nil")
	}
	// Asynchronous calls cannot be resolved by a single-pass evaluation. Reject before doing any
	// work (this also covers ContextEval, which delegates here); ConcurrentEval does not call Eval.
	if p.hasAsync {
		return nil, nil, errAsyncRequiresConcurrentEval
	}
	// Build a hierarchical activation if there are default vars set.
	var frame *interpreter.ExecutionFrame
	var mustClose bool
	if f, ok := input.(*interpreter.ExecutionFrame); ok {
		frame = f
	} else {
		frame, err = p.newExecutionFrame(input)
		if err != nil {
			return nil, nil, err
		}
		mustClose = true
	}
	// Configure error recovery and details capture for evaluation.
	defer func() {
		det = captureCostTracker(det, frame)
		if r := recover(); r != nil {
			switch t := r.(type) {
			case interpreter.EvalCancelledError:
				err = t
			case cost.LimitExceededError:
				err = interpreter.EvalCancelledError{
					Cause:   interpreter.CostLimitExceeded,
					Message: t.Message,
				}
			default:
				err = fmt.Errorf("internal error: %v", r)
			}
		}
		if mustClose {
			frame.Close()
		}
	}()

	if p.observable != nil {
		det = &EvalDetails{}
		out = p.observable.ObserveExec(frame, func(observed any) {
			switch o := observed.(type) {
			case interpreter.EvalState:
				det.state = o
			case *types.MemoryTracker:
				det.memTracker = o
			}
		})
	} else {
		out = p.interpretable.Exec(frame)
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
	if p == nil {
		return nil, nil, errors.New("program is nil")
	}
	if ctx == nil {
		return nil, nil, errors.New("context can not be nil")
	}
	frame, err := p.newExecutionFrame(input)
	if err != nil {
		return nil, nil, err
	}
	defer frame.Close()
	frame.SetContext(ctx, p.interruptCheckFrequency)
	out, det, errEval := p.Eval(frame)
	if errEval != nil && errors.Is(errEval, interpreter.InterruptError{}) {
		return out, det, fmt.Errorf("%w: %w", errEval, context.Cause(ctx))
	}
	return out, det, errEval
}

// newExecutionFrame creates an ExecutionFrame for the given input without a timeout context.
func (p *prog) newExecutionFrame(input any) (*interpreter.ExecutionFrame, error) {
	frame, err := interpreter.NewExecutionFrame(input)
	if err != nil {
		return nil, err
	}
	if p.defaultVars != nil {
		if err := frame.SetDefaultVars(p.defaultVars); err != nil {
			frame.Close()
			return nil, err
		}
	}
	return frame, nil
}

// newAsyncFrame creates an ExecutionFrame configured for asynchronous evaluation under the
// given context, wiring the observer and concurrency limit from the program options.
func (p *prog) newAsyncFrame(ctx context.Context, input any) (*interpreter.ExecutionFrame, error) {
	frame, err := p.newExecutionFrame(input)
	if err != nil {
		return nil, err
	}
	if err := frame.SetContext(ctx, p.interruptCheckFrequency); err != nil {
		frame.Close()
		return nil, err
	}
	frame.SetAsyncObserver(p.asyncObserver)
	frame.SetAsyncMaxConcurrency(resolveAsyncMaxConcurrency(p.asyncMaxConcurrency))
	return frame, nil
}

// defaultAsyncMaxConcurrency bounds the number of concurrently launched async calls when the
// program does not configure AsyncMaxConcurrency. It exists so that a wide fan-out (e.g. an async
// call inside a comprehension over a large list) cannot spawn an unbounded number of goroutines.
const defaultAsyncMaxConcurrency = 100

// resolveAsyncMaxConcurrency maps the configured concurrency to the effective launch limit:
//   - 0 (unset): apply defaultAsyncMaxConcurrency.
//   - >0: use the configured value.
//   - <0: unlimited (no launch limiter); use only if the caller bounds concurrency another way.
func resolveAsyncMaxConcurrency(configured int) int {
	if configured == 0 {
		return defaultAsyncMaxConcurrency
	}
	return configured
}

// resolveCompletionBufferSize returns the size of the async completion channel. When unset, it
// defaults to the effective launch concurrency so that all in-flight calls can report completion
// without blocking. An unbuffered channel would make a completed call hold its launch slot until
// the evaluator drained it, throttling effective concurrency to the drain rate.
func (p *prog) resolveCompletionBufferSize() int {
	if p.asyncCompletionBufferSize > 0 {
		return p.asyncCompletionBufferSize
	}
	limit := resolveAsyncMaxConcurrency(p.asyncMaxConcurrency)
	if limit < 0 {
		// Unlimited launches: fall back to the default bound for the buffer so it stays finite.
		return defaultAsyncMaxConcurrency
	}
	return limit
}

// ConcurrentEval implements the Program interface.
func (p *prog) ConcurrentEval(ctx context.Context, input any) <-chan EvalResult {
	resCh := make(chan EvalResult, 1)
	if p == nil {
		resCh <- EvalResult{Err: errors.New("program is nil")}
		close(resCh)
		return resCh
	}
	if ctx == nil {
		resCh <- EvalResult{Err: errors.New("context can not be nil")}
		close(resCh)
		return resCh
	}

	go func() {
		defer close(resCh)
		frame, err := p.newAsyncFrame(ctx, input)
		if err != nil {
			resCh <- EvalResult{Err: err}
			return
		}
		defer frame.Close()

		var det *EvalDetails
		// Ensure concurrent eval handles panic / recovery properly.
		//
		// The details are only captured and published here for the panic case. Every normal
		// return path populates the details and sends them on the result channel itself, and
		// the receiver may read them as soon as the send completes. Mutating det from this
		// defer, which runs after the send, would race with those reads.
		defer func() {
			r := recover()
			if r == nil {
				return
			}
			det = captureCostTracker(det, frame)
			switch t := r.(type) {
			case interpreter.EvalCancelledError:
				resCh <- EvalResult{EvalDetails: det, Err: t}
			case cost.LimitExceededError:
				resCh <- EvalResult{
					EvalDetails: det,
					Err: interpreter.EvalCancelledError{
						Cause:   interpreter.CostLimitExceeded,
						Message: t.Message,
					},
				}
			default:
				resCh <- EvalResult{EvalDetails: det, Err: fmt.Errorf("internal error: %v", r)}
			}
		}()

		// Completions are signaled to this channel as async calls finish. The asyncCallState
		// fan-in also selects on ctx.Done(), so the sender will not leak if this loop returns early.
		completions := make(chan int64, p.resolveCompletionBufferSize())
		frame.SetCompletions(completions)

		for {
			var out ref.Val

			if p.observable != nil {
				det = &EvalDetails{}
				out = p.observable.ObserveExec(frame, func(observed any) {
					switch o := observed.(type) {
					case interpreter.EvalState:
						det.state = o
					case *types.MemoryTracker:
						det.memTracker = o
					}
				})
			} else {
				out = p.interpretable.Exec(frame)
			}
			// Capture the cost tracker before the result is published on the channel: once the
			// send completes the receiver owns the details and they must not be mutated.
			det = captureCostTracker(det, frame)

			// Communicate errors quickly.
			if types.IsError(out) {
				var err error = out.(*types.Err)
				if errors.Is(err, interpreter.InterruptError{}) {
					err = fmt.Errorf("%w: %w", err, context.Cause(ctx))
				}
				resCh <- EvalResult{Val: out, EvalDetails: det, Err: err}
				return
			}

			// A concrete (non-unknown) result is final.
			unk, isUnknown := out.(*types.Unknown)
			if !isUnknown || !unk.HasUnknownFunction() {
				resCh <- EvalResult{Val: out, EvalDetails: det, Err: nil}
				return
			}

			// Post-execution dispatch: launch only the async calls required by the unknown result.
			frame.DispatchPendingAsyncCalls(unk.IDs())

			// The result depends on one or more unresolved async calls. Wait for completions and
			// re-evaluate according to the configured drain strategy.
			var batch []async.Call

			// Wait for at least one completion (or cancellation).
			select {
			case id := <-completions:
				if call := frame.AsyncCall(id); call != nil {
					batch = append(batch, call)
				}
			case <-ctx.Done():
				resCh <- EvalResult{Val: out, EvalDetails: det, Err: ctx.Err()}
				return
			}

			// Accumulate completions and consult the strategy.
			var timer *time.Timer
			reevaluate := false
			for !reevaluate {
				active := frame.ActiveAsyncCalls()
				action := p.drainStrategy.NextAction(batch, active)
				if action.Reevaluate {
					break
				}

				var timeoutCh <-chan time.Time
				if action.WaitDuration > 0 {
					if timer == nil {
						timer = time.NewTimer(action.WaitDuration)
					} else {
						if !timer.Stop() {
							select {
							case <-timer.C:
							default:
							}
						}
						timer.Reset(action.WaitDuration)
					}
					timeoutCh = timer.C
				}

				select {
				case id := <-completions:
					if call := frame.AsyncCall(id); call != nil {
						batch = append(batch, call)
					}
				case <-timeoutCh:
					reevaluate = true
				case <-ctx.Done():
					if timer != nil {
						timer.Stop()
					}
					resCh <- EvalResult{Val: out, EvalDetails: det, Err: ctx.Err()}
					return
				}
			}
			if timer != nil {
				timer.Stop()
			}
		}
	}()

	return resCh
}

// errAsyncRequiresConcurrentEval is returned by the synchronous entry points (Eval, ContextEval)
// when the expression contains asynchronous function calls, which only ConcurrentEval can resolve.
var errAsyncRequiresConcurrentEval = errors.New(
	"expression contains asynchronous function calls; use ConcurrentEval")
