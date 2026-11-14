// Copyright 2026 Google LLC
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
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"cel.dev/cel-go/common/functions"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
)

// evalContext contains the stateful information needed for a single evaluation.
//
// This state is shared across all frames within a single evaluation, including
// child frames created for comprehension blocks.
type evalContext struct {
	// interrupt exposes a callback channel for cancellation.
	interrupt <-chan struct{}

	// interruptCheckCount is the number of times the interrupt channel has been checked.
	interruptCheckCount atomic.Uint64

	// interruptCheckFrequency is the frequency at which the interrupt channel is checked.
	interruptCheckFrequency uint

	// interrupted indicates whether the evaluation has been interrupted.
	interrupted atomic.Bool

	// state provides the context for tracking the evaluation state.
	state EvalState

	// costs provides the context for tracking the evaluation costs.
	costs *CostTracker

	// memory provides the context for tracking peak memory during evaluation.
	memory *types.MemoryTracker

	// ctx is the context for async call implementations to use.
	ctx context.Context

	// cancel cancels the context when the evaluation is finished.
	cancel context.CancelFunc

	// asyncCalls tracks the state of async call invocations across re-evaluations.
	asyncCalls *asyncCallStateTracker

	// gate coordinates async call admission control and completion signaling.
	gate *asyncGate

	// observer for monitoring async calls.
	observer AsyncObserver
}

// ExecutionFrame provides the context for a single evaluation of an expression.
//
// The execution frame must not be stored in any fashion as its lifecycle is completely
// controlled by the CEL evaluation process.
type ExecutionFrame struct {
	// parent provides the context for parent scopes (used for comprehension iterators and nested blocks).
	parent *ExecutionFrame

	// scope provides the local activation for this frame (e.g. comprehension folder, block slots, or input Activation).
	scope Activation

	// functions supplies the late-bound function implementations for the evaluation, resolved
	// once from the input activation. Scopes introduced during evaluation never supply function
	// bindings, so a child frame inherits this value from its parent rather than resolving again.
	functions FunctionActivation

	// vars holds map-based input variables directly on the frame to eliminate pool allocations.
	vars map[string]any

	// lazyVars caches evaluations of lazy variables (func() any / func() ref.Val) in vars.
	lazyVars map[string]any

	// ctx provides the shared evaluation state across frames.
	ctx *evalContext

	// costTracker provides direct access to the active CostTracker for this evaluation pass.
	costTracker *CostTracker
}

// Scope returns the local activation scope for this frame, if present.
func (f *ExecutionFrame) Scope() Activation {
	return f.scope
}

// SetScope sets the local activation scope for this frame.
func (f *ExecutionFrame) SetScope(scope Activation) {
	f.scope = scope
}

// SetDefaultVars sets the default variables activation for the frame, composing it
// with any existing scope activation.
func (f *ExecutionFrame) SetDefaultVars(defaultVars Activation) error {
	if defaultVars == nil {
		return nil
	}
	fns, err := FindFunctionActivation(defaultVars)
	if err != nil {
		return err
	}
	if fns != nil {
		if f.functions != nil {
			return errNestedFunctionActivation
		}
		f.functions = fns
	}
	if f.scope != nil {
		f.scope = NewHierarchicalActivation(defaultVars, f.scope)
	} else {
		f.scope = defaultVars
	}
	return nil
}

// NewExecutionFrame creates a new execution frame from the pool.
func NewExecutionFrame(input any) (*ExecutionFrame, error) {
	f := frameStack.Get().(*ExecutionFrame)
	switch v := input.(type) {
	case emptyActivation:
		// empty frame, no backing scope needed
	case Activation:
		f.scope = v
		fns, err := FindFunctionActivation(f.scope)
		if err != nil {
			frameStack.Put(f)
			return nil, err
		}
		f.functions = fns
	case map[string]any:
		f.vars = v
	default:
		frameStack.Put(f)
		return nil, fmt.Errorf("invalid input, wanted Activation or map[string]any, got: (%T)%v", input, input)
	}
	return f, nil
}

// SetContext sets the context for the execution frame.
func (f *ExecutionFrame) SetContext(ctx context.Context, interruptCheckFrequency uint) error {
	if f.parent != nil {
		return errors.New("SetContext() called on child frame")
	}
	if f.ctx != nil {
		return errors.New("SetContext() called more than once")
	}
	f.ctx = evalContextPool.Get().(*evalContext)
	f.ctx.ctx, f.ctx.cancel = context.WithCancel(ctx)
	f.ctx.asyncCalls = asyncCallStateTrackerPool.create()
	f.ctx.gate = &asyncGate{}
	f.ctx.interrupt = ctx.Done()
	f.ctx.interruptCheckFrequency = interruptCheckFrequency
	f.ctx.interruptCheckCount.Store(0)
	f.ctx.interrupted.Store(false)
	return nil
}

// Close releases the resources held by the execution frame and returns it to the pool.
func (f *ExecutionFrame) Close() {
	if f.parent == nil && f.ctx != nil {
		if f.ctx.cancel != nil {
			f.ctx.cancel()
			f.ctx.cancel = nil
		}
		f.ctx.ctx = nil
		f.ctx.gate = nil
		asyncCallStateTrackerPool.release(f.ctx.asyncCalls)
		f.ctx.asyncCalls = nil
		f.ctx.observer = nil
		f.ctx.interrupt = nil
		f.ctx.state = nil
		f.ctx.costs = nil
		f.ctx.memory = nil
		f.ctx.interrupted.Store(false)
		f.ctx.interruptCheckCount.Store(0)
		f.ctx.interruptCheckFrequency = 0
		evalContextPool.Put(f.ctx)
	}
	f.ctx = nil
	f.parent = nil
	f.costTracker = nil
	f.functions = nil
	if f.vars != nil {
		f.vars = nil
		clear(f.lazyVars)
	}
	f.scope = nil
	frameStack.Put(f)
}

// Push pushes the given activation onto the activation stack and returns the new frame.
//
// This operation is internal to the interpreter and is used to handle comprehension
// and block scoping. The child frame inherits the shared evalContext from the parent.
func (f *ExecutionFrame) Push(activation Activation) *ExecutionFrame {
	child := frameStack.Get().(*ExecutionFrame)
	child.parent = f
	child.ctx = f.ctx
	child.costTracker = f.costTracker
	child.scope = activation
	// Scopes pushed during evaluation never supply late-bound functions, so the child inherits
	// the bindings resolved for the evaluation rather than searching its own hierarchy.
	child.functions = f.functions
	return child
}

// Pop returns the parent frame, releasing the current frame back to the pool.
func (f *ExecutionFrame) Pop() *ExecutionFrame {
	if f.parent == nil {
		return f
	}
	parent := f.parent
	f.scope = nil
	f.parent = nil
	f.functions = nil
	f.ctx = nil
	f.costTracker = nil
	if f.vars != nil {
		f.vars = nil
		clear(f.lazyVars)
	}
	frameStack.Put(f)
	return parent
}

// ResolveName implements the Activation interface by proxying to the internal activation.
func (f *ExecutionFrame) ResolveName(name string) (any, bool) {
	if f.vars != nil {
		v, found := f.vars[name]
		if found {
			if f.lazyVars != nil {
				if resolved, found := f.lazyVars[name]; found {
					return resolved, true
				}
			}
			var lazy any
			switch obj := v.(type) {
			case func() ref.Val:
				lazy = obj()
			case func() any:
				lazy = obj()
			default:
				return obj, true
			}
			if f.lazyVars == nil {
				f.lazyVars = make(map[string]any, 4)
			}
			f.lazyVars[name] = lazy
			return lazy, true
		}
	}
	if f.scope != nil {
		if val, found := f.scope.ResolveName(name); found {
			return val, true
		}
	}
	if f.parent != nil {
		return f.parent.ResolveName(name)
	}
	return nil, false
}

// Parent implements the Activation interface by proxying to the parent frame or scope.
func (f *ExecutionFrame) Parent() Activation {
	if f.parent != nil {
		if f.parent.scope != nil {
			return f.parent.scope
		}
		return f.parent
	}
	if f.scope != nil {
		return f.scope.Parent()
	}
	return nil
}

// ResolveFunction implements the FunctionActivation interface using the bindings resolved when
// the evaluation began, so the cost of a lookup does not depend on the number of enclosing scopes.
func (f *ExecutionFrame) ResolveFunction(name string) (functions.LateBoundOp, bool) {
	if f.functions == nil {
		return nil, false
	}
	return f.functions.ResolveFunction(name)
}

// AsPartialActivation implements the PartialActivation interface by proxying to the internal scope or parent.
func (f *ExecutionFrame) AsPartialActivation() (PartialActivation, bool) {
	if f.scope != nil {
		if pa, ok := AsPartialActivation(f.scope); ok {
			return pa, true
		}
	}
	if f.parent != nil {
		return f.parent.AsPartialActivation()
	}
	return nil, false
}

// UnknownAttributePatterns implements the PartialActivation interface returning the unknown patterns
// if they were provided to the input activation, or an empty set if the frame is not partial.
func (f *ExecutionFrame) UnknownAttributePatterns() []*AttributePattern {
	if pa, ok := f.AsPartialActivation(); ok {
		return pa.UnknownAttributePatterns()
	}
	return []*AttributePattern{}
}

// Unwrap returns the local activation scope if present, or the parent frame.
func (f *ExecutionFrame) Unwrap() Activation {
	if f.scope != nil {
		return f.scope
	}
	if f.parent != nil {
		return f.parent
	}
	return nil
}

// IsLocalVariable reports whether the variable name is locally bound in the frame.
func (f *ExecutionFrame) IsLocalVariable(name string) bool {
	if holder, ok := f.scope.(localVariableHolder); ok {
		if holder.IsLocalVariable(name) {
			return true
		}
	}
	// Search parent scopes
	if f.parent != nil {
		return f.parent.IsLocalVariable(name)
	}
	return false
}

// CheckInterrupt returns whether the evaluation has been interrupted.
func (f *ExecutionFrame) CheckInterrupt() bool {
	if f.ctx == nil {
		return false
	}
	if f.ctx.interrupted.Load() {
		return true
	}
	count := f.ctx.interruptCheckCount.Add(1)
	if f.ctx.interruptCheckFrequency > 0 && count%uint64(f.ctx.interruptCheckFrequency) == 0 {
		select {
		case <-f.ctx.interrupt:
			f.ctx.interrupted.Store(true)
			return true
		default:
			return false
		}
	}
	return false
}

// CostTracker returns the active CostTracker for this evaluation pass, or nil.
func (f *ExecutionFrame) CostTracker() *CostTracker {
	if f == nil {
		return nil
	}
	return f.costTracker
}

// SetCostTracker sets the active CostTracker for this evaluation pass.
func (f *ExecutionFrame) SetCostTracker(tracker *CostTracker) {
	if f.ctx == nil {
		f.ctx = evalContextPool.Get().(*evalContext)
	}
	f.ctx.costs = tracker
	f.costTracker = tracker
}

// ComputeResult tracks and computes the result of the given asynchronous function.
//
// The first invocation for a given (node id, args) tuple registers the call state and returns an
// Unknown which references the call's unique callID. Subsequent invocations return the cached
// result once the call has completed. Launching background execution is deferred to post-execution
// dispatch via DispatchPendingAsyncCalls.
func (f *ExecutionFrame) ComputeResult(id int64, function, overload string, impl functions.AsyncOp, argVals []ref.Val) ref.Val {
	if f.ctx == nil || f.ctx.asyncCalls == nil {
		return types.NewErrWithNodeID(id, "asynchronous function calls require concurrent evaluation and cannot be resolved by a synchronous Eval")
	}
	t := f.ctx.asyncCalls
	acs := t.getOrCreate(id, function, overload, argVals, impl, f.ctx.gate)
	if res := acs.ResultOrUnknown(); res != nil {
		return res
	}
	return types.NewUnknown(acs.callID, nil)
}

// DispatchPendingAsyncCalls launches pending asynchronous calls for the specified required call IDs.
func (f *ExecutionFrame) DispatchPendingAsyncCalls(callIDs []int64) {
	if f.ctx == nil || f.ctx.asyncCalls == nil {
		return
	}
	t := f.ctx.asyncCalls
	for _, callID := range callIDs {
		if acs := t.getByID(callID); acs != nil {
			t.launch(f.ctx.ctx, acs, f.ctx.observer)
		}
	}
}

// ActiveAsyncCalls returns the number of async function calls that have been launched
// but whose completions have not yet been drained.
func (f *ExecutionFrame) ActiveAsyncCalls() int {
	if f.ctx == nil || f.ctx.gate == nil {
		return 0
	}
	return f.ctx.gate.ActiveCalls()
}

// AsyncCall returns the state of an async call by its callID, or nil if not found.
func (f *ExecutionFrame) AsyncCall(callID int64) AsyncCall {
	if f.ctx == nil || f.ctx.asyncCalls == nil {
		return nil
	}
	acs := f.ctx.asyncCalls.getByID(callID)
	if acs == nil {
		return nil
	}
	return acs
}

// SetCompletions configures a channel to receive callIDs when asynchronous evaluations finish.
func (f *ExecutionFrame) SetCompletions(ch chan<- int64) error {
	if f.ctx == nil {
		return errors.New("asynchronous evaluation options require the execution frame to have a context configured")
	}
	f.ctx.gate.completions = ch
	return nil
}

// SetAsyncObserver sets the observer for monitoring asynchronous function calls.
func (f *ExecutionFrame) SetAsyncObserver(observer AsyncObserver) error {
	if f.ctx == nil {
		return errors.New("asynchronous evaluation options require the execution frame to have a context configured")
	}
	f.ctx.observer = observer
	return nil
}

// SetAsyncMaxConcurrency sets the maximum concurrency for asynchronous function calls.
//
// A non-positive value indicates that concurrency is unbounded.
func (f *ExecutionFrame) SetAsyncMaxConcurrency(n int) error {
	if f.ctx == nil {
		return errors.New("asynchronous evaluation options require the execution frame to have a context configured")
	}
	if n > 0 {
		f.ctx.gate.semaphore = make(chan struct{}, n)
	} else {
		f.ctx.gate.semaphore = nil
	}
	return nil
}

// frameStack provides a synchronized pool of ExecutionFrames.
var frameStack = &sync.Pool{
	New: func() any {
		return &ExecutionFrame{}
	},
}

// evalContextPool provides a synchronized pool of evalContexts.
var evalContextPool = &sync.Pool{
	New: func() any {
		return &evalContext{}
	},
}
