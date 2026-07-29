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
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"sync"
	"sync/atomic"

	"github.com/google/cel-go/common/functions"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// Async extension function support.
//
// CEL supports `types.Unknown` as a first-class value, and concurrent (async) function execution
// in CEL invokes a stub function which checks for the presence of an existing result which matches
// the function call and call arguments, or which records the 'unexecuted' function and call arguments
// for concurrent execution in a later phase if the result is `types.Unknown` and indicates the
// expression ids of the functions necessary to advance the execution.
//
// This call pattern is repeated iteratively until there are either no more functions to call or no
// progress is made toward resolving the unknowns.

// AsyncObserver provides callbacks for monitoring the lifecycle of asynchronous function calls.
//
// Implementations must be safe for concurrent use: OnCallStarted is invoked from the evaluator
// goroutine when a call is launched, while OnCallFinished is invoked from the call's own goroutine
// when it completes. The two callbacks therefore run on different goroutines, and OnCallFinished
// callbacks for distinct calls may run concurrently with each other.
type AsyncObserver interface {
	// OnCallStarted is called when an asynchronous function is first launched.
	OnCallStarted(callID int64, function, overload string, args []ref.Val)
	// OnCallFinished is called when an asynchronous function completes.
	OnCallFinished(callID int64, function, overload string, res ref.Val)
}

// AsyncCall describes a pending or completed asynchronous function call.
type AsyncCall interface {
	// CallID returns the unique identifier for this async call invocation.
	CallID() int64
	// Function returns the name of the function being called.
	Function() string
	// Overload returns the specific overload ID being invoked.
	Overload() string
}

// evalAsyncFunc is the planned Interpretable for an asynchronous function call.
type evalAsyncFunc struct {
	id       int64
	function string
	overload string
	args     []InterpretableV2
	impl     functions.AsyncOp
}

// ID implements the Interpretable interface method.
func (fn *evalAsyncFunc) ID() int64 {
	return fn.id
}

// Function returns the name of the function being invoked.
func (fn *evalAsyncFunc) Function() string {
	return fn.function
}

// OverloadID returns the overload id of the function being invoked.
func (fn *evalAsyncFunc) OverloadID() string {
	return fn.overload
}

// Args returns the argument Interpretables for the function call.
func (fn *evalAsyncFunc) Args() []InterpretableV2 {
	return fn.args
}

// Eval implements the Interpretable interface method.
func (fn *evalAsyncFunc) Eval(vars Activation) ref.Val {
	return fn.Exec(AsFrame(vars))
}

// Exec implements the InterpretableV2 interface method.
func (fn *evalAsyncFunc) Exec(frame *ExecutionFrame) ref.Val {
	argVals := make([]ref.Val, len(fn.args))
	var unk *types.Unknown
	for i, arg := range fn.args {
		argVals[i] = arg.Exec(frame)
		if types.IsError(argVals[i]) {
			return argVals[i]
		}
		unk, _ = types.MaybeMergeUnknowns(argVals[i], unk)
	}
	if unk != nil {
		return unk
	}
	result := frame.ComputeResult(fn.ID(), fn.Function(), fn.OverloadID(), fn.impl, argVals)
	return types.LabelErrNode(fn.id, result)
}

// asyncCallStateTracker manages async call states across re-evaluations of a single program.
type asyncCallStateTracker struct {
	mu sync.RWMutex
	// calls buckets call states by a composite hash of (node id, overload, string/int/double/uint/bool args).
	// A single AST node id may host many concurrently-live calls when it is evaluated inside a
	// comprehension (once per element with different arguments), so each bucket may hold more
	// than one state. The exact match within a bucket is resolved via asyncCallState.matches,
	// which applies CEL's full equality semantics to the arguments.
	calls      map[uint64][]*asyncCallState
	callsByID  map[int64]*asyncCallState
	nextCallID atomic.Int64
}

func newAsyncCallStateTracker() *asyncCallStateTracker {
	return &asyncCallStateTracker{
		calls:     make(map[uint64][]*asyncCallState),
		callsByID: make(map[int64]*asyncCallState),
	}
}

var (
	hashZeroMarker      = []byte{0}
	hashStringMarker    = []byte{'s'}
	hashBoolTrueMarker  = []byte{'b', 1}
	hashBoolFalseMarker = []byte{'b', 0}
	hashNumberMarker    = []byte{'n'}
	hashDefaultMarker   = []byte{'x'}
)

// hashCall computes the composite bucket key for an async call.
//
// Only string, int, double, uint, and bool argument values contribute to the hash. More complex types
// rely on a richer notion of equivalence (e.g. unordered maps, proto equality, custom types)
// that a byte-level hash cannot capture safely, so they are intentionally excluded from the key
// and are instead disambiguated within the bucket by asyncCallState.matches.
func hashCall(id int64, overload string, args []ref.Val) uint64 {
	h := fnv.New64a()
	var idBuf [8]byte
	binary.LittleEndian.PutUint64(idBuf[:], uint64(id))
	h.Write(idBuf[:])
	h.Write([]byte(overload))
	h.Write(hashZeroMarker)
	for _, arg := range args {
		switch v := arg.(type) {
		case types.String:
			h.Write(hashStringMarker)
			h.Write([]byte(string(v)))
		case types.Bool:
			if bool(v) {
				h.Write(hashBoolTrueMarker)
			} else {
				h.Write(hashBoolFalseMarker)
			}
		case types.Int:
			h.Write(hashNumberMarker)
			var buf [8]byte
			binary.LittleEndian.PutUint64(buf[:], math.Float64bits(float64(v)))
			h.Write(buf[:])
		case types.Uint:
			h.Write(hashNumberMarker)
			var buf [8]byte
			binary.LittleEndian.PutUint64(buf[:], math.Float64bits(float64(v)))
			h.Write(buf[:])
		case types.Double:
			h.Write(hashNumberMarker)
			if math.IsNaN(float64(v)) {
				h.Write([]byte("NaN"))
				h.Write(hashZeroMarker)
				continue
			}
			// Normalize -0.0 to 0.0. Go will treat -0.0 as 0.0 at compile time,
			// but the function math.Copysign(0.0, -1.0) can be used to test the -0.0 case.
			if v == types.Double(0.0) && math.Signbit(float64(v)) {
				v = types.Double(0.0)
			}
			var buf [8]byte
			binary.LittleEndian.PutUint64(buf[:], math.Float64bits(float64(v)))
			h.Write(buf[:])
		default:
			// Value intentionally omitted; bucket membership falls back to matches.
			h.Write(hashDefaultMarker)
		}
		// Separator to avoid cross-argument collisions, e.g. ("a", "bc") vs ("ab", "c").
		h.Write(hashZeroMarker)
	}
	return h.Sum64()
}

// findInBucket returns the call state in the bucket matching the same node id and call identity,
// or nil if no match is present.
func findInBucket(bucket []*asyncCallState, id int64, function, overload string, args []ref.Val) *asyncCallState {
	for _, acs := range bucket {
		if acs.matches(id, function, overload, args) {
			return acs
		}
	}
	return nil
}

// getOrCreate returns the existing call state for the (node id, args) tuple, or registers and
// returns a new one. A newly registered call is assigned a unique callID and counted as pending.
func (t *asyncCallStateTracker) getOrCreate(id int64, function, overload string, argVals []ref.Val, impl functions.AsyncOp, gate *asyncGate) *asyncCallState {
	key := hashCall(id, overload, argVals)

	t.mu.RLock()
	acs := findInBucket(t.calls[key], id, function, overload, argVals)
	t.mu.RUnlock()
	if acs != nil {
		return acs
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	// Check again in case it was created while waiting for the lock.
	if acs := findInBucket(t.calls[key], id, function, overload, argVals); acs != nil {
		return acs
	}

	// Assign a new unique call ID for this async call.
	acs = newAsyncCallState(id, function, overload, argVals, impl)
	callID := t.nextCallID.Add(1)
	acs.callID = callID
	acs.gate = gate
	t.calls[key] = append(t.calls[key], acs)
	t.callsByID[callID] = acs
	return acs
}

func (t *asyncCallStateTracker) getByID(callID int64) *asyncCallState {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.callsByID[callID]
}

func newAsyncCallState(id int64, function, overload string, argVals []ref.Val, impl functions.AsyncOp) *asyncCallState {
	return &asyncCallState{
		id:       id,
		function: function,
		overload: overload,
		argVals:  argVals,
		impl:     impl,
	}
}

// asyncCallState tracks the result of a single async function call across multiple re-evaluations.
type asyncCallState struct {
	id       int64 // AST expression node ID where the call is defined.
	callID   int64 // Unique incremental tracking ID assigned to this call.
	function string
	overload string
	argVals  []ref.Val
	impl     functions.AsyncOp

	mu      sync.RWMutex
	started bool
	result  ref.Val

	gate *asyncGate
}

// CallID returns the unique identifier for this async call invocation.
func (acs *asyncCallState) CallID() int64 {
	return acs.callID
}

// Function returns the name of the function being called.
func (acs *asyncCallState) Function() string {
	return acs.function
}

// Overload returns the specific overload ID being invoked.
func (acs *asyncCallState) Overload() string {
	return acs.overload
}

// ResultOrUnknown returns the cached result if the call has completed, an Unknown
// with the call ID if pending, or nil if the call has not been started.
func (acs *asyncCallState) ResultOrUnknown() ref.Val {
	if acs == nil {
		return nil
	}
	acs.mu.RLock()
	defer acs.mu.RUnlock()
	if acs.result == nil && acs.started {
		return types.NewUnknown(acs.callID, nil)
	}
	return acs.result
}

// SetResult sets the completed result for an asynchronous function call.
func (acs *asyncCallState) SetResult(res ref.Val) {
	if acs == nil {
		return
	}
	acs.mu.Lock()
	defer acs.mu.Unlock()
	acs.result = res
}

// launch returns a call's cached result, or starts the call (subject to the launch limiter) and
// returns an Unknown referencing its callID while the result is pending.
//
// Admission control: when a concurrency semaphore is configured, a launch slot is reserved with a
// non-blocking send. If no slot is free the call is left unstarted and an Unknown is returned; the
// call is retried on a later re-evaluation pass once an in-flight call completes and frees a slot.
// The reservation is non-blocking on purpose — the evaluator runs on a single goroutine, and
// blocking it here while completing calls block on an undrained completion channel would deadlock.
// The slot is held by the launched goroutine and released when it exits, so the number of live
// async goroutines is bounded by the semaphore capacity.
func (t *asyncCallStateTracker) launch(ctx context.Context, acs *asyncCallState, observer AsyncObserver) ref.Val {
	if res := acs.ResultOrUnknown(); res != nil {
		return res
	}
	gate := acs.gate
	if !gate.TryAcquire() {
		return types.NewUnknown(acs.callID, nil)
	}
	acs.mu.Lock()
	if acs.started || acs.result != nil {
		// Defensive: the evaluator is single-threaded so this should not happen, but if it does,
		// return the reserved slot rather than leak it.
		acs.mu.Unlock()
		gate.Release()
		return types.NewUnknown(acs.callID, nil)
	}
	acs.started = true
	acs.mu.Unlock()

	if observer != nil {
		observer.OnCallStarted(acs.callID, acs.function, acs.overload, acs.argVals)
	}
	go func() {
		defer func() {
			if observer != nil {
				observer.OnCallFinished(acs.callID, acs.function, acs.overload, acs.ResultOrUnknown())
			}
			gate.Complete(ctx, acs.callID)
		}()

		ch := acs.impl(ctx, acs.argVals...)
		// Early terminate with a CEL error when an implementation returns an empty channel.
		if ch == nil {
			acs.SetResult(types.NewErrFromString(
				fmt.Sprintf("function %s returned an empty channel", acs.function)))
			return
		}
		// Wait for the async computation to finish or for the context to be cancelled.
		select {
		case r, ok := <-ch:
			if !ok {
				acs.SetResult(types.NewErrFromString(
					fmt.Sprintf("function %s returned an empty channel", acs.function)))
				return
			}
			acs.SetResult(r)
		case <-ctx.Done():
			// Evaluation context cancelled before the async operation completed.
			acs.SetResult(types.WrapErr(context.Cause(ctx)))
		}
	}()
	return types.NewUnknown(acs.callID, nil)
}

// matches reports whether two call states refer to the same function, overload, and arguments.
func (acs *asyncCallState) matches(id int64, function, overload string, args []ref.Val) bool {
	if acs == nil {
		return false
	}
	if acs.id != id || acs.function != function || acs.overload != overload {
		return false
	}
	if len(acs.argVals) != len(args) {
		return false
	}
	for i, v := range acs.argVals {
		otherV := args[i]
		if types.Equal(v, otherV) == types.True {
			continue
		}
		if n, ok := v.(types.Double); ok {
			// Treat NaN as equivalent for the sake of function dispatch equality.
			if otherN, ok := otherV.(types.Double); ok && math.IsNaN(float64(n)) && math.IsNaN(float64(otherN)) {
				continue
			}
		}
		return false
	}
	return true
}

// trackerShrinkThreshold is the entry count above which a released tracker's maps are reallocated
// rather than cleared in place, so the pool does not retain a large backing array indefinitely.
const trackerShrinkThreshold = 1024

// asyncCallStateTrackerPool provides a synchronized pool of asyncCallStateTrackers.
type asyncCallTrackerPool struct {
	sync.Pool
}

func (pool *asyncCallTrackerPool) create() *asyncCallStateTracker {
	return pool.Get().(*asyncCallStateTracker)
}

func (pool *asyncCallTrackerPool) release(tracker *asyncCallStateTracker) {
	if tracker == nil {
		return
	}
	tracker.mu.Lock()
	// Clearing with delete reuses the backing arrays, which is ideal for the common case but pins
	// a large allocation in the pool after a wide fan-out (e.g. an async call over a big list).
	// Past a threshold, reallocate so the high-water-mark memory is released to the GC instead of
	// being retained by the pooled tracker.
	if len(tracker.calls) > trackerShrinkThreshold || len(tracker.callsByID) > trackerShrinkThreshold {
		tracker.calls = make(map[uint64][]*asyncCallState)
		tracker.callsByID = make(map[int64]*asyncCallState)
	} else {
		for k := range tracker.calls {
			delete(tracker.calls, k)
		}
		for k := range tracker.callsByID {
			delete(tracker.callsByID, k)
		}
	}
	tracker.nextCallID.Store(0)
	tracker.mu.Unlock()
	pool.Pool.Put(tracker)
}

func newAsyncCallTrackerPool() *asyncCallTrackerPool {
	return &asyncCallTrackerPool{
		Pool: sync.Pool{
			New: func() any {
				return newAsyncCallStateTracker()
			},
		},
	}
}

var asyncCallStateTrackerPool = newAsyncCallTrackerPool()

// asyncGate coordinates async call admission control and completion signaling.
type asyncGate struct {
	semaphore   chan struct{}
	completions chan<- int64
	activeCalls atomic.Int32
}

func newAsyncGate(maxConcurrency int, completions chan<- int64) *asyncGate {
	var sem chan struct{}
	if maxConcurrency > 0 {
		sem = make(chan struct{}, maxConcurrency)
	}
	return &asyncGate{
		semaphore:   sem,
		completions: completions,
	}
}

// TryAcquire attempts to acquire a concurrency slot and increments the active calls count.
func (g *asyncGate) TryAcquire() bool {
	if g == nil {
		return true
	}
	if g.semaphore != nil {
		select {
		case g.semaphore <- struct{}{}:
		default:
			return false
		}
	}
	g.activeCalls.Add(1)
	return true
}

// Release releases a concurrency slot and decrements the active calls count (used for defensive recovery).
func (g *asyncGate) Release() {
	if g == nil {
		return
	}
	if g.semaphore != nil {
		select {
		case <-g.semaphore:
		default:
		}
	}
	g.activeCalls.Add(-1)
}

// Complete releases a concurrency slot and notifies completions.
func (g *asyncGate) Complete(ctx context.Context, callID int64) {
	if g == nil {
		return
	}
	g.Release()

	if g.completions != nil {
		// Prioritize context cancellation to prevent racy completion signals.
		if ctx.Err() != nil {
			return
		}
		select {
		case g.completions <- callID:
		case <-ctx.Done():
		}
	}
}

// ActiveCalls returns the number of active asynchronous calls.
func (g *asyncGate) ActiveCalls() int {
	if g == nil {
		return 0
	}
	return int(g.activeCalls.Load())
}
