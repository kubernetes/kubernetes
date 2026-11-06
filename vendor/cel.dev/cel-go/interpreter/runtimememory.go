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
	"errors"

	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
)

// memoryTrackPlanOption modifies the memory tracking factory associated with the MemoryObserver.
type memoryTrackPlanOption func(*memoryTrackerFactory) *memoryTrackerFactory

// MemoryTrackerFactory configures the factory method to generate a new memory-tracker
// per-evaluation.
func MemoryTrackerFactory(factory func() (*types.MemoryTracker, error)) memoryTrackPlanOption {
	return func(fac *memoryTrackerFactory) *memoryTrackerFactory {
		fac.factory = factory
		return fac
	}
}

// MemoryObserver provides an observer that tracks runtime peak memory.
func MemoryObserver(opts ...memoryTrackPlanOption) PlannerOption {
	mt := &memoryTrackerFactory{}
	for _, o := range opts {
		mt = o(mt)
	}
	return func(p *planner) (*planner, error) {
		if mt.factory == nil {
			return nil, errors.New("memory tracker factory not configured")
		}
		p.observers = append(p.observers, mt)
		return p, nil
	}
}

// memoryTrackerFactory holds a factory for producing new MemoryTracker instances on each Eval call.
type memoryTrackerFactory struct {
	factory func() (*types.MemoryTracker, error)
}

// InitState produces a MemoryTracker and bundles it into the ExecutionFrame in a way which is
// not visible to expression evaluation.
func (mt *memoryTrackerFactory) InitState(frame *ExecutionFrame) (any, error) {
	if frame.ctx != nil && frame.ctx.memory != nil {
		return frame.ctx.memory, nil
	}
	tracker, err := mt.factory()
	if err != nil {
		return nil, err
	}
	if frame.ctx == nil {
		frame.ctx = evalContextPool.Get().(*evalContext)
	}
	frame.ctx.memory = tracker
	return tracker, nil
}

// GetState extracts the MemoryTracker from the ExecutionFrame.
func (mt *memoryTrackerFactory) GetState(frame *ExecutionFrame) any {
	if frame == nil || frame.ctx == nil {
		return nil
	}
	return frame.ctx.memory
}

// Observe records the peak memory watermarks associated with each evaluation step.
//
// Watermarks are observed at the points where values materialize during evaluation: resolved
// attributes, function call results, constructed aggregate literals, and comprehension results.
// Since every intermediate Interpretable is observed, the inputs to a call contribute to the
// peak at the expression nodes which produced them; constants are part of the program image
// rather than runtime-materialized memory and are not observed.
func (mt *memoryTrackerFactory) Observe(vars Activation, id int64, programStep any, val ref.Val) {
	frame, ok := vars.(*ExecutionFrame)
	if !ok || frame.ctx == nil || frame.ctx.memory == nil {
		return
	}
	tracker := frame.ctx.memory
	switch programStep.(type) {
	case InterpretableAttribute, InterpretableConst, InterpretableCall, InterpretableConstructor, *evalFold:
		if frame.parent != nil {
			tracker.Sample(id, val)
		} else {
			tracker.Track(val)
		}
		if tracker.ExceedsLimit() {
			panic(EvalCancelledError{Cause: MemoryLimitExceeded, Message: "operation cancelled: memory limit exceeded"})
		}
	}
}
