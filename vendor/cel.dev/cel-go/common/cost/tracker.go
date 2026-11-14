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

package cost

import (
	"cel.dev/cel-go/common/types/ref"
	"fmt"
)

// WARNING: Any changes to cost calculations in this file require a corresponding change in estimator.go

// ActualCostEstimator provides function call cost estimations at runtime.
//
// CallCost returns an estimated cost for the function overload invocation with the given args, or nil if it has no
// estimate to provide. CEL attempts to provide reasonable estimates for its standard function library, so CallCost
// should typically not need to provide an estimate for CELs standard function.
type ActualCostEstimator interface {
	CallCost(function, overloadID string, args []ref.Val, result ref.Val) *uint64
}

// FunctionTracker computes the actual cost of evaluating the functions with the given arguments and result.
type FunctionTracker func(args []ref.Val, result ref.Val) *uint64

// Call represents an invocable function with a name and overload ID.
type Call interface {
	// Function returns the name of the function being called.
	Function() string

	// OverloadID returns the specific overload ID being invoked.
	OverloadID() string
}

// TrackerOption configures the behavior of CostTracker objects.
type TrackerOption interface {
	applyTracker(*trackerBuilder) error
}

type trackerOption func(*Tracker) error

func (opt trackerOption) applyTracker(b *trackerBuilder) error {
	return opt(b.tracker)
}

type trackerBuilderOption func(*trackerBuilder) error

func (opt trackerBuilderOption) applyTracker(b *trackerBuilder) error {
	return opt(b)
}

// trackerBuilder holds the configuration which matters only while NewTracker builds a Tracker.
//
// It is kept out of the Tracker, which is copied for every evaluation by Clone, so that the
// per-evaluation copy carries only what pricing a call reads.
type trackerBuilder struct {
	tracker *Tracker
	modelRules
	overloadTrackers overloadTable[FunctionTracker]
}

// TrackerLimit sets the runtime limit on the evaluation cost during execution and will terminate the expression
// evaluation if the limit is exceeded.
func TrackerLimit(limit uint64) TrackerOption {
	return trackerOption(func(tracker *Tracker) error {
		tracker.Limit = &limit
		return nil
	})
}

// TrackerPresenceTestHasCost determines whether presence testing has a cost of one or zero.
// Defaults to presence test has a cost of one.
func TrackerPresenceTestHasCost(hasCost bool) TrackerOption {
	return trackerOption(func(tracker *Tracker) error {
		tracker.presenceTestHasCost = hasCost
		return nil
	})
}

// TrackerLimitExceededHandler sets a custom handler invoked when the cost limit is exceeded.
func TrackerLimitExceededHandler(handler func()) TrackerOption {
	return trackerOption(func(tracker *Tracker) error {
		tracker.limitExceededHandler = handler
		return nil
	})
}

// OverloadTracker binds an overload ID to a runtime FunctionTracker implementation.
//
// OverloadTracker instances augment or override ActualCostEstimator decisions, allowing for
// versioned and/or optional cost tracking changes.
func OverloadTracker(overloadID string, fnTracker FunctionTracker) TrackerOption {
	return trackerBuilderOption(func(b *trackerBuilder) error {
		if fnTracker == nil {
			return fmt.Errorf("OverloadTracker(%q) requires a FunctionTracker", overloadID)
		}
		b.overloadTrackers.mutable()[overloadID] = fnTracker
		return nil
	})
}

// LimitExceededError indicates that the actual cost limit was exceeded during evaluation.
type LimitExceededError struct {
	Message string
}

// Error returns the error message for LimitExceededError.
func (e LimitExceededError) Error() string {
	return e.Message
}

// Tracker represents the information needed for tracking runtime cost.
type Tracker struct {
	Estimator                ActualCostEstimator
	overloadTrackers         map[string]FunctionTracker
	standardOverloadTrackers map[string]FunctionTracker
	Limit                    *uint64
	limitExceededHandler     func()
	cost                     uint64
	presenceTestHasCost      bool
	modelVersion             uint32
}

// NewTracker creates a new Tracker with a given estimator and a set of functional TrackerOption values.
//
// The Tracker's configuration is fixed once it is returned; only its accumulated cost changes.
func NewTracker(estimator ActualCostEstimator, opts ...TrackerOption) (*Tracker, error) {
	b := &trackerBuilder{
		tracker:    &Tracker{Estimator: estimator, presenceTestHasCost: true},
		modelRules: defaultModelRules(),
	}
	for _, opt := range opts {
		if mo, ok := opt.(ModelOption); ok {
			b.modelRules.override(mo)
		}
	}
	for _, opt := range opts {
		if err := opt.applyTracker(b); err != nil {
			return nil, err
		}
	}
	tracker := b.tracker
	tracker.modelVersion = b.modelVersion
	tracker.overloadTrackers = b.overloadTrackers.fns
	// The package-level trackers are compiled under the default rules.
	tracker.standardOverloadTrackers = stdOverloadTrackers
	if !b.modelRules.isDefault() {
		tracker.standardOverloadTrackers = StandardOverloadTrackersWithOptions(b.modelRules.modelOptions()...)
	}
	return tracker, nil
}

// Clone makes a shallow copy of the tracker with no accumulated cost.
// The different clones can be used independently from each other: they share only configuration,
// which is not modified after NewTracker returns.
func (c *Tracker) Clone() (*Tracker, error) {
	tracker := *c
	tracker.cost = 0
	return &tracker, nil
}

// ActualCost returns the runtime cost.
func (c *Tracker) ActualCost() uint64 {
	return c.cost
}

// PresenceTestHasCost returns whether presence testing has a cost.
func (c *Tracker) PresenceTestHasCost() bool {
	return c.presenceTestHasCost
}

// ModelVersion returns the cost model version configured on the tracker.
func (c *Tracker) ModelVersion() uint32 {
	if c == nil {
		return defaultModelVersion
	}
	return c.modelVersion
}

// CreateList records list literal construction cost.
func (c *Tracker) CreateList(id int64, res ref.Val) {
	c.cost = SafeAdd(c.cost, ListCreateBaseCost)
	c.checkLimit()
}

// CreateMap records map literal construction cost.
func (c *Tracker) CreateMap(id int64, res ref.Val) {
	c.cost = SafeAdd(c.cost, MapCreateBaseCost)
	c.checkLimit()
}

// CreateStruct records struct/object construction cost.
func (c *Tracker) CreateStruct(id int64, res ref.Val) {
	c.cost = SafeAdd(c.cost, StructCreateBaseCost)
	c.checkLimit()
}

// EvalAttribute records attribute resolution cost (ident / select).
func (c *Tracker) EvalAttribute(id int64, isTestOnly bool, res ref.Val) {
	if !isTestOnly || c.presenceTestHasCost {
		c.cost = SafeAdd(c.cost, SelectAndIdentCost)
		c.checkLimit()
	}
}

// Qualify records qualifier cost.
func (c *Tracker) Qualify(id int64) {
	c.cost = SafeAdd(c.cost, 1)
	c.checkLimit()
}

func (c *Tracker) recordCallCost(call Call, args []ref.Val, result ref.Val) {
	c.cost = SafeAdd(c.cost, c.CostCall(call, args, result))
	c.checkLimit()
}

// EvalZeroArity records the cost for a 0-arity call expression.
func (c *Tracker) EvalZeroArity(vars any, id int64, call Call, result ref.Val) {
	c.recordCallCost(call, nil, result)
}

// EvalUnary records the cost for a unary call expression.
func (c *Tracker) EvalUnary(vars any, id int64, call Call, arg ref.Val, result ref.Val) {
	var buf [1]ref.Val
	buf[0] = arg
	c.recordCallCost(call, buf[:], result)
}

// EvalBinary records the cost for a binary call expression.
func (c *Tracker) EvalBinary(vars any, id int64, call Call, lhs, rhs ref.Val, result ref.Val) {
	var buf [2]ref.Val
	buf[0] = lhs
	buf[1] = rhs
	c.recordCallCost(call, buf[:], result)
}

// EvalVarArgs records the cost for a variadic call expression.
func (c *Tracker) EvalVarArgs(vars any, id int64, call Call, args []ref.Val, result ref.Val) {
	c.recordCallCost(call, args, result)
}

func (c *Tracker) checkLimit() {
	if c.Limit != nil && c.cost > *c.Limit {
		if c.limitExceededHandler != nil {
			c.limitExceededHandler()
		}
		panic(LimitExceededError{Message: "operation cancelled: actual cost limit exceeded"})
	}
}

// CostCall calculates the runtime cost for a function call.
func (c *Tracker) CostCall(call Call, args []ref.Val, result ref.Val) uint64 {
	var total uint64
	if tracker, found := c.overloadTrackers[call.OverloadID()]; found {
		callCost := tracker(args, result)
		if callCost != nil {
			total = SafeAdd(total, *callCost)
			return total
		}
	}
	if c.Estimator != nil {
		callCost := c.Estimator.CallCost(call.Function(), call.OverloadID(), args, result)
		if callCost != nil {
			total = SafeAdd(total, *callCost)
			return total
		}
	}
	if tracker, found := c.standardOverloadTrackers[call.OverloadID()]; found {
		callCost := tracker(args, result)
		if callCost != nil {
			total = SafeAdd(total, *callCost)
			return total
		}
	}
	// The following operations are assumed to have O(1) complexity.
	// - AddList due to the implementation. Index lookup can be O(c) the
	//    number of concatenated lists, but we don't track that in cost calculations.
	// - Conversions, since none perform a traversal of a type of unbound length.
	// - Computing the size of strings, byte sequences, lists and maps.
	// - Logical operations and all operators on fixed width scalars (comparisons, equality)
	// - Any functions that don't have a declared cost either here or in provided ActualCostEstimator.
	return SafeAdd(total, 1)
}

var (
	stdOverloadTrackers = StandardOverloadTrackers()
)
