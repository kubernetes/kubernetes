// Copyright 2022 Google LLC
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

	"cel.dev/cel-go/common/cost"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
)

// WARNING: Any changes to cost calculations in this file require a corresponding change in checker/cost.go

type (
	// ActualCostEstimator provides function call cost estimations at runtime
	//
	// Deprecated: use cost.ActualCostEstimator
	ActualCostEstimator = cost.ActualCostEstimator

	// FunctionTracker computes the actual cost of evaluating the functions with the given arguments and result.
	//
	// Deprecated: use cost.FunctionTracker
	FunctionTracker = cost.FunctionTracker

	// CostTrackerOption configures the behavior of CostTracker objects.
	//
	// Deprecated: use cost.CostTrackerOption
	CostTrackerOption = cost.TrackerOption

	// CostTracker represents the information needed for tracking runtime cost.
	//
	// Deprecated: use cost.CostTracker
	CostTracker = cost.Tracker
)

// CostTrackerLimit sets the runtime limit on the evaluation cost during execution and will terminate the expression
// evaluation if the limit is exceeded.
//
// Deprecated: use cost.CostTrackerLimit
func CostTrackerLimit(limit uint64) CostTrackerOption {
	return cost.TrackerLimit(limit)
}

// PresenceTestHasCost determines whether presence testing has a cost of one or zero.
// Defaults to presence test has a cost of one.
//
// Deprecated: use cost.CostTrackerPresenceTestHasCost
func PresenceTestHasCost(hasCost bool) CostTrackerOption {
	return cost.TrackerPresenceTestHasCost(hasCost)
}

// OverloadCostTracker binds an overload ID to a runtime FunctionTracker implementation.
//
// Deprecated: use cost.OverloadCostTracker
func OverloadCostTracker(overloadID string, fnTracker FunctionTracker) CostTrackerOption {
	return cost.OverloadTracker(overloadID, fnTracker)
}

// NewCostTracker creates a new CostTracker with a given estimator and a set of functional CostTrackerOption values.
//
// Deprecated: use cost.NewCostTracker
func NewCostTracker(estimator ActualCostEstimator, opts ...CostTrackerOption) (*CostTracker, error) {
	evalCancelHandler := cost.TrackerLimitExceededHandler(func() {
		panic(EvalCancelledError{Cause: CostLimitExceeded, Message: "operation cancelled: actual cost limit exceeded"})
	})
	allOpts := make([]cost.TrackerOption, 0, len(opts)+1)
	allOpts = append(allOpts, evalCancelHandler)
	allOpts = append(allOpts, opts...)
	return cost.NewTracker(estimator, allOpts...)
}

// CostTrackPlanOption modifies the cost tracking factory associated with the CostObserver.
type CostTrackPlanOption func(*costTrackerFactory) *costTrackerFactory

type costTrackPlanOption = CostTrackPlanOption

// CostTrackerFactory configures the factory method to generate a new cost-tracker per-evaluation.
func CostTrackerFactory(factory func() (*CostTracker, error)) CostTrackPlanOption {
	return func(fac *costTrackerFactory) *costTrackerFactory {
		fac.factory = factory
		return fac
	}
}

// CostObserver provides an observer that tracks runtime cost.
func CostObserver(opts ...CostTrackPlanOption) PlannerOption {
	ct := &costTrackerFactory{}
	for _, o := range opts {
		ct = o(ct)
	}
	return func(p *planner) (*planner, error) {
		if ct.factory == nil {
			return nil, errors.New("cost tracker factory not configured")
		}
		p.costTrackerFactory = ct.factory
		return p, nil
	}
}

// costTrackerFactory holds a factory for producing new CostTracker instances on each Eval call.
type costTrackerFactory struct {
	factory func() (*CostTracker, error)
}

type costTrackingInterpretable struct {
	InterpretableV2
	factory func() (*CostTracker, error)
}

func (c *costTrackingInterpretable) Exec(frame *ExecutionFrame) ref.Val {
	if frame.CostTracker() == nil {
		tracker, err := c.factory()
		if err != nil {
			return types.NewErr("cost tracker factory: %v", err)
		}
		frame.SetCostTracker(tracker)
	}
	return c.InterpretableV2.Exec(frame)
}

func (c *costTrackingInterpretable) Eval(ctx Activation) ref.Val {
	return EvalActivation(ctx, c.Exec)
}
