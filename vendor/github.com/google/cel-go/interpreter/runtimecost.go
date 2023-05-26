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
	"math"

	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

// WARNING: Any changes to cost calculations in this file require a corresponding change in checker/cost.go

// ActualCostEstimator provides function call cost estimations at runtime
// CallCost returns an estimated cost for the function overload invocation with the given args, or nil if it has no
// estimate to provide. CEL attempts to provide reasonable estimates for its standard function library, so CallCost
// should typically not need to provide an estimate for CELs standard function.
type ActualCostEstimator interface {
	CallCost(function, overloadID string, args []ref.Val, result ref.Val) *uint64
}

// CostObserver provides an observer that tracks runtime cost.
func CostObserver(tracker *CostTracker) EvalObserver {
	observer := func(id int64, programStep any, val ref.Val) {
		switch t := programStep.(type) {
		case ConstantQualifier:
			// TODO: Push identifiers on to the stack before observing constant qualifiers that apply to them
			// and enable the below pop. Once enabled this can case can be collapsed into the Qualifier case.
			tracker.cost++
		case InterpretableConst:
			// zero cost
		case InterpretableAttribute:
			switch a := t.Attr().(type) {
			case *conditionalAttribute:
				// Ternary has no direct cost. All cost is from the conditional and the true/false branch expressions.
				tracker.stack.drop(a.falsy.ID(), a.truthy.ID(), a.expr.ID())
			default:
				tracker.stack.drop(t.Attr().ID())
				tracker.cost += common.SelectAndIdentCost
			}
			if !tracker.presenceTestHasCost {
				if _, isTestOnly := programStep.(*evalTestOnly); isTestOnly {
					tracker.cost -= common.SelectAndIdentCost
				}
			}
		case *evalExhaustiveConditional:
			// Ternary has no direct cost. All cost is from the conditional and the true/false branch expressions.
			tracker.stack.drop(t.attr.falsy.ID(), t.attr.truthy.ID(), t.attr.expr.ID())

		// While the field names are identical, the boolean operation eval structs do not share an interface and so
		// must be handled individually.
		case *evalOr:
			tracker.stack.drop(t.rhs.ID(), t.lhs.ID())
		case *evalAnd:
			tracker.stack.drop(t.rhs.ID(), t.lhs.ID())
		case *evalExhaustiveOr:
			tracker.stack.drop(t.rhs.ID(), t.lhs.ID())
		case *evalExhaustiveAnd:
			tracker.stack.drop(t.rhs.ID(), t.lhs.ID())
		case *evalFold:
			tracker.stack.drop(t.iterRange.ID())
		case Qualifier:
			tracker.cost++
		case InterpretableCall:
			if argVals, ok := tracker.stack.dropArgs(t.Args()); ok {
				tracker.cost += tracker.costCall(t, argVals, val)
			}
		case InterpretableConstructor:
			tracker.stack.dropArgs(t.InitVals())
			switch t.Type() {
			case types.ListType:
				tracker.cost += common.ListCreateBaseCost
			case types.MapType:
				tracker.cost += common.MapCreateBaseCost
			default:
				tracker.cost += common.StructCreateBaseCost
			}
		}
		tracker.stack.push(val, id)

		if tracker.Limit != nil && tracker.cost > *tracker.Limit {
			panic(EvalCancelledError{Cause: CostLimitExceeded, Message: "operation cancelled: actual cost limit exceeded"})
		}
	}
	return observer
}

// CostTrackerOption configures the behavior of CostTracker objects.
type CostTrackerOption func(*CostTracker) error

// CostTrackerLimit sets the runtime limit on the evaluation cost during execution and will terminate the expression
// evaluation if the limit is exceeded.
func CostTrackerLimit(limit uint64) CostTrackerOption {
	return func(tracker *CostTracker) error {
		tracker.Limit = &limit
		return nil
	}
}

// PresenceTestHasCost determines whether presence testing has a cost of one or zero.
// Defaults to presence test has a cost of one.
func PresenceTestHasCost(hasCost bool) CostTrackerOption {
	return func(tracker *CostTracker) error {
		tracker.presenceTestHasCost = hasCost
		return nil
	}
}

// NewCostTracker creates a new CostTracker with a given estimator and a set of functional CostTrackerOption values.
func NewCostTracker(estimator ActualCostEstimator, opts ...CostTrackerOption) (*CostTracker, error) {
	tracker := &CostTracker{
		Estimator:           estimator,
		presenceTestHasCost: true,
	}
	for _, opt := range opts {
		err := opt(tracker)
		if err != nil {
			return nil, err
		}
	}
	return tracker, nil
}

// CostTracker represents the information needed for tracking runtime cost.
type CostTracker struct {
	Estimator           ActualCostEstimator
	Limit               *uint64
	presenceTestHasCost bool

	cost  uint64
	stack refValStack
}

// ActualCost returns the runtime cost
func (c *CostTracker) ActualCost() uint64 {
	return c.cost
}

func (c *CostTracker) costCall(call InterpretableCall, argValues []ref.Val, result ref.Val) uint64 {
	var cost uint64
	if c.Estimator != nil {
		callCost := c.Estimator.CallCost(call.Function(), call.OverloadID(), argValues, result)
		if callCost != nil {
			cost += *callCost
			return cost
		}
	}
	// if user didn't specify, the default way of calculating runtime cost would be used.
	// if user has their own implementation of ActualCostEstimator, make sure to cover the mapping between overloadId and cost calculation
	switch call.OverloadID() {
	// O(n) functions
	case overloads.StartsWithString, overloads.EndsWithString, overloads.StringToBytes, overloads.BytesToString, overloads.ExtQuoteString, overloads.ExtFormatString:
		cost += uint64(math.Ceil(float64(c.actualSize(argValues[0])) * common.StringTraversalCostFactor))
	case overloads.InList:
		// If a list is composed entirely of constant values this is O(1), but we don't account for that here.
		// We just assume all list containment checks are O(n).
		cost += c.actualSize(argValues[1])
	// O(min(m, n)) functions
	case overloads.LessString, overloads.GreaterString, overloads.LessEqualsString, overloads.GreaterEqualsString,
		overloads.LessBytes, overloads.GreaterBytes, overloads.LessEqualsBytes, overloads.GreaterEqualsBytes,
		overloads.Equals, overloads.NotEquals:
		// When we check the equality of 2 scalar values (e.g. 2 integers, 2 floating-point numbers, 2 booleans etc.),
		// the CostTracker.actualSize() function by definition returns 1 for each operand, resulting in an overall cost
		// of 1.
		lhsSize := c.actualSize(argValues[0])
		rhsSize := c.actualSize(argValues[1])
		minSize := lhsSize
		if rhsSize < minSize {
			minSize = rhsSize
		}
		cost += uint64(math.Ceil(float64(minSize) * common.StringTraversalCostFactor))
	// O(m+n) functions
	case overloads.AddString, overloads.AddBytes:
		// In the worst case scenario, we would need to reallocate a new backing store and copy both operands over.
		cost += uint64(math.Ceil(float64(c.actualSize(argValues[0])+c.actualSize(argValues[1])) * common.StringTraversalCostFactor))
	// O(nm) functions
	case overloads.MatchesString:
		// https://swtch.com/~rsc/regexp/regexp1.html applies to RE2 implementation supported by CEL
		// Add one to string length for purposes of cost calculation to prevent product of string and regex to be 0
		// in case where string is empty but regex is still expensive.
		strCost := uint64(math.Ceil((1.0 + float64(c.actualSize(argValues[0]))) * common.StringTraversalCostFactor))
		// We don't know how many expressions are in the regex, just the string length (a huge
		// improvement here would be to somehow get a count the number of expressions in the regex or
		// how many states are in the regex state machine and use that to measure regex cost).
		// For now, we're making a guess that each expression in a regex is typically at least 4 chars
		// in length.
		regexCost := uint64(math.Ceil(float64(c.actualSize(argValues[1])) * common.RegexStringLengthCostFactor))
		cost += strCost * regexCost
	case overloads.ContainsString:
		strCost := uint64(math.Ceil(float64(c.actualSize(argValues[0])) * common.StringTraversalCostFactor))
		substrCost := uint64(math.Ceil(float64(c.actualSize(argValues[1])) * common.StringTraversalCostFactor))
		cost += strCost * substrCost

	default:
		// The following operations are assumed to have O(1) complexity.
		// - AddList due to the implementation. Index lookup can be O(c) the
		//    number of concatenated lists, but we don't track that is cost calculations.
		// - Conversions, since none perform a traversal of a type of unbound length.
		// - Computing the size of strings, byte sequences, lists and maps.
		// - Logical operations and all operators on fixed width scalars (comparisons, equality)
		// - Any functions that don't have a declared cost either here or in provided ActualCostEstimator.
		cost++

	}
	return cost
}

// actualSize returns the size of value
func (c *CostTracker) actualSize(value ref.Val) uint64 {
	if sz, ok := value.(traits.Sizer); ok {
		return uint64(sz.Size().(types.Int))
	}
	return 1
}

type stackVal struct {
	Val ref.Val
	ID  int64
}

// refValStack keeps track of values of the stack for cost calculation purposes
type refValStack []stackVal

func (s *refValStack) push(val ref.Val, id int64) {
	value := stackVal{Val: val, ID: id}
	*s = append(*s, value)
}

// TODO: Allowing drop and dropArgs to remove stack items above the IDs they are provided is a workaround. drop and dropArgs
// should find and remove only the stack items matching the provided IDs once all attributes are properly pushed and popped from stack.

// drop searches the stack for each ID and removes the ID and all stack items above it.
// If none of the IDs are found, the stack is not modified.
// WARNING: It is possible for multiple expressions with the same ID to exist (due to how macros are implemented) so it's
// possible that a dropped ID will remain on the stack.  They should be removed when IDs on the stack are popped.
func (s *refValStack) drop(ids ...int64) {
	for _, id := range ids {
		for idx := len(*s) - 1; idx >= 0; idx-- {
			if (*s)[idx].ID == id {
				*s = (*s)[:idx]
				break
			}
		}
	}
}

// dropArgs searches the stack for all the args by their IDs, accumulates their associated ref.Vals and drops any
// stack items above any of the arg IDs. If any of the IDs are not found the stack, false is returned.
// Args are assumed to be found in the stack in reverse order, i.e. the last arg is expected to be found highest in
// the stack.
// WARNING: It is possible for multiple expressions with the same ID to exist (due to how macros are implemented) so it's
// possible that a dropped ID will remain on the stack.  They should be removed when IDs on the stack are popped.
func (s *refValStack) dropArgs(args []Interpretable) ([]ref.Val, bool) {
	result := make([]ref.Val, len(args))
argloop:
	for nIdx := len(args) - 1; nIdx >= 0; nIdx-- {
		for idx := len(*s) - 1; idx >= 0; idx-- {
			if (*s)[idx].ID == args[nIdx].ID() {
				el := (*s)[idx]
				*s = (*s)[:idx]
				result[nIdx] = el.Val
				continue argloop
			}
		}
		return nil, false
	}
	return result, true
}
