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

import "cel.dev/cel-go/common/overloads"

// StandardOverloadModels defines the cost models for standard CEL functions.
var StandardOverloadModels = []OverloadModel{
	// O(1) container index operations
	Overload(overloads.IndexList,
		EvalCost(Const(1)),
		ResultSize(ArgElem(0)),
	),
	Overload(overloads.IndexMap,
		EvalCost(Const(1)),
		ResultSize(ArgElem(0)),
	),

	// O(n) prefix/suffix functions
	MemberOverload(overloads.StartsWithString,
		EvalCost(Scale(Arg(0), StringTraversalCostFactor)),
	),
	MemberOverload(overloads.EndsWithString,
		EvalCost(Scale(Arg(0), StringTraversalCostFactor)),
	),

	// O(n) conversion & format functions
	Overload(overloads.StringToBytes,
		EvalCost(Scale(Arg(0), StringTraversalCostFactor)),
		ResultSize(Ranged(Arg(0), Scale(Arg(0), 4.0))),
	),
	Overload(overloads.BytesToString,
		EvalCost(Scale(Arg(0), StringTraversalCostFactor)),
		ResultSize(Ranged(Scale(Arg(0), 0.25), Arg(0))),
	),
	Overload(overloads.ExtQuoteString,
		EvalCost(Scale(Arg(0), StringTraversalCostFactor)),
		ResultSize(Ranged(Sum(Arg(0), Const(2)), Sum(Scale(Arg(0), 2.0), Const(2)))),
	),
	MemberOverload(overloads.ExtFormatString,
		EvalCost(Scale(Target(), StringTraversalCostFactor)),
	),

	// O(n) containment
	Overload(overloads.InList, EvalCost(Arg(1))),

	// Short-circuiting optional selection. The planner rewrites these calls into a branch which
	// evaluates the alternative only when the receiver is empty, and which is not charged at
	// runtime, so the model reports no cost for the dispatch itself. The alternative's cost is
	// bounded by calculateArgCost.
	//
	// From version 1 the result is whichever of the two is present: or() returns one of the two
	// optionals, and orValue() the receiver's value or the alternative. Before the revision these
	// were unmodeled calls, which v0.32.0 estimated at 1 with no result size, although it charged
	// them nothing.
	MemberOverload(overloads.OptionalOrOptional,
		EvalCost(Const(0)),
		argumentsOnly(),
		ResultSize(Union(Target(), Arg(0))),
		BeforeVersion(1, ManualEstimator(estimateUnmodeledCall)),
	),
	MemberOverload(overloads.OptionalOrValueValue,
		EvalCost(Const(0)),
		argumentsOnly(),
		ResultSize(Union(ElemOf(Target()), Arg(0))),
		BeforeVersion(1, ManualEstimator(estimateUnmodeledCall)),
	),

	// Creating, wrapping and unwrapping an optional. These cost what an unmodeled call costs; they
	// are modeled from version 1 so that the wrapped value's size survives the round trip, with an
	// optional sized as 1 plus the value it holds (see Optional). Without them an estimate dies at
	// optional.of(x) and everything downstream of the optional is unbounded.
	Overload(overloads.OptionalNone,
		EvalCost(Const(1)),
		ResultSize(Optional(Const(0))),
		BeforeVersion(1, ResultSize(nil)),
	),
	Overload(overloads.OptionalOf,
		EvalCost(Const(1)),
		ResultSize(Optional(Arg(0))),
		BeforeVersion(1, ResultSize(nil)),
	),
	// ofNonZeroValue is empty for a zero value, so it holds a value of at most the argument's size.
	Overload(overloads.OptionalOfNonZeroValue,
		EvalCost(Const(1)),
		ResultSize(Optional(AtMost(Arg(0)))),
		BeforeVersion(1, ResultSize(nil)),
	),
	MemberOverload(overloads.OptionalValue,
		EvalCost(Const(1)),
		ResultSize(ElemOf(Target())),
		BeforeVersion(1, ResultSize(nil)),
	),

	// O(min(m, n)) comparison / equality. Before version 1 the estimate keeps the v0.32.0 bounds of
	// estimateComparisonV0; the charge is the same at every version.
	Overload(overloads.LessString,
		EvalCost(Scale(Min(Arg(0), Arg(1)), StringTraversalCostFactor)),
		BeforeVersion(1, ManualEstimator(estimateComparisonV0)),
	),
	Overload(overloads.GreaterString,
		EvalCost(Scale(Min(Arg(0), Arg(1)), StringTraversalCostFactor)),
		BeforeVersion(1, ManualEstimator(estimateComparisonV0)),
	),
	Overload(overloads.LessEqualsString,
		EvalCost(Scale(Min(Arg(0), Arg(1)), StringTraversalCostFactor)),
		BeforeVersion(1, ManualEstimator(estimateComparisonV0)),
	),
	Overload(overloads.GreaterEqualsString,
		EvalCost(Scale(Min(Arg(0), Arg(1)), StringTraversalCostFactor)),
		BeforeVersion(1, ManualEstimator(estimateComparisonV0)),
	),
	Overload(overloads.LessBytes,
		EvalCost(Scale(Min(Arg(0), Arg(1)), StringTraversalCostFactor)),
		BeforeVersion(1, ManualEstimator(estimateComparisonV0)),
	),
	Overload(overloads.GreaterBytes,
		EvalCost(Scale(Min(Arg(0), Arg(1)), StringTraversalCostFactor)),
		BeforeVersion(1, ManualEstimator(estimateComparisonV0)),
	),
	Overload(overloads.LessEqualsBytes,
		EvalCost(Scale(Min(Arg(0), Arg(1)), StringTraversalCostFactor)),
		BeforeVersion(1, ManualEstimator(estimateComparisonV0)),
	),
	Overload(overloads.GreaterEqualsBytes,
		EvalCost(Scale(Min(Arg(0), Arg(1)), StringTraversalCostFactor)),
		BeforeVersion(1, ManualEstimator(estimateComparisonV0)),
	),
	Overload(overloads.Equals,
		EvalCost(Scale(Min(Arg(0), Arg(1)), StringTraversalCostFactor)),
		BeforeVersion(1, ManualEstimator(estimateComparisonV0)),
	),
	Overload(overloads.NotEquals,
		EvalCost(Scale(Min(Arg(0), Arg(1)), StringTraversalCostFactor)),
		BeforeVersion(1, ManualEstimator(estimateComparisonV0)),
	),

	// O(m+n) string & bytes concatenation
	Overload(overloads.AddString,
		EvalCost(Scale(Sum(Arg(0), Arg(1)), StringTraversalCostFactor)),
		ResultSize(Sum(Arg(0), Arg(1))),
	),
	Overload(overloads.AddBytes,
		EvalCost(Scale(Sum(Arg(0), Arg(1)), StringTraversalCostFactor)),
		ResultSize(Sum(Arg(0), Arg(1))),
	),

	// O(1) list concatenation with size tracking
	Overload(overloads.AddList,
		EvalCost(Const(1)),
		ResultSize(Sum(Arg(0), Arg(1))),
	),

	// O(nm) regex matches
	Overload(overloads.Matches,
		EvalCost(Mul(
			Scale(Sum(Arg(0), Const(1)), StringTraversalCostFactor),
			Scale(Arg(1), RegexStringLengthCostFactor),
		)),
		BeforeVersion(1, EvalCost(Mul(
			roundScale(Sum(Arg(0), Const(1)), StringTraversalCostFactor),
			roundScale(Arg(1), RegexStringLengthCostFactor),
		))),
	),
	MemberOverload(overloads.MatchesString,
		EvalCost(Mul(
			Scale(Sum(Target(), Const(1)), StringTraversalCostFactor),
			Scale(Arg(0), RegexStringLengthCostFactor),
		)),
		BeforeVersion(1, EvalCost(Mul(
			roundScale(Sum(Target(), Const(1)), StringTraversalCostFactor),
			roundScale(Arg(0), RegexStringLengthCostFactor),
		))),
	),

	// O(nm) substring contains
	MemberOverload(overloads.ContainsString,
		EvalCost(Mul(
			Scale(Target(), StringTraversalCostFactor),
			Scale(Arg(0), StringTraversalCostFactor),
		)),
		BeforeVersion(1, EvalCost(Mul(
			roundScale(Target(), StringTraversalCostFactor),
			roundScale(Arg(0), StringTraversalCostFactor),
		))),
	),

	// The arg cost for logical and conditional operators is special-cased for
	// short-circuiting (see CalculateArgCost in estimator.go), so the arg cost is 0.

	// Logical short-circuiting operators
	Overload(overloads.LogicalOr, EvalCost(Const(0)), argumentsOnly()),
	Overload(overloads.LogicalAnd, EvalCost(Const(0)), argumentsOnly()),

	// Conditional operator
	Overload(overloads.Conditional,
		EvalCost(Const(0)),
		argumentsOnly(),
		ResultSize(Union(Arg(1), Arg(2))),
	),
}

// roundScale scales e by f and rounds before any outer multiplication, reproducing the version 0
// rounding of products of scaled factors.
func roundScale(e QuantityExpr, f float64) QuantityExpr {
	s := Scale(e, f)
	return Ranged(s, s)
}

// StandardOverloadEstimators returns the map of FunctionEstimator instances for standard overloads.
func StandardOverloadEstimators() map[string]FunctionEstimator {
	return StandardOverloadEstimatorsWithOptions()
}

// StandardOverloadEstimatorsWithOptions returns the map of FunctionEstimator instances for standard
// overloads, configured by the supplied options.
func StandardOverloadEstimatorsWithOptions(opts ...ModelOption) map[string]FunctionEstimator {
	estimators := make(map[string]FunctionEstimator, len(StandardOverloadModels))
	for _, m := range StandardOverloadModels {
		estimators[m.ID] = m.FunctionEstimatorWithOptions(opts...)
	}
	return estimators
}

// StandardOverloadTrackers returns the map of FunctionTracker instances for standard overloads.
func StandardOverloadTrackers() map[string]FunctionTracker {
	return StandardOverloadTrackersWithOptions()
}

// StandardOverloadTrackersWithOptions returns the map of FunctionTracker instances for standard
// overloads, configured by the supplied options.
//
// The models are tracked under the configured revision, so that each charge is computed by the same
// rules as its estimate.
func StandardOverloadTrackersWithOptions(opts ...ModelOption) map[string]FunctionTracker {
	trackers := make(map[string]FunctionTracker, len(StandardOverloadModels))
	for _, m := range StandardOverloadModels {
		trackers[m.ID] = m.FunctionTrackerWithOptions(opts...)
	}
	return trackers
}

// estimateUnmodeledCall estimates a call as the estimator does a call it has no model for: a cost
// of 1, with no result size.
func estimateUnmodeledCall(Estimator, *AstNode, []AstNode) *CallEstimate {
	return NewCallEstimate(CallCostEstimate, nil)
}

// estimateComparisonV0 estimates a string, bytes or equality comparison as cel-go v0.32.0 did:
// ceil(0.1 * [m, min(lhs.Max, rhs.Max)]), where m is 1 when the smaller maximum is non-zero, and 0
// otherwise.
//
// The lower bound is the charge of comparing any non-empty operands, whatever their minimum sizes,
// where the version 1 model reports ceil(0.1 * min(lhs.Min, rhs.Min)). The upper bound scales an
// unknown size like any other, where the version 1 model saturates it at math.MaxUint64.
func estimateComparisonV0(estimator Estimator, _ *AstNode, args []AstNode) *CallEstimate {
	if len(args) != 2 {
		return nil
	}
	smallestMax := min(EstimateSize(estimator, args[0]).Max, EstimateSize(estimator, args[1]).Max)
	lo := uint64(0)
	if smallestMax > 0 {
		lo = 1
	}
	return NewCallEstimate(CostEstimate{Min: lo, Max: smallestMax}.MultiplyByCostFactor(StringTraversalCostFactor), nil)
}
