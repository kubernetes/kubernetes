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

package checker

import (
	"cel.dev/cel-go/common/ast"
	"cel.dev/cel-go/common/cost"
	"cel.dev/cel-go/common/types"
)

type (
	// CostEstimator estimates the sizes of variable length input data and the costs of functions.
	//
	// Deprecated: use cost.CostEstimator
	CostEstimator = cost.Estimator

	// CallEstimate includes a CostEstimate for the call, and an optional estimate of the result object size.
	//
	// Deprecated: use cost.CallEstimate
	CallEstimate = cost.CallEstimate

	// AstNode represents an AST node for the purpose of cost estimations.
	//
	// Deprecated: use cost.AstNode
	AstNode = cost.AstNode

	// SizeEstimate represents an estimated size of a variable length string, bytes, map or list.
	//
	// Deprecated: use cost.SizeEstimate
	SizeEstimate = cost.SizeEstimate

	// CostEstimate represents an estimated cost range and provides add and multiply operations
	// that do not overflow.
	//
	// Deprecated: use cost.CostEstimate
	CostEstimate = cost.CostEstimate

	// CostOption configures flags which affect cost computations.
	//
	// Deprecated: use cost.CostOption
	CostOption = cost.CostOption

	// FunctionEstimator provides a CallEstimate given the target and arguments for a specific function, overload pair.
	//
	// Deprecated: use cost.FunctionEstimator
	FunctionEstimator = cost.FunctionEstimator
)

// UnknownSizeEstimate returns a size between 0 and max uint.
//
// Deprecated: use cost.UnknownSizeEstimate
func UnknownSizeEstimate() SizeEstimate {
	return cost.UnknownSizeEstimate()
}

// FixedSizeEstimate returns a size estimate with a fixed min and max range.
//
// Deprecated: use cost.FixedSizeEstimate
func FixedSizeEstimate(size uint64) SizeEstimate {
	return cost.FixedSizeEstimate(size)
}

// UnknownCostEstimate returns a cost with an unknown impact.
//
// Deprecated: use cost.UnknownCostEstimate
func UnknownCostEstimate() CostEstimate {
	return cost.UnknownCostEstimate()
}

// FixedCostEstimate returns a cost with a fixed min and max range.
//
// Deprecated: use cost.FixedCostEstimate
func FixedCostEstimate(fixedCost uint64) CostEstimate {
	return cost.FixedCostEstimate(fixedCost)
}

// PresenceTestHasCost determines whether presence testing has a cost of one or zero.
//
// Deprecated: use cost.PresenceTestHasCost
func PresenceTestHasCost(hasCost bool) CostOption {
	return cost.PresenceTestHasCost(hasCost)
}

// OverloadCostEstimate binds a FunctionEstimator to a specific function overload ID.
//
// Deprecated: use cost.OverloadCostEstimate
func OverloadCostEstimate(overloadID string, functionCoster FunctionEstimator) CostOption {
	return cost.OverloadCostEstimate(overloadID, functionCoster)
}

// NewAstNode creates a new AstNode for cost estimation.
//
// Deprecated: use cost.NewAstNode
func NewAstNode(expr ast.Expr, path []string, t *types.Type, derivedSize *SizeEstimate) AstNode {
	return cost.NewAstNode(expr, path, t, derivedSize)
}

// Cost estimates the cost of the parsed and type checked CEL expression.
//
// Deprecated: use cost.Cost
func Cost(checked *ast.AST, estimator CostEstimator, opts ...CostOption) (CostEstimate, error) {
	return cost.Cost(checked, estimator, opts...)
}
