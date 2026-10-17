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

package ext

import (
	"math"

	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

var (
	callCostEstimate = checker.FixedCostEstimate(1)
	callCost         = uint64(1)
	listAllocCost    = checker.FixedCostEstimate(common.ListCreateBaseCost)
	stringCostFactor = common.StringTraversalCostFactor
)

func estimateStringScan(sz checker.SizeEstimate) (checker.CostEstimate, *checker.SizeEstimate) {
	return estimateTraversal(sz, stringCostFactor, nil)
}

func estimateListAlloc(sz checker.SizeEstimate, costFactor float64) (checker.CostEstimate, *checker.SizeEstimate) {
	return estimateTraversal(sz, costFactor, &listAllocCost)
}

// estimateTraversal computes cost as a function of the size of the target object and whether the call allocates memory.
func estimateTraversal(nodeSize checker.SizeEstimate, costFactor float64, allocationCost *checker.CostEstimate) (checker.CostEstimate, *checker.SizeEstimate) {
	cost := nodeSize.MultiplyByCostFactor(costFactor)
	if allocationCost != nil {
		cost = cost.Add(*allocationCost)
	}
	return cost, &nodeSize
}

func estimateSize(estimator checker.CostEstimator, node checker.AstNode) checker.SizeEstimate {
	if l := node.ComputedSize(); l != nil {
		return *l
	}
	if l := estimator.EstimateSize(node); l != nil {
		return *l
	}
	return checker.SizeEstimate{Min: 0, Max: math.MaxUint64}
}

func actualSize(value ref.Val) uint64 {
	if sz, ok := value.(traits.Sizer); ok {
		return uint64(sz.Size().(types.Int))
	}
	return 1
}

func nodeAsUintValue(node checker.AstNode, defaultVal uint64) uint64 {
	if node.Expr().Kind() != ast.LiteralKind {
		return defaultVal
	}
	lit := node.Expr().AsLiteral()
	if lit.Type() != types.IntType {
		return defaultVal
	}
	val := lit.(types.Int)
	if val < types.IntZero {
		return 0
	}
	return uint64(lit.(types.Int))
}

func callEstimate(cost checker.CostEstimate, sz *checker.SizeEstimate) *checker.CallEstimate {
	return &checker.CallEstimate{CostEstimate: cost, ResultSize: sz}
}

func rangedSizeEstimate(min, max uint64) checker.SizeEstimate {
	return checker.SizeEstimate{Min: min, Max: max}
}

func fixedSizeEstimate(val uint64) checker.SizeEstimate {
	return checker.FixedSizeEstimate(val)
}

func atLeastOne(size checker.SizeEstimate) checker.SizeEstimate {
	if size.Min == 0 {
		size.Min = 1
	}
	if size.Max == 0 {
		size.Max = 1
	}
	return size
}

func safeAdd(x, y uint64, rest ...uint64) uint64 {
	if y > 0 && x > math.MaxUint64-y {
		return math.MaxUint64
	}
	next := x + y
	if len(rest) == 0 {
		return next
	}
	return safeAdd(next, rest[0], rest[1:]...)
}

func safeMul(x, y uint64) uint64 {
	if y != 0 && x > math.MaxUint64/y {
		return math.MaxUint64
	}
	return x * y
}
