/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package library

import (
	"math"

	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// CostEstimator implements CEL's interpretable.ActualCostEstimator and checker.CostEstimator.
type CostEstimator struct {
	// SizeEstimator provides a CostEstimator.EstimateSize that this CostEstimator will delegate size estimation
	// calculations to if the size is not well known (i.e. a constant).
	SizeEstimator checker.CostEstimator
}

func (l *CostEstimator) CallCost(function, overloadId string, args []ref.Val, result ref.Val) *uint64 {
	switch function {
	case "check":
		// An authorization check has a fixed cost
		// This cost is set to allow for only two authorization checks per expression
		cost := uint64(350000)
		return &cost
	case "serviceAccount", "path", "group", "resource", "subresource", "namespace", "name", "allowed", "denied", "reason":
		// All authorization builder and accessor functions have a nominal cost
		cost := uint64(1)
		return &cost
	case "isSorted", "sum", "max", "min", "indexOf", "lastIndexOf":
		var cost uint64
		if len(args) > 0 {
			cost += traversalCost(args[0]) // these O(n) operations all cost roughly the cost of a single traversal
		}
		return &cost
	case "url", "lowerAscii", "upperAscii", "substring", "trim":
		if len(args) >= 1 {
			cost := uint64(math.Ceil(float64(actualSize(args[0])) * common.StringTraversalCostFactor))
			return &cost
		}
	case "replace", "split":
		if len(args) >= 1 {
			// cost is the traversal plus the construction of the result
			cost := uint64(math.Ceil(float64(actualSize(args[0])) * 2 * common.StringTraversalCostFactor))
			return &cost
		}
	case "join":
		if len(args) >= 1 {
			cost := uint64(math.Ceil(float64(actualSize(result)) * 2 * common.StringTraversalCostFactor))
			return &cost
		}
	case "find", "findAll":
		if len(args) >= 2 {
			strCost := uint64(math.Ceil((1.0 + float64(actualSize(args[0]))) * common.StringTraversalCostFactor))
			// We don't know how many expressions are in the regex, just the string length (a huge
			// improvement here would be to somehow get a count the number of expressions in the regex or
			// how many states are in the regex state machine and use that to measure regex cost).
			// For now, we're making a guess that each expression in a regex is typically at least 4 chars
			// in length.
			regexCost := uint64(math.Ceil(float64(actualSize(args[1])) * common.RegexStringLengthCostFactor))
			cost := strCost * regexCost
			return &cost
		}
	}
	return nil
}

func (l *CostEstimator) EstimateCallCost(function, overloadId string, target *checker.AstNode, args []checker.AstNode) *checker.CallEstimate {
	// WARNING: Any changes to this code impact API compatibility! The estimated cost is used to determine which CEL rules may be written to a
	// CRD and any change (cost increases and cost decreases) are breaking.
	switch function {
	case "check":
		// An authorization check has a fixed cost
		// This cost is set to allow for only two authorization checks per expression
		return &checker.CallEstimate{CostEstimate: checker.CostEstimate{Min: 350000, Max: 350000}}
	case "serviceAccount", "path", "group", "resource", "subresource", "namespace", "name", "allowed", "denied", "reason":
		// All authorization builder and accessor functions have a nominal cost
		return &checker.CallEstimate{CostEstimate: checker.CostEstimate{Min: 1, Max: 1}}
	case "isSorted", "sum", "max", "min", "indexOf", "lastIndexOf":
		if target != nil {
			// Charge 1 cost for comparing each element in the list
			elCost := checker.CostEstimate{Min: 1, Max: 1}
			// If the list contains strings or bytes, add the cost of traversing all the strings/bytes as a way
			// of estimating the additional comparison cost.
			if elNode := l.listElementNode(*target); elNode != nil {
				t := elNode.Type().GetPrimitive()
				if t == exprpb.Type_STRING || t == exprpb.Type_BYTES {
					sz := l.sizeEstimate(elNode)
					elCost = elCost.Add(sz.MultiplyByCostFactor(common.StringTraversalCostFactor))
				}
				return &checker.CallEstimate{CostEstimate: l.sizeEstimate(*target).MultiplyByCost(elCost)}
			} else { // the target is a string, which is supported by indexOf and lastIndexOf
				return &checker.CallEstimate{CostEstimate: l.sizeEstimate(*target).MultiplyByCostFactor(common.StringTraversalCostFactor)}
			}
		}
	case "url":
		if len(args) == 1 {
			sz := l.sizeEstimate(args[0])
			return &checker.CallEstimate{CostEstimate: sz.MultiplyByCostFactor(common.StringTraversalCostFactor)}
		}
	case "lowerAscii", "upperAscii", "substring", "trim":
		if target != nil {
			sz := l.sizeEstimate(*target)
			return &checker.CallEstimate{CostEstimate: sz.MultiplyByCostFactor(common.StringTraversalCostFactor), ResultSize: &sz}
		}
	case "replace":
		if target != nil && len(args) >= 2 {
			sz := l.sizeEstimate(*target)
			toReplaceSz := l.sizeEstimate(args[0])
			replaceWithSz := l.sizeEstimate(args[1])
			// smallest possible result: smallest input size composed of the largest possible substrings being replaced by smallest possible replacement
			minSz := uint64(math.Ceil(float64(sz.Min)/float64(toReplaceSz.Max))) * replaceWithSz.Min
			// largest possible result: largest input size composed of the smallest possible substrings being replaced by largest possible replacement
			maxSz := uint64(math.Ceil(float64(sz.Max)/float64(toReplaceSz.Min))) * replaceWithSz.Max

			// cost is the traversal plus the construction of the result
			return &checker.CallEstimate{CostEstimate: sz.MultiplyByCostFactor(2 * common.StringTraversalCostFactor), ResultSize: &checker.SizeEstimate{Min: minSz, Max: maxSz}}
		}
	case "split":
		if target != nil {
			sz := l.sizeEstimate(*target)

			// Worst case size is where is that a separator of "" is used, and each char is returned as a list element.
			max := sz.Max
			if len(args) > 1 {
				if c := args[1].Expr().GetConstExpr(); c != nil {
					max = uint64(c.GetInt64Value())
				}
			}
			// Cost is the traversal plus the construction of the result.
			return &checker.CallEstimate{CostEstimate: sz.MultiplyByCostFactor(2 * common.StringTraversalCostFactor), ResultSize: &checker.SizeEstimate{Min: 0, Max: max}}
		}
	case "join":
		if target != nil {
			var sz checker.SizeEstimate
			listSize := l.sizeEstimate(*target)
			if elNode := l.listElementNode(*target); elNode != nil {
				elemSize := l.sizeEstimate(elNode)
				sz = listSize.Multiply(elemSize)
			}

			if len(args) > 0 {
				sepSize := l.sizeEstimate(args[0])
				minSeparators := uint64(0)
				maxSeparators := uint64(0)
				if listSize.Min > 0 {
					minSeparators = listSize.Min - 1
				}
				if listSize.Max > 0 {
					maxSeparators = listSize.Max - 1
				}
				sz = sz.Add(sepSize.Multiply(checker.SizeEstimate{Min: minSeparators, Max: maxSeparators}))
			}

			return &checker.CallEstimate{CostEstimate: sz.MultiplyByCostFactor(common.StringTraversalCostFactor), ResultSize: &sz}
		}
	case "find", "findAll":
		if target != nil && len(args) >= 1 {
			sz := l.sizeEstimate(*target)
			// Add one to string length for purposes of cost calculation to prevent product of string and regex to be 0
			// in case where string is empty but regex is still expensive.
			strCost := sz.Add(checker.SizeEstimate{Min: 1, Max: 1}).MultiplyByCostFactor(common.StringTraversalCostFactor)
			// We don't know how many expressions are in the regex, just the string length (a huge
			// improvement here would be to somehow get a count the number of expressions in the regex or
			// how many states are in the regex state machine and use that to measure regex cost).
			// For now, we're making a guess that each expression in a regex is typically at least 4 chars
			// in length.
			regexCost := l.sizeEstimate(args[0]).MultiplyByCostFactor(common.RegexStringLengthCostFactor)
			// worst case size of result is that every char is returned as separate find result.
			return &checker.CallEstimate{CostEstimate: strCost.Multiply(regexCost), ResultSize: &checker.SizeEstimate{Min: 0, Max: sz.Max}}
		}
	}
	return nil
}

func actualSize(value ref.Val) uint64 {
	if sz, ok := value.(traits.Sizer); ok {
		return uint64(sz.Size().(types.Int))
	}
	return 1
}

func (l *CostEstimator) sizeEstimate(t checker.AstNode) checker.SizeEstimate {
	if sz := t.ComputedSize(); sz != nil {
		return *sz
	}
	if sz := l.EstimateSize(t); sz != nil {
		return *sz
	}
	return checker.SizeEstimate{Min: 0, Max: math.MaxUint64}
}

func (l *CostEstimator) listElementNode(list checker.AstNode) checker.AstNode {
	if lt := list.Type().GetListType(); lt != nil {
		nodePath := list.Path()
		if nodePath != nil {
			// Provide path if we have it so that a OpenAPIv3 maxLength validation can be looked up, if it exists
			// for this node.
			path := make([]string, len(nodePath)+1)
			copy(path, nodePath)
			path[len(nodePath)] = "@items"
			return &itemsNode{path: path, t: lt.GetElemType(), expr: nil}
		} else {
			// Provide just the type if no path is available so that worst case size can be looked up based on type.
			return &itemsNode{t: lt.GetElemType(), expr: nil}
		}
	}
	return nil
}

func (l *CostEstimator) EstimateSize(element checker.AstNode) *checker.SizeEstimate {
	if l.SizeEstimator != nil {
		return l.SizeEstimator.EstimateSize(element)
	}
	return nil
}

type itemsNode struct {
	path []string
	t    *exprpb.Type
	expr *exprpb.Expr
}

func (i *itemsNode) Path() []string {
	return i.path
}

func (i *itemsNode) Type() *exprpb.Type {
	return i.t
}

func (i *itemsNode) Expr() *exprpb.Expr {
	return i.expr
}

func (i *itemsNode) ComputedSize() *checker.SizeEstimate {
	return nil
}

// traversalCost computes the cost of traversing a ref.Val as a data tree.
func traversalCost(v ref.Val) uint64 {
	// TODO: This could potentially be optimized by sampling maps and lists instead of traversing.
	switch vt := v.(type) {
	case types.String:
		return uint64(float64(len(string(vt))) * common.StringTraversalCostFactor)
	case types.Bytes:
		return uint64(float64(len([]byte(vt))) * common.StringTraversalCostFactor)
	case traits.Lister:
		cost := uint64(0)
		for it := vt.Iterator(); it.HasNext() == types.True; {
			i := it.Next()
			cost += traversalCost(i)
		}
		return cost
	case traits.Mapper: // maps and objects
		cost := uint64(0)
		for it := vt.Iterator(); it.HasNext() == types.True; {
			k := it.Next()
			cost += traversalCost(k) + traversalCost(vt.Get(k))
		}
		return cost
	default:
		return 1
	}
}
