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
	"fmt"
	"math"

	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	"k8s.io/apiserver/pkg/cel"
)

// panicOnUnknown makes cost estimate functions panic on unrecognized functions.
// This is only set to true for unit tests.
var panicOnUnknown = false

// builtInFunctions is a list of functions used in cost tests that are not handled by CostEstimator.
var knownUnhandledFunctions = map[string]bool{
	"@not_strictly_false": true,
	"uint":                true,
	"duration":            true,
	"bytes":               true,
	"cel.@mapInsert":      true,
	"timestamp":           true,
	"strings.quote":       true,
	"value":               true,
	"_==_":                true,
	"_&&_":                true,
	"_||_":                true,
	"_>_":                 true,
	"_>=_":                true,
	"_<_":                 true,
	"_<=_":                true,
	"!_":                  true,
	"_?_:_":               true,
	"_+_":                 true,
	"_-_":                 true,
}

// CostEstimator implements CEL's interpretable.ActualCostEstimator and checker.CostEstimator.
type CostEstimator struct {
	// SizeEstimator provides a CostEstimator.EstimateSize that this CostEstimator will delegate size estimation
	// calculations to if the size is not well known (i.e. a constant).
	SizeEstimator checker.CostEstimator
}

const (
	// shortest repeatable selector requirement that allocates a values slice is 2 characters: k,
	selectorLengthToRequirementCount = float64(.5)
	// the expensive parts to represent each requirement are a struct and a values slice
	costPerRequirement = float64(common.ListCreateBaseCost + common.StructCreateBaseCost)
)

// a selector consists of a list of requirements held in a slice
var baseSelectorCost = checker.CostEstimate{Min: common.ListCreateBaseCost, Max: common.ListCreateBaseCost}

func selectorCostEstimate(selectorLength checker.SizeEstimate) checker.CostEstimate {
	parseCost := selectorLength.MultiplyByCostFactor(common.StringTraversalCostFactor)

	requirementCount := selectorLength.MultiplyByCostFactor(selectorLengthToRequirementCount)
	requirementCost := requirementCount.MultiplyByCostFactor(costPerRequirement)

	return baseSelectorCost.Add(parseCost).Add(requirementCost)
}

func (l *CostEstimator) CallCost(function, overloadId string, args []ref.Val, result ref.Val) *uint64 {
	switch function {
	case "check":
		// An authorization check has a fixed cost
		// This cost is set to allow for only two authorization checks per expression
		cost := uint64(350000)
		return &cost
	case "serviceAccount", "path", "group", "resource", "subresource", "namespace", "name", "allowed", "reason", "error", "errored":
		// All authorization builder and accessor functions have a nominal cost
		cost := uint64(1)
		return &cost
	case "fieldSelector", "labelSelector":
		// field and label selector parse is a string parse into a structured set of requirements
		if len(args) >= 2 {
			selectorLength := actualSize(args[1])
			cost := selectorCostEstimate(checker.SizeEstimate{Min: selectorLength, Max: selectorLength})
			return &cost.Max
		}
	case "isSorted", "sum", "max", "min", "indexOf", "lastIndexOf":
		var cost uint64
		if len(args) > 0 {
			cost += traversalCost(args[0]) // these O(n) operations all cost roughly the cost of a single traversal
		}
		return &cost
	case "url", "lowerAscii", "upperAscii", "substring", "trim", "jsonpatch.escapeKey":
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
	case "cidr", "isIP", "isCIDR":
		// IP and CIDR parsing is a string traversal.
		if len(args) >= 1 {
			cost := uint64(math.Ceil(float64(actualSize(args[0])) * common.StringTraversalCostFactor))
			return &cost
		}
	case "ip":
		// IP and CIDR parsing is a string traversal.
		if len(args) >= 1 {
			if overloadId == "cidr_ip" {
				// The IP member of the CIDR object is just accessing a field.
				// Nominal cost.
				cost := uint64(1)
				return &cost
			}

			cost := uint64(math.Ceil(float64(actualSize(args[0])) * common.StringTraversalCostFactor))
			return &cost
		}
	case "ip.isCanonical":
		if len(args) >= 1 {
			// We have to parse the string and then compare the parsed string to the original string.
			// So we double the cost of parsing the string.
			cost := uint64(math.Ceil(float64(actualSize(args[0])) * 2 * common.StringTraversalCostFactor))
			return &cost
		}
	case "masked", "prefixLength", "family", "isUnspecified", "isLoopback", "isLinkLocalMulticast", "isLinkLocalUnicast", "isGlobalUnicast":
		// IP and CIDR accessors are nominal cost.
		cost := uint64(1)
		return &cost
	case "containsIP":
		if len(args) >= 2 {
			cidrSize := actualSize(args[0])
			otherSize := actualSize(args[1])

			// This is the base cost of comparing two byte lists.
			// We will compare only up to the length of the CIDR prefix in bytes, so use the cidrSize twice.
			cost := uint64(math.Ceil(float64(cidrSize+cidrSize) * common.StringTraversalCostFactor))

			if overloadId == "cidr_contains_ip_string" {
				// If we are comparing a string, we must parse the string to into the right type, so add the cost of traversing the string again.
				cost += uint64(math.Ceil(float64(otherSize) * common.StringTraversalCostFactor))

			}

			return &cost
		}
	case "containsCIDR":
		if len(args) >= 2 {
			cidrSize := actualSize(args[0])
			otherSize := actualSize(args[1])

			// This is the base cost of comparing two byte lists.
			// We will compare only up to the length of the CIDR prefix in bytes, so use the cidrSize twice.
			cost := uint64(math.Ceil(float64(cidrSize+cidrSize) * common.StringTraversalCostFactor))

			// As we are comparing if a CIDR is within another CIDR, we first mask the base CIDR and
			// also compare the CIDR bits.
			// This has an additional cost of the length of the IP being traversed again, plus 1.
			cost += uint64(math.Ceil(float64(cidrSize)*common.StringTraversalCostFactor)) + 1

			if overloadId == "cidr_contains_cidr_string" {
				// If we are comparing a string, we must parse the string to into the right type, so add the cost of traversing the string again.
				cost += uint64(math.Ceil(float64(otherSize) * common.StringTraversalCostFactor))
			}

			return &cost
		}
	case "quantity", "isQuantity", "semver", "isSemver":
		if len(args) >= 1 {
			cost := uint64(math.Ceil(float64(actualSize(args[0])) * common.StringTraversalCostFactor))
			return &cost
		}
	case "validate":
		if len(args) >= 2 {
			format, isFormat := args[0].Value().(cel.Format)
			if isFormat {
				strSize := actualSize(args[1])

				// Dont have access to underlying regex, estimate a long regexp
				regexSize := format.MaxRegexSize

				// Copied from CEL implementation for regex cost
				//
				// https://swtch.com/~rsc/regexp/regexp1.html applies to RE2 implementation supported by CEL
				// Add one to string length for purposes of cost calculation to prevent product of string and regex to be 0
				// in case where string is empty but regex is still expensive.
				strCost := uint64(math.Ceil((1.0 + float64(strSize)) * common.StringTraversalCostFactor))
				// We don't know how many expressions are in the regex, just the string length (a huge
				// improvement here would be to somehow get a count the number of expressions in the regex or
				// how many states are in the regex state machine and use that to measure regex cost).
				// For now, we're making a guess that each expression in a regex is typically at least 4 chars
				// in length.
				regexCost := uint64(math.Ceil(float64(regexSize) * common.RegexStringLengthCostFactor))
				cost := strCost * regexCost
				return &cost
			}
		}
	case "format.named":
		// Simply dictionary lookup
		cost := uint64(1)
		return &cost
	case "sign", "asInteger", "isInteger", "asApproximateFloat", "isGreaterThan", "isLessThan", "compareTo", "add", "sub", "major", "minor", "patch":
		cost := uint64(1)
		return &cost
	case "getScheme", "getHostname", "getHost", "getPort", "getEscapedPath", "getQuery":
		// url accessors
		cost := uint64(1)
		return &cost
	case "_==_":
		if len(args) == 2 {
			unitCost := uint64(1)
			lhs := args[0]
			switch lhs.(type) {
			case *cel.Quantity, cel.Quantity,
				*cel.IP, cel.IP,
				*cel.CIDR, cel.CIDR,
				*cel.Format, cel.Format, // Formats have a small max size. Format takes pointer receiver.
				*cel.URL, cel.URL, // TODO: Computing the actual cost is expensive, and changing this would be a breaking change
				*cel.Semver, cel.Semver,
				*authorizerVal, authorizerVal, *pathCheckVal, pathCheckVal, *groupCheckVal, groupCheckVal,
				*resourceCheckVal, resourceCheckVal, *decisionVal, decisionVal:
				return &unitCost
			default:
				if panicOnUnknown && lhs.Type() != nil && isRegisteredType(lhs.Type().TypeName()) {
					panic(fmt.Errorf("CallCost: unhandled equality for Kubernetes type %T", lhs))
				}
			}
		}
	}
	if panicOnUnknown && !knownUnhandledFunctions[function] {
		panic(fmt.Errorf("CallCost: unhandled function %q or args %v", function, args))
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
	case "serviceAccount", "path", "group", "resource", "subresource", "namespace", "name", "allowed", "reason", "error", "errored":
		// All authorization builder and accessor functions have a nominal cost
		return &checker.CallEstimate{CostEstimate: checker.CostEstimate{Min: 1, Max: 1}}
	case "fieldSelector", "labelSelector":
		// field and label selector parse is a string parse into a structured set of requirements
		if len(args) == 1 {
			return &checker.CallEstimate{CostEstimate: selectorCostEstimate(l.sizeEstimate(args[0]))}
		}
	case "isSorted", "sum", "max", "min", "indexOf", "lastIndexOf":
		if target != nil {
			// Charge 1 cost for comparing each element in the list
			elCost := checker.CostEstimate{Min: 1, Max: 1}
			// If the list contains strings or bytes, add the cost of traversing all the strings/bytes as a way
			// of estimating the additional comparison cost.
			if elNode := l.listElementNode(*target); elNode != nil {
				k := elNode.Type().Kind()
				if k == types.StringKind || k == types.BytesKind {
					sz := l.sizeEstimate(elNode)
					elCost = elCost.Add(sz.MultiplyByCostFactor(common.StringTraversalCostFactor))
				}
				return &checker.CallEstimate{CostEstimate: l.sizeEstimate(*target).MultiplyByCost(elCost)}
			} else { // the target is a string, which is supported by indexOf and lastIndexOf
				return &checker.CallEstimate{CostEstimate: l.sizeEstimate(*target).MultiplyByCostFactor(common.StringTraversalCostFactor)}
			}
		}
	case "url", "jsonpatch.escapeKey":
		if len(args) == 1 {
			sz := l.sizeEstimate(args[0])
			return &checker.CallEstimate{CostEstimate: sz.MultiplyByCostFactor(common.StringTraversalCostFactor), ResultSize: &sz}
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

			var replaceCount, retainedSz checker.SizeEstimate
			// find the longest replacement:
			if toReplaceSz.Min == 0 {
				// if the string being replaced is empty, replace surrounds all characters in the input string with the replacement.
				if sz.Max < math.MaxUint64 {
					replaceCount.Max = sz.Max + 1
				} else {
					replaceCount.Max = sz.Max
				}
				// Include the length of the longest possible original string length.
				retainedSz.Max = sz.Max
			} else if replaceWithSz.Max <= toReplaceSz.Min {
				// If the replacement does not make the result longer, use the original string length.
				replaceCount.Max = 0
				retainedSz.Max = sz.Max
			} else {
				// Replace the smallest possible substrings with the largest possible replacement
				// as many times as possible.
				replaceCount.Max = uint64(math.Ceil(float64(sz.Max) / float64(toReplaceSz.Min)))
			}

			// find the shortest replacement:
			if toReplaceSz.Max == 0 {
				// if the string being replaced is empty, replace surrounds all characters in the input string with the replacement.
				if sz.Min < math.MaxUint64 {
					replaceCount.Min = sz.Min + 1
				} else {
					replaceCount.Min = sz.Min
				}
				// Include the length of the shortest possible original string length.
				retainedSz.Min = sz.Min
			} else if toReplaceSz.Max <= replaceWithSz.Min {
				// If the replacement does not make the result shorter, use the original string length.
				replaceCount.Min = 0
				retainedSz.Min = sz.Min
			} else {
				// Replace the largest possible substrings being with the smallest possible replacement
				// as many times as possible.
				replaceCount.Min = uint64(math.Ceil(float64(sz.Min) / float64(toReplaceSz.Max)))
			}
			size := replaceCount.Multiply(replaceWithSz).Add(retainedSz)

			// cost is the traversal plus the construction of the result
			return &checker.CallEstimate{CostEstimate: sz.MultiplyByCostFactor(2 * common.StringTraversalCostFactor), ResultSize: &size}
		}
	case "split":
		if target != nil {
			sz := l.sizeEstimate(*target)

			// Worst case size is where is that a separator of "" is used, and each char is returned as a list element.
			max := sz.Max
			if len(args) > 1 {
				if v := args[1].Expr().AsLiteral(); v != nil {
					if i, ok := v.Value().(int64); ok {
						max = uint64(i)
					}
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
	case "cidr", "isIP", "isCIDR":
		if target != nil {
			sz := l.sizeEstimate(args[0])
			return &checker.CallEstimate{CostEstimate: sz.MultiplyByCostFactor(common.StringTraversalCostFactor)}
		}
	case "ip":
		if target != nil && len(args) >= 1 {
			if overloadId == "cidr_ip" {
				// The IP member of the CIDR object is just accessing a field.
				// Nominal cost.
				return &checker.CallEstimate{CostEstimate: checker.CostEstimate{Min: 1, Max: 1}}
			}

			sz := l.sizeEstimate(args[0])
			return &checker.CallEstimate{CostEstimate: sz.MultiplyByCostFactor(common.StringTraversalCostFactor)}
		} else if target != nil {
			// The IP member of a CIDR is a just accessing a field, nominal cost.
			return &checker.CallEstimate{CostEstimate: checker.CostEstimate{Min: 1, Max: 1}}
		}
	case "ip.isCanonical":
		if target != nil && len(args) >= 1 {
			sz := l.sizeEstimate(args[0])
			// We have to parse the string and then compare the parsed string to the original string.
			// So we double the cost of parsing the string.
			return &checker.CallEstimate{CostEstimate: sz.MultiplyByCostFactor(2 * common.StringTraversalCostFactor)}
		}
	case "masked", "prefixLength", "family", "isUnspecified", "isLoopback", "isLinkLocalMulticast", "isLinkLocalUnicast", "isGlobalUnicast":
		// IP and CIDR accessors are nominal cost.
		return &checker.CallEstimate{CostEstimate: checker.CostEstimate{Min: 1, Max: 1}}
	case "containsIP":
		if target != nil && len(args) >= 1 {
			// The base cost of the function is the cost of comparing two byte lists.
			// The byte lists will be either ipv4 or ipv6 so will have a length of 4, or 16 bytes.
			sz := checker.SizeEstimate{Min: 4, Max: 16}

			// We have to compare the two strings to determine if the CIDR/IP is in the other CIDR.
			ipCompCost := sz.Add(sz).MultiplyByCostFactor(common.StringTraversalCostFactor)

			if overloadId == "cidr_contains_ip_string" {
				// If we are comparing a string, we must parse the string to into the right type, so add the cost of traversing the string again.
				ipCompCost = ipCompCost.Add(checker.CostEstimate(l.sizeEstimate(args[0])).MultiplyByCostFactor(common.StringTraversalCostFactor))
			}

			return &checker.CallEstimate{CostEstimate: ipCompCost}
		}
	case "containsCIDR":
		if target != nil && len(args) >= 1 {
			// The base cost of the function is the cost of comparing two byte lists.
			// The byte lists will be either ipv4 or ipv6 so will have a length of 4, or 16 bytes.
			sz := checker.SizeEstimate{Min: 4, Max: 16}

			// We have to compare the two strings to determine if the CIDR/IP is in the other CIDR.
			ipCompCost := sz.Add(sz).MultiplyByCostFactor(common.StringTraversalCostFactor)

			// As we are comparing if a CIDR is within another CIDR, we first mask the base CIDR and
			// also compare the CIDR bits.
			// This has an additional cost of the length of the IP being traversed again, plus 1.
			ipCompCost = ipCompCost.Add(sz.MultiplyByCostFactor(common.StringTraversalCostFactor))
			ipCompCost = ipCompCost.Add(checker.CostEstimate{Min: 1, Max: 1})

			if overloadId == "cidr_contains_cidr_string" {
				// If we are comparing a string, we must parse the string to into the right type, so add the cost of traversing the string again.
				ipCompCost = ipCompCost.Add(checker.CostEstimate(l.sizeEstimate(args[0])).MultiplyByCostFactor(common.StringTraversalCostFactor))
			}

			return &checker.CallEstimate{CostEstimate: ipCompCost}
		}
	case "quantity", "isQuantity", "semver", "isSemver":
		if target != nil {
			sz := l.sizeEstimate(args[0])
			return &checker.CallEstimate{CostEstimate: sz.MultiplyByCostFactor(common.StringTraversalCostFactor)}
		}
	case "validate":
		if target != nil {
			sz := l.sizeEstimate(args[0])
			return &checker.CallEstimate{CostEstimate: sz.MultiplyByCostFactor(common.StringTraversalCostFactor).MultiplyByCostFactor(cel.MaxNameFormatRegexSize * common.RegexStringLengthCostFactor)}
		}
	case "format.named":
		return &checker.CallEstimate{CostEstimate: checker.CostEstimate{Min: 1, Max: 1}}
	case "sign", "asInteger", "isInteger", "asApproximateFloat", "isGreaterThan", "isLessThan", "compareTo", "add", "sub", "major", "minor", "patch":
		return &checker.CallEstimate{CostEstimate: checker.CostEstimate{Min: 1, Max: 1}}
	case "getScheme", "getHostname", "getHost", "getPort", "getEscapedPath", "getQuery":
		// url accessors
		return &checker.CallEstimate{CostEstimate: checker.CostEstimate{Min: 1, Max: 1}}
	case "_==_":
		if len(args) == 2 {
			lhs := args[0]
			rhs := args[1]
			if lhs.Type().Equal(rhs.Type()) == types.True {
				t := lhs.Type()
				if t.Kind() == types.OpaqueKind {
					switch t.TypeName() {
					case cel.IPType.TypeName(), cel.CIDRType.TypeName():
						return &checker.CallEstimate{CostEstimate: checker.CostEstimate{Min: 1, Max: 1}}
					}
				}
				if t.Kind() == types.StructKind {
					switch t {
					case cel.QuantityType, AuthorizerType, PathCheckType, // O(1) cost equality checks
						GroupCheckType, ResourceCheckType, DecisionType, cel.SemverType:
						return &checker.CallEstimate{CostEstimate: checker.CostEstimate{Min: 1, Max: 1}}
					case cel.FormatType:
						return &checker.CallEstimate{CostEstimate: checker.CostEstimate{Min: 1, Max: cel.MaxFormatSize}.MultiplyByCostFactor(common.StringTraversalCostFactor)}
					case cel.URLType:
						size := checker.SizeEstimate{Min: 1, Max: 1}
						rhSize := rhs.ComputedSize()
						lhSize := rhs.ComputedSize()
						if rhSize != nil && lhSize != nil {
							size = rhSize.Union(*lhSize)
						}
						return &checker.CallEstimate{CostEstimate: checker.CostEstimate{Min: 1, Max: size.Max}.MultiplyByCostFactor(common.StringTraversalCostFactor)}
					}
				}
				if panicOnUnknown && isRegisteredType(t.TypeName()) {
					panic(fmt.Errorf("EstimateCallCost: unhandled equality for Kubernetes type %v", t))
				}
			}
		}
	}
	if panicOnUnknown && !knownUnhandledFunctions[function] {
		panic(fmt.Errorf("EstimateCallCost: unhandled function %q, target %v, args %v", function, target, args))
	}
	return nil
}

func actualSize(value ref.Val) uint64 {
	if sz, ok := value.(traits.Sizer); ok {
		return uint64(sz.Size().(types.Int))
	}
	if panicOnUnknown {
		// debug.PrintStack()
		panic(fmt.Errorf("actualSize: non-sizer type %T", value))
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
	if params := list.Type().Parameters(); len(params) > 0 {
		lt := params[0]
		nodePath := list.Path()
		if nodePath != nil {
			// Provide path if we have it so that a OpenAPIv3 maxLength validation can be looked up, if it exists
			// for this node.
			path := make([]string, len(nodePath)+1)
			copy(path, nodePath)
			path[len(nodePath)] = "@items"
			return &itemsNode{path: path, t: lt, expr: nil}
		} else {
			// Provide just the type if no path is available so that worst case size can be looked up based on type.
			return &itemsNode{t: lt, expr: nil}
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
	t    *types.Type
	expr ast.Expr
}

func (i *itemsNode) Path() []string {
	return i.path
}

func (i *itemsNode) Type() *types.Type {
	return i.t
}

func (i *itemsNode) Expr() ast.Expr {
	return i.expr
}

func (i *itemsNode) ComputedSize() *checker.SizeEstimate {
	return nil
}

var _ checker.AstNode = (*itemsNode)(nil)

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
