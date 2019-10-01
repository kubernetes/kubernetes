/*
Copyright 2019 The Kubernetes Authors.

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

package generator

import (
	"k8s.io/gengo/types"
)

// an unsafeConversionArbitrator decides whether conversions can be done using unsafe casts.
type unsafeConversionArbitrator struct {
	processedPairs           map[ConversionPair]unsafeConversionDecision
	manualConversionsTracker *ManualConversionsTracker
	functionTagName          string
}

type unsafeConversionDecision int

const (
	// notPossibleOneWay means that we've determined that the unsafe conversion from x to y
	// is not possible, but that decision doesn't hold for conversions from y to x.
	// That happens when there is a non-copy manual conversion function defined from x to y.
	notPossibleOneWay unsafeConversionDecision = iota
	// notPossibleTwoWay means that we've determined that unsafe conversions from x to y
	// and from y to x are both impossible - i.e. x and y don't have the same memory layout.
	notPossibleTwoWay
	// possible means that we've determined that unsafe conversions from x to y
	// and from y to x are both possible - i.e. x and y have the same memory layout.
	possible
)

func newUnsafeConversionArbitrator(manualConversionsTracker *ManualConversionsTracker) *unsafeConversionArbitrator {
	return &unsafeConversionArbitrator{
		processedPairs:           make(map[ConversionPair]unsafeConversionDecision),
		manualConversionsTracker: manualConversionsTracker,
	}
}

// canUseUnsafeConversion returns true iff x can be converted to y using an unsafe conversion.
func (a *unsafeConversionArbitrator) canUseUnsafeConversion(x, y *types.Type) bool {
	// alreadyVisitedTypes holds all the types that have already been checked in the structural type recursion.
	alreadyVisitedTypes := make(map[*types.Type]bool)
	return a.canUseUnsafeConversionWithCaching(x, y, alreadyVisitedTypes) == possible
}

func (a *unsafeConversionArbitrator) canUseUnsafeConversionWithCaching(x, y *types.Type, alreadyVisitedTypes map[*types.Type]bool) unsafeConversionDecision {
	if x == y {
		return possible
	}
	if decision, present := a.processedPairs[ConversionPair{x, y}]; present {
		return decision
	}
	if decision, present := a.processedPairs[ConversionPair{y, x}]; present && decision != notPossibleOneWay {
		return decision
	}

	var result unsafeConversionDecision
	if a.nonCopyOnlyManualConversionFunctionExists(x, y) {
		result = notPossibleOneWay
	} else {
		result = a.canUseUnsafeRecursive(x, y, alreadyVisitedTypes)
	}

	a.processedPairs[ConversionPair{x, y}] = result
	return result
}

// nonCopyOnlyManualConversionFunctionExists returns true iff the manual conversion tracker
// knows of a conversion function from x to y, that is not a copy-only conversion function.
func (a *unsafeConversionArbitrator) nonCopyOnlyManualConversionFunctionExists(x, y *types.Type) bool {
	conversionFunction, exists := a.manualConversionsTracker.preexists(x, y)
	return exists && !isCopyOnlyFunction(conversionFunction, a.functionTagName)
}

// setFunctionTagName sets the function tag name.
// That also invalidates the cache if the new function tag name is different than the previous one.
func (a *unsafeConversionArbitrator) setFunctionTagName(functionTagName string) {
	if a.functionTagName != functionTagName {
		a.functionTagName = functionTagName
		a.processedPairs = make(map[ConversionPair]unsafeConversionDecision)
	}
}

func (a *unsafeConversionArbitrator) canUseUnsafeRecursive(x, y *types.Type, alreadyVisitedTypes map[*types.Type]bool) unsafeConversionDecision {
	in, out := unwrapAlias(x), unwrapAlias(y)
	switch {
	case in == out:
		return possible
	case in.Kind == out.Kind:
		// if the type exists already, return early to avoid recursion
		if alreadyVisitedTypes[in] {
			return possible
		}
		alreadyVisitedTypes[in] = true

		switch in.Kind {
		case types.Struct:
			if len(in.Members) != len(out.Members) {
				return notPossibleTwoWay
			}
			for i, inMember := range in.Members {
				outMember := out.Members[i]
				if decision := a.canUseUnsafeConversionWithCaching(inMember.Type, outMember.Type, alreadyVisitedTypes); decision != possible {
					return decision
				}
			}
			return possible
		case types.Pointer:
			return a.canUseUnsafeConversionWithCaching(in.Elem, out.Elem, alreadyVisitedTypes)
		case types.Map:
			return min(a.canUseUnsafeConversionWithCaching(in.Key, out.Key, alreadyVisitedTypes),
				a.canUseUnsafeConversionWithCaching(in.Elem, out.Elem, alreadyVisitedTypes))
		case types.Slice:
			return a.canUseUnsafeConversionWithCaching(in.Elem, out.Elem, alreadyVisitedTypes)
		case types.Interface:
			// TODO: determine whether the interfaces are actually equivalent - for now, they must have the
			// same type.
			return notPossibleTwoWay
		case types.Builtin:
			if in.Name.Name == out.Name.Name {
				return possible
			}
			return notPossibleTwoWay
		}
	}
	return notPossibleTwoWay
}

func min(a, b unsafeConversionDecision) unsafeConversionDecision {
	if a < b {
		return a
	}
	return b
}
