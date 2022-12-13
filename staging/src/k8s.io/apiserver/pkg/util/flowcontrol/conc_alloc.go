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

package flowcontrol

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// allocProblemItem is one of the classes to which computeConcurrencyAllocation should make an allocation
type allocProblemItem struct {
	target     float64
	lowerBound float64
	upperBound float64
}

// relativeAllocItem is like allocProblemItem but with target avoiding zero and the bounds divided by the target
type relativeAllocItem struct {
	target             float64
	relativeLowerBound float64
	relativeUpperBound float64
}

// relativeAllocProblem collects together all the classes and holds the result of sorting by increasing bounds.
// For J <= K, ascendingIndices[J] identifies a bound that is <= the one of ascendingIndices[K].
// When ascendingIndices[J] = 2*N + 0, this identifies the lower bound of items[N].
// When ascendingIndices[J] = 2*N + 1, this identifies the upper bound of items[N].
type relativeAllocProblem struct {
	items            []relativeAllocItem
	ascendingIndices []int
}

// initIndices fills in ascendingIndices and sorts them
func (rap *relativeAllocProblem) initIndices() *relativeAllocProblem {
	rap.ascendingIndices = make([]int, len(rap.items)*2)
	for idx := 0; idx < len(rap.ascendingIndices); idx++ {
		rap.ascendingIndices[idx] = idx
	}
	sort.Sort(rap)
	return rap
}

func (rap *relativeAllocProblem) getItemIndex(idx int) (int, bool) {
	packedIndex := rap.ascendingIndices[idx]
	itemIndex := packedIndex / 2
	return itemIndex, packedIndex == itemIndex*2
}

// decode(J) returns the bound associated with ascendingIndices[J], the associated items index,
// and a bool indicating whether the bound is the item's lower bound.
func (rap *relativeAllocProblem) decode(idx int) (float64, int, bool) {
	itemIdx, lower := rap.getItemIndex(idx)
	if lower {
		return rap.items[itemIdx].relativeLowerBound, itemIdx, lower
	}
	return rap.items[itemIdx].relativeUpperBound, itemIdx, lower
}

func (rap *relativeAllocProblem) getProportion(idx int) float64 {
	prop, _, _ := rap.decode(idx)
	return prop
}

func (rap *relativeAllocProblem) Len() int { return len(rap.items) * 2 }

func (rap *relativeAllocProblem) Less(i, j int) bool {
	return rap.getProportion(i) < rap.getProportion(j)
}

func (rap *relativeAllocProblem) Swap(i, j int) {
	rap.ascendingIndices[i], rap.ascendingIndices[j] = rap.ascendingIndices[j], rap.ascendingIndices[i]
}

// minMax records the minimum and maximum value seen while scanning a set of numbers
type minMax struct {
	min float64
	max float64
}

// note scans one more number
func (mm *minMax) note(x float64) {
	mm.min = math.Min(mm.min, x)
	mm.max = math.Max(mm.max, x)
}

const MinTarget = 0.001
const epsilon = 0.0000001

// computeConcurrencyAllocation returns the unique `allocs []float64`, and
// an associated `fairProp float64`, that jointly have
// all of the following properties (to the degree that floating point calculations allow)
// if possible otherwise returns an error saying why it is impossible.
// `allocs` sums to `requiredSum`.
// For each J in [0, len(classes)):
// (1) `classes[J].lowerBound <= allocs[J] <= classes[J].upperBound` and
// (2) exactly one of the following is true:
// (2a) `allocs[J] == fairProp * classes[J].target`,
// (2b) `allocs[J] == classes[J].lowerBound && classes[J].lowerBound > fairProp * classes[J].target`, or
// (2c) `allocs[J] == classes[J].upperBound && classes[J].upperBound < fairProp * classes[J].target`.
// Each allocProblemItem is required to have `target >= lowerBound >= 0` and `upperBound >= lowerBound`.
// A target smaller than MinTarget is treated as if it were MinTarget.
func computeConcurrencyAllocation(requiredSum int, classes []allocProblemItem) ([]float64, float64, error) {
	if requiredSum < 0 {
		return nil, 0, errors.New("negative sums are not supported")
	}
	requiredSumF := float64(requiredSum)
	var lowSum, highSum, targetSum float64
	ubRange := minMax{min: float64(math.MaxFloat32)}
	lbRange := minMax{min: float64(math.MaxFloat32)}
	relativeItems := make([]relativeAllocItem, len(classes))
	for idx, item := range classes {
		target := item.target
		if item.lowerBound < 0 {
			return nil, 0, fmt.Errorf("lower bound %d is %v but negative lower bounds are not allowed", idx, item.lowerBound)
		}
		if target < item.lowerBound {
			return nil, 0, fmt.Errorf("target %d is %v, which is below its lower bound of %v", idx, target, item.lowerBound)
		}
		if item.upperBound < item.lowerBound {
			return nil, 0, fmt.Errorf("upper bound %d is %v but should not be less than the lower bound %v", idx, item.upperBound, item.lowerBound)
		}
		if target < MinTarget {
			// tweak this to a non-zero value so avoid dividing by zero
			target = MinTarget
		}
		lowSum += item.lowerBound
		highSum += item.upperBound
		targetSum += target
		relativeItem := relativeAllocItem{
			target:             target,
			relativeLowerBound: item.lowerBound / target,
			relativeUpperBound: item.upperBound / target,
		}
		ubRange.note(relativeItem.relativeUpperBound)
		lbRange.note(relativeItem.relativeLowerBound)
		relativeItems[idx] = relativeItem
	}
	if lbRange.max > 1 {
		return nil, 0, fmt.Errorf("lbRange.max-1=%v, which is impossible because lbRange.max can not be greater than 1", lbRange.max-1)
	}
	if lowSum-requiredSumF > epsilon {
		return nil, 0, fmt.Errorf("lower bounds sum to %v, which is higher than the required sum of %v", lowSum, requiredSum)
	}
	if requiredSumF-highSum > epsilon {
		return nil, 0, fmt.Errorf("upper bounds sum to %v, which is lower than the required sum of %v", highSum, requiredSum)
	}
	ans := make([]float64, len(classes))
	if requiredSum == 0 {
		return ans, 0, nil
	}
	if lowSum-requiredSumF > -epsilon { // no wiggle room, constrained from below
		for idx, item := range classes {
			ans[idx] = item.lowerBound
		}
		return ans, lbRange.min, nil
	}
	if requiredSumF-highSum > -epsilon { // no wiggle room, constrained from above
		for idx, item := range classes {
			ans[idx] = item.upperBound
		}
		return ans, ubRange.max, nil
	}
	// Now we know the solution is a unique fairProp in [lbRange.min, ubRange.max].
	// See if the solution does not run into any bounds.
	fairProp := requiredSumF / targetSum
	if lbRange.max <= fairProp && fairProp <= ubRange.min { // no bounds matter
		for idx := range classes {
			ans[idx] = relativeItems[idx].target * fairProp
		}
		return ans, fairProp, nil
	}
	// Sadly, some bounds matter.
	// We find the solution by sorting the bounds and considering progressively
	// higher values of fairProp, starting from lbRange.min.
	rap := (&relativeAllocProblem{items: relativeItems}).initIndices()
	sumSoFar := lowSum
	fairProp = lbRange.min
	var sensitiveTargetSum, deltaSensitiveTargetSum float64
	var numSensitiveClasses, deltaSensitiveClasses int
	var nextIdx int
	// `nextIdx` is the next `rap` index to consider.
	// `sumSoFar` is what the allocs would sum to if the current
	// value of `fairProp` solves the problem.
	// If the current value of fairProp were the answer then
	// `sumSoFar == requiredSum`.
	// Otherwise the next increment in fairProp involves changing the allocations
	// of `numSensitiveClasses` classes whose targets sum to `sensitiveTargetSum`;
	// for the other classes, an upper or lower bound has applied and will continue to apply.
	// The last increment of nextIdx calls for adding `deltaSensitiveClasses`
	// to `numSensitiveClasses` and adding `deltaSensitiveTargetSum` to `sensitiveTargetSum`.
	for sumSoFar < requiredSumF {
		// There might be more than one bound that is equal to the current value
		// of fairProp; find all of them because they will all be relevant to
		// the next change in fairProp.
		// Set nextBound to the next bound that is NOT equal to fairProp,
		// and advance nextIdx to the index of that bound.
		var nextBound float64
		for {
			sensitiveTargetSum += deltaSensitiveTargetSum
			numSensitiveClasses += deltaSensitiveClasses
			if nextIdx >= rap.Len() {
				return nil, 0, fmt.Errorf("impossible: ran out of bounds to consider in bound-constrained problem")
			}
			var itemIdx int
			var lower bool
			nextBound, itemIdx, lower = rap.decode(nextIdx)
			if lower {
				deltaSensitiveClasses = 1
				deltaSensitiveTargetSum = rap.items[itemIdx].target
			} else {
				deltaSensitiveClasses = -1
				deltaSensitiveTargetSum = -rap.items[itemIdx].target
			}
			nextIdx++
			if nextBound > fairProp {
				break
			}
		}
		// fairProp can increase to nextBound without passing any intermediate bounds.
		if numSensitiveClasses == 0 {
			// No classes are affected by the next range of fairProp; skip right past it
			fairProp = nextBound
			continue
		}
		// See whether fairProp can increase to the solution before passing the next bound.
		deltaFairProp := (requiredSumF - sumSoFar) / sensitiveTargetSum
		nextProp := fairProp + deltaFairProp
		if nextProp <= nextBound {
			fairProp = nextProp
			break
		}
		// No, fairProp has to increase above nextBound
		sumSoFar += (nextBound - fairProp) * sensitiveTargetSum
		fairProp = nextBound
	}
	for idx, item := range classes {
		ans[idx] = math.Max(item.lowerBound, math.Min(item.upperBound, fairProp*relativeItems[idx].target))
	}
	return ans, fairProp, nil
}
