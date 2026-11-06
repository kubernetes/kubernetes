/*
Copyright The Kubernetes Authors.

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

package bridge

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"time"

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v6/value"
)

type elementCase struct {
	Operation string `json:"operation"`
	Left      []int  `json:"left"`
	Right     []int  `json:"right"`
}

type elementOperation struct {
	name string
	run  func(*fieldpath.PathElementSet, *fieldpath.PathElementSet) *fieldpath.PathElementSet
	keep func(bool, bool) bool
}

var elementOperations = []elementOperation{
	{"difference", (*fieldpath.PathElementSet).Difference, func(left, right bool) bool { return left && !right }},
	{"union", (*fieldpath.PathElementSet).Union, func(left, right bool) bool { return left || right }},
	{"intersection", (*fieldpath.PathElementSet).Intersection, func(left, right bool) bool { return left && right }},
}

// Independent ranks exercise every path-element variant and all subset pairs.
func elementComparisons(operations []elementOperation) ([]elementCase, [][]int, error) {
	domain := []fieldpath.PathElement{
		fieldpath.FieldNameElement("left"), fieldpath.FieldNameElement("right"),
		fieldpath.KeyElementByFields("name", "db"),
		fieldpath.ValueElement(value.NewValueInterface("member")),
		fieldpath.IndexElement(0), fieldpath.IndexElement(1),
	}
	for a := range domain {
		for b := range domain {
			if domain[a].Less(domain[b]) != (a < b) || domain[a].Equals(domain[b]) != (a == b) {
				return nil, nil, fmt.Errorf("path-element order/equality differs from independent ranks %d/%d", a, b)
			}
		}
	}

	// Reverse insertion exercises the real sorted representation without deriving its oracle from Less.
	sets := make([]*fieldpath.PathElementSet, 1<<len(domain))
	ranks := make([][]int, len(sets))
	for mask := range sets {
		sets[mask] = &fieldpath.PathElementSet{}
		ranks[mask] = []int{}
		for rank := len(domain) - 1; rank >= 0; rank-- {
			if mask&(1<<rank) != 0 {
				sets[mask].Insert(domain[rank])
			}
		}
		for rank := range domain {
			if mask&(1<<rank) != 0 {
				ranks[mask] = append(ranks[mask], rank)
			}
		}
	}
	cases := make([]elementCase, 0, len(operations)*len(sets)*len(sets))
	results := make([][]int, 0, cap(cases))
	for _, operation := range operations {
		for left := range sets {
			for right := range sets {
				actual := []int{}
				unknown := false
				operation.run(sets[left], sets[right]).Iterate(func(element fieldpath.PathElement) {
					for rank, candidate := range domain {
						if element.Equals(candidate) {
							actual = append(actual, rank)
							return
						}
					}
					unknown = true
				})
				if sets[left].Size() != len(ranks[left]) || sets[right].Size() != len(ranks[right]) {
					return nil, nil, fmt.Errorf("Go element %s mutated input size for masks %d/%d", operation.name, left, right)
				}
				want := []int{}
				for rank, element := range domain {
					if operation.keep(left&(1<<rank) != 0, right&(1<<rank) != 0) {
						want = append(want, rank)
					}
					if sets[left].Has(element) != (left&(1<<rank) != 0) || sets[right].Has(element) != (right&(1<<rank) != 0) {
						return nil, nil, fmt.Errorf("Go element %s mutated input membership for masks %d/%d", operation.name, left, right)
					}
				}
				if unknown || !slices.Equal(actual, want) {
					return nil, nil, fmt.Errorf("Go element %s for masks %d/%d: got %v, want %v", operation.name, left, right, actual, want)
				}
				cases = append(cases, elementCase{Operation: operation.name, Left: ranks[left], Right: ranks[right]})
				results = append(results, actual)
			}
		}
	}
	return cases, results, nil
}

// CompareElementAlgebra checks all three imported methods against Go and independent membership oracles.
func CompareElementAlgebra(ctx context.Context, model string) (int, error) {
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	pairs, err := compareElements(ctx, model, elementOperations)
	if err != nil {
		return 0, err
	}
	// Missing JSON operation fields must retain the original difference-only protocol.
	input := []byte(`[{"left":[0,2],"right":[1,2]},{"operation":"union","left":[0,2],"right":[1,2]}]`)
	output, err := runModel(ctx, model, "--element-difference", input)
	if err != nil {
		return 0, err
	}
	var legacy [][]int
	if err := json.Unmarshal(output, &legacy); err != nil {
		return 0, fmt.Errorf("invalid legacy element-set output: %w", err)
	}
	if len(legacy) != 2 || !slices.Equal(legacy[0], []int{0}) || !slices.Equal(legacy[1], []int{0}) {
		return 0, fmt.Errorf("legacy difference mode changed missing-operation or override behavior: %v", legacy)
	}
	return pairs, nil
}

// CompareElementDifference retains the original single-operation comparison entrypoint.
func CompareElementDifference(ctx context.Context, model string) (int, error) {
	return compareElements(ctx, model, elementOperations[:1])
}

func compareElements(ctx context.Context, model string, operations []elementOperation) (int, error) {
	cases, actual, err := elementComparisons(operations)
	if err != nil {
		return 0, err
	}
	input, err := json.Marshal(cases)
	if err != nil {
		return 0, err
	}
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	output, err := runModel(ctx, model, "--element-algebra", input)
	if err != nil {
		return 0, err
	}
	var lean [][]int
	if err := json.Unmarshal(output, &lean); err != nil {
		return 0, fmt.Errorf("invalid element-set execution output: %w", err)
	}
	if len(lean) != len(cases) {
		return 0, fmt.Errorf("element-set interpreter returned %d results for %d pairs", len(lean), len(cases))
	}
	for index, result := range lean {
		if !slices.Equal(result, actual[index]) {
			return 0, fmt.Errorf("element-set interpreter differs at %+v: got %v, want %v", cases[index], result, actual[index])
		}
	}
	return len(cases), nil
}
