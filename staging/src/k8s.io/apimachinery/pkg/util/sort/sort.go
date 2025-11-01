/*
Copyright 2025 The Kubernetes Authors.

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

package sort

import (
	"sort"

	"k8s.io/apimachinery/pkg/util/sets"
)

// MergePreservingRelativeOrder performs a topological consensus sort of items from multiple sources.
// It merges multiple lists of strings into a single list, preserving the relative order of
// elements within each source list.
//
// For any two items, if one appears before the other in any of the input lists,
// that relative order will be preserved in the output. If no relative ordering is
// defined between two items, they are sorted lexicographically.
//
// This function contains a shortcut optimization that returns the first list
// found to contain all items. If the 'inputLists' slice is in a different order on
// two different calls, this shortcut may return a different, validly ordered list.
//
// Example:
//   - Input {{"a", "b", "c"}, {"b", "c", "a"}} might return {"a", "b", "c"}.
//   - Input {{"b", "c", "a"}, {"a", "b", "c"}} might return {"b", "c", "a"}.
//
// This is useful for creating a stable, consistent ordering when merging data from
// multiple sources that may have partial but not conflicting orderings.
func MergePreservingRelativeOrder(inputLists [][]string) []string {
	items := []string{}
	lessThan := map[string]sets.Set[string]{}
	// Build the unsorted list of items, and populate the graph with all the immediate "less than" relationships.
	for _, list := range inputLists {
		for i, lhs := range list {
			if _, seen := lessThan[lhs]; !seen {
				items = append(items, lhs)
				lessThan[lhs] = sets.New[string]()
			}
			if i < len(list)-1 {
				lessThan[lhs].Insert(list[i+1])
			}
		}
	}

	// Shortcut if one of the lists already has all the items already.
	for _, list := range inputLists {
		if len(list) == len(items) && sets.New(list...).Len() == len(items) {
			copy(items, list)
			return items
		}
	}

	// Sort based on finding paths between pairs.
	sort.Slice(items, func(i, j int) bool {
		itemI := items[i]
		itemJ := items[j]
		if pathFrom(itemI, itemJ, lessThan) {
			return true
		}
		if pathFrom(itemJ, itemI, lessThan) {
			return false
		}
		// if there's no path, sort lexically.
		return itemI < itemJ
	})

	return items
}

func pathFrom(from, to string, links map[string]sets.Set[string]) bool {
	visited := sets.New[string]()
	tovisit := sets.New(from)
	for {
		v, ok := tovisit.PopAny()
		if !ok {
			return false
		}
		visited.Insert(v)
		for next := range links[v] {
			if next == to {
				return true
			}
			if !visited.Has(next) {
				tovisit.Insert(next)
			}
		}
	}
}
