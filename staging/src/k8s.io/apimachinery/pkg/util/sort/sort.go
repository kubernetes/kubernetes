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

// SortDiscoveryGroupsTopo performs a topological consensus sort of API discovery group names.
//
// This is needed in Kubernetes API discovery merging logic, where multiple peers (apiserver instances)
// may report different groups. To present a consistent merged discovery response,
// we use this function to compute a consensus ordering that respects the relative orderings
// reported by all peers. This is called from the merged discovery handler when constructing
// the final APIGroupDiscoveryList for clients.
func SortDiscoveryGroupsTopo(peers [][]string) []string {
	items := []string{}
	lessThan := map[string]sets.Set[string]{}
	// build the unsorted list of items, and populate the graph with all the immediate "less than" relationships
	for _, peerItems := range peers {
		for i, lhs := range peerItems {
			if _, seen := lessThan[lhs]; !seen {
				items = append(items, lhs)
				lessThan[lhs] = sets.New[string]()
			}
			if i < len(peerItems)-1 {
				lessThan[lhs].Insert(peerItems[i+1])
			}
		}
	}

	// shortcut if one of the peers has all the items already
	for _, peerItems := range peers {
		if len(peerItems) == len(items) && sets.New(peerItems...).Len() == len(items) {
			copy(items, peerItems)
			return items
		}
	}

	// sort based on finding paths between pairs
	sort.Slice(items, func(i, j int) bool {
		itemI := items[i]
		itemJ := items[j]
		if pathFrom(itemI, itemJ, lessThan) {
			return true
		}
		if pathFrom(itemJ, itemI, lessThan) {
			return false
		}
		// if there's no path, sort lexically
		return itemI < itemJ
	})

	return items
}

func pathFrom(from, to string, links map[string]sets.Set[string]) bool {
	visited := sets.New[string]()
	tovisit := sets.New[string](from)
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
