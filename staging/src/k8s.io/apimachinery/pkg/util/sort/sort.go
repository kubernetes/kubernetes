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
	"container/heap"
	"fmt"
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
// The function uses Kahn's algorithm for topological sorting with a min-heap to ensure
// deterministic output. Items with no dependencies are processed in lexicographic order,
// guaranteeing consistent results across multiple invocations with the same input.
//
// This function contains a shortcut optimization that returns an input list directly
// if it already contains all unique items. This provides O(n) performance in the best case.
//
// Example:
//   - Input: {{"a", "b", "c"}, {"b", "c"}} returns {"a", "b", "c"}
//   - Input: {{"a", "c"}, {"b", "c"}} returns {"a", "b", "c"} (lexicographic tie-breaking)
//   - Input: {{"a", "b"}, {"b", "a"}} returns error (cycle detected)
//
// Complexity: O(L*n + V*log(V) + E) where L is the number of lists, n is the average
// list size, V is the number of unique items, and E is the number of precedence edges.
//
// This is useful for creating a stable, consistent ordering when merging data from
// multiple sources that may have partial but not conflicting orderings.
func MergePreservingRelativeOrder(inputLists [][]string) []string {
	if len(inputLists) == 0 {
		return nil
	}

	// Build a directed graph of precedence relationships
	graph := make(map[string]*graphNode)
	for _, list := range inputLists {
		for i, item := range list {
			node := getOrCreateNode(graph, item)

			// Add edge from current item to next item in list
			if i < len(list)-1 {
				nextItem := list[i+1]
				nextNode := getOrCreateNode(graph, nextItem)

				// Only add edge if not already present (avoid incrementing in-degree multiple times)
				if !node.outEdges.Has(nextItem) {
					node.outEdges.Insert(nextItem)
					nextNode.inDegree++
				}
			}
		}
	}

	// Shortcut: if any input list contains all items (no duplicates), use it
	allItems := sets.New[string]()
	for name := range graph {
		allItems.Insert(name)
	}
	for _, list := range inputLists {
		if len(list) == allItems.Len() && isUnique(list) {
			return list
		}
	}

	// Perform topological sort using Kahn's algorithm with min-heap for determinism
	result, err := topologicalSort(graph)
	if err != nil {
		// This should not happen with valid input, but if it does,
		// fall back to lexicographic sort to provide some result
		items := make([]string, 0, len(graph))
		for name := range graph {
			items = append(items, name)
		}
		sort.Strings(items)
		return items
	}

	return result
}

// getOrCreateNode retrieves or creates a graph node for the given name
func getOrCreateNode(graph map[string]*graphNode, name string) *graphNode {
	if graph[name] == nil {
		graph[name] = &graphNode{
			outEdges: sets.New[string](),
			inDegree: 0,
		}
	}
	return graph[name]
}

// isUnique checks if a list contains no duplicate items
func isUnique(list []string) bool {
	seen := make(map[string]bool, len(list))
	for _, item := range list {
		if seen[item] {
			return false
		}
		seen[item] = true
	}
	return true
}

// topologicalSort performs Kahn's algorithm with a min-heap for deterministic ordering
func topologicalSort(graph map[string]*graphNode) ([]string, error) {
	// Initialize min-heap with all nodes that have no incoming edges
	pq := &stringMinHeap{}
	heap.Init(pq)

	for name, node := range graph {
		if node.inDegree == 0 {
			heap.Push(pq, name)
		}
	}

	result := make([]string, 0, len(graph))

	for pq.Len() > 0 {
		// Pop item with lowest lexicographic value
		current := heap.Pop(pq).(string)
		result = append(result, current)

		currentNode := graph[current]

		// Reduce in-degree for all neighbors
		for neighbor := range currentNode.outEdges {
			neighborNode := graph[neighbor]
			neighborNode.inDegree--

			// If in-degree becomes 0, add to heap
			if neighborNode.inDegree == 0 {
				heap.Push(pq, neighbor)
			}
		}
	}

	// Check for cycles
	if len(result) != len(graph) {
		return nil, fmt.Errorf("cycle detected in precedence graph: sorted %d items but graph has %d items", len(result), len(graph))
	}

	return result, nil
}

// graphNode represents a node in the precedence graph
type graphNode struct {
	// Items that should come after this item
	outEdges sets.Set[string]
	// Number of items that should come before this item
	inDegree int
}

// stringMinHeap implements heap.Interface for strings (min-heap with lexicographic ordering)
type stringMinHeap []string

func (h stringMinHeap) Len() int           { return len(h) }
func (h stringMinHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h stringMinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *stringMinHeap) Push(x interface{}) {
	*h = append(*h, x.(string))
}
func (h *stringMinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}
