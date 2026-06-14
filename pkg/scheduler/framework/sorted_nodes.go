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

package framework

import (
	"container/heap"

	fwk "k8s.io/kube-scheduler/framework"
)

var _ SortedScoredNodes = (*sortedNodeScores)(nil)

type sortedNodeScores struct {
	nodes nodeScoreHeap
}

// NewSortedScoredNodes creates a SortedScoredNodes backed by a max-heap keyed on TotalScore.
func NewSortedScoredNodes(nodeScoreList []fwk.NodePluginScores) SortedScoredNodes {
	var h nodeScoreHeap = nodeScoreList
	heap.Init(&h)
	return &sortedNodeScores{nodes: h}
}

func (s *sortedNodeScores) Pop() fwk.NodePluginScores {
	return heap.Pop(&s.nodes).(fwk.NodePluginScores)
}

func (s *sortedNodeScores) Len() int {
	return s.nodes.Len()
}

// UnorderedList returns all nodes in heap-internal order (not sorted by score).
func (s *sortedNodeScores) UnorderedList() []fwk.NodePluginScores {
	result := make([]fwk.NodePluginScores, len(s.nodes))
	copy(result, s.nodes)
	return result
}

// nodeScoreHeap is a heap of fwk.NodePluginScores.
type nodeScoreHeap []fwk.NodePluginScores

var _ heap.Interface = &nodeScoreHeap{}

func (h nodeScoreHeap) Len() int { return len(h) }
func (h nodeScoreHeap) Less(i, j int) bool {
	return (h[i].TotalScore > h[j].TotalScore ||
		(h[i].TotalScore == h[j].TotalScore && h[i].Randomizer > h[j].Randomizer))
}
func (h nodeScoreHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *nodeScoreHeap) Push(x interface{}) {
	*h = append(*h, x.(fwk.NodePluginScores))
}

func (h *nodeScoreHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}
