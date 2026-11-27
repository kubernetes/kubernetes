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
	"fmt"

	fwk "k8s.io/kube-scheduler/framework"
)

var _ SortedScoredNodes = (*sortedNodeScores)(nil)

type sortedNodeScores struct {
	nodes nodeScoreHeap
}

// NewSortedScoredNodes creates a SortedScoredNodes backed by a max-heap keyed on TotalScore.
func NewSortedScoredNodes(nodeScoreList []fwk.NodePluginScores) SortedScoredNodes {
	return NewSortedRankedNodes(nodeScoreList, nil)
}

// NewSortedRankedNodes creates a SortedScoredNodes that returns nodes of a lower rank first
// (see NodeWithRank) and orders nodes of the same rank like NewSortedScoredNodes does.
// ranks[i] is the rank of nodeScoreList[i]. Nil ranks mean that the nodes are not ranked.
func NewSortedRankedNodes(nodeScoreList []fwk.NodePluginScores, ranks []int) SortedScoredNodes {
	if ranks != nil && len(ranks) != len(nodeScoreList) {
		panic(fmt.Sprintf("ranks length (%d) must match nodeScoreList length (%d)", len(ranks), len(nodeScoreList)))
	}
	h := nodeScoreHeap{scores: nodeScoreList, ranks: ranks}
	heap.Init(&h)
	return &sortedNodeScores{nodes: h}
}

func (s *sortedNodeScores) Pop() fwk.NodePluginScores {
	return heap.Pop(&s.nodes).(fwk.NodePluginScores)
}

func (s *sortedNodeScores) Len() int {
	return s.nodes.Len()
}

func (s *sortedNodeScores) Ranked() bool {
	return s.nodes.ranks != nil
}

// UnorderedList returns all nodes in heap-internal order (not sorted by score).
// Only the outer slice is copied. Each NodePluginScores.Scores slice is still
// shared with the heap. Callers must not mutate the returned elements.
func (s *sortedNodeScores) UnorderedList() []fwk.NodePluginScores {
	result := make([]fwk.NodePluginScores, len(s.nodes.scores))
	copy(result, s.nodes.scores)
	return result
}

// nodeScoreHeap is a heap of fwk.NodePluginScores.
type nodeScoreHeap struct {
	scores []fwk.NodePluginScores
	// ranks is parallel to scores. Keeping it apart from the scores lets the
	// common, unranked case use the caller's slice as it is.
	ranks []int
}

var _ heap.Interface = &nodeScoreHeap{}

func (h *nodeScoreHeap) Len() int { return len(h.scores) }
func (h *nodeScoreHeap) Less(i, j int) bool {
	if h.ranks != nil && h.ranks[i] != h.ranks[j] {
		return h.ranks[i] < h.ranks[j]
	}
	return (h.scores[i].TotalScore > h.scores[j].TotalScore ||
		(h.scores[i].TotalScore == h.scores[j].TotalScore && h.scores[i].Randomizer > h.scores[j].Randomizer))
}
func (h *nodeScoreHeap) Swap(i, j int) {
	h.scores[i], h.scores[j] = h.scores[j], h.scores[i]
	if h.ranks != nil {
		h.ranks[i], h.ranks[j] = h.ranks[j], h.ranks[i]
	}
}

func (h *nodeScoreHeap) Push(x interface{}) {
	if h.ranks != nil {
		panic("nodeScoreHeap.Push is not supported for ranked nodes")
	}
	h.scores = append(h.scores, x.(fwk.NodePluginScores))
}

func (h *nodeScoreHeap) Pop() interface{} {
	last := len(h.scores) - 1
	n := h.scores[last]
	h.scores[last] = fwk.NodePluginScores{}
	h.scores = h.scores[:last]
	if h.ranks != nil {
		h.ranks = h.ranks[:last]
	}
	return n
}
