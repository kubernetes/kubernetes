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

package cm

import (
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	tmbitmask "k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

type numaUtilizationScorer interface {
	GetNUMAUtilizationScores(current, candidate tmbitmask.BitMask) (currentScore, candidateScore int64)
}

type numaScorerAggregator struct {
	cpu numaUtilizationScorer
	mem numaUtilizationScorer
}

// NewNUMAScorerAggregator combines CPU and memory static-manager utilization
// scores for prefer-most-allocated-numa-node, mirroring kube-scheduler's MostAllocated
// weighted-average: aggregateScore = (cpuScore + memScore) / 2 per NUMA.
// Only exclusive CPUs and pinned regular memory are scored; devices and DRA
// resources are not included in the utilization calculation.
func NewNUMAScorerAggregator(cpu cpumanager.Manager, mem memorymanager.Manager) topologymanager.NUMAScorer {
	return &numaScorerAggregator{cpu: cpu, mem: mem}
}

func (a *numaScorerAggregator) Score(current, candidate tmbitmask.BitMask) topologymanager.NUMAScoringResult {
	cpuCur, cpuCand := a.cpu.GetNUMAUtilizationScores(current, candidate)
	memCur, memCand := a.mem.GetNUMAUtilizationScores(current, candidate)
	aggCur := (cpuCur + memCur) / 2
	aggCand := (cpuCand + memCand) / 2
	return topologymanager.NUMAScoringResult{
		PreferCandidate:         aggCand > aggCur,
		Ok:                      aggCur != aggCand,
		CPUScoreCurrent:         cpuCur,
		CPUScoreCandidate:       cpuCand,
		MemScoreCurrent:         memCur,
		MemScoreCandidate:       memCand,
		AggregateScoreCurrent:   aggCur,
		AggregateScoreCandidate: aggCand,
	}
}
