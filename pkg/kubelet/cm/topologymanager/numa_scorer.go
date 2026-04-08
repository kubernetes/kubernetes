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

package topologymanager

import "k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"

// NUMAScoringResult carries the comparison outcome and the scoring breakdown
// so callers can log why a particular NUMA node was selected.
type NUMAScoringResult struct {
	PreferCandidate         bool
	Ok                      bool
	CPUScoreCurrent         int64
	CPUScoreCandidate       int64
	MemScoreCurrent         int64
	MemScoreCandidate       int64
	AggregateScoreCurrent   int64
	AggregateScoreCandidate int64
}

// NUMAScorer scores two single-NUMA candidates by utilization when
// topologyManagerPolicy is single-numa-node and prefer-most-allocated-numa-node is enabled.
type NUMAScorer interface {
	Score(current, candidate bitmask.BitMask) NUMAScoringResult
}
