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

package podgroupcount

import (
	"context"
	"math"

	"k8s.io/apimachinery/pkg/runtime"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
)

// PodGroupCount is a score plugin that favors placements with more pods from the same PodGroup.
type PodGroupCount struct {
	handle fwk.Handle
}

var _ fwk.PlacementScorePlugin = &PodGroupCount{}

const Name = names.PodGroupCount

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, h fwk.Handle) (fwk.Plugin, error) {
	return &PodGroupCount{handle: h}, nil
}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *PodGroupCount) Name() string {
	return Name
}

// ScorePlacement calculates a score for a given Placement.
func (pl *PodGroupCount) ScorePlacement(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, placement *fwk.PodGroupAssignments) (int64, *fwk.Status) {
	pgState, err := pl.handle.PodGroupManager().PodGroupStates().Get(podGroup.GetNamespace(), podGroup.GetWorkloadReference())
	if err != nil {
		return 0, fwk.AsStatus(err)
	}

	count := len(pgState.AssumedPods()) + len(pgState.AssignedPods()) + len(placement.ProposedAssignments)
	return int64(count), nil
}

// PlacementScoreExtensions returns a PlacementScoreExtensions interface if it implements one, or nil if does not.
func (pl *PodGroupCount) PlacementScoreExtensions() fwk.PlacementScoreExtensions {
	return pl
}

// NormalizePlacementScore normalizes the scores to a range of [MinScore, MaxScore].
func (pl *PodGroupCount) NormalizePlacementScore(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, scores []fwk.PlacementScore) *fwk.Status {
	minCount := int64(0)
	maxCount := int64(math.MinInt64)

	for _, score := range scores {
		maxCount = max(maxCount, score.Score)
	}

	if minCount == maxCount {
		for i := range scores {
			scores[i].Score = fwk.MaxScore
		}
		return nil
	}

	for i := range scores {
		scores[i].Score = fwk.MinScore + (scores[i].Score-minCount)*(fwk.MaxScore-fwk.MinScore)/(maxCount-minCount)
	}

	return nil
}
