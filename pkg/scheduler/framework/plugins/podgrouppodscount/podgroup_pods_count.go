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

package podgrouppodscount

import (
	"context"
	"errors"

	"k8s.io/apimachinery/pkg/runtime"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
)

// PodGroupPodsCount is a placement score plugin that favors placements that can accommodate more pods from the considered PodGroup.
type PodGroupPodsCount struct {
	handle fwk.Handle
}

var _ fwk.PlacementScorePlugin = &PodGroupPodsCount{}

const Name = names.PodGroupPodsCount

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, h fwk.Handle, _ feature.Features) (fwk.Plugin, error) {
	return &PodGroupPodsCount{handle: h}, nil
}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *PodGroupPodsCount) Name() string {
	return Name
}

// ScorePlacement calculates a score for a given Placement.
// Both scheduled (assumed/assigned) pods and the proposed assignments are taken into consideration
// when computing the score. This ensures that the relative difference between choices is reduced,
// and small changes to the total count result in small changes to the score.
func (pl *PodGroupPodsCount) ScorePlacement(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, placement *fwk.PodGroupAssignments) (int64, *fwk.Status) {
	pgState, err := pl.handle.SnapshotSharedLister().PodGroupStates().Get(podGroup.GetNamespace(), podGroup.GetName())
	if err != nil {
		return 0, fwk.AsStatus(err)
	}

	return int64(pgState.ScheduledPodsCount() + len(placement.ProposedAssignments)), nil
}

// PlacementScoreExtensions returns a PlacementScoreExtensions interface if it implements one, or nil if does not.
// PodGroupPodsCount implements this interface.
func (pl *PodGroupPodsCount) PlacementScoreExtensions() fwk.PlacementScoreExtensions {
	return pl
}

// NormalizePlacementScore normalizes the scores to a range of [MinScore, MaxScore].
// The normalization is based on the maximum count among all candidate placements.
// We purposely do not consider MinCount (the minimum pods required for the group) during normalization
// to avoid large gaps in scores when there are minimal differences in pod counts.
func (pl *PodGroupPodsCount) NormalizePlacementScore(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, scores []fwk.PlacementScore) *fwk.Status {
	maxCount := int64(0)

	for _, score := range scores {
		maxCount = max(maxCount, score.Score)
	}

	if maxCount == 0 {
		return fwk.AsStatus(errors.New("no pods from pod group are assigned to any of the candidate placements"))
	}

	for i := range scores {
		scores[i].Score = fwk.MinScore + scores[i].Score*(fwk.MaxScore-fwk.MinScore)/maxCount
	}

	return nil
}
