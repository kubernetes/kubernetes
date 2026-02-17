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

package noderesources

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
)

var _ fwk.PlacementScorePlugin = &PlacementBinPacking{}

type PlacementBinPacking struct {
	handle fwk.Handle
	*resourceAllocationScorer
}

const PlacementBinPackingName = names.PlacementBinPacking

func (pl *PlacementBinPacking) Name() string {
	return PlacementBinPackingName
}

// NewPlacementBinPacking initializes a new placement score plugin and returns it.
func NewPlacementBinPacking(_ context.Context, plArgs runtime.Object, fh fwk.Handle, fts feature.Features) (*PlacementBinPacking, error) {
	args, ok := plArgs.(*config.PlacementBinPackingArgs)
	if !ok {
		return nil, fmt.Errorf("want args to be of type PlacementBinPackingArgs, got %T", plArgs)
	}
	if err := validation.ValidatePlacementBinPackingArgs(nil, args, fts); err != nil {
		return nil, err
	}
	scorer, err := getScorer(args.ScoringStrategy)
	if err != nil {
		return nil, err
	}
	return &PlacementBinPacking{
		handle:                   fh,
		resourceAllocationScorer: scorer,
	}, nil
}

// ScorePlacement scores capacity ratio on all nodes in the placement including all assigned pod group pods' resource requests.
func (pl *PlacementBinPacking) ScorePlacement(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, podGroupAssignments *fwk.PodGroupAssignments) (int64, *fwk.Status) {
	logger := klog.FromContext(ctx)
	requested := make([]int64, len(pl.resources))
	// Calculate requests for the pod group pods scheduled for this placement
	for pod, nodeName := range podGroupAssignments.ProposedAssignments {
		if _, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName); err != nil {
			logger.Error(err, "pod assignment node not found in the snapshot", "pod", klog.KObj(pod), "node", nodeName)
			continue
		}
		podRequests := pl.calculatePodResourceRequestList(pod, pl.resources)
		for i := range len(pl.resources) {
			requested[i] += podRequests[i]
		}
	}
	allocatable := make([]int64, len(pl.resources))
	allocated := make([]int64, len(pl.resources))
	// Calculate resources on all nodes in this placement
	for _, node := range podGroupAssignments.Nodes {
		_, nodeAllocated, nodeAllocatable := pl.calculateNodeAllocatableRequest(ctx, node, requested, nil)
		for i := range pl.resources {
			allocatable[i] += nodeAllocatable[i]
			allocated[i] += nodeAllocated[i]
			// requested includes both the already existing pods
			// and the pod group pods that are being scheduled for this placement
			requested[i] += nodeAllocated[i]
		}
	}
	return pl.scorer(requested, allocated, allocatable), nil
}

func (pl *PlacementBinPacking) PlacementScoreExtensions() fwk.PlacementScoreExtensions {
	return nil
}
