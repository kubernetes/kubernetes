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

var _ fwk.PlacementScorerPlugin = &PlacementBinPacking{}

type PlacementBinPacking struct {
	handle fwk.Handle
	*resourceAllocationScorer
}

const PlacementBinPackingName = names.PlacementBinPacking

func (pl *PlacementBinPacking) Name() string {
	return PlacementBinPackingName
}

func NewPlacementBinPacking(_ context.Context, plArgs runtime.Object, fh fwk.Handle, fts feature.Features) (*PlacementBinPacking, error) {
	args, ok := plArgs.(*config.PlacementBinPackingArgs)
	if !ok {
		return nil, fmt.Errorf("want args to be of type NodeResourcesFitArgs, got %T", plArgs)
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

func (pl *PlacementBinPacking) ScorePlacement(ctx context.Context, state fwk.PodGroupCycleState, podGroup *fwk.PodGroupInfo, placement *fwk.PlacementInfo, podGroupAssignments *fwk.PodGroupAssignments) (int64, *fwk.Status) {
	logger := klog.FromContext(ctx)
	requested := []int64{}
	for _, pod := range podGroup.UnscheduledPods {
		nodeName, ok := podGroupAssignments.UnscheduledPodsToNodes[pod.UID]
		if !ok {
			continue
		}
		if _, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName); err != nil {
			logger.Error(nil, "Pod assignment does not belong to the placement", "pod", klog.KObj(pod), "node", nodeName)
			continue
		}
		podRequests := pl.calculatePodRequestList(pod)
		if len(requested) == 0 {
			requested = make([]int64, len(podRequests))
		}
		for i, request := range podRequests {
			requested[i] += request
		}
	}
	allocatable := make([]int64, len(requested))
	allocated := make([]int64, len(requested))
	for _, node := range placement.PlacementNodes {
		nodeAllocatable, nodeAllocated := pl.calculateNodeAllocatableRequest(ctx, node, requested, nil)
		for i := range requested {
			allocatable[i] += nodeAllocatable[i]
			allocated[i] += nodeAllocated[i]
			requested[i] += nodeAllocated[i]
		}
	}
	return pl.Score(allocatable, allocated, requested), nil
}

func (pl *PlacementBinPacking) PlacementScoreExtensions() fwk.PlacementScoreExtensions {
	return nil
}
