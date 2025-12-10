/*
Copyright 2019 The Kubernetes Authors.

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
	"math"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/dynamic-resource-allocation/structured"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
)

// BalancedAllocation is a score plugin that calculates the difference between the cpu and memory fraction
// of capacity, and prioritizes the host based on how close the two metrics are to each other.
type BalancedAllocation struct {
	handle fwk.Handle
	resourceAllocationScorer
}

var _ fwk.PreScorePlugin = &BalancedAllocation{}
var _ fwk.ScorePlugin = &BalancedAllocation{}
var _ fwk.SignPlugin = &BalancedAllocation{}

// BalancedAllocationName is the name of the plugin used in the plugin registry and configurations.
const (
	BalancedAllocationName = names.NodeResourcesBalancedAllocation

	// balancedAllocationPreScoreStateKey is the key in CycleState to NodeResourcesBalancedAllocation pre-computed data for Scoring.
	balancedAllocationPreScoreStateKey = "PreScore" + BalancedAllocationName
)

// draPreScoreState holds the pre-computed data for DRA extended resources scoring.
type draPreScoreState struct {
	// allocatedState holds the DRA allocated state for DRA extended resources scoring.
	allocatedState *structured.AllocatedState
	// resourceSlices holds the list of resource slices for DRA extended resource scoring.
	resourceSlices []*resourceapi.ResourceSlice
}

// balancedAllocationPreScoreState computed at PreScore and used at Score.
type balancedAllocationPreScoreState struct {
	// podRequests have the same order of the resources defined in NodeResourcesFitArgs.Resources,
	// same for other place we store a list like that.
	podRequests []int64
	// DRA extended resource scoring state.
	*draPreScoreState
}

// Clone implements the mandatory Clone interface. We don't really copy the data since
// there is no need for that.
func (s *balancedAllocationPreScoreState) Clone() fwk.StateData {
	return s
}

// PreScore calculates incoming pod's resource requests and writes them to the cycle state used.
func (ba *BalancedAllocation) PreScore(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) *fwk.Status {
	podRequests := ba.calculatePodResourceRequestList(pod, ba.resources)
	if ba.isBestEffortPod(podRequests) {
		// Skip BalancedAllocation scoring for best-effort pods to
		// prevent a large number of pods from being scheduled to the same node.
		// See https://github.com/kubernetes/kubernetes/issues/129138 for details.
		return fwk.NewStatus(fwk.Skip)
	}
	state := &balancedAllocationPreScoreState{
		podRequests: podRequests,
	}
	if ba.enableDRAExtendedResource {
		draPreScoreState, status := getDRAPreScoredParams(ba.draManager, ba.resources)
		if status != nil {
			return status
		}
		if draPreScoreState != nil {
			state.draPreScoreState = draPreScoreState
		}
	}
	cycleState.Write(balancedAllocationPreScoreStateKey, state)
	return nil
}

func getBalancedAllocationPreScoreState(cycleState fwk.CycleState) (*balancedAllocationPreScoreState, error) {
	c, err := cycleState.Read(balancedAllocationPreScoreStateKey)
	if err != nil {
		return nil, fmt.Errorf("reading %q from cycleState: %w", balancedAllocationPreScoreStateKey, err)
	}

	s, ok := c.(*balancedAllocationPreScoreState)
	if !ok {
		return nil, fmt.Errorf("invalid PreScore state, got type %T", c)
	}
	return s, nil
}

// Name returns name of the plugin. It is used in logs, etc.
func (ba *BalancedAllocation) Name() string {
	return BalancedAllocationName
}

// Filtering and scoring based on the container resources and overheads.
func (pl *BalancedAllocation) SignPod(ctx context.Context, pod *v1.Pod) ([]fwk.SignFragment, *fwk.Status) {
	opts := ResourceRequestsOptions{
		EnablePodLevelResources:   pl.enablePodLevelResources,
		EnableDRAExtendedResource: pl.enableDRAExtendedResource,
	}

	if pl.enableDRAExtendedResource {
		return nil, fwk.NewStatus(fwk.Unschedulable, "signature disabled when dra extended resources enabled")
	}

	return []fwk.SignFragment{
		{Key: fwk.ResourcesSignerName, Value: computePodResourceRequest(pod, opts)},
	}, nil
}

// Score invoked at the score extension point.
func (ba *BalancedAllocation) Score(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) (int64, *fwk.Status) {
	s, err := getBalancedAllocationPreScoreState(state)
	if err != nil {
		s = &balancedAllocationPreScoreState{podRequests: ba.calculatePodResourceRequestList(pod, ba.resources)}
		if ba.isBestEffortPod(s.podRequests) {
			return 0, nil
		}
		if ba.enableDRAExtendedResource {
			draPreScoreState, status := getDRAPreScoredParams(ba.draManager, ba.resources)
			if status != nil {
				return 0, status
			}
			if draPreScoreState != nil {
				s.draPreScoreState = draPreScoreState
			}
		}
	}

	// ba.score favors nodes with balanced resource usage rate.
	// It calculates the standard deviation for those resources and prioritizes the node based on how close the usage of those resources is to each other.
	// Detail: score = (1 - std) * MaxNodeScore, where std is calculated by the root square of Σ((fraction(i)-mean)^2)/len(resources)
	// The algorithm is partly inspired by:
	// "Wei Huang et al. An Energy Efficient Virtual Machine Placement Algorithm with Balanced Resource Utilization"
	return ba.score(ctx, pod, nodeInfo, s.podRequests, s.draPreScoreState)
}

// ScoreExtensions of the Score plugin.
func (ba *BalancedAllocation) ScoreExtensions() fwk.ScoreExtensions {
	return nil
}

// NewBalancedAllocation initializes a new plugin and returns it.
func NewBalancedAllocation(_ context.Context, baArgs runtime.Object, h fwk.Handle, fts feature.Features) (fwk.Plugin, error) {
	args, ok := baArgs.(*config.NodeResourcesBalancedAllocationArgs)
	if !ok {
		return nil, fmt.Errorf("want args to be of type NodeResourcesBalancedAllocationArgs, got %T", baArgs)
	}

	if err := validation.ValidateNodeResourcesBalancedAllocationArgs(nil, args); err != nil {
		return nil, err
	}

	return &BalancedAllocation{
		handle: h,
		resourceAllocationScorer: resourceAllocationScorer{
			Name:                            BalancedAllocationName,
			enableInPlacePodVerticalScaling: fts.EnableInPlacePodVerticalScaling,
			enablePodLevelResources:         fts.EnablePodLevelResources,
			enableDRAExtendedResource:       fts.EnableDRAExtendedResource,
			scorer:                          balancedResourceScorer,
			useRequested:                    true,
			resources:                       args.Resources,
			enableInPlacePodLevelResourcesVerticalScaling: fts.EnableInPlacePodLevelResourcesVerticalScaling,
		},
	}, nil
}

func balancedResourceScorer(requested, allocable []int64) int64 {
	var resourceToFractions []float64
	var totalFraction float64
	for i := range requested {
		if allocable[i] == 0 {
			continue
		}
		fraction := float64(requested[i]) / float64(allocable[i])
		if fraction > 1 {
			fraction = 1
		}
		totalFraction += fraction
		resourceToFractions = append(resourceToFractions, fraction)
	}

	std := 0.0

	// For most cases, resources are limited to cpu and memory, the std could be simplified to std := (fraction1-fraction2)/2
	// len(fractions) > 2: calculate std based on the well-known formula - root square of Σ((fraction(i)-mean)^2)/len(fractions)
	// Otherwise, set the std to zero is enough.
	if len(resourceToFractions) == 2 {
		std = math.Abs((resourceToFractions[0] - resourceToFractions[1]) / 2)
	} else if len(resourceToFractions) > 2 {
		mean := totalFraction / float64(len(resourceToFractions))
		var sum float64
		for _, fraction := range resourceToFractions {
			sum = sum + (fraction-mean)*(fraction-mean)
		}
		std = math.Sqrt(sum / float64(len(resourceToFractions)))
	}

	// STD (standard deviation) is always a positive value. 1-deviation lets the score to be higher for node which has least deviation and
	// multiplying it with `MaxNodeScore` provides the scaling factor needed.
	return int64((1 - std) * float64(fwk.MaxNodeScore))
}
