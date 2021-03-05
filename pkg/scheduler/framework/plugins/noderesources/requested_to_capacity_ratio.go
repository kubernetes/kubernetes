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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
)

const (
	// RequestedToCapacityRatioName is the name of this plugin.
	RequestedToCapacityRatioName = "RequestedToCapacityRatio"
	maxUtilization               = 100
)

// NewRequestedToCapacityRatio initializes a new plugin and returns it.
func NewRequestedToCapacityRatio(plArgs runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	args, err := getRequestedToCapacityRatioArgs(plArgs)
	if err != nil {
		return nil, err
	}

	if err := validation.ValidateRequestedToCapacityRatioArgs(args); err != nil {
		return nil, err
	}

	shape := make([]helper.FunctionShapePoint, 0, len(args.Shape))
	for _, point := range args.Shape {
		shape = append(shape, helper.FunctionShapePoint{
			Utilization: int64(point.Utilization),
			// MaxCustomPriorityScore may diverge from the max score used in the scheduler and defined by MaxNodeScore,
			// therefore we need to scale the score returned by requested to capacity ratio to the score range
			// used by the scheduler.
			Score: int64(point.Score) * (framework.MaxNodeScore / config.MaxCustomPriorityScore),
		})
	}

	resourceToWeightMap := make(resourceToWeightMap)
	for _, resource := range args.Resources {
		resourceToWeightMap[v1.ResourceName(resource.Name)] = resource.Weight
		if resource.Weight == 0 {
			// Apply the default weight.
			resourceToWeightMap[v1.ResourceName(resource.Name)] = 1
		}
	}

	return &RequestedToCapacityRatio{
		handle: handle,
		resourceAllocationScorer: resourceAllocationScorer{
			RequestedToCapacityRatioName,
			buildRequestedToCapacityRatioScorerFunction(shape, resourceToWeightMap),
			resourceToWeightMap,
		},
	}, nil
}

func getRequestedToCapacityRatioArgs(obj runtime.Object) (config.RequestedToCapacityRatioArgs, error) {
	ptr, ok := obj.(*config.RequestedToCapacityRatioArgs)
	if !ok {
		return config.RequestedToCapacityRatioArgs{}, fmt.Errorf("want args to be of type RequestedToCapacityRatioArgs, got %T", obj)
	}
	return *ptr, nil
}

// RequestedToCapacityRatio is a score plugin that allow users to apply bin packing
// on core resources like CPU, Memory as well as extended resources like accelerators.
type RequestedToCapacityRatio struct {
	handle framework.Handle
	resourceAllocationScorer
}

var _ framework.ScorePlugin = &RequestedToCapacityRatio{}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *RequestedToCapacityRatio) Name() string {
	return RequestedToCapacityRatioName
}

// Score invoked at the score extension point.
func (pl *RequestedToCapacityRatio) Score(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.AsStatus(fmt.Errorf("getting node %q from Snapshot: %w", nodeName, err))
	}
	return pl.score(pod, nodeInfo)
}

// ScoreExtensions of the Score plugin.
func (pl *RequestedToCapacityRatio) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

func buildRequestedToCapacityRatioScorerFunction(scoringFunctionShape helper.FunctionShape, resourceToWeightMap resourceToWeightMap) func(resourceToValueMap, resourceToValueMap, bool, int, int) int64 {
	rawScoringFunction := helper.BuildBrokenLinearFunction(scoringFunctionShape)
	resourceScoringFunction := func(requested, capacity int64) int64 {
		if capacity == 0 || requested > capacity {
			return rawScoringFunction(maxUtilization)
		}

		return rawScoringFunction(maxUtilization - (capacity-requested)*maxUtilization/capacity)
	}
	return func(requested, allocable resourceToValueMap, includeVolumes bool, requestedVolumes int, allocatableVolumes int) int64 {
		var nodeScore, weightSum int64
		for resource, weight := range resourceToWeightMap {
			resourceScore := resourceScoringFunction(requested[resource], allocable[resource])
			if resourceScore > 0 {
				nodeScore += resourceScore * weight
				weightSum += weight
			}
		}
		if weightSum == 0 {
			return 0
		}
		return int64(math.Round(float64(nodeScore) / float64(weightSum)))
	}
}
