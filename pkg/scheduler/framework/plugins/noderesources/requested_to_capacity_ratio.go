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
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

const (
	// RequestedToCapacityRatioName is the name of this plugin.
	RequestedToCapacityRatioName = "RequestedToCapacityRatio"
	minUtilization               = 0
	maxUtilization               = 100
	minScore                     = 0
	maxScore                     = framework.MaxNodeScore
)

type functionShape []functionShapePoint

type functionShapePoint struct {
	// Utilization is function argument.
	utilization int64
	// Score is function value.
	score int64
}

// NewRequestedToCapacityRatio initializes a new plugin and returns it.
func NewRequestedToCapacityRatio(plArgs runtime.Object, handle framework.FrameworkHandle) (framework.Plugin, error) {
	args, err := getRequestedToCapacityRatioArgs(plArgs)
	if err != nil {
		return nil, err
	}

	shape := make([]functionShapePoint, 0, len(args.Shape))
	for _, point := range args.Shape {
		shape = append(shape, functionShapePoint{
			utilization: int64(point.Utilization),
			// MaxCustomPriorityScore may diverge from the max score used in the scheduler and defined by MaxNodeScore,
			// therefore we need to scale the score returned by requested to capacity ratio to the score range
			// used by the scheduler.
			score: int64(point.Score) * (framework.MaxNodeScore / config.MaxCustomPriorityScore),
		})
	}

	if err := validateFunctionShape(shape); err != nil {
		return nil, err
	}

	resourceToWeightMap := make(resourceToWeightMap)
	for _, resource := range args.Resources {
		resourceToWeightMap[v1.ResourceName(resource.Name)] = resource.Weight
		if resource.Weight == 0 {
			// Apply the default weight.
			resourceToWeightMap[v1.ResourceName(resource.Name)] = 1
		}
	}
	if len(args.Resources) == 0 {
		// If no resources specified, used the default set.
		resourceToWeightMap = defaultRequestedRatioResources
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
	if obj == nil {
		return config.RequestedToCapacityRatioArgs{}, nil
	}
	ptr, ok := obj.(*config.RequestedToCapacityRatioArgs)
	if !ok {
		return config.RequestedToCapacityRatioArgs{}, fmt.Errorf("want args to be of type RequestedToCapacityRatioArgs, got %T", obj)
	}
	return *ptr, nil
}

// RequestedToCapacityRatio is a score plugin that allow users to apply bin packing
// on core resources like CPU, Memory as well as extended resources like accelerators.
type RequestedToCapacityRatio struct {
	handle framework.FrameworkHandle
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
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}
	return pl.score(pod, nodeInfo)
}

// ScoreExtensions of the Score plugin.
func (pl *RequestedToCapacityRatio) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

func validateFunctionShape(shape functionShape) error {
	if len(shape) == 0 {
		return fmt.Errorf("at least one point must be specified")
	}

	for i := 1; i < len(shape); i++ {
		if shape[i-1].utilization >= shape[i].utilization {
			return fmt.Errorf("utilization values must be sorted. Utilization[%d]==%d >= Utilization[%d]==%d", i-1, shape[i-1].utilization, i, shape[i].utilization)
		}
	}

	for i, point := range shape {
		if point.utilization < minUtilization {
			return fmt.Errorf("utilization values must not be less than %d. Utilization[%d]==%d", minUtilization, i, point.utilization)
		}
		if point.utilization > maxUtilization {
			return fmt.Errorf("utilization values must not be greater than %d. Utilization[%d]==%d", maxUtilization, i, point.utilization)
		}
		if point.score < minScore {
			return fmt.Errorf("score values must not be less than %d. Score[%d]==%d", minScore, i, point.score)
		}
		if int64(point.score) > maxScore {
			return fmt.Errorf("score values not be greater than %d. Score[%d]==%d", maxScore, i, point.score)
		}
	}

	return nil
}

func validateResourceWeightMap(resourceToWeightMap resourceToWeightMap) error {
	if len(resourceToWeightMap) == 0 {
		return fmt.Errorf("resourceToWeightMap cannot be nil")
	}

	for resource, weight := range resourceToWeightMap {
		if weight < 1 {
			return fmt.Errorf("resource %s weight %d must not be less than 1", string(resource), weight)
		}
	}
	return nil
}

func buildRequestedToCapacityRatioScorerFunction(scoringFunctionShape functionShape, resourceToWeightMap resourceToWeightMap) func(resourceToValueMap, resourceToValueMap, bool, int, int) int64 {
	rawScoringFunction := buildBrokenLinearFunction(scoringFunctionShape)
	err := validateResourceWeightMap(resourceToWeightMap)
	if err != nil {
		klog.Error(err)
	}
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

// Creates a function which is built using linear segments. Segments are defined via shape array.
// Shape[i].utilization slice represents points on "utilization" axis where different segments meet.
// Shape[i].score represents function values at meeting points.
//
// function f(p) is defined as:
//   shape[0].score for p < f[0].utilization
//   shape[i].score for p == shape[i].utilization
//   shape[n-1].score for p > shape[n-1].utilization
// and linear between points (p < shape[i].utilization)
func buildBrokenLinearFunction(shape functionShape) func(int64) int64 {
	return func(p int64) int64 {
		for i := 0; i < len(shape); i++ {
			if p <= int64(shape[i].utilization) {
				if i == 0 {
					return shape[0].score
				}
				return shape[i-1].score + (shape[i].score-shape[i-1].score)*(p-shape[i-1].utilization)/(shape[i].utilization-shape[i-1].utilization)
			}
		}
		return shape[len(shape)-1].score
	}
}
