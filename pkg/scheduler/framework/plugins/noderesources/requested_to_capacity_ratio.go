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
	"math"

	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
)

const maxUtilization = 100

// buildRequestedToCapacityRatioScorerFunction allows users to apply bin packing
// on core resources like CPU, Memory as well as extended resources like accelerators.
func buildRequestedToCapacityRatioScorerFunction(scoringFunctionShape helper.FunctionShape, resourceToWeightMap resourceToWeightMap) func(resourceToValueMap, resourceToValueMap) int64 {
	rawScoringFunction := helper.BuildBrokenLinearFunction(scoringFunctionShape)
	resourceScoringFunction := func(requested, capacity int64) int64 {
		if capacity == 0 || requested > capacity {
			return rawScoringFunction(maxUtilization)
		}

		return rawScoringFunction(requested * maxUtilization / capacity)
	}
	return func(requested, allocable resourceToValueMap) int64 {
		var nodeScore, weightSum int64
		for resource := range requested {
			weight := resourceToWeightMap[resource]
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

func requestedToCapacityRatioScorer(weightMap resourceToWeightMap, shape []config.UtilizationShapePoint) func(resourceToValueMap, resourceToValueMap) int64 {
	shapes := make([]helper.FunctionShapePoint, 0, len(shape))
	for _, point := range shape {
		shapes = append(shapes, helper.FunctionShapePoint{
			Utilization: int64(point.Utilization),
			// MaxCustomPriorityScore may diverge from the max score used in the scheduler and defined by MaxNodeScore,
			// therefore we need to scale the score returned by requested to capacity ratio to the score range
			// used by the scheduler.
			Score: int64(point.Score) * (framework.MaxNodeScore / config.MaxCustomPriorityScore),
		})
	}

	return buildRequestedToCapacityRatioScorerFunction(shapes, weightMap)
}
