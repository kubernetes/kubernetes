/*
Copyright 2015 The Kubernetes Authors.

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

package metrics

import (
	"fmt"
)

// GetResourceUtilizationRatio takes in a set of metrics, a set of matching requests,
// and a target utilization percentage, and calcuates the the ratio of
// desired to actual utilization (returning that and the actual utilization)
func GetResourceUtilizationRatio(metrics PodResourceInfo, requests map[string]int64, targetUtilization int32) (float64, int32, error) {
	metricsTotal := int64(0)
	requestsTotal := int64(0)

	for podName, metricValue := range metrics {
		request, hasRequest := requests[podName]
		if !hasRequest {
			// we check for missing requests elsewhere, so assuming missing requests == extraneous metrics
			continue
		}

		metricsTotal += metricValue
		requestsTotal += request
	}

	// if the set of requests is completely disjoint from the set of metrics,
	// then we could have an issue where the requests total is zero
	if requestsTotal == 0 {
		return 0, 0, fmt.Errorf("no metrics returned matched known pods")
	}

	currentUtilization := int32((metricsTotal * 100) / requestsTotal)

	return float64(currentUtilization) / float64(targetUtilization), currentUtilization, nil
}

// GetMetricUtilizationRatio takes in a set of metrics and a target utilization value,
// and calcuates the ratio of desired to actual utilization
// (returning that and the actual utilization)
func GetMetricUtilizationRatio(metrics PodMetricsInfo, targetUtilization float64) (float64, float64) {
	metricsTotal := float64(0)
	for _, metricValue := range metrics {
		metricsTotal += metricValue
	}

	currentUtilization := metricsTotal / float64(len(metrics))

	return currentUtilization / targetUtilization, currentUtilization
}
