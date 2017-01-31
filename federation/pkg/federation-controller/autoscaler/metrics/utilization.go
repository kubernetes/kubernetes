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

// GetAvgUtilizationForCluster takes in a set of metrics, a set of matching requests,
// and calcuates the current average utilization (returning that and the pods used for calculation)
func GetAvgUtilizationForCluster(metrics PodResourceInfo, requests map[string]int64) (float64, int32, error) {
	metricsTotal := int64(0)
	requestsTotal := int64(0)
	numPods := int32(0)

	for podName, metricValue := range metrics {
		request, hasRequest := requests[podName]
		if !hasRequest {
			// we check for missing requests elsewhere, so assuming missing requests == extraneous metrics
			//IRF:TODO do we need to subtract this from all pods used to calculate average
			continue
		}

		metricsTotal += metricValue
		requestsTotal += request
		numPods += 1
	}

	currentUtilization := int32((metricsTotal * 100) / requestsTotal)
	return float64(currentUtilization), numPods, nil

	//return float64(currentUtilization) / float64(targetUtilization), currentUtilization, nil
}
