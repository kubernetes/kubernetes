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
	"math"
	"math/big"
)

// GetResourceUtilizationRatio takes in a set of metrics, a set of matching requests,
// and a target utilization percentage, and calculates the ratio of
// desired to actual utilization (returning that, the actual utilization, and the raw average value)
func GetResourceUtilizationRatio(metrics PodMetricsInfo, requests map[string]int64, targetUtilization int32) (utilizationRatio float64, currentUtilization int32, rawAverageValue int64, err error) {
	// Values near math.MaxInt64 wrap when summed as int64, and so does the *100
	// taken for the percentage, so the totals are accumulated exactly.
	metricsTotal := new(big.Int)
	requestsTotal := new(big.Int)
	numEntries := int64(0)

	for podName, metric := range metrics {
		request, hasRequest := requests[podName]
		if !hasRequest {
			// we check for missing requests elsewhere, so assuming missing requests == extraneous metrics
			continue
		}

		metricsTotal.Add(metricsTotal, big.NewInt(metric.Value))
		requestsTotal.Add(requestsTotal, big.NewInt(request))
		numEntries++
	}

	// if the set of requests is completely disjoint from the set of metrics,
	// then we could have an issue where the requests total is zero
	if requestsTotal.Sign() == 0 {
		return 0, 0, 0, fmt.Errorf("no metrics returned matched known pods")
	}

	// The percentage is reported as an int32, so an oversized utilization is
	// clamped to the rail instead of being truncated into an arbitrary value.
	percentage := new(big.Int).Mul(metricsTotal, big.NewInt(100))
	percentage.Quo(percentage, requestsTotal)
	switch {
	case percentage.Cmp(big.NewInt(math.MaxInt32)) > 0:
		currentUtilization = math.MaxInt32
	case percentage.Cmp(big.NewInt(math.MinInt32)) < 0:
		currentUtilization = math.MinInt32
	default:
		currentUtilization = int32(percentage.Int64())
	}

	// the average of int64 values always fits in an int64
	rawAverageValue = metricsTotal.Quo(metricsTotal, big.NewInt(numEntries)).Int64()

	return float64(currentUtilization) / float64(targetUtilization), currentUtilization, rawAverageValue, nil
}

// GetMetricUsageRatio takes in a set of metrics and a target usage value,
// and calculates the ratio of desired to actual usage
// (returning that and the actual usage)
func GetMetricUsageRatio(metrics PodMetricsInfo, targetUsage int64) (usageRatio float64, currentUsage int64) {
	metricsTotal := int64(0)
	for _, metric := range metrics {
		metricsTotal += metric.Value
	}

	currentUsage = metricsTotal / int64(len(metrics))

	return float64(currentUsage) / float64(targetUsage), currentUsage
}
