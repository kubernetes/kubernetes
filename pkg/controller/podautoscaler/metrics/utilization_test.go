/*
Copyright 2018 The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type resourceUtilizationRatioTestCase struct {
	name              string
	metrics           PodMetricsInfo
	requests          map[string]int64
	targetUtilization int32

	expectedUtilizationRatio   float64
	expectedCurrentUtilization int32
	expectedRawAverageValue    int64
	expectedErr                error
}

func TestResourceUtilizationRatio(t *testing.T) {
	testCases := []resourceUtilizationRatioTestCase{
		{
			name: "base case",
			metrics: PodMetricsInfo{
				"test-pod-0": {Value: 50}, "test-pod-1": {Value: 76},
			},
			requests: map[string]int64{
				"test-pod-0": 100, "test-pod-1": 100,
			},
			targetUtilization:          50,
			expectedUtilizationRatio:   1.26,
			expectedCurrentUtilization: 63,
			expectedRawAverageValue:    63,
			expectedErr:                nil,
		},
		{
			name: "ignore pods with no request",
			metrics: PodMetricsInfo{
				"test-pod-0": {Value: 50}, "test-pod-1": {Value: 76}, "test-pod-no-request": {Value: 100},
			},
			requests: map[string]int64{
				"test-pod-0": 100, "test-pod-1": 100,
			},
			targetUtilization:          50,
			expectedUtilizationRatio:   1.26,
			expectedCurrentUtilization: 63,
			expectedRawAverageValue:    63,
			expectedErr:                nil,
		},
		{
			name: "extra request",
			metrics: PodMetricsInfo{
				"test-pod-0": {Value: 50}, "test-pod-1": {Value: 76},
			},
			requests: map[string]int64{
				"test-pod-0": 100, "test-pod-1": 100, "test-pod-extra-request": 500,
			},
			targetUtilization:          50,
			expectedUtilizationRatio:   1.26,
			expectedCurrentUtilization: 63,
			expectedRawAverageValue:    63,
			expectedErr:                nil,
		},
		{
			name: "no request",
			metrics: PodMetricsInfo{
				"test-pod-0": {Value: 50}, "test-pod-1": {Value: 76},
			},
			requests:          map[string]int64{},
			targetUtilization: 50,

			expectedUtilizationRatio:   0,
			expectedCurrentUtilization: 0,
			expectedRawAverageValue:    0,
			expectedErr:                fmt.Errorf("no metrics returned matched known pods"),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actualUtilizationRatio, actualCurrentUtilization, actualRawAverageValue, actualErr := GetResourceUtilizationRatio(tc.metrics, tc.requests, tc.targetUtilization)
			if tc.expectedErr != nil {
				require.Error(t, actualErr, "there should be an error getting the utilization ratio")
				assert.Contains(t, fmt.Sprintf("%v", actualErr), fmt.Sprintf("%v", tc.expectedErr), "the error message should be as expected")
				return
			}
			require.NoError(t, actualErr, "there should be no error retrieving the utilization ratio")
			assert.InEpsilon(t, tc.expectedUtilizationRatio, actualUtilizationRatio, 0.01, "the utilization ratios should be as expected")
			assert.Equal(t, tc.expectedCurrentUtilization, actualCurrentUtilization, "the current utilization should be as expected")
			assert.Equal(t, tc.expectedRawAverageValue, actualRawAverageValue, "the raw average value should be as expected")
		})
	}
}

type metricUsageRatioTestCase struct {
	name        string
	metrics     PodMetricsInfo
	targetUsage int64

	expectedUsageRatio   float64
	expectedCurrentUsage int64
}

func TestMetricUsageRatio(t *testing.T) {
	// TODO: add more unit tests
	testCases := []metricUsageRatioTestCase{
		{
			name: "base case",
			metrics: PodMetricsInfo{
				"test-pod-0": {Value: 5000}, "test-pod-1": {Value: 10000},
			},
			targetUsage:          10000,
			expectedUsageRatio:   .75,
			expectedCurrentUsage: 7500,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actualUsageRatio, actualCurrentUsage := GetMetricUsageRatio(tc.metrics, tc.targetUsage)
			assert.InEpsilon(t, tc.expectedUsageRatio, actualUsageRatio, 0.01, "the usage ratios should be as expected")
			assert.Equal(t, tc.expectedCurrentUsage, actualCurrentUsage, "the current usage should be as expected")
		})
	}
}
