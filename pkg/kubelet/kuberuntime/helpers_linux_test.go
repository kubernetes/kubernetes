/*
Copyright 2016 The Kubernetes Authors.

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

package kuberuntime

import (
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestMilliCPUToQuota(t *testing.T) {
	for _, testCase := range []struct {
		msg      string
		input    int64
		expected int64
		period   uint64
	}{
		{
			msg:      "all-zero",
			input:    int64(0),
			expected: int64(0),
			period:   uint64(0),
		},
		{
			msg:      "5 input default quota and period",
			input:    int64(5),
			expected: int64(1000),
			period:   uint64(100000),
		},
		{
			msg:      "9 input default quota and period",
			input:    int64(9),
			expected: int64(1000),
			period:   uint64(100000),
		},
		{
			msg:      "10 input default quota and period",
			input:    int64(10),
			expected: int64(1000),
			period:   uint64(100000),
		},
		{
			msg:      "200 input 20k quota and default period",
			input:    int64(200),
			expected: int64(20000),
			period:   uint64(100000),
		},
		{
			msg:      "500 input 50k quota and default period",
			input:    int64(500),
			expected: int64(50000),
			period:   uint64(100000),
		},
		{
			msg:      "1k input 100k quota and default period",
			input:    int64(1000),
			expected: int64(100000),
			period:   uint64(100000),
		},
		{
			msg:      "1500 input 150k quota and default period",
			input:    int64(1500),
			expected: int64(150000),
			period:   uint64(100000),
		}} {
		t.Run(testCase.msg, func(t *testing.T) {
			quota := milliCPUToQuota(testCase.input, int64(testCase.period))
			if quota != testCase.expected {
				t.Errorf("Input %v and %v, expected quota %v, but got quota %v", testCase.input, testCase.period, testCase.expected, quota)
			}
		})
	}
}

func TestMilliCPUToQuotaWithCustomCPUCFSQuotaPeriod(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUCFSQuotaPeriod, true)()

	for _, testCase := range []struct {
		msg      string
		input    int64
		expected int64
		period   uint64
	}{
		{
			msg:      "all-zero",
			input:    int64(0),
			expected: int64(0),
			period:   uint64(0),
		},
		{
			msg:      "5 input default quota and period",
			input:    int64(5),
			expected: minQuotaPeriod,
			period:   uint64(100000),
		},
		{
			msg:      "9 input default quota and period",
			input:    int64(9),
			expected: minQuotaPeriod,
			period:   uint64(100000),
		},
		{
			msg:      "10 input default quota and period",
			input:    int64(10),
			expected: minQuotaPeriod,
			period:   uint64(100000),
		},
		{
			msg:      "200 input 20k quota and default period",
			input:    int64(200),
			expected: int64(20000),
			period:   uint64(100000),
		},
		{
			msg:      "500 input 50k quota and default period",
			input:    int64(500),
			expected: int64(50000),
			period:   uint64(100000),
		},
		{
			msg:      "1k input 100k quota and default period",
			input:    int64(1000),
			expected: int64(100000),
			period:   uint64(100000),
		},
		{
			msg:      "1500 input 150k quota and default period",
			input:    int64(1500),
			expected: int64(150000),
			period:   uint64(100000),
		},
		{
			msg:      "5 input 10k period and default quota expected",
			input:    int64(5),
			period:   uint64(10000),
			expected: minQuotaPeriod,
		},
		{
			msg:      "5 input 5k period and default quota expected",
			input:    int64(5),
			period:   uint64(5000),
			expected: minQuotaPeriod,
		},
		{
			msg:      "9 input 10k period and default quota expected",
			input:    int64(9),
			period:   uint64(10000),
			expected: minQuotaPeriod,
		},
		{
			msg:      "10 input 200k period and 2000 quota expected",
			input:    int64(10),
			period:   uint64(200000),
			expected: int64(2000),
		},
		{
			msg:      "200 input 200k period and 40k quota",
			input:    int64(200),
			period:   uint64(200000),
			expected: int64(40000),
		},
		{
			msg:      "500 input 20k period and 20k expected quota",
			input:    int64(500),
			period:   uint64(20000),
			expected: int64(10000),
		},
		{
			msg:      "1000 input 10k period and 10k expected quota",
			input:    int64(1000),
			period:   uint64(10000),
			expected: int64(10000),
		},
		{
			msg:      "1500 input 5000 period and 7500 expected quota",
			input:    int64(1500),
			period:   uint64(5000),
			expected: int64(7500),
		}} {
		t.Run(testCase.msg, func(t *testing.T) {
			quota := milliCPUToQuota(testCase.input, int64(testCase.period))
			if quota != testCase.expected {
				t.Errorf("Input %v and %v, expected quota %v, but got quota %v", testCase.input, testCase.period, testCase.expected, quota)
			}
		})
	}
}

func TestSharesToMilliCPU(t *testing.T) {
	knownMilliCPUToShares := map[int64]int64{
		0:    2,
		1:    2,
		2:    2,
		3:    3,
		4:    4,
		32:   32,
		64:   65,
		100:  102,
		250:  256,
		500:  512,
		1000: 1024,
		1500: 1536,
		2000: 2048,
	}

	t.Run("sharesToMilliCPUTest", func(t *testing.T) {
		var testMilliCPU int64
		for testMilliCPU = 0; testMilliCPU <= 2000; testMilliCPU++ {
			shares := milliCPUToShares(testMilliCPU)
			if expectedShares, found := knownMilliCPUToShares[testMilliCPU]; found {
				if shares != expectedShares {
					t.Errorf("Test milliCPIToShares: Input milliCPU %v, expected shares %v, but got %v", testMilliCPU, expectedShares, shares)
				}
			}
			expectedMilliCPU := testMilliCPU
			if testMilliCPU < 2 {
				expectedMilliCPU = 2
			}
			milliCPU := sharesToMilliCPU(shares)
			if milliCPU != expectedMilliCPU {
				t.Errorf("Test sharesToMilliCPU: Input shares %v, expected milliCPU %v, but got %v", shares, expectedMilliCPU, milliCPU)
			}
		}
	})
}

func TestQuotaToMilliCPU(t *testing.T) {
	for _, tc := range []struct {
		name     string
		quota    int64
		period   int64
		expected int64
	}{
		{
			name:     "50m",
			quota:    int64(5000),
			period:   int64(100000),
			expected: int64(50),
		},
		{
			name:     "750m",
			quota:    int64(75000),
			period:   int64(100000),
			expected: int64(750),
		},
		{
			name:     "1000m",
			quota:    int64(100000),
			period:   int64(100000),
			expected: int64(1000),
		},
		{
			name:     "1500m",
			quota:    int64(150000),
			period:   int64(100000),
			expected: int64(1500),
		}} {
		t.Run(tc.name, func(t *testing.T) {
			milliCPU := quotaToMilliCPU(tc.quota, tc.period)
			if milliCPU != tc.expected {
				t.Errorf("Test %s: Input quota %v and period %v, expected milliCPU %v, but got %v", tc.name, tc.quota, tc.period, tc.expected, milliCPU)
			}
		})
	}
}
