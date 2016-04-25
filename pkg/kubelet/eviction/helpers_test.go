/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package eviction

import (
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api/resource"
)

func TestParseThresholdConfig(t *testing.T) {
	gracePeriod, _ := time.ParseDuration("30s")
	testCases := map[string]struct {
		evictionHard            string
		evictionSoft            string
		evictionSoftGracePeriod string
		expectErr               bool
		expectThresholds        []Threshold
	}{
		"no values": {
			evictionHard:            "",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			expectErr:               false,
			expectThresholds:        []Threshold{},
		},
		"all flag values": {
			evictionHard:            "memory.available<150Mi",
			evictionSoft:            "memory.available<300Mi",
			evictionSoftGracePeriod: "memory.available=30s",
			expectErr:               false,
			expectThresholds: []Threshold{
				{
					Signal:   SignalMemoryAvailable,
					Operator: OpLessThan,
					Value:    resource.MustParse("150Mi"),
				},
				{
					Signal:      SignalMemoryAvailable,
					Operator:    OpLessThan,
					Value:       resource.MustParse("300Mi"),
					GracePeriod: gracePeriod,
				},
			},
		},
		"invalid-signal": {
			evictionHard:            "mem.available<150Mi",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			expectErr:               true,
			expectThresholds:        []Threshold{},
		},
		"duplicate-signal": {
			evictionHard:            "memory.available<150Mi,memory.available<100Mi",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			expectErr:               true,
			expectThresholds:        []Threshold{},
		},
		"valid-and-invalid-signal": {
			evictionHard:            "memory.available<150Mi,invalid.foo<150Mi",
			evictionSoft:            "",
			evictionSoftGracePeriod: "",
			expectErr:               true,
			expectThresholds:        []Threshold{},
		},
		"soft-no-grace-period": {
			evictionHard:            "",
			evictionSoft:            "memory.available<150Mi",
			evictionSoftGracePeriod: "",
			expectErr:               true,
			expectThresholds:        []Threshold{},
		},
		"soft-neg-grace-period": {
			evictionHard:            "",
			evictionSoft:            "memory.available<150Mi",
			evictionSoftGracePeriod: "memory.available=-30s",
			expectErr:               true,
			expectThresholds:        []Threshold{},
		},
	}
	for testName, testCase := range testCases {
		thresholds, err := ParseThresholdConfig(testCase.evictionHard, testCase.evictionSoft, testCase.evictionSoftGracePeriod)
		if testCase.expectErr != (err != nil) {
			t.Errorf("Err not as expected, test: %v, error expected: %v, actual: %v", testName, testCase.expectErr, err)
		}
		if !thresholdsEqual(testCase.expectThresholds, thresholds) {
			t.Errorf("thresholds not as expected, test: %v, expected: %v, actual: %v", testName, testCase.expectThresholds, thresholds)
		}
	}
}

func thresholdsEqual(expected []Threshold, actual []Threshold) bool {
	if len(expected) != len(actual) {
		return false
	}
	for _, aThreshold := range expected {
		equal := false
		for _, bThreshold := range actual {
			if thresholdEqual(aThreshold, bThreshold) {
				equal = true
			}
		}
		if !equal {
			return false
		}
	}
	for _, aThreshold := range actual {
		equal := false
		for _, bThreshold := range expected {
			if thresholdEqual(aThreshold, bThreshold) {
				equal = true
			}
		}
		if !equal {
			return false
		}
	}
	return true
}

func thresholdEqual(a Threshold, b Threshold) bool {
	return a.GracePeriod == b.GracePeriod &&
		a.Operator == b.Operator &&
		a.Signal == b.Signal &&
		a.Value.Cmp(b.Value) == 0
}
