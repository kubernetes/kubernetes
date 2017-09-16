/*
Copyright 2017 The Kubernetes Authors.

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

package defaults

import (
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"os"
	"testing"
)

func TestGetMaxVols(t *testing.T) {
	previousValue := os.Getenv(KubeMaxPDVols)
	defaultValue := 39

	tests := []struct {
		rawMaxVols string
		expected   int
		test       string
	}{
		{
			rawMaxVols: "invalid",
			expected:   defaultValue,
			test:       "Unable to parse maximum PD volumes value, using default value",
		},
		{
			rawMaxVols: "-2",
			expected:   defaultValue,
			test:       "Maximum PD volumes must be a positive value, using default value",
		},
		{
			rawMaxVols: "40",
			expected:   40,
			test:       "Parse maximum PD volumes value from env",
		},
	}

	for _, test := range tests {
		os.Setenv(KubeMaxPDVols, test.rawMaxVols)
		result := getMaxVols(defaultValue)
		if result != test.expected {
			t.Errorf("%s: expected %v got %v", test.test, test.expected, result)
		}
	}

	os.Unsetenv(KubeMaxPDVols)
	if previousValue != "" {
		os.Setenv(KubeMaxPDVols, previousValue)
	}
}

func TestCopyAndReplace(t *testing.T) {
	testCases := []struct {
		set         sets.String
		replaceWhat string
		replaceWith string
		expected    sets.String
	}{
		{
			set:         sets.String{"A": sets.Empty{}, "B": sets.Empty{}},
			replaceWhat: "A",
			replaceWith: "C",
			expected:    sets.String{"B": sets.Empty{}, "C": sets.Empty{}},
		},
		{
			set:         sets.String{"A": sets.Empty{}, "B": sets.Empty{}},
			replaceWhat: "D",
			replaceWith: "C",
			expected:    sets.String{"A": sets.Empty{}, "B": sets.Empty{}},
		},
	}
	for _, testCase := range testCases {
		result := copyAndReplace(testCase.set, testCase.replaceWhat, testCase.replaceWith)
		if !result.Equal(testCase.expected) {
			t.Errorf("expected %v got %v", testCase.expected, result)
		}
	}
}

func TestDefaultPriorities(t *testing.T) {
	result := sets.NewString(
		"SelectorSpreadPriority",
		"InterPodAffinityPriority",
		"LeastRequestedPriority",
		"BalancedResourceAllocation",
		"NodePreferAvoidPodsPriority",
		"NodeAffinityPriority",
		"TaintTolerationPriority")
	if expected := defaultPriorities(); !result.Equal(expected) {
		t.Errorf("expected %v got %v", expected, result)
	}
}

func TestDefaultPredicates(t *testing.T) {
	testCases := []struct {
		actionFunc  func(value string) error
		actionParam string
		expected    sets.String
	}{
		{
			actionFunc:  utilfeature.DefaultFeatureGate.Set,
			actionParam: "TaintNodesByCondition=true",
			expected: sets.NewString(
				"NoVolumeZoneConflict",
				"MaxEBSVolumeCount",
				"MaxGCEPDVolumeCount",
				"MaxAzureDiskVolumeCount",
				"MatchInterPodAffinity",
				"NoDiskConflict",
				"GeneralPredicates",
				"CheckNodeMemoryPressure",
				"CheckNodeDiskPressure",
				"NoVolumeNodeConflict",
				"PodToleratesNodeTaints",
			),
		},
		{
			actionFunc:  utilfeature.DefaultFeatureGate.Set,
			actionParam: "TaintNodesByCondition=false",
			expected: sets.NewString(
				"NoVolumeZoneConflict",
				"MaxEBSVolumeCount",
				"MaxGCEPDVolumeCount",
				"MaxAzureDiskVolumeCount",
				"MatchInterPodAffinity",
				"NoDiskConflict",
				"GeneralPredicates",
				"CheckNodeMemoryPressure",
				"CheckNodeDiskPressure",
				"NoVolumeNodeConflict",
				"CheckNodeCondition",
				"PodToleratesNodeTaints",
			),
		},
	}
	for _, testCase := range testCases {
		testCase.actionFunc(testCase.actionParam)
		if result := defaultPredicates(); !result.Equal(testCase.expected) {
			t.Errorf("expected %v got %v", testCase.expected, result)
		}
	}
}
