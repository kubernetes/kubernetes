/*
Copyright 2022 The Kubernetes Authors.

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

package topologymanager

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/utils/ktesting"
)

var fancyBetaOption = "fancy-new-option"
var fancyAlphaOption = "fancy-alpha-option"

type optionAvailTest struct {
	option            string
	featureGate       featuregate.Feature
	featureGateEnable bool
	expectedAvailable bool
}

func TestNewTopologyManagerOptions(t *testing.T) {
	testCases := []struct {
		description       string
		policyOptions     map[string]string
		featureGate       featuregate.Feature
		featureGateEnable bool
		expectedErr       error
		expectedOptions   PolicyOptions
	}{
		{
			description: "return TopologyManagerOptions with PreferClosestNUMA set to true",
			expectedOptions: PolicyOptions{
				PreferClosestNUMA:      true,
				MaxAllowableNUMANodes:  8,
				NUMAAllocationStrategy: NUMAAllocationStrategyNone,
			},
			policyOptions: map[string]string{
				PreferClosestNUMANodes: "true",
				MaxAllowableNUMANodes:  "8",
			},
		},
		{
			description: "return TopologyManagerOptions with MaxAllowableNUMANodes set to 12",
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  12,
				NUMAAllocationStrategy: NUMAAllocationStrategyNone,
			},
			policyOptions: map[string]string{
				MaxAllowableNUMANodes: "12",
			},
		},
		{
			description: "return empty TopologyManagerOptions",
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  8,
				NUMAAllocationStrategy: NUMAAllocationStrategyNone,
			},
		},
		{
			description:       "fail to parse options with error PreferClosestNUMANodes",
			featureGateEnable: true,
			policyOptions: map[string]string{
				PreferClosestNUMANodes: "not a boolean",
			},
			expectedErr: fmt.Errorf("bad value for option"),
		},
		{
			description: "fail to parse options with error MaxAllowableNUMANodes",
			policyOptions: map[string]string{
				MaxAllowableNUMANodes: "can't parse to int",
			},
			expectedErr: fmt.Errorf("unable to convert policy option to integer"),
		},
		{
			description:       "test beta options success",
			featureGate:       pkgfeatures.TopologyManagerPolicyBetaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				fancyBetaOption: "true",
			},
			expectedOptions: PolicyOptions{
				PreferClosestNUMA:      false,
				MaxAllowableNUMANodes:  8,
				NUMAAllocationStrategy: NUMAAllocationStrategyNone,
			},
		},
		{
			description: "test beta options fail",
			featureGate: pkgfeatures.TopologyManagerPolicyBetaOptions,
			policyOptions: map[string]string{
				fancyBetaOption: "true",
			},
			expectedErr: fmt.Errorf("topology manager policy beta-level options not enabled,"),
		},
		{
			description:       "test alpha options success",
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				fancyAlphaOption: "true",
			},
			expectedOptions: PolicyOptions{
				PreferClosestNUMA:      false,
				MaxAllowableNUMANodes:  8,
				NUMAAllocationStrategy: NUMAAllocationStrategyNone,
			},
		},
		{
			description: "test alpha options fail",
			policyOptions: map[string]string{
				fancyAlphaOption: "true",
			},
			expectedErr: fmt.Errorf("topology manager policy alpha-level options not enabled,"),
		},
		{
			description:       "return TopologyManagerOptions with NUMAAllocationStrategy set to most-allocated",
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				NUMAAllocationStrategy: NUMAAllocationStrategyMostAllocated,
			},
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  8,
				NUMAAllocationStrategy: NUMAAllocationStrategyMostAllocated,
			},
		},
		{
			description:       "return TopologyManagerOptions with NUMAAllocationStrategy set to least-allocated",
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				NUMAAllocationStrategy: NUMAAllocationStrategyLeastAllocated,
			},
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  8,
				NUMAAllocationStrategy: NUMAAllocationStrategyLeastAllocated,
			},
		},
		{
			description:       "return TopologyManagerOptions with NUMAAllocationStrategy set to none",
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				NUMAAllocationStrategy: NUMAAllocationStrategyNone,
			},
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  8,
				NUMAAllocationStrategy: NUMAAllocationStrategyNone,
			},
		},
		{
			description:       "an empty NUMAAllocationStrategy means none",
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				NUMAAllocationStrategy: "",
			},
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  8,
				NUMAAllocationStrategy: NUMAAllocationStrategyNone,
			},
		},
		{
			description:       "fail to parse options with unknown NUMAAllocationStrategy",
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				NUMAAllocationStrategy: "most-allocated-ish",
			},
			expectedErr: fmt.Errorf("bad value for option"),
		},
		{
			description: "NUMAAllocationStrategy is rejected unless the alpha options gate is on",
			featureGate: pkgfeatures.TopologyManagerPolicyAlphaOptions,
			policyOptions: map[string]string{
				NUMAAllocationStrategy: NUMAAllocationStrategyMostAllocated,
			},
			expectedErr: fmt.Errorf("topology manager policy alpha-level options not enabled,"),
		},
		{
			description:       "return TopologyManagerOptions with NUMAScoreWeights parsed",
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				NUMAScoreWeights: "cpu=3,memory=1,nvidia.com/gpu=6",
			},
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  8,
				NUMAAllocationStrategy: NUMAAllocationStrategyNone,
				NUMAScoreWeights:       map[string]int{"cpu": 3, "memory": 1, "nvidia.com/gpu": 6},
			},
		},
		{
			description:       "NUMAScoreWeights alongside NUMAAllocationStrategy",
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				NUMAAllocationStrategy: NUMAAllocationStrategyLeastAllocated,
				NUMAScoreWeights:       "cpu=0,memory=0",
			},
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  8,
				NUMAAllocationStrategy: NUMAAllocationStrategyLeastAllocated,
				NUMAScoreWeights:       map[string]int{"cpu": 0, "memory": 0},
			},
		},
		{
			description:       "an empty NUMAScoreWeights means no weights",
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				NUMAScoreWeights: "",
			},
			expectedOptions: PolicyOptions{
				MaxAllowableNUMANodes:  8,
				NUMAAllocationStrategy: NUMAAllocationStrategyNone,
				NUMAScoreWeights:       map[string]int{},
			},
		},
		{
			description:       "fail to parse options with a malformed NUMAScoreWeights",
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				NUMAScoreWeights: "cpu:3",
			},
			expectedErr: fmt.Errorf("bad value for option"),
		},
		{
			description:       "fail to parse options with an out-of-range NUMAScoreWeights",
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				NUMAScoreWeights: "cpu=101",
			},
			expectedErr: fmt.Errorf("must be in range [0, 100]"),
		},
		{
			description: "NUMAScoreWeights is rejected unless the alpha options gate is on",
			featureGate: pkgfeatures.TopologyManagerPolicyAlphaOptions,
			policyOptions: map[string]string{
				NUMAScoreWeights: "cpu=3,memory=1",
			},
			expectedErr: fmt.Errorf("topology manager policy alpha-level options not enabled,"),
		},
	}

	setTopologyManagerOptionsDuringTest(t, betaOptions, fancyBetaOption)
	setTopologyManagerOptionsDuringTest(t, alphaOptions, fancyAlphaOption)

	logger, _ := ktesting.NewTestContext(t)

	for _, tcase := range testCases {
		t.Run(tcase.description, func(t *testing.T) {
			if tcase.featureGate != "" {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, tcase.featureGate, tcase.featureGateEnable)
			}
			opts, err := NewPolicyOptions(logger, tcase.policyOptions)
			if tcase.expectedErr != nil {
				if err == nil {
					t.Errorf("expected error %v, got no error", tcase.expectedErr)
				} else if !strings.Contains(err.Error(), tcase.expectedErr.Error()) {
					t.Errorf("Unexpected error message. Have: %s, wants %s", err.Error(), tcase.expectedErr.Error())
				}
				return
			}

			if !reflect.DeepEqual(opts, tcase.expectedOptions) {
				t.Errorf("Expected TopologyManagerOptions to equal %v, not %v", tcase.expectedOptions, opts)
			}
		})
	}
}

// TestParseAndValidateNUMAScoreWeights covers every row of the validation
// table in the KEP: the accepted forms of the weight string, and each way one
// can be malformed. Parsing and validation are exercised as a pair because
// that is how NewPolicyOptions uses them — a string is only accepted when both
// succeed.
func TestParseAndValidateNUMAScoreWeights(t *testing.T) {
	testCases := []struct {
		description     string
		raw             string
		expectedWeights map[string]int
		expectedErr     string
	}{
		{
			description:     "simple ratios",
			raw:             "cpu=3,memory=1",
			expectedWeights: map[string]int{"cpu": 3, "memory": 1},
		},
		{
			description:     "larger ratios are equivalent, only the proportions matter",
			raw:             "cpu=30,memory=10,nvidia.com/gpu=60",
			expectedWeights: map[string]int{"cpu": 30, "memory": 10, "nvidia.com/gpu": 60},
		},
		{
			description:     "device resource names survive the slash and the dots",
			raw:             "nvidia.com/gpu=6,intel.com/sriov-nic=2,cpu=3",
			expectedWeights: map[string]int{"nvidia.com/gpu": 6, "intel.com/sriov-nic": 2, "cpu": 3},
		},
		{
			description:     "a single resource, the rest left at the default weight",
			raw:             "nvidia.com/gpu=10",
			expectedWeights: map[string]int{"nvidia.com/gpu": 10},
		},
		{
			description:     "weight 0 is accepted as an explicit exclusion",
			raw:             "nvidia.com/gpu=10,cpu=0",
			expectedWeights: map[string]int{"nvidia.com/gpu": 10, "cpu": 0},
		},
		{
			description:     "the bounds of the accepted range are inclusive",
			raw:             "cpu=0,memory=100",
			expectedWeights: map[string]int{"cpu": 0, "memory": 100},
		},
		{
			description:     "an empty value means no weights",
			raw:             "",
			expectedWeights: map[string]int{},
		},
		{
			description:     "surrounding whitespace is tolerated",
			raw:             " cpu = 3 , memory = 1 ",
			expectedWeights: map[string]int{"cpu": 3, "memory": 1},
		},
		{
			description:     "a trailing comma is tolerated",
			raw:             "cpu=3,",
			expectedWeights: map[string]int{"cpu": 3},
		},
		{
			description: "reject a negative weight",
			raw:         "cpu=-5",
			expectedErr: "must be in range [0, 100]",
		},
		{
			description: "reject a weight above the range",
			raw:         "cpu=150",
			expectedErr: "must be in range [0, 100]",
		},
		{
			description: "reject a fractional weight",
			raw:         "cpu=3.5",
			expectedErr: "invalid weight value",
		},
		{
			description: "reject a non-numeric weight",
			raw:         "cpu=abc",
			expectedErr: "invalid weight value",
		},
		{
			description: "reject an entry with no separator",
			raw:         "cpu:5",
			expectedErr: "expected resource=weight",
		},
		{
			description: "reject an empty resource name",
			raw:         "=5",
			expectedErr: "empty resource name",
		},
		{
			description: "reject a malformed entry alongside a valid one",
			raw:         "cpu=3,memory",
			expectedErr: "expected resource=weight",
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.description, func(t *testing.T) {
			weights, err := parseNUMAScoreWeights(tcase.raw)
			if err == nil {
				err = validateNUMAScoreWeights(weights)
			}

			if tcase.expectedErr != "" {
				if err == nil {
					t.Fatalf("expected an error containing %q, got none (weights=%v)", tcase.expectedErr, weights)
				}
				if !strings.Contains(err.Error(), tcase.expectedErr) {
					t.Errorf("Unexpected error message. Have: %s, wants %s", err.Error(), tcase.expectedErr)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(weights, tcase.expectedWeights) {
				t.Errorf("Expected weights to equal %v, not %v", tcase.expectedWeights, weights)
			}
		})
	}
}

// TestValidateNUMAScoreWeightsScaleInvariance pins the property the option
// documentation rests on: weights are not validated against their sum, so
// proportionally equivalent strings are equally acceptable however large the
// numbers get.
func TestValidateNUMAScoreWeightsScaleInvariance(t *testing.T) {
	for _, raw := range []string{"cpu=3,memory=1", "cpu=30,memory=10", "cpu=90,memory=30"} {
		t.Run(raw, func(t *testing.T) {
			weights, err := parseNUMAScoreWeights(raw)
			if err != nil {
				t.Fatalf("unexpected parse error: %v", err)
			}
			if err := validateNUMAScoreWeights(weights); err != nil {
				t.Errorf("unexpected validation error: %v", err)
			}
		})
	}
}

func setTopologyManagerOptionsDuringTest(t *testing.T, optionGroup sets.Set[string], opts ...string) {
	t.Helper()
	t.Cleanup(func() {
		optionGroup.Delete(opts...)
	})
	optionGroup.Insert(opts...)
}

func TestPolicyDefaultsAvailable(t *testing.T) {
	testCases := []optionAvailTest{
		{
			option:            "this-option-does-not-exist",
			expectedAvailable: false,
		},
		{
			option:            PreferClosestNUMANodes,
			expectedAvailable: true,
		},
		{
			option:            MaxAllowableNUMANodes,
			expectedAvailable: true,
		},
		{
			option:            NUMAAllocationStrategy,
			expectedAvailable: false,
		},
		{
			option:            NUMAScoreWeights,
			expectedAvailable: false,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.option, func(t *testing.T) {
			err := CheckPolicyOptionAvailable(testCase.option)
			isEnabled := (err == nil)
			if isEnabled != testCase.expectedAvailable {
				t.Errorf("option %q available got=%v expected=%v", testCase.option, isEnabled, testCase.expectedAvailable)
			}
		})
	}
}

func TestPolicyOptionsAvailable(t *testing.T) {
	testCases := []optionAvailTest{
		{
			option:            "this-option-does-not-exist",
			featureGate:       pkgfeatures.TopologyManagerPolicyBetaOptions,
			featureGateEnable: false,
			expectedAvailable: false,
		},
		{
			option:            "this-option-does-not-exist",
			featureGate:       pkgfeatures.TopologyManagerPolicyBetaOptions,
			featureGateEnable: true,
			expectedAvailable: false,
		},
		{
			option:            PreferClosestNUMANodes,
			featureGate:       pkgfeatures.TopologyManagerPolicyBetaOptions,
			featureGateEnable: false,
			expectedAvailable: true,
		},
		{
			option:            PreferClosestNUMANodes,
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: false,
			expectedAvailable: true,
		},
		{
			option:            MaxAllowableNUMANodes,
			featureGate:       pkgfeatures.TopologyManagerPolicyBetaOptions,
			featureGateEnable: false,
			expectedAvailable: true,
		},
		{
			option:            PreferClosestNUMANodes,
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: false,
			expectedAvailable: true,
		},
		{
			option:            fancyAlphaOption,
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			expectedAvailable: true,
		},
		{
			option:            fancyAlphaOption,
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: false,
			expectedAvailable: false,
		},
		{
			option:            fancyBetaOption,
			featureGate:       pkgfeatures.TopologyManagerPolicyBetaOptions,
			featureGateEnable: true,
			expectedAvailable: true,
		},
		{
			option:            fancyBetaOption,
			featureGate:       pkgfeatures.TopologyManagerPolicyBetaOptions,
			featureGateEnable: false,
			expectedAvailable: false,
		},
		{
			option:            NUMAAllocationStrategy,
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			expectedAvailable: true,
		},
		{
			option:            NUMAAllocationStrategy,
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: false,
			expectedAvailable: false,
		},
		{
			option:            NUMAScoreWeights,
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			expectedAvailable: true,
		},
		{
			option:            NUMAScoreWeights,
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: false,
			expectedAvailable: false,
		},
	}
	setTopologyManagerOptionsDuringTest(t, betaOptions, fancyBetaOption)
	setTopologyManagerOptionsDuringTest(t, alphaOptions, fancyAlphaOption)
	for _, testCase := range testCases {
		t.Run(testCase.option, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, testCase.featureGate, testCase.featureGateEnable)
			defer func() {
				// reset feature flag
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, testCase.featureGate, !testCase.featureGateEnable)
			}()
			err := CheckPolicyOptionAvailable(testCase.option)
			isEnabled := (err == nil)
			if isEnabled != testCase.expectedAvailable {
				t.Errorf("option %q available got=%v expected=%v", testCase.option, isEnabled, testCase.expectedAvailable)
			}
		})
	}
}
