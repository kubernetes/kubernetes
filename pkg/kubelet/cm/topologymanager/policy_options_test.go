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
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
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
			description:       "return TopologyManagerOptions with PreferClosestNUMA set to true",
			featureGate:       pkgfeatures.TopologyManagerPolicyBetaOptions,
			featureGateEnable: true,
			expectedOptions: PolicyOptions{
				PreferClosestNUMA: true,
			},
			policyOptions: map[string]string{
				PreferClosestNUMANodes: "true",
			},
		},
		{
			description: "fail to set option when TopologyManagerPolicyBetaOptions feature gate is not set",
			featureGate: pkgfeatures.TopologyManagerPolicyBetaOptions,
			policyOptions: map[string]string{
				PreferClosestNUMANodes: "true",
			},
			expectedErr: fmt.Errorf("Topology Manager Policy Beta-level Options not enabled,"),
		},
		{
			description: "return empty TopologyManagerOptions",
		},
		{
			description:       "fail to parse options",
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				PreferClosestNUMANodes: "not a boolean",
			},
			expectedErr: fmt.Errorf("bad value for option"),
		},
		{
			description:       "test beta options success",
			featureGate:       pkgfeatures.TopologyManagerPolicyBetaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				fancyBetaOption: "true",
			},
		},
		{
			description: "test beta options fail",
			featureGate: pkgfeatures.TopologyManagerPolicyBetaOptions,
			policyOptions: map[string]string{
				fancyBetaOption: "true",
			},
			expectedErr: fmt.Errorf("Topology Manager Policy Beta-level Options not enabled,"),
		},
		{
			description:       "test alpha options success",
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: true,
			policyOptions: map[string]string{
				fancyAlphaOption: "true",
			},
		},
		{
			description: "test alpha options fail",
			policyOptions: map[string]string{
				fancyAlphaOption: "true",
			},
			expectedErr: fmt.Errorf("Topology Manager Policy Alpha-level Options not enabled,"),
		},
	}

	betaOptions.Insert(fancyBetaOption)
	alphaOptions = sets.New[string](fancyAlphaOption)

	for _, tcase := range testCases {
		t.Run(tcase.description, func(t *testing.T) {
			if tcase.featureGate != "" {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, tcase.featureGate, tcase.featureGateEnable)()
			}
			opts, err := NewPolicyOptions(tcase.policyOptions)
			if tcase.expectedErr != nil {
				if !strings.Contains(err.Error(), tcase.expectedErr.Error()) {
					t.Errorf("Unexpected error message. Have: %s wants %s", err.Error(), tcase.expectedErr.Error())
				}
			}

			if opts != tcase.expectedOptions {
				t.Errorf("Expected TopologyManagerOptions to equal %v, not %v", tcase.expectedOptions, opts)

			}
		})
	}
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
			featureGateEnable: true,
			expectedAvailable: true,
		},
		{
			option:            PreferClosestNUMANodes,
			featureGate:       pkgfeatures.TopologyManagerPolicyAlphaOptions,
			featureGateEnable: false,
			expectedAvailable: true,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.option, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, testCase.featureGate, testCase.featureGateEnable)()
			err := CheckPolicyOptionAvailable(testCase.option)
			isEnabled := (err == nil)
			if isEnabled != testCase.expectedAvailable {
				t.Errorf("option %q available got=%v expected=%v", testCase.option, isEnabled, testCase.expectedAvailable)
			}
		})
	}
}
