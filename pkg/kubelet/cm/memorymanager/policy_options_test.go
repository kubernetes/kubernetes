/*
Copyright The Kubernetes Authors.

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

package memorymanager

import (
	"fmt"
	"strings"
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
)

func TestNewPolicyOptions(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	testCases := []struct {
		description       string
		featureGateEnable bool
		policyOptions     map[string]string
		expectedErr       error
		expectedOptions   PolicyOptions
	}{
		{
			description: "no options while the feature gate is disabled",
		},
		{
			description:   "an option while the feature gate is disabled",
			policyOptions: map[string]string{MemoryDriftTolerance: "off"},
			expectedErr:   fmt.Errorf("requires the MemoryManagerDriftTolerance feature gate"),
		},
		{
			description:       "no options",
			featureGateEnable: true,
		},
		{
			description:       "off",
			featureGateEnable: true,
			policyOptions:     map[string]string{MemoryDriftTolerance: "off"},
		},
		{
			description:       "auto",
			featureGateEnable: true,
			policyOptions:     map[string]string{MemoryDriftTolerance: "auto"},
			expectedOptions:   PolicyOptions{MaxMemoryDrift: memoryDriftFromKernelImage(logger, procIomemPath)},
		},
		{
			description:       "an explicit quantity",
			featureGateEnable: true,
			policyOptions:     map[string]string{MemoryDriftTolerance: "128Mi"},
			expectedOptions:   PolicyOptions{MaxMemoryDrift: 128 * mb},
		},
		{
			description:       "zero",
			featureGateEnable: true,
			policyOptions:     map[string]string{MemoryDriftTolerance: "0"},
			expectedErr:       fmt.Errorf("must be greater than zero"),
		},
		{
			description:       "a negative quantity",
			featureGateEnable: true,
			policyOptions:     map[string]string{MemoryDriftTolerance: "-1Mi"},
			expectedErr:       fmt.Errorf("must be greater than zero"),
		},
		{
			description:       "not a quantity",
			featureGateEnable: true,
			policyOptions:     map[string]string{MemoryDriftTolerance: "lots"},
			expectedErr:       fmt.Errorf("bad value for option"),
		},
		{
			description:       "an unknown option",
			featureGateEnable: true,
			policyOptions:     map[string]string{"no-such-option": "true"},
			expectedErr:       fmt.Errorf("unknown Memory Manager Policy option"),
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MemoryManagerDriftTolerance, tcase.featureGateEnable)
			opts, err := NewPolicyOptions(logger, tcase.policyOptions)
			if tcase.expectedErr != nil {
				if err == nil {
					t.Errorf("expected error %v, got no error", tcase.expectedErr)
				} else if !strings.Contains(err.Error(), tcase.expectedErr.Error()) {
					t.Errorf("Unexpected error message. Have: %s, wants %s", err.Error(), tcase.expectedErr.Error())
				}
				return
			}
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if opts != tcase.expectedOptions {
				t.Errorf("Expected PolicyOptions to equal %v, not %v", tcase.expectedOptions, opts)
			}
		})
	}
}
