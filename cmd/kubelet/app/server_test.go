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

package app

import (
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestValueOfAllocatableResources(t *testing.T) {
	testCases := []struct {
		kubeReserved        map[string]string
		systemReserved      map[string]string
		nodeSwapFeatureGate bool
		errorExpected       bool
		name                string
	}{
		{
			kubeReserved:        map[string]string{"cpu": "200m", "memory": "-150G", "ephemeral-storage": "10Gi", "pid": "1000"},
			systemReserved:      map[string]string{"cpu": "200m", "memory": "15Ki"},
			nodeSwapFeatureGate: false,
			errorExpected:       true,
			name:                "negative quantity value",
		},
		{
			kubeReserved:        map[string]string{"cpu": "200m", "memory": "150Gi", "ephemeral-storage": "10Gi", "pid": "0"},
			systemReserved:      map[string]string{"cpu": "200m", "memory": "15Ky"},
			nodeSwapFeatureGate: false,
			errorExpected:       true,
			name:                "invalid quantity unit",
		},
		{
			kubeReserved:        map[string]string{"cpu": "200m", "memory": "15G", "ephemeral-storage": "10Gi", "pid": "10000"},
			systemReserved:      map[string]string{"cpu": "200m", "memory": "15Ki"},
			nodeSwapFeatureGate: false,
			errorExpected:       false,
			name:                "valid resource quantity",
		},
		{
			systemReserved:      map[string]string{"swap": "-100m"},
			nodeSwapFeatureGate: true,
			errorExpected:       true,
			name:                "invalid negative swap",
		},
		{
			systemReserved:      map[string]string{"swap": "-100xx"},
			nodeSwapFeatureGate: true,
			errorExpected:       true,
			name:                "invalid swap unit",
		},
		{
			systemReserved:      map[string]string{"swap": "100m"},
			nodeSwapFeatureGate: true,
			errorExpected:       false,
			name:                "valid swap",
		},
		{
			systemReserved:      map[string]string{"swap": "100m"},
			nodeSwapFeatureGate: false,
			errorExpected:       true,
			name:                "valid swap but feature gate disabled",
		},
		{
			systemReserved:      map[string]string{"swap": "-100m"},
			nodeSwapFeatureGate: false,
			errorExpected:       true,
			name:                "invalid swap but feature gate disabled",
		},
	}

	for _, test := range testCases {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeSwap, test.nodeSwapFeatureGate)()

		_, err1 := parseResourceList(test.kubeReserved)
		_, err2 := parseResourceList(test.systemReserved)
		if test.errorExpected {
			if err1 == nil && err2 == nil {
				t.Errorf("%s: error expected", test.name)
			}
		} else {
			if err1 != nil || err2 != nil {
				t.Errorf("%s: unexpected error: %v, %v", test.name, err1, err2)
			}
		}
	}
}
