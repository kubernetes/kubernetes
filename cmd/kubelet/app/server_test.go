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

	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
)

func TestValueOfAllocatableResources(t *testing.T) {
	testCases := []struct {
		kubeReserved   string
		systemReserved string
		errorExpected  bool
		name           string
	}{
		{
			kubeReserved:   "cpu=200m,memory=-150G",
			systemReserved: "cpu=200m,memory=15Ki",
			errorExpected:  true,
			name:           "negative quantity value",
		},
		{
			kubeReserved:   "cpu=200m,memory=150Gi",
			systemReserved: "cpu=200m,memory=15Ky",
			errorExpected:  true,
			name:           "invalid quantity unit",
		},
		{
			kubeReserved:   "cpu=200m,memory=15G",
			systemReserved: "cpu=200m,memory=15Ki",
			errorExpected:  false,
			name:           "Valid resource quantity",
		},
	}

	for _, test := range testCases {
		kubeReservedCM := make(kubeletconfig.ConfigurationMap)
		systemReservedCM := make(kubeletconfig.ConfigurationMap)

		kubeReservedCM.Set(test.kubeReserved)
		systemReservedCM.Set(test.systemReserved)

		_, err1 := parseResourceList(kubeReservedCM)
		_, err2 := parseResourceList(systemReservedCM)
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
