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

package validation

import (
	"testing"

	resourcequotaapi "k8s.io/kubernetes/plugin/pkg/admission/resourcequota/apis/resourcequota"
)

func TestValidateConfiguration(t *testing.T) {
	successCases := []resourcequotaapi.Configuration{
		{
			LimitedResources: []resourcequotaapi.LimitedResource{
				{
					Resource:      "pods",
					MatchContains: []string{"requests.cpu"},
				},
			},
		},
		{
			LimitedResources: []resourcequotaapi.LimitedResource{
				{
					Resource:      "persistentvolumeclaims",
					MatchContains: []string{"requests.storage"},
				},
			},
		},
	}
	for i := range successCases {
		configuration := successCases[i]
		if errs := ValidateConfiguration(&configuration); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	errorCases := map[string]resourcequotaapi.Configuration{
		"missing apiGroupResource": {LimitedResources: []resourcequotaapi.LimitedResource{
			{MatchContains: []string{"requests.cpu"}},
		}},
	}
	for k, v := range errorCases {
		if errs := ValidateConfiguration(&v); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}
