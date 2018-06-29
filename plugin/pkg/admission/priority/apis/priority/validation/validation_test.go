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

	internalapi "k8s.io/kubernetes/plugin/pkg/admission/priority/apis/priority"
)

func TestValidateConfiguration(t *testing.T) {

	tests := []struct {
		config     internalapi.Configuration
		testName   string
		testStatus bool
	}{
		{
			config: internalapi.Configuration{
				Namespaces: []string{"kube-system", "namespaces1", "namespaces2"},
			},
			testName:   "Valid cases",
			testStatus: true,
		},
		{
			config: internalapi.Configuration{
				Namespaces: []string{"inval%id"},
			},
			testName:   "Invalid case",
			testStatus: false,
		},
		{
			config: internalapi.Configuration{
				Namespaces: []string{"?invalid"},
			},
			testName:   "Invalid case",
			testStatus: false,
		},
	}

	for i := range tests {
		errs := ValidateConfiguration(&tests[i].config)
		if tests[i].testStatus && errs != nil {
			t.Errorf("Test: %s, expected success: %v", tests[i].testName, errs)
		}
		if !tests[i].testStatus && errs == nil {
			t.Errorf("Test: %s, expected errors: %v", tests[i].testName, errs)
		}
	}
}
