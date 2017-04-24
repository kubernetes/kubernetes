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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	internalapi "k8s.io/kubernetes/plugin/pkg/admission/podpresetrestriction/apis/podpresetrestriction"
)

func TestValidateConfiguration(t *testing.T) {

	tests := []struct {
		config     internalapi.Configuration
		testName   string
		testStatus bool
	}{
		{
			config: internalapi.Configuration{
				DefaultSelector: metav1.LabelSelector{MatchLabels: map[string]string{"component": "redis"}},
			},
			testName:   "Valid case",
			testStatus: true,
		},
		{
			config: internalapi.Configuration{
				DefaultSelector: metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "openshift.io/build.name",
							Operator: "DoesNotExist",
						},
					},
				},
			},
			testName:   "Valid case",
			testStatus: true,
		},
		{
			config: internalapi.Configuration{
				DefaultSelector: metav1.LabelSelector{MatchLabels: map[string]string{"": ""}},
			},
			testName:   "Invalid empty case",
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
