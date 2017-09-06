/*
Copyright 2015 The Kubernetes Authors.

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

package factory

import (
	"testing"

	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/api"
)

func TestAlgorithmNameValidation(t *testing.T) {
	algorithmNamesShouldValidate := []string{
		"1SomeAlgo1rithm",
		"someAlgor-ithm1",
	}
	algorithmNamesShouldNotValidate := []string{
		"-SomeAlgorithm",
		"SomeAlgorithm-",
		"Some,Alg:orithm",
	}
	for _, name := range algorithmNamesShouldValidate {
		if !validName.MatchString(name) {
			t.Errorf("%v should be a valid algorithm name but is not valid.", name)
		}
	}
	for _, name := range algorithmNamesShouldNotValidate {
		if validName.MatchString(name) {
			t.Errorf("%v should be an invalid algorithm name but is valid.", name)
		}
	}
}

func TestValidatePriorityConfigOverFlow(t *testing.T) {
	tests := []struct {
		description string
		configs     []algorithm.PriorityConfig
		expected    bool
	}{
		{
			description: "one of the weights is MaxInt",
			configs:     []algorithm.PriorityConfig{{Weight: api.MaxInt}, {Weight: 5}},
			expected:    true,
		},
		{
			description: "after multiplication with MaxPriority the weight is larger than MaxWeight",
			configs:     []algorithm.PriorityConfig{{Weight: api.MaxInt/api.MaxPriority + api.MaxPriority}, {Weight: 5}},
			expected:    true,
		},
		{
			description: "normal weights",
			configs:     []algorithm.PriorityConfig{{Weight: 10000}, {Weight: 5}},
			expected:    false,
		},
	}
	for _, test := range tests {
		err := validateSelectedConfigs(test.configs)
		if test.expected {
			if err == nil {
				t.Errorf("Expected Overflow for %s", test.description)
			}
		} else {
			if err != nil {
				t.Errorf("Did not expect an overflow for %s", test.description)
			}
		}
	}
}
