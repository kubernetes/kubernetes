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

	"k8s.io/apimachinery/pkg/labels"
	internalapi "k8s.io/kubernetes/plugin/pkg/admission/podnodeselector/apis/podnodeselector"
)

func TestValidateConfiguration(t *testing.T) {
	tests := []struct {
		config            internalapi.Configuration
		defaultSelector   labels.Set
		namespaceSpecific map[string]labels.Set
		testName          string
		testStatus        bool
	}{
		{
			config: internalapi.Configuration{
				ClusterDefaultNodeSelectors: "env=default",
				NamespaceSelectorsWhitelists: map[string]string{
					"nA": "env=name_a",
					"nB": "env=name_b, foo=bar",
				},
			},
			defaultSelector: labels.Set{"env": "default"},
			namespaceSpecific: map[string]labels.Set{
				"nA": {"env": "name_a"},
				"nB": {"env": "name_b", "foo": "bar"},
			},
			testName:   "Valid cases",
			testStatus: true,
		},
		{
			config: internalapi.Configuration{
				ClusterDefaultNodeSelectors: "env%=default",
				NamespaceSelectorsWhitelists: map[string]string{
					"nA": "env=name,_a",
				},
			},
			testName:   "Invalid cases",
			testStatus: false,
		},
	}

	for _, test := range tests {
		t.Run(test.testName, func(t *testing.T) {
			errs := ValidateConfiguration(&test.config)
			if test.testStatus && errs != nil {
				t.Errorf("expected success: %v", errs)
			}
			if !test.testStatus && errs == nil {
				t.Errorf("expected errors: %v", errs)
			}
		})
	}
}
