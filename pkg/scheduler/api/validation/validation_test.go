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

package validation

import (
	"errors"
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/scheduler/api"
)

func TestValidatePolicy(t *testing.T) {
	tests := []struct {
		policy   api.Policy
		expected error
		name     string
	}{
		{
			name:     "no weight defined in policy",
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "NoWeightPriority"}}},
			expected: errors.New("Priority NoWeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			name:     "policy weight is not positive",
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "NoWeightPriority", Weight: 0}}},
			expected: errors.New("Priority NoWeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			name:     "valid weight priority",
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: 2}}},
			expected: nil,
		},
		{
			name:     "invalid negative weight policy",
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: -2}}},
			expected: errors.New("Priority WeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			name:     "policy weight exceeds maximum",
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: api.MaxWeight}}},
			expected: errors.New("Priority WeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			name:     "valid weight in policy extender config",
			policy:   api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", PrioritizeVerb: "prioritize", Weight: 2}}},
			expected: nil,
		},
		{
			name:     "invalid negative weight in policy extender config",
			policy:   api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", PrioritizeVerb: "prioritize", Weight: -2}}},
			expected: errors.New("Priority for extender http://127.0.0.1:8081/extender should have a positive weight applied to it"),
		},
		{
			name:     "valid filter verb and url prefix",
			policy:   api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", FilterVerb: "filter"}}},
			expected: nil,
		},
		{
			name:     "valid preemt verb and urlprefix",
			policy:   api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", PreemptVerb: "preempt"}}},
			expected: nil,
		},
		{
			name: "invalid multiple extenders",
			policy: api.Policy{
				ExtenderConfigs: []api.ExtenderConfig{
					{URLPrefix: "http://127.0.0.1:8081/extender", BindVerb: "bind"},
					{URLPrefix: "http://127.0.0.1:8082/extender", BindVerb: "bind"},
				}},
			expected: errors.New("Only one extender can implement bind, found 2"),
		},
		{
			name: "invalid duplicate extender resource name",
			policy: api.Policy{
				ExtenderConfigs: []api.ExtenderConfig{
					{URLPrefix: "http://127.0.0.1:8081/extender", ManagedResources: []api.ExtenderManagedResource{{Name: "foo.com/bar"}}},
					{URLPrefix: "http://127.0.0.1:8082/extender", BindVerb: "bind", ManagedResources: []api.ExtenderManagedResource{{Name: "foo.com/bar"}}},
				}},
			expected: errors.New("Duplicate extender managed resource name foo.com/bar"),
		},
		{
			name: "invalid extended resource name",
			policy: api.Policy{
				ExtenderConfigs: []api.ExtenderConfig{
					{URLPrefix: "http://127.0.0.1:8081/extender", ManagedResources: []api.ExtenderManagedResource{{Name: "kubernetes.io/foo"}}},
				}},
			expected: errors.New("kubernetes.io/foo is an invalid extended resource name"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := ValidatePolicy(test.policy)
			if fmt.Sprint(test.expected) != fmt.Sprint(actual) {
				t.Errorf("expected: %s, actual: %s", test.expected, actual)
			}
		})
	}
}
