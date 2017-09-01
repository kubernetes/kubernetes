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

	"k8s.io/kubernetes/plugin/pkg/scheduler/api"
)

func TestValidatePolicy(t *testing.T) {
	tests := []struct {
		policy   api.Policy
		expected error
	}{
		{
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "NoWeightPriority"}}},
			expected: errors.New("Priority NoWeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "NoWeightPriority", Weight: 0}}},
			expected: errors.New("Priority NoWeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: 2}}},
			expected: nil,
		},
		{
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: -2}}},
			expected: errors.New("Priority WeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			policy:   api.Policy{Priorities: []api.PriorityPolicy{{Name: "WeightPriority", Weight: api.MaxWeight}}},
			expected: errors.New("Priority WeightPriority should have a positive weight applied to it or it has overflown"),
		},
		{
			policy:   api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", FilterVerb: "filter", Weight: 2}}},
			expected: nil,
		},
		{
			policy:   api.Policy{ExtenderConfigs: []api.ExtenderConfig{{URLPrefix: "http://127.0.0.1:8081/extender", FilterVerb: "filter", Weight: -2}}},
			expected: errors.New("Priority for extender http://127.0.0.1:8081/extender should have a positive weight applied to it"),
		},
		{
			policy: api.Policy{
				ExtenderConfigs: []api.ExtenderConfig{
					{URLPrefix: "http://127.0.0.1:8081/extender", BindVerb: "bind", Weight: 2},
					{URLPrefix: "http://127.0.0.1:8082/extender", BindVerb: "bind", Weight: 2},
				}},
			expected: errors.New("Only one extender can implement bind, found 2"),
		},
	}

	for _, test := range tests {
		actual := ValidatePolicy(test.policy)
		if fmt.Sprint(test.expected) != fmt.Sprint(actual) {
			t.Errorf("expected: %s, actual: %s", test.expected, actual)
		}
	}
}
