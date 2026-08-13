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

package format

import (
	"testing"

	v1 "k8s.io/api/flowcontrol/v1"
)

func Test_FmtPriorityLevelConfigurationSpec(t *testing.T) {
	tests := []struct {
		spec     *v1.PriorityLevelConfigurationSpec
		expected string
	}{
		{
			&v1.PriorityLevelConfigurationSpec{},
			`flowcontrolv1.PriorityLevelConfigurationSpec{Type: ""}`,
		},
		{
			&v1.PriorityLevelConfigurationSpec{Limited: &v1.LimitedPriorityLevelConfiguration{}},
			`flowcontrolv1.PriorityLevelConfigurationSpec{Type: "", Limited: &flowcontrol.LimitedPriorityLevelConfiguration{NominalConcurrencyShares:0, LimitResponse:flowcontrol.LimitResponse{Type:"" } }}`,
		},
	}
	for _, test := range tests {
		result := FmtPriorityLevelConfigurationSpec(test.spec)
		if result != test.expected {
			t.Logf("Failed to format %#v, expected %q, got %q", test.spec, test.expected, result)
			t.Fail()
		}
	}
}
