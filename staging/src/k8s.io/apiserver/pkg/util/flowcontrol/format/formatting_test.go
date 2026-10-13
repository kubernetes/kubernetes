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

	flowcontrol "k8s.io/api/flowcontrol/v1"
)

func Test_FmtPriorityLevelConfigurationSpec(t *testing.T) {
	tests := []struct {
		spec     *flowcontrol.PriorityLevelConfigurationSpec
		expected string
	}{
		{
			&flowcontrol.PriorityLevelConfigurationSpec{},
			`flowcontrolv1.PriorityLevelConfigurationSpec{Type: ""}`,
		},
		{
			&flowcontrol.PriorityLevelConfigurationSpec{
				Limited: &flowcontrol.LimitedPriorityLevelConfiguration{
					NominalConcurrencyShares: new(int32(0)),
				},
			},
			`flowcontrolv1.PriorityLevelConfigurationSpec{Type: "", Limited: &flowcontrol.LimitedPriorityLevelConfiguration{NominalConcurrencyShares:0, LimitResponse:flowcontrol.LimitResponse{Type:"" } }}`,
		},
		{
			&flowcontrol.PriorityLevelConfigurationSpec{
				Limited: &flowcontrol.LimitedPriorityLevelConfiguration{
					NominalConcurrencyShares: new(int32(10)),
				},
			},
			`flowcontrolv1.PriorityLevelConfigurationSpec{Type: "", Limited: &flowcontrol.LimitedPriorityLevelConfiguration{NominalConcurrencyShares:10, LimitResponse:flowcontrol.LimitResponse{Type:"" } }}`,
		},
	}

	for _, test := range tests {
		actual := FmtPriorityLevelConfigurationSpec(test.spec)
		if actual != test.expected {
			t.Errorf("Failed to format PriorityLevelConfigurationSpec %#v; expected %q, got %q", test.spec, test.expected, actual)
		}
	}
}
