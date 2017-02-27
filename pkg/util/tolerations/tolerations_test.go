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

package tolerations

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

func TestVerifyAgainstWhitelist(t *testing.T) {
	tests := []struct {
		input      []api.Toleration
		whitelist  []api.Toleration
		testName   string
		testStatus bool
	}{
		{
			input:      []api.Toleration{{Key: "foo", Operator: "Equal", Value: "bar", Effect: "NoSchedule"}},
			whitelist:  []api.Toleration{{Key: "foo", Operator: "Equal", Value: "bar", Effect: "NoSchedule"}},
			testName:   "equal input and whitelist",
			testStatus: true,
		},
		{
			input:      []api.Toleration{{Key: "foo", Operator: "Equal", Value: "bar", Effect: "NoSchedule"}},
			whitelist:  []api.Toleration{{Key: "foo", Operator: "Equal", Value: "bar", Effect: "NoExecute"}},
			testName:   "input does not exist in whitelist",
			testStatus: false,
		},
	}

	for _, c := range tests {
		status := VerifyAgainstWhitelist(c.input, c.whitelist)
		if status != c.testStatus {
			t.Errorf("Test: %s, expected %v", c.testName, status)
		}
	}

}

func TestIsConflict(t *testing.T) {
	tests := []struct {
		input1     []api.Toleration
		input2     []api.Toleration
		testName   string
		testStatus bool
	}{
		{
			input1:     []api.Toleration{{Key: "foo", Operator: "Equal", Value: "bar", Effect: "NoSchedule"}},
			input2:     []api.Toleration{{Key: "foo", Operator: "Equal", Value: "bar", Effect: "NoSchedule"}},
			testName:   "equal inputs",
			testStatus: true,
		},
		{
			input1:     []api.Toleration{{Key: "foo", Operator: "Equal", Value: "foo", Effect: "NoExecute"}},
			input2:     []api.Toleration{{Key: "foo", Operator: "Equal", Value: "bar", Effect: "NoExecute"}},
			testName:   "mismatch values in inputs",
			testStatus: false,
		},
	}

	for _, c := range tests {
		status := IsConflict(c.input1, c.input2)
		if status == c.testStatus {
			t.Errorf("Test: %s, expected %v", c.testName, status)
		}
	}
}
