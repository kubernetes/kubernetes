/*
Copyright 2022 The Kubernetes Authors.

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

package options

import (
	"fmt"
	"testing"

	nodeconfig "k8s.io/cloud-provider/controllers/node/config"
)

func errSliceEq(a []error, b []error) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if a[i].Error() != b[i].Error() {
			return false
		}
	}
	return true
}

func TestNodeControllerConcurrentNodeSyncsValidation(t *testing.T) {
	testCases := []struct {
		desc   string
		input  *NodeControllerOptions
		expect []error
	}{
		{
			desc: "empty options",
		},
		{
			desc:   "negative value",
			input:  &NodeControllerOptions{NodeControllerConfiguration: &nodeconfig.NodeControllerConfiguration{ConcurrentNodeSyncs: -5}},
			expect: []error{fmt.Errorf("concurrent-node-syncs must be a positive number")},
		},
		{
			desc:   "zero value",
			input:  &NodeControllerOptions{NodeControllerConfiguration: &nodeconfig.NodeControllerConfiguration{ConcurrentNodeSyncs: 0}},
			expect: []error{fmt.Errorf("concurrent-node-syncs must be a positive number")},
		},
		{
			desc:  "positive value",
			input: &NodeControllerOptions{NodeControllerConfiguration: &nodeconfig.NodeControllerConfiguration{ConcurrentNodeSyncs: 5}},
		},
	}
	for _, tc := range testCases {
		got := tc.input.Validate()
		if !errSliceEq(tc.expect, got) {
			t.Errorf("%v: expected: %v  got: %v", tc.desc, tc.expect, got)
		}
	}
}
