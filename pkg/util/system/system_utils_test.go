/*
Copyright 2016 The Kubernetes Authors.

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

package system

import (
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
)

func TestIsMasterNode(t *testing.T) {
	testCases := []struct {
		input  string
		result bool
	}{
		{"foo-master", true},
		{"foo-master-", false},
		{"foo-master-a", false},
		{"foo-master-ab", false},
		{"foo-master-abc", true},
		{"foo-master-abdc", false},
		{"foo-bar", false},
	}

	for _, tc := range testCases {
		node := v1.Node{ObjectMeta: v1.ObjectMeta{Name: tc.input}}
		res := IsMasterNode(node.Name)
		if res != tc.result {
			t.Errorf("case \"%s\": expected %t, got %t", tc.input, tc.result, res)
		}
	}
}
