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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestIsMasterNode(t *testing.T) {
	label1 := map[string]string{"foo": "bar"}
	label2 := map[string]string{"foo": "bar", LabelNodeRoleMaster: ""}
	label3 := map[string]string{"foo": "bar", "node-role.kubernetes.io/node": ""}
	testCases := []struct {
		node   *v1.Node
		result bool
	}{
		{&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node1", Labels: label1}}, false},
		{&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node2", Labels: label2}}, true},
		{&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node3", Labels: label3}}, false},
	}

	for _, tc := range testCases {
		res := IsMasterNode(tc.node)
		if res != tc.result {
			t.Errorf("case \"%s\": expected %t, got %t", tc.node.Name, tc.result, res)
		}
	}
}
