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
	labelMater := map[string]string{
		LabelNodeRoleMaster: string(v1.TaintEffectNoSchedule),
	}
	testCases := []struct {
		node   v1.Node
		result bool
	}{
		{
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-master",
				},
			},
			result: true,
		},
		{
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-master-",
				},
			},
			result: false,
		},
		{
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-master-a",
				},
			},
			result: false,
		},
		{
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-master-ab",
				},
			},
			result: false,
		},
		{
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-master-abc",
				},
			},
			result: true,
		},
		{
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-master-abdc",
				},
			},
			result: false,
		},
		{
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-bar",
				},
			},
			result: false,
		},
		{
			node: v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "foo-bar",
					Labels: labelMater,
				},
			},
			result: true,
		},
	}

	for _, tc := range testCases {
		node := tc.node
		res := IsMasterNode(node)
		if res != tc.result {
			t.Errorf("case \"%v\": expected %t, got %t", tc.node, tc.result, res)
		}
	}
}
