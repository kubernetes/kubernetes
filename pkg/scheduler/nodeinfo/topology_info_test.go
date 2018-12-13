/*
Copyright 2018 The Kubernetes Authors.

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

package nodeinfo

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestUpdateNode(t *testing.T) {
	tests := []struct {
		name         string
		info         TopologyInfo
		oldNode      *v1.Node
		newNode      *v1.Node
		expectedInfo TopologyInfo
	}{
		{
			name: "no changes on labels",
			info: map[TopologyPair]sets.String{
				{"k1", "v1"}: sets.NewString("nodeA"),
				{"k2", "v2"}: sets.NewString("nodeA"),
			},
			oldNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "nodeA",
					Labels: map[string]string{"k1": "v1", "k2": "v2"},
				},
			},
			newNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "nodeA",
					Labels: map[string]string{"k1": "v1", "k2": "v2"},
				},
			},
			expectedInfo: map[TopologyPair]sets.String{
				{"k1", "v1"}: sets.NewString("nodeA"),
				{"k2", "v2"}: sets.NewString("nodeA"),
			},
		},
		{
			name: "labels update",
			info: map[TopologyPair]sets.String{
				{"k1", "v1"}: sets.NewString("nodeA"),
				{"k2", "v2"}: sets.NewString("nodeA"),
				{"k3", "v3"}: sets.NewString("nodeA"),
				{"k4", "v4"}: sets.NewString("nodeA"),
			},
			oldNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "nodeA",
					Labels: map[string]string{"k1": "v1", "k2": "v2", "k3": "v3", "k4": "v4"},
				},
			},
			newNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "nodeA",
					Labels: map[string]string{"k1": "v1", "k2": "v2a", "k3": "v3", "k5": "v5"},
				},
			},
			expectedInfo: map[TopologyPair]sets.String{
				{"k1", "v1"}:  sets.NewString("nodeA"),
				{"k2", "v2a"}: sets.NewString("nodeA"),
				{"k3", "v3"}:  sets.NewString("nodeA"),
				{"k5", "v5"}:  sets.NewString("nodeA"),
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			test.info.UpdateNode(test.oldNode, test.newNode)
			if !reflect.DeepEqual(test.info, test.expectedInfo) {
				t.Errorf("TestUpdateNode: Expected %v, Got: %v", test.expectedInfo, test.info)
			}
		})
	}
}
