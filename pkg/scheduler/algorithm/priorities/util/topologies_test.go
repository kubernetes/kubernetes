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

package util

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/sets"
)

func fakePod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "topologies_pod",
			Namespace: metav1.NamespaceDefault,
			UID:       "551f5a43-9f2f-11e7-a589-fa163e148d75",
		},
	}
}

func TestGetNamespacesFromPodAffinityTerm(t *testing.T) {
	tests := []struct {
		name            string
		podAffinityTerm *v1.PodAffinityTerm
		expectedValue   sets.String
	}{
		{
			"podAffinityTerm_namespace_empty",
			&v1.PodAffinityTerm{},
			sets.String{metav1.NamespaceDefault: sets.Empty{}},
		},
		{
			"podAffinityTerm_namespace_not_empty",
			&v1.PodAffinityTerm{
				Namespaces: []string{metav1.NamespacePublic, metav1.NamespaceSystem},
			},
			sets.String{metav1.NamespacePublic: sets.Empty{}, metav1.NamespaceSystem: sets.Empty{}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			realValue := GetNamespacesFromPodAffinityTerm(fakePod(), test.podAffinityTerm)
			assert.EqualValuesf(t, test.expectedValue, realValue, "Failed to test: %s", test.name)
		})
	}
}

func TestPodMatchesTermsNamespaceAndSelector(t *testing.T) {
	fakeNamespaces := sets.String{metav1.NamespacePublic: sets.Empty{}, metav1.NamespaceSystem: sets.Empty{}}
	fakeRequirement, _ := labels.NewRequirement("service", selection.In, []string{"topologies_service1", "topologies_service2"})
	fakeSelector := labels.NewSelector().Add(*fakeRequirement)

	tests := []struct {
		name           string
		podNamespaces  string
		podLabels      map[string]string
		expectedResult bool
	}{
		{
			"namespace_not_in",
			metav1.NamespaceDefault,
			map[string]string{"service": "topologies_service1"},
			false,
		},
		{
			"label_not_match",
			metav1.NamespacePublic,
			map[string]string{"service": "topologies_service3"},
			false,
		},
		{
			"normal_case",
			metav1.NamespacePublic,
			map[string]string{"service": "topologies_service1"},
			true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeTestPod := fakePod()
			fakeTestPod.Namespace = test.podNamespaces
			fakeTestPod.Labels = test.podLabels

			realValue := PodMatchesTermsNamespaceAndSelector(fakeTestPod, fakeNamespaces, fakeSelector)
			assert.EqualValuesf(t, test.expectedResult, realValue, "Faild to test: %s", test.name)
		})
	}

}

func TestNodesHaveSameTopologyKey(t *testing.T) {
	tests := []struct {
		name         string
		nodeA, nodeB *v1.Node
		topologyKey  string
		expected     bool
	}{
		{
			name: "nodeA{'a':'a'} vs. empty label in nodeB",
			nodeA: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "a",
					},
				},
			},
			nodeB:       &v1.Node{},
			expected:    false,
			topologyKey: "a",
		},
		{
			name: "nodeA{'a':'a'} vs. nodeB{'a':'a'}",
			nodeA: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "a",
					},
				},
			},
			nodeB: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "a",
					},
				},
			},
			expected:    true,
			topologyKey: "a",
		},
		{
			name: "nodeA{'a':''} vs. empty label in nodeB",
			nodeA: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "",
					},
				},
			},
			nodeB:       &v1.Node{},
			expected:    false,
			topologyKey: "a",
		},
		{
			name: "nodeA{'a':''} vs. nodeB{'a':''}",
			nodeA: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "",
					},
				},
			},
			nodeB: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "",
					},
				},
			},
			expected:    true,
			topologyKey: "a",
		},
		{
			name: "nodeA{'a':'a'} vs. nodeB{'a':'a'} by key{'b'}",
			nodeA: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "a",
					},
				},
			},
			nodeB: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "a",
					},
				},
			},
			expected:    false,
			topologyKey: "b",
		},
		{
			name: "topologyKey empty",
			nodeA: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "",
					},
				},
			},
			nodeB: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "",
					},
				},
			},
			expected:    false,
			topologyKey: "",
		},
		{
			name: "nodeA lable nil vs. nodeB{'a':''} by key('a')",
			nodeA: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{},
			},
			nodeB: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "",
					},
				},
			},
			expected:    false,
			topologyKey: "a",
		},
		{
			name: "nodeA{'a':''}  vs. nodeB label is nil by key('a')",
			nodeA: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "",
					},
				},
			},
			nodeB: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{},
			},
			expected:    false,
			topologyKey: "a",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := NodesHaveSameTopologyKey(test.nodeA, test.nodeB, test.topologyKey)
			assert.Equalf(t, test.expected, got, "Failed to test: %s", test.name)
		})
	}
}
