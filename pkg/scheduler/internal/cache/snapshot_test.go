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

package cache

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

const mb int64 = 1024 * 1024

func TestGetNodeImageStates(t *testing.T) {
	tests := []struct {
		node              *v1.Node
		imageExistenceMap map[string]sets.Set[string]
		expected          map[string]*framework.ImageStateSummary
	}{
		{
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node-0"},
				Status: v1.NodeStatus{
					Images: []v1.ContainerImage{
						{
							Names: []string{
								"gcr.io/10:v1",
							},
							SizeBytes: int64(10 * mb),
						},
						{
							Names: []string{
								"gcr.io/200:v1",
							},
							SizeBytes: int64(200 * mb),
						},
					},
				},
			},
			imageExistenceMap: map[string]sets.Set[string]{
				"gcr.io/10:v1":  sets.New("node-0", "node-1"),
				"gcr.io/200:v1": sets.New("node-0"),
			},
			expected: map[string]*framework.ImageStateSummary{
				"gcr.io/10:v1": {
					Size:     int64(10 * mb),
					NumNodes: 2,
				},
				"gcr.io/200:v1": {
					Size:     int64(200 * mb),
					NumNodes: 1,
				},
			},
		},
		{
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node-0"},
				Status:     v1.NodeStatus{},
			},
			imageExistenceMap: map[string]sets.Set[string]{
				"gcr.io/10:v1":  sets.New("node-1"),
				"gcr.io/200:v1": sets.New[string](),
			},
			expected: map[string]*framework.ImageStateSummary{},
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			imageStates := getNodeImageStates(test.node, test.imageExistenceMap)
			if diff := cmp.Diff(test.expected, imageStates); diff != "" {
				t.Errorf("Unexpected imageStates (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestCreateImageExistenceMap(t *testing.T) {
	tests := []struct {
		nodes    []*v1.Node
		expected map[string]sets.Set[string]
	}{
		{
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "node-0"},
					Status: v1.NodeStatus{
						Images: []v1.ContainerImage{
							{
								Names: []string{
									"gcr.io/10:v1",
								},
								SizeBytes: int64(10 * mb),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
					Status: v1.NodeStatus{
						Images: []v1.ContainerImage{
							{
								Names: []string{
									"gcr.io/10:v1",
								},
								SizeBytes: int64(10 * mb),
							},
							{
								Names: []string{
									"gcr.io/200:v1",
								},
								SizeBytes: int64(200 * mb),
							},
						},
					},
				},
			},
			expected: map[string]sets.Set[string]{
				"gcr.io/10:v1":  sets.New("node-0", "node-1"),
				"gcr.io/200:v1": sets.New("node-1"),
			},
		},
		{
			nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "node-0"},
					Status:     v1.NodeStatus{},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
					Status: v1.NodeStatus{
						Images: []v1.ContainerImage{
							{
								Names: []string{
									"gcr.io/10:v1",
								},
								SizeBytes: int64(10 * mb),
							},
							{
								Names: []string{
									"gcr.io/200:v1",
								},
								SizeBytes: int64(200 * mb),
							},
						},
					},
				},
			},
			expected: map[string]sets.Set[string]{
				"gcr.io/10:v1":  sets.New("node-1"),
				"gcr.io/200:v1": sets.New("node-1"),
			},
		},
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			imageMap := createImageExistenceMap(test.nodes)
			if diff := cmp.Diff(test.expected, imageMap); diff != "" {
				t.Errorf("Unexpected imageMap (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestCreateUsedPVCSet(t *testing.T) {
	tests := []struct {
		name     string
		pods     []*v1.Pod
		expected sets.Set[string]
	}{
		{
			name:     "empty pods list",
			pods:     []*v1.Pod{},
			expected: sets.New[string](),
		},
		{
			name: "pods not scheduled",
			pods: []*v1.Pod{
				st.MakePod().Name("foo").Namespace("foo").Obj(),
				st.MakePod().Name("bar").Namespace("bar").Obj(),
			},
			expected: sets.New[string](),
		},
		{
			name: "scheduled pods that do not use any PVC",
			pods: []*v1.Pod{
				st.MakePod().Name("foo").Namespace("foo").Node("node-1").Obj(),
				st.MakePod().Name("bar").Namespace("bar").Node("node-2").Obj(),
			},
			expected: sets.New[string](),
		},
		{
			name: "scheduled pods that use PVC",
			pods: []*v1.Pod{
				st.MakePod().Name("foo").Namespace("foo").Node("node-1").PVC("pvc1").Obj(),
				st.MakePod().Name("bar").Namespace("bar").Node("node-2").PVC("pvc2").Obj(),
			},
			expected: sets.New("foo/pvc1", "bar/pvc2"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			usedPVCs := createUsedPVCSet(test.pods)
			if diff := cmp.Diff(test.expected, usedPVCs); diff != "" {
				t.Errorf("Unexpected usedPVCs (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNewSnapshot(t *testing.T) {
	podWithAnnotations := st.MakePod().Name("foo").Namespace("ns").Node("node-1").Annotations(map[string]string{"custom": "annotation"}).Obj()
	podWithPort := st.MakePod().Name("foo").Namespace("foo").Node("node-0").ContainerPort([]v1.ContainerPort{{HostPort: 8080}}).Obj()
	podWithAntiAffitiny := st.MakePod().Name("baz").Namespace("ns").PodAntiAffinity("another", &metav1.LabelSelector{MatchLabels: map[string]string{"another": "label"}}, st.PodAntiAffinityWithRequiredReq).Node("node-0").Obj()
	podsWithAffitiny := []*v1.Pod{
		st.MakePod().Name("bar").Namespace("ns").PodAffinity("baz", &metav1.LabelSelector{MatchLabels: map[string]string{"baz": "qux"}}, st.PodAffinityWithRequiredReq).Node("node-2").Obj(),
		st.MakePod().Name("bar").Namespace("ns").PodAffinity("key", &metav1.LabelSelector{MatchLabels: map[string]string{"key": "value"}}, st.PodAffinityWithRequiredReq).Node("node-0").Obj(),
	}
	podsWithPVCs := []*v1.Pod{
		st.MakePod().Name("foo").Namespace("foo").Node("node-0").PVC("pvc0").Obj(),
		st.MakePod().Name("bar").Namespace("bar").Node("node-1").PVC("pvc1").Obj(),
		st.MakePod().Name("baz").Namespace("baz").Node("node-2").PVC("pvc2").Obj(),
	}
	testCases := []struct {
		name                         string
		pods                         []*v1.Pod
		nodes                        []*v1.Node
		expectedNodesInfos           []*framework.NodeInfo
		expectedNumNodes             int
		expectedPodsWithAffinity     int
		expectedPodsWithAntiAffinity int
		expectedUsedPVCSet           sets.Set[string]
	}{
		{
			name:  "no pods no nodes",
			pods:  nil,
			nodes: nil,
		},
		{
			name: "single pod single node",
			pods: []*v1.Pod{
				podWithPort,
			},
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "node-0"}},
			},
			expectedNodesInfos: []*framework.NodeInfo{
				{
					Pods: []*framework.PodInfo{
						{Pod: podWithPort},
					},
				},
			},
			expectedNumNodes: 1,
		},
		{
			name: "multiple nodes, pods with PVCs",
			pods: podsWithPVCs,
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "node-0"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "node-2"}},
			},
			expectedNodesInfos: []*framework.NodeInfo{
				{
					Pods: []*framework.PodInfo{
						{Pod: podsWithPVCs[0]},
					},
				},
				{
					Pods: []*framework.PodInfo{
						{Pod: podsWithPVCs[1]},
					},
				},
				{
					Pods: []*framework.PodInfo{
						{Pod: podsWithPVCs[2]},
					},
				},
			},
			expectedNumNodes:   3,
			expectedUsedPVCSet: sets.New("foo/pvc0", "bar/pvc1", "baz/pvc2"),
		},
		{
			name: "multiple nodes, pod with affinity",
			pods: []*v1.Pod{
				podWithAnnotations,
				podsWithAffitiny[0],
			},
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "node-0"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "node-2", Labels: map[string]string{"baz": "qux"}}},
			},
			expectedNodesInfos: []*framework.NodeInfo{
				{
					Pods: []*framework.PodInfo{},
				},
				{
					Pods: []*framework.PodInfo{
						{Pod: podWithAnnotations},
					},
				},
				{
					Pods: []*framework.PodInfo{
						{
							Pod: podsWithAffitiny[0],
							RequiredAffinityTerms: []framework.AffinityTerm{
								{
									Namespaces:        sets.New("ns"),
									Selector:          labels.SelectorFromSet(map[string]string{"baz": "qux"}),
									TopologyKey:       "baz",
									NamespaceSelector: labels.Nothing(),
								},
							},
						},
					},
				},
			},
			expectedNumNodes:         3,
			expectedPodsWithAffinity: 1,
		},
		{
			name: "multiple nodes, pod with affinity, pod with anti-affinity",
			pods: []*v1.Pod{
				podsWithAffitiny[1],
				podWithAntiAffitiny,
			},
			nodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "node-0", Labels: map[string]string{"key": "value"}}},
				{ObjectMeta: metav1.ObjectMeta{Name: "node-1", Labels: map[string]string{"another": "label"}}},
			},
			expectedNodesInfos: []*framework.NodeInfo{
				{
					Pods: []*framework.PodInfo{
						{
							Pod: podsWithAffitiny[1],
							RequiredAffinityTerms: []framework.AffinityTerm{
								{
									Namespaces:        sets.New("ns"),
									Selector:          labels.SelectorFromSet(map[string]string{"key": "value"}),
									TopologyKey:       "key",
									NamespaceSelector: labels.Nothing(),
								},
							},
						},
						{
							Pod: podWithAntiAffitiny,
							RequiredAntiAffinityTerms: []framework.AffinityTerm{
								{
									Namespaces:        sets.New("ns"),
									Selector:          labels.SelectorFromSet(map[string]string{"another": "label"}),
									TopologyKey:       "another",
									NamespaceSelector: labels.Nothing(),
								},
							},
						},
					},
				},
				{
					Pods: []*framework.PodInfo{},
				},
			},
			expectedNumNodes:             2,
			expectedPodsWithAffinity:     1,
			expectedPodsWithAntiAffinity: 1,
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			snapshot := NewSnapshot(test.pods, test.nodes)

			if test.expectedNumNodes != snapshot.NumNodes() {
				t.Errorf("unexpected number of nodes, want: %v, got: %v", test.expectedNumNodes, snapshot.NumNodes())
			}

			for i, node := range test.nodes {
				info, err := snapshot.Get(node.Name)
				if err != nil {
					t.Errorf("unexpected error but got %s", err)
				}
				if info == nil {
					t.Error("node infos should not be nil")
				}
				for j := range test.expectedNodesInfos[i].Pods {
					if diff := cmp.Diff(test.expectedNodesInfos[i].Pods[j], info.Pods[j]); diff != "" {
						t.Errorf("Unexpected PodInfo (-want +got):\n%s", diff)
					}
				}
			}

			affinityList, err := snapshot.HavePodsWithAffinityList()
			if err != nil {
				t.Errorf("unexpected error but got %s", err)
			}
			if test.expectedPodsWithAffinity != len(affinityList) {
				t.Errorf("unexpected affinityList number, want: %v, got: %v", test.expectedPodsWithAffinity, len(affinityList))
			}

			antiAffinityList, err := snapshot.HavePodsWithRequiredAntiAffinityList()
			if err != nil {
				t.Errorf("unexpected error but got %s", err)
			}
			if test.expectedPodsWithAntiAffinity != len(antiAffinityList) {
				t.Errorf("unexpected antiAffinityList number, want: %v, got: %v", test.expectedPodsWithAntiAffinity, len(antiAffinityList))
			}

			for key := range test.expectedUsedPVCSet {
				if !snapshot.IsPVCUsedByPods(key) {
					t.Errorf("unexpected IsPVCUsedByPods for %s, want: true, got: false", key)
				}
			}

			if diff := cmp.Diff(test.expectedUsedPVCSet, snapshot.usedPVCSet); diff != "" {
				t.Errorf("Unexpected usedPVCSet (-want +got):\n%s", diff)
			}
		})
	}
}
