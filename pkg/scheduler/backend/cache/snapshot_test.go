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
	"github.com/google/go-cmp/cmp/cmpopts"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

const mb int64 = 1024 * 1024

func TestGetNodeImageStates(t *testing.T) {
	tests := []struct {
		node              *v1.Node
		imageExistenceMap map[string]sets.Set[string]
		expected          map[string]*fwk.ImageStateSummary
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
			expected: map[string]*fwk.ImageStateSummary{
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
			expected: map[string]*fwk.ImageStateSummary{},
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
					Pods: []fwk.PodInfo{
						&framework.PodInfo{Pod: podWithPort},
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
					Pods: []fwk.PodInfo{
						&framework.PodInfo{Pod: podsWithPVCs[0]},
					},
				},
				{
					Pods: []fwk.PodInfo{
						&framework.PodInfo{Pod: podsWithPVCs[1]},
					},
				},
				{
					Pods: []fwk.PodInfo{
						&framework.PodInfo{Pod: podsWithPVCs[2]},
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
					Pods: []fwk.PodInfo{},
				},
				{
					Pods: []fwk.PodInfo{
						&framework.PodInfo{Pod: podWithAnnotations},
					},
				},
				{
					Pods: []fwk.PodInfo{
						&framework.PodInfo{
							Pod: podsWithAffitiny[0],
							RequiredAffinityTerms: []fwk.AffinityTerm{
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
					Pods: []fwk.PodInfo{
						&framework.PodInfo{
							Pod: podsWithAffitiny[1],
							RequiredAffinityTerms: []fwk.AffinityTerm{
								{
									Namespaces:        sets.New("ns"),
									Selector:          labels.SelectorFromSet(map[string]string{"key": "value"}),
									TopologyKey:       "key",
									NamespaceSelector: labels.Nothing(),
								},
							},
						},
						&framework.PodInfo{
							Pod: podWithAntiAffitiny,
							RequiredAntiAffinityTerms: []fwk.AffinityTerm{
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
					Pods: []fwk.PodInfo{},
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

			if test.expectedNumNodes != snapshot.NumNodesInPlacement() {
				t.Errorf("unexpected number of nodes, want: %v, got: %v", test.expectedNumNodes, snapshot.NumNodesInPlacement())
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
					if diff := cmp.Diff(test.expectedNodesInfos[i].Pods[j], info.GetPods()[j], cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
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

func TestSnapshot_AssumeForget(t *testing.T) {
	node1 := st.MakeNode().Name("node-1").Obj()
	node2 := st.MakeNode().Name("node-2").Obj()

	pod1 := st.MakePod().Name("pod-1").UID("pod-1").Node("node-1").Obj()
	pod2 := st.MakePod().Name("pod-2").UID("pod-2").Node("node-1").Obj()
	pod3 := st.MakePod().Name("pod-3").UID("pod-3").Node("node-2").Obj()
	podOnWrongNode := st.MakePod().Name("pod-x").UID("pod-x").Node("node-x").Obj()

	pod2Info, _ := framework.NewPodInfo(pod2)
	pod3Info, _ := framework.NewPodInfo(pod3)
	podOnWrongNodeInfo, _ := framework.NewPodInfo(podOnWrongNode)

	tests := []struct {
		name                string
		initialPods         []*v1.Pod
		initialNodes        []*v1.Node
		podsToAssume        []*framework.PodInfo
		podsToForget        []*v1.Pod
		forgetAll           bool
		expectAssumeErr     bool
		expectForgetErr     bool
		expectedPodsOnNodes map[string]sets.Set[string]
	}{
		{
			name:         "assume a pod successfully",
			initialPods:  []*v1.Pod{pod1},
			initialNodes: []*v1.Node{node1, node2},
			podsToAssume: []*framework.PodInfo{pod2Info, pod3Info},
			expectedPodsOnNodes: map[string]sets.Set[string]{
				"node-1": sets.New("pod-1", "pod-2"),
				"node-2": sets.New("pod-3"),
			},
		},
		{
			name:         "assume a pod on a non-existing node",
			initialPods:  []*v1.Pod{pod1},
			initialNodes: []*v1.Node{node1},
			podsToAssume: []*framework.PodInfo{podOnWrongNodeInfo},
		},
		{
			name:         "forget a pod successfully",
			initialPods:  []*v1.Pod{pod1},
			initialNodes: []*v1.Node{node1, node2},
			podsToAssume: []*framework.PodInfo{pod2Info, pod3Info},
			podsToForget: []*v1.Pod{pod2},
			expectedPodsOnNodes: map[string]sets.Set[string]{
				"node-1": sets.New("pod-1"),
				"node-2": sets.New("pod-3"),
			},
		},
		{
			name:            "forget a pod that was not assumed",
			initialPods:     []*v1.Pod{pod1},
			initialNodes:    []*v1.Node{node1, node2},
			podsToForget:    []*v1.Pod{pod2},
			expectForgetErr: true,
		},
		{
			name:         "forget all assumed pods",
			initialPods:  []*v1.Pod{pod1},
			initialNodes: []*v1.Node{node1, node2},
			podsToAssume: []*framework.PodInfo{pod2Info, pod3Info},
			forgetAll:    true,
			expectedPodsOnNodes: map[string]sets.Set[string]{
				"node-1": sets.New("pod-1"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)

			snapshot := NewSnapshot(tt.initialPods, tt.initialNodes)

			for _, p := range tt.podsToAssume {
				err := snapshot.AssumePod(p)
				if tt.expectAssumeErr {
					if err == nil {
						t.Fatalf("Expected AssumePod to fail but is hasn't")
					}
					return
				}
				if err != nil {
					t.Fatalf("Failed to assume pod %q: %v", p.Pod.Name, err)
				}
			}

			if tt.forgetAll {
				snapshot.forgetAllAssumedPods(logger)
				if len(snapshot.assumedPods) != 0 {
					t.Errorf("Expected assumedPods to be empty, but has %d pods", len(snapshot.assumedPods))
				}
			} else {
				for _, p := range tt.podsToForget {
					err := snapshot.ForgetPod(logger, p)
					if tt.expectForgetErr {
						if err == nil {
							t.Fatalf("Expected ForgetPod to fail but is hasn't")
						}
						return
					}
					if err != nil {
						t.Fatalf("Failed to forget pod %q: %v", p.Name, err)
					}
				}
			}

			nodeInfos, err := snapshot.List()
			if err != nil {
				t.Fatalf("Failed to list snapshotted nodes: %v", err)
			}
			for nodeName, expectedPods := range tt.expectedPodsOnNodes {
				nodeInfo, err := snapshot.Get(nodeName)
				if err != nil {
					t.Fatalf("Failed to get node %q from snapshot: %v", nodeName, err)
				}
				gotPods := nodeInfo.GetPods()
				if len(expectedPods) != len(gotPods) {
					t.Errorf("Unexpected number of pods on node %q: want %d, got %d", nodeName, len(expectedPods), len(gotPods))
				}
				for _, p := range nodeInfo.GetPods() {
					podName := p.GetPod().Name
					if !expectedPods.Has(podName) {
						t.Errorf("Unexpected pod %q on node %q", podName, nodeName)
					}
				}
				// Safety check that nodeInfoList's pods were also updated.
				for _, nInfo := range nodeInfos {
					if nInfo.Node().Name == nodeInfo.Node().Name {
						if diff := cmp.Diff(nodeInfo.GetPods(), nInfo.GetPods(), cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
							t.Errorf("Unexpected nodeInfo state in nodeInfoList (-want +got):\n%s", diff)
						}
					}
				}
			}
		})
	}
}

func TestSnapshot_Placement(t *testing.T) {
	tt := []struct {
		name           string
		initialNodes   []string
		placementNodes []string
		testFn         func(t *testing.T, snapshot *Snapshot, placement *fwk.Placement)
	}{
		{
			name:           "When placement is not set, nodes in placement are same as nodes in snapshot",
			initialNodes:   []string{"n1", "n2", "n3"},
			placementNodes: []string{}, // unused
			testFn: func(t *testing.T, snapshot *Snapshot, placement *fwk.Placement) {
				// We intentionally don't use snapshot.AssumePlacement here
				numNodes := snapshot.NumNodesInPlacement()
				if numNodes != 3 {
					t.Errorf("unexpected number of nodes from NumNodesInPlacement, want: %v, got: %v", 3, numNodes)
				}
				nodes, err := snapshot.ListNodesInPlacement()
				if err != nil {
					t.Fatalf("unexpected error from ListNodesInPlacement %v", err)
				}
				if len(nodes) != 3 {
					t.Errorf("unexpected number of nodes from ListNodesInPlacement, want: %v, got: %v", 3, len(nodes))
				}
				_, err = snapshot.GetNodeInPlacement("n1")
				if err != nil {
					t.Errorf("expected GetNodeInPlacement to find node but instead got unexpected error %v", err)
				}
			},
		},
		{
			name:           "When placement is set, nodes in placement are the nodes from the provided placement",
			initialNodes:   []string{"n1", "n2", "n3"},
			placementNodes: []string{"n2", "n3"},
			testFn: func(t *testing.T, snapshot *Snapshot, placement *fwk.Placement) {
				err := snapshot.AssumePlacement(placement)
				if err != nil {
					t.Fatalf("got unexpected error from AssumePlacement %v", err)
				}
				numNodes := snapshot.NumNodesInPlacement()
				if numNodes != 2 {
					t.Errorf("unexpected number of nodes from NumNodesInPlacement, want: %v, got: %v", 2, numNodes)
				}
				nodes, err := snapshot.ListNodesInPlacement()
				if err != nil {
					t.Fatalf("unexpected error from ListNodesInPlacement %v", err)
				}
				if len(nodes) != 2 {
					t.Errorf("unexpected number of nodes from ListNodesInPlacement, want: %v, got: %v", 2, len(nodes))
				}
				_, err = snapshot.GetNodeInPlacement("n1")
				if err == nil {
					t.Errorf("expected GetNodeInPlacement not to find node but instead got nil error")
				}
				_, err = snapshot.GetNodeInPlacement("n3")
				if err != nil {
					t.Errorf("expected GetNodeInPlacement to find node but instead got unexpected error %v", err)
				}
			},
		},
		{
			name:           "When placement is set, List and Get are not affected and still use snapshot nodes",
			initialNodes:   []string{"n1", "n2", "n3"},
			placementNodes: []string{"n2", "n3"},
			testFn: func(t *testing.T, snapshot *Snapshot, placement *fwk.Placement) {
				err := snapshot.AssumePlacement(placement)
				if err != nil {
					t.Fatalf("got unexpected error from AssumePlacement %v", err)
				}
				nodes, err := snapshot.List()
				if err != nil {
					t.Fatalf("unexpected error from List %v", err)
				}
				if len(nodes) != 3 {
					t.Errorf("unexpected number of nodes from ListNodesInPlacement, want: %v, got: %v", 3, len(nodes))
				}
				_, err = snapshot.Get("n1")
				if err != nil {
					t.Errorf("expected Get to find node but instead got unexpected error %v", err)
				}
			},
		},
		{
			name:           "When placement is cleared through ForgetPlacement, nodes in placement are same as nodes in snapshot",
			initialNodes:   []string{"n1", "n2", "n3"},
			placementNodes: []string{"n2", "n3"},
			testFn: func(t *testing.T, snapshot *Snapshot, placement *fwk.Placement) {
				err := snapshot.AssumePlacement(placement)
				if err != nil {
					t.Fatalf("got unexpected error from AssumePlacement %v", err)
				}
				snapshot.ForgetPlacement()
				numNodes := snapshot.NumNodesInPlacement()
				if numNodes != 3 {
					t.Errorf("unexpected number of nodes from NumNodesInPlacement, want: %v, got: %v", 3, numNodes)
				}
				nodes, err := snapshot.ListNodesInPlacement()
				if err != nil {
					t.Fatalf("unexpected error from ListNodesInPlacement %v", err)
				}
				if len(nodes) != 3 {
					t.Errorf("unexpected number of nodes from ListNodesInPlacement, want: %v, got: %v", 3, len(nodes))
				}
				_, err = snapshot.GetNodeInPlacement("n1")
				if err != nil {
					t.Errorf("expected GetNodeInPlacement to find node but instead got unexpected error %v", err)
				}
			},
		},
		{
			name:           "When placement uses nodes that point to a different instance than snapshot, AssumePlacement returns error and placement is not set",
			initialNodes:   []string{"n1", "n2", "n3"},
			placementNodes: []string{"n2"},
			testFn: func(t *testing.T, snapshot *Snapshot, placement *fwk.Placement) {
				placement.Nodes[0] = placement.Nodes[0].Snapshot()
				err := snapshot.AssumePlacement(placement)
				if err == nil {
					t.Fatalf("expected AssumePlacement to return error due to no match between placement and snapshot node instance but got nil")
				}
				// ensure the placement is cleared
				numNodes := snapshot.NumNodesInPlacement()
				if numNodes != 3 {
					t.Errorf("unexpected number of nodes from NumNodesInPlacement, want: %v, got: %v", 3, numNodes)
				}
				nodes, err := snapshot.ListNodesInPlacement()
				if err != nil {
					t.Fatalf("unexpected error from ListNodesInPlacement %v", err)
				}
				if len(nodes) != 3 {
					t.Errorf("unexpected number of nodes from ListNodesInPlacement, want: %v, got: %v", 3, len(nodes))
				}
				_, err = snapshot.GetNodeInPlacement("n1")
				if err != nil {
					t.Errorf("expected GetNodeInPlacement to find node but instead got unexpected error %v", err)
				}
			},
		},
		{
			name:           "When placement is set, the effect of AssumePod is visible in nodes in placement",
			initialNodes:   []string{"n1", "n2", "n3"},
			placementNodes: []string{"n2"},
			testFn: func(t *testing.T, snapshot *Snapshot, placement *fwk.Placement) {
				err := snapshot.AssumePlacement(placement)
				if err != nil {
					t.Fatalf("unexpected error from AssumePlacement %v", err)
				}
				pod := st.MakePod().Name("pod-1").UID("pod-1").Node("n2").Obj()
				podInfo, err := framework.NewPodInfo(pod)
				if err != nil {
					t.Fatalf("unexpected error from NewPodInfo %v", err)
				}
				err = snapshot.AssumePod(podInfo)
				if err != nil {
					t.Fatalf("unexpected error from AssumePod: %v", err)
				}
				nodes, err := snapshot.ListNodesInPlacement()
				if err != nil {
					t.Fatalf("unexpected error from ListNodesInPlacement %v", err)
				}
				if len(nodes) != 1 {
					t.Fatalf("unexpected number of nodes from ListNodesInPlacement, want: %v, got: %v", 1, len(nodes))
				}
				node, err := snapshot.GetNodeInPlacement("n2")
				if err != nil {
					t.Fatalf("expected GetNodeInPlacement to find node but instead got unexpected error %v", err)
				}
				if nodes[0] != node {
					t.Errorf("node from ListNodesInPlacement is not the same instance as from GetNodeInPlacement")
				}
				snapshotNode, err := snapshot.Get("n2")
				if err != nil {
					t.Fatalf("expected Get to find node but instead of unexpected error: %v", err)
				}
				if snapshotNode != node {
					t.Fatalf("node from Get is not the same instance as from GetNodeInPlacement")
				}
				pods := node.GetPods()
				if len(pods) != 1 {
					t.Fatalf("unexpected number of pods on node, want: %v, got: %v", 1, len(pods))
				}
				if diff := cmp.Diff(podInfo, pods[0], cmpopts.IgnoreUnexported(framework.PodInfo{})); diff != "" {
					t.Errorf("pod from placement is not equal to pod from snapshot (-want +got):\n%s", diff)
				}
			},
		},
	}
	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			initialNodes := []*v1.Node{}
			for _, n := range tc.initialNodes {
				initialNodes = append(initialNodes, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: n}})
			}

			snapshot := NewSnapshot([]*v1.Pod{}, initialNodes)
			placementNodes := []fwk.NodeInfo{}
			for _, n := range tc.placementNodes {
				node, ok := snapshot.nodeInfoMap[n]
				if !ok {
					node = framework.NewNodeInfo()
					node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: n}})
				}
				placementNodes = append(placementNodes, node)
			}

			tc.testFn(t, snapshot, &fwk.Placement{
				Nodes: placementNodes,
			})
		})
	}
}

func TestSnapshot_BackupRestore(t *testing.T) {
	podWithAffinity := st.MakePod().Name("p-aff").Namespace("ns").UID("p-aff").PodAffinity("key", &metav1.LabelSelector{MatchLabels: map[string]string{"key": "value"}}, st.PodAffinityWithRequiredReq).Node("node-1").Obj()
	podWithAntiAffinity := st.MakePod().Name("p-anti").Namespace("ns").UID("p-anti").PodAntiAffinity("key", &metav1.LabelSelector{MatchLabels: map[string]string{"key": "value"}}, st.PodAntiAffinityWithRequiredReq).Node("node-1").Obj()

	tests := []struct {
		name           string
		initialPods    []*v1.Pod
		initialNodes   []*v1.Node
		modifySnapshot func(klog.Logger, *Snapshot)
	}{
		{
			name: "Modify NodeInfo (Add Pod)",
			initialNodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}},
			},
			modifySnapshot: func(_ klog.Logger, s *Snapshot) {
				node := s.nodeInfoMap["node-1"]
				pod := st.MakePod().Name("p1").Node("node-1").Obj()
				node.AddPod(pod)
			},
		},
		{
			name: "Modify havePodsWithAffinityNodeInfoList (Add)",
			initialNodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "node-1", Labels: map[string]string{"key": "value"}}},
			},
			modifySnapshot: func(_ klog.Logger, s *Snapshot) {
				node := s.nodeInfoMap["node-1"]
				node.AddPod(podWithAffinity)
				s.havePodsWithAffinityNodeInfoList = append(s.havePodsWithAffinityNodeInfoList, node)
			},
		},
		{
			name: "Modify havePodsWithRequiredAntiAffinityNodeInfoList (Remove)",
			initialPods: []*v1.Pod{
				podWithAntiAffinity,
			},
			initialNodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "node-1", Labels: map[string]string{"key": "value"}}},
			},
			modifySnapshot: func(logger klog.Logger, s *Snapshot) {
				node := s.nodeInfoMap["node-1"]
				if err := node.RemovePod(logger, podWithAntiAffinity); err != nil {
					t.Fatalf("Failed to remove pod: %v", err)
				}
				s.havePodsWithRequiredAntiAffinityNodeInfoList = []fwk.NodeInfo{}
			},
		},
		{
			name: "Modify nodeInfoList directly",
			initialNodes: []*v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "node-2"}},
			},
			modifySnapshot: func(_ klog.Logger, s *Snapshot) {
				// Reverse the list
				s.nodeInfoList[0], s.nodeInfoList[1] = s.nodeInfoList[1], s.nodeInfoList[0]
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			s := NewSnapshot(tt.initialPods, tt.initialNodes)

			// Store original state for deep verification
			origNodeInfoMap, origNodeInfoList, origAffinityList, origAntiAffinityList := simplifySnapshot(s)

			restore, err := s.BackupSnapshot()
			if err != nil {
				t.Fatalf("failed to prepare a backup")
			}
			tt.modifySnapshot(logger, s)
			restore()

			// Get state after for verification
			postRestoreNodeInfoMap, postRestoreNodeInfoList, postRestoreAffinityList, postRestoreAntiAffinityList := simplifySnapshot(s)

			if cmp.Diff(origNodeInfoMap, postRestoreNodeInfoMap) != "" {
				t.Errorf("nodeInfoMap mismatch: want %v, got %v", origNodeInfoMap, postRestoreNodeInfoMap)
			}
			if cmp.Diff(origNodeInfoList, postRestoreNodeInfoList) != "" {
				t.Errorf("nodeInfoList mismatch: want %v, got %v", origNodeInfoList, postRestoreNodeInfoList)
			}
			if cmp.Diff(origAffinityList, postRestoreAffinityList) != "" {
				t.Errorf("havePodsWithAffinityNodeInfoList mismatch: want %v, got %v", origAffinityList, postRestoreAffinityList)
			}
			if cmp.Diff(origAntiAffinityList, postRestoreAntiAffinityList) != "" {
				t.Errorf("havePodsWithRequiredAntiAffinityNodeInfoList mismatch: want %v, got %v", origAntiAffinityList, postRestoreAntiAffinityList)
			}
		})
	}
}

// simplifySnapshot for comparison in unit tests
func simplifySnapshot(s *Snapshot) (map[string][]string, []string, []string, []string) {
	nodeInfoMap := make(map[string][]string)
	var nodeInfoList []string
	var affinityList []string
	var antiAffinityList []string
	for _, nodeInfo := range s.nodeInfoMap {
		for _, p := range nodeInfo.GetPods() {
			nodeInfoMap[nodeInfo.Node().Name] = append(nodeInfoMap[nodeInfo.Node().Name], p.GetPod().Name)
		}
	}
	for _, nodeInfo := range s.nodeInfoList {
		nodeInfoList = append(nodeInfoList, nodeInfo.Node().Name)
	}
	for _, nodeInfo := range s.havePodsWithAffinityNodeInfoList {
		affinityList = append(affinityList, nodeInfo.Node().Name)
	}
	for _, nodeInfo := range s.havePodsWithRequiredAntiAffinityNodeInfoList {
		antiAffinityList = append(antiAffinityList, nodeInfo.Node().Name)
	}
	return nodeInfoMap, nodeInfoList, affinityList, antiAffinityList
}

func TestSnapshot_MultipleBackups(t *testing.T) {
	s := NewSnapshot(nil, nil)

	restore, err := s.BackupSnapshot()
	if err != nil {
		t.Fatalf("failed to prepare a backup: %v", err)
	}

	_, err = s.BackupSnapshot()
	if err == nil {
		t.Fatalf("expected error when stacking backups, got nil")
	}

	expectedErr := "cannot stack backups"
	if err.Error() != expectedErr {
		t.Errorf("expected error %q, got %q", expectedErr, err.Error())
	}

	// Restore the previous backup, and now it should work again
	restore()

	_, err = s.BackupSnapshot()
	if err != nil {
		t.Fatalf("failed to prepare a backup after restoring: %v", err)
	}
}
