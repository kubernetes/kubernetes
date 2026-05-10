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

package podgrouppodscount

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func init() {
	metrics.Register()
}

type mockProposedAssignment struct {
	nodeName string
	pod      *v1.Pod
}

var _ fwk.ProposedAssignment = &mockProposedAssignment{}

func (pa *mockProposedAssignment) GetNodeName() string {
	return pa.nodeName
}

func (pa *mockProposedAssignment) GetPod() *v1.Pod {
	return pa.pod
}

func TestScorePlacement(t *testing.T) {
	podGroupName := "pg1"

	createPod := func(podName, podGroupName, nodeName string) *v1.Pod {
		return st.MakePod().Name(podName).Namespace("default").UID(podName).PodGroupName(podGroupName).Node(nodeName).Obj()
	}

	createPodWithoutNode := func(podName, podGroupName string) *v1.Pod {
		return createPod(podName, podGroupName, "")
	}

	proposedAssignments := []fwk.ProposedAssignment{
		&mockProposedAssignment{
			nodeName: "node1",
			pod:      createPodWithoutNode("proposed-pod-1", podGroupName),
		},
		&mockProposedAssignment{
			nodeName: "node2",
			pod:      createPodWithoutNode("proposed-pod-2", podGroupName),
		},
	}

	tests := []struct {
		name          string
		pod           *v1.Pod
		assignedPods  []*v1.Pod // Pods to be added to the snapshot
		assumedPods   []*v1.Pod // Pods to be assumed in the snapshot
		placement     *fwk.PodGroupAssignments
		expectedScore int64
	}{
		{
			name: "existing assigned and assumed pods",
			pod:  createPodWithoutNode("p-new", podGroupName),
			assignedPods: []*v1.Pod{
				// Assigned pods
				createPod("p2", podGroupName, "node2"),
				createPod("p3", podGroupName, "node3"),
			},
			assumedPods: []*v1.Pod{
				// Assumed pod
				createPod("p1", podGroupName, "node1"),
			},
			placement: &fwk.PodGroupAssignments{
				ProposedAssignments: proposedAssignments,
			},
			expectedScore: 5, // 1 assumed + 2 assigned + 2 proposed = 5
		},
		{
			name: "no assumed pods",
			pod:  createPodWithoutNode("p-new", podGroupName),
			assignedPods: []*v1.Pod{
				createPod("p1", podGroupName, "node1"),
			},
			placement: &fwk.PodGroupAssignments{
				ProposedAssignments: proposedAssignments,
			},
			expectedScore: 3, // 1 assigned + 2 proposed = 3
		},
		{
			name: "no assigned pods",
			pod:  createPodWithoutNode("p-new", podGroupName),
			assumedPods: []*v1.Pod{
				// Assumed pod
				createPod("p1", podGroupName, "node1"),
			},
			placement: &fwk.PodGroupAssignments{
				ProposedAssignments: proposedAssignments,
			},
			expectedScore: 3, // 1 assumed + 2 proposed = 3
		},
		{
			name: "no assigned pods, no assumed pods",
			pod:  createPodWithoutNode("p-new", podGroupName),
			placement: &fwk.PodGroupAssignments{
				ProposedAssignments: proposedAssignments,
			},
			expectedScore: 2, // 2 proposed
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Enable GenericWorkload feature gate to populate PodGroupState in cache
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: true,
			})

			logger, ctx := ktesting.NewTestContext(t)

			// Setup cache, snapshot and framework
			snapshot := internalcache.NewEmptySnapshot()
			cache := internalcache.New(ctx, nil, true)
			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)

			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithSnapshotSharedLister(snapshot),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			// Add assigned pods to cache
			for _, p := range tt.assignedPods {
				if err := cache.AddPod(logger, p); err != nil {
					t.Fatalf("Failed to add pod %v: %v", p.Name, err)
				}
			}

			// Add assumed pods to cache
			for _, p := range tt.assumedPods {
				cache.AddPodGroupMember(p)
				if err := cache.AssumePod(logger, p); err != nil {
					t.Fatalf("Failed to assume pod %v: %v", p.Name, err)
				}
			}
			// Add proposed pods to cache
			for _, assignment := range tt.placement.ProposedAssignments {
				cache.AddPodGroupMember(assignment.GetPod())
			}

			// Update snapshot
			if err := cache.UpdateSnapshot(logger, snapshot); err != nil {
				t.Fatalf("Failed to update snapshot: %v", err)
			}

			// Create the plugin
			plugin, err := New(ctx, nil, fh, feature.Features{})
			if err != nil {
				t.Fatalf("Failed to create plugin: %v", err)
			}
			pl := plugin.(*PodGroupPodsCount)

			// Construct PodGroupInfo for the test pod
			pgInfo := &framework.PodGroupInfo{
				Namespace: tt.pod.Namespace,
				Name:      *tt.pod.Spec.SchedulingGroup.PodGroupName,
			}

			// Run ScorePlacement
			score, status := pl.ScorePlacement(ctx, nil, pgInfo, tt.placement)
			if !status.IsSuccess() {
				t.Errorf("ScorePlacement failed: %v", status.Message())
			}
			if score != tt.expectedScore {
				t.Errorf("Expected score %d, got %d", tt.expectedScore, score)
			}
		})
	}
}

func TestNormalizePlacementScore(t *testing.T) {
	tests := []struct {
		name                     string
		scores                   []fwk.PlacementScore
		expectedNormalizedScores []fwk.PlacementScore
		expectedError            string
	}{
		{
			name: "distinct scores",
			scores: []fwk.PlacementScore{
				{Score: 10},
				{Score: 50},
				{Score: 110},
			},
			// Normalized score is calculated as: score * (MaxScore - MinScore) / maxCount.
			// With MinScore=0, MaxScore=100, and maxCount=110 (using integer division):
			// 10 * 100 / 110 = 9
			// 50 * 100 / 110 = 45
			// 110 * 100 / 110 = 100
			expectedNormalizedScores: []fwk.PlacementScore{
				{Score: 9},
				{Score: 45},
				{Score: 100},
			},
		},
		{
			name: "equal scores",
			scores: []fwk.PlacementScore{
				{Score: 50},
				{Score: 50},
			},
			expectedNormalizedScores: []fwk.PlacementScore{
				{Score: 100},
				{Score: 100},
			},
		},
		{
			name: "single score",
			scores: []fwk.PlacementScore{
				{Score: 50},
			},
			expectedNormalizedScores: []fwk.PlacementScore{
				{Score: 100},
			},
		},
		{
			name: "some minimal score that is far from a group of scores located closely", // to test that normalization will not distribute it evenly
			scores: []fwk.PlacementScore{
				{Score: 11},
				{Score: 100},
				{Score: 101},
				{Score: 102},
			},
			expectedNormalizedScores: []fwk.PlacementScore{
				{Score: 10},
				{Score: 98},
				{Score: 99},
				{Score: 100},
			},
		},
		{
			name:          "zero scores",
			scores:        []fwk.PlacementScore{{Score: 0}, {Score: 0}},
			expectedError: `no pods from pod group are assigned to any of the candidate placements`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pl := &PodGroupPodsCount{}
			pgInfo := &framework.PodGroupInfo{Name: "pg1"}
			status := pl.NormalizePlacementScore(context.Background(), nil, pgInfo, tt.scores)
			if tt.expectedError != "" {
				if status.IsSuccess() {
					t.Fatal("Expected error, but got success")
				}
				if tt.expectedError != status.Message() {
					t.Errorf("Unexpected error message. Want %s\n, got %s", tt.expectedError, status.Message())
				}
				return
			}

			if !status.IsSuccess() {
				t.Errorf("NormalizePlacementScore failed: %v", status.Message())
			}

			if diff := cmp.Diff(tt.expectedNormalizedScores, tt.scores); diff != "" {
				t.Errorf("Unexpected scores (-want, +got):\n%s", diff)
			}
		})
	}
}
