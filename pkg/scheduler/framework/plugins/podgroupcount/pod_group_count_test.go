/*
Copyright 2025 The Kubernetes Authors.

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

package podgroupcount

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
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func init() {
	metrics.Register()
}

func TestScorePlacement(t *testing.T) {
	tests := []struct {
		name          string
		pod           *v1.Pod
		existingPods  []*v1.Pod // Pods to be added to the cache
		assumedPods   []*v1.Pod // Pods to be assumed in the cache
		placement     *fwk.PodGroupAssignments
		expectedScore int64
	}{
		{
			name: "existing assigned and assumed pods",
			pod: st.MakePod().Name("p-new").Namespace("default").
				WorkloadRef(&v1.WorkloadReference{Name: "pg1", PodGroup: "PodGroup"}).Obj(),
			existingPods: []*v1.Pod{
				// Assigned pods
				st.MakePod().Name("p2").Namespace("default").UID("p2").
					WorkloadRef(&v1.WorkloadReference{Name: "pg1", PodGroup: "PodGroup"}).Node("node2").Obj(),
				st.MakePod().Name("p3").Namespace("default").UID("p3").
					WorkloadRef(&v1.WorkloadReference{Name: "pg1", PodGroup: "PodGroup"}).Node("node3").Obj(),
			},
			assumedPods: []*v1.Pod{
				// Assumed pod
				st.MakePod().Name("p1").Namespace("default").UID("p1").
					WorkloadRef(&v1.WorkloadReference{Name: "pg1", PodGroup: "PodGroup"}).Node("node1").Obj(),
			},
			placement: &fwk.PodGroupAssignments{
				ProposedAssignments: make([]fwk.ProposedAssignment, 2), // 2 new assignments
			},
			expectedScore: 1 + 2 + 2, // 1 assumed + 2 assigned + 2 proposed = 5
		},
		{
			name: "no existing pods",
			pod: st.MakePod().Name("p-new").Namespace("default").
				WorkloadRef(&v1.WorkloadReference{Name: "pg2", PodGroup: "PodGroup"}).Obj(),
			existingPods: nil,
			placement: &fwk.PodGroupAssignments{
				ProposedAssignments: make([]fwk.ProposedAssignment, 3), // 3 new assignments
			},
			expectedScore: 3,
		},
		{
			name: "pods from different pod group ignored",
			pod: st.MakePod().Name("p-new").Namespace("default").
				WorkloadRef(&v1.WorkloadReference{Name: "pg3", PodGroup: "PodGroup"}).Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("p1").Namespace("default").UID("p1").
					WorkloadRef(&v1.WorkloadReference{Name: "other-pg", PodGroup: "PodGroup1"}).Node("node1").Obj(),
			},
			placement: &fwk.PodGroupAssignments{
				ProposedAssignments: make([]fwk.ProposedAssignment, 1),
			},
			expectedScore: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Enable GenericWorkload feature gate to populate PodGroupState in cache
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, true)

			logger, ctx := ktesting.NewTestContext(t)

			// Setup cache and framework
			cache := internalcache.New(ctx, nil)
			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)

			fh, err := frameworkruntime.NewFramework(ctx, nil, nil,
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithPodGroupManager(cache),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			// Add assigned pods to cache
			for _, p := range tt.existingPods {
				if err := cache.AddPod(logger, p); err != nil {
					t.Fatalf("Failed to add pod %v: %v", p.Name, err)
				}
			}

			// Add assumed pods to cache
			for _, p := range tt.assumedPods {
				if err := cache.AssumePod(logger, p); err != nil {
					t.Fatalf("Failed to assume pod %v: %v", p.Name, err)
				}
			}

			// Create the plugin
			plugin, err := New(ctx, nil, fh)
			if err != nil {
				t.Fatalf("Failed to create plugin: %v", err)
			}
			pl := plugin.(*PodGroupCount)

			// Construct PodGroupInfo for the test pod
			pgInfo := &testPodGroupInfo{
				namespace: tt.pod.Namespace,
				workload:  tt.pod.Spec.WorkloadRef,
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
		name     string
		scores   []fwk.PlacementScore
		expected []fwk.PlacementScore
	}{
		{
			name: "distinct scores",
			scores: []fwk.PlacementScore{
				{Score: 10},
				{Score: 50},
				{Score: 110},
			},
			expected: []fwk.PlacementScore{
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
			expected: []fwk.PlacementScore{
				{Score: 100},
				{Score: 100},
			},
		},
		{
			name: "single score",
			scores: []fwk.PlacementScore{
				{Score: 50},
			},
			expected: []fwk.PlacementScore{
				{Score: 100},
			},
		},
		{
			name: "close scores with big gap to min",
			scores: []fwk.PlacementScore{
				{Score: 11},
				{Score: 100},
				{Score: 101},
				{Score: 102},
			},
			expected: []fwk.PlacementScore{
				{Score: 10},
				{Score: 98},
				{Score: 99},
				{Score: 100},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pl := &PodGroupCount{}
			status := pl.NormalizePlacementScore(context.Background(), nil, nil, tt.scores)
			if !status.IsSuccess() {
				t.Errorf("NormalizePlacementScore failed: %v", status.Message())
			}

			if diff := cmp.Diff(tt.expected, tt.scores); diff != "" {
				t.Errorf("Unexpected scores (-want, +got):\n%s", diff)
			}
		})
	}
}

// testPodGroupInfo implements fwk.PodGroupInfo for testing
type testPodGroupInfo struct {
	namespace string
	workload  *v1.WorkloadReference
}

func (t *testPodGroupInfo) GetUnscheduledPods() []*v1.Pod               { return nil }
func (t *testPodGroupInfo) GetWorkloadReference() *v1.WorkloadReference { return t.workload }
func (t *testPodGroupInfo) GetNamespace() string                        { return t.namespace }
