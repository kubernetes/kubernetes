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

package nodedeclaredfeatures

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"

	ndf "k8s.io/component-helpers/nodedeclaredfeatures"
	"k8s.io/component-helpers/nodedeclaredfeatures/features"
	featuretesting "k8s.io/component-helpers/nodedeclaredfeatures/testing"
)

func TestPreFilter(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	testCases := []struct {
		name           string
		pod            *v1.Pod
		features       []ndf.Feature
		expectedStatus *fwk.Status
		expectedState  *preFilterState
	}{
		{
			name: "Pod with feature requirements",
			pod:  st.MakePod().Name("test-pod").Obj(),
			features: []ndf.Feature{
				&featuretesting.MockFeature{
					NameFunc:            func() string { return "TestFeature" },
					InferFromCreateFunc: func(podInfo *ndf.PodInfo) bool { return true },
				},
			},
			expectedStatus: fwk.NewStatus(fwk.Success),
			expectedState:  &preFilterState{reqs: []string{"TestFeature"}},
		},
		{
			name: "Pod with multiple feature requirements",
			pod:  st.MakePod().Name("test-pod").Obj(),
			features: []ndf.Feature{
				&featuretesting.MockFeature{
					NameFunc:            func() string { return "TestFeature1" },
					InferFromCreateFunc: func(podInfo *ndf.PodInfo) bool { return true },
				},
				&featuretesting.MockFeature{
					NameFunc:            func() string { return "TestFeature2" },
					InferFromCreateFunc: func(podInfo *ndf.PodInfo) bool { return true },
				},
			},
			expectedStatus: fwk.NewStatus(fwk.Success),
			expectedState:  &preFilterState{reqs: []string{"TestFeature1", "TestFeature2"}},
		},
		{
			name: "Pod with no requirements",
			pod:  st.MakePod().Name("test-pod").Obj(),
			features: []ndf.Feature{
				&featuretesting.MockFeature{
					NameFunc:            func() string { return "TestFeature" },
					InferFromCreateFunc: func(podInfo *ndf.PodInfo) bool { return false },
				},
			},
			expectedStatus: fwk.NewStatus(fwk.Skip),
			expectedState:  nil,
		},
		{
			name: "Inference error due to version incompatibility",
			pod:  st.MakePod().Name("test-pod").Obj(),
			features: []ndf.Feature{
				&featuretesting.MockFeature{
					NameFunc:            func() string { return "TestFeature" },
					InferFromCreateFunc: func(podInfo *ndf.PodInfo) bool { return true },
					MinVersionFunc:      func() *version.Version { return version.MustParseSemantic("1.31.0") },
				},
			},
			expectedStatus: fwk.AsStatus(fmt.Errorf("inferring pod requirements: feature \"TestFeature\" is not available in version 1.30.0, requires at least 1.31.0")),
			expectedState:  nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			helper, err := ndf.NewHelper(tc.features)
			require.NoError(t, err)

			plugin := &NodeDeclaredFeatures{
				helper:  helper,
				version: version.MustParseSemantic("1.30.0"),
			}
			cycleState := framework.NewCycleState()
			result, status := plugin.PreFilter(ctx, cycleState, tc.pod, nil)
			assert.Nil(t, result, "PreFilter should always return a nil for %s plugin", names.NodeDeclaredFeatures)
			if !status.IsSuccess() {
				if tc.expectedStatus.Code() != status.Code() {
					t.Errorf("unexpected status code: want %d, got %d", tc.expectedStatus.Code(), status.Code())
				}
				if tc.expectedStatus.Message() != status.Message() {
					t.Errorf("unexpected status message: want %q, got %q", tc.expectedStatus.Message(), status.Message())
				}
			} else if diff := cmp.Diff(tc.expectedStatus, status); diff != "" {
				t.Errorf("unexpected status (-want,+got):\n%s", diff)
			}

			if tc.expectedState != nil {
				state, err := getPreFilterState(cycleState)
				require.NoError(t, err)
				if diff := cmp.Diff(tc.expectedState, state, cmp.AllowUnexported(preFilterState{})); diff != "" {
					t.Errorf("unexpected preFilterState (-want,+got):\n%s", diff)
				}
			} else {
				_, err := getPreFilterState(cycleState)
				assert.Error(t, err) // Expect an error if no state was written
			}
		})
	}
}

func TestFilter(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	nodeWithFeatures := st.MakeNode().Name("node-1").Obj()
	nodeWithFeatures.Status.DeclaredFeatures = []string{"TestFeature", "OtherFeature"}

	nodeWithoutFeature := st.MakeNode().Name("node-2").Obj()
	nodeWithoutFeature.Status.DeclaredFeatures = []string{"OtherFeature"}

	nodeWithNoFeatures := st.MakeNode().Name("node-3").Obj()

	testCases := []struct {
		name           string
		pod            *v1.Pod
		node           *v1.Node
		preFilterReqs  []string
		expectedStatus *fwk.Status
	}{
		{
			name:           "Node matches requirements",
			pod:            st.MakePod().Name("test-pod").Obj(),
			node:           nodeWithFeatures,
			preFilterReqs:  []string{"TestFeature"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:           "Node does not match requirements",
			pod:            st.MakePod().Name("test-pod").Obj(),
			node:           nodeWithoutFeature,
			preFilterReqs:  []string{"TestFeature"},
			expectedStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "node declared features check failed - unsatisfied requirements: TestFeature"),
		},
		{
			name:           "Node has no declared features",
			pod:            st.MakePod().Name("test-pod").Obj(),
			node:           nodeWithNoFeatures,
			preFilterReqs:  []string{"TestFeature"},
			expectedStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "node declared features check failed - unsatisfied requirements: TestFeature"),
		},
		{
			name:           "NodeInfo.Node() is nil",
			pod:            st.MakePod().Name("test-pod").Obj(),
			node:           nil, // Simulate nil node
			preFilterReqs:  []string{"TestFeature"},
			expectedStatus: fwk.NewStatus(fwk.Error, "node not found"),
		},
		{
			name:           "Node with some but not all required features",
			pod:            st.MakePod().Name("test-pod").Obj(),
			node:           nodeWithoutFeature, // This node has "OtherFeature" but not "TestFeature"
			preFilterReqs:  []string{"TestFeature", "OtherFeature"},
			expectedStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "node declared features check failed - unsatisfied requirements: TestFeature"),
		},
		{
			name:           "Error getting pre-filter state",
			pod:            st.MakePod().Name("test-pod").Obj(),
			node:           st.MakeNode().Name("node-1").Obj(),
			preFilterReqs:  nil, // This will cause getPreFilterState to fail
			expectedStatus: fwk.AsStatus(fmt.Errorf("error reading %q from cycle-state: %w", preFilterStateKey, fwk.ErrNotFound)),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			helper, err := ndf.NewHelper([]ndf.Feature{}) // Corrected: removed unused features
			require.NoError(t, err)

			plugin := &NodeDeclaredFeatures{
				helper:  helper,
				version: version.MustParseSemantic("1.30.0"),
			}
			cycleState := framework.NewCycleState()
			if tc.preFilterReqs != nil {
				cycleState.Write(preFilterStateKey, &preFilterState{reqs: tc.preFilterReqs})
			}

			nodeInfo := framework.NewNodeInfo()
			if tc.node != nil {
				nodeInfo.SetNode(tc.node)
			}

			status := plugin.Filter(ctx, cycleState, tc.pod, nodeInfo)
			if !status.IsSuccess() {
				if tc.expectedStatus.Code() != status.Code() {
					t.Errorf("unexpected status code: want %d, got %d", tc.expectedStatus.Code(), status.Code())
				}
				if tc.expectedStatus.Message() != status.Message() {
					t.Errorf("unexpected status message: want %q, got %q", tc.expectedStatus.Message(), status.Message())
				}
			} else if diff := cmp.Diff(tc.expectedStatus, status); diff != "" {
				t.Errorf("unexpected status (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestEnqueueExtensions(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	logger, _ := ktesting.NewTestContext(t)

	nodeWithFeature := st.MakeNode().Name("node-1").Obj()
	nodeWithFeature.Status.DeclaredFeatures = []string{"TestFeature"}
	nodeWithOtherFeature := st.MakeNode().Name("node-2").Obj()
	nodeWithOtherFeature.Status.DeclaredFeatures = []string{"OtherFeature"}
	nodeWithoutFeatures := st.MakeNode().Name("node-3").Obj()

	helper, err := ndf.NewHelper([]ndf.Feature{}) // Corrected: removed unused features
	require.NoError(t, err)

	plugin := &NodeDeclaredFeatures{
		helper:  helper,
		version: version.MustParseSemantic("1.30.0"),
	}

	events, err := plugin.EventsToRegister(ctx)
	require.NoError(t, err)
	require.Len(t, events, 2)

	// Test isSchedulableAfterNodeChange
	t.Run("isSchedulableAfterNodeChange", func(t *testing.T) {
		testCases := []struct {
			name         string
			oldNode      *v1.Node
			newNode      *v1.Node
			expectedHint fwk.QueueingHint
		}{
			{
				name:         "Node Add with feature",
				oldNode:      nil,
				newNode:      nodeWithFeature,
				expectedHint: fwk.Queue,
			},
			{
				name:         "Node Update (Features Added)",
				oldNode:      nodeWithoutFeatures,
				newNode:      nodeWithFeature,
				expectedHint: fwk.Queue,
			},
			{
				name:         "Node Update (Features Unchanged)",
				oldNode:      nodeWithFeature,
				newNode:      nodeWithFeature,
				expectedHint: fwk.QueueSkip,
			},
			{
				name:         "Node Update (Irrelevant Features change)",
				oldNode:      nodeWithFeature,
				newNode:      nodeWithOtherFeature,
				expectedHint: fwk.Queue,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				hint, err := plugin.isSchedulableAfterNodeChange(logger, st.MakePod().Name("test-pod").Obj(), tc.oldNode, tc.newNode)
				require.NoError(t, err)
				assert.Equal(t, tc.expectedHint, hint)
			})
		}
	})

	// Test isSchedulableAfterPodUpdate
	t.Run("isSchedulableAfterPodUpdate", func(t *testing.T) {
		testCases := []struct {
			name         string
			oldPod       *v1.Pod
			newPod       *v1.Pod
			features     []ndf.Feature
			expectedHint fwk.QueueingHint
		}{
			{
				name:   "Pod Update adds requirement",
				oldPod: st.MakePod().Name("test-pod").Obj(),
				newPod: st.MakePod().Name("test-pod").Label("foo", "bar").Obj(),
				features: []ndf.Feature{
					&featuretesting.MockFeature{
						InferFromCreateFunc: func(podInfo *ndf.PodInfo) bool {
							return podInfo.Pod.Labels["foo"] == "bar"
						},
					},
				},
				expectedHint: fwk.Queue,
			},
			{
				name:   "Pod Update removes requirement",
				oldPod: st.MakePod().Name("test-pod").Label("foo", "bar").Obj(),
				newPod: st.MakePod().Name("test-pod").Obj(),
				features: []ndf.Feature{
					&featuretesting.MockFeature{
						InferFromCreateFunc: func(podInfo *ndf.PodInfo) bool {
							return podInfo.Pod.Labels["foo"] == "bar"
						},
					},
				},
				expectedHint: fwk.Queue,
			},
			{
				name:   "Pod Update with no change in requirements",
				oldPod: st.MakePod().Name("test-pod").Obj(),
				newPod: st.MakePod().Name("test-pod").Obj(),
				features: []ndf.Feature{
					&featuretesting.MockFeature{
						InferFromCreateFunc: func(podInfo *ndf.PodInfo) bool {
							return podInfo.Pod.Labels["foo"] == "bar"
						},
					},
				},
				expectedHint: fwk.QueueSkip,
			},
			{
				name:   "Inference error",
				oldPod: st.MakePod().Name("test-pod").Obj(),
				newPod: st.MakePod().Name("test-pod").Obj(),
				features: []ndf.Feature{
					&featuretesting.MockFeature{
						InferFromCreateFunc: func(podInfo *ndf.PodInfo) bool {
							return true // This will trigger the version check
						},
						MinVersionFunc: func() *version.Version { return version.MustParseSemantic("1.31.0") },
					},
				},
				expectedHint: fwk.QueueSkip, // Error should result in fwk.QueueSkip
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				helper, err := ndf.NewHelper(tc.features)
				require.NoError(t, err)
				plugin.helper = helper
				hint, err := plugin.isSchedulableAfterPodUpdate(logger, st.MakePod().Name("test-pod").Obj(), tc.oldPod, tc.newPod)
				require.NoError(t, err)
				assert.Equal(t, tc.expectedHint, hint)
			})
		}
	})
}

func TestNew(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	fh, err := runtime.NewFramework(ctx, nil, nil)
	require.NoError(t, err)

	// Test successful creation
	p, err := New(ctx, nil, fh)
	require.NoError(t, err)
	assert.NotNil(t, p)

	// Test error during helper creation by manipulating the global registry
	originalFeatures := features.AllFeatures
	defer func() {
		features.AllFeatures = originalFeatures
	}()
	features.AllFeatures = nil // This will cause NewHelper to return an error

	p, err = New(ctx, nil, fh)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create node feature helper: registry must not be nil")
	assert.Nil(t, p)
}
