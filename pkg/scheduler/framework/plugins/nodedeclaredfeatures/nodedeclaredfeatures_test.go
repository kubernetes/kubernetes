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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/mock"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	ndf "k8s.io/component-helpers/nodedeclaredfeatures"
	ndftesting "k8s.io/component-helpers/nodedeclaredfeatures/testing"
)

// createMockFeature is a helper function to create and configure a MockFeature.
func createMockFeature(t *testing.T, name string, infer bool, maxVersionStr string) *ndftesting.MockFeature {
	m := ndftesting.NewMockFeature(t)
	m.EXPECT().Name().Return(name).Maybe()
	m.EXPECT().InferForScheduling(mock.Anything).Return(infer).Maybe()
	if maxVersionStr != "" {
		minVersion := version.MustParseSemantic(maxVersionStr)
		m.EXPECT().MaxVersion().Return(minVersion).Maybe()
	} else {
		m.EXPECT().MaxVersion().Return(nil).Maybe()
	}
	return m
}

func TestPreFilter(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	testCases := []struct {
		name              string
		pluginEnabled     bool
		pod               *v1.Pod
		nodeFeatures      []ndf.Feature
		expectedStatus    *fwk.Status
		expectedState     *preFilterState
		componenetVersion string
	}{
		{
			name:              "plugin disabled",
			pluginEnabled:     false,
			pod:               st.MakePod().Name("test-pod").Obj(),
			componenetVersion: "1.35.0",
			nodeFeatures: []ndf.Feature{
				createMockFeature(t, "TestFeature", true, ""),
			},
			expectedStatus: fwk.NewStatus(fwk.Skip),
			expectedState:  nil,
		},
		{
			name:              "Pod with feature requirements",
			pluginEnabled:     true,
			pod:               st.MakePod().Name("test-pod").Obj(),
			componenetVersion: "1.35.0",
			nodeFeatures: []ndf.Feature{
				createMockFeature(t, "TestFeature", true, ""),
			},
			expectedStatus: fwk.NewStatus(fwk.Success),
			expectedState:  &preFilterState{reqs: ndf.NewFeatureSet("TestFeature")},
		},
		{
			name:              "Pod with multiple feature requirements",
			pluginEnabled:     true,
			pod:               st.MakePod().Name("test-pod").Obj(),
			componenetVersion: "1.35.0",
			nodeFeatures: []ndf.Feature{
				createMockFeature(t, "TestFeature1", true, "1.38.0"),
				createMockFeature(t, "TestFeature2", true, "1.38.0"),
			},
			expectedStatus: fwk.NewStatus(fwk.Success),
			expectedState:  &preFilterState{reqs: ndf.NewFeatureSet("TestFeature1", "TestFeature2")},
		},
		{
			name:              "Pod with no requirements",
			pluginEnabled:     true,
			pod:               st.MakePod().Name("test-pod").Obj(),
			componenetVersion: "1.35.0",
			nodeFeatures: []ndf.Feature{
				createMockFeature(t, "TestFeature", false, ""),
			},
			expectedStatus: fwk.NewStatus(fwk.Skip),
			expectedState:  nil,
		},
		{
			name:              "Feature not required, version > MaxVersion",
			pluginEnabled:     true,
			pod:               st.MakePod().Name("test-pod").Obj(),
			componenetVersion: "1.34.0",
			nodeFeatures: []ndf.Feature{
				createMockFeature(t, "TestFeature", true, "1.33.0"),
			},
			expectedStatus: fwk.NewStatus(fwk.Skip),
			expectedState:  nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ndfFramework, err := ndf.New(tc.nodeFeatures)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}

			plugin := &NodeDeclaredFeatures{
				ndfFramework: ndfFramework,
				version:      version.MustParseSemantic(tc.componenetVersion),
				enabled:      tc.pluginEnabled,
			}
			cycleState := framework.NewCycleState()
			result, status := plugin.PreFilter(ctx, cycleState, tc.pod, nil)

			if result != nil {
				t.Errorf("PreFilter should always return a nil for result")
			}

			if diff := cmp.Diff(tc.expectedStatus, status); diff != "" {
				t.Errorf("unexpected status (-want,+got):\n%s", diff)
			}

			if tc.expectedState != nil {
				state, err := getPreFilterState(cycleState)
				if err != nil {
					t.Fatalf("getPreFilterState returned unexpected error: %v", err)
				}
				if !tc.expectedState.reqs.Equal(state.reqs) {
					t.Errorf("unexpected preFilterState reqs: want %v, got %v", tc.expectedState.reqs, state.reqs)
				}
			} else {
				_, err := getPreFilterState(cycleState)
				if err == nil {
					t.Fatalf("get prefilter state: %v", err)
				}
			}
		})
	}
}

func TestFilter(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	testCases := []struct {
		name           string
		pluginEnabled  bool
		pod            *v1.Pod
		node           *v1.Node
		preFilterReqs  []string
		expectedStatus *fwk.Status
	}{
		{
			name:           "plugin disabled",
			pluginEnabled:  false,
			pod:            st.MakePod().Name("test-pod").Obj(),
			node:           st.MakeNode().Name("node-1").DeclaredFeatures([]string{"FeatureA", "FeatureB"}).Obj(),
			preFilterReqs:  nil,
			expectedStatus: nil,
		},
		{
			name:           "Node matches requirements",
			pluginEnabled:  true,
			pod:            st.MakePod().Name("test-pod").Obj(),
			node:           st.MakeNode().Name("node-1").DeclaredFeatures([]string{"FeatureA", "FeatureB"}).Obj(),
			preFilterReqs:  []string{"FeatureA"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:           "Node does not match requirements",
			pluginEnabled:  true,
			pod:            st.MakePod().Name("test-pod").Obj(),
			node:           st.MakeNode().Name("node-1").DeclaredFeatures([]string{"FeatureB"}).Obj(),
			preFilterReqs:  []string{"FeatureA"},
			expectedStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "node declared features check failed - unsatisfied requirements: FeatureA"),
		},
		{
			name:           "Node with multiple features, pod requires subset",
			pod:            st.MakePod().Name("test-pod").Obj(),
			node:           st.MakeNode().Name("node-multi").DeclaredFeatures([]string{"FeatureA", "FeatureB", "FeatureC"}).Obj(),
			preFilterReqs:  []string{"FeatureA", "FeatureC"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:           "Node has no declared features",
			pluginEnabled:  true,
			pod:            st.MakePod().Name("test-pod").Obj(),
			node:           st.MakeNode().Name("node-1").Obj(),
			preFilterReqs:  []string{"FeatureA"},
			expectedStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "node declared features check failed - unsatisfied requirements: FeatureA"),
		},
		{
			name:           "Node with some but not all required features",
			pluginEnabled:  true,
			pod:            st.MakePod().Name("test-pod").Obj(),
			node:           st.MakeNode().Name("node-1").DeclaredFeatures([]string{"FeatureA"}).Obj(),
			preFilterReqs:  []string{"FeatureA", "FeatureB"},
			expectedStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "node declared features check failed - unsatisfied requirements: FeatureB"),
		},
		{
			name:           "Error getting pre-filter state",
			pluginEnabled:  true,
			pod:            st.MakePod().Name("test-pod").Obj(),
			node:           st.MakeNode().Name("node-1").Obj(),
			preFilterReqs:  nil, // This will cause getPreFilterState to fail
			expectedStatus: fwk.AsStatus(fmt.Errorf("error reading %q from cycle-state: %w", preFilterStateKey, fwk.ErrNotFound)),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Setting feature gate is still needed as we check for it in SetNode()
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeDeclaredFeatures, tc.pluginEnabled)
			nodeInfo := framework.NewNodeInfo()
			nodeInfo.SetNode(tc.node)

			ndfFramework, err := ndf.New([]ndf.Feature{})
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}
			plugin := &NodeDeclaredFeatures{
				ndfFramework: ndfFramework,
				version:      version.MustParseSemantic("1.35.0"),
				enabled:      tc.pluginEnabled,
			}
			cycleState := framework.NewCycleState()
			if tc.preFilterReqs != nil {
				cycleState.Write(preFilterStateKey, &preFilterState{reqs: ndf.NewFeatureSet(tc.preFilterReqs...)})
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

func TestEnqueueExtensionsNodeUpdate(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	targetPodName := "test-pod"

	testCases := []struct {
		name          string
		pluginEnabled bool
		oldNode       *v1.Node
		newNode       *v1.Node
		expectedHint  fwk.QueueingHint
	}{
		{
			name:         "Node Add with feature",
			oldNode:      nil,
			newNode:      st.MakeNode().Name("node-1").DeclaredFeatures([]string{"FeatureA"}).Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "Node Update (Features Added)",
			oldNode:      st.MakeNode().Name("node-1").Obj(),
			newNode:      st.MakeNode().Name("node-1").DeclaredFeatures([]string{"FeatureA"}).Obj(),
			expectedHint: fwk.Queue,
		},
		{
			name:         "Node Update (Features Unchanged)",
			oldNode:      st.MakeNode().Name("node-1").DeclaredFeatures([]string{"FeatureA"}).Obj(),
			newNode:      st.MakeNode().Name("node-1").DeclaredFeatures([]string{"FeatureA"}).Obj(),
			expectedHint: fwk.QueueSkip,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ndfFramework, err := ndf.New([]ndf.Feature{})
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}
			plugin := &NodeDeclaredFeatures{
				ndfFramework: ndfFramework,
				version:      version.MustParseSemantic("1.35.0"),
				enabled:      true,
			}

			hint, err := plugin.isSchedulableAfterNodeChange(logger, st.MakePod().Name(targetPodName).Obj(), tc.oldNode, tc.newNode)
			if err != nil {
				t.Fatalf("isSchedulableAfterNodeChange returned unexpected error: %v", err)
			}
			if tc.expectedHint != hint {
				t.Errorf("unexpected hint: want %v, got %v", tc.expectedHint, hint)
			}
		})
	}
}

func TestEnqueueExtensionsPodUpdate(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	targetPodName := "test-pod"
	targetPodUID := "123"

	// Test isSchedulableAfterPodUpdate
	testCases := []struct {
		name              string
		oldPod            *v1.Pod
		newPod            *v1.Pod
		setupMock         func(m *ndftesting.MockFeature)
		nodeFeatures      []ndf.Feature
		expectedHint      fwk.QueueingHint
		componenetVersion *version.Version
		expectedErr       string
	}{
		{
			name:              "Pod Update adds requirement",
			oldPod:            st.MakePod().Name(targetPodName).UID(targetPodUID).Obj(),
			newPod:            st.MakePod().Name(targetPodName).UID(targetPodUID).Label("foo", "bar").Obj(),
			componenetVersion: version.MustParseSemantic("1.35.0"),
			setupMock: func(m *ndftesting.MockFeature) {
				m.EXPECT().InferForScheduling(mock.Anything).Return(false).Once()
				m.EXPECT().InferForScheduling(mock.Anything).Return(true).Once()
				m.EXPECT().Name().Return("TestFeature").Maybe()
				m.EXPECT().MaxVersion().Return(nil).Maybe()
			},
			expectedHint: fwk.Queue,
		},
		{
			name:              "Pod Update removes requirement",
			oldPod:            st.MakePod().Name(targetPodName).UID(targetPodUID).Label("foo", "bar").Obj(),
			newPod:            st.MakePod().Name(targetPodName).UID(targetPodUID).Obj(),
			componenetVersion: version.MustParseSemantic("1.35.0"),
			setupMock: func(m *ndftesting.MockFeature) {
				m.EXPECT().InferForScheduling(mock.Anything).Return(true).Once()
				m.EXPECT().InferForScheduling(mock.Anything).Return(false).Once()
				m.EXPECT().Name().Return("TestFeature").Maybe()
				m.EXPECT().MaxVersion().Return(nil).Maybe()
			},
			expectedHint: fwk.Queue,
		},
		{
			name:              "Pod Update with no change in requirements",
			oldPod:            st.MakePod().Name(targetPodName).UID(targetPodUID).Obj(),
			newPod:            st.MakePod().Name(targetPodName).UID(targetPodUID).Obj(),
			componenetVersion: version.MustParseSemantic("1.35.0"),
			setupMock: func(m *ndftesting.MockFeature) {
				m.EXPECT().InferForScheduling(mock.Anything).Return(false)
				m.EXPECT().Name().Return("TestFeature").Maybe()
				m.EXPECT().MaxVersion().Return(nil).Maybe()
			},
			expectedHint: fwk.QueueSkip,
		},
		{
			name:              "Updated pod not the same as target pod",
			oldPod:            st.MakePod().Name("another-test-pod").UID("456").Label("foo", "bar").Obj(),
			newPod:            st.MakePod().Name("another-test-pod").UID("456").Obj(),
			componenetVersion: version.MustParseSemantic("1.35.0"),
			setupMock: func(m *ndftesting.MockFeature) {
				m.EXPECT().InferForScheduling(mock.Anything).Return(false).Maybe()
				m.EXPECT().Name().Return("TestFeature").Maybe()
				m.EXPECT().MaxVersion().Return(nil).Maybe()
			},
			expectedHint: fwk.QueueSkip,
		},
		{
			name:              "Infer returns error",
			oldPod:            st.MakePod().Name(targetPodName).UID(targetPodUID).Obj(),
			newPod:            st.MakePod().Name(targetPodName).UID(targetPodUID).Label("foo", "bar").Obj(),
			componenetVersion: nil,
			setupMock: func(m *ndftesting.MockFeature) {
				m.EXPECT().InferForScheduling(mock.Anything).Return(true).Maybe()
				m.EXPECT().Name().Return("TestFeature").Maybe()
				m.EXPECT().MaxVersion().Return(nil).Maybe()
			},
			expectedHint: fwk.Queue, // Queued again in case of error
			expectedErr:  "target version cannot be nil",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockF := ndftesting.NewMockFeature(t)
			tc.setupMock(mockF)

			ndfFramework, err := ndf.New([]ndf.Feature{mockF})
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}
			plugin := &NodeDeclaredFeatures{
				ndfFramework: ndfFramework,
				version:      tc.componenetVersion,
				enabled:      true,
			}
			hint, err := plugin.isSchedulableAfterPodUpdate(logger, st.MakePod().Name(targetPodName).UID(targetPodUID).Obj(), tc.oldPod, tc.newPod)
			if tc.expectedErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.expectedErr)
				} else if !strings.Contains(err.Error(), tc.expectedErr) {
					t.Fatalf("expected error containing %q, got %v", tc.expectedErr, err)
				}
			} else if err != nil {
				t.Fatalf("expected no error, got %v", err)
			}

			if tc.expectedHint != hint {
				t.Errorf("unexpected hint: want %v, got %v", tc.expectedHint, hint)
			}
		})
	}
}
