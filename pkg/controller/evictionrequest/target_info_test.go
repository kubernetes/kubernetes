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

package evictionrequest

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	lifecycleapply "k8s.io/client-go/applyconfigurations/lifecycle/v1alpha1"
	"k8s.io/utils/ptr"
)

func TestTargetInfo(t *testing.T) {
	testCases := []struct {
		name               string
		target             targetInfo
		expectedName       string
		expectedTargetUID  string
		expectedTargetType targetType
		expectedPodUID     string
		expectedIsGone     bool
	}{
		{
			name:               "pod found",
			target:             newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1")),
			expectedName:       "my-pod",
			expectedTargetUID:  "uid-1",
			expectedTargetType: podTarget,
			expectedPodUID:     "uid-1",
			expectedIsGone:     false,
		},
		{
			name:               "pod found with different UID",
			target:             newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-2")),
			expectedName:       "my-pod",
			expectedTargetUID:  "uid-1",
			expectedTargetType: podTarget,
			expectedPodUID:     "uid-2",
			expectedIsGone:     true,
		},
		{
			name:               "pod not found",
			target:             newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			expectedName:       "my-pod",
			expectedTargetUID:  "uid-1",
			expectedTargetType: podTarget,
			expectedPodUID:     "",
			expectedIsGone:     true,
		},
		{
			name:               "empty target",
			target:             newTargetInfoForEviction(lifecyclev1alpha1.EvictionTarget{}, nil),
			expectedName:       "",
			expectedTargetUID:  "",
			expectedTargetType: noTarget,
			expectedPodUID:     "",
			expectedIsGone:     false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.target.targetType(); tc.expectedTargetType != got {
				t.Errorf("got targetType %v, expected %v", got, tc.expectedTargetType)
			}
			if got := tc.target.targetName(); tc.expectedName != got {
				t.Errorf("got targetName %v, expected %v", got, tc.expectedName)
			}
			if got := tc.target.targetUID(); tc.expectedTargetUID != string(got) {
				t.Errorf("got targetUID %v, expected %v", got, tc.expectedTargetUID)
			}
			isFound := len(tc.expectedPodUID) > 0 && len(tc.expectedTargetUID) > 0
			if got := tc.target.targetFoundByName(); isFound != got {
				t.Errorf("got targetFoundByName %v, expected %v", got, isFound)
			}
			expectedMeta := len(tc.expectedPodUID) > 0
			if got := tc.target.GetObjectMeta(); expectedMeta != (got != nil) ||
				(got != nil && string(got.GetUID()) != tc.expectedPodUID) {
				t.Errorf("got ObjectMeta %v, expected %v", got, expectedMeta)
			}
			if got := tc.target.isGone(); tc.expectedIsGone != got {
				t.Errorf("got isGone %v, expected %v", got, tc.expectedIsGone)
			}
		})
	}
}

func TestIsPartOfPodGroup(t *testing.T) {
	testCases := []struct {
		name                       string
		target                     targetInfo
		expectedHasSchedulingGroup bool
	}{
		{
			name:                       "pod without PodGroup",
			target:                     newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1")),
			expectedHasSchedulingGroup: false,
		},
		{
			name: "pod with PodGroup",
			target: func() targetInfo {
				pod := mkValidPod("my-pod", "uid-1")
				pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: ptr.To("my-podgroup")}
				return newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), pod)
			}(),
			expectedHasSchedulingGroup: true,
		},
		{
			name:                       "pod not found",
			target:                     newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			expectedHasSchedulingGroup: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.target.hasSchedulingGroup(); got != tc.expectedHasSchedulingGroup {
				t.Errorf("got hasSchedulingGroup %v, want %v", got, tc.expectedHasSchedulingGroup)
			}
		})
	}
}

func TestIsTerminal(t *testing.T) {
	testCases := []struct {
		name               string
		target             targetInfo
		expectedIsTerminal bool
	}{
		{
			name:               "running pod",
			target:             newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1")),
			expectedIsTerminal: false,
		},
		{
			name: "succeeded pod",
			target: func() targetInfo {
				pod := mkValidPod("my-pod", "uid-1")
				pod.Status.Phase = v1.PodSucceeded
				return newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), pod)
			}(),
			expectedIsTerminal: true,
		},
		{
			name: "failed pod",
			target: func() targetInfo {
				pod := mkValidPod("my-pod", "uid-1")
				pod.Status.Phase = v1.PodFailed
				return newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), pod)
			}(),
			expectedIsTerminal: true,
		},
		{
			name:               "pod not found",
			target:             newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			expectedIsTerminal: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.target.isTerminal(); got != tc.expectedIsTerminal {
				t.Errorf("got isTerminal %v, want %v", got, tc.expectedIsTerminal)
			}
		})
	}
}

func TestEvictionResponders(t *testing.T) {
	testCases := []struct {
		name           string
		target         targetInfo
		includeDefault bool
		want           []v1.EvictionResponder
	}{
		{
			name: "pod with responders",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"),
				mkValidPod("my-pod", "uid-1", setEvictionResponders(
					v1.EvictionResponder{Name: "responder-a", Priority: new(int32(1000))},
					v1.EvictionResponder{Name: "responder-b", Priority: new(int32(15000))},
				)),
			),
			want: []v1.EvictionResponder{
				{Name: "responder-a", Priority: new(int32(1000))},
				{Name: "responder-b", Priority: new(int32(15000))},
			},
		},
		{
			name: "pod with responders + default ones",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"),
				mkValidPod("my-pod", "uid-1", setEvictionResponders(
					v1.EvictionResponder{Name: "responder-a", Priority: new(int32(1000))},
					v1.EvictionResponder{Name: "responder-b", Priority: new(int32(15000))},
				)),
			),
			includeDefault: true,
			want: []v1.EvictionResponder{
				{Name: "responder-a", Priority: new(int32(1000))},
				{Name: "responder-b", Priority: new(int32(15000))},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100))},
			},
		},
		{
			name:   "pod without responders",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1")),
			want:   nil,
		},
		{
			name:           "pod with just default responders",
			target:         newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1")),
			includeDefault: true,
			want: []v1.EvictionResponder{
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100))},
			},
		},
		{
			name:   "pod not found",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			want:   nil,
		},
		{
			name:           "pod not found with no default responders",
			target:         newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			includeDefault: true,
			want:           nil,
		},
		{
			name:           "no target; pod not found with no default responders",
			target:         newTargetInfoForEviction(lifecyclev1alpha1.EvictionTarget{}, nil),
			includeDefault: true,
			want:           nil,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.target.evictionResponders(tc.includeDefault)
			if diff := cmp.Diff(tc.want, got); len(diff) > 0 {
				t.Fatalf("unexpected evictionResponders: %s", diff)
			}
		})
	}
}

func TestToEvictionTargetApply(t *testing.T) {
	testCases := []struct {
		name                       string
		target                     targetInfo
		includeDefault             bool
		expectedApplyConfiguration *lifecycleapply.EvictionTargetApplyConfiguration
	}{
		{
			name:   "target for Eviction",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			expectedApplyConfiguration: lifecycleapply.EvictionTarget().WithPod(
				lifecycleapply.EvictionPodReference().
					WithName("my-pod").
					WithUID("uid-1"),
			),
		},
		{
			name:   "target for EvictionRequest",
			target: newTargetInfoForEvictionRequest(mkValidEvictionRequestPodTarget("my-pod", "uid-2"), nil),
			expectedApplyConfiguration: lifecycleapply.EvictionTarget().WithPod(
				lifecycleapply.EvictionPodReference().
					WithName("my-pod").
					WithUID("uid-2"),
			),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.target.toEvictionTargetApply()
			if diff := cmp.Diff(tc.expectedApplyConfiguration, got); len(diff) > 0 {
				t.Fatalf("unexpected ApplyConfiguration returned %s", diff)
			}
		})
	}
}
