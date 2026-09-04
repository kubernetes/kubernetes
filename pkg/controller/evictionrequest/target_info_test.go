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
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	lifecycleapply "k8s.io/client-go/applyconfigurations/lifecycle/v1alpha1"
	testing2 "k8s.io/utils/clock/testing"
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
				pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: new("my-podgroup")}
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

func TestTerminalTime(t *testing.T) {
	clock := testing2.NewFakePassiveClock(time.Now())
	testCases := []struct {
		name                 string
		target               targetInfo
		expectedTerminalTime *time.Time
	}{
		{
			name: "no terminal time present",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1",
				setPodDeletionPhase(v1.PodSucceeded))),
			expectedTerminalTime: nil,
		},
		{
			name: "container termination time present",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1",
				setPodDeletionPhase(v1.PodSucceeded), setPodContainerStatuses(v1.ContainerStatus{
					State: v1.ContainerState{
						Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.Time{Time: clock.Now()}},
					},
				}))),
			expectedTerminalTime: new(clock.Now()),
		},
		{
			name: "deletion timestamp present",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1",
				setPodDeletionPhase(v1.PodFailed), setPodDeletionTimestamp(&metav1.Time{Time: clock.Now()}))),
			expectedTerminalTime: new(clock.Now()),
		},
		{
			name: "deletion timestamp older than container termination time",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1",
				setPodDeletionPhase(v1.PodSucceeded),
				setPodDeletionTimestamp(&metav1.Time{Time: clock.Now().Add(-time.Second)}),
				setPodContainerStatuses(v1.ContainerStatus{
					State: v1.ContainerState{
						Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.Time{Time: clock.Now()}},
					},
				}))),
			expectedTerminalTime: new(clock.Now()),
		},
		{
			name: "deletion timestamp newer than container termination time",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1",
				setPodDeletionPhase(v1.PodFailed),
				setPodDeletionTimestamp(&metav1.Time{Time: clock.Now()}),
				setPodContainerStatuses(v1.ContainerStatus{
					State: v1.ContainerState{
						Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.Time{Time: clock.Now().Add(-time.Second)}},
					},
				}))),
			expectedTerminalTime: new(clock.Now()),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.target.terminalTime(); !reflect.DeepEqual(got, tc.expectedTerminalTime) {
				t.Errorf("got terminalTime %v, want %v", got, tc.expectedTerminalTime)
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
				t.Fatalf("unexpected evictionResponders (-want +got):\n%s", diff)
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
				t.Fatalf("unexpected ApplyConfiguration (-want +got):\n%s", diff)
			}
		})
	}
}

// Similar tests in k8s.io/pkg/controller/job.TestGetFinishedTime.
func TestGetFinishedTime(t *testing.T) {
	defaultTestTime := time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC)
	containerRestartPolicyAlways := v1.ContainerRestartPolicyAlways
	testCases := map[string]struct {
		pod            v1.Pod
		wantFinishTime time.Time
	}{
		"Pod with multiple containers and all containers terminated": {
			pod: v1.Pod{
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.NewTime(defaultTestTime.Add(-1 * time.Second))},
							},
						},
						{
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.NewTime(defaultTestTime)},
							},
						},
						{
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.NewTime(defaultTestTime.Add(-2 * time.Second))},
							},
						},
					},
				},
			},
			wantFinishTime: defaultTestTime,
		},
		// In this case, init container is stopped after the regular containers.
		// This is because with the sidecar (restartable init) containers,
		// sidecar containers will always finish later than regular containers.
		"Pod with sidecar container and all containers terminated": {
			pod: v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name:          "sidecar",
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
				},
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						{
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.NewTime(defaultTestTime.Add(-1 * time.Second))},
							},
						},
					},
					InitContainerStatuses: []v1.ContainerStatus{
						{
							Name: "sidecar",
							State: v1.ContainerState{
								Terminated: &v1.ContainerStateTerminated{FinishedAt: metav1.NewTime(defaultTestTime)},
							},
						},
					},
				},
			},
			wantFinishTime: defaultTestTime,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			f := getPodFinishTime(&tc.pod)
			if !f.Equal(tc.wantFinishTime) {
				t.Errorf("Expected value of finishedTime %v; got %v", tc.wantFinishTime, f)
			}
		})
	}
}
