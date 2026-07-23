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
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	testingclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

const (
	targetResponderA = "responder-a"
	targetResponderB = "responder-b"
)

func TestEvictionCondition(t *testing.T) {
	now := time.Now()
	clock := testingclock.NewFakeClock(now)
	tests := []struct {
		name            string
		eviction        *lifecyclev1alpha1.Eviction
		expectSucceeded bool
		expectFailed    bool
	}{
		{
			name: "failed",
			eviction: mkValidEviction("pod-1-foo", "foo", "uid-1",
				addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder),
			),
			expectFailed: true,
		},
		{
			name: "succeeded",
			eviction: mkValidEviction("pod-1-foo", "foo", "uid-1",
				addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionTargetEvicted, lifecyclev1alpha1.EvictionConditionReasonPodDeleted),
			),
			expectSucceeded: true,
		},

		{
			name: "in progress",
			eviction: mkValidEviction("pod-1-foo", "foo", "uid-1",
				addCondition(clock, lifecyclev1alpha1.EvictionConditionFailed, metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder),
			),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := hasEvictionFailed(tc.eviction); tc.expectFailed != got {
				t.Errorf("got eviction failed %v, expected %v", got, tc.expectFailed)
			}
			if got := hasEvictionSucceeded(tc.eviction); tc.expectSucceeded != got {
				t.Errorf("got succeeded failed %v, expected %v", got, tc.expectSucceeded)
			}

			expectCompleted := (tc.expectFailed || tc.expectSucceeded)
			if got := hasEvictionCompleted(tc.eviction); expectCompleted != got {
				t.Errorf("got completed failed %v, expected %v", got, expectCompleted)
			}
		})
	}
}

func TestGetOrInitializeTargetResponders(t *testing.T) {
	testCases := []struct {
		name          string
		target        targetInfo
		eviction      lifecyclev1alpha1.Eviction
		want          []lifecyclev1alpha1.TargetResponder
		expectChanged bool
	}{
		{
			name: "pod with responders + default ones",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"),
				mkValidPod("my-pod", "uid-1", setEvictionResponders(
					v1.EvictionResponder{Name: targetResponderA, Priority: new(int32(1000))},
					v1.EvictionResponder{Name: targetResponderB, Priority: new(int32(15000))},
				)),
			),
			want: []lifecyclev1alpha1.TargetResponder{
				{Name: targetResponderA, Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
				{Name: targetResponderB, Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateInactive},
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectChanged: true,
		},
		{
			name:   "pod with just default responders",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1")),
			want: []lifecyclev1alpha1.TargetResponder{
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectChanged: true,
		},
		{
			name:   "pod not found",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			want:   nil,
		},
		{
			name: "pod with responders + default ones changed, but copied from the last status instead",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"),
				mkValidPod("my-pod", "uid-1", setEvictionResponders(
					v1.EvictionResponder{Name: targetResponderA, Priority: new(int32(1000))},
					v1.EvictionResponder{Name: targetResponderB, Priority: new(int32(15000))},
				)),
			),
			eviction: *mkValidEviction("pod-1-my-pod", "my-pod", "uid-1",
				addTargetResponders("foo.example.com/bar", "foo.example.com/baz"),
				setStateFor(lifecyclev1alpha1.ResponderStateActive, 0)),
			want: []lifecyclev1alpha1.TargetResponder{
				{Name: "foo.example.com/bar", Priority: new(int32(5000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "foo.example.com/baz", Priority: new(int32(4999)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, changed := getOrInitializeTargetResponders(&tc.eviction, tc.target)
			if diff := cmp.Diff(tc.want, got); len(diff) > 0 {
				t.Fatalf("unexpected targetResponders (-want +got):\n%s", diff)
			}
			if changed != tc.expectChanged {
				t.Errorf("got changed %v, want %v", changed, tc.expectChanged)
			}
		})
	}
}

func TestComputeResponderProgression(t *testing.T) {
	clock := testingclock.NewFakePassiveClock(time.Now())
	testCases := []struct {
		name                     string
		target                   targetInfo
		isGone                   bool
		isTerminal               bool
		isCanceled               bool
		statusResponders         []lifecyclev1alpha1.ResponderStatus
		targetResponders         []lifecyclev1alpha1.TargetResponder
		expectedTargetResponders []lifecyclev1alpha1.TargetResponder
		expectProgressionDone    bool
		expectResync             *time.Duration
	}{
		{
			name:                     "empty",
			target:                   newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			targetResponders:         []lifecyclev1alpha1.TargetResponder{},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{},
			expectProgressionDone:    true,
		},

		{
			name:   "activate the first one",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateInactive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectResync: new(ResponderHeartbeatTimeout),
		},
		{
			name:   "activate the second one",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: targetResponderB},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectResync: new(ResponderHeartbeatTimeout),
		},
		{
			name:   "activate the third one",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: targetResponderB, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateActive),
			},
			expectResync: new(ResponderHeartbeatTimeout),
		},
		{
			name:   "complete the third one",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: targetResponderB, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateActive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateCompleted),
			},
		},
		{
			name:   "no change after completion",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: targetResponderB, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateCompleted),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateCompleted),
			},
			expectProgressionDone: true,
		},
		{
			name:   "no change after last interruption",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: targetResponderB, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, StartTime: new(metav1.Now())},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateInterrupted),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateInterrupted),
			},
			expectProgressionDone: true,
		},
		{
			name:   "no change after last cancellation",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: targetResponderB, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, StartTime: new(metav1.Now())},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateCanceled),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(lifecyclev1alpha1.EvictionResponderImperativeEviction, lifecyclev1alpha1.ResponderStateCanceled),
			},
			expectProgressionDone: true,
		},
		{
			name:   "reschedule resync according to the start time",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Time{Time: clock.Now().Add(-1 * time.Minute)})},
				{Name: targetResponderB},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectResync: new(ResponderHeartbeatTimeout - 1*time.Minute),
		},
		{
			name:   "reschedule resync according to the heartbeat time",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{
					Name:          targetResponderA,
					StartTime:     new(metav1.Time{Time: clock.Now().Add(-25 * time.Minute)}),
					HeartbeatTime: new(metav1.Time{Time: clock.Now().Add(-2 * time.Minute)}),
				},
				{Name: targetResponderB},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectResync: new(ResponderHeartbeatTimeout - 2*time.Minute),
		},
		{
			name:   "interrupt the first after the heartbeat has elapsed and activate the second one",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{
					Name:          targetResponderA,
					StartTime:     new(metav1.Time{Time: clock.Now().Add(-25 * time.Minute)}),
					HeartbeatTime: new(metav1.Time{Time: clock.Now().Add(-20 * time.Minute)}),
				},
				{Name: targetResponderB},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateInterrupted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateActive),
			},
			expectResync: new(ResponderHeartbeatTimeout),
		},
		{
			name:   "interrupt the first after the heartbeat (start time fallback has elapsed and activate the second one",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Time{Time: clock.Now().Add(-21 * time.Minute)})},
				{Name: targetResponderB},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateInterrupted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateActive),
			},
			expectResync: new(ResponderHeartbeatTimeout),
		},
		{
			name:   "complete after the target is gone",
			isGone: true,
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: targetResponderB},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
		},
		{
			name:       "complete after the target is terminal with completion time set",
			isTerminal: true,
			target:     newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: targetResponderB},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateCompleted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
		},
		{
			name:   "interrupt after the target is gone without completion time set",
			isGone: true,
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Now())},
				{Name: targetResponderB},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateInterrupted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
		},
		{
			name:       "interrupt after the target is terminal without completion time set",
			isTerminal: true,
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1",
				setPodDeletionTimestamp(new(metav1.Time{Time: clock.Now().Add(-6 * time.Second)})),
			)),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Now())},
				{Name: targetResponderB},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateInterrupted),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
		},
		{
			name:       "wait for completionTime after the target is terminal",
			isTerminal: true,
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1",
				setPodDeletionTimestamp(new(metav1.Time{Time: clock.Now().Add(-4 * time.Second)})),
			)),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Now())},
				{Name: targetResponderB},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectResync: new(1 * time.Second),
		},
		{
			name:       "cancel active responder",
			isCanceled: true,
			target:     newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1")),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Now())},
				{Name: targetResponderB},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateActive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateCanceled),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
		},

		{
			name:       "cancel non active responder",
			isCanceled: true,
			target:     newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1")),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: targetResponderA, StartTime: new(metav1.Now())},
				{Name: targetResponderB},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateInactive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				targetResponder(targetResponderA, lifecyclev1alpha1.ResponderStateInactive),
				targetResponder(targetResponderB, lifecyclev1alpha1.ResponderStateInactive),
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			eviction := mkValidEviction("foo", "foo", "foo-1")
			eviction.Status.Responders = tc.statusResponders
			// also tests computeResponderStateAndNextResync
			isProgressionDone, resyncAfter := computeResponderProgression(clock.Now(), eviction, tc.targetResponders, tc.target, tc.isGone, tc.isTerminal, tc.isCanceled)
			if isProgressionDone != tc.expectProgressionDone {
				t.Errorf("got isProgressionDone %v, want %v", isProgressionDone, tc.expectProgressionDone)
			}
			if !ptr.Equal(resyncAfter, tc.expectResync) {
				t.Errorf("got defer completion %v, expected %v", ptr.Deref(resyncAfter, -1), ptr.Deref(tc.expectResync, -1))
			}
			if diff := cmp.Diff(tc.expectedTargetResponders, tc.targetResponders); len(diff) > 0 {
				t.Fatalf("unexpected targetResponders (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSortEvictionRequestsByRelevance(t *testing.T) {
	now := time.Now()
	clock := testingclock.NewFakeClock(now)
	metaNow := new(metav1.Time{Time: clock.Now()})
	testCases := []struct {
		name             string
		requests         []*lifecyclev1alpha1.EvictionRequest
		expectedRequests []*lifecyclev1alpha1.EvictionRequest
	}{
		{
			name: "no requesters",
		},
		{
			name: "sort requesters",
			requests: []*lifecyclev1alpha1.EvictionRequest{
				mkValidEvictionRequest("requester-2", "pod-2", setERRequester("foo.example.com/2"),
					setERCreationTimestamp(metav1.Time{})),
				mkValidEvictionRequest("requester-4", "pod-4", setERRequester("foo.example.com/4"),
					setERCreationTimestamp(metav1.Time{}),
					setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("requester-1", "pod-1", setERRequester("foo.example.com/1"),
					setERCreationTimestamp(*metaNow)),
				mkValidEvictionRequest("requester-3", "pod-3", setERRequester("foo.example.com/3"),
					setERCreationTimestamp(*metaNow)),
				mkValidEvictionRequest("requester-5", "pod-5", setERRequester("foo.example.com/5"),
					setERCreationTimestamp(*metaNow),
					setERDeletionTimestamp(metaNow)),
				mkValidEvictionRequest("requester-7", "pod-7", setERRequester("foo.example.com/7"),
					setERCreationTimestamp(*metaNow)),
				mkValidEvictionRequest("requester-6", "pod-6", setERRequester("foo.example.com/6"),
					setERCreationTimestamp(*metaNow),
					setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("requester-8", "pod-8", setERRequester("foo.example.com/8"),
					setERCreationTimestamp(*metaNow),
					setERDeletionTimestamp(metaNow)),
				mkValidEvictionRequest("requester-9", "pod-9", setERRequester("foo.example.com/9"),
					setERCreationTimestamp(*metaNow),
					setERDeletionTimestamp(metaNow),
					setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
			},
			expectedRequests: []*lifecyclev1alpha1.EvictionRequest{
				mkValidEvictionRequest("requester-1", "pod-1", setERRequester("foo.example.com/1"), // evictions always first
					setERCreationTimestamp(*metaNow)),
				mkValidEvictionRequest("requester-3", "pod-3", setERRequester("foo.example.com/3"),
					setERCreationTimestamp(*metaNow)),
				mkValidEvictionRequest("requester-7", "pod-7", setERRequester("foo.example.com/7"),
					setERCreationTimestamp(*metaNow)),
				mkValidEvictionRequest("requester-6", "pod-6", setERRequester("foo.example.com/6"), // non deleted
					setERCreationTimestamp(*metaNow),
					setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("requester-5", "pod-5", setERRequester("foo.example.com/5"), // deletionTimestamp
					setERCreationTimestamp(*metaNow),
					setERDeletionTimestamp(metaNow)),
				mkValidEvictionRequest("requester-8", "pod-8", setERRequester("foo.example.com/8"),
					setERCreationTimestamp(*metaNow),
					setERDeletionTimestamp(metaNow)),
				mkValidEvictionRequest("requester-9", "pod-9", setERRequester("foo.example.com/9"),
					setERCreationTimestamp(*metaNow),
					setERDeletionTimestamp(metaNow),
					setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("requester-2", "pod-2", setERRequester("foo.example.com/2"), // moved to the end, since it has no EvictionRequest assigned
					setERCreationTimestamp(metav1.Time{})),
				mkValidEvictionRequest("requester-4", "pod-4", setERRequester("foo.example.com/4"),
					setERCreationTimestamp(metav1.Time{}),
					setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			sortEvictionRequestsByRelevance(tc.requests)
			if diff := cmp.Diff(tc.expectedRequests, tc.requests); len(diff) > 0 {
				t.Fatalf("unexpected EvictionRequests order (-want +got):\n%s", diff)
			}
		})
	}
}

func targetResponder(name string, state lifecyclev1alpha1.ResponderStateType) lifecyclev1alpha1.TargetResponder {
	var priority int32
	switch name {
	case targetResponderA:
		priority = 15000
	case targetResponderB:
		priority = 1000
	case lifecyclev1alpha1.EvictionResponderImperativeEviction:
		priority = 100
	default:
		panic("unknown responder")
	}
	return lifecyclev1alpha1.TargetResponder{
		Name:     name,
		Priority: &priority,
		State:    state,
	}
}
