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
	lifecycleapply "k8s.io/client-go/applyconfigurations/lifecycle/v1alpha1"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	testingclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

func TestGetOrInitializeTargetResponders(t *testing.T) {
	testCases := []struct {
		name     string
		target   targetInfo
		eviction lifecyclev1alpha1.Eviction
		want     []lifecyclev1alpha1.TargetResponder
	}{
		{
			name: "pod with responders + default ones",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"),
				mkValidPod("my-pod", "uid-1", setEvictionResponders(
					v1.EvictionResponder{Name: "responder-a", Priority: new(int32(1000))},
					v1.EvictionResponder{Name: "responder-b", Priority: new(int32(15000))},
				)),
			),
			want: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
				{Name: "responder-b", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateInactive},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
		},
		{
			name:   "pod with just default responders",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1")),
			want: []lifecyclev1alpha1.TargetResponder{
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
		},
		{
			name:   "pod not found",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			want:   nil,
		},
		{
			name:   "pod not found with no default responders",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			want:   nil,
		},
		{
			name: "pod with responders + default ones changed, but copied from the last status instead",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"),
				mkValidPod("my-pod", "uid-1", setEvictionResponders(
					v1.EvictionResponder{Name: "responder-a", Priority: new(int32(1000))},
					v1.EvictionResponder{Name: "responder-b", Priority: new(int32(15000))},
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
			got := getOrInitializeTargetResponders(&tc.eviction, tc.target)
			if diff := cmp.Diff(tc.want, got); len(diff) > 0 {
				t.Fatalf("unexpected targetResponders: %s", diff)
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
		},

		{
			name:   "activate the first one",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateInactive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectResync: new(ResponderHeartbeatTimeout),
		},
		{
			name:   "activate the second one",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: "responder-b"},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectResync: new(ResponderHeartbeatTimeout),
		},
		{
			name:   "activate the third one",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: "responder-b", StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateActive},
			},
			expectResync: new(ResponderHeartbeatTimeout),
		},
		{
			name:   "complete the third one",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: "responder-b", StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateActive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateCompleted},
			},
		},
		{
			name:   "no change after completion",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: "responder-b", StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateCompleted},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateCompleted},
			},
			expectProgressionDone: true,
		},
		{
			name:   "no change after last interruption",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: "responder-b", StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, StartTime: new(metav1.Now())},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateInterrupted},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateInterrupted},
			},
			expectProgressionDone: true,
		},
		{
			name:   "no change after last cancellation",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: "responder-b", StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, StartTime: new(metav1.Now())},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateCanceled},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: lifecyclev1alpha1.EvictionResponderImperativeEviction, Priority: new(int32(100)), State: lifecyclev1alpha1.ResponderStateCanceled},
			},
			expectProgressionDone: true,
		},
		{
			name:   "reschedule resync according to the start time",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Time{clock.Now().Add(-1 * time.Minute)})},
				{Name: "responder-b"},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectResync: new(ResponderHeartbeatTimeout - 1*time.Minute),
		},
		{
			name:   "reschedule resync according to the heartbeat time",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{
					Name:          "responder-a",
					StartTime:     new(metav1.Time{clock.Now().Add(-25 * time.Minute)}),
					HeartbeatTime: new(metav1.Time{clock.Now().Add(-2 * time.Minute)}),
				},
				{Name: "responder-b"},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectResync: new(ResponderHeartbeatTimeout - 2*time.Minute),
		},
		{
			name:   "interrupt the first after the heartbeat has elapsed and activate the second one",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{
					Name:          "responder-a",
					StartTime:     new(metav1.Time{clock.Now().Add(-25 * time.Minute)}),
					HeartbeatTime: new(metav1.Time{clock.Now().Add(-20 * time.Minute)}),
				},
				{Name: "responder-b"},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateInterrupted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateActive},
			},
			expectResync: new(ResponderHeartbeatTimeout),
		},
		{
			name:   "interrupt the first after the heartbeat (start time fallback has elapsed and activate the second one",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Time{clock.Now().Add(-21 * time.Minute)})},
				{Name: "responder-b"},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateInterrupted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateActive},
			},
			expectResync: new(ResponderHeartbeatTimeout),
		},
		{
			name:   "complete after the target is gone",
			isGone: true,
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: "responder-b"},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
		},
		{
			name:       "complete after the target is terminal with completion time set",
			isTerminal: true,
			target:     newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Now()), CompletionTime: new(metav1.Now())},
				{Name: "responder-b"},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateCompleted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
		},
		{
			name:   "interrupt after the target is gone without completion time set",
			isGone: true,
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Now())},
				{Name: "responder-b"},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateInterrupted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
		},
		{
			name:       "interrupt after the target is terminal without completion time set",
			isTerminal: true,
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1",
				setPodDeletionTimestamp(new(metav1.Time{Time: clock.Now().Add(-6 * time.Second)})),
			)),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Now())},
				{Name: "responder-b"},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateInterrupted},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
		},
		{
			name:       "wait for completionTime after the target is terminal",
			isTerminal: true,
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1",
				setPodDeletionTimestamp(new(metav1.Time{Time: clock.Now().Add(-4 * time.Second)})),
			)),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Now())},
				{Name: "responder-b"},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectResync: new(1 * time.Second),
		},
		{
			name:       "cancel active responder",
			isCanceled: true,
			target:     newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1")),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Now())},
				{Name: "responder-b"},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateActive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateCanceled},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
		},

		{
			name:       "cancel non active responder",
			isCanceled: true,
			target:     newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), mkValidPod("my-pod", "uid-1")),
			statusResponders: []lifecyclev1alpha1.ResponderStatus{
				{Name: "responder-a", StartTime: new(metav1.Now())},
				{Name: "responder-b"},
			},
			targetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateInactive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
			},
			expectedTargetResponders: []lifecyclev1alpha1.TargetResponder{
				{Name: "responder-a", Priority: new(int32(15000)), State: lifecyclev1alpha1.ResponderStateInactive},
				{Name: "responder-b", Priority: new(int32(1000)), State: lifecyclev1alpha1.ResponderStateInactive},
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
				t.Fatalf("unexpected targetResponders: %s", diff)
			}
		})
	}
}

func TestValidate(t *testing.T) {
	clock := testingclock.NewFakePassiveClock(time.Now())
	tests := []struct {
		name                string
		hasEvictions        bool
		target              lifecyclev1alpha1.EvictionTarget
		pod                 *v1.Pod
		isDuplicate         bool
		testEviction        bool
		testEvictionRequest bool
		expected            []metav1ac.ConditionApplyConfiguration
	}{
		{
			name:                "valid pod",
			hasEvictions:        false,
			target:              mkValidPodTarget("my-pod", "uid-1"),
			pod:                 mkValidPod("my-pod", "uid-1"),
			expected:            nil,
			testEvictionRequest: true,
		},

		{
			name:                "pod not found after eviction",
			hasEvictions:        true,
			target:              mkValidPodTarget("my-pod", "uid-1"),
			pod:                 nil,
			expected:            nil,
			testEvictionRequest: true,
		},
		{
			name:                "UID mismatch after eviction",
			hasEvictions:        true,
			target:              mkValidPodTarget("my-pod", "uid-1"),
			pod:                 mkValidPod("my-pod", "uid-2"),
			expected:            nil,
			testEvictionRequest: true,
		},
		{
			name:         "pod not found",
			hasEvictions: false,
			target:       mkValidPodTarget("my-pod", "uid-1"),
			pod:          nil,
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid,
					"Target pod not found."),
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			testEvictionRequest: true,
		},
		{
			name:         "UID mismatch",
			hasEvictions: false,
			target:       mkValidPodTarget("my-pod", "uid-1"),
			pod:          mkValidPod("my-pod", "uid-2"),
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid,
					"Target pod UID mismatch: expected uid-1, got uid-2."),
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			testEvictionRequest: true,
		},
		{
			name:         "pod with PodGroup",
			hasEvictions: false,
			target:       mkValidPodTarget("my-pod", "uid-1"),
			pod: func() *v1.Pod {
				pod := mkValidPod("my-pod", "uid-1")
				pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: new("my-podgroup")}
				return pod
			}(),
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid,
					"Target pod references a SchedulingGroup. Eviction is currently not supported."),
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			testEviction:        true,
			testEvictionRequest: true,
		},
		{
			name:         "duplicate eviction",
			hasEvictions: false,
			target:       mkValidPodTarget("my-pod", "uid-1"),
			pod:          mkValidPod("my-pod", "uid-1"),
			isDuplicate:  true,
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid,
					"Active Eviction already exists for the same target."),
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			testEviction: true,
		},
		{
			name:         "empty target",
			hasEvictions: false,
			target:       lifecyclev1alpha1.EvictionTarget{},
			pod:          nil,
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid,
					"Unsupported target type."),
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			testEviction:        true,
			testEvictionRequest: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// always test EvictionRequest, Eviction has only a subset of validations
			if tc.testEviction {
				failed, evicted := validateEviction(clock.Now(), &lifecyclev1alpha1.Eviction{},
					newTargetInfoForEviction(tc.target, tc.pod), tc.isDuplicate)
				var got []metav1ac.ConditionApplyConfiguration
				if failed != nil {
					got = append(got, *failed)
				}
				if evicted != nil {
					got = append(got, *evicted)
				}
				if diff := cmp.Diff(tc.expected, got); diff != "" {
					t.Errorf("unexpected conditions update for validateEviction (-want +got):\n%s", diff)
				}
			}
			if tc.testEvictionRequest {
				failed, evicted := validateEvictionRequest(clock.Now(), &lifecyclev1alpha1.EvictionRequest{},
					newTargetInfoForEviction(tc.target, tc.pod), tc.hasEvictions)
				var got []metav1ac.ConditionApplyConfiguration
				if failed != nil {
					got = append(got, *failed)
				}
				if evicted != nil {
					got = append(got, *evicted)
				}
				if diff := cmp.Diff(tc.expected, got); diff != "" {
					t.Errorf("unexpected conditions update for validateEvictionRequest (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func TestComputeEvictionConditions(t *testing.T) {
	clock := testingclock.NewFakePassiveClock(time.Now())
	tests := []struct {
		name                        string
		isWaitingForResponderUpdate bool
		isGone                      bool
		isTerminal                  bool
		isCanceled                  bool
		isProgressionDone           bool
		expected                    []metav1ac.ConditionApplyConfiguration
	}{
		{
			name:   "pod deleted",
			isGone: true,
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonSucceeded,
					""),
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonPodDeleted, "Target pod has been deleted"),
			},
		},
		{
			name:                        "pod deleted - waiting",
			isGone:                      true,
			isWaitingForResponderUpdate: true,
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction,
					""),
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction, ""),
			},
		},

		{
			name:       "pod terminal",
			isTerminal: true,
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonSucceeded, ""),
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonPodTerminal, "Pod has reached terminal state"),
			},
		},
		{
			name:                        "pod terminal - waiting",
			isTerminal:                  true,
			isWaitingForResponderUpdate: true,
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction, ""),
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction, ""),
			},
		},
		{
			name:       "is canceled",
			isCanceled: true,
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonCanceledDueToNoRequesters, "No active requesters with eviction intent"),
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
		},

		{
			name:              "no progress",
			isProgressionDone: true,
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder, "All responders have completed without evicting the target"),
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
		},
		{
			name: "pending",
			expected: []metav1ac.ConditionApplyConfiguration{
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction, ""),
				*setCondition(clock.Now(), nil, lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction, ""),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			failed, evicted := computeEvictionConditions(clock.Now(), &lifecyclev1alpha1.Eviction{},
				tc.isWaitingForResponderUpdate, tc.isGone, tc.isTerminal, tc.isCanceled, tc.isProgressionDone)
			var got []metav1ac.ConditionApplyConfiguration
			if failed != nil {
				got = append(got, *failed)
			}
			if evicted != nil {
				got = append(got, *evicted)
			}
			if diff := cmp.Diff(tc.expected, got); diff != "" {
				t.Errorf("unexpected conditions update (-want +got):\n%s", diff)
			}
		})
	}
}

func TestUpdateRequestersForEvictionStatusApply(t *testing.T) {
	testCases := []struct {
		name                       string
		existingRequesters         []lifecyclev1alpha1.Requester
		requests                   []*lifecyclev1alpha1.EvictionRequest
		limit                      int
		expectedApplyConfiguration *lifecycleapply.EvictionStatusApplyConfiguration
	}{
		{
			name:                       "no requesters",
			expectedApplyConfiguration: lifecycleapply.EvictionStatus(),
		},
		{
			name: "keep existing requesters, and withdraw if the old ones are missing",
			existingRequesters: []lifecyclev1alpha1.Requester{
				{Name: "foo.example.com/2", Intent: lifecyclev1alpha1.RequesterIntentEviction},
				{Name: "foo.example.com/1", Intent: lifecyclev1alpha1.RequesterIntentWithdrawn},
			},
			limit: 100,
			expectedApplyConfiguration: lifecycleapply.EvictionStatus().WithRequesters(
				lifecycleapply.Requester().WithName("foo.example.com/1").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn),
				lifecycleapply.Requester().WithName("foo.example.com/2").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn),
			),
		},
		{
			name: "keep existing requesters, and withdraw if the old is deleted",
			existingRequesters: []lifecyclev1alpha1.Requester{
				{Name: "foo.example.com/2", Intent: lifecyclev1alpha1.RequesterIntentEviction},
				{Name: "foo.example.com/1", Intent: lifecyclev1alpha1.RequesterIntentWithdrawn},
			},
			requests: []*lifecyclev1alpha1.EvictionRequest{
				mkValidEvictionRequest("requester-2", "pod-2", setERRequester("foo.example.com/2"), setERDeletionTimestamp(new(metav1.Now()))),
			},
			limit: 100,
			expectedApplyConfiguration: lifecycleapply.EvictionStatus().WithRequesters(
				lifecycleapply.Requester().WithName("foo.example.com/2").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn),
				lifecycleapply.Requester().WithName("foo.example.com/1").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn),
			),
		},
		{
			name: "add and delete requesters and change intents",
			existingRequesters: []lifecyclev1alpha1.Requester{
				{Name: "foo.example.com/2", Intent: lifecyclev1alpha1.RequesterIntentEviction},
				{Name: "foo.example.com/3", Intent: lifecyclev1alpha1.RequesterIntentEviction},
				{Name: "foo.example.com/5", Intent: lifecyclev1alpha1.RequesterIntentEviction},
				{Name: "foo.example.com/1", Intent: lifecyclev1alpha1.RequesterIntentWithdrawn},
				{Name: "foo.example.com/4", Intent: lifecyclev1alpha1.RequesterIntentWithdrawn},
				{Name: "foo.example.com/6", Intent: lifecyclev1alpha1.RequesterIntentWithdrawn},
			},
			requests: []*lifecyclev1alpha1.EvictionRequest{
				mkValidEvictionRequest("requester-1", "pod-2", setERRequester("foo.example.com/1")),
				mkValidEvictionRequest("requester-3", "pod-3", setERRequester("foo.example.com/3")),
				mkValidEvictionRequest("requester-5", "pod-5", setERRequester("foo.example.com/5"), setERDeletionTimestamp(new(metav1.Now()))),
				mkValidEvictionRequest("requester-7", "pod-7", setERRequester("foo.example.com/7")),
				mkValidEvictionRequest("requester-6", "pod-6", setERRequester("foo.example.com/6"), setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("requester-8", "pod-8", setERRequester("foo.example.com/8"), setERDeletionTimestamp(new(metav1.Now()))),
			},
			limit: 100,
			expectedApplyConfiguration: lifecycleapply.EvictionStatus().WithRequesters(
				lifecycleapply.Requester().WithName("foo.example.com/1").WithIntent(lifecyclev1alpha1.RequesterIntentEviction), // evictions always first
				lifecycleapply.Requester().WithName("foo.example.com/3").WithIntent(lifecyclev1alpha1.RequesterIntentEviction),
				lifecycleapply.Requester().WithName("foo.example.com/7").WithIntent(lifecyclev1alpha1.RequesterIntentEviction),
				lifecycleapply.Requester().WithName("foo.example.com/6").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn), // non deleted
				lifecycleapply.Requester().WithName("foo.example.com/5").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn), // deletionTimestamp
				lifecycleapply.Requester().WithName("foo.example.com/8").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn),
				lifecycleapply.Requester().WithName("foo.example.com/2").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn), // moved to the end, since it has no EvictionRequest assigned
				lifecycleapply.Requester().WithName("foo.example.com/4").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn),
			),
		},
		{
			name: "add and delete requesters and change intents with limit",
			existingRequesters: []lifecyclev1alpha1.Requester{
				{Name: "foo.example.com/2", Intent: lifecyclev1alpha1.RequesterIntentEviction},
				{Name: "foo.example.com/3", Intent: lifecyclev1alpha1.RequesterIntentEviction},
				{Name: "foo.example.com/5", Intent: lifecyclev1alpha1.RequesterIntentEviction},
				{Name: "foo.example.com/1", Intent: lifecyclev1alpha1.RequesterIntentWithdrawn},
				{Name: "foo.example.com/4", Intent: lifecyclev1alpha1.RequesterIntentWithdrawn},
				{Name: "foo.example.com/6", Intent: lifecyclev1alpha1.RequesterIntentWithdrawn},
			},
			requests: []*lifecyclev1alpha1.EvictionRequest{
				mkValidEvictionRequest("requester-1", "pod-2", setERRequester("foo.example.com/1")),
				mkValidEvictionRequest("requester-3", "pod-3", setERRequester("foo.example.com/3")),
				mkValidEvictionRequest("requester-5", "pod-5", setERRequester("foo.example.com/5"), setERDeletionTimestamp(new(metav1.Now()))),
				mkValidEvictionRequest("requester-7", "pod-7", setERRequester("foo.example.com/7")),
				mkValidEvictionRequest("requester-6", "pod-6", setERRequester("foo.example.com/6"), setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("requester-8", "pod-8", setERRequester("foo.example.com/8"), setERDeletionTimestamp(new(metav1.Now()))),
			},
			limit: 5,
			expectedApplyConfiguration: lifecycleapply.EvictionStatus().WithRequesters(
				lifecycleapply.Requester().WithName("foo.example.com/1").WithIntent(lifecyclev1alpha1.RequesterIntentEviction), // evictions always first
				lifecycleapply.Requester().WithName("foo.example.com/3").WithIntent(lifecyclev1alpha1.RequesterIntentEviction),
				lifecycleapply.Requester().WithName("foo.example.com/7").WithIntent(lifecyclev1alpha1.RequesterIntentEviction),
				lifecycleapply.Requester().WithName("foo.example.com/6").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn), // non deleted
				lifecycleapply.Requester().WithName("foo.example.com/5").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn), // deletionTimestamp
			),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			eviction := mkValidEviction("foo", "foo", "foo-1")
			eviction.Status.Requesters = tc.existingRequesters
			result := lifecycleapply.EvictionStatus()
			updateRequestersForEvictionStatusApply(eviction, tc.requests, tc.limit, result)
			if diff := cmp.Diff(tc.expectedApplyConfiguration, result); len(diff) > 0 {
				t.Fatalf("unexpected ApplyConfiguration returned %s", diff)
			}
		})
	}
}
