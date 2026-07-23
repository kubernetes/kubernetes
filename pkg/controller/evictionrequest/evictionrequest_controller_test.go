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
)

// TesValidate tests early validation
func TestValidate(t *testing.T) {
	now := time.Now()
	clock := testingclock.NewFakePassiveClock(now)
	oldClock := testingclock.NewFakePassiveClock(now.Add(-time.Minute))
	tests := []struct {
		name                string
		hasEvictions        bool
		target              lifecyclev1alpha1.EvictionTarget
		pod                 *v1.Pod
		conditions          []metav1.Condition
		isDuplicate         bool
		testEviction        bool
		testEvictionRequest bool
		expected            []metav1ac.ConditionApplyConfiguration
		expectChanged       bool
	}{
		{
			name:                "valid pod",
			target:              mkValidPodTarget("my-pod", "uid-1"),
			pod:                 mkValidPod("my-pod", "uid-1"),
			expected:            nil,
			testEvictionRequest: true,
		},

		{
			name:                "pod not found after eviction",
			hasEvictions:        true,
			target:              mkValidPodTarget("my-pod", "uid-1"),
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
			name:   "pod not found",
			target: mkValidPodTarget("my-pod", "uid-1"),
			pod:    nil,
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid,
					"Target pod not found."),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			expectChanged:       true,
			testEvictionRequest: true,
		},
		{
			name:   "UID mismatch",
			target: mkValidPodTarget("my-pod", "uid-1"),
			pod:    mkValidPod("my-pod", "uid-2"),
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid,
					"Target pod UID mismatch: expected uid-1, got uid-2."),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			expectChanged:       true,
			testEvictionRequest: true,
		},
		{
			name:   "pod with PodGroup",
			target: mkValidPodTarget("my-pod", "uid-1"),
			pod: func() *v1.Pod {
				pod := mkValidPod("my-pod", "uid-1")
				pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: new("my-podgroup")}
				return pod
			}(),
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid,
					"Target pod references a SchedulingGroup. Eviction is currently not supported."),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			expectChanged:       true,
			testEviction:        true,
			testEvictionRequest: true,
		},
		{
			name:        "duplicate eviction",
			target:      mkValidPodTarget("my-pod", "uid-1"),
			pod:         mkValidPod("my-pod", "uid-1"),
			isDuplicate: true,
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid,
					"Active Eviction already exists for the same target."),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			expectChanged: true,
			testEviction:  true,
		},
		{
			name:   "empty target",
			target: lifecyclev1alpha1.EvictionTarget{},
			pod:    nil,
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid,
					"Unsupported target type."),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			expectChanged:       true,
			testEviction:        true,
			testEvictionRequest: true,
		},
		{
			name:   "empty target - condition already set",
			target: lifecyclev1alpha1.EvictionTarget{},
			pod:    nil,
			conditions: []metav1.Condition{
				{
					Type:    string(lifecyclev1alpha1.EvictionConditionTargetEvicted),
					Status:  metav1.ConditionFalse,
					Reason:  string(lifecyclev1alpha1.EvictionConditionReasonEvictionFailed),
					Message: "",
				},
				{
					Type:    string(lifecyclev1alpha1.EvictionConditionFailed),
					Status:  metav1.ConditionTrue,
					Reason:  string(lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid),
					Message: "Unsupported target type.",
				},
			},
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid,
					"Unsupported target type."),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			testEviction:        true,
			testEvictionRequest: true,
		},
		{
			name:   "empty target - updates existing condition",
			target: lifecyclev1alpha1.EvictionTarget{},
			pod:    nil,
			conditions: []metav1.Condition{
				{
					Type:               string(lifecyclev1alpha1.EvictionConditionFailed),
					Status:             metav1.ConditionFalse,
					Reason:             string(lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid),
					Message:            "Unsupported target type.",
					LastTransitionTime: metav1.Time{Time: oldClock.Now()},
				},
			},
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid,
					"Unsupported target type."),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			expectChanged:       true,
			testEviction:        true,
			testEvictionRequest: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// always test EvictionRequest, Eviction has only a subset of validations
			if tc.testEviction {
				eviction := &lifecyclev1alpha1.Eviction{
					Status: lifecyclev1alpha1.EvictionStatus{
						Conditions: tc.conditions,
					},
				}
				failed, evicted, changed := validateEviction(clock.Now(), eviction, newTargetInfoForEviction(tc.target, tc.pod), tc.isDuplicate)
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
				if tc.expectChanged != changed {
					t.Errorf("got changed %v, want %v", changed, tc.expectChanged)
				}
			}
			if tc.testEvictionRequest {
				evictionRequest := &lifecyclev1alpha1.EvictionRequest{
					Status: lifecyclev1alpha1.EvictionRequestStatus{
						Conditions: tc.conditions,
					},
				}
				failed, evicted, changed := validateEvictionRequest(clock.Now(), evictionRequest,
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
				if tc.expectChanged != changed {
					t.Errorf("got changed %v, want %v", changed, tc.expectChanged)
				}
			}
		})
	}
}

func TestComputeEvictionConditions(t *testing.T) {
	now := time.Now()
	clock := testingclock.NewFakePassiveClock(now)
	oldClock := testingclock.NewFakePassiveClock(now.Add(-time.Minute))
	tests := []struct {
		name                        string
		conditions                  []metav1.Condition
		isWaitingForResponderUpdate bool
		isGone                      bool
		isTerminal                  bool
		isCanceled                  bool
		isProgressionDone           bool
		expected                    []metav1ac.ConditionApplyConfiguration
		expectChanged               bool
	}{
		{
			name:   "pod deleted",
			isGone: true,
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonSucceeded,
					""),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonPodDeleted, "Target pod has been deleted."),
			},
			expectChanged: true,
		},
		{
			name:                        "pod deleted - waiting",
			isGone:                      true,
			isWaitingForResponderUpdate: true,
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction,
					""),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction, ""),
			},
			expectChanged: true,
		},

		{
			name:       "pod terminal",
			isTerminal: true,
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonSucceeded, ""),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonPodTerminal, "Pod has reached terminal state."),
			},
			expectChanged: true,
		},
		{
			name:                        "pod terminal - waiting",
			isTerminal:                  true,
			isWaitingForResponderUpdate: true,
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction, ""),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction, ""),
			},
			expectChanged: true,
		},
		{
			name:       "is canceled",
			isCanceled: true,
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonCanceledDueToNoRequesters, "No active requesters with eviction intent."),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			expectChanged: true,
		},
		{
			name:              "no progress",
			isProgressionDone: true,
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder, "All responders have completed without evicting the target."),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			expectChanged: true,
		},
		{
			name: "pending",
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction, ""),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction, ""),
			},
			expectChanged: true,
		},
		{
			name:              "no progress - condition already set",
			isProgressionDone: true,
			conditions: []metav1.Condition{
				{
					Type:    string(lifecyclev1alpha1.EvictionConditionTargetEvicted),
					Status:  metav1.ConditionFalse,
					Reason:  string(lifecyclev1alpha1.EvictionConditionReasonEvictionFailed),
					Message: "",
				},
				{
					Type:    string(lifecyclev1alpha1.EvictionConditionFailed),
					Status:  metav1.ConditionTrue,
					Reason:  string(lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder),
					Message: "All responders have completed without evicting the target.",
				},
			},
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder, "All responders have completed without evicting the target."),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
		},

		{
			name:              "no progress - updates existing condition",
			isProgressionDone: true,
			conditions: []metav1.Condition{
				{
					Type:               string(lifecyclev1alpha1.EvictionConditionFailed),
					Status:             metav1.ConditionFalse,
					Reason:             string(lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction),
					Message:            "",
					LastTransitionTime: metav1.Time{Time: oldClock.Now()},
				},
			},
			expected: []metav1ac.ConditionApplyConfiguration{
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionFailed,
					metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder, "All responders have completed without evicting the target."),
				*conditionApplyConf(clock.Now(), lifecyclev1alpha1.EvictionConditionTargetEvicted,
					metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, ""),
			},
			expectChanged: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			eviction := &lifecyclev1alpha1.Eviction{
				Status: lifecyclev1alpha1.EvictionStatus{
					Conditions: tc.conditions,
				},
			}
			failed, evicted, changed := computeEvictionConditions(clock.Now(), eviction,
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
			if tc.expectChanged != changed {
				t.Errorf("got changed %v, want %v", changed, tc.expectChanged)
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
		expectChanged              bool
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
			expectChanged: true,
		},
		{
			name: "keep existing requesters, and withdraw if the old is deleted",
			existingRequesters: []lifecyclev1alpha1.Requester{
				{Name: "foo.example.com/2", Intent: lifecyclev1alpha1.RequesterIntentEviction},
				{Name: "foo.example.com/1", Intent: lifecyclev1alpha1.RequesterIntentWithdrawn},
			},
			requests: []*lifecyclev1alpha1.EvictionRequest{
				mkValidEvictionRequest("requester-2", "pod-2", setERRequester("foo.example.com/2"),
					setERDeletionTimestamp(new(metav1.Now()))),
			},
			limit: 100,
			expectedApplyConfiguration: lifecycleapply.EvictionStatus().WithRequesters(
				lifecycleapply.Requester().WithName("foo.example.com/2").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn),
				lifecycleapply.Requester().WithName("foo.example.com/1").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn),
			),
			expectChanged: true,
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
				mkValidEvictionRequest("requester-1", "pod-1", setERRequester("foo.example.com/1")),
				mkValidEvictionRequest("requester-3", "pod-3", setERRequester("foo.example.com/3")),
				mkValidEvictionRequest("requester-5", "pod-5", setERRequester("foo.example.com/5"),
					setERDeletionTimestamp(new(metav1.Now()))),
				mkValidEvictionRequest("requester-7", "pod-7", setERRequester("foo.example.com/7")),
				mkValidEvictionRequest("requester-6", "pod-6", setERRequester("foo.example.com/6"),
					setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("requester-8", "pod-8", setERRequester("foo.example.com/8"),
					setERDeletionTimestamp(new(metav1.Now()))),
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
			expectChanged: true,
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
				mkValidEvictionRequest("requester-1", "pod-1", setERRequester("foo.example.com/1")),
				mkValidEvictionRequest("requester-3", "pod-3", setERRequester("foo.example.com/3")),
				mkValidEvictionRequest("requester-5", "pod-5", setERRequester("foo.example.com/5"),
					setERDeletionTimestamp(new(metav1.Now()))),
				mkValidEvictionRequest("requester-7", "pod-7", setERRequester("foo.example.com/7")),
				mkValidEvictionRequest("requester-6", "pod-6", setERRequester("foo.example.com/6"),
					setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("requester-8", "pod-8", setERRequester("foo.example.com/8"),
					setERDeletionTimestamp(new(metav1.Now()))),
			},
			limit: 5,
			expectedApplyConfiguration: lifecycleapply.EvictionStatus().WithRequesters(
				lifecycleapply.Requester().WithName("foo.example.com/1").WithIntent(lifecyclev1alpha1.RequesterIntentEviction), // evictions always first
				lifecycleapply.Requester().WithName("foo.example.com/3").WithIntent(lifecyclev1alpha1.RequesterIntentEviction),
				lifecycleapply.Requester().WithName("foo.example.com/7").WithIntent(lifecyclev1alpha1.RequesterIntentEviction),
				lifecycleapply.Requester().WithName("foo.example.com/6").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn), // non deleted
				lifecycleapply.Requester().WithName("foo.example.com/5").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn), // deletionTimestamp
			),
			expectChanged: true,
		},
		{
			name: "no changes in requesters",
			existingRequesters: []lifecyclev1alpha1.Requester{
				{Name: "foo.example.com/2", Intent: lifecyclev1alpha1.RequesterIntentEviction},
				{Name: "foo.example.com/1", Intent: lifecyclev1alpha1.RequesterIntentWithdrawn},
			},
			requests: []*lifecyclev1alpha1.EvictionRequest{
				mkValidEvictionRequest("requester-2", "pod-2", setERRequester("foo.example.com/2")),
			},
			limit: 100,
			expectedApplyConfiguration: lifecycleapply.EvictionStatus().WithRequesters(
				lifecycleapply.Requester().WithName("foo.example.com/2").WithIntent(lifecyclev1alpha1.RequesterIntentEviction),
				lifecycleapply.Requester().WithName("foo.example.com/1").WithIntent(lifecyclev1alpha1.RequesterIntentWithdrawn),
			),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			eviction := mkValidEviction("foo", "foo", "foo-1")
			eviction.Status.Requesters = tc.existingRequesters
			result := lifecycleapply.EvictionStatus()
			changed := updateRequestersForEvictionStatusApply(eviction, tc.requests, tc.limit, result)
			if diff := cmp.Diff(tc.expectedApplyConfiguration, result); len(diff) > 0 {
				t.Fatalf("unexpected ApplyConfiguration (-want +got):\n%s", diff)
			}
			if tc.expectChanged != changed {
				t.Errorf("got changed %v, want %v", changed, tc.expectChanged)
			}
		})
	}
}

func conditionApplyConf(
	now time.Time,
	conditionType lifecyclev1alpha1.EvictionConditionType,
	status metav1.ConditionStatus,
	reason lifecyclev1alpha1.EvictionConditionReason,
	message string,
) *metav1ac.ConditionApplyConfiguration {
	return metav1ac.Condition().
		WithType(string(conditionType)).
		WithStatus(status).
		WithReason(string(reason)).
		WithMessage(message).
		WithLastTransitionTime(metav1.Time{Time: now})
}
