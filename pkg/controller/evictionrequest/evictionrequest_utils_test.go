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
	"math/rand"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	apimachinerytypes "k8s.io/apimachinery/pkg/types"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	"k8s.io/utils/clock"
	testing2 "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

func TestNewEvictionName(t *testing.T) {
	tests := []struct {
		name         string
		target       targetInfo
		evictions    []*lifecyclev1alpha1.Eviction
		expectedName string
	}{
		{
			name:         "empty target",
			expectedName: "",
		},
		{
			name:         "picks first name",
			target:       newTargetInfoForEviction(mkValidPodTarget("foo", "uid-1"), nil),
			expectedName: "pod-1-foo",
		},
		{
			name:   "picks the next name",
			target: newTargetInfoForEviction(mkValidPodTarget("foo", "uid-1"), nil),
			evictions: []*lifecyclev1alpha1.Eviction{
				mkValidEviction("pod-0-foo", "foo", "uid-0"),
				mkValidEviction("pod-1-foo", "foo", "uid-1"),
				mkValidEviction("pod-3-foo", "foo", "uid-3"),
				mkValidEviction("pod-4-foo", "foo", "uid-4"),
				mkValidEviction("pod-7-foo", "foo", "uid-7"),
			},
			expectedName: "pod-8-foo",
		},
		{
			name:         "too long name",
			target:       newTargetInfoForEviction(mkValidPodTarget(strings.Repeat("a", 253), "uid-1"), nil),
			expectedName: "pod-1-" + strings.Repeat("a", 247),
		},
		{
			name:   "too long name with eviction",
			target: newTargetInfoForEviction(mkValidPodTarget(strings.Repeat("a", 252)+"b", "uid-2"), nil),
			evictions: []*lifecyclev1alpha1.Eviction{
				mkValidEviction("pod-1-"+strings.Repeat("a", 247), strings.Repeat("a", 252)+"c", "uid-1"),
			},
			expectedName: "pod-2-" + strings.Repeat("a", 247),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := newEvictionName(tc.target, tc.evictions); got != tc.expectedName {
				t.Errorf("got newEvictionName %v, expected %v", got, tc.expectedName)
			}
		})
	}
}

func TestFindRelevantEviction(t *testing.T) {
	now := time.Now()
	clock := testing2.NewFakeClock(now)
	clockOlder := testing2.NewFakeClock(now.Add(-time.Hour))
	clockOldest := testing2.NewFakeClock(now.Add(-time.Hour - time.Second))
	tests := []struct {
		name                 string
		evictions            []*lifecyclev1alpha1.Eviction
		expectedEviction     *lifecyclev1alpha1.Eviction
		expectedShouldCreate bool
	}{
		{
			name:                 "no evictions",
			expectedEviction:     nil,
			expectedShouldCreate: true,
		},
		{
			name: "latest succeeded eviction",
			evictions: []*lifecyclev1alpha1.Eviction{
				mkValidEviction("pod-1-foo", "foo", "uid-1"),
				mkValidEviction("pod-2-foo", "foo", "uid-1", setCreationTimestamp(clockOlder)),
				mkValidEviction("pod-3-foo", "foo", "uid-1",
					addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder)),
				mkValidEviction("pod-4-foo", "foo", "uid-1"),
				mkValidEviction("pod-5-foo", "foo", "uid-1", setCreationTimestamp(clockOldest)),
				mkValidEviction("pod-6-foo", "foo", "uid-1",
					addConditionTrue(clockOldest, lifecyclev1alpha1.EvictionConditionTargetEvicted, lifecyclev1alpha1.EvictionConditionReasonPodDeleted)),
				mkValidEviction("pod-7-foo", "foo", "uid-1",
					addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionTargetEvicted, lifecyclev1alpha1.EvictionConditionReasonPodDeleted)),
				mkValidEviction("pod-8-foo", "foo", "uid-1"),
			},
			expectedEviction: mkValidEviction("pod-7-foo", "foo", "uid-1",
				addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionTargetEvicted, lifecyclev1alpha1.EvictionConditionReasonPodDeleted)),
			expectedShouldCreate: false,
		},
		{
			name: "oldest active eviction",
			evictions: []*lifecyclev1alpha1.Eviction{
				mkValidEviction("pod-1-foo", "foo", "uid-1", setCreationTimestamp(clockOlder)),
				mkValidEviction("pod-2-foo", "foo", "uid-1",
					addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder)),
				mkValidEviction("pod-3-foo", "foo", "uid-1", setCreationTimestamp(clockOldest)),
			},
			expectedEviction:     mkValidEviction("pod-3-foo", "foo", "uid-1", setCreationTimestamp(clockOldest)),
			expectedShouldCreate: false,
		},
		{
			name: "oldest active eviction with conditions",
			evictions: []*lifecyclev1alpha1.Eviction{
				mkValidEviction("pod-1-foo", "foo", "uid-1", setCreationTimestamp(clockOlder)),
				mkValidEviction("pod-2-foo", "foo", "uid-1", setCreationTimestamp(clockOldest),
					addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder)),
				mkValidEviction("pod-3-foo", "foo", "uid-1", setCreationTimestamp(clockOldest),
					addCondition(clockOldest, lifecyclev1alpha1.EvictionConditionTargetEvicted, metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction),
					addCondition(clockOldest, lifecyclev1alpha1.EvictionConditionFailed, metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction)),
				mkValidEviction("pod-4-foo", "foo", "uid-1", setCreationTimestamp(clockOlder)),
			},
			expectedEviction: mkValidEviction("pod-3-foo", "foo", "uid-1", setCreationTimestamp(clockOldest),
				addCondition(clockOldest, lifecyclev1alpha1.EvictionConditionTargetEvicted, metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction),
				addCondition(clockOldest, lifecyclev1alpha1.EvictionConditionFailed, metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction)),
			expectedShouldCreate: false,
		},
		{
			name: "latest failed eviction",
			evictions: []*lifecyclev1alpha1.Eviction{
				mkValidEviction("pod-1-foo", "foo", "uid-1", addConditionTrue(clockOldest, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder)),
				mkValidEviction("pod-2-foo", "foo", "uid-1", addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder)),
			},
			expectedEviction:     mkValidEviction("pod-2-foo", "foo", "uid-1", addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder)),
			expectedShouldCreate: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rand.Shuffle(len(tc.evictions), func(i, j int) {
				tc.evictions[i], tc.evictions[j] = tc.evictions[j], tc.evictions[i]
			})
			// also tests compareConditionTransitionTime
			relevantEviction, shouldCreate := findRelevantEviction(tc.evictions)
			if diff := cmp.Diff(tc.expectedEviction, relevantEviction); len(diff) > 0 {
				t.Fatalf("unexpected relevant eviction: %s", diff)
			}
			if tc.expectedShouldCreate != shouldCreate {
				t.Errorf("got shouldCreate %v, expected %v", shouldCreate, tc.expectedShouldCreate)
			}
		})
	}
}

func TestConvertToEvictionRequestConditions(t *testing.T) {
	now := time.Now()
	clock := testing2.NewFakeClock(now)
	clockOld := testing2.NewFakeClock(now.Add(-time.Minute))
	testCases := []struct {
		name                       string
		existingConditions         []metav1.Condition
		evictionConditions         []metav1.Condition
		conditionType              lifecyclev1alpha1.EvictionConditionType
		expectedApplyConfiguration *metav1ac.ConditionApplyConfiguration
	}{
		{
			name:               "find and set correct condition",
			existingConditions: nil,
			evictionConditions: []metav1.Condition{
				{Type: "ResponderCondition", Status: metav1.ConditionTrue,
					Reason: "foo", Message: "msg2"},
				{Type: string(lifecyclev1alpha1.EvictionConditionTargetEvicted), Status: metav1.ConditionTrue,
					Reason: string(lifecyclev1alpha1.EvictionConditionReasonPodDeleted), Message: "msg"},
				{Type: string(lifecyclev1alpha1.EvictionConditionFailed), Status: metav1.ConditionFalse,
					Reason: string(lifecyclev1alpha1.EvictionConditionReasonSucceeded), Message: "msg2"},
			},
			conditionType: lifecyclev1alpha1.EvictionConditionTargetEvicted,
			expectedApplyConfiguration: metav1ac.Condition().
				WithType(string(lifecyclev1alpha1.EvictionConditionTargetEvicted)).
				WithStatus(metav1.ConditionTrue).
				WithReason(string(lifecyclev1alpha1.EvictionConditionReasonPodDeleted)).
				WithMessage("msg").
				WithLastTransitionTime(metav1.Time{Time: clock.Now()}),
		},
		{
			name: "find and update correct condition",
			existingConditions: []metav1.Condition{
				{Type: string(lifecyclev1alpha1.EvictionConditionFailed), Status: metav1.ConditionFalse,
					Reason: string(lifecyclev1alpha1.EvictionConditionReasonSucceeded), Message: "msg2"},
				{Type: string(lifecyclev1alpha1.EvictionConditionTargetEvicted), Status: metav1.ConditionFalse,
					Reason: string(lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction), Message: "awaiting", LastTransitionTime: metav1.Time{Time: clockOld.Now()}},
			},
			evictionConditions: []metav1.Condition{
				{Type: "ResponderCondition", Status: metav1.ConditionTrue,
					Reason: "foo", Message: "msg2"},
				{Type: string(lifecyclev1alpha1.EvictionConditionTargetEvicted), Status: metav1.ConditionTrue,
					Reason: string(lifecyclev1alpha1.EvictionConditionReasonPodDeleted), Message: "msg"},
				{Type: string(lifecyclev1alpha1.EvictionConditionFailed), Status: metav1.ConditionFalse,
					Reason: string(lifecyclev1alpha1.EvictionConditionReasonSucceeded), Message: "msg2"},
			},
			conditionType: lifecyclev1alpha1.EvictionConditionTargetEvicted,
			expectedApplyConfiguration: metav1ac.Condition().
				WithType(string(lifecyclev1alpha1.EvictionConditionTargetEvicted)).
				WithStatus(metav1.ConditionTrue).
				WithReason(string(lifecyclev1alpha1.EvictionConditionReasonPodDeleted)).
				WithMessage("msg").
				WithLastTransitionTime(metav1.Time{Time: clock.Now()}),
		},
		{
			name: "find and update correct condition without transition",
			existingConditions: []metav1.Condition{
				{Type: string(lifecyclev1alpha1.EvictionConditionFailed), Status: metav1.ConditionFalse,
					Reason: string(lifecyclev1alpha1.EvictionConditionReasonSucceeded), Message: "msg2"},
				{Type: string(lifecyclev1alpha1.EvictionConditionTargetEvicted), Status: metav1.ConditionTrue,
					Reason: string(lifecyclev1alpha1.EvictionConditionReasonPodDeleted), Message: "ms3", LastTransitionTime: metav1.Time{Time: clockOld.Now()}},
			},
			evictionConditions: []metav1.Condition{
				{Type: "ResponderCondition", Status: metav1.ConditionTrue,
					Reason: "foo", Message: "msg2"},
				{Type: string(lifecyclev1alpha1.EvictionConditionTargetEvicted), Status: metav1.ConditionTrue,
					Reason: string(lifecyclev1alpha1.EvictionConditionReasonPodDeleted), Message: "msg"},
				{Type: string(lifecyclev1alpha1.EvictionConditionFailed), Status: metav1.ConditionFalse,
					Reason: string(lifecyclev1alpha1.EvictionConditionReasonSucceeded), Message: "msg2"},
			},
			conditionType: lifecyclev1alpha1.EvictionConditionTargetEvicted,
			expectedApplyConfiguration: metav1ac.Condition().
				WithType(string(lifecyclev1alpha1.EvictionConditionTargetEvicted)).
				WithStatus(metav1.ConditionTrue).
				WithReason(string(lifecyclev1alpha1.EvictionConditionReasonPodDeleted)).
				WithMessage("msg").
				WithLastTransitionTime(metav1.Time{Time: clockOld.Now()}),
		},
		{
			name:               "cannot find the correct condition",
			existingConditions: nil,
			evictionConditions: []metav1.Condition{
				{Type: "ResponderCondition", Status: metav1.ConditionTrue,
					Reason: "foo", Message: "msg2"},
				{Type: string(lifecyclev1alpha1.EvictionConditionFailed), Status: metav1.ConditionFalse,
					Reason: string(lifecyclev1alpha1.EvictionConditionReasonSucceeded), Message: "msg2"},
			},
			conditionType: lifecyclev1alpha1.EvictionConditionTargetEvicted,
			expectedApplyConfiguration: metav1ac.Condition().
				WithType(string(lifecyclev1alpha1.EvictionConditionTargetEvicted)).
				WithStatus(metav1.ConditionFalse).
				WithReason(string(lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction)).
				WithMessage("waiting for an Eviction to report on progress").
				WithLastTransitionTime(metav1.Time{Time: clock.Now()}),
		},

		{
			name:               "cannot find the correct condition: no eviction",
			existingConditions: nil,
			evictionConditions: nil,
			conditionType:      lifecyclev1alpha1.EvictionConditionTargetEvicted,
			expectedApplyConfiguration: metav1ac.Condition().
				WithType(string(lifecyclev1alpha1.EvictionConditionTargetEvicted)).
				WithStatus(metav1.ConditionFalse).
				WithReason(string(lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction)).
				WithMessage("waiting for an Eviction to report on progress").
				WithLastTransitionTime(metav1.Time{Time: clock.Now()}),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			evictionRequest := &lifecyclev1alpha1.EvictionRequest{}
			evictionRequest.Status.Conditions = tc.existingConditions
			var eviction *lifecyclev1alpha1.Eviction
			if len(tc.evictionConditions) > 0 {
				eviction = &lifecyclev1alpha1.Eviction{}
				eviction.Status.Conditions = tc.evictionConditions
			}
			// also tests setCondition
			got := convertToEvictionRequestConditions(clock.Now(), evictionRequest, eviction, tc.conditionType)
			if diff := cmp.Diff(tc.expectedApplyConfiguration, got); len(diff) > 0 {
				t.Fatalf("unexpected ApplyConfiguration returned %s", diff)
			}
		})
	}
}

func TestCompareConditionTransitionTime(t *testing.T) {
	now := time.Now()
	clock := testing2.NewFakeClock(now)
	clockLater := testing2.NewFakeClock(now.Add(time.Hour))
	tests := []struct {
		name        string
		evictionA   *lifecyclev1alpha1.Eviction
		evictionB   *lifecyclev1alpha1.Eviction
		expectedCMP int
	}{
		{
			name:        "no conditions",
			evictionA:   mkValidEviction("pod-1-foo", "foo", "uid-1"),
			evictionB:   mkValidEviction("pod-2-foo", "foo", "uid-1"),
			expectedCMP: 0,
		},
		{
			name: "same transition time",
			evictionA: mkValidEviction("pod-1-foo", "foo", "uid-1",
				addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder),
			),
			evictionB: mkValidEviction("pod-2-foo", "foo", "uid-1",
				addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder),
			),
			expectedCMP: 0,
		},
		{
			name:      "no a condition",
			evictionA: mkValidEviction("pod-1-foo", "foo", "uid-1"),
			evictionB: mkValidEviction("pod-2-foo", "foo", "uid-1",
				addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder),
			),
			expectedCMP: -1,
		},
		{
			name: "no b condition",
			evictionA: mkValidEviction("pod-1-foo", "foo", "uid-1",
				addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder),
			),
			evictionB:   mkValidEviction("pod-2-foo", "foo", "uid-1"),
			expectedCMP: 1,
		},

		{
			name: "a is newest",
			evictionA: mkValidEviction("pod-1-foo", "foo", "uid-1",
				addConditionTrue(clockLater, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder),
			),
			evictionB: mkValidEviction("pod-2-foo", "foo", "uid-1",
				addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder),
			),
			expectedCMP: 1,
		},
		{
			name: "b is newest",
			evictionA: mkValidEviction("pod-1-foo", "foo", "uid-1",
				addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder),
			),
			evictionB: mkValidEviction("pod-2-foo", "foo", "uid-1",
				addConditionTrue(clockLater, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder),
			),
			expectedCMP: -1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if cmp := compareConditionTransitionTime(tc.evictionA, tc.evictionB, lifecyclev1alpha1.EvictionConditionFailed); tc.expectedCMP != cmp {
				t.Errorf("got cmp %v, expected %v", cmp, tc.expectedCMP)
			}
		})
	}
}

func TestEvictionCondition(t *testing.T) {
	now := time.Now()
	clock := testing2.NewFakeClock(now)
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
			expectSucceeded: false,
			expectFailed:    true,
		},
		{
			name: "succeeded",
			eviction: mkValidEviction("pod-1-foo", "foo", "uid-1",
				addConditionTrue(clock, lifecyclev1alpha1.EvictionConditionFailed, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder),
			),
			expectSucceeded: false,
			expectFailed:    true,
		},

		{
			name: "in progress",
			eviction: mkValidEviction("pod-1-foo", "foo", "uid-1",
				addCondition(clock, lifecyclev1alpha1.EvictionConditionFailed, metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder),
			),
			expectSucceeded: false,
			expectFailed:    false,
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

func TestHasEvictionRequestCompleted(t *testing.T) {
	now := time.Now()
	clock := testing2.NewFakeClock(now)
	tests := []struct {
		name             string
		evictionRequest  *lifecyclev1alpha1.EvictionRequest
		expecteCompleted bool
	}{
		{
			name: "completed - target evicted",
			evictionRequest: mkValidEvictionRequest("foo", "uid-1",
				addERCondition(clock, lifecyclev1alpha1.EvictionConditionTargetEvicted, metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonPodDeleted),
			),
			expecteCompleted: true,
		},
		{
			name: "completed - invalid",
			evictionRequest: mkValidEvictionRequest("foo", "uid-1",
				addERCondition(clock, lifecyclev1alpha1.EvictionConditionFailed, metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid),
			),
			expecteCompleted: true,
		},
		{
			name: "in progress - with last failed eviction",
			evictionRequest: mkValidEvictionRequest("foo", "uid-1",
				addERCondition(clock, lifecyclev1alpha1.EvictionConditionFailed, metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder),
			),
			expecteCompleted: false,
		},
		{
			name: "in progress",
			evictionRequest: mkValidEvictionRequest("foo", "uid-1",
				addERCondition(clock, lifecyclev1alpha1.EvictionConditionTargetEvicted, metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction),
				addERCondition(clock, lifecyclev1alpha1.EvictionConditionFailed, metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction),
			),
			expecteCompleted: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := hasEvictionRequestCompleted(tc.evictionRequest); tc.expecteCompleted != got {
				t.Errorf("got eviction request completed %v, expected %v", got, tc.expecteCompleted)
			}
		})
	}
}

func TestHasEvictionRequestEvictionIntent(t *testing.T) {
	tests := []struct {
		name                   string
		eviction               []*lifecyclev1alpha1.EvictionRequest
		expectedEvictionIntent bool
	}{
		{
			name: "multiple intents",
			eviction: []*lifecyclev1alpha1.EvictionRequest{
				mkValidEvictionRequest("foo1", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentEviction)),
				mkValidEvictionRequest("foo2", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentEviction)),
				mkValidEvictionRequest("foo3", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentEviction)),
			},
			expectedEvictionIntent: true,
		},
		{
			name: "single intent",
			eviction: []*lifecyclev1alpha1.EvictionRequest{
				mkValidEvictionRequest("foo1", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("foo2", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("foo3", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("foo4", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentEviction)),
				mkValidEvictionRequest("foo5", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("foo6", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
			},
			expectedEvictionIntent: true,
		},
		{
			name: "withdrawn",
			eviction: []*lifecyclev1alpha1.EvictionRequest{
				mkValidEvictionRequest("foo1", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("foo2", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("foo3", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("foo4", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentEviction), setERDeletionTimestamp(new(metav1.Now()))),
				mkValidEvictionRequest("foo5", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
				mkValidEvictionRequest("foo6", "uid-1", setERIntent(lifecyclev1alpha1.EvictionRequestIntentWithdrawn)),
			},
			expectedEvictionIntent: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := hasEvictionIntent(tc.eviction); tc.expectedEvictionIntent != got {
				t.Errorf("got eviction request completed %v, expected %v", got, tc.expectedEvictionIntent)
			}
		})
	}
}

func TestEvictionRequestAsOwnerReference(t *testing.T) {
	evictionRequest := mkValidEvictionRequest("foo1", "uid-1")
	expectedOwnerRef := metav1ac.OwnerReference().
		WithKind("EvictionRequest").
		WithAPIVersion("lifecycle.k8s.io/v1alpha1").
		WithName(evictionRequest.Name).
		WithUID(evictionRequest.UID)
	gotOwnerRef := evictionRequestAsOwnerReference(evictionRequest)
	if diff := cmp.Diff(expectedOwnerRef, gotOwnerRef); len(diff) > 0 {
		t.Errorf("eviction request ownerRef expected %v, got %v, diff %v", expectedOwnerRef, gotOwnerRef, diff)
	}
}

func TestEvictionLabelsNeedSSAUpdate(t *testing.T) {
	tests := []struct {
		name        string
		oldLabels   map[string]string
		newLabels   map[string]string
		needsUpdate bool
	}{
		{
			name:      "new label",
			oldLabels: nil,
			newLabels: map[string]string{
				"foo": "bar",
			},
			needsUpdate: true,
		},
		{
			name: "label update",
			oldLabels: map[string]string{
				"foo": "bar",
			},
			newLabels: map[string]string{
				"foo": "baz",
			},
			needsUpdate: true,
		},

		{
			name: "removed requester update",
			oldLabels: map[string]string{
				"baz":                 "bar",
				"foo.example.com/bar": string(lifecyclev1alpha1.EvictionParticipantRoleRequester),
			},
			newLabels: map[string]string{
				"baz": "bar",
			},
			needsUpdate: true,
		},
		{
			name: "removal of other values not registered requester update",
			oldLabels: map[string]string{
				"baz":                 "bar",
				"foo.example.com/bar": string(lifecyclev1alpha1.EvictionParticipantRoleResponder),
			},
			newLabels:   nil,
			needsUpdate: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := evictionLabelsNeedSSAUpdate(tc.oldLabels, tc.newLabels); tc.needsUpdate != got {
				t.Errorf("got eviction request needsUpdate %v, expected %v", got, tc.needsUpdate)
			}
		})
	}
}

func TestShouldDeferCompletion(t *testing.T) {
	now := time.Now()

	tests := []struct {
		name                  string
		activeResponderStatus *lifecyclev1alpha1.ResponderStatus
		target                targetInfo
		expectDefer           *time.Duration
	}{
		{
			name: "no responder status",
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"),
				mkValidPod("my-pod", "uid-1", setPodDeletionTimestamp(&metav1.Time{Time: now.Add(-time.Second)})),
			),
			expectDefer: nil,
		},
		{
			name:                  "responder with completion time",
			activeResponderStatus: &lifecyclev1alpha1.ResponderStatus{CompletionTime: &metav1.Time{Time: now}},
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"),
				mkValidPod("my-pod", "uid-1", setPodDeletionTimestamp(&metav1.Time{Time: now.Add(-time.Second)})),
			),
			expectDefer: nil,
		},
		{
			name:                  "no pod",
			activeResponderStatus: &lifecyclev1alpha1.ResponderStatus{},
			target:                newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"), nil),
			expectDefer:           nil,
		},
		{
			name:                  "no deletion timestamp",
			activeResponderStatus: &lifecyclev1alpha1.ResponderStatus{},
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"),
				mkValidPod("my-pod", "uid-1"),
			),
			expectDefer: nil,
		},
		{
			name:                  "planned deletion should defer",
			activeResponderStatus: &lifecyclev1alpha1.ResponderStatus{},
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"),
				mkValidPod("my-pod", "uid-1", setPodDeletionTimestamp(&metav1.Time{Time: now.Add(time.Minute)})),
			),
			expectDefer: new(time.Minute + 5*time.Second),
		},
		{
			name:                  "recent deletion should defer",
			activeResponderStatus: &lifecyclev1alpha1.ResponderStatus{},
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"),
				mkValidPod("my-pod", "uid-1", setPodDeletionTimestamp(&metav1.Time{Time: now.Add(-time.Second)})),
			),
			expectDefer: new(4 * time.Second),
		},
		{
			name:                  "old deletion should not defer",
			activeResponderStatus: &lifecyclev1alpha1.ResponderStatus{},
			target: newTargetInfoForEviction(mkValidPodTarget("my-pod", "uid-1"),
				mkValidPod("my-pod", "uid-1", setPodDeletionTimestamp(&metav1.Time{Time: now.Add(-5 * time.Second)})),
			),
			expectDefer: nil,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := shouldDeferCompletion(now, tc.activeResponderStatus, tc.target); !ptr.Equal(got, tc.expectDefer) {
				t.Errorf("got defer completion %v, expected %v", ptr.Deref(got, -1), ptr.Deref(tc.expectDefer, -1))
			}
		})
	}
}

func TestSortTargetResponders(t *testing.T) {
	tests := []struct {
		name     string
		input    []lifecyclev1alpha1.TargetResponder
		expected []lifecyclev1alpha1.TargetResponder
	}{
		{
			name:     "empty",
			input:    nil,
			expected: nil,
		},
		{
			name: "sort",
			input: []lifecyclev1alpha1.TargetResponder{
				{Name: "bar", Priority: new(int32(3))},
				{Name: "foo", Priority: new(int32(5))},
				{Name: "baz", Priority: new(int32(4))},
			},
			expected: []lifecyclev1alpha1.TargetResponder{
				{Name: "foo", Priority: new(int32(5))},
				{Name: "baz", Priority: new(int32(4))},
				{Name: "bar", Priority: new(int32(3))},
			},
		},
		// Advanced tests are in valdidation.SortTargetResponders
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := sortTargetResponders(tc.input)
			if diff := cmp.Diff(tc.expected, result); len(diff) > 0 {
				t.Errorf("sortTargetResponders expected %v, got %v, diff %v", tc.expected, result, diff)
			}
		})
	}
}
func mkValidEviction(name, podName, podUID string, tweaks ...func(obj *lifecyclev1alpha1.Eviction)) *lifecyclev1alpha1.Eviction {
	obj := &lifecyclev1alpha1.Eviction{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: lifecyclev1alpha1.EvictionSpec{
			Target: lifecyclev1alpha1.EvictionTarget{
				Pod: &lifecyclev1alpha1.EvictionPodReference{
					UID:  apimachinerytypes.UID(podUID),
					Name: podName,
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(obj)
	}

	return obj
}

func setCreationTimestamp(passiveClock clock.PassiveClock) func(obj *lifecyclev1alpha1.Eviction) {
	return func(obj *lifecyclev1alpha1.Eviction) {
		obj.CreationTimestamp = metav1.Time{Time: passiveClock.Now()}
	}
}

func addConditionTrue(passiveClock clock.PassiveClock, conditionName lifecyclev1alpha1.EvictionConditionType, conditionReason lifecyclev1alpha1.EvictionConditionReason) func(obj *lifecyclev1alpha1.Eviction) {
	return addCondition(passiveClock, conditionName, metav1.ConditionTrue, conditionReason)
}

func addCondition(passiveClock clock.PassiveClock, conditionName lifecyclev1alpha1.EvictionConditionType, conditionStatus metav1.ConditionStatus, conditionReason lifecyclev1alpha1.EvictionConditionReason) func(obj *lifecyclev1alpha1.Eviction) {
	return func(obj *lifecyclev1alpha1.Eviction) {
		obj.Status.Conditions = append(obj.Status.Conditions, metav1.Condition{
			Type:               string(conditionName),
			Status:             conditionStatus,
			Reason:             string(conditionReason),
			LastTransitionTime: metav1.Time{Time: passiveClock.Now()},
			Message:            "msg",
		})
	}
}

func mkValidPod(name, uid string, tweaks ...func(obj *v1.Pod)) *v1.Pod {
	obj := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			UID:  types.UID(uid),
		},
	}
	for _, tweak := range tweaks {
		tweak(obj)
	}

	return obj
}
func setPodDeletionTimestamp(intent *metav1.Time) func(obj *v1.Pod) {
	return func(obj *v1.Pod) {
		obj.DeletionTimestamp = intent
	}
}
func setEvictionResponders(evictionResponders ...v1.EvictionResponder) func(obj *v1.Pod) {
	return func(obj *v1.Pod) {
		obj.Spec.EvictionResponders = evictionResponders

	}
}

func mkValidPodTarget(name, uid string) lifecyclev1alpha1.EvictionTarget {
	obj := lifecyclev1alpha1.EvictionTarget{
		Pod: &lifecyclev1alpha1.EvictionPodReference{
			Name: name,
			UID:  types.UID(uid),
		},
	}
	return obj
}

func mkValidEvictionRequestPodTarget(name, uid string) lifecyclev1alpha1.EvictionRequestTarget {
	obj := lifecyclev1alpha1.EvictionRequestTarget{
		Pod: &lifecyclev1alpha1.EvictionRequestPodReference{
			Name: name,
			UID:  types.UID(uid),
		},
	}
	return obj
}

func mkValidEvictionRequest(podName, podUID string, tweaks ...func(obj *lifecyclev1alpha1.EvictionRequest)) *lifecyclev1alpha1.EvictionRequest {
	obj := &lifecyclev1alpha1.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "foo",
			CreationTimestamp: metav1.Now(),
		},
		Spec: lifecyclev1alpha1.EvictionRequestSpec{
			Target: lifecyclev1alpha1.EvictionRequestTarget{
				Pod: &lifecyclev1alpha1.EvictionRequestPodReference{
					UID:  apimachinerytypes.UID(podUID),
					Name: podName,
				},
			},
			Requester: "foo.example.com/bar",
			Intent:    lifecyclev1alpha1.EvictionRequestIntentEviction,
		},
	}
	for _, tweak := range tweaks {
		tweak(obj)
	}

	return obj
}

func addTargetResponders(responders ...string) func(obj *lifecyclev1alpha1.Eviction) {
	return func(obj *lifecyclev1alpha1.Eviction) {
		lastPriority := len(obj.Status.TargetResponders)
		for i, name := range responders {
			obj.Status.TargetResponders = append(obj.Status.TargetResponders, lifecyclev1alpha1.TargetResponder{
				Name:     name,
				Priority: new(5000 - int32(lastPriority+i)),
				State:    lifecyclev1alpha1.ResponderStateInactive,
			})
		}
	}
}
func setStateFor(state lifecyclev1alpha1.ResponderStateType, idx int) func(obj *lifecyclev1alpha1.Eviction) {
	return func(obj *lifecyclev1alpha1.Eviction) {
		obj.Status.TargetResponders[idx].State = state
	}
}
func setERIntent(intent lifecyclev1alpha1.EvictionRequestIntent) func(obj *lifecyclev1alpha1.EvictionRequest) {
	return func(obj *lifecyclev1alpha1.EvictionRequest) {
		obj.Spec.Intent = intent
	}
}
func setERRequester(name string) func(obj *lifecyclev1alpha1.EvictionRequest) {
	return func(obj *lifecyclev1alpha1.EvictionRequest) {
		obj.Spec.Requester = name
	}
}

func setERDeletionTimestamp(intent *metav1.Time) func(obj *lifecyclev1alpha1.EvictionRequest) {
	return func(obj *lifecyclev1alpha1.EvictionRequest) {
		obj.DeletionTimestamp = intent
	}
}
func addERCondition(passiveClock clock.PassiveClock, conditionName lifecyclev1alpha1.EvictionConditionType, conditionStatus metav1.ConditionStatus, conditionReason lifecyclev1alpha1.EvictionConditionReason) func(obj *lifecyclev1alpha1.EvictionRequest) {
	return func(obj *lifecyclev1alpha1.EvictionRequest) {
		obj.Status.Conditions = append(obj.Status.Conditions, metav1.Condition{
			Type:               string(conditionName),
			Status:             conditionStatus,
			Reason:             string(conditionReason),
			LastTransitionTime: metav1.Time{Time: passiveClock.Now()},
			Message:            "msg",
		})
	}
}
