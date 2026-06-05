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

package eviction

import (
	"fmt"
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apimachinerytypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/coordination"
	utilsclock "k8s.io/utils/clock"
	testing2 "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"

	// Ensure all API groups are registered with the scheme
	_ "k8s.io/kubernetes/pkg/apis/coordination/install"
)

const validUID = "a2ee91f4-e13c-44db-9edc-4240e7383ab9"

func TestDeclarativeValidate(t *testing.T) {
	apiVersions := []string{"v1alpha1"} // Eviction is currently only in v1alpha1
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		input  *coordination.Eviction
		errors field.ErrorList
	}{
		"missing target": {
			input: mkValidEviction(clearTarget()),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "", "").WithOrigin("union"),
			},
		},
		"missing target name": {
			input: mkValidEviction(setTarget("", validUID)),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "target", "pod", "name"), ""),
			},
		},
		"invalid target name": {
			input: mkValidEviction(setTarget("_test", validUID)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("spec", "target", "pod", "name"), "", "").WithOrigin("format=k8s-long-name"),
			},
		},
		"missing target uid": {
			input: mkValidEviction(setTarget("bar", "")),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "target", "pod", "uid"), ""),
			},
		},
		"invalid target uid": {
			input: mkValidEviction(setTarget("bar", "invalid-uid")),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target", "pod", "uid"), "", "").WithOrigin("format=k8s-uuid"),
			},
		},
	}
	clock := testing2.NewFakePassiveClock(time.Now())
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "coordination.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "evictions",
				IsResourceRequest: true,
				Verb:              "create",
			})
			strategy := NewStrategy(clock)
			apitesting.VerifyValidationEquivalence(t, ctx, tc.input, strategy, tc.errors)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	apiVersions := []string{"v1alpha1"} // Eviction is currently only in v1alpha1
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		input    *coordination.Eviction
		oldInput *coordination.Eviction
		errors   field.ErrorList
	}{
		"clear target": {
			oldInput: mkValidEviction(),
			input:    mkValidEviction(clearTarget()),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "", "").WithOrigin("immutable"),
			},
		},
		"change target name": {
			oldInput: mkValidEviction(),
			input:    mkValidEviction(setTarget("change", validUID)),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "", "").WithOrigin("immutable"),
			},
		},
		"change target uid": {
			oldInput: mkValidEviction(),
			input:    mkValidEviction(setTarget("bar", "")),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "", "").WithOrigin("immutable"),
			},
		},
	}
	clock := testing2.NewFakePassiveClock(time.Now())
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.oldInput.ResourceVersion = "0"
			tc.input.ResourceVersion = "1"
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "coordination.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "evictions",
				IsResourceRequest: true,
				Verb:              "update",
			})
			strategy := NewStrategy(clock)
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, tc.input, tc.oldInput, strategy, tc.errors)
		})
	}
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	apiVersions := []string{"v1alpha1"} // Eviction is currently only in v1alpha1
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateStatusUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateStatusUpdate(t *testing.T, apiVersion string) {
	clock := testing2.NewFakePassiveClock(time.Now())
	clockAfter := func(duration time.Duration) utilsclock.PassiveClock {
		return testing2.NewFakePassiveClock(clock.Now().Add(duration))
	}

	testCases := map[string]struct {
		input    *coordination.EvictionStatus
		oldInput *coordination.EvictionStatus
		errors   field.ErrorList
	}{
		// conditions
		"too many conditions": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, addConditionsCount(clock, 101)),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "conditions"), 101, 100).WithOrigin("maxItems"),
			},
		},
		"duplicate condition": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(1, addCondition(clock, coordination.EvictionConditionEvicted, true), addCondition(clock, coordination.EvictionConditionEvicted, true)),
			errors: []*field.Error{
				field.Duplicate(field.NewPath("status", "conditions").Index(1), ""),
			},
		},
		// observedGeneration
		"decrease generation to 0": {
			oldInput: mkValidEvictionStatus(0, setObservedGeneration(ptr.To[int64](7))),
			input:    mkValidEvictionStatus(0, setObservedGeneration(ptr.To[int64](0))),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "observedGeneration"), -1, "").WithOrigin("minimum"),
			},
		},
		"decrease generation to negative": {
			oldInput: mkValidEvictionStatus(0, setObservedGeneration(ptr.To[int64](5))),
			input:    mkValidEvictionStatus(0, setObservedGeneration(ptr.To[int64](-1))),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "observedGeneration"), -1, "").WithOrigin("minimum"),
			},
		},
		// requesters
		"add a duplicate requesters": {
			oldInput: mkValidEvictionStatus(0, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz")),
			input:    mkValidEvictionStatus(0, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz", "foo.example.com/baz")),
			errors: field.ErrorList{
				field.Duplicate(field.NewPath("status", "requesters").Index(1), ""),
			},
		},
		"add a requester without a name": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, addRequesters(coordination.RequesterIntentEviction, "")),
			errors: field.ErrorList{
				field.Required(field.NewPath("status", "requesters").Index(0).Child("name"), ""),
			},
		},
		"add a requester without an intent": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, addRequesters("", "foo.example.com/baz")),
			errors: field.ErrorList{
				field.Required(field.NewPath("status", "requesters").Index(0).Child("intent"), ""),
			},
		},
		"add a requester with invalid intent": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, addRequesters("Invalid", "foo.example.com/bar")),
			errors: field.ErrorList{
				field.NotSupported(field.NewPath("status", "requesters").Index(0).Child("intent"), "", []coordination.RequesterIntent{coordination.RequesterIntentEviction, coordination.RequesterIntentWithdrawn}),
			},
		},
		"change to a duplicate requester": {
			oldInput: mkValidEvictionStatus(0, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz", "foo.example.com/bay")),
			input:    mkValidEvictionStatus(0, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz", "foo.example.com/baz")),
			errors: field.ErrorList{
				field.Duplicate(field.NewPath("status", "requesters").Index(1), ""),
				field.Invalid(field.NewPath("status", "requesters"), "", "requesters cannot be removed").MarkFromImperative(),
			},
		},
		"change a valid requester to an invalid one": {
			oldInput: mkValidEvictionStatus(0, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz")),
			input:    mkValidEvictionStatus(0, addRequesters("Invalid", "foo.example.com/baz")),
			errors: field.ErrorList{
				field.NotSupported(field.NewPath("status", "requesters").Index(0).Child("intent"), "", []coordination.RequesterIntent{coordination.RequesterIntentEviction, coordination.RequesterIntentWithdrawn}),
			},
		},
		// targetResponders and responders
		"duplicate targetResponders and responders": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, addTargetResponders("example.com/baz", "example.com/baz"), addStatusResponders("example.com/baz", "example.com/baz")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys in the same order").MarkFromImperative(), // triggered by fallback to oldTargetResponders
				field.Duplicate(field.NewPath("status", "targetResponders").Index(1), ""),
				field.Duplicate(field.NewPath("status", "responders").Index(1), ""),
			},
		},
		"required targetResponder and responder name": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, addTargetResponders(""), setStateFor(coordination.ResponderStateActive, 0), addStatusResponders("")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys in the same order").MarkFromImperative(), // triggered by fallback to oldTargetResponders
				field.Required(field.NewPath("status", "targetResponders").Index(0).Child("name"), ""),
				field.Required(field.NewPath("status", "responders").Index(0).Child("name"), ""),
			},
		},
		// targetResponders
		"too many targetResponders": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatusWithStatuses(18, 0),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "targetResponders"), 18, 17).WithOrigin("maxItems"),
			},
		},
		"required targetResponder state": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(1, setStateFor("", 0)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys in the same order").MarkFromImperative(), // triggered by fallback to oldTargetResponders
				field.Required(field.NewPath("status", "targetResponders").Index(0).Child("state"), ""),
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Canceled, Completed, Interrupted").MarkFromImperative(),
			},
		},
		"invalid targetResponder state": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(1, setStateFor("Invalid", 0)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys in the same order").MarkFromImperative(), // triggered by fallback to oldTargetResponders
				field.NotSupported(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", []string(nil)),
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Canceled, Completed, Interrupted").MarkFromImperative(),
			},
		},
		// responders
		"too many status responders": {
			oldInput: mkValidEvictionStatusWithStatuses(18, 17),
			input:    mkValidEvictionStatus(18),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "responders"), 18, 17).WithOrigin("maxItems"),
			},
		},
		// status responder name
		// startTime
		"startTime cannot be removed once set": {
			oldInput: mkValidEvictionStatus(2,
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("startTime"), nil, "").WithOrigin("update"),
			},
		},
		"startTime cannot be changed once set": {
			oldInput: mkValidEvictionStatus(2,
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setRespondersStartTime(clockAfter(15*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("startTime"), nil, "").WithOrigin("update"),
			},
		},
		// heartbeatTime
		// expectedCompletionTime
		// completionTime
		"completionTime cannot be changed once set": {
			oldInput: mkValidEvictionStatus(2,
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clockAfter(5*time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clockAfter(4*time.Minute), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("completionTime"), nil, "").WithOrigin("update"),
			},
		},
		"completionTime cannot be removed once set": {
			oldInput: mkValidEvictionStatus(2,
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clockAfter(5*time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setRespondersStartTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("completionTime"), nil, "").WithOrigin("update"),
			},
		},
		// message
		"too long message": {
			oldInput: mkValidEvictionStatus(2,
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clock, 0, 1),
				setRespondersMessage(0, 1, strings.Repeat("a", 4000))),
			errors: []*field.Error{
				field.TooLongCharacters(field.NewPath("status", "responders").Index(0).Child("message"), "", 4000).WithOrigin("maxLength"),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			oldEviction := mkValidEviction()
			oldEviction.ResourceVersion = "0"
			oldEviction.Status = *tc.oldInput
			eviction := mkValidEviction()
			eviction.ResourceVersion = "1"
			eviction.Status = *tc.input
			strategy := NewStatusStrategy(NewStrategy(clock))

			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "coordination.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "evictions",
				IsResourceRequest: true,
				Verb:              "update",
				Subresource:       "status",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, eviction, oldEviction, strategy, tc.errors, apitesting.WithSubResources("status"))
		})
	}
}

func mkValidEviction(tweaks ...func(obj *coordination.Eviction)) *coordination.Eviction {
	obj := coordination.Eviction{
		ObjectMeta: metav1.ObjectMeta{Name: "evict-pod-1-foo.pod", Namespace: "foo"},
		Spec: coordination.EvictionSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.EvictionPodReference{
					UID:  validUID,
					Name: "foo.pod",
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return &obj
}

func clearTarget() func(obj *coordination.Eviction) {
	return func(obj *coordination.Eviction) {
		obj.Spec.Target.Pod = nil
	}
}

func setTarget(name, uid string) func(obj *coordination.Eviction) {
	return func(obj *coordination.Eviction) {
		obj.Spec.Target.Pod = &coordination.EvictionPodReference{
			UID:  apimachinerytypes.UID(uid),
			Name: name,
		}
	}
}

func responderName(i int) string {
	return fmt.Sprintf("responder.example.com/bar%d", i)
}
func mkValidEvictionStatus(responders int, tweaks ...func(obj *coordination.EvictionStatus)) *coordination.EvictionStatus {
	return mkValidEvictionStatusWithStatuses(responders, responders, tweaks...)
}
func mkValidEvictionStatusWithStatuses(responders, statuses int, tweaks ...func(obj *coordination.EvictionStatus)) *coordination.EvictionStatus {
	obj := coordination.EvictionStatus{
		ObservedGeneration: ptr.To[int64](1),
	}
	for i := range responders {
		obj.TargetResponders = append(obj.TargetResponders, coordination.TargetResponder{
			Name:  responderName(i),
			State: coordination.ResponderStateInactive,
		})
		if i == 0 {
			obj.TargetResponders[i].State = coordination.ResponderStateActive
		}
	}
	for i := range statuses {
		obj.Responders = append(obj.Responders, coordination.ResponderStatus{
			Name: responderName(i),
		})
		if i == 0 {
			obj.Responders[i].StartTime = ptr.To(metav1.Now())
		}
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return &obj
}
func addCondition(clock utilsclock.PassiveClock, name coordination.EvictionConditionType, status bool) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		newCond := metav1.Condition{
			Type:               string(name),
			Status:             metav1.ConditionFalse,
			Reason:             string(name) + "Reason",
			LastTransitionTime: metav1.Time{Time: clock.Now()},
		}
		if status {
			newCond.Status = metav1.ConditionTrue
		}
		obj.Conditions = append(obj.Conditions, newCond)
	}
}
func addConditionsCount(clock utilsclock.PassiveClock, count int) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for i := range count {
			addCondition(clock, coordination.EvictionConditionType(fmt.Sprintf("Condition%d", i)), true)(obj)
		}
	}
}

func setObservedGeneration(generation *int64) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		obj.ObservedGeneration = generation
	}
}
func addRequesters(intent coordination.RequesterIntent, names ...string) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for _, name := range names {
			obj.Requesters = append(obj.Requesters, coordination.Requester{Name: name, Intent: intent})
		}
	}
}
func addTargetResponders(responders ...string) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for _, name := range responders {
			obj.TargetResponders = append(obj.TargetResponders, coordination.TargetResponder{Name: name, State: coordination.ResponderStateInactive})
		}
	}
}

func setStateFor(state coordination.ResponderStateType, idx int) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		obj.TargetResponders[idx].State = state
	}
}

func addStatusResponders(responders ...string) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for _, responder := range responders {
			obj.Responders = append(obj.Responders, coordination.ResponderStatus{Name: responder})
		}
	}
}
func setRespondersStartTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].StartTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setRespondersHeartBeatTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].HeartbeatTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setRespondersCompletionTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].CompletionTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setRespondersMessage(from, to int, suffixes ...string) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].Message = fmt.Sprintf("message %d", i)
			for _, suffix := range suffixes {
				obj.Responders[i].Message += suffix
			}
		}
	}
}
