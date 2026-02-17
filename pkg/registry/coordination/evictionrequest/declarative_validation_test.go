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

package evictionrequest

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apimachinerytypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/coordination"
	utilsclock "k8s.io/utils/clock"
	testing2 "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"

	// Ensure all API groups are registered with the scheme
	_ "k8s.io/kubernetes/pkg/apis/coordination/install"
)

const valiUIDName = "a2ee91f4-e13c-44db-9edc-4240e7383ab9"

type TestDecisionAuthorizer struct {
	decision authorizer.Decision
}

func (t *TestDecisionAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	return t.decision, "", nil
}

func TestDeclarativeValidate(t *testing.T) {
	apiVersions := []string{"v1alpha1"} // EvictionRequest is currently only in v1alpha1
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		input  *coordination.EvictionRequest
		errors field.ErrorList
	}{
		"name is not valid": {
			input: mkValidEvictionRequest(1, setName("invalid-name-test", "")),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "", "").WithOrigin("format=k8s-uuid"),
				field.Invalid(field.NewPath("metadata", "name"), "", "must be the same value as spec.target.pod.uid").MarkFromImperative(),
			},
		},
		"generateName not supported": {
			input: mkValidEvictionRequest(1, setName("", "invalid-generate-name")),
			errors: field.ErrorList{
				field.Forbidden(field.NewPath("metadata", "generateName"), ""),
				field.Invalid(field.NewPath("metadata", "name"), "", "").WithOrigin("format=k8s-uuid"),
				field.Required(field.NewPath("metadata", "name"), "name or generateName is required").MarkFromImperative(),
				field.Invalid(field.NewPath("metadata", "name"), "", "must be the same value as spec.target.pod.uid").MarkFromImperative(),
			},
		},
		"generateName with name not supported": {
			input: mkValidEvictionRequest(1, setName(valiUIDName, "invalid-generate-name")),
			errors: field.ErrorList{
				field.Forbidden(field.NewPath("metadata", "generateName"), ""),
			},
		},
		"missing target": {
			input: mkValidEvictionRequest(1, clearTarget()),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "", "").WithOrigin("union"),
			},
		},
		"missing target name": {
			input: mkValidEvictionRequest(1, setTarget("", valiUIDName)),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "target", "pod", "name"), ""),
			},
		},
		"invalid target name": {
			input: mkValidEvictionRequest(1, setTarget("_test", valiUIDName)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("spec", "target", "pod", "name"), "", "").WithOrigin("format=k8s-long-name"),
			},
		},
		"missing target uid": {
			input: mkValidEvictionRequest(1, setTarget("bar", "")),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "target", "pod", "uid"), ""),
			},
		},
		"invalid target uid": {
			input: mkValidEvictionRequest(1, setTarget("bar", "invalid-uid")),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target", "pod", "uid"), "", "").WithOrigin("format=k8s-uuid"),
				field.Invalid(field.NewPath("metadata", "name"), "", "must be the same value as spec.target.pod.uid").MarkFromImperative(),
			},
		},
		"too few requesters": {
			input: mkValidEvictionRequest(0),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "requesters"), ""),
				field.Invalid(field.NewPath("spec", "requesters"), "", "must have at least one requester with an intent that is not \"Withdrawn\" on EvictionRequest creation").MarkFromImperative(),
			},
		},
		"too many requesters": {
			input: mkValidEvictionRequest(101),
			errors: field.ErrorList{
				field.TooMany(field.NewPath("spec", "requesters"), 101, 100).WithOrigin("maxItems"),
			},
		},
		"duplicate requesters": {
			input: mkValidEvictionRequest(3, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz", "foo.example.com/baz")),
			errors: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "requesters").Index(4), ""),
			},
		},
		"requester without a name": {
			input: mkValidEvictionRequest(0, addRequesters(coordination.RequesterIntentEviction, "")),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "requesters").Index(0).Child("name"), ""),
			},
		},
		"requester without an intent": {
			input: mkValidEvictionRequest(0, addRequesters("", "foo.example.com/baz")),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "requesters").Index(0).Child("intent"), ""),
			},
		},
		"requester with invalid intent": {
			input: mkValidEvictionRequest(0, addRequesters("Invalid", "foo.example.com/bar")),
			errors: field.ErrorList{
				field.NotSupported(field.NewPath("spec", "requesters").Index(0).Child("intent"), "", []coordination.RequesterIntent{coordination.RequesterIntentEviction, coordination.RequesterIntentWithdrawn}),
			},
		},
	}
	clock := testing2.NewFakePassiveClock(time.Now())
	for _, authDecision := range []authorizer.Decision{authorizer.DecisionAllow, authorizer.DecisionDeny, authorizer.DecisionNoOpinion} {
		for k, tc := range testCases {
			t.Run(fmt.Sprintf("%v authDecision=%v", k, authDecision.String()), func(t *testing.T) {
				ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
					APIGroup:          "coordination.k8s.io",
					APIVersion:        apiVersion,
					Resource:          "evictionrequests",
					IsResourceRequest: true,
					Verb:              "create",
				})
				ctx = genericapirequest.WithUser(ctx, &user.DefaultInfo{Name: "test"})
				strategy := NewStrategy(&TestDecisionAuthorizer{authDecision}, clock)
				if tc.input.Spec.Target.Pod != nil && authDecision != authorizer.DecisionAllow {
					tc.errors = field.ErrorList{field.Forbidden(field.NewPath(""), "User \"test\" must have permission to delete pods in \"foo\" namespace when spec.target.pod is set")}
				}
				apitesting.VerifyValidationEquivalence(t, ctx, tc.input, strategy.Validate, tc.errors)
			})
		}
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	apiVersions := []string{"v1alpha1"} // EvictionRequest is currently only in v1alpha1
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		input             *coordination.EvictionRequest
		oldInput          *coordination.EvictionRequest
		requiresAuthCheck bool
		errors            field.ErrorList
	}{
		"clear target": {
			oldInput: mkValidEvictionRequest(1),
			input:    mkValidEvictionRequest(1, clearTarget()),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "", "").WithOrigin("immutable"),
			},
		},
		"change target name": {
			oldInput: mkValidEvictionRequest(1),
			input:    mkValidEvictionRequest(1, setTarget("change", valiUIDName)),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "", "").WithOrigin("immutable"),
			},
		},
		"change target uid": {
			oldInput: mkValidEvictionRequest(1),
			input:    mkValidEvictionRequest(1, setTarget("bar", "")),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "", "").WithOrigin("immutable"),
			},
		},
		"add too many requesters": {
			oldInput:          mkValidEvictionRequest(1),
			input:             mkValidEvictionRequest(101),
			requiresAuthCheck: true,
			errors: field.ErrorList{
				field.TooMany(field.NewPath("spec", "requesters"), 101, 100).WithOrigin("maxItems"),
			},
		},
		"add a duplicate requesters": {
			oldInput:          mkValidEvictionRequest(3, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz")),
			input:             mkValidEvictionRequest(3, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz", "foo.example.com/baz")),
			requiresAuthCheck: true,
			errors: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "requesters").Index(4), ""),
			},
		},
		"change and remove a requester": {
			oldInput:          mkValidEvictionRequest(1, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz")),
			input:             mkValidEvictionRequest(0, addRequesters(coordination.RequesterIntentEviction, "bar.example.com/bay")),
			requiresAuthCheck: true,
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "requesters"), "", "requesters cannot be removed and must preserve the same keys in the same order").MarkFromImperative(),
			},
		},
		"change to a duplicate requester": {
			oldInput:          mkValidEvictionRequest(1, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz")),
			input:             mkValidEvictionRequest(0, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz", "foo.example.com/baz")),
			requiresAuthCheck: true,
			errors: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "requesters").Index(1), ""),
				field.Invalid(field.NewPath("spec", "requesters"), "", "requesters cannot be removed and must preserve the same keys in the same order").MarkFromImperative(),
			},
		},
		"add a requester without a name": {
			oldInput:          mkValidEvictionRequest(1),
			input:             mkValidEvictionRequest(1, addRequesters(coordination.RequesterIntentEviction, "")),
			requiresAuthCheck: true,
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "requesters").Index(1).Child("name"), ""),
			},
		},
		"requester without an intent": {
			oldInput:          mkValidEvictionRequest(1, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz")),
			input:             mkValidEvictionRequest(0, addRequesters("", "foo.example.com/baz")),
			requiresAuthCheck: true,
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "requesters").Index(0).Child("intent"), ""),
				field.Invalid(field.NewPath("spec", "requesters"), "", "must have at least one requester with an intent that is not \"Withdrawn\" on EvictionRequest creation").MarkFromImperative(),
			},
		},
		"requester with invalid intent": {
			oldInput:          mkValidEvictionRequest(1, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz")),
			input:             mkValidEvictionRequest(0, addRequesters("Invalid", "foo.example.com/baz")),
			requiresAuthCheck: true,
			errors: field.ErrorList{
				field.NotSupported(field.NewPath("spec", "requesters").Index(0).Child("intent"), "", []coordination.RequesterIntent{coordination.RequesterIntentEviction, coordination.RequesterIntentWithdrawn}),
				field.Invalid(field.NewPath("spec", "requesters"), "", "must have at least one requester with an intent that is not \"Withdrawn\" on EvictionRequest creation").MarkFromImperative(),
			},
		},
	}
	clock := testing2.NewFakePassiveClock(time.Now())
	for _, authDecision := range []authorizer.Decision{authorizer.DecisionAllow, authorizer.DecisionDeny, authorizer.DecisionNoOpinion} {
		for k, tc := range testCases {
			t.Run(fmt.Sprintf("%v authDecision=%v", k, authDecision.String()), func(t *testing.T) {
				tc.oldInput.ResourceVersion = "0"
				tc.input.ResourceVersion = "1"
				ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
					APIGroup:          "coordination.k8s.io",
					APIVersion:        apiVersion,
					Resource:          "evictionrequests",
					IsResourceRequest: true,
					Verb:              "update",
				})
				ctx = genericapirequest.WithUser(ctx, &user.DefaultInfo{Name: "test"})
				strategy := NewStrategy(&TestDecisionAuthorizer{authDecision}, clock)
				if tc.input.Spec.Target.Pod != nil && tc.requiresAuthCheck && authDecision != authorizer.DecisionAllow {
					tc.errors = field.ErrorList{field.Forbidden(field.NewPath("spec", "requesters"), "User \"test\" must have permission to delete pods in \"foo\" namespace when spec.target.pod is set")}
				}
				apitesting.VerifyUpdateValidationEquivalence(t, ctx, tc.input, tc.oldInput, strategy.ValidateUpdate, tc.errors)
			})
		}
	}
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	apiVersions := []string{"v1alpha1"} // EvictionRequest is currently only in v1alpha1
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
		input    *coordination.EvictionRequestStatus
		oldInput *coordination.EvictionRequestStatus
		errors   field.ErrorList
	}{
		// conditions
		"too many conditions": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(0, addConditionsCount(clock, 501)),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "conditions"), 501, 500).WithOrigin("maxItems"),
			},
		},
		"duplicate condition": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionEvicted, true), addCondition(clock, coordination.EvictionRequestConditionEvicted, true)),
			errors: []*field.Error{
				field.Duplicate(field.NewPath("status", "conditions").Index(1), ""),
			},
		},
		// observedGeneration
		"decrease generation to 0": {
			oldInput: mkValidEvictionRequestStatus(0, setObservedGeneration(ptr.To[int64](7))),
			input:    mkValidEvictionRequestStatus(0, setObservedGeneration(ptr.To[int64](0))),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "observedGeneration"), -1, "").WithOrigin("minimum"),
			},
		},
		"decrease generation to negative": {
			oldInput: mkValidEvictionRequestStatus(0, setObservedGeneration(ptr.To[int64](5))),
			input:    mkValidEvictionRequestStatus(0, setObservedGeneration(ptr.To[int64](-1))),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "observedGeneration"), -1, "").WithOrigin("minimum"),
			},
		},
		// targetResponders and responders
		"duplicate targetResponders and responders": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(0, addTargetResponders("example.com/baz", "example.com/baz"), addStatusResponders("example.com/baz", "example.com/baz")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys in the same order").MarkFromImperative(), // triggered by fallback to oldTargetResponders
				field.Duplicate(field.NewPath("status", "targetResponders").Index(1), ""),
				field.Duplicate(field.NewPath("status", "responders").Index(1), ""),
			},
		},
		"required targetResponder and responder name": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(0, addTargetResponders(""), setStateFor(coordination.ResponderStateActive, 0), addStatusResponders("")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys in the same order").MarkFromImperative(), // triggered by fallback to oldTargetResponders
				field.Required(field.NewPath("status", "targetResponders").Index(0).Child("name"), ""),
				field.Required(field.NewPath("status", "responders").Index(0).Child("name"), ""),
			},
		},
		// targetResponders
		"too many targetResponders": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatusWithStatuses(18, 0),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "targetResponders"), 18, 17).WithOrigin("maxItems"),
			},
		},
		"required targetResponder state": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(1, setStateFor("", 0)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys in the same order").MarkFromImperative(), // triggered by fallback to oldTargetResponders
				field.Required(field.NewPath("status", "targetResponders").Index(0).Child("state"), ""),
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Canceled, Completed, Interrupted").MarkFromImperative(),
			},
		},
		"invalid targetResponder state": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(1, setStateFor("Invalid", 0)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys in the same order").MarkFromImperative(), // triggered by fallback to oldTargetResponders
				field.NotSupported(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", []string(nil)),
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Canceled, Completed, Interrupted").MarkFromImperative(),
			},
		},
		// responders
		"too many status responders": {
			oldInput: mkValidEvictionRequestStatusWithStatuses(18, 17),
			input:    mkValidEvictionRequestStatus(18),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "responders"), 18, 17).WithOrigin("maxItems"),
			},
		},
		// status responder name
		// startTime
		"startTime cannot be removed once set": {
			oldInput: mkValidEvictionRequestStatus(2,
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("startTime"), nil, "").WithOrigin("update"),
			},
		},
		"startTime cannot be changed once set": {
			oldInput: mkValidEvictionRequestStatus(2,
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				setRespondersStartTime(clockAfter(15*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("startTime"), nil, "").WithOrigin("update"),
			},
		},
		// heartbeatTime
		// expectedCompletionTime
		// completionTime
		"completionTime cannot be changed once set": {
			oldInput: mkValidEvictionRequestStatus(2,
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clockAfter(5*time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clockAfter(4*time.Minute), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("completionTime"), nil, "").WithOrigin("update"),
			},
		},
		"completionTime cannot be removed once set": {
			oldInput: mkValidEvictionRequestStatus(2,
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clockAfter(5*time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				setRespondersStartTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("completionTime"), nil, "").WithOrigin("update"),
			},
		},
		// message
		"too long message": {
			oldInput: mkValidEvictionRequestStatus(2,
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
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
			oldEvictionRequest := mkValidEvictionRequest(2)
			oldEvictionRequest.ResourceVersion = "0"
			oldEvictionRequest.Status = *tc.oldInput
			evictionRequest := mkValidEvictionRequest(2)
			evictionRequest.ResourceVersion = "1"
			evictionRequest.Status = *tc.input
			strategy := NewStatusStrategy(NewStrategy(nil, clock))

			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "coordination.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "evictionrequests",
				IsResourceRequest: true,
				Verb:              "update",
				Subresource:       "status",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, evictionRequest, oldEvictionRequest, strategy.ValidateUpdate, tc.errors, apitesting.WithSubResources("status"))
		})
	}
}

func mkValidEvictionRequest(requesters int, tweaks ...func(obj *coordination.EvictionRequest)) *coordination.EvictionRequest {
	obj := coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: valiUIDName, Namespace: "foo"},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.PodReference{
					UID:  valiUIDName,
					Name: "foo.pod",
				},
			},
		},
	}
	for i := range requesters {
		obj.Spec.Requesters = append(obj.Spec.Requesters, coordination.Requester{
			Name:   responderName(i),
			Intent: coordination.RequesterIntentEviction,
		})
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return &obj
}

func setName(name, generateName string) func(obj *coordination.EvictionRequest) {
	return func(obj *coordination.EvictionRequest) {
		obj.Name = name
		obj.GenerateName = generateName
	}
}

func clearTarget() func(obj *coordination.EvictionRequest) {
	return func(obj *coordination.EvictionRequest) {
		obj.Spec.Target.Pod = nil
	}
}

func setTarget(name, uid string) func(obj *coordination.EvictionRequest) {
	return func(obj *coordination.EvictionRequest) {
		obj.Spec.Target.Pod = &coordination.PodReference{
			UID:  apimachinerytypes.UID(uid),
			Name: name,
		}
	}
}
func addRequesters(intent coordination.RequesterIntent, names ...string) func(obj *coordination.EvictionRequest) {
	return func(obj *coordination.EvictionRequest) {
		for _, name := range names {
			obj.Spec.Requesters = append(obj.Spec.Requesters, coordination.Requester{Name: name, Intent: intent})
		}
	}
}

func responderName(i int) string {
	return fmt.Sprintf("responder.example.com/bar%d", i)
}
func mkValidEvictionRequestStatus(responders int, tweaks ...func(obj *coordination.EvictionRequestStatus)) *coordination.EvictionRequestStatus {
	return mkValidEvictionRequestStatusWithStatuses(responders, responders, tweaks...)
}
func mkValidEvictionRequestStatusWithStatuses(responders, statuses int, tweaks ...func(obj *coordination.EvictionRequestStatus)) *coordination.EvictionRequestStatus {
	obj := coordination.EvictionRequestStatus{
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
func addCondition(clock utilsclock.PassiveClock, name coordination.EvictionRequestConditionType, status bool) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
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
func addConditionsCount(clock utilsclock.PassiveClock, count int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := range count {
			addCondition(clock, coordination.EvictionRequestConditionType(fmt.Sprintf("Condition%d", i)), true)(obj)
		}
	}
}

func setObservedGeneration(generation *int64) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		obj.ObservedGeneration = generation
	}
}
func addTargetResponders(responders ...string) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for _, name := range responders {
			obj.TargetResponders = append(obj.TargetResponders, coordination.TargetResponder{Name: name, State: coordination.ResponderStateInactive})
		}
	}
}

func setStateFor(state coordination.ResponderStateType, idx int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		obj.TargetResponders[idx].State = state
	}
}

func addStatusResponders(responders ...string) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for _, responder := range responders {
			obj.Responders = append(obj.Responders, coordination.ResponderStatus{Name: responder})
		}
	}
}
func setRespondersStartTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].StartTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setRespondersHeartBeatTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].HeartbeatTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setRespondersCompletionTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].CompletionTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setRespondersMessage(from, to int, suffixes ...string) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].Message = fmt.Sprintf("message %d", i)
			for _, suffix := range suffixes {
				obj.Responders[i].Message += suffix
			}
		}
	}
}
