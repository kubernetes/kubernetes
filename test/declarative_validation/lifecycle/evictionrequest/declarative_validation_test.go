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
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apimachinerytypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/lifecycle"
	registry "k8s.io/kubernetes/pkg/registry/lifecycle/evictionrequest"
	utilsclock "k8s.io/utils/clock"
	testing2 "k8s.io/utils/clock/testing"
	// Ensure all API groups are registered with the scheme
	_ "k8s.io/kubernetes/pkg/apis/lifecycle/install"
)

const validUID = "5477c2ff-f59f-4eb9-a0be-e54232323faa"

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
		input  *lifecycle.EvictionRequest
		errors field.ErrorList
	}{
		"missing target": {
			input: mkValidEvictionRequest(clearTarget()),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "", "").WithOrigin("union"),
			},
		},
		"missing target name": {
			input: mkValidEvictionRequest(setTarget("", validUID)),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "target", "pod", "name"), ""),
			},
		},
		"invalid target name": {
			input: mkValidEvictionRequest(setTarget("_test", validUID)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("spec", "target", "pod", "name"), "", "").WithOrigin("format=k8s-long-name"),
			},
		},
		"missing target uid": {
			input: mkValidEvictionRequest(setTarget("bar", "")),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "target", "pod", "uid"), ""),
			},
		},
		"invalid target uid": {
			input: mkValidEvictionRequest(setTarget("bar", "invalid-uid")),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target", "pod", "uid"), "", "").WithOrigin("format=k8s-uuid"),
			},
		},
		"missing requester name": {
			input: mkValidEvictionRequest(setRequesterName("")),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "requesterName"), ""),
			},
		},
		"requester without an intent": {
			input: mkValidEvictionRequest(setRequesterIntent("")),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "intent"), ""),
			},
		},
		"requester with invalid intent": {
			input: mkValidEvictionRequest(setRequesterIntent("Invalid")),
			errors: field.ErrorList{
				field.NotSupported(field.NewPath("spec", "intent"), "", []lifecycle.RequesterIntent{lifecycle.RequesterIntentEviction, lifecycle.RequesterIntentWithdrawn}),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "lifecycle.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "evictionrequests",
				IsResourceRequest: true,
				Verb:              "create",
			})
			strategy := registry.NewStrategy()
			apitesting.VerifyValidationEquivalence(t, ctx, tc.input, strategy, tc.errors)
		})
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
		input    *lifecycle.EvictionRequest
		oldInput *lifecycle.EvictionRequest
		errors   field.ErrorList
	}{
		"clear target": {
			oldInput: mkValidEvictionRequest(),
			input:    mkValidEvictionRequest(clearTarget()),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "", "").WithOrigin("immutable"),
			},
		},
		"change target name": {
			oldInput: mkValidEvictionRequest(),
			input:    mkValidEvictionRequest(setTarget("change", validUID)),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "", "").WithOrigin("immutable"),
			},
		},
		"change target uid": {
			oldInput: mkValidEvictionRequest(),
			input:    mkValidEvictionRequest(setTarget("bar", "")),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "", "").WithOrigin("immutable"),
			},
		},
		"change requester name": {
			oldInput: mkValidEvictionRequest(),
			input:    mkValidEvictionRequest(setRequesterName("foo.example.com/baz")),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "requesterName"), "", "").WithOrigin("immutable"),
			},
		},
		"change to a requester without an intent": {
			oldInput: mkValidEvictionRequest(),
			input:    mkValidEvictionRequest(setRequesterIntent("")),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "intent"), ""),
			},
		},
		"change to a requester with invalid intent": {
			oldInput: mkValidEvictionRequest(),
			input:    mkValidEvictionRequest(setRequesterIntent("Invalid")),
			errors: field.ErrorList{
				field.NotSupported(field.NewPath("spec", "intent"), "", []lifecycle.RequesterIntent{lifecycle.RequesterIntentEviction, lifecycle.RequesterIntentWithdrawn}),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.oldInput.ResourceVersion = "0"
			tc.input.ResourceVersion = "1"
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "lifecycle.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "evictionrequests",
				IsResourceRequest: true,
				Verb:              "update",
			})
			strategy := registry.NewStrategy()
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, tc.input, tc.oldInput, strategy, tc.errors)
		})
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

	testCases := map[string]struct {
		input    *lifecycle.EvictionRequestStatus
		oldInput *lifecycle.EvictionRequestStatus
		errors   field.ErrorList
	}{
		// conditions
		"too many conditions": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(0, addConditionsCount(clock, 101)),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "conditions"), 101, 100).WithOrigin("maxItems"),
			},
		},
		"duplicate condition": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(1, addCondition(clock, lifecycle.EvictionConditionTargetEvicted, true), addCondition(clock, lifecycle.EvictionConditionTargetEvicted, true)),
			errors: []*field.Error{
				field.Duplicate(field.NewPath("status", "conditions").Index(1), ""),
			},
		},

		"required condition type": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(1, addCondition(clock, "", true)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "conditions").Index(0).Child("type"), "").MarkAlpha(),
			},
		},
		"observedGeneration condition minimum": {
			oldInput: mkValidEvictionRequestStatus(0),
			input: mkValidEvictionRequestStatus(1, func(obj *lifecycle.EvictionRequestStatus) {
				newCond := metav1.Condition{
					Type:               string(lifecycle.EvictionConditionTargetEvicted),
					Status:             metav1.ConditionFalse,
					Reason:             string(lifecycle.EvictionConditionTargetEvicted) + "Reason",
					LastTransitionTime: metav1.Time{Time: clock.Now()},
					ObservedGeneration: int64(-1),
				}

				obj.Conditions = append(obj.Conditions, newCond)
			}),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions").Index(0).Child("observedGeneration"), "", "").WithOrigin("minimum").MarkAlpha(),
			},
		},
		// observedGeneration
		"decrease generation to 0": {
			oldInput: mkValidEvictionRequestStatus(0, setObservedGeneration(new(int64(7)))),
			input:    mkValidEvictionRequestStatus(0, setObservedGeneration(new(int64(0)))),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "observedGeneration"), -1, "").WithOrigin("minimum"),
			},
		},
		"decrease generation to negative": {
			oldInput: mkValidEvictionRequestStatus(0, setObservedGeneration(new(int64(5)))),
			input:    mkValidEvictionRequestStatus(0, setObservedGeneration(new(int64(-1)))),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "observedGeneration"), -1, "").WithOrigin("minimum"),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			oldEvictionRequest := mkValidEvictionRequest()
			oldEvictionRequest.ResourceVersion = "0"
			oldEvictionRequest.Status = *tc.oldInput
			evictionRequest := mkValidEvictionRequest()
			evictionRequest.ResourceVersion = "1"
			evictionRequest.Status = *tc.input
			strategy := registry.NewStatusStrategy(registry.NewStrategy())

			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "lifecycle.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "evictionrequests",
				IsResourceRequest: true,
				Verb:              "update",
				Subresource:       "status",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, evictionRequest, oldEvictionRequest, strategy, tc.errors, apitesting.WithSubResources("status"))
		})
	}
}

func mkValidEvictionRequest(tweaks ...func(obj *lifecycle.EvictionRequest)) *lifecycle.EvictionRequest {
	obj := lifecycle.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "foo"},
		Spec: lifecycle.EvictionRequestSpec{
			Target: lifecycle.EvictionRequestTarget{
				Pod: &lifecycle.EvictionRequestPodReference{
					UID:  validUID,
					Name: "foo.pod",
				},
			},
			RequesterName: "foo.example.com/bar",
			Intent:        lifecycle.EvictionRequestIntentEviction,
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return &obj
}

func clearTarget() func(obj *lifecycle.EvictionRequest) {
	return func(obj *lifecycle.EvictionRequest) {
		obj.Spec.Target.Pod = nil
	}
}

func setTarget(name, uid string) func(obj *lifecycle.EvictionRequest) {
	return func(obj *lifecycle.EvictionRequest) {
		obj.Spec.Target.Pod = &lifecycle.EvictionRequestPodReference{
			UID:  apimachinerytypes.UID(uid),
			Name: name,
		}
	}
}
func setRequesterName(requesterName string) func(obj *lifecycle.EvictionRequest) {
	return func(obj *lifecycle.EvictionRequest) {
		obj.Spec.RequesterName = requesterName
	}
}
func setRequesterIntent(intent lifecycle.EvictionRequestIntent) func(obj *lifecycle.EvictionRequest) {
	return func(obj *lifecycle.EvictionRequest) {
		obj.Spec.Intent = intent
	}
}

func mkValidEvictionRequestStatus(responders int, tweaks ...func(obj *lifecycle.EvictionRequestStatus)) *lifecycle.EvictionRequestStatus {
	obj := lifecycle.EvictionRequestStatus{
		ObservedGeneration: new(int64(1)),
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return &obj
}
func addCondition(clock utilsclock.PassiveClock, name lifecycle.EvictionConditionType, status bool) func(obj *lifecycle.EvictionRequestStatus) {
	return func(obj *lifecycle.EvictionRequestStatus) {
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
func addConditionsCount(clock utilsclock.PassiveClock, count int) func(obj *lifecycle.EvictionRequestStatus) {
	return func(obj *lifecycle.EvictionRequestStatus) {
		for i := range count {
			addCondition(clock, lifecycle.EvictionConditionType(fmt.Sprintf("Condition%d", i)), true)(obj)
		}
	}
}

func setObservedGeneration(generation *int64) func(obj *lifecycle.EvictionRequestStatus) {
	return func(obj *lifecycle.EvictionRequestStatus) {
		obj.ObservedGeneration = generation
	}
}
