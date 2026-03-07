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
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/coordination"
	"k8s.io/kubernetes/pkg/apis/core"
	utilsclock "k8s.io/utils/clock"
	testing2 "k8s.io/utils/clock/testing"

	// Ensure all API groups are registered with the scheme
	_ "k8s.io/kubernetes/pkg/apis/coordination/install"
)

const valiUIDdName = "a2ee91f4-e13c-44db-9edc-4240e7383ab9"

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
		"name target uid mismatch": {
			input: mkValidEvictionRequest(1, setName("4fa67f6f-da60-4748-bccd-1525dab1bfee", "")),
			errors: field.ErrorList{
				field.Forbidden(field.NewPath("metadata", "name"), "must be the same value as spec.target.pod.uid").MarkFromImperative(),
			},
		},
		"name is not valid": {
			input: mkValidEvictionRequest(1, setName("invalid-name-test", "")),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), "invalid-name-test", "must be UUID in RFC 4122 normalized form, `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` with lowercase hexadecimal characters").MarkFromImperative(),
				field.Forbidden(field.NewPath("metadata", "name"), "must be the same value as spec.target.pod.uid").MarkFromImperative(),
			},
		},
		"missing namespace": {
			input: mkValidEvictionRequest(1, setNamespace("")),
			errors: field.ErrorList{
				field.Required(field.NewPath("metadata", "namespace"), "").MarkFromImperative(),
			},
		},
		"generateName not supported": {
			input: mkValidEvictionRequest(1, setName("", "invalid-generate-name")),
			errors: field.ErrorList{
				field.Forbidden(field.NewPath("metadata", "generateName"), "").MarkAlpha(),
				field.Forbidden(field.NewPath("metadata", "name"), "must be the same value as spec.target.pod.uid").MarkFromImperative(),
				field.Required(field.NewPath("metadata", "name"), "name or generateName is required").MarkFromImperative(),
			},
		},
		"generateName with name not supported": {
			input: mkValidEvictionRequest(1, setName(valiUIDdName, "invalid-generate-name")),
			errors: field.ErrorList{
				field.Forbidden(field.NewPath("metadata", "generateName"), "").MarkAlpha(),
			},
		},
		"missing target": {
			input: mkValidEvictionRequest(1, clearTarget()),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "invalid-name-test", "must specify one of: `pod`").WithOrigin("union").MarkAlpha(),
			},
		},
		"missing target name": {
			input: mkValidEvictionRequest(1, setTarget("", valiUIDdName)),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "target", "pod", "name"), "").MarkAlpha(),
			},
		},
		"missing target uid": {
			input: mkValidEvictionRequest(1, setTarget("bar", "")),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "target", "pod", "uid"), "").MarkAlpha(),
			},
		},
		"invalid target uid": {
			input: mkValidEvictionRequest(1, setTarget("bar", "invalid-uid")),
			errors: field.ErrorList{
				field.Forbidden(field.NewPath("metadata", "name"), "must be the same value as spec.target.pod.uid").MarkFromImperative(),
				field.Invalid(field.NewPath("spec", "target", "pod", "uid"), "invalid-uid", "must be UUID in RFC 4122 normalized form, `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` with lowercase hexadecimal characters").MarkFromImperative(),
			},
		},
		"requesters are required": {
			input: mkValidEvictionRequest(0),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "requesters"), "must have at least one requester on EvictionRequest creation").MarkFromImperative(),
			},
		},
		"too many requesters": {
			input: mkValidEvictionRequest(101),
			errors: field.ErrorList{
				field.TooMany(field.NewPath("spec", "requesters"), 101, 100).WithOrigin("maxItems").MarkAlpha(),
			},
		},
		"duplicate requesters": {
			input: mkValidEvictionRequest(3, addRequesters("foo.example.com", "foo.example.com")),
			errors: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "requesters").Index(4), "").MarkAlpha(),
			},
		},
		"requester without a name": {
			input: mkValidEvictionRequest(0, addRequesters("")),
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "requesters").Index(0).Child("name"), "").MarkAlpha(),
			},
		},
		"invalid requester, 2 segments": {
			input: mkValidEvictionRequest(1, addRequesters("example.com")),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "requesters").Index(1).Child("name"), "example.com", "should be a domain with at least three segments separated by dots").MarkFromImperative(),
			},
		},
		"invalid requester, reserved domain": {
			input: mkValidEvictionRequest(1, addRequesters("requester.k8s.io")),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "requesters").Index(1).Child("name"), "requester.k8s.io", "domain names *.k8s.io, *.kubernetes.io are reserved").MarkFromImperative(),
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
				field.Invalid(field.NewPath("spec", "target"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkAlpha(),
			},
		},
		"change target name": {
			oldInput: mkValidEvictionRequest(1),
			input:    mkValidEvictionRequest(1, setTarget("change", valiUIDdName)),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkAlpha(),
			},
		},
		"change target uid": {
			oldInput: mkValidEvictionRequest(1),
			input:    mkValidEvictionRequest(1, setTarget("bar", "")),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "target"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkAlpha(),
			},
		},
		"increase requesters in a canceled eviction request": {
			oldInput:          mkValidEvictionRequest(0),
			input:             mkValidEvictionRequest(1),
			requiresAuthCheck: true,
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "requesters"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkFromImperative(),
			},
		},
		"add too many requesters": {
			oldInput:          mkValidEvictionRequest(1),
			input:             mkValidEvictionRequest(101),
			requiresAuthCheck: true,
			errors: field.ErrorList{
				field.TooMany(field.NewPath("spec", "requesters"), 101, 100).WithOrigin("maxItems").MarkAlpha(),
			},
		},
		"add a duplicate requesters": {
			oldInput:          mkValidEvictionRequest(3, addRequesters("foo.example.com")),
			input:             mkValidEvictionRequest(3, addRequesters("foo.example.com", "foo.example.com")),
			requiresAuthCheck: true,
			errors: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "requesters").Index(4), "").MarkAlpha(),
			},
		},
		"change to a new requester": {
			oldInput:          mkValidEvictionRequest(1, addRequesters("foo.example.com")),
			input:             mkValidEvictionRequest(0, addRequesters("bar.example.com", "foo.example.com")),
			requiresAuthCheck: true,
			errors:            nil,
		},
		"change and remove a requester": {
			oldInput:          mkValidEvictionRequest(1, addRequesters("foo.example.com")),
			input:             mkValidEvictionRequest(0, addRequesters("bar.example.com")),
			requiresAuthCheck: true,
			errors:            nil,
		},
		"change to a duplicate requester": {
			oldInput:          mkValidEvictionRequest(1, addRequesters("foo.example.com")),
			input:             mkValidEvictionRequest(0, addRequesters("foo.example.com", "foo.example.com")),
			requiresAuthCheck: true,
			errors: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "requesters").Index(1), "").MarkAlpha(),
			},
		},
		"add a requester without a name": {
			oldInput:          mkValidEvictionRequest(1),
			input:             mkValidEvictionRequest(1, addRequesters("")),
			requiresAuthCheck: true,
			errors: field.ErrorList{
				field.Required(field.NewPath("spec", "requesters").Index(1).Child("name"), "").MarkAlpha(),
			},
		},
		"add an invalid requester, 2 segments": {
			oldInput:          mkValidEvictionRequest(1),
			input:             mkValidEvictionRequest(1, addRequesters("example.com")),
			requiresAuthCheck: true,
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "requesters").Index(1).Child("name"), "example.com", "should be a domain with at least three segments separated by dots").MarkFromImperative(),
			},
		},
		"add an invalid requester, reserved domain": {
			oldInput:          mkValidEvictionRequest(1),
			input:             mkValidEvictionRequest(1, addRequesters("requester.k8s.io")),
			requiresAuthCheck: true,
			errors: field.ErrorList{
				field.Invalid(field.NewPath("spec", "requesters").Index(1).Child("name"), "requester.k8s.io", "domain names *.k8s.io, *.kubernetes.io are reserved").MarkFromImperative(),
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
	clockBefore := func(duration time.Duration) utilsclock.PassiveClock {
		return testing2.NewFakePassiveClock(clock.Now().Add(-duration))
	}
	clockAfter := func(duration time.Duration) utilsclock.PassiveClock {
		return testing2.NewFakePassiveClock(clock.Now().Add(duration))
	}
	clock2 := clockAfter(5 * time.Second)

	testCases := map[string]struct {
		input    *coordination.EvictionRequestStatus
		oldInput *coordination.EvictionRequestStatus
		errors   field.ErrorList
	}{
		"clear generation": {
			oldInput: mkValidEvictionRequestStatus(0, setObservedGeneration(1)),
			input:    mkValidEvictionRequestStatus(0, setObservedGeneration(0)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "observedGeneration"), 0, "cannot decrement, must be greater than or equal to 1").MarkFromImperative(),
			},
		},
		"decrease generation": {
			oldInput: mkValidEvictionRequestStatus(0, setObservedGeneration(5)),
			input:    mkValidEvictionRequestStatus(0, setObservedGeneration(4)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "observedGeneration"), 4, "cannot decrement, must be greater than or equal to 5").MarkFromImperative(),
			},
		},
		"decrease generation to negative": {
			oldInput: mkValidEvictionRequestStatus(0, setObservedGeneration(5)),
			input:    mkValidEvictionRequestStatus(0, setObservedGeneration(-1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "observedGeneration"), -1, "must be greater than or equal to 0").WithOrigin("minimum").MarkAlpha(),
			},
		},
		// all interceptors
		"immutable interceptor fields when targetInterceptors is missing": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatusWithStatuses(0, 3, addActiveInterceptors(interceptorName(1)), addProcessedInterceptorsCount(2)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "activeInterceptors"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkFromImperative(),
				field.Invalid(field.NewPath("status", "processedInterceptors"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkFromImperative(),
				field.Invalid(field.NewPath("status", "interceptors"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkFromImperative(),
			},
		},
		// targetInterceptors
		"too many targetInterceptors": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatusWithStatuses(17, 0),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "targetInterceptors"), 17, 16).WithOrigin("maxItems").MarkAlpha(),
				field.Required(field.NewPath("status", "interceptors"), "").MarkFromImperative(),
			},
		},
		"invalid targetInterceptors and required status interceptors": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(0, addTargetInterceptors("f.ba.com", "f.ba.com", "", "invalid", "foo.k8s.io", "example.com", "foo.example.com/bar")),
			errors: []*field.Error{
				field.Duplicate(field.NewPath("status", "targetInterceptors").Index(1), "").MarkAlpha(),
				field.Required(field.NewPath("status", "targetInterceptors").Index(2).Child("name"), "").MarkFromImperative(),
				field.Invalid(field.NewPath("status", "targetInterceptors").Index(3).Child("name"), "invalid", "should be a domain with at least three segments separated by dots").MarkFromImperative(),
				field.Invalid(field.NewPath("status", "targetInterceptors").Index(5).Child("name"), "example.com", "should be a domain with at least three segments separated by dots").MarkFromImperative(),
				field.Invalid(field.NewPath("status", "targetInterceptors").Index(6).Child("name"), "foo.example.com/bar", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')").MarkFromImperative(),
				field.Required(field.NewPath("status", "interceptors"), "").MarkFromImperative(),
			},
		},
		"immutable targetInterceptors - remove": {
			oldInput: mkValidEvictionRequestStatus(5),
			input:    mkValidEvictionRequestStatus(4),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetInterceptors"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkFromImperative(),
				field.Invalid(field.NewPath("status", "interceptors"), "", "should be the same length as status.targetInterceptors and contain the same keys in the same order").MarkFromImperative(),
			},
		},
		"immutable targetInterceptors - add": {
			oldInput: mkValidEvictionRequestStatus(4),
			input:    mkValidEvictionRequestStatus(5),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetInterceptors"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkFromImperative(),
				field.Invalid(field.NewPath("status", "interceptors"), "", "should be the same length as status.targetInterceptors and contain the same keys in the same order").MarkFromImperative(),
			},
		},
		// activeInterceptors
		"no new activeInterceptors are allowed when the eviction request has evicted status": {
			oldInput: mkValidEvictionRequestStatus(1),
			input: mkValidEvictionRequestStatus(1,
				addActiveInterceptors(interceptorName(0)),
				addCondition(clock, coordination.EvictionRequestConditionEvicted, true)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "activeInterceptors"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkFromImperative(),
			},
		},
		"no new activeInterceptors are allowed when the eviction request has canceled status": {
			oldInput: mkValidEvictionRequestStatus(1),
			input: mkValidEvictionRequestStatus(1,
				addActiveInterceptors(interceptorName(0)),
				addCondition(clock, coordination.EvictionRequestConditionCanceled, true)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "activeInterceptors"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkFromImperative(),
			},
		},
		"invalid activeInterceptors as all interceptors have been processed": {
			oldInput: mkValidEvictionRequestStatus(2,
				addProcessedInterceptorsCount(2),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(0)),
				addProcessedInterceptorsCount(2),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "activeInterceptors"), "must not be set because all interceptors have been processed").MarkFromImperative(),
			},
		},
		"activeInterceptors items cannot be removed unless processed": {
			oldInput: mkValidEvictionRequestStatus(3, addActiveInterceptors(interceptorName(0)), setInterceptorsFullStatus(clock, clock2, 0, 1)),
			input:    mkValidEvictionRequestStatus(3, addActiveInterceptors(), setInterceptorsFullStatus(clock, clock2, 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "activeInterceptors"), "items cannot be removed, unless they are added to status.processedInterceptors").MarkFromImperative(),
			},
		},
		"too many activeInterceptors": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(5, addActiveInterceptorsCount(2)),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "activeInterceptors"), 2, 1).WithOrigin("maxItems").MarkAlpha(),
			},
		},
		"duplicate activeInterceptors -short circuited by too many": {
			oldInput: mkValidEvictionRequestStatus(3),
			input:    mkValidEvictionRequestStatus(3, addActiveInterceptors(interceptorName(0), interceptorName(0))),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "activeInterceptors"), 2, 1).WithOrigin("maxItems").MarkAlpha(),
			},
		},
		"invalid activeInterceptors": {
			oldInput: mkValidEvictionRequestStatus(3),
			input:    mkValidEvictionRequestStatus(3, addActiveInterceptors("invalid")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "activeInterceptors").Index(0), "", "is not a valid interceptor from status.targetInterceptors").MarkFromImperative(),
			},
		},
		"set activeInterceptors to old processed interceptor": {
			oldInput: mkValidEvictionRequestStatus(3,
				addProcessedInterceptorsCount(1),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionRequestStatus(3,
				addActiveInterceptors(interceptorName(0)),
				addProcessedInterceptorsCount(1),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "activeInterceptors").Index(0), "must be \"interceptor-1.example.com\" because this interceptor is next in line").MarkFromImperative(),
			},
		},
		"out of order activeInterceptors": {
			oldInput: mkValidEvictionRequestStatus(3),
			input:    mkValidEvictionRequestStatus(3, addActiveInterceptors(interceptorName(1))),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "activeInterceptors").Index(0), "must be \"interceptor-0.example.com\" because this interceptor is next in line").MarkFromImperative(),
			},
		},
		"out of order activeInterceptors with one interceptor processed": {
			oldInput: mkValidEvictionRequestStatus(3,
				addProcessedInterceptorsCount(1),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionRequestStatus(3,
				addActiveInterceptors(interceptorName(2)),
				addProcessedInterceptorsCount(1),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "activeInterceptors").Index(0), "must be \"interceptor-1.example.com\" because this interceptor is next in line").MarkFromImperative(),
			},
		},
		// processedInterceptors
		"removing processedInterceptors items": {
			oldInput: mkValidEvictionRequestStatus(3,
				addProcessedInterceptors(interceptorName(0), interceptorName(1)),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			input: mkValidEvictionRequestStatus(3,
				addProcessedInterceptors(interceptorName(0)),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors"), "items cannot be removed").MarkFromImperative(),
			},
		},
		"adding processedInterceptors items can not be done in bulk": {
			oldInput: mkValidEvictionRequestStatus(3,
				addActiveInterceptorsCount(2),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			input: mkValidEvictionRequestStatus(3,
				addProcessedInterceptorsCount(2),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors"), "items can only be added one at a time").MarkFromImperative(),
			},
		},
		"too many processedInterceptors": {
			oldInput: mkValidEvictionRequestStatus(16,
				addActiveInterceptorsCount(16),
				addProcessedInterceptorsCount(16),
				setInterceptorsStartTime(clock, 0, 16),
				setInterceptorsCompletionTime(clock, 0, 16)),
			input: mkValidEvictionRequestStatus(16,
				addProcessedInterceptorsCount(17),
				setInterceptorsStartTime(clock, 0, 16),
				setInterceptorsCompletionTime(clock, 0, 16)),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "processedInterceptors"), 17, 16).WithOrigin("maxItems").MarkAlpha(),
			},
		},
		"duplicate processedInterceptors": {
			oldInput: mkValidEvictionRequestStatus(3,
				addActiveInterceptors(interceptorName(0)),
				addProcessedInterceptors(interceptorName(0)),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionRequestStatus(3,
				addProcessedInterceptors(interceptorName(0), interceptorName(0)),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			errors: []*field.Error{
				field.Duplicate(field.NewPath("status", "processedInterceptors").Index(1), "").MarkAlpha(),
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(1), "is immutable because a \"status.interceptors[1]\" does not have a matching name").MarkFromImperative(),
			},
		},
		"old processedInterceptors are immutable": {
			oldInput: mkValidEvictionRequestStatus(3,
				addProcessedInterceptorsCount(1),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			input: mkValidEvictionRequestStatus(3,
				addProcessedInterceptors(interceptorName(1)),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "processedInterceptors").Index(0), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkFromImperative(),
			},
		},
		"invalid processedInterceptors - must be a target interceptor": {
			oldInput: mkValidEvictionRequestStatusWithStatuses(1, 0,
				addActiveInterceptors("invalid"),
				addInterceptorStatusesNames("invalid"),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatusWithStatuses(1, 0,
				addProcessedInterceptors("invalid"),
				addInterceptorStatusesNames("invalid"),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(0), "is not a valid interceptor from status.targetInterceptors").MarkFromImperative(),
				field.Invalid(field.NewPath("status", "interceptors"), "", "should contain the same keys in the same order as status.targetInterceptors").MarkFromImperative(),
			},
		},
		"processedInterceptors require a status status.interceptors": {
			oldInput: mkValidEvictionRequestStatusWithStatuses(1, 0,
				addActiveInterceptorsCount(1)),
			input: mkValidEvictionRequestStatusWithStatuses(1, 0,
				addProcessedInterceptorsCount(1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(0), "is immutable because a \"interceptor-0.example.com\" has to be tracked in status.interceptors first").MarkFromImperative(),
				field.Required(field.NewPath("status", "interceptors"), "").MarkFromImperative(),
			},
		},
		"processedInterceptors must have a status with the same name and status interceptors should contain the same keys in the same order": {
			oldInput: mkValidEvictionRequestStatusWithStatuses(3, 0,
				addActiveInterceptors(interceptorName(1)),
				addProcessedInterceptorsCount(1),
				addInterceptorStatusesNames(interceptorName(0), interceptorName(2), interceptorName(5)),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionRequestStatusWithStatuses(3, 0,
				addProcessedInterceptorsCount(2),
				addInterceptorStatusesNames(interceptorName(0), interceptorName(2), interceptorName(5)),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors"), "", "should contain the same keys in the same order as status.targetInterceptors").MarkFromImperative(),
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(1), "is immutable because a \"status.interceptors[1]\" does not have a matching name").MarkFromImperative(),
			},
		},
		"new processedInterceptors must be moved from activeInterceptors": {
			oldInput: mkValidEvictionRequestStatus(3,
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionRequestStatus(3,
				addProcessedInterceptorsCount(1),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(0), "\"interceptor-0.example.com\" should have been active and present in status.activeInterceptors").MarkFromImperative(),
			},
		},
		"processedInterceptors must have a status with start time": {
			oldInput: mkValidEvictionRequestStatus(1,
				addActiveInterceptorsCount(1)),
			input: mkValidEvictionRequestStatus(1,
				addProcessedInterceptorsCount(1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(0), "is immutable because \"interceptor-0.example.com\" interceptor should have status.interceptors[0].startTime").MarkFromImperative(),
			},
		},
		"processedInterceptors must exceed the deadline (start time fallback) before it is marked processed": {
			oldInput: mkValidEvictionRequestStatus(1,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(1,
				addProcessedInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(0), "is immutable because \"interceptor-0.example.com\" interceptor is in progress and it should report status.interceptors[0].heartbeatTime or status.interceptors[0].completionTime").MarkFromImperative(),
			},
		},
		"processedInterceptors must exceed the deadline before it is marked processed": {
			oldInput: mkValidEvictionRequestStatus(1,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(1,
				addProcessedInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(0), "is immutable because \"interceptor-0.example.com\" interceptor is in progress and it should report status.interceptors[0].heartbeatTime or status.interceptors[0].completionTime").MarkFromImperative(),
			},
		},
		// interceptors
		"required status interceptors when targetInterceptors are set": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatusWithStatuses(1, 0),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "interceptors"), "").MarkFromImperative(),
			},
		},
		"status interceptors should contain the same keys in the same order - different length": {
			oldInput: mkValidEvictionRequestStatus(0),
			input: mkValidEvictionRequestStatusWithStatuses(2, 0,
				addInterceptorStatusesNames(interceptorName(1))),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors"), "", "should be the same length as status.targetInterceptors and contain the same keys in the same order").MarkFromImperative(),
			},
		},
		"status interceptors should contain the same keys in the same order": {
			oldInput: mkValidEvictionRequestStatus(0),
			input: mkValidEvictionRequestStatusWithStatuses(2, 0,
				addInterceptorStatusesNames(interceptorName(1), interceptorName(0))),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors"), "", "should contain the same keys in the same order as status.targetInterceptors").MarkFromImperative(),
			},
		},
		"status interceptors cannot remove an item - target interceptors are immutable": {
			oldInput: mkValidEvictionRequestStatus(5),
			input:    mkValidEvictionRequestStatusWithStatuses(5, 4),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors"), "", "should be the same length as status.targetInterceptors and contain the same keys in the same order").MarkFromImperative(),
			},
		},
		"too many status interceptors": {
			oldInput: mkValidEvictionRequestStatusWithStatuses(17, 16),
			input:    mkValidEvictionRequestStatus(17),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "interceptors"), 17, 16).WithOrigin("maxItems").MarkAlpha(),
			},
		},
		// "duplicate status interceptors ": short circuited by targetInterceptors key order
		"status interceptors cannot be mutated unless present in active interceptors": {
			oldInput: mkValidEvictionRequestStatus(5),
			input: mkValidEvictionRequestStatus(5,
				setInterceptorsHeartBeatTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkFromImperative(),
			},
		},
		// status interceptor name
		"status interceptors cannot mutate name when active - short circuited by targetInterceptors key order": {
			oldInput: mkValidEvictionRequestStatus(2, addActiveInterceptorsCount(1), setInterceptorsStartTime(clock, 0, 2)),
			input: mkValidEvictionRequestStatusWithStatuses(2, 0,
				addActiveInterceptorsCount(1),
				addInterceptorStatusesNames(interceptorName(1), interceptorName(0)),
				setInterceptorsStartTime(clock, 0, 2)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors"), "", "should contain the same keys in the same order as status.targetInterceptors").MarkFromImperative(),
			},
		},
		// "required status interceptors names": short circuited by targetInterceptors key order
		"invalid status interceptors names - short circuited by targetInterceptors key order": {
			oldInput: mkValidEvictionRequestStatus(0),
			input: mkValidEvictionRequestStatusWithStatuses(1, 0,
				addInterceptorStatusesNames("foo.k8s.io")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors"), "", "should contain the same keys in the same order as status.targetInterceptors").MarkFromImperative(),
			},
		},
		// startTime
		"startTime cannot be removed once set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkFromImperative(),
			},
		},
		"startTime cannot be changed once set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockAfter(15*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkFromImperative(),
			},
		},
		"startTime is required for an active interceptor": {
			oldInput: mkValidEvictionRequestStatus(2),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "is required for an active interceptor").MarkFromImperative(),
			},
		},
		"startTime must be set to the present time with skew; too old": {
			oldInput: mkValidEvictionRequestStatus(2),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "", "should be set to the present time").MarkFromImperative(),
			},
		},
		"startTime must be set to the present time with skew; too new": {
			oldInput: mkValidEvictionRequestStatus(2),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockAfter(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "", "should be set to the present time").MarkFromImperative(),
			},
		},
		// heartbeatTime
		"cannot remove heartbeatTime when previously set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsHeartBeatTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "interceptors").Index(0).Child("heartbeatTime"), "is required once set").MarkFromImperative(),
			},
		},
		"heartbeatTime cannot be set before startTime is set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsHeartBeatTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "is required for an active interceptor").MarkFromImperative(),
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("heartbeatTime"), "", "cannot be set before status.interceptors[0].startTime is set").MarkFromImperative(),
			},
		},
		"heartbeatTime cannot be decreased": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Second), 0, 1),
				setInterceptorsHeartBeatTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Second), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("heartbeatTime"), "", "cannot be decreased").MarkFromImperative(),
			},
		},
		"heartbeatTime cannot be set before startTime even with a skew": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("heartbeatTime"), "", "must occur after status.interceptors[0].startTime").MarkFromImperative(),
			},
		},
		"heartbeatTime must be updated after 1m at the earliest": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(59*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("heartbeatTime"), "", "there must be at least 1m0s increments during subsequent updates").MarkFromImperative(),
			},
		},
		"heartbeatTime must be set to the present time with skew": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(19*time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockAfter(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("heartbeatTime"), "", "should be set to the present time").MarkFromImperative(),
			},
		},
		// expectedCompletionTime
		"expectedCompletionTime cannot be set before startTime is set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsExpectedCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "is required for an active interceptor").MarkFromImperative(),
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("expectedCompletionTime"), "", "cannot be set before status.interceptors[0].startTime is set").MarkFromImperative(),
			},
		},
		"expectedCompletionTime cannot be set before startTime even with a skew": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsExpectedCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("expectedCompletionTime"), "", "must occur after status.interceptors[0].startTime").MarkFromImperative(),
			},
		},
		"expectedCompletionTime cannot be set to the past with skew": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(32*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(32*time.Second), 0, 1),
				setInterceptorsExpectedCompletionTime(clockBefore(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("expectedCompletionTime"), "", "cannot be set to the past time").MarkFromImperative(),
			},
		},
		"expectedCompletionTime must complete within 10 years": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(time.Minute), 0, 1),
				setInterceptorsExpectedCompletionTime(clockAfter(time.Hour*24*365*10+time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("expectedCompletionTime"), "", "must complete within 10 years").MarkFromImperative(),
			},
		},
		// completionTime
		"completionTime cannot be changed once set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clockAfter(5*time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clockAfter(4*time.Minute), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("completionTime"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkFromImperative(),
			},
		},
		"completionTime cannot be removed once set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clockAfter(5*time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("completionTime"), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable").MarkFromImperative(),
			},
		},
		"completionTime cannot be set before startTime is set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "is required for an active interceptor").MarkFromImperative(),
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("completionTime"), "", "cannot be set before status.interceptors[0].startTime is set").MarkFromImperative(),
			},
		},
		"completionTime cannot be set before startTime even with a skew": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("completionTime"), "", "must occur after status.interceptors[0].startTime").MarkFromImperative(),
			},
		},
		"completionTime must be set to the present time with skew; too new": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clockAfter(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("completionTime"), "", "should be set to the present time").MarkFromImperative(),
			},
		},
		"completionTime must be set to the present time with skew; too old": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(1*time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(1*time.Minute), 0, 1),
				setInterceptorsCompletionTime(clockBefore(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("completionTime"), "", "should be set to the present time").MarkFromImperative(),
			},
		},
		// conditions
		"duplicate condition": {
			oldInput: mkValidEvictionRequestStatus(1),
			input:    mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionEvicted, true), addCondition(clock, coordination.EvictionRequestConditionEvicted, true)),
			errors: []*field.Error{
				field.Duplicate(field.NewPath("status", "conditions").Index(1), "").MarkAlpha(),
			},
		},
		"add invalid condition": {
			oldInput: mkValidEvictionRequestStatus(1),
			input: mkValidEvictionRequestStatus(1, func(obj *coordination.EvictionRequestStatus) {
				obj.Conditions = append(obj.Conditions, metav1.Condition{
					Type:               "-bad-name",
					Status:             "invalid",
					ObservedGeneration: -1,
					Reason:             "-Reason",
				})
			}),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions").Index(0).Child("type"), "", "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')").WithOrigin("format=k8s-label-key").MarkFromImperative(),
				field.NotSupported(field.NewPath("status", "conditions").Index(0).Child("status"), "", []string{"False", "True", "Unknown"}).MarkFromImperative(),
				field.Invalid(field.NewPath("status", "conditions").Index(0).Child("observedGeneration"), "", "must be greater than or equal to zero").MarkFromImperative(),
				field.Required(field.NewPath("status", "conditions").Index(0).Child("lastTransitionTime"), "").MarkFromImperative(),
				field.Invalid(field.NewPath("status", "conditions").Index(0).Child("reason"), "", "a condition reason must start with alphabetic character, optionally followed by a string of alphanumeric characters or '_,:', and must end with an alphanumeric character or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName',  or 'ReasonA,ReasonB',  or 'ReasonA:ReasonB', regex used for validation is '[A-Za-z]([A-Za-z0-9_,:]*[A-Za-z0-9_])?')").MarkFromImperative(),
			},
		},
		"Evicted condition cannot be removed": {
			oldInput: mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionEvicted, true)),
			input:    mkValidEvictionRequestStatus(1),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Evicted condition is immutable").MarkFromImperative(),
			},
		},
		"Evicted condition cannot be changed": {
			oldInput: mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionEvicted, true)),
			input:    mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionEvicted, false)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Evicted condition is immutable").MarkFromImperative(),
			},
		},
		"Canceled condition cannot be removed": {
			oldInput: mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionCanceled, true)),
			input:    mkValidEvictionRequestStatus(1),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Canceled condition is immutable").MarkFromImperative(),
			},
		},
		"Canceled condition cannot be changed": {
			oldInput: mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionCanceled, true)),
			input:    mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionCanceled, false)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Canceled condition is immutable").MarkFromImperative(),
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
		ObjectMeta: metav1.ObjectMeta{Name: valiUIDdName, Namespace: "foo"},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.LocalTargetReference{
					UID:  valiUIDdName,
					Name: "foo.pod",
				},
			},
		},
	}
	for i := range requesters {
		obj.Spec.Requesters = append(obj.Spec.Requesters, coordination.Requester{
			Name: fmt.Sprintf("requester-%d.example.com", i),
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

func setNamespace(namespace string) func(obj *coordination.EvictionRequest) {
	return func(obj *coordination.EvictionRequest) {
		obj.Namespace = namespace
	}
}

func clearTarget() func(obj *coordination.EvictionRequest) {
	return func(obj *coordination.EvictionRequest) {
		obj.Spec.Target.Pod = nil
	}
}

func setTarget(name, uid string) func(obj *coordination.EvictionRequest) {
	return func(obj *coordination.EvictionRequest) {
		obj.Spec.Target.Pod = &coordination.LocalTargetReference{
			UID:  uid,
			Name: name,
		}
	}
}
func addRequesters(names ...string) func(obj *coordination.EvictionRequest) {
	return func(obj *coordination.EvictionRequest) {
		for _, name := range names {
			obj.Spec.Requesters = append(obj.Spec.Requesters, coordination.Requester{Name: name})
		}
	}
}

func interceptorName(i int) string {
	return fmt.Sprintf("interceptor-%d.example.com", i)
}
func mkValidEvictionRequestStatus(interceptors int, tweaks ...func(obj *coordination.EvictionRequestStatus)) *coordination.EvictionRequestStatus {
	return mkValidEvictionRequestStatusWithStatuses(interceptors, interceptors, tweaks...)
}
func mkValidEvictionRequestStatusWithStatuses(interceptors, statuses int, tweaks ...func(obj *coordination.EvictionRequestStatus)) *coordination.EvictionRequestStatus {
	obj := coordination.EvictionRequestStatus{
		ObservedGeneration: 1,
	}
	for i := range interceptors {
		obj.TargetInterceptors = append(obj.TargetInterceptors, core.EvictionInterceptor{
			Name: interceptorName(i),
		})
	}
	for i := range statuses {
		obj.Interceptors = append(obj.Interceptors, coordination.InterceptorStatus{
			Name: interceptorName(i),
		})
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

func setObservedGeneration(generation int64) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		obj.ObservedGeneration = generation
	}
}
func addTargetInterceptors(interceptors ...string) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for _, interceptorName := range interceptors {
			obj.TargetInterceptors = append(obj.TargetInterceptors, core.EvictionInterceptor{Name: interceptorName})
		}
	}
}
func addActiveInterceptors(interceptors ...string) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		obj.ActiveInterceptors = append(obj.ActiveInterceptors, interceptors...)
	}
}
func addActiveInterceptorsCount(count int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := range count {
			obj.ActiveInterceptors = append(obj.ActiveInterceptors, interceptorName(i))
		}
	}
}
func addProcessedInterceptors(interceptors ...string) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		obj.ProcessedInterceptors = append(obj.ProcessedInterceptors, interceptors...)
	}
}
func addProcessedInterceptorsCount(count int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := range count {
			obj.ProcessedInterceptors = append(obj.ProcessedInterceptors, interceptorName(i))
		}
	}
}
func addInterceptorStatusesNames(interceptors ...string) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for _, interceptor := range interceptors {
			obj.Interceptors = append(obj.Interceptors, coordination.InterceptorStatus{Name: interceptor})
		}
	}
}
func setInterceptorsStartTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := from; i < to; i++ {
			obj.Interceptors[i].StartTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setInterceptorsHeartBeatTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := from; i < to; i++ {
			obj.Interceptors[i].HeartbeatTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setInterceptorsExpectedCompletionTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := from; i < to; i++ {
			obj.Interceptors[i].ExpectedCompletionTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setInterceptorsCompletionTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := from; i < to; i++ {
			obj.Interceptors[i].CompletionTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setInterceptorsMessage(from, to int, suffixes ...string) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := from; i < to; i++ {
			obj.Interceptors[i].Message = fmt.Sprintf("message %d", i)
			for _, suffix := range suffixes {
				obj.Interceptors[i].Message += suffix
			}
		}
	}
}
func setInterceptorsFullStatus(startTimeClock, clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		setInterceptorsStartTime(startTimeClock, from, to)(obj)
		setInterceptorsHeartBeatTime(clock, from, to)(obj)
		setInterceptorsExpectedCompletionTime(clock, from, to)(obj)
		setInterceptorsCompletionTime(clock, from, to)(obj)
		setInterceptorsMessage(from, to, clock.Now().String())(obj)
	}
}
