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

	"k8s.io/utils/ptr"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/coordination"
	testing2 "k8s.io/utils/clock/testing"
)

func TestEvictionRequestStrategy_ResetFields(t *testing.T) {
	strategy := NewStrategy(nil, nil)
	for _, fields := range strategy.GetResetFields() {
		if !fields.Has(fieldpath.MakePathOrDie("status")) {
			t.Errorf("status should be reset on creation and update")
		}
	}
}

func TestEvictionRequestStrategy(t *testing.T) {
	clock := testing2.NewFakePassiveClock(time.Now())
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "coordination.k8s.io",
		APIVersion:        "v1alpha1",
		Resource:          "evictionrequests",
		IsResourceRequest: true,
		Verb:              "create",
	})
	ctx = genericapirequest.WithUser(ctx, &user.DefaultInfo{Name: "test"})
	strategy := NewStrategy(&TestDecisionAuthorizer{authorizer.DecisionAllow}, clock)

	if !strategy.NamespaceScoped() {
		t.Errorf("EvictionRequest must be namespace scoped")
	}
	if strategy.AllowCreateOnUpdate() {
		t.Errorf("EvictionRequest should not allow create on update")
	}
	if strategy.GenerateName("test") != "test" {
		t.Errorf("EvictionRequest should not implement name generation")
	}
	if len(strategy.WarningsOnCreate(ctx, nil)) != 0 {
		t.Errorf("EvictionRequest warnings on create are expected to be empty")
	}

	evictionRequest := &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: valiUIDName, Namespace: "foo"},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.PodReference{
					UID:  valiUIDName,
					Name: "foo.pod",
				},
			},
			Requesters: []coordination.Requester{
				{Name: "requester.example.com/bar", Intent: coordination.RequesterIntentEviction},
			},
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: ptr.To[int64](1),
			TargetResponders: []coordination.TargetResponder{
				{Name: "test", State: coordination.ResponderStateInactive},
			},
		},
	}

	strategy.PrepareForCreate(ctx, evictionRequest)
	if evictionRequest.Generation != int64(1) {
		t.Error("EvictionRequest metadata.generation should be set to 1")
	}
	if len(evictionRequest.Status.TargetResponders) != 0 {
		t.Error("EvictionRequest should not allow setting status.targetResponders on create")
	}
	if evictionRequest.Status.ObservedGeneration != nil {
		t.Error("EvictionRequest should not allow setting status.observedGeneration on create")
	}
	errs := strategy.Validate(ctx, evictionRequest)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
}

func TestEvictionRequestStrategy_Unauthorized(t *testing.T) {
	clock := testing2.NewFakePassiveClock(time.Now())
	for _, authDecision := range []authorizer.Decision{authorizer.DecisionDeny, authorizer.DecisionNoOpinion} {
		t.Run(authDecision.String(), func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "coordination.k8s.io",
				APIVersion:        "v1alpha1",
				Resource:          "evictionrequests",
				IsResourceRequest: true,
				Verb:              "create",
			})
			ctx = genericapirequest.WithUser(ctx, &user.DefaultInfo{Name: "test"})
			strategy := NewStrategy(&TestDecisionAuthorizer{authDecision}, clock)

			evictionRequest := &coordination.EvictionRequest{
				ObjectMeta: metav1.ObjectMeta{Name: valiUIDName, Namespace: "foo"},
				Spec: coordination.EvictionRequestSpec{
					Target: coordination.EvictionTarget{
						Pod: &coordination.PodReference{
							UID:  valiUIDName,
							Name: "foo.pod",
						},
					},
					Requesters: []coordination.Requester{
						{Name: "requester.example.com/bar"},
					},
				},
			}

			strategy.PrepareForCreate(ctx, evictionRequest)
			gotErr := strategy.Validate(ctx, evictionRequest)
			expectedErr := field.ErrorList{field.Forbidden(field.NewPath(""), "User \"test\" must have permission to delete pods in \"foo\" namespace when spec.target.pod is set")}
			errOutputMatcher := field.ErrorMatcher{}.ByType().ByField().ByDetailExact()

			errOutputMatcher.Test(t, expectedErr, gotErr)
		})

	}
}

func TestEvictionRequestStrategy_Update(t *testing.T) {
	clock := testing2.NewFakePassiveClock(time.Now())
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "coordination.k8s.io",
		APIVersion:        "v1alpha1",
		Resource:          "evictionrequests",
		IsResourceRequest: true,
		Verb:              "update",
	})
	ctx = genericapirequest.WithUser(ctx, &user.DefaultInfo{Name: "test"})
	strategy := NewStrategy(&TestDecisionAuthorizer{authorizer.DecisionAllow}, clock)

	if len(strategy.WarningsOnUpdate(ctx, nil, nil)) != 0 {
		t.Errorf("EvictionRequest warnings on update are expected to be empty")
	}

	oldEvictionRequest := &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: valiUIDName, Namespace: "foo", Generation: 1, ResourceVersion: "2"},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.PodReference{
					UID:  valiUIDName,
					Name: "foo.pod",
				},
			},
			Requesters: []coordination.Requester{
				{Name: "requester-1.example.com/bar", Intent: coordination.RequesterIntentEviction},
				{Name: "requester-2.example.com/bar", Intent: coordination.RequesterIntentWithdrawn},
			},
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: ptr.To[int64](1),
		},
	}

	newEvictionRequest := &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: valiUIDName, Namespace: "foo", ResourceVersion: "2"},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.PodReference{
					UID:  valiUIDName,
					Name: "bar.pod",
				},
			},
			Requesters: []coordination.Requester{
				{Name: "requester-1.example.com/bar", Intent: coordination.RequesterIntentEviction},
				{Name: "requester-2.example.com/bar", Intent: coordination.RequesterIntentEviction},
			},
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: ptr.To[int64](10),
			TargetResponders: []coordination.TargetResponder{
				{Name: "test", State: coordination.ResponderStateCanceled},
			},
		},
	}

	strategy.PrepareForUpdate(ctx, newEvictionRequest, oldEvictionRequest)
	if newEvictionRequest.Generation != int64(2) {
		t.Error("EvictionRequest metadata.generation should be set to 2")
	}
	if len(newEvictionRequest.Status.TargetResponders) != 0 {
		t.Error("EvictionRequest should not allow setting status.targetResponders on update")
	}
	errs := strategy.ValidateUpdate(ctx, newEvictionRequest, oldEvictionRequest)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
	newEvictionRequest.Spec.Target.Pod.Name = "foo.pod"
	errs = strategy.ValidateUpdate(ctx, newEvictionRequest, oldEvictionRequest)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
}

func TestEvictionRequestStrategy_UpdateUnauthorized(t *testing.T) {
	clock := testing2.NewFakePassiveClock(time.Now())
	for _, authDecision := range []authorizer.Decision{authorizer.DecisionDeny, authorizer.DecisionNoOpinion} {
		t.Run(authDecision.String(), func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "coordination.k8s.io",
				APIVersion:        "v1alpha1",
				Resource:          "evictionrequests",
				IsResourceRequest: true,
				Verb:              "update",
			})
			ctx = genericapirequest.WithUser(ctx, &user.DefaultInfo{Name: "test"})
			strategy := NewStrategy(&TestDecisionAuthorizer{authDecision}, clock)

			oldEvictionRequest := &coordination.EvictionRequest{
				ObjectMeta: metav1.ObjectMeta{Name: valiUIDName, Namespace: "foo", Generation: 1, ResourceVersion: "2"},
				Spec: coordination.EvictionRequestSpec{
					Target: coordination.EvictionTarget{
						Pod: &coordination.PodReference{
							UID:  valiUIDName,
							Name: "foo.pod",
						},
					},
					Requesters: []coordination.Requester{
						{Name: "requester-1.example.com/bar", Intent: coordination.RequesterIntentEviction},
						{Name: "requester-2.example.com/bar", Intent: coordination.RequesterIntentWithdrawn},
					},
				},
			}

			newEvictionRequest := &coordination.EvictionRequest{
				ObjectMeta: metav1.ObjectMeta{Name: valiUIDName, Namespace: "foo", ResourceVersion: "2"},
				Spec: coordination.EvictionRequestSpec{
					Target: coordination.EvictionTarget{
						Pod: &coordination.PodReference{
							UID:  valiUIDName,
							Name: "foo.pod",
						},
					},
					Requesters: []coordination.Requester{
						{Name: "requester-1.example.com/bar", Intent: coordination.RequesterIntentEviction},
						// Eviction intent is a privileged operation
						{Name: "requester-2.example.com/bar", Intent: coordination.RequesterIntentEviction},
					},
				},
			}

			strategy.PrepareForUpdate(ctx, newEvictionRequest, oldEvictionRequest)
			gotErr := strategy.ValidateUpdate(ctx, newEvictionRequest, oldEvictionRequest)
			expectedErr := field.ErrorList{field.Forbidden(field.NewPath("spec", "requesters"), "User \"test\" must have permission to delete pods in \"foo\" namespace when spec.target.pod is set")}
			errOutputMatcher := field.ErrorMatcher{}.ByType().ByField().ByDetailExact()

			errOutputMatcher.Test(t, expectedErr, gotErr)
		})
	}
}

func TestEvictionRequestStatusStrategy_ResetFields(t *testing.T) {
	strategy := NewStrategy(nil, nil)
	statusStrategy := NewStatusStrategy(strategy)
	for _, fields := range statusStrategy.GetResetFields() {
		if !fields.Has(fieldpath.MakePathOrDie("spec")) {
			t.Errorf("spec should be reset on status update")
		}
		if !fields.Has(fieldpath.MakePathOrDie("metadata")) {
			t.Errorf("metadata should be reset on status update")
		}
	}
}

func TestEvictionRequestStatusStrategy(t *testing.T) {
	clock := testing2.NewFakePassiveClock(time.Now())
	strategy := NewStatusStrategy(NewStrategy(nil, clock))
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "coordination.k8s.io",
		APIVersion:        "v1alpha1",
		Resource:          "evictionrequests",
		IsResourceRequest: true,
		Verb:              "update",
		Subresource:       "status",
	})

	oldEvictionRequest := &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: valiUIDName, Namespace: "foo", Generation: 1, ResourceVersion: "2",
			Annotations: map[string]string{"test": "true"},
			Labels:      map[string]string{"foo": "bar"}},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.PodReference{
					UID:  valiUIDName,
					Name: "foo.pod",
				},
			},
			Requesters: []coordination.Requester{
				{Name: "requester.example.com/bar"},
			},
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: ptr.To[int64](1),
		},
	}

	newEvictionRequest := &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: valiUIDName, Namespace: "foo", Generation: 1, ResourceVersion: "2",
			Annotations: map[string]string{"test": "false"},
			Labels:      map[string]string{"foo": "baz"}},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.PodReference{
					UID:  valiUIDName,
					Name: "foo.pod",
				},
			},
			Requesters: []coordination.Requester{
				{Name: "requester-1.example.com/bar"},
				{Name: "requester-2.example.com/bar"},
			},
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: ptr.To[int64](-5),
			TargetResponders: []coordination.TargetResponder{
				{Name: "responder1.example.com/bar", State: coordination.ResponderStateActive},
			},
			Responders: []coordination.ResponderStatus{
				{Name: "responder1.example.com/bar", Message: "test message", StartTime: ptr.To(metav1.Now())},
			},
		},
	}

	strategy.PrepareForUpdate(ctx, newEvictionRequest, oldEvictionRequest)
	if len(newEvictionRequest.Spec.Requesters) != 1 {
		t.Error("EvictionRequest spec.requesters should not be updated and have a length of 1")
	}
	if newEvictionRequest.Labels["foo"] != "bar" {
		t.Error("EvictionRequest should not allow changing labels")
	}
	if newEvictionRequest.Annotations["test"] != "true" {
		t.Error("EvictionRequest should not allow changing annotations")
	}
	errs := strategy.ValidateUpdate(ctx, newEvictionRequest, oldEvictionRequest)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
	newEvictionRequest.Status.ObservedGeneration = ptr.To[int64](2)
	errs = strategy.ValidateUpdate(ctx, newEvictionRequest, oldEvictionRequest)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
}

func TestHasRequestersIntentChangedExcludingWithdrawal(t *testing.T) {
	testCases := map[string]struct {
		requesters                   []coordination.Requester
		oldRequesters                []coordination.Requester
		expectedHasNewEvictionDemand bool
	}{
		// true
		"new eviction requesters": {
			oldRequesters: []coordination.Requester{},
			requesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			expectedHasNewEvictionDemand: true,
		},
		"new mixed requesters": {
			oldRequesters: []coordination.Requester{},
			requesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentWithdrawn},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
				{Name: "3", Intent: coordination.RequesterIntentWithdrawn},
			},
			expectedHasNewEvictionDemand: true,
		},
		"eviction requesters duplicate added": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			requesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			expectedHasNewEvictionDemand: true,
		},
		"eviction requesters added": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			requesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
				{Name: "3", Intent: coordination.RequesterIntentEviction},
			},
			expectedHasNewEvictionDemand: true,
		},
		"clear requesters": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			requesters:                   []coordination.Requester{},
			expectedHasNewEvictionDemand: true,
		},
		"clear and add withdrawn requesters": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			requesters: []coordination.Requester{
				{Name: "3", Intent: coordination.RequesterIntentWithdrawn},
				{Name: "4", Intent: coordination.RequesterIntentWithdrawn},
			},
			expectedHasNewEvictionDemand: true,
		},
		"eviction requesters withdrawn but one added": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			requesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentWithdrawn},
				{Name: "2", Intent: coordination.RequesterIntentWithdrawn},
				{Name: "3", Intent: coordination.RequesterIntentEviction},
			},
			expectedHasNewEvictionDemand: true,
		},
		"eviction requesters removed but one added": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			requesters: []coordination.Requester{
				{Name: "3", Intent: coordination.RequesterIntentEviction},
			},
			expectedHasNewEvictionDemand: true,
		},
		"eviction requesters new unknown intent changed": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			requesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: "new"},
			},
			expectedHasNewEvictionDemand: true,
		},
		"eviction requesters new unknown intent added": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			requesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
				{Name: "3", Intent: "new"},
			},
			expectedHasNewEvictionDemand: true,
		},
		"eviction requesters new unknown empty intent added": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			requesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
				{Name: "3", Intent: ""},
			},
			expectedHasNewEvictionDemand: true,
		},
		"eviction requesters removed and new unknown intent added": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			requesters: []coordination.Requester{
				{Name: "1", Intent: "new"},
			},
			expectedHasNewEvictionDemand: true,
		},
		"eviction requesters withdrawn changed": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentWithdrawn},
			},
			requesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			expectedHasNewEvictionDemand: true,
		},
		"eviction requesters withdrawn changed to new": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentWithdrawn},
			},
			requesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: "new"},
			},
			expectedHasNewEvictionDemand: true,
		},
		// false
		"eviction requesters don't change": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			requesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			expectedHasNewEvictionDemand: false,
		},
		"eviction requesters withdrawn": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			requesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentWithdrawn},
				{Name: "2", Intent: coordination.RequesterIntentWithdrawn},
			},
			expectedHasNewEvictionDemand: false,
		},
		"eviction requesters partially withdrawn": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			requesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentWithdrawn},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			expectedHasNewEvictionDemand: false,
		},
		"eviction requesters new withdrawn added": {
			oldRequesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
			},
			requesters: []coordination.Requester{
				{Name: "1", Intent: coordination.RequesterIntentEviction},
				{Name: "2", Intent: coordination.RequesterIntentEviction},
				{Name: "3", Intent: coordination.RequesterIntentWithdrawn},
			},
			expectedHasNewEvictionDemand: false,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			got := hasRequestersIntentChangedExcludingWithdrawal(tc.requesters, tc.oldRequesters)
			if tc.expectedHasNewEvictionDemand != got {
				t.Errorf("hasRequestersIntentChanged failed: expected=%v, got=%v", tc.expectedHasNewEvictionDemand, got)
			}
		})
	}
}
