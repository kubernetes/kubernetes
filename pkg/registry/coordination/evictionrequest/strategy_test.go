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
	"strings"
	"testing"
	"time"

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/coordination"
	api "k8s.io/kubernetes/pkg/apis/core"
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
	if "test" != strategy.GenerateName("test") {
		t.Errorf("EvictionRequest should not implement name generation")
	}
	if len(strategy.WarningsOnCreate(nil, nil)) != 0 {
		t.Errorf("EvictionRequest warnings on create are expected to be empty")
	}

	evictionRequest := &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: valiUIDdName, Namespace: "foo"},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.LocalTargetReference{
					UID:  valiUIDdName,
					Name: "foo.pod",
				},
			},
			Requesters: []coordination.Requester{
				{Name: "requester.example.com"},
			},
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: 1,
			TargetInterceptors: []api.EvictionInterceptor{
				{Name: "test"},
			},
		},
	}

	strategy.PrepareForCreate(ctx, evictionRequest)
	if evictionRequest.Generation != int64(1) {
		t.Error("EvictionRequest metadata.generation should be set to 1")
	}
	if len(evictionRequest.Status.TargetInterceptors) != 0 {
		t.Error("EvictionRequest should not allow setting status.targetInterceptors on create")
	}
	if evictionRequest.Status.ObservedGeneration != int64(0) {
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
				ObjectMeta: metav1.ObjectMeta{Name: valiUIDdName, Namespace: "foo"},
				Spec: coordination.EvictionRequestSpec{
					Target: coordination.EvictionTarget{
						Pod: &coordination.LocalTargetReference{
							UID:  valiUIDdName,
							Name: "foo.pod",
						},
					},
					Requesters: []coordination.Requester{
						{Name: "requester.example.com"},
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

	if len(strategy.WarningsOnUpdate(nil, nil, nil)) != 0 {
		t.Errorf("EvictionRequest warnings on update are expected to be empty")
	}

	oldEvictionRequest := &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: valiUIDdName, Namespace: "foo", Generation: 1, ResourceVersion: "2"},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.LocalTargetReference{
					UID:  valiUIDdName,
					Name: "foo.pod",
				},
			},
			Requesters: []coordination.Requester{
				{Name: "requester.example.com"},
			},
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: 1,
		},
	}

	newEvictionRequest := &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: valiUIDdName, Namespace: "foo", ResourceVersion: "2"},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.LocalTargetReference{
					UID:  valiUIDdName,
					Name: "bar.pod",
				},
			},
			Requesters: []coordination.Requester{
				{Name: "requester-1.example.com"},
				{Name: "requester-2.example.com"},
			},
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: 10,
			TargetInterceptors: []api.EvictionInterceptor{
				{Name: "test"},
			},
		},
	}

	strategy.PrepareForUpdate(ctx, newEvictionRequest, oldEvictionRequest)
	if newEvictionRequest.Generation != int64(2) {
		t.Error("EvictionRequest metadata.generation should be set to 2")
	}
	if len(newEvictionRequest.Status.TargetInterceptors) != 0 {
		t.Error("EvictionRequest should not allow setting status.targetInterceptors on update")
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
				ObjectMeta: metav1.ObjectMeta{Name: valiUIDdName, Namespace: "foo", Generation: 1, ResourceVersion: "2"},
				Spec: coordination.EvictionRequestSpec{
					Target: coordination.EvictionTarget{
						Pod: &coordination.LocalTargetReference{
							UID:  valiUIDdName,
							Name: "foo.pod",
						},
					},
					Requesters: []coordination.Requester{
						{Name: "requester.example.com"},
					},
				},
			}

			newEvictionRequest := &coordination.EvictionRequest{
				ObjectMeta: metav1.ObjectMeta{Name: valiUIDdName, Namespace: "foo", ResourceVersion: "2"},
				Spec: coordination.EvictionRequestSpec{
					Target: coordination.EvictionTarget{
						Pod: &coordination.LocalTargetReference{
							UID:  valiUIDdName,
							Name: "foo.pod",
						},
					},
					Requesters: []coordination.Requester{
						{Name: "requester-1.example.com"},
						// adding requesters is a privileged operation
						{Name: "requester-2.example.com"},
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
		if !fields.Has(fieldpath.MakePathOrDie("metadata", "labels")) {
			t.Errorf("metadata.labels should be reset on status update")
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
		ObjectMeta: metav1.ObjectMeta{Name: valiUIDdName, Namespace: "foo", Generation: 1, ResourceVersion: "2",
			Annotations: map[string]string{"test": "false"},
			Labels:      map[string]string{"foo": "bar"}},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.LocalTargetReference{
					UID:  valiUIDdName,
					Name: "foo.pod",
				},
			},
			Requesters: []coordination.Requester{
				{Name: "requester.example.com"},
			},
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: 1,
		},
	}

	newEvictionRequest := &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: valiUIDdName, Namespace: "foo", Generation: 1, ResourceVersion: "2",
			Annotations: map[string]string{"test": "true"},
			Labels:      map[string]string{"foo": "baz"}},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.LocalTargetReference{
					UID:  valiUIDdName,
					Name: "foo.pod",
				},
			},
			Requesters: []coordination.Requester{
				{Name: "requester-1.example.com"},
				{Name: "requester-2.example.com"},
			},
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: -5,
			TargetInterceptors: []api.EvictionInterceptor{
				{Name: "interceptor1.example.com"},
			},
			Interceptors: []coordination.InterceptorStatus{
				{Name: "interceptor1.example.com", Message: strings.Repeat("a", 35000)},
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
		t.Error("EvictionRequest should allow changing annotations")
	}
	if len(newEvictionRequest.Status.Interceptors[0].Message) != 4000 {
		t.Error("EvictionRequest should truncate long messages")
	}
	errs := strategy.ValidateUpdate(ctx, newEvictionRequest, oldEvictionRequest)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
	newEvictionRequest.Status.ObservedGeneration = 2
	errs = strategy.ValidateUpdate(ctx, newEvictionRequest, oldEvictionRequest)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
}
