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

	"k8s.io/utils/ptr"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/coordination"
)

func TestEvictionRequestStrategy_ResetFields(t *testing.T) {
	strategy := NewStrategy()
	for _, fields := range strategy.GetResetFields() {
		if !fields.Has(fieldpath.MakePathOrDie("status")) {
			t.Errorf("status should be reset on creation and update")
		}
	}
}

func TestEvictionRequestStrategy(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "coordination.k8s.io",
		APIVersion:        "v1alpha1",
		Resource:          "evictionrequests",
		IsResourceRequest: true,
		Verb:              "create",
	})
	ctx = genericapirequest.WithUser(ctx, &user.DefaultInfo{Name: "test"})
	strategy := NewStrategy()

	if !strategy.NamespaceScoped() {
		t.Errorf("EvictionRequest must be namespace scoped")
	}
	if strategy.AllowCreateOnUpdate(ctx) {
		t.Errorf("EvictionRequest should not allow create on update")
	}
	if len(strategy.GenerateName("test")) <= len("test") {
		t.Errorf("Eviction should implement name generation")
	}
	if len(strategy.WarningsOnCreate(ctx, nil)) != 0 {
		t.Errorf("EvictionRequest warnings on create are expected to be empty")
	}

	evictionRequest := &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "foo"},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionRequestTarget{
				Pod: &coordination.EvictionRequestPodReference{
					UID:  validUID,
					Name: "foo.pod",
				},
			},
			RequesterName: "requester.domain/requester1",
			Intent:        coordination.EvictionRequestIntentEviction,
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: ptr.To[int64](1),
		},
	}

	strategy.PrepareForCreate(ctx, evictionRequest)
	if evictionRequest.Generation != int64(1) {
		t.Error("EvictionRequest metadata.generation should be set to 1")
	}
	if evictionRequest.Status.ObservedGeneration != nil {
		t.Error("EvictionRequest should not allow setting status.observedGeneration on create")
	}
	errs := strategy.Validate(ctx, evictionRequest)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
}

func TestEvictionRequestStrategy_Update(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "coordination.k8s.io",
		APIVersion:        "v1alpha1",
		Resource:          "evictionrequests",
		IsResourceRequest: true,
		Verb:              "update",
	})
	ctx = genericapirequest.WithUser(ctx, &user.DefaultInfo{Name: "test"})
	strategy := NewStrategy()

	if len(strategy.WarningsOnUpdate(ctx, nil, nil)) != 0 {
		t.Errorf("EvictionRequest warnings on update are expected to be empty")
	}

	oldEvictionRequest := &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "foo", Generation: 1, ResourceVersion: "2"},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionRequestTarget{
				Pod: &coordination.EvictionRequestPodReference{
					UID:  validUID,
					Name: "foo.pod",
				},
			},

			RequesterName: "requester.domain/requester1",
			Intent:        coordination.EvictionRequestIntentWithdrawn,
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: ptr.To[int64](1),
		},
	}

	newEvictionRequest := &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "foo", ResourceVersion: "2"},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionRequestTarget{
				Pod: &coordination.EvictionRequestPodReference{
					UID:  validUID,
					Name: "bar.pod",
				},
			},
			RequesterName: "requester.domain/requester1",
			Intent:        coordination.EvictionRequestIntentEviction,
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: ptr.To[int64](10),
		},
	}

	strategy.PrepareForUpdate(ctx, newEvictionRequest, oldEvictionRequest)
	if newEvictionRequest.Generation != int64(2) {
		t.Error("EvictionRequest metadata.generation should be set to 2")
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

func TestEvictionRequestStatusStrategy_ResetFields(t *testing.T) {
	strategy := NewStrategy()
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
	strategy := NewStatusStrategy(NewStrategy())
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "coordination.k8s.io",
		APIVersion:        "v1alpha1",
		Resource:          "evictionrequests",
		IsResourceRequest: true,
		Verb:              "update",
		Subresource:       "status",
	})

	oldEvictionRequest := &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "foo", Generation: 1, ResourceVersion: "2",
			Annotations: map[string]string{"test": "true"},
			Labels:      map[string]string{"foo": "bar"}},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionRequestTarget{
				Pod: &coordination.EvictionRequestPodReference{
					UID:  validUID,
					Name: "foo.pod",
				},
			},
			RequesterName: "requester.domain/requester1",
			Intent:        coordination.EvictionRequestIntentEviction,
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: ptr.To[int64](1),
		},
	}

	newEvictionRequest := &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "foo", Generation: 1, ResourceVersion: "2",
			Annotations: map[string]string{"test": "false"},
			Labels:      map[string]string{"foo": "baz"}},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionRequestTarget{
				Pod: &coordination.EvictionRequestPodReference{
					UID:  validUID,
					Name: "foo.pod",
				},
			},

			RequesterName: "requester.domain/requester1",
			Intent:        coordination.EvictionRequestIntentWithdrawn,
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: ptr.To[int64](-5),
			Conditions: []metav1.Condition{
				{
					Type:               "Failed",
					Status:             metav1.ConditionFalse,
					LastTransitionTime: metav1.Now(),
					Reason:             "reason",
					Message:            "message",
					ObservedGeneration: 1,
				},
			},
		},
	}

	strategy.PrepareForUpdate(ctx, newEvictionRequest, oldEvictionRequest)
	if newEvictionRequest.Spec.Intent != coordination.EvictionRequestIntentEviction {
		t.Error("EvictionRequest spec.intent should not be updated and have a non Eviction intent")
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
