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

package eviction

import (
	"testing"
	"time"

	"k8s.io/utils/ptr"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/coordination"
	testing2 "k8s.io/utils/clock/testing"
)

func TestEvictionStrategy_ResetFields(t *testing.T) {
	strategy := NewStrategy(nil)
	for _, fields := range strategy.GetResetFields() {
		if !fields.Has(fieldpath.MakePathOrDie("status")) {
			t.Errorf("status should be reset on creation and update")
		}
	}
}

func TestEvictionStrategy(t *testing.T) {
	clock := testing2.NewFakePassiveClock(time.Now())
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "coordination.k8s.io",
		APIVersion:        "v1alpha1",
		Resource:          "evictions",
		IsResourceRequest: true,
		Verb:              "create",
	})
	strategy := NewStrategy(clock)

	if !strategy.NamespaceScoped() {
		t.Errorf("Eviction must be namespace scoped")
	}
	if strategy.AllowCreateOnUpdate(ctx) {
		t.Errorf("Eviction should not allow create on update")
	}
	if len(strategy.GenerateName("test")) <= len("test") {
		t.Errorf("Eviction should implement name generation")
	}
	if len(strategy.WarningsOnCreate(ctx, nil)) != 0 {
		t.Errorf("Eviction warnings on create are expected to be empty")
	}

	eviction := &coordination.Eviction{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "foo"},
		Spec: coordination.EvictionSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.EvictionPodReference{
					UID:  validUID,
					Name: "foo.pod",
				},
			},
		},
		Status: coordination.EvictionStatus{
			ObservedGeneration: ptr.To[int64](1),
			Requesters: []coordination.Requester{
				{Name: "requester.example.com/bar", Intent: coordination.RequesterIntentEviction},
			},
			TargetResponders: []coordination.TargetResponder{
				{Name: "test", State: coordination.ResponderStateInactive},
			},
		},
	}

	strategy.PrepareForCreate(ctx, eviction)
	if eviction.Generation != int64(1) {
		t.Error("Eviction metadata.generation should be set to 1")
	}
	if len(eviction.Status.TargetResponders) != 0 {
		t.Error("Eviction should not allow setting status.targetResponders on create")
	}
	if eviction.Status.ObservedGeneration != nil {
		t.Error("Eviction should not allow setting status.observedGeneration on create")
	}
	errs := strategy.Validate(ctx, eviction)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
}

func TestEvictionStrategy_Update(t *testing.T) {
	clock := testing2.NewFakePassiveClock(time.Now())
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "coordination.k8s.io",
		APIVersion:        "v1alpha1",
		Resource:          "evictions",
		IsResourceRequest: true,
		Verb:              "update",
	})
	ctx = genericapirequest.WithUser(ctx, &user.DefaultInfo{Name: "other-user"})
	strategy := NewStrategy(clock)

	if len(strategy.WarningsOnUpdate(ctx, nil, nil)) != 0 {
		t.Errorf("Eviction warnings on update are expected to be empty")
	}

	oldEviction := &coordination.Eviction{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "foo", Generation: 1, ResourceVersion: "2"},
		Spec: coordination.EvictionSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.EvictionPodReference{
					UID:  validUID,
					Name: "foo.pod",
				},
			},
		},
		Status: coordination.EvictionStatus{
			ObservedGeneration: ptr.To[int64](1),
			Requesters: []coordination.Requester{
				{Name: "requester-1.example.com/bar", Intent: coordination.RequesterIntentEviction},
				{Name: "requester-2.example.com/bar", Intent: coordination.RequesterIntentWithdrawn},
			},
		},
	}

	newEviction := &coordination.Eviction{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "foo", ResourceVersion: "2"},
		Spec: coordination.EvictionSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.EvictionPodReference{
					UID:  validUID,
					Name: "bar.pod",
				},
			},
		},
		Status: coordination.EvictionStatus{
			ObservedGeneration: ptr.To[int64](10),
			Requesters: []coordination.Requester{
				{Name: "requester-1.example.com/bar", Intent: coordination.RequesterIntentEviction},
				{Name: "requester-2.example.com/bar", Intent: coordination.RequesterIntentEviction},
			},
			TargetResponders: []coordination.TargetResponder{
				{Name: "test", State: coordination.ResponderStateCanceled},
			},
		},
	}

	strategy.PrepareForUpdate(ctx, newEviction, oldEviction)
	if newEviction.Generation != int64(2) {
		t.Error("Eviction metadata.generation should be set to 2")
	}
	if len(newEviction.Status.TargetResponders) != 0 {
		t.Error("Eviction should not allow setting status.targetResponders on update")
	}
	errs := strategy.ValidateUpdate(ctx, newEviction, oldEviction)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
	newEviction.Spec.Target.Pod.Name = "foo.pod"
	errs = strategy.ValidateUpdate(ctx, newEviction, oldEviction)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
}

func TestEvictionStatusStrategy_ResetFields(t *testing.T) {
	strategy := NewStrategy(nil)
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

func TestEvictionStatusStrategy(t *testing.T) {
	clock := testing2.NewFakePassiveClock(time.Now())
	strategy := NewStatusStrategy(NewStrategy(clock))
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "coordination.k8s.io",
		APIVersion:        "v1alpha1",
		Resource:          "evictions",
		IsResourceRequest: true,
		Verb:              "update",
		Subresource:       "status",
	})

	oldEviction := &coordination.Eviction{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "foo", Generation: 1, ResourceVersion: "2",
			Annotations: map[string]string{"test": "true"},
			Labels:      map[string]string{"foo": "bar"}},
		Spec: coordination.EvictionSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.EvictionPodReference{
					UID:  validUID,
					Name: "foo.pod",
				},
			},
		},
		Status: coordination.EvictionStatus{
			ObservedGeneration: ptr.To[int64](1),
		},
	}

	newEviction := &coordination.Eviction{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "foo", Generation: 1, ResourceVersion: "2",
			Annotations: map[string]string{"test": "false"},
			Labels:      map[string]string{"foo": "baz"}},
		Spec: coordination.EvictionSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.EvictionPodReference{
					UID:  validUID,
					Name: "foo.update",
				},
			},
		},
		Status: coordination.EvictionStatus{
			ObservedGeneration: ptr.To[int64](-5),
			Requesters: []coordination.Requester{
				{Name: "requester-1.example.com/bar", Intent: coordination.RequesterIntentEviction},
				{Name: "requester-2.example.com/bar", Intent: coordination.RequesterIntentEviction},
			},
			TargetResponders: []coordination.TargetResponder{
				{Name: "responder1.example.com/bar", State: coordination.ResponderStateActive},
			},
			Responders: []coordination.ResponderStatus{
				{Name: "responder1.example.com/bar", Message: "test message", StartTime: ptr.To(metav1.Now())},
			},
		},
	}

	strategy.PrepareForUpdate(ctx, newEviction, oldEviction)
	if newEviction.Spec.Target.Pod.Name != "foo.pod" {
		t.Error("Eviction spec.target.pod.name should not be updated")
	}
	if newEviction.Labels["foo"] != "bar" {
		t.Error("Eviction should not allow changing labels")
	}
	if newEviction.Annotations["test"] != "true" {
		t.Error("Eviction should not allow changing annotations")
	}
	errs := strategy.ValidateUpdate(ctx, newEviction, oldEviction)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
	newEviction.Status.ObservedGeneration = ptr.To[int64](2)
	errs = strategy.ValidateUpdate(ctx, newEviction, oldEviction)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
}
