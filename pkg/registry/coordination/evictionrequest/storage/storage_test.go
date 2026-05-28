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

package storage

import (
	"context"
	"testing"

	"k8s.io/utils/ptr"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/pkg/apis/coordination"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

const validUID = "c88f9680-e6bc-4b18-9a4c-eed4292c4de9"

type TestDecisionAuthorizer struct {
	decision authorizer.Decision
}

func (t *TestDecisionAuthorizer) ConditionsAwareAuthorize(ctx context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
	return authorizer.ConditionsAwareDecisionFromParts(t.Authorize(ctx, a))
}

func (t *TestDecisionAuthorizer) EvaluateConditions(ctx context.Context, decision authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorized authorizer.Decision, reason string, err error) {
	return authorizer.DecisionDeny, "", authorizer.ErrorConditionEvaluationNotSupported
}

func (t *TestDecisionAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	return t.decision, "", nil
}

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, coordination.SchemeGroupVersion.WithResource("evictionrequests").GroupResource())
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "evictionrequests",
	}

	evictionRequestStorage, evictionRequestStatusStorage, err := NewREST(restOptions, &TestDecisionAuthorizer{authorizer.DecisionAllow})
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return evictionRequestStorage, evictionRequestStatusStorage, server
}

func tester(t *testing.T, storage *REST) *genericregistrytest.Tester {
	test := genericregistrytest.New(t, storage.Store)
	requestInfo := &genericapirequest.RequestInfo{
		APIGroup:   "coordination.k8s.io",
		APIVersion: "v1alpha1",
		Resource:   "evictionrequests",
	}
	test.SetRequestInfo(requestInfo)
	test.SetUserInfo(&user.DefaultInfo{Name: "test"})
	return test
}

func newValidEvictionRequest() *coordination.EvictionRequest {
	return &coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionRequestTarget{
				Pod: &coordination.EvictionRequestPodReference{
					UID:  validUID,
					Name: "foo.pod",
				},
			},
			RequesterName: "requester-1.example.com/bar",
			Intent:        coordination.EvictionRequestIntentEviction,
		},
		Status: coordination.EvictionRequestStatus{
			ObservedGeneration: ptr.To[int64](1),
		},
	}
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage)
	validEvictionRequest := newValidEvictionRequest()
	validEvictionRequest.ObjectMeta = metav1.ObjectMeta{}

	invalidEvictionRequest := newValidEvictionRequest()
	invalidEvictionRequest.ObjectMeta = metav1.ObjectMeta{Name: "-foo"}
	test.TestCreate(
		validEvictionRequest,
		invalidEvictionRequest,
	)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage)
	validEvictionRequest := newValidEvictionRequest()
	test.TestUpdate(
		validEvictionRequest,
		func(obj runtime.Object) runtime.Object {
			object := obj.(*coordination.EvictionRequest)
			object.ObjectMeta.Annotations = map[string]string{"foo": "bar"}
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*coordination.EvictionRequest)
			object.Spec.Target.Pod.Name = "bar"
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage)
	test.TestDelete(newValidEvictionRequest())
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage)
	test.TestGet(newValidEvictionRequest())
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage)
	test.TestList(newValidEvictionRequest())
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage)
	test.TestWatch(
		newValidEvictionRequest(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"x": "y"},
		},
		// matching fields
		[]fields.Set{},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "xyz"},
			{"name": "foo"},
		},
	)
}

func TestStatusUpdate(t *testing.T) {
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	evictionRequest := newValidEvictionRequest()
	evictionRequest.Status = coordination.EvictionRequestStatus{}

	ctx := evictionRequestContext()
	key, err := storage.KeyFunc(ctx, "foo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	result := &coordination.EvictionRequest{}
	if err := storage.Storage.Create(ctx, key, evictionRequest, result, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Status.ObservedGeneration != nil {
		t.Errorf("we expected .status.observedGeneration to be nil but it was %v", *result.Status.ObservedGeneration)
	}
	evictionRequestUpdate := newValidEvictionRequest()
	evictionRequestUpdate.ObjectMeta = result.ObjectMeta
	evictionRequestUpdate.Labels = map[string]string{"foo": "bar"}
	evictionRequestUpdate.Spec.Target.Pod.Name = "bax"
	evictionRequestUpdate.Status.ObservedGeneration = ptr.To[int64](1)

	if _, _, err := statusStorage.Update(ctx, evictionRequestUpdate.Name, rest.DefaultUpdatedObjectInfo(evictionRequestUpdate), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, evictionRequestUpdate.Name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result = obj.(*coordination.EvictionRequest)
	if len(result.Labels) != 0 {
		t.Errorf("we expected .status.labels to be empty but it was %v", result.Labels)
	}
	if result.Spec.Target.Pod.Name != "foo.pod" {
		t.Errorf("we expected .spec.target.pod.name to not be updated but it was updated to %v", result.Spec.Target.Pod.Name)
	}
	if ptr.Deref(result.Status.ObservedGeneration, 0) != 1 {
		t.Errorf("we expected .status.observedGeneration to be updated to but it was %v", ptr.Deref(result.Status.ObservedGeneration, -1))
	}
}
func TestGenerationNumber(t *testing.T) {
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	evictionRequest := newValidEvictionRequest()
	evictionRequest.Generation = 100
	evictionRequest.Status.ObservedGeneration = ptr.To[int64](10)
	ctx := evictionRequestContext()
	resultObj, err := storage.Create(ctx, evictionRequest, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	result, _ := resultObj.(*coordination.EvictionRequest)

	// Generation initialization
	if result.Generation != 1 || result.Status.ObservedGeneration != nil {
		t.Fatalf("Unexpected generation number %v, status generation %v", result.Generation, result.Status.ObservedGeneration)
	}

	// Updates to status should not increment either spec or status generation numbers
	result.Status.Conditions = append(result.Status.Conditions, metav1.Condition{Type: "Test", Status: "True", LastTransitionTime: metav1.Now(), Reason: "Reason"})
	if _, _, err := statusStorage.Update(ctx, result.Name, rest.DefaultUpdatedObjectInfo(result), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resultObj, err = storage.Get(ctx, result.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	result, _ = resultObj.(*coordination.EvictionRequest)
	if result.Generation != 1 || result.Status.ObservedGeneration != nil {
		t.Fatalf("Unexpected generation number, spec: %v, status: %v", result.Generation, result.Status.ObservedGeneration)
	}
}

func evictionRequestContext() context.Context {
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	ctx = genericapirequest.WithRequestInfo(ctx, &genericapirequest.RequestInfo{
		APIGroup:   "coordination.k8s.io",
		APIVersion: "v1alpha1",
		Resource:   "evictionrequests",
	})
	ctx = genericapirequest.WithUser(ctx, &user.DefaultInfo{Name: "test"})
	return ctx
}
