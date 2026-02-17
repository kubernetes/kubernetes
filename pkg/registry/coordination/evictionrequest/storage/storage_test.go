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
	"time"

	"github.com/google/uuid"

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
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	testing2 "k8s.io/utils/clock/testing"
)

const valiUIDdName = "8057f54d-455d-4b25-90c6-92a919cff10a"

type TestDecisionAuthorizer struct {
	decision authorizer.Decision
}

func (t *TestDecisionAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	return t.decision, "", nil
}

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	clock := testing2.NewFakePassiveClock(time.Now())
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, coordination.SchemeGroupVersion.WithResource("evictionrequests").GroupResource())
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "evictionrequests",
	}

	evictionRequestStorage, evictionRequestStatusStorage, err := NewREST(restOptions, &TestDecisionAuthorizer{authorizer.DecisionAllow}, clock)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return evictionRequestStorage, evictionRequestStatusStorage, server
}

func tester(t *testing.T, storage *REST, isStatus bool) *genericregistrytest.Tester {
	uuidMap := make(map[int]string)
	test := genericregistrytest.New(t, storage.Store).
		GeneratesName().
		Namer(func(i int) string {
			if result, ok := uuidMap[i]; ok {
				return result
			}
			uuidMap[i] = uuid.NewString()
			return uuidMap[i]
		}).NewObjectModifier(func(object runtime.Object) {
		evictionRequest := object.(*coordination.EvictionRequest)
		evictionRequest.Spec.Target.Pod.UID = evictionRequest.Name
	})
	requestInfo := &genericapirequest.RequestInfo{
		APIGroup:   "coordination.k8s.io",
		APIVersion: "v1alpha1",
		Resource:   "evictionrequests",
	}
	if isStatus {
		requestInfo.Subresource = "status"
	}
	test.SetRequestInfo(requestInfo)
	test.SetUserInfo(&user.DefaultInfo{Name: "test"})
	return test
}

func newValidEvictionRequest() *coordination.EvictionRequest {
	return &coordination.EvictionRequest{
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
			ObservedGeneration: 1,
			TargetInterceptors: []core.EvictionInterceptor{
				{Name: "interceptor1.example.com"},
			},
			Interceptors: []coordination.InterceptorStatus{
				{Name: "interceptor1.example.com"},
			},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage, false)
	validEvictionRequest := newValidEvictionRequest()

	invalidEvictionRequest := newValidEvictionRequest()
	test.TestCreate(
		validEvictionRequest,
		invalidEvictionRequest,
	)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage, false)
	validEvictionRequest := newValidEvictionRequest()
	test.TestUpdate(
		validEvictionRequest,
		func(obj runtime.Object) runtime.Object {
			object := obj.(*coordination.EvictionRequest)
			object.Spec.Requesters = append(object.Spec.Requesters, coordination.Requester{Name: "requester-3.example.com"})
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*coordination.EvictionRequest)
			object.Spec.Target.Pod.Name = "bar"
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*coordination.EvictionRequest)
			object.Spec.Requesters = append(object.Spec.Requesters, coordination.Requester{Name: "requester-4"})
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage, false)
	evictionRequest := newValidEvictionRequest()
	evictionRequest.Name = valiUIDdName
	test.TestDelete(evictionRequest)
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage, false)
	test.TestGet(newValidEvictionRequest())
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage, false)
	test.TestList(newValidEvictionRequest())
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := tester(t, storage, false)
	evictionRequest := newValidEvictionRequest()
	evictionRequest.Name = valiUIDdName
	test.TestWatch(
		evictionRequest,
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
	evictionRequest.Name = valiUIDdName
	evictionRequest.Namespace = metav1.NamespaceDefault
	evictionRequest.Status = coordination.EvictionRequestStatus{}

	ctx := evictionRequestContext()
	key, err := storage.KeyFunc(ctx, valiUIDdName)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	result := &coordination.EvictionRequest{}
	if err := storage.Storage.Create(ctx, key, evictionRequest, result, 0, false); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Status.TargetInterceptors) != 0 {
		t.Errorf("we expected .status.targetInterceptors to be empty but it was %v", result.Status.TargetInterceptors)
	}
	evictionRequestUpdate := newValidEvictionRequest()
	evictionRequestUpdate.ObjectMeta = result.ObjectMeta
	evictionRequestUpdate.Labels = map[string]string{"foo": "bar"}
	evictionRequestUpdate.Spec.Requesters = nil
	evictionRequestUpdate.Status.TargetInterceptors = []core.EvictionInterceptor{
		{Name: "interceptor1.example.com"},
	}
	evictionRequestUpdate.Status.Interceptors = []coordination.InterceptorStatus{
		{Name: "interceptor1.example.com"},
	}

	if _, _, err := statusStorage.Update(ctx, valiUIDdName, rest.DefaultUpdatedObjectInfo(evictionRequestUpdate), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, valiUIDdName, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result = obj.(*coordination.EvictionRequest)
	if len(result.Labels) != 0 {
		t.Errorf("we expected .status.labels to be empty but it was %v", result.Labels)
	}
	if len(result.Spec.Requesters) == 0 {
		t.Errorf("we expected .spec.requesters to not be updated but it was updated to %v", result.Spec.Requesters)
	}
	if len(result.Status.TargetInterceptors) != 1 {
		t.Errorf("we expected .status.targetInterceptors to be updated to but it was %v", result.Status.TargetInterceptors)
	}
}

func TestGenerationNumber(t *testing.T) {
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	evictionRequest := newValidEvictionRequest()
	evictionRequest.Name = valiUIDdName
	evictionRequest.Generation = 100
	evictionRequest.Status.ObservedGeneration = 10
	ctx := evictionRequestContext()
	resultObj, err := storage.Create(ctx, evictionRequest, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	result, _ := resultObj.(*coordination.EvictionRequest)

	// Generation initialization
	if result.Generation != 1 || result.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation number %v, status generation %v", result.Generation, result.Status.ObservedGeneration)
	}

	// Updates to spec should increment the generation number
	result.Spec.Requesters = nil
	if _, _, err := storage.Update(ctx, result.Name, rest.DefaultUpdatedObjectInfo(result), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resultObj, err = storage.Get(ctx, result.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	result, _ = resultObj.(*coordination.EvictionRequest)
	if result.Generation != 2 || result.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation, spec: %v, status: %v", result.Generation, result.Status.ObservedGeneration)
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
	if result.Generation != 2 || result.Status.ObservedGeneration != 0 {
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
