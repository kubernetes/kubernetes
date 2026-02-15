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
	"testing"

	"github.com/google/go-cmp/cmp"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, scheduling.Resource("podgroups"))
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "podgroups",
	}
	rest, statusRest, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("Unable to create REST %v", err)
	}
	return rest, statusRest, server
}

func validNewPodGroup() *scheduling.PodGroup {
	return &scheduling.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: scheduling.PodGroupSpec{
			SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
				Gang: &scheduling.GangSchedulingPolicy{
					MinCount: 5,
				},
			},
		},
	}
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	podGroup := validNewPodGroup()
	podGroup.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		podGroup,
		// invalid cases
		&scheduling.PodGroup{
			ObjectMeta: metav1.ObjectMeta{
				Name: "*badName",
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestUpdate(
		// valid
		validNewPodGroup(),
		// valid update
		// Update Status
		func(obj runtime.Object) runtime.Object {
			pg := obj.(*scheduling.PodGroup)
			pg.Status = scheduling.PodGroupStatus{
				Conditions: []metav1.Condition{
					{
						Type:   "PodGroupScheduled",
						Status: metav1.ConditionTrue,
					},
				},
			}
			return pg
		},
		// invalid update
		// Update MinCount
		func(obj runtime.Object) runtime.Object {
			pg := obj.(*scheduling.PodGroup)
			pg.Spec.SchedulingPolicy.Gang.MinCount = 4
			return pg
		},
		// invalid update
		// Update PodGroupTemplateRef
		func(obj runtime.Object) runtime.Object {
			pg := obj.(*scheduling.PodGroup)
			pg.Spec.PodGroupTemplateRef = &scheduling.PodGroupTemplateReference{
				WorkloadName:         "foo",
				PodGroupTemplateName: "baz",
			}
			return pg
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestDelete(validNewPodGroup())
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestGet(validNewPodGroup())
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestList(validNewPodGroup())
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestWatch(
		validNewPodGroup(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
		},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
		},
	)
}

func TestUpdateStatus(t *testing.T) {
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	ctx := genericapirequest.NewDefaultContext()

	key, _ := storage.KeyFunc(ctx, "foo")
	podGroupStart := validNewPodGroup()
	err := storage.Storage.Create(ctx, key, podGroupStart, nil, 0, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}

	podGroup := podGroupStart.DeepCopy()
	podGroup.Status = scheduling.PodGroupStatus{
		Conditions: []metav1.Condition{
			{
				Type:   "PodGroupScheduled",
				Status: metav1.ConditionTrue,
			},
		},
	}
	_, _, err = statusStorage.Update(ctx, podGroup.Name, rest.DefaultUpdatedObjectInfo(podGroup), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	podGroupOut := obj.(*scheduling.PodGroup)
	// only compare relevant changes b/c of difference in metadata
	if !apiequality.Semantic.DeepEqual(podGroup.Status, podGroupOut.Status) {
		t.Errorf("unexpected object: %s", cmp.Diff(podGroup.Status, podGroupOut.Status))
	}
}
