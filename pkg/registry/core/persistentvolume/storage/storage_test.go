/*
Copyright 2015 The Kubernetes Authors.

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

	"github.com/google/go-cmp/cmp"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/registry/core/persistentvolume"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "persistentvolumes",
	}
	persistentVolumeStorage, statusStorage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return persistentVolumeStorage, statusStorage, server
}

func newHostPathType(pathType string) *api.HostPathType {
	hostPathType := new(api.HostPathType)
	*hostPathType = api.HostPathType(pathType)
	return hostPathType
}

func validNewPersistentVolume(name string) *api.PersistentVolume {
	now := persistentvolume.NowFunc()
	pv := &api.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: api.PersistentVolumeSpec{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
			},
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{Path: "/foo", Type: newHostPathType(string(api.HostPathDirectoryOrCreate))},
			},
			PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimRetain,
		},
		Status: api.PersistentVolumeStatus{
			Phase:                   api.VolumePending,
			Message:                 "bar",
			Reason:                  "foo",
			LastPhaseTransitionTime: &now,
		},
	}
	return pv
}

func TestCreate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	pv := validNewPersistentVolume("foo")
	pv.ObjectMeta = metav1.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		pv,
		// invalid
		&api.PersistentVolume{
			ObjectMeta: metav1.ObjectMeta{Name: "*BadName!"},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestUpdate(
		// valid
		validNewPersistentVolume("foo"),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*api.PersistentVolume)
			object.Spec.Capacity = api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("20G"),
			}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope().ReturnDeletedObject()
	test.TestDelete(validNewPersistentVolume("foo"))
}

func TestGet(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestGet(validNewPersistentVolume("foo"))
}

func TestList(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestList(validNewPersistentVolume("foo"))
}

func TestWatch(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).ClusterScope()
	test.TestWatch(
		validNewPersistentVolume("foo"),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
			{"name": "foo"},
		},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
		},
	)
}

func TestUpdateStatus(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PersistentVolumeLastPhaseTransitionTime, true)
	storage, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceNone)
	key, _ := storage.KeyFunc(ctx, "foo")
	pvStart := validNewPersistentVolume("foo")
	err := storage.Storage.Create(ctx, key, pvStart, nil, 0, false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	pvStartTimestamp, err := getPhaseTranstitionTime(ctx, pvStart.Name, storage)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// We need to set custom timestamp which is not the same as the one on existing PV
	// - doing so will prevent timestamp update on phase change and custom one is used instead.
	pvStartTimestamp = &metav1.Time{Time: pvStartTimestamp.Time.Add(time.Second)}

	pvIn := &api.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Status: api.PersistentVolumeStatus{
			Phase:                   api.VolumeBound,
			LastPhaseTransitionTime: pvStartTimestamp,
		},
	}

	_, _, err = statusStorage.Update(ctx, pvIn.Name, rest.DefaultUpdatedObjectInfo(pvIn), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	obj, err := storage.Get(ctx, "foo", &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	pvOut := obj.(*api.PersistentVolume)
	// only compare the relevant change b/c metadata will differ
	if !apiequality.Semantic.DeepEqual(pvIn.Status, pvOut.Status) {
		t.Errorf("unexpected object: %s", cmp.Diff(pvIn.Status, pvOut.Status))
	}
}

func getPhaseTranstitionTime(ctx context.Context, pvName string, storage *REST) (*metav1.Time, error) {
	obj, err := storage.Get(ctx, pvName, &metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return obj.(*api.PersistentVolume).Status.LastPhaseTransitionTime, nil
}

func TestShortNames(t *testing.T) {
	storage, _, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"pv"}
	registrytest.AssertShortNames(t, storage, expected)
}
