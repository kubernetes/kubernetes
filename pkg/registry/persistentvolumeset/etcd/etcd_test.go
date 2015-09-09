/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package etcd

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/resource"
	//	"k8s.io/kubernetes/pkg/fields"
	//	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"

	"github.com/davecgh/go-spew/spew"
)

func newStorage(t *testing.T) (*REST, *StatusREST, *tools.FakeEtcdClient) {
	etcdStorage, fakeClient := registrytest.NewEtcdStorage(t, "")
	storage, statusStorage := NewREST(etcdStorage)
	return storage, statusStorage, fakeClient
}

func validNewPersistentVolumeSet(name string) *api.PersistentVolumeSet {
	pvs := &api.PersistentVolumeSet{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.PersistentVolumeSetSpec{
			MinimumReplicas: 5,
			MaximumReplicas: 10,
			Selector:        map[string]string{"a": "b"},
			Template: &api.PersistentVolumeTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					GenerateName: "pvs-",
					Labels:       map[string]string{"a": "b"},
				},
				Spec: api.PersistentVolumeSpec{
					Capacity: api.ResourceList{
						api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
					},
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
					PersistentVolumeSource: api.PersistentVolumeSource{
						HostPath: &api.HostPathVolumeSource{Path: "/foo"},
					},
					PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimRecycle,
				},
			},
		},
		Status: api.PersistentVolumeSetStatus{
			BoundReplicas:     5,
			AvailableReplicas: 5,
		},
	}
	return pvs
}

func TestCreate(t *testing.T) {
	storage, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
	pvs := validNewPersistentVolumeSet("foo")
	pvs.ObjectMeta = api.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		pvs,
		// invalid
		&api.PersistentVolumeSet{
			ObjectMeta: api.ObjectMeta{Name: "*BadName!"},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
	test.TestUpdate(
		// valid
		validNewPersistentVolumeSet("foo"),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*api.PersistentVolumeSet)
			object.Spec.MaximumReplicas = 100
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope().ReturnDeletedObject()
	test.TestDelete(validNewPersistentVolumeSet("foo"))
}

func TestGet(t *testing.T) {
	storage, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
	test.TestGet(validNewPersistentVolumeSet("foo"))
}

func TestList(t *testing.T) {
	storage, _, fakeClient := newStorage(t)
	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
	test.TestList(validNewPersistentVolumeSet("foo"))
}

//
//func TestWatch(t *testing.T) {
//	storage, _, fakeClient := newStorage(t)
//	test := registrytest.New(t, fakeClient, storage.Etcd).ClusterScope()
//	test.TestWatch(
//		validNewPersistentVolumeSet("foo"),
//		// matching labels
//		[]labels.Set{},
//		// not matching labels
//		[]labels.Set{
//			{"foo": "bar"},
//		},
//		// matching fields
//		[]fields.Set{
//			{"metadata.name": "foo"},
//			{"name": "foo"},
//		},
//		// not matching fields
//		[]fields.Set{
//			{"metadata.name": "bar"},
//		},
//	)
//}

func TestEtcdUpdateStatus(t *testing.T) {
	storage, statusStorage, fakeClient := newStorage(t)
	fakeClient.TestIndex = true

	ctx := api.NewContext()
	key, _ := storage.KeyFunc(ctx, "bar")
	key = etcdtest.AddPrefix(key)
	pvStart := validNewPersistentVolumeSet("bar")
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, pvStart), 0)

	pvIn := &api.PersistentVolumeSet{
		ObjectMeta: api.ObjectMeta{
			Name:            "bar",
			ResourceVersion: "1",
			Labels:          map[string]string{"a": "b"},
		},
		Status: api.PersistentVolumeSetStatus{
			BoundReplicas: 5,
		},
	}

	expected := *pvStart
	expected.ResourceVersion = "2"
	expected.Labels = pvIn.Labels
	expected.Status = pvIn.Status

	_, _, err := statusStorage.Update(ctx, pvIn)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	pvOut, err := storage.Get(ctx, "bar")

	spew.Dump(pvOut)
	spew.Dump(expected.DeletionTimestamp)

	pv := pvOut.(*api.PersistentVolumeSet)
	spew.Dump(pv.DeletionTimestamp)

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !api.Semantic.DeepEqual(&expected, pvOut) {
		t.Errorf("unexpected object: %s", util.ObjectDiff(expected, pvOut))
	}
}
