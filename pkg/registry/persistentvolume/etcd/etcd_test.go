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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest/resttest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

type testRegistry struct {
	*registrytest.GenericRegistry
}

func newStorage(t *testing.T) (*REST, *StatusREST, *tools.FakeEtcdClient, tools.EtcdHelper) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	helper := tools.NewEtcdHelper(fakeEtcdClient, latest.Codec, etcdtest.PathPrefix())
	storage, statusStorage := NewStorage(helper)
	return storage, statusStorage, fakeEtcdClient, helper
}

func validNewPersistentVolume(name string) *api.PersistentVolume {
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.PersistentVolumeSpec{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
			},
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{Path: "/foo"},
			},
			PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimRetain,
		},
		Status: api.PersistentVolumeStatus{
			Phase:   api.VolumePending,
			Message: "bar",
			Reason:  "foo",
		},
	}
	return pv
}

func validChangedPersistentVolume() *api.PersistentVolume {
	pv := validNewPersistentVolume("foo")
	pv.ResourceVersion = "1"
	return pv
}

func TestCreate(t *testing.T) {
	storage, _, fakeEtcdClient, _ := newStorage(t)
	test := resttest.New(t, storage, fakeEtcdClient.SetError).ClusterScope()
	pv := validNewPersistentVolume("foo")
	pv.ObjectMeta = api.ObjectMeta{GenerateName: "foo"}
	test.TestCreate(
		// valid
		pv,
		// invalid
		&api.PersistentVolume{
			ObjectMeta: api.ObjectMeta{Name: "*BadName!"},
		},
	)
}

func TestDelete(t *testing.T) {
	ctx := api.NewContext()
	storage, _, fakeEtcdClient, _ := newStorage(t)
	test := resttest.New(t, storage, fakeEtcdClient.SetError).ClusterScope()

	pv := validChangedPersistentVolume()
	key, _ := storage.KeyFunc(ctx, pv.Name)
	key = etcdtest.AddPrefix(key)
	createFn := func() runtime.Object {
		fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{
				Node: &etcd.Node{
					Value:         runtime.EncodeOrDie(latest.Codec, pv),
					ModifiedIndex: 1,
				},
			},
		}
		return pv
	}
	gracefulSetFn := func() bool {
		if fakeEtcdClient.Data[key].R.Node == nil {
			return false
		}
		return fakeEtcdClient.Data[key].R.Node.TTL == 30
	}
	test.TestDeleteNoGraceful(createFn, gracefulSetFn)
}

func TestEtcdListPersistentVolumes(t *testing.T) {
	ctx := api.NewContext()
	storage, _, fakeClient, _ := newStorage(t)
	key := storage.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, validNewPersistentVolume("foo")),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, validNewPersistentVolume("bar")),
					},
				},
			},
		},
		E: nil,
	}

	pvObj, err := storage.List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	pvs := pvObj.(*api.PersistentVolumeList)

	if len(pvs.Items) != 2 || pvs.Items[0].Name != "foo" || pvs.Items[1].Name != "bar" {
		t.Errorf("Unexpected persistentVolume list: %#v", pvs)
	}
}

func TestEtcdGetPersistentVolumes(t *testing.T) {
	ctx := api.NewContext()
	storage, _, fakeClient, _ := newStorage(t)
	persistentVolume := validNewPersistentVolume("foo")
	name := persistentVolume.Name
	key, _ := storage.KeyFunc(ctx, name)
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, persistentVolume), 0)

	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var persistentVolumeOut api.PersistentVolume
	err = latest.Codec.DecodeInto([]byte(response.Node.Value), &persistentVolumeOut)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	obj, err := storage.Get(ctx, name)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	got := obj.(*api.PersistentVolume)

	persistentVolume.ObjectMeta.ResourceVersion = got.ObjectMeta.ResourceVersion
	if e, a := persistentVolume, got; !api.Semantic.DeepEqual(*e, *a) {
		t.Errorf("Unexpected persistentVolume: %#v, expected %#v", e, a)
	}
}

func TestListEmptyPersistentVolumesList(t *testing.T) {
	ctx := api.NewContext()
	storage, _, fakeClient, _ := newStorage(t)
	fakeClient.ChangeIndex = 1
	key := storage.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: fakeClient.NewError(tools.EtcdErrorCodeNotFound),
	}

	persistentVolume, err := storage.List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(persistentVolume.(*api.PersistentVolumeList).Items) != 0 {
		t.Errorf("Unexpected non-zero pod list: %#v", persistentVolume)
	}
	if persistentVolume.(*api.PersistentVolumeList).ResourceVersion != "1" {
		t.Errorf("Unexpected resource version: %#v", persistentVolume)
	}
}

func TestListPersistentVolumesList(t *testing.T) {
	ctx := api.NewContext()
	storage, _, fakeClient, _ := newStorage(t)
	fakeClient.ChangeIndex = 1
	key := storage.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.PersistentVolume{
							ObjectMeta: api.ObjectMeta{Name: "foo"},
						}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.PersistentVolume{
							ObjectMeta: api.ObjectMeta{Name: "bar"},
						}),
					},
				},
			},
		},
	}

	persistentVolumeObj, err := storage.List(ctx, labels.Everything(), fields.Everything())
	persistentVolumeList := persistentVolumeObj.(*api.PersistentVolumeList)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(persistentVolumeList.Items) != 2 {
		t.Errorf("Unexpected persistentVolume list: %#v", persistentVolumeList)
	}
	if persistentVolumeList.Items[0].Name != "foo" {
		t.Errorf("Unexpected persistentVolume: %#v", persistentVolumeList.Items[0])
	}
	if persistentVolumeList.Items[1].Name != "bar" {
		t.Errorf("Unexpected persistentVolume: %#v", persistentVolumeList.Items[1])
	}
}

func TestPersistentVolumesDecode(t *testing.T) {
	storage, _, _, _ := newStorage(t)
	expected := validNewPersistentVolume("foo")
	body, err := latest.Codec.Encode(expected)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	actual := storage.New()
	if err := latest.Codec.DecodeInto(body, actual); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !api.Semantic.DeepEqual(expected, actual) {
		t.Errorf("mismatch: %s", util.ObjectDiff(expected, actual))
	}
}

func TestEtcdUpdatePersistentVolumes(t *testing.T) {
	ctx := api.NewContext()
	storage, _, fakeClient, _ := newStorage(t)
	persistentVolume := validChangedPersistentVolume()

	key, _ := storage.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, validNewPersistentVolume("foo")), 0)

	_, _, err := storage.Update(ctx, persistentVolume)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var persistentVolumeOut api.PersistentVolume
	err = latest.Codec.DecodeInto([]byte(response.Node.Value), &persistentVolumeOut)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	persistentVolume.ObjectMeta.ResourceVersion = persistentVolumeOut.ObjectMeta.ResourceVersion
	if !api.Semantic.DeepEqual(persistentVolume, &persistentVolumeOut) {
		t.Errorf("Unexpected persistentVolume: %#v, expected %#v", &persistentVolumeOut, persistentVolume)
	}
}

func TestDeletePersistentVolumes(t *testing.T) {
	ctx := api.NewContext()
	storage, _, fakeClient, _ := newStorage(t)
	persistentVolume := validNewPersistentVolume("foo")
	name := persistentVolume.Name
	key, _ := storage.KeyFunc(ctx, name)
	key = etcdtest.AddPrefix(key)
	fakeClient.ChangeIndex = 1
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(latest.Codec, persistentVolume),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
	}
	_, err := storage.Delete(ctx, name, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEtcdUpdateStatus(t *testing.T) {
	storage, statusStorage, fakeClient, helper := newStorage(t)
	ctx := api.NewContext()
	fakeClient.TestIndex = true

	key, _ := storage.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	pvStart := validNewPersistentVolume("foo")
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, pvStart), 1)

	pvIn := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
		},
		Status: api.PersistentVolumeStatus{
			Phase: api.VolumeBound,
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
	var pvOut api.PersistentVolume
	key, _ = storage.KeyFunc(ctx, "foo")
	if err := helper.ExtractObj(key, &pvOut, false); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !api.Semantic.DeepEqual(expected, pvOut) {
		t.Errorf("unexpected object: %s", util.ObjectDiff(expected, pvOut))
	}
}
