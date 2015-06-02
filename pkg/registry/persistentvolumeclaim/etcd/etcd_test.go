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

func validNewPersistentVolumeClaim(name, ns string) *api.PersistentVolumeClaim {
	pv := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
			},
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimPending,
		},
	}
	return pv
}

func validChangedPersistentVolumeClaim() *api.PersistentVolumeClaim {
	pv := validNewPersistentVolumeClaim("foo", api.NamespaceDefault)
	pv.ResourceVersion = "1"
	return pv
}

func TestCreate(t *testing.T) {

	registry, _, fakeEtcdClient, _ := newStorage(t)
	test := resttest.New(t, registry, fakeEtcdClient.SetError)
	pv := validNewPersistentVolumeClaim("foo", api.NamespaceDefault)
	pv.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		pv,
		// invalid
		&api.PersistentVolumeClaim{
			ObjectMeta: api.ObjectMeta{Name: "*BadName!"},
		},
	)
}

func TestDelete(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, _, fakeEtcdClient, _ := newStorage(t)
	test := resttest.New(t, storage, fakeEtcdClient.SetError)

	pv := validChangedPersistentVolumeClaim()
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

func TestEtcdListPersistentVolumeClaims(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry, _, fakeClient, _ := newStorage(t)
	key := registry.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, validNewPersistentVolumeClaim("foo", api.NamespaceDefault)),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, validNewPersistentVolumeClaim("bar", api.NamespaceDefault)),
					},
				},
			},
		},
		E: nil,
	}

	pvObj, err := registry.List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	pvs := pvObj.(*api.PersistentVolumeClaimList)

	if len(pvs.Items) != 2 || pvs.Items[0].Name != "foo" || pvs.Items[1].Name != "bar" {
		t.Errorf("Unexpected persistentVolume list: %#v", pvs)
	}
}

func TestEtcdGetPersistentVolumeClaims(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry, _, fakeClient, _ := newStorage(t)
	persistentVolume := validNewPersistentVolumeClaim("foo", api.NamespaceDefault)
	name := persistentVolume.Name
	key, _ := registry.KeyFunc(ctx, name)
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, persistentVolume), 0)

	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var persistentVolumeOut api.PersistentVolumeClaim
	err = latest.Codec.DecodeInto([]byte(response.Node.Value), &persistentVolumeOut)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	obj, err := registry.Get(ctx, name)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	got := obj.(*api.PersistentVolumeClaim)

	persistentVolume.ObjectMeta.ResourceVersion = got.ObjectMeta.ResourceVersion
	if e, a := persistentVolume, got; !api.Semantic.DeepEqual(*e, *a) {
		t.Errorf("Unexpected persistentVolume: %#v, expected %#v", e, a)
	}
}

func TestListEmptyPersistentVolumeClaimsList(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry, _, fakeClient, _ := newStorage(t)
	fakeClient.ChangeIndex = 1
	key := registry.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: fakeClient.NewError(tools.EtcdErrorCodeNotFound),
	}

	persistentVolume, err := registry.List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(persistentVolume.(*api.PersistentVolumeClaimList).Items) != 0 {
		t.Errorf("Unexpected non-zero pod list: %#v", persistentVolume)
	}
	if persistentVolume.(*api.PersistentVolumeClaimList).ResourceVersion != "1" {
		t.Errorf("Unexpected resource version: %#v", persistentVolume)
	}
}

func TestListPersistentVolumeClaimsList(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry, _, fakeClient, _ := newStorage(t)
	fakeClient.ChangeIndex = 1
	key := registry.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.PersistentVolumeClaim{
							ObjectMeta: api.ObjectMeta{Name: "foo"},
						}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.PersistentVolumeClaim{
							ObjectMeta: api.ObjectMeta{Name: "bar"},
						}),
					},
				},
			},
		},
	}

	persistentVolumeObj, err := registry.List(ctx, labels.Everything(), fields.Everything())
	persistentVolumeList := persistentVolumeObj.(*api.PersistentVolumeClaimList)
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

func TestPersistentVolumeClaimsDecode(t *testing.T) {
	registry, _, _, _ := newStorage(t)
	expected := validNewPersistentVolumeClaim("foo", api.NamespaceDefault)
	body, err := latest.Codec.Encode(expected)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	actual := registry.New()
	if err := latest.Codec.DecodeInto(body, actual); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !api.Semantic.DeepEqual(expected, actual) {
		t.Errorf("mismatch: %s", util.ObjectDiff(expected, actual))
	}
}

func TestEtcdUpdatePersistentVolumeClaims(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry, _, fakeClient, _ := newStorage(t)
	persistentVolume := validChangedPersistentVolumeClaim()

	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, validNewPersistentVolumeClaim("foo", api.NamespaceDefault)), 0)

	_, _, err := registry.Update(ctx, persistentVolume)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var persistentVolumeOut api.PersistentVolumeClaim
	err = latest.Codec.DecodeInto([]byte(response.Node.Value), &persistentVolumeOut)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	persistentVolume.ObjectMeta.ResourceVersion = persistentVolumeOut.ObjectMeta.ResourceVersion
	if !api.Semantic.DeepEqual(persistentVolume, &persistentVolumeOut) {
		t.Errorf("Unexpected persistentVolume: %#v, expected %#v", &persistentVolumeOut, persistentVolume)
	}
}

func TestDeletePersistentVolumeClaims(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry, _, fakeClient, _ := newStorage(t)
	pvClaim := validNewPersistentVolumeClaim("foo", api.NamespaceDefault)
	name := pvClaim.Name
	key, _ := registry.KeyFunc(ctx, name)
	key = etcdtest.AddPrefix(key)
	fakeClient.ChangeIndex = 1
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(latest.Codec, pvClaim),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
	}
	_, err := registry.Delete(ctx, name, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEtcdUpdateStatus(t *testing.T) {
	storage, statusStorage, fakeClient, helper := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true

	key, _ := storage.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	pvcStart := validNewPersistentVolumeClaim("foo", api.NamespaceDefault)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, pvcStart), 1)

	pvc := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "1",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("3Gi"),
				},
			},
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimBound,
		},
	}

	expected := *pvcStart
	expected.ResourceVersion = "2"
	expected.Labels = pvc.Labels
	expected.Status = pvc.Status

	_, _, err := statusStorage.Update(ctx, pvc)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	var pvcOut api.PersistentVolumeClaim
	key, _ = storage.KeyFunc(ctx, "foo")
	if err := helper.ExtractObj(key, &pvcOut, false); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !api.Semantic.DeepEqual(expected, pvcOut) {
		t.Errorf("unexpected object: %s", util.ObjectDiff(expected, pvcOut))
	}
}
