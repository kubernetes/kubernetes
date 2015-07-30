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
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest/resttest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/resourcequota"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/storage"
	etcdstorage "github.com/GoogleCloudPlatform/kubernetes/pkg/storage/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

func newEtcdStorage(t *testing.T) (*tools.FakeEtcdClient, storage.Interface) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	etcdStorage := etcdstorage.NewEtcdStorage(fakeEtcdClient, latest.Codec, etcdtest.PathPrefix())
	return fakeEtcdClient, etcdStorage
}

func newStorage(t *testing.T) (*REST, *StatusREST, *tools.FakeEtcdClient, storage.Interface) {
	fakeEtcdClient, h := newEtcdStorage(t)
	storage, statusStorage := NewStorage(h)
	return storage, statusStorage, fakeEtcdClient, h
}

func validNewResourceQuota() *api.ResourceQuota {
	return &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: api.ResourceQuotaSpec{
			Hard: api.ResourceList{
				api.ResourceCPU:                    resource.MustParse("100"),
				api.ResourceMemory:                 resource.MustParse("4Gi"),
				api.ResourcePods:                   resource.MustParse("10"),
				api.ResourceServices:               resource.MustParse("10"),
				api.ResourceReplicationControllers: resource.MustParse("10"),
				api.ResourceQuotas:                 resource.MustParse("1"),
			},
		},
	}
}

func validChangedResourceQuota() *api.ResourceQuota {
	resourcequota := validNewResourceQuota()
	resourcequota.ResourceVersion = "1"
	resourcequota.Labels = map[string]string{
		"foo": "bar",
	}
	return resourcequota
}

func TestStorage(t *testing.T) {
	storage, _, _, _ := newStorage(t)
	resourcequota.NewRegistry(storage)
}

func TestCreate(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage, _ := NewStorage(etcdStorage)
	test := resttest.New(t, storage, fakeEtcdClient.SetError)
	resourcequota := validNewResourceQuota()
	resourcequota.ObjectMeta = api.ObjectMeta{}
	test.TestCreate(
		// valid
		resourcequota,
		// invalid
		&api.ResourceQuota{
			ObjectMeta: api.ObjectMeta{Name: "_-a123-a_"},
		},
	)
}

func expectResourceQuota(t *testing.T, out runtime.Object) (*api.ResourceQuota, bool) {
	resourcequota, ok := out.(*api.ResourceQuota)
	if !ok || resourcequota == nil {
		t.Errorf("Expected an api.ResourceQuota object, was %#v", out)
		return nil, false
	}
	return resourcequota, true
}

func TestCreateRegistryError(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	fakeEtcdClient.Err = fmt.Errorf("test error")
	storage, _ := NewStorage(etcdStorage)

	resourcequota := validNewResourceQuota()
	_, err := storage.Create(api.NewDefaultContext(), resourcequota)
	if err != fakeEtcdClient.Err {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCreateSetsFields(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage, _ := NewStorage(etcdStorage)
	ctx := api.NewDefaultContext()
	resourcequota := validNewResourceQuota()
	_, err := storage.Create(api.NewDefaultContext(), resourcequota)
	if err != fakeEtcdClient.Err {
		t.Fatalf("unexpected error: %v", err)
	}

	actual := &api.ResourceQuota{}
	key, _ := storage.Etcd.KeyFunc(ctx, "foo")
	if err := etcdStorage.Get(key, actual, false); err != nil {
		t.Fatalf("unexpected extraction error: %v", err)
	}
	if actual.Name != resourcequota.Name {
		t.Errorf("unexpected resourcequota: %#v", actual)
	}
	if len(actual.UID) == 0 {
		t.Errorf("expected resourcequota UID to be set: %#v", actual)
	}
}

func TestListError(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	fakeEtcdClient.Err = fmt.Errorf("test error")
	storage, _ := NewStorage(etcdStorage)
	resourcequotas, err := storage.List(api.NewDefaultContext(), labels.Everything(), fields.Everything())
	if err != fakeEtcdClient.Err {
		t.Fatalf("Expected %#v, Got %#v", fakeEtcdClient.Err, err)
	}
	if resourcequotas != nil {
		t.Errorf("Unexpected non-nil resourcequota list: %#v", resourcequotas)
	}
}

func TestListEmptyResourceQuotaList(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	fakeEtcdClient.ChangeIndex = 1
	storage, _ := NewStorage(etcdStorage)
	ctx := api.NewContext()
	key := storage.Etcd.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)

	fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: fakeEtcdClient.NewError(tools.EtcdErrorCodeNotFound),
	}

	resourcequotas, err := storage.List(api.NewContext(), labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resourcequotas.(*api.ResourceQuotaList).Items) != 0 {
		t.Errorf("Unexpected non-zero resourcequota list: %#v", resourcequotas)
	}
	if resourcequotas.(*api.ResourceQuotaList).ResourceVersion != "1" {
		t.Errorf("Unexpected resource version: %#v", resourcequotas)
	}
}

func TestListResourceQuotaList(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage, _ := NewStorage(etcdStorage)
	ctx := api.NewDefaultContext()
	key := storage.Etcd.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.ResourceQuota{
							ObjectMeta: api.ObjectMeta{Name: "foo"},
						}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.ResourceQuota{
							ObjectMeta: api.ObjectMeta{Name: "bar"},
						}),
					},
				},
			},
		},
	}
	resourcequotasObj, err := storage.List(api.NewDefaultContext(), labels.Everything(), fields.Everything())
	resourcequotas := resourcequotasObj.(*api.ResourceQuotaList)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resourcequotas.Items) != 2 {
		t.Errorf("Unexpected resourcequota list: %#v", resourcequotas)
	}
	if resourcequotas.Items[0].Name != "foo" {
		t.Errorf("Unexpected resourcequota: %#v", resourcequotas.Items[0])
	}
	if resourcequotas.Items[1].Name != "bar" {
		t.Errorf("Unexpected resourcequota: %#v", resourcequotas.Items[1])
	}
}

func TestListResourceQuotaListSelection(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage, _ := NewStorage(etcdStorage)
	ctx := api.NewDefaultContext()
	key := storage.Etcd.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{Value: runtime.EncodeOrDie(latest.Codec, &api.ResourceQuota{
						ObjectMeta: api.ObjectMeta{Name: "foo"},
					})},
					{Value: runtime.EncodeOrDie(latest.Codec, &api.ResourceQuota{
						ObjectMeta: api.ObjectMeta{
							Name:   "qux",
							Labels: map[string]string{"label": "qux"},
						},
					})},
					{Value: runtime.EncodeOrDie(latest.Codec, &api.ResourceQuota{
						ObjectMeta: api.ObjectMeta{Name: "zot"},
					})},
				},
			},
		},
	}

	table := []struct {
		label, field string
		expectedIDs  util.StringSet
	}{
		{
			expectedIDs: util.NewStringSet("foo", "qux", "zot"),
		}, {
			field:       "name=zot",
			expectedIDs: util.NewStringSet("zot"),
		}, {
			label:       "label=qux",
			expectedIDs: util.NewStringSet("qux"),
		},
	}

	for index, item := range table {
		label, err := labels.Parse(item.label)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		field, err := fields.ParseSelector(item.field)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		resourcequotasObj, err := storage.List(ctx, label, field)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		resourcequotas := resourcequotasObj.(*api.ResourceQuotaList)

		set := util.NewStringSet()
		for i := range resourcequotas.Items {
			set.Insert(resourcequotas.Items[i].Name)
		}
		if e, a := len(item.expectedIDs), len(set); e != a {
			t.Errorf("%v: Expected %v, got %v", index, item.expectedIDs, set)
		}
	}
}

func TestResourceQuotaDecode(t *testing.T) {
	_, etcdStorage := newEtcdStorage(t)
	storage, _ := NewStorage(etcdStorage)
	expected := validNewResourceQuota()
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

func TestGet(t *testing.T) {
	expect := validNewResourceQuota()
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage, _ := NewStorage(etcdStorage)
	key := "/resourcequotas/test/foo"
	key = etcdtest.AddPrefix(key)
	fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: runtime.EncodeOrDie(latest.Codec, expect),
			},
		},
	}
	obj, err := storage.Get(api.WithNamespace(api.NewContext(), "test"), "foo")
	resourcequota := obj.(*api.ResourceQuota)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if e, a := expect, resourcequota; !api.Semantic.DeepEqual(e, a) {
		t.Errorf("Unexpected resourcequota: %s", util.ObjectDiff(e, a))
	}
}

func TestDeleteResourceQuota(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	fakeEtcdClient.ChangeIndex = 1
	storage, _ := NewStorage(etcdStorage)
	ctx := api.NewDefaultContext()
	key, _ := storage.Etcd.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeEtcdClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: runtime.EncodeOrDie(latest.Codec, &api.ResourceQuota{
					ObjectMeta: api.ObjectMeta{
						Name:      "foo",
						Namespace: api.NamespaceDefault,
					},
					Status: api.ResourceQuotaStatus{},
				}),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
	}
	_, err := storage.Delete(api.NewDefaultContext(), "foo", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// TestEtcdGetDifferentNamespace ensures same-name resourcequotas in different namespaces do not clash
func TestEtcdGetDifferentNamespace(t *testing.T) {
	registry, _, fakeClient, _ := newStorage(t)

	ctx1 := api.NewDefaultContext()
	ctx2 := api.WithNamespace(api.NewContext(), "other")

	key1, _ := registry.KeyFunc(ctx1, "foo")
	key2, _ := registry.KeyFunc(ctx2, "foo")

	key1 = etcdtest.AddPrefix(key1)
	key2 = etcdtest.AddPrefix(key2)

	fakeClient.Set(key1, runtime.EncodeOrDie(latest.Codec, &api.ResourceQuota{ObjectMeta: api.ObjectMeta{Namespace: "default", Name: "foo"}}), 0)
	fakeClient.Set(key2, runtime.EncodeOrDie(latest.Codec, &api.ResourceQuota{ObjectMeta: api.ObjectMeta{Namespace: "other", Name: "foo"}}), 0)

	obj, err := registry.Get(ctx1, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resourcequota1 := obj.(*api.ResourceQuota)
	if resourcequota1.Name != "foo" {
		t.Errorf("Unexpected resourcequota: %#v", resourcequota1)
	}
	if resourcequota1.Namespace != "default" {
		t.Errorf("Unexpected resourcequota: %#v", resourcequota1)
	}

	obj, err = registry.Get(ctx2, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resourcequota2 := obj.(*api.ResourceQuota)
	if resourcequota2.Name != "foo" {
		t.Errorf("Unexpected resourcequota: %#v", resourcequota2)
	}
	if resourcequota2.Namespace != "other" {
		t.Errorf("Unexpected resourcequota: %#v", resourcequota2)
	}

}

func TestEtcdGet(t *testing.T) {
	registry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.ResourceQuota{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)
	obj, err := registry.Get(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resourcequota := obj.(*api.ResourceQuota)
	if resourcequota.Name != "foo" {
		t.Errorf("Unexpected resourcequota: %#v", resourcequota)
	}
}

func TestEtcdGetNotFound(t *testing.T) {
	registry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	_, err := registry.Get(ctx, "foo")
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdCreateFailsWithoutNamespace(t *testing.T) {
	registry, _, fakeClient, _ := newStorage(t)
	fakeClient.TestIndex = true
	resourcequota := validNewResourceQuota()
	resourcequota.Namespace = ""
	_, err := registry.Create(api.NewContext(), resourcequota)
	// Accept "namespace" or "Namespace".
	if err == nil || !strings.Contains(err.Error(), "amespace") {
		t.Fatalf("expected error that namespace was missing from context, got: %v", err)
	}
}

func TestEtcdCreateAlreadyExisting(t *testing.T) {
	registry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: runtime.EncodeOrDie(latest.Codec, &api.ResourceQuota{ObjectMeta: api.ObjectMeta{Name: "foo"}}),
			},
		},
		E: nil,
	}
	_, err := registry.Create(ctx, validNewResourceQuota())
	if !errors.IsAlreadyExists(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdUpdateStatus(t *testing.T) {
	registry, status, fakeClient, etcdStorage := newStorage(t)
	ctx := api.NewDefaultContext()
	fakeClient.TestIndex = true

	key, _ := registry.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	resourcequotaStart := validNewResourceQuota()
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, resourcequotaStart), 1)

	resourcequotaIn := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       api.NamespaceDefault,
			ResourceVersion: "1",
		},
		Status: api.ResourceQuotaStatus{
			Used: api.ResourceList{
				api.ResourceCPU:                    resource.MustParse("1"),
				api.ResourceMemory:                 resource.MustParse("1Gi"),
				api.ResourcePods:                   resource.MustParse("1"),
				api.ResourceServices:               resource.MustParse("1"),
				api.ResourceReplicationControllers: resource.MustParse("1"),
				api.ResourceQuotas:                 resource.MustParse("1"),
			},
			Hard: api.ResourceList{
				api.ResourceCPU:                    resource.MustParse("100"),
				api.ResourceMemory:                 resource.MustParse("4Gi"),
				api.ResourcePods:                   resource.MustParse("10"),
				api.ResourceServices:               resource.MustParse("10"),
				api.ResourceReplicationControllers: resource.MustParse("10"),
				api.ResourceQuotas:                 resource.MustParse("1"),
			},
		},
	}

	expected := *resourcequotaStart
	expected.ResourceVersion = "2"
	expected.Labels = resourcequotaIn.Labels
	expected.Status = resourcequotaIn.Status

	_, _, err := status.Update(ctx, resourcequotaIn)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	var resourcequotaOut api.ResourceQuota
	key, _ = registry.KeyFunc(ctx, "foo")
	if err := etcdStorage.Get(key, &resourcequotaOut, false); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !api.Semantic.DeepEqual(expected, resourcequotaOut) {
		t.Errorf("unexpected object: %s", util.ObjectDiff(expected, resourcequotaOut))
	}
}

func TestEtcdEmptyList(t *testing.T) {
	registry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	key := registry.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{},
			},
		},
		E: nil,
	}

	obj, err := registry.List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resourcequotas := obj.(*api.ResourceQuotaList)
	if len(resourcequotas.Items) != 0 {
		t.Errorf("Unexpected resourcequota list: %#v", resourcequotas)
	}
}

func TestEtcdListNotFound(t *testing.T) {
	registry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	key := registry.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}
	obj, err := registry.List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resourcequotas := obj.(*api.ResourceQuotaList)
	if len(resourcequotas.Items) != 0 {
		t.Errorf("Unexpected resourcequota list: %#v", resourcequotas)
	}
}

func TestEtcdList(t *testing.T) {
	registry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	key := registry.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.ResourceQuota{
							ObjectMeta: api.ObjectMeta{Name: "foo"},
						}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.ResourceQuota{
							ObjectMeta: api.ObjectMeta{Name: "bar"},
						}),
					},
				},
			},
		},
		E: nil,
	}
	obj, err := registry.List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resourcequotas := obj.(*api.ResourceQuotaList)

	if len(resourcequotas.Items) != 2 || resourcequotas.Items[0].Name != "foo" || resourcequotas.Items[1].Name != "bar" {
		t.Errorf("Unexpected resourcequota list: %#v", resourcequotas)
	}
}

func TestEtcdWatchResourceQuotas(t *testing.T) {
	registry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	watching, err := registry.Watch(ctx,
		labels.Everything(),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	default:
	}
	fakeClient.WatchInjectError <- nil
	if _, ok := <-watching.ResultChan(); ok {
		t.Errorf("watching channel should be closed")
	}
	watching.Stop()
}

func TestEtcdWatchResourceQuotasMatch(t *testing.T) {
	registry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	watching, err := registry.Watch(ctx,
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	resourcequota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
			Labels: map[string]string{
				"name": "foo",
			},
		},
	}
	resourcequotaBytes, _ := latest.Codec.Encode(resourcequota)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(resourcequotaBytes),
		},
	}
	select {
	case _, ok := <-watching.ResultChan():
		if !ok {
			t.Errorf("watching channel should be open")
		}
	case <-time.After(time.Millisecond * 100):
		t.Error("unexpected timeout from result channel")
	}
	watching.Stop()
}

func TestEtcdWatchResourceQuotasNotMatch(t *testing.T) {
	registry, _, fakeClient, _ := newStorage(t)
	ctx := api.NewDefaultContext()
	watching, err := registry.Watch(ctx,
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	resourcequota := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Name: "bar",
			Labels: map[string]string{
				"name": "bar",
			},
		},
	}
	resourcequotaBytes, _ := latest.Codec.Encode(resourcequota)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(resourcequotaBytes),
		},
	}

	select {
	case <-watching.ResultChan():
		t.Error("unexpected result from result channel")
	case <-time.After(time.Millisecond * 100):
		// expected case
	}
}
