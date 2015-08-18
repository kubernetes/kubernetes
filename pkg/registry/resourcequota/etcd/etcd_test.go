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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/rest/resttest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/registry/resourcequota"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"

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

func TestEtcdGet(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage, _ := NewStorage(etcdStorage)
	test := resttest.New(t, storage, fakeEtcdClient.SetError)
	resourcequota := validNewResourceQuota()
	test.TestGet(resourcequota)
}

func TestEtcdList(t *testing.T) {
	fakeEtcdClient, etcdStorage := newEtcdStorage(t)
	storage, _ := NewStorage(etcdStorage)
	test := resttest.New(t, storage, fakeEtcdClient.SetError)
	key := etcdtest.AddPrefix(storage.Etcd.KeyRootFunc(test.TestContext()))
	resourcequota := validNewResourceQuota()
	test.TestList(
		resourcequota,
		func(objects []runtime.Object) []runtime.Object {
			return registrytest.SetObjectsForKey(fakeEtcdClient, key, objects)
		},
		func(resourceVersion uint64) {
			registrytest.SetResourceVersion(fakeEtcdClient, resourceVersion)
		})
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
