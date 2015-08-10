/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package service

import (
	"strconv"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	etcdservice "k8s.io/kubernetes/pkg/registry/service/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

func NewTestEtcdRegistry(client tools.EtcdClient) (Registry, *etcdservice.REST) {
	storage := etcdstorage.NewEtcdStorage(client, latest.Codec, etcdtest.PathPrefix())
	rest := etcdservice.NewStorage(storage)
	registry := NewRegistry(rest)
	return registry, rest
}

func makeTestService(name string) *api.Service {
	return &api.Service{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: "default"},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{
				{Name: "port", Protocol: api.ProtocolTCP, Port: 12345, TargetPort: util.NewIntOrStringFromInt(12345)},
			},
			Type:            api.ServiceTypeClusterIP,
			SessionAffinity: api.ServiceAffinityNone,
		},
	}
}

func TestEtcdListServicesNotFound(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	registry, rest := NewTestEtcdRegistry(fakeClient)
	ctx := api.NewDefaultContext()
	key := rest.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}
	services, err := registry.ListServices(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(services.Items) != 0 {
		t.Errorf("Unexpected services list: %#v", services)
	}
}

func TestEtcdListServices(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry, rest := NewTestEtcdRegistry(fakeClient)
	key := rest.KeyRootFunc(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, makeTestService("foo")),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, makeTestService("bar")),
					},
				},
			},
		},
		E: nil,
	}
	services, err := registry.ListServices(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(services.Items) != 2 || services.Items[0].Name != "foo" || services.Items[1].Name != "bar" {
		t.Errorf("Unexpected service list: %#v", services)
	}
}

func TestEtcdCreateService(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry, rest := NewTestEtcdRegistry(fakeClient)
	_, err := registry.CreateService(ctx, makeTestService("foo"))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	key, _ := rest.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	var service api.Service
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &service)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if service.Name != "foo" {
		t.Errorf("Unexpected service: %#v %s", service, resp.Node.Value)
	}
}

func TestEtcdCreateServiceAlreadyExisting(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry, rest := NewTestEtcdRegistry(fakeClient)
	key, _ := rest.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, makeTestService("foo")), 0)
	_, err := registry.CreateService(ctx, makeTestService("foo"))
	if !errors.IsAlreadyExists(err) {
		t.Errorf("expected already exists err, got %#v", err)
	}
}

// TestEtcdGetServiceDifferentNamespace ensures same-name services in different namespaces do not clash
func TestEtcdGetServiceDifferentNamespace(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	registry, rest := NewTestEtcdRegistry(fakeClient)

	ctx1 := api.NewDefaultContext()
	ctx2 := api.WithNamespace(api.NewContext(), "other")

	key1, _ := rest.KeyFunc(ctx1, "foo")
	key2, _ := rest.KeyFunc(ctx2, "foo")

	key1 = etcdtest.AddPrefix(key1)
	key2 = etcdtest.AddPrefix(key2)

	fakeClient.Set(key1, runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Namespace: "default", Name: "foo"}}), 0)
	fakeClient.Set(key2, runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Namespace: "other", Name: "foo"}}), 0)

	service1, err := registry.GetService(ctx1, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if service1.Name != "foo" {
		t.Errorf("Unexpected service: %#v", service1)
	}
	if service1.Namespace != "default" {
		t.Errorf("Unexpected service: %#v", service1)
	}

	service2, err := registry.GetService(ctx2, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if service2.Name != "foo" {
		t.Errorf("Unexpected service: %#v", service2)
	}
	if service2.Namespace != "other" {
		t.Errorf("Unexpected service: %#v", service2)
	}

}

func TestEtcdGetService(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry, rest := NewTestEtcdRegistry(fakeClient)
	key, _ := rest.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, makeTestService("foo")), 0)
	service, err := registry.GetService(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if service.Name != "foo" {
		t.Errorf("Unexpected service: %#v", service)
	}
}

func TestEtcdGetServiceNotFound(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry, rest := NewTestEtcdRegistry(fakeClient)
	key, _ := rest.KeyFunc(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	_, err := registry.GetService(ctx, "foo")
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdDeleteService(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry, _ := NewTestEtcdRegistry(fakeClient)
	key, _ := etcdgeneric.NamespaceKeyFunc(ctx, "/services/specs", "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, makeTestService("foo")), 0)
	path, _ := etcdgeneric.NamespaceKeyFunc(ctx, "/services/endpoints", "foo")
	endpointsKey := etcdtest.AddPrefix(path)
	fakeClient.Set(endpointsKey, runtime.EncodeOrDie(latest.Codec, &api.Endpoints{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)

	err := registry.DeleteService(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 2 delete, found %#v", fakeClient.DeletedKeys)
	}
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
}

func TestEtcdUpdateService(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	registry, rest := NewTestEtcdRegistry(fakeClient)
	key, _ := rest.KeyFunc(ctx, "uniquefoo")
	key = etcdtest.AddPrefix(key)
	resp, _ := fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, makeTestService("uniquefoo")), 0)
	testService := api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:            "uniquefoo",
			ResourceVersion: strconv.FormatUint(resp.Node.ModifiedIndex, 10),
			Labels: map[string]string{
				"baz": "bar",
			},
		},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{
				{Name: "port", Protocol: api.ProtocolTCP, Port: 12345, TargetPort: util.NewIntOrStringFromInt(12345)},
			},
			Selector: map[string]string{
				"baz": "bar",
			},
			SessionAffinity: "None",
			Type:            api.ServiceTypeClusterIP,
		},
	}
	_, err := registry.UpdateService(ctx, &testService)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	svc, err := registry.GetService(ctx, "uniquefoo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Clear modified indices before the equality test.
	svc.ResourceVersion = ""
	testService.ResourceVersion = ""
	if !api.Semantic.DeepEqual(*svc, testService) {
		t.Errorf("Unexpected service: got\n %#v\n, wanted\n %#v", svc, testService)
	}
}

func TestEtcdWatchServices(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry, _ := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchServices(ctx,
		labels.Everything(),
		fields.SelectorFromSet(fields.Set{"name": "foo"}),
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

// TODO We need a test for the compare and swap behavior.  This basically requires two things:
//   1) Add a per-operation synchronization channel to the fake etcd client, such that any operation waits on that
//      channel, this will enable us to orchestrate the flow of etcd requests in the test.
//   2) We need to make the map from key to (response, error) actually be a [](response, error) and pop
//      our way through the responses.  That will enable us to hand back multiple different responses for
//      the same key.
//   Once that infrastructure is in place, the test looks something like:
//      Routine #1                               Routine #2
//         Read
//         Wait for sync on update               Read
//                                               Update
//         Update
//   In the buggy case, this will result in lost data.  In the correct case, the second update should fail
//   and be retried.
