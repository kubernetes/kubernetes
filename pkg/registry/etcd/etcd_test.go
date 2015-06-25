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

package etcd

import (
	"strconv"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/endpoint"
	endpointetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/endpoint/etcd"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	podetcd "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"

	"github.com/coreos/go-etcd/etcd"
)

func NewTestEtcdRegistry(client tools.EtcdClient) *Registry {
	helper := tools.NewEtcdHelper(client, latest.Codec, etcdtest.PathPrefix())
	registry := NewRegistry(helper, nil, nil)
	return registry
}

func NewTestEtcdRegistryWithPods(client tools.EtcdClient) *Registry {
	helper := tools.NewEtcdHelper(client, latest.Codec, etcdtest.PathPrefix())
	podStorage := podetcd.NewStorage(helper, nil)
	endpointStorage := endpointetcd.NewStorage(helper)
	registry := NewRegistry(helper, pod.NewRegistry(podStorage.Pod), endpoint.NewRegistry(endpointStorage))
	return registry
}

func TestEtcdListControllersNotFound(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ctx := api.NewDefaultContext()
	key := makeControllerListKey(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	controllers, err := registry.ListControllers(ctx)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(controllers.Items) != 0 {
		t.Errorf("Unexpected controller list: %#v", controllers)
	}
}

func TestEtcdListServicesNotFound(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	ctx := api.NewDefaultContext()
	key := makeServiceListKey(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}
	services, err := registry.ListServices(ctx)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(services.Items) != 0 {
		t.Errorf("Unexpected controller list: %#v", services)
	}
}

func TestEtcdListControllers(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	ctx := api.NewDefaultContext()
	key := makeControllerListKey(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "bar"}}),
					},
				},
			},
		},
		E: nil,
	}
	controllers, err := registry.ListControllers(ctx)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(controllers.Items) != 2 || controllers.Items[0].Name != "foo" || controllers.Items[1].Name != "bar" {
		t.Errorf("Unexpected controller list: %#v", controllers)
	}
}

// TestEtcdGetControllerDifferentNamespace ensures same-name controllers in different namespaces do not clash
func TestEtcdGetControllerDifferentNamespace(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)

	ctx1 := api.NewDefaultContext()
	ctx2 := api.WithNamespace(api.NewContext(), "other")

	key1, _ := makeControllerKey(ctx1, "foo")
	key2, _ := makeControllerKey(ctx2, "foo")

	key1 = etcdtest.AddPrefix(key1)
	key2 = etcdtest.AddPrefix(key2)

	fakeClient.Set(key1, runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Namespace: "default", Name: "foo"}}), 0)
	fakeClient.Set(key2, runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Namespace: "other", Name: "foo"}}), 0)

	ctrl1, err := registry.GetController(ctx1, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if ctrl1.Name != "foo" {
		t.Errorf("Unexpected controller: %#v", ctrl1)
	}
	if ctrl1.Namespace != "default" {
		t.Errorf("Unexpected controller: %#v", ctrl1)
	}

	ctrl2, err := registry.GetController(ctx2, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if ctrl2.Name != "foo" {
		t.Errorf("Unexpected controller: %#v", ctrl2)
	}
	if ctrl2.Namespace != "other" {
		t.Errorf("Unexpected controller: %#v", ctrl2)
	}
}

func TestEtcdGetController(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	key, _ := makeControllerKey(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)
	ctrl, err := registry.GetController(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if ctrl.Name != "foo" {
		t.Errorf("Unexpected controller: %#v", ctrl)
	}
}

func TestEtcdGetControllerNotFound(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	key, _ := makeControllerKey(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	ctrl, err := registry.GetController(ctx, "foo")
	if ctrl != nil {
		t.Errorf("Unexpected non-nil controller: %#v", ctrl)
	}
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdDeleteController(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	key, _ := makeControllerKey(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)
	err := registry.DeleteController(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	}
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
}

func TestEtcdCreateController(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	key, _ := makeControllerKey(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	_, err := registry.CreateController(ctx, &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var ctrl api.ReplicationController
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &ctrl)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if ctrl.Name != "foo" {
		t.Errorf("Unexpected pod: %#v %s", ctrl, resp.Node.Value)
	}
}

func TestEtcdCreateControllerAlreadyExisting(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	key, _ := makeControllerKey(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)

	_, err := registry.CreateController(ctx, &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
	})
	if !errors.IsAlreadyExists(err) {
		t.Errorf("expected already exists err, got %#v", err)
	}
}

func TestEtcdUpdateController(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	fakeClient.TestIndex = true
	key, _ := makeControllerKey(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	resp, _ := fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)
	_, err := registry.UpdateController(ctx, &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: strconv.FormatUint(resp.Node.ModifiedIndex, 10)},
		Spec: api.ReplicationControllerSpec{
			Replicas: 2,
		},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	ctrl, err := registry.GetController(ctx, "foo")
	if ctrl.Spec.Replicas != 2 {
		t.Errorf("Unexpected controller: %#v", ctrl)
	}
}

func TestEtcdWatchController(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchControllers(ctx,
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

func TestEtcdWatchControllersMatch(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistryWithPods(fakeClient)
	path := etcdgeneric.NamespaceKeyRootFunc(ctx, "/pods")
	path = etcdtest.AddPrefix(path)
	fakeClient.ExpectNotFoundGet(path)
	watching, err := registry.WatchControllers(ctx,
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
			Labels: map[string]string{
				"name": "foo",
			},
		},
	}
	controllerBytes, _ := latest.Codec.Encode(controller)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(controllerBytes),
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

func TestEtcdWatchControllersNotMatch(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.ExpectNotFoundGet(etcdgeneric.NamespaceKeyRootFunc(ctx, "/registry/pods"))
	registry := NewTestEtcdRegistryWithPods(fakeClient)
	watching, err := registry.WatchControllers(ctx,
		labels.SelectorFromSet(labels.Set{"name": "foo"}),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "bar",
			Labels: map[string]string{
				"name": "bar",
			},
		},
	}
	controllerBytes, _ := latest.Codec.Encode(controller)
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value: string(controllerBytes),
		},
	}

	select {
	case <-watching.ResultChan():
		t.Error("unexpected result from result channel")
	case <-time.After(time.Millisecond * 100):
		// expected case
	}
}

func TestEtcdListServices(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	key := makeServiceListKey(ctx)
	key = etcdtest.AddPrefix(key)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Name: "foo"}}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Name: "bar"}}),
					},
				},
			},
		},
		E: nil,
	}
	services, err := registry.ListServices(ctx)
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
	registry := NewTestEtcdRegistry(fakeClient)
	_, err := registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	key, _ := makeServiceKey(ctx, "foo")
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
	registry := NewTestEtcdRegistry(fakeClient)
	key, _ := makeServiceKey(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)
	_, err := registry.CreateService(ctx, &api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	})
	if !errors.IsAlreadyExists(err) {
		t.Errorf("expected already exists err, got %#v", err)
	}
}

// TestEtcdGetServiceDifferentNamespace ensures same-name services in different namespaces do not clash
func TestEtcdGetServiceDifferentNamespace(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)

	ctx1 := api.NewDefaultContext()
	ctx2 := api.WithNamespace(api.NewContext(), "other")

	key1, _ := makeServiceKey(ctx1, "foo")
	key2, _ := makeServiceKey(ctx2, "foo")

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
	registry := NewTestEtcdRegistry(fakeClient)
	key, _ := makeServiceKey(ctx, "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)
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
	registry := NewTestEtcdRegistry(fakeClient)
	key, _ := makeServiceKey(ctx, "foo")
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
	registry := NewTestEtcdRegistryWithPods(fakeClient)
	key, _ := etcdgeneric.NamespaceKeyFunc(ctx, "/services/specs", "foo")
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)
	path, _ := etcdgeneric.NamespaceKeyFunc(ctx, "/services/endpoints", "foo")
	endpointsKey := etcdtest.AddPrefix(path)
	fakeClient.Set(endpointsKey, runtime.EncodeOrDie(latest.Codec, &api.Endpoints{ObjectMeta: api.ObjectMeta{Name: "foo"}}), 0)

	err := registry.DeleteService(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeClient.DeletedKeys) != 2 {
		t.Errorf("Expected 2 delete, found %#v", fakeClient.DeletedKeys)
	}
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
	if fakeClient.DeletedKeys[1] != endpointsKey {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[1], endpointsKey)
	}
}

func TestEtcdUpdateService(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	registry := NewTestEtcdRegistry(fakeClient)
	key, _ := makeServiceKey(ctx, "uniquefoo")
	key = etcdtest.AddPrefix(key)
	resp, _ := fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Service{ObjectMeta: api.ObjectMeta{Name: "uniquefoo"}}), 0)
	testService := api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:            "uniquefoo",
			ResourceVersion: strconv.FormatUint(resp.Node.ModifiedIndex, 10),
			Labels: map[string]string{
				"baz": "bar",
			},
		},
		Spec: api.ServiceSpec{
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
	registry := NewTestEtcdRegistry(fakeClient)
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

func TestEtcdWatchServicesBadSelector(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	_, err := registry.WatchServices(
		ctx,
		labels.Everything(),
		fields.SelectorFromSet(fields.Set{"Field.Selector": "foo"}),
		"",
	)
	if err == nil {
		t.Errorf("unexpected non-error: %v", err)
	}

	_, err = registry.WatchServices(
		ctx,
		labels.SelectorFromSet(labels.Set{"Label.Selector": "foo"}),
		fields.Everything(),
		"",
	)
	if err == nil {
		t.Errorf("unexpected non-error: %v", err)
	}
}

func TestEtcdWatchEndpoints(t *testing.T) {
	ctx := api.NewDefaultContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistryWithPods(fakeClient)
	watching, err := registry.endpoints.WatchEndpoints(
		ctx,
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

func TestEtcdWatchEndpointsAcrossNamespaces(t *testing.T) {
	ctx := api.NewContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistryWithPods(fakeClient)
	watching, err := registry.endpoints.WatchEndpoints(
		ctx,
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
