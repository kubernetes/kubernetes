/*
Copyright 2014 Google Inc. All rights reserved.

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
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"

	"github.com/coreos/go-etcd/etcd"
)

func NewTestEtcdRegistry(client tools.EtcdClient) *Registry {
	registry := NewRegistry(tools.EtcdHelper{client, latest.Codec, latest.ResourceVersioner})
	registry.manifestFactory = &BasicManifestFactory{
		serviceRegistry: &registrytest.ServiceRegistry{},
	}
	return registry
}

func TestEtcdGetPod(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ns := api.NamespaceDefault
	id := "foo"
	fakeClient.Set(makePodKey(ns, id), runtime.EncodeOrDie(latest.Codec, &api.Pod{JSONBase: api.JSONBase{ID: id, Namespace: ns}}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	pod, err := registry.GetPod(ns, id)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if pod.ID != id {
		t.Errorf("Unexpected pod: %#v", pod)
	}
}

func TestEtcdGetPodNotFound(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ns := api.NamespaceDefault
	id := "foo"
	fakeClient.Data[makePodKey(ns, id)] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	_, err := registry.GetPod(ns, id)
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdCreatePod(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	ns := api.NamespaceDefault
	id := "foo"
	fakeClient.Data[makePodKey(ns, id)] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	fakeClient.Set("/registry/hosts/machine/kubelet", runtime.EncodeOrDie(latest.Codec, &api.ContainerManifestList{}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreatePod(&api.Pod{
		JSONBase: api.JSONBase{
			ID:        id,
			Namespace: ns,
		},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Containers: []api.Container{
					{
						Name: "foo",
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	err = registry.ApplyBinding(&api.Binding{JSONBase: api.JSONBase{Namespace: ns}, PodID: id, Host: "machine"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	resp, err := fakeClient.Get(makePodKey(ns, id), false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var pod api.Pod
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if pod.ID != id {
		t.Errorf("Unexpected pod: %#v %s", pod, resp.Node.Value)
	}
	var manifests api.ContainerManifestList
	resp, err = fakeClient.Get("/registry/hosts/machine/kubelet", false, false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &manifests)
	if len(manifests.Items) != 1 || manifests.Items[0].ID != "foo" {
		t.Errorf("Unexpected manifest list: %#v", manifests)
	}
}

func TestEtcdCreatePodAlreadyExisting(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ns := api.NamespaceDefault
	id := "foo"
	fakeClient.Data[makePodKey(ns, id)] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: runtime.EncodeOrDie(latest.Codec, &api.Pod{JSONBase: api.JSONBase{ID: id, Namespace: ns}}),
			},
		},
		E: nil,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreatePod(&api.Pod{
		JSONBase: api.JSONBase{
			ID:        id,
			Namespace: ns,
		},
	})
	if !errors.IsAlreadyExists(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdCreatePodWithContainersError(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	ns := api.NamespaceDefault
	id := "foo"
	fakeClient.Data[makePodKey(ns, id)] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	fakeClient.Data["/registry/hosts/machine/kubelet"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNodeExist, // validate that ApplyBinding is translating Create errors
	}
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreatePod(&api.Pod{
		JSONBase: api.JSONBase{
			ID:        id,
			Namespace: ns,
		},
	})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	err = registry.ApplyBinding(&api.Binding{JSONBase: api.JSONBase{Namespace: ns}, PodID: id, Host: "machine"})
	if !errors.IsAlreadyExists(err) {
		t.Fatalf("Unexpected error returned: %#v", err)
	}

	existingPod, err := registry.GetPod(api.NamespaceDefault, "foo")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if existingPod.DesiredState.Host == "machine" {
		t.Fatal("Pod's host changed in response to an non-apply-able binding.")
	}
}

func TestEtcdCreatePodWithContainersNotFound(t *testing.T) {
	id := "foo"
	ns := api.NamespaceDefault
	key := makePodKey(ns, id)
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	fakeClient.Data["/registry/hosts/machine/kubelet"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreatePod(&api.Pod{
		JSONBase: api.JSONBase{
			ID:        id,
			Namespace: ns,
		},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				ID: id,
				Containers: []api.Container{
					{
						Name: "foo",
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	err = registry.ApplyBinding(&api.Binding{JSONBase: api.JSONBase{Namespace: ns}, PodID: id, Host: "machine"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var pod api.Pod
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if pod.ID != id {
		t.Errorf("Unexpected pod: %#v %s", pod, resp.Node.Value)
	}
	var manifests api.ContainerManifestList
	resp, err = fakeClient.Get("/registry/hosts/machine/kubelet", false, false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &manifests)
	if len(manifests.Items) != 1 || manifests.Items[0].ID != "foo" {
		t.Errorf("Unexpected manifest list: %#v", manifests)
	}
}

func TestEtcdCreatePodWithExistingContainers(t *testing.T) {
	id := "foo"
	ns := api.NamespaceDefault
	key := makePodKey(ns, id)
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	fakeClient.Set("/registry/hosts/machine/kubelet", runtime.EncodeOrDie(latest.Codec, &api.ContainerManifestList{
		Items: []api.ContainerManifest{
			{ID: "bar"},
		},
	}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreatePod(&api.Pod{
		JSONBase: api.JSONBase{
			ID:        id,
			Namespace: ns,
		},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				ID: "foo",
				Containers: []api.Container{
					{
						Name: "foo",
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Suddenly, a wild scheduler appears:
	err = registry.ApplyBinding(&api.Binding{JSONBase: api.JSONBase{Namespace: ns}, PodID: "foo", Host: "machine"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var pod api.Pod
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if pod.ID != id {
		t.Errorf("Unexpected pod: %#v %s", pod, resp.Node.Value)
	}
	var manifests api.ContainerManifestList
	resp, err = fakeClient.Get("/registry/hosts/machine/kubelet", false, false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &manifests)
	if len(manifests.Items) != 2 || manifests.Items[1].ID != "foo" {
		t.Errorf("Unexpected manifest list: %#v", manifests)
	}
}

func TestEtcdDeletePod(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true

	ns := api.NamespaceDefault
	id := "foo"
	key := makePodKey(ns, id)

	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Pod{
		JSONBase:     api.JSONBase{ID: "foo", Namespace: api.NamespaceDefault},
		DesiredState: api.PodState{Host: "machine"},
	}), 0)
	fakeClient.Set("/registry/hosts/machine/kubelet", runtime.EncodeOrDie(latest.Codec, &api.ContainerManifestList{
		Items: []api.ContainerManifest{
			{ID: "foo"},
		},
	}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.DeletePod(api.NamespaceDefault, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	} else if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
	response, err := fakeClient.Get("/registry/hosts/machine/kubelet", false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var manifests api.ContainerManifestList
	latest.Codec.DecodeInto([]byte(response.Node.Value), &manifests)
	if len(manifests.Items) != 0 {
		t.Errorf("Unexpected container set: %s, expected empty", response.Node.Value)
	}
}

func TestEtcdDeletePodMultipleContainers(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true

	ns := api.NamespaceDefault
	id := "foo"
	key := makePodKey(ns, id)

	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Pod{
		JSONBase:     api.JSONBase{ID: "foo", Namespace: api.NamespaceDefault},
		DesiredState: api.PodState{Host: "machine"},
	}), 0)
	fakeClient.Set("/registry/hosts/machine/kubelet", runtime.EncodeOrDie(latest.Codec, &api.ContainerManifestList{
		Items: []api.ContainerManifest{
			{ID: "foo"},
			{ID: "bar"},
		},
	}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.DeletePod(api.NamespaceDefault, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	}
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
	response, err := fakeClient.Get("/registry/hosts/machine/kubelet", false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var manifests api.ContainerManifestList
	latest.Codec.DecodeInto([]byte(response.Node.Value), &manifests)
	if len(manifests.Items) != 1 {
		t.Fatalf("Unexpected manifest set: %#v, expected empty", manifests)
	}
	if manifests.Items[0].ID != "bar" {
		t.Errorf("Deleted wrong manifest: %#v", manifests)
	}
}

func TestEtcdEmptyListPods(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ns := api.NamespaceDefault
	key := makePodKey(ns, "")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{},
			},
		},
		E: nil,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	pods, err := registry.ListPods(ns, labels.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(pods.Items) != 0 {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
}

func TestEtcdListPodsNotFound(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ns := api.NamespaceDefault
	key := makePodKey(ns, "")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	pods, err := registry.ListPods(ns, labels.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(pods.Items) != 0 {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
}

func TestEtcdListPods(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ns := api.NamespaceDefault
	key := makePodKey(ns, "")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Pod{
							JSONBase:     api.JSONBase{ID: "foo", Namespace: ns},
							DesiredState: api.PodState{Host: "machine"},
						}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Pod{
							JSONBase:     api.JSONBase{ID: "bar", Namespace: ns},
							DesiredState: api.PodState{Host: "machine"},
						}),
					},
				},
			},
		},
		E: nil,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	pods, err := registry.ListPods(ns, labels.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(pods.Items) != 2 || pods.Items[0].ID != "foo" || pods.Items[1].ID != "bar" {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
	if pods.Items[0].CurrentState.Host != "machine" ||
		pods.Items[1].CurrentState.Host != "machine" {
		t.Errorf("Failed to populate host name.")
	}
}

func TestEtcdListControllersNotFound(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	key := makeControllerKey(api.NamespaceDefault, "")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	controllers, err := registry.ListControllers(api.NamespaceDefault)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(controllers.Items) != 0 {
		t.Errorf("Unexpected controller list: %#v", controllers)
	}
}

func TestEtcdListServicesNotFound(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ns := api.NamespaceDefault
	key := makeServiceKey(ns, "")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	services, err := registry.ListServices(ns)
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
	ns := api.NamespaceDefault
	key := makeControllerKey(ns, "")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{JSONBase: api.JSONBase{ID: "foo", Namespace: ns}}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{JSONBase: api.JSONBase{ID: "bar", Namespace: ns}}),
					},
				},
			},
		},
		E: nil,
	}
	controllers, err := registry.ListControllers(ns)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(controllers.Items) != 2 || controllers.Items[0].ID != "foo" || controllers.Items[1].ID != "bar" {
		t.Errorf("Unexpected controller list: %#v", controllers)
	}
}

func TestEtcdGetController(t *testing.T) {
	ns := api.NamespaceDefault
	id := "foo"
	key := makeControllerKey(ns, id)

	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{JSONBase: api.JSONBase{ID: id, Namespace: ns}}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	ctrl, err := registry.GetController(ns, id)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if ctrl.ID != id {
		t.Errorf("Unexpected controller: %#v", ctrl)
	}
}

func TestEtcdGetControllerNotFound(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.Data["/registry/controllers/"+api.NamespaceDefault+"/foo"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	ctrl, err := registry.GetController(api.NamespaceDefault, "foo")
	if ctrl != nil {
		t.Errorf("Unexpected non-nil controller: %#v", ctrl)
	}
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdDeleteController(t *testing.T) {
	ns := api.NamespaceDefault
	id := "foo"
	key := makeControllerKey(ns, id)
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.DeleteController(ns, id)
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
	ns := api.NamespaceDefault
	id := "foo"
	key := makeControllerKey(ns, id)

	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreateController(&api.ReplicationController{
		JSONBase: api.JSONBase{
			ID:        id,
			Namespace: ns,
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

	if ctrl.ID != "foo" {
		t.Errorf("Unexpected pod: %#v %s", ctrl, resp.Node.Value)
	}
}

func TestEtcdCreateControllerAlreadyExisting(t *testing.T) {
	ns := api.NamespaceDefault
	id := "foo"
	key := makeControllerKey(ns, id)

	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{JSONBase: api.JSONBase{ID: id, Namespace: ns}}), 0)

	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreateController(&api.ReplicationController{
		JSONBase: api.JSONBase{
			ID:        id,
			Namespace: ns,
		},
	})
	if !errors.IsAlreadyExists(err) {
		t.Errorf("expected already exists err, got %#v", err)
	}
}

func TestEtcdUpdateController(t *testing.T) {
	ns := api.NamespaceDefault
	id := "foo"
	key := makeControllerKey(ns, id)

	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true

	resp, _ := fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.ReplicationController{JSONBase: api.JSONBase{ID: id, Namespace: ns}}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.UpdateController(&api.ReplicationController{
		JSONBase: api.JSONBase{ID: id, Namespace: ns, ResourceVersion: resp.Node.ModifiedIndex},
		DesiredState: api.ReplicationControllerState{
			Replicas: 2,
		},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	ctrl, err := registry.GetController(ns, id)
	if ctrl.DesiredState.Replicas != 2 {
		t.Errorf("Unexpected controller: %#v", ctrl)
	}
}

func TestEtcdListServices(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ns := api.NamespaceDefault
	key := makeServiceKey(ns, "")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Service{JSONBase: api.JSONBase{ID: "foo", Namespace: ns}}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Service{JSONBase: api.JSONBase{ID: "bar", Namespace: ns}}),
					},
				},
			},
		},
		E: nil,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	services, err := registry.ListServices(ns)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(services.Items) != 2 || services.Items[0].ID != "foo" || services.Items[1].ID != "bar" {
		t.Errorf("Unexpected service list: %#v", services)
	}
}

func TestEtcdCreateService(t *testing.T) {
	ns := api.NamespaceDefault
	id := "foo"
	key := makeServiceKey(ns, id)

	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreateService(&api.Service{
		JSONBase: api.JSONBase{ID: id, Namespace: ns},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	var service api.Service
	err = latest.Codec.DecodeInto([]byte(resp.Node.Value), &service)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if service.ID != id {
		t.Errorf("Unexpected service: %#v %s", service, resp.Node.Value)
	}
}

func TestEtcdCreateServiceAlreadyExisting(t *testing.T) {
	ns := api.NamespaceDefault
	id := "foo"
	key := makeServiceKey(ns, id)

	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Service{JSONBase: api.JSONBase{ID: id, Namespace: ns}}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.CreateService(&api.Service{
		JSONBase: api.JSONBase{ID: id, Namespace: ns},
	})
	if !errors.IsAlreadyExists(err) {
		t.Errorf("expected already exists err, got %#v", err)
	}
}

func TestEtcdGetService(t *testing.T) {
	ns := api.NamespaceDefault
	id := "foo"
	key := makeServiceKey(ns, id)

	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Service{JSONBase: api.JSONBase{ID: id, Namespace: ns}}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	service, err := registry.GetService(ns, id)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if service.ID != id {
		t.Errorf("Unexpected service: %#v", service)
	}
}

func TestEtcdGetServiceNotFound(t *testing.T) {
	ns := api.NamespaceDefault
	id := "foo"
	key := makeServiceKey(ns, id)

	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: tools.EtcdErrorNotFound,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	_, err := registry.GetService(ns, id)
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error returned: %#v", err)
	}
}

func TestEtcdDeleteService(t *testing.T) {
	ns := api.NamespaceDefault
	id := "foo"
	key := makeServiceKey(ns, id)

	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	err := registry.DeleteService(ns, id)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeClient.DeletedKeys) != 2 {
		t.Errorf("Expected 2 delete, found %#v", fakeClient.DeletedKeys)
	}

	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}

	endpointsKey := makeServiceEndpointsKey(ns, id)
	if fakeClient.DeletedKeys[1] != endpointsKey {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[1], key)
	}
}

func TestEtcdUpdateService(t *testing.T) {
	ns := api.NamespaceDefault
	id := "foo"
	key := makeServiceKey(ns, id)

	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true

	resp, _ := fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Service{JSONBase: api.JSONBase{ID: id, Namespace: ns}}), 0)
	registry := NewTestEtcdRegistry(fakeClient)
	testService := api.Service{
		JSONBase: api.JSONBase{ID: id, Namespace: ns, ResourceVersion: resp.Node.ModifiedIndex},
		Labels: map[string]string{
			"baz": "bar",
		},
		Selector: map[string]string{
			"baz": "bar",
		},
	}
	err := registry.UpdateService(&testService)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	svc, err := registry.GetService(ns, id)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Clear modified indices before the equality test.
	svc.ResourceVersion = 0
	testService.ResourceVersion = 0
	if !reflect.DeepEqual(*svc, testService) {
		t.Errorf("Unexpected service: got\n %#v\n, wanted\n %#v", svc, testService)
	}
}

func TestEtcdListEndpoints(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	ns := api.NamespaceDefault
	key := makeServiceEndpointsKey(ns, "")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Endpoints{JSONBase: api.JSONBase{ID: "foo", Namespace: ns}, Endpoints: []string{"127.0.0.1:8345"}}),
					},
					{
						Value: runtime.EncodeOrDie(latest.Codec, &api.Endpoints{JSONBase: api.JSONBase{ID: "bar", Namespace: ns}}),
					},
				},
			},
		},
		E: nil,
	}
	registry := NewTestEtcdRegistry(fakeClient)
	services, err := registry.ListEndpoints(ns)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(services.Items) != 2 || services.Items[0].ID != "foo" || services.Items[1].ID != "bar" {
		t.Errorf("Unexpected endpoints list: %#v", services)
	}
}

func TestEtcdGetEndpoints(t *testing.T) {
	id := "foo"
	ns := api.NamespaceDefault
	key := makeServiceEndpointsKey(ns, id)

	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	endpoints := &api.Endpoints{
		JSONBase:  api.JSONBase{ID: id, Namespace: ns},
		Endpoints: []string{"127.0.0.1:34855"},
	}

	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, endpoints), 0)

	got, err := registry.GetEndpoints(ns, id)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if e, a := endpoints, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Unexpected endpoints: %#v, expected %#v", e, a)
	}
}

func TestEtcdUpdateEndpoints(t *testing.T) {
	id := "foo"
	ns := api.NamespaceDefault
	key := makeServiceEndpointsKey(ns, id)

	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	registry := NewTestEtcdRegistry(fakeClient)
	endpoints := api.Endpoints{
		JSONBase:  api.JSONBase{ID: id, Namespace: ns},
		Endpoints: []string{"baz", "bar"},
	}

	fakeClient.Set(key, runtime.EncodeOrDie(latest.Codec, &api.Endpoints{}), 0)

	err := registry.UpdateEndpoints(&endpoints)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	response, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var endpointsOut api.Endpoints
	err = latest.Codec.DecodeInto([]byte(response.Node.Value), &endpointsOut)
	if !reflect.DeepEqual(endpoints, endpointsOut) {
		t.Errorf("Unexpected endpoints: %#v, expected %#v", endpointsOut, endpoints)
	}
}

func TestEtcdWatchServices(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchServices(
		labels.Everything(),
		labels.SelectorFromSet(labels.Set{"ID": "foo"}),
		1,
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
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	_, err := registry.WatchServices(
		labels.Everything(),
		labels.SelectorFromSet(labels.Set{"Field.Selector": "foo"}),
		0,
	)
	if err == nil {
		t.Errorf("unexpected non-error: %v", err)
	}

	_, err = registry.WatchServices(
		labels.SelectorFromSet(labels.Set{"Label.Selector": "foo"}),
		labels.Everything(),
		0,
	)
	if err == nil {
		t.Errorf("unexpected non-error: %v", err)
	}
}

func TestEtcdWatchEndpoints(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	watching, err := registry.WatchEndpoints(
		labels.Everything(),
		labels.SelectorFromSet(labels.Set{"ID": "foo"}),
		1,
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

func TestEtcdWatchEndpointsBadSelector(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	registry := NewTestEtcdRegistry(fakeClient)
	_, err := registry.WatchEndpoints(
		labels.Everything(),
		labels.SelectorFromSet(labels.Set{"Field.Selector": "foo"}),
		0,
	)
	if err == nil {
		t.Errorf("unexpected non-error: %v", err)
	}

	_, err = registry.WatchEndpoints(
		labels.SelectorFromSet(labels.Set{"Label.Selector": "foo"}),
		labels.Everything(),
		0,
	)
	if err == nil {
		t.Errorf("unexpected non-error: %v", err)
	}
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
