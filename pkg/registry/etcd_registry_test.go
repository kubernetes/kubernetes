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

package registry

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
)

func MakeTestEtcdRegistry(client util.EtcdClient, machines []string) *EtcdRegistry {
	registry := MakeEtcdRegistry(client, MakeMinionRegistry(machines))
	registry.manifestFactory = &BasicManifestFactory{
		serviceRegistry: &MockServiceRegistry{},
	}
	return registry
}

func TestEtcdGetPod(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	fakeClient.Set("/registry/hosts/machine/pods/foo", util.MakeJSONString(api.Pod{JSONBase: api.JSONBase{ID: "foo"}}), 0)
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	pod, err := registry.GetPod("foo")
	expectNoError(t, err)
	if pod.ID != "foo" {
		t.Errorf("Unexpected pod: %#v", pod)
	}
}

func TestEtcdGetPodNotFound(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	fakeClient.Data["/registry/hosts/machine/pods/foo"] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{
			ErrorCode: 100,
		},
	}
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	_, err := registry.GetPod("foo")
	if err == nil {
		t.Errorf("Unexpected non-error.")
	}
}

func TestEtcdCreatePod(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	fakeClient.Data["/registry/hosts/machine/pods/foo"] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{ErrorCode: 100},
	}
	fakeClient.Set("/registry/hosts/machine/kubelet", util.MakeJSONString([]api.ContainerManifest{}), 0)
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	err := registry.CreatePod("machine", api.Pod{
		JSONBase: api.JSONBase{
			ID: "foo",
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
	expectNoError(t, err)
	resp, err := fakeClient.Get("/registry/hosts/machine/pods/foo", false, false)
	expectNoError(t, err)
	var pod api.Pod
	err = json.Unmarshal([]byte(resp.Node.Value), &pod)
	expectNoError(t, err)
	if pod.ID != "foo" {
		t.Errorf("Unexpected pod: %#v %s", pod, resp.Node.Value)
	}
	var manifests []api.ContainerManifest
	resp, err = fakeClient.Get("/registry/hosts/machine/kubelet", false, false)
	expectNoError(t, err)
	err = json.Unmarshal([]byte(resp.Node.Value), &manifests)
	if len(manifests) != 1 || manifests[0].Id != "foo" {
		t.Errorf("Unexpected manifest list: %#v", manifests)
	}
}

func TestEtcdCreatePodAlreadyExisting(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	fakeClient.Data["/registry/hosts/machine/pods/foo"] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: util.MakeJSONString(api.Pod{JSONBase: api.JSONBase{ID: "foo"}}),
			},
		},
		E: nil,
	}
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	err := registry.CreatePod("machine", api.Pod{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
	})
	if err == nil {
		t.Error("Unexpected non-error")
	}
}

func TestEtcdCreatePodWithContainersError(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	fakeClient.Data["/registry/hosts/machine/pods/foo"] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{ErrorCode: 100},
	}
	fakeClient.Data["/registry/hosts/machine/kubelet"] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{ErrorCode: 200},
	}
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	err := registry.CreatePod("machine", api.Pod{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
	})
	if err == nil {
		t.Error("Unexpected non-error")
	}
	_, err = fakeClient.Get("/registry/hosts/machine/pods/foo", false, false)
	if err == nil {
		t.Error("Unexpected non-error")
	}
	if err != nil && err.(*etcd.EtcdError).ErrorCode != 100 {
		t.Errorf("Unexpected error: %#v", err)
	}
}

func TestEtcdCreatePodWithContainersNotFound(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	fakeClient.Data["/registry/hosts/machine/pods/foo"] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{ErrorCode: 100},
	}
	fakeClient.Data["/registry/hosts/machine/kubelet"] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{ErrorCode: 100},
	}
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	err := registry.CreatePod("machine", api.Pod{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Id: "foo",
				Containers: []api.Container{
					{
						Name: "foo",
					},
				},
			},
		},
	})
	expectNoError(t, err)
	resp, err := fakeClient.Get("/registry/hosts/machine/pods/foo", false, false)
	expectNoError(t, err)
	var pod api.Pod
	err = json.Unmarshal([]byte(resp.Node.Value), &pod)
	expectNoError(t, err)
	if pod.ID != "foo" {
		t.Errorf("Unexpected pod: %#v %s", pod, resp.Node.Value)
	}
	var manifests []api.ContainerManifest
	resp, err = fakeClient.Get("/registry/hosts/machine/kubelet", false, false)
	expectNoError(t, err)
	err = json.Unmarshal([]byte(resp.Node.Value), &manifests)
	if len(manifests) != 1 || manifests[0].Id != "foo" {
		t.Errorf("Unexpected manifest list: %#v", manifests)
	}
}

func TestEtcdCreatePodWithExistingContainers(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	fakeClient.Data["/registry/hosts/machine/pods/foo"] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{ErrorCode: 100},
	}
	fakeClient.Set("/registry/hosts/machine/kubelet", util.MakeJSONString([]api.ContainerManifest{
		{
			Id: "bar",
		},
	}), 0)
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	err := registry.CreatePod("machine", api.Pod{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Id: "foo",
				Containers: []api.Container{
					{
						Name: "foo",
					},
				},
			},
		},
	})
	expectNoError(t, err)
	resp, err := fakeClient.Get("/registry/hosts/machine/pods/foo", false, false)
	expectNoError(t, err)
	var pod api.Pod
	err = json.Unmarshal([]byte(resp.Node.Value), &pod)
	expectNoError(t, err)
	if pod.ID != "foo" {
		t.Errorf("Unexpected pod: %#v %s", pod, resp.Node.Value)
	}
	var manifests []api.ContainerManifest
	resp, err = fakeClient.Get("/registry/hosts/machine/kubelet", false, false)
	expectNoError(t, err)
	err = json.Unmarshal([]byte(resp.Node.Value), &manifests)
	if len(manifests) != 2 || manifests[1].Id != "foo" {
		t.Errorf("Unexpected manifest list: %#v", manifests)
	}
}

func TestEtcdDeletePod(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	key := "/registry/hosts/machine/pods/foo"
	fakeClient.Set(key, util.MakeJSONString(api.Pod{JSONBase: api.JSONBase{ID: "foo"}}), 0)
	fakeClient.Set("/registry/hosts/machine/kubelet", util.MakeJSONString([]api.ContainerManifest{
		{
			Id: "foo",
		},
	}), 0)
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	err := registry.DeletePod("foo")
	expectNoError(t, err)
	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	} else if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
	response, _ := fakeClient.Get("/registry/hosts/machine/kubelet", false, false)
	if response.Node.Value != "[]" {
		t.Errorf("Unexpected container set: %s, expected empty", response.Node.Value)
	}
}

func TestEtcdDeletePodMultipleContainers(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	key := "/registry/hosts/machine/pods/foo"
	fakeClient.Set(key, util.MakeJSONString(api.Pod{JSONBase: api.JSONBase{ID: "foo"}}), 0)
	fakeClient.Set("/registry/hosts/machine/kubelet", util.MakeJSONString([]api.ContainerManifest{
		{Id: "foo"},
		{Id: "bar"},
	}), 0)
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	err := registry.DeletePod("foo")
	expectNoError(t, err)
	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	}
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
	response, _ := fakeClient.Get("/registry/hosts/machine/kubelet", false, false)
	var manifests []api.ContainerManifest
	json.Unmarshal([]byte(response.Node.Value), &manifests)
	if len(manifests) != 1 {
		t.Errorf("Unexpected manifest set: %#v, expected empty", manifests)
	}
	if manifests[0].Id != "bar" {
		t.Errorf("Deleted wrong manifest: %#v", manifests)
	}
}

func TestEtcdEmptyListPods(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	key := "/registry/hosts/machine/pods"
	fakeClient.Data[key] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{},
			},
		},
		E: nil,
	}
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	pods, err := registry.ListPods(labels.Everything())
	expectNoError(t, err)
	if len(pods) != 0 {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
}

func TestEtcdListPodsNotFound(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	key := "/registry/hosts/machine/pods"
	fakeClient.Data[key] = util.EtcdResponseWithError{
		R: &etcd.Response{},
		E: &etcd.EtcdError{ErrorCode: 100},
	}
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	pods, err := registry.ListPods(labels.Everything())
	expectNoError(t, err)
	if len(pods) != 0 {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
}

func TestEtcdListPods(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	key := "/registry/hosts/machine/pods"
	fakeClient.Data[key] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: util.MakeJSONString(api.Pod{JSONBase: api.JSONBase{ID: "foo"}}),
					},
					{
						Value: util.MakeJSONString(api.Pod{JSONBase: api.JSONBase{ID: "bar"}}),
					},
				},
			},
		},
		E: nil,
	}
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	pods, err := registry.ListPods(labels.Everything())
	expectNoError(t, err)
	if len(pods) != 2 || pods[0].ID != "foo" || pods[1].ID != "bar" {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
	if pods[0].CurrentState.Host != "machine" ||
		pods[1].CurrentState.Host != "machine" {
		t.Errorf("Failed to populate host name.")
	}
}

func TestEtcdListControllersNotFound(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	key := "/registry/controllers"
	fakeClient.Data[key] = util.EtcdResponseWithError{
		R: &etcd.Response{},
		E: &etcd.EtcdError{ErrorCode: 100},
	}
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	controllers, err := registry.ListControllers()
	expectNoError(t, err)
	if len(controllers) != 0 {
		t.Errorf("Unexpected controller list: %#v", controllers)
	}
}

func TestEtcdListServicesNotFound(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	key := "/registry/services/specs"
	fakeClient.Data[key] = util.EtcdResponseWithError{
		R: &etcd.Response{},
		E: &etcd.EtcdError{ErrorCode: 100},
	}
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	services, err := registry.ListServices()
	expectNoError(t, err)
	if len(services.Items) != 0 {
		t.Errorf("Unexpected controller list: %#v", services)
	}
}

func TestEtcdListControllers(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	key := "/registry/controllers"
	fakeClient.Data[key] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: util.MakeJSONString(api.ReplicationController{JSONBase: api.JSONBase{ID: "foo"}}),
					},
					{
						Value: util.MakeJSONString(api.ReplicationController{JSONBase: api.JSONBase{ID: "bar"}}),
					},
				},
			},
		},
		E: nil,
	}
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	controllers, err := registry.ListControllers()
	expectNoError(t, err)
	if len(controllers) != 2 || controllers[0].ID != "foo" || controllers[1].ID != "bar" {
		t.Errorf("Unexpected controller list: %#v", controllers)
	}
}

func TestEtcdGetController(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	fakeClient.Set("/registry/controllers/foo", util.MakeJSONString(api.ReplicationController{JSONBase: api.JSONBase{ID: "foo"}}), 0)
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	ctrl, err := registry.GetController("foo")
	expectNoError(t, err)
	if ctrl.ID != "foo" {
		t.Errorf("Unexpected controller: %#v", ctrl)
	}
}

func TestEtcdGetControllerNotFound(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	fakeClient.Data["/registry/controllers/foo"] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{
			ErrorCode: 100,
		},
	}
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	ctrl, err := registry.GetController("foo")
	if ctrl != nil {
		t.Errorf("Unexpected non-nil controller: %#v", ctrl)
	}
	if err == nil {
		t.Error("Unexpected non-error.")
	}
}

func TestEtcdDeleteController(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	err := registry.DeleteController("foo")
	expectNoError(t, err)
	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	}
	key := "/registry/controllers/foo"
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
}

func TestEtcdCreateController(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	err := registry.CreateController(api.ReplicationController{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
	})
	expectNoError(t, err)
	resp, err := fakeClient.Get("/registry/controllers/foo", false, false)
	expectNoError(t, err)
	var ctrl api.ReplicationController
	err = json.Unmarshal([]byte(resp.Node.Value), &ctrl)
	expectNoError(t, err)
	if ctrl.ID != "foo" {
		t.Errorf("Unexpected pod: %#v %s", ctrl, resp.Node.Value)
	}
}

func TestEtcdUpdateController(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	fakeClient.Set("/registry/controllers/foo", util.MakeJSONString(api.ReplicationController{JSONBase: api.JSONBase{ID: "foo"}}), 0)
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	err := registry.UpdateController(api.ReplicationController{
		JSONBase: api.JSONBase{ID: "foo"},
		DesiredState: api.ReplicationControllerState{
			Replicas: 2,
		},
	})
	expectNoError(t, err)
	ctrl, err := registry.GetController("foo")
	if ctrl.DesiredState.Replicas != 2 {
		t.Errorf("Unexpected controller: %#v", ctrl)
	}
}

func TestEtcdListServices(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	key := "/registry/services/specs"
	fakeClient.Data[key] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: util.MakeJSONString(api.Service{JSONBase: api.JSONBase{ID: "foo"}}),
					},
					{
						Value: util.MakeJSONString(api.Service{JSONBase: api.JSONBase{ID: "bar"}}),
					},
				},
			},
		},
		E: nil,
	}
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	services, err := registry.ListServices()
	expectNoError(t, err)
	if len(services.Items) != 2 || services.Items[0].ID != "foo" || services.Items[1].ID != "bar" {
		t.Errorf("Unexpected pod list: %#v", services)
	}
}

func TestEtcdCreateService(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	fakeClient.Data["/registry/services/specs/foo"] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{ErrorCode: 100},
	}
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	err := registry.CreateService(api.Service{
		JSONBase: api.JSONBase{ID: "foo"},
	})
	expectNoError(t, err)
	resp, err := fakeClient.Get("/registry/services/specs/foo", false, false)
	expectNoError(t, err)
	var service api.Service
	err = json.Unmarshal([]byte(resp.Node.Value), &service)
	expectNoError(t, err)
	if service.ID != "foo" {
		t.Errorf("Unexpected service: %#v %s", service, resp.Node.Value)
	}
}

func TestEtcdGetService(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	fakeClient.Set("/registry/services/specs/foo", util.MakeJSONString(api.Service{JSONBase: api.JSONBase{ID: "foo"}}), 0)
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	service, err := registry.GetService("foo")
	expectNoError(t, err)
	if service.ID != "foo" {
		t.Errorf("Unexpected pod: %#v", service)
	}
}

func TestEtcdGetServiceNotFound(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	fakeClient.Data["/registry/services/specs/foo"] = util.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{
			ErrorCode: 100,
		},
	}
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	_, err := registry.GetService("foo")
	if err == nil {
		t.Errorf("Unexpected non-error.")
	}
}

func TestEtcdDeleteService(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	err := registry.DeleteService("foo")
	expectNoError(t, err)
	if len(fakeClient.DeletedKeys) != 2 {
		t.Errorf("Expected 2 delete, found %#v", fakeClient.DeletedKeys)
	}
	key := "/registry/services/specs/foo"
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
	key = "/registry/services/endpoints/foo"
	if fakeClient.DeletedKeys[1] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[1], key)
	}
}

func TestEtcdUpdateService(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	fakeClient.Set("/registry/services/specs/foo", util.MakeJSONString(api.Service{JSONBase: api.JSONBase{ID: "foo"}}), 0)
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	testService := api.Service{
		JSONBase: api.JSONBase{ID: "foo"},
		Labels: map[string]string{
			"baz": "bar",
		},
		Selector: map[string]string{
			"baz": "bar",
		},
	}
	err := registry.UpdateService(testService)
	expectNoError(t, err)
	svc, err := registry.GetService("foo")
	expectNoError(t, err)
	if !reflect.DeepEqual(*svc, testService) {
		t.Errorf("Unexpected service: got %#v, wanted %#v", svc, testService)
	}
}

func TestEtcdUpdateEndpoints(t *testing.T) {
	fakeClient := util.MakeFakeEtcdClient(t)
	registry := MakeTestEtcdRegistry(fakeClient, []string{"machine"})
	endpoints := api.Endpoints{
		Name:      "foo",
		Endpoints: []string{"baz", "bar"},
	}
	err := registry.UpdateEndpoints(endpoints)
	expectNoError(t, err)
	response, err := fakeClient.Get("/registry/services/endpoints/foo", false, false)
	expectNoError(t, err)
	var endpointsOut api.Endpoints
	err = json.Unmarshal([]byte(response.Node.Value), &endpointsOut)
	if !reflect.DeepEqual(endpoints, endpointsOut) {
		t.Errorf("Unexpected endpoints: %#v, expected %#v", endpointsOut, endpoints)
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
