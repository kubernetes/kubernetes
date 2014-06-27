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
	"fmt"
	"io/ioutil"
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

type MockControllerRegistry struct {
	err         error
	controllers []api.ReplicationController
}

func (registry *MockControllerRegistry) ListControllers() ([]api.ReplicationController, error) {
	return registry.controllers, registry.err
}

func (registry *MockControllerRegistry) GetController(ID string) (*api.ReplicationController, error) {
	return &api.ReplicationController{}, registry.err
}

func (registry *MockControllerRegistry) CreateController(controller api.ReplicationController) error {
	return registry.err
}

func (registry *MockControllerRegistry) UpdateController(controller api.ReplicationController) error {
	return registry.err
}
func (registry *MockControllerRegistry) DeleteController(ID string) error {
	return registry.err
}

func TestListControllersError(t *testing.T) {
	mockRegistry := MockControllerRegistry{
		err: fmt.Errorf("test error"),
	}
	storage := ControllerRegistryStorage{
		registry: &mockRegistry,
	}
	controllersObj, err := storage.List(nil)
	controllers := controllersObj.(api.ReplicationControllerList)
	if err != mockRegistry.err {
		t.Errorf("Expected %#v, Got %#v", mockRegistry.err, err)
	}
	if len(controllers.Items) != 0 {
		t.Errorf("Unexpected non-zero ctrl list: %#v", controllers)
	}
}

func TestListEmptyControllerList(t *testing.T) {
	mockRegistry := MockControllerRegistry{}
	storage := ControllerRegistryStorage{
		registry: &mockRegistry,
	}
	controllers, err := storage.List(labels.Everything())
	expectNoError(t, err)
	if len(controllers.(api.ReplicationControllerList).Items) != 0 {
		t.Errorf("Unexpected non-zero ctrl list: %#v", controllers)
	}
}

func TestListControllerList(t *testing.T) {
	mockRegistry := MockControllerRegistry{
		controllers: []api.ReplicationController{
			{
				JSONBase: api.JSONBase{
					ID: "foo",
				},
			},
			{
				JSONBase: api.JSONBase{
					ID: "bar",
				},
			},
		},
	}
	storage := ControllerRegistryStorage{
		registry: &mockRegistry,
	}
	controllersObj, err := storage.List(labels.Everything())
	controllers := controllersObj.(api.ReplicationControllerList)
	expectNoError(t, err)
	if len(controllers.Items) != 2 {
		t.Errorf("Unexpected controller list: %#v", controllers)
	}
	if controllers.Items[0].ID != "foo" {
		t.Errorf("Unexpected controller: %#v", controllers.Items[0])
	}
	if controllers.Items[1].ID != "bar" {
		t.Errorf("Unexpected controller: %#v", controllers.Items[1])
	}
}

func TestExtractControllerJson(t *testing.T) {
	mockRegistry := MockControllerRegistry{}
	storage := ControllerRegistryStorage{
		registry: &mockRegistry,
	}
	controller := api.ReplicationController{
		JSONBase: api.JSONBase{
			ID: "foo",
		},
	}
	body, err := api.Encode(&controller)
	expectNoError(t, err)
	controllerOut, err := storage.Extract(body)
	expectNoError(t, err)
	if !reflect.DeepEqual(controller, controllerOut) {
		t.Errorf("Expected %#v, found %#v", controller, controllerOut)
	}
}

func TestControllerParsing(t *testing.T) {
	expectedController := api.ReplicationController{
		JSONBase: api.JSONBase{
			ID: "nginxController",
		},
		DesiredState: api.ReplicationControllerState{
			Replicas: 2,
			ReplicaSelector: map[string]string{
				"name": "nginx",
			},
			PodTemplate: api.PodTemplate{
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Image: "dockerfile/nginx",
								Ports: []api.Port{
									{
										ContainerPort: 80,
										HostPort:      8080,
									},
								},
							},
						},
					},
				},
				Labels: map[string]string{
					"name": "nginx",
				},
			},
		},
		Labels: map[string]string{
			"name": "nginx",
		},
	}
	file, err := ioutil.TempFile("", "controller")
	fileName := file.Name()
	expectNoError(t, err)
	data, err := json.Marshal(expectedController)
	expectNoError(t, err)
	_, err = file.Write(data)
	expectNoError(t, err)
	err = file.Close()
	expectNoError(t, err)
	data, err = ioutil.ReadFile(fileName)
	expectNoError(t, err)
	var controller api.ReplicationController
	err = json.Unmarshal(data, &controller)
	expectNoError(t, err)

	if !reflect.DeepEqual(controller, expectedController) {
		t.Errorf("Parsing failed: %s %#v %#v", string(data), controller, expectedController)
	}
}

func TestCreateController(t *testing.T) {
	mockRegistry := MockControllerRegistry{}
	mockPodRegistry := MockPodRegistry{
		pods: []api.Pod{
			{
				JSONBase: api.JSONBase{ID: "foo"},
			},
		},
	}
	storage := ControllerRegistryStorage{
		registry:    &mockRegistry,
		podRegistry: &mockPodRegistry,
		pollPeriod:  time.Millisecond * 1,
	}
	controller := api.ReplicationController{
		JSONBase: api.JSONBase{ID: "test"},
		DesiredState: api.ReplicationControllerState{
			Replicas: 2,
		},
	}
	channel, err := storage.Create(controller)
	expectNoError(t, err)

	select {
	case <-time.After(time.Second * 1):
		// Do nothing, this is expected.
	case <-channel:
		t.Error("Unexpected read from async channel")
	}

	mockPodRegistry.pods = []api.Pod{
		{
			JSONBase: api.JSONBase{ID: "foo"},
		},
		{
			JSONBase: api.JSONBase{ID: "bar"},
		},
	}

	time.Sleep(time.Millisecond * 30)

	select {
	case <-time.After(time.Second * 1):
		t.Error("Unexpected timeout")
	case <-channel:
		// Do nothing, this is expected
	}
}
