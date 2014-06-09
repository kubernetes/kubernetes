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

	. "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type MockControllerRegistry struct {
	err         error
	controllers []ReplicationController
}

func (registry *MockControllerRegistry) ListControllers() ([]ReplicationController, error) {
	return registry.controllers, registry.err
}

func (registry *MockControllerRegistry) GetController(ID string) (*ReplicationController, error) {
	return &ReplicationController{}, registry.err
}

func (registry *MockControllerRegistry) CreateController(controller ReplicationController) error {
	return registry.err
}

func (registry *MockControllerRegistry) UpdateController(controller ReplicationController) error {
	return registry.err
}
func (registry *MockControllerRegistry) DeleteController(ID string) error {
	return registry.err
}

func TestListControllersError(t *testing.T) {
	mockRegistry := MockControllerRegistry{
		err: fmt.Errorf("Test Error"),
	}
	storage := ControllerRegistryStorage{
		registry: &mockRegistry,
	}
	controllersObj, err := storage.List(nil)
	controllers := controllersObj.(ReplicationControllerList)
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
	controllers, err := storage.List(nil)
	expectNoError(t, err)
	if len(controllers.(ReplicationControllerList).Items) != 0 {
		t.Errorf("Unexpected non-zero ctrl list: %#v", controllers)
	}
}

func TestListControllerList(t *testing.T) {
	mockRegistry := MockControllerRegistry{
		controllers: []ReplicationController{
			ReplicationController{
				JSONBase: JSONBase{
					ID: "foo",
				},
			},
			ReplicationController{
				JSONBase: JSONBase{
					ID: "bar",
				},
			},
		},
	}
	storage := ControllerRegistryStorage{
		registry: &mockRegistry,
	}
	controllersObj, err := storage.List(nil)
	controllers := controllersObj.(ReplicationControllerList)
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
	controller := ReplicationController{
		JSONBase: JSONBase{
			ID: "foo",
		},
	}
	body, err := json.Marshal(controller)
	expectNoError(t, err)
	controllerOut, err := storage.Extract(string(body))
	expectNoError(t, err)
	// Extract adds a Kind
	controller.Kind = "cluster#replicationController"
	if !reflect.DeepEqual(controller, controllerOut) {
		t.Errorf("Expected %#v, found %#v", controller, controllerOut)
	}
}

func TestControllerParsing(t *testing.T) {
	expectedController := ReplicationController{
		JSONBase: JSONBase{
			ID: "nginxController",
		},
		DesiredState: ReplicationControllerState{
			Replicas: 2,
			ReplicasInSet: map[string]string{
				"name": "nginx",
			},
			PodTemplate: PodTemplate{
				DesiredState: PodState{
					Manifest: ContainerManifest{
						Containers: []Container{
							Container{
								Image: "dockerfile/nginx",
								Ports: []Port{
									Port{
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
	var controller ReplicationController
	err = json.Unmarshal(data, &controller)
	expectNoError(t, err)

	if !reflect.DeepEqual(controller, expectedController) {
		t.Errorf("Parsing failed: %s %#v %#v", string(data), controller, expectedController)
	}
}
