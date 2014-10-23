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

package controller

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
)

func TestListControllersError(t *testing.T) {
	mockRegistry := registrytest.ControllerRegistry{
		Err: fmt.Errorf("test error"),
	}
	storage := REST{
		registry: &mockRegistry,
	}
	ctx := api.NewContext()
	controllers, err := storage.List(ctx, labels.Everything(), labels.Everything())
	if err != mockRegistry.Err {
		t.Errorf("Expected %#v, Got %#v", mockRegistry.Err, err)
	}
	if controllers != nil {
		t.Errorf("Unexpected non-nil ctrl list: %#v", controllers)
	}
}

func TestListEmptyControllerList(t *testing.T) {
	mockRegistry := registrytest.ControllerRegistry{nil, &api.ReplicationControllerList{ListMeta: api.ListMeta{ResourceVersion: "1"}}}
	storage := REST{
		registry: &mockRegistry,
	}
	ctx := api.NewContext()
	controllers, err := storage.List(ctx, labels.Everything(), labels.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(controllers.(*api.ReplicationControllerList).Items) != 0 {
		t.Errorf("Unexpected non-zero ctrl list: %#v", controllers)
	}
	if controllers.(*api.ReplicationControllerList).ResourceVersion != "1" {
		t.Errorf("Unexpected resource version: %#v", controllers)
	}
}

func TestListControllerList(t *testing.T) {
	mockRegistry := registrytest.ControllerRegistry{
		Controllers: &api.ReplicationControllerList{
			Items: []api.ReplicationController{
				{
					ObjectMeta: api.ObjectMeta{
						Name: "foo",
					},
				},
				{
					ObjectMeta: api.ObjectMeta{
						Name: "bar",
					},
				},
			},
		},
	}
	storage := REST{
		registry: &mockRegistry,
	}
	ctx := api.NewContext()
	controllersObj, err := storage.List(ctx, labels.Everything(), labels.Everything())
	controllers := controllersObj.(*api.ReplicationControllerList)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(controllers.Items) != 2 {
		t.Errorf("Unexpected controller list: %#v", controllers)
	}
	if controllers.Items[0].Name != "foo" {
		t.Errorf("Unexpected controller: %#v", controllers.Items[0])
	}
	if controllers.Items[1].Name != "bar" {
		t.Errorf("Unexpected controller: %#v", controllers.Items[1])
	}
}

func TestControllerDecode(t *testing.T) {
	mockRegistry := registrytest.ControllerRegistry{}
	storage := REST{
		registry: &mockRegistry,
	}
	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
	}
	body, err := latest.Codec.Encode(controller)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	controllerOut := storage.New()
	if err := latest.Codec.DecodeInto(body, controllerOut); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(controller, controllerOut) {
		t.Errorf("Expected %#v, found %#v", controller, controllerOut)
	}
}

func TestControllerParsing(t *testing.T) {
	expectedController := api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "nginxController",
			Labels: map[string]string{
				"name": "nginx",
			},
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
	}
	file, err := ioutil.TempFile("", "controller")
	fileName := file.Name()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	data, err := json.Marshal(expectedController)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	_, err = file.Write(data)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	err = file.Close()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	data, err = ioutil.ReadFile(fileName)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	var controller api.ReplicationController
	err = json.Unmarshal(data, &controller)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(controller, expectedController) {
		t.Errorf("Parsing failed: %s %#v %#v", string(data), controller, expectedController)
	}
}

var validPodTemplate = api.PodTemplate{
	DesiredState: api.PodState{
		Manifest: api.ContainerManifest{
			Version: "v1beta1",
			Containers: []api.Container{
				{
					Name:  "test",
					Image: "test_image",
				},
			},
		},
	},
	Labels: map[string]string{"a": "b"},
}

func TestCreateController(t *testing.T) {
	mockRegistry := registrytest.ControllerRegistry{}
	mockPodRegistry := registrytest.PodRegistry{
		Pods: &api.PodList{
			Items: []api.Pod{
				{
					ObjectMeta: api.ObjectMeta{
						Name:   "foo",
						Labels: map[string]string{"a": "b"},
					},
				},
			},
		},
	}
	storage := REST{
		registry:   &mockRegistry,
		podLister:  &mockPodRegistry,
		pollPeriod: time.Millisecond * 1,
	}
	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "test"},
		DesiredState: api.ReplicationControllerState{
			Replicas:        2,
			ReplicaSelector: map[string]string{"a": "b"},
			PodTemplate:     validPodTemplate,
		},
	}
	ctx := api.NewDefaultContext()
	channel, err := storage.Create(ctx, controller)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	select {
	case <-channel:
		// expected case
	case <-time.After(time.Millisecond * 100):
		t.Error("Unexpected timeout from async channel")
	}
}

func TestControllerStorageValidatesCreate(t *testing.T) {
	mockRegistry := registrytest.ControllerRegistry{}
	storage := REST{
		registry:   &mockRegistry,
		podLister:  nil,
		pollPeriod: time.Millisecond * 1,
	}
	failureCases := map[string]api.ReplicationController{
		"empty ID": {
			ObjectMeta: api.ObjectMeta{Name: ""},
			DesiredState: api.ReplicationControllerState{
				ReplicaSelector: map[string]string{"bar": "baz"},
			},
		},
		"empty selector": {
			ObjectMeta:   api.ObjectMeta{Name: "abc"},
			DesiredState: api.ReplicationControllerState{},
		},
	}
	ctx := api.NewDefaultContext()
	for _, failureCase := range failureCases {
		c, err := storage.Create(ctx, &failureCase)
		if c != nil {
			t.Errorf("Expected nil channel")
		}
		if !errors.IsInvalid(err) {
			t.Errorf("Expected to get an invalid resource error, got %v", err)
		}
	}
}

func TestControllerStorageValidatesUpdate(t *testing.T) {
	mockRegistry := registrytest.ControllerRegistry{}
	storage := REST{
		registry:   &mockRegistry,
		podLister:  nil,
		pollPeriod: time.Millisecond * 1,
	}
	failureCases := map[string]api.ReplicationController{
		"empty ID": {
			ObjectMeta: api.ObjectMeta{Name: ""},
			DesiredState: api.ReplicationControllerState{
				ReplicaSelector: map[string]string{"bar": "baz"},
			},
		},
		"empty selector": {
			ObjectMeta:   api.ObjectMeta{Name: "abc"},
			DesiredState: api.ReplicationControllerState{},
		},
	}
	ctx := api.NewDefaultContext()
	for _, failureCase := range failureCases {
		c, err := storage.Update(ctx, &failureCase)
		if c != nil {
			t.Errorf("Expected nil channel")
		}
		if !errors.IsInvalid(err) {
			t.Errorf("Expected to get an invalid resource error, got %v", err)
		}
	}
}

type fakePodLister struct {
	e error
	l api.PodList
	s labels.Selector
}

func (f *fakePodLister) ListPods(ctx api.Context, s labels.Selector) (*api.PodList, error) {
	f.s = s
	return &f.l, f.e
}

func TestFillCurrentState(t *testing.T) {
	fakeLister := fakePodLister{
		l: api.PodList{
			Items: []api.Pod{
				{ObjectMeta: api.ObjectMeta{Name: "foo"}},
				{ObjectMeta: api.ObjectMeta{Name: "bar"}},
			},
		},
	}
	mockRegistry := registrytest.ControllerRegistry{}
	storage := REST{
		registry:  &mockRegistry,
		podLister: &fakeLister,
	}
	controller := api.ReplicationController{
		DesiredState: api.ReplicationControllerState{
			ReplicaSelector: map[string]string{
				"foo": "bar",
			},
		},
	}
	ctx := api.NewContext()
	storage.fillCurrentState(ctx, &controller)
	if controller.CurrentState.Replicas != 2 {
		t.Errorf("expected 2, got: %d", controller.CurrentState.Replicas)
	}
	if !reflect.DeepEqual(fakeLister.s, labels.Set(controller.DesiredState.ReplicaSelector).AsSelector()) {
		t.Errorf("unexpected output: %#v %#v", labels.Set(controller.DesiredState.ReplicaSelector).AsSelector(), fakeLister.s)
	}
}

func TestCreateControllerWithConflictingNamespace(t *testing.T) {
	storage := REST{}
	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "not-default"},
	}

	ctx := api.NewDefaultContext()
	channel, err := storage.Create(ctx, controller)
	if channel != nil {
		t.Error("Expected a nil channel, but we got a value")
	}
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if strings.Index(err.Error(), "Controller.Namespace does not match the provided context") == -1 {
		t.Errorf("Expected 'Controller.Namespace does not match the provided context' error, got '%v'", err.Error())
	}
}

func TestUpdateControllerWithConflictingNamespace(t *testing.T) {
	storage := REST{}
	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "not-default"},
	}

	ctx := api.NewDefaultContext()
	channel, err := storage.Update(ctx, controller)
	if channel != nil {
		t.Error("Expected a nil channel, but we got a value")
	}
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if strings.Index(err.Error(), "Controller.Namespace does not match the provided context") == -1 {
		t.Errorf("Expected 'Controller.Namespace does not match the provided context' error, got '%v'", err.Error())
	}
}
