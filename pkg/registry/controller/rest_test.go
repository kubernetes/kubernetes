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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest/resttest"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func TestListControllersError(t *testing.T) {
	mockRegistry := registrytest.ControllerRegistry{
		Err: fmt.Errorf("test error"),
	}
	storage := REST{
		registry: &mockRegistry,
	}
	ctx := api.NewContext()
	controllers, err := storage.List(ctx, labels.Everything(), fields.Everything())
	if err != mockRegistry.Err {
		t.Errorf("Expected %#v, Got %#v", mockRegistry.Err, err)
	}
	if controllers != nil {
		t.Errorf("Unexpected non-nil ctrl list: %#v", controllers)
	}
}

func TestListEmptyControllerList(t *testing.T) {
	mockRegistry := registrytest.ControllerRegistry{
		Controllers: &api.ReplicationControllerList{ListMeta: api.ListMeta{ResourceVersion: "1"}},
	}
	storage := REST{
		registry: &mockRegistry,
	}
	ctx := api.NewContext()
	controllers, err := storage.List(ctx, labels.Everything(), fields.Everything())
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
	controllersObj, err := storage.List(ctx, labels.Everything(), fields.Everything())
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

// TODO: remove, this is sufficiently covered by other tests
func TestControllerDecode(t *testing.T) {
	mockRegistry := registrytest.ControllerRegistry{}
	storage := REST{
		registry: &mockRegistry,
	}
	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
		Spec: api.ReplicationControllerSpec{
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"name": "nginx",
					},
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
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

	if !api.Semantic.DeepEqual(controller, controllerOut) {
		t.Errorf("Expected %#v, found %#v", controller, controllerOut)
	}
}

// TODO: this is sufficiently covered by other tetss
func TestControllerParsing(t *testing.T) {
	expectedController := api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "nginx-controller",
			Labels: map[string]string{
				"name": "nginx",
			},
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 2,
			Selector: map[string]string{
				"name": "nginx",
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"name": "nginx",
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "dockerfile/nginx",
							Ports: []api.ContainerPort{
								{
									ContainerPort: 80,
									HostPort:      8080,
								},
							},
						},
					},
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

	if !api.Semantic.DeepEqual(controller, expectedController) {
		t.Errorf("Parsing failed: %s %#v %#v", string(data), controller, expectedController)
	}
}

var validPodTemplate = api.PodTemplate{
	Spec: api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Labels: map[string]string{"a": "b"},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            "test",
					Image:           "test_image",
					ImagePullPolicy: api.PullIfNotPresent,
				},
			},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
	},
}

// TODO: remove, this is sufficiently covered by other tests
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
		registry:  &mockRegistry,
		podLister: &mockPodRegistry,
		strategy:  Strategy,
	}
	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "test"},
		Spec: api.ReplicationControllerSpec{
			Replicas: 2,
			Selector: map[string]string{"a": "b"},
			Template: &validPodTemplate.Spec,
		},
	}
	ctx := api.NewDefaultContext()
	obj, err := storage.Create(ctx, controller)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if obj == nil {
		t.Errorf("unexpected object")
	}
	if !api.HasObjectMetaSystemFieldValues(&controller.ObjectMeta) {
		t.Errorf("storage did not populate object meta field values")
	}

}

// TODO: remove, covered by TestCreate
func TestControllerStorageValidatesCreate(t *testing.T) {
	mockRegistry := registrytest.ControllerRegistry{}
	storage := REST{
		registry:  &mockRegistry,
		podLister: nil,
		strategy:  Strategy,
	}
	failureCases := map[string]api.ReplicationController{
		"empty ID": {
			ObjectMeta: api.ObjectMeta{Name: ""},
			Spec: api.ReplicationControllerSpec{
				Selector: map[string]string{"bar": "baz"},
			},
		},
		"empty selector": {
			ObjectMeta: api.ObjectMeta{Name: "abc"},
			Spec:       api.ReplicationControllerSpec{},
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

func TestControllerValidatesUpdate(t *testing.T) {
	mockRegistry := registrytest.ControllerRegistry{
		Controllers: &api.ReplicationControllerList{},
	}
	storage := REST{
		registry:  &mockRegistry,
		podLister: nil,
		strategy:  Strategy,
	}

	var validControllerSpec = api.ReplicationControllerSpec{
		Selector: validPodTemplate.Spec.Labels,
		Template: &validPodTemplate.Spec,
	}

	var validController = api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "default"},
		Spec:       validControllerSpec,
	}

	ctx := api.NewDefaultContext()
	storage.Create(ctx, &validController)
	ns := "newNamespace"

	updaters := []func(rc api.ReplicationController) (runtime.Object, bool, error){
		func(rc api.ReplicationController) (runtime.Object, bool, error) {
			rc.UID = "newUID"
			return storage.Update(ctx, &rc)
		},
		func(rc api.ReplicationController) (runtime.Object, bool, error) {
			rc.Name = ""
			return storage.Update(ctx, &rc)
		},
		func(rc api.ReplicationController) (runtime.Object, bool, error) {
			rc.Namespace = ns
			return storage.Update(api.WithNamespace(ctx, ns), &rc)
		},
		func(rc api.ReplicationController) (runtime.Object, bool, error) {
			rc.Spec.Selector = map[string]string{}
			return storage.Update(ctx, &rc)
		},
	}
	for _, u := range updaters {
		c, updated, err := u(validController)
		if c != nil || updated {
			t.Errorf("Expected nil object and not created")
		}
		if !errors.IsInvalid(err) {
			t.Errorf("Expected to get an invalid resource error, got %v", err)
		}
	}
	// The update should fail if the namespace on the controller is set to something
	// other than the namespace on the given context, even if the namespace on the
	// controller is valid.
	c, updated, err := storage.Update(api.WithNamespace(ctx, ns), &validController)
	if c != nil || updated {
		t.Errorf("Expected nil object and not created")
	}
	errSubString := "namespace"
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if !errors.IsBadRequest(err) ||
		strings.Index(err.Error(), errSubString) == -1 {
		t.Errorf("Expected a Bad Request error with the sub string '%s', got %v", errSubString, err)
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

// TODO: remove, covered by TestCreate
func TestCreateControllerWithGeneratedName(t *testing.T) {
	storage := NewREST(&registrytest.ControllerRegistry{}, nil)
	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Namespace:    api.NamespaceDefault,
			GenerateName: "rc-",
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 2,
			Selector: map[string]string{"a": "b"},
			Template: &validPodTemplate.Spec,
		},
	}

	ctx := api.NewDefaultContext()
	_, err := storage.Create(ctx, controller)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if controller.Name == "rc-" || !strings.HasPrefix(controller.Name, "rc-") {
		t.Errorf("unexpected name: %#v", controller)
	}
}

// TODO: remove, covered by TestCreate
func TestCreateControllerWithConflictingNamespace(t *testing.T) {
	storage := REST{strategy: Strategy}
	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "not-default"},
	}

	ctx := api.NewDefaultContext()
	channel, err := storage.Create(ctx, controller)
	if channel != nil {
		t.Error("Expected a nil channel, but we got a value")
	}
	errSubString := "namespace"
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if !errors.IsBadRequest(err) ||
		strings.Index(err.Error(), errSubString) == -1 {
		t.Errorf("Expected a Bad Request error with the sub string '%s', got %v", errSubString, err)
	}
}

func TestCreate(t *testing.T) {
	registry := &registrytest.ControllerRegistry{}
	test := resttest.New(t, NewREST(registry, nil), registry.SetError)
	test.TestCreate(
		// valid
		&api.ReplicationController{
			Spec: api.ReplicationControllerSpec{
				Replicas: 2,
				Selector: map[string]string{"a": "b"},
				Template: &validPodTemplate.Spec,
			},
		},
		// invalid
		&api.ReplicationController{
			Spec: api.ReplicationControllerSpec{
				Replicas: 2,
				Selector: map[string]string{},
				Template: &validPodTemplate.Spec,
			},
		},
	)
}

func TestBeforeCreate(t *testing.T) {
	failures := []runtime.Object{
		&api.Service{},
		&api.Service{
			ObjectMeta: api.ObjectMeta{
				Name:      "foo",
				Namespace: "#$%%invalid",
			},
		},
		&api.Service{
			ObjectMeta: api.ObjectMeta{
				Name:      "##&*(&invalid",
				Namespace: api.NamespaceDefault,
			},
		},
	}
	for _, test := range failures {
		ctx := api.NewDefaultContext()
		err := rest.BeforeCreate(rest.Services, ctx, test)
		if err == nil {
			t.Errorf("unexpected non-error for %v", test)
		}
	}

	obj := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: api.ReplicationControllerSpec{
			Selector: map[string]string{"name": "foo"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"name": "foo",
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:            "foo",
							Image:           "foo",
							ImagePullPolicy: api.PullAlways,
						},
					},
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSDefault,
				},
			},
		},
		Status: api.ReplicationControllerStatus{
			Replicas: 3,
		},
	}
	ctx := api.NewDefaultContext()
	err := rest.BeforeCreate(Strategy, ctx, obj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(obj.Status, api.ReplicationControllerStatus{}) {
		t.Errorf("status was not cleared as expected.")
	}
	if obj.Name != "foo" || obj.Namespace != api.NamespaceDefault {
		t.Errorf("unexpected object metadata: %v", obj.ObjectMeta)
	}

	obj.Spec.Replicas = -1
	if err := rest.BeforeCreate(Strategy, ctx, obj); err == nil {
		t.Errorf("unexpected non-error for invalid replication controller.")
	}
}
