/*
Copyright 2014 The Kubernetes Authors.

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

package storage

import (
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

const (
	namespace = metav1.NamespaceDefault
	name      = "foo"
)

func newStorage(t *testing.T) (ControllerStorage, *etcdtesting.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "replicationcontrollers",
	}
	storage := NewStorage(restOptions)
	return storage, server
}

// createController is a helper function that returns a controller with the updated resource version.
func createController(storage *REST, rc api.ReplicationController, t *testing.T) (api.ReplicationController, error) {
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), rc.Namespace)
	obj, err := storage.Create(ctx, &rc, rest.ValidateAllObjectFunc, false)
	if err != nil {
		t.Errorf("Failed to create controller, %v", err)
	}
	newRc := obj.(*api.ReplicationController)
	return *newRc, nil
}

func validNewController() *api.ReplicationController {
	return &api.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: api.ReplicationControllerSpec{
			Selector: map[string]string{"a": "b"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:                     "test",
							Image:                    "test_image",
							ImagePullPolicy:          api.PullIfNotPresent,
							TerminationMessagePolicy: api.TerminationMessageReadFile,
						},
					},
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
	}
}

var validController = validNewController()

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Controller.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Controller.Store, legacyscheme.Scheme)
	controller := validNewController()
	controller.ObjectMeta = metav1.ObjectMeta{}
	test.TestCreate(
		// valid
		controller,
		// invalid (invalid selector)
		&api.ReplicationController{
			Spec: api.ReplicationControllerSpec{
				Replicas: 2,
				Selector: map[string]string{},
				Template: validController.Spec.Template,
			},
		},
	)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Controller.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Controller.Store, legacyscheme.Scheme)
	test.TestUpdate(
		// valid
		validNewController(),
		// valid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*api.ReplicationController)
			object.Spec.Replicas = object.Spec.Replicas + 1
			return object
		},
		// invalid updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*api.ReplicationController)
			object.Name = ""
			return object
		},
		func(obj runtime.Object) runtime.Object {
			object := obj.(*api.ReplicationController)
			object.Spec.Selector = map[string]string{}
			return object
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Controller.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Controller.Store, legacyscheme.Scheme)
	test.TestDelete(validNewController())
}

func TestGenerationNumber(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Controller.Store.DestroyFunc()
	modifiedSno := *validNewController()
	modifiedSno.Generation = 100
	modifiedSno.Status.ObservedGeneration = 10
	ctx := genericapirequest.NewDefaultContext()
	rc, err := createController(storage.Controller, modifiedSno, t)
	ctrl, err := storage.Controller.Get(ctx, rc.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	controller, _ := ctrl.(*api.ReplicationController)

	// Generation initialization
	if controller.Generation != 1 && controller.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation number %v, status generation %v", controller.Generation, controller.Status.ObservedGeneration)
	}

	// Updates to spec should increment the generation number
	controller.Spec.Replicas += 1
	storage.Controller.Update(ctx, controller.Name, rest.DefaultUpdatedObjectInfo(controller), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	ctrl, err = storage.Controller.Get(ctx, rc.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	controller, _ = ctrl.(*api.ReplicationController)
	if controller.Generation != 2 || controller.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation, spec: %v, status: %v", controller.Generation, controller.Status.ObservedGeneration)
	}

	// Updates to status should not increment either spec or status generation numbers
	controller.Status.Replicas += 1
	storage.Controller.Update(ctx, controller.Name, rest.DefaultUpdatedObjectInfo(controller), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	ctrl, err = storage.Controller.Get(ctx, rc.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	controller, _ = ctrl.(*api.ReplicationController)
	if controller.Generation != 2 || controller.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation number, spec: %v, status: %v", controller.Generation, controller.Status.ObservedGeneration)
	}
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Controller.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Controller.Store, legacyscheme.Scheme)
	test.TestGet(validNewController())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Controller.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Controller.Store, legacyscheme.Scheme)
	test.TestList(validNewController())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Controller.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Controller.Store, legacyscheme.Scheme)
	test.TestWatch(
		validController,
		// matching labels
		[]labels.Set{
			{"a": "b"},
		},
		// not matching labels
		[]labels.Set{
			{"a": "c"},
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"status.replicas": "0"},
			{"metadata.name": "foo"},
			{"status.replicas": "0", "metadata.name": "foo"},
		},
		// not matchin fields
		[]fields.Set{
			{"status.replicas": "10"},
			{"metadata.name": "bar"},
			{"name": "foo"},
			{"status.replicas": "10", "metadata.name": "foo"},
			{"status.replicas": "0", "metadata.name": "bar"},
		},
	)
}

//TODO TestUpdateStatus

func TestScaleGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Controller.Store.DestroyFunc()

	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)
	rc, err := createController(storage.Controller, *validController, t)
	if err != nil {
		t.Fatalf("error setting new replication controller %v: %v", *validController, err)
	}

	want := &autoscaling.Scale{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			Namespace:         namespace,
			UID:               rc.UID,
			ResourceVersion:   rc.ResourceVersion,
			CreationTimestamp: rc.CreationTimestamp,
		},
		Spec: autoscaling.ScaleSpec{
			Replicas: validController.Spec.Replicas,
		},
		Status: autoscaling.ScaleStatus{
			Replicas: validController.Status.Replicas,
			Selector: labels.SelectorFromSet(validController.Spec.Template.Labels).String(),
		},
	}
	obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}
	got := obj.(*autoscaling.Scale)
	if !apiequality.Semantic.DeepEqual(want, got) {
		t.Errorf("unexpected scale: %s", diff.ObjectDiff(want, got))
	}
}

func TestScaleUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Controller.Store.DestroyFunc()

	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)
	rc, err := createController(storage.Controller, *validController, t)
	if err != nil {
		t.Fatalf("error setting new replication controller %v: %v", *validController, err)
	}
	replicas := int32(12)
	update := autoscaling.Scale{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: autoscaling.ScaleSpec{
			Replicas: replicas,
		},
	}

	if _, _, err := storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc); err != nil {
		t.Fatalf("error updating scale %v: %v", update, err)
	}
	obj, err := storage.Scale.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error fetching scale for %s: %v", name, err)
	}
	scale := obj.(*autoscaling.Scale)
	if scale.Spec.Replicas != replicas {
		t.Errorf("wrong replicas count expected: %d got: %d", replicas, rc.Spec.Replicas)
	}

	update.ResourceVersion = rc.ResourceVersion
	update.Spec.Replicas = 15

	if _, _, err = storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc); err != nil && !errors.IsConflict(err) {
		t.Fatalf("unexpected error, expecting an update conflict but got %v", err)
	}
}

func TestShortNames(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Controller.Store.DestroyFunc()
	expected := []string{"rc"}
	registrytest.AssertShortNames(t, storage.Controller, expected)
}

func TestCategories(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Controller.Store.DestroyFunc()
	expected := []string{"all"}
	registrytest.AssertCategories(t, storage.Controller, expected)
}
