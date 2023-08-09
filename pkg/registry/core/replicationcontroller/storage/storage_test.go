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
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

const (
	namespace = metav1.NamespaceDefault
	name      = "foo"
)

func newStorage(t *testing.T) (ControllerStorage, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "replicationcontrollers",
	}
	storage, err := NewStorage(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return storage, server
}

// createController is a helper function that returns a controller with the updated resource version.
func createController(storage *REST, rc api.ReplicationController, t *testing.T) (api.ReplicationController, error) {
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), rc.Namespace)
	obj, err := storage.Create(ctx, &rc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
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
	test := genericregistrytest.New(t, storage.Controller.Store)
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
	test := genericregistrytest.New(t, storage.Controller.Store)
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
	test := genericregistrytest.New(t, storage.Controller.Store)
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
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	ctrl, err := storage.Controller.Get(ctx, rc.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	controller, _ := ctrl.(*api.ReplicationController)

	// Generation initialization
	if controller.Generation != 1 || controller.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation number %v, status generation %v", controller.Generation, controller.Status.ObservedGeneration)
	}

	// Updates to spec should increment the generation number
	controller.Spec.Replicas++
	if _, _, err := storage.Controller.Update(ctx, controller.Name, rest.DefaultUpdatedObjectInfo(controller), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
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
	controller.Status.Replicas++
	if _, _, err := storage.Controller.Update(ctx, controller.Name, rest.DefaultUpdatedObjectInfo(controller), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
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
	test := genericregistrytest.New(t, storage.Controller.Store)
	test.TestGet(validNewController())
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Controller.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Controller.Store)
	test.TestList(validNewController())
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Controller.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Controller.Store)
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
		// not matching fields
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
		t.Errorf("unexpected scale: %s", cmp.Diff(want, got))
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

	if _, _, err := storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
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

	if _, _, err = storage.Scale.Update(ctx, update.Name, rest.DefaultUpdatedObjectInfo(&update), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil && !errors.IsConflict(err) {
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

func TestScalePatchErrors(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	validObj := validController
	resourceStore := storage.Controller.Store
	scaleStore := storage.Scale

	defer resourceStore.DestroyFunc()
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)

	{
		applyNotFoundPatch := func() rest.TransformFunc {
			return func(_ context.Context, _, currentObject runtime.Object) (objToUpdate runtime.Object, patchErr error) {
				t.Errorf("notfound patch called")
				return currentObject, nil
			}
		}
		_, _, err := scaleStore.Update(ctx, "bad-name", rest.DefaultUpdatedObjectInfo(nil, applyNotFoundPatch()), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
		if !errors.IsNotFound(err) {
			t.Errorf("expected notfound, got %v", err)
		}
	}

	if _, err := resourceStore.Create(ctx, validObj, rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	{
		applyBadUIDPatch := func() rest.TransformFunc {
			return func(_ context.Context, _, currentObject runtime.Object) (objToUpdate runtime.Object, patchErr error) {
				currentObject.(*autoscaling.Scale).UID = "123"
				return currentObject, nil
			}
		}
		_, _, err := scaleStore.Update(ctx, name, rest.DefaultUpdatedObjectInfo(nil, applyBadUIDPatch()), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
		if !errors.IsConflict(err) {
			t.Errorf("expected conflict, got %v", err)
		}
	}

	{
		applyBadResourceVersionPatch := func() rest.TransformFunc {
			return func(_ context.Context, _, currentObject runtime.Object) (objToUpdate runtime.Object, patchErr error) {
				currentObject.(*autoscaling.Scale).ResourceVersion = "123"
				return currentObject, nil
			}
		}
		_, _, err := scaleStore.Update(ctx, name, rest.DefaultUpdatedObjectInfo(nil, applyBadResourceVersionPatch()), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
		if !errors.IsConflict(err) {
			t.Errorf("expected conflict, got %v", err)
		}
	}
}

func TestScalePatchConflicts(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	validObj := validController
	resourceStore := storage.Controller.Store
	scaleStore := storage.Scale

	defer resourceStore.DestroyFunc()
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), namespace)
	if _, err := resourceStore.Create(ctx, validObj, rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	applyLabelPatch := func(labelName, labelValue string) rest.TransformFunc {
		return func(_ context.Context, _, currentObject runtime.Object) (objToUpdate runtime.Object, patchErr error) {
			currentObject.(metav1.Object).SetLabels(map[string]string{labelName: labelValue})
			return currentObject, nil
		}
	}
	stopCh := make(chan struct{})
	wg := &sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		// continuously submits a patch that updates a label and verifies the label update was effective
		labelName := "timestamp"
		for i := 0; ; i++ {
			select {
			case <-stopCh:
				return
			default:
				expectedLabelValue := fmt.Sprint(i)
				updated, _, err := resourceStore.Update(ctx, name, rest.DefaultUpdatedObjectInfo(nil, applyLabelPatch(labelName, fmt.Sprint(i))), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
				if err != nil {
					t.Errorf("error patching main resource: %v", err)
					return
				}
				gotLabelValue := updated.(metav1.Object).GetLabels()[labelName]
				if gotLabelValue != expectedLabelValue {
					t.Errorf("wrong label value: expected: %s, got: %s", expectedLabelValue, gotLabelValue)
					return
				}
			}
		}
	}()

	// continuously submits a scale patch of replicas for a monotonically increasing replica value
	applyReplicaPatch := func(replicas int) rest.TransformFunc {
		return func(_ context.Context, _, currentObject runtime.Object) (objToUpdate runtime.Object, patchErr error) {
			currentObject.(*autoscaling.Scale).Spec.Replicas = int32(replicas)
			return currentObject, nil
		}
	}
	for i := 0; i < 100; i++ {
		result, _, err := scaleStore.Update(ctx, name, rest.DefaultUpdatedObjectInfo(nil, applyReplicaPatch(i)), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("error patching scale: %v", err)
		}
		scale := result.(*autoscaling.Scale)
		if scale.Spec.Replicas != int32(i) {
			t.Errorf("wrong replicas count: expected: %d got: %d", i, scale.Spec.Replicas)
		}
	}
	close(stopCh)
	wg.Wait()
}
