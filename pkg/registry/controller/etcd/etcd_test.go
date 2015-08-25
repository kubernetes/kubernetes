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

	"github.com/coreos/go-etcd/etcd"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest/resttest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
)

const (
	PASS = iota
	FAIL
)

func newStorage(t *testing.T) (*REST, *tools.FakeEtcdClient) {
	etcdStorage, fakeClient := registrytest.NewEtcdStorage(t)
	return NewREST(etcdStorage), fakeClient
}

// createController is a helper function that returns a controller with the updated resource version.
func createController(storage *REST, rc api.ReplicationController, t *testing.T) (api.ReplicationController, error) {
	ctx := api.WithNamespace(api.NewContext(), rc.Namespace)
	obj, err := storage.Create(ctx, &rc)
	if err != nil {
		t.Errorf("Failed to create controller, %v", err)
	}
	newRc := obj.(*api.ReplicationController)
	return *newRc, nil
}

var validPodTemplate = api.PodTemplate{
	Template: api.PodTemplateSpec{
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

var validControllerSpec = api.ReplicationControllerSpec{
	Selector: validPodTemplate.Template.Labels,
	Template: &validPodTemplate.Template,
}

var validController = api.ReplicationController{
	ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "default"},
	Spec:       validControllerSpec,
}

func TestCreate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	test.TestCreate(
		// valid
		&api.ReplicationController{
			Spec: api.ReplicationControllerSpec{
				Replicas: 2,
				Selector: map[string]string{"a": "b"},
				Template: &validPodTemplate.Template,
			},
		},
		func(ctx api.Context, obj runtime.Object) error {
			return registrytest.SetObject(fakeClient, storage.KeyFunc, ctx, obj)
		},
		func(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
			return registrytest.GetObject(fakeClient, storage.KeyFunc, storage.NewFunc, ctx, obj)
		},
		// invalid
		&api.ReplicationController{
			Spec: api.ReplicationControllerSpec{
				Replicas: 2,
				Selector: map[string]string{},
				Template: &validPodTemplate.Template,
			},
		},
	)
}

func TestEtcdControllerValidatesUpdate(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, _ := newStorage(t)

	updateController, err := createController(storage, validController, t)
	if err != nil {
		t.Errorf("Failed to create controller, cannot proceed with test.")
	}

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
			rc.Spec.Selector = map[string]string{}
			return storage.Update(ctx, &rc)
		},
	}
	for _, u := range updaters {
		c, updated, err := u(updateController)
		if c != nil || updated {
			t.Errorf("Expected nil object and not created")
		}
		if !errors.IsInvalid(err) && !errors.IsBadRequest(err) {
			t.Errorf("Expected invalid or bad request error, got %v of type %T", err, err)
		}
	}
}

func TestEtcdControllerValidatesNamespaceOnUpdate(t *testing.T) {
	storage, _ := newStorage(t)
	ns := "newnamespace"

	// The update should fail if the namespace on the controller is set to something
	// other than the namespace on the given context, even if the namespace on the
	// controller is valid.
	updateController, err := createController(storage, validController, t)

	newNamespaceController := validController
	newNamespaceController.Namespace = ns
	_, err = createController(storage, newNamespaceController, t)

	c, updated, err := storage.Update(api.WithNamespace(api.NewContext(), ns), &updateController)
	if c != nil || updated {
		t.Errorf("Expected nil object and not created")
	}
	// TODO: Be more paranoid about the type of error and make sure it has the substring
	// "namespace" in it, once #5684 is fixed. Ideally this would be a NewBadRequest.
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	}
}

func TestGenerationNumber(t *testing.T) {
	storage, _ := newStorage(t)
	modifiedSno := validController
	modifiedSno.Generation = 100
	modifiedSno.Status.ObservedGeneration = 10
	ctx := api.NewDefaultContext()
	rc, err := createController(storage, modifiedSno, t)
	ctrl, err := storage.Get(ctx, rc.Name)
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
	storage.Update(ctx, controller)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	ctrl, err = storage.Get(ctx, rc.Name)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	controller, _ = ctrl.(*api.ReplicationController)
	if controller.Generation != 2 || controller.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation, spec: %v, status: %v", controller.Generation, controller.Status.ObservedGeneration)
	}

	// Updates to status should not increment either spec or status generation numbers
	controller.Status.Replicas += 1
	storage.Update(ctx, controller)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	ctrl, err = storage.Get(ctx, rc.Name)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	controller, _ = ctrl.(*api.ReplicationController)
	if controller.Generation != 2 || controller.Status.ObservedGeneration != 0 {
		t.Fatalf("Unexpected generation number, spec: %v, status: %v", controller.Generation, controller.Status.ObservedGeneration)
	}
}

func TestEtcdGetController(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	copy := validController
	test.TestGet(&copy)
}

func TestEtcdListControllers(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	key := etcdtest.AddPrefix(storage.KeyRootFunc(test.TestContext()))
	copy := validController
	test.TestList(
		&copy,
		func(objects []runtime.Object) []runtime.Object {
			return registrytest.SetObjectsForKey(fakeClient, key, objects)
		},
		func(resourceVersion uint64) {
			registrytest.SetResourceVersion(fakeClient, resourceVersion)
		})
}

func TestEtcdUpdateController(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	key, _ := storage.KeyFunc(ctx, validController.Name)
	key = etcdtest.AddPrefix(key)

	// set a key, then retrieve the current resource version and try updating it
	resp, _ := fakeClient.Set(key, runtime.EncodeOrDie(testapi.Codec(), &validController), 0)
	update := validController
	update.ResourceVersion = strconv.FormatUint(resp.Node.ModifiedIndex, 10)
	update.Spec.Replicas = validController.Spec.Replicas + 1
	_, created, err := storage.Update(ctx, &update)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if created {
		t.Errorf("expected an update but created flag was returned")
	}
	ctrl, err := storage.Get(ctx, validController.Name)
	updatedController, _ := ctrl.(*api.ReplicationController)
	if updatedController.Spec.Replicas != validController.Spec.Replicas+1 {
		t.Errorf("Unexpected controller: %#v", ctrl)
	}
}

func TestEtcdDeleteController(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	key, _ := storage.KeyFunc(ctx, validController.Name)
	key = etcdtest.AddPrefix(key)

	fakeClient.Set(key, runtime.EncodeOrDie(testapi.Codec(), &validController), 0)
	obj, err := storage.Delete(ctx, validController.Name, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if status, ok := obj.(*api.Status); !ok {
		t.Errorf("Expected status of delete, got %#v", status)
	} else if status.Status != api.StatusSuccess {
		t.Errorf("Expected success, got %#v", status.Status)
	}
	if len(fakeClient.DeletedKeys) != 1 {
		t.Errorf("Expected 1 delete, found %#v", fakeClient.DeletedKeys)
	}
	if fakeClient.DeletedKeys[0] != key {
		t.Errorf("Unexpected key: %s, expected %s", fakeClient.DeletedKeys[0], key)
	}
}

func TestEtcdWatchController(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	watching, err := storage.Watch(ctx,
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
	ctx := api.WithNamespace(api.NewDefaultContext(), validController.Namespace)
	storage, fakeClient := newStorage(t)
	fakeClient.ExpectNotFoundGet(etcdgeneric.NamespaceKeyRootFunc(ctx, "/registry/pods"))

	watching, err := storage.Watch(ctx,
		labels.SelectorFromSet(validController.Spec.Selector),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	// The watcher above is waiting for these Labels, on receiving them it should
	// apply the ControllerStatus decorator, which lists pods, causing a query against
	// the /registry/pods endpoint of the etcd client.
	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Labels:    validController.Spec.Selector,
			Namespace: "default",
		},
	}
	controllerBytes, _ := testapi.Codec().Encode(controller)
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

func TestEtcdWatchControllersFields(t *testing.T) {
	ctx := api.WithNamespace(api.NewDefaultContext(), validController.Namespace)
	storage, fakeClient := newStorage(t)
	fakeClient.ExpectNotFoundGet(etcdgeneric.NamespaceKeyRootFunc(ctx, "/registry/pods"))

	testFieldMap := map[int][]fields.Set{
		PASS: {
			{"status.replicas": "0"},
			{"metadata.name": "foo"},
			{"status.replicas": "0", "metadata.name": "foo"},
		},
		FAIL: {
			{"status.replicas": "10"},
			{"metadata.name": "bar"},
			{"name": "foo"},
			{"status.replicas": "10", "metadata.name": "foo"},
			{"status.replicas": "0", "metadata.name": "bar"},
		},
	}
	testEtcdActions := []string{
		etcdstorage.EtcdCreate,
		etcdstorage.EtcdSet,
		etcdstorage.EtcdCAS,
		etcdstorage.EtcdDelete}

	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Labels:    validController.Spec.Selector,
			Namespace: "default",
		},
		Status: api.ReplicationControllerStatus{
			Replicas: 0,
		},
	}
	controllerBytes, _ := testapi.Codec().Encode(controller)

	for expectedResult, fieldSet := range testFieldMap {
		for _, field := range fieldSet {
			for _, action := range testEtcdActions {
				watching, err := storage.Watch(ctx,
					labels.Everything(),
					field.AsSelector(),
					"1",
				)
				var prevNode *etcd.Node = nil
				node := &etcd.Node{
					Value: string(controllerBytes),
				}
				if action == etcdstorage.EtcdDelete {
					prevNode = node
				}
				fakeClient.WaitForWatchCompletion()
				fakeClient.WatchResponse <- &etcd.Response{
					Action:   action,
					Node:     node,
					PrevNode: prevNode,
				}
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}

				select {
				case r, ok := <-watching.ResultChan():
					if expectedResult == FAIL {
						t.Errorf("Unexpected result from channel %#v", r)
					}
					if !ok {
						t.Errorf("watching channel should be open")
					}
				case <-time.After(time.Millisecond * 100):
					if expectedResult == PASS {
						t.Error("unexpected timeout from result channel")
					}
				}
				watching.Stop()
			}
		}
	}
}

func TestEtcdWatchControllersNotMatch(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	fakeClient.ExpectNotFoundGet(etcdgeneric.NamespaceKeyRootFunc(ctx, "/registry/pods"))

	watching, err := storage.Watch(ctx,
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
	controllerBytes, _ := testapi.Codec().Encode(controller)
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

func TestDelete(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	key, _ := storage.KeyFunc(ctx, validController.Name)
	key = etcdtest.AddPrefix(key)

	createFn := func() runtime.Object {
		rc := validController
		rc.ResourceVersion = "1"
		fakeClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{
				Node: &etcd.Node{
					Value:         runtime.EncodeOrDie(testapi.Codec(), &rc),
					ModifiedIndex: 1,
				},
			},
		}
		return &rc
	}
	gracefulSetFn := func() bool {
		// If the controller is still around after trying to delete either the delete
		// failed, or we're deleting it gracefully.
		if fakeClient.Data[key].R.Node != nil {
			return true
		}
		return false
	}

	test.TestDelete(createFn, gracefulSetFn)
}
