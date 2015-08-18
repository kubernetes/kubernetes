/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"strings"
	"testing"
	"time"

	"github.com/coreos/go-etcd/etcd"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest/resttest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/expapi/v1"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
)

const (
	PASS = iota
	FAIL
)

func init() {
	// Ensure that expapi/v1 package is used, sot that it will get initialized and register Job resource
	_ = v1.Job{}
	_ = v1.JobSpec{}
}

// newStorage creates a REST storage backed by etcd helpers
func newStorage(t *testing.T) (*REST, *tools.FakeEtcdClient) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeEtcdClient.TestIndex = true
	etcdStorage := etcdstorage.NewEtcdStorage(fakeEtcdClient, testapi.Codec(), etcdtest.PathPrefix())
	storage := NewREST(etcdStorage)
	return storage, fakeEtcdClient
}

// createJob is a helper function that returns a controller with the updated resource version.
func createJob(storage *REST, rc expapi.Job, t *testing.T) (expapi.Job, error) {
	ctx := api.WithNamespace(api.NewContext(), rc.Namespace)
	obj, err := storage.Create(ctx, &rc)
	if err != nil {
		t.Errorf("Failed to create controller, %v", err)
	}
	newJob := obj.(*expapi.Job)
	return *newJob, nil
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
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
		},
	},
}

var validCompletions = 1
var validParallelism = 1

var validJobSpec = expapi.JobSpec{
	Selector:    validPodTemplate.Template.Labels,
	Template:    &validPodTemplate.Template,
	Completions: &validCompletions,
	Parallelism: &validParallelism,
}

var validJob = expapi.Job{
	ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "default"},
	Spec:       validJobSpec,
}

// makeJobKey constructs etcd paths to controller items enforcing namespace rules.
func makeJobKey(ctx api.Context, id string) (string, error) {
	return etcdgeneric.NamespaceKeyFunc(ctx, jobPrefix, id)
}

// makeJobListKey constructs etcd paths to the root of the resource,
// not a specific controller resource
func makeJobListKey(ctx api.Context) string {
	return etcdgeneric.NamespaceKeyRootFunc(ctx, jobPrefix)
}

func TestEtcdCreateJob(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	_, err := storage.Create(ctx, &validJob)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	key, _ := makeJobKey(ctx, validJob.Name)
	key = etcdtest.AddPrefix(key)
	resp, err := fakeClient.Get(key, false, false)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	var ctrl expapi.Job
	err = testapi.Codec().DecodeInto([]byte(resp.Node.Value), &ctrl)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if ctrl.Name != "foo" {
		t.Errorf("Unexpected controller: %#v %s", ctrl, resp.Node.Value)
	}
}

func TestEtcdCreateJobAlreadyExisting(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	key, _ := makeJobKey(ctx, validJob.Name)
	key = etcdtest.AddPrefix(key)
	fakeClient.Set(key, runtime.EncodeOrDie(testapi.Codec(), &validJob), 0)

	_, err := storage.Create(ctx, &validJob)
	if !errors.IsAlreadyExists(err) {
		t.Errorf("expected already exists err, got %#v", err)
	}
}

func TestEtcdCreateJobValidates(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, _ := newStorage(t)
	emptyName := validJob
	emptyName.Name = ""
	failureCases := []expapi.Job{emptyName}
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

func TestCreateJobWithGeneratedName(t *testing.T) {
	storage, _ := newStorage(t)
	controller := &expapi.Job{
		ObjectMeta: api.ObjectMeta{
			Namespace:    api.NamespaceDefault,
			GenerateName: "rc-",
		},
		Spec: expapi.JobSpec{
			Completions: &validCompletions,
			Selector:    map[string]string{"a": "b"},
			Template:    &validPodTemplate.Template,
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

func TestCreateJobWithConflictingNamespace(t *testing.T) {
	storage, _ := newStorage(t)
	controller := &expapi.Job{
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

func TestEtcdGetJob(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	copy := validJob
	test.TestGet(&copy)
}

func TestEtcdJobValidatesUpdate(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, _ := newStorage(t)

	updateJob, err := createJob(storage, validJob, t)
	if err != nil {
		t.Errorf("Failed to create controller, cannot proceed with test.")
	}

	updaters := []func(rc expapi.Job) (runtime.Object, bool, error){
		func(rc expapi.Job) (runtime.Object, bool, error) {
			rc.UID = "newUID"
			return storage.Update(ctx, &rc)
		},
		func(rc expapi.Job) (runtime.Object, bool, error) {
			rc.Name = ""
			return storage.Update(ctx, &rc)
		},
		func(rc expapi.Job) (runtime.Object, bool, error) {
			rc.Spec.Selector = map[string]string{}
			return storage.Update(ctx, &rc)
		},
	}
	for _, u := range updaters {
		c, updated, err := u(updateJob)
		if c != nil || updated {
			t.Errorf("Expected nil object and not created")
		}
		if !errors.IsInvalid(err) && !errors.IsBadRequest(err) {
			t.Errorf("Expected invalid or bad request error, got %v of type %T", err, err)
		}
	}
}

func TestEtcdJobValidatesNamespaceOnUpdate(t *testing.T) {
	storage, _ := newStorage(t)
	ns := "newnamespace"

	// The update should fail if the namespace on the controller is set to something
	// other than the namespace on the given context, even if the namespace on the
	// controller is valid.
	updateJob, err := createJob(storage, validJob, t)

	newNamespaceJob := validJob
	newNamespaceJob.Namespace = ns
	_, err = createJob(storage, newNamespaceJob, t)

	c, updated, err := storage.Update(api.WithNamespace(api.NewContext(), ns), &updateJob)
	if c != nil || updated {
		t.Errorf("Expected nil object and not created")
	}
	// TODO: Be more paranoid about the type of error and make sure it has the substring
	// "namespace" in it, once #5684 is fixed. Ideally this would be a NewBadRequest.
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	}
}

func TestEtcdUpdateJob(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	key, _ := makeJobKey(ctx, validJob.Name)
	key = etcdtest.AddPrefix(key)

	// set a key, then retrieve the current resource version and try updating it
	resp, _ := fakeClient.Set(key, runtime.EncodeOrDie(testapi.Codec(), &validJob), 0)
	update := validJob
	update.ResourceVersion = strconv.FormatUint(resp.Node.ModifiedIndex, 10)
	completions := *validJob.Spec.Completions + 1
	update.Spec.Completions = &completions
	_, created, err := storage.Update(ctx, &update)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if created {
		t.Errorf("expected an update but created flag was returned")
	}
	ctrl, err := storage.Get(ctx, validJob.Name)
	updatedJob, _ := ctrl.(*expapi.Job)
	if *updatedJob.Spec.Completions != *validJob.Spec.Completions+1 {
		t.Errorf("Unexpected controller: %#v", ctrl)
	}
}

func TestEtcdDeleteJob(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	key, _ := makeJobKey(ctx, validJob.Name)
	key = etcdtest.AddPrefix(key)

	fakeClient.Set(key, runtime.EncodeOrDie(testapi.Codec(), &validJob), 0)
	obj, err := storage.Delete(ctx, validJob.Name, nil)
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

func TestEtcdListJobs(t *testing.T) {
	storage, fakeClient := newStorage(t)
	ctx := api.NewDefaultContext()
	key := makeJobListKey(ctx)
	key = etcdtest.AddPrefix(key)
	controller := validJob
	controller.Name = "bar"
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(testapi.Codec(), &validJob),
					},
					{
						Value: runtime.EncodeOrDie(testapi.Codec(), &controller),
					},
				},
			},
		},
		E: nil,
	}
	objList, err := storage.List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	controllers, _ := objList.(*expapi.JobList)
	if len(controllers.Items) != 2 || controllers.Items[0].Name != validJob.Name || controllers.Items[1].Name != controller.Name {
		t.Errorf("Unexpected controller list: %#v", controllers)
	}
}

func TestEtcdListJobsNotFound(t *testing.T) {
	storage, fakeClient := newStorage(t)
	ctx := api.NewDefaultContext()
	key := makeJobListKey(ctx)
	key = etcdtest.AddPrefix(key)

	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}
	objList, err := storage.List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	controllers, _ := objList.(*expapi.JobList)
	if len(controllers.Items) != 0 {
		t.Errorf("Unexpected controller list: %#v", controllers)
	}
}

func TestEtcdListJobsLabelsMatch(t *testing.T) {
	storage, fakeClient := newStorage(t)
	ctx := api.NewDefaultContext()
	key := makeJobListKey(ctx)
	key = etcdtest.AddPrefix(key)

	controller := validJob
	controller.Labels = map[string]string{"k": "v"}
	controller.Name = "bar"

	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: runtime.EncodeOrDie(testapi.Codec(), &validJob),
					},
					{
						Value: runtime.EncodeOrDie(testapi.Codec(), &controller),
					},
				},
			},
		},
		E: nil,
	}
	testLabels := labels.SelectorFromSet(labels.Set(controller.Labels))
	objList, err := storage.List(ctx, testLabels, fields.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	controllers, _ := objList.(*expapi.JobList)
	if len(controllers.Items) != 1 || controllers.Items[0].Name != controller.Name ||
		!testLabels.Matches(labels.Set(controllers.Items[0].Labels)) {
		t.Errorf("Unexpected controller list: %#v for query with labels %#v",
			controllers, testLabels)
	}
}

func TestEtcdWatchJob(t *testing.T) {
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

func TestEtcdWatchJobFields(t *testing.T) {
	ctx := api.WithNamespace(api.NewDefaultContext(), validJob.Namespace)
	storage, fakeClient := newStorage(t)
	fakeClient.ExpectNotFoundGet(etcdgeneric.NamespaceKeyRootFunc(ctx, "/registry/pods"))

	testFieldMap := map[int][]fields.Set{
		PASS: {
			{"status.successful": "0"},
			{"metadata.name": "foo"},
			{"status.successful": "0", "metadata.name": "foo"},
		},
		FAIL: {
			{"status.successful": "10"},
			{"metadata.name": "bar"},
			{"name": "foo"},
			{"status.successful": "10", "metadata.name": "foo"},
			{"status.successful": "0", "metadata.name": "bar"},
		},
	}
	testEtcdActions := []string{
		etcdstorage.EtcdCreate,
		etcdstorage.EtcdSet,
		etcdstorage.EtcdCAS,
		etcdstorage.EtcdDelete}

	controller := &expapi.Job{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Labels:    validJob.Spec.Selector,
			Namespace: "default",
		},
		Status: expapi.JobStatus{
			Successful: 0,
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

func TestEtcdWatchJobsMatch(t *testing.T) {
	ctx := api.WithNamespace(api.NewDefaultContext(), validJob.Namespace)
	storage, fakeClient := newStorage(t)
	fakeClient.ExpectNotFoundGet(etcdgeneric.NamespaceKeyRootFunc(ctx, "/registry/pods"))

	watching, err := storage.Watch(ctx,
		labels.SelectorFromSet(validJob.Spec.Selector),
		fields.Everything(),
		"1",
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	// The watcher above is waiting for these Labels, on receiving them it should
	// apply the JobStatus decorator, which lists pods, causing a query against
	// the /registry/pods endpoint of the etcd client.
	controller := &expapi.Job{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Labels:    validJob.Spec.Selector,
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

func TestEtcdWatchJobsNotMatch(t *testing.T) {
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

	controller := &expapi.Job{
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

func TestCreate(t *testing.T) {
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	test.TestCreate(
		// valid
		&expapi.Job{
			Spec: expapi.JobSpec{
				Completions: &validCompletions,
				Selector:    map[string]string{"a": "b"},
				Template:    &validPodTemplate.Template,
			},
		},
		// invalid
		&expapi.Job{
			Spec: expapi.JobSpec{
				Completions: &validCompletions,
				Selector:    map[string]string{},
				Template:    &validPodTemplate.Template,
			},
		},
	)
}

func TestDelete(t *testing.T) {
	ctx := api.NewDefaultContext()
	storage, fakeClient := newStorage(t)
	test := resttest.New(t, storage, fakeClient.SetError)
	key, _ := makeJobKey(ctx, validJob.Name)
	key = etcdtest.AddPrefix(key)

	createFn := func() runtime.Object {
		rc := validJob
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
