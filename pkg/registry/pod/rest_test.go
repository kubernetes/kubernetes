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

package pod

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest/resttest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type fakeCache struct {
	requestedNamespace string
	requestedName      string
	clearedNamespace   string
	clearedName        string

	statusToReturn *api.PodStatus
	errorToReturn  error
}

func (f *fakeCache) GetPodStatus(namespace, name string) (*api.PodStatus, error) {
	f.requestedNamespace = namespace
	f.requestedName = name
	return f.statusToReturn, f.errorToReturn
}

func (f *fakeCache) ClearPodStatus(namespace, name string) {
	f.clearedNamespace = namespace
	f.clearedName = name
}

func expectApiStatusError(t *testing.T, ch <-chan apiserver.RESTResult, msg string) {
	out := <-ch
	status, ok := out.Object.(*api.Status)
	if !ok {
		t.Errorf("Expected an api.Status object, was %#v", out)
		return
	}
	if msg != status.Message {
		t.Errorf("Expected %#v, was %s", msg, status.Message)
	}
}

func expectPod(t *testing.T, ch <-chan apiserver.RESTResult) (*api.Pod, bool) {
	out := <-ch
	pod, ok := out.Object.(*api.Pod)
	if !ok || pod == nil {
		t.Errorf("Expected an api.Pod object, was %#v", out)
		return nil, false
	}
	return pod, true
}

func TestCreatePodRegistryError(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := REST{
		registry: podRegistry,
		podCache: &fakeCache{statusToReturn: &api.PodStatus{}},
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
	}
	ctx := api.NewDefaultContext()
	ch, err := storage.Create(ctx, pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectApiStatusError(t, ch, podRegistry.Err.Error())
}

func TestCreatePodSetsIds(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := REST{
		registry: podRegistry,
		podCache: &fakeCache{statusToReturn: &api.PodStatus{}},
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
	}
	ctx := api.NewDefaultContext()
	ch, err := storage.Create(ctx, pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectApiStatusError(t, ch, podRegistry.Err.Error())

	if len(podRegistry.Pod.Name) == 0 {
		t.Errorf("Expected pod ID to be set, Got %#v", pod)
	}
	if pod.Name != podRegistry.Pod.Name {
		t.Errorf("Expected manifest ID to be equal to pod ID, Got %#v", pod)
	}
}

func TestCreatePodSetsUID(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := REST{
		registry: podRegistry,
		podCache: &fakeCache{statusToReturn: &api.PodStatus{}},
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
	}
	ctx := api.NewDefaultContext()
	ch, err := storage.Create(ctx, pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectApiStatusError(t, ch, podRegistry.Err.Error())

	if len(podRegistry.Pod.UID) == 0 {
		t.Errorf("Expected pod UID to be set, Got %#v", pod)
	}
}

func TestListPodsError(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := REST{
		registry: podRegistry,
		podCache: &fakeCache{statusToReturn: &api.PodStatus{}},
	}
	ctx := api.NewContext()
	pods, err := storage.List(ctx, labels.Everything(), labels.Everything())
	if err != podRegistry.Err {
		t.Errorf("Expected %#v, Got %#v", podRegistry.Err, err)
	}
	if pods.(*api.PodList) != nil {
		t.Errorf("Unexpected non-nil pod list: %#v", pods)
	}
}

func TestListPodsCacheError(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Pods = &api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
		},
	}
	storage := REST{
		registry: podRegistry,
		podCache: &fakeCache{errorToReturn: client.ErrPodInfoNotAvailable},
	}
	ctx := api.NewContext()
	pods, err := storage.List(ctx, labels.Everything(), labels.Everything())
	if err != nil {
		t.Fatalf("Expected no error, got %#v", err)
	}
	pl := pods.(*api.PodList)
	if len(pl.Items) != 1 {
		t.Fatalf("Unexpected 0-len pod list: %+v", pl)
	}
	if e, a := api.PodUnknown, pl.Items[0].Status.Phase; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
}

func TestListEmptyPodList(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(&api.PodList{ListMeta: api.ListMeta{ResourceVersion: "1"}})
	storage := REST{
		registry: podRegistry,
		podCache: &fakeCache{statusToReturn: &api.PodStatus{}},
	}
	ctx := api.NewContext()
	pods, err := storage.List(ctx, labels.Everything(), labels.Everything())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(pods.(*api.PodList).Items) != 0 {
		t.Errorf("Unexpected non-zero pod list: %#v", pods)
	}
	if pods.(*api.PodList).ResourceVersion != "1" {
		t.Errorf("Unexpected resource version: %#v", pods)
	}
}

func TestListPodList(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Pods = &api.PodList{
		Items: []api.Pod{
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
	}
	storage := REST{
		registry: podRegistry,
		podCache: &fakeCache{statusToReturn: &api.PodStatus{Phase: api.PodRunning}},
	}
	ctx := api.NewContext()
	podsObj, err := storage.List(ctx, labels.Everything(), labels.Everything())
	pods := podsObj.(*api.PodList)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(pods.Items) != 2 {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
	if pods.Items[0].Name != "foo" || pods.Items[0].Status.Phase != api.PodRunning {
		t.Errorf("Unexpected pod: %#v", pods.Items[0])
	}
	if pods.Items[1].Name != "bar" {
		t.Errorf("Unexpected pod: %#v", pods.Items[1])
	}
}

func TestListPodListSelection(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Pods = &api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
			}, {
				ObjectMeta: api.ObjectMeta{Name: "bar"},
				Status:     api.PodStatus{Host: "barhost"},
			}, {
				ObjectMeta: api.ObjectMeta{Name: "baz"},
				Status:     api.PodStatus{Phase: "bazstatus"},
			}, {
				ObjectMeta: api.ObjectMeta{
					Name:   "qux",
					Labels: map[string]string{"label": "qux"},
				},
			}, {
				ObjectMeta: api.ObjectMeta{Name: "zot"},
			},
		},
	}
	storage := REST{
		registry: podRegistry,
		podCache: &fakeCache{statusToReturn: &api.PodStatus{}},
	}
	ctx := api.NewContext()

	table := []struct {
		label, field string
		expectedIDs  util.StringSet
	}{
		{
			expectedIDs: util.NewStringSet("foo", "bar", "baz", "qux", "zot"),
		}, {
			field:       "name=zot",
			expectedIDs: util.NewStringSet("zot"),
		}, {
			label:       "label=qux",
			expectedIDs: util.NewStringSet("qux"),
		}, {
			field:       "Status.Phase=bazstatus",
			expectedIDs: util.NewStringSet("baz"),
		}, {
			field:       "Status.Host=barhost",
			expectedIDs: util.NewStringSet("bar"),
		}, {
			field:       "Status.Host=",
			expectedIDs: util.NewStringSet("foo", "baz", "qux", "zot"),
		}, {
			field:       "Status.Host!=",
			expectedIDs: util.NewStringSet("bar"),
		},
	}

	for index, item := range table {
		label, err := labels.ParseSelector(item.label)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		field, err := labels.ParseSelector(item.field)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		podsObj, err := storage.List(ctx, label, field)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		pods := podsObj.(*api.PodList)

		if e, a := len(item.expectedIDs), len(pods.Items); e != a {
			t.Errorf("%v: Expected %v, got %v", index, e, a)
		}
		for _, pod := range pods.Items {
			if !item.expectedIDs.Has(pod.Name) {
				t.Errorf("%v: Unexpected pod %v", index, pod.Name)
			}
			t.Logf("%v: Got pod Name: %v", index, pod.Name)
		}
	}
}

func TestPodDecode(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	storage := REST{
		registry: podRegistry,
		podCache: &fakeCache{statusToReturn: &api.PodStatus{}},
	}
	expected := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
	}
	body, err := latest.Codec.Encode(expected)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	actual := storage.New()
	if err := latest.Codec.DecodeInto(body, actual); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Expected %#v, Got %#v", expected, actual)
	}
}

func TestGetPod(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Pod = &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Status:     api.PodStatus{Host: "machine"},
	}
	storage := REST{
		registry: podRegistry,
		podCache: &fakeCache{statusToReturn: &api.PodStatus{Phase: api.PodRunning}},
	}
	ctx := api.NewContext()
	obj, err := storage.Get(ctx, "foo")
	pod := obj.(*api.Pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expect := *podRegistry.Pod
	expect.Status.Phase = api.PodRunning
	// TODO: when host is moved to spec, remove this line.
	expect.Status.Host = "machine"
	if e, a := &expect, pod; !reflect.DeepEqual(e, a) {
		t.Errorf("Unexpected pod. Expected %#v, Got %#v", e, a)
	}
}

func TestGetPodCacheError(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Pod = &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	storage := REST{
		registry: podRegistry,
		podCache: &fakeCache{errorToReturn: client.ErrPodInfoNotAvailable},
	}
	ctx := api.NewContext()
	obj, err := storage.Get(ctx, "foo")
	pod := obj.(*api.Pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expect := *podRegistry.Pod
	expect.Status.Phase = api.PodUnknown
	if e, a := &expect, pod; !reflect.DeepEqual(e, a) {
		t.Errorf("Unexpected pod. Expected %#v, Got %#v", e, a)
	}
}

// TODO: remove, this is covered by RESTTest.TestCreate
func TestPodStorageValidatesCreate(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := REST{
		registry: podRegistry,
		podCache: &fakeCache{statusToReturn: &api.PodStatus{}},
	}
	ctx := api.NewDefaultContext()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels: map[string]string{
				"invalid/label/to/cause/validation/failure": "bar",
			},
		},
	}
	c, err := storage.Create(ctx, pod)
	if c != nil {
		t.Errorf("Expected nil channel")
	}
	if !errors.IsInvalid(err) {
		t.Errorf("Expected to get an invalid resource error, got %v", err)
	}
}

// TODO: remove, this is covered by RESTTest.TestCreate
func TestCreatePod(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Pod = &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Status: api.PodStatus{
			Host: "machine",
		},
	}
	storage := REST{
		registry: podRegistry,
		podCache: &fakeCache{statusToReturn: &api.PodStatus{}},
	}
	pod := &api.Pod{}
	pod.Name = "foo"
	ctx := api.NewDefaultContext()
	channel, err := storage.Create(ctx, pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	select {
	case <-channel:
		// Do nothing, this is expected.
	case <-time.After(time.Millisecond * 100):
		t.Error("Unexpected timeout on async channel")
	}
	if !api.HasObjectMetaSystemFieldValues(&podRegistry.Pod.ObjectMeta) {
		t.Errorf("Expected ObjectMeta field values were populated")
	}
}

// TODO: remove, this is covered by RESTTest.TestCreate
func TestCreatePodWithConflictingNamespace(t *testing.T) {
	storage := REST{}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "not-default"},
	}

	ctx := api.NewDefaultContext()
	channel, err := storage.Create(ctx, pod)
	if channel != nil {
		t.Error("Expected a nil channel, but we got a value")
	}
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if strings.Contains(err.Error(), "Controller.Namespace does not match the provided context") {
		t.Errorf("Expected 'Pod.Namespace does not match the provided context' error, got '%v'", err.Error())
	}
}

func TestUpdatePodWithConflictingNamespace(t *testing.T) {
	storage := REST{}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "not-default"},
	}

	ctx := api.NewDefaultContext()
	channel, err := storage.Update(ctx, pod)
	if channel != nil {
		t.Error("Expected a nil channel, but we got a value")
	}
	if err == nil {
		t.Errorf("Expected an error, but we didn't get one")
	} else if strings.Index(err.Error(), "Pod.Namespace does not match the provided context") == -1 {
		t.Errorf("Expected 'Pod.Namespace does not match the provided context' error, got '%v'", err.Error())
	}
}

func TestResourceLocation(t *testing.T) {
	expectedIP := "1.2.3.4"
	testCases := []struct {
		pod      api.Pod
		query    string
		location string
	}{
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
			},
			query:    "foo",
			location: expectedIP,
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
			},
			query:    "foo:12345",
			location: expectedIP + ":12345",
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "ctr"},
					},
				},
			},
			query:    "foo",
			location: expectedIP,
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "ctr", Ports: []api.Port{{ContainerPort: 9376}}},
					},
				},
			},
			query:    "foo",
			location: expectedIP + ":9376",
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "ctr", Ports: []api.Port{{ContainerPort: 9376}}},
					},
				},
			},
			query:    "foo:12345",
			location: expectedIP + ":12345",
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "ctr1"},
						{Name: "ctr2", Ports: []api.Port{{ContainerPort: 9376}}},
					},
				},
			},
			query:    "foo",
			location: expectedIP + ":9376",
		},
		{
			pod: api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "ctr1", Ports: []api.Port{{ContainerPort: 9376}}},
						{Name: "ctr2", Ports: []api.Port{{ContainerPort: 1234}}},
					},
				},
			},
			query:    "foo",
			location: expectedIP + ":9376",
		},
	}

	for _, tc := range testCases {
		podRegistry := registrytest.NewPodRegistry(nil)
		podRegistry.Pod = &tc.pod
		storage := &REST{
			registry: podRegistry,
			podCache: &fakeCache{statusToReturn: &api.PodStatus{PodIP: expectedIP}},
		}

		redirector := apiserver.Redirector(storage)
		location, err := redirector.ResourceLocation(api.NewDefaultContext(), tc.query)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		if location != tc.location {
			t.Errorf("Expected %v, but got %v", tc.location, location)
		}
	}
}

func TestDeletePod(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Pod = &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Status:     api.PodStatus{Host: "machine"},
	}
	fakeCache := &fakeCache{}
	storage := REST{
		registry: podRegistry,
		podCache: fakeCache,
	}
	ctx := api.NewDefaultContext()
	channel, err := storage.Delete(ctx, "foo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var result apiserver.RESTResult
	select {
	case result = <-channel:
		// Do nothing, this is expected.
	case <-time.After(time.Millisecond * 100):
		t.Error("Unexpected timeout on async channel")
	}
	if fakeCache.clearedNamespace != "default" || fakeCache.clearedName != "foo" {
		t.Errorf("Unexpeceted cache delete: %s %s %#v", fakeCache.clearedName, fakeCache.clearedNamespace, result.Object)
	}
}

func TestCreate(t *testing.T) {
	registry := registrytest.NewPodRegistry(nil)
	test := resttest.New(t, &REST{
		registry: registry,
		podCache: &fakeCache{statusToReturn: &api.PodStatus{}},
	}, registry.SetError)
	test.TestCreate(
		// valid
		&api.Pod{
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "test1",
						Image: "foo",
					},
				},
			},
		},
		// invalid
		&api.Pod{
			Spec: api.PodSpec{
				Containers: []api.Container{},
			},
		},
	)
}
