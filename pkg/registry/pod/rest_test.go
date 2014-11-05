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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/fake"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

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
	}
	desiredState := api.PodState{
		Manifest: api.ContainerManifest{
			Version: "v1beta1",
		},
	}
	pod := &api.Pod{DesiredState: desiredState}
	ctx := api.NewDefaultContext()
	ch, err := storage.Create(ctx, pod)
	if err != nil {
		t.Errorf("Expected %#v, Got %#v", nil, err)
	}
	expectApiStatusError(t, ch, podRegistry.Err.Error())
}

func TestCreatePodSetsIds(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := REST{
		registry: podRegistry,
	}
	desiredState := api.PodState{
		Manifest: api.ContainerManifest{
			Version: "v1beta1",
		},
	}
	pod := &api.Pod{DesiredState: desiredState}
	ctx := api.NewDefaultContext()
	ch, err := storage.Create(ctx, pod)
	if err != nil {
		t.Errorf("Expected %#v, Got %#v", nil, err)
	}
	expectApiStatusError(t, ch, podRegistry.Err.Error())

	if len(podRegistry.Pod.Name) == 0 {
		t.Errorf("Expected pod ID to be set, Got %#v", pod)
	}
	if podRegistry.Pod.DesiredState.Manifest.ID != podRegistry.Pod.Name {
		t.Errorf("Expected manifest ID to be equal to pod ID, Got %#v", pod)
	}
}

func TestCreatePodSetsUUIDs(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := REST{
		registry: podRegistry,
	}
	desiredState := api.PodState{
		Manifest: api.ContainerManifest{
			Version: "v1beta1",
		},
	}
	pod := &api.Pod{DesiredState: desiredState}
	ctx := api.NewDefaultContext()
	ch, err := storage.Create(ctx, pod)
	if err != nil {
		t.Errorf("Expected %#v, Got %#v", nil, err)
	}
	expectApiStatusError(t, ch, podRegistry.Err.Error())

	if len(podRegistry.Pod.DesiredState.Manifest.UUID) == 0 {
		t.Errorf("Expected pod UUID to be set, Got %#v", pod)
	}
}

func TestListPodsError(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := REST{
		registry: podRegistry,
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

func TestListEmptyPodList(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(&api.PodList{ListMeta: api.ListMeta{ResourceVersion: "1"}})
	storage := REST{
		registry: podRegistry,
	}
	ctx := api.NewContext()
	pods, err := storage.List(ctx, labels.Everything(), labels.Everything())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(pods.(*api.PodList).Items) != 0 {
		t.Errorf("Unexpected non-zero pod list: %#v", pods)
	}
	if pods.(*api.PodList).ResourceVersion != "1" {
		t.Errorf("Unexpected resource version: %#v", pods)
	}
}

type fakeClock struct {
	t time.Time
}

func (f *fakeClock) Now() time.Time {
	return f.t
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
		ipCache:  ipCache{},
		clock:    &fakeClock{},
	}
	ctx := api.NewContext()
	podsObj, err := storage.List(ctx, labels.Everything(), labels.Everything())
	pods := podsObj.(*api.PodList)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(pods.Items) != 2 {
		t.Errorf("Unexpected pod list: %#v", pods)
	}
	if pods.Items[0].Name != "foo" {
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
				ObjectMeta:   api.ObjectMeta{Name: "bar"},
				DesiredState: api.PodState{Host: "barhost"},
			}, {
				ObjectMeta:   api.ObjectMeta{Name: "baz"},
				DesiredState: api.PodState{Status: "bazstatus"},
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
		ipCache:  ipCache{},
		clock:    &fakeClock{},
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
			field:       "DesiredState.Status=bazstatus",
			expectedIDs: util.NewStringSet("baz"),
		}, {
			field:       "DesiredState.Host=barhost",
			expectedIDs: util.NewStringSet("bar"),
		}, {
			field:       "DesiredState.Host=",
			expectedIDs: util.NewStringSet("foo", "baz", "qux", "zot"),
		}, {
			field:       "DesiredState.Host!=",
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
	}
	expected := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "foo",
		},
	}
	body, err := latest.Codec.Encode(expected)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	actual := storage.New()
	if err := latest.Codec.DecodeInto(body, actual); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Expected %#v, Got %#v", expected, actual)
	}
}

func TestGetPod(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Pod = &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	storage := REST{
		registry: podRegistry,
		ipCache:  ipCache{},
		clock:    &fakeClock{},
	}
	ctx := api.NewContext()
	obj, err := storage.Get(ctx, "foo")
	pod := obj.(*api.Pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if e, a := podRegistry.Pod, pod; !reflect.DeepEqual(e, a) {
		t.Errorf("Unexpected pod. Expected %#v, Got %#v", e, a)
	}
}

func TestGetPodCloud(t *testing.T) {
	fakeCloud := &fake_cloud.FakeCloud{}
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Pod = &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}, CurrentState: api.PodState{Host: "machine"}}

	clock := &fakeClock{t: time.Now()}

	storage := REST{
		registry:      podRegistry,
		cloudProvider: fakeCloud,
		ipCache:       ipCache{},
		clock:         clock,
	}
	ctx := api.NewContext()
	obj, err := storage.Get(ctx, "foo")
	pod := obj.(*api.Pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if e, a := podRegistry.Pod, pod; !reflect.DeepEqual(e, a) {
		t.Errorf("Unexpected pod. Expected %#v, Got %#v", e, a)
	}

	// This call should hit the cache, so we expect no additional calls to the cloud
	obj, err = storage.Get(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(fakeCloud.Calls) != 1 || fakeCloud.Calls[0] != "ip-address" {
		t.Errorf("Unexpected calls: %#v", fakeCloud.Calls)
	}

	// Advance the clock, this call should miss the cache, so expect one more call.
	clock.t = clock.t.Add(60 * time.Second)
	obj, err = storage.Get(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(fakeCloud.Calls) != 2 || fakeCloud.Calls[1] != "ip-address" {
		t.Errorf("Unexpected calls: %#v", fakeCloud.Calls)
	}
}

func TestMakePodStatus(t *testing.T) {
	fakeClient := client.Fake{
		MinionsList: api.MinionList{
			Items: []api.Minion{
				{
					ObjectMeta: api.ObjectMeta{Name: "machine"},
				},
			},
		},
	}
	desiredState := api.PodState{
		Manifest: api.ContainerManifest{
			Version: "v1beta1",
			Containers: []api.Container{
				{Name: "containerA"},
				{Name: "containerB"},
			},
		},
	}
	currentState := api.PodState{
		Host: "machine",
	}
	runningState := api.ContainerStatus{
		State: api.ContainerState{
			Running: &api.ContainerStateRunning{},
		},
	}
	stoppedState := api.ContainerStatus{
		State: api.ContainerState{
			Termination: &api.ContainerStateTerminated{},
		},
	}

	tests := []struct {
		pod    *api.Pod
		status api.PodCondition
		test   string
	}{
		{&api.Pod{DesiredState: desiredState, CurrentState: currentState}, api.PodPending, "waiting"},
		{
			&api.Pod{
				DesiredState: desiredState,
				CurrentState: api.PodState{
					Host: "machine-2",
				},
			},
			api.PodFailed,
			"no info, but bad machine",
		},
		{
			&api.Pod{
				DesiredState: desiredState,
				CurrentState: api.PodState{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
						"containerB": runningState,
					},
					Host: "machine",
				},
			},
			api.PodRunning,
			"all running",
		},
		{
			&api.Pod{
				DesiredState: desiredState,
				CurrentState: api.PodState{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
						"containerB": runningState,
					},
					Host: "machine-two",
				},
			},
			api.PodFailed,
			"all running but minion is missing",
		},
		{
			&api.Pod{
				DesiredState: desiredState,
				CurrentState: api.PodState{
					Info: map[string]api.ContainerStatus{
						"containerA": stoppedState,
						"containerB": stoppedState,
					},
					Host: "machine",
				},
			},
			api.PodFailed,
			"all stopped",
		},
		{
			&api.Pod{
				DesiredState: desiredState,
				CurrentState: api.PodState{
					Info: map[string]api.ContainerStatus{
						"containerA": stoppedState,
						"containerB": stoppedState,
					},
					Host: "machine-two",
				},
			},
			api.PodFailed,
			"all stopped but minion missing",
		},
		{
			&api.Pod{
				DesiredState: desiredState,
				CurrentState: api.PodState{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
						"containerB": stoppedState,
					},
					Host: "machine",
				},
			},
			api.PodRunning,
			"mixed state #1",
		},
		{
			&api.Pod{
				DesiredState: desiredState,
				CurrentState: api.PodState{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
					},
					Host: "machine",
				},
			},
			api.PodPending,
			"mixed state #2",
		},
	}
	for _, test := range tests {
		if status, err := getPodStatus(test.pod, fakeClient.Minions()); status != test.status {
			t.Errorf("In test %s, expected %v, got %v", test.test, test.status, status)
			if err != nil {
				t.Errorf("In test %s, unexpected error: %v", test.test, err)
			}
		}
	}
}

func TestPodStorageValidatesCreate(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := REST{
		registry: podRegistry,
	}
	ctx := api.NewDefaultContext()
	pod := &api.Pod{}
	c, err := storage.Create(ctx, pod)
	if c != nil {
		t.Errorf("Expected nil channel")
	}
	if !errors.IsInvalid(err) {
		t.Errorf("Expected to get an invalid resource error, got %v", err)
	}
}

func TestPodStorageValidatesUpdate(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Err = fmt.Errorf("test error")
	storage := REST{
		registry: podRegistry,
	}
	ctx := api.NewDefaultContext()
	pod := &api.Pod{}
	c, err := storage.Update(ctx, pod)
	if c != nil {
		t.Errorf("Expected nil channel")
	}
	if !errors.IsInvalid(err) {
		t.Errorf("Expected to get an invalid resource error, got %v", err)
	}
}

func TestCreatePod(t *testing.T) {
	podRegistry := registrytest.NewPodRegistry(nil)
	podRegistry.Pod = &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		CurrentState: api.PodState{
			Host: "machine",
		},
	}
	storage := REST{
		registry:      podRegistry,
		podPollPeriod: time.Millisecond * 100,
	}
	desiredState := api.PodState{
		Manifest: api.ContainerManifest{
			Version: "v1beta1",
		},
	}
	pod := &api.Pod{
		ObjectMeta:   api.ObjectMeta{Name: "foo"},
		DesiredState: desiredState,
	}
	ctx := api.NewDefaultContext()
	channel, err := storage.Create(ctx, pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	select {
	case <-channel:
		// Do nothing, this is expected.
	case <-time.After(time.Millisecond * 100):
		t.Error("Unexpected timeout on async channel")
	}
}

type FakePodInfoGetter struct {
	info api.PodInfo
	err  error
}

func (f *FakePodInfoGetter) GetPodInfo(host, podNamespace string, podID string) (api.PodInfo, error) {
	return f.info, f.err
}

func TestFillPodInfo(t *testing.T) {
	expectedIP := "1.2.3.4"
	expectedTime, _ := time.Parse("2013-Feb-03", "2013-Feb-03")
	fakeGetter := FakePodInfoGetter{
		info: map[string]api.ContainerStatus{
			"net": {
				State: api.ContainerState{
					Running: &api.ContainerStateRunning{
						StartedAt: expectedTime,
					},
				},
				RestartCount: 1,
				PodIP:        expectedIP,
			},
		},
	}
	storage := REST{
		podCache: &fakeGetter,
	}
	pod := api.Pod{DesiredState: api.PodState{Host: "foo"}}
	storage.fillPodInfo(&pod)
	if !reflect.DeepEqual(fakeGetter.info, pod.CurrentState.Info) {
		t.Errorf("Expected: %#v, Got %#v", fakeGetter.info, pod.CurrentState.Info)
	}
	if pod.CurrentState.PodIP != expectedIP {
		t.Errorf("Expected %s, Got %s", expectedIP, pod.CurrentState.PodIP)
	}
}

func TestFillPodInfoNoData(t *testing.T) {
	expectedIP := ""
	fakeGetter := FakePodInfoGetter{
		info: map[string]api.ContainerStatus{
			"net": {
				State: api.ContainerState{},
			},
		},
	}
	storage := REST{
		podCache: &fakeGetter,
	}
	pod := api.Pod{DesiredState: api.PodState{Host: "foo"}}
	storage.fillPodInfo(&pod)
	if !reflect.DeepEqual(fakeGetter.info, pod.CurrentState.Info) {
		t.Errorf("Expected %#v, Got %#v", fakeGetter.info, pod.CurrentState.Info)
	}
	if pod.CurrentState.PodIP != expectedIP {
		t.Errorf("Expected %s, Got %s", expectedIP, pod.CurrentState.PodIP)
	}
}

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
	} else if strings.Index(err.Error(), "Pod.Namespace does not match the provided context") == -1 {
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
