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
package kubelet

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/fsouza/go-dockerclient"
)

// TODO: This doesn't reduce typing enough to make it worth the less readable errors. Remove.
func expectNoError(t *testing.T, err error) {
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
}

// These are used for testing extract json (below)
type TestData struct {
	Value  string
	Number int
}

type TestObject struct {
	Name string
	Data TestData
}

func verifyStringEquals(t *testing.T, actual, expected string) {
	if actual != expected {
		t.Errorf("Verification failed.  Expected: %s, Found %s", expected, actual)
	}
}

func verifyIntEquals(t *testing.T, actual, expected int) {
	if actual != expected {
		t.Errorf("Verification failed.  Expected: %d, Found %d", expected, actual)
	}
}

func verifyNoError(t *testing.T, e error) {
	if e != nil {
		t.Errorf("Expected no error, found %#v", e)
	}
}

func verifyError(t *testing.T, e error) {
	if e == nil {
		t.Errorf("Expected error, found nil")
	}
}

func TestExtractJSON(t *testing.T) {
	obj := TestObject{}
	kubelet := Kubelet{}
	data := `{ "name": "foo", "data": { "value": "bar", "number": 10 } }`
	kubelet.ExtractYAMLData([]byte(data), &obj)

	verifyStringEquals(t, obj.Name, "foo")
	verifyStringEquals(t, obj.Data.Value, "bar")
	verifyIntEquals(t, obj.Data.Number, 10)
}

type FakeDockerClient struct {
	containerList []docker.APIContainers
	container     *docker.Container
	err           error
	called        []string
}

func (f *FakeDockerClient) clearCalls() {
	f.called = []string{}
}

func (f *FakeDockerClient) appendCall(call string) {
	f.called = append(f.called, call)
}

func (f *FakeDockerClient) ListContainers(options docker.ListContainersOptions) ([]docker.APIContainers, error) {
	f.appendCall("list")
	return f.containerList, f.err
}

func (f *FakeDockerClient) InspectContainer(id string) (*docker.Container, error) {
	f.appendCall("inspect")
	return f.container, f.err
}

func (f *FakeDockerClient) CreateContainer(docker.CreateContainerOptions) (*docker.Container, error) {
	f.appendCall("create")
	return nil, nil
}

func (f *FakeDockerClient) StartContainer(id string, hostConfig *docker.HostConfig) error {
	f.appendCall("start")
	return nil
}

func (f *FakeDockerClient) StopContainer(id string, timeout uint) error {
	f.appendCall("stop")
	return nil
}

func verifyCalls(t *testing.T, fakeDocker FakeDockerClient, calls []string) {
	verifyStringArrayEquals(t, fakeDocker.called, calls)
}

func verifyStringArrayEquals(t *testing.T, actual, expected []string) {
	invalid := len(actual) != len(expected)
	for ix, value := range actual {
		if expected[ix] != value {
			invalid = true
		}
	}
	if invalid {
		t.Errorf("Expected: %#v, Actual: %#v", expected, actual)
	}
}

func verifyPackUnpack(t *testing.T, manifestId, containerName string) {
	name := manifestAndContainerToDockerName(
		&api.ContainerManifest{Id: manifestId},
		&api.Container{Name: containerName},
	)
	returnedManifestId, returnedContainerName := dockerNameToManifestAndContainer(name)
	if manifestId != returnedManifestId || containerName != returnedContainerName {
		t.Errorf("For (%s, %s), unpacked (%s, %s)", manifestId, containerName, returnedManifestId, returnedContainerName)
	}
}

func TestContainerManifestNaming(t *testing.T) {
	verifyPackUnpack(t, "manifest1234", "container5678")
	verifyPackUnpack(t, "manifest--", "container__")
	verifyPackUnpack(t, "--manifest", "__container")
	verifyPackUnpack(t, "m___anifest_", "container-_-")
	verifyPackUnpack(t, "_m___anifest", "-_-container")
}

func TestContainerExists(t *testing.T) {
	fakeDocker := FakeDockerClient{
		err: nil,
	}
	kubelet := Kubelet{
		DockerClient: &fakeDocker,
	}
	manifest := api.ContainerManifest{
		Id: "qux",
	}
	container := api.Container{
		Name: "foo",
	}
	fakeDocker.containerList = []docker.APIContainers{
		{
			Names: []string{"foo--qux--1234"},
		},
		{
			Names: []string{"bar--qux--1234"},
		},
	}
	fakeDocker.container = &docker.Container{
		ID: "foobar",
	}

	exists, _, err := kubelet.ContainerExists(&manifest, &container)
	verifyCalls(t, fakeDocker, []string{"list", "list", "inspect"})
	if !exists {
		t.Errorf("Failed to find container %#v", container)
	}
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
}

func TestGetContainerID(t *testing.T) {
	fakeDocker := FakeDockerClient{
		err: nil,
	}
	kubelet := Kubelet{
		DockerClient: &fakeDocker,
	}
	fakeDocker.containerList = []docker.APIContainers{
		{
			Names: []string{"foo"},
			ID:    "1234",
		},
		{
			Names: []string{"bar"},
			ID:    "4567",
		},
	}

	id, err := kubelet.GetContainerID("foo")
	verifyStringEquals(t, id, "1234")
	verifyNoError(t, err)
	verifyCalls(t, fakeDocker, []string{"list"})
	fakeDocker.clearCalls()

	id, err = kubelet.GetContainerID("bar")
	verifyStringEquals(t, id, "4567")
	verifyNoError(t, err)
	verifyCalls(t, fakeDocker, []string{"list"})
	fakeDocker.clearCalls()

	id, err = kubelet.GetContainerID("NotFound")
	verifyError(t, err)
	verifyCalls(t, fakeDocker, []string{"list"})
}

func TestGetContainerByName(t *testing.T) {
	fakeDocker := FakeDockerClient{
		err: nil,
	}
	kubelet := Kubelet{
		DockerClient: &fakeDocker,
	}
	fakeDocker.containerList = []docker.APIContainers{
		{
			Names: []string{"foo"},
		},
		{
			Names: []string{"bar"},
		},
	}
	fakeDocker.container = &docker.Container{
		ID: "foobar",
	}

	container, err := kubelet.GetContainerByName("foo")
	verifyCalls(t, fakeDocker, []string{"list", "inspect"})
	if container == nil {
		t.Errorf("Unexpected nil container")
	}
	verifyStringEquals(t, container.ID, "foobar")
	verifyNoError(t, err)
}

func TestListContainers(t *testing.T) {
	fakeDocker := FakeDockerClient{
		err: nil,
	}
	kubelet := Kubelet{
		DockerClient: &fakeDocker,
	}
	fakeDocker.containerList = []docker.APIContainers{
		{
			Names: []string{"foo"},
		},
		{
			Names: []string{"bar"},
		},
	}

	containers, err := kubelet.ListContainers()
	verifyStringArrayEquals(t, containers, []string{"foo", "bar"})
	verifyNoError(t, err)
	verifyCalls(t, fakeDocker, []string{"list"})
}

func TestKillContainerWithError(t *testing.T) {
	fakeDocker := FakeDockerClient{
		err: fmt.Errorf("Sample Error"),
		containerList: []docker.APIContainers{
			{
				Names: []string{"foo"},
			},
			{
				Names: []string{"bar"},
			},
		},
	}
	kubelet := Kubelet{
		DockerClient: &fakeDocker,
	}
	err := kubelet.KillContainer("foo")
	verifyError(t, err)
	verifyCalls(t, fakeDocker, []string{"list"})
}

func TestKillContainer(t *testing.T) {
	fakeDocker := FakeDockerClient{
		err: nil,
	}
	kubelet := Kubelet{
		DockerClient: &fakeDocker,
	}
	fakeDocker.containerList = []docker.APIContainers{
		{
			Names: []string{"foo"},
		},
		{
			Names: []string{"bar"},
		},
	}
	fakeDocker.container = &docker.Container{
		ID: "foobar",
	}

	err := kubelet.KillContainer("foo")
	verifyNoError(t, err)
	verifyCalls(t, fakeDocker, []string{"list", "stop"})
}

func TestSyncHTTP(t *testing.T) {
	containers := api.ContainerManifest{
		Containers: []api.Container{
			{
				Name:  "foo",
				Image: "dockerfile/foo",
			},
			{
				Name:  "bar",
				Image: "dockerfile/bar",
			},
		},
	}
	data, _ := json.Marshal(containers)
	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(data),
	}
	testServer := httptest.NewServer(&fakeHandler)
	kubelet := Kubelet{}

	var containersOut api.ContainerManifest
	data, err := kubelet.SyncHTTP(&http.Client{}, testServer.URL, &containersOut)
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if len(containers.Containers) != len(containersOut.Containers) {
		t.Errorf("Container sizes don't match.  Expected: %d Received %d, %#v", len(containers.Containers), len(containersOut.Containers), containersOut)
	}
	expectedData, _ := json.Marshal(containers)
	actualData, _ := json.Marshal(containersOut)
	if string(expectedData) != string(actualData) {
		t.Errorf("Container data doesn't match.  Expected: %s Received %s", string(expectedData), string(actualData))
	}
}

func TestResponseToContainersNil(t *testing.T) {
	kubelet := Kubelet{}
	list, err := kubelet.ResponseToManifests(&etcd.Response{Node: nil})
	if len(list) != 0 {
		t.Errorf("Unexpected non-zero list: %#v", list)
	}
	if err == nil {
		t.Error("Unexpected non-error")
	}
}

func TestResponseToManifests(t *testing.T) {
	kubelet := Kubelet{}
	list, err := kubelet.ResponseToManifests(&etcd.Response{
		Node: &etcd.Node{
			Value: util.MakeJSONString([]api.ContainerManifest{
				{Id: "foo"},
				{Id: "bar"},
			}),
		},
	})
	if len(list) != 2 || list[0].Id != "foo" || list[1].Id != "bar" {
		t.Errorf("Unexpected list: %#v", list)
	}
	expectNoError(t, err)
}

type channelReader struct {
	list [][]api.ContainerManifest
	wg   sync.WaitGroup
}

func startReading(channel <-chan []api.ContainerManifest) *channelReader {
	cr := &channelReader{}
	cr.wg.Add(1)
	go func() {
		for {
			containers, ok := <-channel
			if !ok {
				break
			}
			cr.list = append(cr.list, containers)
		}
		cr.wg.Done()
	}()
	return cr
}

func (cr *channelReader) GetList() [][]api.ContainerManifest {
	cr.wg.Wait()
	return cr.list
}

func TestGetKubeletStateFromEtcdNoData(t *testing.T) {
	fakeClient := registry.MakeFakeEtcdClient(t)
	kubelet := Kubelet{
		Client: fakeClient,
	}
	channel := make(chan []api.ContainerManifest)
	reader := startReading(channel)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = registry.EtcdResponseWithError{
		R: &etcd.Response{},
		E: nil,
	}
	err := kubelet.getKubeletStateFromEtcd("/registry/hosts/machine", channel)
	if err == nil {
		t.Error("Unexpected no err.")
	}
	close(channel)
	list := reader.GetList()
	if len(list) != 0 {
		t.Errorf("Unexpected list: %#v", list)
	}
}

func TestGetKubeletStateFromEtcd(t *testing.T) {
	fakeClient := registry.MakeFakeEtcdClient(t)
	kubelet := Kubelet{
		Client: fakeClient,
	}
	channel := make(chan []api.ContainerManifest)
	reader := startReading(channel)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = registry.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: util.MakeJSONString([]api.Container{}),
			},
		},
		E: nil,
	}
	err := kubelet.getKubeletStateFromEtcd("/registry/hosts/machine", channel)
	expectNoError(t, err)
	close(channel)
	list := reader.GetList()
	if len(list) != 1 {
		t.Errorf("Unexpected list: %#v", list)
	}
}

func TestGetKubeletStateFromEtcdNotFound(t *testing.T) {
	fakeClient := registry.MakeFakeEtcdClient(t)
	kubelet := Kubelet{
		Client: fakeClient,
	}
	channel := make(chan []api.ContainerManifest)
	reader := startReading(channel)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = registry.EtcdResponseWithError{
		R: &etcd.Response{},
		E: &etcd.EtcdError{
			ErrorCode: 100,
		},
	}
	err := kubelet.getKubeletStateFromEtcd("/registry/hosts/machine", channel)
	expectNoError(t, err)
	close(channel)
	list := reader.GetList()
	if len(list) != 0 {
		t.Errorf("Unexpected list: %#v", list)
	}
}

func TestGetKubeletStateFromEtcdError(t *testing.T) {
	fakeClient := registry.MakeFakeEtcdClient(t)
	kubelet := Kubelet{
		Client: fakeClient,
	}
	channel := make(chan []api.ContainerManifest)
	reader := startReading(channel)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = registry.EtcdResponseWithError{
		R: &etcd.Response{},
		E: &etcd.EtcdError{
			ErrorCode: 200, // non not found error
		},
	}
	err := kubelet.getKubeletStateFromEtcd("/registry/hosts/machine", channel)
	if err == nil {
		t.Error("Unexpected non-error")
	}
	close(channel)
	list := reader.GetList()
	if len(list) != 0 {
		t.Errorf("Unexpected list: %#v", list)
	}
}

func TestSyncManifestsDoesNothing(t *testing.T) {
	fakeDocker := FakeDockerClient{
		err: nil,
	}
	fakeDocker.containerList = []docker.APIContainers{
		{
			// format is <container-id>--<manifest-id>
			Names: []string{"bar--foo"},
			ID:    "1234",
		},
	}
	fakeDocker.container = &docker.Container{
		ID: "1234",
	}
	kubelet := Kubelet{
		DockerClient: &fakeDocker,
	}
	err := kubelet.SyncManifests([]api.ContainerManifest{
		{
			Id: "foo",
			Containers: []api.Container{
				{Name: "bar"},
			},
		},
	})
	expectNoError(t, err)
	if len(fakeDocker.called) != 4 ||
		fakeDocker.called[0] != "list" ||
		fakeDocker.called[1] != "list" ||
		fakeDocker.called[2] != "inspect" ||
		fakeDocker.called[3] != "list" {
		t.Errorf("Unexpected call sequence: %#v", fakeDocker.called)
	}
}

func TestSyncManifestsDeletes(t *testing.T) {
	fakeDocker := FakeDockerClient{
		err: nil,
	}
	fakeDocker.containerList = []docker.APIContainers{
		{
			Names: []string{"foo"},
			ID:    "1234",
		},
	}
	kubelet := Kubelet{
		DockerClient: &fakeDocker,
	}
	err := kubelet.SyncManifests([]api.ContainerManifest{})
	expectNoError(t, err)
	if len(fakeDocker.called) != 3 ||
		fakeDocker.called[0] != "list" ||
		fakeDocker.called[1] != "list" ||
		fakeDocker.called[2] != "stop" {
		t.Errorf("Unexpected call sequence: %#v", fakeDocker.called)
	}
}
