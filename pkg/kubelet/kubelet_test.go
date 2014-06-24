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
	"io/ioutil"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/fsouza/go-dockerclient"
	"github.com/google/cadvisor/info"
	"github.com/stretchr/testify/mock"
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
	stopped       []string
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
	f.stopped = append(f.stopped, id)
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

func verifyBoolean(t *testing.T, expected, value bool) {
	if expected != value {
		t.Errorf("Unexpected boolean.  Expected %s.  Found %s", expected, value)
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
			Names: []string{"/k8s--foo--qux--1234"},
		},
		{
			Names: []string{"/k8s--bar--qux--1234"},
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

	fakeDocker.clearCalls()
	missingManifest := api.ContainerManifest{Id: "foobar"}
	exists, _, err = kubelet.ContainerExists(&missingManifest, &container)
	verifyCalls(t, fakeDocker, []string{"list"})
	if exists {
		t.Errorf("Failed to not find container %#v, missingManifest")
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

	id, found, err := kubelet.GetContainerID("foo")
	verifyBoolean(t, true, found)
	verifyStringEquals(t, id, "1234")
	verifyNoError(t, err)
	verifyCalls(t, fakeDocker, []string{"list"})
	fakeDocker.clearCalls()

	id, found, err = kubelet.GetContainerID("bar")
	verifyBoolean(t, true, found)
	verifyStringEquals(t, id, "4567")
	verifyNoError(t, err)
	verifyCalls(t, fakeDocker, []string{"list"})
	fakeDocker.clearCalls()

	id, found, err = kubelet.GetContainerID("NotFound")
	verifyBoolean(t, false, found)
	verifyNoError(t, err)
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
		err: fmt.Errorf("sample error"),
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

func startReading(channel <-chan manifestUpdate) *channelReader {
	cr := &channelReader{}
	cr.wg.Add(1)
	go func() {
		for {
			update, ok := <-channel
			if !ok {
				break
			}
			cr.list = append(cr.list, update.manifests)
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
	fakeClient := util.MakeFakeEtcdClient(t)
	kubelet := Kubelet{
		EtcdClient: fakeClient,
	}
	channel := make(chan manifestUpdate)
	reader := startReading(channel)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = util.EtcdResponseWithError{
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
	fakeClient := util.MakeFakeEtcdClient(t)
	kubelet := Kubelet{
		EtcdClient: fakeClient,
	}
	channel := make(chan manifestUpdate)
	reader := startReading(channel)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = util.EtcdResponseWithError{
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
	fakeClient := util.MakeFakeEtcdClient(t)
	kubelet := Kubelet{
		EtcdClient: fakeClient,
	}
	channel := make(chan manifestUpdate)
	reader := startReading(channel)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = util.EtcdResponseWithError{
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
	fakeClient := util.MakeFakeEtcdClient(t)
	kubelet := Kubelet{
		EtcdClient: fakeClient,
	}
	channel := make(chan manifestUpdate)
	reader := startReading(channel)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = util.EtcdResponseWithError{
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
			// format is k8s--<container-id>--<manifest-id>
			Names: []string{"/k8s--bar--foo"},
			ID:    "1234",
		},
		{
			// network container
			Names: []string{"/k8s--net--foo--"},
			ID:    "9876",
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
	if len(fakeDocker.called) != 5 ||
		fakeDocker.called[0] != "list" ||
		fakeDocker.called[1] != "list" ||
		fakeDocker.called[2] != "list" ||
		fakeDocker.called[3] != "inspect" ||
		fakeDocker.called[4] != "list" {
		t.Errorf("Unexpected call sequence: %#v", fakeDocker.called)
	}
}

func TestSyncManifestsDeletes(t *testing.T) {
	fakeDocker := FakeDockerClient{
		err: nil,
	}
	fakeDocker.containerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s--foo--bar"},
			ID:    "1234",
		},
		{
			// network container
			Names: []string{"/k8s--net--foo--"},
			ID:    "9876",
		},
		{
			Names: []string{"foo"},
			ID:    "4567",
		},
	}
	kubelet := Kubelet{
		DockerClient: &fakeDocker,
	}
	err := kubelet.SyncManifests([]api.ContainerManifest{})
	expectNoError(t, err)
	if len(fakeDocker.called) != 5 ||
		fakeDocker.called[0] != "list" ||
		fakeDocker.called[1] != "list" ||
		fakeDocker.called[2] != "stop" ||
		fakeDocker.called[3] != "list" ||
		fakeDocker.called[4] != "stop" ||
		fakeDocker.stopped[0] != "1234" ||
		fakeDocker.stopped[1] != "9876" {
		t.Errorf("Unexpected call sequence: %#v %s", fakeDocker.called, fakeDocker.stopped)
	}
}

func TestEventWriting(t *testing.T) {
	fakeEtcd := util.MakeFakeEtcdClient(t)
	kubelet := &Kubelet{
		EtcdClient: fakeEtcd,
	}
	expectedEvent := api.Event{
		Event: "test",
		Container: &api.Container{
			Name: "foo",
		},
	}
	err := kubelet.LogEvent(&expectedEvent)
	expectNoError(t, err)
	if fakeEtcd.Ix != 1 {
		t.Errorf("Unexpected number of children added: %d, expected 1", fakeEtcd.Ix)
	}
	response, err := fakeEtcd.Get("/events/foo/1", false, false)
	expectNoError(t, err)
	var event api.Event
	err = json.Unmarshal([]byte(response.Node.Value), &event)
	expectNoError(t, err)
	if event.Event != expectedEvent.Event ||
		event.Container.Name != expectedEvent.Container.Name {
		t.Errorf("Event's don't match.  Expected: %#v Saw: %#v", expectedEvent, event)
	}
}

func TestEventWritingError(t *testing.T) {
	fakeEtcd := util.MakeFakeEtcdClient(t)
	kubelet := &Kubelet{
		EtcdClient: fakeEtcd,
	}
	fakeEtcd.Err = fmt.Errorf("test error")
	err := kubelet.LogEvent(&api.Event{
		Event: "test",
		Container: &api.Container{
			Name: "foo",
		},
	})
	if err == nil {
		t.Errorf("Unexpected non-error")
	}
}

func TestMakeEnvVariables(t *testing.T) {
	container := api.Container{
		Env: []api.EnvVar{
			{
				Name:  "foo",
				Value: "bar",
			},
			{
				Name:  "baz",
				Value: "blah",
			},
		},
	}
	vars := makeEnvironmentVariables(&container)
	if len(vars) != len(container.Env) {
		t.Errorf("Vars don't match.  Expected: %#v Found: %#v", container.Env, vars)
	}
	for ix, env := range container.Env {
		value := fmt.Sprintf("%s=%s", env.Name, env.Value)
		if value != vars[ix] {
			t.Errorf("Unexpected value: %s.  Expected: %s", vars[ix], value)
		}
	}
}

func TestMakeVolumesAndBinds(t *testing.T) {
	container := api.Container{
		VolumeMounts: []api.VolumeMount{
			{
				MountPath: "/mnt/path",
				Name:      "disk",
				ReadOnly:  false,
			},
			{
				MountPath: "/mnt/path2",
				Name:      "disk2",
				ReadOnly:  true,
				MountType: "LOCAL",
			},
			{
				MountPath: "/mnt/path3",
				Name:      "disk3",
				ReadOnly:  false,
				MountType: "HOST",
			},
		},
	}
	volumes, binds := makeVolumesAndBinds(&container)

	expectedVolumes := []string{"/mnt/path", "/mnt/path2"}
	expectedBinds := []string{"/exports/disk:/mnt/path", "/exports/disk2:/mnt/path2:ro", "/mnt/path3:/mnt/path3"}
	if len(volumes) != len(expectedVolumes) {
		t.Errorf("Unexpected volumes. Expected %#v got %#v.  Container was: %#v", expectedVolumes, volumes, container)
	}
	for _, expectedVolume := range expectedVolumes {
		if _, ok := volumes[expectedVolume]; !ok {
			t.Errorf("Volumes map is missing key: %s. %#v", expectedVolume, volumes)
		}
	}
	if len(binds) != len(expectedBinds) {
		t.Errorf("Unexpected binds: Expected %# got %#v.  Container was: %#v", expectedBinds, binds, container)
	}
	verifyStringArrayEquals(t, binds, expectedBinds)
}

func TestMakePortsAndBindings(t *testing.T) {
	container := api.Container{
		Ports: []api.Port{
			{
				ContainerPort: 80,
				HostPort:      8080,
			},
			{
				ContainerPort: 443,
				HostPort:      443,
				Protocol:      "tcp",
			},
			{
				ContainerPort: 444,
				HostPort:      444,
				Protocol:      "udp",
			},
			{
				ContainerPort: 445,
				HostPort:      445,
				Protocol:      "foobar",
			},
		},
	}
	exposedPorts, bindings := makePortsAndBindings(&container)
	if len(container.Ports) != len(exposedPorts) ||
		len(container.Ports) != len(bindings) {
		t.Errorf("Unexpected ports and bindings, %#v %#v %#v", container, exposedPorts, bindings)
	}
	for key, value := range bindings {
		switch value[0].HostPort {
		case "8080":
			if !reflect.DeepEqual(docker.Port("80/tcp"), key) {
				t.Errorf("Unexpected docker port: %#v", key)
			}
		case "443":
			if !reflect.DeepEqual(docker.Port("443/tcp"), key) {
				t.Errorf("Unexpected docker port: %#v", key)
			}
		case "444":
			if !reflect.DeepEqual(docker.Port("444/udp"), key) {
				t.Errorf("Unexpected docker port: %#v", key)
			}
		case "445":
			if !reflect.DeepEqual(docker.Port("445/tcp"), key) {
				t.Errorf("Unexpected docker port: %#v", key)
			}
		}
	}

}

func TestExtractFromNonExistentFile(t *testing.T) {
	kubelet := Kubelet{}
	_, err := kubelet.extractFromFile("/some/fake/file")
	if err == nil {
		t.Error("Unexpected non-error.")
	}
}

func TestExtractFromBadDataFile(t *testing.T) {
	kubelet := Kubelet{}

	badData := []byte{1, 2, 3}
	file, err := ioutil.TempFile("", "foo")
	expectNoError(t, err)
	name := file.Name()
	file.Close()
	ioutil.WriteFile(name, badData, 0755)
	_, err = kubelet.extractFromFile(name)

	if err == nil {
		t.Error("Unexpected non-error.")
	}

}

func TestExtractFromValidDataFile(t *testing.T) {
	kubelet := Kubelet{}

	manifest := api.ContainerManifest{Id: "bar"}
	data, err := json.Marshal(manifest)
	expectNoError(t, err)
	file, err := ioutil.TempFile("", "foo")
	expectNoError(t, err)
	name := file.Name()
	expectNoError(t, file.Close())
	ioutil.WriteFile(name, data, 0755)

	read, err := kubelet.extractFromFile(name)
	expectNoError(t, err)
	if !reflect.DeepEqual(read, manifest) {
		t.Errorf("Unexpected difference.  Expected %#v, got %#v", manifest, read)
	}
}

func TestExtractFromEmptyDir(t *testing.T) {
	kubelet := Kubelet{}

	dirName, err := ioutil.TempDir("", "foo")
	expectNoError(t, err)

	_, err = kubelet.extractFromDir(dirName)
	expectNoError(t, err)
}

func TestExtractFromDir(t *testing.T) {
	kubelet := Kubelet{}

	manifests := []api.ContainerManifest{
		{Id: "aaaa"},
		{Id: "bbbb"},
	}

	dirName, err := ioutil.TempDir("", "foo")
	expectNoError(t, err)

	for _, manifest := range manifests {
		data, err := json.Marshal(manifest)
		expectNoError(t, err)
		file, err := ioutil.TempFile(dirName, manifest.Id)
		expectNoError(t, err)
		name := file.Name()
		expectNoError(t, file.Close())
		ioutil.WriteFile(name, data, 0755)
	}

	read, err := kubelet.extractFromDir(dirName)
	expectNoError(t, err)
	if !reflect.DeepEqual(read, manifests) {
		t.Errorf("Unexpected difference.  Expected %#v, got %#v", manifests, read)
	}
}

func TestExtractFromHttpBadness(t *testing.T) {
	kubelet := Kubelet{}
	updateChannel := make(chan manifestUpdate)
	reader := startReading(updateChannel)

	err := kubelet.extractFromHTTP("http://localhost:12345", updateChannel)
	if err == nil {
		t.Error("Unexpected non-error.")
	}
	close(updateChannel)
	list := reader.GetList()

	if len(list) != 0 {
		t.Errorf("Unexpected list: %#v", list)
	}
}

func TestExtractFromHttp(t *testing.T) {
	kubelet := Kubelet{}
	updateChannel := make(chan manifestUpdate)
	reader := startReading(updateChannel)

	manifests := []api.ContainerManifest{
		{Id: "foo"},
	}
	// TODO: provide a mechanism for taking arrays of
	// manifests or a single manifest.
	data, err := json.Marshal(manifests[0])

	fakeHandler := util.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(data),
	}
	testServer := httptest.NewServer(&fakeHandler)

	err = kubelet.extractFromHTTP(testServer.URL, updateChannel)
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	close(updateChannel)

	read := reader.GetList()

	if len(read) != 1 {
		t.Errorf("Unexpected list: %#v", read)
	}
	if !reflect.DeepEqual(manifests, read[0]) {
		t.Errorf("Unexpected difference.  Expected: %#v, Saw: %#v", manifests, read[0])
	}
}

func TestWatchEtcd(t *testing.T) {
	watchChannel := make(chan *etcd.Response)
	updateChannel := make(chan manifestUpdate)
	kubelet := Kubelet{}
	reader := startReading(updateChannel)

	manifest := []api.ContainerManifest{
		{
			Id: "foo",
		},
	}
	data, err := json.Marshal(manifest)
	expectNoError(t, err)

	go kubelet.WatchEtcd(watchChannel, updateChannel)

	watchChannel <- &etcd.Response{
		Node: &etcd.Node{
			Value: string(data),
		},
	}
	close(watchChannel)
	close(updateChannel)

	read := reader.GetList()
	if len(read) != 1 ||
		!reflect.DeepEqual(read[0], manifest) {
		t.Errorf("Unexpected manifest(s) %#v %#v", read[0], manifest)
	}
}

type mockCadvisorClient struct {
	mock.Mock
}

func (self *mockCadvisorClient) ContainerInfo(name string) (*info.ContainerInfo, error) {
	args := self.Called(name)
	return args.Get(0).(*info.ContainerInfo), args.Error(1)
}

func (self *mockCadvisorClient) MachineInfo() (*info.MachineInfo, error) {
	args := self.Called()
	return args.Get(0).(*info.MachineInfo), args.Error(1)
}

func areSamePercentiles(
	cadvisorPercentiles []info.Percentile,
	kubePercentiles []api.Percentile,
	t *testing.T,
) {
	if len(cadvisorPercentiles) != len(kubePercentiles) {
		t.Errorf("cadvisor gives %v percentiles; kubelet got %v", len(cadvisorPercentiles), len(kubePercentiles))
		return
	}
	for _, ap := range cadvisorPercentiles {
		found := false
		for _, kp := range kubePercentiles {
			if ap.Percentage == kp.Percentage {
				found = true
				if ap.Value != kp.Value {
					t.Errorf("%v percentile from cadvisor is %v; kubelet got %v",
						ap.Percentage,
						ap.Value,
						kp.Value)
				}
			}
		}
		if !found {
			t.Errorf("Unable to find %v percentile in kubelet's data", ap.Percentage)
		}
	}
}

func TestGetContainerStats(t *testing.T) {
	containerId := "ab2cdf"
	containerPath := fmt.Sprintf("/docker/%v", containerId)
	containerInfo := &info.ContainerInfo{
		ContainerReference: info.ContainerReference{
			Name: containerPath,
		},
		StatsPercentiles: &info.ContainerStatsPercentiles{
			MaxMemoryUsage: 1024000,
			MemoryUsagePercentiles: []info.Percentile{
				{50, 100},
				{80, 180},
				{90, 190},
			},
			CpuUsagePercentiles: []info.Percentile{
				{51, 101},
				{81, 181},
				{91, 191},
			},
		},
	}
	fakeDocker := FakeDockerClient{
		err: nil,
	}

	mockCadvisor := &mockCadvisorClient{}
	mockCadvisor.On("ContainerInfo", containerPath).Return(containerInfo, nil)

	kubelet := Kubelet{
		DockerClient:   &fakeDocker,
		CadvisorClient: mockCadvisor,
	}
	fakeDocker.containerList = []docker.APIContainers{
		{
			Names: []string{"foo"},
			ID:    containerId,
		},
	}

	stats, err := kubelet.GetContainerStats("foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if stats.MaxMemoryUsage != containerInfo.StatsPercentiles.MaxMemoryUsage {
		t.Errorf("wrong max memory usage")
	}
	areSamePercentiles(containerInfo.StatsPercentiles.CpuUsagePercentiles, stats.CpuUsagePercentiles, t)
	areSamePercentiles(containerInfo.StatsPercentiles.MemoryUsagePercentiles, stats.MemoryUsagePercentiles, t)
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerStatsWithoutCadvisor(t *testing.T) {
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
	}

	stats, _ := kubelet.GetContainerStats("foo")
	// When there's no cAdvisor, the stats should be either nil or empty
	if stats == nil {
		return
	}
	if stats.MaxMemoryUsage != 0 {
		t.Errorf("MaxMemoryUsage is %v even if there's no cadvisor", stats.MaxMemoryUsage)
	}
	if len(stats.CpuUsagePercentiles) > 0 {
		t.Errorf("Cpu usage percentiles is not empty (%+v) even if there's no cadvisor", stats.CpuUsagePercentiles)
	}
	if len(stats.MemoryUsagePercentiles) > 0 {
		t.Errorf("Memory usage percentiles is not empty (%+v) even if there's no cadvisor", stats.MemoryUsagePercentiles)
	}
}

func TestGetContainerStatsWhenCadvisorFailed(t *testing.T) {
	containerId := "ab2cdf"
	containerPath := fmt.Sprintf("/docker/%v", containerId)
	fakeDocker := FakeDockerClient{
		err: nil,
	}

	containerInfo := &info.ContainerInfo{}
	mockCadvisor := &mockCadvisorClient{}
	expectedErr := fmt.Errorf("some error")
	mockCadvisor.On("ContainerInfo", containerPath).Return(containerInfo, expectedErr)

	kubelet := Kubelet{
		DockerClient:   &fakeDocker,
		CadvisorClient: mockCadvisor,
	}
	fakeDocker.containerList = []docker.APIContainers{
		{
			Names: []string{"foo"},
			ID:    containerId,
		},
	}

	stats, err := kubelet.GetContainerStats("foo")
	if stats != nil {
		t.Errorf("non-nil stats on error")
	}
	if err == nil {
		t.Errorf("expect error but received nil error")
		return
	}
	if err.Error() != expectedErr.Error() {
		t.Errorf("wrong error message. expect %v, got %v", err, expectedErr)
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerStatsOnNonExistContainer(t *testing.T) {
	fakeDocker := FakeDockerClient{
		err: nil,
	}

	mockCadvisor := &mockCadvisorClient{}

	kubelet := Kubelet{
		DockerClient:   &fakeDocker,
		CadvisorClient: mockCadvisor,
	}
	fakeDocker.containerList = []docker.APIContainers{}

	stats, _ := kubelet.GetContainerStats("foo")
	if stats != nil {
		t.Errorf("non-nil stats on non exist container")
	}
	mockCadvisor.AssertExpectations(t)
}
