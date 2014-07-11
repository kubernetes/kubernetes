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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
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

func makeTestKubelet(t *testing.T) (*Kubelet, *tools.FakeEtcdClient, *FakeDockerClient) {
	fakeEtcdClient := tools.MakeFakeEtcdClient(t)
	fakeDocker := &FakeDockerClient{
		err: nil,
	}

	kubelet := New()
	kubelet.DockerClient = fakeDocker
	kubelet.DockerPuller = &FakeDockerPuller{}
	kubelet.EtcdClient = fakeEtcdClient
	return kubelet, fakeEtcdClient, fakeDocker
}

func TestExtractJSON(t *testing.T) {
	obj := TestObject{}
	kubelet, _, _ := makeTestKubelet(t)
	data := `{ "name": "foo", "data": { "value": "bar", "number": 10 } }`
	kubelet.ExtractYAMLData([]byte(data), &obj)

	verifyStringEquals(t, obj.Name, "foo")
	verifyStringEquals(t, obj.Data.Value, "bar")
	verifyIntEquals(t, obj.Data.Number, 10)
}

func verifyCalls(t *testing.T, fakeDocker *FakeDockerClient, calls []string) {
	verifyStringArrayEquals(t, fakeDocker.called, calls)
}

func verifyStringArrayEquals(t *testing.T, actual, expected []string) {
	invalid := len(actual) != len(expected)
	if !invalid {
		for ix, value := range actual {
			if expected[ix] != value {
				invalid = true
			}
		}
	}
	if invalid {
		t.Errorf("Expected: %#v, Actual: %#v", expected, actual)
	}
}

func verifyPackUnpack(t *testing.T, manifestID, containerName string) {
	name := buildDockerName(
		&api.ContainerManifest{ID: manifestID},
		&api.Container{Name: containerName},
	)
	returnedManifestID, returnedContainerName := parseDockerName(name)
	if manifestID != returnedManifestID || containerName != returnedContainerName {
		t.Errorf("For (%s, %s), unpacked (%s, %s)", manifestID, containerName, returnedManifestID, returnedContainerName)
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

func TestGetContainerID(t *testing.T) {
	kubelet, _, fakeDocker := makeTestKubelet(t)
	manifest := api.ContainerManifest{
		ID: "qux",
	}
	container := api.Container{
		Name: "foo",
	}
	fakeDocker.containerList = []docker.APIContainers{
		{
			ID:    "foobar",
			Names: []string{"/k8s--foo--qux--1234"},
		},
		{
			ID:    "barbar",
			Names: []string{"/k8s--bar--qux--2565"},
		},
	}
	fakeDocker.container = &docker.Container{
		ID: "foobar",
	}

	id, err := kubelet.getContainerID(&manifest, &container)
	verifyCalls(t, fakeDocker, []string{"list"})
	if id == "" {
		t.Errorf("Failed to find container %#v", container)
	}
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}

	fakeDocker.clearCalls()
	missingManifest := api.ContainerManifest{ID: "foobar"}
	id, err = kubelet.getContainerID(&missingManifest, &container)
	verifyCalls(t, fakeDocker, []string{"list"})
	if id != "" {
		t.Errorf("Failed to not find container %#v", missingManifest)
	}
}

func TestKillContainerWithError(t *testing.T) {
	fakeDocker := &FakeDockerClient{
		err: fmt.Errorf("sample error"),
		containerList: []docker.APIContainers{
			{
				ID:    "1234",
				Names: []string{"/k8s--foo--qux--1234"},
			},
			{
				ID:    "5678",
				Names: []string{"/k8s--bar--qux--5678"},
			},
		},
	}
	kubelet, _, _ := makeTestKubelet(t)
	kubelet.DockerClient = fakeDocker
	err := kubelet.killContainer(fakeDocker.containerList[0])
	verifyError(t, err)
	verifyCalls(t, fakeDocker, []string{"stop"})
}

func TestKillContainer(t *testing.T) {
	kubelet, _, fakeDocker := makeTestKubelet(t)
	fakeDocker.containerList = []docker.APIContainers{
		{
			ID:    "1234",
			Names: []string{"/k8s--foo--qux--1234"},
		},
		{
			ID:    "5678",
			Names: []string{"/k8s--bar--qux--5678"},
		},
	}
	fakeDocker.container = &docker.Container{
		ID: "foobar",
	}

	err := kubelet.killContainer(fakeDocker.containerList[0])
	verifyNoError(t, err)
	verifyCalls(t, fakeDocker, []string{"stop"})
}

func TestResponseToContainersNil(t *testing.T) {
	kubelet, _, _ := makeTestKubelet(t)
	list, err := kubelet.ResponseToManifests(&etcd.Response{Node: nil})
	if len(list) != 0 {
		t.Errorf("Unexpected non-zero list: %#v", list)
	}
	if err == nil {
		t.Error("Unexpected non-error")
	}
}

func TestResponseToManifests(t *testing.T) {
	kubelet, _, _ := makeTestKubelet(t)
	list, err := kubelet.ResponseToManifests(&etcd.Response{
		Node: &etcd.Node{
			Value: util.MakeJSONString([]api.ContainerManifest{
				{ID: "foo"},
				{ID: "bar"},
			}),
		},
	})
	if len(list) != 2 || list[0].ID != "foo" || list[1].ID != "bar" {
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
	kubelet, fakeClient, _ := makeTestKubelet(t)
	channel := make(chan manifestUpdate)
	reader := startReading(channel)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: nil,
	}
	err := kubelet.getKubeletStateFromEtcd("/registry/hosts/machine/kubelet", channel)
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
	kubelet, fakeClient, _ := makeTestKubelet(t)
	channel := make(chan manifestUpdate)
	reader := startReading(channel)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: util.MakeJSONString([]api.Container{}),
			},
		},
		E: nil,
	}
	err := kubelet.getKubeletStateFromEtcd("/registry/hosts/machine/kubelet", channel)
	expectNoError(t, err)
	close(channel)
	list := reader.GetList()
	if len(list) != 1 {
		t.Errorf("Unexpected list: %#v", list)
	}
}

func TestGetKubeletStateFromEtcdNotFound(t *testing.T) {
	kubelet, fakeClient, _ := makeTestKubelet(t)
	channel := make(chan manifestUpdate)
	reader := startReading(channel)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: &etcd.EtcdError{
			ErrorCode: 100,
		},
	}
	err := kubelet.getKubeletStateFromEtcd("/registry/hosts/machine/kubelet", channel)
	expectNoError(t, err)
	close(channel)
	list := reader.GetList()
	if len(list) != 0 {
		t.Errorf("Unexpected list: %#v", list)
	}
}

func TestGetKubeletStateFromEtcdError(t *testing.T) {
	kubelet, fakeClient, _ := makeTestKubelet(t)
	channel := make(chan manifestUpdate)
	reader := startReading(channel)
	fakeClient.Data["/registry/hosts/machine/kubelet"] = tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: &etcd.EtcdError{
			ErrorCode: 200, // non not found error
		},
	}
	err := kubelet.getKubeletStateFromEtcd("/registry/hosts/machine/kubelet", channel)
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
	kubelet, _, fakeDocker := makeTestKubelet(t)
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
	err := kubelet.SyncManifests([]api.ContainerManifest{
		{
			ID: "foo",
			Containers: []api.Container{
				{Name: "bar"},
			},
		},
	})
	expectNoError(t, err)
	verifyCalls(t, fakeDocker, []string{"list", "list", "list", "list"})
}

func TestSyncManifestsDeletes(t *testing.T) {
	kubelet, _, fakeDocker := makeTestKubelet(t)
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
	err := kubelet.SyncManifests([]api.ContainerManifest{})
	expectNoError(t, err)
	verifyCalls(t, fakeDocker, []string{"list", "stop", "stop"})

	// A map interation is used to delete containers, so must not depend on
	// order here.
	expectedToStop := map[string]bool{
		"1234": true,
		"9876": true,
	}
	if len(fakeDocker.stopped) != 2 ||
		!expectedToStop[fakeDocker.stopped[0]] ||
		!expectedToStop[fakeDocker.stopped[1]] {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.stopped)
	}
}

type FalseHealthChecker struct{}

func (f *FalseHealthChecker) HealthCheck(container api.Container) (HealthCheckStatus, error) {
	return CheckUnhealthy, nil
}

func TestSyncManifestsUnhealthy(t *testing.T) {
	kubelet, _, fakeDocker := makeTestKubelet(t)
	kubelet.HealthChecker = &FalseHealthChecker{}
	fakeDocker.containerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s--bar--foo"},
			ID:    "1234",
		},
		{
			// network container
			Names: []string{"/k8s--net--foo--"},
			ID:    "9876",
		},
	}
	err := kubelet.SyncManifests([]api.ContainerManifest{
		{
			ID: "foo",
			Containers: []api.Container{
				{Name: "bar",
					LivenessProbe: &api.LivenessProbe{
						// Always returns healthy == false
						Type: "false",
					},
				},
			},
		}})
	expectNoError(t, err)
	verifyCalls(t, fakeDocker, []string{"list", "list", "list", "stop", "create", "start", "list"})

	// A map interation is used to delete containers, so must not depend on
	// order here.
	expectedToStop := map[string]bool{
		"1234": true,
	}
	if len(fakeDocker.stopped) != 1 ||
		!expectedToStop[fakeDocker.stopped[0]] {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.stopped)
	}
}

func TestEventWriting(t *testing.T) {
	kubelet, fakeEtcd, _ := makeTestKubelet(t)
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
	kubelet, fakeEtcd, _ := makeTestKubelet(t)
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
	volumes, binds := makeVolumesAndBinds("pod", &container)

	expectedVolumes := []string{"/mnt/path", "/mnt/path2"}
	expectedBinds := []string{"/exports/pod/disk:/mnt/path", "/exports/pod/disk2:/mnt/path2:ro", "/mnt/path3:/mnt/path3"}
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
				HostIP:        "127.0.0.1",
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
			if value[0].HostIp != "127.0.0.1" {
				t.Errorf("Unexpected host IP: %s", value[0].HostIp)
			}
		case "443":
			if !reflect.DeepEqual(docker.Port("443/tcp"), key) {
				t.Errorf("Unexpected docker port: %#v", key)
			}
			if value[0].HostIp != "" {
				t.Errorf("Unexpected host IP: %s", value[0].HostIp)
			}
		case "444":
			if !reflect.DeepEqual(docker.Port("444/udp"), key) {
				t.Errorf("Unexpected docker port: %#v", key)
			}
			if value[0].HostIp != "" {
				t.Errorf("Unexpected host IP: %s", value[0].HostIp)
			}
		case "445":
			if !reflect.DeepEqual(docker.Port("445/tcp"), key) {
				t.Errorf("Unexpected docker port: %#v", key)
			}
			if value[0].HostIp != "" {
				t.Errorf("Unexpected host IP: %s", value[0].HostIp)
			}
		}
	}
}

func TestCheckHostPortConflicts(t *testing.T) {
	successCaseAll := []api.ContainerManifest{
		{Containers: []api.Container{{Ports: []api.Port{{HostPort: 80}}}}},
		{Containers: []api.Container{{Ports: []api.Port{{HostPort: 81}}}}},
		{Containers: []api.Container{{Ports: []api.Port{{HostPort: 82}}}}},
	}
	successCaseNew := api.ContainerManifest{
		Containers: []api.Container{{Ports: []api.Port{{HostPort: 83}}}},
	}
	if errs := checkHostPortConflicts(successCaseAll, &successCaseNew); len(errs) != 0 {
		t.Errorf("Expected success: %v", errs)
	}

	failureCaseAll := []api.ContainerManifest{
		{Containers: []api.Container{{Ports: []api.Port{{HostPort: 80}}}}},
		{Containers: []api.Container{{Ports: []api.Port{{HostPort: 81}}}}},
		{Containers: []api.Container{{Ports: []api.Port{{HostPort: 82}}}}},
	}
	failureCaseNew := api.ContainerManifest{
		Containers: []api.Container{{Ports: []api.Port{{HostPort: 81}}}},
	}
	if errs := checkHostPortConflicts(failureCaseAll, &failureCaseNew); len(errs) == 0 {
		t.Errorf("Expected failure")
	}
}

func TestExtractFromNonExistentFile(t *testing.T) {
	kubelet := New()
	_, err := kubelet.extractFromFile("/some/fake/file")
	if err == nil {
		t.Error("Unexpected non-error.")
	}
}

func TestExtractFromBadDataFile(t *testing.T) {
	kubelet := New()

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
	kubelet := New()

	manifest := api.ContainerManifest{ID: "bar"}
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
	kubelet := New()

	dirName, err := ioutil.TempDir("", "foo")
	expectNoError(t, err)

	_, err = kubelet.extractFromDir(dirName)
	expectNoError(t, err)
}

func TestExtractFromDir(t *testing.T) {
	kubelet := New()

	manifests := []api.ContainerManifest{
		{ID: "aaaa"},
		{ID: "bbbb"},
	}

	dirName, err := ioutil.TempDir("", "foo")
	expectNoError(t, err)

	for _, manifest := range manifests {
		data, err := json.Marshal(manifest)
		expectNoError(t, err)
		file, err := ioutil.TempFile(dirName, manifest.ID)
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
	kubelet := New()
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

func TestExtractFromHttpSingle(t *testing.T) {
	kubelet := New()
	updateChannel := make(chan manifestUpdate)
	reader := startReading(updateChannel)

	manifests := []api.ContainerManifest{
		{Version: "v1beta1", ID: "foo"},
	}
	// Taking a single-manifest from a URL allows kubelet to be used
	// in the implementation of google's container VM image.
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
		return
	}
	if !reflect.DeepEqual(manifests, read[0]) {
		t.Errorf("Unexpected difference.  Expected: %#v, Saw: %#v", manifests, read[0])
	}
}

func TestExtractFromHttpMultiple(t *testing.T) {
	kubelet := New()
	updateChannel := make(chan manifestUpdate)
	reader := startReading(updateChannel)

	manifests := []api.ContainerManifest{
		{Version: "v1beta1", ID: "foo"},
		{Version: "v1beta1", ID: "bar"},
	}
	data, err := json.Marshal(manifests)
	if err != nil {
		t.Fatalf("Some weird json problem: %v", err)
	}

	t.Logf("Serving: %v", string(data))

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
		return
	}
	if !reflect.DeepEqual(manifests, read[0]) {
		t.Errorf("Unexpected difference.  Expected: %#v, Saw: %#v", manifests, read[0])
	}
}

func TestExtractFromHttpEmptyArray(t *testing.T) {
	kubelet := New()
	updateChannel := make(chan manifestUpdate)
	reader := startReading(updateChannel)

	manifests := []api.ContainerManifest{}
	data, err := json.Marshal(manifests)
	if err != nil {
		t.Fatalf("Some weird json problem: %v", err)
	}

	t.Logf("Serving: %v", string(data))

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
		return
	}
	if len(read[0]) != 0 {
		t.Errorf("Unexpected manifests: %#v", read[0])
	}
}

func TestWatchEtcd(t *testing.T) {
	watchChannel := make(chan *etcd.Response)
	updateChannel := make(chan manifestUpdate)
	kubelet := New()
	reader := startReading(updateChannel)

	manifest := []api.ContainerManifest{
		{
			ID: "foo",
		},
	}
	data, err := json.Marshal(manifest)
	expectNoError(t, err)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		kubelet.WatchEtcd(watchChannel, updateChannel)
		wg.Done()
	}()

	watchChannel <- &etcd.Response{
		Node: &etcd.Node{
			Value: string(data),
		},
	}
	close(watchChannel)
	wg.Wait()
	close(updateChannel)

	read := reader.GetList()
	if len(read) != 1 {
		t.Errorf("Expected number of results: %v", len(read))
	} else if !reflect.DeepEqual(read[0], manifest) {
		t.Errorf("Unexpected manifest(s) %#v %#v", read[0], manifest)
	}
}

type mockCadvisorClient struct {
	mock.Mock
}

// ContainerInfo is a mock implementation of CadvisorInterface.ContainerInfo.
func (c *mockCadvisorClient) ContainerInfo(name string) (*info.ContainerInfo, error) {
	args := c.Called(name)
	return args.Get(0).(*info.ContainerInfo), args.Error(1)
}

// MachineInfo is a mock implementation of CadvisorInterface.MachineInfo.
func (c *mockCadvisorClient) MachineInfo() (*info.MachineInfo, error) {
	args := c.Called()
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
	containerID := "ab2cdf"
	containerPath := fmt.Sprintf("/docker/%v", containerID)
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

	mockCadvisor := &mockCadvisorClient{}
	mockCadvisor.On("ContainerInfo", containerPath).Return(containerInfo, nil)

	kubelet, _, fakeDocker := makeTestKubelet(t)
	kubelet.CadvisorClient = mockCadvisor
	fakeDocker.containerList = []docker.APIContainers{
		{
			ID: containerID,
			// pod id: qux
			// container id: foo
			Names: []string{"/k8s--foo--qux--1234"},
		},
	}

	stats, err := kubelet.GetContainerStats("qux", "foo")
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

func TestGetMachineStats(t *testing.T) {
	containerPath := "/"
	containerInfo := &info.ContainerInfo{
		ContainerReference: info.ContainerReference{
			Name: containerPath,
		}, StatsPercentiles: &info.ContainerStatsPercentiles{MaxMemoryUsage: 1024000, MemoryUsagePercentiles: []info.Percentile{{50, 100}, {80, 180},
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
		DockerPuller:   &FakeDockerPuller{},
		CadvisorClient: mockCadvisor,
	}

	// If the container name is an empty string, then it means the root container.
	stats, err := kubelet.GetMachineStats()
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
	kubelet, _, fakeDocker := makeTestKubelet(t)
	fakeDocker.containerList = []docker.APIContainers{
		{
			ID: "foobar",
			// pod id: qux
			// container id: foo
			Names: []string{"/k8s--foo--qux--1234"},
		},
	}

	stats, _ := kubelet.GetContainerStats("qux", "foo")
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
	containerID := "ab2cdf"
	containerPath := fmt.Sprintf("/docker/%v", containerID)

	containerInfo := &info.ContainerInfo{}
	mockCadvisor := &mockCadvisorClient{}
	expectedErr := fmt.Errorf("some error")
	mockCadvisor.On("ContainerInfo", containerPath).Return(containerInfo, expectedErr)

	kubelet, _, fakeDocker := makeTestKubelet(t)
	kubelet.CadvisorClient = mockCadvisor
	fakeDocker.containerList = []docker.APIContainers{
		{
			ID: containerID,
			// pod id: qux
			// container id: foo
			Names: []string{"/k8s--foo--qux--1234"},
		},
	}

	stats, err := kubelet.GetContainerStats("qux", "foo")
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
	mockCadvisor := &mockCadvisorClient{}

	kubelet, _, fakeDocker := makeTestKubelet(t)
	kubelet.CadvisorClient = mockCadvisor
	fakeDocker.containerList = []docker.APIContainers{}

	stats, _ := kubelet.GetContainerStats("qux", "foo")
	if stats != nil {
		t.Errorf("non-nil stats on non exist container")
	}
	mockCadvisor.AssertExpectations(t)
}

func TestParseImageName(t *testing.T) {
	name, tag := parseImageName("ubuntu")
	if name != "ubuntu" || tag != "" {
		t.Fatal("Unexpected name/tag: %s/%s", name, tag)
	}

	name, tag = parseImageName("ubuntu:2342")
	if name != "ubuntu" || tag != "2342" {
		t.Fatal("Unexpected name/tag: %s/%s", name, tag)
	}

	name, tag = parseImageName("foo/bar:445566")
	if name != "foo/bar" || tag != "445566" {
		t.Fatal("Unexpected name/tag: %s/%s", name, tag)
	}

	name, tag = parseImageName("registry.example.com:5000/foobar")
	if name != "registry.example.com:5000/foobar" || tag != "" {
		t.Fatal("Unexpected name/tag: %s/%s", name, tag)
	}

	name, tag = parseImageName("registry.example.com:5000/foobar:5342")
	if name != "registry.example.com:5000/foobar" || tag != "5342" {
		t.Fatal("Unexpected name/tag: %s/%s", name, tag)
	}
}
