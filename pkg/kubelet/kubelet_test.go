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
	"reflect"
	"sync"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
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

	kubelet := &Kubelet{}
	kubelet.dockerClient = fakeDocker
	kubelet.dockerPuller = &FakeDockerPuller{}
	kubelet.etcdClient = fakeEtcdClient
	kubelet.rootDirectory = "/tmp/kubelet"
	kubelet.podWorkers = newPodWorkers()
	return kubelet, fakeEtcdClient, fakeDocker
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

func verifyPackUnpack(t *testing.T, podNamespace, podName, containerName string) {
	name := buildDockerName(
		&Pod{Name: podName, Namespace: podNamespace},
		&api.Container{Name: containerName},
	)
	podFullName := fmt.Sprintf("%s.%s", podName, podNamespace)
	returnedPodFullName, returnedContainerName := parseDockerName(name)
	if podFullName != returnedPodFullName || containerName != returnedContainerName {
		t.Errorf("For (%s, %s), unpacked (%s, %s)", podFullName, containerName, returnedPodFullName, returnedContainerName)
	}
}

func verifyBoolean(t *testing.T, expected, value bool) {
	if expected != value {
		t.Errorf("Unexpected boolean.  Expected %t.  Found %t", expected, value)
	}
}

func TestContainerManifestNaming(t *testing.T) {
	verifyPackUnpack(t, "file", "manifest1234", "container5678")
	verifyPackUnpack(t, "file", "manifest--", "container__")
	verifyPackUnpack(t, "file", "--manifest", "__container")
	verifyPackUnpack(t, "", "m___anifest_", "container-_-")
	verifyPackUnpack(t, "other", "_m___anifest", "-_-container")
}

func TestGetContainerID(t *testing.T) {
	_, _, fakeDocker := makeTestKubelet(t)
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

	dockerContainers, err := getKubeletDockerContainers(fakeDocker)
	if err != nil {
		t.Errorf("Expected no error, Got %#v", err)
	}
	if len(dockerContainers) != 2 {
		t.Errorf("Expected %#v, Got %#v", fakeDocker.containerList, dockerContainers)
	}
	verifyCalls(t, fakeDocker, []string{"list"})
	dockerContainer, found := dockerContainers.FindPodContainer("qux", "foo")
	if dockerContainer == nil || !found {
		t.Errorf("Failed to find container %#v", dockerContainer)
	}

	fakeDocker.clearCalls()
	dockerContainer, found = dockerContainers.FindPodContainer("foobar", "foo")
	verifyCalls(t, fakeDocker, []string{})
	if dockerContainer != nil || found {
		t.Errorf("Should not have found container %#v", dockerContainer)
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
	kubelet.dockerClient = fakeDocker
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

type channelReader struct {
	list [][]Pod
	wg   sync.WaitGroup
}

func startReading(channel <-chan interface{}) *channelReader {
	cr := &channelReader{}
	cr.wg.Add(1)
	go func() {
		for {
			update, ok := <-channel
			if !ok {
				break
			}
			cr.list = append(cr.list, update.(PodUpdate).Pods)
		}
		cr.wg.Done()
	}()
	return cr
}

func (cr *channelReader) GetList() [][]Pod {
	cr.wg.Wait()
	return cr.list
}

func TestSyncPodsDoesNothing(t *testing.T) {
	kubelet, _, fakeDocker := makeTestKubelet(t)
	fakeDocker.containerList = []docker.APIContainers{
		{
			// format is k8s--<container-id>--<pod-fullname>
			Names: []string{"/k8s--bar--foo.test"},
			ID:    "1234",
		},
		{
			// network container
			Names: []string{"/k8s--net--foo.test--"},
			ID:    "9876",
		},
	}
	fakeDocker.container = &docker.Container{
		ID: "1234",
	}
	err := kubelet.SyncPods([]Pod{
		{
			Name:      "foo",
			Namespace: "test",
			Manifest: api.ContainerManifest{
				ID: "foo",
				Containers: []api.Container{
					{Name: "bar"},
				},
			},
		},
	})
	expectNoError(t, err)
	verifyCalls(t, fakeDocker, []string{"list", "list"})
}

func TestSyncPodsDeletes(t *testing.T) {
	kubelet, _, fakeDocker := makeTestKubelet(t)
	fakeDocker.containerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s--foo--bar.test"},
			ID:    "1234",
		},
		{
			// network container
			Names: []string{"/k8s--net--foo.test--"},
			ID:    "9876",
		},
		{
			Names: []string{"foo"},
			ID:    "4567",
		},
	}
	err := kubelet.SyncPods([]Pod{})
	expectNoError(t, err)
	verifyCalls(t, fakeDocker, []string{"list", "list", "stop", "stop"})

	// A map iteration is used to delete containers, so must not depend on
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

func TestSyncPodDeletesDuplicate(t *testing.T) {
	kubelet, _, fakeDocker := makeTestKubelet(t)
	dockerContainers := DockerContainers{
		"1234": &docker.APIContainers{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s--foo--bar.test--1"},
			ID:    "1234",
		},
		"9876": &docker.APIContainers{
			// network container
			Names: []string{"/k8s--net--bar.test--"},
			ID:    "9876",
		},
		"4567": &docker.APIContainers{
			// Duplicate for the same container.
			Names: []string{"/k8s--foo--bar.test--2"},
			ID:    "4567",
		},
		"2304": &docker.APIContainers{
			// Container for another pod, untouched.
			Names: []string{"/k8s--baz--fiz.test--6"},
			ID:    "2304",
		},
	}
	err := kubelet.syncPod(&Pod{
		Name:      "bar",
		Namespace: "test",
		Manifest: api.ContainerManifest{
			ID: "bar",
			Containers: []api.Container{
				{Name: "foo"},
			},
		},
	}, dockerContainers)
	expectNoError(t, err)
	verifyCalls(t, fakeDocker, []string{"stop"})

	// Expect one of the duplicates to be killed.
	if len(fakeDocker.stopped) != 1 || (len(fakeDocker.stopped) != 0 && fakeDocker.stopped[0] != "1234" && fakeDocker.stopped[0] != "4567") {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.stopped)
	}
}

type FalseHealthChecker struct{}

func (f *FalseHealthChecker) HealthCheck(container api.Container) (health.Status, error) {
	return health.Unhealthy, nil
}

func TestSyncPodUnhealthy(t *testing.T) {
	kubelet, _, fakeDocker := makeTestKubelet(t)
	kubelet.healthChecker = &FalseHealthChecker{}
	dockerContainers := DockerContainers{
		"1234": &docker.APIContainers{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s--bar--foo.test"},
			ID:    "1234",
		},
		"9876": &docker.APIContainers{
			// network container
			Names: []string{"/k8s--net--foo.test--"},
			ID:    "9876",
		},
	}
	err := kubelet.syncPod(&Pod{
		Name:      "foo",
		Namespace: "test",
		Manifest: api.ContainerManifest{
			ID: "foo",
			Containers: []api.Container{
				{Name: "bar",
					LivenessProbe: &api.LivenessProbe{
						// Always returns healthy == false
						Type: "false",
					},
				},
			},
		},
	}, dockerContainers)
	expectNoError(t, err)
	verifyCalls(t, fakeDocker, []string{"stop", "create", "start"})

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

func TestMountExternalVolumes(t *testing.T) {
	kubelet, _, _ := makeTestKubelet(t)
	manifest := api.ContainerManifest{
		Volumes: []api.Volume{
			{
				Name: "host-dir",
				Source: &api.VolumeSource{
					HostDirectory: &api.HostDirectory{"/dir/path"},
				},
			},
		},
	}
	podVolumes, _ := kubelet.mountExternalVolumes(&manifest)
	expectedPodVolumes := make(volumeMap)
	expectedPodVolumes["host-dir"] = &volume.HostDirectory{"/dir/path"}
	if len(expectedPodVolumes) != len(podVolumes) {
		t.Errorf("Unexpected volumes. Expected %#v got %#v.  Manifest was: %#v", expectedPodVolumes, podVolumes, manifest)
	}
	for name, expectedVolume := range expectedPodVolumes {
		if _, ok := podVolumes[name]; !ok {
			t.Errorf("Pod volumes map is missing key: %s. %#v", expectedVolume, podVolumes)
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
			{
				MountPath: "/mnt/path4",
				Name:      "disk4",
				ReadOnly:  false,
			},
			{
				MountPath: "/mnt/path5",
				Name:      "disk5",
				ReadOnly:  false,
			},
		},
	}

	pod := Pod{
		Name:      "pod",
		Namespace: "test",
	}

	podVolumes := make(volumeMap)
	podVolumes["disk4"] = &volume.HostDirectory{"/mnt/host"}
	podVolumes["disk5"] = &volume.EmptyDirectory{"disk5", "podID", "/var/lib/kubelet"}

	volumes, binds := makeVolumesAndBinds(&pod, &container, podVolumes)

	expectedVolumes := []string{"/mnt/path", "/mnt/path2"}
	expectedBinds := []string{"/exports/pod.test/disk:/mnt/path", "/exports/pod.test/disk2:/mnt/path2:ro", "/mnt/path3:/mnt/path3",
		"/mnt/host:/mnt/path4", "/var/lib/kubelet/podID/volumes/empty/disk5:/mnt/path5"}

	if len(volumes) != len(expectedVolumes) {
		t.Errorf("Unexpected volumes. Expected %#v got %#v.  Container was: %#v", expectedVolumes, volumes, container)
	}
	for _, expectedVolume := range expectedVolumes {
		if _, ok := volumes[expectedVolume]; !ok {
			t.Errorf("Volumes map is missing key: %s. %#v", expectedVolume, volumes)
		}
	}
	if len(binds) != len(expectedBinds) {
		t.Errorf("Unexpected binds: Expected %#v got %#v.  Container was: %#v", expectedBinds, binds, container)
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
	successCaseAll := []Pod{
		{Manifest: api.ContainerManifest{Containers: []api.Container{{Ports: []api.Port{{HostPort: 80}}}}}},
		{Manifest: api.ContainerManifest{Containers: []api.Container{{Ports: []api.Port{{HostPort: 81}}}}}},
		{Manifest: api.ContainerManifest{Containers: []api.Container{{Ports: []api.Port{{HostPort: 82}}}}}},
	}
	successCaseNew := Pod{
		Manifest: api.ContainerManifest{Containers: []api.Container{{Ports: []api.Port{{HostPort: 83}}}}},
	}
	expected := append(successCaseAll, successCaseNew)
	if actual := filterHostPortConflicts(expected); !reflect.DeepEqual(actual, expected) {
		t.Errorf("Expected %#v, Got %#v", expected, actual)
	}

	failureCaseAll := []Pod{
		{Manifest: api.ContainerManifest{Containers: []api.Container{{Ports: []api.Port{{HostPort: 80}}}}}},
		{Manifest: api.ContainerManifest{Containers: []api.Container{{Ports: []api.Port{{HostPort: 81}}}}}},
		{Manifest: api.ContainerManifest{Containers: []api.Container{{Ports: []api.Port{{HostPort: 82}}}}}},
	}
	failureCaseNew := Pod{
		Manifest: api.ContainerManifest{Containers: []api.Container{{Ports: []api.Port{{HostPort: 81}}}}},
	}
	if actual := filterHostPortConflicts(append(failureCaseAll, failureCaseNew)); !reflect.DeepEqual(failureCaseAll, actual) {
		t.Errorf("Expected %#v, Got %#v", expected, actual)
	}
}

type mockCadvisorClient struct {
	mock.Mock
}

// ContainerInfo is a mock implementation of CadvisorInterface.ContainerInfo.
func (c *mockCadvisorClient) ContainerInfo(name string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	args := c.Called(name, req)
	return args.Get(0).(*info.ContainerInfo), args.Error(1)
}

// MachineInfo is a mock implementation of CadvisorInterface.MachineInfo.
func (c *mockCadvisorClient) MachineInfo() (*info.MachineInfo, error) {
	args := c.Called()
	return args.Get(0).(*info.MachineInfo), args.Error(1)
}

func areSamePercentiles(
	cadvisorPercentiles []info.Percentile,
	kubePercentiles []info.Percentile,
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

func TestGetContainerInfo(t *testing.T) {
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
	req := &info.ContainerInfoRequest{}
	cadvisorReq := getCadvisorContainerInfoRequest(req)
	mockCadvisor.On("ContainerInfo", containerPath, cadvisorReq).Return(containerInfo, nil)

	kubelet, _, fakeDocker := makeTestKubelet(t)
	kubelet.cadvisorClient = mockCadvisor
	fakeDocker.containerList = []docker.APIContainers{
		{
			ID: containerID,
			// pod id: qux
			// container id: foo
			Names: []string{"/k8s--foo--qux--1234"},
		},
	}

	stats, err := kubelet.GetContainerInfo("qux", "foo", req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if stats == nil {
		t.Fatalf("stats should not be nil")
	}
	if stats.StatsPercentiles.MaxMemoryUsage != containerInfo.StatsPercentiles.MaxMemoryUsage {
		t.Errorf("wrong max memory usage")
	}
	areSamePercentiles(containerInfo.StatsPercentiles.CpuUsagePercentiles, stats.StatsPercentiles.CpuUsagePercentiles, t)
	areSamePercentiles(containerInfo.StatsPercentiles.MemoryUsagePercentiles, stats.StatsPercentiles.MemoryUsagePercentiles, t)
	mockCadvisor.AssertExpectations(t)
}

func TestGetRooInfo(t *testing.T) {
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
	req := &info.ContainerInfoRequest{}
	cadvisorReq := getCadvisorContainerInfoRequest(req)
	mockCadvisor.On("ContainerInfo", containerPath, cadvisorReq).Return(containerInfo, nil)

	kubelet := Kubelet{
		dockerClient:   &fakeDocker,
		dockerPuller:   &FakeDockerPuller{},
		cadvisorClient: mockCadvisor,
		podWorkers:     newPodWorkers(),
	}

	// If the container name is an empty string, then it means the root container.
	stats, err := kubelet.GetRootInfo(req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if stats.StatsPercentiles.MaxMemoryUsage != containerInfo.StatsPercentiles.MaxMemoryUsage {
		t.Errorf("wrong max memory usage")
	}
	areSamePercentiles(containerInfo.StatsPercentiles.CpuUsagePercentiles, stats.StatsPercentiles.CpuUsagePercentiles, t)
	areSamePercentiles(containerInfo.StatsPercentiles.MemoryUsagePercentiles, stats.StatsPercentiles.MemoryUsagePercentiles, t)
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoWithoutCadvisor(t *testing.T) {
	kubelet, _, fakeDocker := makeTestKubelet(t)
	fakeDocker.containerList = []docker.APIContainers{
		{
			ID: "foobar",
			// pod id: qux
			// container id: foo
			Names: []string{"/k8s--foo--qux--1234"},
		},
	}

	stats, _ := kubelet.GetContainerInfo("qux", "foo", nil)
	// When there's no cAdvisor, the stats should be either nil or empty
	if stats == nil {
		return
	}
	if stats.StatsPercentiles.MaxMemoryUsage != 0 {
		t.Errorf("MaxMemoryUsage is %v even if there's no cadvisor", stats.StatsPercentiles.MaxMemoryUsage)
	}
	if len(stats.StatsPercentiles.CpuUsagePercentiles) > 0 {
		t.Errorf("CPU usage percentiles is not empty (%+v) even if there's no cadvisor", stats.StatsPercentiles.CpuUsagePercentiles)
	}
	if len(stats.StatsPercentiles.MemoryUsagePercentiles) > 0 {
		t.Errorf("Memory usage percentiles is not empty (%+v) even if there's no cadvisor", stats.StatsPercentiles.MemoryUsagePercentiles)
	}
}

func TestGetContainerInfoWhenCadvisorFailed(t *testing.T) {
	containerID := "ab2cdf"
	containerPath := fmt.Sprintf("/docker/%v", containerID)

	containerInfo := &info.ContainerInfo{}
	mockCadvisor := &mockCadvisorClient{}
	req := &info.ContainerInfoRequest{}
	cadvisorReq := getCadvisorContainerInfoRequest(req)
	expectedErr := fmt.Errorf("some error")
	mockCadvisor.On("ContainerInfo", containerPath, cadvisorReq).Return(containerInfo, expectedErr)

	kubelet, _, fakeDocker := makeTestKubelet(t)
	kubelet.cadvisorClient = mockCadvisor
	fakeDocker.containerList = []docker.APIContainers{
		{
			ID: containerID,
			// pod id: qux
			// container id: foo
			Names: []string{"/k8s--foo--qux--1234"},
		},
	}

	stats, err := kubelet.GetContainerInfo("qux", "foo", req)
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

func TestGetContainerInfoOnNonExistContainer(t *testing.T) {
	mockCadvisor := &mockCadvisorClient{}

	kubelet, _, fakeDocker := makeTestKubelet(t)
	kubelet.cadvisorClient = mockCadvisor
	fakeDocker.containerList = []docker.APIContainers{}

	stats, _ := kubelet.GetContainerInfo("qux", "foo", nil)
	if stats != nil {
		t.Errorf("non-nil stats on non exist container")
	}
	mockCadvisor.AssertExpectations(t)
}

var parseImageNameTests = []struct {
	imageName string
	name      string
	tag       string
}{
	{"ubuntu", "ubuntu", ""},
	{"ubuntu:2342", "ubuntu", "2342"},
	{"ubuntu:latest", "ubuntu", "latest"},
	{"foo/bar:445566", "foo/bar", "445566"},
	{"registry.example.com:5000/foobar", "registry.example.com:5000/foobar", ""},
	{"registry.example.com:5000/foobar:5342", "registry.example.com:5000/foobar", "5342"},
	{"registry.example.com:5000/foobar:latest", "registry.example.com:5000/foobar", "latest"},
}

func TestParseImageName(t *testing.T) {
	for _, tt := range parseImageNameTests {
		name, tag := parseImageName(tt.imageName)
		if name != tt.name || tag != tt.tag {
			t.Errorf("Expected name/tag: %s/%s, got %s/%s", tt.name, tt.tag, name, tag)
		}
	}
}
