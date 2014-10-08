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
	"fmt"
	"net/http"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/fsouza/go-dockerclient"
	"github.com/google/cadvisor/info"
	"github.com/stretchr/testify/mock"
)

func newTestKubelet(t *testing.T) (*Kubelet, *tools.FakeEtcdClient, *dockertools.FakeDockerClient) {
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	fakeDocker := &dockertools.FakeDockerClient{}

	kubelet := &Kubelet{}
	kubelet.dockerClient = fakeDocker
	kubelet.dockerPuller = &dockertools.FakeDockerPuller{}
	kubelet.etcdClient = fakeEtcdClient
	kubelet.rootDirectory = "/tmp/kubelet"
	kubelet.podWorkers = newPodWorkers()
	return kubelet, fakeEtcdClient, fakeDocker
}

func verifyCalls(t *testing.T, fakeDocker *dockertools.FakeDockerClient, calls []string) {
	err := fakeDocker.AssertCalls(calls)
	if err != nil {
		t.Error(err)
	}
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

func verifyBoolean(t *testing.T, expected, value bool) {
	if expected != value {
		t.Errorf("Unexpected boolean.  Expected %t.  Found %t", expected, value)
	}
}

func TestKillContainerWithError(t *testing.T) {
	fakeDocker := &dockertools.FakeDockerClient{
		Err: fmt.Errorf("sample error"),
		ContainerList: []docker.APIContainers{
			{
				ID:    "1234",
				Names: []string{"/k8s_foo_qux_1234"},
			},
			{
				ID:    "5678",
				Names: []string{"/k8s_bar_qux_5678"},
			},
		},
	}
	kubelet, _, _ := newTestKubelet(t)
	kubelet.dockerClient = fakeDocker
	err := kubelet.killContainer(&fakeDocker.ContainerList[0])
	if err == nil {
		t.Errorf("expected error, found nil")
	}
	verifyCalls(t, fakeDocker, []string{"stop"})
}

func TestKillContainer(t *testing.T) {
	kubelet, _, fakeDocker := newTestKubelet(t)
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    "1234",
			Names: []string{"/k8s_foo_qux_1234"},
		},
		{
			ID:    "5678",
			Names: []string{"/k8s_bar_qux_5678"},
		},
	}
	fakeDocker.Container = &docker.Container{
		ID: "foobar",
	}

	err := kubelet.killContainer(&fakeDocker.ContainerList[0])
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
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
	kubelet, _, fakeDocker := newTestKubelet(t)
	container := api.Container{Name: "bar"}
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// format is k8s_<container-id>_<pod-fullname>
			Names: []string{"/k8s_bar." + strconv.FormatUint(dockertools.HashContainer(&container), 16) + "_foo.test"},
			ID:    "1234",
		},
		{
			// network container
			Names: []string{"/k8s_net_foo.test_"},
			ID:    "9876",
		},
	}
	err := kubelet.SyncPods([]Pod{
		{
			Name:      "foo",
			Namespace: "test",
			Manifest: api.ContainerManifest{
				ID: "foo",
				Containers: []api.Container{
					container,
				},
			},
		},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{"list", "list"})
}

// drainWorkers waits until all workers are done.  Should only used for testing.
func (kl *Kubelet) drainWorkers() {
	for {
		kl.podWorkers.lock.Lock()
		length := len(kl.podWorkers.workers)
		kl.podWorkers.lock.Unlock()
		if length == 0 {
			return
		}
		time.Sleep(time.Millisecond * 100)
	}
}

func matchString(t *testing.T, pattern, str string) bool {
	match, err := regexp.MatchString(pattern, str)
	if err != nil {
		t.Logf("unexpected error: %v", err)
	}
	return match
}

func TestSyncPodsCreatesNetAndContainer(t *testing.T) {
	kubelet, _, fakeDocker := newTestKubelet(t)
	kubelet.networkContainerImage = "custom_image_name"
	fakeDocker.ContainerList = []docker.APIContainers{}
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
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	kubelet.drainWorkers()

	verifyCalls(t, fakeDocker, []string{
		"list", "list", "create", "start", "list", "inspect_container", "list", "create", "start"})

	fakeDocker.Lock()

	found := false
	for _, c := range fakeDocker.ContainerList {
		if c.Image == "custom_image_name" && strings.HasPrefix(c.Names[0], "/k8s_net") {
			found = true
		}
	}
	if !found {
		t.Errorf("Custom net container not found: %v", fakeDocker.ContainerList)
	}

	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "k8s_net\\.[a-f0-9]+_foo.test_", fakeDocker.Created[0]) ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo.test_", fakeDocker.Created[1]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
}

func TestSyncPodsCreatesNetAndContainerPullsImage(t *testing.T) {
	kubelet, _, fakeDocker := newTestKubelet(t)
	puller := kubelet.dockerPuller.(*dockertools.FakeDockerPuller)
	puller.HasImages = []string{}
	kubelet.networkContainerImage = "custom_image_name"
	fakeDocker.ContainerList = []docker.APIContainers{}
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
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	kubelet.drainWorkers()

	verifyCalls(t, fakeDocker, []string{
		"list", "list", "create", "start", "list", "inspect_container", "list", "create", "start"})

	fakeDocker.Lock()

	if !reflect.DeepEqual(puller.ImagesPulled, []string{"custom_image_name", ""}) {
		t.Errorf("Unexpected pulled containers: %v", puller.ImagesPulled)
	}

	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "k8s_net\\.[a-f0-9]+_foo.test_", fakeDocker.Created[0]) ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo.test_", fakeDocker.Created[1]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
}

func TestSyncPodsWithNetCreatesContainer(t *testing.T) {
	kubelet, _, fakeDocker := newTestKubelet(t)
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// network container
			Names: []string{"/k8s_net_foo.test_"},
			ID:    "9876",
		},
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
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	kubelet.drainWorkers()

	verifyCalls(t, fakeDocker, []string{
		"list", "list", "list", "inspect_container", "list", "create", "start"})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 1 ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo.test_", fakeDocker.Created[0]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
}

func TestSyncPodsWithNetCreatesContainerCallsHandler(t *testing.T) {
	kubelet, _, fakeDocker := newTestKubelet(t)
	fakeHttp := fakeHTTP{}
	kubelet.httpClient = &fakeHttp
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// network container
			Names: []string{"/k8s_net_foo.test_"},
			ID:    "9876",
		},
	}
	err := kubelet.SyncPods([]Pod{
		{
			Name:      "foo",
			Namespace: "test",
			Manifest: api.ContainerManifest{
				ID: "foo",
				Containers: []api.Container{
					{
						Name: "bar",
						Lifecycle: &api.Lifecycle{
							PostStart: &api.Handler{
								HTTPGet: &api.HTTPGetAction{
									Host: "foo",
									Port: util.IntOrString{IntVal: 8080, Kind: util.IntstrInt},
									Path: "bar",
								},
							},
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	kubelet.drainWorkers()

	verifyCalls(t, fakeDocker, []string{
		"list", "list", "list", "inspect_container", "list", "create", "start"})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 1 ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo.test_", fakeDocker.Created[0]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
	if fakeHttp.url != "http://foo:8080/bar" {
		t.Errorf("Unexpected handler: %s", fakeHttp.url)
	}
}

func TestSyncPodsDeletesWithNoNetContainer(t *testing.T) {
	kubelet, _, fakeDocker := newTestKubelet(t)
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// format is k8s_<container-id>_<pod-fullname>
			Names: []string{"/k8s_bar_foo.test"},
			ID:    "1234",
		},
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
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	kubelet.drainWorkers()

	verifyCalls(t, fakeDocker, []string{
		"list", "list", "stop", "create", "start", "list", "list", "inspect_container", "list", "create", "start"})

	// A map iteration is used to delete containers, so must not depend on
	// order here.
	expectedToStop := map[string]bool{
		"1234": true,
	}
	fakeDocker.Lock()
	if len(fakeDocker.Stopped) != 1 || !expectedToStop[fakeDocker.Stopped[0]] {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
	fakeDocker.Unlock()
}

func TestSyncPodsDeletes(t *testing.T) {
	kubelet, _, fakeDocker := newTestKubelet(t)
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_foo_bar.test"},
			ID:    "1234",
		},
		{
			// network container
			Names: []string{"/k8s_net_foo.test_"},
			ID:    "9876",
		},
		{
			Names: []string{"foo"},
			ID:    "4567",
		},
	}
	err := kubelet.SyncPods([]Pod{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{"list", "list", "stop", "stop"})

	// A map iteration is used to delete containers, so must not depend on
	// order here.
	expectedToStop := map[string]bool{
		"1234": true,
		"9876": true,
	}
	if len(fakeDocker.Stopped) != 2 ||
		!expectedToStop[fakeDocker.Stopped[0]] ||
		!expectedToStop[fakeDocker.Stopped[1]] {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
}

func TestSyncPodDeletesDuplicate(t *testing.T) {
	kubelet, _, fakeDocker := newTestKubelet(t)
	dockerContainers := dockertools.DockerContainers{
		"1234": &docker.APIContainers{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_foo_bar.test_1"},
			ID:    "1234",
		},
		"9876": &docker.APIContainers{
			// network container
			Names: []string{"/k8s_net_bar.test_"},
			ID:    "9876",
		},
		"4567": &docker.APIContainers{
			// Duplicate for the same container.
			Names: []string{"/k8s_foo_bar.test_2"},
			ID:    "4567",
		},
		"2304": &docker.APIContainers{
			// Container for another pod, untouched.
			Names: []string{"/k8s_baz_fiz.test_6"},
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
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{"list", "stop"})

	// Expect one of the duplicates to be killed.
	if len(fakeDocker.Stopped) != 1 || (len(fakeDocker.Stopped) != 0 && fakeDocker.Stopped[0] != "1234" && fakeDocker.Stopped[0] != "4567") {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
}

type FalseHealthChecker struct{}

func (f *FalseHealthChecker) HealthCheck(podFullName, podUUID string, state api.PodState, container api.Container) (health.Status, error) {
	return health.Unhealthy, nil
}

func (f *FalseHealthChecker) CanCheck(probe *api.LivenessProbe) bool {
	return true
}

func TestSyncPodBadHash(t *testing.T) {
	kubelet, _, fakeDocker := newTestKubelet(t)
	kubelet.healthChecker = &FalseHealthChecker{}
	dockerContainers := dockertools.DockerContainers{
		"1234": &docker.APIContainers{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_bar.1234_foo.test"},
			ID:    "1234",
		},
		"9876": &docker.APIContainers{
			// network container
			Names: []string{"/k8s_net_foo.test_"},
			ID:    "9876",
		},
	}
	err := kubelet.syncPod(&Pod{
		Name:      "foo",
		Namespace: "test",
		Manifest: api.ContainerManifest{
			ID: "foo",
			Containers: []api.Container{
				{Name: "bar"},
			},
		},
	}, dockerContainers)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{"list", "stop", "list", "create", "start"})

	// A map interation is used to delete containers, so must not depend on
	// order here.
	expectedToStop := map[string]bool{
		"1234": true,
	}
	if len(fakeDocker.Stopped) != 1 ||
		!expectedToStop[fakeDocker.Stopped[0]] {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
}

func TestSyncPodUnhealthy(t *testing.T) {
	kubelet, _, fakeDocker := newTestKubelet(t)
	kubelet.healthChecker = &FalseHealthChecker{}
	dockerContainers := dockertools.DockerContainers{
		"1234": &docker.APIContainers{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_bar_foo.test"},
			ID:    "1234",
		},
		"9876": &docker.APIContainers{
			// network container
			Names: []string{"/k8s_net_foo.test_"},
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
					},
				},
			},
		},
	}, dockerContainers)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{"list", "stop", "list", "create", "start"})

	// A map interation is used to delete containers, so must not depend on
	// order here.
	expectedToStop := map[string]bool{
		"1234": true,
	}
	if len(fakeDocker.Stopped) != 1 ||
		!expectedToStop[fakeDocker.Stopped[0]] {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
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
	kubelet, _, _ := newTestKubelet(t)
	manifest := api.ContainerManifest{
		Volumes: []api.Volume{
			{
				Name: "host-dir",
				Source: &api.VolumeSource{
					HostDir: &api.HostDir{"/dir/path"},
				},
			},
		},
	}
	podVolumes, _ := kubelet.mountExternalVolumes(&manifest)
	expectedPodVolumes := make(volumeMap)
	expectedPodVolumes["host-dir"] = &volume.HostDir{"/dir/path"}
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
				MountPath: "/mnt/path3",
				Name:      "disk",
				ReadOnly:  true,
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

	podVolumes := volumeMap{
		"disk":  &volume.HostDir{"/mnt/disk"},
		"disk4": &volume.HostDir{"/mnt/host"},
		"disk5": &volume.EmptyDir{"disk5", "podID", "/var/lib/kubelet"},
	}

	binds := makeBinds(&pod, &container, podVolumes)

	expectedBinds := []string{
		"/mnt/disk:/mnt/path",
		"/mnt/disk:/mnt/path3:ro",
		"/mnt/host:/mnt/path4",
		"/var/lib/kubelet/podID/volumes/empty/disk5:/mnt/path5",
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

func TestGetContainerInfo(t *testing.T) {
	containerID := "ab2cdf"
	containerPath := fmt.Sprintf("/docker/%v", containerID)
	containerInfo := &info.ContainerInfo{
		ContainerReference: info.ContainerReference{
			Name: containerPath,
		},
	}

	mockCadvisor := &mockCadvisorClient{}
	req := &info.ContainerInfoRequest{}
	cadvisorReq := getCadvisorContainerInfoRequest(req)
	mockCadvisor.On("ContainerInfo", containerPath, cadvisorReq).Return(containerInfo, nil)

	kubelet, _, fakeDocker := newTestKubelet(t)
	kubelet.cadvisorClient = mockCadvisor
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID: containerID,
			// pod id: qux
			// container id: foo
			Names: []string{"/k8s_foo_qux_1234"},
		},
	}

	stats, err := kubelet.GetContainerInfo("qux", "", "foo", req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if stats == nil {
		t.Fatalf("stats should not be nil")
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetRootInfo(t *testing.T) {
	containerPath := "/"
	containerInfo := &info.ContainerInfo{
		ContainerReference: info.ContainerReference{
			Name: containerPath,
		},
	}
	fakeDocker := dockertools.FakeDockerClient{}

	mockCadvisor := &mockCadvisorClient{}
	req := &info.ContainerInfoRequest{}
	cadvisorReq := getCadvisorContainerInfoRequest(req)
	mockCadvisor.On("ContainerInfo", containerPath, cadvisorReq).Return(containerInfo, nil)

	kubelet := Kubelet{
		dockerClient:   &fakeDocker,
		dockerPuller:   &dockertools.FakeDockerPuller{},
		cadvisorClient: mockCadvisor,
		podWorkers:     newPodWorkers(),
	}

	// If the container name is an empty string, then it means the root container.
	_, err := kubelet.GetRootInfo(req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoWithoutCadvisor(t *testing.T) {
	kubelet, _, fakeDocker := newTestKubelet(t)
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID: "foobar",
			// pod id: qux
			// container id: foo
			Names: []string{"/k8s_foo_qux_uuid_1234"},
		},
	}

	stats, _ := kubelet.GetContainerInfo("qux", "uuid", "foo", nil)
	// When there's no cAdvisor, the stats should be either nil or empty
	if stats == nil {
		return
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

	kubelet, _, fakeDocker := newTestKubelet(t)
	kubelet.cadvisorClient = mockCadvisor
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID: containerID,
			// pod id: qux
			// container id: foo
			Names: []string{"/k8s_foo_qux_uuid_1234"},
		},
	}

	stats, err := kubelet.GetContainerInfo("qux", "uuid", "foo", req)
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

	kubelet, _, fakeDocker := newTestKubelet(t)
	kubelet.cadvisorClient = mockCadvisor
	fakeDocker.ContainerList = []docker.APIContainers{}

	stats, _ := kubelet.GetContainerInfo("qux", "", "foo", nil)
	if stats != nil {
		t.Errorf("non-nil stats on non exist container")
	}
	mockCadvisor.AssertExpectations(t)
}

type fakeContainerCommandRunner struct {
	Cmd []string
	ID  string
	E   error
}

func (f *fakeContainerCommandRunner) RunInContainer(id string, cmd []string) ([]byte, error) {
	f.Cmd = cmd
	f.ID = id
	return []byte{}, f.E
}

func TestRunInContainerNoSuchPod(t *testing.T) {
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet, _, fakeDocker := newTestKubelet(t)
	fakeDocker.ContainerList = []docker.APIContainers{}
	kubelet.runner = &fakeCommandRunner

	podName := "podFoo"
	podNamespace := "etcd"
	containerName := "containerFoo"
	output, err := kubelet.RunInContainer(
		GetPodFullName(&Pod{Name: podName, Namespace: podNamespace}),
		"",
		containerName,
		[]string{"ls"})
	if output != nil {
		t.Errorf("unexpected non-nil command: %v", output)
	}
	if err == nil {
		t.Error("unexpected non-error")
	}
}

func TestRunInContainer(t *testing.T) {
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet, _, fakeDocker := newTestKubelet(t)
	kubelet.runner = &fakeCommandRunner

	containerID := "abc1234"
	podName := "podFoo"
	podNamespace := "etcd"
	containerName := "containerFoo"

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    containerID,
			Names: []string{"/k8s_" + containerName + "_" + podName + "." + podNamespace + "_1234"},
		},
	}

	cmd := []string{"ls"}
	_, err := kubelet.RunInContainer(
		GetPodFullName(&Pod{Name: podName, Namespace: podNamespace}),
		"",
		containerName,
		cmd)
	if fakeCommandRunner.ID != containerID {
		t.Errorf("unexected ID: %s", fakeCommandRunner.ID)
	}
	if !reflect.DeepEqual(fakeCommandRunner.Cmd, cmd) {
		t.Errorf("unexpected commnd: %s", fakeCommandRunner.Cmd)
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRunHandlerExec(t *testing.T) {
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet, _, fakeDocker := newTestKubelet(t)
	kubelet.runner = &fakeCommandRunner

	containerID := "abc1234"
	podName := "podFoo"
	podNamespace := "etcd"
	containerName := "containerFoo"

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    containerID,
			Names: []string{"/k8s_" + containerName + "_" + podName + "." + podNamespace + "_1234"},
		},
	}

	container := api.Container{
		Name: containerName,
		Lifecycle: &api.Lifecycle{
			PostStart: &api.Handler{
				Exec: &api.ExecAction{
					Command: []string{"ls", "-a"},
				},
			},
		},
	}
	err := kubelet.runHandler(podName+"."+podNamespace, "", &container, container.Lifecycle.PostStart)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if fakeCommandRunner.ID != containerID ||
		!reflect.DeepEqual(container.Lifecycle.PostStart.Exec.Command, fakeCommandRunner.Cmd) {
		t.Errorf("unexpected commands: %v", fakeCommandRunner)
	}
}

type fakeHTTP struct {
	url string
	err error
}

func (f *fakeHTTP) Get(url string) (*http.Response, error) {
	f.url = url
	return nil, f.err
}

func TestRunHandlerHttp(t *testing.T) {
	fakeHttp := fakeHTTP{}

	kubelet, _, _ := newTestKubelet(t)
	kubelet.httpClient = &fakeHttp

	podName := "podFoo"
	podNamespace := "etcd"
	containerName := "containerFoo"

	container := api.Container{
		Name: containerName,
		Lifecycle: &api.Lifecycle{
			PostStart: &api.Handler{
				HTTPGet: &api.HTTPGetAction{
					Host: "foo",
					Port: util.IntOrString{IntVal: 8080, Kind: util.IntstrInt},
					Path: "bar",
				},
			},
		},
	}
	err := kubelet.runHandler(podName+"."+podNamespace, "", &container, container.Lifecycle.PostStart)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if fakeHttp.url != "http://foo:8080/bar" {
		t.Errorf("unexpected url: %s", fakeHttp.url)
	}
}

func TestNewHandler(t *testing.T) {
	kubelet, _, _ := newTestKubelet(t)
	handler := &api.Handler{
		HTTPGet: &api.HTTPGetAction{
			Host: "foo",
			Port: util.IntOrString{IntVal: 8080, Kind: util.IntstrInt},
			Path: "bar",
		},
	}
	actionHandler := kubelet.newActionHandler(handler)
	if actionHandler == nil {
		t.Error("unexpected nil action handler.")
	}

	handler = &api.Handler{
		Exec: &api.ExecAction{
			Command: []string{"ls", "-l"},
		},
	}
	actionHandler = kubelet.newActionHandler(handler)
	if actionHandler == nil {
		t.Error("unexpected nil action handler.")
	}

	handler = &api.Handler{}
	actionHandler = kubelet.newActionHandler(handler)
	if actionHandler != nil {
		t.Errorf("unexpected non-nil action handler: %v", actionHandler)
	}
}

func TestSyncPodEventHandlerFails(t *testing.T) {
	kubelet, _, fakeDocker := newTestKubelet(t)
	kubelet.httpClient = &fakeHTTP{
		err: fmt.Errorf("test error"),
	}
	dockerContainers := dockertools.DockerContainers{
		"9876": &docker.APIContainers{
			// network container
			Names: []string{"/k8s_net_foo.test_"},
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
					Lifecycle: &api.Lifecycle{
						PostStart: &api.Handler{
							HTTPGet: &api.HTTPGetAction{
								Host: "does.no.exist",
								Port: util.IntOrString{IntVal: 8080, Kind: util.IntstrInt},
								Path: "bar",
							},
						},
					},
				},
			},
		},
	}, dockerContainers)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{"list", "list", "create", "start", "stop"})

	if len(fakeDocker.Stopped) != 1 {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
}
