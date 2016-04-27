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

package dockertools

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	dockertypes "github.com/docker/engine-api/types"
	dockercontainer "github.com/docker/engine-api/types/container"
	dockerstrslice "github.com/docker/engine-api/types/strslice"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/network"
	nettest "k8s.io/kubernetes/pkg/kubelet/network/testing"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	uexec "k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/sets"
)

type fakeHTTP struct {
	url string
	err error
}

func (f *fakeHTTP) Get(url string) (*http.Response, error) {
	f.url = url
	return nil, f.err
}

// fakeRuntimeHelper implementes kubecontainer.RuntimeHelper inter
// faces for testing purposes.
type fakeRuntimeHelper struct{}

var _ kubecontainer.RuntimeHelper = &fakeRuntimeHelper{}

var testPodContainerDir string

func (f *fakeRuntimeHelper) GenerateRunContainerOptions(pod *api.Pod, container *api.Container, podIP string) (*kubecontainer.RunContainerOptions, error) {
	var opts kubecontainer.RunContainerOptions
	var err error
	if len(container.TerminationMessagePath) != 0 {
		testPodContainerDir, err = ioutil.TempDir("", "fooPodContainerDir")
		if err != nil {
			return nil, err
		}
		opts.PodContainerDir = testPodContainerDir
	}
	return &opts, nil
}

func (f *fakeRuntimeHelper) GetClusterDNS(pod *api.Pod) ([]string, []string, error) {
	return nil, nil, fmt.Errorf("not implemented")
}

// This is not used by docker runtime.
func (f *fakeRuntimeHelper) GeneratePodHostNameAndDomain(pod *api.Pod) (string, string, error) {
	return "", "", nil
}

func (f *fakeRuntimeHelper) GetPodDir(types.UID) string {
	return ""
}

func createTestDockerManager(fakeHTTPClient *fakeHTTP, fakeDocker *FakeDockerClient) (*DockerManager, *FakeDockerClient) {
	if fakeHTTPClient == nil {
		fakeHTTPClient = &fakeHTTP{}
	}
	if fakeDocker == nil {
		fakeDocker = NewFakeDockerClient()
	}
	fakeRecorder := &record.FakeRecorder{}
	containerRefManager := kubecontainer.NewRefManager()
	networkPlugin, _ := network.InitNetworkPlugin([]network.NetworkPlugin{}, "", nettest.NewFakeHost(nil))
	dockerManager := NewFakeDockerManager(
		fakeDocker,
		fakeRecorder,
		proberesults.NewManager(),
		containerRefManager,
		&cadvisorapi.MachineInfo{},
		options.GetDefaultPodInfraContainerImage(),
		0, 0, "",
		containertest.FakeOS{},
		networkPlugin,
		&fakeRuntimeHelper{},
		fakeHTTPClient,
		flowcontrol.NewBackOff(time.Second, 300*time.Second))

	return dockerManager, fakeDocker
}

func newTestDockerManagerWithHTTPClient(fakeHTTPClient *fakeHTTP) (*DockerManager, *FakeDockerClient) {
	return createTestDockerManager(fakeHTTPClient, nil)
}

func newTestDockerManagerWithVersion(version, apiVersion string) (*DockerManager, *FakeDockerClient) {
	fakeDocker := NewFakeDockerClientWithVersion(version, apiVersion)
	return createTestDockerManager(nil, fakeDocker)
}

func newTestDockerManager() (*DockerManager, *FakeDockerClient) {
	return createTestDockerManager(nil, nil)
}

func matchString(t *testing.T, pattern, str string) bool {
	match, err := regexp.MatchString(pattern, str)
	if err != nil {
		t.Logf("unexpected error: %v", err)
	}
	return match
}

func TestSetEntrypointAndCommand(t *testing.T) {
	cases := []struct {
		name      string
		container *api.Container
		envs      []kubecontainer.EnvVar
		expected  *dockertypes.ContainerCreateConfig
	}{
		{
			name:      "none",
			container: &api.Container{},
			expected: &dockertypes.ContainerCreateConfig{
				Config: &dockercontainer.Config{},
			},
		},
		{
			name: "command",
			container: &api.Container{
				Command: []string{"foo", "bar"},
			},
			expected: &dockertypes.ContainerCreateConfig{
				Config: &dockercontainer.Config{
					Entrypoint: dockerstrslice.StrSlice([]string{"foo", "bar"}),
				},
			},
		},
		{
			name: "command expanded",
			container: &api.Container{
				Command: []string{"foo", "$(VAR_TEST)", "$(VAR_TEST2)"},
			},
			envs: []kubecontainer.EnvVar{
				{
					Name:  "VAR_TEST",
					Value: "zoo",
				},
				{
					Name:  "VAR_TEST2",
					Value: "boo",
				},
			},
			expected: &dockertypes.ContainerCreateConfig{
				Config: &dockercontainer.Config{
					Entrypoint: dockerstrslice.StrSlice([]string{"foo", "zoo", "boo"}),
				},
			},
		},
		{
			name: "args",
			container: &api.Container{
				Args: []string{"foo", "bar"},
			},
			expected: &dockertypes.ContainerCreateConfig{
				Config: &dockercontainer.Config{
					Cmd: []string{"foo", "bar"},
				},
			},
		},
		{
			name: "args expanded",
			container: &api.Container{
				Args: []string{"zap", "$(VAR_TEST)", "$(VAR_TEST2)"},
			},
			envs: []kubecontainer.EnvVar{
				{
					Name:  "VAR_TEST",
					Value: "hap",
				},
				{
					Name:  "VAR_TEST2",
					Value: "trap",
				},
			},
			expected: &dockertypes.ContainerCreateConfig{
				Config: &dockercontainer.Config{
					Cmd: dockerstrslice.StrSlice([]string{"zap", "hap", "trap"}),
				},
			},
		},
		{
			name: "both",
			container: &api.Container{
				Command: []string{"foo"},
				Args:    []string{"bar", "baz"},
			},
			expected: &dockertypes.ContainerCreateConfig{
				Config: &dockercontainer.Config{
					Entrypoint: dockerstrslice.StrSlice([]string{"foo"}),
					Cmd:        dockerstrslice.StrSlice([]string{"bar", "baz"}),
				},
			},
		},
		{
			name: "both expanded",
			container: &api.Container{
				Command: []string{"$(VAR_TEST2)--$(VAR_TEST)", "foo", "$(VAR_TEST3)"},
				Args:    []string{"foo", "$(VAR_TEST)", "$(VAR_TEST2)"},
			},
			envs: []kubecontainer.EnvVar{
				{
					Name:  "VAR_TEST",
					Value: "zoo",
				},
				{
					Name:  "VAR_TEST2",
					Value: "boo",
				},
				{
					Name:  "VAR_TEST3",
					Value: "roo",
				},
			},
			expected: &dockertypes.ContainerCreateConfig{
				Config: &dockercontainer.Config{
					Entrypoint: dockerstrslice.StrSlice([]string{"boo--zoo", "foo", "roo"}),
					Cmd:        dockerstrslice.StrSlice([]string{"foo", "zoo", "boo"}),
				},
			},
		},
	}

	for _, tc := range cases {
		opts := &kubecontainer.RunContainerOptions{
			Envs: tc.envs,
		}

		actualOpts := dockertypes.ContainerCreateConfig{
			Config: &dockercontainer.Config{},
		}
		setEntrypointAndCommand(tc.container, opts, actualOpts)

		if e, a := tc.expected.Config.Entrypoint, actualOpts.Config.Entrypoint; !api.Semantic.DeepEqual(e, a) {
			t.Errorf("%v: unexpected entrypoint: expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.expected.Config.Cmd, actualOpts.Config.Cmd; !api.Semantic.DeepEqual(e, a) {
			t.Errorf("%v: unexpected command: expected %v, got %v", tc.name, e, a)
		}
	}
}

// verifyPods returns true if the two pod slices are equal.
func verifyPods(a, b []*kubecontainer.Pod) bool {
	if len(a) != len(b) {
		return false
	}

	// Sort the containers within a pod.
	for i := range a {
		sort.Sort(containersByID(a[i].Containers))
	}
	for i := range b {
		sort.Sort(containersByID(b[i].Containers))
	}

	// Sort the pods by UID.
	sort.Sort(podsByID(a))
	sort.Sort(podsByID(b))

	return reflect.DeepEqual(a, b)
}

func TestGetPods(t *testing.T) {
	manager, fakeDocker := newTestDockerManager()
	dockerContainers := []*FakeContainer{
		{
			ID:   "1111",
			Name: "/k8s_foo_qux_new_1234_42",
		},
		{
			ID:   "2222",
			Name: "/k8s_bar_qux_new_1234_42",
		},
		{
			ID:   "3333",
			Name: "/k8s_bar_jlk_wen_5678_42",
		},
	}

	// Convert the docker containers. This does not affect the test coverage
	// because the conversion is tested separately in convert_test.go
	containers := make([]*kubecontainer.Container, len(dockerContainers))
	for i := range containers {
		c, err := toRuntimeContainer(&dockertypes.Container{
			ID:    dockerContainers[i].ID,
			Names: []string{dockerContainers[i].Name},
		})
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		containers[i] = c
	}

	expected := []*kubecontainer.Pod{
		{
			ID:         types.UID("1234"),
			Name:       "qux",
			Namespace:  "new",
			Containers: []*kubecontainer.Container{containers[0], containers[1]},
		},
		{
			ID:         types.UID("5678"),
			Name:       "jlk",
			Namespace:  "wen",
			Containers: []*kubecontainer.Container{containers[2]},
		},
	}

	fakeDocker.SetFakeRunningContainers(dockerContainers)
	actual, err := manager.GetPods(false)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if !verifyPods(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}
}

func TestListImages(t *testing.T) {
	manager, fakeDocker := newTestDockerManager()
	dockerImages := []dockertypes.Image{{ID: "1111"}, {ID: "2222"}, {ID: "3333"}}
	expected := sets.NewString([]string{"1111", "2222", "3333"}...)

	fakeDocker.Images = dockerImages
	actualImages, err := manager.ListImages()
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	actual := sets.NewString()
	for _, i := range actualImages {
		actual.Insert(i.ID)
	}
	// We can compare the two sets directly because util.StringSet.List()
	// returns a "sorted" list.
	if !reflect.DeepEqual(expected.List(), actual.List()) {
		t.Errorf("expected %#v, got %#v", expected.List(), actual.List())
	}
}

func TestKillContainerInPod(t *testing.T) {
	manager, fakeDocker := newTestDockerManager()

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "qux",
			Namespace: "new",
		},
		Spec: api.PodSpec{Containers: []api.Container{{Name: "foo"}, {Name: "bar"}}},
	}
	containers := []*FakeContainer{
		{
			ID:   "1111",
			Name: "/k8s_foo_qux_new_1234_42",
		},
		{
			ID:   "2222",
			Name: "/k8s_bar_qux_new_1234_42",
		},
	}
	containerToKill := containers[0]
	containerToSpare := containers[1]

	fakeDocker.SetFakeRunningContainers(containers)

	if err := manager.KillContainerInPod(kubecontainer.ContainerID{}, &pod.Spec.Containers[0], pod, "test kill container in pod.", nil); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	// Assert the container has been stopped.
	if err := fakeDocker.AssertStopped([]string{containerToKill.ID}); err != nil {
		t.Errorf("container was not stopped correctly: %v", err)
	}
	// Assert the container has been spared.
	if err := fakeDocker.AssertStopped([]string{containerToSpare.ID}); err == nil {
		t.Errorf("container unexpectedly stopped: %v", containerToSpare.ID)
	}
}

func TestKillContainerInPodWithPreStop(t *testing.T) {
	manager, fakeDocker := newTestDockerManager()
	fakeDocker.ExecInspect = &dockertypes.ContainerExecInspect{
		Running:  false,
		ExitCode: 0,
	}
	expectedCmd := []string{"foo.sh", "bar"}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "qux",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "foo",
					Lifecycle: &api.Lifecycle{
						PreStop: &api.Handler{
							Exec: &api.ExecAction{
								Command: expectedCmd,
							},
						},
					},
				},
				{Name: "bar"}}},
	}
	podString, err := runtime.Encode(testapi.Default.Codec(), pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	containers := []*FakeContainer{
		{
			ID:   "1111",
			Name: "/k8s_foo_qux_new_1234_42",
			Config: &dockercontainer.Config{
				Labels: map[string]string{
					kubernetesPodLabel:           string(podString),
					kubernetesContainerNameLabel: "foo",
				},
			},
		},
		{
			ID:   "2222",
			Name: "/k8s_bar_qux_new_1234_42",
		},
	}
	containerToKill := containers[0]
	fakeDocker.SetFakeRunningContainers(containers)

	if err := manager.KillContainerInPod(kubecontainer.ContainerID{}, &pod.Spec.Containers[0], pod, "test kill container with preStop.", nil); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	// Assert the container has been stopped.
	if err := fakeDocker.AssertStopped([]string{containerToKill.ID}); err != nil {
		t.Errorf("container was not stopped correctly: %v", err)
	}
	verifyCalls(t, fakeDocker, []string{"list", "create_exec", "start_exec", "stop"})
	if !reflect.DeepEqual(expectedCmd, fakeDocker.execCmd) {
		t.Errorf("expected: %v, got %v", expectedCmd, fakeDocker.execCmd)
	}
}

func TestKillContainerInPodWithError(t *testing.T) {
	manager, fakeDocker := newTestDockerManager()

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "qux",
			Namespace: "new",
		},
		Spec: api.PodSpec{Containers: []api.Container{{Name: "foo"}, {Name: "bar"}}},
	}
	containers := []*FakeContainer{
		{
			ID:   "1111",
			Name: "/k8s_foo_qux_new_1234_42",
		},
		{
			ID:   "2222",
			Name: "/k8s_bar_qux_new_1234_42",
		},
	}
	fakeDocker.SetFakeRunningContainers(containers)
	fakeDocker.InjectError("stop", fmt.Errorf("sample error"))

	if err := manager.KillContainerInPod(kubecontainer.ContainerID{}, &pod.Spec.Containers[0], pod, "test kill container with error.", nil); err == nil {
		t.Errorf("expected error, found nil")
	}
}

func TestIsAExitError(t *testing.T) {
	var err error
	err = &dockerExitError{nil}
	_, ok := err.(uexec.ExitError)
	if !ok {
		t.Error("couldn't cast dockerExitError to exec.ExitError")
	}
}

func generatePodInfraContainerHash(pod *api.Pod) uint64 {
	var ports []api.ContainerPort
	if pod.Spec.SecurityContext == nil || !pod.Spec.SecurityContext.HostNetwork {
		for _, container := range pod.Spec.Containers {
			ports = append(ports, container.Ports...)
		}
	}

	container := &api.Container{
		Name:            PodInfraContainerName,
		Image:           options.GetDefaultPodInfraContainerImage(),
		Ports:           ports,
		ImagePullPolicy: podInfraContainerImagePullPolicy,
	}
	return kubecontainer.HashContainer(container)
}

// runSyncPod is a helper function to retrieve the running pods from the fake
// docker client and runs SyncPod for the given pod.
func runSyncPod(t *testing.T, dm *DockerManager, fakeDocker *FakeDockerClient, pod *api.Pod, backOff *flowcontrol.Backoff, expectErr bool) kubecontainer.PodSyncResult {
	podStatus, err := dm.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fakeDocker.ClearCalls()
	if backOff == nil {
		backOff = flowcontrol.NewBackOff(time.Second, time.Minute)
	}
	// api.PodStatus is not used in SyncPod now, pass in an empty one.
	result := dm.SyncPod(pod, api.PodStatus{}, podStatus, []api.Secret{}, backOff)
	err = result.Error()
	if err != nil && !expectErr {
		t.Errorf("unexpected error: %v", err)
	} else if err == nil && expectErr {
		t.Errorf("expected error didn't occur")
	}
	return result
}

func TestSyncPodCreateNetAndContainer(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	dm.podInfraContainerImage = "pod_infra_image"

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "bar"},
			},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil, false)
	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})
	fakeDocker.Lock()

	found := false
	for _, c := range fakeDocker.RunningContainerList {
		if c.Image == "pod_infra_image" && strings.HasPrefix(c.Names[0], "/k8s_POD") {
			found = true
		}
	}
	if !found {
		t.Errorf("Custom pod infra container not found: %v", fakeDocker.RunningContainerList)
	}

	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
}

func TestSyncPodCreatesNetAndContainerPullsImage(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	dm.podInfraContainerImage = "pod_infra_image"
	puller := dm.dockerPuller.(*FakeDockerPuller)
	puller.HasImages = []string{}
	dm.podInfraContainerImage = "pod_infra_image"
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "bar", Image: "something", ImagePullPolicy: "IfNotPresent"},
			},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()

	if !reflect.DeepEqual(puller.ImagesPulled, []string{"pod_infra_image", "something"}) {
		t.Errorf("unexpected pulled containers: %v", puller.ImagesPulled)
	}

	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
}

func TestSyncPodWithPodInfraCreatesContainer(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "bar"},
			},
		},
	}

	fakeDocker.SetFakeRunningContainers([]*FakeContainer{{
		ID: "9876",
		// Pod infra container.
		Name: "/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0",
	}})
	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 1 ||
		!matchString(t, "/k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
}

func TestSyncPodDeletesWithNoPodInfraContainer(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo1",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "bar1"},
			},
		},
	}
	fakeDocker.SetFakeRunningContainers([]*FakeContainer{{
		ID:   "1234",
		Name: "/k8s_bar1_foo1_new_12345678_0",
	}})

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Kill the container since pod infra container is not running.
		"stop",
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

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

func TestSyncPodDeletesDuplicate(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "foo"},
			},
		},
	}

	fakeDocker.SetFakeRunningContainers([]*FakeContainer{
		{
			ID:   "1234",
			Name: "/k8s_foo_bar_new_12345678_1111",
		},
		{
			ID:   "9876",
			Name: "/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_bar_new_12345678_2222",
		},
		{
			ID:   "4567",
			Name: "/k8s_foo_bar_new_12345678_3333",
		}})

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Kill the duplicated container.
		"stop",
	})
	// Expect one of the duplicates to be killed.
	if len(fakeDocker.Stopped) != 1 || (fakeDocker.Stopped[0] != "1234" && fakeDocker.Stopped[0] != "4567") {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
}

func TestSyncPodBadHash(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "bar"},
			},
		},
	}

	fakeDocker.SetFakeRunningContainers([]*FakeContainer{
		{
			ID:   "1234",
			Name: "/k8s_bar.1234_foo_new_12345678_42",
		},
		{
			ID:   "9876",
			Name: "/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_42",
		}})
	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Kill and restart the bad hash container.
		"stop", "create", "start", "inspect_container",
	})

	if err := fakeDocker.AssertStopped([]string{"1234"}); err != nil {
		t.Errorf("%v", err)
	}
}

func TestSyncPodsUnhealthy(t *testing.T) {
	const (
		unhealthyContainerID = "1234"
		infraContainerID     = "9876"
	)
	dm, fakeDocker := newTestDockerManager()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{{Name: "unhealthy"}},
		},
	}

	fakeDocker.SetFakeRunningContainers([]*FakeContainer{
		{
			ID:   unhealthyContainerID,
			Name: "/k8s_unhealthy_foo_new_12345678_42",
		},
		{
			ID:   infraContainerID,
			Name: "/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_42",
		}})
	dm.livenessManager.Set(kubecontainer.DockerID(unhealthyContainerID).ContainerID(), proberesults.Failure, pod)

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Kill the unhealthy container.
		"stop",
		// Restart the unhealthy container.
		"create", "start", "inspect_container",
	})

	if err := fakeDocker.AssertStopped([]string{unhealthyContainerID}); err != nil {
		t.Errorf("%v", err)
	}
}

func TestSyncPodsDoesNothing(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	container := api.Container{Name: "bar"}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				container,
			},
		},
	}
	fakeDocker.SetFakeRunningContainers([]*FakeContainer{
		{
			ID:   "1234",
			Name: "/k8s_bar." + strconv.FormatUint(kubecontainer.HashContainer(&container), 16) + "_foo_new_12345678_0",
		},
		{
			ID:   "9876",
			Name: "/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0",
		}})

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{})
}

func TestSyncPodWithRestartPolicy(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	containers := []api.Container{
		{Name: "succeeded"},
		{Name: "failed"},
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: containers,
		},
	}
	dockerContainers := []*FakeContainer{
		{
			ID:        "9876",
			Name:      "/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0",
			StartedAt: time.Now(),
			Running:   true,
		},
		{
			ID:         "1234",
			Name:       "/k8s_succeeded." + strconv.FormatUint(kubecontainer.HashContainer(&containers[0]), 16) + "_foo_new_12345678_0",
			ExitCode:   0,
			StartedAt:  time.Now(),
			FinishedAt: time.Now(),
		},
		{
			ID:         "5678",
			Name:       "/k8s_failed." + strconv.FormatUint(kubecontainer.HashContainer(&containers[1]), 16) + "_foo_new_12345678_0",
			ExitCode:   42,
			StartedAt:  time.Now(),
			FinishedAt: time.Now(),
		}}

	tests := []struct {
		policy  api.RestartPolicy
		calls   []string
		created []string
		stopped []string
	}{
		{
			api.RestartPolicyAlways,
			[]string{
				// Restart both containers.
				"create", "start", "inspect_container", "create", "start", "inspect_container",
			},
			[]string{"succeeded", "failed"},
			[]string{},
		},
		{
			api.RestartPolicyOnFailure,
			[]string{
				// Restart the failed container.
				"create", "start", "inspect_container",
			},
			[]string{"failed"},
			[]string{},
		},
		{
			api.RestartPolicyNever,
			[]string{
				// Check the pod infra container.
				"inspect_container", "inspect_container",
				// Stop the last pod infra container.
				"stop",
			},
			[]string{},
			[]string{"9876"},
		},
	}

	for i, tt := range tests {
		fakeDocker.SetFakeContainers(dockerContainers)
		pod.Spec.RestartPolicy = tt.policy
		runSyncPod(t, dm, fakeDocker, pod, nil, false)
		// 'stop' is because the pod infra container is killed when no container is running.
		verifyCalls(t, fakeDocker, tt.calls)

		if err := fakeDocker.AssertCreated(tt.created); err != nil {
			t.Errorf("case [%d]: %v", i, err)
		}
		if err := fakeDocker.AssertStopped(tt.stopped); err != nil {
			t.Errorf("case [%d]: %v", i, err)
		}
	}
}

func TestSyncPodBackoff(t *testing.T) {
	var fakeClock = util.NewFakeClock(time.Now())
	startTime := fakeClock.Now()

	dm, fakeDocker := newTestDockerManager()
	containers := []api.Container{
		{Name: "good"},
		{Name: "bad"},
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "podfoo",
			Namespace: "nsnew",
		},
		Spec: api.PodSpec{
			Containers: containers,
		},
	}

	stableId := "k8s_bad." + strconv.FormatUint(kubecontainer.HashContainer(&containers[1]), 16) + "_podfoo_nsnew_12345678"
	dockerContainers := []*FakeContainer{
		{
			ID:        "9876",
			Name:      "/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_podfoo_nsnew_12345678_0",
			StartedAt: startTime,
			Running:   true,
		},
		{
			ID:        "1234",
			Name:      "/k8s_good." + strconv.FormatUint(kubecontainer.HashContainer(&containers[0]), 16) + "_podfoo_nsnew_12345678_0",
			StartedAt: startTime,
			Running:   true,
		},
		{
			ID:         "5678",
			Name:       "/k8s_bad." + strconv.FormatUint(kubecontainer.HashContainer(&containers[1]), 16) + "_podfoo_nsnew_12345678_0",
			ExitCode:   42,
			StartedAt:  startTime,
			FinishedAt: fakeClock.Now(),
		},
	}

	startCalls := []string{"create", "start", "inspect_container"}
	backOffCalls := []string{}
	startResult := &kubecontainer.SyncResult{Action: kubecontainer.StartContainer, Target: "bad", Error: nil, Message: ""}
	backoffResult := &kubecontainer.SyncResult{Action: kubecontainer.StartContainer, Target: "bad", Error: kubecontainer.ErrCrashLoopBackOff, Message: ""}
	tests := []struct {
		tick      int
		backoff   int
		killDelay int
		result    []string
		expectErr bool
	}{
		{1, 1, 1, startCalls, false},
		{2, 2, 2, startCalls, false},
		{3, 2, 3, backOffCalls, true},
		{4, 4, 4, startCalls, false},
		{5, 4, 5, backOffCalls, true},
		{6, 4, 6, backOffCalls, true},
		{7, 4, 7, backOffCalls, true},
		{8, 8, 129, startCalls, false},
		{130, 1, 0, startCalls, false},
	}

	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	backOff.Clock = fakeClock
	for _, c := range tests {
		fakeDocker.SetFakeContainers(dockerContainers)
		fakeClock.SetTime(startTime.Add(time.Duration(c.tick) * time.Second))

		result := runSyncPod(t, dm, fakeDocker, pod, backOff, c.expectErr)
		verifyCalls(t, fakeDocker, c.result)

		// Verify whether the correct sync pod result is generated
		if c.expectErr {
			verifySyncResults(t, []*kubecontainer.SyncResult{backoffResult}, result)
		} else {
			verifySyncResults(t, []*kubecontainer.SyncResult{startResult}, result)
		}

		if backOff.Get(stableId) != time.Duration(c.backoff)*time.Second {
			t.Errorf("At tick %s expected backoff=%s got=%s", time.Duration(c.tick)*time.Second, time.Duration(c.backoff)*time.Second, backOff.Get(stableId))
		}

		if len(fakeDocker.Created) > 0 {
			// pretend kill the container
			fakeDocker.Created = nil
			dockerContainers[2].FinishedAt = startTime.Add(time.Duration(c.killDelay) * time.Second)
		}
	}
}

func TestGetRestartCount(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	containerName := "bar"
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: containerName},
			},
			RestartPolicy: "Always",
		},
		Status: api.PodStatus{
			ContainerStatuses: []api.ContainerStatus{
				{
					Name:         containerName,
					RestartCount: 3,
				},
			},
		},
	}

	// Helper function for verifying the restart count.
	verifyRestartCount := func(pod *api.Pod, expectedCount int) {
		runSyncPod(t, dm, fakeDocker, pod, nil, false)
		status, err := dm.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		cs := status.FindContainerStatusByName(containerName)
		if cs == nil {
			t.Fatalf("Can't find status for container %q", containerName)
		}
		restartCount := cs.RestartCount
		if restartCount != expectedCount {
			t.Errorf("expected %d restart count, got %d", expectedCount, restartCount)
		}
	}

	killOneContainer := func(pod *api.Pod) {
		status, err := dm.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		cs := status.FindContainerStatusByName(containerName)
		if cs == nil {
			t.Fatalf("Can't find status for container %q", containerName)
		}
		dm.KillContainerInPod(cs.ID, &pod.Spec.Containers[0], pod, "test container restart count.", nil)
	}
	// Container "bar" starts the first time.
	// TODO: container lists are expected to be sorted reversely by time.
	// We should fix FakeDockerClient to sort the list before returning.
	// (randome-liu) Just partially sorted now.
	verifyRestartCount(&pod, 0)
	killOneContainer(&pod)

	// Poor container "bar" has been killed, and should be restarted with restart count 1
	verifyRestartCount(&pod, 1)
	killOneContainer(&pod)

	// Poor container "bar" has been killed again, and should be restarted with restart count 2
	verifyRestartCount(&pod, 2)
	killOneContainer(&pod)

	// Poor container "bar" has been killed again ang again, and should be restarted with restart count 3
	verifyRestartCount(&pod, 3)

	// The oldest container has been garbage collected
	exitedContainers := fakeDocker.ExitedContainerList
	fakeDocker.ExitedContainerList = exitedContainers[:len(exitedContainers)-1]
	verifyRestartCount(&pod, 3)

	// The last two oldest containers have been garbage collected
	fakeDocker.ExitedContainerList = exitedContainers[:len(exitedContainers)-2]
	verifyRestartCount(&pod, 3)

	// All exited containers have been garbage collected, restart count should be got from old api pod status
	fakeDocker.ExitedContainerList = []dockertypes.Container{}
	verifyRestartCount(&pod, 3)
	killOneContainer(&pod)

	// Poor container "bar" has been killed again ang again and again, and should be restarted with restart count 4
	verifyRestartCount(&pod, 4)
}

func TestGetTerminationMessagePath(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	containers := []api.Container{
		{
			Name: "bar",
			TerminationMessagePath: "/dev/somepath",
		},
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: containers,
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	containerList := fakeDocker.RunningContainerList
	if len(containerList) != 2 {
		// One for infra container, one for container "bar"
		t.Fatalf("unexpected container list length %d", len(containerList))
	}
	inspectResult, err := fakeDocker.InspectContainer(containerList[0].ID)
	if err != nil {
		t.Fatalf("unexpected inspect error: %v", err)
	}
	containerInfo := getContainerInfoFromLabel(inspectResult.Config.Labels)
	terminationMessagePath := containerInfo.TerminationMessagePath
	if terminationMessagePath != containers[0].TerminationMessagePath {
		t.Errorf("expected termination message path %s, got %s", containers[0].TerminationMessagePath, terminationMessagePath)
	}
}

func TestSyncPodWithPodInfraCreatesContainerCallsHandler(t *testing.T) {
	fakeHTTPClient := &fakeHTTP{}
	dm, fakeDocker := newTestDockerManagerWithHTTPClient(fakeHTTPClient)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "bar",
					Lifecycle: &api.Lifecycle{
						PostStart: &api.Handler{
							HTTPGet: &api.HTTPGetAction{
								Host: "foo",
								Port: intstr.FromInt(8080),
								Path: "bar",
							},
						},
					},
				},
			},
		},
	}
	fakeDocker.SetFakeRunningContainers([]*FakeContainer{{
		ID:   "9876",
		Name: "/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0",
	}})
	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 1 ||
		!matchString(t, "/k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
	if fakeHTTPClient.url != "http://foo:8080/bar" {
		t.Errorf("unexpected handler: %q", fakeHTTPClient.url)
	}
}

func TestSyncPodEventHandlerFails(t *testing.T) {
	// Simulate HTTP failure.
	fakeHTTPClient := &fakeHTTP{err: fmt.Errorf("test error")}
	dm, fakeDocker := newTestDockerManagerWithHTTPClient(fakeHTTPClient)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "bar",
					Lifecycle: &api.Lifecycle{
						PostStart: &api.Handler{
							HTTPGet: &api.HTTPGetAction{
								Host: "does.no.exist",
								Port: intstr.FromInt(8080),
								Path: "bar",
							},
						},
					},
				},
			},
		},
	}

	fakeDocker.SetFakeRunningContainers([]*FakeContainer{{
		ID:   "9876",
		Name: "/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0",
	}})
	runSyncPod(t, dm, fakeDocker, pod, nil, true)

	verifyCalls(t, fakeDocker, []string{
		// Create the container.
		"create", "start",
		// Kill the container since event handler fails.
		"stop",
	})

	// TODO(yifan): Check the stopped container's name.
	if len(fakeDocker.Stopped) != 1 {
		t.Fatalf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
	dockerName, _, err := ParseDockerName(fakeDocker.Stopped[0])
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if dockerName.ContainerName != "bar" {
		t.Errorf("Wrong stopped container, expected: bar, get: %q", dockerName.ContainerName)
	}
}

type fakeReadWriteCloser struct{}

func (*fakeReadWriteCloser) Read([]byte) (int, error)  { return 0, nil }
func (*fakeReadWriteCloser) Write([]byte) (int, error) { return 0, nil }
func (*fakeReadWriteCloser) Close() error              { return nil }

func TestPortForwardNoSuchContainer(t *testing.T) {
	dm, _ := newTestDockerManager()

	podName, podNamespace := "podName", "podNamespace"
	err := dm.PortForward(
		&kubecontainer.Pod{
			ID:         "podID",
			Name:       podName,
			Namespace:  podNamespace,
			Containers: nil,
		},
		5000,
		// need a valid io.ReadWriteCloser here
		&fakeReadWriteCloser{},
	)
	if err == nil {
		t.Fatal("unexpected non-error")
	}
	expectedErr := noPodInfraContainerError(podName, podNamespace)
	if !reflect.DeepEqual(err, expectedErr) {
		t.Fatalf("expected %v, but saw %v", expectedErr, err)
	}
}

func TestSyncPodWithTerminationLog(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	container := api.Container{
		Name: "bar",
		TerminationMessagePath: "/dev/somepath",
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				container,
			},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil, false)
	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	defer os.Remove(testPodContainerDir)

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
	newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	parts := strings.Split(newContainer.HostConfig.Binds[0], ":")
	if !matchString(t, testPodContainerDir+"/[a-f0-9]", parts[0]) {
		t.Errorf("unexpected host path: %s", parts[0])
	}
	if parts[1] != "/dev/somepath" {
		t.Errorf("unexpected container path: %s", parts[1])
	}
}

func TestSyncPodWithHostNetwork(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "bar"},
			},
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: true,
			},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()

	newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	utsMode := newContainer.HostConfig.UTSMode
	if utsMode != "host" {
		t.Errorf("Pod with host network must have \"host\" utsMode, actual: \"%v\"", utsMode)
	}
}

func TestVerifyNonRoot(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()

	// setup test cases.
	var rootUid int64 = 0
	var nonRootUid int64 = 1

	tests := map[string]struct {
		container     *api.Container
		inspectImage  *dockertypes.ImageInspect
		expectedError string
	}{
		// success cases
		"non-root runAsUser": {
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: &nonRootUid,
				},
			},
		},
		"numeric non-root image user": {
			container: &api.Container{},
			inspectImage: &dockertypes.ImageInspect{
				Config: &dockercontainer.Config{
					User: "1",
				},
			},
		},
		"numeric non-root image user with gid": {
			container: &api.Container{},
			inspectImage: &dockertypes.ImageInspect{
				Config: &dockercontainer.Config{
					User: "1:2",
				},
			},
		},

		// failure cases
		"root runAsUser": {
			container: &api.Container{
				SecurityContext: &api.SecurityContext{
					RunAsUser: &rootUid,
				},
			},
			expectedError: "container's runAsUser breaks non-root policy",
		},
		"non-numeric image user": {
			container: &api.Container{},
			inspectImage: &dockertypes.ImageInspect{
				Config: &dockercontainer.Config{
					User: "foo",
				},
			},
			expectedError: "non-numeric user",
		},
		"numeric root image user": {
			container: &api.Container{},
			inspectImage: &dockertypes.ImageInspect{
				Config: &dockercontainer.Config{
					User: "0",
				},
			},
			expectedError: "container has no runAsUser and image will run as root",
		},
		"numeric root image user with gid": {
			container: &api.Container{},
			inspectImage: &dockertypes.ImageInspect{
				Config: &dockercontainer.Config{
					User: "0:1",
				},
			},
			expectedError: "container has no runAsUser and image will run as root",
		},
		"nil image in inspect": {
			container:     &api.Container{},
			expectedError: "unable to inspect image",
		},
		"nil config in image inspect": {
			container:     &api.Container{},
			inspectImage:  &dockertypes.ImageInspect{},
			expectedError: "unable to inspect image",
		},
	}

	for k, v := range tests {
		fakeDocker.Image = v.inspectImage
		err := dm.verifyNonRoot(v.container)
		if v.expectedError == "" && err != nil {
			t.Errorf("case[%q]: unexpected error: %v", k, err)
		}
		if v.expectedError != "" && !strings.Contains(err.Error(), v.expectedError) {
			t.Errorf("case[%q]: expected: %q, got: %q", k, v.expectedError, err.Error())
		}
	}
}

func TestGetUidFromUser(t *testing.T) {
	tests := map[string]struct {
		input  string
		expect string
	}{
		"no gid": {
			input:  "0",
			expect: "0",
		},
		"uid/gid": {
			input:  "0:1",
			expect: "0",
		},
		"empty input": {
			input:  "",
			expect: "",
		},
		"multiple spearators": {
			input:  "1:2:3",
			expect: "1",
		},
	}
	for k, v := range tests {
		actual := getUidFromUser(v.input)
		if actual != v.expect {
			t.Errorf("%s failed.  Expected %s but got %s", k, v.expect, actual)
		}
	}
}

func TestGetPidMode(t *testing.T) {
	// test false
	pod := &api.Pod{}
	pidMode := getPidMode(pod)

	if pidMode != "" {
		t.Errorf("expected empty pid mode for pod but got %v", pidMode)
	}

	// test true
	pod.Spec.SecurityContext = &api.PodSecurityContext{}
	pod.Spec.SecurityContext.HostPID = true
	pidMode = getPidMode(pod)
	if pidMode != "host" {
		t.Errorf("expected host pid mode for pod but got %v", pidMode)
	}
}

func TestGetIPCMode(t *testing.T) {
	// test false
	pod := &api.Pod{}
	ipcMode := getIPCMode(pod)

	if ipcMode != "" {
		t.Errorf("expected empty ipc mode for pod but got %v", ipcMode)
	}

	// test true
	pod.Spec.SecurityContext = &api.PodSecurityContext{}
	pod.Spec.SecurityContext.HostIPC = true
	ipcMode = getIPCMode(pod)
	if ipcMode != "host" {
		t.Errorf("expected host ipc mode for pod but got %v", ipcMode)
	}
}

func TestSyncPodWithPullPolicy(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	puller := dm.dockerPuller.(*FakeDockerPuller)
	puller.HasImages = []string{"existing_one", "want:latest"}
	dm.podInfraContainerImage = "pod_infra_image"

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "bar", Image: "pull_always_image", ImagePullPolicy: api.PullAlways},
				{Name: "bar2", Image: "pull_if_not_present_image", ImagePullPolicy: api.PullIfNotPresent},
				{Name: "bar3", Image: "existing_one", ImagePullPolicy: api.PullIfNotPresent},
				{Name: "bar4", Image: "want:latest", ImagePullPolicy: api.PullIfNotPresent},
				{Name: "bar5", Image: "pull_never_image", ImagePullPolicy: api.PullNever},
			},
		},
	}

	expectedResults := []*kubecontainer.SyncResult{
		//Sync result for infra container
		{kubecontainer.StartContainer, PodInfraContainerName, nil, ""},
		{kubecontainer.SetupNetwork, kubecontainer.GetPodFullName(pod), nil, ""},
		//Sync result for user containers
		{kubecontainer.StartContainer, "bar", nil, ""},
		{kubecontainer.StartContainer, "bar2", nil, ""},
		{kubecontainer.StartContainer, "bar3", nil, ""},
		{kubecontainer.StartContainer, "bar4", nil, ""},
		{kubecontainer.StartContainer, "bar5", kubecontainer.ErrImageNeverPull,
			"Container image \"pull_never_image\" is not present with pull policy of Never"},
	}

	result := runSyncPod(t, dm, fakeDocker, pod, nil, true)
	verifySyncResults(t, expectedResults, result)

	fakeDocker.Lock()
	defer fakeDocker.Unlock()

	pulledImageSorted := puller.ImagesPulled[:]
	sort.Strings(pulledImageSorted)
	assert.Equal(t, []string{"pod_infra_image", "pull_always_image", "pull_if_not_present_image"}, pulledImageSorted)

	if len(fakeDocker.Created) != 5 {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
}

// This test only covers SyncPod with PullImageFailure, CreateContainerFailure and StartContainerFailure.
// There are still quite a few failure cases not covered.
// TODO(random-liu): Better way to test the SyncPod failures.
func TestSyncPodWithFailure(t *testing.T) {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
	}
	tests := map[string]struct {
		container   api.Container
		dockerError map[string]error
		pullerError []error
		expected    []*kubecontainer.SyncResult
	}{
		"PullImageFailure": {
			api.Container{Name: "bar", Image: "realImage", ImagePullPolicy: api.PullAlways},
			map[string]error{},
			[]error{fmt.Errorf("can't pull image")},
			[]*kubecontainer.SyncResult{{kubecontainer.StartContainer, "bar", kubecontainer.ErrImagePull, "can't pull image"}},
		},
		"CreateContainerFailure": {
			api.Container{Name: "bar", Image: "alreadyPresent"},
			map[string]error{"create": fmt.Errorf("can't create container")},
			[]error{},
			[]*kubecontainer.SyncResult{{kubecontainer.StartContainer, "bar", kubecontainer.ErrRunContainer, "can't create container"}},
		},
		"StartContainerFailure": {
			api.Container{Name: "bar", Image: "alreadyPresent"},
			map[string]error{"start": fmt.Errorf("can't start container")},
			[]error{},
			[]*kubecontainer.SyncResult{{kubecontainer.StartContainer, "bar", kubecontainer.ErrRunContainer, "can't start container"}},
		},
	}

	for _, test := range tests {
		dm, fakeDocker := newTestDockerManager()
		puller := dm.dockerPuller.(*FakeDockerPuller)
		puller.HasImages = []string{test.container.Image}
		// Pretend that the pod infra container has already been created, so that
		// we can run the user containers.
		fakeDocker.SetFakeRunningContainers([]*FakeContainer{{
			ID:   "9876",
			Name: "/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0",
		}})
		fakeDocker.InjectErrors(test.dockerError)
		puller.ErrorsToInject = test.pullerError
		pod.Spec.Containers = []api.Container{test.container}
		result := runSyncPod(t, dm, fakeDocker, pod, nil, true)
		verifySyncResults(t, test.expected, result)
	}
}

// Verify whether all the expected results appear exactly only once in real result.
func verifySyncResults(t *testing.T, expectedResults []*kubecontainer.SyncResult, realResult kubecontainer.PodSyncResult) {
	if len(expectedResults) != len(realResult.SyncResults) {
		t.Errorf("expected sync result number %d, got %d", len(expectedResults), len(realResult.SyncResults))
		for _, r := range expectedResults {
			t.Errorf("expected result: %+v", r)
		}
		for _, r := range realResult.SyncResults {
			t.Errorf("real result: %+v", r)
		}
		return
	}
	// The container start order is not fixed, because SyncPod() uses a map to store the containers to start.
	// Here we should make sure each expected result appears only once in the real result.
	for _, expectR := range expectedResults {
		found := 0
		for _, realR := range realResult.SyncResults {
			// For the same action of the same container, the result should be the same
			if realR.Target == expectR.Target && realR.Action == expectR.Action {
				// We use Contains() here because the message format may be changed, but at least we should
				// make sure that the expected message is contained.
				if realR.Error != expectR.Error || !strings.Contains(realR.Message, expectR.Message) {
					t.Errorf("expected sync result %+v, got %+v", expectR, realR)
				}
				found++
			}
		}
		if found == 0 {
			t.Errorf("not found expected result %+v", expectR)
		}
		if found > 1 {
			t.Errorf("got %d duplicate expected result %+v", found, expectR)
		}
	}
}

func TestSeccompIsDisabledWithDockerV110(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithVersion("1.10.1", "1.22")
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "bar"},
			},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()

	newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	assert.Contains(t, newContainer.HostConfig.SecurityOpt, "seccomp:unconfined", "Pods with Docker versions >= 1.10 must have seccomp disabled.")
}

func TestSecurityOptsAreNilWithDockerV19(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithVersion("1.9.1", "1.21")
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "bar"},
			},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()

	newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	assert.NotContains(t, newContainer.HostConfig.SecurityOpt, "seccomp:unconfined", "Pods with Docker versions < 1.10 must not have seccomp disabled by default")
}

func TestCheckVersionCompatibility(t *testing.T) {
	type test struct {
		version    string
		compatible bool
	}
	tests := []test{
		// Minimum apiversion
		{minimumDockerAPIVersion, true},
		// Invalid apiversion
		{"invalid_api_version", false},
		// Older apiversion
		{"1.0.0", false},
		// Newer apiversion
		// NOTE(random-liu): We need to bump up the newer apiversion,
		// if docker apiversion really reaches "9.9.9" someday. But I
		// really doubt whether the test could live that long.
		{"9.9.9", true},
	}
	for i, tt := range tests {
		testCase := fmt.Sprintf("test case #%d test version %q", i, tt.version)
		dm, fakeDocker := newTestDockerManagerWithVersion("", tt.version)
		err := dm.checkVersionCompatibility()
		assert.Equal(t, tt.compatible, err == nil, testCase)
		if tt.compatible == true {
			// Get docker version error
			fakeDocker.InjectError("version", fmt.Errorf("injected version error"))
			err := dm.checkVersionCompatibility()
			assert.NotNil(t, err, testCase+" version error check")
		}
	}
}

func TestVersion(t *testing.T) {
	expectedVersion := "1.8.1"
	expectedAPIVersion := "1.20"
	dm, _ := newTestDockerManagerWithVersion(expectedVersion, expectedAPIVersion)
	version, err := dm.Version()
	if err != nil {
		t.Errorf("got error while getting docker server version - %v", err)
	}
	if e, a := expectedVersion, version.String(); e != a {
		t.Errorf("expect docker server version %q, got %q", e, a)
	}

	apiVersion, err := dm.APIVersion()
	if err != nil {
		t.Errorf("got error while getting docker api version - %v", err)
	}
	if e, a := expectedAPIVersion, apiVersion.String(); e != a {
		t.Errorf("expect docker api version %q, got %q", e, a)
	}
}

func TestGetPodStatusNoSuchContainer(t *testing.T) {
	const (
		noSuchContainerID = "nosuchcontainer"
		infraContainerID  = "9876"
	)
	dm, fakeDocker := newTestDockerManager()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{{Name: "nosuchcontainer"}},
		},
	}

	fakeDocker.SetFakeContainers([]*FakeContainer{
		{
			ID:         noSuchContainerID,
			Name:       "/k8s_nosuchcontainer_foo_new_12345678_42",
			ExitCode:   0,
			StartedAt:  time.Now(),
			FinishedAt: time.Now(),
			Running:    false,
		},
		{
			ID:         infraContainerID,
			Name:       "/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_42",
			ExitCode:   0,
			StartedAt:  time.Now(),
			FinishedAt: time.Now(),
			Running:    false,
		},
	})
	fakeDocker.InjectErrors(map[string]error{"inspect_container": containerNotFoundError{}})
	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	// Verify that we will try to start new contrainers even if the inspections
	// failed.
	verifyCalls(t, fakeDocker, []string{
		// Start a new infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Start a new container.
		"create", "start", "inspect_container",
	})
}
