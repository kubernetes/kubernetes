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

	docker "github.com/fsouza/go-dockerclient"
	cadvisorApi "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	uexec "k8s.io/kubernetes/pkg/util/exec"
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

type fakeOptionGenerator struct{}

var _ kubecontainer.RunContainerOptionsGenerator = &fakeOptionGenerator{}

var testPodContainerDir string

func (*fakeOptionGenerator) GenerateRunContainerOptions(pod *api.Pod, container *api.Container) (*kubecontainer.RunContainerOptions, error) {
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

func newTestDockerManagerWithHTTPClient(fakeHTTPClient *fakeHTTP) (*DockerManager, *FakeDockerClient) {
	fakeDocker := &FakeDockerClient{VersionInfo: docker.Env{"Version=1.1.3", "ApiVersion=1.15"}, Errors: make(map[string]error), RemovedImages: sets.String{}}
	fakeRecorder := &record.FakeRecorder{}
	containerRefManager := kubecontainer.NewRefManager()
	networkPlugin, _ := network.InitNetworkPlugin([]network.NetworkPlugin{}, "", network.NewFakeHost(nil))
	optionGenerator := &fakeOptionGenerator{}
	dockerManager := NewFakeDockerManager(
		fakeDocker,
		fakeRecorder,
		prober.FakeProber{},
		containerRefManager,
		&cadvisorApi.MachineInfo{},
		PodInfraContainerImage,
		0, 0, "",
		kubecontainer.FakeOS{},
		networkPlugin,
		optionGenerator,
		fakeHTTPClient,
		util.NewBackOff(time.Second, 300*time.Second))

	return dockerManager, fakeDocker
}

func newTestDockerManager() (*DockerManager, *FakeDockerClient) {
	return newTestDockerManagerWithHTTPClient(&fakeHTTP{})
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
		expected  *docker.CreateContainerOptions
	}{
		{
			name:      "none",
			container: &api.Container{},
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{},
			},
		},
		{
			name: "command",
			container: &api.Container{
				Command: []string{"foo", "bar"},
			},
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{
					Entrypoint: []string{"foo", "bar"},
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
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{
					Entrypoint: []string{"foo", "zoo", "boo"},
				},
			},
		},
		{
			name: "args",
			container: &api.Container{
				Args: []string{"foo", "bar"},
			},
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{
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
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{
					Cmd: []string{"zap", "hap", "trap"},
				},
			},
		},
		{
			name: "both",
			container: &api.Container{
				Command: []string{"foo"},
				Args:    []string{"bar", "baz"},
			},
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{
					Entrypoint: []string{"foo"},
					Cmd:        []string{"bar", "baz"},
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
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{
					Entrypoint: []string{"boo--zoo", "foo", "roo"},
					Cmd:        []string{"foo", "zoo", "boo"},
				},
			},
		},
	}

	for _, tc := range cases {
		opts := &kubecontainer.RunContainerOptions{
			Envs: tc.envs,
		}

		actualOpts := &docker.CreateContainerOptions{
			Config: &docker.Config{},
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
	dockerContainers := []docker.APIContainers{
		{
			ID:    "1111",
			Names: []string{"/k8s_foo_qux_new_1234_42"},
		},
		{
			ID:    "2222",
			Names: []string{"/k8s_bar_qux_new_1234_42"},
		},
		{
			ID:    "3333",
			Names: []string{"/k8s_bar_jlk_wen_5678_42"},
		},
	}

	// Convert the docker containers. This does not affect the test coverage
	// because the conversion is tested separately in convert_test.go
	containers := make([]*kubecontainer.Container, len(dockerContainers))
	for i := range containers {
		c, err := toRuntimeContainer(&dockerContainers[i])
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

	fakeDocker.ContainerList = dockerContainers
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
	dockerImages := []docker.APIImages{{ID: "1111"}, {ID: "2222"}, {ID: "3333"}}
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

func apiContainerToContainer(c docker.APIContainers) kubecontainer.Container {
	dockerName, hash, err := ParseDockerName(c.Names[0])
	if err != nil {
		return kubecontainer.Container{}
	}
	return kubecontainer.Container{
		ID:   kubecontainer.ContainerID{"docker", c.ID},
		Name: dockerName.ContainerName,
		Hash: hash,
	}
}

func dockerContainersToPod(containers DockerContainers) kubecontainer.Pod {
	var pod kubecontainer.Pod
	for _, c := range containers {
		dockerName, hash, err := ParseDockerName(c.Names[0])
		if err != nil {
			continue
		}
		pod.Containers = append(pod.Containers, &kubecontainer.Container{
			ID:    kubecontainer.ContainerID{"docker", c.ID},
			Name:  dockerName.ContainerName,
			Hash:  hash,
			Image: c.Image,
		})
		// TODO(yifan): Only one evaluation is enough.
		pod.ID = dockerName.PodUID
		name, namespace, _ := kubecontainer.ParsePodFullName(dockerName.PodFullName)
		pod.Name = name
		pod.Namespace = namespace
	}
	return pod
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
	containers := []docker.APIContainers{
		{
			ID:    "1111",
			Names: []string{"/k8s_foo_qux_new_1234_42"},
		},
		{
			ID:    "2222",
			Names: []string{"/k8s_bar_qux_new_1234_42"},
		},
	}
	containerToKill := &containers[0]
	containerToSpare := &containers[1]
	fakeDocker.ContainerList = containers

	if err := manager.KillContainerInPod(kubecontainer.ContainerID{}, &pod.Spec.Containers[0], pod); err != nil {
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
	fakeDocker.ExecInspect = &docker.ExecInspect{
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
	podString, err := testapi.Default.Codec().Encode(pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	containers := []docker.APIContainers{
		{
			ID:    "1111",
			Names: []string{"/k8s_foo_qux_new_1234_42"},
		},
		{
			ID:    "2222",
			Names: []string{"/k8s_bar_qux_new_1234_42"},
		},
	}
	containerToKill := &containers[0]
	fakeDocker.ContainerList = containers
	fakeDocker.Container = &docker.Container{
		Config: &docker.Config{
			Labels: map[string]string{
				kubernetesPodLabel:       string(podString),
				kubernetesContainerLabel: "foo",
			},
		},
	}

	if err := manager.KillContainerInPod(kubecontainer.ContainerID{}, &pod.Spec.Containers[0], pod); err != nil {
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
	containers := []docker.APIContainers{
		{
			ID:    "1111",
			Names: []string{"/k8s_foo_qux_new_1234_42"},
		},
		{
			ID:    "2222",
			Names: []string{"/k8s_bar_qux_new_1234_42"},
		},
	}
	fakeDocker.ContainerList = containers
	fakeDocker.Errors["stop"] = fmt.Errorf("sample error")

	if err := manager.KillContainerInPod(kubecontainer.ContainerID{}, &pod.Spec.Containers[0], pod); err == nil {
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
		Image:           PodInfraContainerImage,
		Ports:           ports,
		ImagePullPolicy: podInfraContainerImagePullPolicy,
	}
	return kubecontainer.HashContainer(container)
}

// runSyncPod is a helper function to retrieve the running pods from the fake
// docker client and runs SyncPod for the given pod.
func runSyncPod(t *testing.T, dm *DockerManager, fakeDocker *FakeDockerClient, pod *api.Pod, backOff *util.Backoff) {
	runningPods, err := dm.GetPods(false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	runningPod := kubecontainer.Pods(runningPods).FindPodByID(pod.UID)
	podStatus, err := dm.GetPodStatus(pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fakeDocker.ClearCalls()
	if backOff == nil {
		backOff = util.NewBackOff(time.Second, time.Minute)
	}
	err = dm.SyncPod(pod, runningPod, *podStatus, []api.Secret{}, backOff)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestSyncPodCreateNetAndContainer(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	dm.podInfraContainerImage = "pod_infra_image"
	fakeDocker.ContainerList = []docker.APIContainers{}
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

	runSyncPod(t, dm, fakeDocker, pod, nil)
	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})
	fakeDocker.Lock()

	found := false
	for _, c := range fakeDocker.ContainerList {
		if c.Image == "pod_infra_image" && strings.HasPrefix(c.Names[0], "/k8s_POD") {
			found = true
		}
	}
	if !found {
		t.Errorf("Custom pod infra container not found: %v", fakeDocker.ContainerList)
	}

	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "k8s_POD\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
}

func TestSyncPodCreatesNetAndContainerPullsImage(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	dm.podInfraContainerImage = "pod_infra_image"
	puller := dm.dockerPuller.(*FakeDockerPuller)
	puller.HasImages = []string{}
	dm.podInfraContainerImage = "pod_infra_image"
	fakeDocker.ContainerList = []docker.APIContainers{}
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

	runSyncPod(t, dm, fakeDocker, pod, nil)

	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()

	if !reflect.DeepEqual(puller.ImagesPulled, []string{"pod_infra_image", "something"}) {
		t.Errorf("Unexpected pulled containers: %v", puller.ImagesPulled)
	}

	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "k8s_POD\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
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
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"9876": {
			ID:         "9876",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil)

	verifyCalls(t, fakeDocker, []string{
		// Inspect pod infra container (but does not create)"
		"inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 1 ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
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
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
			Names: []string{"/k8s_bar1_foo1_new_12345678_0"},
			ID:    "1234",
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil)

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

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_foo_bar_new_12345678_1111"},
			ID:    "1234",
		},
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_bar_new_12345678_2222"},
			ID:    "9876",
		},
		{
			// Duplicate for the same container.
			Names: []string{"/k8s_foo_bar_new_12345678_3333"},
			ID:    "4567",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"1234": {
			ID:         "1234",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
		"9876": {
			ID:         "9876",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
		"4567": {
			ID:         "4567",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil)

	verifyCalls(t, fakeDocker, []string{
		// Check the pod infra container.
		"inspect_container",
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

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_bar.1234_foo_new_12345678_42"},
			ID:    "1234",
		},
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_42"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"1234": {
			ID:         "1234",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
		"9876": {
			ID:         "9876",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil)

	verifyCalls(t, fakeDocker, []string{
		// Check the pod infra container.
		"inspect_container",
		// Kill and restart the bad hash container.
		"stop", "create", "start", "inspect_container",
	})

	if err := fakeDocker.AssertStopped([]string{"1234"}); err != nil {
		t.Errorf("%v", err)
	}
}

func TestSyncPodsUnhealthy(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "bar",
					LivenessProbe: &api.Probe{
					// Always returns healthy == false
					},
				},
			},
		},
	}

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_bar_foo_new_12345678_42"},
			ID:    "1234",
		},
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_42"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"1234": {
			ID:         "1234",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
		"9876": {
			ID:         "9876",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil)

	verifyCalls(t, fakeDocker, []string{
		// Check the pod infra container.
		"inspect_container",
		// Kill the unhealthy container.
		"stop",
		// Restart the unhealthy container.
		"create", "start", "inspect_container",
	})

	if err := fakeDocker.AssertStopped([]string{"1234"}); err != nil {
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

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>_<random>
			Names: []string{"/k8s_bar." + strconv.FormatUint(kubecontainer.HashContainer(&container), 16) + "_foo_new_12345678_0"},
			ID:    "1234",
		},
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"1234": {
			ID:         "1234",
			HostConfig: &docker.HostConfig{},
			Config:     &docker.Config{},
		},
		"9876": {
			ID:         "9876",
			HostConfig: &docker.HostConfig{},
			Config:     &docker.Config{},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil)

	verifyCalls(t, fakeDocker, []string{
		// Check the pod infra contianer.
		"inspect_container",
	})
}

func TestSyncPodWithPullPolicy(t *testing.T) {
	api.ForTesting_ReferencesAllowBlankSelfLinks = true
	dm, fakeDocker := newTestDockerManager()
	puller := dm.dockerPuller.(*FakeDockerPuller)
	puller.HasImages = []string{"existing_one", "want:latest"}
	dm.podInfraContainerImage = "pod_infra_image"
	fakeDocker.ContainerList = []docker.APIContainers{}

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

	expectedStatusMap := map[string]api.ContainerState{
		"bar":  {Running: &api.ContainerStateRunning{unversioned.Now()}},
		"bar2": {Running: &api.ContainerStateRunning{unversioned.Now()}},
		"bar3": {Running: &api.ContainerStateRunning{unversioned.Now()}},
		"bar4": {Running: &api.ContainerStateRunning{unversioned.Now()}},
		"bar5": {Waiting: &api.ContainerStateWaiting{Reason: kubecontainer.ErrImageNeverPull.Error(),
			Message: "Container image \"pull_never_image\" is not present with pull policy of Never"}},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil)
	statuses, err := dm.GetPodStatus(pod)
	if err != nil {
		t.Errorf("unable to get pod status")
	}
	for _, c := range pod.Spec.Containers {
		if containerStatus, ok := api.GetContainerStatus(statuses.ContainerStatuses, c.Name); ok {
			// copy the StartedAt time, to make the structs match
			if containerStatus.State.Running != nil && expectedStatusMap[c.Name].Running != nil {
				expectedStatusMap[c.Name].Running.StartedAt = containerStatus.State.Running.StartedAt
			}
			assert.Equal(t, containerStatus.State, expectedStatusMap[c.Name], "for container %s", c.Name)
		}
	}

	fakeDocker.Lock()
	defer fakeDocker.Unlock()

	pulledImageSorted := puller.ImagesPulled[:]
	sort.Strings(pulledImageSorted)
	assert.Equal(t, []string{"pod_infra_image", "pull_always_image", "pull_if_not_present_image"}, pulledImageSorted)

	if len(fakeDocker.Created) != 5 {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
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

	runningAPIContainers := []docker.APIContainers{
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0"},
			ID:    "9876",
		},
	}
	exitedAPIContainers := []docker.APIContainers{
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
			Names: []string{"/k8s_succeeded." + strconv.FormatUint(kubecontainer.HashContainer(&containers[0]), 16) + "_foo_new_12345678_0"},
			ID:    "1234",
		},
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
			Names: []string{"/k8s_failed." + strconv.FormatUint(kubecontainer.HashContainer(&containers[1]), 16) + "_foo_new_12345678_0"},
			ID:    "5678",
		},
	}

	containerMap := map[string]*docker.Container{
		"9876": {
			ID:     "9876",
			Name:   "POD",
			Config: &docker.Config{},
			State: docker.State{
				StartedAt: time.Now(),
				Running:   true,
			},
		},
		"1234": {
			ID:     "1234",
			Name:   "succeeded",
			Config: &docker.Config{},
			State: docker.State{
				ExitCode:   0,
				StartedAt:  time.Now(),
				FinishedAt: time.Now(),
			},
		},
		"5678": {
			ID:     "5678",
			Name:   "failed",
			Config: &docker.Config{},
			State: docker.State{
				ExitCode:   42,
				StartedAt:  time.Now(),
				FinishedAt: time.Now(),
			},
		},
	}

	tests := []struct {
		policy  api.RestartPolicy
		calls   []string
		created []string
		stopped []string
	}{
		{
			api.RestartPolicyAlways,
			[]string{
				// Check the pod infra container.
				"inspect_container",
				// Restart both containers.
				"create", "start", "inspect_container", "create", "start", "inspect_container",
			},
			[]string{"succeeded", "failed"},
			[]string{},
		},
		{
			api.RestartPolicyOnFailure,
			[]string{
				// Check the pod infra container.
				"inspect_container",
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
		fakeDocker.ContainerList = runningAPIContainers
		fakeDocker.ExitedContainerList = exitedAPIContainers
		fakeDocker.ContainerMap = containerMap
		pod.Spec.RestartPolicy = tt.policy

		runSyncPod(t, dm, fakeDocker, pod, nil)

		// 'stop' is because the pod infra container is killed when no container is running.
		verifyCalls(t, fakeDocker, tt.calls)

		if err := fakeDocker.AssertCreated(tt.created); err != nil {
			t.Errorf("%d: %v", i, err)
		}
		if err := fakeDocker.AssertStopped(tt.stopped); err != nil {
			t.Errorf("%d: %v", i, err)
		}
	}
}

func TestGetPodStatusWithLastTermination(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	containers := []api.Container{
		{Name: "succeeded"},
		{Name: "failed"},
	}

	exitedAPIContainers := []docker.APIContainers{
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
			Names: []string{"/k8s_succeeded." + strconv.FormatUint(kubecontainer.HashContainer(&containers[0]), 16) + "_foo_new_12345678_0"},
			ID:    "1234",
		},
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
			Names: []string{"/k8s_failed." + strconv.FormatUint(kubecontainer.HashContainer(&containers[1]), 16) + "_foo_new_12345678_0"},
			ID:    "5678",
		},
	}

	containerMap := map[string]*docker.Container{
		"9876": {
			ID:         "9876",
			Name:       "POD",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
			State: docker.State{
				StartedAt:  time.Now(),
				FinishedAt: time.Now(),
				Running:    true,
			},
		},
		"1234": {
			ID:         "1234",
			Name:       "succeeded",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
			State: docker.State{
				ExitCode:   0,
				StartedAt:  time.Now(),
				FinishedAt: time.Now(),
			},
		},
		"5678": {
			ID:         "5678",
			Name:       "failed",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
			State: docker.State{
				ExitCode:   42,
				StartedAt:  time.Now(),
				FinishedAt: time.Now(),
			},
		},
	}

	tests := []struct {
		policy           api.RestartPolicy
		created          []string
		stopped          []string
		lastTerminations []string
	}{
		{
			api.RestartPolicyAlways,
			[]string{"succeeded", "failed"},
			[]string{},
			[]string{"docker://1234", "docker://5678"},
		},
		{
			api.RestartPolicyOnFailure,
			[]string{"failed"},
			[]string{},
			[]string{"docker://5678"},
		},
		{
			api.RestartPolicyNever,
			[]string{},
			[]string{"9876"},
			[]string{},
		},
	}

	for i, tt := range tests {
		fakeDocker.ExitedContainerList = exitedAPIContainers
		fakeDocker.ContainerMap = containerMap
		fakeDocker.ClearCalls()
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "foo",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers:    containers,
				RestartPolicy: tt.policy,
			},
		}
		fakeDocker.ContainerList = []docker.APIContainers{
			{
				// pod infra container
				Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0"},
				ID:    "9876",
			},
		}

		runSyncPod(t, dm, fakeDocker, pod, nil)

		// Check if we can retrieve the pod status.
		status, err := dm.GetPodStatus(pod)
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		terminatedContainers := []string{}
		for _, cs := range status.ContainerStatuses {
			if cs.LastTerminationState.Terminated != nil {
				terminatedContainers = append(terminatedContainers, cs.LastTerminationState.Terminated.ContainerID)
			}
		}
		sort.StringSlice(terminatedContainers).Sort()
		sort.StringSlice(tt.lastTerminations).Sort()
		if !reflect.DeepEqual(terminatedContainers, tt.lastTerminations) {
			t.Errorf("Expected(sorted): %#v, Actual(sorted): %#v", tt.lastTerminations, terminatedContainers)
		}

		if err := fakeDocker.AssertCreated(tt.created); err != nil {
			t.Errorf("%d: %v", i, err)
		}
		if err := fakeDocker.AssertStopped(tt.stopped); err != nil {
			t.Errorf("%d: %v", i, err)
		}
	}
}

func TestSyncPodBackoff(t *testing.T) {
	var fakeClock = &util.FakeClock{Time: time.Now()}
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

	containerList := []docker.APIContainers{
		// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>_<random>
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_podfoo_nsnew_12345678_0"},
			ID:    "9876",
		},
		{
			Names: []string{"/k8s_good." + strconv.FormatUint(kubecontainer.HashContainer(&containers[0]), 16) + "_podfoo_nsnew_12345678_0"},
			ID:    "1234",
		},
	}

	exitedAPIContainers := []docker.APIContainers{
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
			Names: []string{"/k8s_bad." + strconv.FormatUint(kubecontainer.HashContainer(&containers[1]), 16) + "_podfoo_nsnew_12345678_0"},
			ID:    "5678",
		},
	}
	stableId := "k8s_bad." + strconv.FormatUint(kubecontainer.HashContainer(&containers[1]), 16) + "_podfoo_nsnew_12345678"
	containerMap := map[string]*docker.Container{
		"9876": {
			ID:         "9876",
			Name:       "POD",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
			State: docker.State{
				StartedAt: startTime,
				Running:   true,
			},
		},
		"1234": {
			ID:         "1234",
			Name:       "good",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
			State: docker.State{
				StartedAt: startTime,
				Running:   true,
			},
		},
		"5678": {
			ID:         "5678",
			Name:       "bad",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
			State: docker.State{
				ExitCode:   42,
				StartedAt:  startTime,
				FinishedAt: fakeClock.Now(),
			},
		},
	}

	startCalls := []string{"inspect_container", "create", "start", "inspect_container"}
	backOffCalls := []string{"inspect_container"}
	tests := []struct {
		tick      int
		backoff   int
		killDelay int
		result    []string
	}{
		{1, 1, 1, startCalls},
		{2, 2, 2, startCalls},
		{3, 2, 3, backOffCalls},
		{4, 4, 4, startCalls},
		{5, 4, 5, backOffCalls},
		{6, 4, 6, backOffCalls},
		{7, 4, 7, backOffCalls},
		{8, 8, 129, startCalls},
		{130, 1, 0, startCalls},
	}

	backOff := util.NewBackOff(time.Second, time.Minute)
	backOff.Clock = fakeClock
	for _, c := range tests {
		fakeDocker.ContainerMap = containerMap
		fakeDocker.ExitedContainerList = exitedAPIContainers
		fakeDocker.ContainerList = containerList
		fakeClock.Time = startTime.Add(time.Duration(c.tick) * time.Second)

		runSyncPod(t, dm, fakeDocker, pod, backOff)
		verifyCalls(t, fakeDocker, c.result)

		if backOff.Get(stableId) != time.Duration(c.backoff)*time.Second {
			t.Errorf("At tick %s expected backoff=%s got=%s", time.Duration(c.tick)*time.Second, time.Duration(c.backoff)*time.Second, backOff.Get(stableId))
		}

		if len(fakeDocker.Created) > 0 {
			// pretend kill the container
			fakeDocker.Created = nil
			containerMap["5678"].State.FinishedAt = startTime.Add(time.Duration(c.killDelay) * time.Second)
		}
	}
}

func TestGetPodCreationFailureReason(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()

	// Inject the creation failure error to docker.
	failureReason := "RunContainerError"
	fakeDocker.Errors = map[string]error{
		"create": fmt.Errorf("%s", failureReason),
	}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{{Name: "bar"}},
		},
	}

	// Pretend that the pod infra container has already been created, so that
	// we can run the user containers.
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"9876": {
			ID:         "9876",
			HostConfig: &docker.HostConfig{},
			Config:     &docker.Config{},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil)
	// Check if we can retrieve the pod status.
	status, err := dm.GetPodStatus(pod)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	if len(status.ContainerStatuses) < 1 {
		t.Errorf("expected 1 container status, got %d", len(status.ContainerStatuses))
	} else {
		state := status.ContainerStatuses[0].State
		if state.Waiting == nil {
			t.Errorf("expected waiting state, got %#v", state)
		} else if state.Waiting.Reason != failureReason {
			t.Errorf("expected reason %q, got %q", failureReason, state.Waiting.Reason)
		}
	}
}

func TestGetPodPullImageFailureReason(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	// Initialize the FakeDockerPuller so that it'd try to pull non-existent
	// images.
	puller := dm.dockerPuller.(*FakeDockerPuller)
	puller.HasImages = []string{}
	// Inject the pull image failure error.
	failureReason := kubecontainer.ErrImagePull.Error()
	puller.ErrorsToInject = []error{fmt.Errorf("%s", failureReason)}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{{Name: "bar", Image: "realImage", ImagePullPolicy: api.PullAlways}},
		},
	}

	// Pretend that the pod infra container has already been created, so that
	// we can run the user containers.
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"9876": {
			ID:         "9876",
			HostConfig: &docker.HostConfig{},
			Config:     &docker.Config{},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil)
	// Check if we can retrieve the pod status.
	status, err := dm.GetPodStatus(pod)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	if len(status.ContainerStatuses) < 1 {
		t.Errorf("expected 1 container status, got %d", len(status.ContainerStatuses))
	} else {
		state := status.ContainerStatuses[0].State
		if state.Waiting == nil {
			t.Errorf("expected waiting state, got %#v", state)
		} else if state.Waiting.Reason != failureReason {
			t.Errorf("expected reason %q, got %q", failureReason, state.Waiting.Reason)
		}
	}
}

func TestGetRestartCount(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	containers := []api.Container{
		{Name: "bar"},
	}
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: containers,
		},
	}

	// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
	names := []string{"/k8s_bar." + strconv.FormatUint(kubecontainer.HashContainer(&containers[0]), 16) + "_foo_new_12345678_0"}
	currTime := time.Now()
	containerMap := map[string]*docker.Container{
		"1234": {
			ID:     "1234",
			Name:   "bar",
			Config: &docker.Config{},
			State: docker.State{
				ExitCode:   42,
				StartedAt:  currTime.Add(-60 * time.Second),
				FinishedAt: currTime.Add(-60 * time.Second),
			},
		},
		"5678": {
			ID:     "5678",
			Name:   "bar",
			Config: &docker.Config{},
			State: docker.State{
				ExitCode:   42,
				StartedAt:  currTime.Add(-30 * time.Second),
				FinishedAt: currTime.Add(-30 * time.Second),
			},
		},
		"9101": {
			ID:     "9101",
			Name:   "bar",
			Config: &docker.Config{},
			State: docker.State{
				ExitCode:   42,
				StartedAt:  currTime.Add(30 * time.Minute),
				FinishedAt: currTime.Add(30 * time.Minute),
			},
		},
	}
	fakeDocker.ContainerMap = containerMap

	// Helper function for verifying the restart count.
	verifyRestartCount := func(pod *api.Pod, expectedCount int) api.PodStatus {
		status, err := dm.GetPodStatus(pod)
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		restartCount := status.ContainerStatuses[0].RestartCount
		if restartCount != expectedCount {
			t.Errorf("expected %d restart count, got %d", expectedCount, restartCount)
		}
		return *status
	}

	// Container "bar" has failed twice; create two dead docker containers.
	// TODO: container lists are expected to be sorted reversely by time.
	// We should fix FakeDockerClient to sort the list before returning.
	fakeDocker.ExitedContainerList = []docker.APIContainers{{Names: names, ID: "5678"}, {Names: names, ID: "1234"}}
	pod.Status = verifyRestartCount(&pod, 1)

	// Found a new dead container. The restart count should be incremented.
	fakeDocker.ExitedContainerList = []docker.APIContainers{
		{Names: names, ID: "9101"}, {Names: names, ID: "5678"}, {Names: names, ID: "1234"}}
	pod.Status = verifyRestartCount(&pod, 2)

	// All dead containers have been GC'd. The restart count should persist
	// (i.e., remain the same).
	fakeDocker.ExitedContainerList = []docker.APIContainers{}
	verifyRestartCount(&pod, 2)
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
								Port: util.IntOrString{IntVal: 8080, Kind: util.IntstrInt},
								Path: "bar",
							},
						},
					},
				},
			},
		},
	}
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"9876": {
			ID:         "9876",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil)

	verifyCalls(t, fakeDocker, []string{
		// Check the pod infra container.
		"inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 1 ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
	if fakeHTTPClient.url != "http://foo:8080/bar" {
		t.Errorf("Unexpected handler: %q", fakeHTTPClient.url)
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
								Port: util.IntOrString{IntVal: 8080, Kind: util.IntstrInt},
								Path: "bar",
							},
						},
					},
				},
			},
		},
	}

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_42"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"9876": {
			ID:         "9876",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}

	runSyncPod(t, dm, fakeDocker, pod, nil)

	verifyCalls(t, fakeDocker, []string{
		// Check the pod infra container.
		"inspect_container",
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
		t.Errorf("Unexpected error: %v", err)
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
	fakeDocker.ContainerList = []docker.APIContainers{}
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

	runSyncPod(t, dm, fakeDocker, pod, nil)
	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	defer os.Remove(testPodContainerDir)

	fakeDocker.Lock()
	defer fakeDocker.Unlock()

	parts := strings.Split(fakeDocker.Container.HostConfig.Binds[0], ":")
	if !matchString(t, testPodContainerDir+"/k8s_bar\\.[a-f0-9]", parts[0]) {
		t.Errorf("Unexpected host path: %s", parts[0])
	}
	if parts[1] != "/dev/somepath" {
		t.Errorf("Unexpected container path: %s", parts[1])
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

	runSyncPod(t, dm, fakeDocker, pod, nil)

	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "k8s_POD\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}

	utsMode := fakeDocker.Container.HostConfig.UTSMode
	if utsMode != "host" {
		t.Errorf("Pod with host network must have \"host\" utsMode, actual: \"%v\"", utsMode)
	}

	fakeDocker.Unlock()
}

func TestGetPodStatusSortedContainers(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()
	dockerInspect := map[string]*docker.Container{}
	dockerList := []docker.APIContainers{}
	specContainerList := []api.Container{}
	expectedOrder := []string{}

	numContainers := 10
	podName := "foo"
	podNs := "test"
	podUID := "uid1"
	fakeConfig := &docker.Config{
		Image: "some:latest",
	}

	for i := 0; i < numContainers; i++ {
		id := fmt.Sprintf("%v", i)
		containerName := fmt.Sprintf("%vcontainer", id)
		expectedOrder = append(expectedOrder, containerName)
		dockerInspect[id] = &docker.Container{
			ID:     id,
			Name:   containerName,
			Config: fakeConfig,
			Image:  fmt.Sprintf("%vimageid", id),
		}
		dockerList = append(dockerList, docker.APIContainers{
			ID:    id,
			Names: []string{fmt.Sprintf("/k8s_%v_%v_%v_%v_42", containerName, podName, podNs, podUID)},
		})
		specContainerList = append(specContainerList, api.Container{Name: containerName})
	}

	fakeDocker.ContainerMap = dockerInspect
	fakeDocker.ContainerList = dockerList
	fakeDocker.ClearCalls()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       types.UID(podUID),
			Name:      podName,
			Namespace: podNs,
		},
		Spec: api.PodSpec{
			Containers: specContainerList,
		},
	}
	for i := 0; i < 5; i++ {
		status, err := dm.GetPodStatus(pod)
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		for i, c := range status.ContainerStatuses {
			if expectedOrder[i] != c.Name {
				t.Fatalf("Container status not sorted, expected %v at index %d, but found %v", expectedOrder[i], i, c.Name)
			}
		}
	}
}

func TestVerifyNonRoot(t *testing.T) {
	dm, fakeDocker := newTestDockerManager()

	// setup test cases.
	var rootUid int64 = 0
	var nonRootUid int64 = 1

	tests := map[string]struct {
		container     *api.Container
		inspectImage  *docker.Image
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
			inspectImage: &docker.Image{
				Config: &docker.Config{
					User: "1",
				},
			},
		},
		"numeric non-root image user with gid": {
			container: &api.Container{},
			inspectImage: &docker.Image{
				Config: &docker.Config{
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
			inspectImage: &docker.Image{
				Config: &docker.Config{
					User: "foo",
				},
			},
			expectedError: "unable to validate image is non-root, non-numeric user",
		},
		"numeric root image user": {
			container: &api.Container{},
			inspectImage: &docker.Image{
				Config: &docker.Config{
					User: "0",
				},
			},
			expectedError: "container has no runAsUser and image will run as root",
		},
		"numeric root image user with gid": {
			container: &api.Container{},
			inspectImage: &docker.Image{
				Config: &docker.Config{
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
			inspectImage:  &docker.Image{},
			expectedError: "unable to inspect image",
		},
	}

	for k, v := range tests {
		fakeDocker.Image = v.inspectImage
		err := dm.verifyNonRoot(v.container)
		if v.expectedError == "" && err != nil {
			t.Errorf("%s had unexpected error %v", k, err)
		}
		if v.expectedError != "" && !strings.Contains(err.Error(), v.expectedError) {
			t.Errorf("%s expected error %s but received %s", k, v.expectedError, err.Error())
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

func TestPodDependsOnPodIP(t *testing.T) {
	tests := []struct {
		name     string
		expected bool
		env      api.EnvVar
	}{
		{
			name:     "depends on pod IP",
			expected: true,
			env: api.EnvVar{
				Name: "POD_IP",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						APIVersion: testapi.Default.Version(),
						FieldPath:  "status.podIP",
					},
				},
			},
		},
		{
			name:     "literal value",
			expected: false,
			env: api.EnvVar{
				Name:  "SOME_VAR",
				Value: "foo",
			},
		},
		{
			name:     "other downward api field",
			expected: false,
			env: api.EnvVar{
				Name: "POD_NAME",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						APIVersion: testapi.Default.Version(),
						FieldPath:  "metadata.name",
					},
				},
			},
		},
	}

	for _, tc := range tests {
		pod := &api.Pod{
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Env: []api.EnvVar{tc.env}},
				},
			},
		}

		result := podDependsOnPodIP(pod)
		if e, a := tc.expected, result; e != a {
			t.Errorf("%v: Unexpected result; expected %v, got %v", tc.name, e, a)
		}
	}
}
