/*
Copyright 2014 The Kubernetes Authors.

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
	"net"
	"net/http"
	"os"
	"path"
	"reflect"
	"regexp"
	goruntime "runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	dockertypes "github.com/docker/engine-api/types"
	dockercontainer "github.com/docker/engine-api/types/container"
	dockerstrslice "github.com/docker/engine-api/types/strslice"
	"github.com/golang/mock/gomock"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/network/mock_network"
	nettest "k8s.io/kubernetes/pkg/kubelet/network/testing"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/security/apparmor"
	kubetypes "k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/clock"
	uexec "k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/sets"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
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

func (f *fakeRuntimeHelper) GenerateRunContainerOptions(pod *v1.Pod, container *v1.Container, podIP string) (*kubecontainer.RunContainerOptions, error) {
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

func (f *fakeRuntimeHelper) GetClusterDNS(pod *v1.Pod) ([]string, []string, error) {
	return nil, nil, fmt.Errorf("not implemented")
}

// This is not used by docker runtime.
func (f *fakeRuntimeHelper) GeneratePodHostNameAndDomain(pod *v1.Pod) (string, string, error) {
	return "", "", nil
}

func (f *fakeRuntimeHelper) GetPodDir(kubetypes.UID) string {
	return ""
}

func (f *fakeRuntimeHelper) GetExtraSupplementalGroupsForPod(pod *v1.Pod) []int64 {
	return nil
}

type fakeImageManager struct{}

func newFakeImageManager() images.ImageManager {
	return &fakeImageManager{}
}

func (m *fakeImageManager) EnsureImageExists(pod *v1.Pod, container *v1.Container, pullSecrets []v1.Secret) (error, string) {
	return nil, ""
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
	networkPlugin, _ := network.InitNetworkPlugin(
		[]network.NetworkPlugin{},
		"",
		nettest.NewFakeHost(nil),
		componentconfig.HairpinNone,
		"10.0.0.0/8",
		network.UseDefaultMTU)

	dockerManager := NewFakeDockerManager(
		fakeDocker,
		fakeRecorder,
		proberesults.NewManager(),
		containerRefManager,
		&cadvisorapi.MachineInfo{},
		"",
		0, 0, "",
		&containertest.FakeOS{},
		networkPlugin,
		&fakeRuntimeHelper{},
		fakeHTTPClient,
		flowcontrol.NewBackOff(time.Second, 300*time.Second))

	return dockerManager, fakeDocker
}

func createTestDockerManagerWithFakeImageManager(fakeHTTPClient *fakeHTTP, fakeDocker *FakeDockerClient) (*DockerManager, *FakeDockerClient) {
	dm, fd := createTestDockerManager(fakeHTTPClient, fakeDocker)
	dm.imagePuller = newFakeImageManager()
	return dm, fd
}

func newTestDockerManagerWithRealImageManager() (*DockerManager, *FakeDockerClient) {
	return createTestDockerManager(nil, nil)
}
func newTestDockerManagerWithHTTPClient(fakeHTTPClient *fakeHTTP) (*DockerManager, *FakeDockerClient) {
	return createTestDockerManagerWithFakeImageManager(fakeHTTPClient, nil)
}

func newTestDockerManagerWithVersion(version, apiVersion string) (*DockerManager, *FakeDockerClient) {
	fakeDocker := NewFakeDockerClientWithVersion(version, apiVersion)
	return createTestDockerManagerWithFakeImageManager(nil, fakeDocker)
}

func newTestDockerManager() (*DockerManager, *FakeDockerClient) {
	return createTestDockerManagerWithFakeImageManager(nil, nil)
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
		container *v1.Container
		envs      []kubecontainer.EnvVar
		expected  *dockertypes.ContainerCreateConfig
	}{
		{
			name:      "none",
			container: &v1.Container{},
			expected: &dockertypes.ContainerCreateConfig{
				Config: &dockercontainer.Config{},
			},
		},
		{
			name: "command",
			container: &v1.Container{
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
			container: &v1.Container{
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
			container: &v1.Container{
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
			container: &v1.Container{
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
			container: &v1.Container{
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
			container: &v1.Container{
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
			ID:         kubetypes.UID("1234"),
			Name:       "qux",
			Namespace:  "new",
			Containers: []*kubecontainer.Container{containers[0], containers[1]},
		},
		{
			ID:         kubetypes.UID("5678"),
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

func TestDeleteImage(t *testing.T) {
	manager, fakeDocker := newTestDockerManager()
	fakeDocker.Image = &dockertypes.ImageInspect{ID: "1111", RepoTags: []string{"foo"}}
	manager.RemoveImage(kubecontainer.ImageSpec{Image: "1111"})
	fakeDocker.AssertCallDetails(NewCalledDetail("inspect_image", nil), NewCalledDetail("remove_image",
		[]interface{}{"1111", dockertypes.ImageRemoveOptions{PruneChildren: true}}))
}

func TestDeleteImageWithMultipleTags(t *testing.T) {
	manager, fakeDocker := newTestDockerManager()
	fakeDocker.Image = &dockertypes.ImageInspect{ID: "1111", RepoTags: []string{"foo", "bar"}}
	manager.RemoveImage(kubecontainer.ImageSpec{Image: "1111"})
	fakeDocker.AssertCallDetails(NewCalledDetail("inspect_image", nil),
		NewCalledDetail("remove_image", []interface{}{"foo", dockertypes.ImageRemoveOptions{PruneChildren: true}}),
		NewCalledDetail("remove_image", []interface{}{"bar", dockertypes.ImageRemoveOptions{PruneChildren: true}}))
}

func TestKillContainerInPod(t *testing.T) {
	manager, fakeDocker := newTestDockerManager()

	pod := makePod("qux", nil)
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
	pod := makePod("qux", &v1.PodSpec{
		Containers: []v1.Container{
			{
				Name: "foo",
				Lifecycle: &v1.Lifecycle{
					PreStop: &v1.Handler{
						Exec: &v1.ExecAction{
							Command: expectedCmd,
						},
					},
				},
			},
			{Name: "bar"}}})

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
					kubernetesPodLabel:                 string(podString),
					types.KubernetesContainerNameLabel: "foo",
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
	verifyCalls(t, fakeDocker, []string{"list", "inspect_container", "create_exec", "start_exec", "stop"})
	if !reflect.DeepEqual(expectedCmd, fakeDocker.execCmd) {
		t.Errorf("expected: %v, got %v", expectedCmd, fakeDocker.execCmd)
	}
}

func TestKillContainerInPodWithError(t *testing.T) {
	manager, fakeDocker := newTestDockerManager()

	pod := makePod("qux", nil)
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

func generatePodInfraContainerHash(pod *v1.Pod) uint64 {
	var ports []v1.ContainerPort
	if pod.Spec.SecurityContext == nil || !pod.Spec.HostNetwork {
		for _, container := range pod.Spec.Containers {
			ports = append(ports, container.Ports...)
		}
	}

	container := &v1.Container{
		Name:            PodInfraContainerName,
		Image:           "",
		Ports:           ports,
		ImagePullPolicy: podInfraContainerImagePullPolicy,
	}
	return kubecontainer.HashContainer(container)
}

// runSyncPod is a helper function to retrieve the running pods from the fake
// docker client and runs SyncPod for the given pod.
func runSyncPod(t *testing.T, dm *DockerManager, fakeDocker *FakeDockerClient, pod *v1.Pod, backOff *flowcontrol.Backoff, expectErr bool) kubecontainer.PodSyncResult {
	podStatus, err := dm.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fakeDocker.ClearCalls()
	if backOff == nil {
		backOff = flowcontrol.NewBackOff(time.Second, time.Minute)
	}
	// v1.PodStatus is not used in SyncPod now, pass in an empty one.
	result := dm.SyncPod(pod, v1.PodStatus{}, podStatus, []v1.Secret{}, backOff)
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

	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar"},
		},
	})

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
			break
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
	dm, fakeDocker := newTestDockerManagerWithRealImageManager()
	dm.podInfraContainerImage = "foo/infra_image:v1"
	puller := dm.dockerPuller.(*FakeDockerPuller)
	puller.HasImages = []string{}
	dm.podInfraContainerImage = "foo/infra_image:v1"
	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar", Image: "foo/something:v0", ImagePullPolicy: "IfNotPresent"},
		},
	})

	runSyncPod(t, dm, fakeDocker, pod, nil, false)

	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})

	fakeDocker.Lock()

	if !reflect.DeepEqual(puller.ImagesPulled, []string{"foo/infra_image:v1", "foo/something:v0"}) {
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
	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar"},
		},
	})

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
	pod := makePod("foo1", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar1"},
		},
	})
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
	pod := makePod("bar", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "foo"},
		},
	})

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
	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar"},
		},
	})

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
	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{{Name: "unhealthy"}},
	})

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
	container := v1.Container{Name: "bar"}
	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			container,
		},
	})
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
	containers := []v1.Container{
		{Name: "succeeded"},
		{Name: "failed"},
	}
	pod := makePod("foo", &v1.PodSpec{
		Containers: containers,
	})
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
		policy  v1.RestartPolicy
		calls   []string
		created []string
		stopped []string
	}{
		{
			v1.RestartPolicyAlways,
			[]string{
				// Restart both containers.
				"create", "start", "inspect_container", "create", "start", "inspect_container",
			},
			[]string{"succeeded", "failed"},
			[]string{},
		},
		{
			v1.RestartPolicyOnFailure,
			[]string{
				// Restart the failed container.
				"create", "start", "inspect_container",
			},
			[]string{"failed"},
			[]string{},
		},
		{
			v1.RestartPolicyNever,
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
	var fakeClock = clock.NewFakeClock(time.Now())
	startTime := fakeClock.Now()

	dm, fakeDocker := newTestDockerManager()
	containers := []v1.Container{
		{Name: "good"},
		{Name: "bad"},
	}
	pod := makePod("podfoo", &v1.PodSpec{
		Containers: containers,
	})

	stableId := "k8s_bad." + strconv.FormatUint(kubecontainer.HashContainer(&containers[1]), 16) + "_podfoo_new_12345678"
	dockerContainers := []*FakeContainer{
		{
			ID:        "9876",
			Name:      "/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_podfoo_new_12345678_0",
			StartedAt: startTime,
			Running:   true,
		},
		{
			ID:        "1234",
			Name:      "/k8s_good." + strconv.FormatUint(kubecontainer.HashContainer(&containers[0]), 16) + "_podfoo_new_12345678_0",
			StartedAt: startTime,
			Running:   true,
		},
		{
			ID:         "5678",
			Name:       "/k8s_bad." + strconv.FormatUint(kubecontainer.HashContainer(&containers[1]), 16) + "_podfoo_new_12345678_0",
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
	pod := *makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: containerName},
		},
		RestartPolicy: "Always",
	})
	pod.Status = v1.PodStatus{
		ContainerStatuses: []v1.ContainerStatus{
			{
				Name:         containerName,
				RestartCount: 3,
			},
		},
	}

	// Helper function for verifying the restart count.
	verifyRestartCount := func(pod *v1.Pod, expectedCount int) {
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

	killOneContainer := func(pod *v1.Pod) {
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
	containers := []v1.Container{
		{
			Name: "bar",
			TerminationMessagePath: "/dev/somepath",
		},
	}
	pod := makePod("foo", &v1.PodSpec{
		Containers: containers,
	})

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

	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			{
				Name: "bar",
				Lifecycle: &v1.Lifecycle{
					PostStart: &v1.Handler{
						HTTPGet: &v1.HTTPGetAction{
							Host: "foo",
							Port: intstr.FromInt(8080),
							Path: "bar",
						},
					},
				},
			},
		},
	})
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

	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar",
				Lifecycle: &v1.Lifecycle{
					PostStart: &v1.Handler{
						HTTPGet: &v1.HTTPGetAction{
							Host: "does.no.exist",
							Port: intstr.FromInt(8080),
							Path: "bar",
						},
					},
				},
			},
		},
	})

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
	container := v1.Container{
		Name: "bar",
		TerminationMessagePath: "/dev/somepath",
	}
	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			container,
		},
	})

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
	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar"},
		},
		HostNetwork: true,
	})

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
		container     *v1.Container
		inspectImage  *dockertypes.ImageInspect
		expectedError string
	}{
		// success cases
		"non-root runAsUser": {
			container: &v1.Container{
				SecurityContext: &v1.SecurityContext{
					RunAsUser: &nonRootUid,
				},
			},
		},
		"numeric non-root image user": {
			container: &v1.Container{},
			inspectImage: &dockertypes.ImageInspect{
				Config: &dockercontainer.Config{
					User: "1",
				},
			},
		},
		"numeric non-root image user with gid": {
			container: &v1.Container{},
			inspectImage: &dockertypes.ImageInspect{
				Config: &dockercontainer.Config{
					User: "1:2",
				},
			},
		},

		// failure cases
		"root runAsUser": {
			container: &v1.Container{
				SecurityContext: &v1.SecurityContext{
					RunAsUser: &rootUid,
				},
			},
			expectedError: "container's runAsUser breaks non-root policy",
		},
		"non-numeric image user": {
			container: &v1.Container{},
			inspectImage: &dockertypes.ImageInspect{
				Config: &dockercontainer.Config{
					User: "foo",
				},
			},
			expectedError: "non-numeric user",
		},
		"numeric root image user": {
			container: &v1.Container{},
			inspectImage: &dockertypes.ImageInspect{
				Config: &dockercontainer.Config{
					User: "0",
				},
			},
			expectedError: "container has no runAsUser and image will run as root",
		},
		"numeric root image user with gid": {
			container: &v1.Container{},
			inspectImage: &dockertypes.ImageInspect{
				Config: &dockercontainer.Config{
					User: "0:1",
				},
			},
			expectedError: "container has no runAsUser and image will run as root",
		},
		"nil image in inspect": {
			container:     &v1.Container{},
			inspectImage:  nil,
			expectedError: "unable to inspect image",
		},
		"nil config in image inspect": {
			container:     &v1.Container{},
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

func TestGetUserFromImageUser(t *testing.T) {
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
		"root username": {
			input:  "root:root",
			expect: "root",
		},
		"username": {
			input:  "test:test",
			expect: "test",
		},
	}
	for k, v := range tests {
		actual := GetUserFromImageUser(v.input)
		if actual != v.expect {
			t.Errorf("%s failed.  Expected %s but got %s", k, v.expect, actual)
		}
	}
}

func TestGetPidMode(t *testing.T) {
	// test false
	pod := &v1.Pod{}
	pidMode := getPidMode(pod)

	if pidMode != "" {
		t.Errorf("expected empty pid mode for pod but got %v", pidMode)
	}

	// test true
	pod.Spec.SecurityContext = &v1.PodSecurityContext{}
	pod.Spec.HostPID = true
	pidMode = getPidMode(pod)
	if pidMode != "host" {
		t.Errorf("expected host pid mode for pod but got %v", pidMode)
	}
}

func TestGetIPCMode(t *testing.T) {
	// test false
	pod := &v1.Pod{}
	ipcMode := getIPCMode(pod)

	if ipcMode != "" {
		t.Errorf("expected empty ipc mode for pod but got %v", ipcMode)
	}

	// test true
	pod.Spec.SecurityContext = &v1.PodSecurityContext{}
	pod.Spec.HostIPC = true
	ipcMode = getIPCMode(pod)
	if ipcMode != "host" {
		t.Errorf("expected host ipc mode for pod but got %v", ipcMode)
	}
}

func TestSyncPodWithPullPolicy(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithRealImageManager()
	puller := dm.dockerPuller.(*FakeDockerPuller)
	puller.HasImages = []string{"foo/existing_one:v1", "foo/want:latest"}
	dm.podInfraContainerImage = "foo/infra_image:v1"

	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar", Image: "foo/pull_always_image:v1", ImagePullPolicy: v1.PullAlways},
			{Name: "bar2", Image: "foo/pull_if_not_present_image:v1", ImagePullPolicy: v1.PullIfNotPresent},
			{Name: "bar3", Image: "foo/existing_one:v1", ImagePullPolicy: v1.PullIfNotPresent},
			{Name: "bar4", Image: "foo/want:latest", ImagePullPolicy: v1.PullIfNotPresent},
			{Name: "bar5", Image: "foo/pull_never_image:v1", ImagePullPolicy: v1.PullNever},
		},
	})

	expectedResults := []*kubecontainer.SyncResult{
		//Sync result for infra container
		{kubecontainer.StartContainer, PodInfraContainerName, nil, ""},
		{kubecontainer.SetupNetwork, kubecontainer.GetPodFullName(pod), nil, ""},
		//Sync result for user containers
		{kubecontainer.StartContainer, "bar", nil, ""},
		{kubecontainer.StartContainer, "bar2", nil, ""},
		{kubecontainer.StartContainer, "bar3", nil, ""},
		{kubecontainer.StartContainer, "bar4", nil, ""},
		{kubecontainer.StartContainer, "bar5", images.ErrImageNeverPull,
			"Container image \"foo/pull_never_image:v1\" is not present with pull policy of Never"},
	}

	result := runSyncPod(t, dm, fakeDocker, pod, nil, true)
	verifySyncResults(t, expectedResults, result)

	fakeDocker.Lock()
	defer fakeDocker.Unlock()

	pulledImageSorted := puller.ImagesPulled[:]
	sort.Strings(pulledImageSorted)
	assert.Equal(t, []string{"foo/infra_image:v1", "foo/pull_always_image:v1", "foo/pull_if_not_present_image:v1"}, pulledImageSorted)

	if len(fakeDocker.Created) != 5 {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
}

// This test only covers SyncPod with PullImageFailure, CreateContainerFailure and StartContainerFailure.
// There are still quite a few failure cases not covered.
// TODO(random-liu): Better way to test the SyncPod failures.
func TestSyncPodWithFailure(t *testing.T) {
	pod := makePod("foo", nil)
	tests := map[string]struct {
		container   v1.Container
		dockerError map[string]error
		pullerError []error
		expected    []*kubecontainer.SyncResult
	}{
		"PullImageFailure": {
			v1.Container{Name: "bar", Image: "foo/real_image:v1", ImagePullPolicy: v1.PullAlways},
			map[string]error{},
			[]error{fmt.Errorf("can't pull image")},
			[]*kubecontainer.SyncResult{{kubecontainer.StartContainer, "bar", images.ErrImagePull, "can't pull image"}},
		},
		"CreateContainerFailure": {
			v1.Container{Name: "bar", Image: "foo/already_present:v2"},
			map[string]error{"create": fmt.Errorf("can't create container")},
			[]error{},
			[]*kubecontainer.SyncResult{{kubecontainer.StartContainer, "bar", kubecontainer.ErrRunContainer, "can't create container"}},
		},
		"StartContainerFailure": {
			v1.Container{Name: "bar", Image: "foo/already_present:v2"},
			map[string]error{"start": fmt.Errorf("can't start container")},
			[]error{},
			[]*kubecontainer.SyncResult{{kubecontainer.StartContainer, "bar", kubecontainer.ErrRunContainer, "can't start container"}},
		},
	}

	for _, test := range tests {
		dm, fakeDocker := newTestDockerManagerWithRealImageManager()
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
		pod.Spec.Containers = []v1.Container{test.container}
		result := runSyncPod(t, dm, fakeDocker, pod, nil, true)
		verifySyncResults(t, test.expected, result)
	}
}

// Verify whether all the expected results appear exactly only once in real result.
func verifySyncResults(t *testing.T, expectedResults []*kubecontainer.SyncResult, realResult kubecontainer.PodSyncResult) {
	if len(expectedResults) != len(realResult.SyncResults) {
		t.Errorf("expected sync result number %d, got %d", len(expectedResults), len(realResult.SyncResults))
		for _, r := range expectedResults {
			t.Errorf("expected result: %#v", r)
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
					t.Errorf("expected sync result %#v, got %+v", expectR, realR)
				}
				found++
			}
		}
		if found == 0 {
			t.Errorf("not found expected result %#v", expectR)
		}
		if found > 1 {
			t.Errorf("got %d duplicate expected result %#v", found, expectR)
		}
	}
}

func TestSecurityOptsOperator(t *testing.T) {
	dm110, _ := newTestDockerManagerWithVersion("1.10.1", "1.22")
	dm111, _ := newTestDockerManagerWithVersion("1.11.0", "1.23")

	secOpts := []dockerOpt{{"seccomp", "unconfined", ""}}
	opts, err := dm110.fmtDockerOpts(secOpts)
	if err != nil {
		t.Fatalf("error getting security opts for Docker 1.10: %v", err)
	}
	if expected := []string{"seccomp:unconfined"}; len(opts) != 1 || opts[0] != expected[0] {
		t.Fatalf("security opts for Docker 1.10: expected %v, got: %v", expected, opts)
	}

	opts, err = dm111.fmtDockerOpts(secOpts)
	if err != nil {
		t.Fatalf("error getting security opts for Docker 1.11: %v", err)
	}
	if expected := []string{"seccomp=unconfined"}; len(opts) != 1 || opts[0] != expected[0] {
		t.Fatalf("security opts for Docker 1.11: expected %v, got: %v", expected, opts)
	}
}

func TestGetSecurityOpts(t *testing.T) {
	const containerName = "bar"
	pod := func(annotations map[string]string) *v1.Pod {
		p := makePod("foo", &v1.PodSpec{
			Containers: []v1.Container{
				{Name: containerName},
			},
		})
		p.Annotations = annotations
		return p
	}

	tests := []struct {
		msg          string
		pod          *v1.Pod
		expectedOpts []string
	}{{
		msg:          "No security annotations",
		pod:          pod(nil),
		expectedOpts: []string{"seccomp=unconfined"},
	}, {
		msg: "Seccomp default",
		pod: pod(map[string]string{
			v1.SeccompContainerAnnotationKeyPrefix + containerName: "docker/default",
		}),
		expectedOpts: nil,
	}, {
		msg: "AppArmor runtime/default",
		pod: pod(map[string]string{
			apparmor.ContainerAnnotationKeyPrefix + containerName: apparmor.ProfileRuntimeDefault,
		}),
		expectedOpts: []string{"seccomp=unconfined"},
	}, {
		msg: "AppArmor local profile",
		pod: pod(map[string]string{
			apparmor.ContainerAnnotationKeyPrefix + containerName: apparmor.ProfileNamePrefix + "foo",
		}),
		expectedOpts: []string{"seccomp=unconfined", "apparmor=foo"},
	}, {
		msg: "AppArmor and seccomp profile",
		pod: pod(map[string]string{
			v1.SeccompContainerAnnotationKeyPrefix + containerName: "docker/default",
			apparmor.ContainerAnnotationKeyPrefix + containerName:  apparmor.ProfileNamePrefix + "foo",
		}),
		expectedOpts: []string{"apparmor=foo"},
	}}

	dm, _ := newTestDockerManagerWithVersion("1.11.1", "1.23")
	for i, test := range tests {
		securityOpts, err := dm.getSecurityOpts(test.pod, containerName)
		assert.NoError(t, err, "TestCase[%d]: %s", i, test.msg)
		opts, err := dm.fmtDockerOpts(securityOpts)
		assert.NoError(t, err, "TestCase[%d]: %s", i, test.msg)
		assert.Len(t, opts, len(test.expectedOpts), "TestCase[%d]: %s", i, test.msg)
		for _, opt := range test.expectedOpts {
			assert.Contains(t, opts, opt, "TestCase[%d]: %s", i, test.msg)
		}
	}
}

func TestSeccompIsUnconfinedByDefaultWithDockerV110(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithVersion("1.10.1", "1.22")
	// We want to capture events.
	recorder := record.NewFakeRecorder(20)
	dm.recorder = recorder

	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar"},
		},
	})

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
	assert.Contains(t, newContainer.HostConfig.SecurityOpt, "seccomp:unconfined", "Pods with Docker versions >= 1.10 must not have seccomp disabled by default")

	cid := utilstrings.ShortenString(fakeDocker.Created[1], 12)
	assert.NoError(t, expectEvent(recorder, v1.EventTypeNormal, events.CreatedContainer,
		fmt.Sprintf("Created container with docker id %s; Security:[seccomp=unconfined]", cid)))
}

func TestUnconfinedSeccompProfileWithDockerV110(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithVersion("1.10.1", "1.22")
	pod := makePod("foo4", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar4"},
		},
	})
	pod.Annotations = map[string]string{
		v1.SeccompPodAnnotationKey: "unconfined",
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
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo4_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_bar4\\.[a-f0-9]+_foo4_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()

	newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	assert.Contains(t, newContainer.HostConfig.SecurityOpt, "seccomp:unconfined", "Pods created with a secccomp annotation of unconfined should have seccomp:unconfined.")
}

func TestDefaultSeccompProfileWithDockerV110(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithVersion("1.10.1", "1.22")
	pod := makePod("foo1", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar1"},
		},
	})
	pod.Annotations = map[string]string{
		v1.SeccompPodAnnotationKey: "docker/default",
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
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo1_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_bar1\\.[a-f0-9]+_foo1_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()

	newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	assert.NotContains(t, newContainer.HostConfig.SecurityOpt, "seccomp:unconfined", "Pods created with a secccomp annotation of docker/default should have empty security opt.")
}

func TestSeccompContainerAnnotationTrumpsPod(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithVersion("1.10.1", "1.22")
	pod := makePod("foo2", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar2"},
		},
	})
	pod.Annotations = map[string]string{
		v1.SeccompPodAnnotationKey:                      "unconfined",
		v1.SeccompContainerAnnotationKeyPrefix + "bar2": "docker/default",
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
		!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo2_new_", fakeDocker.Created[0]) ||
		!matchString(t, "/k8s_bar2\\.[a-f0-9]+_foo2_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()

	newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	assert.NotContains(t, newContainer.HostConfig.SecurityOpt, "seccomp:unconfined", "Container annotation should trump the pod annotation for seccomp.")
}

func TestSeccompLocalhostProfileIsLoaded(t *testing.T) {
	tests := []struct {
		annotations    map[string]string
		expectedSecOpt string
		expectedSecMsg string
		expectedError  string
	}{
		{
			annotations: map[string]string{
				v1.SeccompPodAnnotationKey: "localhost/test",
			},
			expectedSecOpt: `seccomp={"foo":"bar"}`,
			expectedSecMsg: "seccomp=test(md5:21aeae45053385adebd25311f9dd9cb1)",
		},
		{
			annotations: map[string]string{
				v1.SeccompPodAnnotationKey: "localhost/sub/subtest",
			},
			expectedSecOpt: `seccomp={"abc":"def"}`,
			expectedSecMsg: "seccomp=sub/subtest(md5:07c9bcb4db631f7ca191d6e0bca49f76)",
		},
		{
			annotations: map[string]string{
				v1.SeccompPodAnnotationKey: "localhost/not-existing",
			},
			expectedError: "cannot load seccomp profile",
		},
	}

	for i, test := range tests {
		dm, fakeDocker := newTestDockerManagerWithVersion("1.11.0", "1.23")
		// We want to capture events.
		recorder := record.NewFakeRecorder(20)
		dm.recorder = recorder

		_, filename, _, _ := goruntime.Caller(0)
		dm.seccompProfileRoot = path.Join(path.Dir(filename), "fixtures", "seccomp")

		pod := makePod("foo2", &v1.PodSpec{
			Containers: []v1.Container{
				{Name: "bar2"},
			},
		})
		pod.Annotations = test.annotations

		result := runSyncPod(t, dm, fakeDocker, pod, nil, test.expectedError != "")
		if test.expectedError != "" {
			assert.Contains(t, result.Error().Error(), test.expectedError)
			continue
		}

		verifyCalls(t, fakeDocker, []string{
			// Create pod infra container.
			"create", "start", "inspect_container", "inspect_container",
			// Create container.
			"create", "start", "inspect_container",
		})

		fakeDocker.Lock()
		if len(fakeDocker.Created) != 2 ||
			!matchString(t, "/k8s_POD\\.[a-f0-9]+_foo2_new_", fakeDocker.Created[0]) ||
			!matchString(t, "/k8s_bar2\\.[a-f0-9]+_foo2_new_", fakeDocker.Created[1]) {
			t.Errorf("unexpected containers created %v", fakeDocker.Created)
		}
		fakeDocker.Unlock()

		newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		assert.Contains(t, newContainer.HostConfig.SecurityOpt, test.expectedSecOpt, "The compacted seccomp json profile should be loaded.")

		cid := utilstrings.ShortenString(fakeDocker.Created[1], 12)
		assert.NoError(t, expectEvent(recorder, v1.EventTypeNormal, events.CreatedContainer,
			fmt.Sprintf("Created container with docker id %s; Security:[%s]", cid, test.expectedSecMsg)),
			"testcase %d", i)
	}
}

func TestSecurityOptsAreNilWithDockerV19(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithVersion("1.9.1", "1.21")
	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar"},
		},
	})

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

func TestCreateAppArmorContanier(t *testing.T) {
	dm, fakeDocker := newTestDockerManagerWithVersion("1.11.1", "1.23")
	// We want to capture events.
	recorder := record.NewFakeRecorder(20)
	dm.recorder = recorder

	pod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
			Annotations: map[string]string{
				apparmor.ContainerAnnotationKeyPrefix + "test": apparmor.ProfileNamePrefix + "test-profile",
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Name: "test"},
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
		!matchString(t, "/k8s_test\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()

	// Verify security opts.
	newContainer, err := fakeDocker.InspectContainer(fakeDocker.Created[1])
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	securityOpts := newContainer.HostConfig.SecurityOpt
	assert.Contains(t, securityOpts, "apparmor=test-profile", "Container should have apparmor security opt")

	cid := utilstrings.ShortenString(fakeDocker.Created[1], 12)
	assert.NoError(t, expectEvent(recorder, v1.EventTypeNormal, events.CreatedContainer,
		fmt.Sprintf("Created container with docker id %s; Security:[seccomp=unconfined apparmor=test-profile]", cid)))
}

func expectEvent(recorder *record.FakeRecorder, eventType, reason, msg string) error {
	expected := fmt.Sprintf("%s %s %s", eventType, reason, msg)
	var events []string
	// Drain the event channel.
	for {
		select {
		case event := <-recorder.Events:
			if event == expected {
				return nil
			}
			events = append(events, event)
		default:
			// No more events!
			return fmt.Errorf("Event %q not found in [%s]", expected, strings.Join(events, ", "))
		}
	}
}

func TestNewDockerVersion(t *testing.T) {
	cases := []struct {
		value string
		out   string
		err   bool
	}{
		{value: "1", err: true},
		{value: "1.8", err: true},
		{value: "1.8.1", out: "1.8.1"},
		{value: "1.8.1-fc21.other", out: "1.8.1-fc21.other"},
		{value: "1.8.1-beta.12", out: "1.8.1-beta.12"},
	}
	for _, test := range cases {
		v, err := newDockerVersion(test.value)
		switch {
		case err != nil && test.err:
			continue
		case (err != nil) != test.err:
			t.Errorf("error for %q: expected %t, got %v", test.value, test.err, err)
			continue
		}
		if v.String() != test.out {
			t.Errorf("unexpected parsed version %q for %q", v, test.value)
		}
	}
}

func TestDockerVersionComparison(t *testing.T) {
	v, err := newDockerVersion("1.10.3")
	assert.NoError(t, err)
	for i, test := range []struct {
		version string
		compare int
		err     bool
	}{
		{version: "1.9.2", compare: 1},
		{version: "1.9.2-rc2", compare: 1},
		{version: "1.10.3", compare: 0},
		{version: "1.10.3-rc3", compare: 1},
		{version: "1.10.4", compare: -1},
		{version: "1.10.4-rc1", compare: -1},
		{version: "1.11.1", compare: -1},
		{version: "1.11.1-rc4", compare: -1},
		{version: "invalid", compare: -1, err: true},
	} {
		testCase := fmt.Sprintf("test case #%d test version %q", i, test.version)
		res, err := v.Compare(test.version)
		assert.Equal(t, test.compare, res, testCase)
		assert.Equal(t, test.err, err != nil, testCase)
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
	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{{Name: "nosuchcontainer"}},
	})

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

func TestPruneInitContainers(t *testing.T) {
	dm, fake := newTestDockerManager()
	pod := makePod("", &v1.PodSpec{
		InitContainers: []v1.Container{
			{Name: "init1"},
			{Name: "init2"},
		},
	})
	status := &kubecontainer.PodStatus{
		ContainerStatuses: []*kubecontainer.ContainerStatus{
			{Name: "init2", ID: kubecontainer.ContainerID{ID: "init2-new-1"}, State: kubecontainer.ContainerStateExited},
			{Name: "init1", ID: kubecontainer.ContainerID{ID: "init1-new-1"}, State: kubecontainer.ContainerStateExited},
			{Name: "init1", ID: kubecontainer.ContainerID{ID: "init1-new-2"}, State: kubecontainer.ContainerStateExited},
			{Name: "init1", ID: kubecontainer.ContainerID{ID: "init1-old-1"}, State: kubecontainer.ContainerStateExited},
			{Name: "init2", ID: kubecontainer.ContainerID{ID: "init2-old-1"}, State: kubecontainer.ContainerStateExited},
		},
	}
	fake.ExitedContainerList = []dockertypes.Container{
		{ID: "init1-new-1"},
		{ID: "init1-new-2"},
		{ID: "init1-old-1"},
		{ID: "init2-new-1"},
		{ID: "init2-old-1"},
	}
	keep := map[kubecontainer.DockerID]int{}
	dm.pruneInitContainersBeforeStart(pod, status, keep)
	sort.Sort(sort.StringSlice(fake.Removed))
	if !reflect.DeepEqual([]string{"init1-new-2", "init1-old-1", "init2-old-1"}, fake.Removed) {
		t.Fatal(fake.Removed)
	}
}

func TestGetPodStatusFromNetworkPlugin(t *testing.T) {
	cases := []struct {
		pod                *v1.Pod
		fakePodIP          string
		containerID        string
		infraContainerID   string
		networkStatusError error
		expectRunning      bool
		expectUnknown      bool
	}{
		{
			pod: &v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					UID:       "12345678",
					Name:      "foo",
					Namespace: "new",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container"}},
				},
			},
			fakePodIP:          "10.10.10.10",
			containerID:        "123",
			infraContainerID:   "9876",
			networkStatusError: nil,
			expectRunning:      true,
			expectUnknown:      false,
		},
		{
			pod: &v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					UID:       "12345678",
					Name:      "foo",
					Namespace: "new",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "container"}},
				},
			},
			fakePodIP:          "",
			containerID:        "123",
			infraContainerID:   "9876",
			networkStatusError: fmt.Errorf("CNI plugin error"),
			expectRunning:      false,
			expectUnknown:      true,
		},
	}
	for _, test := range cases {
		dm, fakeDocker := newTestDockerManager()
		ctrl := gomock.NewController(t)
		fnp := mock_network.NewMockNetworkPlugin(ctrl)
		dm.networkPlugin = fnp

		fakeDocker.SetFakeRunningContainers([]*FakeContainer{
			{
				ID:      test.containerID,
				Name:    fmt.Sprintf("/k8s_container_%s_%s_%s_42", test.pod.Name, test.pod.Namespace, test.pod.UID),
				Running: true,
			},
			{
				ID:      test.infraContainerID,
				Name:    fmt.Sprintf("/k8s_POD.%s_%s_%s_%s_42", strconv.FormatUint(generatePodInfraContainerHash(test.pod), 16), test.pod.Name, test.pod.Namespace, test.pod.UID),
				Running: true,
			},
		})

		fnp.EXPECT().Name().Return("someNetworkPlugin").AnyTimes()
		var podNetworkStatus *network.PodNetworkStatus
		if test.fakePodIP != "" {
			podNetworkStatus = &network.PodNetworkStatus{IP: net.ParseIP(test.fakePodIP)}
		}
		fnp.EXPECT().GetPodNetworkStatus(test.pod.Namespace, test.pod.Name, kubecontainer.DockerID(test.infraContainerID).ContainerID()).Return(podNetworkStatus, test.networkStatusError)

		podStatus, err := dm.GetPodStatus(test.pod.UID, test.pod.Name, test.pod.Namespace)
		if err != nil {
			t.Fatal(err)
		}
		if podStatus.IP != test.fakePodIP {
			t.Errorf("Got wrong ip, expected %v, got %v", test.fakePodIP, podStatus.IP)
		}

		expectedStatesCount := 0
		var expectedState kubecontainer.ContainerState
		if test.expectRunning {
			expectedState = kubecontainer.ContainerStateRunning
		} else if test.expectUnknown {
			expectedState = kubecontainer.ContainerStateUnknown
		} else {
			t.Errorf("Some state has to be expected")
		}
		for _, containerStatus := range podStatus.ContainerStatuses {
			if containerStatus.State == expectedState {
				expectedStatesCount++
			}
		}
		if expectedStatesCount < 1 {
			t.Errorf("Invalid count of containers with expected state")
		}
	}
}

func TestSyncPodGetsPodIPFromNetworkPlugin(t *testing.T) {
	const (
		containerID      = "123"
		infraContainerID = "9876"
		fakePodIP        = "10.10.10.10"
	)
	dm, fakeDocker := newTestDockerManager()
	dm.podInfraContainerImage = "pod_infra_image"
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()
	fnp := mock_network.NewMockNetworkPlugin(ctrl)
	dm.networkPlugin = fnp

	pod := makePod("foo", &v1.PodSpec{
		Containers: []v1.Container{
			{Name: "bar"},
		},
	})

	// Can be called multiple times due to GetPodStatus
	fnp.EXPECT().Name().Return("someNetworkPlugin").AnyTimes()
	fnp.EXPECT().GetPodNetworkStatus("new", "foo", gomock.Any()).Return(&network.PodNetworkStatus{IP: net.ParseIP(fakePodIP)}, nil).AnyTimes()
	fnp.EXPECT().SetUpPod("new", "foo", gomock.Any()).Return(nil)

	runSyncPod(t, dm, fakeDocker, pod, nil, false)
	verifyCalls(t, fakeDocker, []string{
		// Create pod infra container.
		"create", "start", "inspect_container", "inspect_container",
		// Create container.
		"create", "start", "inspect_container",
	})
}

// only test conditions "if inspect == nil || inspect.Config == nil || inspect.Config.Labels == nil" now
func TestContainerAndPodFromLabels(t *testing.T) {
	tests := []struct {
		inspect       *dockertypes.ContainerJSON
		expectedError error
	}{
		{
			inspect:       nil,
			expectedError: errNoPodOnContainer,
		},
		{
			inspect:       &dockertypes.ContainerJSON{},
			expectedError: errNoPodOnContainer,
		},
		{
			inspect: &dockertypes.ContainerJSON{
				Config: &dockercontainer.Config{
					Hostname: "foo",
				},
			},
			expectedError: errNoPodOnContainer,
		},
	}

	for k, v := range tests {
		pod, container, err := containerAndPodFromLabels(v.inspect)
		if pod != nil || container != nil || err != v.expectedError {
			t.Errorf("case[%q]: expected: nil, nil, %v, got: %v, %v, %v", k, v.expectedError, pod, container, err)
		}
	}
}

func makePod(name string, spec *v1.PodSpec) *v1.Pod {
	if spec == nil {
		spec = &v1.PodSpec{Containers: []v1.Container{{Name: "foo"}, {Name: "bar"}}}
	}
	pod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			UID:       "12345678",
			Name:      name,
			Namespace: "new",
		},
		Spec: *spec,
	}
	return pod
}
