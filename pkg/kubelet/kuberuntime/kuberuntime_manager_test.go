/*
Copyright 2016 The Kubernetes Authors.

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

package kuberuntime

import (
	"reflect"
	"sort"
	"testing"
	"time"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	apitest "k8s.io/kubernetes/pkg/kubelet/api/testing"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/network"
	nettest "k8s.io/kubernetes/pkg/kubelet/network/testing"
	"k8s.io/kubernetes/pkg/types"
	kubetypes "k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

var (
	fakeCreatedAt int64 = 1
)

func createTestRuntimeManager() (*apitest.FakeRuntimeService, *apitest.FakeImageService, *kubeGenericRuntimeManager, error) {
	fakeRuntimeService := apitest.NewFakeRuntimeService()
	fakeImageService := apitest.NewFakeImageService()
	// Only an empty machineInfo is needed here, because in unit test all containers are besteffort,
	// data in machineInfo is not used. If burstable containers are used in unit test in the future,
	// we may want to set memory capacity.
	machineInfo := &cadvisorapi.MachineInfo{}
	networkPlugin, _ := network.InitNetworkPlugin(
		[]network.NetworkPlugin{},
		"",
		nettest.NewFakeHost(nil),
		componentconfig.HairpinNone,
		"10.0.0.0/8",
		network.UseDefaultMTU,
	)
	osInterface := &containertest.FakeOS{}
	manager, err := NewFakeKubeRuntimeManager(fakeRuntimeService, fakeImageService, machineInfo, networkPlugin, osInterface)
	return fakeRuntimeService, fakeImageService, manager, err
}

// sandboxTemplate is a sandbox template to create fake sandbox.
type sandboxTemplate struct {
	pod       *api.Pod
	attempt   uint32
	createdAt int64
	state     runtimeApi.PodSandboxState
}

// containerTemplate is a container template to create fake container.
type containerTemplate struct {
	pod            *api.Pod
	container      *api.Container
	sandboxAttempt uint32
	attempt        int
	createdAt      int64
	state          runtimeApi.ContainerState
}

// makeAndSetFakePod is a helper function to create and set one fake sandbox for a pod and
// one fake container for each of its container.
func makeAndSetFakePod(t *testing.T, m *kubeGenericRuntimeManager, fakeRuntime *apitest.FakeRuntimeService,
	pod *api.Pod) (*apitest.FakePodSandbox, []*apitest.FakeContainer) {
	sandbox := makeFakePodSandbox(t, m, sandboxTemplate{
		pod:       pod,
		createdAt: fakeCreatedAt,
		state:     runtimeApi.PodSandboxState_SANDBOX_READY,
	})

	var containers []*apitest.FakeContainer
	newTemplate := func(c *api.Container) containerTemplate {
		return containerTemplate{
			pod:       pod,
			container: c,
			createdAt: fakeCreatedAt,
			state:     runtimeApi.ContainerState_CONTAINER_RUNNING,
		}
	}
	for i := range pod.Spec.Containers {
		containers = append(containers, makeFakeContainer(t, m, newTemplate(&pod.Spec.Containers[i])))
	}
	for i := range pod.Spec.InitContainers {
		containers = append(containers, makeFakeContainer(t, m, newTemplate(&pod.Spec.InitContainers[i])))
	}

	fakeRuntime.SetFakeSandboxes([]*apitest.FakePodSandbox{sandbox})
	fakeRuntime.SetFakeContainers(containers)
	return sandbox, containers
}

// makeFakePodSandbox creates a fake pod sandbox based on a sandbox template.
func makeFakePodSandbox(t *testing.T, m *kubeGenericRuntimeManager, template sandboxTemplate) *apitest.FakePodSandbox {
	config, err := m.generatePodSandboxConfig(template.pod, template.attempt)
	assert.NoError(t, err, "generatePodSandboxConfig for sandbox template %+v", template)

	podSandboxID := apitest.BuildSandboxName(config.Metadata)
	return &apitest.FakePodSandbox{
		PodSandboxStatus: runtimeApi.PodSandboxStatus{
			Id:        &podSandboxID,
			Metadata:  config.Metadata,
			State:     &template.state,
			CreatedAt: &template.createdAt,
			Network: &runtimeApi.PodSandboxNetworkStatus{
				Ip: &apitest.FakePodSandboxIP,
			},
			Labels: config.Labels,
		},
	}
}

// makeFakePodSandboxes creates a group of fake pod sandboxes based on the sandbox templates.
// The function guarantees the order of the fake pod sandboxes is the same with the templates.
func makeFakePodSandboxes(t *testing.T, m *kubeGenericRuntimeManager, templates []sandboxTemplate) []*apitest.FakePodSandbox {
	var fakePodSandboxes []*apitest.FakePodSandbox
	for _, template := range templates {
		fakePodSandboxes = append(fakePodSandboxes, makeFakePodSandbox(t, m, template))
	}
	return fakePodSandboxes
}

// makeFakeContainer creates a fake container based on a container template.
func makeFakeContainer(t *testing.T, m *kubeGenericRuntimeManager, template containerTemplate) *apitest.FakeContainer {
	sandboxConfig, err := m.generatePodSandboxConfig(template.pod, template.sandboxAttempt)
	assert.NoError(t, err, "generatePodSandboxConfig for container template %+v", template)

	containerConfig, err := m.generateContainerConfig(template.container, template.pod, template.attempt, "")
	assert.NoError(t, err, "generateContainerConfig for container template %+v", template)

	podSandboxID := apitest.BuildSandboxName(sandboxConfig.Metadata)
	containerID := apitest.BuildContainerName(containerConfig.Metadata, podSandboxID)
	imageRef := containerConfig.Image.GetImage()
	return &apitest.FakeContainer{
		ContainerStatus: runtimeApi.ContainerStatus{
			Id:          &containerID,
			Metadata:    containerConfig.Metadata,
			Image:       containerConfig.Image,
			ImageRef:    &imageRef,
			CreatedAt:   &template.createdAt,
			State:       &template.state,
			Labels:      containerConfig.Labels,
			Annotations: containerConfig.Annotations,
		},
		SandboxID: podSandboxID,
	}
}

// makeFakeContainers creates a group of fake containers based on the container templates.
// The function guarantees the order of the fake containers is the same with the templates.
func makeFakeContainers(t *testing.T, m *kubeGenericRuntimeManager, templates []containerTemplate) []*apitest.FakeContainer {
	var fakeContainers []*apitest.FakeContainer
	for _, template := range templates {
		fakeContainers = append(fakeContainers, makeFakeContainer(t, m, template))
	}
	return fakeContainers
}

// makeTestContainer creates a test api container.
func makeTestContainer(name, image string) api.Container {
	return api.Container{
		Name:  name,
		Image: image,
	}
}

// makeTestPod creates a test api pod.
func makeTestPod(podName, podNamespace, podUID string, containers []api.Container) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       types.UID(podUID),
			Name:      podName,
			Namespace: podNamespace,
		},
		Spec: api.PodSpec{
			Containers: containers,
		},
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

func verifyFakeContainerList(fakeRuntime *apitest.FakeRuntimeService, expected []string) ([]string, bool) {
	actual := []string{}
	for _, c := range fakeRuntime.Containers {
		actual = append(actual, c.GetId())
	}
	sort.Sort(sort.StringSlice(actual))
	sort.Sort(sort.StringSlice(expected))

	return actual, reflect.DeepEqual(expected, actual)
}

func TestNewKubeRuntimeManager(t *testing.T) {
	_, _, _, err := createTestRuntimeManager()
	assert.NoError(t, err)
}

func TestVersion(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	version, err := m.Version()
	assert.NoError(t, err)
	assert.Equal(t, kubeRuntimeAPIVersion, version.String())
}

func TestContainerRuntimeType(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	runtimeType := m.Type()
	assert.Equal(t, apitest.FakeRuntimeName, runtimeType)
}

func TestGetPodStatus(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	containers := []api.Container{
		{
			Name:            "foo1",
			Image:           "busybox",
			ImagePullPolicy: api.PullIfNotPresent,
		},
		{
			Name:            "foo2",
			Image:           "busybox",
			ImagePullPolicy: api.PullIfNotPresent,
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

	// Set fake sandbox and faked containers to fakeRuntime.
	makeAndSetFakePod(t, m, fakeRuntime, pod)

	podStatus, err := m.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	assert.Equal(t, pod.UID, podStatus.ID)
	assert.Equal(t, pod.Name, podStatus.Name)
	assert.Equal(t, pod.Namespace, podStatus.Namespace)
	assert.Equal(t, apitest.FakePodSandboxIP, podStatus.IP)
}

func TestGetPods(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "foo1",
					Image: "busybox",
				},
				{
					Name:  "foo2",
					Image: "busybox",
				},
			},
		},
	}

	// Set fake sandbox and fake containers to fakeRuntime.
	fakeSandbox, fakeContainers := makeAndSetFakePod(t, m, fakeRuntime, pod)

	// Convert the fakeContainers to kubecontainer.Container
	containers := make([]*kubecontainer.Container, len(fakeContainers))
	for i := range containers {
		fakeContainer := fakeContainers[i]
		c, err := m.toKubeContainer(&runtimeApi.Container{
			Id:          fakeContainer.Id,
			Metadata:    fakeContainer.Metadata,
			State:       fakeContainer.State,
			Image:       fakeContainer.Image,
			ImageRef:    fakeContainer.ImageRef,
			Labels:      fakeContainer.Labels,
			Annotations: fakeContainer.Annotations,
		})
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		containers[i] = c
	}
	// Convert fakeSandbox to kubecontainer.Container
	sandbox, err := m.sandboxToKubeContainer(&runtimeApi.PodSandbox{
		Id:          fakeSandbox.Id,
		Metadata:    fakeSandbox.Metadata,
		State:       fakeSandbox.State,
		CreatedAt:   fakeSandbox.CreatedAt,
		Labels:      fakeSandbox.Labels,
		Annotations: fakeSandbox.Annotations,
	})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	expected := []*kubecontainer.Pod{
		{
			ID:         kubetypes.UID("12345678"),
			Name:       "foo",
			Namespace:  "new",
			Containers: []*kubecontainer.Container{containers[0], containers[1]},
			Sandboxes:  []*kubecontainer.Container{sandbox},
		},
	}

	actual, err := m.GetPods(false)
	assert.NoError(t, err)

	if !verifyPods(expected, actual) {
		t.Errorf("expected %q, got %q", expected, actual)
	}
}

func TestGetPodContainerID(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "foo1",
					Image: "busybox",
				},
				{
					Name:  "foo2",
					Image: "busybox",
				},
			},
		},
	}
	// Set fake sandbox and fake containers to fakeRuntime.
	fakeSandbox, _ := makeAndSetFakePod(t, m, fakeRuntime, pod)

	// Convert fakeSandbox to kubecontainer.Container
	sandbox, err := m.sandboxToKubeContainer(&runtimeApi.PodSandbox{
		Id:        fakeSandbox.Id,
		Metadata:  fakeSandbox.Metadata,
		State:     fakeSandbox.State,
		CreatedAt: fakeSandbox.CreatedAt,
		Labels:    fakeSandbox.Labels,
	})
	assert.NoError(t, err)

	expectedPod := &kubecontainer.Pod{
		ID:         pod.UID,
		Name:       pod.Name,
		Namespace:  pod.Namespace,
		Containers: []*kubecontainer.Container{},
		Sandboxes:  []*kubecontainer.Container{sandbox},
	}
	actual, err := m.GetPodContainerID(expectedPod)
	assert.Equal(t, fakeSandbox.GetId(), actual.ID)
}

func TestGetNetNS(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "foo1",
					Image: "busybox",
				},
				{
					Name:  "foo2",
					Image: "busybox",
				},
			},
		},
	}

	// Set fake sandbox and fake containers to fakeRuntime.
	sandbox, _ := makeAndSetFakePod(t, m, fakeRuntime, pod)

	actual, err := m.GetNetNS(kubecontainer.ContainerID{ID: sandbox.GetId()})
	assert.Equal(t, "", actual)
	assert.Equal(t, "not supported", err.Error())
}

func TestKillPod(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "foo1",
					Image: "busybox",
				},
				{
					Name:  "foo2",
					Image: "busybox",
				},
			},
		},
	}

	// Set fake sandbox and fake containers to fakeRuntime.
	fakeSandbox, fakeContainers := makeAndSetFakePod(t, m, fakeRuntime, pod)

	// Convert the fakeContainers to kubecontainer.Container
	containers := make([]*kubecontainer.Container, len(fakeContainers))
	for i := range containers {
		fakeContainer := fakeContainers[i]
		c, err := m.toKubeContainer(&runtimeApi.Container{
			Id:       fakeContainer.Id,
			Metadata: fakeContainer.Metadata,
			State:    fakeContainer.State,
			Image:    fakeContainer.Image,
			ImageRef: fakeContainer.ImageRef,
			Labels:   fakeContainer.Labels,
		})
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		containers[i] = c
	}
	runningPod := kubecontainer.Pod{
		ID:         pod.UID,
		Name:       pod.Name,
		Namespace:  pod.Namespace,
		Containers: []*kubecontainer.Container{containers[0], containers[1]},
		Sandboxes: []*kubecontainer.Container{
			{
				ID: kubecontainer.ContainerID{
					ID:   fakeSandbox.GetId(),
					Type: apitest.FakeRuntimeName,
				},
			},
		},
	}

	err = m.KillPod(pod, runningPod, nil)
	assert.NoError(t, err)
	assert.Equal(t, 2, len(fakeRuntime.Containers))
	assert.Equal(t, 1, len(fakeRuntime.Sandboxes))
	for _, sandbox := range fakeRuntime.Sandboxes {
		assert.Equal(t, runtimeApi.PodSandboxState_SANDBOX_NOTREADY, sandbox.GetState())
	}
	for _, c := range fakeRuntime.Containers {
		assert.Equal(t, runtimeApi.ContainerState_CONTAINER_EXITED, c.GetState())
	}
}

func TestSyncPod(t *testing.T) {
	fakeRuntime, fakeImage, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	containers := []api.Container{
		{
			Name:            "foo1",
			Image:           "busybox",
			ImagePullPolicy: api.PullIfNotPresent,
		},
		{
			Name:            "foo2",
			Image:           "alpine",
			ImagePullPolicy: api.PullIfNotPresent,
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

	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	result := m.SyncPod(pod, api.PodStatus{}, &kubecontainer.PodStatus{}, []api.Secret{}, backOff)
	assert.NoError(t, result.Error())
	assert.Equal(t, 2, len(fakeRuntime.Containers))
	assert.Equal(t, 2, len(fakeImage.Images))
	assert.Equal(t, 1, len(fakeRuntime.Sandboxes))
	for _, sandbox := range fakeRuntime.Sandboxes {
		assert.Equal(t, runtimeApi.PodSandboxState_SANDBOX_READY, sandbox.GetState())
	}
	for _, c := range fakeRuntime.Containers {
		assert.Equal(t, runtimeApi.ContainerState_CONTAINER_RUNNING, c.GetState())
	}
}

func TestPruneInitContainers(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	init1 := makeTestContainer("init1", "busybox")
	init2 := makeTestContainer("init2", "busybox")
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			InitContainers: []api.Container{init1, init2},
		},
	}

	templates := []containerTemplate{
		{pod: pod, container: &init1, attempt: 2, createdAt: 2, state: runtimeApi.ContainerState_CONTAINER_EXITED},
		{pod: pod, container: &init1, attempt: 1, createdAt: 1, state: runtimeApi.ContainerState_CONTAINER_EXITED},
		{pod: pod, container: &init2, attempt: 1, createdAt: 1, state: runtimeApi.ContainerState_CONTAINER_EXITED},
		{pod: pod, container: &init2, attempt: 0, createdAt: 0, state: runtimeApi.ContainerState_CONTAINER_EXITED},
		{pod: pod, container: &init1, attempt: 0, createdAt: 0, state: runtimeApi.ContainerState_CONTAINER_EXITED},
	}
	fakes := makeFakeContainers(t, m, templates)
	fakeRuntime.SetFakeContainers(fakes)
	podStatus, err := m.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)

	keep := map[kubecontainer.ContainerID]int{}
	m.pruneInitContainersBeforeStart(pod, podStatus, keep)
	expectedContainers := []string{fakes[0].GetId(), fakes[2].GetId()}
	if actual, ok := verifyFakeContainerList(fakeRuntime, expectedContainers); !ok {
		t.Errorf("expected %q, got %q", expectedContainers, actual)
	}
}

func TestSyncPodWithInitContainers(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	initContainers := []api.Container{
		{
			Name:            "init1",
			Image:           "init",
			ImagePullPolicy: api.PullIfNotPresent,
		},
	}
	containers := []api.Container{
		{
			Name:            "foo1",
			Image:           "busybox",
			ImagePullPolicy: api.PullIfNotPresent,
		},
		{
			Name:            "foo2",
			Image:           "alpine",
			ImagePullPolicy: api.PullIfNotPresent,
		},
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers:     containers,
			InitContainers: initContainers,
		},
	}

	// buildContainerID is an internal helper function to build container id from api pod
	// and container with default attempt number 0.
	buildContainerID := func(pod *api.Pod, container api.Container) string {
		uid := string(pod.UID)
		sandboxID := apitest.BuildSandboxName(&runtimeApi.PodSandboxMetadata{
			Name:      &pod.Name,
			Uid:       &uid,
			Namespace: &pod.Namespace,
		})
		return apitest.BuildContainerName(&runtimeApi.ContainerMetadata{Name: &container.Name}, sandboxID)
	}

	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)

	// 1. should only create the init container.
	podStatus, err := m.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	result := m.SyncPod(pod, api.PodStatus{}, podStatus, []api.Secret{}, backOff)
	assert.NoError(t, result.Error())
	assert.Equal(t, 1, len(fakeRuntime.Containers))
	initContainerID := buildContainerID(pod, initContainers[0])
	expectedContainers := []string{initContainerID}
	if actual, ok := verifyFakeContainerList(fakeRuntime, expectedContainers); !ok {
		t.Errorf("expected %q, got %q", expectedContainers, actual)
	}

	// 2. should not create app container because init container is still running.
	podStatus, err = m.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	result = m.SyncPod(pod, api.PodStatus{}, podStatus, []api.Secret{}, backOff)
	assert.NoError(t, result.Error())
	assert.Equal(t, 1, len(fakeRuntime.Containers))
	expectedContainers = []string{initContainerID}
	if actual, ok := verifyFakeContainerList(fakeRuntime, expectedContainers); !ok {
		t.Errorf("expected %q, got %q", expectedContainers, actual)
	}

	// 3. should create all app containers because init container finished.
	fakeRuntime.StopContainer(initContainerID, 0)
	podStatus, err = m.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	result = m.SyncPod(pod, api.PodStatus{}, podStatus, []api.Secret{}, backOff)
	assert.NoError(t, result.Error())
	assert.Equal(t, 3, len(fakeRuntime.Containers))
	expectedContainers = []string{initContainerID, buildContainerID(pod, containers[0]),
		buildContainerID(pod, containers[1])}
	if actual, ok := verifyFakeContainerList(fakeRuntime, expectedContainers); !ok {
		t.Errorf("expected %q, got %q", expectedContainers, actual)
	}
}
