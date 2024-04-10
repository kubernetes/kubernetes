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
	"context"
	"fmt"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	oteltrace "go.opentelemetry.io/otel/trace"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/util/flowcontrol"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	apitest "k8s.io/cri-api/pkg/apis/testing"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
)

var (
	fakeCreatedAt                int64 = 1
	containerRestartPolicyAlways       = v1.ContainerRestartPolicyAlways
)

func createTestRuntimeManager() (*apitest.FakeRuntimeService, *apitest.FakeImageService, *kubeGenericRuntimeManager, error) {
	return customTestRuntimeManager(&credentialprovider.BasicDockerKeyring{})
}

func customTestRuntimeManager(keyring *credentialprovider.BasicDockerKeyring) (*apitest.FakeRuntimeService, *apitest.FakeImageService, *kubeGenericRuntimeManager, error) {
	fakeRuntimeService := apitest.NewFakeRuntimeService()
	fakeImageService := apitest.NewFakeImageService()
	// Only an empty machineInfo is needed here, because in unit test all containers are besteffort,
	// data in machineInfo is not used. If burstable containers are used in unit test in the future,
	// we may want to set memory capacity.
	memoryCapacityQuantity := resource.MustParse(fakeNodeAllocatableMemory)
	machineInfo := &cadvisorapi.MachineInfo{
		MemoryCapacity: uint64(memoryCapacityQuantity.Value()),
	}
	osInterface := &containertest.FakeOS{}
	manager, err := newFakeKubeRuntimeManager(fakeRuntimeService, fakeImageService, machineInfo, osInterface, &containertest.FakeRuntimeHelper{}, keyring, oteltrace.NewNoopTracerProvider().Tracer(""))
	return fakeRuntimeService, fakeImageService, manager, err
}

// sandboxTemplate is a sandbox template to create fake sandbox.
type sandboxTemplate struct {
	pod         *v1.Pod
	attempt     uint32
	createdAt   int64
	state       runtimeapi.PodSandboxState
	running     bool
	terminating bool
}

// containerTemplate is a container template to create fake container.
type containerTemplate struct {
	pod            *v1.Pod
	container      *v1.Container
	sandboxAttempt uint32
	attempt        int
	createdAt      int64
	state          runtimeapi.ContainerState
}

// makeAndSetFakePod is a helper function to create and set one fake sandbox for a pod and
// one fake container for each of its container.
func makeAndSetFakePod(t *testing.T, m *kubeGenericRuntimeManager, fakeRuntime *apitest.FakeRuntimeService,
	pod *v1.Pod) (*apitest.FakePodSandbox, []*apitest.FakeContainer) {
	sandbox := makeFakePodSandbox(t, m, sandboxTemplate{
		pod:       pod,
		createdAt: fakeCreatedAt,
		state:     runtimeapi.PodSandboxState_SANDBOX_READY,
	})

	var containers []*apitest.FakeContainer
	newTemplate := func(c *v1.Container) containerTemplate {
		return containerTemplate{
			pod:       pod,
			container: c,
			createdAt: fakeCreatedAt,
			state:     runtimeapi.ContainerState_CONTAINER_RUNNING,
		}
	}
	podutil.VisitContainers(&pod.Spec, podutil.AllFeatureEnabledContainers(), func(c *v1.Container, containerType podutil.ContainerType) bool {
		containers = append(containers, makeFakeContainer(t, m, newTemplate(c)))
		return true
	})

	fakeRuntime.SetFakeSandboxes([]*apitest.FakePodSandbox{sandbox})
	fakeRuntime.SetFakeContainers(containers)
	return sandbox, containers
}

// makeFakePodSandbox creates a fake pod sandbox based on a sandbox template.
func makeFakePodSandbox(t *testing.T, m *kubeGenericRuntimeManager, template sandboxTemplate) *apitest.FakePodSandbox {
	config, err := m.generatePodSandboxConfig(template.pod, template.attempt)
	assert.NoError(t, err, "generatePodSandboxConfig for sandbox template %+v", template)

	podSandboxID := apitest.BuildSandboxName(config.Metadata)
	podSandBoxStatus := &apitest.FakePodSandbox{
		PodSandboxStatus: runtimeapi.PodSandboxStatus{
			Id:        podSandboxID,
			Metadata:  config.Metadata,
			State:     template.state,
			CreatedAt: template.createdAt,
			Network: &runtimeapi.PodSandboxNetworkStatus{
				Ip: apitest.FakePodSandboxIPs[0],
			},
			Labels: config.Labels,
		},
	}
	// assign additional IPs
	additionalIPs := apitest.FakePodSandboxIPs[1:]
	additionalPodIPs := make([]*runtimeapi.PodIP, 0, len(additionalIPs))
	for _, ip := range additionalIPs {
		additionalPodIPs = append(additionalPodIPs, &runtimeapi.PodIP{
			Ip: ip,
		})
	}
	podSandBoxStatus.Network.AdditionalIps = additionalPodIPs
	return podSandBoxStatus

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
	ctx := context.Background()
	sandboxConfig, err := m.generatePodSandboxConfig(template.pod, template.sandboxAttempt)
	assert.NoError(t, err, "generatePodSandboxConfig for container template %+v", template)

	containerConfig, _, err := m.generateContainerConfig(ctx, template.container, template.pod, template.attempt, "", template.container.Image, []string{}, nil)
	assert.NoError(t, err, "generateContainerConfig for container template %+v", template)

	podSandboxID := apitest.BuildSandboxName(sandboxConfig.Metadata)
	containerID := apitest.BuildContainerName(containerConfig.Metadata, podSandboxID)
	imageRef := containerConfig.Image.Image
	return &apitest.FakeContainer{
		ContainerStatus: runtimeapi.ContainerStatus{
			Id:          containerID,
			Metadata:    containerConfig.Metadata,
			Image:       containerConfig.Image,
			ImageRef:    imageRef,
			CreatedAt:   template.createdAt,
			State:       template.state,
			Labels:      containerConfig.Labels,
			Annotations: containerConfig.Annotations,
			LogPath:     filepath.Join(sandboxConfig.GetLogDirectory(), containerConfig.GetLogPath()),
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
func makeTestContainer(name, image string) v1.Container {
	return v1.Container{
		Name:  name,
		Image: image,
	}
}

// makeTestPod creates a test api pod.
func makeTestPod(podName, podNamespace, podUID string, containers []v1.Container) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       types.UID(podUID),
			Name:      podName,
			Namespace: podNamespace,
		},
		Spec: v1.PodSpec{
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

func verifyFakeContainerList(fakeRuntime *apitest.FakeRuntimeService, expected sets.String) (sets.String, bool) {
	actual := sets.NewString()
	for _, c := range fakeRuntime.Containers {
		actual.Insert(c.Id)
	}
	return actual, actual.Equal(expected)
}

// Only extract the fields of interests.
type cRecord struct {
	name    string
	attempt uint32
	state   runtimeapi.ContainerState
}

type cRecordList []*cRecord

func (b cRecordList) Len() int      { return len(b) }
func (b cRecordList) Swap(i, j int) { b[i], b[j] = b[j], b[i] }
func (b cRecordList) Less(i, j int) bool {
	if b[i].name != b[j].name {
		return b[i].name < b[j].name
	}
	return b[i].attempt < b[j].attempt
}

func verifyContainerStatuses(t *testing.T, runtime *apitest.FakeRuntimeService, expected []*cRecord, desc string) {
	actual := []*cRecord{}
	for _, cStatus := range runtime.Containers {
		actual = append(actual, &cRecord{name: cStatus.Metadata.Name, attempt: cStatus.Metadata.Attempt, state: cStatus.State})
	}
	sort.Sort(cRecordList(expected))
	sort.Sort(cRecordList(actual))
	assert.Equal(t, expected, actual, desc)
}

func TestNewKubeRuntimeManager(t *testing.T) {
	_, _, _, err := createTestRuntimeManager()
	assert.NoError(t, err)
}

func TestVersion(t *testing.T) {
	ctx := context.Background()
	_, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	version, err := m.Version(ctx)
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
	ctx := context.Background()
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	containers := []v1.Container{
		{
			Name:            "foo1",
			Image:           "busybox",
			ImagePullPolicy: v1.PullIfNotPresent,
		},
		{
			Name:            "foo2",
			Image:           "busybox",
			ImagePullPolicy: v1.PullIfNotPresent,
		},
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: containers,
		},
	}

	// Set fake sandbox and faked containers to fakeRuntime.
	makeAndSetFakePod(t, m, fakeRuntime, pod)

	podStatus, err := m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	assert.Equal(t, pod.UID, podStatus.ID)
	assert.Equal(t, pod.Name, podStatus.Name)
	assert.Equal(t, pod.Namespace, podStatus.Namespace)
	assert.Equal(t, apitest.FakePodSandboxIPs, podStatus.IPs)
}

func TestStopContainerWithNotFoundError(t *testing.T) {
	ctx := context.Background()
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	containers := []v1.Container{
		{
			Name:            "foo1",
			Image:           "busybox",
			ImagePullPolicy: v1.PullIfNotPresent,
		},
		{
			Name:            "foo2",
			Image:           "busybox",
			ImagePullPolicy: v1.PullIfNotPresent,
		},
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: containers,
		},
	}

	// Set fake sandbox and faked containers to fakeRuntime.
	makeAndSetFakePod(t, m, fakeRuntime, pod)
	fakeRuntime.InjectError("StopContainer", status.Error(codes.NotFound, "No such container"))
	podStatus, err := m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	require.NoError(t, err)
	p := kubecontainer.ConvertPodStatusToRunningPod("", podStatus)
	gracePeriod := int64(1)
	err = m.KillPod(ctx, pod, p, &gracePeriod)
	require.NoError(t, err)
}

func TestGetPodStatusWithNotFoundError(t *testing.T) {
	ctx := context.Background()
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	containers := []v1.Container{
		{
			Name:            "foo1",
			Image:           "busybox",
			ImagePullPolicy: v1.PullIfNotPresent,
		},
		{
			Name:            "foo2",
			Image:           "busybox",
			ImagePullPolicy: v1.PullIfNotPresent,
		},
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: containers,
		},
	}

	// Set fake sandbox and faked containers to fakeRuntime.
	makeAndSetFakePod(t, m, fakeRuntime, pod)
	fakeRuntime.InjectError("ContainerStatus", status.Error(codes.NotFound, "No such container"))
	podStatus, err := m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	require.NoError(t, err)
	require.Equal(t, pod.UID, podStatus.ID)
	require.Equal(t, pod.Name, podStatus.Name)
	require.Equal(t, pod.Namespace, podStatus.Namespace)
	require.Equal(t, apitest.FakePodSandboxIPs, podStatus.IPs)
}

func TestGetPods(t *testing.T) {
	ctx := context.Background()
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
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
		c, err := m.toKubeContainer(&runtimeapi.Container{
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
	sandbox, err := m.sandboxToKubeContainer(&runtimeapi.PodSandbox{
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
			ID:         types.UID("12345678"),
			Name:       "foo",
			Namespace:  "new",
			CreatedAt:  uint64(fakeSandbox.CreatedAt),
			Containers: []*kubecontainer.Container{containers[0], containers[1]},
			Sandboxes:  []*kubecontainer.Container{sandbox},
		},
	}

	actual, err := m.GetPods(ctx, false)
	assert.NoError(t, err)

	if !verifyPods(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}
}

func TestGetPodsSorted(t *testing.T) {
	ctx := context.Background()
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"}}

	createdTimestamps := []uint64{10, 5, 20}
	fakeSandboxes := []*apitest.FakePodSandbox{}
	for i, createdAt := range createdTimestamps {
		pod.UID = types.UID(fmt.Sprint(i))
		fakeSandboxes = append(fakeSandboxes, makeFakePodSandbox(t, m, sandboxTemplate{
			pod:       pod,
			createdAt: int64(createdAt),
			state:     runtimeapi.PodSandboxState_SANDBOX_READY,
		}))
	}
	fakeRuntime.SetFakeSandboxes(fakeSandboxes)

	actual, err := m.GetPods(ctx, false)
	assert.NoError(t, err)

	assert.Len(t, actual, 3)

	// Verify that the pods are sorted by their creation time (newest/biggest timestamp first)
	assert.Equal(t, uint64(createdTimestamps[2]), actual[0].CreatedAt)
	assert.Equal(t, uint64(createdTimestamps[0]), actual[1].CreatedAt)
	assert.Equal(t, uint64(createdTimestamps[1]), actual[2].CreatedAt)
}

func TestKillPod(t *testing.T) {
	ctx := context.Background()
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "foo1",
					Image: "busybox",
				},
				{
					Name:  "foo2",
					Image: "busybox",
				},
			},
			EphemeralContainers: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:  "debug",
						Image: "busybox",
					},
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
		c, err := m.toKubeContainer(&runtimeapi.Container{
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
		Containers: []*kubecontainer.Container{containers[0], containers[1], containers[2]},
		Sandboxes: []*kubecontainer.Container{
			{
				ID: kubecontainer.ContainerID{
					ID:   fakeSandbox.Id,
					Type: apitest.FakeRuntimeName,
				},
			},
		},
	}

	err = m.KillPod(ctx, pod, runningPod, nil)
	assert.NoError(t, err)
	assert.Equal(t, 3, len(fakeRuntime.Containers))
	assert.Equal(t, 1, len(fakeRuntime.Sandboxes))
	for _, sandbox := range fakeRuntime.Sandboxes {
		assert.Equal(t, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, sandbox.State)
	}
	for _, c := range fakeRuntime.Containers {
		assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, c.State)
	}
}

func TestSyncPod(t *testing.T) {
	fakeRuntime, fakeImage, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	containers := []v1.Container{
		{
			Name:            "foo1",
			Image:           "busybox",
			ImagePullPolicy: v1.PullIfNotPresent,
		},
		{
			Name:            "foo2",
			Image:           "alpine",
			ImagePullPolicy: v1.PullIfNotPresent,
		},
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: containers,
		},
	}

	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	result := m.SyncPod(context.Background(), pod, &kubecontainer.PodStatus{}, []v1.Secret{}, backOff)
	assert.NoError(t, result.Error())
	assert.Equal(t, 2, len(fakeRuntime.Containers))
	assert.Equal(t, 2, len(fakeImage.Images))
	assert.Equal(t, 1, len(fakeRuntime.Sandboxes))
	for _, sandbox := range fakeRuntime.Sandboxes {
		assert.Equal(t, runtimeapi.PodSandboxState_SANDBOX_READY, sandbox.State)
	}
	for _, c := range fakeRuntime.Containers {
		assert.Equal(t, runtimeapi.ContainerState_CONTAINER_RUNNING, c.State)
	}
}

func TestSyncPodWithConvertedPodSysctls(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	containers := []v1.Container{
		{
			Name:            "foo",
			Image:           "busybox",
			ImagePullPolicy: v1.PullIfNotPresent,
		},
	}

	securityContext := &v1.PodSecurityContext{
		Sysctls: []v1.Sysctl{
			{
				Name:  "kernel/shm_rmid_forced",
				Value: "1",
			},
			{
				Name:  "net/ipv4/ip_local_port_range",
				Value: "1024 65535",
			},
		},
	}
	exceptSysctls := []v1.Sysctl{
		{
			Name:  "kernel.shm_rmid_forced",
			Value: "1",
		},
		{
			Name:  "net.ipv4.ip_local_port_range",
			Value: "1024 65535",
		},
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers:      containers,
			SecurityContext: securityContext,
		},
	}

	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	result := m.SyncPod(context.Background(), pod, &kubecontainer.PodStatus{}, []v1.Secret{}, backOff)
	assert.NoError(t, result.Error())
	assert.Equal(t, exceptSysctls, pod.Spec.SecurityContext.Sysctls)
	for _, sandbox := range fakeRuntime.Sandboxes {
		assert.Equal(t, runtimeapi.PodSandboxState_SANDBOX_READY, sandbox.State)
	}
	for _, c := range fakeRuntime.Containers {
		assert.Equal(t, runtimeapi.ContainerState_CONTAINER_RUNNING, c.State)
	}
}

func TestPruneInitContainers(t *testing.T) {
	ctx := context.Background()
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	init1 := makeTestContainer("init1", "busybox")
	init2 := makeTestContainer("init2", "busybox")
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{init1, init2},
		},
	}

	templates := []containerTemplate{
		{pod: pod, container: &init1, attempt: 3, createdAt: 3, state: runtimeapi.ContainerState_CONTAINER_EXITED},
		{pod: pod, container: &init1, attempt: 2, createdAt: 2, state: runtimeapi.ContainerState_CONTAINER_EXITED},
		{pod: pod, container: &init2, attempt: 1, createdAt: 1, state: runtimeapi.ContainerState_CONTAINER_EXITED},
		{pod: pod, container: &init1, attempt: 1, createdAt: 1, state: runtimeapi.ContainerState_CONTAINER_UNKNOWN},
		{pod: pod, container: &init2, attempt: 0, createdAt: 0, state: runtimeapi.ContainerState_CONTAINER_EXITED},
		{pod: pod, container: &init1, attempt: 0, createdAt: 0, state: runtimeapi.ContainerState_CONTAINER_EXITED},
	}
	fakes := makeFakeContainers(t, m, templates)
	fakeRuntime.SetFakeContainers(fakes)
	podStatus, err := m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)

	m.pruneInitContainersBeforeStart(ctx, pod, podStatus)
	expectedContainers := sets.NewString(fakes[0].Id, fakes[2].Id)
	if actual, ok := verifyFakeContainerList(fakeRuntime, expectedContainers); !ok {
		t.Errorf("expected %v, got %v", expectedContainers, actual)
	}
}

func TestSyncPodWithInitContainers(t *testing.T) {
	ctx := context.Background()
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	initContainers := []v1.Container{
		{
			Name:            "init1",
			Image:           "init",
			ImagePullPolicy: v1.PullIfNotPresent,
		},
	}
	containers := []v1.Container{
		{
			Name:            "foo1",
			Image:           "busybox",
			ImagePullPolicy: v1.PullIfNotPresent,
		},
		{
			Name:            "foo2",
			Image:           "alpine",
			ImagePullPolicy: v1.PullIfNotPresent,
		},
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers:     containers,
			InitContainers: initContainers,
		},
	}

	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)

	// 1. should only create the init container.
	podStatus, err := m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	result := m.SyncPod(context.Background(), pod, podStatus, []v1.Secret{}, backOff)
	assert.NoError(t, result.Error())
	expected := []*cRecord{
		{name: initContainers[0].Name, attempt: 0, state: runtimeapi.ContainerState_CONTAINER_RUNNING},
	}
	verifyContainerStatuses(t, fakeRuntime, expected, "start only the init container")

	// 2. should not create app container because init container is still running.
	podStatus, err = m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	result = m.SyncPod(context.Background(), pod, podStatus, []v1.Secret{}, backOff)
	assert.NoError(t, result.Error())
	verifyContainerStatuses(t, fakeRuntime, expected, "init container still running; do nothing")

	// 3. should create all app containers because init container finished.
	// Stop init container instance 0.
	sandboxIDs, err := m.getSandboxIDByPodUID(ctx, pod.UID, nil)
	require.NoError(t, err)
	sandboxID := sandboxIDs[0]
	initID0, err := fakeRuntime.GetContainerID(sandboxID, initContainers[0].Name, 0)
	require.NoError(t, err)
	fakeRuntime.StopContainer(ctx, initID0, 0)
	// Sync again.
	podStatus, err = m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	result = m.SyncPod(ctx, pod, podStatus, []v1.Secret{}, backOff)
	assert.NoError(t, result.Error())
	expected = []*cRecord{
		{name: initContainers[0].Name, attempt: 0, state: runtimeapi.ContainerState_CONTAINER_EXITED},
		{name: containers[0].Name, attempt: 0, state: runtimeapi.ContainerState_CONTAINER_RUNNING},
		{name: containers[1].Name, attempt: 0, state: runtimeapi.ContainerState_CONTAINER_RUNNING},
	}
	verifyContainerStatuses(t, fakeRuntime, expected, "init container completed; all app containers should be running")

	// 4. should restart the init container if needed to create a new podsandbox
	// Stop the pod sandbox.
	fakeRuntime.StopPodSandbox(ctx, sandboxID)
	// Sync again.
	podStatus, err = m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	result = m.SyncPod(ctx, pod, podStatus, []v1.Secret{}, backOff)
	assert.NoError(t, result.Error())
	expected = []*cRecord{
		// The first init container instance is purged and no longer visible.
		// The second (attempt == 1) instance has been started and is running.
		{name: initContainers[0].Name, attempt: 1, state: runtimeapi.ContainerState_CONTAINER_RUNNING},
		// All containers are killed.
		{name: containers[0].Name, attempt: 0, state: runtimeapi.ContainerState_CONTAINER_EXITED},
		{name: containers[1].Name, attempt: 0, state: runtimeapi.ContainerState_CONTAINER_EXITED},
	}
	verifyContainerStatuses(t, fakeRuntime, expected, "kill all app containers, purge the existing init container, and restart a new one")
}

// A helper function to get a basic pod and its status assuming all sandbox and
// containers are running and ready.
func makeBasePodAndStatus() (*v1.Pod, *kubecontainer.PodStatus) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "foo-ns",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "foo1",
					Image: "busybox",
				},
				{
					Name:  "foo2",
					Image: "busybox",
				},
				{
					Name:  "foo3",
					Image: "busybox",
				},
			},
		},
		Status: v1.PodStatus{
			ContainerStatuses: []v1.ContainerStatus{
				{
					ContainerID: "://id1",
					Name:        "foo1",
					Image:       "busybox",
					State:       v1.ContainerState{Running: &v1.ContainerStateRunning{}},
				},
				{
					ContainerID: "://id2",
					Name:        "foo2",
					Image:       "busybox",
					State:       v1.ContainerState{Running: &v1.ContainerStateRunning{}},
				},
				{
					ContainerID: "://id3",
					Name:        "foo3",
					Image:       "busybox",
					State:       v1.ContainerState{Running: &v1.ContainerStateRunning{}},
				},
			},
		},
	}
	status := &kubecontainer.PodStatus{
		ID:        pod.UID,
		Name:      pod.Name,
		Namespace: pod.Namespace,
		SandboxStatuses: []*runtimeapi.PodSandboxStatus{
			{
				Id:       "sandboxID",
				State:    runtimeapi.PodSandboxState_SANDBOX_READY,
				Metadata: &runtimeapi.PodSandboxMetadata{Name: pod.Name, Namespace: pod.Namespace, Uid: "sandboxuid", Attempt: uint32(0)},
				Network:  &runtimeapi.PodSandboxNetworkStatus{Ip: "10.0.0.1"},
			},
		},
		ContainerStatuses: []*kubecontainer.Status{
			{
				ID:   kubecontainer.ContainerID{ID: "id1"},
				Name: "foo1", State: kubecontainer.ContainerStateRunning,
				Hash: kubecontainer.HashContainer(&pod.Spec.Containers[0]),
			},
			{
				ID:   kubecontainer.ContainerID{ID: "id2"},
				Name: "foo2", State: kubecontainer.ContainerStateRunning,
				Hash: kubecontainer.HashContainer(&pod.Spec.Containers[1]),
			},
			{
				ID:   kubecontainer.ContainerID{ID: "id3"},
				Name: "foo3", State: kubecontainer.ContainerStateRunning,
				Hash: kubecontainer.HashContainer(&pod.Spec.Containers[2]),
			},
		},
	}
	return pod, status
}

func TestComputePodActions(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	// Creating a pair reference pod and status for the test cases to refer
	// the specific fields.
	basePod, baseStatus := makeBasePodAndStatus()
	noAction := podActions{
		SandboxID:         baseStatus.SandboxStatuses[0].Id,
		ContainersToStart: []int{},
		ContainersToKill:  map[kubecontainer.ContainerID]containerToKillInfo{},
	}

	for desc, test := range map[string]struct {
		mutatePodFn    func(*v1.Pod)
		mutateStatusFn func(*kubecontainer.PodStatus)
		actions        podActions
		resetStatusFn  func(*kubecontainer.PodStatus)
	}{
		"everything is good; do nothing": {
			actions: noAction,
		},
		"start pod sandbox and all containers for a new pod": {
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				// No container or sandbox exists.
				status.SandboxStatuses = []*runtimeapi.PodSandboxStatus{}
				status.ContainerStatuses = []*kubecontainer.Status{}
			},
			actions: podActions{
				KillPod:           true,
				CreateSandbox:     true,
				Attempt:           uint32(0),
				ContainersToStart: []int{0, 1, 2},
				ContainersToKill:  getKillMap(basePod, baseStatus, []int{}),
			},
		},
		"restart exited containers if RestartPolicy == Always": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				// The first container completed, restart it,
				status.ContainerStatuses[0].State = kubecontainer.ContainerStateExited
				status.ContainerStatuses[0].ExitCode = 0

				// The second container exited with failure, restart it,
				status.ContainerStatuses[1].State = kubecontainer.ContainerStateExited
				status.ContainerStatuses[1].ExitCode = 111
			},
			actions: podActions{
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				ContainersToStart: []int{0, 1},
				ContainersToKill:  getKillMap(basePod, baseStatus, []int{}),
			},
		},
		"restart failed containers if RestartPolicy == OnFailure": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyOnFailure },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				// The first container completed, don't restart it,
				status.ContainerStatuses[0].State = kubecontainer.ContainerStateExited
				status.ContainerStatuses[0].ExitCode = 0

				// The second container exited with failure, restart it,
				status.ContainerStatuses[1].State = kubecontainer.ContainerStateExited
				status.ContainerStatuses[1].ExitCode = 111
			},
			actions: podActions{
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				ContainersToStart: []int{1},
				ContainersToKill:  getKillMap(basePod, baseStatus, []int{}),
			},
		},
		"don't restart containers if RestartPolicy == Never": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyNever },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				// Don't restart any containers.
				status.ContainerStatuses[0].State = kubecontainer.ContainerStateExited
				status.ContainerStatuses[0].ExitCode = 0
				status.ContainerStatuses[1].State = kubecontainer.ContainerStateExited
				status.ContainerStatuses[1].ExitCode = 111
			},
			actions: noAction,
		},
		"Kill pod and recreate everything if the pod sandbox is dead, and RestartPolicy == Always": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
			},
			actions: podActions{
				KillPod:           true,
				CreateSandbox:     true,
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				Attempt:           uint32(1),
				ContainersToStart: []int{0, 1, 2},
				ContainersToKill:  getKillMap(basePod, baseStatus, []int{}),
			},
		},
		"Kill pod and recreate all containers (except for the succeeded one) if the pod sandbox is dead, and RestartPolicy == OnFailure": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyOnFailure },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
				status.ContainerStatuses[1].State = kubecontainer.ContainerStateExited
				status.ContainerStatuses[1].ExitCode = 0
			},
			actions: podActions{
				KillPod:           true,
				CreateSandbox:     true,
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				Attempt:           uint32(1),
				ContainersToStart: []int{0, 2},
				ContainersToKill:  getKillMap(basePod, baseStatus, []int{}),
			},
		},
		"Kill pod and recreate all containers if the PodSandbox does not have an IP": {
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.SandboxStatuses[0].Network.Ip = ""
			},
			actions: podActions{
				KillPod:           true,
				CreateSandbox:     true,
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				Attempt:           uint32(1),
				ContainersToStart: []int{0, 1, 2},
				ContainersToKill:  getKillMap(basePod, baseStatus, []int{}),
			},
		},
		"Kill and recreate the container if the container's spec changed": {
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyAlways
			},
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses[1].Hash = uint64(432423432)
			},
			actions: podActions{
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				ContainersToKill:  getKillMap(basePod, baseStatus, []int{1}),
				ContainersToStart: []int{1},
			},
		},
		"Kill and recreate the container if the liveness check has failed": {
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyAlways
			},
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				m.livenessManager.Set(status.ContainerStatuses[1].ID, proberesults.Failure, basePod)
			},
			actions: podActions{
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				ContainersToKill:  getKillMap(basePod, baseStatus, []int{1}),
				ContainersToStart: []int{1},
			},
			resetStatusFn: func(status *kubecontainer.PodStatus) {
				m.livenessManager.Remove(status.ContainerStatuses[1].ID)
			},
		},
		"Kill and recreate the container if the startup check has failed": {
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyAlways
			},
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				m.startupManager.Set(status.ContainerStatuses[1].ID, proberesults.Failure, basePod)
			},
			actions: podActions{
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				ContainersToKill:  getKillMap(basePod, baseStatus, []int{1}),
				ContainersToStart: []int{1},
			},
			resetStatusFn: func(status *kubecontainer.PodStatus) {
				m.startupManager.Remove(status.ContainerStatuses[1].ID)
			},
		},
		"Verify we do not create a pod sandbox if no ready sandbox for pod with RestartPolicy=Never and all containers exited": {
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyNever
			},
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				// no ready sandbox
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
				status.SandboxStatuses[0].Metadata.Attempt = uint32(1)
				// all containers exited
				for i := range status.ContainerStatuses {
					status.ContainerStatuses[i].State = kubecontainer.ContainerStateExited
					status.ContainerStatuses[i].ExitCode = 0
				}
			},
			actions: podActions{
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				Attempt:           uint32(2),
				CreateSandbox:     false,
				KillPod:           true,
				ContainersToStart: []int{},
				ContainersToKill:  map[kubecontainer.ContainerID]containerToKillInfo{},
			},
		},
		"Verify we do not create a pod sandbox if no ready sandbox for pod with RestartPolicy=OnFailure and all containers succeeded": {
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyOnFailure
			},
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				// no ready sandbox
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
				status.SandboxStatuses[0].Metadata.Attempt = uint32(1)
				// all containers succeeded
				for i := range status.ContainerStatuses {
					status.ContainerStatuses[i].State = kubecontainer.ContainerStateExited
					status.ContainerStatuses[i].ExitCode = 0
				}
			},
			actions: podActions{
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				Attempt:           uint32(2),
				CreateSandbox:     false,
				KillPod:           true,
				ContainersToStart: []int{},
				ContainersToKill:  map[kubecontainer.ContainerID]containerToKillInfo{},
			},
		},
		"Verify we create a pod sandbox if no ready sandbox for pod with RestartPolicy=Never and no containers have ever been created": {
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyNever
			},
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				// no ready sandbox
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
				status.SandboxStatuses[0].Metadata.Attempt = uint32(2)
				// no visible containers
				status.ContainerStatuses = []*kubecontainer.Status{}
			},
			actions: podActions{
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				Attempt:           uint32(3),
				CreateSandbox:     true,
				KillPod:           true,
				ContainersToStart: []int{0, 1, 2},
				ContainersToKill:  map[kubecontainer.ContainerID]containerToKillInfo{},
			},
		},
		"Kill and recreate the container if the container is in unknown state": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyNever },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses[1].State = kubecontainer.ContainerStateUnknown
			},
			actions: podActions{
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				ContainersToKill:  getKillMap(basePod, baseStatus, []int{1}),
				ContainersToStart: []int{1},
			},
		},
	} {
		pod, status := makeBasePodAndStatus()
		if test.mutatePodFn != nil {
			test.mutatePodFn(pod)
		}
		if test.mutateStatusFn != nil {
			test.mutateStatusFn(status)
		}
		ctx := context.Background()
		actions := m.computePodActions(ctx, pod, status)
		verifyActions(t, &test.actions, &actions, desc)
		if test.resetStatusFn != nil {
			test.resetStatusFn(status)
		}
	}
}

func getKillMap(pod *v1.Pod, status *kubecontainer.PodStatus, cIndexes []int) map[kubecontainer.ContainerID]containerToKillInfo {
	m := map[kubecontainer.ContainerID]containerToKillInfo{}
	for _, i := range cIndexes {
		m[status.ContainerStatuses[i].ID] = containerToKillInfo{
			container: &pod.Spec.Containers[i],
			name:      pod.Spec.Containers[i].Name,
		}
	}
	return m
}

func getKillMapWithInitContainers(pod *v1.Pod, status *kubecontainer.PodStatus, cIndexes []int) map[kubecontainer.ContainerID]containerToKillInfo {
	m := map[kubecontainer.ContainerID]containerToKillInfo{}
	for _, i := range cIndexes {
		m[status.ContainerStatuses[i].ID] = containerToKillInfo{
			container: &pod.Spec.InitContainers[i],
			name:      pod.Spec.InitContainers[i].Name,
		}
	}
	return m
}

func verifyActions(t *testing.T, expected, actual *podActions, desc string) {
	if actual.ContainersToKill != nil {
		// Clear the message and reason fields since we don't need to verify them.
		for k, info := range actual.ContainersToKill {
			info.message = ""
			info.reason = ""
			actual.ContainersToKill[k] = info
		}
	}
	assert.Equal(t, expected, actual, desc)
}

func TestComputePodActionsWithInitContainers(t *testing.T) {
	t.Run("sidecar containers disabled", func(t *testing.T) {
		testComputePodActionsWithInitContainers(t, false)
	})
	t.Run("sidecar containers enabled", func(t *testing.T) {
		testComputePodActionsWithInitContainers(t, true)
	})
}

func testComputePodActionsWithInitContainers(t *testing.T, sidecarContainersEnabled bool) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SidecarContainers, sidecarContainersEnabled)()
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	// Creating a pair reference pod and status for the test cases to refer
	// the specific fields.
	basePod, baseStatus := makeBasePodAndStatusWithInitContainers()
	noAction := podActions{
		SandboxID:         baseStatus.SandboxStatuses[0].Id,
		ContainersToStart: []int{},
		ContainersToKill:  map[kubecontainer.ContainerID]containerToKillInfo{},
	}

	for desc, test := range map[string]struct {
		mutatePodFn    func(*v1.Pod)
		mutateStatusFn func(*kubecontainer.PodStatus)
		actions        podActions
	}{
		"initialization completed; start all containers": {
			actions: podActions{
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				ContainersToStart: []int{0, 1, 2},
				ContainersToKill:  getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"no init containers have been started; start the first one": {
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses = nil
			},
			actions: podActions{
				SandboxID:                baseStatus.SandboxStatuses[0].Id,
				NextInitContainerToStart: &basePod.Spec.InitContainers[0],
				InitContainersToStart:    []int{0},
				ContainersToStart:        []int{},
				ContainersToKill:         getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"initialization in progress; do nothing": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses[2].State = kubecontainer.ContainerStateRunning
			},
			actions: noAction,
		},
		"Kill pod and restart the first init container if the pod sandbox is dead": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
			},
			actions: podActions{
				KillPod:                  true,
				CreateSandbox:            true,
				SandboxID:                baseStatus.SandboxStatuses[0].Id,
				Attempt:                  uint32(1),
				NextInitContainerToStart: &basePod.Spec.InitContainers[0],
				InitContainersToStart:    []int{0},
				ContainersToStart:        []int{},
				ContainersToKill:         getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"initialization failed; restart the last init container if RestartPolicy == Always": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses[2].ExitCode = 137
			},
			actions: podActions{
				SandboxID:                baseStatus.SandboxStatuses[0].Id,
				NextInitContainerToStart: &basePod.Spec.InitContainers[2],
				InitContainersToStart:    []int{2},
				ContainersToStart:        []int{},
				ContainersToKill:         getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"initialization failed; restart the last init container if RestartPolicy == OnFailure": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyOnFailure },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses[2].ExitCode = 137
			},
			actions: podActions{
				SandboxID:                baseStatus.SandboxStatuses[0].Id,
				NextInitContainerToStart: &basePod.Spec.InitContainers[2],
				InitContainersToStart:    []int{2},
				ContainersToStart:        []int{},
				ContainersToKill:         getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"initialization failed; kill pod if RestartPolicy == Never": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyNever },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses[2].ExitCode = 137
			},
			actions: podActions{
				KillPod:           true,
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				ContainersToStart: []int{},
				ContainersToKill:  getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"init container state unknown; kill and recreate the last init container if RestartPolicy == Always": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses[2].State = kubecontainer.ContainerStateUnknown
			},
			actions: podActions{
				SandboxID:                baseStatus.SandboxStatuses[0].Id,
				NextInitContainerToStart: &basePod.Spec.InitContainers[2],
				InitContainersToStart:    []int{2},
				ContainersToStart:        []int{},
				ContainersToKill:         getKillMapWithInitContainers(basePod, baseStatus, []int{2}),
			},
		},
		"init container state unknown; kill and recreate the last init container if RestartPolicy == OnFailure": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyOnFailure },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses[2].State = kubecontainer.ContainerStateUnknown
			},
			actions: podActions{
				SandboxID:                baseStatus.SandboxStatuses[0].Id,
				NextInitContainerToStart: &basePod.Spec.InitContainers[2],
				InitContainersToStart:    []int{2},
				ContainersToStart:        []int{},
				ContainersToKill:         getKillMapWithInitContainers(basePod, baseStatus, []int{2}),
			},
		},
		"init container state unknown; kill pod if RestartPolicy == Never": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyNever },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses[2].State = kubecontainer.ContainerStateUnknown
			},
			actions: podActions{
				KillPod:           true,
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				ContainersToStart: []int{},
				ContainersToKill:  getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"Pod sandbox not ready, init container failed, but RestartPolicy == Never; kill pod only": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyNever },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
			},
			actions: podActions{
				KillPod:           true,
				CreateSandbox:     false,
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				Attempt:           uint32(1),
				ContainersToStart: []int{},
				ContainersToKill:  getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"Pod sandbox not ready, and RestartPolicy == Never, but no visible init containers;  create a new pod sandbox": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyNever },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
				status.ContainerStatuses = []*kubecontainer.Status{}
			},
			actions: podActions{
				KillPod:                  true,
				CreateSandbox:            true,
				SandboxID:                baseStatus.SandboxStatuses[0].Id,
				Attempt:                  uint32(1),
				NextInitContainerToStart: &basePod.Spec.InitContainers[0],
				InitContainersToStart:    []int{0},
				ContainersToStart:        []int{},
				ContainersToKill:         getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"Pod sandbox not ready, init container failed, and RestartPolicy == OnFailure; create a new pod sandbox": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyOnFailure },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
				status.ContainerStatuses[2].ExitCode = 137
			},
			actions: podActions{
				KillPod:                  true,
				CreateSandbox:            true,
				SandboxID:                baseStatus.SandboxStatuses[0].Id,
				Attempt:                  uint32(1),
				NextInitContainerToStart: &basePod.Spec.InitContainers[0],
				InitContainersToStart:    []int{0},
				ContainersToStart:        []int{},
				ContainersToKill:         getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"some of the init container statuses are missing but the last init container is running, don't restart preceding ones": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses[2].State = kubecontainer.ContainerStateRunning
				status.ContainerStatuses = status.ContainerStatuses[2:]
			},
			actions: podActions{
				KillPod:           false,
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				ContainersToStart: []int{},
				ContainersToKill:  getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
	} {
		pod, status := makeBasePodAndStatusWithInitContainers()
		if test.mutatePodFn != nil {
			test.mutatePodFn(pod)
		}
		if test.mutateStatusFn != nil {
			test.mutateStatusFn(status)
		}
		ctx := context.Background()
		actions := m.computePodActions(ctx, pod, status)
		if !sidecarContainersEnabled {
			// If sidecar containers are disabled, we should not see any
			// InitContainersToStart in the actions.
			test.actions.InitContainersToStart = nil
		} else {
			// If sidecar containers are enabled, we should not see any
			// NextInitContainerToStart in the actions.
			test.actions.NextInitContainerToStart = nil
		}
		verifyActions(t, &test.actions, &actions, desc)
	}
}

func makeBasePodAndStatusWithInitContainers() (*v1.Pod, *kubecontainer.PodStatus) {
	pod, status := makeBasePodAndStatus()
	pod.Spec.InitContainers = []v1.Container{
		{
			Name:  "init1",
			Image: "bar-image",
		},
		{
			Name:  "init2",
			Image: "bar-image",
		},
		{
			Name:  "init3",
			Image: "bar-image",
		},
	}
	// Replace the original statuses of the containers with those for the init
	// containers.
	status.ContainerStatuses = []*kubecontainer.Status{
		{
			ID:   kubecontainer.ContainerID{ID: "initid1"},
			Name: "init1", State: kubecontainer.ContainerStateExited,
			Hash: kubecontainer.HashContainer(&pod.Spec.InitContainers[0]),
		},
		{
			ID:   kubecontainer.ContainerID{ID: "initid2"},
			Name: "init2", State: kubecontainer.ContainerStateExited,
			Hash: kubecontainer.HashContainer(&pod.Spec.InitContainers[0]),
		},
		{
			ID:   kubecontainer.ContainerID{ID: "initid3"},
			Name: "init3", State: kubecontainer.ContainerStateExited,
			Hash: kubecontainer.HashContainer(&pod.Spec.InitContainers[0]),
		},
	}
	return pod, status
}

func TestComputePodActionsWithRestartableInitContainers(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SidecarContainers, true)()
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	// Creating a pair reference pod and status for the test cases to refer
	// the specific fields.
	basePod, baseStatus := makeBasePodAndStatusWithRestartableInitContainers()
	noAction := podActions{
		SandboxID:         baseStatus.SandboxStatuses[0].Id,
		ContainersToStart: []int{},
		ContainersToKill:  map[kubecontainer.ContainerID]containerToKillInfo{},
	}

	for desc, test := range map[string]struct {
		mutatePodFn    func(*v1.Pod)
		mutateStatusFn func(*v1.Pod, *kubecontainer.PodStatus)
		actions        podActions
		resetStatusFn  func(*kubecontainer.PodStatus)
	}{
		"initialization completed; start all containers": {
			actions: podActions{
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				ContainersToStart: []int{0, 1, 2},
				ContainersToKill:  getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"no init containers have been started; start the first one": {
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.ContainerStatuses = nil
			},
			actions: podActions{
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				InitContainersToStart: []int{0},
				ContainersToStart:     []int{},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"initialization in progress; do nothing": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.ContainerStatuses[2].State = kubecontainer.ContainerStateCreated
			},
			actions: noAction,
		},
		"restartable init container has started; start the next": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.ContainerStatuses = status.ContainerStatuses[:1]
			},
			actions: podActions{
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				InitContainersToStart: []int{1},
				ContainersToStart:     []int{},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"livenessProbe has not been run; start the nothing": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				m.livenessManager.Remove(status.ContainerStatuses[1].ID)
				status.ContainerStatuses = status.ContainerStatuses[:2]
			},
			actions: podActions{
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				InitContainersToStart: []int{2},
				ContainersToStart:     []int{},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"livenessProbe in progress; start the next": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				m.livenessManager.Set(status.ContainerStatuses[1].ID, proberesults.Unknown, basePod)
				status.ContainerStatuses = status.ContainerStatuses[:2]
			},
			actions: podActions{
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				InitContainersToStart: []int{2},
				ContainersToStart:     []int{},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
			resetStatusFn: func(status *kubecontainer.PodStatus) {
				m.livenessManager.Remove(status.ContainerStatuses[1].ID)
			},
		},
		"livenessProbe has completed; start the next": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.ContainerStatuses = status.ContainerStatuses[:2]
			},
			actions: podActions{
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				InitContainersToStart: []int{2},
				ContainersToStart:     []int{},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"kill and recreate the restartable init container if the liveness check has failed": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				m.livenessManager.Set(status.ContainerStatuses[2].ID, proberesults.Failure, basePod)
			},
			actions: podActions{
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				InitContainersToStart: []int{2},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{2}),
				ContainersToStart:     []int{0, 1, 2},
			},
			resetStatusFn: func(status *kubecontainer.PodStatus) {
				m.livenessManager.Remove(status.ContainerStatuses[2].ID)
			},
		},
		"startupProbe has not been run; do nothing": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				m.startupManager.Remove(status.ContainerStatuses[1].ID)
				status.ContainerStatuses = status.ContainerStatuses[:2]
			},
			actions: noAction,
		},
		"startupProbe in progress; do nothing": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				m.startupManager.Set(status.ContainerStatuses[1].ID, proberesults.Unknown, basePod)
				status.ContainerStatuses = status.ContainerStatuses[:2]
			},
			actions: noAction,
			resetStatusFn: func(status *kubecontainer.PodStatus) {
				m.startupManager.Remove(status.ContainerStatuses[1].ID)
			},
		},
		"startupProbe has completed; start the next": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.ContainerStatuses = status.ContainerStatuses[:2]
			},
			actions: podActions{
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				InitContainersToStart: []int{2},
				ContainersToStart:     []int{},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"kill and recreate the restartable init container if the startup check has failed": {
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyAlways
				pod.Spec.InitContainers[2].StartupProbe = &v1.Probe{}
			},
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				m.startupManager.Set(status.ContainerStatuses[2].ID, proberesults.Failure, basePod)
			},
			actions: podActions{
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				InitContainersToStart: []int{2},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{2}),
				ContainersToStart:     []int{},
			},
			resetStatusFn: func(status *kubecontainer.PodStatus) {
				m.startupManager.Remove(status.ContainerStatuses[2].ID)
			},
		},
		"restart terminated restartable init container and next init container": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.ContainerStatuses[0].State = kubecontainer.ContainerStateExited
				status.ContainerStatuses[2].State = kubecontainer.ContainerStateExited
			},
			actions: podActions{
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				InitContainersToStart: []int{0, 2},
				ContainersToStart:     []int{},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"restart terminated restartable init container and regular containers": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.ContainerStatuses[0].State = kubecontainer.ContainerStateExited
			},
			actions: podActions{
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				InitContainersToStart: []int{0},
				ContainersToStart:     []int{0, 1, 2},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"Pod sandbox not ready, restartable init container failed, but RestartPolicy == Never; kill pod only": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyNever },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
				status.ContainerStatuses[2].State = kubecontainer.ContainerStateExited
				status.ContainerStatuses[2].ExitCode = 137
			},
			actions: podActions{
				KillPod:           true,
				CreateSandbox:     false,
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				Attempt:           uint32(1),
				ContainersToStart: []int{},
				ContainersToKill:  getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"Pod sandbox not ready, and RestartPolicy == Never, but no visible restartable init containers;  create a new pod sandbox": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyNever },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
				status.ContainerStatuses = []*kubecontainer.Status{}
			},
			actions: podActions{
				KillPod:               true,
				CreateSandbox:         true,
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				Attempt:               uint32(1),
				InitContainersToStart: []int{0},
				ContainersToStart:     []int{},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"Pod sandbox not ready, restartable init container failed, and RestartPolicy == OnFailure; create a new pod sandbox": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyOnFailure },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
				status.ContainerStatuses[2].State = kubecontainer.ContainerStateExited
				status.ContainerStatuses[2].ExitCode = 137
			},
			actions: podActions{
				KillPod:               true,
				CreateSandbox:         true,
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				Attempt:               uint32(1),
				InitContainersToStart: []int{0},
				ContainersToStart:     []int{},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"Pod sandbox not ready, restartable init container failed, and RestartPolicy == Always; create a new pod sandbox": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
				status.ContainerStatuses[2].State = kubecontainer.ContainerStateExited
				status.ContainerStatuses[2].ExitCode = 137
			},
			actions: podActions{
				KillPod:               true,
				CreateSandbox:         true,
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				Attempt:               uint32(1),
				InitContainersToStart: []int{0},
				ContainersToStart:     []int{},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"initialization failed; restart the last restartable init container even if pod's RestartPolicy == Never": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyNever },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.ContainerStatuses[2].State = kubecontainer.ContainerStateExited
				status.ContainerStatuses[2].ExitCode = 137
			},
			actions: podActions{
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				InitContainersToStart: []int{2},
				ContainersToStart:     []int{},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"restartable init container state unknown; kill and recreate the last restartable init container even if pod's RestartPolicy == Never": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyNever },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.ContainerStatuses[2].State = kubecontainer.ContainerStateUnknown
			},
			actions: podActions{
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				InitContainersToStart: []int{2},
				ContainersToStart:     []int{},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{2}),
			},
		},
		"restart restartable init container if regular containers are running even if pod's RestartPolicy == Never": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyNever },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.ContainerStatuses[2].State = kubecontainer.ContainerStateExited
				status.ContainerStatuses[2].ExitCode = 137
				// all main containers are running
				for i := 1; i <= 3; i++ {
					status.ContainerStatuses = append(status.ContainerStatuses, &kubecontainer.Status{
						ID:    kubecontainer.ContainerID{ID: fmt.Sprintf("id%d", i)},
						Name:  fmt.Sprintf("foo%d", i),
						State: kubecontainer.ContainerStateRunning,
						Hash:  kubecontainer.HashContainer(&pod.Spec.Containers[i-1]),
					})
				}
			},
			actions: podActions{
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				InitContainersToStart: []int{2},
				ContainersToStart:     []int{},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"kill the pod if all main containers succeeded if pod's RestartPolicy == Never": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyNever },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				// all main containers succeeded
				for i := 1; i <= 3; i++ {
					status.ContainerStatuses = append(status.ContainerStatuses, &kubecontainer.Status{
						ID:       kubecontainer.ContainerID{ID: fmt.Sprintf("id%d", i)},
						Name:     fmt.Sprintf("foo%d", i),
						State:    kubecontainer.ContainerStateExited,
						ExitCode: 0,
						Hash:     kubecontainer.HashContainer(&pod.Spec.Containers[i-1]),
					})
				}
			},
			actions: podActions{
				KillPod:           true,
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				ContainersToStart: []int{},
				ContainersToKill:  getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"some of the init container statuses are missing but the last init container is running, restart restartable init and regular containers": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(pod *v1.Pod, status *kubecontainer.PodStatus) {
				status.ContainerStatuses = status.ContainerStatuses[2:]
			},
			actions: podActions{
				SandboxID:             baseStatus.SandboxStatuses[0].Id,
				InitContainersToStart: []int{0, 1},
				ContainersToStart:     []int{0, 1, 2},
				ContainersToKill:      getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
	} {
		pod, status := makeBasePodAndStatusWithRestartableInitContainers()
		m.livenessManager.Set(status.ContainerStatuses[1].ID, proberesults.Success, basePod)
		m.startupManager.Set(status.ContainerStatuses[1].ID, proberesults.Success, basePod)
		m.livenessManager.Set(status.ContainerStatuses[2].ID, proberesults.Success, basePod)
		m.startupManager.Set(status.ContainerStatuses[2].ID, proberesults.Success, basePod)
		if test.mutatePodFn != nil {
			test.mutatePodFn(pod)
		}
		if test.mutateStatusFn != nil {
			test.mutateStatusFn(pod, status)
		}
		ctx := context.Background()
		actions := m.computePodActions(ctx, pod, status)
		verifyActions(t, &test.actions, &actions, desc)
		if test.resetStatusFn != nil {
			test.resetStatusFn(status)
		}
	}
}

func makeBasePodAndStatusWithRestartableInitContainers() (*v1.Pod, *kubecontainer.PodStatus) {
	pod, status := makeBasePodAndStatus()
	pod.Spec.InitContainers = []v1.Container{
		{
			Name:          "restartable-init-1",
			Image:         "bar-image",
			RestartPolicy: &containerRestartPolicyAlways,
		},
		{
			Name:          "restartable-init-2",
			Image:         "bar-image",
			RestartPolicy: &containerRestartPolicyAlways,
			LivenessProbe: &v1.Probe{},
			StartupProbe:  &v1.Probe{},
		},
		{
			Name:          "restartable-init-3",
			Image:         "bar-image",
			RestartPolicy: &containerRestartPolicyAlways,
			LivenessProbe: &v1.Probe{},
			StartupProbe:  &v1.Probe{},
		},
	}
	// Replace the original statuses of the containers with those for the init
	// containers.
	status.ContainerStatuses = []*kubecontainer.Status{
		{
			ID:   kubecontainer.ContainerID{ID: "initid1"},
			Name: "restartable-init-1", State: kubecontainer.ContainerStateRunning,
			Hash: kubecontainer.HashContainer(&pod.Spec.InitContainers[0]),
		},
		{
			ID:   kubecontainer.ContainerID{ID: "initid2"},
			Name: "restartable-init-2", State: kubecontainer.ContainerStateRunning,
			Hash: kubecontainer.HashContainer(&pod.Spec.InitContainers[0]),
		},
		{
			ID:   kubecontainer.ContainerID{ID: "initid3"},
			Name: "restartable-init-3", State: kubecontainer.ContainerStateRunning,
			Hash: kubecontainer.HashContainer(&pod.Spec.InitContainers[0]),
		},
	}
	return pod, status
}

func TestComputePodActionsWithInitAndEphemeralContainers(t *testing.T) {
	// Make sure existing test cases pass with feature enabled
	TestComputePodActions(t)
	TestComputePodActionsWithInitContainers(t)

	t.Run("sidecar containers disabled", func(t *testing.T) {
		testComputePodActionsWithInitAndEphemeralContainers(t, false)
	})
	t.Run("sidecar containers enabled", func(t *testing.T) {
		testComputePodActionsWithInitAndEphemeralContainers(t, true)
	})
}

func testComputePodActionsWithInitAndEphemeralContainers(t *testing.T, sidecarContainersEnabled bool) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SidecarContainers, sidecarContainersEnabled)()
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	basePod, baseStatus := makeBasePodAndStatusWithInitAndEphemeralContainers()
	noAction := podActions{
		SandboxID:         baseStatus.SandboxStatuses[0].Id,
		ContainersToStart: []int{},
		ContainersToKill:  map[kubecontainer.ContainerID]containerToKillInfo{},
	}

	for desc, test := range map[string]struct {
		mutatePodFn    func(*v1.Pod)
		mutateStatusFn func(*kubecontainer.PodStatus)
		actions        podActions
	}{
		"steady state; do nothing; ignore ephemeral container": {
			actions: noAction,
		},
		"No ephemeral containers running; start one": {
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses = status.ContainerStatuses[:4]
			},
			actions: podActions{
				SandboxID:                  baseStatus.SandboxStatuses[0].Id,
				ContainersToStart:          []int{},
				ContainersToKill:           map[kubecontainer.ContainerID]containerToKillInfo{},
				EphemeralContainersToStart: []int{0},
			},
		},
		"Start second ephemeral container": {
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.EphemeralContainers = append(pod.Spec.EphemeralContainers, v1.EphemeralContainer{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:  "debug2",
						Image: "busybox",
					},
				})
			},
			actions: podActions{
				SandboxID:                  baseStatus.SandboxStatuses[0].Id,
				ContainersToStart:          []int{},
				ContainersToKill:           map[kubecontainer.ContainerID]containerToKillInfo{},
				EphemeralContainersToStart: []int{1},
			},
		},
		"Ephemeral container exited; do not restart": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses[4].State = kubecontainer.ContainerStateExited
			},
			actions: podActions{
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				ContainersToStart: []int{},
				ContainersToKill:  map[kubecontainer.ContainerID]containerToKillInfo{},
			},
		},
		"initialization in progress; start ephemeral container": {
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses[3].State = kubecontainer.ContainerStateRunning
				status.ContainerStatuses = status.ContainerStatuses[:4]
			},
			actions: podActions{
				SandboxID:                  baseStatus.SandboxStatuses[0].Id,
				ContainersToStart:          []int{},
				ContainersToKill:           map[kubecontainer.ContainerID]containerToKillInfo{},
				EphemeralContainersToStart: []int{0},
			},
		},
		"Create a new pod sandbox if the pod sandbox is dead, init container failed and RestartPolicy == OnFailure": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyOnFailure },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
				status.ContainerStatuses = status.ContainerStatuses[3:]
				status.ContainerStatuses[0].ExitCode = 137
			},
			actions: podActions{
				KillPod:                  true,
				CreateSandbox:            true,
				SandboxID:                baseStatus.SandboxStatuses[0].Id,
				Attempt:                  uint32(1),
				NextInitContainerToStart: &basePod.Spec.InitContainers[0],
				InitContainersToStart:    []int{0},
				ContainersToStart:        []int{},
				ContainersToKill:         getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"Kill pod and do not restart ephemeral container if the pod sandbox is dead": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyAlways },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.SandboxStatuses[0].State = runtimeapi.PodSandboxState_SANDBOX_NOTREADY
			},
			actions: podActions{
				KillPod:                  true,
				CreateSandbox:            true,
				SandboxID:                baseStatus.SandboxStatuses[0].Id,
				Attempt:                  uint32(1),
				NextInitContainerToStart: &basePod.Spec.InitContainers[0],
				InitContainersToStart:    []int{0},
				ContainersToStart:        []int{},
				ContainersToKill:         getKillMapWithInitContainers(basePod, baseStatus, []int{}),
			},
		},
		"Kill pod if all containers exited except ephemeral container": {
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.RestartPolicy = v1.RestartPolicyNever
			},
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				// all regular containers exited
				for i := 0; i < 3; i++ {
					status.ContainerStatuses[i].State = kubecontainer.ContainerStateExited
					status.ContainerStatuses[i].ExitCode = 0
				}
			},
			actions: podActions{
				SandboxID:         baseStatus.SandboxStatuses[0].Id,
				CreateSandbox:     false,
				KillPod:           true,
				ContainersToStart: []int{},
				ContainersToKill:  map[kubecontainer.ContainerID]containerToKillInfo{},
			},
		},
		"Ephemeral container is in unknown state; leave it alone": {
			mutatePodFn: func(pod *v1.Pod) { pod.Spec.RestartPolicy = v1.RestartPolicyNever },
			mutateStatusFn: func(status *kubecontainer.PodStatus) {
				status.ContainerStatuses[4].State = kubecontainer.ContainerStateUnknown
			},
			actions: noAction,
		},
	} {
		pod, status := makeBasePodAndStatusWithInitAndEphemeralContainers()
		if test.mutatePodFn != nil {
			test.mutatePodFn(pod)
		}
		if test.mutateStatusFn != nil {
			test.mutateStatusFn(status)
		}
		ctx := context.Background()
		actions := m.computePodActions(ctx, pod, status)
		if !sidecarContainersEnabled {
			// If sidecar containers are disabled, we should not see any
			// InitContainersToStart in the actions.
			test.actions.InitContainersToStart = nil
		} else {
			// If sidecar containers are enabled, we should not see any
			// NextInitContainerToStart in the actions.
			test.actions.NextInitContainerToStart = nil
		}
		verifyActions(t, &test.actions, &actions, desc)
	}
}

func TestSyncPodWithSandboxAndDeletedPod(t *testing.T) {
	ctx := context.Background()
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)
	fakeRuntime.ErrorOnSandboxCreate = true

	containers := []v1.Container{
		{
			Name:            "foo1",
			Image:           "busybox",
			ImagePullPolicy: v1.PullIfNotPresent,
		},
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: containers,
		},
	}

	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	m.podStateProvider.(*fakePodStateProvider).removed = map[types.UID]struct{}{pod.UID: {}}

	// GetPodStatus and the following SyncPod will not return errors in the
	// case where the pod has been deleted. We are not adding any pods into
	// the fakePodProvider so they are 'deleted'.
	podStatus, err := m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	result := m.SyncPod(context.Background(), pod, podStatus, []v1.Secret{}, backOff)
	// This will return an error if the pod has _not_ been deleted.
	assert.NoError(t, result.Error())
}

func makeBasePodAndStatusWithInitAndEphemeralContainers() (*v1.Pod, *kubecontainer.PodStatus) {
	pod, status := makeBasePodAndStatus()
	pod.Spec.InitContainers = []v1.Container{
		{
			Name:  "init1",
			Image: "bar-image",
		},
	}
	pod.Spec.EphemeralContainers = []v1.EphemeralContainer{
		{
			EphemeralContainerCommon: v1.EphemeralContainerCommon{
				Name:  "debug",
				Image: "busybox",
			},
		},
	}
	status.ContainerStatuses = append(status.ContainerStatuses, &kubecontainer.Status{
		ID:   kubecontainer.ContainerID{ID: "initid1"},
		Name: "init1", State: kubecontainer.ContainerStateExited,
		Hash: kubecontainer.HashContainer(&pod.Spec.InitContainers[0]),
	}, &kubecontainer.Status{
		ID:   kubecontainer.ContainerID{ID: "debug1"},
		Name: "debug", State: kubecontainer.ContainerStateRunning,
		Hash: kubecontainer.HashContainer((*v1.Container)(&pod.Spec.EphemeralContainers[0].EphemeralContainerCommon)),
	})
	return pod, status
}

func TestComputePodActionsForPodResize(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)()
	fakeRuntime, _, m, err := createTestRuntimeManager()
	m.machineInfo.MemoryCapacity = 17179860387 // 16GB
	assert.NoError(t, err)

	cpu100m := resource.MustParse("100m")
	cpu200m := resource.MustParse("200m")
	mem100M := resource.MustParse("100Mi")
	mem200M := resource.MustParse("200Mi")
	cpuPolicyRestartNotRequired := v1.ContainerResizePolicy{ResourceName: v1.ResourceCPU, RestartPolicy: v1.NotRequired}
	memPolicyRestartNotRequired := v1.ContainerResizePolicy{ResourceName: v1.ResourceMemory, RestartPolicy: v1.NotRequired}
	cpuPolicyRestartRequired := v1.ContainerResizePolicy{ResourceName: v1.ResourceCPU, RestartPolicy: v1.RestartContainer}
	memPolicyRestartRequired := v1.ContainerResizePolicy{ResourceName: v1.ResourceMemory, RestartPolicy: v1.RestartContainer}

	for desc, test := range map[string]struct {
		podResizePolicyFn       func(*v1.Pod)
		mutatePodFn             func(*v1.Pod)
		getExpectedPodActionsFn func(*v1.Pod, *kubecontainer.PodStatus) *podActions
	}{
		"Update container CPU and memory resources": {
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.Containers[1].Resources = v1.ResourceRequirements{
					Limits: v1.ResourceList{v1.ResourceCPU: cpu100m, v1.ResourceMemory: mem100M},
				}
				if idx, found := podutil.GetIndexOfContainerStatus(pod.Status.ContainerStatuses, pod.Spec.Containers[1].Name); found {
					pod.Status.ContainerStatuses[idx].Resources = &v1.ResourceRequirements{
						Limits: v1.ResourceList{v1.ResourceCPU: cpu200m, v1.ResourceMemory: mem200M},
					}
				}
			},
			getExpectedPodActionsFn: func(pod *v1.Pod, podStatus *kubecontainer.PodStatus) *podActions {
				kcs := podStatus.FindContainerStatusByName(pod.Spec.Containers[1].Name)
				pa := podActions{
					SandboxID:         podStatus.SandboxStatuses[0].Id,
					ContainersToStart: []int{},
					ContainersToKill:  getKillMap(pod, podStatus, []int{}),
					ContainersToUpdate: map[v1.ResourceName][]containerToUpdateInfo{
						v1.ResourceMemory: {
							{
								apiContainerIdx: 1,
								kubeContainerID: kcs.ID,
								desiredContainerResources: containerResources{
									memoryLimit: mem100M.Value(),
									cpuLimit:    cpu100m.MilliValue(),
								},
								currentContainerResources: &containerResources{
									memoryLimit: mem200M.Value(),
									cpuLimit:    cpu200m.MilliValue(),
								},
							},
						},
						v1.ResourceCPU: {
							{
								apiContainerIdx: 1,
								kubeContainerID: kcs.ID,
								desiredContainerResources: containerResources{
									memoryLimit: mem100M.Value(),
									cpuLimit:    cpu100m.MilliValue(),
								},
								currentContainerResources: &containerResources{
									memoryLimit: mem200M.Value(),
									cpuLimit:    cpu200m.MilliValue(),
								},
							},
						},
					},
				}
				return &pa
			},
		},
		"Update container CPU resources": {
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.Containers[1].Resources = v1.ResourceRequirements{
					Limits: v1.ResourceList{v1.ResourceCPU: cpu100m, v1.ResourceMemory: mem100M},
				}
				if idx, found := podutil.GetIndexOfContainerStatus(pod.Status.ContainerStatuses, pod.Spec.Containers[1].Name); found {
					pod.Status.ContainerStatuses[idx].Resources = &v1.ResourceRequirements{
						Limits: v1.ResourceList{v1.ResourceCPU: cpu200m, v1.ResourceMemory: mem100M},
					}
				}
			},
			getExpectedPodActionsFn: func(pod *v1.Pod, podStatus *kubecontainer.PodStatus) *podActions {
				kcs := podStatus.FindContainerStatusByName(pod.Spec.Containers[1].Name)
				pa := podActions{
					SandboxID:         podStatus.SandboxStatuses[0].Id,
					ContainersToStart: []int{},
					ContainersToKill:  getKillMap(pod, podStatus, []int{}),
					ContainersToUpdate: map[v1.ResourceName][]containerToUpdateInfo{
						v1.ResourceCPU: {
							{
								apiContainerIdx: 1,
								kubeContainerID: kcs.ID,
								desiredContainerResources: containerResources{
									memoryLimit: mem100M.Value(),
									cpuLimit:    cpu100m.MilliValue(),
								},
								currentContainerResources: &containerResources{
									memoryLimit: mem100M.Value(),
									cpuLimit:    cpu200m.MilliValue(),
								},
							},
						},
					},
				}
				return &pa
			},
		},
		"Update container memory resources": {
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.Containers[2].Resources = v1.ResourceRequirements{
					Limits: v1.ResourceList{v1.ResourceCPU: cpu200m, v1.ResourceMemory: mem200M},
				}
				if idx, found := podutil.GetIndexOfContainerStatus(pod.Status.ContainerStatuses, pod.Spec.Containers[2].Name); found {
					pod.Status.ContainerStatuses[idx].Resources = &v1.ResourceRequirements{
						Limits: v1.ResourceList{v1.ResourceCPU: cpu200m, v1.ResourceMemory: mem100M},
					}
				}
			},
			getExpectedPodActionsFn: func(pod *v1.Pod, podStatus *kubecontainer.PodStatus) *podActions {
				kcs := podStatus.FindContainerStatusByName(pod.Spec.Containers[2].Name)
				pa := podActions{
					SandboxID:         podStatus.SandboxStatuses[0].Id,
					ContainersToStart: []int{},
					ContainersToKill:  getKillMap(pod, podStatus, []int{}),
					ContainersToUpdate: map[v1.ResourceName][]containerToUpdateInfo{
						v1.ResourceMemory: {
							{
								apiContainerIdx: 2,
								kubeContainerID: kcs.ID,
								desiredContainerResources: containerResources{
									memoryLimit: mem200M.Value(),
									cpuLimit:    cpu200m.MilliValue(),
								},
								currentContainerResources: &containerResources{
									memoryLimit: mem100M.Value(),
									cpuLimit:    cpu200m.MilliValue(),
								},
							},
						},
					},
				}
				return &pa
			},
		},
		"Nothing when spec.Resources and status.Resources are equal": {
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.Containers[1].Resources = v1.ResourceRequirements{
					Limits: v1.ResourceList{v1.ResourceCPU: cpu200m},
				}
				pod.Status.ContainerStatuses[1].Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{v1.ResourceCPU: cpu200m},
				}
			},
			getExpectedPodActionsFn: func(pod *v1.Pod, podStatus *kubecontainer.PodStatus) *podActions {
				pa := podActions{
					SandboxID:          podStatus.SandboxStatuses[0].Id,
					ContainersToKill:   getKillMap(pod, podStatus, []int{}),
					ContainersToStart:  []int{},
					ContainersToUpdate: map[v1.ResourceName][]containerToUpdateInfo{},
				}
				return &pa
			},
		},
		"Update container CPU and memory resources with Restart policy for CPU": {
			podResizePolicyFn: func(pod *v1.Pod) {
				pod.Spec.Containers[0].ResizePolicy = []v1.ContainerResizePolicy{cpuPolicyRestartRequired, memPolicyRestartNotRequired}
			},
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.Containers[0].Resources = v1.ResourceRequirements{
					Limits: v1.ResourceList{v1.ResourceCPU: cpu200m, v1.ResourceMemory: mem200M},
				}
				if idx, found := podutil.GetIndexOfContainerStatus(pod.Status.ContainerStatuses, pod.Spec.Containers[0].Name); found {
					pod.Status.ContainerStatuses[idx].Resources = &v1.ResourceRequirements{
						Limits: v1.ResourceList{v1.ResourceCPU: cpu100m, v1.ResourceMemory: mem100M},
					}
				}
			},
			getExpectedPodActionsFn: func(pod *v1.Pod, podStatus *kubecontainer.PodStatus) *podActions {
				kcs := podStatus.FindContainerStatusByName(pod.Spec.Containers[0].Name)
				killMap := make(map[kubecontainer.ContainerID]containerToKillInfo)
				killMap[kcs.ID] = containerToKillInfo{
					container: &pod.Spec.Containers[0],
					name:      pod.Spec.Containers[0].Name,
				}
				pa := podActions{
					SandboxID:          podStatus.SandboxStatuses[0].Id,
					ContainersToStart:  []int{0},
					ContainersToKill:   killMap,
					ContainersToUpdate: map[v1.ResourceName][]containerToUpdateInfo{},
					UpdatePodResources: true,
				}
				return &pa
			},
		},
		"Update container CPU and memory resources with Restart policy for memory": {
			podResizePolicyFn: func(pod *v1.Pod) {
				pod.Spec.Containers[2].ResizePolicy = []v1.ContainerResizePolicy{cpuPolicyRestartNotRequired, memPolicyRestartRequired}
			},
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.Containers[2].Resources = v1.ResourceRequirements{
					Limits: v1.ResourceList{v1.ResourceCPU: cpu200m, v1.ResourceMemory: mem200M},
				}
				if idx, found := podutil.GetIndexOfContainerStatus(pod.Status.ContainerStatuses, pod.Spec.Containers[2].Name); found {
					pod.Status.ContainerStatuses[idx].Resources = &v1.ResourceRequirements{
						Limits: v1.ResourceList{v1.ResourceCPU: cpu100m, v1.ResourceMemory: mem100M},
					}
				}
			},
			getExpectedPodActionsFn: func(pod *v1.Pod, podStatus *kubecontainer.PodStatus) *podActions {
				kcs := podStatus.FindContainerStatusByName(pod.Spec.Containers[2].Name)
				killMap := make(map[kubecontainer.ContainerID]containerToKillInfo)
				killMap[kcs.ID] = containerToKillInfo{
					container: &pod.Spec.Containers[2],
					name:      pod.Spec.Containers[2].Name,
				}
				pa := podActions{
					SandboxID:          podStatus.SandboxStatuses[0].Id,
					ContainersToStart:  []int{2},
					ContainersToKill:   killMap,
					ContainersToUpdate: map[v1.ResourceName][]containerToUpdateInfo{},
					UpdatePodResources: true,
				}
				return &pa
			},
		},
		"Update container memory resources with Restart policy for CPU": {
			podResizePolicyFn: func(pod *v1.Pod) {
				pod.Spec.Containers[1].ResizePolicy = []v1.ContainerResizePolicy{cpuPolicyRestartRequired, memPolicyRestartNotRequired}
			},
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.Containers[1].Resources = v1.ResourceRequirements{
					Limits: v1.ResourceList{v1.ResourceCPU: cpu100m, v1.ResourceMemory: mem200M},
				}
				if idx, found := podutil.GetIndexOfContainerStatus(pod.Status.ContainerStatuses, pod.Spec.Containers[1].Name); found {
					pod.Status.ContainerStatuses[idx].Resources = &v1.ResourceRequirements{
						Limits: v1.ResourceList{v1.ResourceCPU: cpu100m, v1.ResourceMemory: mem100M},
					}
				}
			},
			getExpectedPodActionsFn: func(pod *v1.Pod, podStatus *kubecontainer.PodStatus) *podActions {
				kcs := podStatus.FindContainerStatusByName(pod.Spec.Containers[1].Name)
				pa := podActions{
					SandboxID:         podStatus.SandboxStatuses[0].Id,
					ContainersToStart: []int{},
					ContainersToKill:  getKillMap(pod, podStatus, []int{}),
					ContainersToUpdate: map[v1.ResourceName][]containerToUpdateInfo{
						v1.ResourceMemory: {
							{
								apiContainerIdx: 1,
								kubeContainerID: kcs.ID,
								desiredContainerResources: containerResources{
									memoryLimit: mem200M.Value(),
									cpuLimit:    cpu100m.MilliValue(),
								},
								currentContainerResources: &containerResources{
									memoryLimit: mem100M.Value(),
									cpuLimit:    cpu100m.MilliValue(),
								},
							},
						},
					},
				}
				return &pa
			},
		},
		"Update container CPU resources with Restart policy for memory": {
			podResizePolicyFn: func(pod *v1.Pod) {
				pod.Spec.Containers[2].ResizePolicy = []v1.ContainerResizePolicy{cpuPolicyRestartNotRequired, memPolicyRestartRequired}
			},
			mutatePodFn: func(pod *v1.Pod) {
				pod.Spec.Containers[2].Resources = v1.ResourceRequirements{
					Limits: v1.ResourceList{v1.ResourceCPU: cpu200m, v1.ResourceMemory: mem100M},
				}
				if idx, found := podutil.GetIndexOfContainerStatus(pod.Status.ContainerStatuses, pod.Spec.Containers[2].Name); found {
					pod.Status.ContainerStatuses[idx].Resources = &v1.ResourceRequirements{
						Limits: v1.ResourceList{v1.ResourceCPU: cpu100m, v1.ResourceMemory: mem100M},
					}
				}
			},
			getExpectedPodActionsFn: func(pod *v1.Pod, podStatus *kubecontainer.PodStatus) *podActions {
				kcs := podStatus.FindContainerStatusByName(pod.Spec.Containers[2].Name)
				pa := podActions{
					SandboxID:         podStatus.SandboxStatuses[0].Id,
					ContainersToStart: []int{},
					ContainersToKill:  getKillMap(pod, podStatus, []int{}),
					ContainersToUpdate: map[v1.ResourceName][]containerToUpdateInfo{
						v1.ResourceCPU: {
							{
								apiContainerIdx: 2,
								kubeContainerID: kcs.ID,
								desiredContainerResources: containerResources{
									memoryLimit: mem100M.Value(),
									cpuLimit:    cpu200m.MilliValue(),
								},
								currentContainerResources: &containerResources{
									memoryLimit: mem100M.Value(),
									cpuLimit:    cpu100m.MilliValue(),
								},
							},
						},
					},
				}
				return &pa
			},
		},
	} {
		pod, kps := makeBasePodAndStatus()
		for idx := range pod.Spec.Containers {
			// default resize policy when pod resize feature is enabled
			pod.Spec.Containers[idx].ResizePolicy = []v1.ContainerResizePolicy{cpuPolicyRestartNotRequired, memPolicyRestartNotRequired}
		}
		if test.podResizePolicyFn != nil {
			test.podResizePolicyFn(pod)
		}
		for idx := range pod.Spec.Containers {
			// compute hash
			if kcs := kps.FindContainerStatusByName(pod.Spec.Containers[idx].Name); kcs != nil {
				kcs.Hash = kubecontainer.HashContainer(&pod.Spec.Containers[idx])
			}
		}
		makeAndSetFakePod(t, m, fakeRuntime, pod)
		ctx := context.Background()
		status, _ := m.GetPodStatus(ctx, kps.ID, pod.Name, pod.Namespace)
		for idx := range pod.Spec.Containers {
			if rcs := status.FindContainerStatusByName(pod.Spec.Containers[idx].Name); rcs != nil {
				if csIdx, found := podutil.GetIndexOfContainerStatus(pod.Status.ContainerStatuses, pod.Spec.Containers[idx].Name); found {
					pod.Status.ContainerStatuses[csIdx].ContainerID = rcs.ID.String()
				}
			}
		}
		for idx := range pod.Spec.Containers {
			if kcs := kps.FindContainerStatusByName(pod.Spec.Containers[idx].Name); kcs != nil {
				kcs.Hash = kubecontainer.HashContainer(&pod.Spec.Containers[idx])
			}
		}
		if test.mutatePodFn != nil {
			test.mutatePodFn(pod)
		}
		expectedActions := test.getExpectedPodActionsFn(pod, status)
		actions := m.computePodActions(ctx, pod, status)
		verifyActions(t, expectedActions, &actions, desc)
	}
}

func TestUpdatePodContainerResources(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)()
	fakeRuntime, _, m, err := createTestRuntimeManager()
	m.machineInfo.MemoryCapacity = 17179860387 // 16GB
	assert.NoError(t, err)

	cpu100m := resource.MustParse("100m")
	cpu150m := resource.MustParse("150m")
	cpu200m := resource.MustParse("200m")
	cpu250m := resource.MustParse("250m")
	cpu300m := resource.MustParse("300m")
	cpu350m := resource.MustParse("350m")
	mem100M := resource.MustParse("100Mi")
	mem150M := resource.MustParse("150Mi")
	mem200M := resource.MustParse("200Mi")
	mem250M := resource.MustParse("250Mi")
	mem300M := resource.MustParse("300Mi")
	mem350M := resource.MustParse("350Mi")
	res100m100Mi := v1.ResourceList{v1.ResourceCPU: cpu100m, v1.ResourceMemory: mem100M}
	res150m100Mi := v1.ResourceList{v1.ResourceCPU: cpu150m, v1.ResourceMemory: mem100M}
	res100m150Mi := v1.ResourceList{v1.ResourceCPU: cpu100m, v1.ResourceMemory: mem150M}
	res150m150Mi := v1.ResourceList{v1.ResourceCPU: cpu150m, v1.ResourceMemory: mem150M}
	res200m200Mi := v1.ResourceList{v1.ResourceCPU: cpu200m, v1.ResourceMemory: mem200M}
	res250m200Mi := v1.ResourceList{v1.ResourceCPU: cpu250m, v1.ResourceMemory: mem200M}
	res200m250Mi := v1.ResourceList{v1.ResourceCPU: cpu200m, v1.ResourceMemory: mem250M}
	res250m250Mi := v1.ResourceList{v1.ResourceCPU: cpu250m, v1.ResourceMemory: mem250M}
	res300m300Mi := v1.ResourceList{v1.ResourceCPU: cpu300m, v1.ResourceMemory: mem300M}
	res350m300Mi := v1.ResourceList{v1.ResourceCPU: cpu350m, v1.ResourceMemory: mem300M}
	res300m350Mi := v1.ResourceList{v1.ResourceCPU: cpu300m, v1.ResourceMemory: mem350M}
	res350m350Mi := v1.ResourceList{v1.ResourceCPU: cpu350m, v1.ResourceMemory: mem350M}

	pod, _ := makeBasePodAndStatus()
	makeAndSetFakePod(t, m, fakeRuntime, pod)

	for dsc, tc := range map[string]struct {
		resourceName            v1.ResourceName
		apiSpecResources        []v1.ResourceRequirements
		apiStatusResources      []v1.ResourceRequirements
		requiresRestart         []bool
		invokeUpdateResources   bool
		expectedCurrentLimits   []v1.ResourceList
		expectedCurrentRequests []v1.ResourceList
	}{
		"Guaranteed QoS Pod - CPU & memory resize requested, update CPU": {
			resourceName: v1.ResourceCPU,
			apiSpecResources: []v1.ResourceRequirements{
				{Limits: res150m150Mi, Requests: res150m150Mi},
				{Limits: res250m250Mi, Requests: res250m250Mi},
				{Limits: res350m350Mi, Requests: res350m350Mi},
			},
			apiStatusResources: []v1.ResourceRequirements{
				{Limits: res100m100Mi, Requests: res100m100Mi},
				{Limits: res200m200Mi, Requests: res200m200Mi},
				{Limits: res300m300Mi, Requests: res300m300Mi},
			},
			requiresRestart:         []bool{false, false, false},
			invokeUpdateResources:   true,
			expectedCurrentLimits:   []v1.ResourceList{res150m100Mi, res250m200Mi, res350m300Mi},
			expectedCurrentRequests: []v1.ResourceList{res150m100Mi, res250m200Mi, res350m300Mi},
		},
		"Guaranteed QoS Pod - CPU & memory resize requested, update memory": {
			resourceName: v1.ResourceMemory,
			apiSpecResources: []v1.ResourceRequirements{
				{Limits: res150m150Mi, Requests: res150m150Mi},
				{Limits: res250m250Mi, Requests: res250m250Mi},
				{Limits: res350m350Mi, Requests: res350m350Mi},
			},
			apiStatusResources: []v1.ResourceRequirements{
				{Limits: res100m100Mi, Requests: res100m100Mi},
				{Limits: res200m200Mi, Requests: res200m200Mi},
				{Limits: res300m300Mi, Requests: res300m300Mi},
			},
			requiresRestart:         []bool{false, false, false},
			invokeUpdateResources:   true,
			expectedCurrentLimits:   []v1.ResourceList{res100m150Mi, res200m250Mi, res300m350Mi},
			expectedCurrentRequests: []v1.ResourceList{res100m150Mi, res200m250Mi, res300m350Mi},
		},
	} {
		var containersToUpdate []containerToUpdateInfo
		for idx := range pod.Spec.Containers {
			// default resize policy when pod resize feature is enabled
			pod.Spec.Containers[idx].Resources = tc.apiSpecResources[idx]
			pod.Status.ContainerStatuses[idx].Resources = &tc.apiStatusResources[idx]
			cInfo := containerToUpdateInfo{
				apiContainerIdx: idx,
				kubeContainerID: kubecontainer.ContainerID{},
				desiredContainerResources: containerResources{
					memoryLimit:   tc.apiSpecResources[idx].Limits.Memory().Value(),
					memoryRequest: tc.apiSpecResources[idx].Requests.Memory().Value(),
					cpuLimit:      tc.apiSpecResources[idx].Limits.Cpu().MilliValue(),
					cpuRequest:    tc.apiSpecResources[idx].Requests.Cpu().MilliValue(),
				},
				currentContainerResources: &containerResources{
					memoryLimit:   tc.apiStatusResources[idx].Limits.Memory().Value(),
					memoryRequest: tc.apiStatusResources[idx].Requests.Memory().Value(),
					cpuLimit:      tc.apiStatusResources[idx].Limits.Cpu().MilliValue(),
					cpuRequest:    tc.apiStatusResources[idx].Requests.Cpu().MilliValue(),
				},
			}
			containersToUpdate = append(containersToUpdate, cInfo)
		}
		fakeRuntime.Called = []string{}
		err := m.updatePodContainerResources(pod, tc.resourceName, containersToUpdate)
		assert.NoError(t, err, dsc)

		if tc.invokeUpdateResources {
			assert.Contains(t, fakeRuntime.Called, "UpdateContainerResources", dsc)
		}
		for idx := range pod.Spec.Containers {
			assert.Equal(t, tc.expectedCurrentLimits[idx].Memory().Value(), containersToUpdate[idx].currentContainerResources.memoryLimit, dsc)
			assert.Equal(t, tc.expectedCurrentRequests[idx].Memory().Value(), containersToUpdate[idx].currentContainerResources.memoryRequest, dsc)
			assert.Equal(t, tc.expectedCurrentLimits[idx].Cpu().MilliValue(), containersToUpdate[idx].currentContainerResources.cpuLimit, dsc)
			assert.Equal(t, tc.expectedCurrentRequests[idx].Cpu().MilliValue(), containersToUpdate[idx].currentContainerResources.cpuRequest, dsc)
		}
	}
}
