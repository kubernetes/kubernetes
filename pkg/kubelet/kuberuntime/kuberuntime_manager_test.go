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

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	nettest "k8s.io/kubernetes/pkg/kubelet/network/testing"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

func createTestFakeRuntimeManager() (*fakeKubeRuntime, *kubeGenericRuntimeManager) {
	fakeRuntime, manager, _ := createTestFakeRuntimeManagerWithBackOff()
	return fakeRuntime, manager
}

func createTestFakeRuntimeManagerWithBackOff() (*fakeKubeRuntime, *kubeGenericRuntimeManager, *flowcontrol.Backoff) {
	fakeRuntime := NewFakeKubeRuntime()
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	manager := NewFakeKubeRuntimeManager(fakeRuntime, nettest.NewFakeHost(nil), &containertest.FakeOS{})
	return fakeRuntime, manager, backOff
}

func makeAndSetFakePod(m *kubeGenericRuntimeManager, fakeRuntime *fakeKubeRuntime, pod *api.Pod) (*PodSandboxWithState, []*ContainerWithState, error) {
	fakePodSandbox, err := makeFakePodSandboxWithState(m, pod)
	if err != nil {
		return nil, nil, err
	}

	fakeContainers, err := makeFakeContainersWithState(m, pod, pod.Spec.Containers)
	if err != nil {
		return nil, nil, err
	}

	fakeRuntime.SetFakeSandboxes([]*PodSandboxWithState{fakePodSandbox})
	fakeRuntime.SetFakeContainers(fakeContainers)
	return fakePodSandbox, fakeContainers, nil
}

func makeFakePodSandboxWithState(m *kubeGenericRuntimeManager, pod *api.Pod) (*PodSandboxWithState, error) {
	config, err := m.generatePodSandboxConfig(pod)
	if err != nil {
		return nil, err
	}

	return &PodSandboxWithState{
		config:    config,
		state:     runtimeApi.PodSandBoxState_READY,
		createdAt: 10000,
	}, nil
}

func makeFakeContainerWithState(m *kubeGenericRuntimeManager, pod *api.Pod, container api.Container) (*ContainerWithState, error) {
	sandboxConfig, err := m.generatePodSandboxConfig(pod)
	if err != nil {
		return nil, err
	}

	containerConfig, err := m.generateContainerConfig(&container, pod, 0)
	if err != nil {
		return nil, err
	}

	return &ContainerWithState{
		createdAt:       10000,
		containerID:     containerConfig.GetName(),
		podSandBoxID:    sandboxConfig.GetName(),
		containerConfig: containerConfig,
		sandboxConfig:   sandboxConfig,
		imageID:         containerConfig.Image.GetImage(),
		state:           runtimeApi.ContainerState_RUNNING,
	}, nil
}

func makeFakeContainersWithState(m *kubeGenericRuntimeManager, pod *api.Pod, containers []api.Container) ([]*ContainerWithState, error) {
	result := make([]*ContainerWithState, len(containers))

	for idx, c := range containers {
		containerWithState, err := makeFakeContainerWithState(m, pod, c)
		if err != nil {
			return nil, err
		}

		result[idx] = containerWithState
	}

	return result, nil
}

func TestNewKubeRuntimeManager(t *testing.T) {
	_, m := createTestFakeRuntimeManager()
	version, err := m.Version()
	assert.NoError(t, err)
	assert.Equal(t, kubeRuntimeAPIVersion, version.String())
}

func TestContainerRuntimeType(t *testing.T) {
	_, m := createTestFakeRuntimeManager()
	runtimeType := m.Type()
	assert.Equal(t, fakeRuntimeName, runtimeType)
}

func TestContainerRuntimeStatus(t *testing.T) {
	_, m := createTestFakeRuntimeManager()
	err := m.Status()
	assert.NoError(t, err)
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

// verifyPodStatus returns true if the two pod status are equal.
func verifyPodStatus(a, b *kubecontainer.PodStatus) bool {
	if a == nil || b == nil {
		return false
	}

	sort.Sort(containerStatusByID(a.ContainerStatuses))
	sort.Sort(containerStatusByID(b.ContainerStatuses))

	return reflect.DeepEqual(a, b)
}

func TestSyncPod(t *testing.T) {
	fakeRuntime, m, backOff := createTestFakeRuntimeManagerWithBackOff()

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

	result := m.SyncPod(pod, api.PodStatus{}, &kubecontainer.PodStatus{}, []api.Secret{}, backOff)
	assert.NoError(t, result.Error())
	assert.Equal(t, 2, len(fakeRuntime.Containers))
	assert.Equal(t, 1, len(fakeRuntime.Images))
	assert.Equal(t, 1, len(fakeRuntime.Sandboxes))
	for _, sandbox := range fakeRuntime.Sandboxes {
		assert.Equal(t, runtimeApi.PodSandBoxState_READY, sandbox.state)
	}
	for _, c := range fakeRuntime.Containers {
		assert.Equal(t, runtimeApi.ContainerState_RUNNING, c.state)
	}
}

func TestGetPods(t *testing.T) {
	fakeRuntime, m := createTestFakeRuntimeManager()

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
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
			},
		},
	}
	_, fakeContainers, err := makeAndSetFakePod(m, fakeRuntime, pod)
	assert.NoError(t, err)

	// make expected pod
	expectedContainers := make([]*kubecontainer.Container, len(fakeContainers))
	for i := range expectedContainers {
		expectedContainers[i] = fakeContainers[i].toKubeContainer()
	}
	expected := []*kubecontainer.Pod{
		{
			ID:         types.UID("12345678"),
			Name:       "foo",
			Namespace:  "new",
			Containers: []*kubecontainer.Container{expectedContainers[0], expectedContainers[1]},
		},
	}

	actual, err := m.GetPods(false)
	assert.NoError(t, err)
	if !verifyPods(expected, actual) {
		t.Errorf("expected %#q, got %#q", expected, actual)
	}
}

func TestGetPodStatus(t *testing.T) {
	fakeRuntime, m, backOff := createTestFakeRuntimeManagerWithBackOff()

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

	result := m.SyncPod(pod, api.PodStatus{}, &kubecontainer.PodStatus{}, []api.Secret{}, backOff)
	assert.NoError(t, result.Error())

	expected, err := fakeRuntime.getExpectedKubePodStatus(pod)
	assert.NoError(t, err)
	actual, err := m.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	if !verifyPodStatus(expected, actual) {
		t.Errorf("expected %#q, got %#q", expected, actual)
	}
}

func TestKillPod(t *testing.T) {
	fakeRuntime, m := createTestFakeRuntimeManager()

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
	_, fakeContainers, err := makeAndSetFakePod(m, fakeRuntime, pod)
	assert.NoError(t, err)

	// make running pod
	expectedContainers := make([]*kubecontainer.Container, len(fakeContainers))
	for i := range expectedContainers {
		expectedContainers[i] = fakeContainers[i].toKubeContainer()
	}
	runningPod := kubecontainer.Pod{
		ID:         types.UID("12345678"),
		Name:       "foo",
		Namespace:  "new",
		Containers: []*kubecontainer.Container{expectedContainers[0], expectedContainers[1]},
	}

	err = m.KillPod(pod, runningPod, nil)
	assert.NoError(t, err)
	assert.Equal(t, 2, len(fakeRuntime.Containers))
	assert.Equal(t, 1, len(fakeRuntime.Sandboxes))
	for _, sandbox := range fakeRuntime.Sandboxes {
		assert.Equal(t, runtimeApi.PodSandBoxState_NOTREADY, sandbox.state)
	}
	for _, c := range fakeRuntime.Containers {
		assert.Equal(t, runtimeApi.ContainerState_EXITED, c.state)
	}
}
