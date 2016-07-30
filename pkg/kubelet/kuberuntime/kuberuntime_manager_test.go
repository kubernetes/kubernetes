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
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

func createTestFakeRuntimeManager() (*fakeKubeRuntime, *kubeGenericRuntimeManager, error) {
	fakeRuntime, manager, _, err := createTestFakeRuntimeManagerWithBackOff()
	return fakeRuntime, manager, err
}

func createTestFakeRuntimeManagerWithBackOff() (*fakeKubeRuntime, *kubeGenericRuntimeManager, *flowcontrol.Backoff, error) {
	fakeRuntime := NewFakeKubeRuntime()
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	manager, err := NewFakeKubeRuntimeManager(fakeRuntime, nettest.NewFakeHost(nil), &containertest.FakeOS{})
	return fakeRuntime, manager, backOff, err
}

func makeAndSetFakePod(m *kubeGenericRuntimeManager, fakeRuntime *fakeKubeRuntime, pod *api.Pod) error {
	fakePodSandbox, err := makeFakePodSandboxWithState(m, pod)
	if err != nil {
		return err
	}

	fakeContainers, err := makeFakeContainersWithState(m, pod, pod.Spec.Containers)
	if err != nil {
		return err
	}

	fakeRuntime.SetFakeSandboxes([]*PodSandboxWithState{fakePodSandbox})
	fakeRuntime.SetFakeContainers(fakeContainers)
	return nil
}

func makeFakePodSandboxWithState(m *kubeGenericRuntimeManager, pod *api.Pod) (*PodSandboxWithState, error) {
	config, err := m.generatePodSandboxConfig(pod, "")
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
	sandboxConfig, err := m.generatePodSandboxConfig(pod, "")
	if err != nil {
		return nil, err
	}

	containerConfig, err := m.generateContainerConfig(&container, pod, 0, "")
	if err != nil {
		return nil, err
	}

	return &ContainerWithState{
		createdAt:       10000,
		containerID:     containerConfig.GetName(),
		podSandboxID:    sandboxConfig.GetName(),
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
	_, m, err := createTestFakeRuntimeManager()
	assert.NoError(t, err)

	version, err := m.Version()
	assert.NoError(t, err)
	assert.Equal(t, kubeRuntimeAPIVersion, version.String())
}

func TestContainerRuntimeType(t *testing.T) {
	_, m, err := createTestFakeRuntimeManager()
	assert.NoError(t, err)

	runtimeType := m.Type()
	assert.Equal(t, fakeRuntimeName, runtimeType)
}

func TestContainerRuntimeStatus(t *testing.T) {
	_, m, err := createTestFakeRuntimeManager()
	assert.NoError(t, err)

	err = m.Status()
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
	fakeRuntime, m, backOff, err := createTestFakeRuntimeManagerWithBackOff()
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
	fakeRuntime, m, err := createTestFakeRuntimeManager()
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

	err = makeAndSetFakePod(m, fakeRuntime, pod)
	assert.NoError(t, err)

	actual, err := m.GetPods(false)
	assert.NoError(t, err)

	expected := []*kubecontainer.Pod{fakeRuntime.getExpectedKubePod(pod)}
	if !verifyPods(expected, actual) {
		t.Errorf("expected %#q, got %#q", expected, actual)
	}
}

func TestGetPodStatus(t *testing.T) {
	fakeRuntime, m, backOff, err := createTestFakeRuntimeManagerWithBackOff()
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
	fakeRuntime, m, err := createTestFakeRuntimeManager()
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

	err = makeAndSetFakePod(m, fakeRuntime, pod)
	assert.NoError(t, err)

	runningPod := fakeRuntime.getExpectedKubePod(pod)
	err = m.KillPod(pod, *runningPod, nil)
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
