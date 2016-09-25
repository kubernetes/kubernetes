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
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	apitest "k8s.io/kubernetes/pkg/kubelet/api/testing"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/network"
	nettest "k8s.io/kubernetes/pkg/kubelet/network/testing"
	kubetypes "k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

var (
	fakeCreatedAt int64 = 1
)

func createTestRuntimeManager() (*apitest.FakeRuntimeService, *apitest.FakeImageService, *kubeGenericRuntimeManager, error) {
	fakeRuntimeService := apitest.NewFakeRuntimeService()
	fakeImageService := apitest.NewFakeImageService()
	networkPlugin, _ := network.InitNetworkPlugin(
		[]network.NetworkPlugin{},
		"",
		nettest.NewFakeHost(nil),
		componentconfig.HairpinNone,
		"10.0.0.0/8",
		network.UseDefaultMTU,
	)
	osInterface := &containertest.FakeOS{}
	manager, err := NewFakeKubeRuntimeManager(fakeRuntimeService, fakeImageService, networkPlugin, osInterface)
	return fakeRuntimeService, fakeImageService, manager, err
}

func makeAndSetFakePod(m *kubeGenericRuntimeManager, fakeRuntime *apitest.FakeRuntimeService, pod *api.Pod) (*apitest.FakePodSandbox, []*apitest.FakeContainer, error) {
	fakePodSandbox, err := makeFakePodSandbox(m, pod, fakeCreatedAt)
	if err != nil {
		return nil, nil, err
	}

	fakeContainers, err := makeFakeContainers(m, pod, pod.Spec.Containers, fakeCreatedAt)
	if err != nil {
		return nil, nil, err
	}

	fakeRuntime.SetFakeSandboxes([]*apitest.FakePodSandbox{fakePodSandbox})
	fakeRuntime.SetFakeContainers(fakeContainers)
	return fakePodSandbox, fakeContainers, nil
}

func makeFakePodSandbox(m *kubeGenericRuntimeManager, pod *api.Pod, createdAt int64) (*apitest.FakePodSandbox, error) {
	config, err := m.generatePodSandboxConfig(pod, 0)
	if err != nil {
		return nil, err
	}

	podSandboxID := apitest.BuildSandboxName(config.Metadata)
	readyState := runtimeApi.PodSandBoxState_READY
	return &apitest.FakePodSandbox{
		PodSandboxStatus: runtimeApi.PodSandboxStatus{
			Id:        &podSandboxID,
			Metadata:  config.Metadata,
			State:     &readyState,
			CreatedAt: &createdAt,
			Labels:    config.Labels,
		},
	}, nil
}

func makeFakeContainer(m *kubeGenericRuntimeManager, pod *api.Pod, container api.Container, sandboxConfig *runtimeApi.PodSandboxConfig, createdAt int64) (*apitest.FakeContainer, error) {
	containerConfig, err := m.generateContainerConfig(&container, pod, 0, "")
	if err != nil {
		return nil, err
	}

	containerID := apitest.BuildContainerName(containerConfig.Metadata)
	podSandboxID := apitest.BuildSandboxName(sandboxConfig.Metadata)
	runningState := runtimeApi.ContainerState_RUNNING
	imageRef := containerConfig.Image.GetImage()
	return &apitest.FakeContainer{
		ContainerStatus: runtimeApi.ContainerStatus{
			Id:          &containerID,
			Metadata:    containerConfig.Metadata,
			Image:       containerConfig.Image,
			ImageRef:    &imageRef,
			CreatedAt:   &createdAt,
			State:       &runningState,
			Labels:      containerConfig.Labels,
			Annotations: containerConfig.Annotations,
		},
		SandboxID: podSandboxID,
	}, nil
}

func makeFakeContainers(m *kubeGenericRuntimeManager, pod *api.Pod, containers []api.Container, createdAt int64) ([]*apitest.FakeContainer, error) {
	sandboxConfig, err := m.generatePodSandboxConfig(pod, 0)
	if err != nil {
		return nil, err
	}

	result := make([]*apitest.FakeContainer, len(containers))
	for idx, c := range containers {
		containerWithState, err := makeFakeContainer(m, pod, c, sandboxConfig, createdAt)
		if err != nil {
			return nil, err
		}

		result[idx] = containerWithState
	}

	return result, nil
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
	_, _, err = makeAndSetFakePod(m, fakeRuntime, pod)
	assert.NoError(t, err)

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
	fakeSandbox, fakeContainers, err := makeAndSetFakePod(m, fakeRuntime, pod)
	assert.NoError(t, err)

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
	// Convert fakeSandbox to kubecontainer.Container
	sandbox, err := m.sandboxToKubeContainer(&runtimeApi.PodSandbox{
		Id:        fakeSandbox.Id,
		Metadata:  fakeSandbox.Metadata,
		State:     fakeSandbox.State,
		CreatedAt: fakeSandbox.CreatedAt,
		Labels:    fakeSandbox.Labels,
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
		t.Errorf("expected %#v, got %#v", expected, actual)
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
	fakeSandbox, _, err := makeAndSetFakePod(m, fakeRuntime, pod)
	assert.NoError(t, err)

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
	sandbox, _, err := makeAndSetFakePod(m, fakeRuntime, pod)
	assert.NoError(t, err)

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
	fakeSandbox, fakeContainers, err := makeAndSetFakePod(m, fakeRuntime, pod)
	assert.NoError(t, err)

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
		assert.Equal(t, runtimeApi.PodSandBoxState_NOTREADY, sandbox.GetState())
	}
	for _, c := range fakeRuntime.Containers {
		assert.Equal(t, runtimeApi.ContainerState_EXITED, c.GetState())
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
		assert.Equal(t, runtimeApi.PodSandBoxState_READY, sandbox.GetState())
	}
	for _, c := range fakeRuntime.Containers {
		assert.Equal(t, runtimeApi.ContainerState_RUNNING, c.GetState())
	}
}
