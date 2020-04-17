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
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
)

func TestSandboxGC(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	podStateProvider := m.containerGC.podStateProvider.(*fakePodStateProvider)
	makeGCSandbox := func(pod *v1.Pod, attempt uint32, state runtimeapi.PodSandboxState, withPodStateProvider bool, createdAt int64) sandboxTemplate {
		if withPodStateProvider {
			// initialize the pod getter
			podStateProvider.existingPods[pod.UID] = struct{}{}
		}
		return sandboxTemplate{
			pod:       pod,
			state:     state,
			attempt:   attempt,
			createdAt: createdAt,
		}
	}

	pods := []*v1.Pod{
		makeTestPod("foo1", "new", "1234", []v1.Container{
			makeTestContainer("bar1", "busybox"),
			makeTestContainer("bar2", "busybox"),
		}),
		makeTestPod("foo2", "new", "5678", []v1.Container{
			makeTestContainer("bar3", "busybox"),
		}),
		makeTestPod("deleted", "new", "9012", []v1.Container{
			makeTestContainer("bar4", "busybox"),
		}),
	}

	for c, test := range []struct {
		description         string              // description of the test case
		sandboxes           []sandboxTemplate   // templates of sandboxes
		containers          []containerTemplate // templates of containers
		remain              []int               // template indexes of remaining sandboxes
		evictTerminatedPods bool
	}{
		{
			description: "notready sandboxes without containers for deleted pods should be garbage collected.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[2], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, false, 0),
			},
			containers:          []containerTemplate{},
			remain:              []int{},
			evictTerminatedPods: false,
		},
		{
			description: "ready sandboxes without containers for deleted pods should not be garbage collected.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[2], 0, runtimeapi.PodSandboxState_SANDBOX_READY, false, 0),
			},
			containers:          []containerTemplate{},
			remain:              []int{0},
			evictTerminatedPods: false,
		},
		{
			description: "sandboxes for existing pods should not be garbage collected.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[0], 0, runtimeapi.PodSandboxState_SANDBOX_READY, true, 0),
				makeGCSandbox(pods[1], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, 0),
			},
			containers:          []containerTemplate{},
			remain:              []int{0, 1},
			evictTerminatedPods: false,
		},
		{
			description: "older exited sandboxes without containers for existing pods should be garbage collected if there are more than one exited sandboxes.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[0], 1, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, 1),
				makeGCSandbox(pods[0], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, 0),
			},
			containers:          []containerTemplate{},
			remain:              []int{0},
			evictTerminatedPods: false,
		},
		{
			description: "older exited sandboxes with containers for existing pods should not be garbage collected even if there are more than one exited sandboxes.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[0], 1, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, 1),
				makeGCSandbox(pods[0], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, 0),
			},
			containers: []containerTemplate{
				{pod: pods[0], container: &pods[0].Spec.Containers[0], sandboxAttempt: 0, state: runtimeapi.ContainerState_CONTAINER_EXITED},
			},
			remain:              []int{0, 1},
			evictTerminatedPods: false,
		},
		{
			description: "non-running sandboxes for existing pods should be garbage collected if evictTerminatedPods is set.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[0], 0, runtimeapi.PodSandboxState_SANDBOX_READY, true, 0),
				makeGCSandbox(pods[1], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, 0),
			},
			containers:          []containerTemplate{},
			remain:              []int{0},
			evictTerminatedPods: true,
		},
		{
			description: "sandbox with containers should not be garbage collected.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[0], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, false, 0),
			},
			containers: []containerTemplate{
				{pod: pods[0], container: &pods[0].Spec.Containers[0], state: runtimeapi.ContainerState_CONTAINER_EXITED},
			},
			remain:              []int{0},
			evictTerminatedPods: false,
		},
		{
			description: "multiple sandboxes should be handled properly.",
			sandboxes: []sandboxTemplate{
				// running sandbox.
				makeGCSandbox(pods[0], 1, runtimeapi.PodSandboxState_SANDBOX_READY, true, 1),
				// exited sandbox without containers.
				makeGCSandbox(pods[0], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, 0),
				// exited sandbox with containers.
				makeGCSandbox(pods[1], 1, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, 1),
				// exited sandbox without containers.
				makeGCSandbox(pods[1], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, 0),
				// exited sandbox without containers for deleted pods.
				makeGCSandbox(pods[2], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, false, 0),
			},
			containers: []containerTemplate{
				{pod: pods[1], container: &pods[1].Spec.Containers[0], sandboxAttempt: 1, state: runtimeapi.ContainerState_CONTAINER_EXITED},
			},
			remain:              []int{0, 2},
			evictTerminatedPods: false,
		},
	} {
		t.Logf("TestCase #%d: %+v", c, test)
		fakeSandboxes := makeFakePodSandboxes(t, m, test.sandboxes)
		fakeContainers := makeFakeContainers(t, m, test.containers)
		fakeRuntime.SetFakeSandboxes(fakeSandboxes)
		fakeRuntime.SetFakeContainers(fakeContainers)

		err := m.containerGC.evictSandboxes(test.evictTerminatedPods)
		assert.NoError(t, err)
		realRemain, err := fakeRuntime.ListPodSandbox(nil)
		assert.NoError(t, err)
		assert.Len(t, realRemain, len(test.remain))
		for _, remain := range test.remain {
			status, err := fakeRuntime.PodSandboxStatus(fakeSandboxes[remain].Id)
			assert.NoError(t, err)
			assert.Equal(t, &fakeSandboxes[remain].PodSandboxStatus, status)
		}
	}
}

func makeGCContainer(podStateProvider *fakePodStateProvider, podName, containerName string, attempt int, createdAt int64, state runtimeapi.ContainerState) containerTemplate {
	container := makeTestContainer(containerName, "test-image")
	pod := makeTestPod(podName, "test-ns", podName, []v1.Container{container})
	if podName == "running" {
		podStateProvider.runningPods[pod.UID] = struct{}{}
	}
	if podName != "deleted" {
		podStateProvider.existingPods[pod.UID] = struct{}{}
	}
	return containerTemplate{
		pod:       pod,
		container: &container,
		attempt:   attempt,
		createdAt: createdAt,
		state:     state,
	}
}

func TestContainerGC(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	podStateProvider := m.containerGC.podStateProvider.(*fakePodStateProvider)
	defaultGCPolicy := kubecontainer.ContainerGCPolicy{MinAge: time.Hour, MaxPerPodContainer: 2, MaxContainers: 6}

	for c, test := range []struct {
		description         string                           // description of the test case
		containers          []containerTemplate              // templates of containers
		policy              *kubecontainer.ContainerGCPolicy // container gc policy
		remain              []int                            // template indexes of remaining containers
		evictTerminatedPods bool
		allSourcesReady     bool
	}{
		{
			description: "all containers should be removed when max container limit is 0",
			containers: []containerTemplate{
				makeGCContainer(podStateProvider, "foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			policy:              &kubecontainer.ContainerGCPolicy{MinAge: time.Minute, MaxPerPodContainer: 1, MaxContainers: 0},
			remain:              []int{},
			evictTerminatedPods: false,
			allSourcesReady:     true,
		},
		{
			description: "max containers should be complied when no max per pod container limit is set",
			containers: []containerTemplate{
				makeGCContainer(podStateProvider, "foo", "bar", 4, 4, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 3, 3, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			policy:              &kubecontainer.ContainerGCPolicy{MinAge: time.Minute, MaxPerPodContainer: -1, MaxContainers: 4},
			remain:              []int{0, 1, 2, 3},
			evictTerminatedPods: false,
			allSourcesReady:     true,
		},
		{
			description: "no containers should be removed if both max container and per pod container limits are not set",
			containers: []containerTemplate{
				makeGCContainer(podStateProvider, "foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			policy:              &kubecontainer.ContainerGCPolicy{MinAge: time.Minute, MaxPerPodContainer: -1, MaxContainers: -1},
			remain:              []int{0, 1, 2},
			evictTerminatedPods: false,
			allSourcesReady:     true,
		},
		{
			description: "recently started containers should not be removed",
			containers: []containerTemplate{
				makeGCContainer(podStateProvider, "foo", "bar", 2, time.Now().UnixNano(), runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 1, time.Now().UnixNano(), runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 0, time.Now().UnixNano(), runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0, 1, 2},
			evictTerminatedPods: false,
			allSourcesReady:     true,
		},
		{
			description: "oldest containers should be removed when per pod container limit exceeded",
			containers: []containerTemplate{
				makeGCContainer(podStateProvider, "foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0, 1},
			evictTerminatedPods: false,
			allSourcesReady:     true,
		},
		{
			description: "running containers should not be removed",
			containers: []containerTemplate{
				makeGCContainer(podStateProvider, "foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_RUNNING),
			},
			remain:              []int{0, 1, 2},
			evictTerminatedPods: false,
			allSourcesReady:     true,
		},
		{
			description: "no containers should be removed when limits are not exceeded",
			containers: []containerTemplate{
				makeGCContainer(podStateProvider, "foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0, 1},
			evictTerminatedPods: false,
			allSourcesReady:     true,
		},
		{
			description: "max container count should apply per (UID, container) pair",
			containers: []containerTemplate{
				makeGCContainer(podStateProvider, "foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo1", "baz", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo1", "baz", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo1", "baz", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo2", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo2", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo2", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0, 1, 3, 4, 6, 7},
			evictTerminatedPods: false,
			allSourcesReady:     true,
		},
		{
			description: "max limit should apply and try to keep from every pod",
			containers: []containerTemplate{
				makeGCContainer(podStateProvider, "foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo1", "bar1", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo1", "bar1", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo2", "bar2", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo2", "bar2", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo3", "bar3", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo3", "bar3", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo4", "bar4", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo4", "bar4", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0, 2, 4, 6, 8},
			evictTerminatedPods: false,
			allSourcesReady:     true,
		},
		{
			description: "oldest pods should be removed if limit exceeded",
			containers: []containerTemplate{
				makeGCContainer(podStateProvider, "foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo1", "bar1", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo1", "bar1", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo2", "bar2", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo3", "bar3", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo4", "bar4", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo5", "bar5", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo6", "bar6", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo7", "bar7", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0, 2, 4, 6, 8, 9},
			evictTerminatedPods: false,
			allSourcesReady:     true,
		},
		{
			description: "all non-running containers should be removed when evictTerminatedPods is set",
			containers: []containerTemplate{
				makeGCContainer(podStateProvider, "foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo1", "bar1", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo1", "bar1", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "running", "bar2", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo3", "bar3", 0, 0, runtimeapi.ContainerState_CONTAINER_RUNNING),
			},
			remain:              []int{4, 5},
			evictTerminatedPods: true,
			allSourcesReady:     true,
		},
		{
			description: "containers for deleted pods should be removed",
			containers: []containerTemplate{
				makeGCContainer(podStateProvider, "foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				// deleted pods still respect MinAge.
				makeGCContainer(podStateProvider, "deleted", "bar1", 2, time.Now().UnixNano(), runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "deleted", "bar1", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer(podStateProvider, "deleted", "bar1", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0, 1, 2},
			evictTerminatedPods: false,
			allSourcesReady:     true,
		},
		{
			description: "containers for deleted pods may not be removed if allSourcesReady is set false ",
			containers: []containerTemplate{
				makeGCContainer(podStateProvider, "deleted", "bar1", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0},
			evictTerminatedPods: true,
			allSourcesReady:     false,
		},
	} {
		t.Logf("TestCase #%d: %+v", c, test)
		fakeContainers := makeFakeContainers(t, m, test.containers)
		fakeRuntime.SetFakeContainers(fakeContainers)

		if test.policy == nil {
			test.policy = &defaultGCPolicy
		}
		err := m.containerGC.evictContainers(*test.policy, test.allSourcesReady, test.evictTerminatedPods)
		assert.NoError(t, err)
		realRemain, err := fakeRuntime.ListContainers(nil)
		assert.NoError(t, err)
		assert.Len(t, realRemain, len(test.remain))
		for _, remain := range test.remain {
			status, err := fakeRuntime.ContainerStatus(fakeContainers[remain].Id)
			assert.NoError(t, err)
			assert.Equal(t, &fakeContainers[remain].ContainerStatus, status)
		}
	}
}

// Notice that legacy container symlink is not tested since it may be deprecated soon.
func TestPodLogDirectoryGC(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)
	fakeOS := m.osInterface.(*containertest.FakeOS)
	podStateProvider := m.containerGC.podStateProvider.(*fakePodStateProvider)

	// pod log directories without corresponding pods should be removed.
	podStateProvider.existingPods["123"] = struct{}{}
	podStateProvider.existingPods["456"] = struct{}{}
	podStateProvider.existingPods["321"] = struct{}{}
	podStateProvider.runningPods["123"] = struct{}{}
	podStateProvider.runningPods["456"] = struct{}{}
	podStateProvider.existingPods["321"] = struct{}{}
	files := []string{"123", "456", "789", "012", "name_namespace_321", "name_namespace_654"}
	removed := []string{
		filepath.Join(podLogsRootDirectory, "789"),
		filepath.Join(podLogsRootDirectory, "012"),
		filepath.Join(podLogsRootDirectory, "name_namespace_654"),
	}

	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	fakeOS.ReadDirFn = func(string) ([]os.FileInfo, error) {
		var fileInfos []os.FileInfo
		for _, file := range files {
			mockFI := containertest.NewMockFileInfo(ctrl)
			mockFI.EXPECT().Name().Return(file)
			fileInfos = append(fileInfos, mockFI)
		}
		return fileInfos, nil
	}

	// allSourcesReady == true, pod log directories without corresponding pod should be removed.
	err = m.containerGC.evictPodLogsDirectories(true)
	assert.NoError(t, err)
	assert.Equal(t, removed, fakeOS.Removes)

	// allSourcesReady == false, pod log directories should not be removed.
	fakeOS.Removes = []string{}
	err = m.containerGC.evictPodLogsDirectories(false)
	assert.NoError(t, err)
	assert.Empty(t, fakeOS.Removes)
}

func TestUnknownStateContainerGC(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	podStateProvider := m.containerGC.podStateProvider.(*fakePodStateProvider)
	defaultGCPolicy := kubecontainer.ContainerGCPolicy{MinAge: time.Hour, MaxPerPodContainer: 0, MaxContainers: 0}

	fakeContainers := makeFakeContainers(t, m, []containerTemplate{
		makeGCContainer(podStateProvider, "foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_UNKNOWN),
	})
	fakeRuntime.SetFakeContainers(fakeContainers)

	err = m.containerGC.evictContainers(defaultGCPolicy, true, false)
	assert.NoError(t, err)

	assert.Contains(t, fakeRuntime.GetCalls(), "StopContainer", "RemoveContainer",
		"container in unknown state should be stopped before being removed")

	remain, err := fakeRuntime.ListContainers(nil)
	assert.NoError(t, err)
	assert.Empty(t, remain)
}
