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
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
)

func TestSandboxGC(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	podDeletionProvider := m.containerGC.podDeletionProvider.(*fakePodDeletionProvider)
	makeGCSandbox := func(pod *v1.Pod, attempt uint32, state runtimeapi.PodSandboxState, withPodGetter bool, createdAt int64) sandboxTemplate {
		if withPodGetter {
			// initialize the pod getter
			podDeletionProvider.pods[pod.UID] = struct{}{}
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
		minAge              time.Duration       // sandboxMinGCAge
		remain              []int               // template indexes of remaining sandboxes
		evictNonDeletedPods bool
	}{
		{
			description: "notready sandboxes without containers for deleted pods should be garbage collected.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[2], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, false, 0),
			},
			containers:          []containerTemplate{},
			remain:              []int{},
			evictNonDeletedPods: false,
		},
		{
			description: "ready sandboxes without containers for deleted pods should not be garbage collected.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[2], 0, runtimeapi.PodSandboxState_SANDBOX_READY, false, 0),
			},
			containers:          []containerTemplate{},
			remain:              []int{0},
			evictNonDeletedPods: false,
		},
		{
			description: "sandboxes for existing pods should not be garbage collected.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[0], 0, runtimeapi.PodSandboxState_SANDBOX_READY, true, 0),
				makeGCSandbox(pods[1], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, 0),
			},
			containers:          []containerTemplate{},
			remain:              []int{0, 1},
			evictNonDeletedPods: false,
		},
		{
			description: "non-running sandboxes for existing pods should be garbage collected if evictNonDeletedPods is set.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[0], 0, runtimeapi.PodSandboxState_SANDBOX_READY, true, 0),
				makeGCSandbox(pods[1], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, 0),
			},
			containers:          []containerTemplate{},
			remain:              []int{0},
			evictNonDeletedPods: true,
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
			evictNonDeletedPods: false,
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
			evictNonDeletedPods: false,
		},
	} {
		t.Logf("TestCase #%d: %+v", c, test)
		fakeSandboxes := makeFakePodSandboxes(t, m, test.sandboxes)
		fakeContainers := makeFakeContainers(t, m, test.containers)
		fakeRuntime.SetFakeSandboxes(fakeSandboxes)
		fakeRuntime.SetFakeContainers(fakeContainers)

		err := m.containerGC.evictSandboxes(test.evictNonDeletedPods)
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

func TestContainerGC(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	podDeletionProvider := m.containerGC.podDeletionProvider.(*fakePodDeletionProvider)
	makeGCContainer := func(podName, containerName string, attempt int, createdAt int64, state runtimeapi.ContainerState) containerTemplate {
		container := makeTestContainer(containerName, "test-image")
		pod := makeTestPod(podName, "test-ns", podName, []v1.Container{container})
		if podName != "deleted" {
			// initialize the pod getter, explicitly exclude deleted pod
			podDeletionProvider.pods[pod.UID] = struct{}{}
		}
		return containerTemplate{
			pod:       pod,
			container: &container,
			attempt:   attempt,
			createdAt: createdAt,
			state:     state,
		}
	}
	defaultGCPolicy := kubecontainer.ContainerGCPolicy{MinAge: time.Hour, MaxPerPodContainer: 2, MaxContainers: 6}

	for c, test := range []struct {
		description         string                           // description of the test case
		containers          []containerTemplate              // templates of containers
		policy              *kubecontainer.ContainerGCPolicy // container gc policy
		remain              []int                            // template indexes of remaining containers
		evictNonDeletedPods bool
	}{
		{
			description: "all containers should be removed when max container limit is 0",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			policy:              &kubecontainer.ContainerGCPolicy{MinAge: time.Minute, MaxPerPodContainer: 1, MaxContainers: 0},
			remain:              []int{},
			evictNonDeletedPods: false,
		},
		{
			description: "max containers should be complied when no max per pod container limit is set",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 4, 4, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 3, 3, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			policy:              &kubecontainer.ContainerGCPolicy{MinAge: time.Minute, MaxPerPodContainer: -1, MaxContainers: 4},
			remain:              []int{0, 1, 2, 3},
			evictNonDeletedPods: false,
		},
		{
			description: "no containers should be removed if both max container and per pod container limits are not set",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			policy:              &kubecontainer.ContainerGCPolicy{MinAge: time.Minute, MaxPerPodContainer: -1, MaxContainers: -1},
			remain:              []int{0, 1, 2},
			evictNonDeletedPods: false,
		},
		{
			description: "recently started containers should not be removed",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 2, time.Now().UnixNano(), runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 1, time.Now().UnixNano(), runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 0, time.Now().UnixNano(), runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0, 1, 2},
			evictNonDeletedPods: false,
		},
		{
			description: "oldest containers should be removed when per pod container limit exceeded",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0, 1},
			evictNonDeletedPods: false,
		},
		{
			description: "running containers should not be removed",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_RUNNING),
			},
			remain:              []int{0, 1, 2},
			evictNonDeletedPods: false,
		},
		{
			description: "no containers should be removed when limits are not exceeded",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0, 1},
			evictNonDeletedPods: false,
		},
		{
			description: "max container count should apply per (UID, container) pair",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo1", "baz", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo1", "baz", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo1", "baz", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo2", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo2", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo2", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0, 1, 3, 4, 6, 7},
			evictNonDeletedPods: false,
		},
		{
			description: "max limit should apply and try to keep from every pod",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo1", "bar1", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo1", "bar1", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo2", "bar2", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo2", "bar2", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo3", "bar3", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo3", "bar3", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo4", "bar4", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo4", "bar4", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0, 2, 4, 6, 8},
			evictNonDeletedPods: false,
		},
		{
			description: "oldest pods should be removed if limit exceeded",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo1", "bar1", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo1", "bar1", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo2", "bar2", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo3", "bar3", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo4", "bar4", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo5", "bar5", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo6", "bar6", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo7", "bar7", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0, 2, 4, 6, 8, 9},
			evictNonDeletedPods: false,
		},
		{
			description: "all non-running containers should be removed when evictNonDeletedPods is set",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo1", "bar1", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo1", "bar1", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo2", "bar2", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo3", "bar3", 0, 0, runtimeapi.ContainerState_CONTAINER_RUNNING),
			},
			remain:              []int{5},
			evictNonDeletedPods: true,
		},
		{
			description: "containers for deleted pods should be removed",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
				// deleted pods still respect MinAge.
				makeGCContainer("deleted", "bar1", 2, time.Now().UnixNano(), runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("deleted", "bar1", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("deleted", "bar1", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:              []int{0, 1, 2},
			evictNonDeletedPods: false,
		},
	} {
		t.Logf("TestCase #%d: %+v", c, test)
		fakeContainers := makeFakeContainers(t, m, test.containers)
		fakeRuntime.SetFakeContainers(fakeContainers)

		if test.policy == nil {
			test.policy = &defaultGCPolicy
		}
		err := m.containerGC.evictContainers(*test.policy, true, test.evictNonDeletedPods)
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
	podDeletionProvider := m.containerGC.podDeletionProvider.(*fakePodDeletionProvider)

	// pod log directories without corresponding pods should be removed.
	podDeletionProvider.pods["123"] = struct{}{}
	podDeletionProvider.pods["456"] = struct{}{}
	files := []string{"123", "456", "789", "012"}
	removed := []string{filepath.Join(podLogsRootDirectory, "789"), filepath.Join(podLogsRootDirectory, "012")}

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
