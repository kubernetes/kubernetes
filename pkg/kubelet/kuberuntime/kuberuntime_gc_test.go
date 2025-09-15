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

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestSandboxGC(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	podStateProvider := m.containerGC.podStateProvider.(*fakePodStateProvider)
	makeGCSandbox := func(pod *v1.Pod, attempt uint32, state runtimeapi.PodSandboxState, hasRunningContainers, isTerminating bool, createdAt int64) sandboxTemplate {
		return sandboxTemplate{
			pod:         pod,
			state:       state,
			attempt:     attempt,
			createdAt:   createdAt,
			running:     hasRunningContainers,
			terminating: isTerminating,
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

	for _, test := range []struct {
		description          string              // description of the test case
		sandboxes            []sandboxTemplate   // templates of sandboxes
		containers           []containerTemplate // templates of containers
		remain               []int               // template indexes of remaining sandboxes
		evictTerminatingPods bool
	}{
		{
			description: "notready sandboxes without containers for deleted pods should be garbage collected.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[2], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, false, false, 0),
			},
			containers:           []containerTemplate{},
			remain:               []int{},
			evictTerminatingPods: false,
		},
		{
			description: "ready sandboxes without containers for deleted pods should not be garbage collected.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[2], 0, runtimeapi.PodSandboxState_SANDBOX_READY, false, false, 0),
			},
			containers:           []containerTemplate{},
			remain:               []int{0},
			evictTerminatingPods: false,
		},
		{
			description: "sandboxes for existing pods should not be garbage collected.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[0], 0, runtimeapi.PodSandboxState_SANDBOX_READY, true, false, 0),
				makeGCSandbox(pods[1], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, false, 0),
			},
			containers:           []containerTemplate{},
			remain:               []int{0, 1},
			evictTerminatingPods: false,
		},
		{
			description: "older exited sandboxes without containers for existing pods should be garbage collected if there are more than one exited sandboxes.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[0], 1, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, false, 1),
				makeGCSandbox(pods[0], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, false, 0),
			},
			containers:           []containerTemplate{},
			remain:               []int{0},
			evictTerminatingPods: false,
		},
		{
			description: "older exited sandboxes with containers for existing pods should not be garbage collected even if there are more than one exited sandboxes.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[0], 1, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, false, 1),
				makeGCSandbox(pods[0], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, false, 0),
			},
			containers: []containerTemplate{
				{pod: pods[0], container: &pods[0].Spec.Containers[0], sandboxAttempt: 0, state: runtimeapi.ContainerState_CONTAINER_EXITED},
			},
			remain:               []int{0, 1},
			evictTerminatingPods: false,
		},
		{
			description: "non-running sandboxes for existing pods should be garbage collected if evictTerminatingPods is set.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[0], 0, runtimeapi.PodSandboxState_SANDBOX_READY, true, true, 0),
				makeGCSandbox(pods[1], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, true, 0),
			},
			containers:           []containerTemplate{},
			remain:               []int{0},
			evictTerminatingPods: true,
		},
		{
			description: "sandbox with containers should not be garbage collected.",
			sandboxes: []sandboxTemplate{
				makeGCSandbox(pods[0], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, false, false, 0),
			},
			containers: []containerTemplate{
				{pod: pods[0], container: &pods[0].Spec.Containers[0], state: runtimeapi.ContainerState_CONTAINER_EXITED},
			},
			remain:               []int{0},
			evictTerminatingPods: false,
		},
		{
			description: "multiple sandboxes should be handled properly.",
			sandboxes: []sandboxTemplate{
				// running sandbox.
				makeGCSandbox(pods[0], 1, runtimeapi.PodSandboxState_SANDBOX_READY, true, false, 1),
				// exited sandbox without containers.
				makeGCSandbox(pods[0], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, false, 0),
				// exited sandbox with containers.
				makeGCSandbox(pods[1], 1, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, false, 1),
				// exited sandbox without containers.
				makeGCSandbox(pods[1], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, true, false, 0),
				// exited sandbox without containers for deleted pods.
				makeGCSandbox(pods[2], 0, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, false, true, 0),
			},
			containers: []containerTemplate{
				{pod: pods[1], container: &pods[1].Spec.Containers[0], sandboxAttempt: 1, state: runtimeapi.ContainerState_CONTAINER_EXITED},
			},
			remain:               []int{0, 2},
			evictTerminatingPods: false,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			podStateProvider.removed = make(map[types.UID]struct{})
			podStateProvider.terminated = make(map[types.UID]struct{})
			fakeSandboxes := makeFakePodSandboxes(t, m, test.sandboxes)
			fakeContainers := makeFakeContainers(t, m, test.containers)
			for _, s := range test.sandboxes {
				if !s.running && s.pod.Name == "deleted" {
					podStateProvider.removed[s.pod.UID] = struct{}{}
				}
				if s.terminating {
					podStateProvider.terminated[s.pod.UID] = struct{}{}
				}
			}
			fakeRuntime.SetFakeSandboxes(fakeSandboxes)
			fakeRuntime.SetFakeContainers(fakeContainers)

			err := m.containerGC.evictSandboxes(tCtx, test.evictTerminatingPods)
			assert.NoError(t, err)
			realRemain, err := fakeRuntime.ListPodSandbox(tCtx, nil)
			assert.NoError(t, err)
			assert.Len(t, realRemain, len(test.remain))
			for _, remain := range test.remain {
				resp, err := fakeRuntime.PodSandboxStatus(tCtx, fakeSandboxes[remain].Id, false)
				assert.NoError(t, err)
				assert.Equal(t, &fakeSandboxes[remain].PodSandboxStatus, resp.Status)
			}
		})
	}
}

func makeGCContainer(podName, containerName string, attempt int, createdAt int64, state runtimeapi.ContainerState) containerTemplate {
	container := makeTestContainer(containerName, "test-image")
	pod := makeTestPod(podName, "test-ns", podName, []v1.Container{container})
	return containerTemplate{
		pod:       pod,
		container: &container,
		attempt:   attempt,
		createdAt: createdAt,
		state:     state,
	}
}

func TestContainerGC(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	podStateProvider := m.containerGC.podStateProvider.(*fakePodStateProvider)
	defaultGCPolicy := kubecontainer.GCPolicy{MinAge: time.Hour, MaxPerPodContainer: 2, MaxContainers: 6}

	for _, test := range []struct {
		description          string                  // description of the test case
		containers           []containerTemplate     // templates of containers
		policy               *kubecontainer.GCPolicy // container gc policy
		remain               []int                   // template indexes of remaining containers
		evictTerminatingPods bool
		allSourcesReady      bool
	}{
		{
			description: "all containers should be removed when max container limit is 0",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			policy:               &kubecontainer.GCPolicy{MinAge: time.Minute, MaxPerPodContainer: 1, MaxContainers: 0},
			remain:               []int{},
			evictTerminatingPods: false,
			allSourcesReady:      true,
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
			policy:               &kubecontainer.GCPolicy{MinAge: time.Minute, MaxPerPodContainer: -1, MaxContainers: 4},
			remain:               []int{0, 1, 2, 3},
			evictTerminatingPods: false,
			allSourcesReady:      true,
		},
		{
			description: "no containers should be removed if both max container and per pod container limits are not set",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			policy:               &kubecontainer.GCPolicy{MinAge: time.Minute, MaxPerPodContainer: -1, MaxContainers: -1},
			remain:               []int{0, 1, 2},
			evictTerminatingPods: false,
			allSourcesReady:      true,
		},
		{
			description: "recently started containers should not be removed",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 2, time.Now().UnixNano(), runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 1, time.Now().UnixNano(), runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 0, time.Now().UnixNano(), runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:               []int{0, 1, 2},
			evictTerminatingPods: false,
			allSourcesReady:      true,
		},
		{
			description: "oldest containers should be removed when per pod container limit exceeded",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:               []int{0, 1},
			evictTerminatingPods: false,
			allSourcesReady:      true,
		},
		{
			description: "running containers should not be removed",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_RUNNING),
			},
			remain:               []int{0, 1, 2},
			evictTerminatingPods: false,
			allSourcesReady:      true,
		},
		{
			description: "no containers should be removed when limits are not exceeded",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:               []int{0, 1},
			evictTerminatingPods: false,
			allSourcesReady:      true,
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
			remain:               []int{0, 1, 3, 4, 6, 7},
			evictTerminatingPods: false,
			allSourcesReady:      true,
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
			remain:               []int{0, 2, 4, 6, 8},
			evictTerminatingPods: false,
			allSourcesReady:      true,
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
			remain:               []int{0, 2, 4, 6, 8, 9},
			evictTerminatingPods: false,
			allSourcesReady:      true,
		},
		{
			description: "all non-running containers should be removed when evictTerminatingPods is set",
			containers: []containerTemplate{
				makeGCContainer("foo", "bar", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo", "bar", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo1", "bar1", 2, 2, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo1", "bar1", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("running", "bar2", 1, 1, runtimeapi.ContainerState_CONTAINER_EXITED),
				makeGCContainer("foo3", "bar3", 0, 0, runtimeapi.ContainerState_CONTAINER_RUNNING),
			},
			remain:               []int{4, 5},
			evictTerminatingPods: true,
			allSourcesReady:      true,
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
			remain:               []int{0, 1, 2},
			evictTerminatingPods: false,
			allSourcesReady:      true,
		},
		{
			description: "containers for deleted pods may not be removed if allSourcesReady is set false ",
			containers: []containerTemplate{
				makeGCContainer("deleted", "bar1", 0, 0, runtimeapi.ContainerState_CONTAINER_EXITED),
			},
			remain:               []int{0},
			evictTerminatingPods: true,
			allSourcesReady:      false,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			podStateProvider.removed = make(map[types.UID]struct{})
			podStateProvider.terminated = make(map[types.UID]struct{})
			fakeContainers := makeFakeContainers(t, m, test.containers)
			for _, s := range test.containers {
				if s.pod.Name == "deleted" {
					podStateProvider.removed[s.pod.UID] = struct{}{}
				}
				if s.pod.Name != "running" {
					podStateProvider.terminated[s.pod.UID] = struct{}{}
				}
			}
			fakeRuntime.SetFakeContainers(fakeContainers)

			if test.policy == nil {
				test.policy = &defaultGCPolicy
			}
			err := m.containerGC.evictContainers(tCtx, *test.policy, test.allSourcesReady, test.evictTerminatingPods)
			assert.NoError(t, err)
			realRemain, err := fakeRuntime.ListContainers(tCtx, nil)
			assert.NoError(t, err)
			assert.Len(t, realRemain, len(test.remain))
			for _, remain := range test.remain {
				resp, err := fakeRuntime.ContainerStatus(tCtx, fakeContainers[remain].Id, false)
				assert.NoError(t, err)
				assert.Equal(t, &fakeContainers[remain].ContainerStatus, resp.Status)
			}
		})
	}
}

// Notice that legacy container symlink is not tested since it may be deprecated soon.
func TestPodLogDirectoryGC(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)
	fakeOS := m.osInterface.(*containertest.FakeOS)
	podStateProvider := m.containerGC.podStateProvider.(*fakePodStateProvider)

	// pod log directories without corresponding pods should be removed.
	files := []string{"123", "456", "789", "012", "name_namespace_321", "name_namespace_654"}
	podLogsDirectory := "/var/log/pods"
	removed := []string{
		filepath.Join(podLogsDirectory, "789"),
		filepath.Join(podLogsDirectory, "012"),
		filepath.Join(podLogsDirectory, "name_namespace_654"),
	}
	podStateProvider.removed["012"] = struct{}{}
	podStateProvider.removed["789"] = struct{}{}
	podStateProvider.removed["654"] = struct{}{}

	fakeOS.ReadDirFn = func(string) ([]os.DirEntry, error) {
		var dirEntries []os.DirEntry
		for _, file := range files {
			mockDE := containertest.NewMockDirEntry(t)
			mockDE.EXPECT().Name().Return(file)
			dirEntries = append(dirEntries, mockDE)
		}
		return dirEntries, nil
	}

	// allSourcesReady == true, pod log directories without corresponding pod should be removed.
	err = m.containerGC.evictPodLogsDirectories(tCtx, true)
	assert.NoError(t, err)
	assert.Equal(t, removed, fakeOS.Removes)

	// allSourcesReady == false, pod log directories should not be removed.
	fakeOS.Removes = []string{}
	err = m.containerGC.evictPodLogsDirectories(tCtx, false)
	assert.NoError(t, err)
	assert.Empty(t, fakeOS.Removes)
}

func TestUnknownStateContainerGC(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	// podStateProvider := m.containerGC.podStateProvider.(*fakePodStateProvider)
	defaultGCPolicy := kubecontainer.GCPolicy{MinAge: time.Hour, MaxPerPodContainer: 0, MaxContainers: 0}

	fakeContainers := makeFakeContainers(t, m, []containerTemplate{
		makeGCContainer("foo", "bar", 0, 0, runtimeapi.ContainerState_CONTAINER_UNKNOWN),
	})
	fakeRuntime.SetFakeContainers(fakeContainers)

	err = m.containerGC.evictContainers(tCtx, defaultGCPolicy, true, false)
	assert.NoError(t, err)

	assert.Contains(t, fakeRuntime.GetCalls(), "StopContainer", "RemoveContainer",
		"container in unknown state should be stopped before being removed")

	remain, err := fakeRuntime.ListContainers(tCtx, nil)
	assert.NoError(t, err)
	assert.Empty(t, remain)
}
