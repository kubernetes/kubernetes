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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	apitest "k8s.io/kubernetes/pkg/kubelet/api/testing"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
)

type apiPodWithCreatedAt struct {
	apiPod    *api.Pod
	createdAt int64
}

func makeAndSetFakeEvictablePod(m *kubeGenericRuntimeManager, fakeRuntime *apitest.FakeRuntimeService, pods []*apiPodWithCreatedAt) error {
	sandboxes := make([]*apitest.FakePodSandbox, 0)
	containers := make([]*apitest.FakeContainer, 0)
	for _, pod := range pods {
		fakePodSandbox, err := makeFakePodSandbox(m, pod.apiPod, pod.createdAt)
		if err != nil {
			return err
		}

		fakeContainers, err := makeFakeContainers(m, pod.apiPod, pod.apiPod.Spec.Containers, pod.createdAt, runtimeApi.ContainerState_EXITED)
		if err != nil {
			return err
		}

		// Set sandbox to not ready state
		sandboxNotReady := runtimeApi.PodSandBoxState_NOTREADY
		fakePodSandbox.State = &sandboxNotReady
		sandboxes = append(sandboxes, fakePodSandbox)

		// Set containers to exited state
		containerExited := runtimeApi.ContainerState_EXITED
		for _, c := range fakeContainers {
			c.State = &containerExited
			containers = append(containers, c)
		}

	}

	fakeRuntime.SetFakeSandboxes(sandboxes)
	fakeRuntime.SetFakeContainers(containers)
	return nil
}

func makeTestContainer(name, image string) api.Container {
	return api.Container{
		Name:  name,
		Image: image,
	}
}

func makeTestPod(podName, podNamespace, podUID string, containers []api.Container, createdAt int64) *apiPodWithCreatedAt {
	return &apiPodWithCreatedAt{
		createdAt: createdAt,
		apiPod: &api.Pod{
			ObjectMeta: api.ObjectMeta{
				UID:       types.UID(podUID),
				Name:      podName,
				Namespace: podNamespace,
			},
			Spec: api.PodSpec{
				Containers: containers,
			},
		},
	}
}

func TestGarbageCollect(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	pods := []*apiPodWithCreatedAt{
		makeTestPod("123456", "foo1", "new", []api.Container{
			makeTestContainer("foo1", "busybox"),
			makeTestContainer("foo2", "busybox"),
			makeTestContainer("foo3", "busybox"),
		}, 1),
		makeTestPod("1234567", "foo2", "new", []api.Container{
			makeTestContainer("foo4", "busybox"),
			makeTestContainer("foo5", "busybox"),
		}, 2),
		makeTestPod("12345678", "foo3", "new", []api.Container{
			makeTestContainer("foo6", "busybox"),
		}, 3),
	}
	err = makeAndSetFakeEvictablePod(m, fakeRuntime, pods)
	assert.NoError(t, err)
	assert.NoError(t, m.GarbageCollect(kubecontainer.ContainerGCPolicy{
		MinAge:             time.Second,
		MaxPerPodContainer: 1,
		MaxContainers:      3,
	}, false))
	assert.Equal(t, 3, len(fakeRuntime.Containers))
	assert.Equal(t, 2, len(fakeRuntime.Sandboxes))

	// no containers should be removed.
	err = makeAndSetFakeEvictablePod(m, fakeRuntime, pods)
	assert.NoError(t, err)
	assert.NoError(t, m.GarbageCollect(kubecontainer.ContainerGCPolicy{
		MinAge:             time.Second,
		MaxPerPodContainer: 10,
		MaxContainers:      100,
	}, false))
	assert.Equal(t, 6, len(fakeRuntime.Containers))
	assert.Equal(t, 3, len(fakeRuntime.Sandboxes))

	// all containers should be removed.
	err = makeAndSetFakeEvictablePod(m, fakeRuntime, pods)
	assert.NoError(t, err)
	assert.NoError(t, m.GarbageCollect(kubecontainer.ContainerGCPolicy{
		MinAge:             time.Second,
		MaxPerPodContainer: 0,
		MaxContainers:      0,
	}, false))
	assert.Equal(t, 0, len(fakeRuntime.Containers))
	assert.Equal(t, 0, len(fakeRuntime.Sandboxes))
}
