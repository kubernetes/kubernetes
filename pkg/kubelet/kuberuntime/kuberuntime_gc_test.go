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
)

func makeAndSetFakeEvictablePod(m *kubeGenericRuntimeManager, fakeRuntime *apitest.FakeRuntimeService, pods []*api.Pod) error {
	sandboxes := make([]*apitest.FakePodSandbox, 0)
	containers := make([]*apitest.FakeContainer, 0)
	for _, pod := range pods {
		fakePodSandbox, err := makeFakePodSandbox(m, pod)
		if err != nil {
			return err
		}

		fakeContainers, err := makeFakeContainers(m, pod, pod.Spec.Containers)
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

func TestGarbageCollect(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "foo1",
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
					{
						Name:  "foo3",
						Image: "busybox",
					},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "123456789",
				Name:      "foo2",
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
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "1234567890",
				Name:      "foo1",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "foo1",
						Image: "busybox",
					},
				},
			},
		},
	}

	// Set fake sandbox and fake containers to fakeRuntime.
	err = makeAndSetFakeEvictablePod(m, fakeRuntime, pods)
	assert.NoError(t, err)

	err = m.GarbageCollect(kubecontainer.ContainerGCPolicy{
		MinAge:             time.Second,
		MaxPerPodContainer: 1,
		MaxContainers:      2,
	}, true)
	assert.NoError(t, err)
	assert.Equal(t, 2, len(fakeRuntime.Containers))
	assert.Equal(t, 2, len(fakeRuntime.Sandboxes))
}
