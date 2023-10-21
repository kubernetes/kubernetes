/*
Copyright 2023 The Kubernetes Authors.

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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/flowcontrol"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	critesting "k8s.io/cri-api/pkg/apis/testing"
	kubelettypes "k8s.io/kubelet/pkg/types"
)

func TestSyncTerminatingPod(t *testing.T) {
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
	sandboxID := "sandbox-1"
	fakeRuntime.SetFakeSandboxes([]*critesting.FakePodSandbox{
		{
			PodSandboxStatus: runtimeapi.PodSandboxStatus{
				Id:       sandboxID,
				Metadata: &runtimeapi.PodSandboxMetadata{},
				Labels: map[string]string{
					kubelettypes.KubernetesPodUIDLabel: string(pod.UID),
				},
			},
		},
	})

	var fakeContainers []*critesting.FakeContainer
	for _, c := range containers {
		fakeContainers = append(fakeContainers, &critesting.FakeContainer{
			ContainerStatus: runtimeapi.ContainerStatus{
				Id: c.Name,
				Metadata: &runtimeapi.ContainerMetadata{
					Name: c.Name,
				},
				State: runtimeapi.ContainerState_CONTAINER_RUNNING,
				Image: &runtimeapi.ImageSpec{
					Image: c.Image,
				},
				Labels: map[string]string{
					kubelettypes.KubernetesPodUIDLabel:        string(pod.UID),
					kubelettypes.KubernetesContainerNameLabel: c.Name,
				},
			},
			SandboxID: sandboxID,
		})
	}
	fakeRuntime.SetFakeContainers(fakeContainers)

	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)

	ctx := context.Background()
	// assign enough grace period for containers to not be terminated because of it.
	gracePeriod := int64(300)

	podStatus, err := m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	t.Logf("podStatus: %+v", podStatus)
	result, terminated := m.SyncTerminatingPod(context.Background(), pod, podStatus, &gracePeriod, []v1.Secret{}, backOff)
	drainWorkers(m.containerTermination, pod)
	assert.NoError(t, result.Error())
	assert.False(t, terminated)
	assert.Equal(t, 2, len(fakeRuntime.Containers))
	assert.Equal(t, 1, len(fakeRuntime.Sandboxes))
	for _, sandbox := range fakeRuntime.Sandboxes {
		assert.Equal(t, runtimeapi.PodSandboxState_SANDBOX_READY, sandbox.State)
	}
	for _, c := range fakeRuntime.Containers {
		assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, c.State)
	}

	podStatus, err = m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	t.Logf("podStatus: %+v", podStatus)
	result, terminated = m.SyncTerminatingPod(context.Background(), pod, podStatus, &gracePeriod, []v1.Secret{}, backOff)
	drainWorkers(m.containerTermination, pod)
	assert.NoError(t, result.Error())
	assert.True(t, terminated)
	assert.Equal(t, 2, len(fakeRuntime.Containers))
	assert.Equal(t, 1, len(fakeRuntime.Sandboxes))
	for _, sandbox := range fakeRuntime.Sandboxes {
		assert.Equal(t, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, sandbox.State)
	}
	for _, c := range fakeRuntime.Containers {
		assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, c.State)
	}
}

func TestSyncTerminatingPodWithRestartableInitContainers(t *testing.T) {
	fakeRuntime, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	containerRestartPolicyAlways := v1.ContainerRestartPolicyAlways
	restartableInit1 := "restartable-init-1"
	restartableInit2 := "restartable-init-2"
	regular1 := "regular-1"
	regular2 := "regular-2"
	initContainers := []v1.Container{
		{
			Name:            restartableInit1,
			Image:           "busybox",
			ImagePullPolicy: v1.PullIfNotPresent,
			RestartPolicy:   &containerRestartPolicyAlways,
		},
		{
			Name:            restartableInit2,
			Image:           "busybox",
			ImagePullPolicy: v1.PullIfNotPresent,
			RestartPolicy:   &containerRestartPolicyAlways,
		},
	}
	containers := []v1.Container{
		{
			Name:            regular1,
			Image:           "busybox",
			ImagePullPolicy: v1.PullIfNotPresent,
		},
		{
			Name:            regular2,
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
			InitContainers: initContainers,
			Containers:     containers,
		},
	}
	sandboxID := "sandbox-1"
	fakeRuntime.SetFakeSandboxes([]*critesting.FakePodSandbox{
		{
			PodSandboxStatus: runtimeapi.PodSandboxStatus{
				Id:       sandboxID,
				Metadata: &runtimeapi.PodSandboxMetadata{},
				Labels: map[string]string{
					kubelettypes.KubernetesPodUIDLabel: string(pod.UID),
				},
			},
		},
	})

	var fakeContainers []*critesting.FakeContainer
	for _, c := range append(initContainers, containers...) {
		fakeContainers = append(fakeContainers, &critesting.FakeContainer{
			ContainerStatus: runtimeapi.ContainerStatus{
				Id: c.Name,
				Metadata: &runtimeapi.ContainerMetadata{
					Name: c.Name,
				},
				State: runtimeapi.ContainerState_CONTAINER_RUNNING,
				Image: &runtimeapi.ImageSpec{
					Image: c.Image,
				},
				Labels: map[string]string{
					kubelettypes.KubernetesPodUIDLabel:        string(pod.UID),
					kubelettypes.KubernetesContainerNameLabel: c.Name,
				},
			},
			SandboxID: sandboxID,
		})
	}
	fakeRuntime.SetFakeContainers(fakeContainers)

	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)

	ctx := context.Background()
	// assign enough grace period for containers to not be terminated because of it.
	gracePeriod := int64(300)

	podStatus, err := m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	t.Logf("podStatus: %+v", podStatus)
	result, terminated := m.SyncTerminatingPod(context.Background(), pod, podStatus, &gracePeriod, []v1.Secret{}, backOff)
	drainWorkers(m.containerTermination, pod)
	assert.NoError(t, result.Error())
	assert.False(t, terminated)
	assert.Equal(t, 4, len(fakeRuntime.Containers))
	assert.Equal(t, 1, len(fakeRuntime.Sandboxes))
	for _, sandbox := range fakeRuntime.Sandboxes {
		assert.Equal(t, runtimeapi.PodSandboxState_SANDBOX_READY, sandbox.State)
	}
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_RUNNING, fakeRuntime.Containers[restartableInit1].State)
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_RUNNING, fakeRuntime.Containers[restartableInit2].State)
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, fakeRuntime.Containers[regular1].State)
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, fakeRuntime.Containers[regular2].State)

	t.Logf("Let's assume the first init container exited for some reason during the pod termination")
	fakeRuntime.Lock()
	fakeRuntime.Containers[restartableInit1].State = runtimeapi.ContainerState_CONTAINER_EXITED
	fakeRuntime.Unlock()

	podStatus, err = m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	t.Logf("podStatus: %+v", podStatus)
	result, terminated = m.SyncTerminatingPod(context.Background(), pod, podStatus, &gracePeriod, []v1.Secret{}, backOff)
	drainWorkers(m.containerTermination, pod)
	assert.NoError(t, result.Error())
	assert.False(t, terminated)
	assert.Equal(t, 5, len(fakeRuntime.Containers))
	assert.Equal(t, 1, len(fakeRuntime.Sandboxes))
	for _, sandbox := range fakeRuntime.Sandboxes {
		assert.Equal(t, runtimeapi.PodSandboxState_SANDBOX_READY, sandbox.State)
	}

	var newRestartableInit1 string
	for id, c := range fakeRuntime.Containers {
		if c.Labels[kubelettypes.KubernetesContainerNameLabel] != restartableInit1 {
			continue
		}
		if id == restartableInit1 {
			continue
		}
		newRestartableInit1 = id
	}
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_RUNNING, fakeRuntime.Containers[newRestartableInit1].State)
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, fakeRuntime.Containers[restartableInit2].State)
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, fakeRuntime.Containers[regular1].State)
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, fakeRuntime.Containers[regular2].State)

	podStatus, err = m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	t.Logf("podStatus: %+v", podStatus)
	result, terminated = m.SyncTerminatingPod(context.Background(), pod, podStatus, &gracePeriod, []v1.Secret{}, backOff)
	drainWorkers(m.containerTermination, pod)
	assert.NoError(t, result.Error())
	assert.False(t, terminated)
	assert.Equal(t, 5, len(fakeRuntime.Containers))
	assert.Equal(t, 1, len(fakeRuntime.Sandboxes))
	for _, sandbox := range fakeRuntime.Sandboxes {
		assert.Equal(t, runtimeapi.PodSandboxState_SANDBOX_READY, sandbox.State)
	}
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, fakeRuntime.Containers[newRestartableInit1].State)
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, fakeRuntime.Containers[restartableInit2].State)
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, fakeRuntime.Containers[regular1].State)
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, fakeRuntime.Containers[regular2].State)

	podStatus, err = m.GetPodStatus(ctx, pod.UID, pod.Name, pod.Namespace)
	assert.NoError(t, err)
	t.Logf("podStatus: %+v", podStatus)
	result, terminated = m.SyncTerminatingPod(context.Background(), pod, podStatus, &gracePeriod, []v1.Secret{}, backOff)
	drainWorkers(m.containerTermination, pod)
	assert.NoError(t, result.Error())
	assert.True(t, terminated)
	assert.Equal(t, 5, len(fakeRuntime.Containers))
	assert.Equal(t, 1, len(fakeRuntime.Sandboxes))
	for _, sandbox := range fakeRuntime.Sandboxes {
		assert.Equal(t, runtimeapi.PodSandboxState_SANDBOX_NOTREADY, sandbox.State)
	}
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, fakeRuntime.Containers[newRestartableInit1].State)
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, fakeRuntime.Containers[restartableInit2].State)
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, fakeRuntime.Containers[regular1].State)
	assert.Equal(t, runtimeapi.ContainerState_CONTAINER_EXITED, fakeRuntime.Containers[regular2].State)
}

func drainWorkers(ct *containerTermination, pod *v1.Pod) {
	allContainers := make([]v1.Container, 0, len(pod.Spec.InitContainers)+len(pod.Spec.Containers))
	allContainers = append(allContainers, pod.Spec.InitContainers...)
	allContainers = append(allContainers, pod.Spec.Containers...)
	for {
		ct.lock.Lock()
		_, terminating := ct.isTerminating[pod.UID]
		if !terminating {
			ct.lock.Unlock()
			return
		}

		stillWorking := false
		for _, c := range allContainers {
			_, cannotStop := ct.canStop[pod.UID][c.Name]
			if cannotStop {
				continue
			}

			_, terminating := ct.isTerminating[pod.UID][c.Name]
			if terminating {
				stillWorking = true
				break
			}
		}
		ct.lock.Unlock()
		if !stillWorking {
			return
		}

		time.Sleep(100 * time.Millisecond)
	}
}
