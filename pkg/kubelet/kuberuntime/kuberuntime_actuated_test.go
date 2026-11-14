/*
Copyright The Kubernetes Authors.

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
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/util/flowcontrol"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/utils/ptr"
)

// TestInitializeActuatedPodSeedsResources checks that the containers seeded from the allocated pod
// hold only their resources, and no image, so they aren't taken for a spec they were started with.
func TestInitializeActuatedPodSeedsResources(t *testing.T) {
	restartAlways := v1.ContainerRestartPolicyAlways
	resources := v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")}}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "uid", Name: "foo", Namespace: "ns"},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{{Name: "sidecar", Image: "busybox", RestartPolicy: &restartAlways, Resources: resources}},
			Containers:     []v1.Container{{Name: "main", Image: "busybox", Resources: resources}},
		},
	}
	setDynamicContainersGate(t, true)
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	m.InitializeActuatedPod(klog.Background(), pod)
	seeded, ok := m.actuatedState.GetContainer(pod.UID, "main", podutil.Containers)
	require.True(t, ok)
	assert.Equal(t, &v1.Container{Name: "main", Resources: resources}, seeded)
	seeded, ok = m.actuatedState.GetContainer(pod.UID, "sidecar", podutil.InitContainers)
	require.True(t, ok)
	assert.Equal(t, &v1.Container{Name: "sidecar", Resources: resources}, seeded)
}

// cgroupContainerManager provides pod cgroup configs, which the fake pod container manager lacks,
// so that pods can be resized.
type cgroupContainerManager struct {
	cm.ContainerManager
}

func (c *cgroupContainerManager) NewPodContainerManager() cm.PodContainerManager {
	return &cgroupPodContainerManager{FakePodContainerManager: cm.NewFakePodContainerManager()}
}

type cgroupPodContainerManager struct {
	*cm.FakePodContainerManager
}

func (p *cgroupPodContainerManager) GetPodCgroupConfig(_ *v1.Pod, resourceName v1.ResourceName) (*cm.ResourceConfig, error) {
	if resourceName == v1.ResourceMemory {
		return &cm.ResourceConfig{Memory: ptr.To[int64](1 << 30)}, nil
	}
	return &cm.ResourceConfig{
		CPUShares: new(cm.MilliCPUToShares(200)),
		CPUQuota:  ptr.To[int64](-1),
	}, nil
}

// The actuated record of a container removed from the pod spec lives as long as its runtime
// container, and the name can be reused afterwards.
func TestRemovedContainerActuatedRecordLifecycle(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicContainers, true)
	tCtx := ktesting.Init(t)
	fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)
	m.containerManager = &cgroupContainerManager{ContainerManager: m.containerManager}

	resources := v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")}}
	removed := v1.Container{
		Name:         "removed",
		Image:        "busybox",
		Resources:    resources,
		VolumeMounts: []v1.VolumeMount{{Name: "vol", MountPath: "/vol"}},
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "12345678", Name: "foo", Namespace: "new"},
		Spec: v1.PodSpec{Containers: []v1.Container{
			{Name: "main", Image: "busybox", Resources: resources},
		}},
	}
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	podStatus := func() *kubecontainer.PodStatus {
		t.Helper()
		runtimePod, err := m.GetPod(tCtx, pod.UID)
		require.NoError(t, err)
		status, err := m.GetPodStatus(tCtx, runtimePod)
		require.NoError(t, err)
		return status
	}
	syncPod := func(pod *v1.Pod, status *kubecontainer.PodStatus) {
		t.Helper()
		result := m.SyncPod(tCtx, pod, status, []v1.Secret{}, backOff, false)
		require.NoError(t, result.Error())
	}
	m.InitializeActuatedPod(tCtx.Logger(), pod)
	syncPod(pod, &kubecontainer.PodStatus{})

	// An added container isn't actuated until it is started.
	withRemoved := pod.DeepCopy()
	withRemoved.Spec.Containers = append(withRemoved.Spec.Containers, removed)
	_, ok := m.getActuatedContainer(pod.UID, "removed", podutil.Containers)
	require.False(t, ok, "not started yet")
	syncPod(withRemoved, podStatus())
	actuated, ok := m.getActuatedContainer(pod.UID, "removed", podutil.Containers)
	require.True(t, ok, "started")
	// The resources are recorded as actuated (e.g. with pod-level limits applied).
	actuated.Resources = removed.Resources
	assert.Equal(t, removed, *actuated)

	// Removing the container kills it, but keeps the record.
	syncPod(pod, podStatus())
	syncPod(pod, podStatus())
	_, ok = m.getActuatedContainer(pod.UID, "removed", podutil.Containers)
	assert.True(t, ok, "the record should outlive the kill")

	// Once the runtime container is gone (e.g. garbage collected), the record is dropped.
	for id, c := range fakeRuntime.Containers {
		if c.Metadata.Name == "removed" {
			require.NoError(t, m.removeContainer(tCtx, id, false))
		}
	}
	syncPod(pod, podStatus())
	_, ok = m.getActuatedContainer(pod.UID, "removed", podutil.Containers)
	assert.False(t, ok, "the record should be dropped with the runtime container")
	_, ok = m.actuatedState.GetContainerResources(pod.UID, "removed")
	assert.False(t, ok)

	// Adding the name back is treated as a new container, which resizes the pod.
	actions := m.computePodActions(tCtx, withRemoved, podStatus(), false)
	assert.True(t, actions.UpdatePodResources, "a re-added container should resize the pod")
	syncPod(withRemoved, podStatus())
	_, ok = m.getActuatedContainer(pod.UID, "removed", podutil.Containers)
	assert.True(t, ok)
}

// A container removed before it was ever started has no runtime container, so its record (here
// seeded when the pod was first synced) is dropped right away. The same reconciliation drops stale
// records after a kubelet restart.
func TestRemovedContainerNeverStarted(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicContainers, true)
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	resources := v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")}}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "12345678", Name: "foo", Namespace: "new"},
		Spec: v1.PodSpec{Containers: []v1.Container{
			{Name: "main", Image: "busybox", Resources: resources},
			{Name: "never-started", Image: "busybox", Resources: resources},
		}},
	}
	m.InitializeActuatedPod(tCtx.Logger(), pod)
	updated := pod.DeepCopy()
	updated.Spec.Containers = updated.Spec.Containers[:1]
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	// The first sync creates the sandbox and starts "main" only; the next one drops the record.
	result := m.SyncPod(tCtx, updated, &kubecontainer.PodStatus{}, []v1.Secret{}, backOff, false)
	require.NoError(t, result.Error())
	runtimePod, err := m.GetPod(tCtx, pod.UID)
	require.NoError(t, err)
	podStatus, err := m.GetPodStatus(tCtx, runtimePod)
	require.NoError(t, err)
	result = m.SyncPod(tCtx, updated, podStatus, []v1.Secret{}, backOff, false)
	require.NoError(t, result.Error())
	_, ok := m.actuatedState.GetContainerResources(pod.UID, "never-started")
	assert.False(t, ok, "the record of the removed container should be dropped")
	_, ok = m.actuatedState.GetContainerResources(pod.UID, "main")
	assert.True(t, ok)
}

// killContainer uses the spec a removed container was started with, even if DynamicContainers was
// disabled after the removal.
func TestKillRemovedContainerUsesActuatedSpec(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(map[bool]string{false: "gate disabled", true: "gate enabled"}[enabled], func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicContainers, enabled)
			tCtx := ktesting.Init(t)
			fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
			require.NoError(t, err)
			fakeRunner := &containertest.FakeContainerCommandRunner{}
			m.runner = lifecycle.NewHandlerRunner(&fakeHTTP{}, fakeRunner, nil, nil)

			preStop := func(cmd string) *v1.Lifecycle {
				return &v1.Lifecycle{PreStop: &v1.LifecycleHandler{Exec: &v1.ExecAction{Command: []string{cmd}}}}
			}
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{UID: "12345678", Name: "foo", Namespace: "new"},
				Spec: v1.PodSpec{
					TerminationGracePeriodSeconds: ptr.To[int64](30),
					Containers: []v1.Container{
						{Name: "main", Image: "busybox"},
						{Name: "removed", Image: "busybox", Lifecycle: preStop("from-labels")},
					},
				},
			}
			_, fakeContainers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
			removedID := kubecontainer.ContainerID{Type: "fake", ID: fakeContainers[1].Id}
			actuated := pod.Spec.Containers[1].DeepCopy()
			actuated.Lifecycle = preStop("from-actuated")
			require.NoError(t, m.actuatedState.SetContainer(klog.Background(), pod.UID, podutil.Containers, *actuated))

			pod.Spec.Containers = pod.Spec.Containers[:1]
			require.NoError(t, m.killContainer(tCtx, pod, removedID, "removed", "removed from pod spec", "", nil, nil))
			assert.Equal(t, []string{"from-actuated"}, fakeRunner.Cmd)
		})
	}
}
