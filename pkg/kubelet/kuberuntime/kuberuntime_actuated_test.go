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
	goruntime "runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
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

	m.InitializeActuatedPod(klog.Background(), pod, nil)
	seeded, ok := m.actuatedState.GetContainer(pod.UID, "main", podutil.Containers)
	require.True(t, ok)
	assert.True(t, apiequality.Semantic.DeepEqual(&v1.Container{Name: "main", Resources: resources}, seeded), "%+v", seeded)
	seeded, ok = m.actuatedState.GetContainer(pod.UID, "sidecar", podutil.InitContainers)
	require.True(t, ok)
	assert.True(t, apiequality.Semantic.DeepEqual(&v1.Container{Name: "sidecar", Resources: resources}, seeded), "%+v", seeded)
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
	m.InitializeActuatedPod(tCtx.Logger(), pod, nil)
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
	if goruntime.GOOS == "linux" {
		assert.Equal(t, sets.New(v1.ResourceCPU), actions.PodCgroupResourcesToUpdate, "a re-added container should resize the pod")
	} else {
		assert.Empty(t, actions.PodCgroupResourcesToUpdate, "pods are not resized on this platform")
	}
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
	m.containerManager = &cgroupContainerManager{ContainerManager: m.containerManager}

	resources := v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")}}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "12345678", Name: "foo", Namespace: "new"},
		Spec: v1.PodSpec{Containers: []v1.Container{
			{Name: "main", Image: "busybox", Resources: resources},
			{Name: "never-started", Image: "busybox", Resources: resources},
		}},
	}
	m.InitializeActuatedPod(tCtx.Logger(), pod, nil)
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

func TestExitedRemovedContainerClearsActuatedVolumeMounts(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicContainers, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicNodeLocalEphemeralVolumes, true)
	tCtx := ktesting.Init(t)
	_, _, m, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)
	m.containerManager = &cgroupContainerManager{ContainerManager: m.containerManager}

	resources := v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")}}
	removed := v1.Container{
		Name:         "removed",
		Image:        "busybox",
		Resources:    resources,
		VolumeMounts: []v1.VolumeMount{{Name: "vol", MountPath: "/vol", SubPath: "sub1"}},
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "12345678", Name: "foo", Namespace: "new"},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{{Name: "vol", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}}},
			Containers: []v1.Container{
				{Name: "main", Image: "busybox", Resources: resources},
				removed,
			},
		},
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
	syncPod := func(p *v1.Pod, status *kubecontainer.PodStatus) {
		t.Helper()
		result := m.SyncPod(tCtx, p, status, []v1.Secret{}, backOff, false)
		require.NoError(t, result.Error())
	}

	require.NoError(t, m.actuatedState.SetPodVolume(tCtx.Logger(), pod.UID, pod.Spec.Volumes[0]))
	m.InitializeActuatedPod(tCtx.Logger(), pod, nil)
	syncPod(pod, &kubecontainer.PodStatus{})

	mounts, found := m.actuatedState.GetContainerVolumeMounts(pod.UID, "removed")
	require.True(t, found)
	require.Equal(t, removed.VolumeMounts, mounts)

	curStatus := podStatus()

	// Remove "removed" from pod.Spec: first sync starts killing it (if still Running in status,
	// VolumeMounts are retained while draining); once ContainerStateExited is observed, actuated
	// VolumeMounts are cleared while preserving the container record for the last-10 history.
	withoutRemoved := pod.DeepCopy()
	withoutRemoved.Spec.Containers = withoutRemoved.Spec.Containers[:1]
	withoutRemoved.Spec.Volumes = nil

	// While "removed" is still Running (gracefully terminating), actuated mounts remain populated.
	m.computePodActions(tCtx, withoutRemoved, curStatus, false)
	mounts, found = m.actuatedState.GetContainerVolumeMounts(pod.UID, "removed")
	require.True(t, found)
	assert.Equal(t, removed.VolumeMounts, mounts, "terminating removed container must retain VolumeMounts until it stops")

	syncPod(withoutRemoved, podStatus())
	syncPod(withoutRemoved, podStatus())

	actuated, ok := m.getActuatedContainer(pod.UID, "removed", podutil.Containers)
	require.True(t, ok, "exited removed container record must remain in actuatedState while runtime container exists")
	assert.Equal(t, "busybox", actuated.Image)
	mounts, found = m.actuatedState.GetContainerVolumeMounts(pod.UID, "removed")
	require.True(t, found)
	assert.Empty(t, mounts, "exited removed container must clear VolumeMounts so dynamically removed volumes can unmount")
	fakeHelper := m.runtimeHelper.(*containertest.FakeRuntimeHelper)
	assert.Equal(t, []string{"removed"}, fakeHelper.CleanedSubPathContainers, "exited removed container with subPath mount must clean container subPaths")

	// A removed dynamic container stuck in ContainerStateCreated (e.g. RunContainerError after CreateContainer)
	// must also clear its actuated mounts and subPaths so volume teardown is not wedged, while a container
	// still present in pod.Spec retains its immutable VolumeMounts.
	fakeHelper.CleanedSubPathContainers = nil
	createdPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "created-pod-uid", Name: "created", Namespace: "new"},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyOnFailure,
			Volumes:       []v1.Volume{{Name: "vol-keep", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}}},
			Containers: []v1.Container{{
				Name:         "c1",
				Image:        "busybox",
				VolumeMounts: []v1.VolumeMount{{Name: "vol-keep", MountPath: "/keep", SubPath: "sub-keep"}},
			}},
		},
	}
	m.InitializeActuatedPod(tCtx.Logger(), createdPod, nil)
	require.NoError(t, m.actuatedState.SetContainer(tCtx.Logger(), createdPod.UID, podutil.Containers, createdPod.Spec.Containers[0]))
	require.NoError(t, m.actuatedState.SetContainer(tCtx.Logger(), createdPod.UID, podutil.Containers, v1.Container{
		Name:         "removed-created",
		Image:        "busybox",
		VolumeMounts: []v1.VolumeMount{{Name: "vol-removed-created", MountPath: "/rm", SubPath: "sub"}},
	}))
	createdPodStatus := &kubecontainer.PodStatus{
		ID:              createdPod.UID,
		Name:            createdPod.Name,
		Namespace:       createdPod.Namespace,
		SandboxStatuses: curStatus.SandboxStatuses,
		ContainerStatuses: []*kubecontainer.Status{
			{
				ID:       kubecontainer.ContainerID{Type: "fake", ID: "c1-exited"},
				Name:     "c1",
				State:    kubecontainer.ContainerStateExited,
				ExitCode: 0,
			},
			{
				ID:    kubecontainer.ContainerID{Type: "fake", ID: "created-id"},
				Name:  "removed-created",
				State: kubecontainer.ContainerStateCreated,
			},
		},
	}
	m.computePodActions(tCtx, createdPod, createdPodStatus, false)
	keptExitedMounts, found := m.actuatedState.GetContainerVolumeMounts(createdPod.UID, "c1")
	require.True(t, found)
	assert.Equal(t, createdPod.Spec.Containers[0].VolumeMounts, keptExitedMounts, "container still in pod.Spec must retain immutable VolumeMounts")
	removedCreatedMounts, found := m.actuatedState.GetContainerVolumeMounts(createdPod.UID, "removed-created")
	require.True(t, found)
	assert.Empty(t, removedCreatedMounts, "removed ContainerStateCreated container must clear actuated mounts")
	assert.Equal(t, []string{"removed-created"}, fakeHelper.CleanedSubPathContainers)

	// Absent removed container (cs == nil, including when createPodSandbox=true) clears actuated mounts
	// and cleans subPaths, while a container still in pod.Spec retains its mounts across sandbox/node restart.
	fakeHelper.CleanedSubPathContainers = nil
	gcPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "gc-pod-uid", Name: "gc", Namespace: "new"},
		Spec: v1.PodSpec{
			Volumes:    []v1.Volume{{Name: "vol-empty", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}}},
			Containers: []v1.Container{{Name: "kept", Image: "busybox", VolumeMounts: []v1.VolumeMount{{Name: "vol-empty", MountPath: "/mnt", SubPath: "sub"}}}},
		},
	}
	m.InitializeActuatedPod(tCtx.Logger(), gcPod, nil)
	require.NoError(t, m.actuatedState.SetContainer(tCtx.Logger(), gcPod.UID, podutil.Containers, gcPod.Spec.Containers[0]))
	require.NoError(t, m.actuatedState.SetContainer(tCtx.Logger(), gcPod.UID, podutil.Containers, v1.Container{
		Name:         "gc-removed",
		Image:        "busybox",
		VolumeMounts: []v1.VolumeMount{{Name: "vol-old", MountPath: "/old", SubPath: "sub-old"}},
	}))
	m.computePodActions(tCtx, gcPod, &kubecontainer.PodStatus{ID: gcPod.UID, Name: gcPod.Name, Namespace: gcPod.Namespace}, false)
	keptMounts, found := m.actuatedState.GetContainerVolumeMounts(gcPod.UID, "kept")
	require.True(t, found)
	assert.Equal(t, gcPod.Spec.Containers[0].VolumeMounts, keptMounts, "container still in pod.Spec must retain actuated mounts across node/sandbox restart")
	gcRemovedMounts, found := m.actuatedState.GetContainerVolumeMounts(gcPod.UID, "gc-removed")
	require.True(t, found)
	assert.Empty(t, gcRemovedMounts, "absent removed container (cs == nil, createPodSandbox=true) must clear actuated mounts")
	assert.Equal(t, []string{"gc-removed"}, fakeHelper.CleanedSubPathContainers)

	// Containers migrated from a V1 checkpoint (stored with Name == "" and resources only) must
	// backfill active specs and mounted OCI Image volumes in InitializeActuatedPod across Running,
	// Exited/Created, and already-GC'd completed init containers (once regular containers started).
	v1Pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "v1-pod-uid", Name: "v1", Namespace: "new"},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyOnFailure,
			Volumes: []v1.Volume{
				{Name: "vol-empty", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}},
				{Name: "vol-img", VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: "busybox"}}},
				{Name: "vol-init-img", VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{Reference: "busybox:init"}}},
			},
			InitContainers: []v1.Container{{
				Name:         "gc-init-v1",
				Image:        "busybox",
				Resources:    resources,
				VolumeMounts: []v1.VolumeMount{{Name: "vol-init-img", MountPath: "/init-img"}},
			}},
			Containers: []v1.Container{
				{
					Name:      "running-v1",
					Image:     "busybox",
					Resources: resources,
					VolumeMounts: []v1.VolumeMount{
						{Name: "vol-empty", MountPath: "/empty"},
						{Name: "vol-img", MountPath: "/img"},
					},
				},
				{
					Name:         "exited-v1",
					Image:        "busybox",
					Resources:    resources,
					VolumeMounts: []v1.VolumeMount{{Name: "vol-empty", MountPath: "/done", SubPath: "sub"}},
				},
				{
					Name:         "created-v1",
					Image:        "busybox",
					Resources:    resources,
					VolumeMounts: []v1.VolumeMount{{Name: "vol-empty", MountPath: "/created"}},
				},
			},
		},
	}
	// Before any regular container has been created, an absent init container is not backfilled.
	preInitPod := v1Pod.DeepCopy()
	preInitPod.UID = "v1-pre-init-uid"
	m.InitializeActuatedPod(tCtx.Logger(), preInitPod, &kubecontainer.PodStatus{ID: preInitPod.UID, Name: preInitPod.Name, Namespace: preInitPod.Namespace})
	preInitMounts, found := m.actuatedState.GetContainerVolumeMounts(preInitPod.UID, "gc-init-v1")
	require.True(t, found)
	assert.Empty(t, preInitMounts, "absent init container must not be backfilled before any regular container is created")

	require.NoError(t, m.actuatedState.SetContainerResources(tCtx.Logger(), v1Pod.UID, "running-v1", podutil.Containers, resources))
	v1RunningStatus := &kubecontainer.PodStatus{
		ID:              v1Pod.UID,
		Name:            v1Pod.Name,
		Namespace:       v1Pod.Namespace,
		SandboxStatuses: curStatus.SandboxStatuses,
		ContainerStatuses: []*kubecontainer.Status{
			{
				ID:    kubecontainer.ContainerID{Type: "fake", ID: "v1-running-id"},
				Name:  "running-v1",
				State: kubecontainer.ContainerStateRunning,
			},
			{
				ID:       kubecontainer.ContainerID{Type: "fake", ID: "v1-exited-id"},
				Name:     "exited-v1",
				State:    kubecontainer.ContainerStateExited,
				ExitCode: 0,
			},
			{
				ID:    kubecontainer.ContainerID{Type: "fake", ID: "v1-created-id"},
				Name:  "created-v1",
				State: kubecontainer.ContainerStateCreated,
			},
		},
	}
	m.InitializeActuatedPod(tCtx.Logger(), v1Pod, v1RunningStatus)
	v1Mounts, found := m.actuatedState.GetContainerVolumeMounts(v1Pod.UID, "running-v1")
	require.True(t, found)
	assert.Equal(t, v1Pod.Spec.Containers[0].VolumeMounts, v1Mounts, "running container migrated from V1 checkpoint must backfill VolumeMounts")
	v1ExitedMounts, found := m.actuatedState.GetContainerVolumeMounts(v1Pod.UID, "exited-v1")
	require.True(t, found)
	assert.Equal(t, v1Pod.Spec.Containers[1].VolumeMounts, v1ExitedMounts, "completed container migrated from V1 checkpoint must backfill VolumeMounts")
	v1CreatedMounts, found := m.actuatedState.GetContainerVolumeMounts(v1Pod.UID, "created-v1")
	require.True(t, found)
	assert.Equal(t, v1Pod.Spec.Containers[2].VolumeMounts, v1CreatedMounts, "created container migrated from V1 checkpoint must backfill VolumeMounts")
	v1InitMounts, found := m.actuatedState.GetContainerVolumeMounts(v1Pod.UID, "gc-init-v1")
	require.True(t, found)
	assert.Equal(t, v1Pod.Spec.InitContainers[0].VolumeMounts, v1InitMounts, "GC'd completed init container migrated from V1 checkpoint must backfill VolumeMounts")
	v1Vols, found := m.actuatedState.GetPodVolumes(v1Pod.UID)
	require.True(t, found)
	assert.ElementsMatch(t, []v1.Volume{v1Pod.Spec.Volumes[1], v1Pod.Spec.Volumes[2]}, v1Vols, "containers migrated from V1 checkpoint must backfill mounted OCI Image volumes")

	// A named V2 checkpoint record (!state.IsMigratedFromV1) is left untouched by InitializeActuatedPod.
	namedV2Pod := v1Pod.DeepCopy()
	namedV2Pod.UID = "named-v2-uid"
	m.InitializeActuatedPod(tCtx.Logger(), namedV2Pod, nil)
	m.InitializeActuatedPod(tCtx.Logger(), namedV2Pod, v1RunningStatus)
	namedV2Mounts, found := m.actuatedState.GetContainerVolumeMounts(namedV2Pod.UID, "running-v1")
	require.True(t, found)
	assert.Empty(t, namedV2Mounts, "existing named V2 checkpoint must be left untouched by InitializeActuatedPod")

	// A removed container with a newer Exited status alongside an older Unknown/Running instance must
	// retain its actuated VolumeMounts and subPaths until all instances have stopped.
	fakeHelper.CleanedSubPathContainers = nil
	multiInstPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "multi-inst-uid", Name: "multi", Namespace: "new"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{Name: "main", Image: "busybox"}},
		},
	}
	m.InitializeActuatedPod(tCtx.Logger(), multiInstPod, nil)
	require.NoError(t, m.actuatedState.SetContainer(tCtx.Logger(), multiInstPod.UID, podutil.Containers, removed))
	multiRunningStatus := &kubecontainer.PodStatus{
		ID:              multiInstPod.UID,
		Name:            multiInstPod.Name,
		Namespace:       multiInstPod.Namespace,
		SandboxStatuses: curStatus.SandboxStatuses,
		ContainerStatuses: []*kubecontainer.Status{
			{
				ID:        kubecontainer.ContainerID{Type: "fake", ID: "newer-exited"},
				Name:      "removed",
				State:     kubecontainer.ContainerStateExited,
				CreatedAt: time.Unix(20, 0),
			},
			{
				ID:        kubecontainer.ContainerID{Type: "fake", ID: "older-unknown"},
				Name:      "removed",
				State:     kubecontainer.ContainerStateUnknown,
				CreatedAt: time.Unix(10, 0),
			},
		},
	}
	m.computePodActions(tCtx, multiInstPod, multiRunningStatus, false)
	multiMounts, found := m.actuatedState.GetContainerVolumeMounts(multiInstPod.UID, "removed")
	require.True(t, found)
	assert.Equal(t, removed.VolumeMounts, multiMounts, "older Unknown instance of removed container must keep actuated VolumeMounts")
	assert.Empty(t, fakeHelper.CleanedSubPathContainers, "older Unknown instance of removed container must not clean subPaths prematurely")

	multiRunningStatus.ContainerStatuses[1].State = kubecontainer.ContainerStateExited
	m.computePodActions(tCtx, multiInstPod, multiRunningStatus, false)
	multiMounts, found = m.actuatedState.GetContainerVolumeMounts(multiInstPod.UID, "removed")
	require.True(t, found)
	assert.Empty(t, multiMounts, "once all instances of removed container stop, actuated VolumeMounts must be cleared")
	assert.Equal(t, []string{"removed"}, fakeHelper.CleanedSubPathContainers)

	// InitializeActuatedPod (KeepOnlyResourceValues) must not seed VolumeMounts for containers that
	// have never started yet, and pod termination (DeletionTimestamp != nil) must retain actuated
	// mounts on stopped containers still in pod.Spec so ContainerStatus.VolumeMounts is preserved.
	unstartedPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "unstarted-pod-uid", Name: "unstarted", Namespace: "new"},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyOnFailure,
			Volumes:       []v1.Volume{{Name: "vol-init", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}}},
			Containers: []v1.Container{{
				Name:         "never-created",
				Image:        "busybox",
				Resources:    resources,
				VolumeMounts: []v1.VolumeMount{{Name: "vol-init", MountPath: "/mnt"}},
			}},
		},
	}
	m.InitializeActuatedPod(tCtx.Logger(), unstartedPod, nil)
	unstartedMounts, found := m.actuatedState.GetContainerVolumeMounts(unstartedPod.UID, "never-created")
	require.True(t, found)
	assert.Empty(t, unstartedMounts, "InitializeActuatedPod must not seed VolumeMounts before container starts")

	startedCtr := unstartedPod.Spec.Containers[0]
	require.NoError(t, m.actuatedState.SetContainer(tCtx.Logger(), unstartedPod.UID, podutil.Containers, startedCtr))
	terminatingPod := unstartedPod.DeepCopy()
	now := metav1.Now()
	terminatingPod.DeletionTimestamp = &now
	stoppedStatus := &kubecontainer.PodStatus{
		ID:        terminatingPod.UID,
		Name:      terminatingPod.Name,
		Namespace: terminatingPod.Namespace,
		ContainerStatuses: []*kubecontainer.Status{{
			ID:       kubecontainer.ContainerID{Type: "fake", ID: "done-ctr"},
			Name:     "never-created",
			State:    kubecontainer.ContainerStateExited,
			ExitCode: 0,
		}},
	}
	m.computePodActions(tCtx, terminatingPod, stoppedStatus, false)
	afterTermMounts, found := m.actuatedState.GetContainerVolumeMounts(terminatingPod.UID, "never-created")
	require.True(t, found)
	assert.Equal(t, startedCtr.VolumeMounts, afterTermMounts, "terminating pod must retain actuated mounts on stopped containers still in pod.Spec")

	// When startContainer fails in generateContainerConfig after PrepareSafeSubpath (before
	// setActuatedContainer), CleanContainerSubPaths is called immediately so no host subPath bind mount leaks.
	fakeHelper.CleanedSubPathContainers = nil
	runAsNonRoot := true
	rootUser := int64(0)
	badConfigPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "bad-cfg-uid", Name: "bad-cfg", Namespace: "new"},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{{Name: "vol-sub", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}}},
			Containers: []v1.Container{{
				Name:         "bad-subpath-ctr",
				Image:        "busybox",
				VolumeMounts: []v1.VolumeMount{{Name: "vol-sub", MountPath: "/sub", SubPath: "dir"}},
				SecurityContext: &v1.SecurityContext{
					RunAsNonRoot: &runAsNonRoot,
					RunAsUser:    &rootUser,
				},
			}},
		},
	}
	require.NoError(t, m.actuatedState.SetPodVolume(tCtx.Logger(), badConfigPod.UID, badConfigPod.Spec.Volumes[0]))
	m.InitializeActuatedPod(tCtx.Logger(), badConfigPod, nil)
	badSyncRes := m.SyncPod(tCtx, badConfigPod, &kubecontainer.PodStatus{}, []v1.Secret{}, backOff, false)
	require.ErrorIs(t, badSyncRes.Error(), ErrCreateContainerConfig)
	assert.Equal(t, []string{"bad-subpath-ctr"}, fakeHelper.CleanedSubPathContainers, "failed generateContainerConfig must clean container subPaths")

	// With both DynamicNodeLocalEphemeralVolumes and DynamicContainers disabled, InitializeActuatedPod
	// skips V1 container-spec backfill (matching master gate-off behavior).
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicNodeLocalEphemeralVolumes, false)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicContainers, false)
	gateOffV1Pod := v1Pod.DeepCopy()
	gateOffV1Pod.UID = "gate-off-v1-uid"
	require.NoError(t, m.actuatedState.SetContainerResources(tCtx.Logger(), gateOffV1Pod.UID, "running-v1", podutil.Containers, resources))
	m.InitializeActuatedPod(tCtx.Logger(), gateOffV1Pod, v1RunningStatus)
	gateOffMounts, found := m.actuatedState.GetContainerVolumeMounts(gateOffV1Pod.UID, "running-v1")
	require.True(t, found)
	assert.Empty(t, gateOffMounts, "gate-off InitializeActuatedPod must leave V1 container records resource-only")
}
