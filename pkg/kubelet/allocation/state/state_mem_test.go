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

package state

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

func TestStateMemory_EmptyDirVolumeLimits(t *testing.T) {
	logger := klog.TODO()
	state := NewStateMemory(logger, PodMap{})

	podUID := types.UID("pod-1")
	volName := "volume-1"

	// Get on nonexistent pod should return (nil, false)
	qty, exists := state.GetEmptyDirVolumeLimit(podUID, volName)
	assert.Nil(t, qty)
	assert.False(t, exists)

	// Set volume limit on nonexistent pod should implicitly initialize the pod and insert it
	targetLimit := resource.MustParse("256Mi")
	err := state.SetEmptyDirVolumeLimit(podUID, volName, &targetLimit)
	require.NoError(t, err)

	// Get volume limit should return the parsed value and true
	qty, exists = state.GetEmptyDirVolumeLimit(podUID, volName)
	require.True(t, exists)
	require.NotNil(t, qty)
	assert.True(t, targetLimit.Equal(*qty))

	// Get on existing pod with nonexistent volume should return (nil, false)
	qtyNonexistent, existsNonexistent := state.GetEmptyDirVolumeLimit(podUID, "nonexistent-volume")
	assert.Nil(t, qtyNonexistent)
	assert.False(t, existsNonexistent)

	// Returned quantity should be a deep copy (mutability check)
	qty.Set(1024 * 1024 * 512) // Modify the returned Quantity value to 512Mi
	refreshedQty, exists := state.GetEmptyDirVolumeLimit(podUID, volName)
	require.True(t, exists)
	assert.True(t, targetLimit.Equal(*refreshedQty), "Modifying the returned Quantity pointer should not alter Kubelet's internal memory state")

	// Set another volume on the same pod should keep existing limits intact
	anotherVolName := "volume-2"
	anotherLimit := resource.MustParse("128Mi")
	err = state.SetEmptyDirVolumeLimit(podUID, anotherVolName, &anotherLimit)
	require.NoError(t, err)

	// Verify both volumes are present and correct
	qty1, exists1 := state.GetEmptyDirVolumeLimit(podUID, volName)
	assert.True(t, exists1)
	assert.True(t, targetLimit.Equal(*qty1))

	qty2, exists2 := state.GetEmptyDirVolumeLimit(podUID, anotherVolName)
	assert.True(t, exists2)
	assert.True(t, anotherLimit.Equal(*qty2))
}

func TestStateMemory_ResourceIsolation(t *testing.T) {
	logger := klog.TODO()
	state := NewStateMemory(logger, PodMap{})

	podUID := types.UID("pod-1")
	containerName := "container-1"
	volumeName := "volume-1"

	// Set container resources
	containerResources := v1.ResourceRequirements{
		Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("250m")},
	}
	require.NoError(t, state.SetContainerResources(logger, podUID, containerName, podutil.Containers, containerResources))

	// Verify container resources are set, and others are not
	res, found := state.GetContainerResources(podUID, containerName)
	assert.True(t, found)
	assert.True(t, containerResources.Requests.Cpu().Equal(*res.Requests.Cpu()))
	podRes, _ := state.GetPodLevelResources(podUID)
	assert.Nil(t, podRes)
	volLimit, exists := state.GetEmptyDirVolumeLimit(podUID, volumeName)
	assert.Nil(t, volLimit)
	assert.False(t, exists)

	// Set pod-level resources
	podResources := &v1.ResourceRequirements{
		Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("512Mi")},
	}
	require.NoError(t, state.SetPodLevelResources(logger, podUID, podResources))

	// Verify pod-level resources are set, AND container resources are still intact
	podRes, _ = state.GetPodLevelResources(podUID)
	require.NotNil(t, podRes)
	assert.True(t, podResources.Requests.Memory().Equal(*podRes.Requests.Memory()))
	res, found = state.GetContainerResources(podUID, containerName)
	assert.True(t, found)
	assert.True(t, containerResources.Requests.Cpu().Equal(*res.Requests.Cpu()))
	volLimit, exists = state.GetEmptyDirVolumeLimit(podUID, volumeName)
	assert.Nil(t, volLimit)
	assert.False(t, exists)

	// Set emptyDir volume limit
	targetLimit := resource.MustParse("256Mi")
	require.NoError(t, state.SetEmptyDirVolumeLimit(podUID, volumeName, &targetLimit))

	// Verify volume limit is set, AND both container and pod-level resources are still intact
	volLimit, exists = state.GetEmptyDirVolumeLimit(podUID, volumeName)
	require.True(t, exists)
	require.NotNil(t, volLimit)
	assert.True(t, targetLimit.Equal(*volLimit))
	res, found = state.GetContainerResources(podUID, containerName)
	assert.True(t, found)
	assert.True(t, containerResources.Requests.Cpu().Equal(*res.Requests.Cpu()))
	podRes, _ = state.GetPodLevelResources(podUID)
	require.NotNil(t, podRes)
	assert.True(t, podResources.Requests.Memory().Equal(*podRes.Requests.Memory()))

	// Update container resources again
	updatedContainerResources := v1.ResourceRequirements{
		Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("500m")},
	}
	require.NoError(t, state.SetContainerResources(logger, podUID, containerName, podutil.Containers, updatedContainerResources))

	// Verify container resources are updated, AND both pod-level resources and volume limits are still intact
	res, found = state.GetContainerResources(podUID, containerName)
	assert.True(t, found)
	assert.True(t, updatedContainerResources.Requests.Cpu().Equal(*res.Requests.Cpu()))
	podRes, _ = state.GetPodLevelResources(podUID)
	require.NotNil(t, podRes)
	assert.True(t, podResources.Requests.Memory().Equal(*podRes.Requests.Memory()))
	volLimit, exists = state.GetEmptyDirVolumeLimit(podUID, volumeName)
	require.True(t, exists)
	require.NotNil(t, volLimit)
	assert.True(t, targetLimit.Equal(*volLimit))
}

func TestStateMemory_SetContainerResources_ContainerTypes(t *testing.T) {
	logger := klog.TODO()
	state := NewStateMemory(logger, PodMap{})
	podUID := types.UID("pod-1")

	resources := func(cpu string) v1.ResourceRequirements {
		return v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse(cpu)}}
	}
	cpuOf := func(name string) resource.Quantity {
		res, found := state.GetContainerResources(podUID, name)
		require.True(t, found, name)
		return *res.Requests.Cpu()
	}

	require.NoError(t, state.SetContainerResources(logger, podUID, "init", podutil.InitContainers, resources("1")))
	require.NoError(t, state.SetContainerResources(logger, podUID, "main", podutil.Containers, resources("2")))
	require.NoError(t, state.SetContainerResources(logger, podUID, "debug", podutil.EphemeralContainers, resources("3")))

	pod, ok := state.GetPod(podUID)
	require.True(t, ok)
	require.Len(t, pod.Spec.InitContainers, 1)
	assert.Equal(t, "init", pod.Spec.InitContainers[0].Name)
	require.Len(t, pod.Spec.Containers, 1)
	assert.Equal(t, "main", pod.Spec.Containers[0].Name)
	require.Len(t, pod.Spec.EphemeralContainers, 1)
	assert.Equal(t, "debug", pod.Spec.EphemeralContainers[0].Name)

	// A container is found whatever kind it is, and updated instead of added again. That matters for a
	// pod migrated from a V1 checkpoint, which keeps all of its containers in Spec.Containers.
	for _, containerType := range []podutil.ContainerType{podutil.InitContainers, podutil.Containers, podutil.EphemeralContainers} {
		for _, name := range []string{"init", "main", "debug"} {
			require.NoError(t, state.SetContainerResources(logger, podUID, name, containerType, resources("4")))
			expected := resource.MustParse("4")
			assert.True(t, expected.Equal(cpuOf(name)), "%s as %d", name, containerType)
		}
	}
	pod, _ = state.GetPod(podUID)
	assert.Len(t, pod.Spec.InitContainers, 1)
	assert.Len(t, pod.Spec.Containers, 1)
	assert.Len(t, pod.Spec.EphemeralContainers, 1)

	// A container that is not stored cannot be added with a type that does not say where it belongs.
	err := state.SetContainerResources(logger, "pod-2", "main", podutil.Containers|podutil.InitContainers, resources("1"))
	require.Error(t, err)
	assert.False(t, state.HasPod("pod-2"), "a failed update should not create the pod")
	err = state.SetContainerResources(logger, podUID, "new", 0, resources("1"))
	require.Error(t, err)
	_, found := state.GetContainerResources(podUID, "new")
	assert.False(t, found)
}

func TestStateMemory_SetPod(t *testing.T) {
	logger := klog.TODO()
	state := NewStateMemory(logger, PodMap{})

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "pod-1", Name: "name", Namespace: "namespace", Labels: map[string]string{"a": "b"}},
		Spec: v1.PodSpec{
			NodeName: "node",
			Containers: []v1.Container{{
				Name:      "main",
				Image:     "image",
				Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")}},
			}},
		},
		Status: v1.PodStatus{Phase: v1.PodRunning},
	}
	require.NoError(t, state.SetPod(logger, pod))

	// Only the identity and the spec are stored.
	stored, ok := state.GetPod(pod.UID)
	require.True(t, ok)
	expected := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: pod.UID, Name: pod.Name, Namespace: pod.Namespace},
		Spec:       *pod.Spec.DeepCopy(),
	}
	assert.True(t, apiequality.Semantic.DeepEqual(expected, stored))

	// The stored pod does not change when the pod it was made from does.
	pod.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse("2")
	pod.Spec.NodeName = "other"
	res, found := state.GetContainerResources(pod.UID, "main")
	require.True(t, found)
	assert.True(t, resource.MustParse("1").Equal(*res.Requests.Cpu()))
	stored, _ = state.GetPod(pod.UID)
	assert.Equal(t, "node", stored.Spec.NodeName)

	// Setting a pod replaces what was stored for it.
	require.NoError(t, state.SetPod(logger, &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: pod.UID}}))
	_, found = state.GetContainerResources(pod.UID, "main")
	assert.False(t, found)
}

func TestStateMemory_SettersCopyTheirInput(t *testing.T) {
	logger := klog.TODO()
	state := NewStateMemory(logger, PodMap{})
	podUID := types.UID("pod-1")

	containerResources := v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")}}
	require.NoError(t, state.SetContainerResources(logger, podUID, "main", podutil.Containers, containerResources))
	podResources := &v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("1Gi")}}
	require.NoError(t, state.SetPodLevelResources(logger, podUID, podResources))
	limit := resource.MustParse("1Gi")
	require.NoError(t, state.SetEmptyDirVolumeLimit(podUID, "volume", &limit))

	containerResources.Requests[v1.ResourceCPU] = resource.MustParse("2")
	podResources.Requests[v1.ResourceMemory] = resource.MustParse("2Gi")
	limit.Set(2 << 30)

	res, _ := state.GetContainerResources(podUID, "main")
	assert.True(t, resource.MustParse("1").Equal(*res.Requests.Cpu()))
	podRes, _ := state.GetPodLevelResources(podUID)
	assert.True(t, resource.MustParse("1Gi").Equal(*podRes.Requests.Memory()))
	volLimit, _ := state.GetEmptyDirVolumeLimit(podUID, "volume")
	assert.True(t, resource.MustParse("1Gi").Equal(*volLimit))

	// Nor does what the getters return share anything with the state.
	res.Requests[v1.ResourceCPU] = resource.MustParse("3")
	podRes.Requests[v1.ResourceMemory] = resource.MustParse("3Gi")
	volLimit.Set(3 << 30)
	pod, _ := state.GetPod(podUID)
	pod.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse("3")
	pod.Spec.Resources.Requests[v1.ResourceMemory] = resource.MustParse("3Gi")
	pod.Spec.Volumes[0].EmptyDir.SizeLimit.Set(3 << 30)
	res, _ = state.GetContainerResources(podUID, "main")
	assert.True(t, resource.MustParse("1").Equal(*res.Requests.Cpu()))
	podRes, _ = state.GetPodLevelResources(podUID)
	assert.True(t, resource.MustParse("1Gi").Equal(*podRes.Requests.Memory()))
	volLimit, _ = state.GetEmptyDirVolumeLimit(podUID, "volume")
	assert.True(t, resource.MustParse("1Gi").Equal(*volLimit))
}

// A stored pod is shared with toPodList, so every update has to leave it, and anything it refers to, alone.
func TestStateMemory_UpdatesDoNotModifyStoredPods(t *testing.T) {
	logger := klog.TODO()
	sm := newStateMemory(logger, PodMap{})
	podUID := types.UID("pod-1")

	cpu := func(q string) v1.ResourceRequirements {
		return v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse(q)}}
	}
	limit := resource.MustParse("2Gi")
	require.NoError(t, sm.SetPod(logger, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: podUID},
		Spec: v1.PodSpec{
			InitContainers:      []v1.Container{{Name: "init", Resources: cpu("1")}},
			Containers:          []v1.Container{{Name: "main", Resources: cpu("1")}},
			EphemeralContainers: []v1.EphemeralContainer{{EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "debug", Resources: cpu("1")}}},
			Resources:           &v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("1Gi")}},
			Volumes: []v1.Volume{{
				Name:         "volume",
				VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{Medium: v1.StorageMediumMemory, SizeLimit: &limit}},
			}},
		},
	}))

	newLimit := resource.MustParse("4Gi")
	updates := map[string]func() error{
		"init container": func() error {
			return sm.SetContainerResources(logger, podUID, "init", podutil.InitContainers, cpu("2"))
		},
		"container": func() error { return sm.SetContainerResources(logger, podUID, "main", podutil.Containers, cpu("2")) },
		"ephemeral container": func() error {
			return sm.SetContainerResources(logger, podUID, "debug", podutil.EphemeralContainers, cpu("2"))
		},
		"new container": func() error { return sm.SetContainerResources(logger, podUID, "new", podutil.Containers, cpu("2")) },
		"pod-level resources": func() error {
			return sm.SetPodLevelResources(logger, podUID, &v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("2Gi")}})
		},
		"volume limit":     func() error { return sm.SetEmptyDirVolumeLimit(podUID, "volume", &newLimit) },
		"new volume limit": func() error { return sm.SetEmptyDirVolumeLimit(podUID, "new-volume", &newLimit) },
	}
	for _, name := range []string{"init container", "container", "ephemeral container", "new container", "pod-level resources", "volume limit", "new volume limit"} {
		// Not GetPod: that returns a copy, which would never show a modification.
		stored := sm.pods[podUID]
		storedBefore := stored.DeepCopy()
		podListBefore := sm.toPodList()
		podListCopy := podListBefore.DeepCopy()

		require.NoError(t, updates[name](), name)

		assert.True(t, apiequality.Semantic.DeepEqual(storedBefore, stored), "%s: the stored pod was modified", name)
		assert.True(t, apiequality.Semantic.DeepEqual(podListCopy, podListBefore), "%s: the pod list handed out earlier was modified", name)
		updated, _ := sm.GetPod(podUID)
		assert.False(t, apiequality.Semantic.DeepEqual(storedBefore, updated), "%s: the update was not applied", name)
	}

	// The medium of a volume is not touched when its limit is updated.
	pod, _ := sm.GetPod(podUID)
	assert.Equal(t, v1.StorageMediumMemory, pod.Spec.Volumes[0].EmptyDir.Medium)
}

func TestStateMemory_GetPodUIDs(t *testing.T) {
	logger := klog.TODO()
	state := NewStateMemory(logger, PodMap{})

	assert.Empty(t, state.GetPodUIDs())

	pod1 := types.UID("pod-1")
	pod2 := types.UID("pod-2")
	require.NoError(t, state.SetPod(logger, &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: pod1}}))
	require.NoError(t, state.SetPod(logger, &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: pod2}}))
	assert.ElementsMatch(t, []types.UID{pod1, pod2}, state.GetPodUIDs())

	require.NoError(t, state.RemovePod(logger, pod1))
	assert.Equal(t, []types.UID{pod2}, state.GetPodUIDs())
}

func TestStateMemory_HasPod(t *testing.T) {
	logger := klog.TODO()
	state := NewStateMemory(logger, PodMap{})

	podUID := types.UID("pod-1")
	assert.False(t, state.HasPod(podUID))

	// A pod is created by the first update of any kind.
	require.NoError(t, state.SetPodLevelResources(logger, podUID, nil))
	assert.True(t, state.HasPod(podUID))

	require.NoError(t, state.RemovePod(logger, podUID))
	assert.False(t, state.HasPod(podUID))
}

func TestStateMemory_RemoveOrphanedPods(t *testing.T) {
	logger := klog.TODO()
	state := NewStateMemory(logger, PodMap{})
	for _, uid := range []types.UID{"pod-1", "pod-2", "pod-3"} {
		require.NoError(t, state.SetPod(logger, &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: uid}}))
	}

	state.RemoveOrphanedPods(sets.New[types.UID]("pod-2"))
	assert.Equal(t, []types.UID{"pod-2"}, state.GetPodUIDs())
}

func TestStateMemory_ToPodList(t *testing.T) {
	logger := klog.TODO()
	sm := newStateMemory(logger, PodMap{})
	assert.Empty(t, sm.toPodList().Items)

	require.NoError(t, sm.SetPod(logger, &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod-1"}}))
	require.NoError(t, sm.SetPod(logger, &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "pod-2"}}))

	uids := []types.UID{}
	for _, pod := range sm.toPodList().Items {
		uids = append(uids, pod.UID)
	}
	assert.ElementsMatch(t, []types.UID{"pod-1", "pod-2"}, uids)
}
