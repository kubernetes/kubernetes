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

package kubelet

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/allocation"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/utils/ptr"
)

func startupTestPod(uid string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: types.UID(uid), Name: uid, Namespace: "default",
			Annotations: map[string]string{kubetypes.ConfigSourceAnnotationKey: kubetypes.ApiserverSource}},
		Spec: v1.PodSpec{NodeName: testKubeletHostname, Containers: []v1.Container{{Name: "c", Image: "image",
			Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")}}}}},
		Status: v1.PodStatus{Phase: v1.PodRunning, StartTime: ptr.To(metav1.Now())},
	}
}

func TestRestorePodState(t *testing.T) {
	tCtx := ktesting.Init(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NotEvictPodOnKubeletRestart, true)
	tk := newTestKubelet(t, false)
	defer tk.Cleanup()
	kl := tk.kubelet
	kl.sourcesReady = config.NewSourcesReady(func(sets.Set[string]) bool { return false })
	kl.allocationManager = allocation.NewInMemoryManager(tCtx.Logger(), kl.statusManager, kl, kl.sourcesReady, record.NewFakeRecorder(20))

	running := startupTestPod("running")
	running.Status.StartTime = nil
	running.Annotations = nil
	terminal := startupTestPod("terminal")
	terminating := startupTestPod("terminating")
	deleted := startupTestPod("deleted")
	unknownSource := startupTestPod("old-v2")
	unknownSource.Annotations = nil
	static := startupTestPod("static")
	static.Annotations[kubetypes.ConfigSourceAnnotationKey] = kubetypes.FileSource
	legacy := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "legacy"}}
	for _, pod := range []*v1.Pod{running, terminal, terminating, deleted, unknownSource, static, legacy} {
		require.NoError(t, kl.allocationManager.SetAllocatedResources(tCtx.Logger(), pod))
	}
	// These pods would fail admission if either restoration or the first ADD
	// accidentally ran the admission handlers again.
	newPod := startupTestPod("new")
	kl.allocationManager.AddPodAdmitHandlers(lifecycle.PodAdmitHandlers{&testPodAdmitHandler{
		podsToReject: []*v1.Pod{running, terminal, terminating, newPod},
	}})
	desired := running.DeepCopy()
	desired.Labels = map[string]string{"current": "label"}
	desired.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse("2")
	desired.Status.Conditions = []v1.PodCondition{{Type: v1.PodResizePending, Status: v1.ConditionTrue, Reason: v1.PodReasonDeferred}}
	terminal.Status.Phase = v1.PodSucceeded
	terminating.DeletionTimestamp = ptr.To(metav1.Now())
	terminating.DeletionGracePeriodSeconds = ptr.To(int64(10))
	replacement := deleted.DeepCopy()
	replacement.UID = "replacement"

	client := fake.NewClientset()
	client.PrependReactor("list", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		opts := action.(clienttesting.ListActionImpl).ListOptions
		assert.Equal(t, "spec.nodeName="+string(kl.nodeName), opts.FieldSelector)
		assert.Empty(t, opts.ResourceVersion)
		return true, &v1.PodList{Items: []v1.Pod{*desired, *terminal, *terminating, *replacement, *newPod}}, nil
	})
	kl.kubeClient = client
	workers := kl.podWorkers.(*fakePodWorkers)
	var synced []types.UID
	workers.syncPodFn = func(_ context.Context, _ kubetypes.SyncPodType, pod, _ *v1.Pod, _ *kubecontainer.PodStatus) (bool, func(), error) {
		// Hydration of the whole set precedes the first worker update.
		for _, p := range []*v1.Pod{desired, terminal, terminating} {
			status, found := kl.statusManager.GetPodStatus(p.UID)
			assert.True(t, found)
			assert.Equal(t, p.Status.Phase, status.Phase)
		}
		synced = append(synced, pod.UID)
		return false, nil, nil
	}
	pending, err := kl.restorePodState(tCtx, nil)
	require.NoError(t, err)
	assert.Empty(t, pending)
	assert.ElementsMatch(t, []types.UID{running.UID, terminal.UID, terminating.UID}, synced)
	assert.Equal(t, []types.UID{deleted.UID}, workers.triggeredDeletion)
	assert.False(t, kl.sourcesReady.AllReady(), "deletion must not depend on informer readiness")
	assert.False(t, kl.allocationManager.HasPodAllocatedResources(deleted.UID))
	assert.True(t, kl.allocationManager.HasPodAllocatedResources(unknownSource.UID))
	assert.True(t, kl.allocationManager.HasPendingResizes())
	for _, checkpointed := range kl.allocationManager.GetCheckpointedPods() {
		if checkpointed.UID == running.UID {
			assert.Equal(t, kubetypes.ApiserverSource, checkpointed.Annotations[kubetypes.ConfigSourceAnnotationKey])
		}
	}
	restored, found := kl.podManager.GetPodByUID(running.UID)
	require.True(t, found)
	assert.Equal(t, desired.Labels, restored.Labels)
	allocated, changed := kl.allocationManager.UpdatePodFromAllocation(restored)
	assert.True(t, changed)
	assert.Equal(t, "1", allocated.Spec.Containers[0].Resources.Requests.Cpu().String())
	assert.True(t, kl.statusManager.IsPodResizeDeferred(running.UID))
	for _, uid := range []types.UID{deleted.UID, replacement.UID, newPod.UID, unknownSource.UID, static.UID, legacy.UID} {
		_, found := kl.podManager.GetPodByUID(uid)
		assert.False(t, found, "unexpected restored pod %s", uid)
	}

	kl.HandlePodAdditions(tCtx, []*v1.Pod{desired, terminal, terminating, newPod})
	checkPodStatus(t, kl, desired, v1.PodRunning)
	checkPodStatus(t, kl, newPod, v1.PodFailed)
}

func TestRestorePodStateRetriesWhileStartingStaticPods(t *testing.T) {
	tCtx := ktesting.Init(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NotEvictPodOnKubeletRestart, true)
	tk := newTestKubelet(t, false)
	defer tk.Cleanup()
	kl := tk.kubelet
	pod := startupTestPod("api-pod")
	static := startupTestPod("static-pod")
	static.Annotations[kubetypes.ConfigSourceAnnotationKey] = kubetypes.FileSource
	require.NoError(t, kl.allocationManager.SetAllocatedResources(tCtx.Logger(), pod))
	require.NoError(t, kl.allocationManager.SetAllocatedResources(tCtx.Logger(), static))
	kl.allocationManager.AddPodAdmitHandlers(lifecycle.PodAdmitHandlers{&testPodAdmitHandler{podsToReject: []*v1.Pod{static}}})

	ctx, cancel := context.WithTimeout(tCtx, 10*time.Second)
	defer cancel()
	staticStarted := make(chan struct{})
	staticAttempts := 0
	kl.podWorkers.(*fakePodWorkers).syncPodFn = func(_ context.Context, _ kubetypes.SyncPodType, p, _ *v1.Pod, _ *kubecontainer.PodStatus) (bool, func(), error) {
		if p.UID == static.UID {
			staticAttempts++
			if staticAttempts == 1 {
				kl.workQueue.Enqueue(static.UID, -time.Second)
			} else {
				close(staticStarted)
			}
		}
		return false, nil, nil
	}
	client := fake.NewClientset()
	attempts := 0
	client.PrependReactor("list", "pods", func(clienttesting.Action) (bool, runtime.Object, error) {
		attempts++
		if attempts == 1 {
			return true, nil, fmt.Errorf("API server is starting")
		}
		select {
		case <-staticStarted:
		case <-ctx.Done():
			return true, nil, ctx.Err()
		}
		return true, &v1.PodList{Items: []v1.Pod{*pod}}, nil
	})
	kl.kubeClient = client
	updates := make(chan kubetypes.PodUpdate, 2)
	apiUpdate := kubetypes.PodUpdate{Pods: []*v1.Pod{pod}, Op: kubetypes.ADD, Source: kubetypes.ApiserverSource}
	updates <- apiUpdate
	updates <- kubetypes.PodUpdate{Pods: []*v1.Pod{static}, Op: kubetypes.ADD, Source: kubetypes.FileSource}
	pending, err := kl.restorePodState(ctx, updates)
	require.NoError(t, err)
	assert.Equal(t, []kubetypes.PodUpdate{apiUpdate}, pending)
	assert.Equal(t, 2, attempts)
	assert.Equal(t, 2, staticAttempts)
	assert.Empty(t, kl.podWorkers.(*fakePodWorkers).triggeredDeletion)
}

func TestRestorePodStateCancellation(t *testing.T) {
	tCtx := ktesting.Init(t)
	tk := newTestKubelet(t, false)
	defer tk.Cleanup()
	kl := tk.kubelet
	pod := startupTestPod("pod")
	require.NoError(t, kl.allocationManager.SetAllocatedResources(tCtx.Logger(), pod))
	ctx, cancel := context.WithCancel(tCtx)
	client := fake.NewClientset()
	client.PrependReactor("list", "pods", func(clienttesting.Action) (bool, runtime.Object, error) {
		cancel()
		return true, nil, fmt.Errorf("API server unavailable")
	})
	kl.kubeClient = client
	_, err := kl.restorePodState(ctx, nil)
	require.ErrorIs(t, err, context.Canceled)
	assert.Empty(t, kl.podManager.GetPods())
	assert.Empty(t, kl.podWorkers.(*fakePodWorkers).triggeredDeletion)
	assert.True(t, kl.allocationManager.HasPodAllocatedResources(pod.UID))
}

func TestHandlePodAdditionsRestorationDisabled(t *testing.T) {
	tCtx := ktesting.Init(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NotEvictPodOnKubeletRestart, false)
	tk := newTestKubelet(t, false)
	defer tk.Cleanup()
	kl := tk.kubelet
	pod := startupTestPod("pod")
	kl.podManager.AddPod(pod)
	require.NoError(t, kl.allocationManager.SetAllocatedResources(tCtx.Logger(), pod))
	kl.allocationManager.AddPodAdmitHandlers(lifecycle.PodAdmitHandlers{&testPodAdmitHandler{podsToReject: []*v1.Pod{pod}}})
	kl.HandlePodAdditions(tCtx, []*v1.Pod{pod})
	checkPodStatus(t, kl, pod, v1.PodFailed)
}

func TestRestorePodStateWithoutAPIPods(t *testing.T) {
	for _, tc := range []struct {
		name       string
		checkpoint *v1.Pod
		standalone bool
	}{
		{name: "empty checkpoint"},
		{name: "legacy checkpoint", checkpoint: &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "legacy"}}},
		{name: "static pod", checkpoint: &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "static", Name: "static",
			Annotations: map[string]string{kubetypes.ConfigSourceAnnotationKey: kubetypes.FileSource}}}},
		{name: "standalone kubelet", checkpoint: startupTestPod("pod"), standalone: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			tk := newTestKubelet(t, false)
			defer tk.Cleanup()
			kl := tk.kubelet
			if tc.checkpoint != nil {
				require.NoError(t, kl.allocationManager.SetAllocatedResources(tCtx.Logger(), tc.checkpoint))
			}
			client := fake.NewClientset()
			kl.kubeClient = client
			if tc.standalone {
				kl.kubeClient = nil
			}
			pending, err := kl.restorePodState(tCtx, nil)
			require.NoError(t, err)
			assert.Empty(t, pending)
			assert.Empty(t, client.Actions())
			assert.Empty(t, kl.podManager.GetPods())
			assert.Empty(t, kl.podWorkers.(*fakePodWorkers).triggeredDeletion)
		})
	}
}
