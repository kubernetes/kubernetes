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
	"errors"
	"os"
	"path/filepath"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/ktesting"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

func checkpointOperationFixture(t *testing.T) (*Kubelet, *containertest.FakeRuntime, *v1.Pod, *nodev1alpha1.PodCheckpoint) {
	t.Helper()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tk := newTestKubelet(t, false)
	t.Cleanup(tk.Cleanup)
	kl, runtime := tk.kubelet, tk.fakeRuntime
	pod := podWithUIDNameNsSpec("source-uid", "source", "ns", v1.PodSpec{NodeName: "testNode", Containers: []v1.Container{{Name: "app"}}})
	kl.podManager.SetPods([]*v1.Pod{pod})
	runtime.PodList = []*containertest.FakePod{{Pod: &kubecontainer.Pod{ID: pod.UID, Name: pod.Name, Namespace: pod.Namespace}}}
	runtime.PodStatus = kubecontainer.PodStatus{
		ID: pod.UID, Name: pod.Name, Namespace: pod.Namespace,
		SandboxStatuses:   []*runtimeapi.PodSandboxStatus{{Id: "sandbox", State: runtimeapi.PodSandboxState_SANDBOX_READY}},
		ContainerStatuses: []*kubecontainer.Status{{ID: kubecontainer.ContainerID{Type: "test", ID: "app-id"}, Name: "app", State: kubecontainer.ContainerStateRunning, ImageRef: "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}},
	}
	pc := &nodev1alpha1.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Namespace: pod.Namespace, Name: "checkpoint", UID: "checkpoint-uid"}}
	kl.kubeClient = fake.NewClientset(pc)
	kl.kubeletConfiguration.PodCheckpointTimeout.Duration = time.Minute
	useCheckpointPodWorkers(kl)
	return kl, runtime, pod, pc
}

func TestCheckpointOperationWaitsForRuntimeCompletion(t *testing.T) {
	for _, completionErr := range []error{errors.New("writer is still active"), context.DeadlineExceeded, context.Canceled} {
		t.Run(completionErr.Error(), func(t *testing.T) { testCheckpointOperationWaitsForRuntimeCompletion(t, completionErr) })
	}
}

func testCheckpointOperationWaitsForRuntimeCompletion(t *testing.T, completionErr error) {
	t.Helper()
	var ctx context.Context = ktesting.Init(t)
	kl, runtime, pod, pc := checkpointOperationFixture(t)
	kl.checkpointOperationContext.Store(&ctx)
	probes := &checkpointProbeManager{resumed: make(chan types.UID, 8), suspend: func(context.Context, types.UID) error { return nil }}
	kl.probeManager = probes
	var acknowledged atomic.Bool
	runtime.CancelPodOperationFunc = func(cancelCtx context.Context, request *runtimeapi.CancelPodOperationRequest) error {
		deadline, ok := cancelCtx.Deadline()
		require.True(t, ok)
		require.True(t, deadline.After(time.Now()))
		require.LessOrEqual(t, time.Until(deadline), podCheckpointAPITimeout)
		require.NoError(t, cancelCtx.Err(), "settlement must work after capture cancellation")
		require.Equal(t, string(pc.UID), request.OperationId)
		require.Equal(t, runtimeapi.PodOperationType_POD_CHECKPOINT, request.OperationType)
		if !acknowledged.Load() {
			return completionErr
		}
		return nil
	}
	captureCalls := 0
	kl.containerRuntime = &checkpointRuntime{Runtime: runtime, checkpointPod: func(_ context.Context, request *runtimeapi.CheckpointPodRequest) error {
		captureCalls++
		require.Equal(t, string(pc.UID), request.OperationId)
		require.NoError(t, os.WriteFile(filepath.Join(request.OutputPath, "partial"), []byte("still used"), 0600))
		return context.DeadlineExceeded
	}}
	require.NoError(t, checkpointPod(ctx, kl, pod, pc.Name, pc.UID, 0))
	require.Empty(t, probes.resumed)
	require.True(t, kl.podCheckpointInFlight(pod.UID))
	operations, err := kl.readPodCheckpointOperations(pod.UID)
	require.NoError(t, err)
	require.Len(t, operations, 1)
	require.False(t, operations[0].Complete)
	_, output, err := kl.podCheckpointOutputPath(pc.UID)
	require.NoError(t, err)
	require.FileExists(t, filepath.Join(output, "partial"))
	cached, _ := kl.statusManager.GetPodStatus(pod.UID)
	_, condition := podutil.GetPodCondition(&cached, v1.PodCheckpointing)
	require.NotNil(t, condition)

	_, _, err = kl.SyncPod(ctx, kubetypes.SyncPodSync, pod, nil, &runtime.PodStatus)
	require.ErrorIs(t, err, errPodCheckpointCompletionPending)
	grace := int64(0)
	require.ErrorIs(t, kl.SyncTerminatingPod(ctx, pod, &runtime.PodStatus, &grace, nil), errPodCheckpointCompletionPending)
	require.ErrorIs(t, kl.SyncTerminatedPod(ctx, pod, &runtime.PodStatus), errPodCheckpointCompletionPending)
	runtime.Lock()
	called := slices.Clone(runtime.CalledFunctions)
	runtime.Unlock()
	require.False(t, slices.Contains(called, "SyncPod"))
	require.False(t, slices.Contains(called, "KillPod"))
	require.Equal(t, 1, captureCalls)

	acknowledged.Store(true)
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	require.NoError(t, kl.reconcilePodCheckpointOperation(cancelled, pod))
	require.Eventually(t, func() bool { return !kl.podCheckpointInFlight(pod.UID) }, 5*time.Second, time.Millisecond)
	require.NoFileExists(t, filepath.Join(output, "partial"))
	require.NotEmpty(t, probes.resumed)
	cached, _ = kl.statusManager.GetPodStatus(pod.UID)
	_, condition = podutil.GetPodCondition(&cached, v1.PodCheckpointing)
	require.Nil(t, condition)
	require.Equal(t, 1, captureCalls, "settlement must not capture again")
	stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(ctx, pc.Name, metav1.GetOptions{})
	require.NoError(t, err)
	require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady).Reason)
}

func TestCheckpointOperationRecoverySurvivesObjectDeletion(t *testing.T) {
	var ctx context.Context = ktesting.Init(t)
	kl, runtime, pod, _ := checkpointOperationFixture(t)
	kl.kubeClient = fake.NewClientset()
	kl.checkpointOperationContext.Store(&ctx)
	for _, uid := range []types.UID{"first-checkpoint", "second-checkpoint"} {
		require.NoError(t, kl.writePodCheckpointOperation(podCheckpointOperation{CheckpointUID: uid, PodUID: pod.UID, PodName: pod.Name, Namespace: pod.Namespace, Name: string(uid)}))
		root, output, err := kl.podCheckpointOutputPath(uid)
		require.NoError(t, err)
		require.NoError(t, os.MkdirAll(root, 0700))
		require.NoError(t, os.Mkdir(output, 0700))
		require.NoError(t, os.WriteFile(filepath.Join(output, "partial"), []byte("partial"), 0600))
	}
	operations, err := kl.readPodCheckpointOperations(pod.UID)
	require.NoError(t, err)
	require.Len(t, operations, 2, "a later checkpoint must not overwrite an older operation")
	var calls []string
	runtime.CancelPodOperationFunc = func(_ context.Context, request *runtimeapi.CancelPodOperationRequest) error {
		calls = append(calls, request.OperationId)
		return nil
	}
	other := pod.DeepCopy()
	other.UID = "unrelated-uid"
	require.NoError(t, kl.reconcilePodCheckpointOperation(ctx, other))
	require.Empty(t, calls, "unrelated Pod must progress without waiting for this source")
	kl.recoverPodCheckpointOperations(ctx)
	require.ElementsMatch(t, []string{"first-checkpoint", "second-checkpoint"}, calls)
	require.Eventually(t, func() bool { ops, err := kl.readPodCheckpointOperations(pod.UID); return err == nil && len(ops) == 0 }, 5*time.Second, time.Millisecond)
	for _, uid := range []types.UID{"first-checkpoint", "second-checkpoint"} {
		_, output, err := kl.podCheckpointOutputPath(uid)
		require.NoError(t, err)
		require.NoDirExists(t, output)
	}
	require.Empty(t, runtime.StartedPods)
}

func TestCheckpointOperationProtectsUndiscoveredPodResources(t *testing.T) {
	ctx := ktesting.Init(t)
	kl, runtime, pod, _ := checkpointOperationFixture(t)
	kl.kubeClient = nil
	kl.podManager.SetPods(nil)
	runtime.PodList = nil
	base := &podWorkers{}
	base.SyncKnownPods(ctx.Logger(), nil)
	kl.podWorkers = &podOperationAwarePodWorkers{PodWorkers: base, kubelet: kl}
	require.True(t, base.ShouldPodRuntimeBeRemoved(pod.UID), "no source Pod was rediscovered after restart")
	require.True(t, base.ShouldPodContentBeRemoved(pod.UID))
	require.False(t, base.CouldHaveRunningContainers(pod.UID))
	for _, uid := range []types.UID{"first-checkpoint", "second-checkpoint"} {
		require.NoError(t, kl.writePodCheckpointOperation(podCheckpointOperation{CheckpointUID: uid, PodUID: pod.UID, PodName: pod.Name, Namespace: pod.Namespace, Name: string(uid)}))
	}
	assertRetained := func() {
		t.Helper()
		require.False(t, kl.podWorkers.ShouldPodRuntimeBeRemoved(pod.UID), "volume cleanup must retain resources")
		require.False(t, kl.podWorkers.ShouldPodContentBeRemoved(pod.UID), "Pod garbage collection must retain content")
		require.True(t, kl.podWorkers.CouldHaveRunningContainers(pod.UID))
		require.True(t, kl.podWorkers.ShouldPodRuntimeBeRemoved("unrelated-pod"))
		require.True(t, kl.podWorkers.ShouldPodContentBeRemoved("unrelated-pod"))
		require.False(t, kl.podWorkers.CouldHaveRunningContainers("unrelated-pod"))
	}
	assertRetained()
	runtime.CancelPodOperationFunc = func(_ context.Context, request *runtimeapi.CancelPodOperationRequest) error {
		if request.OperationId == "second-checkpoint" {
			return errors.New("writer is still active")
		}
		return nil
	}
	require.ErrorIs(t, kl.reconcilePodCheckpointOperation(ctx, pod), errPodCheckpointCompletionPending)
	assertRetained()
	runtime.CancelPodOperationFunc = func(context.Context, *runtimeapi.CancelPodOperationRequest) error { return nil }
	require.NoError(t, kl.reconcilePodCheckpointOperation(ctx, pod))
	operations, err := kl.readPodCheckpointOperations(pod.UID)
	require.NoError(t, err)
	require.Len(t, operations, 2, "publication-only records may remain after runtime completion")
	require.True(t, kl.podWorkers.ShouldPodRuntimeBeRemoved(pod.UID))
	require.True(t, kl.podWorkers.ShouldPodContentBeRemoved(pod.UID))
	require.False(t, kl.podWorkers.CouldHaveRunningContainers(pod.UID))
}

func TestCheckpointOperationRetainsResourcesForInvalidRecords(t *testing.T) {
	for _, kind := range []string{"corrupt", "not a directory"} {
		t.Run(kind, func(t *testing.T) {
			ctx := ktesting.Init(t)
			kl, _, pod, pc := checkpointOperationFixture(t)
			base := &podWorkers{}
			base.SyncKnownPods(ctx.Logger(), nil)
			workers := &podOperationAwarePodWorkers{PodWorkers: base, kubelet: kl}
			operation := podCheckpointOperation{CheckpointUID: pc.UID, PodUID: pod.UID}
			require.NoError(t, kl.writePodCheckpointOperation(operation))
			dir, key, err := kl.podCheckpointOperationPath(pod.UID, pc.UID)
			require.NoError(t, err)
			switch kind {
			case "corrupt":
				require.NoError(t, os.WriteFile(filepath.Join(dir, key), []byte("invalid JSON"), 0600))
			case "not a directory":
				require.NoError(t, os.RemoveAll(dir))
				require.NoError(t, os.WriteFile(dir, []byte("invalid operation directory"), 0600))
			}
			require.False(t, workers.ShouldPodRuntimeBeRemoved(pod.UID))
			require.False(t, workers.ShouldPodContentBeRemoved(pod.UID))
			require.True(t, workers.CouldHaveRunningContainers(pod.UID))
			require.True(t, workers.ShouldPodRuntimeBeRemoved("unrelated-pod"))
			require.True(t, workers.ShouldPodContentBeRemoved("unrelated-pod"))
			require.False(t, workers.CouldHaveRunningContainers("unrelated-pod"))
		})
	}
}

func TestCheckpointOperationRequiresRuntimeCapability(t *testing.T) {
	for _, known := range []bool{false, true} {
		t.Run(map[bool]string{false: "status not observed", true: "capability unsupported"}[known], func(t *testing.T) {
			var ctx context.Context = ktesting.Init(t)
			kl, runtime, pod, pc := checkpointOperationFixture(t)
			var capability *kubecontainer.RuntimeFeatures
			if known {
				capability = &kubecontainer.RuntimeFeatures{}
			}
			kl.runtimeState.setRuntimeFeatures(capability)
			calls := 0
			kl.containerRuntime = &checkpointRuntime{Runtime: runtime, checkpointPod: func(context.Context, *runtimeapi.CheckpointPodRequest) error { calls++; return nil }}
			require.NoError(t, checkpointPod(ctx, kl, pod, pc.Name, pc.UID, 0))
			require.Zero(t, calls)
			operations, err := kl.readPodCheckpointOperations(pod.UID)
			require.NoError(t, err)
			require.Empty(t, operations)
			if known {
				require.Eventually(t, func() bool { return !kl.podCheckpointInFlight(pod.UID) }, 5*time.Second, time.Millisecond)
			} else {
				_, pending := kl.podCheckpointRequests.Load(pod.UID)
				require.True(t, pending)
			}
		})
	}
}

func TestCheckpointOperationPublishesCompletedDataAfterRestart(t *testing.T) {
	var ctx context.Context = ktesting.Init(t)
	kl, runtime, pod, pc := checkpointOperationFixture(t)
	kl.checkpointOperationContext.Store(&ctx)
	root, output, err := kl.podCheckpointOutputPath(pc.UID)
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(root, 0700))
	require.NoError(t, os.Mkdir(output, 0700))
	require.NoError(t, os.WriteFile(filepath.Join(output, "saved"), []byte("checkpoint"), 0600))
	require.NoError(t, kl.writePodCheckpointOperation(podCheckpointOperation{CheckpointUID: pc.UID, PodUID: pod.UID, PodName: pod.Name, Namespace: pc.Namespace, Name: pc.Name, Complete: true, Succeeded: true, Location: filepath.Base(output)}))
	runtime.CancelPodOperationFunc = func(context.Context, *runtimeapi.CancelPodOperationRequest) error {
		t.Error("confirmed successful capture must not be canceled after restart")
		return nil
	}
	require.NoError(t, kl.reconcilePodCheckpointOperation(ctx, pod))
	require.Eventually(t, func() bool {
		records, err := kl.readPodCheckpointOperations(pod.UID)
		return err == nil && len(records) == 0
	}, 5*time.Second, time.Millisecond)
	require.FileExists(t, filepath.Join(output, "saved"))
	stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(ctx, pc.Name, metav1.GetOptions{})
	require.NoError(t, err)
	require.Equal(t, metav1.ConditionTrue, apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady).Status)
	require.Empty(t, runtime.CanceledPodOperations)
}

func TestCheckpointRecoveryKeepsQueuedCaptureBehindProbeOwner(t *testing.T) {
	ctx := ktesting.Init(t)
	kl, runtime, pod, _ := checkpointOperationFixture(t)
	kl.kubeClient = nil
	probes := &checkpointProbeManager{resumed: make(chan types.UID, 8), suspend: func(context.Context, types.UID) error { return nil }}
	kl.probeManager = probes
	require.NoError(t, kl.writePodCheckpointOperation(podCheckpointOperation{CheckpointUID: "interrupted", PodUID: pod.UID, PodName: pod.Name, Namespace: pod.Namespace, Name: "old"}))
	settling, acknowledged := make(chan struct{}), make(chan struct{})
	runtime.CancelPodOperationFunc = func(context.Context, *runtimeapi.CancelPodOperationRequest) error {
		close(settling)
		<-acknowledged
		return nil
	}
	done := make(chan error, 1)
	go func() { done <- kl.reconcilePodCheckpointOperations(ctx, pod, false) }()
	defer func() {
		close(acknowledged)
		require.NoError(t, <-done)
	}()
	select {
	case <-settling:
	case <-time.After(5 * time.Second):
		t.Fatal("checkpoint recovery did not reach runtime settlement")
	}
	request := &podCheckpointRequest{ctx: ctx, namespace: pod.Namespace, name: "next", uid: "next-checkpoint"}
	kl.podCheckpointRequests.Store(pod.UID, request)
	require.ErrorIs(t, kl.syncPodCheckpointRequest(ctx, pod, &runtime.PodStatus), errPodCheckpointCompletionPending)
	queued, ok := kl.podCheckpointRequests.Load(pod.UID)
	require.True(t, ok, "recovery must not consume the next capture request")
	require.Same(t, request, queued)
	require.Empty(t, probes.resumed, "probes must remain suspended while runtime completion is pending")
	unrelated := pod.DeepCopy()
	unrelated.UID = "unrelated"
	require.NoError(t, kl.syncPodCheckpointRequest(ctx, unrelated, &runtime.PodStatus))
}

func TestCheckpointPublicationCannotReleaseNewCaptureOwnership(t *testing.T) {
	kl, _, pod, _ := checkpointOperationFixture(t)
	kl.checkpointsInFlight.Store(pod.UID, types.UID("new-checkpoint"))
	kl.finishPodCheckpoint(pod.UID, "old-checkpoint")
	require.True(t, kl.podCheckpointInFlight(pod.UID), "publishing an older outcome must preserve a queued capture")
	kl.finishPodCheckpoint(pod.UID, "new-checkpoint")
	require.False(t, kl.podCheckpointInFlight(pod.UID))
}
