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
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	grpccodes "google.golang.org/grpc/codes"
	grpcstatus "google.golang.org/grpc/status"
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
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
	testingclock "k8s.io/utils/clock/testing"
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

func TestCheckpointOperationRecoveryBacksOff(t *testing.T) {
	var ctx context.Context = ktesting.Init(t)
	kl, runtime, pod, _ := checkpointOperationFixture(t)
	kl.kubeClient = nil
	kl.checkpointOperationContext.Store(&ctx)
	require.NoError(t, kl.writePodCheckpointOperation(podCheckpointOperation{CheckpointUID: "checkpoint", PodUID: pod.UID, PodName: pod.Name, Namespace: pod.Namespace, Name: "checkpoint"}))
	var calls atomic.Int32
	var confirmed atomic.Bool
	runtime.CancelPodOperationFunc = func(context.Context, *runtimeapi.CancelPodOperationRequest) error {
		calls.Add(1)
		if !confirmed.Load() {
			return errors.New("runtime cleanup failed")
		}
		return nil
	}
	clock := testingclock.NewFakeClock(time.Now())
	kl.clock = clock
	kl.checkpointRecoveryBackOff.Clock = clock

	kl.recoverPodCheckpointOperations(ctx)
	require.EqualValues(t, 1, calls.Load())
	kl.recoverPodCheckpointOperations(ctx)
	require.EqualValues(t, 1, calls.Load(), "a failed attempt waits for its backoff")
	clock.Step(time.Second)
	kl.recoverPodCheckpointOperations(ctx)
	require.EqualValues(t, 2, calls.Load())
	clock.Step(time.Second)
	kl.recoverPodCheckpointOperations(ctx)
	require.EqualValues(t, 2, calls.Load(), "the delay grows after each failure")

	confirmed.Store(true)
	clock.Step(time.Second)
	kl.recoverPodCheckpointOperations(ctx)
	require.EqualValues(t, 3, calls.Load())
	operations, err := kl.readPodCheckpointOperations(pod.UID)
	require.NoError(t, err)
	require.Len(t, operations, 1)
	require.True(t, operations[0].Complete, "confirmed cleanup completes the operation")
	require.False(t, kl.checkpointRecoveryBackOff.IsInBackOffSinceUpdate(string(pod.UID), clock.Now()), "success clears the backoff")
}

func TestPodOperationWorkersRetainActivePodWithoutJournalLookup(t *testing.T) {
	uid := types.UID("active-pod")
	base := &podWorkers{
		podsSynced:      true,
		podSyncStatuses: map[types.UID]*podSyncStatus{uid: {}},
	}
	// Active Pods already retain resources; no Kubelet journal reader is needed.
	workers := &podOperationAwarePodWorkers{PodWorkers: base}
	require.False(t, workers.ShouldPodRuntimeBeRemoved(uid))
	require.False(t, workers.ShouldPodContentBeRemoved(uid))
	require.True(t, workers.CouldHaveRunningContainers(uid))
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

// A runtime without CheckpointPod starts no work, so the checkpoint fails
// without waiting for CancelPodOperation.
func TestCheckpointOperationUnimplementedSkipsCancellation(t *testing.T) {
	var ctx context.Context = ktesting.Init(t)
	kl, runtime, pod, pc := checkpointOperationFixture(t)
	runtime.CancelPodOperationFunc = func(context.Context, *runtimeapi.CancelPodOperationRequest) error {
		t.Error("CancelPodOperation must not be called after Unimplemented")
		return nil
	}
	outputs := make(chan string, 1)
	kl.containerRuntime = &checkpointRuntime{Runtime: runtime, checkpointPod: func(_ context.Context, request *runtimeapi.CheckpointPodRequest) error {
		outputs <- request.OutputPath
		return grpcstatus.Error(grpccodes.Unimplemented, "unknown method CheckpointPod")
	}}
	require.NoError(t, checkpointPod(ctx, kl, pod, pc.Name, pc.UID, 0))
	output := <-outputs
	require.Eventually(t, func() bool { return !kl.podCheckpointInFlight(pod.UID) }, 5*time.Second, time.Millisecond)
	current, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(ctx, pc.Name, metav1.GetOptions{})
	require.NoError(t, err)
	ready := apimeta.FindStatusCondition(current.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
	require.NotNil(t, ready)
	require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, ready.Reason)
	require.Contains(t, ready.Message, "runtime does not support Pod checkpoints")
	require.NoDirExists(t, output)
}

func TestCheckpointOperationHoldsProbesUntilAcknowledgement(t *testing.T) {
	var ctx context.Context = ktesting.Init(t)
	kl, runtime, pod, pc := checkpointOperationFixture(t)
	kl.checkpointOperationContext.Store(&ctx)
	kl.podWorkers.(*checkpointPodWorkers).async = true
	probes := &checkpointProbeManager{resumed: make(chan types.UID, 8), suspend: func(context.Context, types.UID) error { return nil }}
	kl.probeManager = probes
	entered := make(chan struct{})
	release := make(chan struct{})
	var cancelCalls atomic.Int32
	runtime.CancelPodOperationFunc = func(cancelCtx context.Context, _ *runtimeapi.CancelPodOperationRequest) error {
		if cancelCalls.Add(1) == 1 {
			close(entered)
		}
		select {
		case <-release:
			return nil
		case <-cancelCtx.Done():
			return cancelCtx.Err()
		}
	}
	defer close(release)
	kl.containerRuntime = &checkpointRuntime{Runtime: runtime, checkpointPod: func(_ context.Context, request *runtimeapi.CheckpointPodRequest) error {
		require.NoError(t, os.WriteFile(filepath.Join(request.OutputPath, "partial"), []byte("runtime owns output"), 0600))
		return context.DeadlineExceeded
	}}
	require.NoError(t, checkpointPod(ctx, kl, pod, pc.Name, pc.UID, 0))
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("completion request did not reach runtime")
	}
	_, output, err := kl.podCheckpointOutputPath(pc.UID)
	require.NoError(t, err)
	require.FileExists(t, filepath.Join(output, "partial"))
	require.Empty(t, probes.resumed)
	// The orphan worker must not become a second owner of an active capture.
	kl.recoverPodCheckpointOperations(ctx)
	require.EqualValues(t, 1, cancelCalls.Load())
	release <- struct{}{}
	require.Eventually(t, func() bool { return !kl.podCheckpointInFlight(pod.UID) }, 5*time.Second, time.Millisecond)
	require.NotEmpty(t, probes.resumed)
	require.NoDirExists(t, output)
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

type checkpointFullFilesystem struct {
	utilfs.DefaultFs
	full atomic.Bool
}

func (fs *checkpointFullFilesystem) TempFile(dir, prefix string) (utilfs.File, error) {
	if fs.full.Load() {
		return nil, &os.PathError{Op: "create", Path: dir, Err: syscall.ENOSPC}
	}
	return fs.DefaultFs.TempFile(dir, prefix)
}

func TestCheckpointCleanupProgressesWhenCompletionStoreIsFull(t *testing.T) {
	for _, recovering := range []bool{false, true} {
		t.Run(map[bool]string{false: "capture", true: "restart recovery"}[recovering], func(t *testing.T) {
			ctx := ktesting.Init(t)
			kl, runtime, pod, pc := checkpointOperationFixture(t)
			fs := &checkpointFullFilesystem{}
			kl.podCheckpointFilesystem = fs
			probes := &checkpointProbeManager{resumed: make(chan types.UID, 8), suspend: func(context.Context, types.UID) error { return nil }}
			kl.probeManager = probes
			_, output, err := kl.createPodCheckpointOutput(pc.UID)
			require.NoError(t, err)
			partial := filepath.Join(output, "partial")
			require.NoError(t, os.WriteFile(partial, []byte("incomplete archive"), 0600))
			operation := podCheckpointOperation{CheckpointUID: pc.UID, PodUID: pod.UID, PodName: pod.Name, Namespace: pc.Namespace, Name: pc.Name}
			if recovering {
				require.NoError(t, kl.writePodCheckpointOperation(operation))
				fs.full.Store(true)
				require.ErrorIs(t, kl.reconcilePodCheckpointOperation(ctx, pod), errPodCheckpointCompletionPending)
			} else {
				require.NoError(t, os.RemoveAll(output))
				kl.containerRuntime = &checkpointRuntime{Runtime: runtime, checkpointPod: func(_ context.Context, request *runtimeapi.CheckpointPodRequest) error {
					require.NoError(t, os.WriteFile(filepath.Join(request.OutputPath, "partial"), []byte("incomplete archive"), 0600))
					fs.full.Store(true)
					return syscall.ENOSPC
				}}
				require.NoError(t, checkpointPod(ctx, kl, pod, pc.Name, pc.UID, 0))
			}
			require.NoDirExists(t, output, "confirmed failure cleanup must not depend on another allocation in the full filesystem")
			require.NotEmpty(t, probes.resumed, "confirmed runtime completion must release probes even while metadata retries")
			records, err := kl.readPodCheckpointOperations(pod.UID)
			require.NoError(t, err)
			require.Len(t, records, 1, "the original durable identity must survive a completion write failure")
			fs.full.Store(false)
			require.NoError(t, kl.reconcilePodCheckpointOperation(ctx, pod))
		})
	}
}
