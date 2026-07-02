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
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	testingclock "k8s.io/utils/clock/testing"
)

func TestRestoreOperationRejectsInvalidRecords(t *testing.T) {
	for _, data := range []string{
		"", "invalid JSON", "{}",
		`{"version":2,"operationID":"test-pod","state":"started"}`,
		`{"version":1,"operationID":"wrong-pod","state":"started"}`,
		`{"version":1,"operationID":"test-pod","state":""}`,
		`{"version":1,"operationID":"test-pod","state":"legacy"}`,
		`{"version":1,"operationID":"test-pod","state":"started","containerIDs":[""]}`,
		`{"version":1,"operationID":"test-pod","state":"started","containerIDs":["same","same"]}`,
	} {
		t.Run(data, func(t *testing.T) {
			ctx := ktesting.Init(t)
			kl, runtime, pod, _ := checkpointOperationFixture(t)
			pod.UID = "test-pod"
			require.NoError(t, os.MkdirAll(kl.getPodDir(pod.UID), 0700))
			path := filepath.Join(kl.getPodDir(pod.UID), podRestoreStartedKey)
			require.NoError(t, os.WriteFile(path, []byte(data), 0600))
			_, err := kl.ReadPodRestoreState(pod.UID)
			require.Error(t, err)
			require.True(t, kl.podOperationMayUsePodResources(pod.UID))
			_, err = kl.settlePodRestore(ctx, pod, nil)
			require.Error(t, err)
			require.Empty(t, runtime.CanceledPodOperations)
			require.Error(t, kl.SetPodRestoreState(pod.UID, kubecontainer.PodRestoreCleanupCompleted))
			after, err := os.ReadFile(path)
			require.NoError(t, err)
			require.Equal(t, []byte(data), after, "corrupt state must not be replaced with a completed claim")
		})
	}
}

func TestRestoreOperationCleanupCannotBeReplacedBySuccess(t *testing.T) {
	for _, from := range []kubecontainer.PodRestoreState{kubecontainer.PodRestoreCleanupPending, kubecontainer.PodRestoreCleanupCompleted} {
		for _, to := range []kubecontainer.PodRestoreState{kubecontainer.PodRestoreStarted, kubecontainer.PodRestoreAdopted} {
			t.Run(fmt.Sprintf("%s to %s", from, to), func(t *testing.T) {
				kl, _, pod, _ := checkpointOperationFixture(t)
				require.NoError(t, kl.SetPodRestoreState(pod.UID, from))
				require.Error(t, kl.SetPodRestoreState(pod.UID, to))
				state, err := kl.ReadPodRestoreState(pod.UID)
				require.NoError(t, err)
				require.Equal(t, from, state)
			})
		}
	}

	kl, _, pod, _ := checkpointOperationFixture(t)
	require.NoError(t, kl.MarkPodRestoreStarted(pod.UID))
	require.NoError(t, kl.SetPodRestoreState(pod.UID, kubecontainer.PodRestoreCleanupPending))
	// The runtime manager and the kubelet both save the confirmed cleanup.
	require.NoError(t, kl.SetPodRestoreState(pod.UID, kubecontainer.PodRestoreCleanupCompleted))
	require.NoError(t, kl.SetPodRestoreState(pod.UID, kubecontainer.PodRestoreCleanupCompleted))
}

func TestRestoreOperationCleanupIdentitiesSurviveRestart(t *testing.T) {
	ctx := ktesting.Init(t)
	kl, _, pod, _ := checkpointOperationFixture(t)
	require.NoError(t, kl.MarkPodRestoreStarted(pod.UID))
	require.NoError(t, kl.RecordPodRestoreContainers(pod.UID, []string{"second", "first"}))
	require.NoError(t, kl.recordPodRestoreCleanup(pod.UID, []string{"first", "third"}))
	before, err := kl.ReadPodRestoreContainers(pod.UID)
	require.NoError(t, err)
	require.Equal(t, []string{"first", "second", "third"}, before)

	tk := newTestKubelet(t, false)
	defer tk.Cleanup()
	restarted := tk.kubelet
	restarted.rootDirectory = kl.rootDirectory
	calls := 0
	tk.fakeRuntime.CancelPodOperationFunc = func(_ context.Context, request *runtimeapi.CancelPodOperationRequest) error {
		calls++
		require.Equal(t, string(pod.UID), request.OperationId)
		ids, err := restarted.ReadPodRestoreContainers(pod.UID)
		require.NoError(t, err)
		require.Equal(t, before, ids, "runtime metadata loss must not lose kubelet cleanup identities")
		if calls == 1 {
			return errors.New("host cleanup is still pending")
		}
		return nil
	}
	_, err = restarted.settlePodRestore(ctx, pod, &kubecontainer.PodStatus{ID: pod.UID})
	require.ErrorContains(t, err, "host cleanup is still pending")
	state, err := restarted.ReadPodRestoreState(pod.UID)
	require.NoError(t, err)
	require.Equal(t, kubecontainer.PodRestoreCleanupPending, state)
	require.True(t, restarted.podOperationMayUsePodResources(pod.UID))
	_, err = restarted.settlePodRestore(ctx, pod, nil)
	require.NoError(t, err)
	state, err = restarted.ReadPodRestoreState(pod.UID)
	require.NoError(t, err)
	require.Equal(t, kubecontainer.PodRestoreCleanupCompleted, state)
	require.False(t, restarted.podOperationMayUsePodResources(pod.UID))
	after, err := restarted.ReadPodRestoreContainers(pod.UID)
	require.NoError(t, err)
	require.Equal(t, before, after)
	require.Empty(t, tk.fakeRuntime.StartedPods)
	_, err = restarted.settlePodRestore(ctx, pod, nil)
	require.NoError(t, err)
	require.Equal(t, 2, calls, "a completed cleanup must not call the runtime again")
}

func TestRestoreOperationRetainsUndiscoveredResources(t *testing.T) {
	for _, tc := range []struct {
		name    string
		state   kubecontainer.PodRestoreState
		corrupt bool
		retain  bool
	}{
		{name: "no operation"},
		{name: "started", state: kubecontainer.PodRestoreStarted, retain: true},
		{name: "cleanup pending", state: kubecontainer.PodRestoreCleanupPending, retain: true},
		{name: "cleanup completed", state: kubecontainer.PodRestoreCleanupCompleted},
		{name: "adopted", state: kubecontainer.PodRestoreAdopted},
		{name: "corrupt", state: kubecontainer.PodRestoreStarted, corrupt: true, retain: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := ktesting.Init(t)
			kl, runtime, pod, _ := checkpointOperationFixture(t)
			kl.podManager.SetPods(nil)
			runtime.PodList = nil
			base := &podWorkers{}
			base.SyncKnownPods(ctx.Logger(), nil)
			workers := &podOperationAwarePodWorkers{PodWorkers: base, kubelet: kl}
			require.True(t, base.ShouldPodRuntimeBeRemoved(pod.UID))
			require.True(t, base.ShouldPodContentBeRemoved(pod.UID))
			require.False(t, base.CouldHaveRunningContainers(pod.UID))
			if tc.state != kubecontainer.PodRestoreNotStarted {
				require.NoError(t, kl.SetPodRestoreState(pod.UID, tc.state))
			}
			if tc.corrupt {
				require.NoError(t, os.WriteFile(filepath.Join(kl.getPodDir(pod.UID), podRestoreStartedKey), []byte("invalid JSON"), 0600))
			}
			require.Equal(t, !tc.retain, workers.ShouldPodRuntimeBeRemoved(pod.UID))
			require.Equal(t, !tc.retain, workers.ShouldPodContentBeRemoved(pod.UID))
			require.Equal(t, tc.retain, workers.CouldHaveRunningContainers(pod.UID))
			require.True(t, workers.ShouldPodRuntimeBeRemoved("unrelated-pod"))
			require.True(t, workers.ShouldPodContentBeRemoved("unrelated-pod"))
			require.False(t, workers.CouldHaveRunningContainers("unrelated-pod"))
			if tc.corrupt {
				require.Error(t, kl.SetPodRestoreState(pod.UID, kubecontainer.PodRestoreCleanupCompleted))
				return
			}
			if tc.retain {
				require.NoError(t, kl.SetPodRestoreState(pod.UID, kubecontainer.PodRestoreCleanupCompleted))
				require.True(t, workers.ShouldPodRuntimeBeRemoved(pod.UID))
				require.True(t, workers.ShouldPodContentBeRemoved(pod.UID))
				require.False(t, workers.CouldHaveRunningContainers(pod.UID))
			}
		})
	}
}

func TestRestoreOrphanRecoveryWaitsForCleanupAcknowledgement(t *testing.T) {
	ctx := ktesting.Init(t)
	kl, runtime, pod, _ := checkpointOperationFixture(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, false)
	kl.podManager.SetPods(nil)
	runtime.PodList = nil
	require.NoError(t, kl.SetPodRestoreState(pod.UID, kubecontainer.PodRestoreStarted))
	acknowledged, calls := false, 0
	runtime.CancelPodOperationFunc = func(cancelCtx context.Context, request *runtimeapi.CancelPodOperationRequest) error {
		calls++
		require.NoError(t, cancelCtx.Err())
		deadline, ok := cancelCtx.Deadline()
		require.True(t, ok)
		require.LessOrEqual(t, time.Until(deadline), podCheckpointAPITimeout)
		require.Equal(t, string(pod.UID), request.OperationId)
		require.Equal(t, runtimeapi.PodOperationType_POD_RESTORE, request.OperationType)
		if !acknowledged {
			return errors.New("restore reader still active")
		}
		return nil
	}
	clock := testingclock.NewFakeClock(time.Now())
	kl.clock = clock
	kl.restoreRecoveryBackOff.Clock = clock
	kl.recoverOrphanedPodRestores(ctx)
	state, err := kl.ReadPodRestoreState(pod.UID)
	require.NoError(t, err)
	require.Equal(t, kubecontainer.PodRestoreCleanupPending, state)
	require.True(t, kl.podOperationMayUsePodResources(pod.UID))
	require.Equal(t, 1, calls)
	acknowledged = true
	kl.recoverOrphanedPodRestores(ctx)
	require.Equal(t, 1, calls, "a failed cleanup waits for its backoff")
	clock.Step(time.Second)
	kl.recoverOrphanedPodRestores(ctx)
	state, err = kl.ReadPodRestoreState(pod.UID)
	require.NoError(t, err)
	require.Equal(t, kubecontainer.PodRestoreCleanupCompleted, state)
	require.False(t, kl.podOperationMayUsePodResources(pod.UID))
	require.Equal(t, 2, calls)
	kl.recoverOrphanedPodRestores(ctx)
	require.Equal(t, 2, calls, "settled restore must not be replayed or canceled again")
}

func TestRestoreOrphanRecoveryPreservesOtherOwners(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		state                 kubecontainer.PodRestoreState
		known, owned, corrupt bool
	}{
		{name: "known Pod", state: kubecontainer.PodRestoreStarted, known: true},
		{name: "active worker", state: kubecontainer.PodRestoreStarted, owned: true},
		{name: "adopted", state: kubecontainer.PodRestoreAdopted},
		{name: "completed", state: kubecontainer.PodRestoreCleanupCompleted},
		{name: "corrupt", state: kubecontainer.PodRestoreStarted, corrupt: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := ktesting.Init(t)
			kl, runtime, pod, _ := checkpointOperationFixture(t)
			if !tc.known {
				kl.podManager.SetPods(nil)
			}
			if tc.owned {
				kl.restoreOperationOwners.Store(pod.UID, struct{}{})
			}
			require.NoError(t, kl.SetPodRestoreState(pod.UID, tc.state))
			path := filepath.Join(kl.getPodDir(pod.UID), podRestoreStartedKey)
			if tc.corrupt {
				require.NoError(t, os.WriteFile(path, []byte("invalid JSON"), 0600))
			}
			before, err := os.ReadFile(path)
			require.NoError(t, err)
			calls := 0
			runtime.CancelPodOperationFunc = func(context.Context, *runtimeapi.CancelPodOperationRequest) error { calls++; return nil }
			kl.recoverOrphanedPodRestores(ctx)
			require.Zero(t, calls)
			after, err := os.ReadFile(path)
			require.NoError(t, err)
			require.Equal(t, before, after)
		})
	}
}

func TestRestoreMarkerProtectsPodWithoutRestoreSpec(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		for _, initial := range []kubecontainer.PodRestoreState{kubecontainer.PodRestoreStarted, kubecontainer.PodRestoreCleanupPending} {
			for _, terminating := range []bool{false, true} {
				t.Run(fmt.Sprintf("gate=%t/state=%s/terminating=%t", enabled, initial, terminating), func(t *testing.T) {
					ctx := ktesting.Init(t)
					kl, runtime, pod, _ := checkpointOperationFixture(t)
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, enabled)
					runtime.PodStatus.SandboxStatuses[0].Metadata = &runtimeapi.PodSandboxMetadata{}
					require.Nil(t, pod.Spec.RestoreFrom)
					require.Empty(t, pod.Finalizers)
					require.NoError(t, kl.SetPodRestoreState(pod.UID, initial))
					acknowledged, calls := false, 0
					runtime.CancelPodOperationFunc = func(_ context.Context, request *runtimeapi.CancelPodOperationRequest) error {
						calls++
						require.Equal(t, string(pod.UID), request.OperationId)
						if !acknowledged {
							return errors.New("restore cleanup is still active")
						}
						return nil
					}
					var err error
					if terminating {
						grace := int64(0)
						err = kl.SyncTerminatingPod(ctx, pod, &runtime.PodStatus, &grace, nil)
					} else {
						_, _, err = kl.SyncPod(ctx, kubetypes.SyncPodSync, pod, nil, &runtime.PodStatus)
					}
					require.Error(t, err, "a missing API field must not release durable restore ownership")
					require.Equal(t, 1, calls)
					runtime.Lock()
					called := slices.Clone(runtime.CalledFunctions)
					runtime.Unlock()
					require.NotContains(t, called, "SyncPod")
					require.NotContains(t, called, "KillPod")
					require.True(t, kl.podOperationMayUsePodResources(pod.UID))
					acknowledged = true
					terminal, _, err := kl.SyncPod(ctx, kubetypes.SyncPodSync, pod, nil, &runtime.PodStatus)
					require.NoError(t, err)
					require.True(t, terminal, "an interrupted restore must fail after confirmed cleanup without cold-starting")
					state, err := kl.ReadPodRestoreState(pod.UID)
					require.NoError(t, err)
					require.Equal(t, kubecontainer.PodRestoreCleanupCompleted, state)
					require.False(t, kl.podOperationMayUsePodResources(pod.UID))
				})
			}
		}
	}
}

func TestRestoreKeepsCheckpointDataUntilSettled(t *testing.T) {
	for _, settled := range []kubecontainer.PodRestoreState{kubecontainer.PodRestoreAdopted, kubecontainer.PodRestoreCleanupCompleted} {
		t.Run(string(settled), func(t *testing.T) {
			ctx := ktesting.Init(t)
			kl, _, pod, _ := checkpointOperationFixture(t)
			// The PodCheckpoint was deleted while the restore used its data.
			kl.kubeClient = fake.NewClientset()
			seedCheckpointCache(t, kl)
			checkpointUID := types.UID("deleted-checkpoint")
			_, data, err := kl.podCheckpointOutputPath(checkpointUID)
			require.NoError(t, err)
			require.NoError(t, os.MkdirAll(data, 0o700))

			// ValidateRestoreCheckpoint resolved the checkpoint before RestorePod.
			kl.restoreCheckpointUIDs.Store(pod.UID, checkpointUID)
			require.NoError(t, kl.sweepPodCheckpointData(ctx))
			require.DirExists(t, data)
			require.NoError(t, kl.MarkPodRestoreStarted(pod.UID))
			_, pending := kl.restoreCheckpointUIDs.Load(pod.UID)
			require.False(t, pending, "the restore record takes over from the handoff")
			operation, err := kl.readPodRestoreOperation(pod.UID)
			require.NoError(t, err)
			require.Equal(t, checkpointUID, operation.CheckpointUID)
			require.NoError(t, kl.sweepPodCheckpointData(ctx))
			require.DirExists(t, data)

			if settled == kubecontainer.PodRestoreCleanupCompleted {
				require.NoError(t, kl.SetPodRestoreState(pod.UID, kubecontainer.PodRestoreCleanupPending))
				require.NoError(t, kl.sweepPodCheckpointData(ctx))
				require.DirExists(t, data, "the runtime may still read data until cleanup is confirmed")
			}
			require.NoError(t, kl.SetPodRestoreState(pod.UID, settled))
			operation, err = kl.readPodRestoreOperation(pod.UID)
			require.NoError(t, err)
			require.Equal(t, checkpointUID, operation.CheckpointUID)
			require.NoError(t, kl.sweepPodCheckpointData(ctx))
			require.NoDirExists(t, data)
		})
	}
}

func TestRestoreRecordWithoutCheckpointKeepsAllData(t *testing.T) {
	ctx := ktesting.Init(t)
	kl, _, pod, _ := checkpointOperationFixture(t)
	kl.kubeClient = fake.NewClientset()
	seedCheckpointCache(t, kl)
	_, data, err := kl.podCheckpointOutputPath("deleted-checkpoint")
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(data, 0o700))
	// The restore was saved before the kubelet stored its checkpoint UID.
	require.NoError(t, os.MkdirAll(kl.getPodDir(pod.UID), 0o700))
	record := fmt.Sprintf(`{"version":1,"operationID":%q,"state":"started"}`, pod.UID)
	require.NoError(t, os.WriteFile(filepath.Join(kl.getPodDir(pod.UID), podRestoreStartedKey), []byte(record), 0o600))

	require.NoError(t, kl.sweepPodCheckpointData(ctx))
	require.DirExists(t, data, "the restore may read any checkpoint")
	require.NoError(t, kl.SetPodRestoreState(pod.UID, kubecontainer.PodRestoreCleanupCompleted))
	require.NoError(t, kl.sweepPodCheckpointData(ctx))
	require.NoDirExists(t, data)
}

func TestReleaseRestoreCheckpointGeneration(t *testing.T) {
	for _, tc := range []struct {
		name         string
		generation   *int64
		unregistered bool
		want         int64
		wantError    bool
	}{
		{name: "unset", want: 1},
		{name: "zero", generation: new(int64(0)), want: 1},
		{name: "advanced", generation: new(int64(7)), want: 8},
		{name: "registration response lost", generation: new(int64(7)), unregistered: true, want: 8},
		{name: "exhausted", generation: new(int64(math.MaxInt64)), wantError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := ktesting.Init(t)
			kl, _, pod, _ := checkpointOperationFixture(t)
			other := nodev1alpha1.PodCheckpointConsumerReference{Name: "other", UID: "other-uid"}
			checkpoint := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: pod.Namespace, UID: "checkpoint-uid"},
				Status: nodev1alpha1.PodCheckpointStatus{
					RestoreGeneration: tc.generation,
					ActiveRestores:    []nodev1alpha1.PodCheckpointConsumerReference{other},
				},
			}
			if !tc.unregistered {
				checkpoint.Status.ActiveRestores = append(checkpoint.Status.ActiveRestores, nodev1alpha1.PodCheckpointConsumerReference{Name: pod.Name, UID: pod.UID})
			}
			kl.kubeClient = fake.NewClientset(checkpoint)
			operation := podRestoreOperationRecord{
				Version:             1,
				OperationID:         pod.UID,
				State:               kubecontainer.PodRestoreAdopted,
				CheckpointUID:       checkpoint.UID,
				CheckpointNamespace: checkpoint.Namespace,
				CheckpointName:      checkpoint.Name,
				ReservationPending:  true,
			}
			require.NoError(t, kl.writePodRestoreOperation(operation))
			err := kl.releaseRestoreCheckpoint(ctx, operation)
			if tc.wantError {
				require.ErrorContains(t, err, "restore generation is exhausted")
			} else {
				require.NoError(t, err)
			}
			current, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pod.Namespace).Get(ctx, checkpoint.Name, metav1.GetOptions{})
			require.NoError(t, err)
			recorded, err := kl.readPodRestoreOperation(pod.UID)
			require.NoError(t, err)
			if tc.wantError {
				require.Equal(t, checkpoint.Status, current.Status)
				require.True(t, recorded.ReservationPending)
				return
			}
			require.Equal(t, new(tc.want), current.Status.RestoreGeneration)
			require.Equal(t, []nodev1alpha1.PodCheckpointConsumerReference{other}, current.Status.ActiveRestores)
			require.False(t, recorded.ReservationPending)
			require.NoError(t, kl.releaseRestoreCheckpoint(ctx, recorded))
			afterRetry, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pod.Namespace).Get(ctx, checkpoint.Name, metav1.GetOptions{})
			require.NoError(t, err)
			require.Equal(t, current.Status, afterRetry.Status)
		})
	}
}
