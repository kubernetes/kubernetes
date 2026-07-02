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
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
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
	kl.recoverOrphanedPodRestores(ctx)
	state, err := kl.ReadPodRestoreState(pod.UID)
	require.NoError(t, err)
	require.Equal(t, kubecontainer.PodRestoreCleanupPending, state)
	require.True(t, kl.podOperationMayUsePodResources(pod.UID))
	require.Equal(t, 1, calls)
	acknowledged = true
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
