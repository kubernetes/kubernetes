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
	"testing"

	"github.com/stretchr/testify/require"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/ktesting"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
)

func TestRestoreReservationLostReplies(t *testing.T) {
	ctx := ktesting.Init(t)
	kl, _, pod, pc := checkpointOperationFixture(t)
	pc.Finalizers = []string{nodeapi.PodCheckpointRestoreProtectionFinalizer}
	pc.Status.NodeName = new(string(kl.nodeName))
	pc.Status.Conditions = []metav1.Condition{{Type: "Ready", Status: metav1.ConditionTrue}}
	client := fake.NewClientset(pc)
	kl.kubeClient = client
	writes := 0
	client.PrependReactor("update", "podcheckpoints", func(a clienttesting.Action) (bool, apiruntime.Object, error) {
		writes++
		record, err := kl.readPodRestoreOperation(pod.UID)
		require.NoError(t, err)
		require.True(t, record.ReservationPending, "durable ownership must precede the API write")
		if writes == 1 || writes == 2 {
			require.NoError(t, client.Tracker().Update(nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints"), a.(clienttesting.UpdateAction).GetObject(), pc.Namespace))
			return true, nil, context.DeadlineExceeded
		}
		return false, nil, nil
	})
	require.ErrorIs(t, kl.reserveRestoreCheckpoint(ctx, pod, pc), context.DeadlineExceeded)
	current, err := client.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(ctx, pc.Name, metav1.GetOptions{})
	require.NoError(t, err)
	require.Len(t, current.Status.ActiveRestores, 1)
	current.DeletionTimestamp = new(metav1.Now())
	require.NoError(t, kl.reserveRestoreCheckpoint(ctx, pod, current), "an existing consumer is recoverable during deletion")
	require.Equal(t, 1, writes, "a lost response must not create another consumer")
	record, err := kl.readPodRestoreOperation(pod.UID)
	require.NoError(t, err)
	require.Error(t, kl.releaseRestoreCheckpoint(ctx, record), "runtime uncertainty cannot release ownership")
	require.NoError(t, kl.SetPodRestoreState(pod.UID, kubecontainer.PodRestoreCleanupCompleted))
	record, err = kl.readPodRestoreOperation(pod.UID)
	require.NoError(t, err)
	require.ErrorIs(t, kl.releaseRestoreCheckpoint(ctx, record), context.DeadlineExceeded)
	record, err = kl.readPodRestoreOperation(pod.UID)
	require.NoError(t, err)
	require.True(t, record.ReservationPending, "lost release reply must retain the recovery record")
	require.NoError(t, kl.releaseRestoreCheckpoint(ctx, record))
	record, err = kl.readPodRestoreOperation(pod.UID)
	require.NoError(t, err)
	require.False(t, record.ReservationPending)
	require.Equal(t, 3, writes)
}

func TestRestoreReservationRejectsNewConsumers(t *testing.T) {
	for _, tc := range []struct {
		name       string
		deleting   bool
		full       bool
		wantReason string
	}{
		{name: "deleting", deleting: true, wantReason: events.RestoreFailed},
		{name: "at capacity", full: true, wantReason: events.RestoreInProgress},
	} {
		t.Run(tc.name, func(t *testing.T) {
			kl, _, pod, pc := checkpointOperationFixture(t)
			pc.Finalizers = []string{nodeapi.PodCheckpointRestoreProtectionFinalizer}
			pc.Status.NodeName = new(string(kl.nodeName))
			pc.Status.Conditions = []metav1.Condition{{Type: "Ready", Status: metav1.ConditionTrue}}
			if tc.deleting {
				pc.DeletionTimestamp = new(metav1.Now())
			}
			if tc.full {
				for i := range nodev1alpha1.PodCheckpointActiveRestoresMaxSize {
					pc.Status.ActiveRestores = append(pc.Status.ActiveRestores, nodev1alpha1.PodCheckpointConsumerReference{Name: fmt.Sprintf("restore-%d", i), UID: types.UID(fmt.Sprintf("uid-%d", i))})
				}
			}
			before := pc.DeepCopy()
			err := kl.reserveRestoreCheckpoint(context.Background(), pod, pc)
			require.Equal(t, tc.wantReason, kubecontainer.RestoreErrorReason(err))
			require.Equal(t, before, pc, "capacity checks must not evict active readers")
			record, err := kl.readPodRestoreOperation(pod.UID)
			require.NoError(t, err)
			require.False(t, record.ReservationPending)
		})
	}
}

func TestRestoreReservationRecoveryAfterPodRemoval(t *testing.T) {
	for _, state := range []kubecontainer.PodRestoreState{kubecontainer.PodRestoreReserved, kubecontainer.PodRestoreCleanupPending, kubecontainer.PodRestoreAdopted} {
		t.Run(string(state), func(t *testing.T) {
			ctx := ktesting.Init(t)
			kl, runtime, pod, pc := checkpointOperationFixture(t)
			pc.Finalizers = []string{nodeapi.PodCheckpointRestoreProtectionFinalizer}
			pc.Status.NodeName = new(string(kl.nodeName))
			pc.Status.Conditions = []metav1.Condition{{Type: "Ready", Status: metav1.ConditionTrue}}
			client := fake.NewClientset(pc)
			kl.kubeClient = client
			require.NoError(t, kl.reserveRestoreCheckpoint(ctx, pod, pc))
			if state != kubecontainer.PodRestoreReserved {
				require.NoError(t, kl.SetPodRestoreState(pod.UID, state))
			}
			kl.podManager.SetPods(nil)
			runtime.PodList = nil
			calls := 0
			runtime.CancelPodOperationFunc = func(context.Context, *runtimeapi.CancelPodOperationRequest) error {
				calls++
				return errors.New("runtime still reads data")
			}
			kl.recoverOrphanedPodRestores(ctx)
			record, err := kl.readPodRestoreOperation(pod.UID)
			require.NoError(t, err)
			require.Equal(t, state != kubecontainer.PodRestoreAdopted, record.ReservationPending, "runtime cleanup and API release are separate handoffs")
			if state == kubecontainer.PodRestoreCleanupPending {
				require.Equal(t, 1, calls)
				runtime.CancelPodOperationFunc = func(context.Context, *runtimeapi.CancelPodOperationRequest) error { return nil }
				kl.restoreRecoveryBackOff.Reset(string(pod.UID))
				kl.recoverOrphanedPodRestores(ctx)
			} else {
				require.Zero(t, calls, "never-started and adopted operations do not call runtime cancellation")
			}
			kl.recoverOrphanedPodRestores(ctx)
			record, err = kl.readPodRestoreOperation(pod.UID)
			require.NoError(t, err)
			require.False(t, record.ReservationPending)
			stored, err := client.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(ctx, pc.Name, metav1.GetOptions{})
			require.NoError(t, err)
			require.Empty(t, stored.Status.ActiveRestores)
		})
	}
}

func TestRestoreReservationSurvivesAPIRollback(t *testing.T) {
	ctx := ktesting.Init(t)
	kl, _, pod, pc := checkpointOperationFixture(t)
	pc.Finalizers = []string{nodeapi.PodCheckpointRestoreProtectionFinalizer}
	pc.Status.NodeName = new(string(kl.nodeName))
	pc.Status.Conditions = []metav1.Condition{{Type: "Ready", Status: metav1.ConditionTrue}}
	client := fake.NewClientset(pc)
	kl.kubeClient = client
	require.NoError(t, kl.reserveRestoreCheckpoint(ctx, pod, pc))
	require.NoError(t, kl.SetPodRestoreState(pod.UID, kubecontainer.PodRestoreCleanupCompleted))
	client.PrependReactor("get", "podcheckpoints", func(clienttesting.Action) (bool, apiruntime.Object, error) {
		return true, nil, apierrors.NewNotFound(nodev1alpha1.Resource("podcheckpoints"), pc.Name)
	})
	client.PrependReactor("list", "podcheckpoints", func(clienttesting.Action) (bool, apiruntime.Object, error) {
		return true, nil, apierrors.NewNotFound(nodev1alpha1.Resource("podcheckpoints"), "")
	})
	record, err := kl.readPodRestoreOperation(pod.UID)
	require.NoError(t, err)
	require.Error(t, kl.releaseRestoreCheckpoint(ctx, record), "an unavailable API is not proof that the checkpoint was removed")
	record, err = kl.readPodRestoreOperation(pod.UID)
	require.NoError(t, err)
	require.True(t, record.ReservationPending)
	client.ReactionChain = client.ReactionChain[2:]
	require.NoError(t, kl.releaseRestoreCheckpoint(ctx, record))
	record, err = kl.readPodRestoreOperation(pod.UID)
	require.NoError(t, err)
	require.False(t, record.ReservationPending)
}

func TestAbandonedRestoreReservationReleasesLocalHandoff(t *testing.T) {
	ctx := ktesting.Init(t)
	kl, _, pod, pc := checkpointOperationFixture(t)
	pc.Finalizers = []string{nodeapi.PodCheckpointRestoreProtectionFinalizer}
	pc.Status.NodeName = new(string(kl.nodeName))
	pc.Status.Conditions = []metav1.Condition{{Type: "Ready", Status: metav1.ConditionTrue}}
	kl.kubeClient = fake.NewClientset(pc)
	require.NoError(t, kl.reserveRestoreCheckpoint(ctx, pod, pc))
	// Path resolution succeeded but preparation failed before the runtime start.
	kl.restoreCheckpointUIDs.Store(pod.UID, pc.UID)
	reason, err := kl.settlePodRestore(ctx, pod, nil)
	require.NoError(t, err)
	require.Equal(t, nodeapi.PodRestoreReasonNotStarted, reason)
	_, pinned := kl.restoreCheckpointUIDs.Load(pod.UID)
	require.False(t, pinned, "a failed never-started Pod must not retain the handoff forever")
	inUse := sets.New[types.UID]()
	require.NoError(t, kl.collectRestoreCheckpointUIDs(inUse))
	require.True(t, inUse.Has(pc.UID), "durable release intent retains ownership until the API acknowledges it")
	record, err := kl.readPodRestoreOperation(pod.UID)
	require.NoError(t, err)
	require.NoError(t, kl.releaseRestoreCheckpoint(ctx, record))
	inUse = sets.New[types.UID]()
	require.NoError(t, kl.collectRestoreCheckpointUIDs(inUse))
	require.False(t, inUse.Has(pc.UID), "confirmed release must permit eventual archive collection")
}
