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
	"math"
	"slices"

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/klog/v2"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
)

// A runtime can lose discoverable resources before its abandoned restore has
// stopped using prepared paths. Recover from the durable marker as well as the
// API/runtime lists, without replaying RestorePod or occupying Pod workers.
func (kl *Kubelet) recoverOrphanedPodRestores(ctx context.Context) {
	uids, err := kl.listPodsFromDisk()
	if err != nil {
		klog.FromContext(ctx).Error(err, "Cannot list restore recovery records")
		return
	}
	kl.restoreRecoveryBackOff.GC()
	for _, uid := range uids {
		if ctx.Err() != nil {
			return
		}
		// A runtime that keeps failing cleanup is retried with a growing delay
		// instead of on every pass.
		if kl.restoreRecoveryBackOff.IsInBackOffSinceUpdate(string(uid), kl.clock.Now()) {
			continue
		}
		if _, owned := kl.restoreOperationOwners.LoadOrStore(uid, struct{}{}); owned {
			continue
		}
		func() {
			defer kl.restoreOperationOwners.Delete(uid)
			operation, err := kl.readPodRestoreOperation(uid)
			if err != nil {
				return
			}
			state := operation.State
			if state == kubecontainer.PodRestoreAdopted || state == kubecontainer.PodRestoreCleanupCompleted {
				if operation.ReservationPending {
					if err := kl.releaseRestoreCheckpoint(ctx, operation); err != nil {
						kl.restoreRecoveryBackOff.Next(string(uid), kl.clock.Now())
						klog.FromContext(ctx).Error(err, "Restore reservation release remains pending", "podUID", uid)
					} else {
						kl.restoreRecoveryBackOff.Reset(string(uid))
					}
				}
				return
			}
			// Active Pod workers own reservation intent and runtime execution.
			if pod, exists := kl.podManager.GetPodByUID(uid); exists && (pod.DeletionTimestamp == nil || !kl.podWorkers.ShouldPodBeFinished(uid)) {
				return
			}
			if state != kubecontainer.PodRestoreReserved && state != kubecontainer.PodRestoreStarted && state != kubecontainer.PodRestoreCleanupPending {
				return
			}
			pod := &v1.Pod{}
			pod.UID = uid
			// Keep each orphan attempt short; later passes retry uncertainty.
			timeout := min(kl.kubeletConfiguration.PodRestoreTimeout.Duration, podCheckpointAPITimeout)
			if _, err := kl.settlePodRestoreWithTimeout(ctx, pod, nil, timeout); err != nil {
				kl.restoreRecoveryBackOff.Next(string(uid), kl.clock.Now())
				klog.FromContext(ctx).Error(err, "Orphaned restore runtime recovery remains pending", "podUID", uid)
				return
			}
			kl.restoreRecoveryBackOff.Reset(string(uid))
		}()
	}
}

// reserveRestoreCheckpoint orders execution against deletion on the checkpoint
// object. Durable intent precedes the write so a lost reply cannot orphan a pin.
func (kl *Kubelet) reserveRestoreCheckpoint(ctx context.Context, pod *v1.Pod, pc *nodev1alpha1.PodCheckpoint) error {
	failed := func(err error) error { return &kubecontainer.RestoreError{Reason: events.RestoreFailed, Err: err} }
	pending := func(err error) error { return &kubecontainer.RestoreError{Reason: events.RestoreInProgress, Err: err} }
	if pc.Status.NodeName == nil || *pc.Status.NodeName != string(kl.nodeName) {
		return &kubecontainer.RestoreError{Reason: events.CheckpointWrongNode, Err: fmt.Errorf("PodCheckpoint %q is not assigned to this node", pc.Name)}
	}
	if !apimeta.IsStatusConditionTrue(pc.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady) {
		return &kubecontainer.RestoreError{Reason: events.CheckpointNotReady, Err: fmt.Errorf("PodCheckpoint %q is not ready", pc.Name)}
	}
	consumer := nodev1alpha1.PodCheckpointConsumerReference{Name: pod.Name, UID: pod.UID}
	reserved := slices.Contains(pc.Status.ActiveRestores, consumer)
	if !reserved && (pc.DeletionTimestamp != nil || !slices.Contains(pc.Finalizers, nodeapi.PodCheckpointRestoreProtectionFinalizer)) {
		return failed(fmt.Errorf("PodCheckpoint %q is being deleted or lacks restore protection", pc.Name))
	}
	if !reserved && len(pc.Status.ActiveRestores) >= nodev1alpha1.PodCheckpointActiveRestoresMaxSize {
		return pending(fmt.Errorf("PodCheckpoint %q has reached its active restore limit", pc.Name))
	}
	operation, err := kl.readPodRestoreOperation(pod.UID)
	if err != nil {
		return failed(err)
	}
	if operation.State != kubecontainer.PodRestoreNotStarted && operation.State != kubecontainer.PodRestoreReserved {
		return failed(fmt.Errorf("restore operation %q already reached state %q", pod.UID, operation.State))
	}
	if operation.CheckpointUID != "" && operation.CheckpointUID != pc.UID {
		return failed(fmt.Errorf("restore operation %q already reserved checkpoint UID %q", pod.UID, operation.CheckpointUID))
	}
	operation.State = kubecontainer.PodRestoreReserved
	operation.CheckpointUID = pc.UID
	operation.CheckpointNamespace = pc.Namespace
	operation.CheckpointName = pc.Name
	operation.ReservationPending = true
	kl.checkpointDataLock.Lock()
	err = kl.writePodRestoreOperation(operation)
	kl.checkpointDataLock.Unlock()
	if err != nil {
		return pending(err)
	}
	if reserved {
		return nil
	}
	pc = pc.DeepCopy()
	pc.Status.ActiveRestores = append(pc.Status.ActiveRestores, consumer)
	if _, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).UpdateStatus(ctx, pc, metav1.UpdateOptions{}); err != nil {
		// Return to the Pod worker. A subsequent live read reconciles an update
		// whose response was lost; it never repeats RestorePod.
		return pending(fmt.Errorf("reserve PodCheckpoint %q for restore: %w", pc.Name, err))
	}
	return nil
}

// releaseRestoreCheckpoint runs under restoreOperationOwners, including after
// the Pod has disappeared. Only confirmed runtime outcomes may release its pin.
func (kl *Kubelet) releaseRestoreCheckpoint(ctx context.Context, operation podRestoreOperationRecord) error {
	if !operation.ReservationPending {
		return nil
	}
	if operation.State != kubecontainer.PodRestoreAdopted && operation.State != kubecontainer.PodRestoreCleanupCompleted {
		return fmt.Errorf("restore operation %q has not settled", operation.OperationID)
	}
	if kl.kubeClient == nil {
		return fmt.Errorf("cannot release restore reservation without an API client")
	}
	ctx, cancel := context.WithTimeout(ctx, podCheckpointAPITimeout)
	defer cancel()
	pc, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(operation.CheckpointNamespace).Get(ctx, operation.CheckpointName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) || apierrors.IsForbidden(err) {
		// A disabled API group also returns 404. A successful node-scoped list
		// distinguishes removal of this UID from API unavailability or skew.
		live, listErr := kl.kubeClient.NodeV1alpha1().PodCheckpoints(operation.CheckpointNamespace).List(ctx, metav1.ListOptions{
			FieldSelector: fields.Set{"metadata.name": operation.CheckpointName, "status.nodeName": string(kl.nodeName)}.AsSelector().String(),
		})
		if listErr != nil {
			return listErr
		}
		pc, err = nil, nil
		for i := range live.Items {
			if live.Items[i].UID == operation.CheckpointUID {
				pc = &live.Items[i]
				break
			}
		}
	}
	if err != nil {
		return err
	}
	if pc != nil && pc.UID == operation.CheckpointUID {
		generation := int64(0)
		if pc.Status.RestoreGeneration != nil {
			generation = *pc.Status.RestoreGeneration
		}
		if generation == math.MaxInt64 {
			return fmt.Errorf("PodCheckpoint %q restore generation is exhausted", pc.Name)
		}
		pc.Status.ActiveRestores = slices.DeleteFunc(pc.Status.ActiveRestores, func(consumer nodev1alpha1.PodCheckpointConsumerReference) bool {
			return consumer.UID == operation.OperationID
		})
		// A GET that finds no consumer can precede a delayed registration. Even
		// then change the object, so the stale registration must conflict.
		nextGeneration := generation + 1
		pc.Status.RestoreGeneration = &nextGeneration
		if _, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).UpdateStatus(ctx, pc, metav1.UpdateOptions{}); err != nil {
			return err
		}
	}
	operation.ReservationPending = false
	return kl.writePodRestoreOperation(operation)
}
