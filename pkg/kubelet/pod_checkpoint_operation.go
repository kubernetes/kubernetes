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
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"
	"unicode/utf8"

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/events"
	utilstore "k8s.io/kubernetes/pkg/kubelet/util/store"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

var (
	errPodCheckpointOutputCleanupPending = errors.New("checkpoint output cleanup has not completed")
	errPodCheckpointCompletionPending    = errors.New("runtime checkpoint completion has not been confirmed")
)

// podOperationRecoveryMaxBackOff caps the delay between recovery attempts for
// an operation whose runtime cleanup keeps failing.
const podOperationRecoveryMaxBackOff = time.Minute

// Independent resource managers can run before a restarted kubelet rediscovers
// the source Pod. Runtime completion records must protect its resources too.
type podOperationAwarePodWorkers struct {
	PodWorkers
	kubelet *Kubelet
}

// PodOperationMayUsePodResources excludes operation-owned resources from
// garbage collection, including quota eviction for an otherwise active Pod.
func (p *podOperationAwarePodWorkers) PodOperationMayUsePodResources(uid types.UID) bool {
	return p.kubelet.podOperationMayUsePodResources(uid)
}

func (p *podOperationAwarePodWorkers) ShouldPodRuntimeBeRemoved(uid types.UID) bool {
	return p.PodWorkers.ShouldPodRuntimeBeRemoved(uid) && !p.kubelet.podOperationMayUsePodResources(uid)
}

func (p *podOperationAwarePodWorkers) ShouldPodContentBeRemoved(uid types.UID) bool {
	return p.PodWorkers.ShouldPodContentBeRemoved(uid) && !p.kubelet.podOperationMayUsePodResources(uid)
}

func (p *podOperationAwarePodWorkers) CouldHaveRunningContainers(uid types.UID) bool {
	return p.PodWorkers.CouldHaveRunningContainers(uid) || p.kubelet.podOperationMayUsePodResources(uid)
}

func (kl *Kubelet) podOperationMayUsePodResources(uid types.UID) bool {
	if _, owned := kl.checkpointOperationOwners.Load(uid); owned {
		return true
	}
	podKey, err := checkpointOperationKey(uid)
	if err != nil {
		return true
	}
	for _, root := range kl.podCheckpointOperationRoots() {
		info, err := os.Stat(filepath.Join(root, podKey))
		if os.IsNotExist(err) || errors.Is(err, syscall.ENOTDIR) {
			// ENOTDIR on the per-Pod leaf is corruption, not absence.
			if err != nil && errors.Is(err, syscall.ENOTDIR) {
				if _, parentErr := os.Stat(root); parentErr == nil {
					return true
				}
			}
			continue
		}
		if err != nil || !info.IsDir() {
			return true
		}
	}
	operations, err := kl.readPodCheckpointOperations(uid)
	if err != nil {
		return true
	}
	for _, operation := range operations {
		if !operation.Complete {
			return true
		}
	}
	return false
}

// Each operation has its own file so failed publication or an older operation
// cannot be overwritten by another checkpoint of the same source Pod.
type podCheckpointOperation struct {
	Version       int       `json:"version"`
	CheckpointUID types.UID `json:"checkpointUID"`
	PodUID        types.UID `json:"podUID"`
	PodName       string    `json:"podName"`
	Namespace     string    `json:"namespace"`
	Name          string    `json:"name"`
	Complete      bool      `json:"complete,omitempty"`
	Succeeded     bool      `json:"succeeded,omitempty"`
	Location      string    `json:"location,omitempty"`
	Message       string    `json:"message,omitempty"`
}

func checkpointOperationKey(uid types.UID) (string, error) {
	if uid == "" || len(uid) > 120 {
		return "", fmt.Errorf("invalid checkpoint operation UID %q", uid)
	}
	return hex.EncodeToString([]byte(uid)), nil
}

func (kl *Kubelet) podCheckpointOperationsDir() string {
	return filepath.Join(kl.getRootDir(), "pod-checkpoint-operations")
}

// Read the legacy journal as well, including while the feature gate is off.
// New intent lives outside the filesystem that checkpoint bulk data can fill.
func (kl *Kubelet) podCheckpointOperationRoots() []string {
	return []string{filepath.Join(kl.getPodCheckpointsDir(), "operations"), kl.podCheckpointOperationsDir()}
}

const maxPodCheckpointOperationSize = 8192

func (kl *Kubelet) podCheckpointOperationPath(podUID, checkpointUID types.UID) (string, string, error) {
	podKey, err := checkpointOperationKey(podUID)
	if err != nil {
		return "", "", err
	}
	key, err := checkpointOperationKey(checkpointUID)
	return filepath.Join(kl.podCheckpointOperationsDir(), podKey), key, err
}

func (kl *Kubelet) writePodCheckpointOperation(operation podCheckpointOperation) error {
	dir, key, err := kl.podCheckpointOperationPath(operation.PodUID, operation.CheckpointUID)
	if err != nil {
		return err
	}
	// Runtime errors are untrusted and can otherwise grow the control journal.
	if len(operation.Message) > 1024 {
		operation.Message = operation.Message[:1024]
		for !utf8.ValidString(operation.Message) {
			operation.Message = operation.Message[:len(operation.Message)-1]
		}
	}
	operation.Version = 1
	data, err := json.Marshal(operation)
	if err != nil {
		return err
	}
	if len(data) > maxPodCheckpointOperationSize {
		return fmt.Errorf("checkpoint operation record exceeds %d bytes", maxPodCheckpointOperationSize)
	}
	if err := utilfs.MkdirAll(dir, 0700); err != nil {
		return err
	}
	fs := kl.podCheckpointFilesystem
	if fs == nil {
		fs = &utilfs.DefaultFs{}
	}
	store, err := utilstore.NewFileStore(dir, fs)
	if err != nil {
		return err
	}
	return store.Write(key, data)
}

func (kl *Kubelet) readPodCheckpointOperations(podUID types.UID) ([]podCheckpointOperation, error) {
	key, err := checkpointOperationKey(podUID)
	if err != nil {
		return nil, err
	}
	byUID := make(map[types.UID]podCheckpointOperation)
	for _, root := range kl.podCheckpointOperationRoots() {
		operations, err := kl.readPodCheckpointOperationsAt(podUID, filepath.Join(root, key))
		if err != nil {
			return nil, err
		}
		for _, operation := range operations {
			if old, exists := byUID[operation.CheckpointUID]; exists && (old.PodUID != operation.PodUID || old.Name != operation.Name || old.Namespace != operation.Namespace || old.PodName != operation.PodName || old.Complete && old != operation) {
				return nil, fmt.Errorf("conflicting checkpoint operation records for %q", operation.CheckpointUID)
			}
			byUID[operation.CheckpointUID] = operation
		}
	}
	operations := make([]podCheckpointOperation, 0, len(byUID))
	for _, operation := range byUID {
		operations = append(operations, operation)
	}
	return operations, nil
}

func (kl *Kubelet) readPodCheckpointOperationsAt(podUID types.UID, dir string) ([]podCheckpointOperation, error) {
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if errors.Is(err, syscall.ENOTDIR) {
		// A non-directory journal leaf is corrupt. A missing legacy parent
		// means this Pod has no legacy operation index.
		if _, parentErr := os.Stat(filepath.Dir(dir)); os.IsNotExist(parentErr) || errors.Is(parentErr, syscall.ENOTDIR) {
			return nil, nil
		}
	}
	if err != nil {
		return nil, err
	}
	operations := make([]podCheckpointOperation, 0, len(entries))
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".") {
			continue
		}
		file, err := os.Open(filepath.Join(dir, entry.Name()))
		if os.IsNotExist(err) || errors.Is(err, syscall.ENOTDIR) {
			continue
		}
		if err != nil {
			return nil, err
		}
		data, err := io.ReadAll(io.LimitReader(file, maxPodCheckpointOperationSize+1))
		_ = file.Close()
		if err != nil {
			return nil, err
		}
		if len(data) > maxPodCheckpointOperationSize {
			return nil, fmt.Errorf("checkpoint operation record for Pod %q exceeds size limit", podUID)
		}
		var operation podCheckpointOperation
		if err := json.Unmarshal(data, &operation); err != nil {
			return nil, fmt.Errorf("read checkpoint operation for Pod %q: %w", podUID, err)
		}
		_, expected, err := kl.podCheckpointOperationPath(operation.PodUID, operation.CheckpointUID)
		if err != nil || operation.Version != 1 || operation.PodUID != podUID || expected != entry.Name() {
			return nil, fmt.Errorf("invalid checkpoint operation record for Pod %q", podUID)
		}
		operations = append(operations, operation)
	}
	return operations, nil
}

func (kl *Kubelet) podCheckpointOperationPods() ([]types.UID, error) {
	pods := make(map[types.UID]struct{})
	for _, root := range kl.podCheckpointOperationRoots() {
		entries, err := os.ReadDir(root)
		if os.IsNotExist(err) || errors.Is(err, syscall.ENOTDIR) {
			continue
		}
		if err != nil {
			return nil, err
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			decoded, err := hex.DecodeString(entry.Name())
			if err == nil {
				pods[types.UID(decoded)] = struct{}{}
			}
		}
	}
	result := make([]types.UID, 0, len(pods))
	for uid := range pods {
		result = append(result, uid)
	}
	return result, nil
}

func (kl *Kubelet) findPodCheckpointOperation(checkpointUID types.UID) (*podCheckpointOperation, error) {
	pods, err := kl.podCheckpointOperationPods()
	if err != nil {
		return nil, err
	}
	for _, uid := range pods {
		operations, err := kl.readPodCheckpointOperations(uid)
		if err != nil {
			return nil, err
		}
		for _, operation := range operations {
			if operation.CheckpointUID == checkpointUID {
				return &operation, nil
			}
		}
	}
	return nil, nil
}

func (kl *Kubelet) clearPodCheckpointOperation(podUID, checkpointUID types.UID) error {
	dir, key, err := kl.podCheckpointOperationPath(podUID, checkpointUID)
	if err != nil {
		return err
	}
	for _, root := range kl.podCheckpointOperationRoots() {
		legacyDir := filepath.Join(root, filepath.Base(dir))
		if err := os.Remove(filepath.Join(legacyDir, key)); err != nil && !os.IsNotExist(err) && !errors.Is(err, syscall.ENOTDIR) {
			return err
		}
		_ = os.Remove(legacyDir)
	}
	return nil
}

func (kl *Kubelet) checkpointOperation(run *podCheckpointRun) podCheckpointOperation {
	return podCheckpointOperation{Version: 1, CheckpointUID: run.uid, PodUID: run.pod.UID, PodName: run.pod.Name, Namespace: run.namespace, Name: run.name}
}

func (kl *Kubelet) confirmPodCheckpointCompletion(ctx context.Context, checkpointUID types.UID) error {
	ctx, cancel := context.WithTimeout(ctx, podCheckpointAPITimeout)
	defer cancel()
	if err := kl.containerRuntime.CancelPodOperation(ctx, &runtimeapi.CancelPodOperationRequest{
		OperationId: string(checkpointUID), OperationType: runtimeapi.PodOperationType_POD_CHECKPOINT,
	}); err != nil {
		return fmt.Errorf("%w: %w", errPodCheckpointCompletionPending, err)
	}
	return nil
}

func (kl *Kubelet) podCheckpointPublicationContext(fallback context.Context) context.Context {
	if ctx := kl.checkpointOperationContext.Load(); ctx != nil {
		return *ctx
	}
	return fallback
}

// reconcilePodCheckpointOperation runs from the source Pod's worker, before
// probing, container reconciliation or termination can overtake runtime work.
func (kl *Kubelet) reconcilePodCheckpointOperation(ctx context.Context, pod *v1.Pod) error {
	return kl.reconcilePodCheckpointOperations(ctx, pod, true)
}

func (kl *Kubelet) reconcilePodCheckpointOperations(ctx context.Context, pod *v1.Pod, publish bool) error {
	if pod.UID == "" {
		return nil
	}
	if _, owned := kl.checkpointOperationOwners.LoadOrStore(pod.UID, struct{}{}); owned {
		return errPodCheckpointCompletionPending
	}
	defer kl.checkpointOperationOwners.Delete(pod.UID)
	operations, err := kl.readPodCheckpointOperations(pod.UID)
	if err != nil {
		return fmt.Errorf("%w: %w", errPodCheckpointCompletionPending, err)
	}
	if len(operations) == 0 {
		return nil
	}
	settleCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), podCheckpointAPITimeout)
	defer cancel()
	completed := make(map[types.UID]bool)
	for i := range operations {
		operation := &operations[i]
		if !operation.Complete {
			kl.statusManager.SetPodCheckpointingCondition(klog.FromContext(ctx), pod.UID, &v1.PodCondition{
				Type: v1.PodCheckpointing, Status: v1.ConditionTrue, Reason: events.CheckpointingPod,
				Message: "Waiting for runtime checkpoint completion", ObservedGeneration: pod.Generation,
			})
			if err := kl.probeManager.SuspendProbes(settleCtx, pod.UID); err != nil {
				return err
			}
			if err := kl.confirmPodCheckpointCompletion(settleCtx, operation.CheckpointUID); err != nil {
				return err
			}
			operation.Complete = true
			operation.Message = "checkpoint failed: runtime cleanup completed after an interrupted checkpoint"
			completed[operation.CheckpointUID] = true
		}
	}
	// Settle every operation before releasing shared probe ownership. Cleanup
	// and metadata retries do not imply that runtime work is still active.
	kl.probeManager.ResumeProbes(pod.UID)
	kl.statusManager.SetPodCheckpointingCondition(klog.FromContext(ctx), pod.UID, nil)
	for _, operation := range operations {
		_, output, err := kl.podCheckpointOutputPath(operation.CheckpointUID)
		if err != nil {
			return err
		}
		run := &podCheckpointRun{pod: pod, namespace: operation.Namespace, name: operation.Name, uid: operation.CheckpointUID, outputPath: output}
		if !operation.Succeeded {
			// Publication and recovery share one owner for archive cleanup.
			if _, publishing := kl.checkpointPublications.LoadOrStore(operation.CheckpointUID, struct{}{}); !publishing {
				removed := kl.removePodCheckpointOutput(ctx, run, "confirmed runtime recovery")
				kl.checkpointPublications.Delete(operation.CheckpointUID)
				if !removed {
					return errPodCheckpointOutputCleanupPending
				}
			}
		}
		if completed[operation.CheckpointUID] {
			if err := kl.writePodCheckpointOperation(operation); err != nil {
				return fmt.Errorf("%w: %w", errPodCheckpointCompletionPending, err)
			}
		}
		_, queued := kl.checkpointRecoveryPending.Load(operation.CheckpointUID)
		if publish && !queued && kl.kubeClient != nil {
			if err := kl.publishRecordedPodCheckpointOutcome(kl.podCheckpointPublicationContext(ctx), operation); err != nil {
				return err
			}
		}
	}
	return nil
}

// API object deletion does not remove these records. This worker can settle
// orphaned operations after restart without occupying the informer workers.
func (kl *Kubelet) recoverPodCheckpointOperations(ctx context.Context) {
	pods, err := kl.podCheckpointOperationPods()
	if err != nil {
		klog.FromContext(ctx).Error(err, "Cannot list checkpoint recovery records")
		return
	}
	kl.checkpointRecoveryBackOff.GC()
	for _, uid := range pods {
		if _, owned := kl.checkpointOperationOwners.Load(uid); owned {
			continue
		}
		// A runtime that keeps failing cleanup is retried with a growing delay
		// instead of on every pass.
		if kl.checkpointRecoveryBackOff.IsInBackOffSinceUpdate(string(uid), kl.clock.Now()) {
			continue
		}
		operations, err := kl.readPodCheckpointOperations(uid)
		if err != nil {
			kl.checkpointRecoveryBackOff.Next(string(uid), kl.clock.Now())
			klog.FromContext(ctx).Error(err, "Cannot read checkpoint recovery records")
			continue
		}
		if len(operations) == 0 {
			continue
		}
		pod := &v1.Pod{}
		pod.UID, pod.Namespace, pod.Name = operations[0].PodUID, operations[0].Namespace, operations[0].PodName
		if err := kl.reconcilePodCheckpointOperation(ctx, pod); err != nil {
			kl.checkpointRecoveryBackOff.Next(string(uid), kl.clock.Now())
			klog.FromContext(ctx).Error(err, "Checkpoint runtime recovery remains pending", "pod", klog.KObj(pod))
			continue
		}
		kl.checkpointRecoveryBackOff.Reset(string(uid))
	}
}

// finalizePodCheckpointPublication retains the recovery record until both the
// API outcome and any abandoned archive cleanup have been settled by its owner.
func (kl *Kubelet) finalizePodCheckpointPublication(ctx context.Context, run *podCheckpointRun, succeeded bool, location, message string, recordObserved bool) error {
	if !succeeded && run.outputPath != "" && !kl.removePodCheckpointOutput(ctx, run, "confirmed runtime failure") {
		return errPodCheckpointOutputCleanupPending
	}
	err := kl.finalizePodCheckpointOutcome(ctx, run.namespace, run.name, run.uid, succeeded, location, message, recordObserved)
	if err != nil {
		if !apierrors.IsNotFound(err) && !errors.Is(err, errPodCheckpointNotPending) {
			return err
		}
		if succeeded && !kl.removePodCheckpointOutput(ctx, run, "checkpoint object removed or replaced") {
			return errPodCheckpointOutputCleanupPending
		}
	}
	if err := kl.clearPodCheckpointOperation(run.pod.UID, run.uid); err != nil {
		return fmt.Errorf("remove completed checkpoint operation record: %w", err)
	}
	kl.checkpointRecoveryPending.Delete(run.uid)
	kl.checkpointCleanupBlocked.Delete(run.uid)
	return nil
}

// A stored runtime outcome belongs to this checkpoint and its original source,
// even when the API name now identifies a different Pod or no Pod at all.
func (kl *Kubelet) recordedPodCheckpointPending(ctx context.Context, pc *nodev1alpha1.PodCheckpoint) (bool, error) {
	if pc.Status.SourcePodUID == nil || *pc.Status.SourcePodUID == "" {
		return false, nil
	}
	sourceUID := *pc.Status.SourcePodUID
	operations, err := kl.readPodCheckpointOperations(sourceUID)
	if err != nil {
		return true, fmt.Errorf("read runtime operation for PodCheckpoint %s/%s: %w", pc.Namespace, pc.Name, err)
	}
	for _, operation := range operations {
		if operation.CheckpointUID != pc.UID {
			continue
		}
		if !operation.Complete {
			if pod, found := kl.podManager.GetPodByUID(sourceUID); found {
				kl.SyncPodNow(ctx, pod)
			}
			return true, errPodCheckpointCompletionPending
		}
		if _, owned := kl.checkpointOperationOwners.Load(sourceUID); owned {
			return true, errPodCheckpointInFlight
		}
		if err := kl.publishRecordedPodCheckpointOutcome(kl.podCheckpointPublicationContext(ctx), operation); err != nil {
			return true, err
		}
		return true, errPodCheckpointInFlight
	}
	if kl.podCheckpointInFlight(sourceUID) {
		return true, errPodCheckpointInFlight
	}
	return false, nil
}

// Re-read after claiming publication: an earlier owner may have published and
// retired the record after a watch/recovery pass took its snapshot.
func (kl *Kubelet) publishRecordedPodCheckpointOutcome(ctx context.Context, snapshot podCheckpointOperation) error {
	if _, publishing := kl.checkpointPublications.LoadOrStore(snapshot.CheckpointUID, struct{}{}); publishing {
		return nil
	}
	release := true
	defer func() {
		if release {
			kl.checkpointPublications.Delete(snapshot.CheckpointUID)
		}
	}()
	operations, err := kl.readPodCheckpointOperations(snapshot.PodUID)
	if err != nil {
		return err
	}
	for _, operation := range operations {
		if operation.CheckpointUID != snapshot.CheckpointUID {
			continue
		}
		if !operation.Complete {
			return errPodCheckpointCompletionPending
		}
		_, output, err := kl.podCheckpointOutputPath(operation.CheckpointUID)
		if err != nil {
			return err
		}
		pod := &v1.Pod{}
		pod.UID, pod.Namespace, pod.Name = operation.PodUID, operation.Namespace, operation.PodName
		run := &podCheckpointRun{pod: pod, namespace: operation.Namespace, name: operation.Name, uid: operation.CheckpointUID, outputPath: output}
		release = false
		go kl.doPublishPodCheckpointOutcome(ctx, run, operation.Succeeded, operation.Location, operation.Message)
		return nil
	}
	return nil
}
