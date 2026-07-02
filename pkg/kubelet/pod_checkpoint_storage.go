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
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
)

// podCheckpointDataSweepPeriod bounds how long data of a deleted PodCheckpoint
// stays after the last operation that used it has settled.
const podCheckpointDataSweepPeriod = time.Minute

type podCheckpointStorageInfo struct {
	CapacityBytes, AvailableBytes   uint64
	CapacityInodes, AvailableInodes uint64
	Isolated                        bool
}

// admitPodCheckpointStorage runs before probe suspension or runtime work. The
// filesystem capacity is the hard limit; headroom only avoids starting a capture
// that is already likely to fail. Retained archives remain charged by the kernel.
func (kl *Kubelet) admitPodCheckpointStorage() error {
	policy := kl.kubeletConfiguration.PodCheckpointStorage
	if policy == nil {
		return fmt.Errorf("configure podCheckpointStorage and mount a dedicated filesystem at %q before checkpointing", kl.getPodCheckpointsDir())
	}
	inspect := kl.podCheckpointStorageInfo
	if inspect == nil {
		inspect = inspectPodCheckpointStorage
	}
	info, err := inspect(kl.getPodCheckpointsDir())
	if err != nil {
		return fmt.Errorf("inspect Pod checkpoint storage: %w", err)
	}
	if !info.Isolated {
		return fmt.Errorf("pod checkpoint storage %q must be a dedicated filesystem separate from the kubelet root", kl.getPodCheckpointsDir())
	}
	if info.CapacityBytes == 0 || info.CapacityInodes == 0 || policy.MaxBytes <= 0 || policy.MaxInodes <= 0 || info.CapacityBytes > uint64(policy.MaxBytes) || info.CapacityInodes > uint64(policy.MaxInodes) {
		return fmt.Errorf("pod checkpoint filesystem capacity exceeds podCheckpointStorage or cannot be bounded")
	}
	if policy.MinAvailableBytes <= 0 || policy.MinAvailableInodes <= 0 || info.AvailableBytes < uint64(policy.MinAvailableBytes) || info.AvailableInodes < uint64(policy.MinAvailableInodes) {
		return fmt.Errorf("pod checkpoint storage has insufficient available bytes or inodes")
	}
	// The control journal is separate from bulk storage and has a bounded record
	// count and record size. Failed publication must not grow it without limit.
	if policy.MaxOperations <= 0 || policy.MaxOperations > 1024 {
		return fmt.Errorf("podCheckpointStorage.maxOperations must be between 1 and 1024")
	}
	count := 0
	for _, root := range kl.podCheckpointOperationRoots() {
		err := countCheckpointOperationFiles(root, &count, int(policy.MaxOperations))
		if err != nil {
			return err
		}
	}
	return nil
}

func countCheckpointOperationFiles(root string, count *int, limit int) error {
	dir, err := os.Open(root)
	if os.IsNotExist(err) || errors.Is(err, syscall.ENOTDIR) {
		return nil
	}
	if err != nil {
		return err
	}
	defer func() { _ = dir.Close() }()
	// The population is bounded during admission, including temporary or malformed
	// records. Recovery still reads existing records when a policy is lowered.
	for {
		entries, err := dir.Readdir(1)
		if len(entries) == 0 {
			if err != nil && !errors.Is(err, io.EOF) {
				return err
			}
			return nil
		}
		entry := entries[0]
		if entry.IsDir() {
			child, err := os.Open(filepath.Join(root, entry.Name()))
			if err != nil {
				return err
			}
			files, readErr := child.Readdirnames(limit - *count + 1)
			_ = child.Close()
			if readErr != nil && !errors.Is(readErr, io.EOF) {
				return readErr
			}
			*count += max(1, len(files))
		} else {
			*count++
		}
		if *count >= limit {
			return fmt.Errorf("pod checkpoint operation limit reached; finish publication or cleanup before starting another capture")
		}
	}
}

// sweepPodCheckpointData removes the data of PodCheckpoints that left this
// node's synced cache. Deleting the object releases its data; data of an
// existing object is never removed. Data stays while a capture, a publication
// or an operation record may still use it.
func (kl *Kubelet) sweepPodCheckpointData(ctx context.Context) error {
	state := kl.podCheckpointWatch.Load()
	if state == nil || state.hasSynced == nil || !state.hasSynced() {
		return nil
	}
	checkpointsDir, err := filepath.Abs(kl.getPodCheckpointsDir())
	if err != nil {
		return err
	}
	entries, err := os.ReadDir(checkpointsDir)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	deleted := make(map[types.UID]string)
	for _, entry := range entries {
		if uid, found := strings.CutPrefix(entry.Name(), podCheckpointDataPrefix); found && uid != "" {
			deleted[types.UID(uid)] = filepath.Join(checkpointsDir, entry.Name())
		}
	}
	// Read the cache after the directory: a capture creates data only for a
	// PodCheckpoint that the cache already holds.
	for _, object := range state.indexer.List() {
		if checkpoint, ok := object.(*nodev1alpha1.PodCheckpoint); ok {
			delete(deleted, checkpoint.UID)
		}
	}
	if len(deleted) == 0 {
		return nil
	}
	logger := klog.FromContext(ctx)
	inUse, err := kl.podCheckpointDataInUse()
	if err != nil {
		logger.V(2).Info("Keeping data of deleted PodCheckpoints until its use is known", "err", err)
		return nil
	}
	for uid := range inUse {
		delete(deleted, uid)
	}
	if len(deleted) == 0 || kl.kubeClient == nil {
		return nil
	}
	// Removal cannot be undone, and the cache can lag behind the API server.
	listCtx, cancel := context.WithTimeout(ctx, podCheckpointAPITimeout)
	defer cancel()
	live, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(metav1.NamespaceAll).List(listCtx, metav1.ListOptions{
		FieldSelector: fields.OneTermEqualSelector("status.nodeName", string(kl.nodeName)).String(),
	})
	if err != nil {
		return fmt.Errorf("confirm that PodCheckpoints were deleted: %w", err)
	}
	for i := range live.Items {
		delete(deleted, live.Items[i].UID)
	}
	var errs []error
	for uid, path := range deleted {
		if err := os.RemoveAll(path); err != nil {
			errs = append(errs, err)
			continue
		}
		logger.V(2).Info("Removed data of a deleted PodCheckpoint", "podCheckpointUID", uid)
	}
	return errors.Join(errs...)
}

// podCheckpointDataInUse returns the UIDs of PodCheckpoints whose data a
// capture, a publication or an operation record may still use. It fails when
// that use cannot be determined.
func (kl *Kubelet) podCheckpointDataInUse() (sets.Set[types.UID], error) {
	inUse := sets.New[types.UID]()
	insert := func(value any) bool {
		if uid, ok := value.(types.UID); ok {
			inUse.Insert(uid)
		}
		return true
	}
	// Read the in-memory owners before the saved operations: an owner is
	// released only after its operation is saved or no longer needed.
	kl.checkpointsInFlight.Range(func(_, checkpointUID any) bool { return insert(checkpointUID) })
	kl.checkpointPublications.Range(func(checkpointUID, _ any) bool { return insert(checkpointUID) })
	pods, err := kl.podCheckpointOperationPods()
	if err != nil {
		return nil, err
	}
	for _, podUID := range pods {
		operations, err := kl.readPodCheckpointOperations(podUID)
		if err != nil {
			return nil, err
		}
		for _, operation := range operations {
			inUse.Insert(operation.CheckpointUID)
		}
	}
	return inUse, nil
}
