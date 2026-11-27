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

package podcheckpoint

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/util/removeall"
)

// podCheckpointDataSweepPeriod bounds how long data of a deleted PodCheckpoint
// stays after the capture or publication that used it has finished.
const podCheckpointDataSweepPeriod = time.Minute

type podCheckpointStorageInfo struct {
	CapacityBytes, AvailableBytes   uint64
	CapacityInodes, AvailableInodes uint64
	Isolated                        bool
}

// admitPodCheckpointStorage runs before probe suspension or runtime work. The
// filesystem capacity is the hard limit; headroom only avoids starting a capture
// that is already likely to fail. Retained archives remain charged by the kernel.
func (m *manager) admitPodCheckpointStorage() error {
	policy := m.kubeletConfiguration.PodCheckpointStorage
	if policy == nil {
		return fmt.Errorf("configure podCheckpointStorage and mount a dedicated filesystem at %q before checkpointing", m.checkpointsDir)
	}
	inspect := m.podCheckpointStorageInfo
	if inspect == nil {
		inspect = inspectPodCheckpointStorage
	}
	info, err := inspect(m.checkpointsDir)
	if err != nil {
		return fmt.Errorf("inspect Pod checkpoint storage: %w", err)
	}
	if !info.Isolated {
		return fmt.Errorf("pod checkpoint storage %q must be a dedicated filesystem separate from the kubelet root", m.checkpointsDir)
	}
	if info.CapacityBytes == 0 || info.CapacityInodes == 0 || policy.MaxBytes <= 0 || policy.MaxInodes <= 0 || info.CapacityBytes > uint64(policy.MaxBytes) || info.CapacityInodes > uint64(policy.MaxInodes) {
		return fmt.Errorf("pod checkpoint filesystem capacity exceeds podCheckpointStorage or cannot be bounded")
	}
	if policy.MinAvailableBytes <= 0 || policy.MinAvailableInodes <= 0 || info.AvailableBytes < uint64(policy.MinAvailableBytes) || info.AvailableInodes < uint64(policy.MinAvailableInodes) {
		return fmt.Errorf("pod checkpoint storage has insufficient available bytes or inodes")
	}
	return nil
}

// sweepPodCheckpointData removes the data of PodCheckpoints that left this
// node's synced cache. Deleting the object releases its data; data of an
// existing object is never removed. Data stays while a capture or a
// publication may still use it.
func (m *manager) sweepPodCheckpointData(ctx context.Context) error {
	state := m.podCheckpointWatch
	if state == nil || state.hasSynced == nil || !state.hasSynced() {
		return nil
	}
	checkpointsDir, err := filepath.Abs(m.checkpointsDir)
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
	for uid := range m.podCheckpointDataInUse() {
		delete(deleted, uid)
	}
	// A synced cache cannot miss a PodCheckpoint whose data exists, and a
	// deleted PodCheckpoint cannot return, so cache lag only delays removal.
	var errs []error
	for uid, path := range deleted {
		if err := removeall.RemoveAllOneFilesystem(m.mounter, path); err != nil {
			errs = append(errs, err)
			continue
		}
		logger.V(2).Info("Removed data of a deleted PodCheckpoint", "podCheckpointUID", uid)
	}
	return utilerrors.NewAggregate(errs)
}

// podCheckpointDataInUse returns the UIDs of PodCheckpoints whose data a
// capture or a publication may still use.
func (m *manager) podCheckpointDataInUse() sets.Set[types.UID] {
	m.stateLock.Lock()
	defer m.stateLock.Unlock()
	inUse := sets.New[types.UID]()
	for _, state := range m.podCheckpoints {
		inUse.Insert(state.checkpointUID)
	}
	return inUse
}
