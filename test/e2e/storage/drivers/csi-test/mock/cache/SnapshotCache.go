/*
Copyright 2021 The Kubernetes Authors.

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

package cache

import (
	"strings"
	"sync"

	"github.com/container-storage-interface/spec/lib/go/csi"
)

type SnapshotCache interface {
	Add(snapshot Snapshot)

	Delete(i int)

	List(ready bool) []csi.Snapshot

	FindSnapshot(k, v string) (int, Snapshot)
}

type Snapshot struct {
	Name        string
	Parameters  map[string]string
	SnapshotCSI csi.Snapshot
}

type snapshotCache struct {
	snapshotsRWL sync.RWMutex
	snapshots    []Snapshot
}

func NewSnapshotCache() SnapshotCache {
	return &snapshotCache{
		snapshots: make([]Snapshot, 0),
	}
}

func (snap *snapshotCache) Add(snapshot Snapshot) {
	snap.snapshotsRWL.Lock()
	defer snap.snapshotsRWL.Unlock()

	snap.snapshots = append(snap.snapshots, snapshot)
}

func (snap *snapshotCache) Delete(i int) {
	snap.snapshotsRWL.Lock()
	defer snap.snapshotsRWL.Unlock()

	copy(snap.snapshots[i:], snap.snapshots[i+1:])
	snap.snapshots = snap.snapshots[:len(snap.snapshots)-1]
}

func (snap *snapshotCache) List(ready bool) []csi.Snapshot {
	snap.snapshotsRWL.RLock()
	defer snap.snapshotsRWL.RUnlock()

	snapshots := make([]csi.Snapshot, 0)
	for _, v := range snap.snapshots {
		if v.SnapshotCSI.GetReadyToUse() {
			snapshots = append(snapshots, v.SnapshotCSI)
		}
	}

	return snapshots
}

func (snap *snapshotCache) FindSnapshot(k, v string) (int, Snapshot) {
	snap.snapshotsRWL.RLock()
	defer snap.snapshotsRWL.RUnlock()

	snapshotIdx := -1
	for i, vi := range snap.snapshots {
		switch k {
		case "id":
			if strings.EqualFold(v, vi.SnapshotCSI.GetSnapshotId()) {
				return i, vi
			}
		case "sourceVolumeId":
			if strings.EqualFold(v, vi.SnapshotCSI.SourceVolumeId) {
				return i, vi
			}
		case "name":
			if vi.Name == v {
				return i, vi
			}
		}
	}

	return snapshotIdx, Snapshot{}
}
