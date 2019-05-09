// Copyright 2017 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package etcdserver

import (
	"fmt"
	"os"
	"time"

	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/mvcc"
	"github.com/coreos/etcd/mvcc/backend"
	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/snap"
)

func newBackend(cfg ServerConfig) backend.Backend {
	bcfg := backend.DefaultBackendConfig()
	bcfg.Path = cfg.backendPath()
	if cfg.QuotaBackendBytes > 0 && cfg.QuotaBackendBytes != DefaultQuotaBytes {
		// permit 10% excess over quota for disarm
		bcfg.MmapSize = uint64(cfg.QuotaBackendBytes + cfg.QuotaBackendBytes/10)
	}
	return backend.New(bcfg)
}

// openSnapshotBackend renames a snapshot db to the current etcd db and opens it.
func openSnapshotBackend(cfg ServerConfig, ss *snap.Snapshotter, snapshot raftpb.Snapshot) (backend.Backend, error) {
	snapPath, err := ss.DBFilePath(snapshot.Metadata.Index)
	if err != nil {
		return nil, fmt.Errorf("database snapshot file path error: %v", err)
	}
	if err := os.Rename(snapPath, cfg.backendPath()); err != nil {
		return nil, fmt.Errorf("rename snapshot file error: %v", err)
	}
	return openBackend(cfg), nil
}

// openBackend returns a backend using the current etcd db.
func openBackend(cfg ServerConfig) backend.Backend {
	fn := cfg.backendPath()
	beOpened := make(chan backend.Backend)
	go func() {
		beOpened <- newBackend(cfg)
	}()
	select {
	case be := <-beOpened:
		return be
	case <-time.After(10 * time.Second):
		plog.Warningf("another etcd process is using %q and holds the file lock, or loading backend file is taking >10 seconds", fn)
		plog.Warningf("waiting for it to exit before starting...")
	}
	return <-beOpened
}

// recoverBackendSnapshot recovers the DB from a snapshot in case etcd crashes
// before updating the backend db after persisting raft snapshot to disk,
// violating the invariant snapshot.Metadata.Index < db.consistentIndex. In this
// case, replace the db with the snapshot db sent by the leader.
func recoverSnapshotBackend(cfg ServerConfig, oldbe backend.Backend, snapshot raftpb.Snapshot) (backend.Backend, error) {
	var cIndex consistentIndex
	kv := mvcc.New(oldbe, &lease.FakeLessor{}, &cIndex)
	defer kv.Close()
	if snapshot.Metadata.Index <= kv.ConsistentIndex() {
		return oldbe, nil
	}
	oldbe.Close()
	return openSnapshotBackend(cfg, snap.New(cfg.SnapDir()), snapshot)
}
