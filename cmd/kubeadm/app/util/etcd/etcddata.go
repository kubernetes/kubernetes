/*
Copyright 2020 The Kubernetes Authors.

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

package etcd

import (
	"context"
	"encoding/json"
	stderrs "errors"
	"io"
	"os"
	"path/filepath"
	"time"

	"go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/client/pkg/v3/transport"
	"go.etcd.io/etcd/client/pkg/v3/types"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/pkg/v3/pbutil"
	"go.etcd.io/etcd/server/v3/etcdserver/api/membership"
	"go.etcd.io/etcd/server/v3/etcdserver/api/snap"
	"go.etcd.io/etcd/server/v3/storage/backend"
	"go.etcd.io/etcd/server/v3/storage/schema"
	"go.etcd.io/etcd/server/v3/storage/wal"
	"go.etcd.io/etcd/server/v3/storage/wal/walpb"
	"go.etcd.io/raft/v3/raftpb"
	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

const (
	rollbackSnapshotFileName = "etcd.db"
	rollbackMetadataFileName = "restore-metadata.json"
)

// CreateDataDirectory creates the etcd data directory (commonly /var/lib/etcd) with the right permissions.
func CreateDataDirectory(dir string) error {
	if err := os.MkdirAll(dir, 0700); err != nil {
		return errors.Wrapf(err, "failed to create the etcd data directory: %q", dir)
	}
	return nil
}

// BackupDataForRollback creates a rollback backup in backupDir, including both
// the etcd snapshot and restore metadata required to preserve identity.
func BackupDataForRollback(endpoint, ca, cert, key, dataDir, manifestDir, backupDir string) error {
	if err := os.MkdirAll(backupDir, 0700); err != nil {
		return errors.Wrapf(err, "failed to create backup directory %q", backupDir)
	}

	snapshotPath := filepath.Join(backupDir, rollbackSnapshotFileName)
	if err := snapshotSave(endpoint, ca, cert, key, snapshotPath); err != nil {
		return errors.Wrapf(err, "failed to save etcd snapshot")
	}

	initialClusterToken, tokenErr := getInitialClusterTokenFromStaticPod(manifestDir)
	if tokenErr != nil {
		klog.V(4).Infof("[etcd] could not read --initial-cluster-token from manifest: %v", tokenErr)
	}

	metadataPath := filepath.Join(backupDir, rollbackMetadataFileName)
	if err := snapshotSaveRestoreMetadata(dataDir, metadataPath, initialClusterToken); err != nil {
		return errors.Wrapf(err, "failed to save etcd restore metadata")
	}

	return nil
}

// RestoreDataForRollback restores etcd data from backupDir.
// It consumes rollback metadata when present and falls back to runtime metadata
// detection for older backups that do not contain metadata files.
func RestoreDataForRollback(backupDir, dataDir, memberName, peerURL, initialClusterToken string) error {
	snapshotPath := filepath.Join(backupDir, rollbackSnapshotFileName)
	metadataPath := filepath.Join(backupDir, rollbackMetadataFileName)
	if _, err := os.Stat(metadataPath); err != nil {
		if os.IsNotExist(err) {
			metadataPath = ""
		} else {
			return errors.Wrapf(err, "failed to stat restore metadata file %q", metadataPath)
		}
	}

	if err := snapshotRestore(snapshotPath, metadataPath, dataDir, memberName, peerURL, initialClusterToken); err != nil {
		return errors.Wrapf(err, "failed to restore etcd data from backup dir %q", backupDir)
	}
	return nil
}

// snapshotSave saves a consistent, point-in-time snapshot of etcd to snapshotPath
// using the etcd v3 Maintenance API.
func snapshotSave(endpoint, ca, cert, key, snapshotPath string) error {
	tlsInfo := transport.TLSInfo{
		CertFile:      cert,
		KeyFile:       key,
		TrustedCAFile: ca,
	}
	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		return errors.Wrapf(err, "failed to build TLS config for etcd snapshot")
	}

	client, err := clientv3.New(clientv3.Config{
		Endpoints:   []string{endpoint},
		DialTimeout: etcdTimeout,
		TLS:         tlsConfig,
	})
	if err != nil {
		return errors.Wrapf(err, "failed to create etcd client for snapshot")
	}
	defer func() { _ = client.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	rc, err := client.Snapshot(ctx)
	if err != nil {
		return errors.Wrapf(err, "failed to request etcd snapshot")
	}
	defer func() { _ = rc.Close() }()

	f, err := os.Create(snapshotPath)
	if err != nil {
		return errors.Wrapf(err, "failed to create snapshot file %q", snapshotPath)
	}
	defer func() { _ = f.Close() }()

	if _, err = io.Copy(f, rc); err != nil {
		return errors.Wrapf(err, "failed to write snapshot to %q", snapshotPath)
	}
	return nil
}

// snapshotSaveRestoreMetadata persists restore-critical raft metadata from the
// current etcd data directory into metadataPath.
//
// The saved payload is coupled with a snapshot backup and should be restored
// together with that snapshot to guarantee member/cluster identity stability.
func snapshotSaveRestoreMetadata(dataDir, metadataPath, initialClusterToken string) error {
	lg := zap.NewNop()
	nodeID, clusterID, confState, ok, err := loadExistingRestoreMetadata(dataDir, lg)
	if err != nil {
		return errors.Wrapf(err, "failed to load current restore metadata from %q", dataDir)
	}
	if !ok {
		return errors.Errorf("no restore metadata found in %q", dataDir)
	}

	payload := restoreMetadataPayload{
		NodeID:              nodeID,
		ClusterID:           clusterID,
		InitialClusterToken: initialClusterToken,
	}
	if confState != nil {
		payload.ConfState = pbutil.MustMarshalMessage(confState)
	}

	if err := os.MkdirAll(filepath.Dir(metadataPath), 0700); err != nil {
		return errors.Wrapf(err, "failed to create metadata directory for %q", metadataPath)
	}

	b, err := json.Marshal(&payload)
	if err != nil {
		return errors.Wrapf(err, "failed to marshal restore metadata payload")
	}
	if err := os.WriteFile(metadataPath, b, 0600); err != nil {
		return errors.Wrapf(err, "failed to write restore metadata file %q", metadataPath)
	}
	return nil
}

func getInitialClusterTokenFromStaticPod(manifestDir string) (string, error) {
	manifestPath := constants.GetStaticPodFilepath(constants.Etcd, manifestDir)
	pod, err := staticpod.ReadStaticPodFromDisk(manifestPath)
	if err != nil {
		return "", err
	}
	if len(pod.Spec.Containers) == 0 {
		return "", errors.Errorf("etcd static pod has no containers: %q", manifestPath)
	}

	container := pod.Spec.Containers[0]
	cmd := append([]string{}, container.Command...)
	cmd = append(cmd, container.Args...)
	args := kubeadmutil.ArgumentsFromCommand(cmd)
	token, _ := kubeadmapi.GetArgValue(args, "initial-cluster-token", -1)
	return token, nil
}

// snapshotRestore reconstructs an etcd data directory from a snapshot file.
// It reads consistent-index metadata from the snapshot's bbolt database,
// then writes a fresh member/wal and member/snap layout that etcd can boot from.
//
// If an existing etcd data directory is present, it reuses the previous
// NodeID, ClusterID and raft ConfState from WAL metadata so member identity and
// cluster membership remain unchanged across rollback restore.
//
// dataDir is cleared and recreated from the snapshot. restoreMetadataPath, when
// set, must point to the metadata captured at snapshot backup time and is used
// as the source of truth for NodeID/ClusterID/ConfState.
// memberName is the etcd
// --name flag value. peerURL is the --initial-advertise-peer-urls value.
// initialClusterToken is the --initial-cluster-token value used only when
// restore metadata is unavailable and IDs must be re-derived.
func snapshotRestore(snapshotPath, restoreMetadataPath, dataDir, memberName, peerURL, initialClusterToken string) error {
	lg := zap.NewNop()

	nodeID, clusterID, confState, tokenFromMetadata, haveExistingMetadata, err := loadRestoreMetadataForSnapshot(restoreMetadataPath, dataDir, lg)
	if err != nil {
		return errors.Wrapf(err, "failed to load restore metadata")
	}
	if initialClusterToken == "" {
		initialClusterToken = tokenFromMetadata
	}

	// Read raft metadata from the snapshot's bbolt database.
	be := backend.NewDefaultBackend(lg, snapshotPath)
	tx := be.BatchTx()
	tx.Lock()
	index, term := schema.UnsafeReadConsistentIndex(tx)
	tx.Unlock()
	be.Close()

	if !haveExistingMetadata {
		// Fall back to deterministic ID derivation when there is no prior local WAL metadata.
		if initialClusterToken == "" {
			return errors.Errorf("cannot derive member/cluster IDs: initialClusterToken is empty and restore metadata is unavailable")
		}
		peerURLs, err := types.NewURLs([]string{peerURL})
		if err != nil {
			return errors.Wrapf(err, "failed to parse peer URL %q", peerURL)
		}
		member := membership.NewMember(memberName, peerURLs, initialClusterToken, nil)
		urlsmap := types.URLsMap{memberName: peerURLs}
		cluster, err := membership.NewClusterFromURLsMap(lg, initialClusterToken, urlsmap)
		if err != nil {
			return errors.Wrapf(err, "failed to create cluster config for snapshot restore")
		}
		nodeID = uint64(member.ID)
		clusterID = uint64(cluster.ID())
	}

	if confState == nil || len(confState.GetVoters()) == 0 {
		// Non-initial snapshots require a non-empty ConfState; default to local node.
		confState = &raftpb.ConfState{Voters: []uint64{nodeID}}
	}

	// Clear and recreate the data directory.
	if err := os.RemoveAll(dataDir); err != nil {
		return errors.Wrapf(err, "failed to remove existing data directory %q", dataDir)
	}
	snapDir := filepath.Join(dataDir, "member", "snap")
	walDir := filepath.Join(dataDir, "member", "wal")
	for _, dir := range []string{snapDir, walDir} {
		if err := os.MkdirAll(dir, 0700); err != nil {
			return errors.Wrapf(err, "failed to create directory %q", dir)
		}
	}

	// Copy the snapshot database to member/snap/db.
	dbPath := filepath.Join(snapDir, "db")
	if err := copySnapshotFile(snapshotPath, dbPath); err != nil {
		return errors.Wrapf(err, "failed to copy snapshot database to %q", dbPath)
	}

	// Create the WAL with member/cluster metadata and a snapshot entry.
	walMetadata := pbutil.MustMarshalMessage(&etcdserverpb.Metadata{
		NodeID:    &nodeID,
		ClusterID: &clusterID,
	})
	w, err := wal.Create(lg, walDir, walMetadata)
	if err != nil {
		return errors.Wrapf(err, "failed to create WAL in %q", walDir)
	}
	if err := w.SaveSnapshot(&walpb.Snapshot{Index: &index, Term: &term, ConfState: confState}); err != nil {
		_ = w.Close()
		return errors.Wrapf(err, "failed to save snapshot entry to WAL")
	}
	commit := index
	if err := w.Save(&raftpb.HardState{
		Term:   &term,
		Vote:   &nodeID,
		Commit: &commit,
	}, nil); err != nil {
		_ = w.Close()
		return errors.Wrapf(err, "failed to save hard state to WAL")
	}
	if err := w.Close(); err != nil {
		return errors.Wrapf(err, "failed to close WAL after restore")
	}

	// Create the raft snapshot file that references the restored index/term.
	snapshotter := snap.New(lg, snapDir)
	if err := snapshotter.SaveSnap(&raftpb.Snapshot{
		Metadata: &raftpb.SnapshotMetadata{
			Index:     &index,
			Term:      &term,
			ConfState: confState,
		},
	}); err != nil {
		return errors.Wrapf(err, "failed to save raft snapshot file")
	}
	return nil
}

func copySnapshotFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer func() { _ = srcFile.Close() }()

	dstFile, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0600)
	if err != nil {
		return err
	}
	defer func() { _ = dstFile.Close() }()

	_, err = io.Copy(dstFile, srcFile)
	return err
}

func loadExistingRestoreMetadata(dataDir string, lg *zap.Logger) (nodeID, clusterID uint64, confState *raftpb.ConfState, ok bool, err error) {
	walDir := filepath.Join(dataDir, "member", "wal")
	if _, statErr := os.Stat(walDir); statErr != nil {
		return 0, 0, nil, false, statErr
	}

	walSnaps, err := wal.ValidSnapshotEntries(lg, walDir)
	if err != nil {
		return 0, 0, nil, false, err
	}

	startSnap := &walpb.Snapshot{}
	if len(walSnaps) > 0 {
		startSnap = walSnaps[len(walSnaps)-1]
		confState = cloneConfState(startSnap.GetConfState())
	}

	w, err := wal.OpenForRead(lg, walDir, startSnap)
	if err != nil {
		return 0, 0, nil, false, err
	}
	defer func() { _ = w.Close() }()

	walMetadata, _, _, readErr := w.ReadAll()
	if readErr != nil && !stderrs.Is(readErr, wal.ErrSnapshotNotFound) {
		return 0, 0, nil, false, readErr
	}
	if len(walMetadata) == 0 {
		return 0, 0, nil, false, errors.Errorf("empty WAL metadata in %q", walDir)
	}

	md := &etcdserverpb.Metadata{}
	if err := proto.Unmarshal(walMetadata, md); err != nil {
		return 0, 0, nil, false, err
	}
	if md.NodeID == nil || md.ClusterID == nil {
		return 0, 0, nil, false, errors.Errorf("incomplete WAL metadata in %q", walDir)
	}

	return md.GetNodeID(), md.GetClusterID(), confState, true, nil
}

func cloneConfState(in *raftpb.ConfState) *raftpb.ConfState {
	if in == nil {
		return nil
	}
	out := &raftpb.ConfState{
		Voters:         append([]uint64(nil), in.GetVoters()...),
		Learners:       append([]uint64(nil), in.GetLearners()...),
		VotersOutgoing: append([]uint64(nil), in.GetVotersOutgoing()...),
		LearnersNext:   append([]uint64(nil), in.GetLearnersNext()...),
	}
	if in.AutoLeave != nil {
		autoLeave := in.GetAutoLeave()
		out.AutoLeave = &autoLeave
	}
	return out
}

type restoreMetadataPayload struct {
	NodeID              uint64 `json:"nodeID"`
	ClusterID           uint64 `json:"clusterID"`
	ConfState           []byte `json:"confState,omitempty"`
	InitialClusterToken string `json:"initialClusterToken,omitempty"`
}

func loadRestoreMetadataForSnapshot(restoreMetadataPath, dataDir string, lg *zap.Logger) (nodeID, clusterID uint64, confState *raftpb.ConfState, initialClusterToken string, ok bool, err error) {
	if restoreMetadataPath != "" {
		return loadRestoreMetadataFile(restoreMetadataPath)
	}

	nodeID, clusterID, confState, ok, err = loadExistingRestoreMetadata(dataDir, lg)
	if err != nil {
		klog.V(4).Infof("[etcd] could not load existing WAL metadata from %q, restore will compute IDs: %v", dataDir, err)
	}
	return nodeID, clusterID, confState, "", ok, nil
}

func loadRestoreMetadataFile(path string) (nodeID, clusterID uint64, confState *raftpb.ConfState, initialClusterToken string, ok bool, err error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, 0, nil, "", false, err
	}
	var payload restoreMetadataPayload
	if err := json.Unmarshal(b, &payload); err != nil {
		return 0, 0, nil, "", false, err
	}
	if payload.NodeID == 0 || payload.ClusterID == 0 {
		return 0, 0, nil, "", false, errors.Errorf("invalid restore metadata file %q: empty node/cluster IDs", path)
	}

	if len(payload.ConfState) > 0 {
		decoded := &raftpb.ConfState{}
		if err := proto.Unmarshal(payload.ConfState, decoded); err != nil {
			return 0, 0, nil, "", false, err
		}
		confState = decoded
	}
	return payload.NodeID, payload.ClusterID, confState, payload.InitialClusterToken, true, nil
}
