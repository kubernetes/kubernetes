/*
Copyright 2016 The Kubernetes Authors.

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

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/coreos/etcd/etcdserver"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/etcdserver/membership"
	"github.com/coreos/etcd/mvcc/backend"
	"github.com/coreos/etcd/mvcc/mvccpb"
	"github.com/coreos/etcd/pkg/pbutil"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/snap"
	"github.com/coreos/etcd/store"
	"github.com/coreos/etcd/wal"
	"github.com/coreos/etcd/wal/walpb"
	"github.com/golang/glog"
)

const rollbackVersion = "2.3.7"

var (
	migrateDatadir = flag.String("data-dir", "", "Path to the data directory")
	ttl            = flag.Duration("ttl", time.Hour, "TTL of event keys (default 1 hour)")
)

func main() {
	flag.Parse()
	if len(*migrateDatadir) == 0 {
		glog.Fatal("need to set '--data-dir'")
	}
	dbpath := path.Join(*migrateDatadir, "member", "snap", "db")

	// etcd3 store backend. We will use it to parse v3 data files and extract information.
	be := backend.NewDefaultBackend(dbpath)
	tx := be.BatchTx()

	// etcd2 store backend. We will use v3 data to update this and then save snapshot to disk.
	st := store.New(etcdserver.StoreClusterPrefix, etcdserver.StoreKeysPrefix)
	expireTime := time.Now().Add(*ttl)

	tx.Lock()
	err := tx.UnsafeForEach([]byte("key"), func(k, v []byte) error {
		kv := &mvccpb.KeyValue{}
		kv.Unmarshal(v)

		// This is compact key.
		if !strings.HasPrefix(string(kv.Key), "/") {
			return nil
		}

		ttlOpt := store.TTLOptionSet{}
		if kv.Lease != 0 {
			ttlOpt = store.TTLOptionSet{ExpireTime: expireTime}
		}

		if !isTombstone(k) {
			sk := path.Join(strings.Trim(etcdserver.StoreKeysPrefix, "/"), string(kv.Key))
			_, err := st.Set(sk, false, string(kv.Value), ttlOpt)
			if err != nil {
				return err
			}
		} else {
			st.Delete(string(kv.Key), false, false)
		}

		return nil
	})
	if err != nil {
		glog.Fatal(err)
	}
	tx.Unlock()

	if err := traverseAndDeleteEmptyDir(st, "/"); err != nil {
		glog.Fatal(err)
	}

	// rebuild cluster state.
	metadata, hardstate, oldSt, err := rebuild(*migrateDatadir)
	if err != nil {
		glog.Fatal(err)
	}

	// In the following, it's low level logic that saves metadata and data into v2 snapshot.
	backupPath := *migrateDatadir + ".rollback.backup"
	if err := os.Rename(*migrateDatadir, backupPath); err != nil {
		glog.Fatal(err)
	}
	if err := os.MkdirAll(path.Join(*migrateDatadir, "member", "snap"), 0700); err != nil {
		glog.Fatal(err)
	}
	walDir := path.Join(*migrateDatadir, "member", "wal")

	w, err := wal.Create(walDir, metadata)
	if err != nil {
		glog.Fatal(err)
	}
	err = w.SaveSnapshot(walpb.Snapshot{Index: hardstate.Commit, Term: hardstate.Term})
	if err != nil {
		glog.Fatal(err)
	}
	w.Close()

	event, err := oldSt.Get(etcdserver.StoreClusterPrefix, true, false)
	if err != nil {
		glog.Fatal(err)
	}
	// nodes (members info) for ConfState
	nodes := []uint64{}
	traverseMetadata(event.Node, func(n *store.NodeExtern) {
		if n.Key != etcdserver.StoreClusterPrefix {
			// update store metadata
			v := ""
			if !n.Dir {
				v = *n.Value
			}
			if n.Key == path.Join(etcdserver.StoreClusterPrefix, "version") {
				v = rollbackVersion
			}
			if _, err := st.Set(n.Key, n.Dir, v, store.TTLOptionSet{}); err != nil {
				glog.Fatal(err)
			}

			// update nodes
			fields := strings.Split(n.Key, "/")
			if len(fields) == 4 && fields[2] == "members" {
				nodeID, err := strconv.ParseUint(fields[3], 16, 64)
				if err != nil {
					glog.Fatalf("failed to parse member ID (%s): %v", fields[3], err)
				}
				nodes = append(nodes, nodeID)
			}
		}
	})

	data, err := st.Save()
	if err != nil {
		glog.Fatal(err)
	}
	raftSnap := raftpb.Snapshot{
		Data: data,
		Metadata: raftpb.SnapshotMetadata{
			Index: hardstate.Commit,
			Term:  hardstate.Term,
			ConfState: raftpb.ConfState{
				Nodes: nodes,
			},
		},
	}
	snapshotter := snap.New(path.Join(*migrateDatadir, "member", "snap"))
	if err := snapshotter.SaveSnap(raftSnap); err != nil {
		glog.Fatal(err)
	}
	fmt.Println("Finished successfully")
}

func traverseMetadata(head *store.NodeExtern, handleFunc func(*store.NodeExtern)) {
	q := []*store.NodeExtern{head}

	for len(q) > 0 {
		n := q[0]
		q = q[1:]

		handleFunc(n)

		for _, next := range n.Nodes {
			q = append(q, next)
		}
	}
}

const (
	revBytesLen       = 8 + 1 + 8
	markedRevBytesLen = revBytesLen + 1
	markBytePosition  = markedRevBytesLen - 1

	markTombstone byte = 't'
)

func isTombstone(b []byte) bool {
	return len(b) == markedRevBytesLen && b[markBytePosition] == markTombstone
}

func traverseAndDeleteEmptyDir(st store.Store, dir string) error {
	e, err := st.Get(dir, true, false)
	if err != nil {
		return err
	}
	if len(e.Node.Nodes) == 0 {
		st.Delete(dir, true, true)
		return nil
	}
	for _, node := range e.Node.Nodes {
		if !node.Dir {
			glog.V(2).Infof("key: %s", node.Key[len(etcdserver.StoreKeysPrefix):])
		} else {
			err := traverseAndDeleteEmptyDir(st, node.Key)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func rebuild(datadir string) ([]byte, *raftpb.HardState, store.Store, error) {
	waldir := path.Join(datadir, "member", "wal")
	snapdir := path.Join(datadir, "member", "snap")

	ss := snap.New(snapdir)
	snapshot, err := ss.Load()
	if err != nil && err != snap.ErrNoSnapshot {
		return nil, nil, nil, err
	}

	var walsnap walpb.Snapshot
	if snapshot != nil {
		walsnap.Index, walsnap.Term = snapshot.Metadata.Index, snapshot.Metadata.Term
	}

	w, err := wal.OpenForRead(waldir, walsnap)
	if err != nil {
		return nil, nil, nil, err
	}
	defer w.Close()

	meta, hardstate, ents, err := w.ReadAll()
	if err != nil {
		return nil, nil, nil, err
	}

	st := store.New(etcdserver.StoreClusterPrefix, etcdserver.StoreKeysPrefix)
	if snapshot != nil {
		err := st.Recovery(snapshot.Data)
		if err != nil {
			return nil, nil, nil, err
		}
	}

	cluster := membership.NewCluster("")
	cluster.SetStore(st)

	applier := etcdserver.NewApplierV2(st, cluster)
	for _, ent := range ents {
		if ent.Type == raftpb.EntryConfChange {
			var cc raftpb.ConfChange
			pbutil.MustUnmarshal(&cc, ent.Data)
			switch cc.Type {
			case raftpb.ConfChangeAddNode:
				m := new(membership.Member)
				if err := json.Unmarshal(cc.Context, m); err != nil {
					return nil, nil, nil, err
				}
				cluster.AddMember(m)
			case raftpb.ConfChangeRemoveNode:
				id := types.ID(cc.NodeID)
				cluster.RemoveMember(id)
			case raftpb.ConfChangeUpdateNode:
				m := new(membership.Member)
				if err := json.Unmarshal(cc.Context, m); err != nil {
					return nil, nil, nil, err
				}
				cluster.UpdateRaftAttributes(m.ID, m.RaftAttributes)
			}
			continue
		}

		var raftReq pb.InternalRaftRequest
		if !pbutil.MaybeUnmarshal(&raftReq, ent.Data) { // backward compatible
			var r pb.Request
			pbutil.MustUnmarshal(&r, ent.Data)
			applyRequest(&r, applier)
		} else {
			if raftReq.V2 != nil {
				req := raftReq.V2
				applyRequest(req, applier)
			}
		}
	}

	return meta, &hardstate, st, nil
}

func toTTLOptions(r *pb.Request) store.TTLOptionSet {
	refresh, _ := pbutil.GetBool(r.Refresh)
	ttlOptions := store.TTLOptionSet{Refresh: refresh}
	if r.Expiration != 0 {
		ttlOptions.ExpireTime = time.Unix(0, r.Expiration)
	}
	return ttlOptions
}

func applyRequest(r *pb.Request, applyV2 etcdserver.ApplierV2) {
	toTTLOptions(r)
	switch r.Method {
	case "PUT":
		applyV2.Put(r)
	case "DELETE":
		applyV2.Delete(r)
	case "POST", "QGET", "SYNC":
		return
	default:
		glog.Fatal("unknown command")
	}
}
