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
	"log"
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
)

var (
	migrateDatadir string
	ttl            time.Duration
)

func init() {
	flag.StringVar(&migrateDatadir, "data-dir", "", "Path to the data directory")
	flag.DurationVar(&ttl, "ttl", time.Hour, "TTL of event keys (default 1 hour)")
	flag.Parse()
}

func main() {
	dbpath := path.Join(migrateDatadir, "member", "snap", "db")

	be := backend.New(dbpath, time.Second, 10000)
	tx := be.BatchTx()

	st := store.New(etcdserver.StoreClusterPrefix, etcdserver.StoreKeysPrefix)
	expireTime := time.Now().Add(ttl)

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
			_, err := st.Set(path.Join("1", string(kv.Key)), false, string(kv.Value), ttlOpt)
			if err != nil {
				return err
			}
		} else {
			st.Delete(string(kv.Key), false, false)
		}

		return nil
	})
	if err != nil {
		panic(err)
	}
	tx.Unlock()

	traverseAndDeleteEmptyDir(st, "/")

	metadata, hardstate, oldSt := rebuild(migrateDatadir)

	// In the following, it's low level logic that saves metadata and data into v2 snapshot.

	if err := os.RemoveAll(migrateDatadir); err != nil {
		panic(err)
	}
	if err := os.MkdirAll(path.Join(migrateDatadir, "member", "snap"), 0700); err != nil {
		panic(err)
	}
	walDir := path.Join(migrateDatadir, "member", "wal")

	w, err := wal.Create(walDir, metadata)
	if err != nil {
		panic(err)
	}
	err = w.SaveSnapshot(walpb.Snapshot{Index: hardstate.Commit, Term: hardstate.Term})
	if err != nil {
		panic(err)
	}
	w.Close()

	nodeIDs := []uint64{}
	event, err := oldSt.Get(etcdserver.StoreClusterPrefix, true, false)
	if err != nil {
		panic(err)
	}
	// searching all metadata nodes and:
	// - update store
	// - update Nodes for ConfState
	q := []*store.NodeExtern{}
	q = append(q, event.Node)
	for len(q) > 0 {
		n := q[0]
		q = q[1:]
		if n.Key != etcdserver.StoreClusterPrefix {
			v := ""
			if !n.Dir {
				v = *n.Value
			}
			if n.Key == path.Join(etcdserver.StoreClusterPrefix, "version") {
				v = "2.3.7"
			}
			if _, err := st.Set(n.Key, n.Dir, v, store.TTLOptionSet{}); err != nil {
				panic(err)
			}

			fields := strings.Split(n.Key, "/")
			if len(fields) == 4 && fields[2] == "members" {
				nodeID, err := strconv.ParseUint(fields[3], 16, 64)
				if err != nil {
					fmt.Println("wrong ID: %s", fields[3])
					panic(err)
				}
				nodeIDs = append(nodeIDs, nodeID)
			}
		}
		for _, next := range n.Nodes {
			q = append(q, next)
		}
	}

	data, err := st.Save()
	if err != nil {
		panic(err)
	}
	raftSnap := raftpb.Snapshot{
		Data: data,
		Metadata: raftpb.SnapshotMetadata{
			Index: hardstate.Commit,
			Term:  hardstate.Term,
			ConfState: raftpb.ConfState{
				Nodes: nodeIDs,
			},
		},
	}
	snapshotter := snap.New(path.Join(migrateDatadir, "member", "snap"))
	if err := snapshotter.SaveSnap(raftSnap); err != nil {
		panic(err)
	}
	fmt.Println("Finished.")
}

const (
	revBytesLen            = 8 + 1 + 8
	markedRevBytesLen      = revBytesLen + 1
	markBytePosition       = markedRevBytesLen - 1
	markTombstone     byte = 't'
)

func isTombstone(b []byte) bool {
	return len(b) == markedRevBytesLen && b[markBytePosition] == markTombstone
}

func traverseAndDeleteEmptyDir(st store.Store, dir string) {
	e, err := st.Get(dir, true, false)
	if err != nil {
		panic(err)
	}
	if len(e.Node.Nodes) == 0 {
		st.Delete(dir, true, true)
		return
	}
	for _, node := range e.Node.Nodes {
		if !node.Dir {
			printNode(node)
		} else {
			traverseAndDeleteEmptyDir(st, node.Key)
		}
	}
}

func printNode(node *store.NodeExtern) {
	fmt.Printf("key:%s\n", node.Key[len("/1"):])
}

func rebuild(datadir string) ([]byte, raftpb.HardState, store.Store) {
	waldir := path.Join(datadir, "member", "wal")
	snapdir := path.Join(datadir, "member", "snap")

	ss := snap.New(snapdir)
	snapshot, err := ss.Load()
	if err != nil && err != snap.ErrNoSnapshot {
		panic(err)
	}

	var walsnap walpb.Snapshot
	if snapshot != nil {
		walsnap.Index, walsnap.Term = snapshot.Metadata.Index, snapshot.Metadata.Term
	}

	w, err := wal.OpenForRead(waldir, walsnap)
	if err != nil {
		panic(err)
	}
	defer w.Close()

	meta, hardstate, ents, err := w.ReadAll()
	if err != nil {
		panic(err)
	}

	st := store.New(etcdserver.StoreClusterPrefix, etcdserver.StoreKeysPrefix)
	if snapshot != nil {
		err := st.Recovery(snapshot.Data)
		if err != nil {
			panic(err)
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
					log.Panicf("unmarshal member should never fail: %v", err)
				}
				cluster.AddMember(m)
			case raftpb.ConfChangeRemoveNode:
				id := types.ID(cc.NodeID)
				cluster.RemoveMember(id)
			case raftpb.ConfChangeUpdateNode:
				m := new(membership.Member)
				if err := json.Unmarshal(cc.Context, m); err != nil {
					log.Panicf("unmarshal member should never fail: %v", err)
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

	return meta, hardstate, st
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
	case "POST":
		applyV2.Post(r)
	case "PUT":
		// fmt.Println("put", r.Path)
		applyV2.Put(r)
	case "DELETE":
		applyV2.Delete(r)
	case "QGET":
		applyV2.QGet(r)
	case "SYNC":
		applyV2.Sync(r)
	default:
		panic("unknown command")
	}
}
