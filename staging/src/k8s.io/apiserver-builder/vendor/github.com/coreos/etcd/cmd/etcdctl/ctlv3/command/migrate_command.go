// Copyright 2016 The etcd Authors
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

package command

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"time"

	"github.com/coreos/etcd/client"
	etcdErr "github.com/coreos/etcd/error"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/etcdserver/membership"
	"github.com/coreos/etcd/mvcc"
	"github.com/coreos/etcd/mvcc/backend"
	"github.com/coreos/etcd/mvcc/mvccpb"
	"github.com/coreos/etcd/pkg/pbutil"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/snap"
	"github.com/coreos/etcd/store"
	"github.com/coreos/etcd/wal"
	"github.com/coreos/etcd/wal/walpb"
	"github.com/gogo/protobuf/proto"
	"github.com/spf13/cobra"
)

var (
	migrateExcludeTTLKey bool
	migrateDatadir       string
	migrateWALdir        string
	migrateTransformer   string
)

// NewMigrateCommand returns the cobra command for "migrate".
func NewMigrateCommand() *cobra.Command {
	mc := &cobra.Command{
		Use:   "migrate",
		Short: "Migrates keys in a v2 store to a mvcc store",
		Run:   migrateCommandFunc,
	}

	mc.Flags().BoolVar(&migrateExcludeTTLKey, "no-ttl", false, "Do not convert TTL keys")
	mc.Flags().StringVar(&migrateDatadir, "data-dir", "", "Path to the data directory")
	mc.Flags().StringVar(&migrateWALdir, "wal-dir", "", "Path to the WAL directory")
	mc.Flags().StringVar(&migrateTransformer, "transformer", "", "Path to the user-provided transformer program")
	return mc
}

func migrateCommandFunc(cmd *cobra.Command, args []string) {
	var (
		writer io.WriteCloser
		reader io.ReadCloser
		errc   chan error
	)
	if migrateTransformer != "" {
		writer, reader, errc = startTransformer()
	} else {
		fmt.Println("using default transformer")
		writer, reader, errc = defaultTransformer()
	}

	st, index := rebuildStoreV2()
	be := prepareBackend()
	defer be.Close()

	go func() {
		writeStore(writer, st)
		writer.Close()
	}()

	readKeys(reader, be)
	mvcc.UpdateConsistentIndex(be, index)
	err := <-errc
	if err != nil {
		fmt.Println("failed to transform keys")
		ExitWithError(ExitError, err)
	}

	fmt.Println("finished transforming keys")
}

func prepareBackend() backend.Backend {
	dbpath := path.Join(migrateDatadir, "member", "snap", "db")
	be := backend.New(dbpath, time.Second, 10000)
	tx := be.BatchTx()
	tx.Lock()
	tx.UnsafeCreateBucket([]byte("key"))
	tx.UnsafeCreateBucket([]byte("meta"))
	tx.Unlock()
	return be
}

func rebuildStoreV2() (store.Store, uint64) {
	var index uint64
	cl := membership.NewCluster("")

	waldir := migrateWALdir
	if len(waldir) == 0 {
		waldir = path.Join(migrateDatadir, "member", "wal")
	}
	snapdir := path.Join(migrateDatadir, "member", "snap")

	ss := snap.New(snapdir)
	snapshot, err := ss.Load()
	if err != nil && err != snap.ErrNoSnapshot {
		ExitWithError(ExitError, err)
	}

	var walsnap walpb.Snapshot
	if snapshot != nil {
		walsnap.Index, walsnap.Term = snapshot.Metadata.Index, snapshot.Metadata.Term
		index = snapshot.Metadata.Index
	}

	w, err := wal.OpenForRead(waldir, walsnap)
	if err != nil {
		ExitWithError(ExitError, err)
	}
	defer w.Close()

	_, _, ents, err := w.ReadAll()
	if err != nil {
		ExitWithError(ExitError, err)
	}

	st := store.New()
	if snapshot != nil {
		err := st.Recovery(snapshot.Data)
		if err != nil {
			ExitWithError(ExitError, err)
		}
	}

	cl.SetStore(st)
	cl.Recover(api.UpdateCapability)

	applier := etcdserver.NewApplierV2(st, cl)
	for _, ent := range ents {
		if ent.Type == raftpb.EntryConfChange {
			var cc raftpb.ConfChange
			pbutil.MustUnmarshal(&cc, ent.Data)
			applyConf(cc, cl)
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
		if ent.Index > index {
			index = ent.Index
		}
	}

	return st, index
}

func applyConf(cc raftpb.ConfChange, cl *membership.RaftCluster) {
	if err := cl.ValidateConfigurationChange(cc); err != nil {
		return
	}
	switch cc.Type {
	case raftpb.ConfChangeAddNode:
		m := new(membership.Member)
		if err := json.Unmarshal(cc.Context, m); err != nil {
			panic(err)
		}
		cl.AddMember(m)
	case raftpb.ConfChangeRemoveNode:
		cl.RemoveMember(types.ID(cc.NodeID))
	case raftpb.ConfChangeUpdateNode:
		m := new(membership.Member)
		if err := json.Unmarshal(cc.Context, m); err != nil {
			panic(err)
		}
		cl.UpdateRaftAttributes(m.ID, m.RaftAttributes)
	}
}

func applyRequest(r *pb.Request, applyV2 etcdserver.ApplierV2) {
	toTTLOptions(r)
	switch r.Method {
	case "POST":
		applyV2.Post(r)
	case "PUT":
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

func toTTLOptions(r *pb.Request) store.TTLOptionSet {
	refresh, _ := pbutil.GetBool(r.Refresh)
	ttlOptions := store.TTLOptionSet{Refresh: refresh}
	if r.Expiration != 0 {
		ttlOptions.ExpireTime = time.Unix(0, r.Expiration)
	}
	return ttlOptions
}

func writeStore(w io.Writer, st store.Store) uint64 {
	all, err := st.Get("/1", true, true)
	if err != nil {
		if eerr, ok := err.(*etcdErr.Error); ok && eerr.ErrorCode == etcdErr.EcodeKeyNotFound {
			fmt.Println("no v2 keys to migrate")
			os.Exit(0)
		}
		ExitWithError(ExitError, err)
	}
	return writeKeys(w, all.Node)
}

func writeKeys(w io.Writer, n *store.NodeExtern) uint64 {
	maxIndex := n.ModifiedIndex

	nodes := n.Nodes
	// remove store v2 bucket prefix
	n.Key = n.Key[2:]
	if n.Key == "" {
		n.Key = "/"
	}
	if n.Dir {
		n.Nodes = nil
	}
	if !migrateExcludeTTLKey || n.TTL == 0 {
		b, err := json.Marshal(n)
		if err != nil {
			ExitWithError(ExitError, err)
		}
		fmt.Fprint(w, string(b))
	}
	for _, nn := range nodes {
		max := writeKeys(w, nn)
		if max > maxIndex {
			maxIndex = max
		}
	}
	return maxIndex
}

func readKeys(r io.Reader, be backend.Backend) error {
	for {
		length64, err := readInt64(r)
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}

		buf := make([]byte, int(length64))
		if _, err = io.ReadFull(r, buf); err != nil {
			return err
		}

		var kv mvccpb.KeyValue
		err = proto.Unmarshal(buf, &kv)
		if err != nil {
			return err
		}

		mvcc.WriteKV(be, kv)
	}
}

func readInt64(r io.Reader) (int64, error) {
	var n int64
	err := binary.Read(r, binary.LittleEndian, &n)
	return n, err
}

func startTransformer() (io.WriteCloser, io.ReadCloser, chan error) {
	cmd := exec.Command(migrateTransformer)
	cmd.Stderr = os.Stderr

	writer, err := cmd.StdinPipe()
	if err != nil {
		ExitWithError(ExitError, err)
	}

	reader, rerr := cmd.StdoutPipe()
	if rerr != nil {
		ExitWithError(ExitError, rerr)
	}

	if err := cmd.Start(); err != nil {
		ExitWithError(ExitError, err)
	}

	errc := make(chan error, 1)

	go func() {
		errc <- cmd.Wait()
	}()

	return writer, reader, errc
}

func defaultTransformer() (io.WriteCloser, io.ReadCloser, chan error) {
	// transformer decodes v2 keys from sr
	sr, sw := io.Pipe()
	// transformer encodes v3 keys into dw
	dr, dw := io.Pipe()

	decoder := json.NewDecoder(sr)

	errc := make(chan error, 1)

	go func() {
		defer func() {
			sr.Close()
			dw.Close()
		}()

		for decoder.More() {
			node := &client.Node{}
			if err := decoder.Decode(node); err != nil {
				errc <- err
				return
			}

			kv := transform(node)
			if kv == nil {
				continue
			}

			data, err := proto.Marshal(kv)
			if err != nil {
				errc <- err
				return
			}
			buf := make([]byte, 8)
			binary.LittleEndian.PutUint64(buf, uint64(len(data)))
			if _, err := dw.Write(buf); err != nil {
				errc <- err
				return
			}
			if _, err := dw.Write(data); err != nil {
				errc <- err
				return
			}
		}

		errc <- nil
	}()

	return sw, dr, errc
}

func transform(n *client.Node) *mvccpb.KeyValue {
	const unKnownVersion = 1
	if n.Dir {
		return nil
	}
	kv := &mvccpb.KeyValue{
		Key:            []byte(n.Key),
		Value:          []byte(n.Value),
		CreateRevision: int64(n.CreatedIndex),
		ModRevision:    int64(n.ModifiedIndex),
		Version:        unKnownVersion,
	}
	return kv
}
