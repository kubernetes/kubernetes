// Copyright 2016 CoreOS, Inc.
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
	"fmt"
	"io"
	"os"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/mirror"
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	"github.com/spf13/cobra"
	"golang.org/x/net/context"
)

// NewSnapshotCommand returns the cobra command for "snapshot".
func NewSnapshotCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "snapshot [filename]",
		Short: "Snapshot streams a point-in-time snapshot of the store",
		Run:   snapshotCommandFunc,
	}
}

// snapshotCommandFunc watches for the length of the entire store and records
// to a file.
func snapshotCommandFunc(cmd *cobra.Command, args []string) {
	switch {
	case len(args) == 0:
		snapshotToStdout(mustClientFromCmd(cmd))
	case len(args) == 1:
		snapshotToFile(mustClientFromCmd(cmd), args[0])
	default:
		err := fmt.Errorf("snapshot takes at most one argument")
		ExitWithError(ExitBadArgs, err)
	}
}

// snapshotToStdout streams a snapshot over stdout
func snapshotToStdout(c *clientv3.Client) {
	// must explicitly fetch first revision since no retry on stdout
	wr := <-c.Watch(context.TODO(), "", clientv3.WithPrefix(), clientv3.WithRev(1))
	if wr.Err() == nil {
		wr.CompactRevision = 1
	}
	if rev := snapshot(os.Stdout, c, wr.CompactRevision+1); rev != 0 {
		err := fmt.Errorf("snapshot interrupted by compaction %v", rev)
		ExitWithError(ExitInterrupted, err)
	}
	os.Stdout.Sync()
}

// snapshotToFile atomically writes a snapshot to a file
func snapshotToFile(c *clientv3.Client, path string) {
	partpath := path + ".part"
	f, err := os.Create(partpath)
	defer f.Close()
	if err != nil {
		exiterr := fmt.Errorf("could not open %s (%v)", partpath, err)
		ExitWithError(ExitBadArgs, exiterr)
	}
	rev := int64(1)
	for rev != 0 {
		f.Seek(0, 0)
		f.Truncate(0)
		rev = snapshot(f, c, rev)
	}
	f.Sync()
	if err := os.Rename(partpath, path); err != nil {
		exiterr := fmt.Errorf("could not rename %s to %s (%v)", partpath, path, err)
		ExitWithError(ExitIO, exiterr)
	}
}

// snapshot reads all of a watcher; returns compaction revision if incomplete
// TODO: stabilize snapshot format
func snapshot(w io.Writer, c *clientv3.Client, rev int64) int64 {
	s := mirror.NewSyncer(c, "", rev)

	rc, errc := s.SyncBase(context.TODO())

	for r := range rc {
		for _, kv := range r.Kvs {
			fmt.Fprintln(w, kv)
		}
	}

	err := <-errc
	if err != nil {
		if err == rpctypes.ErrCompacted {
			// will get correct compact revision on retry
			return rev + 1
		}
		// failed for some unknown reason, retry on same revision
		return rev
	}

	wc := s.SyncUpdates(context.TODO())

	for wr := range wc {
		if wr.Err() != nil {
			return wr.CompactRevision
		}
		for _, ev := range wr.Events {
			fmt.Fprintln(w, ev)
		}
		rev := wr.Events[len(wr.Events)-1].Kv.ModRevision
		if rev >= wr.Header.Revision {
			break
		}
	}

	return 0
}
