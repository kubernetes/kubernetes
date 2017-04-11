// Copyright 2015 The etcd Authors
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
	"log"
	"path"
	"time"

	"github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/pkg/fileutil"
	"github.com/coreos/etcd/pkg/idutil"
	"github.com/coreos/etcd/pkg/pbutil"
	"github.com/coreos/etcd/snap"
	"github.com/coreos/etcd/wal"
	"github.com/coreos/etcd/wal/walpb"
	"github.com/urfave/cli"
)

func NewBackupCommand() cli.Command {
	return cli.Command{
		Name:      "backup",
		Usage:     "backup an etcd directory",
		ArgsUsage: " ",
		Flags: []cli.Flag{
			cli.StringFlag{Name: "data-dir", Value: "", Usage: "Path to the etcd data dir"},
			cli.StringFlag{Name: "wal-dir", Value: "", Usage: "Path to the etcd wal dir"},
			cli.StringFlag{Name: "backup-dir", Value: "", Usage: "Path to the backup dir"},
			cli.StringFlag{Name: "backup-wal-dir", Value: "", Usage: "Path to the backup wal dir"},
		},
		Action: handleBackup,
	}
}

// handleBackup handles a request that intends to do a backup.
func handleBackup(c *cli.Context) error {
	var srcWAL string
	var destWAL string

	srcSnap := path.Join(c.String("data-dir"), "member", "snap")
	destSnap := path.Join(c.String("backup-dir"), "member", "snap")

	if c.String("wal-dir") != "" {
		srcWAL = c.String("wal-dir")
	} else {
		srcWAL = path.Join(c.String("data-dir"), "member", "wal")
	}

	if c.String("backup-wal-dir") != "" {
		destWAL = c.String("backup-wal-dir")
	} else {
		destWAL = path.Join(c.String("backup-dir"), "member", "wal")
	}

	if err := fileutil.CreateDirAll(destSnap); err != nil {
		log.Fatalf("failed creating backup snapshot dir %v: %v", destSnap, err)
	}
	ss := snap.New(srcSnap)
	snapshot, err := ss.Load()
	if err != nil && err != snap.ErrNoSnapshot {
		log.Fatal(err)
	}
	var walsnap walpb.Snapshot
	if snapshot != nil {
		walsnap.Index, walsnap.Term = snapshot.Metadata.Index, snapshot.Metadata.Term
		newss := snap.New(destSnap)
		if err = newss.SaveSnap(*snapshot); err != nil {
			log.Fatal(err)
		}
	}

	w, err := wal.OpenForRead(srcWAL, walsnap)
	if err != nil {
		log.Fatal(err)
	}
	defer w.Close()
	wmetadata, state, ents, err := w.ReadAll()
	switch err {
	case nil:
	case wal.ErrSnapshotNotFound:
		fmt.Printf("Failed to find the match snapshot record %+v in wal %v.", walsnap, srcWAL)
		fmt.Printf("etcdctl will add it back. Start auto fixing...")
	default:
		log.Fatal(err)
	}
	var metadata etcdserverpb.Metadata
	pbutil.MustUnmarshal(&metadata, wmetadata)
	idgen := idutil.NewGenerator(0, time.Now())
	metadata.NodeID = idgen.Next()
	metadata.ClusterID = idgen.Next()

	neww, err := wal.Create(destWAL, pbutil.MustMarshal(&metadata))
	if err != nil {
		log.Fatal(err)
	}
	defer neww.Close()
	if err := neww.Save(state, ents); err != nil {
		log.Fatal(err)
	}
	if err := neww.SaveSnapshot(walsnap); err != nil {
		log.Fatal(err)
	}

	return nil
}
