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

package e2e

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
)

func TestCtlV3Snapshot(t *testing.T) { testCtl(t, snapshotTest) }

func snapshotTest(cx ctlCtx) {
	var kvs = []kv{{"key", "val1"}, {"key", "val2"}, {"key", "val3"}}
	for i := range kvs {
		if err := ctlV3Put(cx, kvs[i].key, kvs[i].val, ""); err != nil {
			cx.t.Fatal(err)
		}
	}

	leaseID, err := ctlV3LeaseGrant(cx, 100)
	if err != nil {
		cx.t.Fatalf("snapshot: ctlV3LeaseGrant error (%v)", err)
	}
	if err = ctlV3Put(cx, "withlease", "withlease", leaseID); err != nil {
		cx.t.Fatalf("snapshot: ctlV3Put error (%v)", err)
	}

	fpath := "test.snapshot"
	defer os.RemoveAll(fpath)

	if err = ctlV3SnapshotSave(cx, fpath); err != nil {
		cx.t.Fatalf("snapshotTest ctlV3SnapshotSave error (%v)", err)
	}

	st, err := getSnapshotStatus(cx, fpath)
	if err != nil {
		cx.t.Fatalf("snapshotTest getSnapshotStatus error (%v)", err)
	}
	if st.Revision != 5 {
		cx.t.Fatalf("expected 4, got %d", st.Revision)
	}
	if st.TotalKey < 4 {
		cx.t.Fatalf("expected at least 4, got %d", st.TotalKey)
	}
}

func TestCtlV3SnapshotCorrupt(t *testing.T) { testCtl(t, snapshotCorruptTest) }

func snapshotCorruptTest(cx ctlCtx) {
	fpath := "test.snapshot"
	defer os.RemoveAll(fpath)

	if err := ctlV3SnapshotSave(cx, fpath); err != nil {
		cx.t.Fatalf("snapshotTest ctlV3SnapshotSave error (%v)", err)
	}

	// corrupt file
	f, oerr := os.OpenFile(fpath, os.O_WRONLY, 0)
	if oerr != nil {
		cx.t.Fatal(oerr)
	}
	if _, err := f.Write(make([]byte, 512)); err != nil {
		cx.t.Fatal(err)
	}
	f.Close()

	defer os.RemoveAll("snap.etcd")
	serr := spawnWithExpect(
		append(cx.PrefixArgs(), "snapshot", "restore",
			"--data-dir", "snap.etcd",
			fpath),
		"expected sha256")

	if serr != nil {
		cx.t.Fatal(serr)
	}
}

func ctlV3SnapshotSave(cx ctlCtx, fpath string) error {
	cmdArgs := append(cx.PrefixArgs(), "snapshot", "save", fpath)
	return spawnWithExpect(cmdArgs, fmt.Sprintf("Snapshot saved at %s", fpath))
}

type snapshotStatus struct {
	Hash      uint32 `json:"hash"`
	Revision  int64  `json:"revision"`
	TotalKey  int    `json:"totalKey"`
	TotalSize int64  `json:"totalSize"`
}

func getSnapshotStatus(cx ctlCtx, fpath string) (snapshotStatus, error) {
	cmdArgs := append(cx.PrefixArgs(), "--write-out", "json", "snapshot", "status", fpath)

	proc, err := spawnCmd(cmdArgs)
	if err != nil {
		return snapshotStatus{}, err
	}
	var txt string
	txt, err = proc.Expect("totalKey")
	if err != nil {
		return snapshotStatus{}, err
	}
	if err = proc.Close(); err != nil {
		return snapshotStatus{}, err
	}

	resp := snapshotStatus{}
	dec := json.NewDecoder(strings.NewReader(txt))
	if err := dec.Decode(&resp); err == io.EOF {
		return snapshotStatus{}, err
	}
	return resp, nil
}
