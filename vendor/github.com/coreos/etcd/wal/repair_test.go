// Copyright 2015 CoreOS, Inc.
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

package wal

import (
	"io"
	"io/ioutil"
	"os"
	"testing"

	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/wal/walpb"
)

func TestRepair(t *testing.T) {
	p, err := ioutil.TempDir(os.TempDir(), "waltest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(p)
	// create WAL
	w, err := Create(p, nil)
	defer w.Close()
	if err != nil {
		t.Fatal(err)
	}

	n := 10
	for i := 1; i <= n; i++ {
		es := []raftpb.Entry{{Index: uint64(i)}}
		if err = w.Save(raftpb.HardState{}, es); err != nil {
			t.Fatal(err)
		}
	}
	w.Close()

	// break the wal.
	f, err := openLast(p)
	if err != nil {
		t.Fatal(err)
	}
	offset, err := f.Seek(-4, os.SEEK_END)
	if err != nil {
		t.Fatal(err)
	}
	err = f.Truncate(offset)
	if err != nil {
		t.Fatal(err)
	}

	// verify we have broke the wal
	w, err = Open(p, walpb.Snapshot{})
	if err != nil {
		t.Fatal(err)
	}
	_, _, _, err = w.ReadAll()
	if err != io.ErrUnexpectedEOF {
		t.Fatalf("err = %v, want %v", err, io.ErrUnexpectedEOF)
	}
	w.Close()

	// repair the wal
	ok := Repair(p)
	if !ok {
		t.Fatalf("fix = %t, want %t", ok, true)
	}

	w, err = Open(p, walpb.Snapshot{})
	if err != nil {
		t.Fatal(err)
	}
	_, _, ents, err := w.ReadAll()
	if err != nil {
		t.Fatalf("err = %v, want %v", err, nil)
	}
	if len(ents) != n-1 {
		t.Fatalf("len(ents) = %d, want %d", len(ents), n-1)
	}
}
