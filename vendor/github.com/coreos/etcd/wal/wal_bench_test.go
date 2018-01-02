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

package wal

import (
	"io/ioutil"
	"os"
	"testing"

	"github.com/coreos/etcd/raft/raftpb"
)

func BenchmarkWrite100EntryWithoutBatch(b *testing.B) { benchmarkWriteEntry(b, 100, 0) }
func BenchmarkWrite100EntryBatch10(b *testing.B)      { benchmarkWriteEntry(b, 100, 10) }
func BenchmarkWrite100EntryBatch100(b *testing.B)     { benchmarkWriteEntry(b, 100, 100) }
func BenchmarkWrite100EntryBatch500(b *testing.B)     { benchmarkWriteEntry(b, 100, 500) }
func BenchmarkWrite100EntryBatch1000(b *testing.B)    { benchmarkWriteEntry(b, 100, 1000) }

func BenchmarkWrite1000EntryWithoutBatch(b *testing.B) { benchmarkWriteEntry(b, 1000, 0) }
func BenchmarkWrite1000EntryBatch10(b *testing.B)      { benchmarkWriteEntry(b, 1000, 10) }
func BenchmarkWrite1000EntryBatch100(b *testing.B)     { benchmarkWriteEntry(b, 1000, 100) }
func BenchmarkWrite1000EntryBatch500(b *testing.B)     { benchmarkWriteEntry(b, 1000, 500) }
func BenchmarkWrite1000EntryBatch1000(b *testing.B)    { benchmarkWriteEntry(b, 1000, 1000) }

func benchmarkWriteEntry(b *testing.B, size int, batch int) {
	p, err := ioutil.TempDir(os.TempDir(), "waltest")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(p)

	w, err := Create(p, []byte("somedata"))
	if err != nil {
		b.Fatalf("err = %v, want nil", err)
	}
	data := make([]byte, size)
	for i := 0; i < size; i++ {
		data[i] = byte(i)
	}
	e := &raftpb.Entry{Data: data}

	b.ResetTimer()
	n := 0
	b.SetBytes(int64(e.Size()))
	for i := 0; i < b.N; i++ {
		err := w.saveEntry(e)
		if err != nil {
			b.Fatal(err)
		}
		n++
		if n > batch {
			w.sync()
			n = 0
		}
	}
}
