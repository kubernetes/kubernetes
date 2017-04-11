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

package mvcc

import (
	"sync/atomic"
	"testing"

	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/mvcc/backend"
)

type fakeConsistentIndex uint64

func (i *fakeConsistentIndex) ConsistentIndex() uint64 {
	return atomic.LoadUint64((*uint64)(i))
}

func BenchmarkStorePut(b *testing.B) {
	var i fakeConsistentIndex
	be, tmpPath := backend.NewDefaultTmpBackend()
	s := NewStore(be, &lease.FakeLessor{}, &i)
	defer cleanup(s, be, tmpPath)

	// arbitrary number of bytes
	bytesN := 64
	keys := createBytesSlice(bytesN, b.N)
	vals := createBytesSlice(bytesN, b.N)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Put(keys[i], vals[i], lease.NoLease)
	}
}

// BenchmarkStoreTxnPut benchmarks the Put operation
// with transaction begin and end, where transaction involves
// some synchronization operations, such as mutex locking.
func BenchmarkStoreTxnPut(b *testing.B) {
	var i fakeConsistentIndex
	be, tmpPath := backend.NewDefaultTmpBackend()
	s := NewStore(be, &lease.FakeLessor{}, &i)
	defer cleanup(s, be, tmpPath)

	// arbitrary number of bytes
	bytesN := 64
	keys := createBytesSlice(bytesN, b.N)
	vals := createBytesSlice(bytesN, b.N)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := s.TxnBegin()
		if _, err := s.TxnPut(id, keys[i], vals[i], lease.NoLease); err != nil {
			plog.Fatalf("txn put error: %v", err)
		}
		s.TxnEnd(id)
	}
}
