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

package storage

import (
	"testing"

	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/storage/backend"
)

type indexVal uint64

func (v *indexVal) ConsistentIndex() uint64 { return uint64(*v) }

func TestConsistentWatchableStoreConsistentIndex(t *testing.T) {
	var idx indexVal
	b, tmpPath := backend.NewDefaultTmpBackend()
	s := newConsistentWatchableStore(b, &lease.FakeLessor{}, &idx)
	defer cleanup(s, b, tmpPath)

	tests := []uint64{1, 2, 3, 5, 10}
	for i, tt := range tests {
		idx = indexVal(tt)
		s.Put([]byte("foo"), []byte("bar"), lease.NoLease)

		id := s.TxnBegin()
		g := s.consistentIndex()
		s.TxnEnd(id)
		if g != tt {
			t.Errorf("#%d: index = %d, want %d", i, g, tt)
		}
	}
}

func TestConsistentWatchableStoreSkip(t *testing.T) {
	idx := indexVal(5)
	b, tmpPath := backend.NewDefaultTmpBackend()
	s := newConsistentWatchableStore(b, &lease.FakeLessor{}, &idx)
	defer cleanup(s, b, tmpPath)

	s.Put([]byte("foo"), []byte("bar"), lease.NoLease)

	// put is skipped
	rev := s.Put([]byte("foo"), []byte("bar"), lease.NoLease)
	if rev != 0 {
		t.Errorf("rev = %d, want 0", rev)
	}
}
