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
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/mvcc/backend"
)

func TestScheduleCompaction(t *testing.T) {
	revs := []revision{{1, 0}, {2, 0}, {3, 0}}

	tests := []struct {
		rev   int64
		keep  map[revision]struct{}
		wrevs []revision
	}{
		// compact at 1 and discard all history
		{
			1,
			nil,
			revs[1:],
		},
		// compact at 3 and discard all history
		{
			3,
			nil,
			nil,
		},
		// compact at 1 and keeps history one step earlier
		{
			1,
			map[revision]struct{}{
				{main: 1}: {},
			},
			revs,
		},
		// compact at 1 and keeps history two steps earlier
		{
			3,
			map[revision]struct{}{
				{main: 2}: {},
				{main: 3}: {},
			},
			revs[1:],
		},
	}
	for i, tt := range tests {
		b, tmpPath := backend.NewDefaultTmpBackend()
		s := NewStore(b, &lease.FakeLessor{}, nil)
		tx := s.b.BatchTx()

		tx.Lock()
		ibytes := newRevBytes()
		for _, rev := range revs {
			revToBytes(rev, ibytes)
			tx.UnsafePut(keyBucketName, ibytes, []byte("bar"))
		}
		tx.Unlock()

		s.scheduleCompaction(tt.rev, tt.keep)

		tx.Lock()
		for _, rev := range tt.wrevs {
			revToBytes(rev, ibytes)
			keys, _ := tx.UnsafeRange(keyBucketName, ibytes, nil, 0)
			if len(keys) != 1 {
				t.Errorf("#%d: range on %v = %d, want 1", i, rev, len(keys))
			}
		}
		_, vals := tx.UnsafeRange(metaBucketName, finishedCompactKeyName, nil, 0)
		revToBytes(revision{main: tt.rev}, ibytes)
		if w := [][]byte{ibytes}; !reflect.DeepEqual(vals, w) {
			t.Errorf("#%d: vals on %v = %+v, want %+v", i, finishedCompactKeyName, vals, w)
		}
		tx.Unlock()

		cleanup(s, b, tmpPath)
	}
}

func TestCompactAllAndRestore(t *testing.T) {
	b, tmpPath := backend.NewDefaultTmpBackend()
	s0 := NewStore(b, &lease.FakeLessor{}, nil)
	defer os.Remove(tmpPath)

	s0.Put([]byte("foo"), []byte("bar"), lease.NoLease)
	s0.Put([]byte("foo"), []byte("bar1"), lease.NoLease)
	s0.Put([]byte("foo"), []byte("bar2"), lease.NoLease)
	s0.DeleteRange([]byte("foo"), nil)

	rev := s0.Rev()
	// compact all keys
	done, err := s0.Compact(rev)
	if err != nil {
		t.Fatal(err)
	}

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for compaction to finish")
	}

	err = s0.Close()
	if err != nil {
		t.Fatal(err)
	}

	s1 := NewStore(b, &lease.FakeLessor{}, nil)
	if s1.Rev() != rev {
		t.Errorf("rev = %v, want %v", s1.Rev(), rev)
	}
	_, err = s1.Range([]byte("foo"), nil, RangeOptions{})
	if err != nil {
		t.Errorf("unexpect range error %v", err)
	}
}
