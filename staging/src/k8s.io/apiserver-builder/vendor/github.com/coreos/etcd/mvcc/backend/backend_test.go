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

package backend

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"
	"time"

	"github.com/boltdb/bolt"
)

func TestBackendClose(t *testing.T) {
	b, tmpPath := NewTmpBackend(time.Hour, 10000)
	defer os.Remove(tmpPath)

	// check close could work
	done := make(chan struct{})
	go func() {
		err := b.Close()
		if err != nil {
			t.Errorf("close error = %v, want nil", err)
		}
		done <- struct{}{}
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Errorf("failed to close database in 10s")
	}
}

func TestBackendSnapshot(t *testing.T) {
	b, tmpPath := NewTmpBackend(time.Hour, 10000)
	defer cleanup(b, tmpPath)

	tx := b.BatchTx()
	tx.Lock()
	tx.UnsafeCreateBucket([]byte("test"))
	tx.UnsafePut([]byte("test"), []byte("foo"), []byte("bar"))
	tx.Unlock()
	b.ForceCommit()

	// write snapshot to a new file
	f, err := ioutil.TempFile(os.TempDir(), "etcd_backend_test")
	if err != nil {
		t.Fatal(err)
	}
	snap := b.Snapshot()
	defer snap.Close()
	if _, err := snap.WriteTo(f); err != nil {
		t.Fatal(err)
	}
	f.Close()

	// bootstrap new backend from the snapshot
	nb := New(f.Name(), time.Hour, 10000)
	defer cleanup(nb, f.Name())

	newTx := b.BatchTx()
	newTx.Lock()
	ks, _ := newTx.UnsafeRange([]byte("test"), []byte("foo"), []byte("goo"), 0)
	if len(ks) != 1 {
		t.Errorf("len(kvs) = %d, want 1", len(ks))
	}
	newTx.Unlock()
}

func TestBackendBatchIntervalCommit(t *testing.T) {
	// start backend with super short batch interval so
	// we do not need to wait long before commit to happen.
	b, tmpPath := NewTmpBackend(time.Nanosecond, 10000)
	defer cleanup(b, tmpPath)

	pc := b.Commits()

	tx := b.BatchTx()
	tx.Lock()
	tx.UnsafeCreateBucket([]byte("test"))
	tx.UnsafePut([]byte("test"), []byte("foo"), []byte("bar"))
	tx.Unlock()

	for i := 0; i < 10; i++ {
		if b.Commits() >= pc+1 {
			break
		}
		time.Sleep(time.Duration(i*100) * time.Millisecond)
	}

	// check whether put happens via db view
	b.db.View(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte("test"))
		if bucket == nil {
			t.Errorf("bucket test does not exit")
			return nil
		}
		v := bucket.Get([]byte("foo"))
		if v == nil {
			t.Errorf("foo key failed to written in backend")
		}
		return nil
	})
}

func TestBackendDefrag(t *testing.T) {
	b, tmpPath := NewDefaultTmpBackend()
	defer cleanup(b, tmpPath)

	tx := b.BatchTx()
	tx.Lock()
	tx.UnsafeCreateBucket([]byte("test"))
	for i := 0; i < defragLimit+100; i++ {
		tx.UnsafePut([]byte("test"), []byte(fmt.Sprintf("foo_%d", i)), []byte("bar"))
	}
	tx.Unlock()
	b.ForceCommit()

	// remove some keys to ensure the disk space will be reclaimed after defrag
	tx = b.BatchTx()
	tx.Lock()
	for i := 0; i < 50; i++ {
		tx.UnsafeDelete([]byte("test"), []byte(fmt.Sprintf("foo_%d", i)))
	}
	tx.Unlock()
	b.ForceCommit()

	size := b.Size()

	// shrink and check hash
	oh, err := b.Hash(nil)
	if err != nil {
		t.Fatal(err)
	}

	err = b.Defrag()
	if err != nil {
		t.Fatal(err)
	}

	nh, err := b.Hash(nil)
	if err != nil {
		t.Fatal(err)
	}
	if oh != nh {
		t.Errorf("hash = %v, want %v", nh, oh)
	}

	nsize := b.Size()
	if nsize >= size {
		t.Errorf("new size = %v, want < %d", nsize, size)
	}

	// try put more keys after shrink.
	tx = b.BatchTx()
	tx.Lock()
	tx.UnsafeCreateBucket([]byte("test"))
	tx.UnsafePut([]byte("test"), []byte("more"), []byte("bar"))
	tx.Unlock()
	b.ForceCommit()
}

func cleanup(b Backend, path string) {
	b.Close()
	os.Remove(path)
}
