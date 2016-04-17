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

package backend

import (
	"bytes"
	"log"
	"sync"
	"sync/atomic"

	"github.com/boltdb/bolt"
)

type BatchTx interface {
	Lock()
	Unlock()
	UnsafeCreateBucket(name []byte)
	UnsafePut(bucketName []byte, key []byte, value []byte)
	UnsafeSeqPut(bucketName []byte, key []byte, value []byte)
	UnsafeRange(bucketName []byte, key, endKey []byte, limit int64) (keys [][]byte, vals [][]byte)
	UnsafeDelete(bucketName []byte, key []byte)
	UnsafeForEach(bucketName []byte, visitor func(k, v []byte) error) error
	Commit()
	CommitAndStop()
}

type batchTx struct {
	sync.Mutex
	tx      *bolt.Tx
	backend *backend
	pending int
}

func newBatchTx(backend *backend) *batchTx {
	tx := &batchTx{backend: backend}
	tx.Commit()
	return tx
}

func (t *batchTx) UnsafeCreateBucket(name []byte) {
	_, err := t.tx.CreateBucket(name)
	if err != nil && err != bolt.ErrBucketExists {
		log.Fatalf("storage: cannot create bucket %s (%v)", string(name), err)
	}
	t.pending++
}

// UnsafePut must be called holding the lock on the tx.
func (t *batchTx) UnsafePut(bucketName []byte, key []byte, value []byte) {
	t.unsafePut(bucketName, key, value, false)
}

// UnsafeSeqPut must be called holding the lock on the tx.
func (t *batchTx) UnsafeSeqPut(bucketName []byte, key []byte, value []byte) {
	t.unsafePut(bucketName, key, value, true)
}

func (t *batchTx) unsafePut(bucketName []byte, key []byte, value []byte, seq bool) {
	bucket := t.tx.Bucket(bucketName)
	if bucket == nil {
		log.Fatalf("storage: bucket %s does not exist", string(bucketName))
	}
	if seq {
		// it is useful to increase fill percent when the workloads are mostly append-only.
		// this can delay the page split and reduce space usage.
		bucket.FillPercent = 0.9
	}
	if err := bucket.Put(key, value); err != nil {
		log.Fatalf("storage: cannot put key into bucket (%v)", err)
	}
	t.pending++
}

// UnsafeRange must be called holding the lock on the tx.
func (t *batchTx) UnsafeRange(bucketName []byte, key, endKey []byte, limit int64) (keys [][]byte, vs [][]byte) {
	bucket := t.tx.Bucket(bucketName)
	if bucket == nil {
		log.Fatalf("storage: bucket %s does not exist", string(bucketName))
	}

	if len(endKey) == 0 {
		if v := bucket.Get(key); v == nil {
			return keys, vs
		} else {
			return append(keys, key), append(vs, v)
		}
	}

	c := bucket.Cursor()
	for ck, cv := c.Seek(key); ck != nil && bytes.Compare(ck, endKey) < 0; ck, cv = c.Next() {
		vs = append(vs, cv)
		keys = append(keys, ck)
		if limit > 0 && limit == int64(len(keys)) {
			break
		}
	}

	return keys, vs
}

// UnsafeDelete must be called holding the lock on the tx.
func (t *batchTx) UnsafeDelete(bucketName []byte, key []byte) {
	bucket := t.tx.Bucket(bucketName)
	if bucket == nil {
		log.Fatalf("storage: bucket %s does not exist", string(bucketName))
	}
	err := bucket.Delete(key)
	if err != nil {
		log.Fatalf("storage: cannot delete key from bucket (%v)", err)
	}
	t.pending++
}

// UnsafeForEach must be called holding the lock on the tx.
func (t *batchTx) UnsafeForEach(bucketName []byte, visitor func(k, v []byte) error) error {
	return t.tx.Bucket(bucketName).ForEach(visitor)
}

// Commit commits a previous tx and begins a new writable one.
func (t *batchTx) Commit() {
	t.Lock()
	defer t.Unlock()
	t.commit(false)
}

// CommitAndStop commits the previous tx and do not create a new one.
func (t *batchTx) CommitAndStop() {
	t.Lock()
	defer t.Unlock()
	t.commit(true)
}

func (t *batchTx) Unlock() {
	if t.pending >= t.backend.batchLimit {
		t.commit(false)
		t.pending = 0
	}
	t.Mutex.Unlock()
}

func (t *batchTx) commit(stop bool) {
	var err error
	// commit the last tx
	if t.tx != nil {
		if t.pending == 0 && !stop {
			return
		}
		err = t.tx.Commit()
		atomic.AddInt64(&t.backend.commits, 1)

		t.pending = 0
		if err != nil {
			log.Fatalf("storage: cannot commit tx (%s)", err)
		}
	}

	if stop {
		return
	}

	t.backend.mu.RLock()
	defer t.backend.mu.RUnlock()
	// begin a new tx
	t.tx, err = t.backend.db.Begin(true)
	if err != nil {
		log.Fatalf("storage: cannot begin tx (%s)", err)
	}
	atomic.StoreInt64(&t.backend.size, t.tx.Size())
}
