// Copyright 2017 The etcd Authors
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
	"math"
	"sync"

	bolt "github.com/coreos/bbolt"
)

// safeRangeBucket is a hack to avoid inadvertently reading duplicate keys;
// overwrites on a bucket should only fetch with limit=1, but safeRangeBucket
// is known to never overwrite any key so range is safe.
var safeRangeBucket = []byte("key")

type ReadTx interface {
	Lock()
	Unlock()

	UnsafeRange(bucketName []byte, key, endKey []byte, limit int64) (keys [][]byte, vals [][]byte)
	UnsafeForEach(bucketName []byte, visitor func(k, v []byte) error) error
}

type readTx struct {
	// mu protects accesses to the txReadBuffer
	mu  sync.RWMutex
	buf txReadBuffer

	// txmu protects accesses to the Tx on Range requests
	txmu sync.Mutex
	tx   *bolt.Tx
}

func (rt *readTx) Lock()   { rt.mu.RLock() }
func (rt *readTx) Unlock() { rt.mu.RUnlock() }

func (rt *readTx) UnsafeRange(bucketName, key, endKey []byte, limit int64) ([][]byte, [][]byte) {
	if endKey == nil {
		// forbid duplicates for single keys
		limit = 1
	}
	if limit <= 0 {
		limit = math.MaxInt64
	}
	if limit > 1 && !bytes.Equal(bucketName, safeRangeBucket) {
		panic("do not use unsafeRange on non-keys bucket")
	}
	keys, vals := rt.buf.Range(bucketName, key, endKey, limit)
	if int64(len(keys)) == limit {
		return keys, vals
	}
	rt.txmu.Lock()
	// ignore error since bucket may have been created in this batch
	k2, v2, _ := unsafeRange(rt.tx, bucketName, key, endKey, limit-int64(len(keys)))
	rt.txmu.Unlock()
	return append(k2, keys...), append(v2, vals...)
}

func (rt *readTx) UnsafeForEach(bucketName []byte, visitor func(k, v []byte) error) error {
	dups := make(map[string]struct{})
	f1 := func(k, v []byte) error {
		dups[string(k)] = struct{}{}
		return visitor(k, v)
	}
	f2 := func(k, v []byte) error {
		if _, ok := dups[string(k)]; ok {
			return nil
		}
		return visitor(k, v)
	}
	if err := rt.buf.ForEach(bucketName, f1); err != nil {
		return err
	}
	rt.txmu.Lock()
	err := unsafeForEach(rt.tx, bucketName, f2)
	rt.txmu.Unlock()
	return err
}
