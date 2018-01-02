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

package main

import (
	"fmt"
	"path/filepath"

	"github.com/boltdb/bolt"
	"github.com/coreos/etcd/mvcc"
	"github.com/coreos/etcd/mvcc/backend"
)

func snapDir(dataDir string) string {
	return filepath.Join(dataDir, "member", "snap")
}

func getBuckets(dbPath string) (buckets []string, err error) {
	db, derr := bolt.Open(dbPath, 0600, &bolt.Options{})
	if derr != nil {
		return nil, derr
	}
	defer db.Close()

	err = db.View(func(tx *bolt.Tx) error {
		return tx.ForEach(func(b []byte, _ *bolt.Bucket) error {
			buckets = append(buckets, string(b))
			return nil
		})
	})
	return
}

func iterateBucket(dbPath, bucket string, limit uint64) (err error) {
	db, derr := bolt.Open(dbPath, 0600, &bolt.Options{})
	if derr != nil {
		return derr
	}
	defer db.Close()

	err = db.View(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(bucket))
		if b == nil {
			return fmt.Errorf("got nil bucket for %s", bucket)
		}

		c := b.Cursor()

		// iterate in reverse order (use First() and Next() for ascending order)
		for k, v := c.Last(); k != nil; k, v = c.Prev() {
			fmt.Printf("key=%q, value=%q\n", k, v)

			limit--
			if limit == 0 {
				break
			}
		}

		return nil
	})
	return
}

func getHash(dbPath string) (hash uint32, err error) {
	b := backend.NewDefaultBackend(dbPath)
	return b.Hash(mvcc.DefaultIgnores)
}

// TODO: revert by revision and find specified hash value
// currently, it's hard because lease is in separate bucket
// and does not modify revision
