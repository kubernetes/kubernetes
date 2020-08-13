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

package mvcc

import (
	"encoding/binary"
	"fmt"

	"go.etcd.io/etcd/mvcc/backend"
	"go.etcd.io/etcd/mvcc/mvccpb"
)

func UpdateConsistentIndex(be backend.Backend, index uint64) {
	tx := be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	var oldi uint64
	_, vs := tx.UnsafeRange(metaBucketName, consistentIndexKeyName, nil, 0)
	if len(vs) != 0 {
		oldi = binary.BigEndian.Uint64(vs[0])
	}

	if index <= oldi {
		return
	}

	bs := make([]byte, 8)
	binary.BigEndian.PutUint64(bs, index)
	tx.UnsafePut(metaBucketName, consistentIndexKeyName, bs)
}

func WriteKV(be backend.Backend, kv mvccpb.KeyValue) {
	ibytes := newRevBytes()
	revToBytes(revision{main: kv.ModRevision}, ibytes)

	d, err := kv.Marshal()
	if err != nil {
		panic(fmt.Errorf("cannot marshal event: %v", err))
	}

	be.BatchTx().Lock()
	be.BatchTx().UnsafePut(keyBucketName, ibytes, d)
	be.BatchTx().Unlock()
}
