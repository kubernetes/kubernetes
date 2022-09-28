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
	"fmt"

	"go.etcd.io/etcd/api/v3/mvccpb"
	"go.etcd.io/etcd/server/v3/mvcc/backend"
	"go.etcd.io/etcd/server/v3/mvcc/buckets"
)

func WriteKV(be backend.Backend, kv mvccpb.KeyValue) {
	ibytes := newRevBytes()
	revToBytes(revision{main: kv.ModRevision}, ibytes)

	d, err := kv.Marshal()
	if err != nil {
		panic(fmt.Errorf("cannot marshal event: %v", err))
	}

	be.BatchTx().LockOutsideApply()
	be.BatchTx().UnsafePut(buckets.Key, ibytes, d)
	be.BatchTx().Unlock()
}
