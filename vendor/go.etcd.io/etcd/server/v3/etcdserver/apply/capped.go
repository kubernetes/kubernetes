// Copyright 2025 The etcd Authors
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

package apply

import (
	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/pkg/v3/traceutil"
	"go.etcd.io/etcd/server/v3/etcdserver/errors"
	serverstorage "go.etcd.io/etcd/server/v3/storage"
)

type applierV3Capped struct {
	applierV3
	q serverstorage.BackendQuota
}

// newApplierV3Capped creates an applyV3 that will reject Puts and transactions
// with Puts so that the number of keys in the store is capped.
func newApplierV3Capped(base applierV3) applierV3 { return &applierV3Capped{applierV3: base} }

func (a *applierV3Capped) Put(_ *pb.PutRequest) (*pb.PutResponse, *traceutil.Trace, error) {
	return nil, nil, errors.ErrNoSpace
}

func (a *applierV3Capped) Txn(r *pb.TxnRequest) (*pb.TxnResponse, *traceutil.Trace, error) {
	if a.q.Cost(r) > 0 {
		return nil, nil, errors.ErrNoSpace
	}
	return a.applierV3.Txn(r)
}

func (a *applierV3Capped) LeaseGrant(_ *pb.LeaseGrantRequest) (*pb.LeaseGrantResponse, error) {
	return nil, errors.ErrNoSpace
}
