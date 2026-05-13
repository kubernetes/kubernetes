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
	"go.uber.org/zap"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/pkg/v3/traceutil"
	"go.etcd.io/etcd/server/v3/etcdserver/errors"
	serverstorage "go.etcd.io/etcd/server/v3/storage"
	"go.etcd.io/etcd/server/v3/storage/backend"
)

type quotaApplierV3 struct {
	applierV3
	q serverstorage.Quota
}

func newQuotaApplierV3(lg *zap.Logger, quotaBackendBytesCfg int64, be backend.Backend, app applierV3) applierV3 {
	return &quotaApplierV3{app, serverstorage.NewBackendQuota(lg, quotaBackendBytesCfg, be, "v3-applier")}
}

func (a *quotaApplierV3) Put(p *pb.PutRequest) (*pb.PutResponse, *traceutil.Trace, error) {
	ok := a.q.Available(p)
	resp, trace, err := a.applierV3.Put(p)
	if err == nil && !ok {
		err = errors.ErrNoSpace
	}
	return resp, trace, err
}

func (a *quotaApplierV3) Txn(rt *pb.TxnRequest) (*pb.TxnResponse, *traceutil.Trace, error) {
	ok := a.q.Available(rt)
	resp, trace, err := a.applierV3.Txn(rt)
	if err == nil && !ok {
		err = errors.ErrNoSpace
	}
	return resp, trace, err
}

func (a *quotaApplierV3) LeaseGrant(lc *pb.LeaseGrantRequest) (*pb.LeaseGrantResponse, error) {
	ok := a.q.Available(lc)
	resp, err := a.applierV3.LeaseGrant(lc)
	if err == nil && !ok {
		err = errors.ErrNoSpace
	}
	return resp, err
}
