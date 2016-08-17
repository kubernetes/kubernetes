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

package etcdserver

import (
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/mvcc/backend"
)

// Quota represents an arbitrary quota against arbitrary requests. Each request
// costs some charge; if there is not enough remaining charge, then there are
// too few resources available within the quota to apply the request.
type Quota interface {
	// Available judges whether the given request fits within the quota.
	Available(req interface{}) bool
	// Cost computes the charge against the quota for a given request.
	Cost(req interface{}) int
	// Remaining is the amount of charge left for the quota.
	Remaining() int64
}

type passthroughQuota struct{}

func (*passthroughQuota) Available(interface{}) bool { return true }
func (*passthroughQuota) Cost(interface{}) int       { return 0 }
func (*passthroughQuota) Remaining() int64           { return 1 }

type backendQuota struct {
	s               *EtcdServer
	maxBackendBytes int64
}

const (
	// leaseOverhead is an estimate for the cost of storing a lease
	leaseOverhead = 64
	// kvOverhead is an estimate for the cost of storing a key's metadata
	kvOverhead = 256
)

func NewBackendQuota(s *EtcdServer) Quota {
	if s.Cfg.QuotaBackendBytes < 0 {
		// disable quotas if negative
		plog.Warningf("disabling backend quota")
		return &passthroughQuota{}
	}
	if s.Cfg.QuotaBackendBytes == 0 {
		// use default size if no quota size given
		return &backendQuota{s, backend.DefaultQuotaBytes}
	}
	if s.Cfg.QuotaBackendBytes > backend.MaxQuotaBytes {
		plog.Warningf("backend quota %v exceeds maximum quota %v; using maximum", s.Cfg.QuotaBackendBytes, backend.MaxQuotaBytes)
		return &backendQuota{s, backend.MaxQuotaBytes}
	}
	return &backendQuota{s, s.Cfg.QuotaBackendBytes}
}

func (b *backendQuota) Available(v interface{}) bool {
	// TODO: maybe optimize backend.Size()
	return b.s.Backend().Size()+int64(b.Cost(v)) < b.maxBackendBytes
}

func (b *backendQuota) Cost(v interface{}) int {
	switch r := v.(type) {
	case *pb.PutRequest:
		return costPut(r)
	case *pb.TxnRequest:
		return costTxn(r)
	case *pb.LeaseGrantRequest:
		return leaseOverhead
	default:
		panic("unexpected cost")
	}
}

func costPut(r *pb.PutRequest) int { return kvOverhead + len(r.Key) + len(r.Value) }

func costTxnReq(u *pb.RequestOp) int {
	r := u.GetRequestPut()
	if r == nil {
		return 0
	}
	return costPut(r)
}

func costTxn(r *pb.TxnRequest) int {
	sizeSuccess := 0
	for _, u := range r.Success {
		sizeSuccess += costTxnReq(u)
	}
	sizeFailure := 0
	for _, u := range r.Failure {
		sizeFailure += costTxnReq(u)
	}
	if sizeFailure > sizeSuccess {
		return sizeFailure
	}
	return sizeSuccess
}

func (b *backendQuota) Remaining() int64 {
	return b.maxBackendBytes - b.s.Backend().Size()
}
