// Copyright 2016 CoreOS, Inc.
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
	"bytes"
	"fmt"
	"sort"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/pkg/types"
	dstorage "github.com/coreos/etcd/storage"
	"github.com/coreos/etcd/storage/storagepb"
	"github.com/gogo/protobuf/proto"
)

const (
	// noTxn is an invalid txn ID.
	// To apply with independent Range, Put, Delete, you can pass noTxn
	// to apply functions instead of a valid txn ID.
	noTxn = -1
)

type applyResult struct {
	resp proto.Message
	err  error
	// physc signals the physical effect of the request has completed in addition
	// to being logically reflected by the node. Currently only used for
	// Compaction requests.
	physc <-chan struct{}
}

// applierV3 is the interface for processing V3 raft messages
type applierV3 interface {
	Put(txnID int64, p *pb.PutRequest) (*pb.PutResponse, error)
	Range(txnID int64, r *pb.RangeRequest) (*pb.RangeResponse, error)
	DeleteRange(txnID int64, dr *pb.DeleteRangeRequest) (*pb.DeleteRangeResponse, error)
	Txn(rt *pb.TxnRequest) (*pb.TxnResponse, error)
	Compaction(compaction *pb.CompactionRequest) (*pb.CompactionResponse, <-chan struct{}, error)
	LeaseGrant(lc *pb.LeaseGrantRequest) (*pb.LeaseGrantResponse, error)
	LeaseRevoke(lc *pb.LeaseRevokeRequest) (*pb.LeaseRevokeResponse, error)
	Alarm(*pb.AlarmRequest) (*pb.AlarmResponse, error)
	AuthEnable() (*pb.AuthEnableResponse, error)
	UserAdd(ua *pb.AuthUserAddRequest) (*pb.AuthUserAddResponse, error)
	UserDelete(ua *pb.AuthUserDeleteRequest) (*pb.AuthUserDeleteResponse, error)
	UserChangePassword(ua *pb.AuthUserChangePasswordRequest) (*pb.AuthUserChangePasswordResponse, error)
	RoleAdd(ua *pb.AuthRoleAddRequest) (*pb.AuthRoleAddResponse, error)
}

type applierV3backend struct {
	s *EtcdServer
}

func (s *EtcdServer) applyV3Request(r *pb.InternalRaftRequest) *applyResult {
	ar := &applyResult{}
	switch {
	case r.Range != nil:
		ar.resp, ar.err = s.applyV3.Range(noTxn, r.Range)
	case r.Put != nil:
		ar.resp, ar.err = s.applyV3.Put(noTxn, r.Put)
	case r.DeleteRange != nil:
		ar.resp, ar.err = s.applyV3.DeleteRange(noTxn, r.DeleteRange)
	case r.Txn != nil:
		ar.resp, ar.err = s.applyV3.Txn(r.Txn)
	case r.Compaction != nil:
		ar.resp, ar.physc, ar.err = s.applyV3.Compaction(r.Compaction)
	case r.LeaseGrant != nil:
		ar.resp, ar.err = s.applyV3.LeaseGrant(r.LeaseGrant)
	case r.LeaseRevoke != nil:
		ar.resp, ar.err = s.applyV3.LeaseRevoke(r.LeaseRevoke)
	case r.Alarm != nil:
		ar.resp, ar.err = s.applyV3.Alarm(r.Alarm)
	case r.AuthEnable != nil:
		ar.resp, ar.err = s.applyV3.AuthEnable()
	case r.AuthUserAdd != nil:
		ar.resp, ar.err = s.applyV3.UserAdd(r.AuthUserAdd)
	case r.AuthUserDelete != nil:
		ar.resp, ar.err = s.applyV3.UserDelete(r.AuthUserDelete)
	case r.AuthUserChangePassword != nil:
		ar.resp, ar.err = s.applyV3.UserChangePassword(r.AuthUserChangePassword)
	case r.AuthRoleAdd != nil:
		ar.resp, ar.err = s.applyV3.RoleAdd(r.AuthRoleAdd)
	default:
		panic("not implemented")
	}
	return ar
}

func (a *applierV3backend) Put(txnID int64, p *pb.PutRequest) (*pb.PutResponse, error) {
	resp := &pb.PutResponse{}
	resp.Header = &pb.ResponseHeader{}
	var (
		rev int64
		err error
	)
	if txnID != noTxn {
		rev, err = a.s.KV().TxnPut(txnID, p.Key, p.Value, lease.LeaseID(p.Lease))
		if err != nil {
			return nil, err
		}
	} else {
		leaseID := lease.LeaseID(p.Lease)
		if leaseID != lease.NoLease {
			if l := a.s.lessor.Lookup(leaseID); l == nil {
				return nil, lease.ErrLeaseNotFound
			}
		}
		rev = a.s.KV().Put(p.Key, p.Value, leaseID)
	}
	resp.Header.Revision = rev
	return resp, nil
}

func (a *applierV3backend) DeleteRange(txnID int64, dr *pb.DeleteRangeRequest) (*pb.DeleteRangeResponse, error) {
	resp := &pb.DeleteRangeResponse{}
	resp.Header = &pb.ResponseHeader{}

	var (
		n   int64
		rev int64
		err error
	)

	if isGteRange(dr.RangeEnd) {
		dr.RangeEnd = []byte{}
	}

	if txnID != noTxn {
		n, rev, err = a.s.KV().TxnDeleteRange(txnID, dr.Key, dr.RangeEnd)
		if err != nil {
			return nil, err
		}
	} else {
		n, rev = a.s.KV().DeleteRange(dr.Key, dr.RangeEnd)
	}

	resp.Deleted = n
	resp.Header.Revision = rev
	return resp, nil
}

func (a *applierV3backend) Range(txnID int64, r *pb.RangeRequest) (*pb.RangeResponse, error) {
	resp := &pb.RangeResponse{}
	resp.Header = &pb.ResponseHeader{}

	var (
		kvs []storagepb.KeyValue
		rev int64
		err error
	)

	if isGteRange(r.RangeEnd) {
		r.RangeEnd = []byte{}
	}

	limit := r.Limit
	if r.SortOrder != pb.RangeRequest_NONE {
		// fetch everything; sort and truncate afterwards
		limit = 0
	}
	if limit > 0 {
		// fetch one extra for 'more' flag
		limit = limit + 1
	}

	if txnID != noTxn {
		kvs, rev, err = a.s.KV().TxnRange(txnID, r.Key, r.RangeEnd, limit, r.Revision)
		if err != nil {
			return nil, err
		}
	} else {
		kvs, rev, err = a.s.KV().Range(r.Key, r.RangeEnd, limit, r.Revision)
		if err != nil {
			return nil, err
		}
	}

	if r.SortOrder != pb.RangeRequest_NONE {
		var sorter sort.Interface
		switch {
		case r.SortTarget == pb.RangeRequest_KEY:
			sorter = &kvSortByKey{&kvSort{kvs}}
		case r.SortTarget == pb.RangeRequest_VERSION:
			sorter = &kvSortByVersion{&kvSort{kvs}}
		case r.SortTarget == pb.RangeRequest_CREATE:
			sorter = &kvSortByCreate{&kvSort{kvs}}
		case r.SortTarget == pb.RangeRequest_MOD:
			sorter = &kvSortByMod{&kvSort{kvs}}
		case r.SortTarget == pb.RangeRequest_VALUE:
			sorter = &kvSortByValue{&kvSort{kvs}}
		}
		switch {
		case r.SortOrder == pb.RangeRequest_ASCEND:
			sort.Sort(sorter)
		case r.SortOrder == pb.RangeRequest_DESCEND:
			sort.Sort(sort.Reverse(sorter))
		}
	}

	if r.Limit > 0 && len(kvs) > int(r.Limit) {
		kvs = kvs[:r.Limit]
		resp.More = true
	}

	resp.Header.Revision = rev
	for i := range kvs {
		resp.Kvs = append(resp.Kvs, &kvs[i])
	}
	return resp, nil
}

func (a *applierV3backend) Txn(rt *pb.TxnRequest) (*pb.TxnResponse, error) {
	var revision int64

	ok := true
	for _, c := range rt.Compare {
		if revision, ok = a.applyCompare(c); !ok {
			break
		}
	}

	var reqs []*pb.RequestUnion
	if ok {
		reqs = rt.Success
	} else {
		reqs = rt.Failure
	}

	if err := a.checkRequestLeases(reqs); err != nil {
		return nil, err
	}
	if err := a.checkRequestRange(reqs); err != nil {
		return nil, err
	}

	// When executing the operations of txn, we need to hold the txn lock.
	// So the reader will not see any intermediate results.
	txnID := a.s.KV().TxnBegin()
	defer func() {
		err := a.s.KV().TxnEnd(txnID)
		if err != nil {
			panic(fmt.Sprint("unexpected error when closing txn", txnID))
		}
	}()

	resps := make([]*pb.ResponseUnion, len(reqs))
	for i := range reqs {
		resps[i] = a.applyUnion(txnID, reqs[i])
	}

	if len(resps) != 0 {
		revision += 1
	}

	txnResp := &pb.TxnResponse{}
	txnResp.Header = &pb.ResponseHeader{}
	txnResp.Header.Revision = revision
	txnResp.Responses = resps
	txnResp.Succeeded = ok
	return txnResp, nil
}

// applyCompare applies the compare request.
// It returns the revision at which the comparison happens. If the comparison
// succeeds, the it returns true. Otherwise it returns false.
func (a *applierV3backend) applyCompare(c *pb.Compare) (int64, bool) {
	ckvs, rev, err := a.s.KV().Range(c.Key, nil, 1, 0)
	if err != nil {
		if err == dstorage.ErrTxnIDMismatch {
			panic("unexpected txn ID mismatch error")
		}
		return rev, false
	}
	var ckv storagepb.KeyValue
	if len(ckvs) != 0 {
		ckv = ckvs[0]
	} else {
		// Use the zero value of ckv normally. However...
		if c.Target == pb.Compare_VALUE {
			// Always fail if we're comparing a value on a key that doesn't exist.
			// We can treat non-existence as the empty set explicitly, such that
			// even a key with a value of length 0 bytes is still a real key
			// that was written that way
			return rev, false
		}
	}

	// -1 is less, 0 is equal, 1 is greater
	var result int
	switch c.Target {
	case pb.Compare_VALUE:
		tv, _ := c.TargetUnion.(*pb.Compare_Value)
		if tv != nil {
			result = bytes.Compare(ckv.Value, tv.Value)
		}
	case pb.Compare_CREATE:
		tv, _ := c.TargetUnion.(*pb.Compare_CreateRevision)
		if tv != nil {
			result = compareInt64(ckv.CreateRevision, tv.CreateRevision)
		}

	case pb.Compare_MOD:
		tv, _ := c.TargetUnion.(*pb.Compare_ModRevision)
		if tv != nil {
			result = compareInt64(ckv.ModRevision, tv.ModRevision)
		}
	case pb.Compare_VERSION:
		tv, _ := c.TargetUnion.(*pb.Compare_Version)
		if tv != nil {
			result = compareInt64(ckv.Version, tv.Version)
		}
	}

	switch c.Result {
	case pb.Compare_EQUAL:
		if result != 0 {
			return rev, false
		}
	case pb.Compare_GREATER:
		if result != 1 {
			return rev, false
		}
	case pb.Compare_LESS:
		if result != -1 {
			return rev, false
		}
	}
	return rev, true
}

func (a *applierV3backend) applyUnion(txnID int64, union *pb.RequestUnion) *pb.ResponseUnion {
	switch tv := union.Request.(type) {
	case *pb.RequestUnion_RequestRange:
		if tv.RequestRange != nil {
			resp, err := a.Range(txnID, tv.RequestRange)
			if err != nil {
				panic("unexpected error during txn")
			}
			return &pb.ResponseUnion{Response: &pb.ResponseUnion_ResponseRange{ResponseRange: resp}}
		}
	case *pb.RequestUnion_RequestPut:
		if tv.RequestPut != nil {
			resp, err := a.Put(txnID, tv.RequestPut)
			if err != nil {
				panic("unexpected error during txn")
			}
			return &pb.ResponseUnion{Response: &pb.ResponseUnion_ResponsePut{ResponsePut: resp}}
		}
	case *pb.RequestUnion_RequestDeleteRange:
		if tv.RequestDeleteRange != nil {
			resp, err := a.DeleteRange(txnID, tv.RequestDeleteRange)
			if err != nil {
				panic("unexpected error during txn")
			}
			return &pb.ResponseUnion{Response: &pb.ResponseUnion_ResponseDeleteRange{ResponseDeleteRange: resp}}
		}
	default:
		// empty union
		return nil
	}
	return nil

}

func (a *applierV3backend) Compaction(compaction *pb.CompactionRequest) (*pb.CompactionResponse, <-chan struct{}, error) {
	resp := &pb.CompactionResponse{}
	resp.Header = &pb.ResponseHeader{}
	ch, err := a.s.KV().Compact(compaction.Revision)
	if err != nil {
		return nil, ch, err
	}
	// get the current revision. which key to get is not important.
	_, resp.Header.Revision, _ = a.s.KV().Range([]byte("compaction"), nil, 1, 0)
	return resp, ch, err
}

func (a *applierV3backend) LeaseGrant(lc *pb.LeaseGrantRequest) (*pb.LeaseGrantResponse, error) {
	l, err := a.s.lessor.Grant(lease.LeaseID(lc.ID), lc.TTL)
	resp := &pb.LeaseGrantResponse{}
	if err == nil {
		resp.ID = int64(l.ID)
		resp.TTL = l.TTL
	}
	return resp, err
}

func (a *applierV3backend) LeaseRevoke(lc *pb.LeaseRevokeRequest) (*pb.LeaseRevokeResponse, error) {
	err := a.s.lessor.Revoke(lease.LeaseID(lc.ID))
	return &pb.LeaseRevokeResponse{}, err
}

func (a *applierV3backend) Alarm(ar *pb.AlarmRequest) (*pb.AlarmResponse, error) {
	resp := &pb.AlarmResponse{}
	oldCount := len(a.s.alarmStore.Get(ar.Alarm))

	switch ar.Action {
	case pb.AlarmRequest_GET:
		resp.Alarms = a.s.alarmStore.Get(ar.Alarm)
	case pb.AlarmRequest_ACTIVATE:
		m := a.s.alarmStore.Activate(types.ID(ar.MemberID), ar.Alarm)
		if m == nil {
			break
		}
		resp.Alarms = append(resp.Alarms, m)
		activated := oldCount == 0 && len(a.s.alarmStore.Get(m.Alarm)) == 1
		if !activated {
			break
		}

		switch m.Alarm {
		case pb.AlarmType_NOSPACE:
			plog.Warningf("alarm raised %+v", m)
			a.s.applyV3 = newApplierV3Capped(a)
		default:
			plog.Errorf("unimplemented alarm activation (%+v)", m)
		}
	case pb.AlarmRequest_DEACTIVATE:
		m := a.s.alarmStore.Deactivate(types.ID(ar.MemberID), ar.Alarm)
		if m == nil {
			break
		}
		resp.Alarms = append(resp.Alarms, m)
		deactivated := oldCount > 0 && len(a.s.alarmStore.Get(ar.Alarm)) == 0
		if !deactivated {
			break
		}

		switch m.Alarm {
		case pb.AlarmType_NOSPACE:
			plog.Infof("alarm disarmed %+v", ar)
			a.s.applyV3 = newQuotaApplierV3(a.s, &applierV3backend{a.s})
		default:
			plog.Errorf("unimplemented alarm deactivation (%+v)", m)
		}
	default:
		return nil, nil
	}
	return resp, nil
}

type applierV3Capped struct {
	applierV3
	q backendQuota
}

// newApplierV3Capped creates an applyV3 that will reject Puts and transactions
// with Puts so that the number of keys in the store is capped.
func newApplierV3Capped(base applierV3) applierV3 { return &applierV3Capped{applierV3: base} }

func (a *applierV3Capped) Put(txnID int64, p *pb.PutRequest) (*pb.PutResponse, error) {
	return nil, ErrNoSpace
}

func (a *applierV3Capped) Txn(r *pb.TxnRequest) (*pb.TxnResponse, error) {
	if a.q.Cost(r) > 0 {
		return nil, ErrNoSpace
	}
	return a.applierV3.Txn(r)
}

func (a *applierV3Capped) LeaseGrant(lc *pb.LeaseGrantRequest) (*pb.LeaseGrantResponse, error) {
	return nil, ErrNoSpace
}

func (a *applierV3backend) AuthEnable() (*pb.AuthEnableResponse, error) {
	a.s.AuthStore().AuthEnable()
	return &pb.AuthEnableResponse{}, nil
}

func (a *applierV3backend) UserAdd(r *pb.AuthUserAddRequest) (*pb.AuthUserAddResponse, error) {
	return a.s.AuthStore().UserAdd(r)
}

func (a *applierV3backend) UserDelete(r *pb.AuthUserDeleteRequest) (*pb.AuthUserDeleteResponse, error) {
	return a.s.AuthStore().UserDelete(r)
}

func (a *applierV3backend) UserChangePassword(r *pb.AuthUserChangePasswordRequest) (*pb.AuthUserChangePasswordResponse, error) {
	return a.s.AuthStore().UserChangePassword(r)
}

func (a *applierV3backend) RoleAdd(r *pb.AuthRoleAddRequest) (*pb.AuthRoleAddResponse, error) {
	return a.s.AuthStore().RoleAdd(r)
}

type quotaApplierV3 struct {
	applierV3
	q Quota
}

func newQuotaApplierV3(s *EtcdServer, app applierV3) applierV3 {
	return &quotaApplierV3{app, NewBackendQuota(s)}
}

func (a *quotaApplierV3) Put(txnID int64, p *pb.PutRequest) (*pb.PutResponse, error) {
	ok := a.q.Available(p)
	resp, err := a.applierV3.Put(txnID, p)
	if err == nil && !ok {
		err = ErrNoSpace
	}
	return resp, err
}

func (a *quotaApplierV3) Txn(rt *pb.TxnRequest) (*pb.TxnResponse, error) {
	ok := a.q.Available(rt)
	resp, err := a.applierV3.Txn(rt)
	if err == nil && !ok {
		err = ErrNoSpace
	}
	return resp, err
}

func (a *quotaApplierV3) LeaseGrant(lc *pb.LeaseGrantRequest) (*pb.LeaseGrantResponse, error) {
	ok := a.q.Available(lc)
	resp, err := a.applierV3.LeaseGrant(lc)
	if err == nil && !ok {
		err = ErrNoSpace
	}
	return resp, err
}

type kvSort struct{ kvs []storagepb.KeyValue }

func (s *kvSort) Swap(i, j int) {
	t := s.kvs[i]
	s.kvs[i] = s.kvs[j]
	s.kvs[j] = t
}
func (s *kvSort) Len() int { return len(s.kvs) }

type kvSortByKey struct{ *kvSort }

func (s *kvSortByKey) Less(i, j int) bool {
	return bytes.Compare(s.kvs[i].Key, s.kvs[j].Key) < 0
}

type kvSortByVersion struct{ *kvSort }

func (s *kvSortByVersion) Less(i, j int) bool {
	return (s.kvs[i].Version - s.kvs[j].Version) < 0
}

type kvSortByCreate struct{ *kvSort }

func (s *kvSortByCreate) Less(i, j int) bool {
	return (s.kvs[i].CreateRevision - s.kvs[j].CreateRevision) < 0
}

type kvSortByMod struct{ *kvSort }

func (s *kvSortByMod) Less(i, j int) bool {
	return (s.kvs[i].ModRevision - s.kvs[j].ModRevision) < 0
}

type kvSortByValue struct{ *kvSort }

func (s *kvSortByValue) Less(i, j int) bool {
	return bytes.Compare(s.kvs[i].Value, s.kvs[j].Value) < 0
}

func (a *applierV3backend) checkRequestLeases(reqs []*pb.RequestUnion) error {
	for _, requ := range reqs {
		tv, ok := requ.Request.(*pb.RequestUnion_RequestPut)
		if !ok {
			continue
		}
		preq := tv.RequestPut
		if preq == nil || lease.LeaseID(preq.Lease) == lease.NoLease {
			continue
		}
		if l := a.s.lessor.Lookup(lease.LeaseID(preq.Lease)); l == nil {
			return lease.ErrLeaseNotFound
		}
	}
	return nil
}

func (a *applierV3backend) checkRequestRange(reqs []*pb.RequestUnion) error {
	for _, requ := range reqs {
		tv, ok := requ.Request.(*pb.RequestUnion_RequestRange)
		if !ok {
			continue
		}
		greq := tv.RequestRange
		if greq == nil || greq.Revision == 0 {
			continue
		}

		if greq.Revision > a.s.KV().Rev() {
			return dstorage.ErrFutureRev
		}
		if greq.Revision < a.s.KV().FirstRev() {
			return dstorage.ErrCompacted
		}
	}
	return nil
}

func compareInt64(a, b int64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

// isGteRange determines if the range end is a >= range. This works around grpc
// sending empty byte strings as nil; >= is encoded in the range end as '\0'.
func isGteRange(rangeEnd []byte) bool {
	return len(rangeEnd) == 1 && rangeEnd[0] == 0
}
