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
	"bytes"
	"sort"
	"time"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/mvcc"
	"github.com/coreos/etcd/mvcc/mvccpb"
	"github.com/coreos/etcd/pkg/types"
	"github.com/gogo/protobuf/proto"
	"golang.org/x/net/context"
)

const (
	warnApplyDuration = 100 * time.Millisecond
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
	Apply(r *pb.InternalRaftRequest) *applyResult

	Put(txn mvcc.TxnWrite, p *pb.PutRequest) (*pb.PutResponse, error)
	Range(txn mvcc.TxnRead, r *pb.RangeRequest) (*pb.RangeResponse, error)
	DeleteRange(txn mvcc.TxnWrite, dr *pb.DeleteRangeRequest) (*pb.DeleteRangeResponse, error)
	Txn(rt *pb.TxnRequest) (*pb.TxnResponse, error)
	Compaction(compaction *pb.CompactionRequest) (*pb.CompactionResponse, <-chan struct{}, error)

	LeaseGrant(lc *pb.LeaseGrantRequest) (*pb.LeaseGrantResponse, error)
	LeaseRevoke(lc *pb.LeaseRevokeRequest) (*pb.LeaseRevokeResponse, error)

	Alarm(*pb.AlarmRequest) (*pb.AlarmResponse, error)

	Authenticate(r *pb.InternalAuthenticateRequest) (*pb.AuthenticateResponse, error)

	AuthEnable() (*pb.AuthEnableResponse, error)
	AuthDisable() (*pb.AuthDisableResponse, error)

	UserAdd(ua *pb.AuthUserAddRequest) (*pb.AuthUserAddResponse, error)
	UserDelete(ua *pb.AuthUserDeleteRequest) (*pb.AuthUserDeleteResponse, error)
	UserChangePassword(ua *pb.AuthUserChangePasswordRequest) (*pb.AuthUserChangePasswordResponse, error)
	UserGrantRole(ua *pb.AuthUserGrantRoleRequest) (*pb.AuthUserGrantRoleResponse, error)
	UserGet(ua *pb.AuthUserGetRequest) (*pb.AuthUserGetResponse, error)
	UserRevokeRole(ua *pb.AuthUserRevokeRoleRequest) (*pb.AuthUserRevokeRoleResponse, error)
	RoleAdd(ua *pb.AuthRoleAddRequest) (*pb.AuthRoleAddResponse, error)
	RoleGrantPermission(ua *pb.AuthRoleGrantPermissionRequest) (*pb.AuthRoleGrantPermissionResponse, error)
	RoleGet(ua *pb.AuthRoleGetRequest) (*pb.AuthRoleGetResponse, error)
	RoleRevokePermission(ua *pb.AuthRoleRevokePermissionRequest) (*pb.AuthRoleRevokePermissionResponse, error)
	RoleDelete(ua *pb.AuthRoleDeleteRequest) (*pb.AuthRoleDeleteResponse, error)
	UserList(ua *pb.AuthUserListRequest) (*pb.AuthUserListResponse, error)
	RoleList(ua *pb.AuthRoleListRequest) (*pb.AuthRoleListResponse, error)
}

type applierV3backend struct {
	s *EtcdServer
}

func (s *EtcdServer) newApplierV3() applierV3 {
	return newAuthApplierV3(
		s.AuthStore(),
		newQuotaApplierV3(s, &applierV3backend{s}),
	)
}

func (a *applierV3backend) Apply(r *pb.InternalRaftRequest) *applyResult {
	ar := &applyResult{}
	defer func(start time.Time) {
		warnOfExpensiveRequest(start, &pb.InternalRaftStringer{Request: r}, ar.resp, ar.err)
	}(time.Now())

	// call into a.s.applyV3.F instead of a.F so upper appliers can check individual calls
	switch {
	case r.Range != nil:
		ar.resp, ar.err = a.s.applyV3.Range(nil, r.Range)
	case r.Put != nil:
		ar.resp, ar.err = a.s.applyV3.Put(nil, r.Put)
	case r.DeleteRange != nil:
		ar.resp, ar.err = a.s.applyV3.DeleteRange(nil, r.DeleteRange)
	case r.Txn != nil:
		ar.resp, ar.err = a.s.applyV3.Txn(r.Txn)
	case r.Compaction != nil:
		ar.resp, ar.physc, ar.err = a.s.applyV3.Compaction(r.Compaction)
	case r.LeaseGrant != nil:
		ar.resp, ar.err = a.s.applyV3.LeaseGrant(r.LeaseGrant)
	case r.LeaseRevoke != nil:
		ar.resp, ar.err = a.s.applyV3.LeaseRevoke(r.LeaseRevoke)
	case r.Alarm != nil:
		ar.resp, ar.err = a.s.applyV3.Alarm(r.Alarm)
	case r.Authenticate != nil:
		ar.resp, ar.err = a.s.applyV3.Authenticate(r.Authenticate)
	case r.AuthEnable != nil:
		ar.resp, ar.err = a.s.applyV3.AuthEnable()
	case r.AuthDisable != nil:
		ar.resp, ar.err = a.s.applyV3.AuthDisable()
	case r.AuthUserAdd != nil:
		ar.resp, ar.err = a.s.applyV3.UserAdd(r.AuthUserAdd)
	case r.AuthUserDelete != nil:
		ar.resp, ar.err = a.s.applyV3.UserDelete(r.AuthUserDelete)
	case r.AuthUserChangePassword != nil:
		ar.resp, ar.err = a.s.applyV3.UserChangePassword(r.AuthUserChangePassword)
	case r.AuthUserGrantRole != nil:
		ar.resp, ar.err = a.s.applyV3.UserGrantRole(r.AuthUserGrantRole)
	case r.AuthUserGet != nil:
		ar.resp, ar.err = a.s.applyV3.UserGet(r.AuthUserGet)
	case r.AuthUserRevokeRole != nil:
		ar.resp, ar.err = a.s.applyV3.UserRevokeRole(r.AuthUserRevokeRole)
	case r.AuthRoleAdd != nil:
		ar.resp, ar.err = a.s.applyV3.RoleAdd(r.AuthRoleAdd)
	case r.AuthRoleGrantPermission != nil:
		ar.resp, ar.err = a.s.applyV3.RoleGrantPermission(r.AuthRoleGrantPermission)
	case r.AuthRoleGet != nil:
		ar.resp, ar.err = a.s.applyV3.RoleGet(r.AuthRoleGet)
	case r.AuthRoleRevokePermission != nil:
		ar.resp, ar.err = a.s.applyV3.RoleRevokePermission(r.AuthRoleRevokePermission)
	case r.AuthRoleDelete != nil:
		ar.resp, ar.err = a.s.applyV3.RoleDelete(r.AuthRoleDelete)
	case r.AuthUserList != nil:
		ar.resp, ar.err = a.s.applyV3.UserList(r.AuthUserList)
	case r.AuthRoleList != nil:
		ar.resp, ar.err = a.s.applyV3.RoleList(r.AuthRoleList)
	default:
		panic("not implemented")
	}
	return ar
}

func (a *applierV3backend) Put(txn mvcc.TxnWrite, p *pb.PutRequest) (resp *pb.PutResponse, err error) {
	resp = &pb.PutResponse{}
	resp.Header = &pb.ResponseHeader{}

	val, leaseID := p.Value, lease.LeaseID(p.Lease)
	if txn == nil {
		if leaseID != lease.NoLease {
			if l := a.s.lessor.Lookup(leaseID); l == nil {
				return nil, lease.ErrLeaseNotFound
			}
		}
		txn = a.s.KV().Write()
		defer txn.End()
	}

	var rr *mvcc.RangeResult
	if p.IgnoreValue || p.IgnoreLease || p.PrevKv {
		rr, err = txn.Range(p.Key, nil, mvcc.RangeOptions{})
		if err != nil {
			return nil, err
		}
	}
	if p.IgnoreValue || p.IgnoreLease {
		if rr == nil || len(rr.KVs) == 0 {
			// ignore_{lease,value} flag expects previous key-value pair
			return nil, ErrKeyNotFound
		}
	}
	if p.IgnoreValue {
		val = rr.KVs[0].Value
	}
	if p.IgnoreLease {
		leaseID = lease.LeaseID(rr.KVs[0].Lease)
	}
	if p.PrevKv {
		if rr != nil && len(rr.KVs) != 0 {
			resp.PrevKv = &rr.KVs[0]
		}
	}

	resp.Header.Revision = txn.Put(p.Key, val, leaseID)
	return resp, nil
}

func (a *applierV3backend) DeleteRange(txn mvcc.TxnWrite, dr *pb.DeleteRangeRequest) (*pb.DeleteRangeResponse, error) {
	resp := &pb.DeleteRangeResponse{}
	resp.Header = &pb.ResponseHeader{}

	if txn == nil {
		txn = a.s.kv.Write()
		defer txn.End()
	}

	if isGteRange(dr.RangeEnd) {
		dr.RangeEnd = []byte{}
	}

	if dr.PrevKv {
		rr, err := txn.Range(dr.Key, dr.RangeEnd, mvcc.RangeOptions{})
		if err != nil {
			return nil, err
		}
		if rr != nil {
			for i := range rr.KVs {
				resp.PrevKvs = append(resp.PrevKvs, &rr.KVs[i])
			}
		}
	}

	resp.Deleted, resp.Header.Revision = txn.DeleteRange(dr.Key, dr.RangeEnd)
	return resp, nil
}

func (a *applierV3backend) Range(txn mvcc.TxnRead, r *pb.RangeRequest) (*pb.RangeResponse, error) {
	resp := &pb.RangeResponse{}
	resp.Header = &pb.ResponseHeader{}

	if txn == nil {
		txn = a.s.kv.Read()
		defer txn.End()
	}

	if isGteRange(r.RangeEnd) {
		r.RangeEnd = []byte{}
	}

	limit := r.Limit
	if r.SortOrder != pb.RangeRequest_NONE ||
		r.MinModRevision != 0 || r.MaxModRevision != 0 ||
		r.MinCreateRevision != 0 || r.MaxCreateRevision != 0 {
		// fetch everything; sort and truncate afterwards
		limit = 0
	}
	if limit > 0 {
		// fetch one extra for 'more' flag
		limit = limit + 1
	}

	ro := mvcc.RangeOptions{
		Limit: limit,
		Rev:   r.Revision,
		Count: r.CountOnly,
	}

	rr, err := txn.Range(r.Key, r.RangeEnd, ro)
	if err != nil {
		return nil, err
	}

	if r.MaxModRevision != 0 {
		f := func(kv *mvccpb.KeyValue) bool { return kv.ModRevision > r.MaxModRevision }
		pruneKVs(rr, f)
	}
	if r.MinModRevision != 0 {
		f := func(kv *mvccpb.KeyValue) bool { return kv.ModRevision < r.MinModRevision }
		pruneKVs(rr, f)
	}
	if r.MaxCreateRevision != 0 {
		f := func(kv *mvccpb.KeyValue) bool { return kv.CreateRevision > r.MaxCreateRevision }
		pruneKVs(rr, f)
	}
	if r.MinCreateRevision != 0 {
		f := func(kv *mvccpb.KeyValue) bool { return kv.CreateRevision < r.MinCreateRevision }
		pruneKVs(rr, f)
	}

	sortOrder := r.SortOrder
	if r.SortTarget != pb.RangeRequest_KEY && sortOrder == pb.RangeRequest_NONE {
		// Since current mvcc.Range implementation returns results
		// sorted by keys in lexiographically ascending order,
		// sort ASCEND by default only when target is not 'KEY'
		sortOrder = pb.RangeRequest_ASCEND
	}
	if sortOrder != pb.RangeRequest_NONE {
		var sorter sort.Interface
		switch {
		case r.SortTarget == pb.RangeRequest_KEY:
			sorter = &kvSortByKey{&kvSort{rr.KVs}}
		case r.SortTarget == pb.RangeRequest_VERSION:
			sorter = &kvSortByVersion{&kvSort{rr.KVs}}
		case r.SortTarget == pb.RangeRequest_CREATE:
			sorter = &kvSortByCreate{&kvSort{rr.KVs}}
		case r.SortTarget == pb.RangeRequest_MOD:
			sorter = &kvSortByMod{&kvSort{rr.KVs}}
		case r.SortTarget == pb.RangeRequest_VALUE:
			sorter = &kvSortByValue{&kvSort{rr.KVs}}
		}
		switch {
		case sortOrder == pb.RangeRequest_ASCEND:
			sort.Sort(sorter)
		case sortOrder == pb.RangeRequest_DESCEND:
			sort.Sort(sort.Reverse(sorter))
		}
	}

	if r.Limit > 0 && len(rr.KVs) > int(r.Limit) {
		rr.KVs = rr.KVs[:r.Limit]
		resp.More = true
	}

	resp.Header.Revision = rr.Rev
	resp.Count = int64(rr.Count)
	for i := range rr.KVs {
		if r.KeysOnly {
			rr.KVs[i].Value = nil
		}
		resp.Kvs = append(resp.Kvs, &rr.KVs[i])
	}
	return resp, nil
}

func (a *applierV3backend) Txn(rt *pb.TxnRequest) (*pb.TxnResponse, error) {
	isWrite := !isTxnReadonly(rt)
	txn := mvcc.NewReadOnlyTxnWrite(a.s.KV().Read())

	reqs, ok := a.compareToOps(txn, rt)
	if isWrite {
		if err := a.checkRequestPut(txn, reqs); err != nil {
			txn.End()
			return nil, err
		}
	}
	if err := checkRequestRange(txn, reqs); err != nil {
		txn.End()
		return nil, err
	}

	resps := make([]*pb.ResponseOp, len(reqs))
	txnResp := &pb.TxnResponse{
		Responses: resps,
		Succeeded: ok,
		Header:    &pb.ResponseHeader{},
	}

	// When executing mutable txn ops, etcd must hold the txn lock so
	// readers do not see any intermediate results. Since writes are
	// serialized on the raft loop, the revision in the read view will
	// be the revision of the write txn.
	if isWrite {
		txn.End()
		txn = a.s.KV().Write()
	}
	for i := range reqs {
		resps[i] = a.applyUnion(txn, reqs[i])
	}
	rev := txn.Rev()
	if len(txn.Changes()) != 0 {
		rev++
	}
	txn.End()

	txnResp.Header.Revision = rev
	return txnResp, nil
}

func (a *applierV3backend) compareToOps(rv mvcc.ReadView, rt *pb.TxnRequest) ([]*pb.RequestOp, bool) {
	for _, c := range rt.Compare {
		if !applyCompare(rv, c) {
			return rt.Failure, false
		}
	}
	return rt.Success, true
}

// applyCompare applies the compare request.
// If the comparison succeeds, it returns true. Otherwise, returns false.
func applyCompare(rv mvcc.ReadView, c *pb.Compare) bool {
	rr, err := rv.Range(c.Key, nil, mvcc.RangeOptions{})
	if err != nil {
		return false
	}
	var ckv mvccpb.KeyValue
	if len(rr.KVs) != 0 {
		ckv = rr.KVs[0]
	} else {
		// Use the zero value of ckv normally. However...
		if c.Target == pb.Compare_VALUE {
			// Always fail if we're comparing a value on a key that doesn't exist.
			// We can treat non-existence as the empty set explicitly, such that
			// even a key with a value of length 0 bytes is still a real key
			// that was written that way
			return false
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
		return result == 0
	case pb.Compare_NOT_EQUAL:
		return result != 0
	case pb.Compare_GREATER:
		return result > 0
	case pb.Compare_LESS:
		return result < 0
	}
	return true
}

func (a *applierV3backend) applyUnion(txn mvcc.TxnWrite, union *pb.RequestOp) *pb.ResponseOp {
	switch tv := union.Request.(type) {
	case *pb.RequestOp_RequestRange:
		if tv.RequestRange != nil {
			resp, err := a.Range(txn, tv.RequestRange)
			if err != nil {
				plog.Panicf("unexpected error during txn: %v", err)
			}
			return &pb.ResponseOp{Response: &pb.ResponseOp_ResponseRange{ResponseRange: resp}}
		}
	case *pb.RequestOp_RequestPut:
		if tv.RequestPut != nil {
			resp, err := a.Put(txn, tv.RequestPut)
			if err != nil {
				plog.Panicf("unexpected error during txn: %v", err)
			}
			return &pb.ResponseOp{Response: &pb.ResponseOp_ResponsePut{ResponsePut: resp}}
		}
	case *pb.RequestOp_RequestDeleteRange:
		if tv.RequestDeleteRange != nil {
			resp, err := a.DeleteRange(txn, tv.RequestDeleteRange)
			if err != nil {
				plog.Panicf("unexpected error during txn: %v", err)
			}
			return &pb.ResponseOp{Response: &pb.ResponseOp_ResponseDeleteRange{ResponseDeleteRange: resp}}
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
	rr, _ := a.s.KV().Range([]byte("compaction"), nil, mvcc.RangeOptions{})
	resp.Header.Revision = rr.Rev
	return resp, ch, err
}

func (a *applierV3backend) LeaseGrant(lc *pb.LeaseGrantRequest) (*pb.LeaseGrantResponse, error) {
	l, err := a.s.lessor.Grant(lease.LeaseID(lc.ID), lc.TTL)
	resp := &pb.LeaseGrantResponse{}
	if err == nil {
		resp.ID = int64(l.ID)
		resp.TTL = l.TTL()
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

func (a *applierV3backend) LeaseRevoke(lc *pb.LeaseRevokeRequest) (*pb.LeaseRevokeResponse, error) {
	err := a.s.lessor.Revoke(lease.LeaseID(lc.ID))
	return &pb.LeaseRevokeResponse{Header: newHeader(a.s)}, err
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
			a.s.applyV3 = a.s.newApplierV3()
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

func (a *applierV3Capped) Put(txn mvcc.TxnWrite, p *pb.PutRequest) (*pb.PutResponse, error) {
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
	err := a.s.AuthStore().AuthEnable()
	if err != nil {
		return nil, err
	}
	return &pb.AuthEnableResponse{Header: newHeader(a.s)}, nil
}

func (a *applierV3backend) AuthDisable() (*pb.AuthDisableResponse, error) {
	a.s.AuthStore().AuthDisable()
	return &pb.AuthDisableResponse{Header: newHeader(a.s)}, nil
}

func (a *applierV3backend) Authenticate(r *pb.InternalAuthenticateRequest) (*pb.AuthenticateResponse, error) {
	ctx := context.WithValue(context.WithValue(a.s.ctx, "index", a.s.consistIndex.ConsistentIndex()), "simpleToken", r.SimpleToken)
	resp, err := a.s.AuthStore().Authenticate(ctx, r.Name, r.Password)
	if resp != nil {
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

func (a *applierV3backend) UserAdd(r *pb.AuthUserAddRequest) (*pb.AuthUserAddResponse, error) {
	resp, err := a.s.AuthStore().UserAdd(r)
	if resp != nil {
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

func (a *applierV3backend) UserDelete(r *pb.AuthUserDeleteRequest) (*pb.AuthUserDeleteResponse, error) {
	resp, err := a.s.AuthStore().UserDelete(r)
	if resp != nil {
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

func (a *applierV3backend) UserChangePassword(r *pb.AuthUserChangePasswordRequest) (*pb.AuthUserChangePasswordResponse, error) {
	resp, err := a.s.AuthStore().UserChangePassword(r)
	if resp != nil {
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

func (a *applierV3backend) UserGrantRole(r *pb.AuthUserGrantRoleRequest) (*pb.AuthUserGrantRoleResponse, error) {
	resp, err := a.s.AuthStore().UserGrantRole(r)
	if resp != nil {
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

func (a *applierV3backend) UserGet(r *pb.AuthUserGetRequest) (*pb.AuthUserGetResponse, error) {
	resp, err := a.s.AuthStore().UserGet(r)
	if resp != nil {
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

func (a *applierV3backend) UserRevokeRole(r *pb.AuthUserRevokeRoleRequest) (*pb.AuthUserRevokeRoleResponse, error) {
	resp, err := a.s.AuthStore().UserRevokeRole(r)
	if resp != nil {
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

func (a *applierV3backend) RoleAdd(r *pb.AuthRoleAddRequest) (*pb.AuthRoleAddResponse, error) {
	resp, err := a.s.AuthStore().RoleAdd(r)
	if resp != nil {
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

func (a *applierV3backend) RoleGrantPermission(r *pb.AuthRoleGrantPermissionRequest) (*pb.AuthRoleGrantPermissionResponse, error) {
	resp, err := a.s.AuthStore().RoleGrantPermission(r)
	if resp != nil {
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

func (a *applierV3backend) RoleGet(r *pb.AuthRoleGetRequest) (*pb.AuthRoleGetResponse, error) {
	resp, err := a.s.AuthStore().RoleGet(r)
	if resp != nil {
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

func (a *applierV3backend) RoleRevokePermission(r *pb.AuthRoleRevokePermissionRequest) (*pb.AuthRoleRevokePermissionResponse, error) {
	resp, err := a.s.AuthStore().RoleRevokePermission(r)
	if resp != nil {
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

func (a *applierV3backend) RoleDelete(r *pb.AuthRoleDeleteRequest) (*pb.AuthRoleDeleteResponse, error) {
	resp, err := a.s.AuthStore().RoleDelete(r)
	if resp != nil {
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

func (a *applierV3backend) UserList(r *pb.AuthUserListRequest) (*pb.AuthUserListResponse, error) {
	resp, err := a.s.AuthStore().UserList(r)
	if resp != nil {
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

func (a *applierV3backend) RoleList(r *pb.AuthRoleListRequest) (*pb.AuthRoleListResponse, error) {
	resp, err := a.s.AuthStore().RoleList(r)
	if resp != nil {
		resp.Header = newHeader(a.s)
	}
	return resp, err
}

type quotaApplierV3 struct {
	applierV3
	q Quota
}

func newQuotaApplierV3(s *EtcdServer, app applierV3) applierV3 {
	return &quotaApplierV3{app, NewBackendQuota(s)}
}

func (a *quotaApplierV3) Put(txn mvcc.TxnWrite, p *pb.PutRequest) (*pb.PutResponse, error) {
	ok := a.q.Available(p)
	resp, err := a.applierV3.Put(txn, p)
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

type kvSort struct{ kvs []mvccpb.KeyValue }

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

func (a *applierV3backend) checkRequestPut(rv mvcc.ReadView, reqs []*pb.RequestOp) error {
	for _, requ := range reqs {
		tv, ok := requ.Request.(*pb.RequestOp_RequestPut)
		if !ok {
			continue
		}
		preq := tv.RequestPut
		if preq == nil {
			continue
		}
		if preq.IgnoreValue || preq.IgnoreLease {
			// expects previous key-value, error if not exist
			rr, err := rv.Range(preq.Key, nil, mvcc.RangeOptions{})
			if err != nil {
				return err
			}
			if rr == nil || len(rr.KVs) == 0 {
				return ErrKeyNotFound
			}
		}
		if lease.LeaseID(preq.Lease) == lease.NoLease {
			continue
		}
		if l := a.s.lessor.Lookup(lease.LeaseID(preq.Lease)); l == nil {
			return lease.ErrLeaseNotFound
		}
	}
	return nil
}

func checkRequestRange(rv mvcc.ReadView, reqs []*pb.RequestOp) error {
	for _, requ := range reqs {
		tv, ok := requ.Request.(*pb.RequestOp_RequestRange)
		if !ok {
			continue
		}
		greq := tv.RequestRange
		if greq == nil || greq.Revision == 0 {
			continue
		}

		if greq.Revision > rv.Rev() {
			return mvcc.ErrFutureRev
		}
		if greq.Revision < rv.FirstRev() {
			return mvcc.ErrCompacted
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

func noSideEffect(r *pb.InternalRaftRequest) bool {
	return r.Range != nil || r.AuthUserGet != nil || r.AuthRoleGet != nil
}

func removeNeedlessRangeReqs(txn *pb.TxnRequest) {
	f := func(ops []*pb.RequestOp) []*pb.RequestOp {
		j := 0
		for i := 0; i < len(ops); i++ {
			if _, ok := ops[i].Request.(*pb.RequestOp_RequestRange); ok {
				continue
			}
			ops[j] = ops[i]
			j++
		}

		return ops[:j]
	}

	txn.Success = f(txn.Success)
	txn.Failure = f(txn.Failure)
}

func pruneKVs(rr *mvcc.RangeResult, isPrunable func(*mvccpb.KeyValue) bool) {
	j := 0
	for i := range rr.KVs {
		rr.KVs[j] = rr.KVs[i]
		if !isPrunable(&rr.KVs[i]) {
			j++
		}
	}
	rr.KVs = rr.KVs[:j]
}

func newHeader(s *EtcdServer) *pb.ResponseHeader {
	return &pb.ResponseHeader{
		ClusterId: uint64(s.Cluster().ID()),
		MemberId:  uint64(s.ID()),
		Revision:  s.KV().Rev(),
		RaftTerm:  s.Term(),
	}
}
