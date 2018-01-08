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
	"sync"

	"github.com/coreos/etcd/auth"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/mvcc"
)

type authApplierV3 struct {
	applierV3
	as auth.AuthStore

	// mu serializes Apply so that user isn't corrupted and so that
	// serialized requests don't leak data from TOCTOU errors
	mu sync.Mutex

	authInfo auth.AuthInfo
}

func newAuthApplierV3(as auth.AuthStore, base applierV3) *authApplierV3 {
	return &authApplierV3{applierV3: base, as: as}
}

func (aa *authApplierV3) Apply(r *pb.InternalRaftRequest) *applyResult {
	aa.mu.Lock()
	defer aa.mu.Unlock()
	if r.Header != nil {
		// backward-compatible with pre-3.0 releases when internalRaftRequest
		// does not have header field
		aa.authInfo.Username = r.Header.Username
		aa.authInfo.Revision = r.Header.AuthRevision
	}
	if needAdminPermission(r) {
		if err := aa.as.IsAdminPermitted(&aa.authInfo); err != nil {
			aa.authInfo.Username = ""
			aa.authInfo.Revision = 0
			return &applyResult{err: err}
		}
	}
	ret := aa.applierV3.Apply(r)
	aa.authInfo.Username = ""
	aa.authInfo.Revision = 0
	return ret
}

func (aa *authApplierV3) Put(txn mvcc.TxnWrite, r *pb.PutRequest) (*pb.PutResponse, error) {
	if err := aa.as.IsPutPermitted(&aa.authInfo, r.Key); err != nil {
		return nil, err
	}
	if r.PrevKv {
		err := aa.as.IsRangePermitted(&aa.authInfo, r.Key, nil)
		if err != nil {
			return nil, err
		}
	}
	return aa.applierV3.Put(txn, r)
}

func (aa *authApplierV3) Range(txn mvcc.TxnRead, r *pb.RangeRequest) (*pb.RangeResponse, error) {
	if err := aa.as.IsRangePermitted(&aa.authInfo, r.Key, r.RangeEnd); err != nil {
		return nil, err
	}
	return aa.applierV3.Range(txn, r)
}

func (aa *authApplierV3) DeleteRange(txn mvcc.TxnWrite, r *pb.DeleteRangeRequest) (*pb.DeleteRangeResponse, error) {
	if err := aa.as.IsDeleteRangePermitted(&aa.authInfo, r.Key, r.RangeEnd); err != nil {
		return nil, err
	}
	if r.PrevKv {
		err := aa.as.IsRangePermitted(&aa.authInfo, r.Key, r.RangeEnd)
		if err != nil {
			return nil, err
		}
	}

	return aa.applierV3.DeleteRange(txn, r)
}

func checkTxnReqsPermission(as auth.AuthStore, ai *auth.AuthInfo, reqs []*pb.RequestOp) error {
	for _, requ := range reqs {
		switch tv := requ.Request.(type) {
		case *pb.RequestOp_RequestRange:
			if tv.RequestRange == nil {
				continue
			}

			if err := as.IsRangePermitted(ai, tv.RequestRange.Key, tv.RequestRange.RangeEnd); err != nil {
				return err
			}

		case *pb.RequestOp_RequestPut:
			if tv.RequestPut == nil {
				continue
			}

			if err := as.IsPutPermitted(ai, tv.RequestPut.Key); err != nil {
				return err
			}

		case *pb.RequestOp_RequestDeleteRange:
			if tv.RequestDeleteRange == nil {
				continue
			}

			if tv.RequestDeleteRange.PrevKv {
				err := as.IsRangePermitted(ai, tv.RequestDeleteRange.Key, tv.RequestDeleteRange.RangeEnd)
				if err != nil {
					return err
				}
			}

			err := as.IsDeleteRangePermitted(ai, tv.RequestDeleteRange.Key, tv.RequestDeleteRange.RangeEnd)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func checkTxnAuth(as auth.AuthStore, ai *auth.AuthInfo, rt *pb.TxnRequest) error {
	for _, c := range rt.Compare {
		if err := as.IsRangePermitted(ai, c.Key, nil); err != nil {
			return err
		}
	}
	if err := checkTxnReqsPermission(as, ai, rt.Success); err != nil {
		return err
	}
	if err := checkTxnReqsPermission(as, ai, rt.Failure); err != nil {
		return err
	}
	return nil
}

func (aa *authApplierV3) Txn(rt *pb.TxnRequest) (*pb.TxnResponse, error) {
	if err := checkTxnAuth(aa.as, &aa.authInfo, rt); err != nil {
		return nil, err
	}
	return aa.applierV3.Txn(rt)
}

func needAdminPermission(r *pb.InternalRaftRequest) bool {
	switch {
	case r.AuthEnable != nil:
		return true
	case r.AuthDisable != nil:
		return true
	case r.AuthUserAdd != nil:
		return true
	case r.AuthUserDelete != nil:
		return true
	case r.AuthUserChangePassword != nil:
		return true
	case r.AuthUserGrantRole != nil:
		return true
	case r.AuthUserGet != nil:
		return true
	case r.AuthUserRevokeRole != nil:
		return true
	case r.AuthRoleAdd != nil:
		return true
	case r.AuthRoleGrantPermission != nil:
		return true
	case r.AuthRoleGet != nil:
		return true
	case r.AuthRoleRevokePermission != nil:
		return true
	case r.AuthRoleDelete != nil:
		return true
	case r.AuthUserList != nil:
		return true
	case r.AuthRoleList != nil:
		return true
	default:
		return false
	}
}
