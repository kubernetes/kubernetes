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

package clientv3

import (
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

type rpcFunc func(ctx context.Context) error
type retryRpcFunc func(context.Context, rpcFunc)

func (c *Client) newRetryWrapper() retryRpcFunc {
	return func(rpcCtx context.Context, f rpcFunc) {
		for {
			err := f(rpcCtx)
			if err == nil {
				return
			}
			// only retry if unavailable
			if grpc.Code(err) != codes.Unavailable {
				return
			}
			// always stop retry on etcd errors
			eErr := rpctypes.Error(err)
			if _, ok := eErr.(rpctypes.EtcdError); ok {
				return
			}
			select {
			case <-c.balancer.ConnectNotify():
			case <-rpcCtx.Done():
			case <-c.ctx.Done():
				return
			}
		}
	}
}

type retryKVClient struct {
	pb.KVClient
	retryf retryRpcFunc
}

// RetryKVClient implements a KVClient that uses the client's FailFast retry policy.
func RetryKVClient(c *Client) pb.KVClient {
	return &retryKVClient{pb.NewKVClient(c.conn), c.retryWrapper}
}

func (rkv *retryKVClient) Put(ctx context.Context, in *pb.PutRequest, opts ...grpc.CallOption) (resp *pb.PutResponse, err error) {
	rkv.retryf(ctx, func(rctx context.Context) error {
		resp, err = rkv.KVClient.Put(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rkv *retryKVClient) DeleteRange(ctx context.Context, in *pb.DeleteRangeRequest, opts ...grpc.CallOption) (resp *pb.DeleteRangeResponse, err error) {
	rkv.retryf(ctx, func(rctx context.Context) error {
		resp, err = rkv.KVClient.DeleteRange(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rkv *retryKVClient) Txn(ctx context.Context, in *pb.TxnRequest, opts ...grpc.CallOption) (resp *pb.TxnResponse, err error) {
	rkv.retryf(ctx, func(rctx context.Context) error {
		resp, err = rkv.KVClient.Txn(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rkv *retryKVClient) Compact(ctx context.Context, in *pb.CompactionRequest, opts ...grpc.CallOption) (resp *pb.CompactionResponse, err error) {
	rkv.retryf(ctx, func(rctx context.Context) error {
		resp, err = rkv.KVClient.Compact(rctx, in, opts...)
		return err
	})
	return resp, err
}

type retryLeaseClient struct {
	pb.LeaseClient
	retryf retryRpcFunc
}

// RetryLeaseClient implements a LeaseClient that uses the client's FailFast retry policy.
func RetryLeaseClient(c *Client) pb.LeaseClient {
	return &retryLeaseClient{pb.NewLeaseClient(c.conn), c.retryWrapper}
}

func (rlc *retryLeaseClient) LeaseGrant(ctx context.Context, in *pb.LeaseGrantRequest, opts ...grpc.CallOption) (resp *pb.LeaseGrantResponse, err error) {
	rlc.retryf(ctx, func(rctx context.Context) error {
		resp, err = rlc.LeaseClient.LeaseGrant(rctx, in, opts...)
		return err
	})
	return resp, err

}

func (rlc *retryLeaseClient) LeaseRevoke(ctx context.Context, in *pb.LeaseRevokeRequest, opts ...grpc.CallOption) (resp *pb.LeaseRevokeResponse, err error) {
	rlc.retryf(ctx, func(rctx context.Context) error {
		resp, err = rlc.LeaseClient.LeaseRevoke(rctx, in, opts...)
		return err
	})
	return resp, err
}

type retryClusterClient struct {
	pb.ClusterClient
	retryf retryRpcFunc
}

// RetryClusterClient implements a ClusterClient that uses the client's FailFast retry policy.
func RetryClusterClient(c *Client) pb.ClusterClient {
	return &retryClusterClient{pb.NewClusterClient(c.conn), c.retryWrapper}
}

func (rcc *retryClusterClient) MemberAdd(ctx context.Context, in *pb.MemberAddRequest, opts ...grpc.CallOption) (resp *pb.MemberAddResponse, err error) {
	rcc.retryf(ctx, func(rctx context.Context) error {
		resp, err = rcc.ClusterClient.MemberAdd(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rcc *retryClusterClient) MemberRemove(ctx context.Context, in *pb.MemberRemoveRequest, opts ...grpc.CallOption) (resp *pb.MemberRemoveResponse, err error) {
	rcc.retryf(ctx, func(rctx context.Context) error {
		resp, err = rcc.ClusterClient.MemberRemove(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rcc *retryClusterClient) MemberUpdate(ctx context.Context, in *pb.MemberUpdateRequest, opts ...grpc.CallOption) (resp *pb.MemberUpdateResponse, err error) {
	rcc.retryf(ctx, func(rctx context.Context) error {
		resp, err = rcc.ClusterClient.MemberUpdate(rctx, in, opts...)
		return err
	})
	return resp, err
}

type retryAuthClient struct {
	pb.AuthClient
	retryf retryRpcFunc
}

// RetryAuthClient implements a AuthClient that uses the client's FailFast retry policy.
func RetryAuthClient(c *Client) pb.AuthClient {
	return &retryAuthClient{pb.NewAuthClient(c.conn), c.retryWrapper}
}

func (rac *retryAuthClient) AuthEnable(ctx context.Context, in *pb.AuthEnableRequest, opts ...grpc.CallOption) (resp *pb.AuthEnableResponse, err error) {
	rac.retryf(ctx, func(rctx context.Context) error {
		resp, err = rac.AuthClient.AuthEnable(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *retryAuthClient) AuthDisable(ctx context.Context, in *pb.AuthDisableRequest, opts ...grpc.CallOption) (resp *pb.AuthDisableResponse, err error) {
	rac.retryf(ctx, func(rctx context.Context) error {
		resp, err = rac.AuthClient.AuthDisable(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *retryAuthClient) UserAdd(ctx context.Context, in *pb.AuthUserAddRequest, opts ...grpc.CallOption) (resp *pb.AuthUserAddResponse, err error) {
	rac.retryf(ctx, func(rctx context.Context) error {
		resp, err = rac.AuthClient.UserAdd(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *retryAuthClient) UserDelete(ctx context.Context, in *pb.AuthUserDeleteRequest, opts ...grpc.CallOption) (resp *pb.AuthUserDeleteResponse, err error) {
	rac.retryf(ctx, func(rctx context.Context) error {
		resp, err = rac.AuthClient.UserDelete(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *retryAuthClient) UserChangePassword(ctx context.Context, in *pb.AuthUserChangePasswordRequest, opts ...grpc.CallOption) (resp *pb.AuthUserChangePasswordResponse, err error) {
	rac.retryf(ctx, func(rctx context.Context) error {
		resp, err = rac.AuthClient.UserChangePassword(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *retryAuthClient) UserGrantRole(ctx context.Context, in *pb.AuthUserGrantRoleRequest, opts ...grpc.CallOption) (resp *pb.AuthUserGrantRoleResponse, err error) {
	rac.retryf(ctx, func(rctx context.Context) error {
		resp, err = rac.AuthClient.UserGrantRole(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *retryAuthClient) UserRevokeRole(ctx context.Context, in *pb.AuthUserRevokeRoleRequest, opts ...grpc.CallOption) (resp *pb.AuthUserRevokeRoleResponse, err error) {
	rac.retryf(ctx, func(rctx context.Context) error {
		resp, err = rac.AuthClient.UserRevokeRole(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *retryAuthClient) RoleAdd(ctx context.Context, in *pb.AuthRoleAddRequest, opts ...grpc.CallOption) (resp *pb.AuthRoleAddResponse, err error) {
	rac.retryf(ctx, func(rctx context.Context) error {
		resp, err = rac.AuthClient.RoleAdd(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *retryAuthClient) RoleDelete(ctx context.Context, in *pb.AuthRoleDeleteRequest, opts ...grpc.CallOption) (resp *pb.AuthRoleDeleteResponse, err error) {
	rac.retryf(ctx, func(rctx context.Context) error {
		resp, err = rac.AuthClient.RoleDelete(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *retryAuthClient) RoleGrantPermission(ctx context.Context, in *pb.AuthRoleGrantPermissionRequest, opts ...grpc.CallOption) (resp *pb.AuthRoleGrantPermissionResponse, err error) {
	rac.retryf(ctx, func(rctx context.Context) error {
		resp, err = rac.AuthClient.RoleGrantPermission(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *retryAuthClient) RoleRevokePermission(ctx context.Context, in *pb.AuthRoleRevokePermissionRequest, opts ...grpc.CallOption) (resp *pb.AuthRoleRevokePermissionResponse, err error) {
	rac.retryf(ctx, func(rctx context.Context) error {
		resp, err = rac.AuthClient.RoleRevokePermission(rctx, in, opts...)
		return err
	})
	return resp, err
}
