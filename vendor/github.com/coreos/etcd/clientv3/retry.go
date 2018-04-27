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
	"google.golang.org/grpc/status"
)

type rpcFunc func(ctx context.Context) error
type retryRPCFunc func(context.Context, rpcFunc) error
type retryStopErrFunc func(error) bool

func isRepeatableStopError(err error) bool {
	eErr := rpctypes.Error(err)
	// always stop retry on etcd errors
	if serverErr, ok := eErr.(rpctypes.EtcdError); ok && serverErr.Code() != codes.Unavailable {
		return true
	}
	// only retry if unavailable
	ev, _ := status.FromError(err)
	return ev.Code() != codes.Unavailable
}

func isNonRepeatableStopError(err error) bool {
	ev, _ := status.FromError(err)
	if ev.Code() != codes.Unavailable {
		return true
	}
	desc := rpctypes.ErrorDesc(err)
	return desc != "there is no address available" && desc != "there is no connection available"
}

func (c *Client) newRetryWrapper(isStop retryStopErrFunc) retryRPCFunc {
	return func(rpcCtx context.Context, f rpcFunc) error {
		for {
			if err := readyWait(rpcCtx, c.ctx, c.balancer.ConnectNotify()); err != nil {
				return err
			}
			pinned := c.balancer.pinned()
			err := f(rpcCtx)
			if err == nil {
				return nil
			}
			if logger.V(4) {
				logger.Infof("clientv3/retry: error %q on pinned endpoint %q", err.Error(), pinned)
			}

			if s, ok := status.FromError(err); ok && (s.Code() == codes.Unavailable || s.Code() == codes.DeadlineExceeded || s.Code() == codes.Internal) {
				// mark this before endpoint switch is triggered
				c.balancer.hostPortError(pinned, err)
				c.balancer.next()
				if logger.V(4) {
					logger.Infof("clientv3/retry: switching from %q due to error %q", pinned, err.Error())
				}
			}

			if isStop(err) {
				return err
			}
		}
	}
}

func (c *Client) newAuthRetryWrapper() retryRPCFunc {
	return func(rpcCtx context.Context, f rpcFunc) error {
		for {
			pinned := c.balancer.pinned()
			err := f(rpcCtx)
			if err == nil {
				return nil
			}
			if logger.V(4) {
				logger.Infof("clientv3/auth-retry: error %q on pinned endpoint %q", err.Error(), pinned)
			}
			// always stop retry on etcd errors other than invalid auth token
			if rpctypes.Error(err) == rpctypes.ErrInvalidAuthToken {
				gterr := c.getToken(rpcCtx)
				if gterr != nil {
					if logger.V(4) {
						logger.Infof("clientv3/auth-retry: cannot retry due to error %q(%q) on pinned endpoint %q", err.Error(), gterr.Error(), pinned)
					}
					return err // return the original error for simplicity
				}
				continue
			}
			return err
		}
	}
}

// RetryKVClient implements a KVClient.
func RetryKVClient(c *Client) pb.KVClient {
	repeatableRetry := c.newRetryWrapper(isRepeatableStopError)
	nonRepeatableRetry := c.newRetryWrapper(isNonRepeatableStopError)
	conn := pb.NewKVClient(c.conn)
	retryBasic := &retryKVClient{&nonRepeatableKVClient{conn, nonRepeatableRetry}, repeatableRetry}
	retryAuthWrapper := c.newAuthRetryWrapper()
	return &retryKVClient{
		&nonRepeatableKVClient{retryBasic, retryAuthWrapper},
		retryAuthWrapper}
}

type retryKVClient struct {
	*nonRepeatableKVClient
	repeatableRetry retryRPCFunc
}

func (rkv *retryKVClient) Range(ctx context.Context, in *pb.RangeRequest, opts ...grpc.CallOption) (resp *pb.RangeResponse, err error) {
	err = rkv.repeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rkv.kc.Range(rctx, in, opts...)
		return err
	})
	return resp, err
}

type nonRepeatableKVClient struct {
	kc                 pb.KVClient
	nonRepeatableRetry retryRPCFunc
}

func (rkv *nonRepeatableKVClient) Put(ctx context.Context, in *pb.PutRequest, opts ...grpc.CallOption) (resp *pb.PutResponse, err error) {
	err = rkv.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rkv.kc.Put(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rkv *nonRepeatableKVClient) DeleteRange(ctx context.Context, in *pb.DeleteRangeRequest, opts ...grpc.CallOption) (resp *pb.DeleteRangeResponse, err error) {
	err = rkv.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rkv.kc.DeleteRange(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rkv *nonRepeatableKVClient) Txn(ctx context.Context, in *pb.TxnRequest, opts ...grpc.CallOption) (resp *pb.TxnResponse, err error) {
	// TODO: repeatableRetry if read-only txn
	err = rkv.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rkv.kc.Txn(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rkv *nonRepeatableKVClient) Compact(ctx context.Context, in *pb.CompactionRequest, opts ...grpc.CallOption) (resp *pb.CompactionResponse, err error) {
	err = rkv.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rkv.kc.Compact(rctx, in, opts...)
		return err
	})
	return resp, err
}

type retryLeaseClient struct {
	lc              pb.LeaseClient
	repeatableRetry retryRPCFunc
}

// RetryLeaseClient implements a LeaseClient.
func RetryLeaseClient(c *Client) pb.LeaseClient {
	retry := &retryLeaseClient{
		pb.NewLeaseClient(c.conn),
		c.newRetryWrapper(isRepeatableStopError),
	}
	return &retryLeaseClient{retry, c.newAuthRetryWrapper()}
}

func (rlc *retryLeaseClient) LeaseTimeToLive(ctx context.Context, in *pb.LeaseTimeToLiveRequest, opts ...grpc.CallOption) (resp *pb.LeaseTimeToLiveResponse, err error) {
	err = rlc.repeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rlc.lc.LeaseTimeToLive(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rlc *retryLeaseClient) LeaseGrant(ctx context.Context, in *pb.LeaseGrantRequest, opts ...grpc.CallOption) (resp *pb.LeaseGrantResponse, err error) {
	err = rlc.repeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rlc.lc.LeaseGrant(rctx, in, opts...)
		return err
	})
	return resp, err

}

func (rlc *retryLeaseClient) LeaseRevoke(ctx context.Context, in *pb.LeaseRevokeRequest, opts ...grpc.CallOption) (resp *pb.LeaseRevokeResponse, err error) {
	err = rlc.repeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rlc.lc.LeaseRevoke(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rlc *retryLeaseClient) LeaseKeepAlive(ctx context.Context, opts ...grpc.CallOption) (stream pb.Lease_LeaseKeepAliveClient, err error) {
	err = rlc.repeatableRetry(ctx, func(rctx context.Context) error {
		stream, err = rlc.lc.LeaseKeepAlive(rctx, opts...)
		return err
	})
	return stream, err
}

type retryClusterClient struct {
	*nonRepeatableClusterClient
	repeatableRetry retryRPCFunc
}

// RetryClusterClient implements a ClusterClient.
func RetryClusterClient(c *Client) pb.ClusterClient {
	repeatableRetry := c.newRetryWrapper(isRepeatableStopError)
	nonRepeatableRetry := c.newRetryWrapper(isNonRepeatableStopError)
	cc := pb.NewClusterClient(c.conn)
	return &retryClusterClient{&nonRepeatableClusterClient{cc, nonRepeatableRetry}, repeatableRetry}
}

func (rcc *retryClusterClient) MemberList(ctx context.Context, in *pb.MemberListRequest, opts ...grpc.CallOption) (resp *pb.MemberListResponse, err error) {
	err = rcc.repeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rcc.cc.MemberList(rctx, in, opts...)
		return err
	})
	return resp, err
}

type nonRepeatableClusterClient struct {
	cc                 pb.ClusterClient
	nonRepeatableRetry retryRPCFunc
}

func (rcc *nonRepeatableClusterClient) MemberAdd(ctx context.Context, in *pb.MemberAddRequest, opts ...grpc.CallOption) (resp *pb.MemberAddResponse, err error) {
	err = rcc.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rcc.cc.MemberAdd(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rcc *nonRepeatableClusterClient) MemberRemove(ctx context.Context, in *pb.MemberRemoveRequest, opts ...grpc.CallOption) (resp *pb.MemberRemoveResponse, err error) {
	err = rcc.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rcc.cc.MemberRemove(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rcc *nonRepeatableClusterClient) MemberUpdate(ctx context.Context, in *pb.MemberUpdateRequest, opts ...grpc.CallOption) (resp *pb.MemberUpdateResponse, err error) {
	err = rcc.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rcc.cc.MemberUpdate(rctx, in, opts...)
		return err
	})
	return resp, err
}

// RetryMaintenanceClient implements a Maintenance.
func RetryMaintenanceClient(c *Client, conn *grpc.ClientConn) pb.MaintenanceClient {
	repeatableRetry := c.newRetryWrapper(isRepeatableStopError)
	nonRepeatableRetry := c.newRetryWrapper(isNonRepeatableStopError)
	mc := pb.NewMaintenanceClient(conn)
	return &retryMaintenanceClient{&nonRepeatableMaintenanceClient{mc, nonRepeatableRetry}, repeatableRetry}
}

type retryMaintenanceClient struct {
	*nonRepeatableMaintenanceClient
	repeatableRetry retryRPCFunc
}

func (rmc *retryMaintenanceClient) Alarm(ctx context.Context, in *pb.AlarmRequest, opts ...grpc.CallOption) (resp *pb.AlarmResponse, err error) {
	err = rmc.repeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rmc.mc.Alarm(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rmc *retryMaintenanceClient) Status(ctx context.Context, in *pb.StatusRequest, opts ...grpc.CallOption) (resp *pb.StatusResponse, err error) {
	err = rmc.repeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rmc.mc.Status(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rmc *retryMaintenanceClient) Hash(ctx context.Context, in *pb.HashRequest, opts ...grpc.CallOption) (resp *pb.HashResponse, err error) {
	err = rmc.repeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rmc.mc.Hash(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rmc *retryMaintenanceClient) Snapshot(ctx context.Context, in *pb.SnapshotRequest, opts ...grpc.CallOption) (stream pb.Maintenance_SnapshotClient, err error) {
	err = rmc.repeatableRetry(ctx, func(rctx context.Context) error {
		stream, err = rmc.mc.Snapshot(rctx, in, opts...)
		return err
	})
	return stream, err
}

type nonRepeatableMaintenanceClient struct {
	mc                 pb.MaintenanceClient
	nonRepeatableRetry retryRPCFunc
}

func (rmc *nonRepeatableMaintenanceClient) Defragment(ctx context.Context, in *pb.DefragmentRequest, opts ...grpc.CallOption) (resp *pb.DefragmentResponse, err error) {
	err = rmc.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rmc.mc.Defragment(rctx, in, opts...)
		return err
	})
	return resp, err
}

type retryAuthClient struct {
	*nonRepeatableAuthClient
	repeatableRetry retryRPCFunc
}

// RetryAuthClient implements a AuthClient.
func RetryAuthClient(c *Client) pb.AuthClient {
	repeatableRetry := c.newRetryWrapper(isRepeatableStopError)
	nonRepeatableRetry := c.newRetryWrapper(isNonRepeatableStopError)
	ac := pb.NewAuthClient(c.conn)
	return &retryAuthClient{&nonRepeatableAuthClient{ac, nonRepeatableRetry}, repeatableRetry}
}

func (rac *retryAuthClient) UserList(ctx context.Context, in *pb.AuthUserListRequest, opts ...grpc.CallOption) (resp *pb.AuthUserListResponse, err error) {
	err = rac.repeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.UserList(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *retryAuthClient) UserGet(ctx context.Context, in *pb.AuthUserGetRequest, opts ...grpc.CallOption) (resp *pb.AuthUserGetResponse, err error) {
	err = rac.repeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.UserGet(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *retryAuthClient) RoleGet(ctx context.Context, in *pb.AuthRoleGetRequest, opts ...grpc.CallOption) (resp *pb.AuthRoleGetResponse, err error) {
	err = rac.repeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.RoleGet(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *retryAuthClient) RoleList(ctx context.Context, in *pb.AuthRoleListRequest, opts ...grpc.CallOption) (resp *pb.AuthRoleListResponse, err error) {
	err = rac.repeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.RoleList(rctx, in, opts...)
		return err
	})
	return resp, err
}

type nonRepeatableAuthClient struct {
	ac                 pb.AuthClient
	nonRepeatableRetry retryRPCFunc
}

func (rac *nonRepeatableAuthClient) AuthEnable(ctx context.Context, in *pb.AuthEnableRequest, opts ...grpc.CallOption) (resp *pb.AuthEnableResponse, err error) {
	err = rac.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.AuthEnable(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *nonRepeatableAuthClient) AuthDisable(ctx context.Context, in *pb.AuthDisableRequest, opts ...grpc.CallOption) (resp *pb.AuthDisableResponse, err error) {
	err = rac.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.AuthDisable(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *nonRepeatableAuthClient) UserAdd(ctx context.Context, in *pb.AuthUserAddRequest, opts ...grpc.CallOption) (resp *pb.AuthUserAddResponse, err error) {
	err = rac.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.UserAdd(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *nonRepeatableAuthClient) UserDelete(ctx context.Context, in *pb.AuthUserDeleteRequest, opts ...grpc.CallOption) (resp *pb.AuthUserDeleteResponse, err error) {
	err = rac.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.UserDelete(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *nonRepeatableAuthClient) UserChangePassword(ctx context.Context, in *pb.AuthUserChangePasswordRequest, opts ...grpc.CallOption) (resp *pb.AuthUserChangePasswordResponse, err error) {
	err = rac.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.UserChangePassword(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *nonRepeatableAuthClient) UserGrantRole(ctx context.Context, in *pb.AuthUserGrantRoleRequest, opts ...grpc.CallOption) (resp *pb.AuthUserGrantRoleResponse, err error) {
	err = rac.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.UserGrantRole(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *nonRepeatableAuthClient) UserRevokeRole(ctx context.Context, in *pb.AuthUserRevokeRoleRequest, opts ...grpc.CallOption) (resp *pb.AuthUserRevokeRoleResponse, err error) {
	err = rac.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.UserRevokeRole(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *nonRepeatableAuthClient) RoleAdd(ctx context.Context, in *pb.AuthRoleAddRequest, opts ...grpc.CallOption) (resp *pb.AuthRoleAddResponse, err error) {
	err = rac.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.RoleAdd(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *nonRepeatableAuthClient) RoleDelete(ctx context.Context, in *pb.AuthRoleDeleteRequest, opts ...grpc.CallOption) (resp *pb.AuthRoleDeleteResponse, err error) {
	err = rac.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.RoleDelete(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *nonRepeatableAuthClient) RoleGrantPermission(ctx context.Context, in *pb.AuthRoleGrantPermissionRequest, opts ...grpc.CallOption) (resp *pb.AuthRoleGrantPermissionResponse, err error) {
	err = rac.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.RoleGrantPermission(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *nonRepeatableAuthClient) RoleRevokePermission(ctx context.Context, in *pb.AuthRoleRevokePermissionRequest, opts ...grpc.CallOption) (resp *pb.AuthRoleRevokePermissionResponse, err error) {
	err = rac.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.RoleRevokePermission(rctx, in, opts...)
		return err
	})
	return resp, err
}

func (rac *nonRepeatableAuthClient) Authenticate(ctx context.Context, in *pb.AuthenticateRequest, opts ...grpc.CallOption) (resp *pb.AuthenticateResponse, err error) {
	err = rac.nonRepeatableRetry(ctx, func(rctx context.Context) error {
		resp, err = rac.ac.Authenticate(rctx, in, opts...)
		return err
	})
	return resp, err
}
