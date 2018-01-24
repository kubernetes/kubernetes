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
	"context"
	"io"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"

	"google.golang.org/grpc"
)

type (
	DefragmentResponse pb.DefragmentResponse
	AlarmResponse      pb.AlarmResponse
	AlarmMember        pb.AlarmMember
	StatusResponse     pb.StatusResponse
	HashKVResponse     pb.HashKVResponse
	MoveLeaderResponse pb.MoveLeaderResponse
)

type Maintenance interface {
	// AlarmList gets all active alarms.
	AlarmList(ctx context.Context) (*AlarmResponse, error)

	// AlarmDisarm disarms a given alarm.
	AlarmDisarm(ctx context.Context, m *AlarmMember) (*AlarmResponse, error)

	// Defragment releases wasted space from internal fragmentation on a given etcd member.
	// Defragment is only needed when deleting a large number of keys and want to reclaim
	// the resources.
	// Defragment is an expensive operation. User should avoid defragmenting multiple members
	// at the same time.
	// To defragment multiple members in the cluster, user need to call defragment multiple
	// times with different endpoints.
	Defragment(ctx context.Context, endpoint string) (*DefragmentResponse, error)

	// Status gets the status of the endpoint.
	Status(ctx context.Context, endpoint string) (*StatusResponse, error)

	// HashKV returns a hash of the KV state at the time of the RPC.
	// If revision is zero, the hash is computed on all keys. If the revision
	// is non-zero, the hash is computed on all keys at or below the given revision.
	HashKV(ctx context.Context, endpoint string, rev int64) (*HashKVResponse, error)

	// Snapshot provides a reader for a point-in-time snapshot of etcd.
	Snapshot(ctx context.Context) (io.ReadCloser, error)

	// MoveLeader requests current leader to transfer its leadership to the transferee.
	// Request must be made to the leader.
	MoveLeader(ctx context.Context, transfereeID uint64) (*MoveLeaderResponse, error)
}

type maintenance struct {
	dial     func(endpoint string) (pb.MaintenanceClient, func(), error)
	remote   pb.MaintenanceClient
	callOpts []grpc.CallOption
}

func NewMaintenance(c *Client) Maintenance {
	api := &maintenance{
		dial: func(endpoint string) (pb.MaintenanceClient, func(), error) {
			conn, err := c.dial(endpoint)
			if err != nil {
				return nil, nil, err
			}
			cancel := func() { conn.Close() }
			return RetryMaintenanceClient(c, conn), cancel, nil
		},
		remote: RetryMaintenanceClient(c, c.conn),
	}
	if c != nil {
		api.callOpts = c.callOpts
	}
	return api
}

func NewMaintenanceFromMaintenanceClient(remote pb.MaintenanceClient, c *Client) Maintenance {
	api := &maintenance{
		dial: func(string) (pb.MaintenanceClient, func(), error) {
			return remote, func() {}, nil
		},
		remote: remote,
	}
	if c != nil {
		api.callOpts = c.callOpts
	}
	return api
}

func (m *maintenance) AlarmList(ctx context.Context) (*AlarmResponse, error) {
	req := &pb.AlarmRequest{
		Action:   pb.AlarmRequest_GET,
		MemberID: 0,                 // all
		Alarm:    pb.AlarmType_NONE, // all
	}
	resp, err := m.remote.Alarm(ctx, req, m.callOpts...)
	if err == nil {
		return (*AlarmResponse)(resp), nil
	}
	return nil, toErr(ctx, err)
}

func (m *maintenance) AlarmDisarm(ctx context.Context, am *AlarmMember) (*AlarmResponse, error) {
	req := &pb.AlarmRequest{
		Action:   pb.AlarmRequest_DEACTIVATE,
		MemberID: am.MemberID,
		Alarm:    am.Alarm,
	}

	if req.MemberID == 0 && req.Alarm == pb.AlarmType_NONE {
		ar, err := m.AlarmList(ctx)
		if err != nil {
			return nil, toErr(ctx, err)
		}
		ret := AlarmResponse{}
		for _, am := range ar.Alarms {
			dresp, derr := m.AlarmDisarm(ctx, (*AlarmMember)(am))
			if derr != nil {
				return nil, toErr(ctx, derr)
			}
			ret.Alarms = append(ret.Alarms, dresp.Alarms...)
		}
		return &ret, nil
	}

	resp, err := m.remote.Alarm(ctx, req, m.callOpts...)
	if err == nil {
		return (*AlarmResponse)(resp), nil
	}
	return nil, toErr(ctx, err)
}

func (m *maintenance) Defragment(ctx context.Context, endpoint string) (*DefragmentResponse, error) {
	remote, cancel, err := m.dial(endpoint)
	if err != nil {
		return nil, toErr(ctx, err)
	}
	defer cancel()
	resp, err := remote.Defragment(ctx, &pb.DefragmentRequest{}, m.callOpts...)
	if err != nil {
		return nil, toErr(ctx, err)
	}
	return (*DefragmentResponse)(resp), nil
}

func (m *maintenance) Status(ctx context.Context, endpoint string) (*StatusResponse, error) {
	remote, cancel, err := m.dial(endpoint)
	if err != nil {
		return nil, toErr(ctx, err)
	}
	defer cancel()
	resp, err := remote.Status(ctx, &pb.StatusRequest{}, m.callOpts...)
	if err != nil {
		return nil, toErr(ctx, err)
	}
	return (*StatusResponse)(resp), nil
}

func (m *maintenance) HashKV(ctx context.Context, endpoint string, rev int64) (*HashKVResponse, error) {
	remote, cancel, err := m.dial(endpoint)
	if err != nil {
		return nil, toErr(ctx, err)
	}
	defer cancel()
	resp, err := remote.HashKV(ctx, &pb.HashKVRequest{Revision: rev}, m.callOpts...)
	if err != nil {
		return nil, toErr(ctx, err)
	}
	return (*HashKVResponse)(resp), nil
}

func (m *maintenance) Snapshot(ctx context.Context) (io.ReadCloser, error) {
	ss, err := m.remote.Snapshot(ctx, &pb.SnapshotRequest{}, m.callOpts...)
	if err != nil {
		return nil, toErr(ctx, err)
	}

	pr, pw := io.Pipe()
	go func() {
		for {
			resp, err := ss.Recv()
			if err != nil {
				pw.CloseWithError(err)
				return
			}
			if resp == nil && err == nil {
				break
			}
			if _, werr := pw.Write(resp.Blob); werr != nil {
				pw.CloseWithError(werr)
				return
			}
		}
		pw.Close()
	}()
	return &snapshotReadCloser{ctx: ctx, ReadCloser: pr}, nil
}

type snapshotReadCloser struct {
	ctx context.Context
	io.ReadCloser
}

func (rc *snapshotReadCloser) Read(p []byte) (n int, err error) {
	n, err = rc.ReadCloser.Read(p)
	return n, toErr(rc.ctx, err)
}

func (m *maintenance) MoveLeader(ctx context.Context, transfereeID uint64) (*MoveLeaderResponse, error) {
	resp, err := m.remote.MoveLeader(ctx, &pb.MoveLeaderRequest{TargetID: transfereeID}, m.callOpts...)
	return (*MoveLeaderResponse)(resp), toErr(ctx, err)
}
