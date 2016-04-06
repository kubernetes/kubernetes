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

package clientv3

import (
	"sync"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

type (
	DefragmentResponse pb.DefragmentResponse
	AlarmResponse      pb.AlarmResponse
	AlarmMember        pb.AlarmMember
	StatusResponse     pb.StatusResponse
)

type Maintenance interface {
	// AlarmList gets all active alarms.
	AlarmList(ctx context.Context) (*AlarmResponse, error)

	// AlarmDisarm disarms a given alarm.
	AlarmDisarm(ctx context.Context, m *AlarmMember) (*AlarmResponse, error)

	// Defragment defragments storage backend of the etcd member with given endpoint.
	// Defragment is only needed when deleting a large number of keys and want to reclaim
	// the resources.
	// Defragment is an expensive operation. User should avoid defragmenting multiple members
	// at the same time.
	// To defragment multiple members in the cluster, user need to call defragment multiple
	// times with different endpoints.
	Defragment(ctx context.Context, endpoint string) (*DefragmentResponse, error)

	// Status gets the status of the member.
	Status(ctx context.Context, endpoint string) (*StatusResponse, error)
}

type maintenance struct {
	c *Client

	mu     sync.Mutex
	conn   *grpc.ClientConn // conn in-use
	remote pb.MaintenanceClient
}

func NewMaintenance(c *Client) Maintenance {
	conn := c.ActiveConnection()
	return &maintenance{
		c:      c,
		conn:   conn,
		remote: pb.NewMaintenanceClient(conn),
	}
}

func (m *maintenance) AlarmList(ctx context.Context) (*AlarmResponse, error) {
	req := &pb.AlarmRequest{
		Action:   pb.AlarmRequest_GET,
		MemberID: 0,                 // all
		Alarm:    pb.AlarmType_NONE, // all
	}
	for {
		resp, err := m.getRemote().Alarm(ctx, req)
		if err == nil {
			return (*AlarmResponse)(resp), nil
		}
		if isHalted(ctx, err) {
			return nil, err
		}
		if err = m.switchRemote(err); err != nil {
			return nil, err
		}
	}
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
			return nil, err
		}
		ret := AlarmResponse{}
		for _, am := range ar.Alarms {
			dresp, derr := m.AlarmDisarm(ctx, (*AlarmMember)(am))
			if derr != nil {
				return nil, derr
			}
			ret.Alarms = append(ret.Alarms, dresp.Alarms...)
		}
		return &ret, nil
	}

	resp, err := m.getRemote().Alarm(ctx, req)
	if err == nil {
		return (*AlarmResponse)(resp), nil
	}
	if !isHalted(ctx, err) {
		go m.switchRemote(err)
	}
	return nil, err
}

func (m *maintenance) Defragment(ctx context.Context, endpoint string) (*DefragmentResponse, error) {
	conn, err := m.c.Dial(endpoint)
	if err != nil {
		return nil, err
	}
	remote := pb.NewMaintenanceClient(conn)
	resp, err := remote.Defragment(ctx, &pb.DefragmentRequest{})
	if err != nil {
		return nil, err
	}
	return (*DefragmentResponse)(resp), nil
}

func (m *maintenance) Status(ctx context.Context, endpoint string) (*StatusResponse, error) {
	conn, err := m.c.Dial(endpoint)
	if err != nil {
		return nil, err
	}
	remote := pb.NewMaintenanceClient(conn)
	resp, err := remote.Status(ctx, &pb.StatusRequest{})
	if err != nil {
		return nil, err
	}
	return (*StatusResponse)(resp), nil
}

func (m *maintenance) getRemote() pb.MaintenanceClient {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.remote
}

func (m *maintenance) switchRemote(prevErr error) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	newConn, err := m.c.retryConnection(m.conn, prevErr)
	if err != nil {
		return err
	}
	m.conn = newConn
	m.remote = pb.NewMaintenanceClient(m.conn)
	return nil
}
