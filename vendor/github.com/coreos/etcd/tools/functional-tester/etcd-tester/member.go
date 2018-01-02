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

package main

import (
	"fmt"
	"net"
	"net/url"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc"

	"github.com/coreos/etcd/clientv3"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/tools/functional-tester/etcd-agent/client"
)

type member struct {
	Agent        client.Agent
	Endpoint     string
	Name         string
	ClientURL    string
	PeerURL      string
	FailpointURL string
}

func (m *member) ClusterEntry() string { return m.Name + "=" + m.PeerURL }

func (m *member) Flags() []string {
	return []string{
		"--name", m.Name,
		"--listen-client-urls", m.ClientURL,
		"--advertise-client-urls", m.ClientURL,
		"--listen-peer-urls", m.PeerURL,
		"--initial-advertise-peer-urls", m.PeerURL,
		"--initial-cluster-state", "new",
	}
}

func (m *member) CheckCompact(rev int64) error {
	cli, err := m.newClientV3()
	if err != nil {
		return fmt.Errorf("%v (endpoint %s)", err, m.ClientURL)
	}
	defer cli.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	wch := cli.Watch(ctx, "\x00", clientv3.WithFromKey(), clientv3.WithRev(rev-1))
	wr, ok := <-wch
	cancel()

	if !ok {
		return fmt.Errorf("watch channel terminated (endpoint %s)", m.ClientURL)
	}
	if wr.CompactRevision != rev {
		return fmt.Errorf("got compact revision %v, wanted %v (endpoint %s)", wr.CompactRevision, rev, m.ClientURL)
	}

	return nil
}

func (m *member) Defrag() error {
	plog.Printf("defragmenting %s\n", m.ClientURL)
	cli, err := m.newClientV3()
	if err != nil {
		return err
	}
	defer cli.Close()
	if _, err = cli.Defragment(context.Background(), m.ClientURL); err != nil {
		return err
	}
	plog.Printf("defragmented %s\n", m.ClientURL)
	return nil
}

func (m *member) RevHash() (int64, int64, error) {
	conn, err := m.dialGRPC()
	if err != nil {
		return 0, 0, err
	}
	mt := pb.NewMaintenanceClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	resp, err := mt.Hash(ctx, &pb.HashRequest{}, grpc.FailFast(false))
	cancel()
	conn.Close()

	if err != nil {
		return 0, 0, err
	}

	return resp.Header.Revision, int64(resp.Hash), nil
}

func (m *member) Rev(ctx context.Context) (int64, error) {
	cli, err := m.newClientV3()
	if err != nil {
		return 0, err
	}
	defer cli.Close()
	resp, err := cli.Status(ctx, m.ClientURL)
	if err != nil {
		return 0, err
	}
	return resp.Header.Revision, nil
}

func (m *member) IsLeader() (bool, error) {
	cli, err := m.newClientV3()
	if err != nil {
		return false, err
	}
	defer cli.Close()
	resp, err := cli.Status(context.Background(), m.ClientURL)
	if err != nil {
		return false, err
	}
	return resp.Header.MemberId == resp.Leader, nil
}

func (m *member) SetHealthKeyV3() error {
	cli, err := m.newClientV3()
	if err != nil {
		return fmt.Errorf("%v (%s)", err, m.ClientURL)
	}
	defer cli.Close()
	// give enough time-out in case expensive requests (range/delete) are pending
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	_, err = cli.Put(ctx, "health", "good")
	cancel()
	if err != nil {
		return fmt.Errorf("%v (%s)", err, m.ClientURL)
	}
	return nil
}

func (m *member) newClientV3() (*clientv3.Client, error) {
	return clientv3.New(clientv3.Config{
		Endpoints:   []string{m.ClientURL},
		DialTimeout: 5 * time.Second,
	})
}

func (m *member) dialGRPC() (*grpc.ClientConn, error) {
	return grpc.Dial(m.grpcAddr(), grpc.WithInsecure(), grpc.WithTimeout(5*time.Second), grpc.WithBlock())
}

// grpcAddr gets the host from clientURL so it works with grpc.Dial()
func (m *member) grpcAddr() string {
	u, err := url.Parse(m.ClientURL)
	if err != nil {
		panic(err)
	}
	return u.Host
}

func (m *member) peerPort() (port int) {
	u, err := url.Parse(m.PeerURL)
	if err != nil {
		panic(err)
	}
	_, portStr, err := net.SplitHostPort(u.Host)
	if err != nil {
		panic(err)
	}
	if _, err = fmt.Sscanf(portStr, "%d", &port); err != nil {
		panic(err)
	}
	return port
}
