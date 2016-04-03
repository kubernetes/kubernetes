// Copyright 2015 CoreOS, Inc.
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
	"math/rand"
	"net"
	"strings"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc"

	clientv2 "github.com/coreos/etcd/client"
	"github.com/coreos/etcd/clientv3"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/tools/functional-tester/etcd-agent/client"
)

const peerURLPort = 2380

type cluster struct {
	v2Only bool // to be deprecated

	agentEndpoints       []string
	datadir              string
	stressKeySize        int
	stressKeySuffixRange int

	Size       int
	Agents     []client.Agent
	Stressers  []Stresser
	Names      []string
	GRPCURLs   []string
	ClientURLs []string
}

type ClusterStatus struct {
	AgentStatuses map[string]client.Status
}

// newCluster starts and returns a new cluster. The caller should call Terminate when finished, to shut it down.
func newCluster(agentEndpoints []string, datadir string, stressKeySize, stressKeySuffixRange int, isV2Only bool) (*cluster, error) {
	c := &cluster{
		v2Only:               isV2Only,
		agentEndpoints:       agentEndpoints,
		datadir:              datadir,
		stressKeySize:        stressKeySize,
		stressKeySuffixRange: stressKeySuffixRange,
	}
	if err := c.Bootstrap(); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *cluster) Bootstrap() error {
	size := len(c.agentEndpoints)

	agents := make([]client.Agent, size)
	names := make([]string, size)
	grpcURLs := make([]string, size)
	clientURLs := make([]string, size)
	peerURLs := make([]string, size)
	members := make([]string, size)
	for i, u := range c.agentEndpoints {
		var err error
		agents[i], err = client.NewAgent(u)
		if err != nil {
			return err
		}

		names[i] = fmt.Sprintf("etcd-%d", i)

		host, _, err := net.SplitHostPort(u)
		if err != nil {
			return err
		}
		grpcURLs[i] = fmt.Sprintf("%s:2378", host)
		clientURLs[i] = fmt.Sprintf("http://%s:2379", host)
		peerURLs[i] = fmt.Sprintf("http://%s:%d", host, peerURLPort)

		members[i] = fmt.Sprintf("%s=%s", names[i], peerURLs[i])
	}
	clusterStr := strings.Join(members, ",")
	token := fmt.Sprint(rand.Int())

	for i, a := range agents {
		flags := []string{
			"--name", names[i],
			"--data-dir", c.datadir,

			"--listen-client-urls", clientURLs[i],
			"--advertise-client-urls", clientURLs[i],

			"--listen-peer-urls", peerURLs[i],
			"--initial-advertise-peer-urls", peerURLs[i],

			"--initial-cluster-token", token,
			"--initial-cluster", clusterStr,
			"--initial-cluster-state", "new",
		}
		if !c.v2Only {
			flags = append(flags,
				"--experimental-v3demo",
				"--experimental-gRPC-addr", grpcURLs[i],
			)
		}

		if _, err := a.Start(flags...); err != nil {
			// cleanup
			for j := 0; j < i; j++ {
				agents[j].Terminate()
			}
			return err
		}
	}

	// TODO: Too intensive stressers can panic etcd member with
	// 'out of memory' error. Put rate limits in server side.
	stressN := 100
	var stressers []Stresser
	if c.v2Only {
		for _, u := range clientURLs {
			s := &stresserV2{
				Endpoint:       u,
				KeySize:        c.stressKeySize,
				KeySuffixRange: c.stressKeySuffixRange,
				N:              stressN,
			}
			go s.Stress()
			stressers = append(stressers, s)
		}
	} else {
		for _, u := range grpcURLs {
			s := &stresser{
				Endpoint:       u,
				KeySize:        c.stressKeySize,
				KeySuffixRange: c.stressKeySuffixRange,
				N:              stressN,
			}
			go s.Stress()
			stressers = append(stressers, s)
		}
	}

	c.Size = size
	c.Agents = agents
	c.Stressers = stressers
	c.Names = names
	c.GRPCURLs = grpcURLs
	c.ClientURLs = clientURLs
	return nil
}

func (c *cluster) WaitHealth() error {
	var err error
	// wait 60s to check cluster health.
	// TODO: set it to a reasonable value. It is set that high because
	// follower may use long time to catch up the leader when reboot under
	// reasonable workload (https://github.com/coreos/etcd/issues/2698)
	healthFunc, urls := setHealthKey, c.GRPCURLs
	if c.v2Only {
		healthFunc, urls = setHealthKeyV2, c.ClientURLs
	}
	for i := 0; i < 60; i++ {
		err = healthFunc(urls)
		if err == nil {
			return nil
		}
		time.Sleep(time.Second)
	}
	return err
}

// GetLeader returns the index of leader and error if any.
func (c *cluster) GetLeader() (int, error) {
	if c.v2Only {
		return 0, nil
	}
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   c.GRPCURLs,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		return 0, err
	}
	defer cli.Close()
	clus := clientv3.NewCluster(cli)
	mem, err := clus.MemberLeader(context.Background())
	if err != nil {
		return 0, err
	}
	for i, name := range c.Names {
		if name == mem.Name {
			return i, nil
		}
	}
	return 0, fmt.Errorf("no leader found")
}

func (c *cluster) Report() (success, failure int) {
	for _, stress := range c.Stressers {
		s, f := stress.Report()
		success += s
		failure += f
	}
	return
}

func (c *cluster) Cleanup() error {
	var lasterr error
	for _, a := range c.Agents {
		if err := a.Cleanup(); err != nil {
			lasterr = err
		}
	}
	for _, s := range c.Stressers {
		s.Cancel()
	}
	return lasterr
}

func (c *cluster) Terminate() {
	for _, a := range c.Agents {
		a.Terminate()
	}
	for _, s := range c.Stressers {
		s.Cancel()
	}
}

func (c *cluster) Status() ClusterStatus {
	cs := ClusterStatus{
		AgentStatuses: make(map[string]client.Status),
	}

	for i, a := range c.Agents {
		s, err := a.Status()
		// TODO: add a.Desc() as a key of the map
		desc := c.agentEndpoints[i]
		if err != nil {
			cs.AgentStatuses[desc] = client.Status{State: "unknown"}
			plog.Printf("failed to get the status of agent [%s]", desc)
		}
		cs.AgentStatuses[desc] = s
	}
	return cs
}

// setHealthKey sets health key on all given urls.
func setHealthKey(us []string) error {
	for _, u := range us {
		conn, err := grpc.Dial(u, grpc.WithInsecure(), grpc.WithTimeout(5*time.Second))
		if err != nil {
			return fmt.Errorf("%v (%s)", err, u)
		}
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		kvc := pb.NewKVClient(conn)
		_, err = kvc.Put(ctx, &pb.PutRequest{Key: []byte("health"), Value: []byte("good")})
		cancel()
		conn.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

// setHealthKeyV2 sets health key on all given urls.
func setHealthKeyV2(us []string) error {
	for _, u := range us {
		cfg := clientv2.Config{
			Endpoints: []string{u},
		}
		c, err := clientv2.New(cfg)
		if err != nil {
			return err
		}
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		kapi := clientv2.NewKeysAPI(c)
		_, err = kapi.Set(ctx, "health", "good", nil)
		cancel()
		if err != nil {
			return err
		}
	}
	return nil
}

func (c *cluster) getRevisionHash() (map[string]int64, map[string]int64, error) {
	revs := make(map[string]int64)
	hashes := make(map[string]int64)
	for _, u := range c.GRPCURLs {
		conn, err := grpc.Dial(u, grpc.WithInsecure(), grpc.WithTimeout(5*time.Second))
		if err != nil {
			return nil, nil, err
		}
		kvc := pb.NewKVClient(conn)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		resp, err := kvc.Hash(ctx, &pb.HashRequest{})
		cancel()
		conn.Close()
		if err != nil {
			return nil, nil, err
		}
		revs[u] = resp.Header.Revision
		hashes[u] = int64(resp.Hash)
	}
	return revs, hashes, nil
}

func (c *cluster) compactKV(rev int64) error {
	var (
		conn *grpc.ClientConn
		err  error
	)
	for _, u := range c.GRPCURLs {
		conn, err = grpc.Dial(u, grpc.WithInsecure(), grpc.WithTimeout(5*time.Second))
		if err != nil {
			continue
		}
		kvc := pb.NewKVClient(conn)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_, err = kvc.Compact(ctx, &pb.CompactionRequest{Revision: rev})
		cancel()
		conn.Close()
		if err == nil {
			return nil
		}
	}
	return err
}
