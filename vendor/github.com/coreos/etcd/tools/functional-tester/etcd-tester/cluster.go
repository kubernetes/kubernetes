// Copyright 2015 The etcd Authors
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

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/tools/functional-tester/etcd-agent/client"
	"google.golang.org/grpc"
)

// agentConfig holds information needed to interact/configure an agent and its etcd process
type agentConfig struct {
	endpoint      string
	clientPort    int
	peerPort      int
	failpointPort int

	datadir string
}

type cluster struct {
	agents  []agentConfig
	Size    int
	Members []*member
}

type ClusterStatus struct {
	AgentStatuses map[string]client.Status
}

func (c *cluster) bootstrap() error {
	size := len(c.agents)

	members := make([]*member, size)
	memberNameURLs := make([]string, size)
	for i, a := range c.agents {
		agent, err := client.NewAgent(a.endpoint)
		if err != nil {
			return err
		}
		host, _, err := net.SplitHostPort(a.endpoint)
		if err != nil {
			return err
		}
		members[i] = &member{
			Agent:        agent,
			Endpoint:     a.endpoint,
			Name:         fmt.Sprintf("etcd-%d", i),
			ClientURL:    fmt.Sprintf("http://%s:%d", host, a.clientPort),
			PeerURL:      fmt.Sprintf("http://%s:%d", host, a.peerPort),
			FailpointURL: fmt.Sprintf("http://%s:%d", host, a.failpointPort),
		}
		memberNameURLs[i] = members[i].ClusterEntry()
	}
	clusterStr := strings.Join(memberNameURLs, ",")
	token := fmt.Sprint(rand.Int())

	for i, m := range members {
		flags := append(
			m.Flags(),
			"--data-dir", c.agents[i].datadir,
			"--initial-cluster-token", token,
			"--initial-cluster", clusterStr)

		if _, err := m.Agent.Start(flags...); err != nil {
			// cleanup
			for _, m := range members[:i] {
				m.Agent.Terminate()
			}
			return err
		}
	}

	c.Size = size
	c.Members = members
	return nil
}

func (c *cluster) Reset() error { return c.bootstrap() }

func (c *cluster) WaitHealth() error {
	var err error
	// wait 60s to check cluster health.
	// TODO: set it to a reasonable value. It is set that high because
	// follower may use long time to catch up the leader when reboot under
	// reasonable workload (https://github.com/coreos/etcd/issues/2698)
	for i := 0; i < 60; i++ {
		for _, m := range c.Members {
			if err = m.SetHealthKeyV3(); err != nil {
				break
			}
		}
		if err == nil {
			return nil
		}
		plog.Warningf("#%d setHealthKey error (%v)", i, err)
		time.Sleep(time.Second)
	}
	return err
}

// GetLeader returns the index of leader and error if any.
func (c *cluster) GetLeader() (int, error) {
	for i, m := range c.Members {
		isLeader, err := m.IsLeader()
		if isLeader || err != nil {
			return i, err
		}
	}
	return 0, fmt.Errorf("no leader found")
}

func (c *cluster) Cleanup() error {
	var lasterr error
	for _, m := range c.Members {
		if err := m.Agent.Cleanup(); err != nil {
			lasterr = err
		}
	}
	return lasterr
}

func (c *cluster) Terminate() {
	for _, m := range c.Members {
		m.Agent.Terminate()
	}
}

func (c *cluster) Status() ClusterStatus {
	cs := ClusterStatus{
		AgentStatuses: make(map[string]client.Status),
	}

	for _, m := range c.Members {
		s, err := m.Agent.Status()
		// TODO: add a.Desc() as a key of the map
		desc := m.Endpoint
		if err != nil {
			cs.AgentStatuses[desc] = client.Status{State: "unknown"}
			plog.Printf("failed to get the status of agent [%s]", desc)
		}
		cs.AgentStatuses[desc] = s
	}
	return cs
}

// maxRev returns the maximum revision found on the cluster.
func (c *cluster) maxRev() (rev int64, err error) {
	ctx, cancel := context.WithTimeout(context.TODO(), time.Second)
	defer cancel()
	revc, errc := make(chan int64, len(c.Members)), make(chan error, len(c.Members))
	for i := range c.Members {
		go func(m *member) {
			mrev, merr := m.Rev(ctx)
			revc <- mrev
			errc <- merr
		}(c.Members[i])
	}
	for i := 0; i < len(c.Members); i++ {
		if merr := <-errc; merr != nil {
			err = merr
		}
		if mrev := <-revc; mrev > rev {
			rev = mrev
		}
	}
	return rev, err
}

func (c *cluster) getRevisionHash() (map[string]int64, map[string]int64, error) {
	revs := make(map[string]int64)
	hashes := make(map[string]int64)
	for _, m := range c.Members {
		rev, hash, err := m.RevHash()
		if err != nil {
			return nil, nil, err
		}
		revs[m.ClientURL] = rev
		hashes[m.ClientURL] = hash
	}
	return revs, hashes, nil
}

func (c *cluster) compactKV(rev int64, timeout time.Duration) (err error) {
	if rev <= 0 {
		return nil
	}

	for i, m := range c.Members {
		u := m.ClientURL
		conn, derr := m.dialGRPC()
		if derr != nil {
			plog.Printf("[compact kv #%d] dial error %v (endpoint %s)", i, derr, u)
			err = derr
			continue
		}
		kvc := pb.NewKVClient(conn)
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		plog.Printf("[compact kv #%d] starting (endpoint %s)", i, u)
		_, cerr := kvc.Compact(ctx, &pb.CompactionRequest{Revision: rev, Physical: true}, grpc.FailFast(false))
		cancel()
		conn.Close()
		succeed := true
		if cerr != nil {
			if strings.Contains(cerr.Error(), "required revision has been compacted") && i > 0 {
				plog.Printf("[compact kv #%d] already compacted (endpoint %s)", i, u)
			} else {
				plog.Warningf("[compact kv #%d] error %v (endpoint %s)", i, cerr, u)
				err = cerr
				succeed = false
			}
		}
		if succeed {
			plog.Printf("[compact kv #%d] done (endpoint %s)", i, u)
		}
	}
	return err
}

func (c *cluster) checkCompact(rev int64) error {
	if rev == 0 {
		return nil
	}
	for _, m := range c.Members {
		if err := m.CheckCompact(rev); err != nil {
			return err
		}
	}
	return nil
}

func (c *cluster) defrag() error {
	for _, m := range c.Members {
		if err := m.Defrag(); err != nil {
			return err
		}
	}
	return nil
}
