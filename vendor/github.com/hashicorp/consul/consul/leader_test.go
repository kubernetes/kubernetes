package consul

import (
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/testutil"
	"github.com/hashicorp/net-rpc-msgpackrpc"
	"github.com/hashicorp/serf/serf"
)

func TestLeader_RegisterMember(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, c1 := testClient(t)
	defer os.RemoveAll(dir2)
	defer c1.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := c1.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Client should be registered
	state := s1.fsm.State()
	testutil.WaitForResult(func() (bool, error) {
		_, node, err := state.GetNode(c1.config.NodeName)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		return node != nil, nil
	}, func(err error) {
		t.Fatalf("client not registered")
	})

	// Should have a check
	_, checks, err := state.NodeChecks(c1.config.NodeName)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(checks) != 1 {
		t.Fatalf("client missing check")
	}
	if checks[0].CheckID != SerfCheckID {
		t.Fatalf("bad check: %v", checks[0])
	}
	if checks[0].Name != SerfCheckName {
		t.Fatalf("bad check: %v", checks[0])
	}
	if checks[0].Status != structs.HealthPassing {
		t.Fatalf("bad check: %v", checks[0])
	}

	// Server should be registered
	_, node, err := state.GetNode(s1.config.NodeName)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if node == nil {
		t.Fatalf("server not registered")
	}

	// Service should be registered
	_, services, err := state.NodeServices(s1.config.NodeName)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if _, ok := services.Services["consul"]; !ok {
		t.Fatalf("consul service not registered: %v", services)
	}
}

func TestLeader_FailedMember(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, c1 := testClient(t)
	defer os.RemoveAll(dir2)
	defer c1.Shutdown()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := c1.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Fail the member
	c1.Shutdown()

	// Should be registered
	state := s1.fsm.State()
	testutil.WaitForResult(func() (bool, error) {
		_, node, err := state.GetNode(c1.config.NodeName)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		return node != nil, nil
	}, func(err error) {
		t.Fatalf("client not registered")
	})

	// Should have a check
	_, checks, err := state.NodeChecks(c1.config.NodeName)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(checks) != 1 {
		t.Fatalf("client missing check")
	}
	if checks[0].CheckID != SerfCheckID {
		t.Fatalf("bad check: %v", checks[0])
	}
	if checks[0].Name != SerfCheckName {
		t.Fatalf("bad check: %v", checks[0])
	}

	testutil.WaitForResult(func() (bool, error) {
		_, checks, err = state.NodeChecks(c1.config.NodeName)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		return checks[0].Status == structs.HealthCritical, errors.New(checks[0].Status)
	}, func(err error) {
		t.Fatalf("check status is %v, should be critical", err)
	})
}

func TestLeader_LeftMember(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, c1 := testClient(t)
	defer os.RemoveAll(dir2)
	defer c1.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := c1.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	state := s1.fsm.State()

	// Should be registered
	testutil.WaitForResult(func() (bool, error) {
		_, node, err := state.GetNode(c1.config.NodeName)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		return node != nil, nil
	}, func(err error) {
		t.Fatalf("client should be registered")
	})

	// Node should leave
	c1.Leave()
	c1.Shutdown()

	// Should be deregistered
	testutil.WaitForResult(func() (bool, error) {
		_, node, err := state.GetNode(c1.config.NodeName)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		return node == nil, nil
	}, func(err error) {
		t.Fatalf("client should not be registered")
	})
}

func TestLeader_ReapMember(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, c1 := testClient(t)
	defer os.RemoveAll(dir2)
	defer c1.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := c1.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	state := s1.fsm.State()

	// Should be registered
	testutil.WaitForResult(func() (bool, error) {
		_, node, err := state.GetNode(c1.config.NodeName)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		return node != nil, nil
	}, func(err error) {
		t.Fatalf("client should be registered")
	})

	// Simulate a node reaping
	mems := s1.LANMembers()
	var c1mem serf.Member
	for _, m := range mems {
		if m.Name == c1.config.NodeName {
			c1mem = m
			c1mem.Status = StatusReap
			break
		}
	}
	s1.reconcileCh <- c1mem

	// Should be deregistered
	testutil.WaitForResult(func() (bool, error) {
		_, node, err := state.GetNode(c1.config.NodeName)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		return node == nil, nil
	}, func(err error) {
		t.Fatalf("client should not be registered")
	})
}

func TestLeader_Reconcile_ReapMember(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Register a non-existing member
	dead := structs.RegisterRequest{
		Datacenter: s1.config.Datacenter,
		Node:       "no-longer-around",
		Address:    "127.1.1.1",
		Check: &structs.HealthCheck{
			Node:    "no-longer-around",
			CheckID: SerfCheckID,
			Name:    SerfCheckName,
			Status:  structs.HealthCritical,
		},
	}
	var out struct{}
	if err := s1.RPC("Catalog.Register", &dead, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Force a reconciliation
	if err := s1.reconcile(); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Node should be gone
	state := s1.fsm.State()
	_, node, err := state.GetNode("no-longer-around")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if node != nil {
		t.Fatalf("client registered")
	}
}

func TestLeader_Reconcile(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, c1 := testClient(t)
	defer os.RemoveAll(dir2)
	defer c1.Shutdown()

	// Join before we have a leader, this should cause a reconcile!
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := c1.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should not be registered
	state := s1.fsm.State()
	_, node, err := state.GetNode(c1.config.NodeName)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if node != nil {
		t.Fatalf("client registered")
	}

	// Should be registered
	testutil.WaitForResult(func() (bool, error) {
		_, node, err = state.GetNode(c1.config.NodeName)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		return node != nil, nil
	}, func(err error) {
		t.Fatalf("client should be registered")
	})
}

func TestLeader_LeftServer(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, s2 := testServerDCBootstrap(t, "dc1", false)
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()

	dir3, s3 := testServerDCBootstrap(t, "dc1", false)
	defer os.RemoveAll(dir3)
	defer s3.Shutdown()
	servers := []*Server{s1, s2, s3}

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if _, err := s3.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	for _, s := range servers {
		testutil.WaitForResult(func() (bool, error) {
			peers, _ := s.raftPeers.Peers()
			return len(peers) == 3, nil
		}, func(err error) {
			t.Fatalf("should have 3 peers")
		})
	}

	testutil.WaitForResult(func() (bool, error) {
		// Kill any server
		servers[0].Shutdown()

		// Force remove the non-leader (transition to left state)
		if err := servers[1].RemoveFailedNode(servers[0].config.NodeName); err != nil {
			t.Fatalf("err: %v", err)
		}

		for _, s := range servers[1:] {
			peers, _ := s.raftPeers.Peers()
			return len(peers) == 2, errors.New(fmt.Sprintf("%v", peers))
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}

func TestLeader_LeftLeader(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, s2 := testServerDCBootstrap(t, "dc1", false)
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()

	dir3, s3 := testServerDCBootstrap(t, "dc1", false)
	defer os.RemoveAll(dir3)
	defer s3.Shutdown()
	servers := []*Server{s1, s2, s3}

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if _, err := s3.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	for _, s := range servers {
		testutil.WaitForResult(func() (bool, error) {
			peers, _ := s.raftPeers.Peers()
			return len(peers) == 3, nil
		}, func(err error) {
			t.Fatalf("should have 3 peers")
		})
	}

	// Kill the leader!
	var leader *Server
	for _, s := range servers {
		if s.IsLeader() {
			leader = s
			break
		}
	}
	if leader == nil {
		t.Fatalf("Should have a leader")
	}
	leader.Leave()
	leader.Shutdown()
	time.Sleep(100 * time.Millisecond)

	var remain *Server
	for _, s := range servers {
		if s == leader {
			continue
		}
		remain = s
		testutil.WaitForResult(func() (bool, error) {
			peers, _ := s.raftPeers.Peers()
			return len(peers) == 2, errors.New(fmt.Sprintf("%v", peers))
		}, func(err error) {
			t.Fatalf("should have 2 peers: %v", err)
		})
	}

	// Verify the old leader is deregistered
	state := remain.fsm.State()
	testutil.WaitForResult(func() (bool, error) {
		_, node, err := state.GetNode(leader.config.NodeName)
		if err != nil {
			t.Fatalf("err: %v", err)
		}
		return node == nil, nil
	}, func(err error) {
		t.Fatalf("leader should be deregistered")
	})
}

func TestLeader_MultiBootstrap(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, s2 := testServer(t)
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()

	servers := []*Server{s1, s2}

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	for _, s := range servers {
		testutil.WaitForResult(func() (bool, error) {
			peers := s.serfLAN.Members()
			return len(peers) == 2, nil
		}, func(err error) {
			t.Fatalf("should have 2 peers")
		})
	}

	// Ensure we don't have multiple raft peers
	for _, s := range servers {
		peers, _ := s.raftPeers.Peers()
		if len(peers) != 1 {
			t.Fatalf("should only have 1 raft peer!")
		}
	}
}

func TestLeader_TombstoneGC_Reset(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, s2 := testServerDCBootstrap(t, "dc1", false)
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()

	dir3, s3 := testServerDCBootstrap(t, "dc1", false)
	defer os.RemoveAll(dir3)
	defer s3.Shutdown()
	servers := []*Server{s1, s2, s3}

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if _, err := s3.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	for _, s := range servers {
		testutil.WaitForResult(func() (bool, error) {
			peers, _ := s.raftPeers.Peers()
			return len(peers) == 3, nil
		}, func(err error) {
			t.Fatalf("should have 3 peers")
		})
	}

	var leader *Server
	for _, s := range servers {
		if s.IsLeader() {
			leader = s
			break
		}
	}
	if leader == nil {
		t.Fatalf("Should have a leader")
	}

	// Check that the leader has a pending GC expiration
	if !leader.tombstoneGC.PendingExpiration() {
		t.Fatalf("should have pending expiration")
	}

	// Kill the leader
	leader.Shutdown()
	time.Sleep(100 * time.Millisecond)

	// Wait for a new leader
	leader = nil
	testutil.WaitForResult(func() (bool, error) {
		for _, s := range servers {
			if s.IsLeader() {
				leader = s
				return true, nil
			}
		}
		return false, nil
	}, func(err error) {
		t.Fatalf("should have leader")
	})

	// Check that the new leader has a pending GC expiration
	testutil.WaitForResult(func() (bool, error) {
		return leader.tombstoneGC.PendingExpiration(), nil
	}, func(err error) {
		t.Fatalf("should have pending expiration")
	})
}

func TestLeader_ReapTombstones(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.TombstoneTTL = 50 * time.Millisecond
		c.TombstoneTTLGranularity = 10 * time.Millisecond
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Create a KV entry
	arg := structs.KVSRequest{
		Datacenter: "dc1",
		Op:         structs.KVSSet,
		DirEnt: structs.DirEntry{
			Key:   "test",
			Value: []byte("test"),
		},
	}
	var out bool
	if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Delete the KV entry (tombstoned).
	arg.Op = structs.KVSDelete
	if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Make sure there's a tombstone.
	state := s1.fsm.State()
	func() {
		snap := state.Snapshot()
		defer snap.Close()
		stones, err := snap.Tombstones()
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if stones.Next() == nil {
			t.Fatalf("missing tombstones")
		}
		if stones.Next() != nil {
			t.Fatalf("unexpected extra tombstones")
		}
	}()

	// Check that the new leader has a pending GC expiration by
	// watching for the tombstone to get removed.
	testutil.WaitForResult(func() (bool, error) {
		snap := state.Snapshot()
		defer snap.Close()
		stones, err := snap.Tombstones()
		if err != nil {
			return false, err
		}
		return stones.Next() == nil, nil
	}, func(err error) {
		t.Fatalf("err: %v", err)
	})
}
