package consul

import (
	"fmt"
	"net"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/testutil"
	"github.com/hashicorp/net-rpc-msgpackrpc"
	"github.com/hashicorp/serf/serf"
)

func testClientConfig(t *testing.T, NodeName string) (string, *Config) {
	dir := tmpDir(t)
	config := DefaultConfig()
	config.Datacenter = "dc1"
	config.DataDir = dir
	config.NodeName = NodeName
	config.RPCAddr = &net.TCPAddr{
		IP:   []byte{127, 0, 0, 1},
		Port: getPort(),
	}
	config.SerfLANConfig.MemberlistConfig.BindAddr = "127.0.0.1"
	config.SerfLANConfig.MemberlistConfig.BindPort = getPort()
	config.SerfLANConfig.MemberlistConfig.ProbeTimeout = 200 * time.Millisecond
	config.SerfLANConfig.MemberlistConfig.ProbeInterval = time.Second
	config.SerfLANConfig.MemberlistConfig.GossipInterval = 100 * time.Millisecond

	return dir, config
}

func testClient(t *testing.T) (string, *Client) {
	return testClientDC(t, "dc1")
}

func testClientDC(t *testing.T, dc string) (string, *Client) {
	dir, config := testClientConfig(t, "testco.internal")
	config.Datacenter = dc

	client, err := NewClient(config)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	return dir, client
}

func testClientWithConfig(t *testing.T, cb func(c *Config)) (string, *Client) {
	name := fmt.Sprintf("Client %d", getPort())
	dir, config := testClientConfig(t, name)
	cb(config)
	client, err := NewClient(config)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	return dir, client
}

func TestClient_StartStop(t *testing.T) {
	dir, client := testClient(t)
	defer os.RemoveAll(dir)

	if err := client.Shutdown(); err != nil {
		t.Fatalf("err: %v", err)
	}
}

func TestClient_JoinLAN(t *testing.T) {
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

	// Check the members
	testutil.WaitForResult(func() (bool, error) {
		server_check := len(s1.LANMembers()) == 2
		client_check := len(c1.LANMembers()) == 2
		return server_check && client_check, nil
	}, func(err error) {
		t.Fatalf("bad len")
	})

	// Check we have a new consul
	testutil.WaitForResult(func() (bool, error) {
		return len(c1.consuls) == 1, nil
	}, func(err error) {
		t.Fatalf("expected consul server")
	})
}

func TestClient_JoinLAN_Invalid(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, c1 := testClientDC(t, "other")
	defer os.RemoveAll(dir2)
	defer c1.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := c1.JoinLAN([]string{addr}); err == nil {
		t.Fatalf("should error")
	}

	time.Sleep(50 * time.Millisecond)
	if len(s1.LANMembers()) != 1 {
		t.Fatalf("should not join")
	}
	if len(c1.LANMembers()) != 1 {
		t.Fatalf("should not join")
	}
}

func TestClient_JoinWAN_Invalid(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, c1 := testClientDC(t, "dc2")
	defer os.RemoveAll(dir2)
	defer c1.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfWANConfig.MemberlistConfig.BindPort)
	if _, err := c1.JoinLAN([]string{addr}); err == nil {
		t.Fatalf("should error")
	}

	time.Sleep(50 * time.Millisecond)
	if len(s1.WANMembers()) != 1 {
		t.Fatalf("should not join")
	}
	if len(c1.LANMembers()) != 1 {
		t.Fatalf("should not join")
	}
}

func TestClient_RPC(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, c1 := testClient(t)
	defer os.RemoveAll(dir2)
	defer c1.Shutdown()

	// Try an RPC
	var out struct{}
	if err := c1.RPC("Status.Ping", struct{}{}, &out); err != structs.ErrNoServers {
		t.Fatalf("err: %v", err)
	}

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := c1.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Check the members
	if len(s1.LANMembers()) != 2 {
		t.Fatalf("bad len")
	}

	if len(c1.LANMembers()) != 2 {
		t.Fatalf("bad len")
	}

	// RPC should succeed
	testutil.WaitForResult(func() (bool, error) {
		err := c1.RPC("Status.Ping", struct{}{}, &out)
		return err == nil, err
	}, func(err error) {
		t.Fatalf("err: %v", err)
	})
}

func TestClient_RPC_Pool(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, c1 := testClient(t)
	defer os.RemoveAll(dir2)
	defer c1.Shutdown()

	// Try to join.
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := c1.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(s1.LANMembers()) != 2 || len(c1.LANMembers()) != 2 {
		t.Fatalf("bad len")
	}

	// Blast out a bunch of RPC requests at the same time to try to get
	// contention opening new connections.
	var wg sync.WaitGroup
	for i := 0; i < 150; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()
			var out struct{}
			testutil.WaitForResult(func() (bool, error) {
				err := c1.RPC("Status.Ping", struct{}{}, &out)
				return err == nil, err
			}, func(err error) {
				t.Fatalf("err: %v", err)
			})
		}()
	}

	wg.Wait()
}

func TestClient_RPC_TLS(t *testing.T) {
	dir1, conf1 := testServerConfig(t, "a.testco.internal")
	conf1.VerifyIncoming = true
	conf1.VerifyOutgoing = true
	configureTLS(conf1)
	s1, err := NewServer(conf1)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, conf2 := testClientConfig(t, "b.testco.internal")
	conf2.VerifyOutgoing = true
	configureTLS(conf2)
	c1, err := NewClient(conf2)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer os.RemoveAll(dir2)
	defer c1.Shutdown()

	// Try an RPC
	var out struct{}
	if err := c1.RPC("Status.Ping", struct{}{}, &out); err != structs.ErrNoServers {
		t.Fatalf("err: %v", err)
	}

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := c1.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Check the members
	if len(s1.LANMembers()) != 2 {
		t.Fatalf("bad len")
	}

	if len(c1.LANMembers()) != 2 {
		t.Fatalf("bad len")
	}

	// RPC should succeed
	testutil.WaitForResult(func() (bool, error) {
		err := c1.RPC("Status.Ping", struct{}{}, &out)
		return err == nil, err
	}, func(err error) {
		t.Fatalf("err: %v", err)
	})
}

func TestClientServer_UserEvent(t *testing.T) {
	clientOut := make(chan serf.UserEvent, 2)
	dir1, c1 := testClientWithConfig(t, func(conf *Config) {
		conf.UserEventHandler = func(e serf.UserEvent) {
			clientOut <- e
		}
	})
	defer os.RemoveAll(dir1)
	defer c1.Shutdown()

	serverOut := make(chan serf.UserEvent, 2)
	dir2, s1 := testServerWithConfig(t, func(conf *Config) {
		conf.UserEventHandler = func(e serf.UserEvent) {
			serverOut <- e
		}
	})
	defer os.RemoveAll(dir2)
	defer s1.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := c1.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Wait for the leader
	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Check the members
	testutil.WaitForResult(func() (bool, error) {
		return len(c1.LANMembers()) == 2 && len(s1.LANMembers()) == 2, nil
	}, func(err error) {
		t.Fatalf("bad len")
	})

	// Fire the user event
	codec := rpcClient(t, s1)
	event := structs.EventFireRequest{
		Name:       "foo",
		Datacenter: "dc1",
		Payload:    []byte("baz"),
	}
	if err := msgpackrpc.CallWithCodec(codec, "Internal.EventFire", &event, nil); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Wait for all the events
	var clientReceived, serverReceived bool
	for i := 0; i < 2; i++ {
		select {
		case e := <-clientOut:
			switch e.Name {
			case "foo":
				clientReceived = true
			default:
				t.Fatalf("Bad: %#v", e)
			}

		case e := <-serverOut:
			switch e.Name {
			case "foo":
				serverReceived = true
			default:
				t.Fatalf("Bad: %#v", e)
			}

		case <-time.After(10 * time.Second):
			t.Fatalf("timeout")
		}
	}

	if !serverReceived || !clientReceived {
		t.Fatalf("missing events")
	}
}

func TestClient_Encrypted(t *testing.T) {
	dir1, c1 := testClient(t)
	defer os.RemoveAll(dir1)
	defer c1.Shutdown()

	key := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	dir2, c2 := testClientWithConfig(t, func(c *Config) {
		c.SerfLANConfig.MemberlistConfig.SecretKey = key
	})
	defer os.RemoveAll(dir2)
	defer c2.Shutdown()

	if c1.Encrypted() {
		t.Fatalf("should not be encrypted")
	}
	if !c2.Encrypted() {
		t.Fatalf("should be encrypted")
	}
}
