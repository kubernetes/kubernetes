package agent

import (
	"errors"
	"fmt"
	"github.com/hashicorp/consul/testutil"
	"github.com/hashicorp/serf/serf"
	"io"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

type rpcParts struct {
	dir    string
	client *RPCClient
	agent  *Agent
	rpc    *AgentRPC
}

func (r *rpcParts) Close() {
	r.client.Close()
	r.rpc.Shutdown()
	r.agent.Shutdown()
	os.RemoveAll(r.dir)
}

// testRPCClient returns an RPCClient connected to an RPC server that
// serves only this connection.
func testRPCClient(t *testing.T) *rpcParts {
	return testRPCClientWithConfig(t, func(c *Config) {})
}

func testRPCClientWithConfig(t *testing.T, cb func(c *Config)) *rpcParts {
	lw := NewLogWriter(512)
	mult := io.MultiWriter(os.Stderr, lw)

	conf := nextConfig()
	cb(conf)

	rpcAddr, err := conf.ClientListener(conf.Addresses.RPC, conf.Ports.RPC)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	l, err := net.Listen(rpcAddr.Network(), rpcAddr.String())
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	dir, agent := makeAgentLog(t, conf, mult)
	rpc := NewAgentRPC(agent, l, mult, lw)

	rpcClient, err := NewRPCClient(l.Addr().String())
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	return &rpcParts{
		dir:    dir,
		client: rpcClient,
		agent:  agent,
		rpc:    rpc,
	}
}

func TestRPCClient_UnixSocket(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.SkipNow()
	}

	tempDir, err := ioutil.TempDir("", "consul")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	defer os.RemoveAll(tempDir)
	socket := filepath.Join(tempDir, "test.sock")

	p1 := testRPCClientWithConfig(t, func(c *Config) {
		c.Addresses.RPC = "unix://" + socket
	})
	defer p1.Close()

	// Ensure the socket was created
	if _, err := os.Stat(socket); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Ensure we can talk with the socket
	mem, err := p1.client.LANMembers()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if len(mem) != 1 {
		t.Fatalf("bad: %#v", mem)
	}
}

func TestRPCClientForceLeave(t *testing.T) {
	p1 := testRPCClient(t)
	p2 := testRPCClient(t)
	defer p1.Close()
	defer p2.Close()

	s2Addr := fmt.Sprintf("127.0.0.1:%d", p2.agent.config.Ports.SerfLan)
	if _, err := p1.agent.JoinLAN([]string{s2Addr}); err != nil {
		t.Fatalf("err: %s", err)
	}

	if err := p2.agent.Shutdown(); err != nil {
		t.Fatalf("err: %s", err)
	}

	if err := p1.client.ForceLeave(p2.agent.config.NodeName); err != nil {
		t.Fatalf("err: %s", err)
	}

	m := p1.agent.LANMembers()
	if len(m) != 2 {
		t.Fatalf("should have 2 members: %#v", m)
	}

	testutil.WaitForResult(func() (bool, error) {
		m := p1.agent.LANMembers()
		success := m[1].Status == serf.StatusLeft
		return success, errors.New(m[1].Status.String())
	}, func(err error) {
		t.Fatalf("member status is %v, should be left", err)
	})
}

func TestRPCClientJoinLAN(t *testing.T) {
	p1 := testRPCClient(t)
	p2 := testRPCClient(t)
	defer p1.Close()
	defer p2.Close()

	s2Addr := fmt.Sprintf("127.0.0.1:%d", p2.agent.config.Ports.SerfLan)
	n, err := p1.client.Join([]string{s2Addr}, false)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if n != 1 {
		t.Fatalf("n != 1: %d", n)
	}
}

func TestRPCClientJoinWAN(t *testing.T) {
	p1 := testRPCClient(t)
	p2 := testRPCClient(t)
	defer p1.Close()
	defer p2.Close()

	s2Addr := fmt.Sprintf("127.0.0.1:%d", p2.agent.config.Ports.SerfWan)
	n, err := p1.client.Join([]string{s2Addr}, true)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if n != 1 {
		t.Fatalf("n != 1: %d", n)
	}
}

func TestRPCClientLANMembers(t *testing.T) {
	p1 := testRPCClient(t)
	p2 := testRPCClient(t)
	defer p1.Close()
	defer p2.Close()

	mem, err := p1.client.LANMembers()
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if len(mem) != 1 {
		t.Fatalf("bad: %#v", mem)
	}

	s2Addr := fmt.Sprintf("127.0.0.1:%d", p2.agent.config.Ports.SerfLan)
	_, err = p1.client.Join([]string{s2Addr}, false)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	mem, err = p1.client.LANMembers()
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if len(mem) != 2 {
		t.Fatalf("bad: %#v", mem)
	}
}

func TestRPCClientWANMembers(t *testing.T) {
	p1 := testRPCClient(t)
	p2 := testRPCClient(t)
	defer p1.Close()
	defer p2.Close()

	mem, err := p1.client.WANMembers()
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if len(mem) != 1 {
		t.Fatalf("bad: %#v", mem)
	}

	s2Addr := fmt.Sprintf("127.0.0.1:%d", p2.agent.config.Ports.SerfWan)
	_, err = p1.client.Join([]string{s2Addr}, true)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	mem, err = p1.client.WANMembers()
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if len(mem) != 2 {
		t.Fatalf("bad: %#v", mem)
	}
}

func TestRPCClientStats(t *testing.T) {
	p1 := testRPCClient(t)
	defer p1.Close()

	stats, err := p1.client.Stats()
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if _, ok := stats["agent"]; !ok {
		t.Fatalf("bad: %#v", stats)
	}

	if _, ok := stats["consul"]; !ok {
		t.Fatalf("bad: %#v", stats)
	}
}

func TestRPCClientLeave(t *testing.T) {
	p1 := testRPCClient(t)
	defer p1.Close()

	if err := p1.client.Leave(); err != nil {
		t.Fatalf("err: %s", err)
	}

	time.Sleep(1 * time.Second)

	select {
	case <-p1.agent.ShutdownCh():
	default:
		t.Fatalf("agent should be shutdown!")
	}
}

func TestRPCClientMonitor(t *testing.T) {
	p1 := testRPCClient(t)
	defer p1.Close()

	eventCh := make(chan string, 64)
	if handle, err := p1.client.Monitor("debug", eventCh); err != nil {
		t.Fatalf("err: %s", err)
	} else {
		defer p1.client.Stop(handle)
	}

	found := false
OUTER1:
	for i := 0; ; i++ {
		select {
		case e := <-eventCh:
			if strings.Contains(e, "Accepted client") {
				found = true
				break OUTER1
			}
		default:
			if i > 100 {
				break OUTER1
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
	if !found {
		t.Fatalf("should log client accept")
	}

	// Join a bad thing to generate more events
	p1.agent.JoinLAN(nil)

	found = false
OUTER2:
	for i := 0; ; i++ {
		select {
		case e := <-eventCh:
			if strings.Contains(e, "joining") {
				found = true
				break OUTER2
			}
		default:
			if i > 100 {
				break OUTER2
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
	if !found {
		t.Fatalf("should log joining")
	}
}

func TestRPCClientListKeys(t *testing.T) {
	key1 := "tbLJg26ZJyJ9pK3qhc9jig=="
	p1 := testRPCClientWithConfig(t, func(c *Config) {
		c.EncryptKey = key1
		c.Datacenter = "dc1"
		c.ACLDatacenter = ""
	})
	defer p1.Close()

	// Key is initially installed to both wan/lan
	keys := listKeys(t, p1.client)
	if _, ok := keys["dc1"][key1]; !ok {
		t.Fatalf("bad: %#v", keys)
	}
	if _, ok := keys["WAN"][key1]; !ok {
		t.Fatalf("bad: %#v", keys)
	}
}

func TestRPCClientInstallKey(t *testing.T) {
	key1 := "tbLJg26ZJyJ9pK3qhc9jig=="
	key2 := "xAEZ3uVHRMZD9GcYMZaRQw=="
	p1 := testRPCClientWithConfig(t, func(c *Config) {
		c.EncryptKey = key1
		c.ACLDatacenter = ""
	})
	defer p1.Close()

	// key2 is not installed yet
	testutil.WaitForResult(func() (bool, error) {
		keys := listKeys(t, p1.client)
		if num, ok := keys["dc1"][key2]; ok || num != 0 {
			return false, fmt.Errorf("bad: %#v", keys)
		}
		if num, ok := keys["WAN"][key2]; ok || num != 0 {
			return false, fmt.Errorf("bad: %#v", keys)
		}
		return true, nil
	}, func(err error) {
		t.Fatal(err.Error())
	})

	// install key2
	r, err := p1.client.InstallKey(key2, "")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	keyringSuccess(t, r)

	// key2 should now be installed
	testutil.WaitForResult(func() (bool, error) {
		keys := listKeys(t, p1.client)
		if num, ok := keys["dc1"][key2]; !ok || num != 1 {
			return false, fmt.Errorf("bad: %#v", keys)
		}
		if num, ok := keys["WAN"][key2]; !ok || num != 1 {
			return false, fmt.Errorf("bad: %#v", keys)
		}
		return true, nil
	}, func(err error) {
		t.Fatal(err.Error())
	})
}

func TestRPCClientUseKey(t *testing.T) {
	key1 := "tbLJg26ZJyJ9pK3qhc9jig=="
	key2 := "xAEZ3uVHRMZD9GcYMZaRQw=="
	p1 := testRPCClientWithConfig(t, func(c *Config) {
		c.EncryptKey = key1
		c.ACLDatacenter = ""
	})
	defer p1.Close()

	// add a second key to the ring
	r, err := p1.client.InstallKey(key2, "")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	keyringSuccess(t, r)

	// key2 is installed
	testutil.WaitForResult(func() (bool, error) {
		keys := listKeys(t, p1.client)
		if num, ok := keys["dc1"][key2]; !ok || num != 1 {
			return false, fmt.Errorf("bad: %#v", keys)
		}
		if num, ok := keys["WAN"][key2]; !ok || num != 1 {
			return false, fmt.Errorf("bad: %#v", keys)
		}
		return true, nil
	}, func(err error) {
		t.Fatal(err.Error())
	})

	// can't remove key1 yet
	r, err = p1.client.RemoveKey(key1, "")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	keyringError(t, r)

	// change primary key
	r, err = p1.client.UseKey(key2, "")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	keyringSuccess(t, r)

	// can remove key1 now
	r, err = p1.client.RemoveKey(key1, "")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	keyringSuccess(t, r)
}

func TestRPCClientKeyOperation_encryptionDisabled(t *testing.T) {
	p1 := testRPCClientWithConfig(t, func(c *Config) {
		c.ACLDatacenter = ""
	})
	defer p1.Close()

	r, err := p1.client.ListKeys("")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	keyringError(t, r)
}

func listKeys(t *testing.T, c *RPCClient) map[string]map[string]int {
	resp, err := c.ListKeys("")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	out := make(map[string]map[string]int)
	for _, k := range resp.Keys {
		respID := k.Datacenter
		if k.Pool == "WAN" {
			respID = k.Pool
		}
		out[respID] = map[string]int{k.Key: k.Count}
	}
	return out
}

func keyringError(t *testing.T, r keyringResponse) {
	for _, i := range r.Info {
		if i.Error == "" {
			t.Fatalf("no error reported from %s (%s)", i.Datacenter, i.Pool)
		}
	}
}

func keyringSuccess(t *testing.T, r keyringResponse) {
	for _, i := range r.Info {
		if i.Error != "" {
			t.Fatalf("error from %s (%s): %s", i.Datacenter, i.Pool, i.Error)
		}
	}
}
