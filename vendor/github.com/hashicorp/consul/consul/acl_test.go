package consul

import (
	"errors"
	"fmt"
	"os"
	"reflect"
	"testing"

	"github.com/hashicorp/consul/acl"
	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/testutil"
)

func TestACL_Disabled(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	client := rpcClient(t, s1)
	defer client.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	acl, err := s1.resolveToken("does not exist")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if acl != nil {
		t.Fatalf("got acl")
	}
}

func TestACL_ResolveRootACL(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1" // Enable ACLs!
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	acl, err := s1.resolveToken("allow")
	if err == nil || err.Error() != rootDenied {
		t.Fatalf("err: %v", err)
	}
	if acl != nil {
		t.Fatalf("bad: %v", acl)
	}

	acl, err = s1.resolveToken("deny")
	if err == nil || err.Error() != rootDenied {
		t.Fatalf("err: %v", err)
	}
	if acl != nil {
		t.Fatalf("bad: %v", acl)
	}
}

func TestACL_Authority_NotFound(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1" // Enable ACLs!
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	client := rpcClient(t, s1)
	defer client.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	acl, err := s1.resolveToken("does not exist")
	if err == nil || err.Error() != aclNotFound {
		t.Fatalf("err: %v", err)
	}
	if acl != nil {
		t.Fatalf("got acl")
	}
}

func TestACL_Authority_Found(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1" // Enable ACLs!
		c.ACLMasterToken = "root"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	client := rpcClient(t, s1)
	defer client.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Create a new token
	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name:  "User token",
			Type:  structs.ACLTypeClient,
			Rules: testACLPolicy,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var id string
	if err := s1.RPC("ACL.Apply", &arg, &id); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Resolve the token
	acl, err := s1.resolveToken(id)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if acl == nil {
		t.Fatalf("missing acl")
	}

	// Check the policy
	if acl.KeyRead("bar") {
		t.Fatalf("unexpected read")
	}
	if !acl.KeyRead("foo/test") {
		t.Fatalf("unexpected failed read")
	}
}

func TestACL_Authority_Anonymous_Found(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1" // Enable ACLs!
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	client := rpcClient(t, s1)
	defer client.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Resolve the token
	acl, err := s1.resolveToken("")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if acl == nil {
		t.Fatalf("missing acl")
	}

	// Check the policy, should allow all
	if !acl.KeyRead("foo/test") {
		t.Fatalf("unexpected failed read")
	}
}

func TestACL_Authority_Master_Found(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1" // Enable ACLs!
		c.ACLMasterToken = "foobar"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	client := rpcClient(t, s1)
	defer client.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Resolve the token
	acl, err := s1.resolveToken("foobar")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if acl == nil {
		t.Fatalf("missing acl")
	}

	// Check the policy, should allow all
	if !acl.KeyRead("foo/test") {
		t.Fatalf("unexpected failed read")
	}
}

func TestACL_Authority_Management(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1" // Enable ACLs!
		c.ACLMasterToken = "foobar"
		c.ACLDefaultPolicy = "deny"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	client := rpcClient(t, s1)
	defer client.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Resolve the token
	acl, err := s1.resolveToken("foobar")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if acl == nil {
		t.Fatalf("missing acl")
	}

	// Check the policy, should allow all
	if !acl.KeyRead("foo/test") {
		t.Fatalf("unexpected failed read")
	}
}

func TestACL_NonAuthority_NotFound(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	dir2, s2 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1" // Enable ACLs!
		c.Bootstrap = false     // Disable bootstrap
	})
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForResult(func() (bool, error) {
		p1, _ := s1.raftPeers.Peers()
		return len(p1) == 2, errors.New(fmt.Sprintf("%v", p1))
	}, func(err error) {
		t.Fatalf("should have 2 peers: %v", err)
	})

	client := rpcClient(t, s1)
	defer client.Close()
	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// find the non-authoritative server
	var nonAuth *Server
	if !s1.IsLeader() {
		nonAuth = s1
	} else {
		nonAuth = s2
	}

	acl, err := nonAuth.resolveToken("does not exist")
	if err == nil || err.Error() != aclNotFound {
		t.Fatalf("err: %v", err)
	}
	if acl != nil {
		t.Fatalf("got acl")
	}
}

func TestACL_NonAuthority_Found(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLMasterToken = "root"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	client := rpcClient(t, s1)
	defer client.Close()

	dir2, s2 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1" // Enable ACLs!
		c.Bootstrap = false     // Disable bootstrap
	})
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForResult(func() (bool, error) {
		p1, _ := s1.raftPeers.Peers()
		return len(p1) == 2, errors.New(fmt.Sprintf("%v", p1))
	}, func(err error) {
		t.Fatalf("should have 2 peers: %v", err)
	})
	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Create a new token
	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name:  "User token",
			Type:  structs.ACLTypeClient,
			Rules: testACLPolicy,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var id string
	if err := s1.RPC("ACL.Apply", &arg, &id); err != nil {
		t.Fatalf("err: %v", err)
	}

	// find the non-authoritative server
	var nonAuth *Server
	if !s1.IsLeader() {
		nonAuth = s1
	} else {
		nonAuth = s2
	}

	// Token should resolve
	acl, err := nonAuth.resolveToken(id)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if acl == nil {
		t.Fatalf("missing acl")
	}

	// Check the policy
	if acl.KeyRead("bar") {
		t.Fatalf("unexpected read")
	}
	if !acl.KeyRead("foo/test") {
		t.Fatalf("unexpected failed read")
	}
}

func TestACL_NonAuthority_Management(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1" // Enable ACLs!
		c.ACLMasterToken = "foobar"
		c.ACLDefaultPolicy = "deny"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	client := rpcClient(t, s1)
	defer client.Close()

	dir2, s2 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1" // Enable ACLs!
		c.ACLDefaultPolicy = "deny"
		c.Bootstrap = false // Disable bootstrap
	})
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForResult(func() (bool, error) {
		p1, _ := s1.raftPeers.Peers()
		return len(p1) == 2, errors.New(fmt.Sprintf("%v", p1))
	}, func(err error) {
		t.Fatalf("should have 2 peers: %v", err)
	})
	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// find the non-authoritative server
	var nonAuth *Server
	if !s1.IsLeader() {
		nonAuth = s1
	} else {
		nonAuth = s2
	}

	// Resolve the token
	acl, err := nonAuth.resolveToken("foobar")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if acl == nil {
		t.Fatalf("missing acl")
	}

	// Check the policy, should allow all
	if !acl.KeyRead("foo/test") {
		t.Fatalf("unexpected failed read")
	}
}

func TestACL_DownPolicy_Deny(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLDownPolicy = "deny"
		c.ACLMasterToken = "root"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	client := rpcClient(t, s1)
	defer client.Close()

	dir2, s2 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1" // Enable ACLs!
		c.ACLDownPolicy = "deny"
		c.Bootstrap = false // Disable bootstrap
	})
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForResult(func() (bool, error) {
		p1, _ := s1.raftPeers.Peers()
		return len(p1) == 2, errors.New(fmt.Sprintf("%v", p1))
	}, func(err error) {
		t.Fatalf("should have 2 peers: %v", err)
	})
	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Create a new token
	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name:  "User token",
			Type:  structs.ACLTypeClient,
			Rules: testACLPolicy,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var id string
	if err := s1.RPC("ACL.Apply", &arg, &id); err != nil {
		t.Fatalf("err: %v", err)
	}

	// find the non-authoritative server
	var nonAuth *Server
	var auth *Server
	if !s1.IsLeader() {
		nonAuth = s1
		auth = s2
	} else {
		nonAuth = s2
		auth = s1
	}

	// Kill the authoritative server
	auth.Shutdown()

	// Token should resolve into a DenyAll
	aclR, err := nonAuth.resolveToken(id)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if aclR != acl.DenyAll() {
		t.Fatalf("bad acl: %#v", aclR)
	}
}

func TestACL_DownPolicy_Allow(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLDownPolicy = "allow"
		c.ACLMasterToken = "root"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	client := rpcClient(t, s1)
	defer client.Close()

	dir2, s2 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1" // Enable ACLs!
		c.ACLDownPolicy = "allow"
		c.Bootstrap = false // Disable bootstrap
	})
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForResult(func() (bool, error) {
		p1, _ := s1.raftPeers.Peers()
		return len(p1) == 2, errors.New(fmt.Sprintf("%v", p1))
	}, func(err error) {
		t.Fatalf("should have 2 peers: %v", err)
	})
	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Create a new token
	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name:  "User token",
			Type:  structs.ACLTypeClient,
			Rules: testACLPolicy,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var id string
	if err := s1.RPC("ACL.Apply", &arg, &id); err != nil {
		t.Fatalf("err: %v", err)
	}

	// find the non-authoritative server
	var nonAuth *Server
	var auth *Server
	if !s1.IsLeader() {
		nonAuth = s1
		auth = s2
	} else {
		nonAuth = s2
		auth = s1
	}

	// Kill the authoritative server
	auth.Shutdown()

	// Token should resolve into a AllowAll
	aclR, err := nonAuth.resolveToken(id)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if aclR != acl.AllowAll() {
		t.Fatalf("bad acl: %#v", aclR)
	}
}

func TestACL_DownPolicy_ExtendCache(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLTTL = 0
		c.ACLDownPolicy = "extend-cache"
		c.ACLMasterToken = "root"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	client := rpcClient(t, s1)
	defer client.Close()

	dir2, s2 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1" // Enable ACLs!
		c.ACLTTL = 0
		c.ACLDownPolicy = "extend-cache"
		c.Bootstrap = false // Disable bootstrap
	})
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForResult(func() (bool, error) {
		p1, _ := s1.raftPeers.Peers()
		return len(p1) == 2, errors.New(fmt.Sprintf("%v", p1))
	}, func(err error) {
		t.Fatalf("should have 2 peers: %v", err)
	})
	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Create a new token
	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name:  "User token",
			Type:  structs.ACLTypeClient,
			Rules: testACLPolicy,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var id string
	if err := s1.RPC("ACL.Apply", &arg, &id); err != nil {
		t.Fatalf("err: %v", err)
	}

	// find the non-authoritative server
	var nonAuth *Server
	var auth *Server
	if !s1.IsLeader() {
		nonAuth = s1
		auth = s2
	} else {
		nonAuth = s2
		auth = s1
	}

	// Warm the caches
	aclR, err := nonAuth.resolveToken(id)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if aclR == nil {
		t.Fatalf("bad acl: %#v", aclR)
	}

	// Kill the authoritative server
	auth.Shutdown()

	// Token should resolve into cached copy
	aclR2, err := nonAuth.resolveToken(id)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if aclR2 != aclR {
		t.Fatalf("bad acl: %#v", aclR)
	}
}

func TestACL_MultiDC_Found(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLMasterToken = "root"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	client := rpcClient(t, s1)
	defer client.Close()

	dir2, s2 := testServerWithConfig(t, func(c *Config) {
		c.Datacenter = "dc2"
		c.ACLDatacenter = "dc1" // Enable ACLs!
	})
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfWANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinWAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc1")
	testutil.WaitForLeader(t, s1.RPC, "dc2")

	// Create a new token
	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name:  "User token",
			Type:  structs.ACLTypeClient,
			Rules: testACLPolicy,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var id string
	if err := s1.RPC("ACL.Apply", &arg, &id); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Token should resolve
	acl, err := s2.resolveToken(id)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if acl == nil {
		t.Fatalf("missing acl")
	}

	// Check the policy
	if acl.KeyRead("bar") {
		t.Fatalf("unexpected read")
	}
	if !acl.KeyRead("foo/test") {
		t.Fatalf("unexpected failed read")
	}
}

func TestACL_filterHealthChecks(t *testing.T) {
	// Create some health checks
	hc := structs.HealthChecks{
		&structs.HealthCheck{
			Node:        "node1",
			CheckID:     "check1",
			ServiceName: "foo",
		},
	}

	// Try permissive filtering
	filt := newAclFilter(acl.AllowAll(), nil)
	filt.filterHealthChecks(&hc)
	if len(hc) != 1 {
		t.Fatalf("bad: %#v", hc)
	}

	// Try restrictive filtering
	filt = newAclFilter(acl.DenyAll(), nil)
	filt.filterHealthChecks(&hc)
	if len(hc) != 0 {
		t.Fatalf("bad: %#v", hc)
	}
}

func TestACL_filterServices(t *testing.T) {
	// Create some services
	services := structs.Services{
		"service1": []string{},
		"service2": []string{},
	}

	// Try permissive filtering
	filt := newAclFilter(acl.AllowAll(), nil)
	filt.filterServices(services)
	if len(services) != 2 {
		t.Fatalf("bad: %#v", services)
	}

	// Try restrictive filtering
	filt = newAclFilter(acl.DenyAll(), nil)
	filt.filterServices(services)
	if len(services) != 0 {
		t.Fatalf("bad: %#v", services)
	}
}

func TestACL_filterServiceNodes(t *testing.T) {
	// Create some service nodes
	nodes := structs.ServiceNodes{
		&structs.ServiceNode{
			Node:        "node1",
			ServiceName: "foo",
		},
	}

	// Try permissive filtering
	filt := newAclFilter(acl.AllowAll(), nil)
	filt.filterServiceNodes(&nodes)
	if len(nodes) != 1 {
		t.Fatalf("bad: %#v", nodes)
	}

	// Try restrictive filtering
	filt = newAclFilter(acl.DenyAll(), nil)
	filt.filterServiceNodes(&nodes)
	if len(nodes) != 0 {
		t.Fatalf("bad: %#v", nodes)
	}
}

func TestACL_filterNodeServices(t *testing.T) {
	// Create some node services
	services := structs.NodeServices{
		Node: &structs.Node{
			Node: "node1",
		},
		Services: map[string]*structs.NodeService{
			"foo": &structs.NodeService{
				ID:      "foo",
				Service: "foo",
			},
		},
	}

	// Try permissive filtering
	filt := newAclFilter(acl.AllowAll(), nil)
	filt.filterNodeServices(&services)
	if len(services.Services) != 1 {
		t.Fatalf("bad: %#v", services.Services)
	}

	// Try restrictive filtering
	filt = newAclFilter(acl.DenyAll(), nil)
	filt.filterNodeServices(&services)
	if len(services.Services) != 0 {
		t.Fatalf("bad: %#v", services.Services)
	}
}

func TestACL_filterCheckServiceNodes(t *testing.T) {
	// Create some nodes
	nodes := structs.CheckServiceNodes{
		structs.CheckServiceNode{
			Node: &structs.Node{
				Node: "node1",
			},
			Service: &structs.NodeService{
				ID:      "foo",
				Service: "foo",
			},
			Checks: structs.HealthChecks{
				&structs.HealthCheck{
					Node:        "node1",
					CheckID:     "check1",
					ServiceName: "foo",
				},
			},
		},
	}

	// Try permissive filtering
	filt := newAclFilter(acl.AllowAll(), nil)
	filt.filterCheckServiceNodes(&nodes)
	if len(nodes) != 1 {
		t.Fatalf("bad: %#v", nodes)
	}
	if len(nodes[0].Checks) != 1 {
		t.Fatalf("bad: %#v", nodes[0].Checks)
	}

	// Try restrictive filtering
	filt = newAclFilter(acl.DenyAll(), nil)
	filt.filterCheckServiceNodes(&nodes)
	if len(nodes) != 0 {
		t.Fatalf("bad: %#v", nodes)
	}
}

func TestACL_filterNodeDump(t *testing.T) {
	// Create a node dump
	dump := structs.NodeDump{
		&structs.NodeInfo{
			Node: "node1",
			Services: []*structs.NodeService{
				&structs.NodeService{
					ID:      "foo",
					Service: "foo",
				},
			},
			Checks: []*structs.HealthCheck{
				&structs.HealthCheck{
					Node:        "node1",
					CheckID:     "check1",
					ServiceName: "foo",
				},
			},
		},
	}

	// Try permissive filtering
	filt := newAclFilter(acl.AllowAll(), nil)
	filt.filterNodeDump(&dump)
	if len(dump) != 1 {
		t.Fatalf("bad: %#v", dump)
	}
	if len(dump[0].Services) != 1 {
		t.Fatalf("bad: %#v", dump[0].Services)
	}
	if len(dump[0].Checks) != 1 {
		t.Fatalf("bad: %#v", dump[0].Checks)
	}

	// Try restrictive filtering
	filt = newAclFilter(acl.DenyAll(), nil)
	filt.filterNodeDump(&dump)
	if len(dump) != 1 {
		t.Fatalf("bad: %#v", dump)
	}
	if len(dump[0].Services) != 0 {
		t.Fatalf("bad: %#v", dump[0].Services)
	}
	if len(dump[0].Checks) != 0 {
		t.Fatalf("bad: %#v", dump[0].Checks)
	}
}

func TestACL_redactPreparedQueryTokens(t *testing.T) {
	query := &structs.PreparedQuery{
		ID:    "f004177f-2c28-83b7-4229-eacc25fe55d1",
		Token: "root",
	}

	expected := &structs.PreparedQuery{
		ID:    "f004177f-2c28-83b7-4229-eacc25fe55d1",
		Token: "root",
	}

	// Try permissive filtering with a management token. This will allow the
	// embedded token to be seen.
	filt := newAclFilter(acl.ManageAll(), nil)
	filt.redactPreparedQueryTokens(&query)
	if !reflect.DeepEqual(query, expected) {
		t.Fatalf("bad: %#v", &query)
	}

	// Hang on to the entry with a token, which needs to survive the next
	// operation.
	original := query

	// Now try permissive filtering with a client token, which should cause
	// the embedded token to get redacted.
	filt = newAclFilter(acl.AllowAll(), nil)
	filt.redactPreparedQueryTokens(&query)
	expected.Token = redactedToken
	if !reflect.DeepEqual(query, expected) {
		t.Fatalf("bad: %#v", *query)
	}

	// Make sure that the original object didn't lose its token.
	if original.Token != "root" {
		t.Fatalf("bad token: %s", original.Token)
	}
}

func TestACL_filterPreparedQueries(t *testing.T) {
	queries := structs.PreparedQueries{
		&structs.PreparedQuery{
			ID: "f004177f-2c28-83b7-4229-eacc25fe55d1",
		},
		&structs.PreparedQuery{
			ID:   "f004177f-2c28-83b7-4229-eacc25fe55d2",
			Name: "query-with-no-token",
		},
		&structs.PreparedQuery{
			ID:    "f004177f-2c28-83b7-4229-eacc25fe55d3",
			Name:  "query-with-a-token",
			Token: "root",
		},
	}

	expected := structs.PreparedQueries{
		&structs.PreparedQuery{
			ID: "f004177f-2c28-83b7-4229-eacc25fe55d1",
		},
		&structs.PreparedQuery{
			ID:   "f004177f-2c28-83b7-4229-eacc25fe55d2",
			Name: "query-with-no-token",
		},
		&structs.PreparedQuery{
			ID:    "f004177f-2c28-83b7-4229-eacc25fe55d3",
			Name:  "query-with-a-token",
			Token: "root",
		},
	}

	// Try permissive filtering with a management token. This will allow the
	// embedded token to be seen.
	filt := newAclFilter(acl.ManageAll(), nil)
	filt.filterPreparedQueries(&queries)
	if !reflect.DeepEqual(queries, expected) {
		t.Fatalf("bad: %#v", queries)
	}

	// Hang on to the entry with a token, which needs to survive the next
	// operation.
	original := queries[2]

	// Now try permissive filtering with a client token, which should cause
	// the embedded token to get redacted, and the query with no name to get
	// filtered out.
	filt = newAclFilter(acl.AllowAll(), nil)
	filt.filterPreparedQueries(&queries)
	expected[2].Token = redactedToken
	expected = append(structs.PreparedQueries{}, expected[1], expected[2])
	if !reflect.DeepEqual(queries, expected) {
		t.Fatalf("bad: %#v", queries)
	}

	// Make sure that the original object didn't lose its token.
	if original.Token != "root" {
		t.Fatalf("bad token: %s", original.Token)
	}

	// Now try restrictive filtering.
	filt = newAclFilter(acl.DenyAll(), nil)
	filt.filterPreparedQueries(&queries)
	if len(queries) != 0 {
		t.Fatalf("bad: %#v", queries)
	}
}

func TestACL_unhandledFilterType(t *testing.T) {
	defer func(t *testing.T) {
		if recover() == nil {
			t.Fatalf("should panic")
		}
	}(t)

	// Create the server
	dir, token, srv, client := testACLFilterServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer client.Close()

	// Pass an unhandled type into the ACL filter.
	srv.filterACL(token, &structs.HealthCheck{})
}

var testACLPolicy = `
key "" {
	policy = "deny"
}
key "foo/" {
	policy = "write"
}
`
