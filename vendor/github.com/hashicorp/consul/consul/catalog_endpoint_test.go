package consul

import (
	"fmt"
	"net/rpc"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/lib"
	"github.com/hashicorp/consul/testutil"
	"github.com/hashicorp/net-rpc-msgpackrpc"
)

func TestCatalogRegister(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	arg := structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			Service: "db",
			Tags:    []string{"master"},
			Port:    8000,
		},
	}
	var out struct{}

	err := msgpackrpc.CallWithCodec(codec, "Catalog.Register", &arg, &out)
	if err == nil || err.Error() != "No cluster leader" {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForResult(func() (bool, error) {
		err := msgpackrpc.CallWithCodec(codec, "Catalog.Register", &arg, &out)
		return err == nil, err
	}, func(err error) {
		t.Fatalf("err: %v", err)
	})
}

func TestCatalogRegister_ACLDeny(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLMasterToken = "root"
		c.ACLDefaultPolicy = "deny"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Create the ACL
	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name:  "User token",
			Type:  structs.ACLTypeClient,
			Rules: testRegisterRules,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var out string
	if err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	id := out

	argR := structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			Service: "db",
			Tags:    []string{"master"},
			Port:    8000,
		},
		WriteRequest: structs.WriteRequest{Token: id},
	}
	var outR struct{}

	err := msgpackrpc.CallWithCodec(codec, "Catalog.Register", &argR, &outR)
	if err == nil || !strings.Contains(err.Error(), permissionDenied) {
		t.Fatalf("err: %v", err)
	}

	argR.Service.Service = "foo"
	err = msgpackrpc.CallWithCodec(codec, "Catalog.Register", &argR, &outR)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
}

func TestCatalogRegister_ForwardLeader(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec1 := rpcClient(t, s1)
	defer codec1.Close()

	dir2, s2 := testServer(t)
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()
	codec2 := rpcClient(t, s2)
	defer codec2.Close()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc1")
	testutil.WaitForLeader(t, s2.RPC, "dc1")

	// Use the follower as the client
	var codec rpc.ClientCodec
	if !s1.IsLeader() {
		codec = codec1
	} else {
		codec = codec2
	}

	arg := structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			Service: "db",
			Tags:    []string{"master"},
			Port:    8000,
		},
	}
	var out struct{}
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.Register", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
}

func TestCatalogRegister_ForwardDC(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	dir2, s2 := testServerDC(t, "dc2")
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfWANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinWAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc2")

	arg := structs.RegisterRequest{
		Datacenter: "dc2", // Should forward through s1
		Node:       "foo",
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			Service: "db",
			Tags:    []string{"master"},
			Port:    8000,
		},
	}
	var out struct{}
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.Register", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
}

func TestCatalogDeregister(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	arg := structs.DeregisterRequest{
		Datacenter: "dc1",
		Node:       "foo",
	}
	var out struct{}

	err := msgpackrpc.CallWithCodec(codec, "Catalog.Deregister", &arg, &out)
	if err == nil || err.Error() != "No cluster leader" {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	if err := msgpackrpc.CallWithCodec(codec, "Catalog.Deregister", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
}

func TestCatalogListDatacenters(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	dir2, s2 := testServerDC(t, "dc2")
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfWANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinWAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	var out []string
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ListDatacenters", struct{}{}, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// The DCs should come out sorted by default.
	if len(out) != 2 {
		t.Fatalf("bad: %v", out)
	}
	if out[0] != "dc1" {
		t.Fatalf("bad: %v", out)
	}
	if out[1] != "dc2" {
		t.Fatalf("bad: %v", out)
	}
}

func TestCatalogListDatacenters_DistanceSort(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	dir2, s2 := testServerDC(t, "dc2")
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()

	dir3, s3 := testServerDC(t, "acdc")
	defer os.RemoveAll(dir3)
	defer s3.Shutdown()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfWANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinWAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if _, err := s3.JoinWAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}
	testutil.WaitForLeader(t, s1.RPC, "dc1")

	var out []string
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ListDatacenters", struct{}{}, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// It's super hard to force the Serfs into a known configuration of
	// coordinates, so the best we can do is make sure that the sorting
	// function is getting called (it's tested extensively in rtt_test.go).
	// Since this is relative to dc1, it will be listed first (proving we
	// went into the sort fn) and the other two will be sorted by name since
	// there are no known coordinates for them.
	if len(out) != 3 {
		t.Fatalf("bad: %v", out)
	}
	if out[0] != "dc1" {
		t.Fatalf("bad: %v", out)
	}
	if out[1] != "acdc" {
		t.Fatalf("bad: %v", out)
	}
	if out[2] != "dc2" {
		t.Fatalf("bad: %v", out)
	}
}

func TestCatalogListNodes(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	args := structs.DCSpecificRequest{
		Datacenter: "dc1",
	}
	var out structs.IndexedNodes
	err := msgpackrpc.CallWithCodec(codec, "Catalog.ListNodes", &args, &out)
	if err == nil || err.Error() != "No cluster leader" {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Just add a node
	if err := s1.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForResult(func() (bool, error) {
		msgpackrpc.CallWithCodec(codec, "Catalog.ListNodes", &args, &out)
		return len(out.Nodes) == 2, nil
	}, func(err error) {
		t.Fatalf("err: %v", err)
	})

	// Server node is auto added from Serf
	if out.Nodes[1].Node != s1.config.NodeName {
		t.Fatalf("bad: %v", out)
	}
	if out.Nodes[0].Node != "foo" {
		t.Fatalf("bad: %v", out)
	}
	if out.Nodes[0].Address != "127.0.0.1" {
		t.Fatalf("bad: %v", out)
	}
}

func TestCatalogListNodes_StaleRaad(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec1 := rpcClient(t, s1)
	defer codec1.Close()

	dir2, s2 := testServerDCBootstrap(t, "dc1", false)
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()
	codec2 := rpcClient(t, s2)
	defer codec2.Close()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc1")
	testutil.WaitForLeader(t, s2.RPC, "dc1")

	// Use the follower as the client
	var codec rpc.ClientCodec
	if !s1.IsLeader() {
		codec = codec1

		// Inject fake data on the follower!
		if err := s1.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
			t.Fatalf("err: %v", err)
		}
	} else {
		codec = codec2

		// Inject fake data on the follower!
		if err := s2.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	args := structs.DCSpecificRequest{
		Datacenter:   "dc1",
		QueryOptions: structs.QueryOptions{AllowStale: true},
	}
	var out structs.IndexedNodes
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ListNodes", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	found := false
	for _, n := range out.Nodes {
		if n.Node == "foo" {
			found = true
		}
	}
	if !found {
		t.Fatalf("failed to find foo")
	}

	if out.QueryMeta.LastContact == 0 {
		t.Fatalf("should have a last contact time")
	}
	if !out.QueryMeta.KnownLeader {
		t.Fatalf("should have known leader")
	}
}

func TestCatalogListNodes_ConsistentRead_Fail(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec1 := rpcClient(t, s1)
	defer codec1.Close()

	dir2, s2 := testServerDCBootstrap(t, "dc1", false)
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()
	codec2 := rpcClient(t, s2)
	defer codec2.Close()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc1")
	testutil.WaitForLeader(t, s2.RPC, "dc1")

	// Use the leader as the client, kill the follower
	var codec rpc.ClientCodec
	if s1.IsLeader() {
		codec = codec1
		s2.Shutdown()
	} else {
		codec = codec2
		s1.Shutdown()
	}

	args := structs.DCSpecificRequest{
		Datacenter:   "dc1",
		QueryOptions: structs.QueryOptions{RequireConsistent: true},
	}
	var out structs.IndexedNodes
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ListNodes", &args, &out); !strings.HasPrefix(err.Error(), "leadership lost") {
		t.Fatalf("err: %v", err)
	}

	if out.QueryMeta.LastContact != 0 {
		t.Fatalf("should not have a last contact time")
	}
	if out.QueryMeta.KnownLeader {
		t.Fatalf("should have no known leader")
	}
}

func TestCatalogListNodes_ConsistentRead(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec1 := rpcClient(t, s1)
	defer codec1.Close()

	dir2, s2 := testServerDCBootstrap(t, "dc1", false)
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()
	codec2 := rpcClient(t, s2)
	defer codec2.Close()

	// Try to join
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfLANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinLAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc1")
	testutil.WaitForLeader(t, s2.RPC, "dc1")

	// Use the leader as the client, kill the follower
	var codec rpc.ClientCodec
	if s1.IsLeader() {
		codec = codec1
	} else {
		codec = codec2
	}

	args := structs.DCSpecificRequest{
		Datacenter:   "dc1",
		QueryOptions: structs.QueryOptions{RequireConsistent: true},
	}
	var out structs.IndexedNodes
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ListNodes", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	if out.QueryMeta.LastContact != 0 {
		t.Fatalf("should not have a last contact time")
	}
	if !out.QueryMeta.KnownLeader {
		t.Fatalf("should have known leader")
	}
}

func TestCatalogListNodes_DistanceSort(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")
	if err := s1.fsm.State().EnsureNode(1, &structs.Node{Node: "aaa", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := s1.fsm.State().EnsureNode(2, &structs.Node{Node: "foo", Address: "127.0.0.2"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := s1.fsm.State().EnsureNode(3, &structs.Node{Node: "bar", Address: "127.0.0.3"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := s1.fsm.State().EnsureNode(4, &structs.Node{Node: "baz", Address: "127.0.0.4"}); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Set all but one of the nodes to known coordinates.
	updates := structs.Coordinates{
		{"foo", generateCoordinate(2 * time.Millisecond)},
		{"bar", generateCoordinate(5 * time.Millisecond)},
		{"baz", generateCoordinate(1 * time.Millisecond)},
	}
	if err := s1.fsm.State().CoordinateBatchUpdate(5, updates); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Query with no given source node, should get the natural order from
	// the index.
	args := structs.DCSpecificRequest{
		Datacenter: "dc1",
	}
	var out structs.IndexedNodes
	testutil.WaitForResult(func() (bool, error) {
		msgpackrpc.CallWithCodec(codec, "Catalog.ListNodes", &args, &out)
		return len(out.Nodes) == 5, nil
	}, func(err error) {
		t.Fatalf("err: %v", err)
	})
	if out.Nodes[0].Node != "aaa" {
		t.Fatalf("bad: %v", out)
	}
	if out.Nodes[1].Node != "bar" {
		t.Fatalf("bad: %v", out)
	}
	if out.Nodes[2].Node != "baz" {
		t.Fatalf("bad: %v", out)
	}
	if out.Nodes[3].Node != "foo" {
		t.Fatalf("bad: %v", out)
	}
	if out.Nodes[4].Node != s1.config.NodeName {
		t.Fatalf("bad: %v", out)
	}

	// Query relative to foo, note that there's no known coordinate for the
	// default-added Serf node nor "aaa" so they will go at the end.
	args = structs.DCSpecificRequest{
		Datacenter: "dc1",
		Source:     structs.QuerySource{Datacenter: "dc1", Node: "foo"},
	}
	testutil.WaitForResult(func() (bool, error) {
		msgpackrpc.CallWithCodec(codec, "Catalog.ListNodes", &args, &out)
		return len(out.Nodes) == 5, nil
	}, func(err error) {
		t.Fatalf("err: %v", err)
	})
	if out.Nodes[0].Node != "foo" {
		t.Fatalf("bad: %v", out)
	}
	if out.Nodes[1].Node != "baz" {
		t.Fatalf("bad: %v", out)
	}
	if out.Nodes[2].Node != "bar" {
		t.Fatalf("bad: %v", out)
	}
	if out.Nodes[3].Node != "aaa" {
		t.Fatalf("bad: %v", out)
	}
	if out.Nodes[4].Node != s1.config.NodeName {
		t.Fatalf("bad: %v", out)
	}
}

func BenchmarkCatalogListNodes(t *testing.B) {
	dir1, s1 := testServer(nil)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(nil, s1)
	defer codec.Close()

	// Just add a node
	if err := s1.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}

	args := structs.DCSpecificRequest{
		Datacenter: "dc1",
	}
	for i := 0; i < t.N; i++ {
		var out structs.IndexedNodes
		if err := msgpackrpc.CallWithCodec(codec, "Catalog.ListNodes", &args, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}
}

func TestCatalogListServices(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	args := structs.DCSpecificRequest{
		Datacenter: "dc1",
	}
	var out structs.IndexedServices
	err := msgpackrpc.CallWithCodec(codec, "Catalog.ListServices", &args, &out)
	if err == nil || err.Error() != "No cluster leader" {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Just add a node
	if err := s1.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := s1.fsm.State().EnsureService(2, "foo", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"primary"}, Address: "127.0.0.1", Port: 5000}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ListServices", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(out.Services) != 2 {
		t.Fatalf("bad: %v", out)
	}
	for _, s := range out.Services {
		if s == nil {
			t.Fatalf("bad: %v", s)
		}
	}
	// Consul service should auto-register
	if _, ok := out.Services["consul"]; !ok {
		t.Fatalf("bad: %v", out)
	}
	if len(out.Services["db"]) != 1 {
		t.Fatalf("bad: %v", out)
	}
	if out.Services["db"][0] != "primary" {
		t.Fatalf("bad: %v", out)
	}
}

func TestCatalogListServices_Blocking(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	args := structs.DCSpecificRequest{
		Datacenter: "dc1",
	}
	var out structs.IndexedServices

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Run the query
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ListServices", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Setup a blocking query
	args.MinQueryIndex = out.Index
	args.MaxQueryTime = time.Second

	// Async cause a change
	idx := out.Index
	start := time.Now()
	go func() {
		time.Sleep(100 * time.Millisecond)
		if err := s1.fsm.State().EnsureNode(idx+1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
			t.Fatalf("err: %v", err)
		}
		if err := s1.fsm.State().EnsureService(idx+2, "foo", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"primary"}, Address: "127.0.0.1", Port: 5000}); err != nil {
			t.Fatalf("err: %v", err)
		}
	}()

	// Re-run the query
	out = structs.IndexedServices{}
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ListServices", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should block at least 100ms
	if time.Now().Sub(start) < 100*time.Millisecond {
		t.Fatalf("too fast")
	}

	// Check the indexes
	if out.Index != idx+2 {
		t.Fatalf("bad: %v", out)
	}

	// Should find the service
	if len(out.Services) != 2 {
		t.Fatalf("bad: %v", out)
	}
}

func TestCatalogListServices_Timeout(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	args := structs.DCSpecificRequest{
		Datacenter: "dc1",
	}
	var out structs.IndexedServices

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Run the query
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ListServices", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Setup a blocking query
	args.MinQueryIndex = out.Index
	args.MaxQueryTime = 100 * time.Millisecond

	// Re-run the query
	start := time.Now()
	out = structs.IndexedServices{}
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ListServices", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should block at least 100ms
	if time.Now().Sub(start) < 100*time.Millisecond {
		t.Fatalf("too fast")
	}

	// Check the indexes, should not change
	if out.Index != args.MinQueryIndex {
		t.Fatalf("bad: %v", out)
	}
}

func TestCatalogListServices_Stale(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	args := structs.DCSpecificRequest{
		Datacenter: "dc1",
	}
	args.AllowStale = true
	var out structs.IndexedServices

	// Inject a fake service
	if err := s1.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := s1.fsm.State().EnsureService(2, "foo", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"primary"}, Address: "127.0.0.1", Port: 5000}); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Run the query, do not wait for leader!
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ListServices", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should find the service
	if len(out.Services) != 1 {
		t.Fatalf("bad: %v", out)
	}

	// Should not have a leader! Stale read
	if out.KnownLeader {
		t.Fatalf("bad: %v", out)
	}
}

func TestCatalogListServiceNodes(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	args := structs.ServiceSpecificRequest{
		Datacenter:  "dc1",
		ServiceName: "db",
		ServiceTag:  "slave",
		TagFilter:   false,
	}
	var out structs.IndexedServiceNodes
	err := msgpackrpc.CallWithCodec(codec, "Catalog.ServiceNodes", &args, &out)
	if err == nil || err.Error() != "No cluster leader" {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Just add a node
	if err := s1.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := s1.fsm.State().EnsureService(2, "foo", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"primary"}, Address: "127.0.0.1", Port: 5000}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ServiceNodes", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(out.ServiceNodes) != 1 {
		t.Fatalf("bad: %v", out)
	}

	// Try with a filter
	args.TagFilter = true
	out = structs.IndexedServiceNodes{}

	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ServiceNodes", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out.ServiceNodes) != 0 {
		t.Fatalf("bad: %v", out)
	}
}

func TestCatalogListServiceNodes_DistanceSort(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	args := structs.ServiceSpecificRequest{
		Datacenter:  "dc1",
		ServiceName: "db",
	}
	var out structs.IndexedServiceNodes
	err := msgpackrpc.CallWithCodec(codec, "Catalog.ServiceNodes", &args, &out)
	if err == nil || err.Error() != "No cluster leader" {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Add a few nodes for the associated services.
	s1.fsm.State().EnsureNode(1, &structs.Node{Node: "aaa", Address: "127.0.0.1"})
	s1.fsm.State().EnsureService(2, "aaa", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"primary"}, Address: "127.0.0.1", Port: 5000})
	s1.fsm.State().EnsureNode(3, &structs.Node{Node: "foo", Address: "127.0.0.2"})
	s1.fsm.State().EnsureService(4, "foo", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"primary"}, Address: "127.0.0.2", Port: 5000})
	s1.fsm.State().EnsureNode(5, &structs.Node{Node: "bar", Address: "127.0.0.3"})
	s1.fsm.State().EnsureService(6, "bar", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"primary"}, Address: "127.0.0.3", Port: 5000})
	s1.fsm.State().EnsureNode(7, &structs.Node{Node: "baz", Address: "127.0.0.4"})
	s1.fsm.State().EnsureService(8, "baz", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"primary"}, Address: "127.0.0.4", Port: 5000})

	// Set all but one of the nodes to known coordinates.
	updates := structs.Coordinates{
		{"foo", generateCoordinate(2 * time.Millisecond)},
		{"bar", generateCoordinate(5 * time.Millisecond)},
		{"baz", generateCoordinate(1 * time.Millisecond)},
	}
	if err := s1.fsm.State().CoordinateBatchUpdate(9, updates); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Query with no given source node, should get the natural order from
	// the index.
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ServiceNodes", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out.ServiceNodes) != 4 {
		t.Fatalf("bad: %v", out)
	}
	if out.ServiceNodes[0].Node != "aaa" {
		t.Fatalf("bad: %v", out)
	}
	if out.ServiceNodes[1].Node != "bar" {
		t.Fatalf("bad: %v", out)
	}
	if out.ServiceNodes[2].Node != "baz" {
		t.Fatalf("bad: %v", out)
	}
	if out.ServiceNodes[3].Node != "foo" {
		t.Fatalf("bad: %v", out)
	}

	// Query relative to foo, note that there's no known coordinate for "aaa"
	// so it will go at the end.
	args = structs.ServiceSpecificRequest{
		Datacenter:  "dc1",
		ServiceName: "db",
		Source:      structs.QuerySource{Datacenter: "dc1", Node: "foo"},
	}
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ServiceNodes", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(out.ServiceNodes) != 4 {
		t.Fatalf("bad: %v", out)
	}
	if out.ServiceNodes[0].Node != "foo" {
		t.Fatalf("bad: %v", out)
	}
	if out.ServiceNodes[1].Node != "baz" {
		t.Fatalf("bad: %v", out)
	}
	if out.ServiceNodes[2].Node != "bar" {
		t.Fatalf("bad: %v", out)
	}
	if out.ServiceNodes[3].Node != "aaa" {
		t.Fatalf("bad: %v", out)
	}
}

func TestCatalogNodeServices(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	args := structs.NodeSpecificRequest{
		Datacenter: "dc1",
		Node:       "foo",
	}
	var out structs.IndexedNodeServices
	err := msgpackrpc.CallWithCodec(codec, "Catalog.NodeServices", &args, &out)
	if err == nil || err.Error() != "No cluster leader" {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Just add a node
	if err := s1.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := s1.fsm.State().EnsureService(2, "foo", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"primary"}, Address: "127.0.0.1", Port: 5000}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := s1.fsm.State().EnsureService(3, "foo", &structs.NodeService{ID: "web", Service: "web", Tags: nil, Address: "127.0.0.1", Port: 80}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := msgpackrpc.CallWithCodec(codec, "Catalog.NodeServices", &args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	if out.NodeServices.Node.Address != "127.0.0.1" {
		t.Fatalf("bad: %v", out)
	}
	if len(out.NodeServices.Services) != 2 {
		t.Fatalf("bad: %v", out)
	}
	services := out.NodeServices.Services
	if !lib.StrContains(services["db"].Tags, "primary") || services["db"].Port != 5000 {
		t.Fatalf("bad: %v", out)
	}
	if len(services["web"].Tags) != 0 || services["web"].Port != 80 {
		t.Fatalf("bad: %v", out)
	}
}

// Used to check for a regression against a known bug
func TestCatalogRegister_FailedCase1(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	arg := structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "bar",
		Address:    "127.0.0.2",
		Service: &structs.NodeService{
			Service: "web",
			Tags:    nil,
			Port:    8000,
		},
	}
	var out struct{}

	err := msgpackrpc.CallWithCodec(codec, "Catalog.Register", &arg, &out)
	if err == nil || err.Error() != "No cluster leader" {
		t.Fatalf("err: %v", err)
	}

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	if err := msgpackrpc.CallWithCodec(codec, "Catalog.Register", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Check we can get this back
	query := &structs.ServiceSpecificRequest{
		Datacenter:  "dc1",
		ServiceName: "web",
	}
	var out2 structs.IndexedServiceNodes
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ServiceNodes", query, &out2); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Check the output
	if len(out2.ServiceNodes) != 1 {
		t.Fatalf("Bad: %v", out2)
	}
}

func testACLFilterServer(t *testing.T) (dir, token string, srv *Server, codec rpc.ClientCodec) {
	dir, srv = testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLMasterToken = "root"
		c.ACLDefaultPolicy = "deny"
	})

	codec = rpcClient(t, srv)
	testutil.WaitForLeader(t, srv.RPC, "dc1")

	// Create a new token
	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name:  "User token",
			Type:  structs.ACLTypeClient,
			Rules: testRegisterRules,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	if err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &token); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Register a service
	regArg := structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       srv.config.NodeName,
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			ID:      "foo",
			Service: "foo",
		},
		Check: &structs.HealthCheck{
			CheckID:   "service:foo",
			Name:      "service:foo",
			ServiceID: "foo",
			Status:    structs.HealthPassing,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.Register", &regArg, nil); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Register a service which should be denied
	regArg = structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       srv.config.NodeName,
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			ID:      "bar",
			Service: "bar",
		},
		Check: &structs.HealthCheck{
			CheckID:   "service:bar",
			Name:      "service:bar",
			ServiceID: "bar",
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.Register", &regArg, nil); err != nil {
		t.Fatalf("err: %s", err)
	}
	return
}

func TestCatalog_ListServices_FilterACL(t *testing.T) {
	dir, token, srv, codec := testACLFilterServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer codec.Close()

	opt := structs.DCSpecificRequest{
		Datacenter:   "dc1",
		QueryOptions: structs.QueryOptions{Token: token},
	}
	reply := structs.IndexedServices{}
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ListServices", &opt, &reply); err != nil {
		t.Fatalf("err: %s", err)
	}
	if _, ok := reply.Services["foo"]; !ok {
		t.Fatalf("bad: %#v", reply.Services)
	}
	if _, ok := reply.Services["bar"]; ok {
		t.Fatalf("bad: %#v", reply.Services)
	}
}

func TestCatalog_ServiceNodes_FilterACL(t *testing.T) {
	dir, token, srv, codec := testACLFilterServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer codec.Close()

	opt := structs.ServiceSpecificRequest{
		Datacenter:   "dc1",
		ServiceName:  "foo",
		QueryOptions: structs.QueryOptions{Token: token},
	}
	reply := structs.IndexedServiceNodes{}
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ServiceNodes", &opt, &reply); err != nil {
		t.Fatalf("err: %s", err)
	}
	found := false
	for _, sn := range reply.ServiceNodes {
		if sn.ServiceID == "foo" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("bad: %#v", reply.ServiceNodes)
	}

	// Filters services we can't access
	opt = structs.ServiceSpecificRequest{
		Datacenter:   "dc1",
		ServiceName:  "bar",
		QueryOptions: structs.QueryOptions{Token: token},
	}
	reply = structs.IndexedServiceNodes{}
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.ServiceNodes", &opt, &reply); err != nil {
		t.Fatalf("err: %s", err)
	}
	for _, sn := range reply.ServiceNodes {
		if sn.ServiceID == "bar" {
			t.Fatalf("bad: %#v", reply.ServiceNodes)
		}
	}
}

func TestCatalog_NodeServices_FilterACL(t *testing.T) {
	dir, token, srv, codec := testACLFilterServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer codec.Close()

	opt := structs.NodeSpecificRequest{
		Datacenter:   "dc1",
		Node:         srv.config.NodeName,
		QueryOptions: structs.QueryOptions{Token: token},
	}
	reply := structs.IndexedNodeServices{}
	if err := msgpackrpc.CallWithCodec(codec, "Catalog.NodeServices", &opt, &reply); err != nil {
		t.Fatalf("err: %s", err)
	}
	found := false
	for _, svc := range reply.NodeServices.Services {
		if svc.ID == "bar" {
			t.Fatalf("bad: %#v", reply.NodeServices.Services)
		}
		if svc.ID == "foo" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("bad: %#v", reply.NodeServices)
	}
}

var testRegisterRules = `
service "foo" {
	policy = "write"
}
`
