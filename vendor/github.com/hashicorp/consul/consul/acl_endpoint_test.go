package consul

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/lib"
	"github.com/hashicorp/consul/testutil"
	"github.com/hashicorp/net-rpc-msgpackrpc"
)

func TestACLEndpoint_Apply(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLMasterToken = "root"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name: "User token",
			Type: structs.ACLTypeClient,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var out string
	if err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	id := out

	// Verify
	state := s1.fsm.State()
	_, s, err := state.ACLGet(out)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s == nil {
		t.Fatalf("should not be nil")
	}
	if s.ID != out {
		t.Fatalf("bad: %v", s)
	}
	if s.Name != "User token" {
		t.Fatalf("bad: %v", s)
	}

	// Do a delete
	arg.Op = structs.ACLDelete
	arg.ACL.ID = out
	if err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Verify
	_, s, err = state.ACLGet(id)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s != nil {
		t.Fatalf("bad: %v", s)
	}
}

func TestACLEndpoint_Update_PurgeCache(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLMasterToken = "root"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name: "User token",
			Type: structs.ACLTypeClient,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var out string
	if err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	id := out

	// Resolve
	acl1, err := s1.resolveToken(id)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if acl1 == nil {
		t.Fatalf("should not be nil")
	}
	if !acl1.KeyRead("foo") {
		t.Fatalf("should be allowed")
	}

	// Do an update
	arg.ACL.ID = out
	arg.ACL.Rules = `{"key": {"": {"policy": "deny"}}}`
	if err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Resolve again
	acl2, err := s1.resolveToken(id)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if acl2 == nil {
		t.Fatalf("should not be nil")
	}
	if acl2 == acl1 {
		t.Fatalf("should not be cached")
	}
	if acl2.KeyRead("foo") {
		t.Fatalf("should not be allowed")
	}

	// Do a delete
	arg.Op = structs.ACLDelete
	arg.ACL.Rules = ""
	if err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Resolve again
	acl3, err := s1.resolveToken(id)
	if err == nil || err.Error() != aclNotFound {
		t.Fatalf("err: %v", err)
	}
	if acl3 != nil {
		t.Fatalf("should be nil")
	}
}

func TestACLEndpoint_Apply_CustomID(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLMasterToken = "root"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			ID:   "foobarbaz", // Specify custom ID, does not exist
			Name: "User token",
			Type: structs.ACLTypeClient,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var out string
	if err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	if out != "foobarbaz" {
		t.Fatalf("bad token ID: %s", out)
	}

	// Verify
	state := s1.fsm.State()
	_, s, err := state.ACLGet(out)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s == nil {
		t.Fatalf("should not be nil")
	}
	if s.ID != out {
		t.Fatalf("bad: %v", s)
	}
	if s.Name != "User token" {
		t.Fatalf("bad: %v", s)
	}
}

func TestACLEndpoint_Apply_Denied(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name: "User token",
			Type: structs.ACLTypeClient,
		},
	}
	var out string
	err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out)
	if err == nil || !strings.Contains(err.Error(), permissionDenied) {
		t.Fatalf("err: %v", err)
	}
}

func TestACLEndpoint_Apply_DeleteAnon(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLMasterToken = "root"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLDelete,
		ACL: structs.ACL{
			ID:   anonymousToken,
			Name: "User token",
			Type: structs.ACLTypeClient,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var out string
	err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out)
	if err == nil || !strings.Contains(err.Error(), "delete anonymous") {
		t.Fatalf("err: %v", err)
	}
}

func TestACLEndpoint_Apply_RootChange(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLMasterToken = "root"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			ID:   "manage",
			Name: "User token",
			Type: structs.ACLTypeClient,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var out string
	err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out)
	if err == nil || !strings.Contains(err.Error(), "root ACL") {
		t.Fatalf("err: %v", err)
	}
}

func TestACLEndpoint_Get(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLMasterToken = "root"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name: "User token",
			Type: structs.ACLTypeClient,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var out string
	if err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	getR := structs.ACLSpecificRequest{
		Datacenter: "dc1",
		ACL:        out,
	}
	var acls structs.IndexedACLs
	if err := msgpackrpc.CallWithCodec(codec, "ACL.Get", &getR, &acls); err != nil {
		t.Fatalf("err: %v", err)
	}

	if acls.Index == 0 {
		t.Fatalf("Bad: %v", acls)
	}
	if len(acls.ACLs) != 1 {
		t.Fatalf("Bad: %v", acls)
	}
	s := acls.ACLs[0]
	if s.ID != out {
		t.Fatalf("bad: %v", s)
	}
}

func TestACLEndpoint_GetPolicy(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLMasterToken = "root"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name: "User token",
			Type: structs.ACLTypeClient,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var out string
	if err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	getR := structs.ACLPolicyRequest{
		Datacenter: "dc1",
		ACL:        out,
	}
	var acls structs.ACLPolicy
	if err := msgpackrpc.CallWithCodec(codec, "ACL.GetPolicy", &getR, &acls); err != nil {
		t.Fatalf("err: %v", err)
	}

	if acls.Policy == nil {
		t.Fatalf("Bad: %v", acls)
	}
	if acls.TTL != 30*time.Second {
		t.Fatalf("bad: %v", acls)
	}

	// Do a conditional lookup with etag
	getR.ETag = acls.ETag
	var out2 structs.ACLPolicy
	if err := msgpackrpc.CallWithCodec(codec, "ACL.GetPolicy", &getR, &out2); err != nil {
		t.Fatalf("err: %v", err)
	}

	if out2.Policy != nil {
		t.Fatalf("Bad: %v", out2)
	}
	if out2.TTL != 30*time.Second {
		t.Fatalf("bad: %v", out2)
	}
}

func TestACLEndpoint_List(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
		c.ACLMasterToken = "root"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	ids := []string{}
	for i := 0; i < 5; i++ {
		arg := structs.ACLRequest{
			Datacenter: "dc1",
			Op:         structs.ACLSet,
			ACL: structs.ACL{
				Name: "User token",
				Type: structs.ACLTypeClient,
			},
			WriteRequest: structs.WriteRequest{Token: "root"},
		}
		var out string
		if err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
		ids = append(ids, out)
	}

	getR := structs.DCSpecificRequest{
		Datacenter:   "dc1",
		QueryOptions: structs.QueryOptions{Token: "root"},
	}
	var acls structs.IndexedACLs
	if err := msgpackrpc.CallWithCodec(codec, "ACL.List", &getR, &acls); err != nil {
		t.Fatalf("err: %v", err)
	}

	if acls.Index == 0 {
		t.Fatalf("Bad: %v", acls)
	}

	// 5 + anonymous + master
	if len(acls.ACLs) != 7 {
		t.Fatalf("Bad: %v", acls.ACLs)
	}
	for i := 0; i < len(acls.ACLs); i++ {
		s := acls.ACLs[i]
		if s.ID == anonymousToken || s.ID == "root" {
			continue
		}
		if !lib.StrContains(ids, s.ID) {
			t.Fatalf("bad: %v", s)
		}
		if s.Name != "User token" {
			t.Fatalf("bad: %v", s)
		}
	}
}

func TestACLEndpoint_List_Denied(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.ACLDatacenter = "dc1"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	getR := structs.DCSpecificRequest{
		Datacenter: "dc1",
	}
	var acls structs.IndexedACLs
	err := msgpackrpc.CallWithCodec(codec, "ACL.List", &getR, &acls)
	if err == nil || !strings.Contains(err.Error(), permissionDenied) {
		t.Fatalf("err: %v", err)
	}
}
