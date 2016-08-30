package consul

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/testutil"
	"github.com/hashicorp/net-rpc-msgpackrpc"
)

func TestKVS_Apply(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	arg := structs.KVSRequest{
		Datacenter: "dc1",
		Op:         structs.KVSSet,
		DirEnt: structs.DirEntry{
			Key:   "test",
			Flags: 42,
			Value: []byte("test"),
		},
	}
	var out bool
	if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Verify
	state := s1.fsm.State()
	_, d, err := state.KVSGet("test")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if d == nil {
		t.Fatalf("should not be nil")
	}

	// Do a check and set
	arg.Op = structs.KVSCAS
	arg.DirEnt.ModifyIndex = d.ModifyIndex
	arg.DirEnt.Flags = 43
	if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Check this was applied
	if out != true {
		t.Fatalf("bad: %v", out)
	}

	// Verify
	_, d, err = state.KVSGet("test")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if d.Flags != 43 {
		t.Fatalf("bad: %v", d)
	}
}

func TestKVS_Apply_ACLDeny(t *testing.T) {
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
			Rules: testListRules,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var out string
	if err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	id := out

	// Try a write
	argR := structs.KVSRequest{
		Datacenter: "dc1",
		Op:         structs.KVSSet,
		DirEnt: structs.DirEntry{
			Key:   "foo/bar",
			Flags: 42,
			Value: []byte("test"),
		},
		WriteRequest: structs.WriteRequest{Token: id},
	}
	var outR bool
	err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &argR, &outR)
	if err == nil || !strings.Contains(err.Error(), permissionDenied) {
		t.Fatalf("err: %v", err)
	}

	// Try a recursive delete
	argR = structs.KVSRequest{
		Datacenter: "dc1",
		Op:         structs.KVSDeleteTree,
		DirEnt: structs.DirEntry{
			Key: "test",
		},
		WriteRequest: structs.WriteRequest{Token: id},
	}
	err = msgpackrpc.CallWithCodec(codec, "KVS.Apply", &argR, &outR)
	if err == nil || !strings.Contains(err.Error(), permissionDenied) {
		t.Fatalf("err: %v", err)
	}
}

func TestKVS_Get(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	arg := structs.KVSRequest{
		Datacenter: "dc1",
		Op:         structs.KVSSet,
		DirEnt: structs.DirEntry{
			Key:   "test",
			Flags: 42,
			Value: []byte("test"),
		},
	}
	var out bool
	if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	getR := structs.KeyRequest{
		Datacenter: "dc1",
		Key:        "test",
	}
	var dirent structs.IndexedDirEntries
	if err := msgpackrpc.CallWithCodec(codec, "KVS.Get", &getR, &dirent); err != nil {
		t.Fatalf("err: %v", err)
	}

	if dirent.Index == 0 {
		t.Fatalf("Bad: %v", dirent)
	}
	if len(dirent.Entries) != 1 {
		t.Fatalf("Bad: %v", dirent)
	}
	d := dirent.Entries[0]
	if d.Flags != 42 {
		t.Fatalf("bad: %v", d)
	}
	if string(d.Value) != "test" {
		t.Fatalf("bad: %v", d)
	}
}

func TestKVS_Get_ACLDeny(t *testing.T) {
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

	arg := structs.KVSRequest{
		Datacenter: "dc1",
		Op:         structs.KVSSet,
		DirEnt: structs.DirEntry{
			Key:   "zip",
			Flags: 42,
			Value: []byte("test"),
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var out bool
	if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	getR := structs.KeyRequest{
		Datacenter: "dc1",
		Key:        "zip",
	}
	var dirent structs.IndexedDirEntries
	if err := msgpackrpc.CallWithCodec(codec, "KVS.Get", &getR, &dirent); err != nil {
		t.Fatalf("err: %v", err)
	}

	if dirent.Index == 0 {
		t.Fatalf("Bad: %v", dirent)
	}
	if len(dirent.Entries) != 0 {
		t.Fatalf("Bad: %v", dirent)
	}
}

func TestKVSEndpoint_List(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	keys := []string{
		"/test/key1",
		"/test/key2",
		"/test/sub/key3",
	}

	for _, key := range keys {
		arg := structs.KVSRequest{
			Datacenter: "dc1",
			Op:         structs.KVSSet,
			DirEnt: structs.DirEntry{
				Key:   key,
				Flags: 1,
			},
		}
		var out bool
		if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	getR := structs.KeyRequest{
		Datacenter: "dc1",
		Key:        "/test",
	}
	var dirent structs.IndexedDirEntries
	if err := msgpackrpc.CallWithCodec(codec, "KVS.List", &getR, &dirent); err != nil {
		t.Fatalf("err: %v", err)
	}

	if dirent.Index == 0 {
		t.Fatalf("Bad: %v", dirent)
	}
	if len(dirent.Entries) != 3 {
		t.Fatalf("Bad: %v", dirent.Entries)
	}
	for i := 0; i < len(dirent.Entries); i++ {
		d := dirent.Entries[i]
		if d.Key != keys[i] {
			t.Fatalf("bad: %v", d)
		}
		if d.Flags != 1 {
			t.Fatalf("bad: %v", d)
		}
		if d.Value != nil {
			t.Fatalf("bad: %v", d)
		}
	}

	// Try listing a nonexistent prefix
	getR.Key = "/nope"
	if err := msgpackrpc.CallWithCodec(codec, "KVS.List", &getR, &dirent); err != nil {
		t.Fatalf("err: %v", err)
	}
	if dirent.Index == 0 {
		t.Fatalf("Bad: %v", dirent)
	}
	if len(dirent.Entries) != 0 {
		t.Fatalf("Bad: %v", dirent.Entries)
	}
}

func TestKVSEndpoint_List_Blocking(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	keys := []string{
		"/test/key1",
		"/test/key2",
		"/test/sub/key3",
	}

	for _, key := range keys {
		arg := structs.KVSRequest{
			Datacenter: "dc1",
			Op:         structs.KVSSet,
			DirEnt: structs.DirEntry{
				Key:   key,
				Flags: 1,
			},
		}
		var out bool
		if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	getR := structs.KeyRequest{
		Datacenter: "dc1",
		Key:        "/test",
	}
	var dirent structs.IndexedDirEntries
	if err := msgpackrpc.CallWithCodec(codec, "KVS.List", &getR, &dirent); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Setup a blocking query
	getR.MinQueryIndex = dirent.Index
	getR.MaxQueryTime = time.Second

	// Async cause a change
	start := time.Now()
	go func() {
		time.Sleep(100 * time.Millisecond)
		codec := rpcClient(t, s1)
		defer codec.Close()
		arg := structs.KVSRequest{
			Datacenter: "dc1",
			Op:         structs.KVSDelete,
			DirEnt: structs.DirEntry{
				Key: "/test/sub/key3",
			},
		}
		var out bool
		if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}()

	// Re-run the query
	dirent = structs.IndexedDirEntries{}
	if err := msgpackrpc.CallWithCodec(codec, "KVS.List", &getR, &dirent); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should block at least 100ms
	if time.Now().Sub(start) < 100*time.Millisecond {
		t.Fatalf("too fast")
	}

	if dirent.Index == 0 {
		t.Fatalf("Bad: %v", dirent)
	}
	if len(dirent.Entries) != 2 {
		for _, ent := range dirent.Entries {
			t.Errorf("Bad: %#v", *ent)
		}
	}
	for i := 0; i < len(dirent.Entries); i++ {
		d := dirent.Entries[i]
		if d.Key != keys[i] {
			t.Fatalf("bad: %v", d)
		}
		if d.Flags != 1 {
			t.Fatalf("bad: %v", d)
		}
		if d.Value != nil {
			t.Fatalf("bad: %v", d)
		}
	}
}

func TestKVSEndpoint_List_ACLDeny(t *testing.T) {
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

	keys := []string{
		"abe",
		"bar",
		"foo",
		"test",
		"zip",
	}

	for _, key := range keys {
		arg := structs.KVSRequest{
			Datacenter: "dc1",
			Op:         structs.KVSSet,
			DirEnt: structs.DirEntry{
				Key:   key,
				Flags: 1,
			},
			WriteRequest: structs.WriteRequest{Token: "root"},
		}
		var out bool
		if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name:  "User token",
			Type:  structs.ACLTypeClient,
			Rules: testListRules,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var out string
	if err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	id := out

	getR := structs.KeyRequest{
		Datacenter:   "dc1",
		Key:          "",
		QueryOptions: structs.QueryOptions{Token: id},
	}
	var dirent structs.IndexedDirEntries
	if err := msgpackrpc.CallWithCodec(codec, "KVS.List", &getR, &dirent); err != nil {
		t.Fatalf("err: %v", err)
	}

	if dirent.Index == 0 {
		t.Fatalf("Bad: %v", dirent)
	}
	if len(dirent.Entries) != 2 {
		t.Fatalf("Bad: %v", dirent.Entries)
	}
	for i := 0; i < len(dirent.Entries); i++ {
		d := dirent.Entries[i]
		switch i {
		case 0:
			if d.Key != "foo" {
				t.Fatalf("bad key")
			}
		case 1:
			if d.Key != "test" {
				t.Fatalf("bad key")
			}
		}
	}
}

func TestKVSEndpoint_ListKeys(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	keys := []string{
		"/test/key1",
		"/test/key2",
		"/test/sub/key3",
	}

	for _, key := range keys {
		arg := structs.KVSRequest{
			Datacenter: "dc1",
			Op:         structs.KVSSet,
			DirEnt: structs.DirEntry{
				Key:   key,
				Flags: 1,
			},
		}
		var out bool
		if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	getR := structs.KeyListRequest{
		Datacenter: "dc1",
		Prefix:     "/test/",
		Seperator:  "/",
	}
	var dirent structs.IndexedKeyList
	if err := msgpackrpc.CallWithCodec(codec, "KVS.ListKeys", &getR, &dirent); err != nil {
		t.Fatalf("err: %v", err)
	}

	if dirent.Index == 0 {
		t.Fatalf("Bad: %v", dirent)
	}
	if len(dirent.Keys) != 3 {
		t.Fatalf("Bad: %v", dirent.Keys)
	}
	if dirent.Keys[0] != "/test/key1" {
		t.Fatalf("Bad: %v", dirent.Keys)
	}
	if dirent.Keys[1] != "/test/key2" {
		t.Fatalf("Bad: %v", dirent.Keys)
	}
	if dirent.Keys[2] != "/test/sub/" {
		t.Fatalf("Bad: %v", dirent.Keys)
	}

	// Try listing a nonexistent prefix
	getR.Prefix = "/nope"
	if err := msgpackrpc.CallWithCodec(codec, "KVS.ListKeys", &getR, &dirent); err != nil {
		t.Fatalf("err: %v", err)
	}
	if dirent.Index == 0 {
		t.Fatalf("Bad: %v", dirent)
	}
	if len(dirent.Keys) != 0 {
		t.Fatalf("Bad: %v", dirent.Keys)
	}
}

func TestKVSEndpoint_ListKeys_ACLDeny(t *testing.T) {
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

	keys := []string{
		"abe",
		"bar",
		"foo",
		"test",
		"zip",
	}

	for _, key := range keys {
		arg := structs.KVSRequest{
			Datacenter: "dc1",
			Op:         structs.KVSSet,
			DirEnt: structs.DirEntry{
				Key:   key,
				Flags: 1,
			},
			WriteRequest: structs.WriteRequest{Token: "root"},
		}
		var out bool
		if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	arg := structs.ACLRequest{
		Datacenter: "dc1",
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Name:  "User token",
			Type:  structs.ACLTypeClient,
			Rules: testListRules,
		},
		WriteRequest: structs.WriteRequest{Token: "root"},
	}
	var out string
	if err := msgpackrpc.CallWithCodec(codec, "ACL.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	id := out

	getR := structs.KeyListRequest{
		Datacenter:   "dc1",
		Prefix:       "",
		Seperator:    "/",
		QueryOptions: structs.QueryOptions{Token: id},
	}
	var dirent structs.IndexedKeyList
	if err := msgpackrpc.CallWithCodec(codec, "KVS.ListKeys", &getR, &dirent); err != nil {
		t.Fatalf("err: %v", err)
	}

	if dirent.Index == 0 {
		t.Fatalf("Bad: %v", dirent)
	}
	if len(dirent.Keys) != 2 {
		t.Fatalf("Bad: %v", dirent.Keys)
	}
	if dirent.Keys[0] != "foo" {
		t.Fatalf("Bad: %v", dirent.Keys)
	}
	if dirent.Keys[1] != "test" {
		t.Fatalf("Bad: %v", dirent.Keys)
	}
}

func TestKVS_Apply_LockDelay(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Create and invalidate a session with a lock
	state := s1.fsm.State()
	if err := state.EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	session := &structs.Session{
		ID:        generateUUID(),
		Node:      "foo",
		LockDelay: 50 * time.Millisecond,
	}
	if err := state.SessionCreate(2, session); err != nil {
		t.Fatalf("err: %v", err)
	}
	id := session.ID
	d := &structs.DirEntry{
		Key:     "test",
		Session: id,
	}
	if ok, err := state.KVSLock(3, d); err != nil || !ok {
		t.Fatalf("err: %v", err)
	}
	if err := state.SessionDestroy(4, id); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Make a new session that is valid
	if err := state.SessionCreate(5, session); err != nil {
		t.Fatalf("err: %v", err)
	}
	validId := session.ID

	// Make a lock request
	arg := structs.KVSRequest{
		Datacenter: "dc1",
		Op:         structs.KVSLock,
		DirEnt: structs.DirEntry{
			Key:     "test",
			Session: validId,
		},
	}
	var out bool
	if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	if out != false {
		t.Fatalf("should not acquire")
	}

	// Wait for lock-delay
	time.Sleep(50 * time.Millisecond)

	// Should acquire
	if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	if out != true {
		t.Fatalf("should acquire")
	}
}

func TestKVS_Issue_1626(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Set up the first key.
	{
		arg := structs.KVSRequest{
			Datacenter: "dc1",
			Op:         structs.KVSSet,
			DirEnt: structs.DirEntry{
				Key:   "foo/test",
				Value: []byte("test"),
			},
		}
		var out bool
		if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Retrieve the base key and snag the index.
	var index uint64
	{
		getR := structs.KeyRequest{
			Datacenter: "dc1",
			Key:        "foo/test",
		}
		var dirent structs.IndexedDirEntries
		if err := msgpackrpc.CallWithCodec(codec, "KVS.Get", &getR, &dirent); err != nil {
			t.Fatalf("err: %v", err)
		}
		if dirent.Index == 0 {
			t.Fatalf("Bad: %v", dirent)
		}
		if len(dirent.Entries) != 1 {
			t.Fatalf("Bad: %v", dirent)
		}
		d := dirent.Entries[0]
		if string(d.Value) != "test" {
			t.Fatalf("bad: %v", d)
		}

		index = dirent.Index
	}

	// Set up a blocking query on the base key.
	doneCh := make(chan *structs.IndexedDirEntries, 1)
	go func() {
		codec := rpcClient(t, s1)
		defer codec.Close()

		getR := structs.KeyRequest{
			Datacenter: "dc1",
			Key:        "foo/test",
			QueryOptions: structs.QueryOptions{
				MinQueryIndex: index,
				MaxQueryTime:  3 * time.Second,
			},
		}
		var dirent structs.IndexedDirEntries
		if err := msgpackrpc.CallWithCodec(codec, "KVS.Get", &getR, &dirent); err != nil {
			t.Fatalf("err: %v", err)
		}
		doneCh <- &dirent
	}()

	// Now update a second key with a prefix that has the first key name
	// as part of it.
	{
		arg := structs.KVSRequest{
			Datacenter: "dc1",
			Op:         structs.KVSSet,
			DirEnt: structs.DirEntry{
				Key:   "foo/test2",
				Value: []byte("test"),
			},
		}
		var out bool
		if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Make sure the blocking query didn't wake up for this update.
	select {
	case <-doneCh:
		t.Fatalf("Blocking query should not have completed")
	case <-time.After(1 * time.Second):
	}

	// Now update the first key's payload.
	{
		arg := structs.KVSRequest{
			Datacenter: "dc1",
			Op:         structs.KVSSet,
			DirEnt: structs.DirEntry{
				Key:   "foo/test",
				Value: []byte("updated"),
			},
		}
		var out bool
		if err := msgpackrpc.CallWithCodec(codec, "KVS.Apply", &arg, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Make sure the blocking query wakes up for the final update.
	select {
	case dirent := <-doneCh:
		if dirent.Index <= index {
			t.Fatalf("Bad: %v", dirent)
		}
		if len(dirent.Entries) != 1 {
			t.Fatalf("Bad: %v", dirent)
		}
		d := dirent.Entries[0]
		if string(d.Value) != "updated" {
			t.Fatalf("bad: %v", d)
		}
	case <-time.After(1 * time.Second):
		t.Fatalf("Blocking query should have completed")
	}
}

var testListRules = `
key "" {
	policy = "deny"
}
key "foo" {
	policy = "read"
}
key "test" {
	policy = "write"
}
key "test/priv" {
	policy = "read"
}
`
