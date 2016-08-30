package consul

import (
	"os"
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/lib"
	"github.com/hashicorp/consul/testutil"
	"github.com/hashicorp/net-rpc-msgpackrpc"
)

func TestSessionEndpoint_Apply(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Just add a node
	s1.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"})

	arg := structs.SessionRequest{
		Datacenter: "dc1",
		Op:         structs.SessionCreate,
		Session: structs.Session{
			Node: "foo",
			Name: "my-session",
		},
	}
	var out string
	if err := msgpackrpc.CallWithCodec(codec, "Session.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	id := out

	// Verify
	state := s1.fsm.State()
	_, s, err := state.SessionGet(out)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s == nil {
		t.Fatalf("should not be nil")
	}
	if s.Node != "foo" {
		t.Fatalf("bad: %v", s)
	}
	if s.Name != "my-session" {
		t.Fatalf("bad: %v", s)
	}

	// Do a delete
	arg.Op = structs.SessionDestroy
	arg.Session.ID = out
	if err := msgpackrpc.CallWithCodec(codec, "Session.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Verify
	_, s, err = state.SessionGet(id)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s != nil {
		t.Fatalf("bad: %v", s)
	}
}

func TestSessionEndpoint_DeleteApply(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Just add a node
	s1.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"})

	arg := structs.SessionRequest{
		Datacenter: "dc1",
		Op:         structs.SessionCreate,
		Session: structs.Session{
			Node:     "foo",
			Name:     "my-session",
			Behavior: structs.SessionKeysDelete,
		},
	}
	var out string
	if err := msgpackrpc.CallWithCodec(codec, "Session.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	id := out

	// Verify
	state := s1.fsm.State()
	_, s, err := state.SessionGet(out)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s == nil {
		t.Fatalf("should not be nil")
	}
	if s.Node != "foo" {
		t.Fatalf("bad: %v", s)
	}
	if s.Name != "my-session" {
		t.Fatalf("bad: %v", s)
	}
	if s.Behavior != structs.SessionKeysDelete {
		t.Fatalf("bad: %v", s)
	}

	// Do a delete
	arg.Op = structs.SessionDestroy
	arg.Session.ID = out
	if err := msgpackrpc.CallWithCodec(codec, "Session.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Verify
	_, s, err = state.SessionGet(id)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s != nil {
		t.Fatalf("bad: %v", s)
	}
}

func TestSessionEndpoint_Get(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	s1.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"})
	arg := structs.SessionRequest{
		Datacenter: "dc1",
		Op:         structs.SessionCreate,
		Session: structs.Session{
			Node: "foo",
		},
	}
	var out string
	if err := msgpackrpc.CallWithCodec(codec, "Session.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	getR := structs.SessionSpecificRequest{
		Datacenter: "dc1",
		Session:    out,
	}
	var sessions structs.IndexedSessions
	if err := msgpackrpc.CallWithCodec(codec, "Session.Get", &getR, &sessions); err != nil {
		t.Fatalf("err: %v", err)
	}

	if sessions.Index == 0 {
		t.Fatalf("Bad: %v", sessions)
	}
	if len(sessions.Sessions) != 1 {
		t.Fatalf("Bad: %v", sessions)
	}
	s := sessions.Sessions[0]
	if s.ID != out {
		t.Fatalf("bad: %v", s)
	}
}

func TestSessionEndpoint_List(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	s1.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"})
	ids := []string{}
	for i := 0; i < 5; i++ {
		arg := structs.SessionRequest{
			Datacenter: "dc1",
			Op:         structs.SessionCreate,
			Session: structs.Session{
				Node: "foo",
			},
		}
		var out string
		if err := msgpackrpc.CallWithCodec(codec, "Session.Apply", &arg, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
		ids = append(ids, out)
	}

	getR := structs.DCSpecificRequest{
		Datacenter: "dc1",
	}
	var sessions structs.IndexedSessions
	if err := msgpackrpc.CallWithCodec(codec, "Session.List", &getR, &sessions); err != nil {
		t.Fatalf("err: %v", err)
	}

	if sessions.Index == 0 {
		t.Fatalf("Bad: %v", sessions)
	}
	if len(sessions.Sessions) != 5 {
		t.Fatalf("Bad: %v", sessions.Sessions)
	}
	for i := 0; i < len(sessions.Sessions); i++ {
		s := sessions.Sessions[i]
		if !lib.StrContains(ids, s.ID) {
			t.Fatalf("bad: %v", s)
		}
		if s.Node != "foo" {
			t.Fatalf("bad: %v", s)
		}
	}
}

func TestSessionEndpoint_ApplyTimers(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	s1.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"})
	arg := structs.SessionRequest{
		Datacenter: "dc1",
		Op:         structs.SessionCreate,
		Session: structs.Session{
			Node: "foo",
			TTL:  "10s",
		},
	}
	var out string
	if err := msgpackrpc.CallWithCodec(codec, "Session.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Check the session map
	if _, ok := s1.sessionTimers[out]; !ok {
		t.Fatalf("missing session timer")
	}

	// Destroy the session
	arg.Op = structs.SessionDestroy
	arg.Session.ID = out
	if err := msgpackrpc.CallWithCodec(codec, "Session.Apply", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Check the session map
	if _, ok := s1.sessionTimers[out]; ok {
		t.Fatalf("session timer exists")
	}
}

func TestSessionEndpoint_Renew(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")
	TTL := "10s" // the minimum allowed ttl
	ttl := 10 * time.Second

	s1.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"})
	ids := []string{}
	for i := 0; i < 5; i++ {
		arg := structs.SessionRequest{
			Datacenter: "dc1",
			Op:         structs.SessionCreate,
			Session: structs.Session{
				Node: "foo",
				TTL:  TTL,
			},
		}
		var out string
		if err := msgpackrpc.CallWithCodec(codec, "Session.Apply", &arg, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
		ids = append(ids, out)
	}

	// Verify the timer map is setup
	if len(s1.sessionTimers) != 5 {
		t.Fatalf("missing session timers")
	}

	getR := structs.DCSpecificRequest{
		Datacenter: "dc1",
	}

	var sessions structs.IndexedSessions
	if err := msgpackrpc.CallWithCodec(codec, "Session.List", &getR, &sessions); err != nil {
		t.Fatalf("err: %v", err)
	}

	if sessions.Index == 0 {
		t.Fatalf("Bad: %v", sessions)
	}
	if len(sessions.Sessions) != 5 {
		t.Fatalf("Bad: %v", sessions.Sessions)
	}
	for i := 0; i < len(sessions.Sessions); i++ {
		s := sessions.Sessions[i]
		if !lib.StrContains(ids, s.ID) {
			t.Fatalf("bad: %v", s)
		}
		if s.Node != "foo" {
			t.Fatalf("bad: %v", s)
		}
		if s.TTL != TTL {
			t.Fatalf("bad session TTL: %s %v", s.TTL, s)
		}
		t.Logf("Created session '%s'", s.ID)
	}

	// Sleep for time shorter than internal destroy ttl
	time.Sleep(ttl * structs.SessionTTLMultiplier / 2)

	// renew 3 out of 5 sessions
	for i := 0; i < 3; i++ {
		renewR := structs.SessionSpecificRequest{
			Datacenter: "dc1",
			Session:    ids[i],
		}
		var session structs.IndexedSessions
		if err := msgpackrpc.CallWithCodec(codec, "Session.Renew", &renewR, &session); err != nil {
			t.Fatalf("err: %v", err)
		}

		if session.Index == 0 {
			t.Fatalf("Bad: %v", session)
		}
		if len(session.Sessions) != 1 {
			t.Fatalf("Bad: %v", session.Sessions)
		}

		s := session.Sessions[0]
		if !lib.StrContains(ids, s.ID) {
			t.Fatalf("bad: %v", s)
		}
		if s.Node != "foo" {
			t.Fatalf("bad: %v", s)
		}

		t.Logf("Renewed session '%s'", s.ID)
	}

	// now sleep for 2/3 the internal destroy TTL time for renewed sessions
	// which is more than the internal destroy TTL time for the non-renewed sessions
	time.Sleep((ttl * structs.SessionTTLMultiplier) * 2.0 / 3.0)

	var sessionsL1 structs.IndexedSessions
	if err := msgpackrpc.CallWithCodec(codec, "Session.List", &getR, &sessionsL1); err != nil {
		t.Fatalf("err: %v", err)
	}

	if sessionsL1.Index == 0 {
		t.Fatalf("Bad: %v", sessionsL1)
	}

	t.Logf("Expect 2 sessions to be destroyed")

	for i := 0; i < len(sessionsL1.Sessions); i++ {
		s := sessionsL1.Sessions[i]
		if !lib.StrContains(ids, s.ID) {
			t.Fatalf("bad: %v", s)
		}
		if s.Node != "foo" {
			t.Fatalf("bad: %v", s)
		}
		if s.TTL != TTL {
			t.Fatalf("bad: %v", s)
		}
		if i > 2 {
			t.Errorf("session '%s' should be destroyed", s.ID)
		}
	}

	if len(sessionsL1.Sessions) != 3 {
		t.Fatalf("Bad: %v", sessionsL1.Sessions)
	}

	// now sleep again for ttl*2 - no sessions should still be alive
	time.Sleep(ttl * structs.SessionTTLMultiplier)

	var sessionsL2 structs.IndexedSessions
	if err := msgpackrpc.CallWithCodec(codec, "Session.List", &getR, &sessionsL2); err != nil {
		t.Fatalf("err: %v", err)
	}

	if sessionsL2.Index == 0 {
		t.Fatalf("Bad: %v", sessionsL2)
	}
	if len(sessionsL2.Sessions) != 0 {
		for i := 0; i < len(sessionsL2.Sessions); i++ {
			s := sessionsL2.Sessions[i]
			if !lib.StrContains(ids, s.ID) {
				t.Fatalf("bad: %v", s)
			}
			if s.Node != "foo" {
				t.Fatalf("bad: %v", s)
			}
			if s.TTL != TTL {
				t.Fatalf("bad: %v", s)
			}
			t.Errorf("session '%s' should be destroyed", s.ID)
		}

		t.Fatalf("Bad: %v", sessionsL2.Sessions)
	}
}

func TestSessionEndpoint_NodeSessions(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	s1.fsm.State().EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"})
	s1.fsm.State().EnsureNode(1, &structs.Node{Node: "bar", Address: "127.0.0.1"})
	ids := []string{}
	for i := 0; i < 10; i++ {
		arg := structs.SessionRequest{
			Datacenter: "dc1",
			Op:         structs.SessionCreate,
			Session: structs.Session{
				Node: "bar",
			},
		}
		if i < 5 {
			arg.Session.Node = "foo"
		}
		var out string
		if err := msgpackrpc.CallWithCodec(codec, "Session.Apply", &arg, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
		if i < 5 {
			ids = append(ids, out)
		}
	}

	getR := structs.NodeSpecificRequest{
		Datacenter: "dc1",
		Node:       "foo",
	}
	var sessions structs.IndexedSessions
	if err := msgpackrpc.CallWithCodec(codec, "Session.NodeSessions", &getR, &sessions); err != nil {
		t.Fatalf("err: %v", err)
	}

	if sessions.Index == 0 {
		t.Fatalf("Bad: %v", sessions)
	}
	if len(sessions.Sessions) != 5 {
		t.Fatalf("Bad: %v", sessions.Sessions)
	}
	for i := 0; i < len(sessions.Sessions); i++ {
		s := sessions.Sessions[i]
		if !lib.StrContains(ids, s.ID) {
			t.Fatalf("bad: %v", s)
		}
		if s.Node != "foo" {
			t.Fatalf("bad: %v", s)
		}
	}
}

func TestSessionEndpoint_Apply_BadTTL(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec := rpcClient(t, s1)
	defer codec.Close()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	arg := structs.SessionRequest{
		Datacenter: "dc1",
		Op:         structs.SessionCreate,
		Session: structs.Session{
			Node: "foo",
			Name: "my-session",
		},
	}

	// Session with illegal TTL
	arg.Session.TTL = "10z"

	var out string
	err := msgpackrpc.CallWithCodec(codec, "Session.Apply", &arg, &out)
	if err == nil {
		t.Fatal("expected error")
	}
	if err.Error() != "Session TTL '10z' invalid: time: unknown unit z in duration 10z" {
		t.Fatalf("incorrect error message: %s", err.Error())
	}

	// less than SessionTTLMin
	arg.Session.TTL = "5s"

	err = msgpackrpc.CallWithCodec(codec, "Session.Apply", &arg, &out)
	if err == nil {
		t.Fatal("expected error")
	}
	if err.Error() != "Invalid Session TTL '5000000000', must be between [10s=24h0m0s]" {
		t.Fatalf("incorrect error message: %s", err.Error())
	}

	// more than SessionTTLMax
	arg.Session.TTL = "100000s"

	err = msgpackrpc.CallWithCodec(codec, "Session.Apply", &arg, &out)
	if err == nil {
		t.Fatal("expected error")
	}
	if err.Error() != "Invalid Session TTL '100000000000000', must be between [10s=24h0m0s]" {
		t.Fatalf("incorrect error message: %s", err.Error())
	}
}
