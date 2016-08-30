package consul

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/testutil"
	"github.com/hashicorp/net-rpc-msgpackrpc"
)

func TestInitializeSessionTimers(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	state := s1.fsm.State()
	if err := state.EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %s", err)
	}
	session := &structs.Session{
		ID:   generateUUID(),
		Node: "foo",
		TTL:  "10s",
	}
	if err := state.SessionCreate(100, session); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Reset the session timers
	err := s1.initializeSessionTimers()
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Check that we have a timer
	_, ok := s1.sessionTimers[session.ID]
	if !ok {
		t.Fatalf("missing session timer")
	}
}

func TestResetSessionTimer_Fault(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Should not exist
	err := s1.resetSessionTimer(generateUUID(), nil)
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("err: %v", err)
	}

	// Create a session
	state := s1.fsm.State()
	if err := state.EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %s", err)
	}
	session := &structs.Session{
		ID:   generateUUID(),
		Node: "foo",
		TTL:  "10s",
	}
	if err := state.SessionCreate(100, session); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Reset the session timer
	err = s1.resetSessionTimer(session.ID, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Check that we have a timer
	_, ok := s1.sessionTimers[session.ID]
	if !ok {
		t.Fatalf("missing session timer")
	}
}

func TestResetSessionTimer_NoTTL(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Create a session
	state := s1.fsm.State()
	if err := state.EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %s", err)
	}
	session := &structs.Session{
		ID:   generateUUID(),
		Node: "foo",
		TTL:  "0000s",
	}
	if err := state.SessionCreate(100, session); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Reset the session timer
	err := s1.resetSessionTimer(session.ID, session)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Check that we have a timer
	_, ok := s1.sessionTimers[session.ID]
	if ok {
		t.Fatalf("should not have session timer")
	}
}

func TestResetSessionTimer_InvalidTTL(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	// Create a session
	session := &structs.Session{
		ID:   generateUUID(),
		Node: "foo",
		TTL:  "foo",
	}

	// Reset the session timer
	err := s1.resetSessionTimer(session.ID, session)
	if err == nil || !strings.Contains(err.Error(), "Invalid Session TTL") {
		t.Fatalf("err: %v", err)
	}
}

func TestResetSessionTimerLocked(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	s1.sessionTimersLock.Lock()
	s1.resetSessionTimerLocked("foo", 5*time.Millisecond)
	s1.sessionTimersLock.Unlock()

	if _, ok := s1.sessionTimers["foo"]; !ok {
		t.Fatalf("missing timer")
	}

	time.Sleep(10 * time.Millisecond * structs.SessionTTLMultiplier)

	if _, ok := s1.sessionTimers["foo"]; ok {
		t.Fatalf("timer should be gone")
	}
}

func TestResetSessionTimerLocked_Renew(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	s1.sessionTimersLock.Lock()
	s1.resetSessionTimerLocked("foo", 5*time.Millisecond)
	s1.sessionTimersLock.Unlock()

	if _, ok := s1.sessionTimers["foo"]; !ok {
		t.Fatalf("missing timer")
	}

	time.Sleep(5 * time.Millisecond)

	// Renew the session
	s1.sessionTimersLock.Lock()
	renew := time.Now()
	s1.resetSessionTimerLocked("foo", 5*time.Millisecond)
	s1.sessionTimersLock.Unlock()

	// Watch for invalidation
	for time.Now().Sub(renew) < 20*time.Millisecond {
		s1.sessionTimersLock.Lock()
		_, ok := s1.sessionTimers["foo"]
		s1.sessionTimersLock.Unlock()
		if !ok {
			end := time.Now()
			if end.Sub(renew) < 5*time.Millisecond {
				t.Fatalf("early invalidate")
			}
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("should have expired")
}

func TestInvalidateSession(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	testutil.WaitForLeader(t, s1.RPC, "dc1")

	// Create a session
	state := s1.fsm.State()
	if err := state.EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %s", err)
	}
	session := &structs.Session{
		ID:   generateUUID(),
		Node: "foo",
		TTL:  "10s",
	}
	if err := state.SessionCreate(100, session); err != nil {
		t.Fatalf("err: %v", err)
	}

	// This should cause a destroy
	s1.invalidateSession(session.ID)

	// Check it is gone
	_, sess, err := state.SessionGet(session.ID)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if sess != nil {
		t.Fatalf("should destroy session")
	}
}

func TestClearSessionTimer(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	s1.sessionTimersLock.Lock()
	s1.resetSessionTimerLocked("foo", 5*time.Millisecond)
	s1.sessionTimersLock.Unlock()

	err := s1.clearSessionTimer("foo")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if _, ok := s1.sessionTimers["foo"]; ok {
		t.Fatalf("timer should be gone")
	}
}

func TestClearAllSessionTimers(t *testing.T) {
	dir1, s1 := testServer(t)
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()

	s1.sessionTimersLock.Lock()
	s1.resetSessionTimerLocked("foo", 10*time.Millisecond)
	s1.resetSessionTimerLocked("bar", 10*time.Millisecond)
	s1.resetSessionTimerLocked("baz", 10*time.Millisecond)
	s1.sessionTimersLock.Unlock()

	err := s1.clearAllSessionTimers()
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(s1.sessionTimers) != 0 {
		t.Fatalf("timers should be gone")
	}
}

func TestServer_SessionTTL_Failover(t *testing.T) {
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

	testutil.WaitForResult(func() (bool, error) {
		peers, _ := s1.raftPeers.Peers()
		return len(peers) == 3, nil
	}, func(err error) {
		t.Fatalf("should have 3 peers")
	})

	// Find the leader
	var leader *Server
	for _, s := range servers {
		// Check that s.sessionTimers is empty
		if len(s.sessionTimers) != 0 {
			t.Fatalf("should have no sessionTimers")
		}
		// Find the leader too
		if s.IsLeader() {
			leader = s
		}
	}
	if leader == nil {
		t.Fatalf("Should have a leader")
	}

	codec := rpcClient(t, leader)
	defer codec.Close()

	// Register a node
	node := structs.RegisterRequest{
		Datacenter: s1.config.Datacenter,
		Node:       "foo",
		Address:    "127.0.0.1",
	}
	var out struct{}
	if err := s1.RPC("Catalog.Register", &node, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Create a TTL session
	arg := structs.SessionRequest{
		Datacenter: "dc1",
		Op:         structs.SessionCreate,
		Session: structs.Session{
			Node: "foo",
			TTL:  "10s",
		},
	}
	var id1 string
	if err := msgpackrpc.CallWithCodec(codec, "Session.Apply", &arg, &id1); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Check that sessionTimers has the session ID
	if _, ok := leader.sessionTimers[id1]; !ok {
		t.Fatalf("missing session timer")
	}

	// Shutdown the leader!
	leader.Shutdown()

	// sessionTimers should be cleared on leader shutdown
	if len(leader.sessionTimers) != 0 {
		t.Fatalf("session timers should be empty on the shutdown leader")
	}

	// Find the new leader
	testutil.WaitForResult(func() (bool, error) {
		leader = nil
		for _, s := range servers {
			if s.IsLeader() {
				leader = s
			}
		}
		if leader == nil {
			return false, fmt.Errorf("Should have a new leader")
		}

		// Ensure session timer is restored
		if _, ok := leader.sessionTimers[id1]; !ok {
			return false, fmt.Errorf("missing session timer")
		}

		return true, nil
	}, func(err error) {
		t.Fatalf("err: %s", err)
	})
}
