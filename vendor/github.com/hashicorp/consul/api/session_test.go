package api

import (
	"testing"
	"time"
)

func TestSession_CreateDestroy(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	session := c.Session()

	id, meta, err := session.Create(nil, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if meta.RequestTime == 0 {
		t.Fatalf("bad: %v", meta)
	}

	if id == "" {
		t.Fatalf("invalid: %v", id)
	}

	meta, err = session.Destroy(id, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if meta.RequestTime == 0 {
		t.Fatalf("bad: %v", meta)
	}
}

func TestSession_CreateRenewDestroy(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	session := c.Session()

	se := &SessionEntry{
		TTL: "10s",
	}

	id, meta, err := session.Create(se, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer session.Destroy(id, nil)

	if meta.RequestTime == 0 {
		t.Fatalf("bad: %v", meta)
	}

	if id == "" {
		t.Fatalf("invalid: %v", id)
	}

	if meta.RequestTime == 0 {
		t.Fatalf("bad: %v", meta)
	}

	renew, meta, err := session.Renew(id, nil)

	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if meta.RequestTime == 0 {
		t.Fatalf("bad: %v", meta)
	}

	if renew == nil {
		t.Fatalf("should get session")
	}

	if renew.ID != id {
		t.Fatalf("should have matching id")
	}

	if renew.TTL != "10s" {
		t.Fatalf("should get session with TTL")
	}
}

func TestSession_CreateRenewDestroyRenew(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	session := c.Session()

	entry := &SessionEntry{
		Behavior: SessionBehaviorDelete,
		TTL:      "500s", // disable ttl
	}

	id, meta, err := session.Create(entry, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if meta.RequestTime == 0 {
		t.Fatalf("bad: %v", meta)
	}

	if id == "" {
		t.Fatalf("invalid: %v", id)
	}

	// Extend right after create. Everything should be fine.
	entry, _, err = session.Renew(id, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if entry == nil {
		t.Fatal("session unexpectedly vanished")
	}

	// Simulate TTL loss by manually destroying the session.
	meta, err = session.Destroy(id, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if meta.RequestTime == 0 {
		t.Fatalf("bad: %v", meta)
	}

	// Extend right after delete. The 404 should proxy as a nil.
	entry, _, err = session.Renew(id, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if entry != nil {
		t.Fatal("session still exists")
	}
}

func TestSession_CreateDestroyRenewPeriodic(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	session := c.Session()

	entry := &SessionEntry{
		Behavior: SessionBehaviorDelete,
		TTL:      "500s", // disable ttl
	}

	id, meta, err := session.Create(entry, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if meta.RequestTime == 0 {
		t.Fatalf("bad: %v", meta)
	}

	if id == "" {
		t.Fatalf("invalid: %v", id)
	}

	// This only tests Create/Destroy/RenewPeriodic to avoid the more
	// difficult case of testing all of the timing code.

	// Simulate TTL loss by manually destroying the session.
	meta, err = session.Destroy(id, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if meta.RequestTime == 0 {
		t.Fatalf("bad: %v", meta)
	}

	// Extend right after delete. The 404 should terminate the loop quickly and return ErrSessionExpired.
	errCh := make(chan error, 1)
	doneCh := make(chan struct{})
	go func() { errCh <- session.RenewPeriodic("1s", id, nil, doneCh) }()
	defer close(doneCh)

	select {
	case <-time.After(1 * time.Second):
		t.Fatal("timedout: missing session did not terminate renewal loop")
	case err = <-errCh:
		if err != ErrSessionExpired {
			t.Fatalf("err: %v", err)
		}
	}
}

func TestSession_Info(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	session := c.Session()

	id, _, err := session.Create(nil, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer session.Destroy(id, nil)

	info, qm, err := session.Info(id, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if qm.LastIndex == 0 {
		t.Fatalf("bad: %v", qm)
	}
	if !qm.KnownLeader {
		t.Fatalf("bad: %v", qm)
	}

	if info == nil {
		t.Fatalf("should get session")
	}
	if info.CreateIndex == 0 {
		t.Fatalf("bad: %v", info)
	}
	if info.ID != id {
		t.Fatalf("bad: %v", info)
	}
	if info.Name != "" {
		t.Fatalf("bad: %v", info)
	}
	if info.Node == "" {
		t.Fatalf("bad: %v", info)
	}
	if len(info.Checks) == 0 {
		t.Fatalf("bad: %v", info)
	}
	if info.LockDelay == 0 {
		t.Fatalf("bad: %v", info)
	}
	if info.Behavior != "release" {
		t.Fatalf("bad: %v", info)
	}
	if info.TTL != "" {
		t.Fatalf("bad: %v", info)
	}
}

func TestSession_Node(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	session := c.Session()

	id, _, err := session.Create(nil, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer session.Destroy(id, nil)

	info, qm, err := session.Info(id, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	sessions, qm, err := session.Node(info.Node, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(sessions) != 1 {
		t.Fatalf("bad: %v", sessions)
	}

	if qm.LastIndex == 0 {
		t.Fatalf("bad: %v", qm)
	}
	if !qm.KnownLeader {
		t.Fatalf("bad: %v", qm)
	}
}

func TestSession_List(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	session := c.Session()

	id, _, err := session.Create(nil, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	defer session.Destroy(id, nil)

	sessions, qm, err := session.List(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if len(sessions) != 1 {
		t.Fatalf("bad: %v", sessions)
	}

	if qm.LastIndex == 0 {
		t.Fatalf("bad: %v", qm)
	}
	if !qm.KnownLeader {
		t.Fatalf("bad: %v", qm)
	}
}
