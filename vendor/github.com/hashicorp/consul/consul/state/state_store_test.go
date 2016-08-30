package state

import (
	crand "crypto/rand"
	"fmt"
	"math/rand"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/lib"
	"github.com/hashicorp/serf/coordinate"
)

func testUUID() string {
	buf := make([]byte, 16)
	if _, err := crand.Read(buf); err != nil {
		panic(fmt.Errorf("failed to read random bytes: %v", err))
	}

	return fmt.Sprintf("%08x-%04x-%04x-%04x-%12x",
		buf[0:4],
		buf[4:6],
		buf[6:8],
		buf[8:10],
		buf[10:16])
}

func testStateStore(t *testing.T) *StateStore {
	s, err := NewStateStore(nil)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if s == nil {
		t.Fatalf("missing state store")
	}
	return s
}

func testRegisterNode(t *testing.T, s *StateStore, idx uint64, nodeID string) {
	node := &structs.Node{Node: nodeID}
	if err := s.EnsureNode(idx, node); err != nil {
		t.Fatalf("err: %s", err)
	}

	tx := s.db.Txn(false)
	defer tx.Abort()
	n, err := tx.First("nodes", "id", nodeID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result, ok := n.(*structs.Node); !ok || result.Node != nodeID {
		t.Fatalf("bad node: %#v", result)
	}
}

func testRegisterService(t *testing.T, s *StateStore, idx uint64, nodeID, serviceID string) {
	svc := &structs.NodeService{
		ID:      serviceID,
		Service: serviceID,
		Address: "1.1.1.1",
		Port:    1111,
	}
	if err := s.EnsureService(idx, nodeID, svc); err != nil {
		t.Fatalf("err: %s", err)
	}

	tx := s.db.Txn(false)
	defer tx.Abort()
	service, err := tx.First("services", "id", nodeID, serviceID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result, ok := service.(*structs.ServiceNode); !ok ||
		result.Node != nodeID ||
		result.ServiceID != serviceID {
		t.Fatalf("bad service: %#v", result)
	}
}

func testRegisterCheck(t *testing.T, s *StateStore, idx uint64,
	nodeID, serviceID, checkID, state string) {
	chk := &structs.HealthCheck{
		Node:      nodeID,
		CheckID:   checkID,
		ServiceID: serviceID,
		Status:    state,
	}
	if err := s.EnsureCheck(idx, chk); err != nil {
		t.Fatalf("err: %s", err)
	}

	tx := s.db.Txn(false)
	defer tx.Abort()
	c, err := tx.First("checks", "id", nodeID, checkID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result, ok := c.(*structs.HealthCheck); !ok ||
		result.Node != nodeID ||
		result.ServiceID != serviceID ||
		result.CheckID != checkID {
		t.Fatalf("bad check: %#v", result)
	}
}

func testSetKey(t *testing.T, s *StateStore, idx uint64, key, value string) {
	entry := &structs.DirEntry{Key: key, Value: []byte(value)}
	if err := s.KVSSet(idx, entry); err != nil {
		t.Fatalf("err: %s", err)
	}

	tx := s.db.Txn(false)
	defer tx.Abort()
	e, err := tx.First("kvs", "id", key)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result, ok := e.(*structs.DirEntry); !ok || result.Key != key {
		t.Fatalf("bad kvs entry: %#v", result)
	}
}

func TestStateStore_Restore_Abort(t *testing.T) {
	s := testStateStore(t)

	// The detailed restore functions are tested below, this just checks
	// that abort works.
	restore := s.Restore()
	entry := &structs.DirEntry{
		Key:   "foo",
		Value: []byte("bar"),
		RaftIndex: structs.RaftIndex{
			ModifyIndex: 5,
		},
	}
	if err := restore.KVS(entry); err != nil {
		t.Fatalf("err: %s", err)
	}
	restore.Abort()

	idx, entries, err := s.KVSList("")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}
	if len(entries) != 0 {
		t.Fatalf("bad: %#v", entries)
	}
}

func TestStateStore_maxIndex(t *testing.T) {
	s := testStateStore(t)

	testRegisterNode(t, s, 0, "foo")
	testRegisterNode(t, s, 1, "bar")
	testRegisterService(t, s, 2, "foo", "consul")

	if max := s.maxIndex("nodes", "services"); max != 2 {
		t.Fatalf("bad max: %d", max)
	}
}

func TestStateStore_indexUpdateMaxTxn(t *testing.T) {
	s := testStateStore(t)

	testRegisterNode(t, s, 0, "foo")
	testRegisterNode(t, s, 1, "bar")

	tx := s.db.Txn(true)
	if err := indexUpdateMaxTxn(tx, 3, "nodes"); err != nil {
		t.Fatalf("err: %s", err)
	}
	tx.Commit()

	if max := s.maxIndex("nodes"); max != 3 {
		t.Fatalf("bad max: %d", max)
	}
}

func TestStateStore_GC(t *testing.T) {
	// Build up a fast GC.
	ttl := 10 * time.Millisecond
	gran := 5 * time.Millisecond
	gc, err := NewTombstoneGC(ttl, gran)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	// Enable it and attach it to the state store.
	gc.SetEnabled(true)
	s, err := NewStateStore(gc)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	// Create some KV pairs.
	testSetKey(t, s, 1, "foo", "foo")
	testSetKey(t, s, 2, "foo/bar", "bar")
	testSetKey(t, s, 3, "foo/baz", "bar")
	testSetKey(t, s, 4, "foo/moo", "bar")
	testSetKey(t, s, 5, "foo/zoo", "bar")

	// Delete a key and make sure the GC sees it.
	if err := s.KVSDelete(6, "foo/zoo"); err != nil {
		t.Fatalf("err: %s", err)
	}
	select {
	case idx := <-gc.ExpireCh():
		if idx != 6 {
			t.Fatalf("bad index: %d", idx)
		}
	case <-time.After(2 * ttl):
		t.Fatalf("GC never fired")
	}

	// Check for the same behavior with a tree delete.
	if err := s.KVSDeleteTree(7, "foo/moo"); err != nil {
		t.Fatalf("err: %s", err)
	}
	select {
	case idx := <-gc.ExpireCh():
		if idx != 7 {
			t.Fatalf("bad index: %d", idx)
		}
	case <-time.After(2 * ttl):
		t.Fatalf("GC never fired")
	}

	// Check for the same behavior with a CAS delete.
	if ok, err := s.KVSDeleteCAS(8, 3, "foo/baz"); !ok || err != nil {
		t.Fatalf("err: %s", err)
	}
	select {
	case idx := <-gc.ExpireCh():
		if idx != 8 {
			t.Fatalf("bad index: %d", idx)
		}
	case <-time.After(2 * ttl):
		t.Fatalf("GC never fired")
	}

	// Finally, try it with an expiring session.
	testRegisterNode(t, s, 9, "node1")
	session := &structs.Session{
		ID:       testUUID(),
		Node:     "node1",
		Behavior: structs.SessionKeysDelete,
	}
	if err := s.SessionCreate(10, session); err != nil {
		t.Fatalf("err: %s", err)
	}
	d := &structs.DirEntry{
		Key:     "lock",
		Session: session.ID,
	}
	if ok, err := s.KVSLock(11, d); !ok || err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := s.SessionDestroy(12, session.ID); err != nil {
		t.Fatalf("err: %s", err)
	}
	select {
	case idx := <-gc.ExpireCh():
		if idx != 12 {
			t.Fatalf("bad index: %d", idx)
		}
	case <-time.After(2 * ttl):
		t.Fatalf("GC never fired")
	}
}

func TestStateStore_ReapTombstones(t *testing.T) {
	s := testStateStore(t)

	// Create some KV pairs.
	testSetKey(t, s, 1, "foo", "foo")
	testSetKey(t, s, 2, "foo/bar", "bar")
	testSetKey(t, s, 3, "foo/baz", "bar")
	testSetKey(t, s, 4, "foo/moo", "bar")
	testSetKey(t, s, 5, "foo/zoo", "bar")

	// Call a delete on some specific keys.
	if err := s.KVSDelete(6, "foo/baz"); err != nil {
		t.Fatalf("err: %s", err)
	}
	if err := s.KVSDelete(7, "foo/moo"); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Pull out the list and check the index, which should come from the
	// tombstones.
	idx, _, err := s.KVSList("foo/")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 7 {
		t.Fatalf("bad index: %d", idx)
	}

	// Reap the tombstones <= 6.
	if err := s.ReapTombstones(6); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Should still be good because 7 is in there.
	idx, _, err = s.KVSList("foo/")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 7 {
		t.Fatalf("bad index: %d", idx)
	}

	// Now reap them all.
	if err := s.ReapTombstones(7); err != nil {
		t.Fatalf("err: %s", err)
	}

	// At this point the sub index will slide backwards.
	idx, _, err = s.KVSList("foo/")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}

	// Make sure the tombstones are actually gone.
	snap := s.Snapshot()
	defer snap.Close()
	stones, err := snap.Tombstones()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if stones.Next() != nil {
		t.Fatalf("unexpected extra tombstones")
	}
}

func TestStateStore_GetWatches(t *testing.T) {
	s := testStateStore(t)

	// This test does two things - it makes sure there's no full table
	// watch for KVS, and it makes sure that asking for a watch that
	// doesn't exist causes a panic.
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatalf("didn't get expected panic")
			}
		}()
		s.getTableWatch("kvs")
	}()

	// Similar for tombstones; those don't support watches at all.
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatalf("didn't get expected panic")
			}
		}()
		s.getTableWatch("tombstones")
	}()

	// Make sure requesting a bogus method causes a panic.
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatalf("didn't get expected panic")
			}
		}()
		s.GetQueryWatch("dogs")
	}()

	// Request valid watches.
	if w := s.GetQueryWatch("Nodes"); w == nil {
		t.Fatalf("didn't get a watch")
	}
	if w := s.GetQueryWatch("NodeDump"); w == nil {
		t.Fatalf("didn't get a watch")
	}
	if w := s.GetKVSWatch("/dogs"); w == nil {
		t.Fatalf("didn't get a watch")
	}
}

func TestStateStore_EnsureRegistration(t *testing.T) {
	s := testStateStore(t)

	// Start with just a node.
	req := &structs.RegisterRequest{
		Node:    "node1",
		Address: "1.2.3.4",
		TaggedAddresses: map[string]string{
			"hello": "world",
		},
	}
	if err := s.EnsureRegistration(1, req); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Retrieve the node and verify its contents.
	verifyNode := func(created, modified uint64) {
		_, out, err := s.GetNode("node1")
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if out.Node != "node1" || out.Address != "1.2.3.4" ||
			len(out.TaggedAddresses) != 1 ||
			out.TaggedAddresses["hello"] != "world" ||
			out.CreateIndex != created || out.ModifyIndex != modified {
			t.Fatalf("bad node returned: %#v", out)
		}
	}
	verifyNode(1, 1)

	// Add in a service definition.
	req.Service = &structs.NodeService{
		ID:      "redis1",
		Service: "redis",
		Address: "1.1.1.1",
		Port:    8080,
	}
	if err := s.EnsureRegistration(2, req); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Verify that the service got registered.
	verifyService := func(created, modified uint64) {
		idx, out, err := s.NodeServices("node1")
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != modified {
			t.Fatalf("bad index: %d", idx)
		}
		if len(out.Services) != 1 {
			t.Fatalf("bad: %#v", out.Services)
		}
		s := out.Services["redis1"]
		if s.ID != "redis1" || s.Service != "redis" ||
			s.Address != "1.1.1.1" || s.Port != 8080 ||
			s.CreateIndex != created || s.ModifyIndex != modified {
			t.Fatalf("bad service returned: %#v", s)
		}
	}
	verifyNode(1, 2)
	verifyService(2, 2)

	// Add in a top-level check.
	req.Check = &structs.HealthCheck{
		Node:    "node1",
		CheckID: "check1",
		Name:    "check",
	}
	if err := s.EnsureRegistration(3, req); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Verify that the check got registered.
	verifyCheck := func(created, modified uint64) {
		idx, out, err := s.NodeChecks("node1")
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != modified {
			t.Fatalf("bad index: %d", idx)
		}
		if len(out) != 1 {
			t.Fatalf("bad: %#v", out)
		}
		c := out[0]
		if c.Node != "node1" || c.CheckID != "check1" || c.Name != "check" ||
			c.CreateIndex != created || c.ModifyIndex != modified {
			t.Fatalf("bad check returned: %#v", c)
		}
	}
	verifyNode(1, 3)
	verifyService(2, 3)
	verifyCheck(3, 3)

	// Add in another check via the slice.
	req.Checks = structs.HealthChecks{
		&structs.HealthCheck{
			Node:    "node1",
			CheckID: "check2",
			Name:    "check",
		},
	}
	if err := s.EnsureRegistration(4, req); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Verify that the additional check got registered.
	verifyNode(1, 4)
	verifyService(2, 4)
	func() {
		idx, out, err := s.NodeChecks("node1")
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 4 {
			t.Fatalf("bad index: %d", idx)
		}
		if len(out) != 2 {
			t.Fatalf("bad: %#v", out)
		}
		c1 := out[0]
		if c1.Node != "node1" || c1.CheckID != "check1" || c1.Name != "check" ||
			c1.CreateIndex != 3 || c1.ModifyIndex != 4 {
			t.Fatalf("bad check returned: %#v", c1)
		}

		c2 := out[1]
		if c2.Node != "node1" || c2.CheckID != "check2" || c2.Name != "check" ||
			c2.CreateIndex != 4 || c2.ModifyIndex != 4 {
			t.Fatalf("bad check returned: %#v", c2)
		}
	}()
}

func TestStateStore_EnsureRegistration_Restore(t *testing.T) {
	s := testStateStore(t)

	// Start with just a node.
	req := &structs.RegisterRequest{
		Node:    "node1",
		Address: "1.2.3.4",
	}
	restore := s.Restore()
	if err := restore.Registration(1, req); err != nil {
		t.Fatalf("err: %s", err)
	}
	restore.Commit()

	// Retrieve the node and verify its contents.
	verifyNode := func(created, modified uint64) {
		_, out, err := s.GetNode("node1")
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if out.Node != "node1" || out.Address != "1.2.3.4" ||
			out.CreateIndex != created || out.ModifyIndex != modified {
			t.Fatalf("bad node returned: %#v", out)
		}
	}
	verifyNode(1, 1)

	// Add in a service definition.
	req.Service = &structs.NodeService{
		ID:      "redis1",
		Service: "redis",
		Address: "1.1.1.1",
		Port:    8080,
	}
	restore = s.Restore()
	if err := restore.Registration(2, req); err != nil {
		t.Fatalf("err: %s", err)
	}
	restore.Commit()

	// Verify that the service got registered.
	verifyService := func(created, modified uint64) {
		idx, out, err := s.NodeServices("node1")
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != modified {
			t.Fatalf("bad index: %d", idx)
		}
		if len(out.Services) != 1 {
			t.Fatalf("bad: %#v", out.Services)
		}
		s := out.Services["redis1"]
		if s.ID != "redis1" || s.Service != "redis" ||
			s.Address != "1.1.1.1" || s.Port != 8080 ||
			s.CreateIndex != created || s.ModifyIndex != modified {
			t.Fatalf("bad service returned: %#v", s)
		}
	}
	verifyNode(1, 2)
	verifyService(2, 2)

	// Add in a top-level check.
	req.Check = &structs.HealthCheck{
		Node:    "node1",
		CheckID: "check1",
		Name:    "check",
	}
	restore = s.Restore()
	if err := restore.Registration(3, req); err != nil {
		t.Fatalf("err: %s", err)
	}
	restore.Commit()

	// Verify that the check got registered.
	verifyCheck := func(created, modified uint64) {
		idx, out, err := s.NodeChecks("node1")
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != modified {
			t.Fatalf("bad index: %d", idx)
		}
		if len(out) != 1 {
			t.Fatalf("bad: %#v", out)
		}
		c := out[0]
		if c.Node != "node1" || c.CheckID != "check1" || c.Name != "check" ||
			c.CreateIndex != created || c.ModifyIndex != modified {
			t.Fatalf("bad check returned: %#v", c)
		}
	}
	verifyNode(1, 3)
	verifyService(2, 3)
	verifyCheck(3, 3)

	// Add in another check via the slice.
	req.Checks = structs.HealthChecks{
		&structs.HealthCheck{
			Node:    "node1",
			CheckID: "check2",
			Name:    "check",
		},
	}
	restore = s.Restore()
	if err := restore.Registration(4, req); err != nil {
		t.Fatalf("err: %s", err)
	}
	restore.Commit()

	// Verify that the additional check got registered.
	verifyNode(1, 4)
	verifyService(2, 4)
	func() {
		idx, out, err := s.NodeChecks("node1")
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 4 {
			t.Fatalf("bad index: %d", idx)
		}
		if len(out) != 2 {
			t.Fatalf("bad: %#v", out)
		}
		c1 := out[0]
		if c1.Node != "node1" || c1.CheckID != "check1" || c1.Name != "check" ||
			c1.CreateIndex != 3 || c1.ModifyIndex != 4 {
			t.Fatalf("bad check returned: %#v", c1)
		}

		c2 := out[1]
		if c2.Node != "node1" || c2.CheckID != "check2" || c2.Name != "check" ||
			c2.CreateIndex != 4 || c2.ModifyIndex != 4 {
			t.Fatalf("bad check returned: %#v", c2)
		}
	}()
}

func TestStateStore_EnsureRegistration_Watches(t *testing.T) {
	s := testStateStore(t)

	req := &structs.RegisterRequest{
		Node:    "node1",
		Address: "1.2.3.4",
	}

	// The nodes watch should fire for this one.
	verifyWatch(t, s.getTableWatch("nodes"), func() {
		verifyNoWatch(t, s.getTableWatch("services"), func() {
			verifyNoWatch(t, s.getTableWatch("checks"), func() {
				if err := s.EnsureRegistration(1, req); err != nil {
					t.Fatalf("err: %s", err)
				}
			})
		})
	})
	// The nodes watch should fire for this one.
	verifyWatch(t, s.getTableWatch("nodes"), func() {
		verifyNoWatch(t, s.getTableWatch("services"), func() {
			verifyNoWatch(t, s.getTableWatch("checks"), func() {
				restore := s.Restore()
				if err := restore.Registration(1, req); err != nil {
					t.Fatalf("err: %s", err)
				}
				restore.Commit()
			})
		})
	})

	// With a service definition added it should fire nodes and
	// services.
	req.Service = &structs.NodeService{
		ID:      "redis1",
		Service: "redis",
		Address: "1.1.1.1",
		Port:    8080,
	}
	verifyWatch(t, s.getTableWatch("nodes"), func() {
		verifyWatch(t, s.getTableWatch("services"), func() {
			verifyNoWatch(t, s.getTableWatch("checks"), func() {
				if err := s.EnsureRegistration(2, req); err != nil {
					t.Fatalf("err: %s", err)
				}
			})
		})
	})
	verifyWatch(t, s.getTableWatch("nodes"), func() {
		verifyWatch(t, s.getTableWatch("services"), func() {
			verifyNoWatch(t, s.getTableWatch("checks"), func() {
				restore := s.Restore()
				if err := restore.Registration(2, req); err != nil {
					t.Fatalf("err: %s", err)
				}
				restore.Commit()
			})
		})
	})

	// Now with a check it should hit all three.
	req.Check = &structs.HealthCheck{
		Node:    "node1",
		CheckID: "check1",
		Name:    "check",
	}
	verifyWatch(t, s.getTableWatch("nodes"), func() {
		verifyWatch(t, s.getTableWatch("services"), func() {
			verifyWatch(t, s.getTableWatch("checks"), func() {
				if err := s.EnsureRegistration(3, req); err != nil {
					t.Fatalf("err: %s", err)
				}
			})
		})
	})
	verifyWatch(t, s.getTableWatch("nodes"), func() {
		verifyWatch(t, s.getTableWatch("services"), func() {
			verifyWatch(t, s.getTableWatch("checks"), func() {
				restore := s.Restore()
				if err := restore.Registration(3, req); err != nil {
					t.Fatalf("err: %s", err)
				}
				restore.Commit()
			})
		})
	})
}

func TestStateStore_EnsureNode(t *testing.T) {
	s := testStateStore(t)

	// Fetching a non-existent node returns nil
	if _, node, err := s.GetNode("node1"); node != nil || err != nil {
		t.Fatalf("expected (nil, nil), got: (%#v, %#v)", node, err)
	}

	// Create a node registration request
	in := &structs.Node{
		Node:    "node1",
		Address: "1.1.1.1",
	}

	// Ensure the node is registered in the db
	if err := s.EnsureNode(1, in); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Retrieve the node again
	idx, out, err := s.GetNode("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	// Correct node was returned
	if out.Node != "node1" || out.Address != "1.1.1.1" {
		t.Fatalf("bad node returned: %#v", out)
	}

	// Indexes are set properly
	if out.CreateIndex != 1 || out.ModifyIndex != 1 {
		t.Fatalf("bad node index: %#v", out)
	}
	if idx != 1 {
		t.Fatalf("bad index: %d", idx)
	}

	// Update the node registration
	in.Address = "1.1.1.2"
	if err := s.EnsureNode(2, in); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Retrieve the node
	idx, out, err = s.GetNode("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	// Node and indexes were updated
	if out.CreateIndex != 1 || out.ModifyIndex != 2 || out.Address != "1.1.1.2" {
		t.Fatalf("bad: %#v", out)
	}
	if idx != 2 {
		t.Fatalf("bad index: %d", idx)
	}

	// Node upsert preserves the create index
	if err := s.EnsureNode(3, in); err != nil {
		t.Fatalf("err: %s", err)
	}
	idx, out, err = s.GetNode("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if out.CreateIndex != 1 || out.ModifyIndex != 3 || out.Address != "1.1.1.2" {
		t.Fatalf("node was modified: %#v", out)
	}
	if idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_GetNodes(t *testing.T) {
	s := testStateStore(t)

	// Listing with no results returns nil
	idx, res, err := s.Nodes()
	if idx != 0 || res != nil || err != nil {
		t.Fatalf("expected (0, nil, nil), got: (%d, %#v, %#v)", idx, res, err)
	}

	// Create some nodes in the state store
	testRegisterNode(t, s, 0, "node0")
	testRegisterNode(t, s, 1, "node1")
	testRegisterNode(t, s, 2, "node2")

	// Retrieve the nodes
	idx, nodes, err := s.Nodes()
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	// Highest index was returned
	if idx != 2 {
		t.Fatalf("bad index: %d", idx)
	}

	// All nodes were returned
	if n := len(nodes); n != 3 {
		t.Fatalf("bad node count: %d", n)
	}

	// Make sure the nodes match
	for i, node := range nodes {
		if node.CreateIndex != uint64(i) || node.ModifyIndex != uint64(i) {
			t.Fatalf("bad node index: %d, %d", node.CreateIndex, node.ModifyIndex)
		}
		name := fmt.Sprintf("node%d", i)
		if node.Node != name {
			t.Fatalf("bad: %#v", node)
		}
	}
}

func BenchmarkGetNodes(b *testing.B) {
	s, err := NewStateStore(nil)
	if err != nil {
		b.Fatalf("err: %s", err)
	}

	if err := s.EnsureNode(100, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		b.Fatalf("err: %v", err)
	}
	if err := s.EnsureNode(101, &structs.Node{Node: "bar", Address: "127.0.0.2"}); err != nil {
		b.Fatalf("err: %v", err)
	}

	for i := 0; i < b.N; i++ {
		s.Nodes()
	}
}

func TestStateStore_DeleteNode(t *testing.T) {
	s := testStateStore(t)

	// Create a node and register a service and health check with it.
	testRegisterNode(t, s, 0, "node1")
	testRegisterService(t, s, 1, "node1", "service1")
	testRegisterCheck(t, s, 2, "node1", "", "check1", structs.HealthPassing)

	// Delete the node
	if err := s.DeleteNode(3, "node1"); err != nil {
		t.Fatalf("err: %s", err)
	}

	// The node was removed
	if idx, n, err := s.GetNode("node1"); err != nil || n != nil || idx != 3 {
		t.Fatalf("bad: %#v %d (err: %#v)", n, idx, err)
	}

	// Associated service was removed. Need to query this directly out of
	// the DB to make sure it is actually gone.
	tx := s.db.Txn(false)
	defer tx.Abort()
	services, err := tx.Get("services", "id", "node1", "service1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if service := services.Next(); service != nil {
		t.Fatalf("bad: %#v", service)
	}

	// Associated health check was removed.
	checks, err := tx.Get("checks", "id", "node1", "check1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if check := checks.Next(); check != nil {
		t.Fatalf("bad: %#v", check)
	}

	// Indexes were updated.
	for _, tbl := range []string{"nodes", "services", "checks"} {
		if idx := s.maxIndex(tbl); idx != 3 {
			t.Fatalf("bad index: %d (%s)", idx, tbl)
		}
	}

	// Deleting a nonexistent node should be idempotent and not return
	// an error
	if err := s.DeleteNode(4, "node1"); err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx := s.maxIndex("nodes"); idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_Node_Snapshot(t *testing.T) {
	s := testStateStore(t)

	// Create some nodes in the state store.
	testRegisterNode(t, s, 0, "node0")
	testRegisterNode(t, s, 1, "node1")
	testRegisterNode(t, s, 2, "node2")

	// Snapshot the nodes.
	snap := s.Snapshot()
	defer snap.Close()

	// Alter the real state store.
	testRegisterNode(t, s, 3, "node3")

	// Verify the snapshot.
	if idx := snap.LastIndex(); idx != 2 {
		t.Fatalf("bad index: %d", idx)
	}
	nodes, err := snap.Nodes()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	for i := 0; i < 3; i++ {
		node := nodes.Next().(*structs.Node)
		if node == nil {
			t.Fatalf("unexpected end of nodes")
		}

		if node.CreateIndex != uint64(i) || node.ModifyIndex != uint64(i) {
			t.Fatalf("bad node index: %d, %d", node.CreateIndex, node.ModifyIndex)
		}
		if node.Node != fmt.Sprintf("node%d", i) {
			t.Fatalf("bad: %#v", node)
		}
	}
	if nodes.Next() != nil {
		t.Fatalf("unexpected extra nodes")
	}
}

func TestStateStore_Node_Watches(t *testing.T) {
	s := testStateStore(t)

	// Call functions that update the nodes table and make sure a watch fires
	// each time.
	verifyWatch(t, s.getTableWatch("nodes"), func() {
		req := &structs.RegisterRequest{
			Node: "node1",
		}
		if err := s.EnsureRegistration(1, req); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
	verifyWatch(t, s.getTableWatch("nodes"), func() {
		node := &structs.Node{Node: "node2"}
		if err := s.EnsureNode(2, node); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
	verifyWatch(t, s.getTableWatch("nodes"), func() {
		if err := s.DeleteNode(3, "node2"); err != nil {
			t.Fatalf("err: %s", err)
		}
	})

	// Check that a delete of a node + service + check + coordinate triggers
	// all tables in one shot.
	testRegisterNode(t, s, 4, "node1")
	testRegisterService(t, s, 5, "node1", "service1")
	testRegisterCheck(t, s, 6, "node1", "service1", "check3", structs.HealthPassing)
	updates := structs.Coordinates{
		&structs.Coordinate{
			Node:  "node1",
			Coord: generateRandomCoordinate(),
		},
	}
	if err := s.CoordinateBatchUpdate(7, updates); err != nil {
		t.Fatalf("err: %s", err)
	}
	verifyWatch(t, s.getTableWatch("nodes"), func() {
		verifyWatch(t, s.getTableWatch("services"), func() {
			verifyWatch(t, s.getTableWatch("checks"), func() {
				verifyWatch(t, s.getTableWatch("coordinates"), func() {
					if err := s.DeleteNode(7, "node1"); err != nil {
						t.Fatalf("err: %s", err)
					}
				})
			})
		})
	})
}

func TestStateStore_EnsureService(t *testing.T) {
	s := testStateStore(t)

	// Fetching services for a node with none returns nil
	idx, res, err := s.NodeServices("node1")
	if err != nil || res != nil || idx != 0 {
		t.Fatalf("expected (0, nil, nil), got: (%d, %#v, %#v)", idx, res, err)
	}

	// Create the service registration
	ns1 := &structs.NodeService{
		ID:      "service1",
		Service: "redis",
		Tags:    []string{"prod"},
		Address: "1.1.1.1",
		Port:    1111,
	}

	// Creating a service without a node returns an error
	if err := s.EnsureService(1, "node1", ns1); err != ErrMissingNode {
		t.Fatalf("expected %#v, got: %#v", ErrMissingNode, err)
	}

	// Register the nodes
	testRegisterNode(t, s, 0, "node1")
	testRegisterNode(t, s, 1, "node2")

	// Service successfully registers into the state store
	if err = s.EnsureService(10, "node1", ns1); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Register a similar service against both nodes
	ns2 := *ns1
	ns2.ID = "service2"
	for _, n := range []string{"node1", "node2"} {
		if err := s.EnsureService(20, n, &ns2); err != nil {
			t.Fatalf("err: %s", err)
		}
	}

	// Register a different service on the bad node
	ns3 := *ns1
	ns3.ID = "service3"
	if err := s.EnsureService(30, "node2", &ns3); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Retrieve the services
	idx, out, err := s.NodeServices("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 30 {
		t.Fatalf("bad index: %d", idx)
	}

	// Only the services for the requested node are returned
	if out == nil || len(out.Services) != 2 {
		t.Fatalf("bad services: %#v", out)
	}

	// Results match the inserted services and have the proper indexes set
	expect1 := *ns1
	expect1.CreateIndex, expect1.ModifyIndex = 10, 10
	if svc := out.Services["service1"]; !reflect.DeepEqual(&expect1, svc) {
		t.Fatalf("bad: %#v", svc)
	}

	expect2 := ns2
	expect2.CreateIndex, expect2.ModifyIndex = 20, 20
	if svc := out.Services["service2"]; !reflect.DeepEqual(&expect2, svc) {
		t.Fatalf("bad: %#v %#v", ns2, svc)
	}

	// Index tables were updated
	if idx := s.maxIndex("services"); idx != 30 {
		t.Fatalf("bad index: %d", idx)
	}

	// Update a service registration
	ns1.Address = "1.1.1.2"
	if err := s.EnsureService(40, "node1", ns1); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Retrieve the service again and ensure it matches
	idx, out, err = s.NodeServices("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 40 {
		t.Fatalf("bad index: %d", idx)
	}
	if out == nil || len(out.Services) != 2 {
		t.Fatalf("bad: %#v", out)
	}
	expect1.Address = "1.1.1.2"
	expect1.ModifyIndex = 40
	if svc := out.Services["service1"]; !reflect.DeepEqual(&expect1, svc) {
		t.Fatalf("bad: %#v", svc)
	}

	// Index tables were updated
	if idx := s.maxIndex("services"); idx != 40 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_Services(t *testing.T) {
	s := testStateStore(t)

	// Register several nodes and services.
	testRegisterNode(t, s, 1, "node1")
	ns1 := &structs.NodeService{
		ID:      "service1",
		Service: "redis",
		Tags:    []string{"prod", "master"},
		Address: "1.1.1.1",
		Port:    1111,
	}
	if err := s.EnsureService(2, "node1", ns1); err != nil {
		t.Fatalf("err: %s", err)
	}
	testRegisterService(t, s, 3, "node1", "dogs")
	testRegisterNode(t, s, 4, "node2")
	ns2 := &structs.NodeService{
		ID:      "service3",
		Service: "redis",
		Tags:    []string{"prod", "slave"},
		Address: "1.1.1.1",
		Port:    1111,
	}
	if err := s.EnsureService(5, "node2", ns2); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Pull all the services.
	idx, services, err := s.Services()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}

	// Verify the result. We sort the lists since the order is
	// non-deterministic (it's built using a map internally).
	expected := structs.Services{
		"redis": []string{"prod", "master", "slave"},
		"dogs":  []string{},
	}
	sort.Strings(expected["redis"])
	for _, tags := range services {
		sort.Strings(tags)
	}
	if !reflect.DeepEqual(expected, services) {
		t.Fatalf("bad: %#v", services)
	}
}

func TestStateStore_ServiceNodes(t *testing.T) {
	s := testStateStore(t)

	if err := s.EnsureNode(10, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := s.EnsureNode(11, &structs.Node{Node: "bar", Address: "127.0.0.2"}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := s.EnsureService(12, "foo", &structs.NodeService{ID: "api", Service: "api", Tags: nil, Address: "", Port: 5000}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := s.EnsureService(13, "bar", &structs.NodeService{ID: "api", Service: "api", Tags: nil, Address: "", Port: 5000}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := s.EnsureService(14, "foo", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"master"}, Address: "", Port: 8000}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := s.EnsureService(15, "bar", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"slave"}, Address: "", Port: 8000}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := s.EnsureService(16, "bar", &structs.NodeService{ID: "db2", Service: "db", Tags: []string{"slave"}, Address: "", Port: 8001}); err != nil {
		t.Fatalf("err: %v", err)
	}

	idx, nodes, err := s.ServiceNodes("db")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 16 {
		t.Fatalf("bad: %v", 16)
	}
	if len(nodes) != 3 {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].Node != "bar" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].Address != "127.0.0.2" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].ServiceID != "db" {
		t.Fatalf("bad: %v", nodes)
	}
	if !lib.StrContains(nodes[0].ServiceTags, "slave") {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].ServicePort != 8000 {
		t.Fatalf("bad: %v", nodes)
	}

	if nodes[1].Node != "bar" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[1].Address != "127.0.0.2" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[1].ServiceID != "db2" {
		t.Fatalf("bad: %v", nodes)
	}
	if !lib.StrContains(nodes[1].ServiceTags, "slave") {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[1].ServicePort != 8001 {
		t.Fatalf("bad: %v", nodes)
	}

	if nodes[2].Node != "foo" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[2].Address != "127.0.0.1" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[2].ServiceID != "db" {
		t.Fatalf("bad: %v", nodes)
	}
	if !lib.StrContains(nodes[2].ServiceTags, "master") {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[2].ServicePort != 8000 {
		t.Fatalf("bad: %v", nodes)
	}
}

func TestStateStore_ServiceTagNodes(t *testing.T) {
	s := testStateStore(t)

	if err := s.EnsureNode(15, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := s.EnsureNode(16, &structs.Node{Node: "bar", Address: "127.0.0.2"}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := s.EnsureService(17, "foo", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"master"}, Address: "", Port: 8000}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := s.EnsureService(18, "foo", &structs.NodeService{ID: "db2", Service: "db", Tags: []string{"slave"}, Address: "", Port: 8001}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := s.EnsureService(19, "bar", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"slave"}, Address: "", Port: 8000}); err != nil {
		t.Fatalf("err: %v", err)
	}

	idx, nodes, err := s.ServiceTagNodes("db", "master")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 19 {
		t.Fatalf("bad: %v", idx)
	}
	if len(nodes) != 1 {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].Node != "foo" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].Address != "127.0.0.1" {
		t.Fatalf("bad: %v", nodes)
	}
	if !lib.StrContains(nodes[0].ServiceTags, "master") {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].ServicePort != 8000 {
		t.Fatalf("bad: %v", nodes)
	}
}

func TestStateStore_ServiceTagNodes_MultipleTags(t *testing.T) {
	s := testStateStore(t)

	if err := s.EnsureNode(15, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := s.EnsureNode(16, &structs.Node{Node: "bar", Address: "127.0.0.2"}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := s.EnsureService(17, "foo", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"master", "v2"}, Address: "", Port: 8000}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := s.EnsureService(18, "foo", &structs.NodeService{ID: "db2", Service: "db", Tags: []string{"slave", "v2", "dev"}, Address: "", Port: 8001}); err != nil {
		t.Fatalf("err: %v", err)
	}

	if err := s.EnsureService(19, "bar", &structs.NodeService{ID: "db", Service: "db", Tags: []string{"slave", "v2"}, Address: "", Port: 8000}); err != nil {
		t.Fatalf("err: %v", err)
	}

	idx, nodes, err := s.ServiceTagNodes("db", "master")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 19 {
		t.Fatalf("bad: %v", idx)
	}
	if len(nodes) != 1 {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].Node != "foo" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].Address != "127.0.0.1" {
		t.Fatalf("bad: %v", nodes)
	}
	if !lib.StrContains(nodes[0].ServiceTags, "master") {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].ServicePort != 8000 {
		t.Fatalf("bad: %v", nodes)
	}

	idx, nodes, err = s.ServiceTagNodes("db", "v2")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 19 {
		t.Fatalf("bad: %v", idx)
	}
	if len(nodes) != 3 {
		t.Fatalf("bad: %v", nodes)
	}

	idx, nodes, err = s.ServiceTagNodes("db", "dev")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 19 {
		t.Fatalf("bad: %v", idx)
	}
	if len(nodes) != 1 {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].Node != "foo" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].Address != "127.0.0.1" {
		t.Fatalf("bad: %v", nodes)
	}
	if !lib.StrContains(nodes[0].ServiceTags, "dev") {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].ServicePort != 8001 {
		t.Fatalf("bad: %v", nodes)
	}
}

func TestStateStore_DeleteService(t *testing.T) {
	s := testStateStore(t)

	// Register a node with one service and a check
	testRegisterNode(t, s, 1, "node1")
	testRegisterService(t, s, 2, "node1", "service1")
	testRegisterCheck(t, s, 3, "node1", "service1", "check1", structs.HealthPassing)

	// Delete the service
	if err := s.DeleteService(4, "node1", "service1"); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Service doesn't exist.
	_, ns, err := s.NodeServices("node1")
	if err != nil || ns == nil || len(ns.Services) != 0 {
		t.Fatalf("bad: %#v (err: %#v)", ns, err)
	}

	// Check doesn't exist. Check using the raw DB so we can test
	// that it actually is removed in the state store.
	tx := s.db.Txn(false)
	defer tx.Abort()
	check, err := tx.First("checks", "id", "node1", "check1")
	if err != nil || check != nil {
		t.Fatalf("bad: %#v (err: %s)", check, err)
	}

	// Index tables were updated
	if idx := s.maxIndex("services"); idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}
	if idx := s.maxIndex("checks"); idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}

	// Deleting a nonexistent service should be idempotent and not return an
	// error
	if err := s.DeleteService(5, "node1", "service1"); err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx := s.maxIndex("services"); idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_Service_Snapshot(t *testing.T) {
	s := testStateStore(t)

	// Register a node with two services.
	testRegisterNode(t, s, 0, "node1")
	ns := []*structs.NodeService{
		&structs.NodeService{
			ID:      "service1",
			Service: "redis",
			Tags:    []string{"prod"},
			Address: "1.1.1.1",
			Port:    1111,
		},
		&structs.NodeService{
			ID:      "service2",
			Service: "nomad",
			Tags:    []string{"dev"},
			Address: "1.1.1.2",
			Port:    1112,
		},
	}
	for i, svc := range ns {
		if err := s.EnsureService(uint64(i+1), "node1", svc); err != nil {
			t.Fatalf("err: %s", err)
		}
	}

	// Create a second node/service to make sure node filtering works. This
	// will affect the index but not the dump.
	testRegisterNode(t, s, 3, "node2")
	testRegisterService(t, s, 4, "node2", "service2")

	// Snapshot the service.
	snap := s.Snapshot()
	defer snap.Close()

	// Alter the real state store.
	testRegisterService(t, s, 5, "node2", "service3")

	// Verify the snapshot.
	if idx := snap.LastIndex(); idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}
	services, err := snap.Services("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	for i := 0; i < len(ns); i++ {
		svc := services.Next().(*structs.ServiceNode)
		if svc == nil {
			t.Fatalf("unexpected end of services")
		}

		ns[i].CreateIndex, ns[i].ModifyIndex = uint64(i+1), uint64(i+1)
		if !reflect.DeepEqual(ns[i], svc.ToNodeService()) {
			t.Fatalf("bad: %#v != %#v", svc, ns[i])
		}
	}
	if services.Next() != nil {
		t.Fatalf("unexpected extra services")
	}
}

func TestStateStore_Service_Watches(t *testing.T) {
	s := testStateStore(t)

	testRegisterNode(t, s, 0, "node1")
	ns := &structs.NodeService{
		ID:      "service2",
		Service: "nomad",
		Address: "1.1.1.2",
		Port:    8000,
	}

	// Call functions that update the services table and make sure a watch
	// fires each time.
	verifyWatch(t, s.getTableWatch("services"), func() {
		if err := s.EnsureService(2, "node1", ns); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
	verifyWatch(t, s.getTableWatch("services"), func() {
		if err := s.DeleteService(3, "node1", "service2"); err != nil {
			t.Fatalf("err: %s", err)
		}
	})

	// Check that a delete of a service + check triggers both tables in one
	// shot.
	testRegisterService(t, s, 4, "node1", "service1")
	testRegisterCheck(t, s, 5, "node1", "service1", "check3", structs.HealthPassing)
	verifyWatch(t, s.getTableWatch("services"), func() {
		verifyWatch(t, s.getTableWatch("checks"), func() {
			if err := s.DeleteService(6, "node1", "service1"); err != nil {
				t.Fatalf("err: %s", err)
			}
		})
	})
}

func TestStateStore_EnsureCheck(t *testing.T) {
	s := testStateStore(t)

	// Create a check associated with the node
	check := &structs.HealthCheck{
		Node:        "node1",
		CheckID:     "check1",
		Name:        "redis check",
		Status:      structs.HealthPassing,
		Notes:       "test check",
		Output:      "aaa",
		ServiceID:   "service1",
		ServiceName: "redis",
	}

	// Creating a check without a node returns error
	if err := s.EnsureCheck(1, check); err != ErrMissingNode {
		t.Fatalf("expected %#v, got: %#v", ErrMissingNode, err)
	}

	// Register the node
	testRegisterNode(t, s, 1, "node1")

	// Creating a check with a bad services returns error
	if err := s.EnsureCheck(1, check); err != ErrMissingService {
		t.Fatalf("expected: %#v, got: %#v", ErrMissingService, err)
	}

	// Register the service
	testRegisterService(t, s, 2, "node1", "service1")

	// Inserting the check with the prerequisites succeeds
	if err := s.EnsureCheck(3, check); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Retrieve the check and make sure it matches
	idx, checks, err := s.NodeChecks("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}
	if len(checks) != 1 {
		t.Fatalf("wrong number of checks: %d", len(checks))
	}
	if !reflect.DeepEqual(checks[0], check) {
		t.Fatalf("bad: %#v", checks[0])
	}

	// Modify the health check
	check.Output = "bbb"
	if err := s.EnsureCheck(4, check); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Check that we successfully updated
	idx, checks, err = s.NodeChecks("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}
	if len(checks) != 1 {
		t.Fatalf("wrong number of checks: %d", len(checks))
	}
	if checks[0].Output != "bbb" {
		t.Fatalf("wrong check output: %#v", checks[0])
	}
	if checks[0].CreateIndex != 3 || checks[0].ModifyIndex != 4 {
		t.Fatalf("bad index: %#v", checks[0])
	}

	// Index tables were updated
	if idx := s.maxIndex("checks"); idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_EnsureCheck_defaultStatus(t *testing.T) {
	s := testStateStore(t)

	// Register a node
	testRegisterNode(t, s, 1, "node1")

	// Create and register a check with no health status
	check := &structs.HealthCheck{
		Node:    "node1",
		CheckID: "check1",
		Status:  "",
	}
	if err := s.EnsureCheck(2, check); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Get the check again
	_, result, err := s.NodeChecks("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	// Check that the status was set to the proper default
	if len(result) != 1 || result[0].Status != structs.HealthCritical {
		t.Fatalf("bad: %#v", result)
	}
}

func TestStateStore_NodeChecks(t *testing.T) {
	s := testStateStore(t)

	// Create the first node and service with some checks
	testRegisterNode(t, s, 0, "node1")
	testRegisterService(t, s, 1, "node1", "service1")
	testRegisterCheck(t, s, 2, "node1", "service1", "check1", structs.HealthPassing)
	testRegisterCheck(t, s, 3, "node1", "service1", "check2", structs.HealthPassing)

	// Create a second node/service with a different set of checks
	testRegisterNode(t, s, 4, "node2")
	testRegisterService(t, s, 5, "node2", "service2")
	testRegisterCheck(t, s, 6, "node2", "service2", "check3", structs.HealthPassing)

	// Try querying for all checks associated with node1
	idx, checks, err := s.NodeChecks("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}
	if len(checks) != 2 || checks[0].CheckID != "check1" || checks[1].CheckID != "check2" {
		t.Fatalf("bad checks: %#v", checks)
	}

	// Try querying for all checks associated with node2
	idx, checks, err = s.NodeChecks("node2")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}
	if len(checks) != 1 || checks[0].CheckID != "check3" {
		t.Fatalf("bad checks: %#v", checks)
	}
}

func TestStateStore_ServiceChecks(t *testing.T) {
	s := testStateStore(t)

	// Create the first node and service with some checks
	testRegisterNode(t, s, 0, "node1")
	testRegisterService(t, s, 1, "node1", "service1")
	testRegisterCheck(t, s, 2, "node1", "service1", "check1", structs.HealthPassing)
	testRegisterCheck(t, s, 3, "node1", "service1", "check2", structs.HealthPassing)

	// Create a second node/service with a different set of checks
	testRegisterNode(t, s, 4, "node2")
	testRegisterService(t, s, 5, "node2", "service2")
	testRegisterCheck(t, s, 6, "node2", "service2", "check3", structs.HealthPassing)

	// Try querying for all checks associated with service1
	idx, checks, err := s.ServiceChecks("service1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}
	if len(checks) != 2 || checks[0].CheckID != "check1" || checks[1].CheckID != "check2" {
		t.Fatalf("bad checks: %#v", checks)
	}
}

func TestStateStore_ChecksInState(t *testing.T) {
	s := testStateStore(t)

	// Querying with no results returns nil
	idx, res, err := s.ChecksInState(structs.HealthPassing)
	if idx != 0 || res != nil || err != nil {
		t.Fatalf("expected (0, nil, nil), got: (%d, %#v, %#v)", idx, res, err)
	}

	// Register a node with checks in varied states
	testRegisterNode(t, s, 0, "node1")
	testRegisterCheck(t, s, 1, "node1", "", "check1", structs.HealthPassing)
	testRegisterCheck(t, s, 2, "node1", "", "check2", structs.HealthCritical)
	testRegisterCheck(t, s, 3, "node1", "", "check3", structs.HealthPassing)

	// Query the state store for passing checks.
	_, checks, err := s.ChecksInState(structs.HealthPassing)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	// Make sure we only get the checks which match the state
	if n := len(checks); n != 2 {
		t.Fatalf("expected 2 checks, got: %d", n)
	}
	if checks[0].CheckID != "check1" || checks[1].CheckID != "check3" {
		t.Fatalf("bad: %#v", checks)
	}

	// HealthAny just returns everything.
	_, checks, err = s.ChecksInState(structs.HealthAny)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if n := len(checks); n != 3 {
		t.Fatalf("expected 3 checks, got: %d", n)
	}
}

func TestStateStore_DeleteCheck(t *testing.T) {
	s := testStateStore(t)

	// Register a node and a node-level health check
	testRegisterNode(t, s, 1, "node1")
	testRegisterCheck(t, s, 2, "node1", "", "check1", structs.HealthPassing)

	// Delete the check
	if err := s.DeleteCheck(3, "node1", "check1"); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Check is gone
	_, checks, err := s.NodeChecks("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if len(checks) != 0 {
		t.Fatalf("bad: %#v", checks)
	}

	// Index tables were updated
	if idx := s.maxIndex("checks"); idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}

	// Deleting a nonexistent check should be idempotent and not return an
	// error
	if err := s.DeleteCheck(4, "node1", "check1"); err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx := s.maxIndex("checks"); idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_CheckServiceNodes(t *testing.T) {
	s := testStateStore(t)

	// Querying with no matches gives an empty response
	idx, res, err := s.CheckServiceNodes("service1")
	if idx != 0 || res != nil || err != nil {
		t.Fatalf("expected (0, nil, nil), got: (%d, %#v, %#v)", idx, res, err)
	}

	// Register some nodes
	testRegisterNode(t, s, 0, "node1")
	testRegisterNode(t, s, 1, "node2")

	// Register node-level checks. These should not be returned
	// in the final result.
	testRegisterCheck(t, s, 2, "node1", "", "check1", structs.HealthPassing)
	testRegisterCheck(t, s, 3, "node2", "", "check2", structs.HealthPassing)

	// Register a service against the nodes
	testRegisterService(t, s, 4, "node1", "service1")
	testRegisterService(t, s, 5, "node2", "service2")

	// Register checks against the services
	testRegisterCheck(t, s, 6, "node1", "service1", "check3", structs.HealthPassing)
	testRegisterCheck(t, s, 7, "node2", "service2", "check4", structs.HealthPassing)

	// Query the state store for nodes and checks which
	// have been registered with a specific service.
	idx, results, err := s.CheckServiceNodes("service1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 7 {
		t.Fatalf("bad index: %d", idx)
	}

	// Make sure we get the expected result (service check + node check)
	if n := len(results); n != 1 {
		t.Fatalf("expected 1 result, got: %d", n)
	}
	csn := results[0]
	if csn.Node == nil || csn.Service == nil || len(csn.Checks) != 2 {
		t.Fatalf("bad output: %#v", csn)
	}

	// Node updates alter the returned index
	testRegisterNode(t, s, 8, "node1")
	idx, results, err = s.CheckServiceNodes("service1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 8 {
		t.Fatalf("bad index: %d", idx)
	}

	// Service updates alter the returned index
	testRegisterService(t, s, 9, "node1", "service1")
	idx, results, err = s.CheckServiceNodes("service1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 9 {
		t.Fatalf("bad index: %d", idx)
	}

	// Check updates alter the returned index
	testRegisterCheck(t, s, 10, "node1", "service1", "check1", structs.HealthCritical)
	idx, results, err = s.CheckServiceNodes("service1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 10 {
		t.Fatalf("bad index: %d", idx)
	}
}

func BenchmarkCheckServiceNodes(b *testing.B) {
	s, err := NewStateStore(nil)
	if err != nil {
		b.Fatalf("err: %s", err)
	}

	if err := s.EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		b.Fatalf("err: %v", err)
	}
	if err := s.EnsureService(2, "foo", &structs.NodeService{ID: "db1", Service: "db", Tags: []string{"master"}, Address: "", Port: 8000}); err != nil {
		b.Fatalf("err: %v", err)
	}
	check := &structs.HealthCheck{
		Node:      "foo",
		CheckID:   "db",
		Name:      "can connect",
		Status:    structs.HealthPassing,
		ServiceID: "db1",
	}
	if err := s.EnsureCheck(3, check); err != nil {
		b.Fatalf("err: %v", err)
	}
	check = &structs.HealthCheck{
		Node:    "foo",
		CheckID: "check1",
		Name:    "check1",
		Status:  structs.HealthPassing,
	}
	if err := s.EnsureCheck(4, check); err != nil {
		b.Fatalf("err: %v", err)
	}

	for i := 0; i < b.N; i++ {
		s.CheckServiceNodes("db")
	}
}

func TestStateStore_CheckServiceTagNodes(t *testing.T) {
	s := testStateStore(t)

	if err := s.EnsureNode(1, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := s.EnsureService(2, "foo", &structs.NodeService{ID: "db1", Service: "db", Tags: []string{"master"}, Address: "", Port: 8000}); err != nil {
		t.Fatalf("err: %v", err)
	}
	check := &structs.HealthCheck{
		Node:      "foo",
		CheckID:   "db",
		Name:      "can connect",
		Status:    structs.HealthPassing,
		ServiceID: "db1",
	}
	if err := s.EnsureCheck(3, check); err != nil {
		t.Fatalf("err: %v", err)
	}
	check = &structs.HealthCheck{
		Node:    "foo",
		CheckID: "check1",
		Name:    "another check",
		Status:  structs.HealthPassing,
	}
	if err := s.EnsureCheck(4, check); err != nil {
		t.Fatalf("err: %v", err)
	}

	idx, nodes, err := s.CheckServiceTagNodes("db", "master")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 4 {
		t.Fatalf("bad: %v", idx)
	}
	if len(nodes) != 1 {
		t.Fatalf("Bad: %v", nodes)
	}
	if nodes[0].Node.Node != "foo" {
		t.Fatalf("Bad: %v", nodes[0])
	}
	if nodes[0].Service.ID != "db1" {
		t.Fatalf("Bad: %v", nodes[0])
	}
	if len(nodes[0].Checks) != 2 {
		t.Fatalf("Bad: %v", nodes[0])
	}
	if nodes[0].Checks[0].CheckID != "check1" {
		t.Fatalf("Bad: %v", nodes[0])
	}
	if nodes[0].Checks[1].CheckID != "db" {
		t.Fatalf("Bad: %v", nodes[0])
	}
}

func TestStateStore_Check_Snapshot(t *testing.T) {
	s := testStateStore(t)

	// Create a node, a service, and a service check as well as a node check.
	testRegisterNode(t, s, 0, "node1")
	testRegisterService(t, s, 1, "node1", "service1")
	checks := structs.HealthChecks{
		&structs.HealthCheck{
			Node:    "node1",
			CheckID: "check1",
			Name:    "node check",
			Status:  structs.HealthPassing,
		},
		&structs.HealthCheck{
			Node:      "node1",
			CheckID:   "check2",
			Name:      "service check",
			Status:    structs.HealthCritical,
			ServiceID: "service1",
		},
	}
	for i, hc := range checks {
		if err := s.EnsureCheck(uint64(i+1), hc); err != nil {
			t.Fatalf("err: %s", err)
		}
	}

	// Create a second node/service to make sure node filtering works. This
	// will affect the index but not the dump.
	testRegisterNode(t, s, 3, "node2")
	testRegisterService(t, s, 4, "node2", "service2")
	testRegisterCheck(t, s, 5, "node2", "service2", "check3", structs.HealthPassing)

	// Snapshot the checks.
	snap := s.Snapshot()
	defer snap.Close()

	// Alter the real state store.
	testRegisterCheck(t, s, 6, "node2", "service2", "check4", structs.HealthPassing)

	// Verify the snapshot.
	if idx := snap.LastIndex(); idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}
	iter, err := snap.Checks("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	for i := 0; i < len(checks); i++ {
		check := iter.Next().(*structs.HealthCheck)
		if check == nil {
			t.Fatalf("unexpected end of checks")
		}

		checks[i].CreateIndex, checks[i].ModifyIndex = uint64(i+1), uint64(i+1)
		if !reflect.DeepEqual(check, checks[i]) {
			t.Fatalf("bad: %#v != %#v", check, checks[i])
		}
	}
	if iter.Next() != nil {
		t.Fatalf("unexpected extra checks")
	}
}

func TestStateStore_Check_Watches(t *testing.T) {
	s := testStateStore(t)

	testRegisterNode(t, s, 0, "node1")
	hc := &structs.HealthCheck{
		Node:    "node1",
		CheckID: "check1",
		Status:  structs.HealthPassing,
	}

	// Call functions that update the checks table and make sure a watch fires
	// each time.
	verifyWatch(t, s.getTableWatch("checks"), func() {
		if err := s.EnsureCheck(1, hc); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
	verifyWatch(t, s.getTableWatch("checks"), func() {
		hc.Status = structs.HealthCritical
		if err := s.EnsureCheck(2, hc); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
	verifyWatch(t, s.getTableWatch("checks"), func() {
		if err := s.DeleteCheck(3, "node1", "check1"); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
}

func TestStateStore_NodeInfo_NodeDump(t *testing.T) {
	s := testStateStore(t)

	// Generating a node dump that matches nothing returns empty
	idx, dump, err := s.NodeInfo("node1")
	if idx != 0 || dump != nil || err != nil {
		t.Fatalf("expected (0, nil, nil), got: (%d, %#v, %#v)", idx, dump, err)
	}
	idx, dump, err = s.NodeDump()
	if idx != 0 || dump != nil || err != nil {
		t.Fatalf("expected (0, nil, nil), got: (%d, %#v, %#v)", idx, dump, err)
	}

	// Register some nodes
	testRegisterNode(t, s, 0, "node1")
	testRegisterNode(t, s, 1, "node2")

	// Register services against them
	testRegisterService(t, s, 2, "node1", "service1")
	testRegisterService(t, s, 3, "node1", "service2")
	testRegisterService(t, s, 4, "node2", "service1")
	testRegisterService(t, s, 5, "node2", "service2")

	// Register service-level checks
	testRegisterCheck(t, s, 6, "node1", "service1", "check1", structs.HealthPassing)
	testRegisterCheck(t, s, 7, "node2", "service1", "check1", structs.HealthPassing)

	// Register node-level checks
	testRegisterCheck(t, s, 8, "node1", "", "check2", structs.HealthPassing)
	testRegisterCheck(t, s, 9, "node2", "", "check2", structs.HealthPassing)

	// Check that our result matches what we expect.
	expect := structs.NodeDump{
		&structs.NodeInfo{
			Node: "node1",
			Checks: structs.HealthChecks{
				&structs.HealthCheck{
					Node:        "node1",
					CheckID:     "check1",
					ServiceID:   "service1",
					ServiceName: "service1",
					Status:      structs.HealthPassing,
					RaftIndex: structs.RaftIndex{
						CreateIndex: 6,
						ModifyIndex: 6,
					},
				},
				&structs.HealthCheck{
					Node:        "node1",
					CheckID:     "check2",
					ServiceID:   "",
					ServiceName: "",
					Status:      structs.HealthPassing,
					RaftIndex: structs.RaftIndex{
						CreateIndex: 8,
						ModifyIndex: 8,
					},
				},
			},
			Services: []*structs.NodeService{
				&structs.NodeService{
					ID:      "service1",
					Service: "service1",
					Address: "1.1.1.1",
					Port:    1111,
					RaftIndex: structs.RaftIndex{
						CreateIndex: 2,
						ModifyIndex: 2,
					},
				},
				&structs.NodeService{
					ID:      "service2",
					Service: "service2",
					Address: "1.1.1.1",
					Port:    1111,
					RaftIndex: structs.RaftIndex{
						CreateIndex: 3,
						ModifyIndex: 3,
					},
				},
			},
		},
		&structs.NodeInfo{
			Node: "node2",
			Checks: structs.HealthChecks{
				&structs.HealthCheck{
					Node:        "node2",
					CheckID:     "check1",
					ServiceID:   "service1",
					ServiceName: "service1",
					Status:      structs.HealthPassing,
					RaftIndex: structs.RaftIndex{
						CreateIndex: 7,
						ModifyIndex: 7,
					},
				},
				&structs.HealthCheck{
					Node:        "node2",
					CheckID:     "check2",
					ServiceID:   "",
					ServiceName: "",
					Status:      structs.HealthPassing,
					RaftIndex: structs.RaftIndex{
						CreateIndex: 9,
						ModifyIndex: 9,
					},
				},
			},
			Services: []*structs.NodeService{
				&structs.NodeService{
					ID:      "service1",
					Service: "service1",
					Address: "1.1.1.1",
					Port:    1111,
					RaftIndex: structs.RaftIndex{
						CreateIndex: 4,
						ModifyIndex: 4,
					},
				},
				&structs.NodeService{
					ID:      "service2",
					Service: "service2",
					Address: "1.1.1.1",
					Port:    1111,
					RaftIndex: structs.RaftIndex{
						CreateIndex: 5,
						ModifyIndex: 5,
					},
				},
			},
		},
	}

	// Get a dump of just a single node
	idx, dump, err = s.NodeInfo("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 9 {
		t.Fatalf("bad index: %d", idx)
	}
	if len(dump) != 1 || !reflect.DeepEqual(dump[0], expect[0]) {
		t.Fatalf("bad: %#v", dump)
	}

	// Generate a dump of all the nodes
	idx, dump, err = s.NodeDump()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 9 {
		t.Fatalf("bad index: %d", 9)
	}
	if !reflect.DeepEqual(dump, expect) {
		t.Fatalf("bad: %#v", dump[0].Services[0])
	}
}

func TestStateStore_KVSSet_KVSGet(t *testing.T) {
	s := testStateStore(t)

	// Get on an nonexistent key returns nil.
	idx, result, err := s.KVSGet("foo")
	if result != nil || err != nil || idx != 0 {
		t.Fatalf("expected (0, nil, nil), got : (%#v, %#v, %#v)", idx, result, err)
	}

	// Write a new K/V entry to the store.
	entry := &structs.DirEntry{
		Key:   "foo",
		Value: []byte("bar"),
	}
	if err := s.KVSSet(1, entry); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Retrieve the K/V entry again.
	idx, result, err = s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result == nil {
		t.Fatalf("expected k/v pair, got nothing")
	}
	if idx != 1 {
		t.Fatalf("bad index: %d", idx)
	}

	// Check that the index was injected into the result.
	if result.CreateIndex != 1 || result.ModifyIndex != 1 {
		t.Fatalf("bad index: %d, %d", result.CreateIndex, result.ModifyIndex)
	}

	// Check that the value matches.
	if v := string(result.Value); v != "bar" {
		t.Fatalf("expected 'bar', got: '%s'", v)
	}

	// Updating the entry works and changes the index.
	update := &structs.DirEntry{
		Key:   "foo",
		Value: []byte("baz"),
	}
	if err := s.KVSSet(2, update); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Fetch the kv pair and check.
	idx, result, err = s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.CreateIndex != 1 || result.ModifyIndex != 2 {
		t.Fatalf("bad index: %d, %d", result.CreateIndex, result.ModifyIndex)
	}
	if v := string(result.Value); v != "baz" {
		t.Fatalf("expected 'baz', got '%s'", v)
	}
	if idx != 2 {
		t.Fatalf("bad index: %d", idx)
	}

	// Attempt to set the session during an update.
	update = &structs.DirEntry{
		Key:     "foo",
		Value:   []byte("zoo"),
		Session: "nope",
	}
	if err := s.KVSSet(3, update); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Fetch the kv pair and check.
	idx, result, err = s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.CreateIndex != 1 || result.ModifyIndex != 3 {
		t.Fatalf("bad index: %d, %d", result.CreateIndex, result.ModifyIndex)
	}
	if v := string(result.Value); v != "zoo" {
		t.Fatalf("expected 'zoo', got '%s'", v)
	}
	if result.Session != "" {
		t.Fatalf("expected empty session, got '%s", result.Session)
	}
	if idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}

	// Make a real session and then lock the key to set the session.
	testRegisterNode(t, s, 4, "node1")
	session := testUUID()
	if err := s.SessionCreate(5, &structs.Session{ID: session, Node: "node1"}); err != nil {
		t.Fatalf("err: %s", err)
	}
	update = &structs.DirEntry{
		Key:     "foo",
		Value:   []byte("locked"),
		Session: session,
	}
	ok, err := s.KVSLock(6, update)
	if !ok || err != nil {
		t.Fatalf("didn't get the lock: %v %s", ok, err)
	}

	// Fetch the kv pair and check.
	idx, result, err = s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.CreateIndex != 1 || result.ModifyIndex != 6 {
		t.Fatalf("bad index: %d, %d", result.CreateIndex, result.ModifyIndex)
	}
	if v := string(result.Value); v != "locked" {
		t.Fatalf("expected 'zoo', got '%s'", v)
	}
	if result.Session != session {
		t.Fatalf("expected session, got '%s", result.Session)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}

	// Now make an update without the session and make sure it gets applied
	// and doesn't take away the session (it is allowed to change the value).
	update = &structs.DirEntry{
		Key:   "foo",
		Value: []byte("stoleit"),
	}
	if err := s.KVSSet(7, update); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Fetch the kv pair and check.
	idx, result, err = s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.CreateIndex != 1 || result.ModifyIndex != 7 {
		t.Fatalf("bad index: %d, %d", result.CreateIndex, result.ModifyIndex)
	}
	if v := string(result.Value); v != "stoleit" {
		t.Fatalf("expected 'zoo', got '%s'", v)
	}
	if result.Session != session {
		t.Fatalf("expected session, got '%s", result.Session)
	}
	if idx != 7 {
		t.Fatalf("bad index: %d", idx)
	}

	// Fetch a key that doesn't exist and make sure we get the right
	// response.
	idx, result, err = s.KVSGet("nope")
	if result != nil || err != nil || idx != 7 {
		t.Fatalf("expected (7, nil, nil), got : (%#v, %#v, %#v)", idx, result, err)
	}
}

func TestStateStore_KVSList(t *testing.T) {
	s := testStateStore(t)

	// Listing an empty KVS returns nothing
	idx, entries, err := s.KVSList("")
	if idx != 0 || entries != nil || err != nil {
		t.Fatalf("expected (0, nil, nil), got: (%d, %#v, %#v)", idx, entries, err)
	}

	// Create some KVS entries
	testSetKey(t, s, 1, "foo", "foo")
	testSetKey(t, s, 2, "foo/bar", "bar")
	testSetKey(t, s, 3, "foo/bar/zip", "zip")
	testSetKey(t, s, 4, "foo/bar/zip/zorp", "zorp")
	testSetKey(t, s, 5, "foo/bar/baz", "baz")

	// List out all of the keys
	idx, entries, err = s.KVSList("")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}

	// Check that all of the keys were returned
	if n := len(entries); n != 5 {
		t.Fatalf("expected 5 kvs entries, got: %d", n)
	}

	// Try listing with a provided prefix
	idx, entries, err = s.KVSList("foo/bar/zip")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}

	// Check that only the keys in the prefix were returned
	if n := len(entries); n != 2 {
		t.Fatalf("expected 2 kvs entries, got: %d", n)
	}
	if entries[0].Key != "foo/bar/zip" || entries[1].Key != "foo/bar/zip/zorp" {
		t.Fatalf("bad: %#v", entries)
	}

	// Delete a key and make sure the index comes from the tombstone.
	if err := s.KVSDelete(6, "foo/bar/baz"); err != nil {
		t.Fatalf("err: %s", err)
	}
	idx, _, err = s.KVSList("foo/bar/baz")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}

	// Set a different key to bump the index.
	testSetKey(t, s, 7, "some/other/key", "")

	// Make sure we get the right index from the tombstone.
	idx, _, err = s.KVSList("foo/bar/baz")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}

	// Now reap the tombstones and make sure we get the latest index
	// since there are no matching keys.
	if err := s.ReapTombstones(6); err != nil {
		t.Fatalf("err: %s", err)
	}
	idx, _, err = s.KVSList("foo/bar/baz")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 7 {
		t.Fatalf("bad index: %d", idx)
	}

	// List all the keys to make sure the index is also correct.
	idx, _, err = s.KVSList("")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 7 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_KVSListKeys(t *testing.T) {
	s := testStateStore(t)

	// Listing keys with no results returns nil.
	idx, keys, err := s.KVSListKeys("", "")
	if idx != 0 || keys != nil || err != nil {
		t.Fatalf("expected (0, nil, nil), got: (%d, %#v, %#v)", idx, keys, err)
	}

	// Create some keys.
	testSetKey(t, s, 1, "foo", "foo")
	testSetKey(t, s, 2, "foo/bar", "bar")
	testSetKey(t, s, 3, "foo/bar/baz", "baz")
	testSetKey(t, s, 4, "foo/bar/zip", "zip")
	testSetKey(t, s, 5, "foo/bar/zip/zam", "zam")
	testSetKey(t, s, 6, "foo/bar/zip/zorp", "zorp")
	testSetKey(t, s, 7, "some/other/prefix", "nack")

	// List all the keys.
	idx, keys, err = s.KVSListKeys("", "")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if len(keys) != 7 {
		t.Fatalf("bad keys: %#v", keys)
	}
	if idx != 7 {
		t.Fatalf("bad index: %d", idx)
	}

	// Query using a prefix and pass a separator.
	idx, keys, err = s.KVSListKeys("foo/bar/", "/")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if len(keys) != 3 {
		t.Fatalf("bad keys: %#v", keys)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}

	// Subset of the keys was returned.
	expect := []string{"foo/bar/baz", "foo/bar/zip", "foo/bar/zip/"}
	if !reflect.DeepEqual(keys, expect) {
		t.Fatalf("bad keys: %#v", keys)
	}

	// Listing keys with no separator returns everything.
	idx, keys, err = s.KVSListKeys("foo", "")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}
	expect = []string{"foo", "foo/bar", "foo/bar/baz", "foo/bar/zip",
		"foo/bar/zip/zam", "foo/bar/zip/zorp"}
	if !reflect.DeepEqual(keys, expect) {
		t.Fatalf("bad keys: %#v", keys)
	}

	// Delete a key and make sure the index comes from the tombstone.
	if err := s.KVSDelete(8, "foo/bar/baz"); err != nil {
		t.Fatalf("err: %s", err)
	}
	idx, _, err = s.KVSListKeys("foo/bar/baz", "")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 8 {
		t.Fatalf("bad index: %d", idx)
	}

	// Set a different key to bump the index.
	testSetKey(t, s, 9, "some/other/key", "")

	// Make sure the index still comes from the tombstone.
	idx, _, err = s.KVSListKeys("foo/bar/baz", "")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 8 {
		t.Fatalf("bad index: %d", idx)
	}

	// Now reap the tombstones and make sure we get the latest index
	// since there are no matching keys.
	if err := s.ReapTombstones(8); err != nil {
		t.Fatalf("err: %s", err)
	}
	idx, _, err = s.KVSListKeys("foo/bar/baz", "")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 9 {
		t.Fatalf("bad index: %d", idx)
	}

	// List all the keys to make sure the index is also correct.
	idx, _, err = s.KVSListKeys("", "")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 9 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_KVSDelete(t *testing.T) {
	s := testStateStore(t)

	// Create some KV pairs
	testSetKey(t, s, 1, "foo", "foo")
	testSetKey(t, s, 2, "foo/bar", "bar")

	// Call a delete on a specific key
	if err := s.KVSDelete(3, "foo"); err != nil {
		t.Fatalf("err: %s", err)
	}

	// The entry was removed from the state store
	tx := s.db.Txn(false)
	defer tx.Abort()
	e, err := tx.First("kvs", "id", "foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if e != nil {
		t.Fatalf("expected kvs entry to be deleted, got: %#v", e)
	}

	// Try fetching the other keys to ensure they still exist
	e, err = tx.First("kvs", "id", "foo/bar")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if e == nil || string(e.(*structs.DirEntry).Value) != "bar" {
		t.Fatalf("bad kvs entry: %#v", e)
	}

	// Check that the index table was updated
	if idx := s.maxIndex("kvs"); idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}

	// Check that the tombstone was created and that prevents the index
	// from sliding backwards.
	idx, _, err := s.KVSList("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}

	// Now reap the tombstone and watch the index revert to the remaining
	// foo/bar key's index.
	if err := s.ReapTombstones(3); err != nil {
		t.Fatalf("err: %s", err)
	}
	idx, _, err = s.KVSList("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 2 {
		t.Fatalf("bad index: %d", idx)
	}

	// Deleting a nonexistent key should be idempotent and not return an
	// error
	if err := s.KVSDelete(4, "foo"); err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx := s.maxIndex("kvs"); idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_KVSDeleteCAS(t *testing.T) {
	s := testStateStore(t)

	// Create some KV entries
	testSetKey(t, s, 1, "foo", "foo")
	testSetKey(t, s, 2, "bar", "bar")
	testSetKey(t, s, 3, "baz", "baz")

	// Do a CAS delete with an index lower than the entry
	ok, err := s.KVSDeleteCAS(4, 1, "bar")
	if ok || err != nil {
		t.Fatalf("expected (false, nil), got: (%v, %#v)", ok, err)
	}

	// Check that the index is untouched and the entry
	// has not been deleted.
	idx, e, err := s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if e == nil {
		t.Fatalf("expected a kvs entry, got nil")
	}
	if idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}

	// Do another CAS delete, this time with the correct index
	// which should cause the delete to take place.
	ok, err = s.KVSDeleteCAS(4, 2, "bar")
	if !ok || err != nil {
		t.Fatalf("expected (true, nil), got: (%v, %#v)", ok, err)
	}

	// Entry was deleted and index was updated
	idx, e, err = s.KVSGet("bar")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if e != nil {
		t.Fatalf("entry should be deleted")
	}
	if idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}

	// Add another key to bump the index.
	testSetKey(t, s, 5, "some/other/key", "baz")

	// Check that the tombstone was created and that prevents the index
	// from sliding backwards.
	idx, _, err = s.KVSList("bar")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}

	// Now reap the tombstone and watch the index move up to the table
	// index since there are no matching keys.
	if err := s.ReapTombstones(4); err != nil {
		t.Fatalf("err: %s", err)
	}
	idx, _, err = s.KVSList("bar")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}

	// A delete on a nonexistent key should be idempotent and not return an
	// error
	ok, err = s.KVSDeleteCAS(6, 2, "bar")
	if !ok || err != nil {
		t.Fatalf("expected (true, nil), got: (%v, %#v)", ok, err)
	}
	if idx := s.maxIndex("kvs"); idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_KVSSetCAS(t *testing.T) {
	s := testStateStore(t)

	// Doing a CAS with ModifyIndex != 0 and no existing entry
	// is a no-op.
	entry := &structs.DirEntry{
		Key:   "foo",
		Value: []byte("foo"),
		RaftIndex: structs.RaftIndex{
			CreateIndex: 1,
			ModifyIndex: 1,
		},
	}
	ok, err := s.KVSSetCAS(2, entry)
	if ok || err != nil {
		t.Fatalf("expected (false, nil), got: (%#v, %#v)", ok, err)
	}

	// Check that nothing was actually stored
	tx := s.db.Txn(false)
	if e, err := tx.First("kvs", "id", "foo"); e != nil || err != nil {
		t.Fatalf("expected (nil, nil), got: (%#v, %#v)", e, err)
	}
	tx.Abort()

	// Index was not updated
	if idx := s.maxIndex("kvs"); idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}

	// Doing a CAS with a ModifyIndex of zero when no entry exists
	// performs the set and saves into the state store.
	entry = &structs.DirEntry{
		Key:   "foo",
		Value: []byte("foo"),
		RaftIndex: structs.RaftIndex{
			CreateIndex: 0,
			ModifyIndex: 0,
		},
	}
	ok, err = s.KVSSetCAS(2, entry)
	if !ok || err != nil {
		t.Fatalf("expected (true, nil), got: (%#v, %#v)", ok, err)
	}

	// Entry was inserted
	idx, entry, err := s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if string(entry.Value) != "foo" || entry.CreateIndex != 2 || entry.ModifyIndex != 2 {
		t.Fatalf("bad entry: %#v", entry)
	}
	if idx != 2 {
		t.Fatalf("bad index: %d", idx)
	}

	// Doing a CAS with a ModifyIndex of zero when an entry exists does
	// not do anything.
	entry = &structs.DirEntry{
		Key:   "foo",
		Value: []byte("foo"),
		RaftIndex: structs.RaftIndex{
			CreateIndex: 0,
			ModifyIndex: 0,
		},
	}
	ok, err = s.KVSSetCAS(3, entry)
	if ok || err != nil {
		t.Fatalf("expected (false, nil), got: (%#v, %#v)", ok, err)
	}

	// Doing a CAS with a ModifyIndex which does not match the current
	// index does not do anything.
	entry = &structs.DirEntry{
		Key:   "foo",
		Value: []byte("bar"),
		RaftIndex: structs.RaftIndex{
			CreateIndex: 3,
			ModifyIndex: 3,
		},
	}
	ok, err = s.KVSSetCAS(3, entry)
	if ok || err != nil {
		t.Fatalf("expected (false, nil), got: (%#v, %#v)", ok, err)
	}

	// Entry was not updated in the store
	idx, entry, err = s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if string(entry.Value) != "foo" || entry.CreateIndex != 2 || entry.ModifyIndex != 2 {
		t.Fatalf("bad entry: %#v", entry)
	}
	if idx != 2 {
		t.Fatalf("bad index: %d", idx)
	}

	// Doing a CAS with the proper current index should make the
	// modification.
	entry = &structs.DirEntry{
		Key:   "foo",
		Value: []byte("bar"),
		RaftIndex: structs.RaftIndex{
			CreateIndex: 2,
			ModifyIndex: 2,
		},
	}
	ok, err = s.KVSSetCAS(3, entry)
	if !ok || err != nil {
		t.Fatalf("expected (true, nil), got: (%#v, %#v)", ok, err)
	}

	// Entry was updated
	idx, entry, err = s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if string(entry.Value) != "bar" || entry.CreateIndex != 2 || entry.ModifyIndex != 3 {
		t.Fatalf("bad entry: %#v", entry)
	}
	if idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}

	// Attempt to update the session during the CAS.
	entry = &structs.DirEntry{
		Key:     "foo",
		Value:   []byte("zoo"),
		Session: "nope",
		RaftIndex: structs.RaftIndex{
			CreateIndex: 2,
			ModifyIndex: 3,
		},
	}
	ok, err = s.KVSSetCAS(4, entry)
	if !ok || err != nil {
		t.Fatalf("expected (true, nil), got: (%#v, %#v)", ok, err)
	}

	// Entry was updated, but the session should have been ignored.
	idx, entry, err = s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if string(entry.Value) != "zoo" || entry.CreateIndex != 2 || entry.ModifyIndex != 4 ||
		entry.Session != "" {
		t.Fatalf("bad entry: %#v", entry)
	}
	if idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}

	// Now lock it and try the update, which should keep the session.
	testRegisterNode(t, s, 5, "node1")
	session := testUUID()
	if err := s.SessionCreate(6, &structs.Session{ID: session, Node: "node1"}); err != nil {
		t.Fatalf("err: %s", err)
	}
	entry = &structs.DirEntry{
		Key:     "foo",
		Value:   []byte("locked"),
		Session: session,
		RaftIndex: structs.RaftIndex{
			CreateIndex: 2,
			ModifyIndex: 4,
		},
	}
	ok, err = s.KVSLock(6, entry)
	if !ok || err != nil {
		t.Fatalf("didn't get the lock: %v %s", ok, err)
	}
	entry = &structs.DirEntry{
		Key:   "foo",
		Value: []byte("locked"),
		RaftIndex: structs.RaftIndex{
			CreateIndex: 2,
			ModifyIndex: 6,
		},
	}
	ok, err = s.KVSSetCAS(7, entry)
	if !ok || err != nil {
		t.Fatalf("expected (true, nil), got: (%#v, %#v)", ok, err)
	}

	// Entry was updated, and the lock status should have stayed the same.
	idx, entry, err = s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if string(entry.Value) != "locked" || entry.CreateIndex != 2 || entry.ModifyIndex != 7 ||
		entry.Session != session {
		t.Fatalf("bad entry: %#v", entry)
	}
	if idx != 7 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_KVSDeleteTree(t *testing.T) {
	s := testStateStore(t)

	// Create kvs entries in the state store
	testSetKey(t, s, 1, "foo/bar", "bar")
	testSetKey(t, s, 2, "foo/bar/baz", "baz")
	testSetKey(t, s, 3, "foo/bar/zip", "zip")
	testSetKey(t, s, 4, "foo/zorp", "zorp")

	// Calling tree deletion which affects nothing does not
	// modify the table index.
	if err := s.KVSDeleteTree(9, "bar"); err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx := s.maxIndex("kvs"); idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}

	// Call tree deletion with a nested prefix.
	if err := s.KVSDeleteTree(5, "foo/bar"); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Check that all the matching keys were deleted
	tx := s.db.Txn(false)
	defer tx.Abort()

	entries, err := tx.Get("kvs", "id")
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	num := 0
	for entry := entries.Next(); entry != nil; entry = entries.Next() {
		if entry.(*structs.DirEntry).Key != "foo/zorp" {
			t.Fatalf("unexpected kvs entry: %#v", entry)
		}
		num++
	}

	if num != 1 {
		t.Fatalf("expected 1 key, got: %d", num)
	}

	// Index should be updated if modifications are made
	if idx := s.maxIndex("kvs"); idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}

	// Check that the tombstones ware created and that prevents the index
	// from sliding backwards.
	idx, _, err := s.KVSList("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}

	// Now reap the tombstones and watch the index revert to the remaining
	// foo/zorp key's index.
	if err := s.ReapTombstones(5); err != nil {
		t.Fatalf("err: %s", err)
	}
	idx, _, err = s.KVSList("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_KVSLockDelay(t *testing.T) {
	s := testStateStore(t)

	// KVSLockDelay is exercised in the lock/unlock and session invalidation
	// cases below, so we just do a basic check on a nonexistent key here.
	expires := s.KVSLockDelay("/not/there")
	if expires.After(time.Now()) {
		t.Fatalf("bad: %v", expires)
	}
}

func TestStateStore_KVSLock(t *testing.T) {
	s := testStateStore(t)

	// Lock with no session should fail.
	ok, err := s.KVSLock(0, &structs.DirEntry{Key: "foo", Value: []byte("foo")})
	if ok || err == nil || !strings.Contains(err.Error(), "missing session") {
		t.Fatalf("didn't detect missing session: %v %s", ok, err)
	}

	// Now try with a bogus session.
	ok, err = s.KVSLock(1, &structs.DirEntry{Key: "foo", Value: []byte("foo"), Session: testUUID()})
	if ok || err == nil || !strings.Contains(err.Error(), "invalid session") {
		t.Fatalf("didn't detect invalid session: %v %s", ok, err)
	}

	// Make a real session.
	testRegisterNode(t, s, 2, "node1")
	session1 := testUUID()
	if err := s.SessionCreate(3, &structs.Session{ID: session1, Node: "node1"}); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Lock and make the key at the same time.
	ok, err = s.KVSLock(4, &structs.DirEntry{Key: "foo", Value: []byte("foo"), Session: session1})
	if !ok || err != nil {
		t.Fatalf("didn't get the lock: %v %s", ok, err)
	}

	// Make sure the indexes got set properly.
	idx, result, err := s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.LockIndex != 1 || result.CreateIndex != 4 || result.ModifyIndex != 4 ||
		string(result.Value) != "foo" {
		t.Fatalf("bad entry: %#v", result)
	}
	if idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}

	// Re-locking with the same session should update the value and report
	// success.
	ok, err = s.KVSLock(5, &structs.DirEntry{Key: "foo", Value: []byte("bar"), Session: session1})
	if !ok || err != nil {
		t.Fatalf("didn't handle locking an already-locked key: %v %s", ok, err)
	}

	// Make sure the indexes got set properly, note that the lock index
	// won't go up since we didn't lock it again.
	idx, result, err = s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.LockIndex != 1 || result.CreateIndex != 4 || result.ModifyIndex != 5 ||
		string(result.Value) != "bar" {
		t.Fatalf("bad entry: %#v", result)
	}
	if idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}

	// Unlock and the re-lock.
	ok, err = s.KVSUnlock(6, &structs.DirEntry{Key: "foo", Value: []byte("baz"), Session: session1})
	if !ok || err != nil {
		t.Fatalf("didn't handle unlocking a locked key: %v %s", ok, err)
	}
	ok, err = s.KVSLock(7, &structs.DirEntry{Key: "foo", Value: []byte("zoo"), Session: session1})
	if !ok || err != nil {
		t.Fatalf("didn't get the lock: %v %s", ok, err)
	}

	// Make sure the indexes got set properly.
	idx, result, err = s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.LockIndex != 2 || result.CreateIndex != 4 || result.ModifyIndex != 7 ||
		string(result.Value) != "zoo" {
		t.Fatalf("bad entry: %#v", result)
	}
	if idx != 7 {
		t.Fatalf("bad index: %d", idx)
	}

	// Lock an existing key.
	testSetKey(t, s, 8, "bar", "bar")
	ok, err = s.KVSLock(9, &structs.DirEntry{Key: "bar", Value: []byte("xxx"), Session: session1})
	if !ok || err != nil {
		t.Fatalf("didn't get the lock: %v %s", ok, err)
	}

	// Make sure the indexes got set properly.
	idx, result, err = s.KVSGet("bar")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.LockIndex != 1 || result.CreateIndex != 8 || result.ModifyIndex != 9 ||
		string(result.Value) != "xxx" {
		t.Fatalf("bad entry: %#v", result)
	}
	if idx != 9 {
		t.Fatalf("bad index: %d", idx)
	}

	// Attempting a re-lock with a different session should also fail.
	session2 := testUUID()
	if err := s.SessionCreate(10, &structs.Session{ID: session2, Node: "node1"}); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Re-locking should not return an error, but will report that it didn't
	// get the lock.
	ok, err = s.KVSLock(11, &structs.DirEntry{Key: "bar", Value: []byte("nope"), Session: session2})
	if ok || err != nil {
		t.Fatalf("didn't handle locking an already-locked key: %v %s", ok, err)
	}

	// Make sure the indexes didn't update.
	idx, result, err = s.KVSGet("bar")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.LockIndex != 1 || result.CreateIndex != 8 || result.ModifyIndex != 9 ||
		string(result.Value) != "xxx" {
		t.Fatalf("bad entry: %#v", result)
	}
	if idx != 9 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_KVSUnlock(t *testing.T) {
	s := testStateStore(t)

	// Unlock with no session should fail.
	ok, err := s.KVSUnlock(0, &structs.DirEntry{Key: "foo", Value: []byte("bar")})
	if ok || err == nil || !strings.Contains(err.Error(), "missing session") {
		t.Fatalf("didn't detect missing session: %v %s", ok, err)
	}

	// Make a real session.
	testRegisterNode(t, s, 1, "node1")
	session1 := testUUID()
	if err := s.SessionCreate(2, &structs.Session{ID: session1, Node: "node1"}); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Unlock with a real session but no key should not return an error, but
	// will report it didn't unlock anything.
	ok, err = s.KVSUnlock(3, &structs.DirEntry{Key: "foo", Value: []byte("bar"), Session: session1})
	if ok || err != nil {
		t.Fatalf("didn't handle unlocking a missing key: %v %s", ok, err)
	}

	// Make a key and unlock it, without it being locked.
	testSetKey(t, s, 4, "foo", "bar")
	ok, err = s.KVSUnlock(5, &structs.DirEntry{Key: "foo", Value: []byte("baz"), Session: session1})
	if ok || err != nil {
		t.Fatalf("didn't handle unlocking a non-locked key: %v %s", ok, err)
	}

	// Make sure the indexes didn't update.
	idx, result, err := s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.LockIndex != 0 || result.CreateIndex != 4 || result.ModifyIndex != 4 ||
		string(result.Value) != "bar" {
		t.Fatalf("bad entry: %#v", result)
	}
	if idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}

	// Lock it with the first session.
	ok, err = s.KVSLock(6, &structs.DirEntry{Key: "foo", Value: []byte("bar"), Session: session1})
	if !ok || err != nil {
		t.Fatalf("didn't get the lock: %v %s", ok, err)
	}

	// Attempt an unlock with another session.
	session2 := testUUID()
	if err := s.SessionCreate(7, &structs.Session{ID: session2, Node: "node1"}); err != nil {
		t.Fatalf("err: %s", err)
	}
	ok, err = s.KVSUnlock(8, &structs.DirEntry{Key: "foo", Value: []byte("zoo"), Session: session2})
	if ok || err != nil {
		t.Fatalf("didn't handle unlocking with the wrong session: %v %s", ok, err)
	}

	// Make sure the indexes didn't update.
	idx, result, err = s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.LockIndex != 1 || result.CreateIndex != 4 || result.ModifyIndex != 6 ||
		string(result.Value) != "bar" {
		t.Fatalf("bad entry: %#v", result)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}

	// Now do the unlock with the correct session.
	ok, err = s.KVSUnlock(9, &structs.DirEntry{Key: "foo", Value: []byte("zoo"), Session: session1})
	if !ok || err != nil {
		t.Fatalf("didn't handle unlocking with the correct session: %v %s", ok, err)
	}

	// Make sure the indexes got set properly.
	idx, result, err = s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.LockIndex != 1 || result.CreateIndex != 4 || result.ModifyIndex != 9 ||
		string(result.Value) != "zoo" {
		t.Fatalf("bad entry: %#v", result)
	}
	if idx != 9 {
		t.Fatalf("bad index: %d", idx)
	}

	// Unlocking again should fail and not change anything.
	ok, err = s.KVSUnlock(10, &structs.DirEntry{Key: "foo", Value: []byte("nope"), Session: session1})
	if ok || err != nil {
		t.Fatalf("didn't handle unlocking with the previous session: %v %s", ok, err)
	}

	// Make sure the indexes didn't update.
	idx, result, err = s.KVSGet("foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.LockIndex != 1 || result.CreateIndex != 4 || result.ModifyIndex != 9 ||
		string(result.Value) != "zoo" {
		t.Fatalf("bad entry: %#v", result)
	}
	if idx != 9 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_KVS_Snapshot_Restore(t *testing.T) {
	s := testStateStore(t)

	// Build up some entries to seed.
	entries := structs.DirEntries{
		&structs.DirEntry{
			Key:   "aaa",
			Flags: 23,
			Value: []byte("hello"),
		},
		&structs.DirEntry{
			Key:   "bar/a",
			Value: []byte("one"),
		},
		&structs.DirEntry{
			Key:   "bar/b",
			Value: []byte("two"),
		},
		&structs.DirEntry{
			Key:   "bar/c",
			Value: []byte("three"),
		},
	}
	for i, entry := range entries {
		if err := s.KVSSet(uint64(i+1), entry); err != nil {
			t.Fatalf("err: %s", err)
		}
	}

	// Make a node and session so we can test a locked key.
	testRegisterNode(t, s, 5, "node1")
	session := testUUID()
	if err := s.SessionCreate(6, &structs.Session{ID: session, Node: "node1"}); err != nil {
		t.Fatalf("err: %s", err)
	}
	entries[3].Session = session
	if ok, err := s.KVSLock(7, entries[3]); !ok || err != nil {
		t.Fatalf("didn't get the lock: %v %s", ok, err)
	}

	// This is required for the compare later.
	entries[3].LockIndex = 1

	// Snapshot the keys.
	snap := s.Snapshot()
	defer snap.Close()

	// Alter the real state store.
	if err := s.KVSSet(8, &structs.DirEntry{Key: "aaa", Value: []byte("nope")}); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Verify the snapshot.
	if idx := snap.LastIndex(); idx != 7 {
		t.Fatalf("bad index: %d", idx)
	}
	iter, err := snap.KVs()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	var dump structs.DirEntries
	for entry := iter.Next(); entry != nil; entry = iter.Next() {
		dump = append(dump, entry.(*structs.DirEntry))
	}
	if !reflect.DeepEqual(dump, entries) {
		t.Fatalf("bad: %#v", dump)
	}

	// Restore the values into a new state store.
	func() {
		s := testStateStore(t)
		restore := s.Restore()
		for _, entry := range dump {
			if err := restore.KVS(entry); err != nil {
				t.Fatalf("err: %s", err)
			}
		}
		restore.Commit()

		// Read the restored keys back out and verify they match.
		idx, res, err := s.KVSList("")
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 7 {
			t.Fatalf("bad index: %d", idx)
		}
		if !reflect.DeepEqual(res, entries) {
			t.Fatalf("bad: %#v", res)
		}

		// Check that the index was updated.
		if idx := s.maxIndex("kvs"); idx != 7 {
			t.Fatalf("bad index: %d", idx)
		}
	}()
}

func TestStateStore_KVS_Watches(t *testing.T) {
	s := testStateStore(t)

	// This is used when locking down below.
	testRegisterNode(t, s, 1, "node1")
	session := testUUID()
	if err := s.SessionCreate(2, &structs.Session{ID: session, Node: "node1"}); err != nil {
		t.Fatalf("err: %s", err)
	}

	// An empty prefix watch should hit on all KVS ops, and some other
	// prefix should not be affected ever. We also add a positive prefix
	// match.
	verifyWatch(t, s.GetKVSWatch(""), func() {
		verifyWatch(t, s.GetKVSWatch("a"), func() {
			verifyNoWatch(t, s.GetKVSWatch("/nope"), func() {
				if err := s.KVSSet(1, &structs.DirEntry{Key: "aaa"}); err != nil {
					t.Fatalf("err: %s", err)
				}
			})
		})
	})
	verifyWatch(t, s.GetKVSWatch(""), func() {
		verifyWatch(t, s.GetKVSWatch("a"), func() {
			verifyNoWatch(t, s.GetKVSWatch("/nope"), func() {
				if err := s.KVSSet(2, &structs.DirEntry{Key: "aaa"}); err != nil {
					t.Fatalf("err: %s", err)
				}
			})
		})
	})

	// Restore just fires off a top-level watch, so we should get hits on
	// any prefix, including ones for keys that aren't in there.
	verifyWatch(t, s.GetKVSWatch(""), func() {
		verifyWatch(t, s.GetKVSWatch("b"), func() {
			verifyWatch(t, s.GetKVSWatch("/nope"), func() {
				restore := s.Restore()
				if err := restore.KVS(&structs.DirEntry{Key: "bbb"}); err != nil {
					t.Fatalf("err: %s", err)
				}
				restore.Commit()
			})
		})
	})

	verifyWatch(t, s.GetKVSWatch(""), func() {
		verifyWatch(t, s.GetKVSWatch("a"), func() {
			verifyNoWatch(t, s.GetKVSWatch("/nope"), func() {
				if err := s.KVSDelete(3, "aaa"); err != nil {
					t.Fatalf("err: %s", err)
				}
			})
		})
	})
	verifyWatch(t, s.GetKVSWatch(""), func() {
		verifyWatch(t, s.GetKVSWatch("a"), func() {
			verifyNoWatch(t, s.GetKVSWatch("/nope"), func() {
				if ok, err := s.KVSSetCAS(4, &structs.DirEntry{Key: "aaa"}); !ok || err != nil {
					t.Fatalf("ok: %v err: %s", ok, err)
				}
			})
		})
	})
	verifyWatch(t, s.GetKVSWatch(""), func() {
		verifyWatch(t, s.GetKVSWatch("a"), func() {
			verifyNoWatch(t, s.GetKVSWatch("/nope"), func() {
				if ok, err := s.KVSLock(5, &structs.DirEntry{Key: "aaa", Session: session}); !ok || err != nil {
					t.Fatalf("ok: %v err: %s", ok, err)
				}
			})
		})
	})
	verifyWatch(t, s.GetKVSWatch(""), func() {
		verifyWatch(t, s.GetKVSWatch("a"), func() {
			verifyNoWatch(t, s.GetKVSWatch("/nope"), func() {
				if ok, err := s.KVSUnlock(6, &structs.DirEntry{Key: "aaa", Session: session}); !ok || err != nil {
					t.Fatalf("ok: %v err: %s", ok, err)
				}
			})
		})
	})
	verifyWatch(t, s.GetKVSWatch(""), func() {
		verifyWatch(t, s.GetKVSWatch("a"), func() {
			verifyNoWatch(t, s.GetKVSWatch("/nope"), func() {
				if err := s.KVSDeleteTree(7, "aaa"); err != nil {
					t.Fatalf("err: %s", err)
				}
			})
		})
	})

	// A delete tree operation at the top level will notify all the watches.
	verifyWatch(t, s.GetKVSWatch(""), func() {
		verifyWatch(t, s.GetKVSWatch("a"), func() {
			verifyWatch(t, s.GetKVSWatch("/nope"), func() {
				if err := s.KVSDeleteTree(8, ""); err != nil {
					t.Fatalf("err: %s", err)
				}
			})
		})
	})

	// Create a more interesting tree.
	testSetKey(t, s, 9, "foo/bar", "bar")
	testSetKey(t, s, 10, "foo/bar/baz", "baz")
	testSetKey(t, s, 11, "foo/bar/zip", "zip")
	testSetKey(t, s, 12, "foo/zorp", "zorp")

	// Deleting just the foo/bar key should not trigger watches on the
	// children.
	verifyWatch(t, s.GetKVSWatch("foo/bar"), func() {
		verifyNoWatch(t, s.GetKVSWatch("foo/bar/baz"), func() {
			verifyNoWatch(t, s.GetKVSWatch("foo/bar/zip"), func() {
				if err := s.KVSDelete(13, "foo/bar"); err != nil {
					t.Fatalf("err: %s", err)
				}
			})
		})
	})

	// But a delete tree from that point should notify the whole subtree,
	// even for keys that don't exist.
	verifyWatch(t, s.GetKVSWatch("foo/bar"), func() {
		verifyWatch(t, s.GetKVSWatch("foo/bar/baz"), func() {
			verifyWatch(t, s.GetKVSWatch("foo/bar/zip"), func() {
				verifyWatch(t, s.GetKVSWatch("foo/bar/uh/nope"), func() {
					if err := s.KVSDeleteTree(14, "foo/bar"); err != nil {
						t.Fatalf("err: %s", err)
					}
				})
			})
		})
	})
}

func TestStateStore_Tombstone_Snapshot_Restore(t *testing.T) {
	s := testStateStore(t)

	// Insert a key and then delete it to create a tombstone.
	testSetKey(t, s, 1, "foo/bar", "bar")
	testSetKey(t, s, 2, "foo/bar/baz", "bar")
	testSetKey(t, s, 3, "foo/bar/zoo", "bar")
	if err := s.KVSDelete(4, "foo/bar"); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Snapshot the Tombstones.
	snap := s.Snapshot()
	defer snap.Close()

	// Alter the real state store.
	if err := s.ReapTombstones(4); err != nil {
		t.Fatalf("err: %s", err)
	}
	idx, _, err := s.KVSList("foo/bar")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}

	// Verify the snapshot.
	stones, err := snap.Tombstones()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	var dump []*Tombstone
	for stone := stones.Next(); stone != nil; stone = stones.Next() {
		dump = append(dump, stone.(*Tombstone))
	}
	if len(dump) != 1 {
		t.Fatalf("bad %#v", dump)
	}
	stone := dump[0]
	if stone.Key != "foo/bar" || stone.Index != 4 {
		t.Fatalf("bad: %#v", stone)
	}

	// Restore the values into a new state store.
	func() {
		s := testStateStore(t)
		restore := s.Restore()
		for _, stone := range dump {
			if err := restore.Tombstone(stone); err != nil {
				t.Fatalf("err: %s", err)
			}
		}
		restore.Commit()

		// See if the stone works properly in a list query.
		idx, _, err := s.KVSList("foo/bar")
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 4 {
			t.Fatalf("bad index: %d", idx)
		}

		// Make sure it reaps correctly. We should still get a 4 for
		// the index here because it will be using the last index from
		// the tombstone table.
		if err := s.ReapTombstones(4); err != nil {
			t.Fatalf("err: %s", err)
		}
		idx, _, err = s.KVSList("foo/bar")
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 4 {
			t.Fatalf("bad index: %d", idx)
		}

		// But make sure the tombstone is actually gone.
		snap := s.Snapshot()
		defer snap.Close()
		stones, err := snap.Tombstones()
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if stones.Next() != nil {
			t.Fatalf("unexpected extra tombstones")
		}
	}()
}

func TestStateStore_SessionCreate_SessionGet(t *testing.T) {
	s := testStateStore(t)

	// SessionGet returns nil if the session doesn't exist
	idx, session, err := s.SessionGet(testUUID())
	if session != nil || err != nil {
		t.Fatalf("expected (nil, nil), got: (%#v, %#v)", session, err)
	}
	if idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}

	// Registering without a session ID is disallowed
	err = s.SessionCreate(1, &structs.Session{})
	if err != ErrMissingSessionID {
		t.Fatalf("expected %#v, got: %#v", ErrMissingSessionID, err)
	}

	// Invalid session behavior throws error
	sess := &structs.Session{
		ID:       testUUID(),
		Behavior: "nope",
	}
	err = s.SessionCreate(1, sess)
	if err == nil || !strings.Contains(err.Error(), "session behavior") {
		t.Fatalf("expected session behavior error, got: %#v", err)
	}

	// Registering with an unknown node is disallowed
	sess = &structs.Session{ID: testUUID()}
	if err := s.SessionCreate(1, sess); err != ErrMissingNode {
		t.Fatalf("expected %#v, got: %#v", ErrMissingNode, err)
	}

	// None of the errored operations modified the index
	if idx := s.maxIndex("sessions"); idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}

	// Valid session is able to register
	testRegisterNode(t, s, 1, "node1")
	sess = &structs.Session{
		ID:   testUUID(),
		Node: "node1",
	}
	if err := s.SessionCreate(2, sess); err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx := s.maxIndex("sessions"); idx != 2 {
		t.Fatalf("bad index: %s", err)
	}

	// Retrieve the session again
	idx, session, err = s.SessionGet(sess.ID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 2 {
		t.Fatalf("bad index: %d", idx)
	}

	// Ensure the session looks correct and was assigned the
	// proper default value for session behavior.
	expect := &structs.Session{
		ID:       sess.ID,
		Behavior: structs.SessionKeysRelease,
		Node:     "node1",
		RaftIndex: structs.RaftIndex{
			CreateIndex: 2,
			ModifyIndex: 2,
		},
	}
	if !reflect.DeepEqual(expect, session) {
		t.Fatalf("bad session: %#v", session)
	}

	// Registering with a non-existent check is disallowed
	sess = &structs.Session{
		ID:     testUUID(),
		Node:   "node1",
		Checks: []string{"check1"},
	}
	err = s.SessionCreate(3, sess)
	if err == nil || !strings.Contains(err.Error(), "Missing check") {
		t.Fatalf("expected missing check error, got: %#v", err)
	}

	// Registering with a critical check is disallowed
	testRegisterCheck(t, s, 3, "node1", "", "check1", structs.HealthCritical)
	err = s.SessionCreate(4, sess)
	if err == nil || !strings.Contains(err.Error(), structs.HealthCritical) {
		t.Fatalf("expected critical state error, got: %#v", err)
	}

	// Registering with a healthy check succeeds
	testRegisterCheck(t, s, 4, "node1", "", "check1", structs.HealthPassing)
	if err := s.SessionCreate(5, sess); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Register a session against two checks.
	testRegisterCheck(t, s, 5, "node1", "", "check2", structs.HealthPassing)
	sess2 := &structs.Session{
		ID:     testUUID(),
		Node:   "node1",
		Checks: []string{"check1", "check2"},
	}
	if err := s.SessionCreate(6, sess2); err != nil {
		t.Fatalf("err: %s", err)
	}

	tx := s.db.Txn(false)
	defer tx.Abort()

	// Check mappings were inserted
	{
		check, err := tx.First("session_checks", "session", sess.ID)
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if check == nil {
			t.Fatalf("missing session check")
		}
		expectCheck := &sessionCheck{
			Node:    "node1",
			CheckID: "check1",
			Session: sess.ID,
		}
		if actual := check.(*sessionCheck); !reflect.DeepEqual(actual, expectCheck) {
			t.Fatalf("expected %#v, got: %#v", expectCheck, actual)
		}
	}
	checks, err := tx.Get("session_checks", "session", sess2.ID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	for i, check := 0, checks.Next(); check != nil; i, check = i+1, checks.Next() {
		expectCheck := &sessionCheck{
			Node:    "node1",
			CheckID: fmt.Sprintf("check%d", i+1),
			Session: sess2.ID,
		}
		if actual := check.(*sessionCheck); !reflect.DeepEqual(actual, expectCheck) {
			t.Fatalf("expected %#v, got: %#v", expectCheck, actual)
		}
	}

	// Pulling a nonexistent session gives the table index.
	idx, session, err = s.SessionGet(testUUID())
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if session != nil {
		t.Fatalf("expected not to get a session: %v", session)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TegstStateStore_SessionList(t *testing.T) {
	s := testStateStore(t)

	// Listing when no sessions exist returns nil
	idx, res, err := s.SessionList()
	if idx != 0 || res != nil || err != nil {
		t.Fatalf("expected (0, nil, nil), got: (%d, %#v, %#v)", idx, res, err)
	}

	// Register some nodes
	testRegisterNode(t, s, 1, "node1")
	testRegisterNode(t, s, 2, "node2")
	testRegisterNode(t, s, 3, "node3")

	// Create some sessions in the state store
	sessions := structs.Sessions{
		&structs.Session{
			ID:       testUUID(),
			Node:     "node1",
			Behavior: structs.SessionKeysDelete,
		},
		&structs.Session{
			ID:       testUUID(),
			Node:     "node2",
			Behavior: structs.SessionKeysRelease,
		},
		&structs.Session{
			ID:       testUUID(),
			Node:     "node3",
			Behavior: structs.SessionKeysDelete,
		},
	}
	for i, session := range sessions {
		if err := s.SessionCreate(uint64(4+i), session); err != nil {
			t.Fatalf("err: %s", err)
		}
	}

	// List out all of the sessions
	idx, sessionList, err := s.SessionList()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(sessionList, sessions) {
		t.Fatalf("bad: %#v", sessions)
	}
}

func TestStateStore_NodeSessions(t *testing.T) {
	s := testStateStore(t)

	// Listing sessions with no results returns nil
	idx, res, err := s.NodeSessions("node1")
	if idx != 0 || res != nil || err != nil {
		t.Fatalf("expected (0, nil, nil), got: (%d, %#v, %#v)", idx, res, err)
	}

	// Create the nodes
	testRegisterNode(t, s, 1, "node1")
	testRegisterNode(t, s, 2, "node2")

	// Register some sessions with the nodes
	sessions1 := structs.Sessions{
		&structs.Session{
			ID:   testUUID(),
			Node: "node1",
		},
		&structs.Session{
			ID:   testUUID(),
			Node: "node1",
		},
	}
	sessions2 := []*structs.Session{
		&structs.Session{
			ID:   testUUID(),
			Node: "node2",
		},
		&structs.Session{
			ID:   testUUID(),
			Node: "node2",
		},
	}
	for i, sess := range append(sessions1, sessions2...) {
		if err := s.SessionCreate(uint64(3+i), sess); err != nil {
			t.Fatalf("err: %s", err)
		}
	}

	// Query all of the sessions associated with a specific
	// node in the state store.
	idx, res, err = s.NodeSessions("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if len(res) != len(sessions1) {
		t.Fatalf("bad: %#v", res)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}

	idx, res, err = s.NodeSessions("node2")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if len(res) != len(sessions2) {
		t.Fatalf("bad: %#v", res)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_SessionDestroy(t *testing.T) {
	s := testStateStore(t)

	// Session destroy is idempotent and returns no error
	// if the session doesn't exist.
	if err := s.SessionDestroy(1, testUUID()); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Ensure the index was not updated if nothing was destroyed.
	if idx := s.maxIndex("sessions"); idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}

	// Register a node.
	testRegisterNode(t, s, 1, "node1")

	// Register a new session
	sess := &structs.Session{
		ID:   testUUID(),
		Node: "node1",
	}
	if err := s.SessionCreate(2, sess); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Destroy the session.
	if err := s.SessionDestroy(3, sess.ID); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Check that the index was updated
	if idx := s.maxIndex("sessions"); idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}

	// Make sure the session is really gone.
	tx := s.db.Txn(false)
	sessions, err := tx.Get("sessions", "id")
	if err != nil || sessions.Next() != nil {
		t.Fatalf("session should not exist")
	}
	tx.Abort()
}

func TestStateStore_Session_Snapshot_Restore(t *testing.T) {
	s := testStateStore(t)

	// Register some nodes and checks.
	testRegisterNode(t, s, 1, "node1")
	testRegisterNode(t, s, 2, "node2")
	testRegisterNode(t, s, 3, "node3")
	testRegisterCheck(t, s, 4, "node1", "", "check1", structs.HealthPassing)

	// Create some sessions in the state store.
	session1 := testUUID()
	sessions := structs.Sessions{
		&structs.Session{
			ID:       session1,
			Node:     "node1",
			Behavior: structs.SessionKeysDelete,
			Checks:   []string{"check1"},
		},
		&structs.Session{
			ID:        testUUID(),
			Node:      "node2",
			Behavior:  structs.SessionKeysRelease,
			LockDelay: 10 * time.Second,
		},
		&structs.Session{
			ID:       testUUID(),
			Node:     "node3",
			Behavior: structs.SessionKeysDelete,
			TTL:      "1.5s",
		},
	}
	for i, session := range sessions {
		if err := s.SessionCreate(uint64(5+i), session); err != nil {
			t.Fatalf("err: %s", err)
		}
	}

	// Snapshot the sessions.
	snap := s.Snapshot()
	defer snap.Close()

	// Alter the real state store.
	if err := s.SessionDestroy(8, session1); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Verify the snapshot.
	if idx := snap.LastIndex(); idx != 7 {
		t.Fatalf("bad index: %d", idx)
	}
	iter, err := snap.Sessions()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	var dump structs.Sessions
	for session := iter.Next(); session != nil; session = iter.Next() {
		sess := session.(*structs.Session)
		dump = append(dump, sess)

		found := false
		for i, _ := range sessions {
			if sess.ID == sessions[i].ID {
				if !reflect.DeepEqual(sess, sessions[i]) {
					t.Fatalf("bad: %#v", sess)
				}
				found = true
			}
		}
		if !found {
			t.Fatalf("bad: %#v", sess)
		}
	}

	// Restore the sessions into a new state store.
	func() {
		s := testStateStore(t)
		restore := s.Restore()
		for _, session := range dump {
			if err := restore.Session(session); err != nil {
				t.Fatalf("err: %s", err)
			}
		}
		restore.Commit()

		// Read the restored sessions back out and verify that they
		// match.
		idx, res, err := s.SessionList()
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 7 {
			t.Fatalf("bad index: %d", idx)
		}
		for _, sess := range res {
			found := false
			for i, _ := range sessions {
				if sess.ID == sessions[i].ID {
					if !reflect.DeepEqual(sess, sessions[i]) {
						t.Fatalf("bad: %#v", sess)
					}
					found = true
				}
			}
			if !found {
				t.Fatalf("bad: %#v", sess)
			}
		}

		// Check that the index was updated.
		if idx := s.maxIndex("sessions"); idx != 7 {
			t.Fatalf("bad index: %d", idx)
		}

		// Manually verify that the session check mapping got restored.
		tx := s.db.Txn(false)
		defer tx.Abort()

		check, err := tx.First("session_checks", "session", session1)
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if check == nil {
			t.Fatalf("missing session check")
		}
		expectCheck := &sessionCheck{
			Node:    "node1",
			CheckID: "check1",
			Session: session1,
		}
		if actual := check.(*sessionCheck); !reflect.DeepEqual(actual, expectCheck) {
			t.Fatalf("expected %#v, got: %#v", expectCheck, actual)
		}
	}()
}

func TestStateStore_Session_Watches(t *testing.T) {
	s := testStateStore(t)

	// Register a test node.
	testRegisterNode(t, s, 1, "node1")

	// This just covers the basics. The session invalidation tests above
	// cover the more nuanced multiple table watches.
	session := testUUID()
	verifyWatch(t, s.getTableWatch("sessions"), func() {
		sess := &structs.Session{
			ID:       session,
			Node:     "node1",
			Behavior: structs.SessionKeysDelete,
		}
		if err := s.SessionCreate(2, sess); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
	verifyWatch(t, s.getTableWatch("sessions"), func() {
		if err := s.SessionDestroy(3, session); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
	verifyWatch(t, s.getTableWatch("sessions"), func() {
		restore := s.Restore()
		sess := &structs.Session{
			ID:       session,
			Node:     "node1",
			Behavior: structs.SessionKeysDelete,
		}
		if err := restore.Session(sess); err != nil {
			t.Fatalf("err: %s", err)
		}
		restore.Commit()
	})
}

func TestStateStore_Session_Invalidate_DeleteNode(t *testing.T) {
	s := testStateStore(t)

	// Set up our test environment.
	if err := s.EnsureNode(3, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	session := &structs.Session{
		ID:   testUUID(),
		Node: "foo",
	}
	if err := s.SessionCreate(14, session); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Delete the node and make sure the watches fire.
	verifyWatch(t, s.getTableWatch("sessions"), func() {
		verifyWatch(t, s.getTableWatch("nodes"), func() {
			if err := s.DeleteNode(15, "foo"); err != nil {
				t.Fatalf("err: %v", err)
			}
		})
	})

	// Lookup by ID, should be nil.
	idx, s2, err := s.SessionGet(session.ID)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s2 != nil {
		t.Fatalf("session should be invalidated")
	}
	if idx != 15 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_Session_Invalidate_DeleteService(t *testing.T) {
	s := testStateStore(t)

	// Set up our test environment.
	if err := s.EnsureNode(11, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := s.EnsureService(12, "foo", &structs.NodeService{ID: "api", Service: "api", Tags: nil, Address: "", Port: 5000}); err != nil {
		t.Fatalf("err: %v", err)
	}
	check := &structs.HealthCheck{
		Node:      "foo",
		CheckID:   "api",
		Name:      "Can connect",
		Status:    structs.HealthPassing,
		ServiceID: "api",
	}
	if err := s.EnsureCheck(13, check); err != nil {
		t.Fatalf("err: %v", err)
	}
	session := &structs.Session{
		ID:     testUUID(),
		Node:   "foo",
		Checks: []string{"api"},
	}
	if err := s.SessionCreate(14, session); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Delete the service and make sure the watches fire.
	verifyWatch(t, s.getTableWatch("sessions"), func() {
		verifyWatch(t, s.getTableWatch("services"), func() {
			verifyWatch(t, s.getTableWatch("checks"), func() {
				if err := s.DeleteService(15, "foo", "api"); err != nil {
					t.Fatalf("err: %v", err)
				}
			})
		})
	})

	// Lookup by ID, should be nil.
	idx, s2, err := s.SessionGet(session.ID)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s2 != nil {
		t.Fatalf("session should be invalidated")
	}
	if idx != 15 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_Session_Invalidate_Critical_Check(t *testing.T) {
	s := testStateStore(t)

	// Set up our test environment.
	if err := s.EnsureNode(3, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	check := &structs.HealthCheck{
		Node:    "foo",
		CheckID: "bar",
		Status:  structs.HealthPassing,
	}
	if err := s.EnsureCheck(13, check); err != nil {
		t.Fatalf("err: %v", err)
	}
	session := &structs.Session{
		ID:     testUUID(),
		Node:   "foo",
		Checks: []string{"bar"},
	}
	if err := s.SessionCreate(14, session); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Invalidate the check and make sure the watches fire.
	verifyWatch(t, s.getTableWatch("sessions"), func() {
		verifyWatch(t, s.getTableWatch("checks"), func() {
			check.Status = structs.HealthCritical
			if err := s.EnsureCheck(15, check); err != nil {
				t.Fatalf("err: %v", err)
			}
		})
	})

	// Lookup by ID, should be nil.
	idx, s2, err := s.SessionGet(session.ID)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s2 != nil {
		t.Fatalf("session should be invalidated")
	}
	if idx != 15 {
		t.Fatalf("bad index: %d", idx)
	}
}

func TestStateStore_Session_Invalidate_DeleteCheck(t *testing.T) {
	s := testStateStore(t)

	// Set up our test environment.
	if err := s.EnsureNode(3, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	check := &structs.HealthCheck{
		Node:    "foo",
		CheckID: "bar",
		Status:  structs.HealthPassing,
	}
	if err := s.EnsureCheck(13, check); err != nil {
		t.Fatalf("err: %v", err)
	}
	session := &structs.Session{
		ID:     testUUID(),
		Node:   "foo",
		Checks: []string{"bar"},
	}
	if err := s.SessionCreate(14, session); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Delete the check and make sure the watches fire.
	verifyWatch(t, s.getTableWatch("sessions"), func() {
		verifyWatch(t, s.getTableWatch("checks"), func() {
			if err := s.DeleteCheck(15, "foo", "bar"); err != nil {
				t.Fatalf("err: %v", err)
			}
		})
	})

	// Lookup by ID, should be nil.
	idx, s2, err := s.SessionGet(session.ID)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s2 != nil {
		t.Fatalf("session should be invalidated")
	}
	if idx != 15 {
		t.Fatalf("bad index: %d", idx)
	}

	// Manually make sure the session checks mapping is clear.
	tx := s.db.Txn(false)
	mapping, err := tx.First("session_checks", "session", session.ID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if mapping != nil {
		t.Fatalf("unexpected session check")
	}
	tx.Abort()
}

func TestStateStore_Session_Invalidate_Key_Unlock_Behavior(t *testing.T) {
	s := testStateStore(t)

	// Set up our test environment.
	if err := s.EnsureNode(3, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	session := &structs.Session{
		ID:        testUUID(),
		Node:      "foo",
		LockDelay: 50 * time.Millisecond,
	}
	if err := s.SessionCreate(4, session); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Lock a key with the session.
	d := &structs.DirEntry{
		Key:     "/foo",
		Flags:   42,
		Value:   []byte("test"),
		Session: session.ID,
	}
	ok, err := s.KVSLock(5, d)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if !ok {
		t.Fatalf("unexpected fail")
	}

	// Delete the node and make sure the watches fire.
	verifyWatch(t, s.getTableWatch("sessions"), func() {
		verifyWatch(t, s.getTableWatch("nodes"), func() {
			verifyWatch(t, s.GetKVSWatch("/f"), func() {
				if err := s.DeleteNode(6, "foo"); err != nil {
					t.Fatalf("err: %v", err)
				}
			})
		})
	})

	// Lookup by ID, should be nil.
	idx, s2, err := s.SessionGet(session.ID)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s2 != nil {
		t.Fatalf("session should be invalidated")
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}

	// Key should be unlocked.
	idx, d2, err := s.KVSGet("/foo")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if d2.ModifyIndex != 6 {
		t.Fatalf("bad index: %v", d2.ModifyIndex)
	}
	if d2.LockIndex != 1 {
		t.Fatalf("bad: %v", *d2)
	}
	if d2.Session != "" {
		t.Fatalf("bad: %v", *d2)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}

	// Key should have a lock delay.
	expires := s.KVSLockDelay("/foo")
	if expires.Before(time.Now().Add(30 * time.Millisecond)) {
		t.Fatalf("Bad: %v", expires)
	}
}

func TestStateStore_Session_Invalidate_Key_Delete_Behavior(t *testing.T) {
	s := testStateStore(t)

	// Set up our test environment.
	if err := s.EnsureNode(3, &structs.Node{Node: "foo", Address: "127.0.0.1"}); err != nil {
		t.Fatalf("err: %v", err)
	}
	session := &structs.Session{
		ID:        testUUID(),
		Node:      "foo",
		LockDelay: 50 * time.Millisecond,
		Behavior:  structs.SessionKeysDelete,
	}
	if err := s.SessionCreate(4, session); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Lock a key with the session.
	d := &structs.DirEntry{
		Key:     "/bar",
		Flags:   42,
		Value:   []byte("test"),
		Session: session.ID,
	}
	ok, err := s.KVSLock(5, d)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if !ok {
		t.Fatalf("unexpected fail")
	}

	// Delete the node and make sure the watches fire.
	verifyWatch(t, s.getTableWatch("sessions"), func() {
		verifyWatch(t, s.getTableWatch("nodes"), func() {
			verifyWatch(t, s.GetKVSWatch("/b"), func() {
				if err := s.DeleteNode(6, "foo"); err != nil {
					t.Fatalf("err: %v", err)
				}
			})
		})
	})

	// Lookup by ID, should be nil.
	idx, s2, err := s.SessionGet(session.ID)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s2 != nil {
		t.Fatalf("session should be invalidated")
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}

	// Key should be deleted.
	idx, d2, err := s.KVSGet("/bar")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if d2 != nil {
		t.Fatalf("unexpected deleted key")
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}

	// Key should have a lock delay.
	expires := s.KVSLockDelay("/bar")
	if expires.Before(time.Now().Add(30 * time.Millisecond)) {
		t.Fatalf("Bad: %v", expires)
	}
}

func TestStateStore_Session_Invalidate_PreparedQuery_Delete(t *testing.T) {
	s := testStateStore(t)

	// Set up our test environment.
	testRegisterNode(t, s, 1, "foo")
	testRegisterService(t, s, 2, "foo", "redis")
	session := &structs.Session{
		ID:   testUUID(),
		Node: "foo",
	}
	if err := s.SessionCreate(3, session); err != nil {
		t.Fatalf("err: %v", err)
	}
	query := &structs.PreparedQuery{
		ID:      testUUID(),
		Session: session.ID,
		Service: structs.ServiceQuery{
			Service: "redis",
		},
	}
	if err := s.PreparedQuerySet(4, query); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Invalidate the session and make sure the watches fire.
	verifyWatch(t, s.getTableWatch("sessions"), func() {
		verifyWatch(t, s.getTableWatch("prepared-queries"), func() {
			if err := s.SessionDestroy(5, session.ID); err != nil {
				t.Fatalf("err: %v", err)
			}
		})
	})

	// Make sure the session is gone.
	idx, s2, err := s.SessionGet(session.ID)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if s2 != nil {
		t.Fatalf("session should be invalidated")
	}
	if idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}

	// Make sure the query is gone and the index is updated.
	idx, q2, err := s.PreparedQueryGet(query.ID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}
	if q2 != nil {
		t.Fatalf("bad: %v", q2)
	}
}

func TestStateStore_ACLSet_ACLGet(t *testing.T) {
	s := testStateStore(t)

	// Querying ACLs with no results returns nil
	idx, res, err := s.ACLGet("nope")
	if idx != 0 || res != nil || err != nil {
		t.Fatalf("expected (0, nil, nil), got: (%d, %#v, %#v)", idx, res, err)
	}

	// Inserting an ACL with empty ID is disallowed
	if err := s.ACLSet(1, &structs.ACL{}); err == nil {
		t.Fatalf("expected %#v, got: %#v", ErrMissingACLID, err)
	}

	// Index is not updated if nothing is saved
	if idx := s.maxIndex("acls"); idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}

	// Inserting valid ACL works
	acl := &structs.ACL{
		ID:    "acl1",
		Name:  "First ACL",
		Type:  structs.ACLTypeClient,
		Rules: "rules1",
	}
	if err := s.ACLSet(1, acl); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Check that the index was updated
	if idx := s.maxIndex("acls"); idx != 1 {
		t.Fatalf("bad index: %d", idx)
	}

	// Retrieve the ACL again
	idx, result, err := s.ACLGet("acl1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 1 {
		t.Fatalf("bad index: %d", idx)
	}

	// Check that the ACL matches the result
	expect := &structs.ACL{
		ID:    "acl1",
		Name:  "First ACL",
		Type:  structs.ACLTypeClient,
		Rules: "rules1",
		RaftIndex: structs.RaftIndex{
			CreateIndex: 1,
			ModifyIndex: 1,
		},
	}
	if !reflect.DeepEqual(result, expect) {
		t.Fatalf("bad: %#v", result)
	}

	// Update the ACL
	acl = &structs.ACL{
		ID:    "acl1",
		Name:  "First ACL",
		Type:  structs.ACLTypeClient,
		Rules: "rules2",
	}
	if err := s.ACLSet(2, acl); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Index was updated
	if idx := s.maxIndex("acls"); idx != 2 {
		t.Fatalf("bad: %d", idx)
	}

	// ACL was updated and matches expected value
	expect = &structs.ACL{
		ID:    "acl1",
		Name:  "First ACL",
		Type:  structs.ACLTypeClient,
		Rules: "rules2",
		RaftIndex: structs.RaftIndex{
			CreateIndex: 1,
			ModifyIndex: 2,
		},
	}
	if !reflect.DeepEqual(acl, expect) {
		t.Fatalf("bad: %#v", acl)
	}
}

func TestStateStore_ACLList(t *testing.T) {
	s := testStateStore(t)

	// Listing when no ACLs exist returns nil
	idx, res, err := s.ACLList()
	if idx != 0 || res != nil || err != nil {
		t.Fatalf("expected (0, nil, nil), got: (%d, %#v, %#v)", idx, res, err)
	}

	// Insert some ACLs
	acls := structs.ACLs{
		&structs.ACL{
			ID:    "acl1",
			Type:  structs.ACLTypeClient,
			Rules: "rules1",
			RaftIndex: structs.RaftIndex{
				CreateIndex: 1,
				ModifyIndex: 1,
			},
		},
		&structs.ACL{
			ID:    "acl2",
			Type:  structs.ACLTypeClient,
			Rules: "rules2",
			RaftIndex: structs.RaftIndex{
				CreateIndex: 2,
				ModifyIndex: 2,
			},
		},
	}
	for _, acl := range acls {
		if err := s.ACLSet(acl.ModifyIndex, acl); err != nil {
			t.Fatalf("err: %s", err)
		}
	}

	// Query the ACLs
	idx, res, err = s.ACLList()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 2 {
		t.Fatalf("bad index: %d", idx)
	}

	// Check that the result matches
	if !reflect.DeepEqual(res, acls) {
		t.Fatalf("bad: %#v", res)
	}
}

func TestStateStore_ACLDelete(t *testing.T) {
	s := testStateStore(t)

	// Calling delete on an ACL which doesn't exist returns nil
	if err := s.ACLDelete(1, "nope"); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Index isn't updated if nothing is deleted
	if idx := s.maxIndex("acls"); idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}

	// Insert an ACL
	if err := s.ACLSet(1, &structs.ACL{ID: "acl1"}); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Delete the ACL and check that the index was updated
	if err := s.ACLDelete(2, "acl1"); err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx := s.maxIndex("acls"); idx != 2 {
		t.Fatalf("bad index: %d", idx)
	}

	tx := s.db.Txn(false)
	defer tx.Abort()

	// Check that the ACL was really deleted
	result, err := tx.First("acls", "id", "acl1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if result != nil {
		t.Fatalf("expected nil, got: %#v", result)
	}
}

func TestStateStore_ACL_Snapshot_Restore(t *testing.T) {
	s := testStateStore(t)

	// Insert some ACLs.
	acls := structs.ACLs{
		&structs.ACL{
			ID:    "acl1",
			Type:  structs.ACLTypeClient,
			Rules: "rules1",
			RaftIndex: structs.RaftIndex{
				CreateIndex: 1,
				ModifyIndex: 1,
			},
		},
		&structs.ACL{
			ID:    "acl2",
			Type:  structs.ACLTypeClient,
			Rules: "rules2",
			RaftIndex: structs.RaftIndex{
				CreateIndex: 2,
				ModifyIndex: 2,
			},
		},
	}
	for _, acl := range acls {
		if err := s.ACLSet(acl.ModifyIndex, acl); err != nil {
			t.Fatalf("err: %s", err)
		}
	}

	// Snapshot the ACLs.
	snap := s.Snapshot()
	defer snap.Close()

	// Alter the real state store.
	if err := s.ACLDelete(3, "acl1"); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Verify the snapshot.
	if idx := snap.LastIndex(); idx != 2 {
		t.Fatalf("bad index: %d", idx)
	}
	iter, err := snap.ACLs()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	var dump structs.ACLs
	for acl := iter.Next(); acl != nil; acl = iter.Next() {
		dump = append(dump, acl.(*structs.ACL))
	}
	if !reflect.DeepEqual(dump, acls) {
		t.Fatalf("bad: %#v", dump)
	}

	// Restore the values into a new state store.
	func() {
		s := testStateStore(t)
		restore := s.Restore()
		for _, acl := range dump {
			if err := restore.ACL(acl); err != nil {
				t.Fatalf("err: %s", err)
			}
		}
		restore.Commit()

		// Read the restored ACLs back out and verify that they match.
		idx, res, err := s.ACLList()
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 2 {
			t.Fatalf("bad index: %d", idx)
		}
		if !reflect.DeepEqual(res, acls) {
			t.Fatalf("bad: %#v", res)
		}

		// Check that the index was updated.
		if idx := s.maxIndex("acls"); idx != 2 {
			t.Fatalf("bad index: %d", idx)
		}
	}()
}

func TestStateStore_ACL_Watches(t *testing.T) {
	s := testStateStore(t)

	// Call functions that update the acls table and make sure a watch fires
	// each time.
	verifyWatch(t, s.getTableWatch("acls"), func() {
		if err := s.ACLSet(1, &structs.ACL{ID: "acl1"}); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
	verifyWatch(t, s.getTableWatch("acls"), func() {
		if err := s.ACLDelete(2, "acl1"); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
	verifyWatch(t, s.getTableWatch("acls"), func() {
		restore := s.Restore()
		if err := restore.ACL(&structs.ACL{ID: "acl1"}); err != nil {
			t.Fatalf("err: %s", err)
		}
		restore.Commit()
	})
}

// generateRandomCoordinate creates a random coordinate. This mucks with the
// underlying structure directly, so it's not really useful for any particular
// position in the network, but it's a good payload to send through to make
// sure things come out the other side or get stored correctly.
func generateRandomCoordinate() *coordinate.Coordinate {
	config := coordinate.DefaultConfig()
	coord := coordinate.NewCoordinate(config)
	for i := range coord.Vec {
		coord.Vec[i] = rand.NormFloat64()
	}
	coord.Error = rand.NormFloat64()
	coord.Adjustment = rand.NormFloat64()
	return coord
}

func TestStateStore_Coordinate_Updates(t *testing.T) {
	s := testStateStore(t)

	// Make sure the coordinates list starts out empty, and that a query for
	// a raw coordinate for a nonexistent node doesn't do anything bad.
	idx, coords, err := s.Coordinates()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}
	if coords != nil {
		t.Fatalf("bad: %#v", coords)
	}
	coord, err := s.CoordinateGetRaw("nope")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if coord != nil {
		t.Fatalf("bad: %#v", coord)
	}

	// Make an update for nodes that don't exist and make sure they get
	// ignored.
	updates := structs.Coordinates{
		&structs.Coordinate{
			Node:  "node1",
			Coord: generateRandomCoordinate(),
		},
		&structs.Coordinate{
			Node:  "node2",
			Coord: generateRandomCoordinate(),
		},
	}
	if err := s.CoordinateBatchUpdate(1, updates); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Should still be empty, though applying an empty batch does bump
	// the table index.
	idx, coords, err = s.Coordinates()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 1 {
		t.Fatalf("bad index: %d", idx)
	}
	if coords != nil {
		t.Fatalf("bad: %#v", coords)
	}

	// Register the nodes then do the update again.
	testRegisterNode(t, s, 1, "node1")
	testRegisterNode(t, s, 2, "node2")
	if err := s.CoordinateBatchUpdate(3, updates); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Should go through now.
	idx, coords, err = s.Coordinates()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(coords, updates) {
		t.Fatalf("bad: %#v", coords)
	}

	// Also verify the raw coordinate interface.
	for _, update := range updates {
		coord, err := s.CoordinateGetRaw(update.Node)
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if !reflect.DeepEqual(coord, update.Coord) {
			t.Fatalf("bad: %#v", coord)
		}
	}

	// Update the coordinate for one of the nodes.
	updates[1].Coord = generateRandomCoordinate()
	if err := s.CoordinateBatchUpdate(4, updates); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Verify it got applied.
	idx, coords, err = s.Coordinates()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(coords, updates) {
		t.Fatalf("bad: %#v", coords)
	}

	// And check the raw coordinate version of the same thing.
	for _, update := range updates {
		coord, err := s.CoordinateGetRaw(update.Node)
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if !reflect.DeepEqual(coord, update.Coord) {
			t.Fatalf("bad: %#v", coord)
		}
	}
}

func TestStateStore_Coordinate_Cleanup(t *testing.T) {
	s := testStateStore(t)

	// Register a node and update its coordinate.
	testRegisterNode(t, s, 1, "node1")
	updates := structs.Coordinates{
		&structs.Coordinate{
			Node:  "node1",
			Coord: generateRandomCoordinate(),
		},
	}
	if err := s.CoordinateBatchUpdate(2, updates); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Make sure it's in there.
	coord, err := s.CoordinateGetRaw("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if !reflect.DeepEqual(coord, updates[0].Coord) {
		t.Fatalf("bad: %#v", coord)
	}

	// Now delete the node.
	if err := s.DeleteNode(3, "node1"); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Make sure the coordinate is gone.
	coord, err = s.CoordinateGetRaw("node1")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if coord != nil {
		t.Fatalf("bad: %#v", coord)
	}

	// Make sure the index got updated.
	idx, coords, err := s.Coordinates()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}
	if coords != nil {
		t.Fatalf("bad: %#v", coords)
	}
}

func TestStateStore_Coordinate_Snapshot_Restore(t *testing.T) {
	s := testStateStore(t)

	// Register two nodes and update their coordinates.
	testRegisterNode(t, s, 1, "node1")
	testRegisterNode(t, s, 2, "node2")
	updates := structs.Coordinates{
		&structs.Coordinate{
			Node:  "node1",
			Coord: generateRandomCoordinate(),
		},
		&structs.Coordinate{
			Node:  "node2",
			Coord: generateRandomCoordinate(),
		},
	}
	if err := s.CoordinateBatchUpdate(3, updates); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Snapshot the coordinates.
	snap := s.Snapshot()
	defer snap.Close()

	// Alter the real state store.
	trash := structs.Coordinates{
		&structs.Coordinate{
			Node:  "node1",
			Coord: generateRandomCoordinate(),
		},
		&structs.Coordinate{
			Node:  "node2",
			Coord: generateRandomCoordinate(),
		},
	}
	if err := s.CoordinateBatchUpdate(4, trash); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Verify the snapshot.
	if idx := snap.LastIndex(); idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}
	iter, err := snap.Coordinates()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	var dump structs.Coordinates
	for coord := iter.Next(); coord != nil; coord = iter.Next() {
		dump = append(dump, coord.(*structs.Coordinate))
	}
	if !reflect.DeepEqual(dump, updates) {
		t.Fatalf("bad: %#v", dump)
	}

	// Restore the values into a new state store.
	func() {
		s := testStateStore(t)
		restore := s.Restore()
		if err := restore.Coordinates(5, dump); err != nil {
			t.Fatalf("err: %s", err)
		}
		restore.Commit()

		// Read the restored coordinates back out and verify that they match.
		idx, res, err := s.Coordinates()
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 5 {
			t.Fatalf("bad index: %d", idx)
		}
		if !reflect.DeepEqual(res, updates) {
			t.Fatalf("bad: %#v", res)
		}

		// Check that the index was updated (note that it got passed
		// in during the restore).
		if idx := s.maxIndex("coordinates"); idx != 5 {
			t.Fatalf("bad index: %d", idx)
		}
	}()

}

func TestStateStore_Coordinate_Watches(t *testing.T) {
	s := testStateStore(t)

	testRegisterNode(t, s, 1, "node1")

	// Call functions that update the coordinates table and make sure a watch fires
	// each time.
	verifyWatch(t, s.getTableWatch("coordinates"), func() {
		updates := structs.Coordinates{
			&structs.Coordinate{
				Node:  "node1",
				Coord: generateRandomCoordinate(),
			},
		}
		if err := s.CoordinateBatchUpdate(2, updates); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
	verifyWatch(t, s.getTableWatch("coordinates"), func() {
		if err := s.DeleteNode(3, "node1"); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
}
