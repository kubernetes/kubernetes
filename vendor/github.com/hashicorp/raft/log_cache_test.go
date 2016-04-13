package raft

import (
	"testing"
)

func TestLogCache(t *testing.T) {
	store := NewInmemStore()
	c, _ := NewLogCache(16, store)

	// Insert into the in-mem store
	for i := 0; i < 32; i++ {
		log := &Log{Index: uint64(i) + 1}
		store.StoreLog(log)
	}

	// Check the indexes
	if idx, _ := c.FirstIndex(); idx != 1 {
		t.Fatalf("bad: %d", idx)
	}
	if idx, _ := c.LastIndex(); idx != 32 {
		t.Fatalf("bad: %d", idx)
	}

	// Try get log with a miss
	var out Log
	err := c.GetLog(1, &out)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out.Index != 1 {
		t.Fatalf("bad: %#v", out)
	}

	// Store logs
	l1 := &Log{Index: 33}
	l2 := &Log{Index: 34}
	err = c.StoreLogs([]*Log{l1, l2})
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	if idx, _ := c.LastIndex(); idx != 34 {
		t.Fatalf("bad: %d", idx)
	}

	// Check that it wrote-through
	err = store.GetLog(33, &out)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	err = store.GetLog(34, &out)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Delete in the backend
	err = store.DeleteRange(33, 34)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should be in the ring buffer
	err = c.GetLog(33, &out)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	err = c.GetLog(34, &out)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Purge the ring buffer
	err = c.DeleteRange(33, 34)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should not be in the ring buffer
	err = c.GetLog(33, &out)
	if err != ErrLogNotFound {
		t.Fatalf("err: %v", err)
	}
	err = c.GetLog(34, &out)
	if err != ErrLogNotFound {
		t.Fatalf("err: %v", err)
	}
}
