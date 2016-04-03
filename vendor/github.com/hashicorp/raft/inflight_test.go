package raft

import (
	"fmt"
	"testing"
)

func TestInflight_StartCommit(t *testing.T) {
	commitCh := make(chan struct{}, 1)
	in := newInflight(commitCh)

	// Commit a transaction as being in flight
	l := &logFuture{log: Log{Index: 1}}
	l.policy = newMajorityQuorum(5)
	in.Start(l)

	// Commit 3 times
	in.Commit(1)
	if in.Committed().Len() != 0 {
		t.Fatalf("should not be commited")
	}

	in.Commit(1)
	if in.Committed().Len() != 1 {
		t.Fatalf("should be commited")
	}

	// Already committed but should work anyways
	in.Commit(1)
}

func TestInflight_Cancel(t *testing.T) {
	commitCh := make(chan struct{}, 1)
	in := newInflight(commitCh)

	// Commit a transaction as being in flight
	l := &logFuture{
		log: Log{Index: 1},
	}
	l.init()
	l.policy = newMajorityQuorum(3)
	in.Start(l)

	// Cancel with an error
	err := fmt.Errorf("error 1")
	in.Cancel(err)

	// Should get an error return
	if l.Error() != err {
		t.Fatalf("expected error")
	}
}

func TestInflight_StartAll(t *testing.T) {
	commitCh := make(chan struct{}, 1)
	in := newInflight(commitCh)

	// Commit a few transaction as being in flight
	l1 := &logFuture{log: Log{Index: 2}}
	l1.policy = newMajorityQuorum(5)
	l2 := &logFuture{log: Log{Index: 3}}
	l2.policy = newMajorityQuorum(5)
	l3 := &logFuture{log: Log{Index: 4}}
	l3.policy = newMajorityQuorum(5)

	// Start all the entries
	in.StartAll([]*logFuture{l1, l2, l3})

	// Commit ranges
	in.CommitRange(1, 5)
	in.CommitRange(1, 4)
	in.CommitRange(1, 10)

	// Should get 3 back
	if in.Committed().Len() != 3 {
		t.Fatalf("expected all 3 to commit")
	}
}

func TestInflight_CommitRange(t *testing.T) {
	commitCh := make(chan struct{}, 1)
	in := newInflight(commitCh)

	// Commit a few transaction as being in flight
	l1 := &logFuture{log: Log{Index: 2}}
	l1.policy = newMajorityQuorum(5)
	in.Start(l1)

	l2 := &logFuture{log: Log{Index: 3}}
	l2.policy = newMajorityQuorum(5)
	in.Start(l2)

	l3 := &logFuture{log: Log{Index: 4}}
	l3.policy = newMajorityQuorum(5)
	in.Start(l3)

	// Commit ranges
	in.CommitRange(1, 5)
	in.CommitRange(1, 4)
	in.CommitRange(1, 10)

	// Should get 3 back
	if in.Committed().Len() != 3 {
		t.Fatalf("expected all 3 to commit")
	}
}

// Should panic if we commit non contiguously!
func TestInflight_NonContiguous(t *testing.T) {
	commitCh := make(chan struct{}, 1)
	in := newInflight(commitCh)

	// Commit a few transaction as being in flight
	l1 := &logFuture{log: Log{Index: 2}}
	l1.policy = newMajorityQuorum(5)
	in.Start(l1)

	l2 := &logFuture{log: Log{Index: 3}}
	l2.policy = newMajorityQuorum(5)
	in.Start(l2)

	in.Commit(3)
	in.Commit(3)
	in.Commit(3) // panic!

	if in.Committed().Len() != 0 {
		t.Fatalf("should not commit")
	}

	in.Commit(2)
	in.Commit(2)
	in.Commit(2) // panic!

	committed := in.Committed()
	if committed.Len() != 2 {
		t.Fatalf("should commit both")
	}

	current := committed.Front()
	l := current.Value.(*logFuture)
	if l.log.Index != 2 {
		t.Fatalf("bad: %v", *l)
	}

	current = current.Next()
	l = current.Value.(*logFuture)
	if l.log.Index != 3 {
		t.Fatalf("bad: %v", *l)
	}
}
