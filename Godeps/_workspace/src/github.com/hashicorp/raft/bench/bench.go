package raftbench

// raftbench provides common benchmarking functions which can be used by
// anything which implements the raft.LogStore and raft.StableStore interfaces.
// All functions accept these interfaces and perform benchmarking. This
// makes comparing backend performance easier by sharing the tests.

import (
	"github.com/hashicorp/raft"
	"testing"
)

func FirstIndex(b *testing.B, store raft.LogStore) {
	// Create some fake data
	var logs []*raft.Log
	for i := 1; i < 10; i++ {
		logs = append(logs, &raft.Log{Index: uint64(i), Data: []byte("data")})
	}
	if err := store.StoreLogs(logs); err != nil {
		b.Fatalf("err: %s", err)
	}
	b.ResetTimer()

	// Run FirstIndex a number of times
	for n := 0; n < b.N; n++ {
		store.FirstIndex()
	}
}

func LastIndex(b *testing.B, store raft.LogStore) {
	// Create some fake data
	var logs []*raft.Log
	for i := 1; i < 10; i++ {
		logs = append(logs, &raft.Log{Index: uint64(i), Data: []byte("data")})
	}
	if err := store.StoreLogs(logs); err != nil {
		b.Fatalf("err: %s", err)
	}
	b.ResetTimer()

	// Run LastIndex a number of times
	for n := 0; n < b.N; n++ {
		store.LastIndex()
	}
}

func GetLog(b *testing.B, store raft.LogStore) {
	// Create some fake data
	var logs []*raft.Log
	for i := 1; i < 10; i++ {
		logs = append(logs, &raft.Log{Index: uint64(i), Data: []byte("data")})
	}
	if err := store.StoreLogs(logs); err != nil {
		b.Fatalf("err: %s", err)
	}
	b.ResetTimer()

	// Run GetLog a number of times
	for n := 0; n < b.N; n++ {
		if err := store.GetLog(5, new(raft.Log)); err != nil {
			b.Fatalf("err: %s", err)
		}
	}
}

func StoreLog(b *testing.B, store raft.LogStore) {
	// Run StoreLog a number of times
	for n := 0; n < b.N; n++ {
		log := &raft.Log{Index: uint64(n), Data: []byte("data")}
		if err := store.StoreLog(log); err != nil {
			b.Fatalf("err: %s", err)
		}
	}
}

func StoreLogs(b *testing.B, store raft.LogStore) {
	// Run StoreLogs a number of times. We want to set multiple logs each
	// run, so we create 3 logs with incrementing indexes for each iteration.
	for n := 0; n < b.N; n++ {
		b.StopTimer()
		offset := 3 * (n + 1)
		logs := []*raft.Log{
			&raft.Log{Index: uint64(offset - 2), Data: []byte("data")},
			&raft.Log{Index: uint64(offset - 1), Data: []byte("data")},
			&raft.Log{Index: uint64(offset), Data: []byte("data")},
		}
		b.StartTimer()

		if err := store.StoreLogs(logs); err != nil {
			b.Fatalf("err: %s", err)
		}
	}
}

func DeleteRange(b *testing.B, store raft.LogStore) {
	// Create some fake data. In this case, we create 3 new log entries for each
	// test case, and separate them by index in multiples of 10. This allows
	// some room so that we can test deleting ranges with "extra" logs to
	// to ensure we stop going to the database once our max index is hit.
	var logs []*raft.Log
	for n := 0; n < b.N; n++ {
		offset := 10 * n
		for i := offset; i < offset+3; i++ {
			logs = append(logs, &raft.Log{Index: uint64(i), Data: []byte("data")})
		}
	}
	if err := store.StoreLogs(logs); err != nil {
		b.Fatalf("err: %s", err)
	}
	b.ResetTimer()

	// Delete a range of the data
	for n := 0; n < b.N; n++ {
		offset := 10 * n
		if err := store.DeleteRange(uint64(offset), uint64(offset+9)); err != nil {
			b.Fatalf("err: %s", err)
		}
	}
}

func Set(b *testing.B, store raft.StableStore) {
	// Run Set a number of times
	for n := 0; n < b.N; n++ {
		if err := store.Set([]byte{byte(n)}, []byte("val")); err != nil {
			b.Fatalf("err: %s", err)
		}
	}
}

func Get(b *testing.B, store raft.StableStore) {
	// Create some fake data
	for i := 1; i < 10; i++ {
		if err := store.Set([]byte{byte(i)}, []byte("val")); err != nil {
			b.Fatalf("err: %s", err)
		}
	}
	b.ResetTimer()

	// Run Get a number of times
	for n := 0; n < b.N; n++ {
		if _, err := store.Get([]byte{0x05}); err != nil {
			b.Fatalf("err: %s", err)
		}
	}
}

func SetUint64(b *testing.B, store raft.StableStore) {
	// Run SetUint64 a number of times
	for n := 0; n < b.N; n++ {
		if err := store.SetUint64([]byte{byte(n)}, uint64(n)); err != nil {
			b.Fatalf("err: %s", err)
		}
	}
}

func GetUint64(b *testing.B, store raft.StableStore) {
	// Create some fake data
	for i := 0; i < 10; i++ {
		if err := store.SetUint64([]byte{byte(i)}, uint64(i)); err != nil {
			b.Fatalf("err: %s", err)
		}
	}
	b.ResetTimer()

	// Run GetUint64 a number of times
	for n := 0; n < b.N; n++ {
		if _, err := store.Get([]byte{0x05}); err != nil {
			b.Fatalf("err: %s", err)
		}
	}
}
