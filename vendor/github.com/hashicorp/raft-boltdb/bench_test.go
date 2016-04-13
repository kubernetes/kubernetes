package raftboltdb

import (
	"os"
	"testing"

	"github.com/hashicorp/raft/bench"
)

func BenchmarkBoltStore_FirstIndex(b *testing.B) {
	store := testBoltStore(b)
	defer store.Close()
	defer os.Remove(store.path)

	raftbench.FirstIndex(b, store)
}

func BenchmarkBoltStore_LastIndex(b *testing.B) {
	store := testBoltStore(b)
	defer store.Close()
	defer os.Remove(store.path)

	raftbench.LastIndex(b, store)
}

func BenchmarkBoltStore_GetLog(b *testing.B) {
	store := testBoltStore(b)
	defer store.Close()
	defer os.Remove(store.path)

	raftbench.GetLog(b, store)
}

func BenchmarkBoltStore_StoreLog(b *testing.B) {
	store := testBoltStore(b)
	defer store.Close()
	defer os.Remove(store.path)

	raftbench.StoreLog(b, store)
}

func BenchmarkBoltStore_StoreLogs(b *testing.B) {
	store := testBoltStore(b)
	defer store.Close()
	defer os.Remove(store.path)

	raftbench.StoreLogs(b, store)
}

func BenchmarkBoltStore_DeleteRange(b *testing.B) {
	store := testBoltStore(b)
	defer store.Close()
	defer os.Remove(store.path)

	raftbench.DeleteRange(b, store)
}

func BenchmarkBoltStore_Set(b *testing.B) {
	store := testBoltStore(b)
	defer store.Close()
	defer os.Remove(store.path)

	raftbench.Set(b, store)
}

func BenchmarkBoltStore_Get(b *testing.B) {
	store := testBoltStore(b)
	defer store.Close()
	defer os.Remove(store.path)

	raftbench.Get(b, store)
}

func BenchmarkBoltStore_SetUint64(b *testing.B) {
	store := testBoltStore(b)
	defer store.Close()
	defer os.Remove(store.path)

	raftbench.SetUint64(b, store)
}

func BenchmarkBoltStore_GetUint64(b *testing.B) {
	store := testBoltStore(b)
	defer store.Close()
	defer os.Remove(store.path)

	raftbench.GetUint64(b, store)
}
