package bitseq

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"testing"
	"time"

	"github.com/docker/libkv/store"
	"github.com/docker/libkv/store/boltdb"
	"github.com/docker/libnetwork/datastore"
	_ "github.com/docker/libnetwork/testutils"
)

const (
	defaultPrefix = "/tmp/libnetwork/test/bitseq"
)

func init() {
	boltdb.Register()
}

func randomLocalStore() (datastore.DataStore, error) {
	tmp, err := ioutil.TempFile("", "libnetwork-")
	if err != nil {
		return nil, fmt.Errorf("Error creating temp file: %v", err)
	}
	if err := tmp.Close(); err != nil {
		return nil, fmt.Errorf("Error closing temp file: %v", err)
	}
	return datastore.NewDataStore(datastore.LocalScope, &datastore.ScopeCfg{
		Client: datastore.ScopeClientCfg{
			Provider: "boltdb",
			Address:  defaultPrefix + tmp.Name(),
			Config: &store.Config{
				Bucket:            "libnetwork",
				ConnectionTimeout: 3 * time.Second,
			},
		},
	})
}

func TestSequenceGetAvailableBit(t *testing.T) {
	input := []struct {
		head    *sequence
		from    uint64
		bytePos uint64
		bitPos  uint64
	}{
		{&sequence{block: 0x0, count: 0}, 0, invalidPos, invalidPos},
		{&sequence{block: 0x0, count: 1}, 0, 0, 0},
		{&sequence{block: 0x0, count: 100}, 0, 0, 0},

		{&sequence{block: 0x80000000, count: 0}, 0, invalidPos, invalidPos},
		{&sequence{block: 0x80000000, count: 1}, 0, 0, 1},
		{&sequence{block: 0x80000000, count: 100}, 0, 0, 1},

		{&sequence{block: 0xFF000000, count: 0}, 0, invalidPos, invalidPos},
		{&sequence{block: 0xFF000000, count: 1}, 0, 1, 0},
		{&sequence{block: 0xFF000000, count: 100}, 0, 1, 0},

		{&sequence{block: 0xFF800000, count: 0}, 0, invalidPos, invalidPos},
		{&sequence{block: 0xFF800000, count: 1}, 0, 1, 1},
		{&sequence{block: 0xFF800000, count: 100}, 0, 1, 1},

		{&sequence{block: 0xFFC0FF00, count: 0}, 0, invalidPos, invalidPos},
		{&sequence{block: 0xFFC0FF00, count: 1}, 0, 1, 2},
		{&sequence{block: 0xFFC0FF00, count: 100}, 0, 1, 2},

		{&sequence{block: 0xFFE0FF00, count: 0}, 0, invalidPos, invalidPos},
		{&sequence{block: 0xFFE0FF00, count: 1}, 0, 1, 3},
		{&sequence{block: 0xFFE0FF00, count: 100}, 0, 1, 3},

		{&sequence{block: 0xFFFEFF00, count: 0}, 0, invalidPos, invalidPos},
		{&sequence{block: 0xFFFEFF00, count: 1}, 0, 1, 7},
		{&sequence{block: 0xFFFEFF00, count: 100}, 0, 1, 7},

		{&sequence{block: 0xFFFFC0FF, count: 0}, 0, invalidPos, invalidPos},
		{&sequence{block: 0xFFFFC0FF, count: 1}, 0, 2, 2},
		{&sequence{block: 0xFFFFC0FF, count: 100}, 0, 2, 2},

		{&sequence{block: 0xFFFFFF00, count: 0}, 0, invalidPos, invalidPos},
		{&sequence{block: 0xFFFFFF00, count: 1}, 0, 3, 0},
		{&sequence{block: 0xFFFFFF00, count: 100}, 0, 3, 0},

		{&sequence{block: 0xFFFFFFFE, count: 0}, 0, invalidPos, invalidPos},
		{&sequence{block: 0xFFFFFFFE, count: 1}, 0, 3, 7},
		{&sequence{block: 0xFFFFFFFE, count: 100}, 0, 3, 7},

		{&sequence{block: 0xFFFFFFFF, count: 0}, 0, invalidPos, invalidPos},
		{&sequence{block: 0xFFFFFFFF, count: 1}, 0, invalidPos, invalidPos},
		{&sequence{block: 0xFFFFFFFF, count: 100}, 0, invalidPos, invalidPos},

		// now test with offset
		{&sequence{block: 0x0, count: 0}, 0, invalidPos, invalidPos},
		{&sequence{block: 0x0, count: 0}, 31, invalidPos, invalidPos},
		{&sequence{block: 0x0, count: 0}, 32, invalidPos, invalidPos},
		{&sequence{block: 0x0, count: 1}, 0, 0, 0},
		{&sequence{block: 0x0, count: 1}, 1, 0, 1},
		{&sequence{block: 0x0, count: 1}, 31, 3, 7},
		{&sequence{block: 0xF0FF0000, count: 1}, 0, 0, 4},
		{&sequence{block: 0xF0FF0000, count: 1}, 8, 2, 0},
		{&sequence{block: 0xFFFFFFFF, count: 1}, 0, invalidPos, invalidPos},
		{&sequence{block: 0xFFFFFFFF, count: 1}, 16, invalidPos, invalidPos},
		{&sequence{block: 0xFFFFFFFF, count: 1}, 31, invalidPos, invalidPos},
		{&sequence{block: 0xFFFFFFFE, count: 1}, 0, 3, 7},
		{&sequence{block: 0xFFFFFFFF, count: 2}, 0, invalidPos, invalidPos},
		{&sequence{block: 0xFFFFFFFF, count: 2}, 32, invalidPos, invalidPos},
	}

	for n, i := range input {
		b, bb, err := i.head.getAvailableBit(i.from)
		if b != i.bytePos || bb != i.bitPos {
			t.Fatalf("Error in sequence.getAvailableBit(%d) (%d).\nExp: (%d, %d)\nGot: (%d, %d), err: %v", i.from, n, i.bytePos, i.bitPos, b, bb, err)
		}
	}
}

func TestSequenceEqual(t *testing.T) {
	input := []struct {
		first    *sequence
		second   *sequence
		areEqual bool
	}{
		{&sequence{block: 0x0, count: 8, next: nil}, &sequence{block: 0x0, count: 8}, true},
		{&sequence{block: 0x0, count: 0, next: nil}, &sequence{block: 0x0, count: 0}, true},
		{&sequence{block: 0x0, count: 2, next: nil}, &sequence{block: 0x0, count: 1, next: &sequence{block: 0x0, count: 1}}, false},
		{&sequence{block: 0x0, count: 2, next: &sequence{block: 0x1, count: 1}}, &sequence{block: 0x0, count: 2}, false},

		{&sequence{block: 0x12345678, count: 8, next: nil}, &sequence{block: 0x12345678, count: 8}, true},
		{&sequence{block: 0x12345678, count: 8, next: nil}, &sequence{block: 0x12345678, count: 9}, false},
		{&sequence{block: 0x12345678, count: 1, next: &sequence{block: 0XFFFFFFFF, count: 1}}, &sequence{block: 0x12345678, count: 1}, false},
		{&sequence{block: 0x12345678, count: 1}, &sequence{block: 0x12345678, count: 1, next: &sequence{block: 0XFFFFFFFF, count: 1}}, false},
	}

	for n, i := range input {
		if i.areEqual != i.first.equal(i.second) {
			t.Fatalf("Error in sequence.equal() (%d).\nExp: %t\nGot: %t,", n, i.areEqual, !i.areEqual)
		}
	}
}

func TestSequenceCopy(t *testing.T) {
	s := getTestSequence()
	n := s.getCopy()
	if !s.equal(n) {
		t.Fatal("copy of s failed")
	}
	if n == s {
		t.Fatal("not true copy of s")
	}
}

func TestGetFirstAvailable(t *testing.T) {
	input := []struct {
		mask    *sequence
		bytePos uint64
		bitPos  uint64
		start   uint64
	}{
		{&sequence{block: 0xffffffff, count: 2048}, invalidPos, invalidPos, 0},
		{&sequence{block: 0x0, count: 8}, 0, 0, 0},
		{&sequence{block: 0x80000000, count: 8}, 0, 1, 0},
		{&sequence{block: 0xC0000000, count: 8}, 0, 2, 0},
		{&sequence{block: 0xE0000000, count: 8}, 0, 3, 0},
		{&sequence{block: 0xF0000000, count: 8}, 0, 4, 0},
		{&sequence{block: 0xF8000000, count: 8}, 0, 5, 0},
		{&sequence{block: 0xFC000000, count: 8}, 0, 6, 0},
		{&sequence{block: 0xFE000000, count: 8}, 0, 7, 0},
		{&sequence{block: 0xFE000000, count: 8}, 3, 0, 24},

		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0x00000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 0, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0x80000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 1, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xC0000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 2, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xE0000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 3, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xF0000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 4, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xF8000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 5, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFC000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 6, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFE000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 7, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0x0E000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 0, 16},

		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFF000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 0, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFF800000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 1, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFC00000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 2, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFE00000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 3, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFF00000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 4, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFF80000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 5, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFFC0000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 6, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFFE0000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 7, 0},

		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xfffffffe, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 7, 7, 0},

		{&sequence{block: 0xffffffff, count: 2, next: &sequence{block: 0x0, count: 6}}, 8, 0, 0},
	}

	for n, i := range input {
		bytePos, bitPos, _ := getFirstAvailable(i.mask, i.start)
		if bytePos != i.bytePos || bitPos != i.bitPos {
			t.Fatalf("Error in (%d) getFirstAvailable(). Expected (%d, %d). Got (%d, %d)", n, i.bytePos, i.bitPos, bytePos, bitPos)
		}
	}
}

func TestFindSequence(t *testing.T) {
	input := []struct {
		head           *sequence
		bytePos        uint64
		precBlocks     uint64
		inBlockBytePos uint64
	}{
		{&sequence{block: 0xffffffff, count: 0}, 0, 0, invalidPos},
		{&sequence{block: 0xffffffff, count: 0}, 31, 0, invalidPos},
		{&sequence{block: 0xffffffff, count: 0}, 100, 0, invalidPos},

		{&sequence{block: 0x0, count: 1}, 0, 0, 0},
		{&sequence{block: 0x0, count: 1}, 1, 0, 1},
		{&sequence{block: 0x0, count: 1}, 31, 0, invalidPos},
		{&sequence{block: 0x0, count: 1}, 60, 0, invalidPos},

		{&sequence{block: 0xffffffff, count: 10}, 0, 0, 0},
		{&sequence{block: 0xffffffff, count: 10}, 3, 0, 3},
		{&sequence{block: 0xffffffff, count: 10}, 4, 1, 0},
		{&sequence{block: 0xffffffff, count: 10}, 7, 1, 3},
		{&sequence{block: 0xffffffff, count: 10}, 8, 2, 0},
		{&sequence{block: 0xffffffff, count: 10}, 39, 9, 3},

		{&sequence{block: 0xffffffff, count: 10, next: &sequence{block: 0xcc000000, count: 10}}, 79, 9, 3},
		{&sequence{block: 0xffffffff, count: 10, next: &sequence{block: 0xcc000000, count: 10}}, 80, 0, invalidPos},
	}

	for n, i := range input {
		_, _, precBlocks, inBlockBytePos := findSequence(i.head, i.bytePos)
		if precBlocks != i.precBlocks || inBlockBytePos != i.inBlockBytePos {
			t.Fatalf("Error in (%d) findSequence(). Expected (%d, %d). Got (%d, %d)", n, i.precBlocks, i.inBlockBytePos, precBlocks, inBlockBytePos)
		}
	}
}

func TestCheckIfAvailable(t *testing.T) {
	input := []struct {
		head    *sequence
		ordinal uint64
		bytePos uint64
		bitPos  uint64
	}{
		{&sequence{block: 0xffffffff, count: 0}, 0, invalidPos, invalidPos},
		{&sequence{block: 0xffffffff, count: 0}, 31, invalidPos, invalidPos},
		{&sequence{block: 0xffffffff, count: 0}, 100, invalidPos, invalidPos},

		{&sequence{block: 0x0, count: 1}, 0, 0, 0},
		{&sequence{block: 0x0, count: 1}, 1, 0, 1},
		{&sequence{block: 0x0, count: 1}, 31, 3, 7},
		{&sequence{block: 0x0, count: 1}, 60, invalidPos, invalidPos},

		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0x800000ff, count: 1}}, 31, invalidPos, invalidPos},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0x800000ff, count: 1}}, 32, invalidPos, invalidPos},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0x800000ff, count: 1}}, 33, 4, 1},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xC00000ff, count: 1}}, 33, invalidPos, invalidPos},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xC00000ff, count: 1}}, 34, 4, 2},

		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xC00000ff, count: 1, next: &sequence{block: 0x0, count: 1}}}, 55, 6, 7},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xC00000ff, count: 1, next: &sequence{block: 0x0, count: 1}}}, 56, invalidPos, invalidPos},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xC00000ff, count: 1, next: &sequence{block: 0x0, count: 1}}}, 63, invalidPos, invalidPos},

		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xC00000ff, count: 1, next: &sequence{block: 0x0, count: 1}}}, 64, 8, 0},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xC00000ff, count: 1, next: &sequence{block: 0x0, count: 1}}}, 95, 11, 7},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xC00000ff, count: 1, next: &sequence{block: 0x0, count: 1}}}, 96, invalidPos, invalidPos},
	}

	for n, i := range input {
		bytePos, bitPos, err := checkIfAvailable(i.head, i.ordinal)
		if bytePos != i.bytePos || bitPos != i.bitPos {
			t.Fatalf("Error in (%d) checkIfAvailable(ord:%d). Expected (%d, %d). Got (%d, %d). err: %v", n, i.ordinal, i.bytePos, i.bitPos, bytePos, bitPos, err)
		}
	}
}

func TestMergeSequences(t *testing.T) {
	input := []struct {
		original *sequence
		merged   *sequence
	}{
		{&sequence{block: 0xFE000000, count: 8, next: &sequence{block: 0xFE000000, count: 2}}, &sequence{block: 0xFE000000, count: 10}},
		{&sequence{block: 0xFFFFFFFF, count: 8, next: &sequence{block: 0xFFFFFFFF, count: 1}}, &sequence{block: 0xFFFFFFFF, count: 9}},
		{&sequence{block: 0xFFFFFFFF, count: 1, next: &sequence{block: 0xFFFFFFFF, count: 8}}, &sequence{block: 0xFFFFFFFF, count: 9}},

		{&sequence{block: 0xFFFFFFF0, count: 8, next: &sequence{block: 0xFFFFFFF0, count: 1}}, &sequence{block: 0xFFFFFFF0, count: 9}},
		{&sequence{block: 0xFFFFFFF0, count: 1, next: &sequence{block: 0xFFFFFFF0, count: 8}}, &sequence{block: 0xFFFFFFF0, count: 9}},

		{&sequence{block: 0xFE, count: 8, next: &sequence{block: 0xFE, count: 1, next: &sequence{block: 0xFE, count: 5}}}, &sequence{block: 0xFE, count: 14}},
		{&sequence{block: 0xFE, count: 8, next: &sequence{block: 0xFE, count: 1, next: &sequence{block: 0xFE, count: 5, next: &sequence{block: 0xFF, count: 1}}}},
			&sequence{block: 0xFE, count: 14, next: &sequence{block: 0xFF, count: 1}}},

		// No merge
		{&sequence{block: 0xFE, count: 8, next: &sequence{block: 0xF8, count: 1, next: &sequence{block: 0xFE, count: 5}}},
			&sequence{block: 0xFE, count: 8, next: &sequence{block: 0xF8, count: 1, next: &sequence{block: 0xFE, count: 5}}}},

		// No merge from head: // Merge function tries to merge from passed head. If it can't merge with next, it does not reattempt with next as head
		{&sequence{block: 0xFE, count: 8, next: &sequence{block: 0xFF, count: 1, next: &sequence{block: 0xFF, count: 5}}},
			&sequence{block: 0xFE, count: 8, next: &sequence{block: 0xFF, count: 6}}},
	}

	for n, i := range input {
		mergeSequences(i.original)
		for !i.merged.equal(i.original) {
			t.Fatalf("Error in (%d) mergeSequences().\nExp: %s\nGot: %s,", n, i.merged.toString(), i.original.toString())
		}
	}
}

func TestPushReservation(t *testing.T) {
	input := []struct {
		mask    *sequence
		bytePos uint64
		bitPos  uint64
		newMask *sequence
	}{
		// Create first sequence and fill in 8 addresses starting from address 0
		{&sequence{block: 0x0, count: 8, next: nil}, 0, 0, &sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 7, next: nil}}},
		{&sequence{block: 0x80000000, count: 8}, 0, 1, &sequence{block: 0xC0000000, count: 1, next: &sequence{block: 0x80000000, count: 7, next: nil}}},
		{&sequence{block: 0xC0000000, count: 8}, 0, 2, &sequence{block: 0xE0000000, count: 1, next: &sequence{block: 0xC0000000, count: 7, next: nil}}},
		{&sequence{block: 0xE0000000, count: 8}, 0, 3, &sequence{block: 0xF0000000, count: 1, next: &sequence{block: 0xE0000000, count: 7, next: nil}}},
		{&sequence{block: 0xF0000000, count: 8}, 0, 4, &sequence{block: 0xF8000000, count: 1, next: &sequence{block: 0xF0000000, count: 7, next: nil}}},
		{&sequence{block: 0xF8000000, count: 8}, 0, 5, &sequence{block: 0xFC000000, count: 1, next: &sequence{block: 0xF8000000, count: 7, next: nil}}},
		{&sequence{block: 0xFC000000, count: 8}, 0, 6, &sequence{block: 0xFE000000, count: 1, next: &sequence{block: 0xFC000000, count: 7, next: nil}}},
		{&sequence{block: 0xFE000000, count: 8}, 0, 7, &sequence{block: 0xFF000000, count: 1, next: &sequence{block: 0xFE000000, count: 7, next: nil}}},

		{&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 7}}, 0, 1, &sequence{block: 0xC0000000, count: 1, next: &sequence{block: 0x0, count: 7, next: nil}}},

		// Create second sequence and fill in 8 addresses starting from address 32
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0x00000000, count: 1, next: &sequence{block: 0xffffffff, count: 6, next: nil}}}, 4, 0,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0x80000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0x80000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 1,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xC0000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xC0000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 2,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xE0000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xE0000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 3,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xF0000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xF0000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 4,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xF8000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xF8000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 5,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFC000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFC000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 6,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFE000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFE000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 4, 7,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFF000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		// fill in 8 addresses starting from address 40
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFF000000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 0,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFF800000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFF800000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 1,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFC00000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFC00000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 2,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFE00000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFE00000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 3,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFF00000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFF00000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 4,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFF80000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFF80000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 5,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFFC0000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFFC0000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 6,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFFE0000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFFE0000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}, 5, 7,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xFFFF0000, count: 1, next: &sequence{block: 0xffffffff, count: 6}}}},

		// Insert new sequence
		{&sequence{block: 0xffffffff, count: 2, next: &sequence{block: 0x0, count: 6}}, 8, 0,
			&sequence{block: 0xffffffff, count: 2, next: &sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 5}}}},
		{&sequence{block: 0xffffffff, count: 2, next: &sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 5}}}, 8, 1,
			&sequence{block: 0xffffffff, count: 2, next: &sequence{block: 0xC0000000, count: 1, next: &sequence{block: 0x0, count: 5}}}},

		// Merge affected with next
		{&sequence{block: 0xffffffff, count: 7, next: &sequence{block: 0xfffffffe, count: 2, next: &sequence{block: 0xffffffff, count: 1}}}, 31, 7,
			&sequence{block: 0xffffffff, count: 8, next: &sequence{block: 0xfffffffe, count: 1, next: &sequence{block: 0xffffffff, count: 1}}}},
		{&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xfffffffc, count: 1, next: &sequence{block: 0xfffffffe, count: 6}}}, 7, 6,
			&sequence{block: 0xffffffff, count: 1, next: &sequence{block: 0xfffffffe, count: 7}}},

		// Merge affected with next and next.next
		{&sequence{block: 0xffffffff, count: 7, next: &sequence{block: 0xfffffffe, count: 1, next: &sequence{block: 0xffffffff, count: 1}}}, 31, 7,
			&sequence{block: 0xffffffff, count: 9}},
		{&sequence{block: 0xffffffff, count: 7, next: &sequence{block: 0xfffffffe, count: 1}}, 31, 7,
			&sequence{block: 0xffffffff, count: 8}},

		// Merge affected with previous and next
		{&sequence{block: 0xffffffff, count: 7, next: &sequence{block: 0xfffffffe, count: 1, next: &sequence{block: 0xffffffff, count: 1}}}, 31, 7,
			&sequence{block: 0xffffffff, count: 9}},

		// Redundant push: No change
		{&sequence{block: 0xffff0000, count: 1}, 0, 0, &sequence{block: 0xffff0000, count: 1}},
		{&sequence{block: 0xffff0000, count: 7}, 25, 7, &sequence{block: 0xffff0000, count: 7}},
		{&sequence{block: 0xffffffff, count: 7, next: &sequence{block: 0xfffffffe, count: 1, next: &sequence{block: 0xffffffff, count: 1}}}, 7, 7,
			&sequence{block: 0xffffffff, count: 7, next: &sequence{block: 0xfffffffe, count: 1, next: &sequence{block: 0xffffffff, count: 1}}}},

		// Set last bit
		{&sequence{block: 0x0, count: 8}, 31, 7, &sequence{block: 0x0, count: 7, next: &sequence{block: 0x1, count: 1}}},

		// Set bit in a middle sequence in the first block, first bit
		{&sequence{block: 0x40000000, count: 1, next: &sequence{block: 0x0, count: 6, next: &sequence{block: 0x1, count: 1}}}, 4, 0,
			&sequence{block: 0x40000000, count: 1, next: &sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 5,
				next: &sequence{block: 0x1, count: 1}}}}},

		// Set bit in a middle sequence in the first block, first bit (merge involved)
		{&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 6, next: &sequence{block: 0x1, count: 1}}}, 4, 0,
			&sequence{block: 0x80000000, count: 2, next: &sequence{block: 0x0, count: 5, next: &sequence{block: 0x1, count: 1}}}},

		// Set bit in a middle sequence in the first block, last bit
		{&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 6, next: &sequence{block: 0x1, count: 1}}}, 4, 31,
			&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x1, count: 1, next: &sequence{block: 0x0, count: 5,
				next: &sequence{block: 0x1, count: 1}}}}},

		// Set bit in a middle sequence in the first block, middle bit
		{&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 6, next: &sequence{block: 0x1, count: 1}}}, 4, 16,
			&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x8000, count: 1, next: &sequence{block: 0x0, count: 5,
				next: &sequence{block: 0x1, count: 1}}}}},

		// Set bit in a middle sequence in a middle block, first bit
		{&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 6, next: &sequence{block: 0x1, count: 1}}}, 16, 0,
			&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 3, next: &sequence{block: 0x80000000, count: 1,
				next: &sequence{block: 0x0, count: 2, next: &sequence{block: 0x1, count: 1}}}}}},

		// Set bit in a middle sequence in a middle block, last bit
		{&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 6, next: &sequence{block: 0x1, count: 1}}}, 16, 31,
			&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 3, next: &sequence{block: 0x1, count: 1,
				next: &sequence{block: 0x0, count: 2, next: &sequence{block: 0x1, count: 1}}}}}},

		// Set bit in a middle sequence in a middle block, middle bit
		{&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 6, next: &sequence{block: 0x1, count: 1}}}, 16, 15,
			&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 3, next: &sequence{block: 0x10000, count: 1,
				next: &sequence{block: 0x0, count: 2, next: &sequence{block: 0x1, count: 1}}}}}},

		// Set bit in a middle sequence in the last block, first bit
		{&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 6, next: &sequence{block: 0x1, count: 1}}}, 24, 0,
			&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 5, next: &sequence{block: 0x80000000, count: 1,
				next: &sequence{block: 0x1, count: 1}}}}},

		// Set bit in a middle sequence in the last block, last bit
		{&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 6, next: &sequence{block: 0x4, count: 1}}}, 24, 31,
			&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 5, next: &sequence{block: 0x1, count: 1,
				next: &sequence{block: 0x4, count: 1}}}}},

		// Set bit in a middle sequence in the last block, last bit (merge involved)
		{&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 6, next: &sequence{block: 0x1, count: 1}}}, 24, 31,
			&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 5, next: &sequence{block: 0x1, count: 2}}}},

		// Set bit in a middle sequence in the last block, middle bit
		{&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 6, next: &sequence{block: 0x1, count: 1}}}, 24, 16,
			&sequence{block: 0x80000000, count: 1, next: &sequence{block: 0x0, count: 5, next: &sequence{block: 0x8000, count: 1,
				next: &sequence{block: 0x1, count: 1}}}}},
	}

	for n, i := range input {
		mask := pushReservation(i.bytePos, i.bitPos, i.mask, false)
		if !mask.equal(i.newMask) {
			t.Fatalf("Error in (%d) pushReservation():\n%s + (%d,%d):\nExp: %s\nGot: %s,",
				n, i.mask.toString(), i.bytePos, i.bitPos, i.newMask.toString(), mask.toString())
		}
	}
}

func TestSerializeDeserialize(t *testing.T) {
	s := getTestSequence()

	data, err := s.toByteArray()
	if err != nil {
		t.Fatal(err)
	}

	r := &sequence{}
	err = r.fromByteArray(data)
	if err != nil {
		t.Fatal(err)
	}

	if !s.equal(r) {
		t.Fatalf("Sequences are different: \n%v\n%v", s, r)
	}
}

func getTestSequence() *sequence {
	// Returns a custom sequence of 1024 * 32 bits
	return &sequence{
		block: 0XFFFFFFFF,
		count: 100,
		next: &sequence{
			block: 0xFFFFFFFE,
			count: 1,
			next: &sequence{
				block: 0xFF000000,
				count: 10,
				next: &sequence{
					block: 0XFFFFFFFF,
					count: 50,
					next: &sequence{
						block: 0XFFFFFFFC,
						count: 1,
						next: &sequence{
							block: 0xFF800000,
							count: 1,
							next: &sequence{
								block: 0XFFFFFFFF,
								count: 87,
								next: &sequence{
									block: 0x0,
									count: 150,
									next: &sequence{
										block: 0XFFFFFFFF,
										count: 200,
										next: &sequence{
											block: 0x0000FFFF,
											count: 1,
											next: &sequence{
												block: 0x0,
												count: 399,
												next: &sequence{
													block: 0XFFFFFFFF,
													count: 23,
													next: &sequence{
														block: 0x1,
														count: 1,
													},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
}

func TestSet(t *testing.T) {
	hnd, err := NewHandle("", nil, "", 1024*32)
	if err != nil {
		t.Fatal(err)
	}
	hnd.head = getTestSequence()

	firstAv := uint64(32*100 + 31)
	last := uint64(1024*32 - 1)

	if hnd.IsSet(100000) {
		t.Fatal("IsSet() returned wrong result")
	}

	if !hnd.IsSet(0) {
		t.Fatal("IsSet() returned wrong result")
	}

	if hnd.IsSet(firstAv) {
		t.Fatal("IsSet() returned wrong result")
	}

	if !hnd.IsSet(last) {
		t.Fatal("IsSet() returned wrong result")
	}

	if err := hnd.Set(0); err == nil {
		t.Fatal("Expected failure, but succeeded")
	}

	os, err := hnd.SetAny()
	if err != nil {
		t.Fatalf("Unexpected failure: %v", err)
	}
	if os != firstAv {
		t.Fatalf("SetAny returned unexpected ordinal. Expected %d. Got %d.", firstAv, os)
	}
	if !hnd.IsSet(firstAv) {
		t.Fatal("IsSet() returned wrong result")
	}

	if err := hnd.Unset(firstAv); err != nil {
		t.Fatalf("Unexpected failure: %v", err)
	}

	if hnd.IsSet(firstAv) {
		t.Fatal("IsSet() returned wrong result")
	}

	if err := hnd.Set(firstAv); err != nil {
		t.Fatalf("Unexpected failure: %v", err)
	}

	if err := hnd.Set(last); err == nil {
		t.Fatal("Expected failure, but succeeded")
	}
}

func TestSetUnset(t *testing.T) {
	numBits := uint64(32 * blockLen)
	hnd, err := NewHandle("", nil, "", numBits)
	if err != nil {
		t.Fatal(err)
	}

	if err := hnd.Set(uint64(32 * blockLen)); err == nil {
		t.Fatal("Expected failure, but succeeded")
	}
	if err := hnd.Unset(uint64(32 * blockLen)); err == nil {
		t.Fatal("Expected failure, but succeeded")
	}

	// set and unset all one by one
	for hnd.Unselected() > 0 {
		if _, err := hnd.SetAny(); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := hnd.SetAny(); err != ErrNoBitAvailable {
		t.Fatal("Expected error. Got success")
	}
	if _, err := hnd.SetAnyInRange(10, 20); err != ErrNoBitAvailable {
		t.Fatal("Expected error. Got success")
	}
	if err := hnd.Set(50); err != ErrBitAllocated {
		t.Fatalf("Expected error. Got %v: %s", err, hnd)
	}
	i := uint64(0)
	for hnd.Unselected() < numBits {
		if err := hnd.Unset(i); err != nil {
			t.Fatal(err)
		}
		i++
	}
}

func TestOffsetSetUnset(t *testing.T) {
	numBits := uint64(32 * blockLen)
	var o uint64
	hnd, err := NewHandle("", nil, "", numBits)
	if err != nil {
		t.Fatal(err)
	}

	// set and unset all one by one
	for hnd.Unselected() > 0 {
		if _, err := hnd.SetAny(); err != nil {
			t.Fatal(err)
		}
	}

	if _, err := hnd.SetAny(); err != ErrNoBitAvailable {
		t.Fatal("Expected error. Got success")
	}

	if _, err := hnd.SetAnyInRange(10, 20); err != ErrNoBitAvailable {
		t.Fatal("Expected error. Got success")
	}

	if err := hnd.Unset(288); err != nil {
		t.Fatal(err)
	}

	//At this point sequence is (0xffffffff, 9)->(0x7fffffff, 1)->(0xffffffff, 22)->end
	if o, err = hnd.SetAnyInRange(32, 500); err != nil {
		t.Fatal(err)
	}

	if o != 288 {
		t.Fatalf("Expected ordinal not received, Received:%d", o)
	}
}

func TestSetInRange(t *testing.T) {
	numBits := uint64(1024 * blockLen)
	hnd, err := NewHandle("", nil, "", numBits)
	if err != nil {
		t.Fatal(err)
	}
	hnd.head = getTestSequence()

	firstAv := uint64(100*blockLen + blockLen - 1)

	if o, err := hnd.SetAnyInRange(4, 3); err == nil {
		t.Fatalf("Expected failure. Got success with ordinal:%d", o)
	}

	if o, err := hnd.SetAnyInRange(0, numBits); err == nil {
		t.Fatalf("Expected failure. Got success with ordinal:%d", o)
	}

	o, err := hnd.SetAnyInRange(100*uint64(blockLen), 101*uint64(blockLen))
	if err != nil {
		t.Fatalf("Unexpected failure: (%d, %v)", o, err)
	}
	if o != firstAv {
		t.Fatalf("Unexpected ordinal: %d", o)
	}

	if o, err := hnd.SetAnyInRange(0, uint64(blockLen)); err == nil {
		t.Fatalf("Expected failure. Got success with ordinal:%d", o)
	}

	if o, err := hnd.SetAnyInRange(0, firstAv-1); err == nil {
		t.Fatalf("Expected failure. Got success with ordinal:%d", o)
	}

	if o, err := hnd.SetAnyInRange(111*uint64(blockLen), 161*uint64(blockLen)); err == nil {
		t.Fatalf("Expected failure. Got success with ordinal:%d", o)
	}

	o, err = hnd.SetAnyInRange(161*uint64(blockLen), 162*uint64(blockLen))
	if err != nil {
		t.Fatal(err)
	}
	if o != 161*uint64(blockLen)+30 {
		t.Fatalf("Unexpected ordinal: %d", o)
	}

	o, err = hnd.SetAnyInRange(161*uint64(blockLen), 162*uint64(blockLen))
	if err != nil {
		t.Fatal(err)
	}
	if o != 161*uint64(blockLen)+31 {
		t.Fatalf("Unexpected ordinal: %d", o)
	}

	o, err = hnd.SetAnyInRange(161*uint64(blockLen), 162*uint64(blockLen))
	if err == nil {
		t.Fatalf("Expected failure. Got success with ordinal:%d", o)
	}

	if _, err := hnd.SetAnyInRange(0, numBits-1); err != nil {
		t.Fatalf("Unexpected failure: %v", err)
	}

	// set one bit using the set range with 1 bit size range
	if _, err := hnd.SetAnyInRange(uint64(163*blockLen-1), uint64(163*blockLen-1)); err != nil {
		t.Fatal(err)
	}

	// create a non multiple of 32 mask
	hnd, err = NewHandle("", nil, "", 30)
	if err != nil {
		t.Fatal(err)
	}

	// set all bit in the first range
	for hnd.Unselected() > 22 {
		if o, err := hnd.SetAnyInRange(0, 7); err != nil {
			t.Fatalf("Unexpected failure: (%d, %v)", o, err)
		}
	}
	// try one more set, which should fail
	o, err = hnd.SetAnyInRange(0, 7)
	if err == nil {
		t.Fatalf("Expected failure. Got success with ordinal:%d", o)
	}
	if err != ErrNoBitAvailable {
		t.Fatalf("Unexpected error: %v", err)
	}

	// set all bit in a second range
	for hnd.Unselected() > 14 {
		if o, err := hnd.SetAnyInRange(8, 15); err != nil {
			t.Fatalf("Unexpected failure: (%d, %v)", o, err)
		}
	}

	// try one more set, which should fail
	o, err = hnd.SetAnyInRange(0, 15)
	if err == nil {
		t.Fatalf("Expected failure. Got success with ordinal:%d", o)
	}
	if err != ErrNoBitAvailable {
		t.Fatalf("Unexpected error: %v", err)
	}

	// set all bit in a range which includes the last bit
	for hnd.Unselected() > 12 {
		if o, err := hnd.SetAnyInRange(28, 29); err != nil {
			t.Fatalf("Unexpected failure: (%d, %v)", o, err)
		}
	}
	o, err = hnd.SetAnyInRange(28, 29)
	if err == nil {
		t.Fatalf("Expected failure. Got success with ordinal:%d", o)
	}
	if err != ErrNoBitAvailable {
		t.Fatalf("Unexpected error: %v", err)
	}
}

// This one tests an allocation pattern which unveiled an issue in pushReservation
// Specifically a failure in detecting when we are in the (B) case (the bit to set
// belongs to the last block of the current sequence). Because of a bug, code
// was assuming the bit belonged to a block in the middle of the current sequence.
// Which in turn caused an incorrect allocation when requesting a bit which is not
// in the first or last sequence block.
func TestSetAnyInRange(t *testing.T) {
	numBits := uint64(8 * blockLen)
	hnd, err := NewHandle("", nil, "", numBits)
	if err != nil {
		t.Fatal(err)
	}

	if err := hnd.Set(0); err != nil {
		t.Fatal(err)
	}

	if err := hnd.Set(255); err != nil {
		t.Fatal(err)
	}

	o, err := hnd.SetAnyInRange(128, 255)
	if err != nil {
		t.Fatal(err)
	}
	if o != 128 {
		t.Fatalf("Unexpected ordinal: %d", o)
	}

	o, err = hnd.SetAnyInRange(128, 255)
	if err != nil {
		t.Fatal(err)
	}

	if o != 129 {
		t.Fatalf("Unexpected ordinal: %d", o)
	}

	o, err = hnd.SetAnyInRange(246, 255)
	if err != nil {
		t.Fatal(err)
	}
	if o != 246 {
		t.Fatalf("Unexpected ordinal: %d", o)
	}

	o, err = hnd.SetAnyInRange(246, 255)
	if err != nil {
		t.Fatal(err)
	}
	if o != 247 {
		t.Fatalf("Unexpected ordinal: %d", o)
	}
}

func TestMethods(t *testing.T) {
	numBits := uint64(256 * blockLen)
	hnd, err := NewHandle("path/to/data", nil, "sequence1", uint64(numBits))
	if err != nil {
		t.Fatal(err)
	}

	if hnd.Bits() != numBits {
		t.Fatalf("Unexpected bit number: %d", hnd.Bits())
	}

	if hnd.Unselected() != numBits {
		t.Fatalf("Unexpected bit number: %d", hnd.Unselected())
	}

	exp := "(0x0, 256)->end"
	if hnd.head.toString() != exp {
		t.Fatalf("Unexpected sequence string: %s", hnd.head.toString())
	}

	for i := 0; i < 192; i++ {
		_, err := hnd.SetAny()
		if err != nil {
			t.Fatal(err)
		}
	}

	exp = "(0xffffffff, 6)->(0x0, 250)->end"
	if hnd.head.toString() != exp {
		t.Fatalf("Unexpected sequence string: %s", hnd.head.toString())
	}
}

func TestRandomAllocateDeallocate(t *testing.T) {
	ds, err := randomLocalStore()
	if err != nil {
		t.Fatal(err)
	}

	numBits := int(16 * blockLen)
	hnd, err := NewHandle("bitseq-test/data/", ds, "test1", uint64(numBits))
	if err != nil {
		t.Fatal(err)
	}

	seed := time.Now().Unix()
	rand.Seed(seed)

	// Allocate all bits using a random pattern
	pattern := rand.Perm(numBits)
	for _, bit := range pattern {
		err := hnd.Set(uint64(bit))
		if err != nil {
			t.Fatalf("Unexpected failure on allocation of %d: %v.\nSeed: %d.\n%s", bit, err, seed, hnd)
		}
	}
	if hnd.Unselected() != 0 {
		t.Fatalf("Expected full sequence. Instead found %d free bits. Seed: %d.\n%s", hnd.unselected, seed, hnd)
	}
	if hnd.head.toString() != "(0xffffffff, 16)->end" {
		t.Fatalf("Unexpected db: %s", hnd.head.toString())
	}

	// Deallocate all bits using a random pattern
	pattern = rand.Perm(numBits)
	for _, bit := range pattern {
		err := hnd.Unset(uint64(bit))
		if err != nil {
			t.Fatalf("Unexpected failure on deallocation of %d: %v.\nSeed: %d.\n%s", bit, err, seed, hnd)
		}
	}
	if hnd.Unselected() != uint64(numBits) {
		t.Fatalf("Expected full sequence. Instead found %d free bits. Seed: %d.\n%s", hnd.unselected, seed, hnd)
	}
	if hnd.head.toString() != "(0x0, 16)->end" {
		t.Fatalf("Unexpected db: %s", hnd.head.toString())
	}

	err = hnd.Destroy()
	if err != nil {
		t.Fatal(err)
	}
}

func TestAllocateRandomDeallocate(t *testing.T) {
	ds, err := randomLocalStore()
	if err != nil {
		t.Fatal(err)
	}

	numBlocks := uint32(8)
	numBits := int(numBlocks * blockLen)
	hnd, err := NewHandle("bitseq-test/data/", ds, "test1", uint64(numBits))
	if err != nil {
		t.Fatal(err)
	}

	expected := &sequence{block: 0xffffffff, count: uint64(numBlocks / 2), next: &sequence{block: 0x0, count: uint64(numBlocks / 2)}}

	// Allocate first half of the bits
	for i := 0; i < numBits/2; i++ {
		_, err := hnd.SetAny()
		if err != nil {
			t.Fatalf("Unexpected failure on allocation %d: %v\n%s", i, err, hnd)
		}
	}
	if hnd.Unselected() != uint64(numBits/2) {
		t.Fatalf("Expected full sequence. Instead found %d free bits. %s", hnd.unselected, hnd)
	}
	if !hnd.head.equal(expected) {
		t.Fatalf("Unexpected sequence. Got:\n%s", hnd)
	}

	seed := time.Now().Unix()
	rand.Seed(seed)

	// Deallocate half of the allocated bits following a random pattern
	pattern := rand.Perm(numBits / 2)
	for i := 0; i < numBits/4; i++ {
		bit := pattern[i]
		err := hnd.Unset(uint64(bit))
		if err != nil {
			t.Fatalf("Unexpected failure on deallocation of %d: %v.\nSeed: %d.\n%s", bit, err, seed, hnd)
		}
	}
	if hnd.Unselected() != uint64(3*numBits/4) {
		t.Fatalf("Expected full sequence. Instead found %d free bits.\nSeed: %d.\n%s", hnd.unselected, seed, hnd)
	}

	// Request a quarter of bits
	for i := 0; i < numBits/4; i++ {
		_, err := hnd.SetAny()
		if err != nil {
			t.Fatalf("Unexpected failure on allocation %d: %v\nSeed: %d\n%s", i, err, seed, hnd)
		}
	}
	if hnd.Unselected() != uint64(numBits/2) {
		t.Fatalf("Expected half sequence. Instead found %d free bits.\nSeed: %d\n%s", hnd.unselected, seed, hnd)
	}
	if !hnd.head.equal(expected) {
		t.Fatalf("Unexpected sequence. Got:\n%s", hnd)
	}

	err = hnd.Destroy()
	if err != nil {
		t.Fatal(err)
	}
}

func TestRetrieveFromStore(t *testing.T) {
	ds, err := randomLocalStore()
	if err != nil {
		t.Fatal(err)
	}

	numBits := int(8 * blockLen)
	hnd, err := NewHandle("bitseq-test/data/", ds, "test1", uint64(numBits))
	if err != nil {
		t.Fatal(err)
	}

	// Allocate first half of the bits
	for i := 0; i < numBits/2; i++ {
		_, err := hnd.SetAny()
		if err != nil {
			t.Fatalf("Unexpected failure on allocation %d: %v\n%s", i, err, hnd)
		}
	}
	hnd0 := hnd.String()

	// Retrieve same handle
	hnd, err = NewHandle("bitseq-test/data/", ds, "test1", uint64(numBits))
	if err != nil {
		t.Fatal(err)
	}
	hnd1 := hnd.String()

	if hnd1 != hnd0 {
		t.Fatalf("%v\n%v", hnd0, hnd1)
	}

	err = hnd.Destroy()
	if err != nil {
		t.Fatal(err)
	}
}

func TestIsCorrupted(t *testing.T) {
	ds, err := randomLocalStore()
	if err != nil {
		t.Fatal(err)
	}
	// Negative test
	hnd, err := NewHandle("bitseq-test/data/", ds, "test_corrupted", 1024)
	if err != nil {
		t.Fatal(err)
	}

	if hnd.runConsistencyCheck() {
		t.Fatalf("Unexpected corrupted for %s", hnd)
	}

	if err := hnd.CheckConsistency(); err != nil {
		t.Fatal(err)
	}

	hnd.Set(0)
	if hnd.runConsistencyCheck() {
		t.Fatalf("Unexpected corrupted for %s", hnd)
	}

	hnd.Set(1023)
	if hnd.runConsistencyCheck() {
		t.Fatalf("Unexpected corrupted for %s", hnd)
	}

	if err := hnd.CheckConsistency(); err != nil {
		t.Fatal(err)
	}

	// Try real corrupted ipam handles found in the local store files reported by three docker users,
	// plus a generic ipam handle from docker 1.9.1. This last will fail as well, because of how the
	// last node in the sequence is expressed (This is true for IPAM handle only, because of the broadcast
	// address reservation: last bit). This will allow an application using bitseq that runs a consistency
	// check to detect and replace the 1.9.0/1 old vulnerable handle with the new one.
	input := []*Handle{
		{
			id:         "LocalDefault/172.17.0.0/16",
			bits:       65536,
			unselected: 65412,
			head: &sequence{
				block: 0xffffffff,
				count: 3,
				next: &sequence{
					block: 0xffffffbf,
					count: 0,
					next: &sequence{
						block: 0xfe98816e,
						count: 1,
						next: &sequence{
							block: 0xffffffff,
							count: 0,
							next: &sequence{
								block: 0xe3bc0000,
								count: 1,
								next: &sequence{
									block: 0x0,
									count: 2042,
									next: &sequence{
										block: 0x1, count: 1,
										next: &sequence{
											block: 0x0, count: 0,
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			id:         "LocalDefault/172.17.0.0/16",
			bits:       65536,
			unselected: 65319,
			head: &sequence{
				block: 0xffffffff,
				count: 7,
				next: &sequence{
					block: 0xffffff7f,
					count: 0,
					next: &sequence{
						block: 0xffffffff,
						count: 0,
						next: &sequence{
							block: 0x2000000,
							count: 1,
							next: &sequence{
								block: 0x0,
								count: 2039,
								next: &sequence{
									block: 0x1,
									count: 1,
									next: &sequence{
										block: 0x0,
										count: 0,
									},
								},
							},
						},
					},
				},
			},
		},
		{
			id:         "LocalDefault/172.17.0.0/16",
			bits:       65536,
			unselected: 65456,
			head: &sequence{
				block: 0xffffffff, count: 2,
				next: &sequence{
					block: 0xfffbffff, count: 0,
					next: &sequence{
						block: 0xffd07000, count: 1,
						next: &sequence{
							block: 0x0, count: 333,
							next: &sequence{
								block: 0x40000000, count: 1,
								next: &sequence{
									block: 0x0, count: 1710,
									next: &sequence{
										block: 0x1, count: 1,
										next: &sequence{
											block: 0x0, count: 0,
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for idx, hnd := range input {
		if !hnd.runConsistencyCheck() {
			t.Fatalf("Expected corrupted for (%d): %s", idx, hnd)
		}
		if hnd.runConsistencyCheck() {
			t.Fatalf("Sequence still marked corrupted (%d): %s", idx, hnd)
		}
	}
}
