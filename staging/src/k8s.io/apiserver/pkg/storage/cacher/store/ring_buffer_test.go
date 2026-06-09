/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package store

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	"go.etcd.io/etcd/api/v3/mvccpb"
	clientv3 "go.etcd.io/etcd/client/v3"
)

func TestPeekLatestAndOldest(t *testing.T) {
	tests := []struct {
		name          string
		capacity      int
		revs          []int64
		wantLatestRev int64
		wantOldestRev int64
	}{
		{
			name:          "empty_buffer",
			capacity:      4,
			revs:          nil,
			wantLatestRev: 0,
			wantOldestRev: 0,
		},
		{
			name:          "single_element",
			capacity:      8,
			revs:          []int64{1},
			wantLatestRev: 1,
			wantOldestRev: 1,
		},
		{
			name:          "ascending_fill",
			capacity:      4,
			revs:          []int64{1, 2, 3, 4},
			wantLatestRev: 4,
			wantOldestRev: 1,
		},
		{
			name:          "overwrite_when_full",
			capacity:      3,
			revs:          []int64{5, 6, 7, 8},
			wantLatestRev: 8,
			wantOldestRev: 6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rb := newRingBuffer(tt.capacity, func(batch []*clientv3.Event) int64 { return batch[0].Kv.ModRevision })
			for _, r := range tt.revs {
				batch, err := makeEventBatch(r, "k", 1)
				if err != nil {
					t.Fatalf("makeEventBatch(%d, k, 1) failed: %v", r, err)
				}
				rb.Append(batch)
			}

			latestRev := rb.PeekLatest()
			oldestRev := rb.PeekOldest()

			gotLatestRev := latestRev
			gotOldestRev := oldestRev

			if tt.wantLatestRev != gotLatestRev {
				t.Fatalf("PeekLatest()=%d, want=%d", gotLatestRev, tt.wantLatestRev)
			}
			if tt.wantOldestRev != gotOldestRev {
				t.Fatalf("PeekOldest()=%d, want=%d", gotOldestRev, tt.wantOldestRev)
			}
		})
	}
}

func TestIterationMethods(t *testing.T) {
	type iterTestCase struct {
		method            iterMethod
		pivot             int64
		wantIterRevisions []int64
	}
	tests := []struct {
		name           string
		capacity       int
		setupRevisions []int64
		cases          []iterTestCase
	}{
		{
			name:           "empty_buffer",
			capacity:       4,
			setupRevisions: nil,
			cases: []iterTestCase{
				{ascendGTE, 0, []int64{}},
				{ascendLT, 10, []int64{}},
				{descendGT, 0, []int64{}},
				{descendLTE, 10, []int64{}},
			},
		},
		{
			name:           "basic_filtering",
			capacity:       5,
			setupRevisions: []int64{1, 2, 3},
			cases: []iterTestCase{
				{ascendGTE, 0, []int64{1, 2, 3}},
				{ascendGTE, 2, []int64{2, 3}},
				{ascendGTE, 100, []int64{}},
				{ascendLT, 3, []int64{1, 2}},
				{ascendLT, 1, []int64{}},
				{ascendLT, 100, []int64{1, 2, 3}},
				{descendGT, 1, []int64{3, 2}},
				{descendGT, 3, []int64{}},
				{descendGT, 0, []int64{3, 2, 1}},
				{descendLTE, 2, []int64{2, 1}},
				{descendLTE, 3, []int64{3, 2, 1}},
				{descendLTE, 0, []int64{}},
			},
		},
		{
			name:           "overflowed stores only entries within capacity",
			capacity:       3,
			setupRevisions: []int64{20, 21, 22, 23, 24}, // stored: 22, 23, 24
			cases: []iterTestCase{
				{ascendGTE, 23, []int64{23, 24}},
				{ascendGTE, 0, []int64{22, 23, 24}},
				{ascendLT, 23, []int64{22}},
				{ascendLT, 25, []int64{22, 23, 24}},
				{descendGT, 22, []int64{24, 23}},
				{descendGT, 25, []int64{}},
				{descendLTE, 23, []int64{23, 22}},
				{descendLTE, 24, []int64{24, 23, 22}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rb := setupRingBuffer(t, tt.capacity, tt.setupRevisions)

			for _, tc := range tt.cases {
				t.Run(fmt.Sprintf("%s_pivot_%d", tc.method, tc.pivot), func(t *testing.T) {
					got := collectRevisions(rb, tc.method, tc.pivot)
					if diff := cmp.Diff(tc.wantIterRevisions, got); diff != "" {
						t.Fatalf("%s(%d) mismatch (-want +got):\n%s", tc.method, tc.pivot, diff)
					}
				})
			}
		})
	}
}

func TestIterationWithBatching(t *testing.T) {
	rb := newRingBuffer(6, func(batch []*clientv3.Event) int64 { return batch[0].Kv.ModRevision })
	batchA := []*clientv3.Event{
		{Kv: &mvccpb.KeyValue{Key: []byte("key-a"), ModRevision: 5}},
	}
	batchB := []*clientv3.Event{
		{Kv: &mvccpb.KeyValue{Key: []byte("key-b-1"), ModRevision: 10}},
		{Kv: &mvccpb.KeyValue{Key: []byte("key-b-2"), ModRevision: 10}},
		{Kv: &mvccpb.KeyValue{Key: []byte("key-b-3"), ModRevision: 10}},
	}
	batchC := []*clientv3.Event{
		{Kv: &mvccpb.KeyValue{Key: []byte("key-c"), ModRevision: 12}},
	}
	rb.Append(batchA)
	rb.Append(batchB)
	rb.Append(batchC)

	tests := []struct {
		name   string
		method iterMethod
		pivot  int64
		want   [][]*clientv3.Event
	}{
		{
			name:   "ascending_gte_includes_batched_revision",
			method: ascendGTE,
			pivot:  10,
			want: [][]*clientv3.Event{
				{
					{Kv: &mvccpb.KeyValue{Key: []byte("key-b-1"), ModRevision: 10}},
					{Kv: &mvccpb.KeyValue{Key: []byte("key-b-2"), ModRevision: 10}},
					{Kv: &mvccpb.KeyValue{Key: []byte("key-b-3"), ModRevision: 10}},
				},
				{
					{Kv: &mvccpb.KeyValue{Key: []byte("key-c"), ModRevision: 12}},
				},
			},
		},
		{
			name:   "ascending_lt_stops_before_batched_revision",
			method: ascendLT,
			pivot:  10,
			want: [][]*clientv3.Event{
				{
					{Kv: &mvccpb.KeyValue{Key: []byte("key-a"), ModRevision: 5}},
				},
			},
		},
		{
			name:   "all_revisions_with_proper_batch_sizes",
			method: ascendGTE,
			pivot:  0,
			want: [][]*clientv3.Event{
				{
					{Kv: &mvccpb.KeyValue{Key: []byte("key-a"), ModRevision: 5}},
				},
				{
					{Kv: &mvccpb.KeyValue{Key: []byte("key-b-1"), ModRevision: 10}},
					{Kv: &mvccpb.KeyValue{Key: []byte("key-b-2"), ModRevision: 10}},
					{Kv: &mvccpb.KeyValue{Key: []byte("key-b-3"), ModRevision: 10}},
				},
				{
					{Kv: &mvccpb.KeyValue{Key: []byte("key-c"), ModRevision: 12}},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got [][]*clientv3.Event

			rb.iterate(tt.method, tt.pivot, func(rev int64, events []*clientv3.Event) bool {
				got = append(got, events)
				return true
			})

			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("Events mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestIterationEarlyStop(t *testing.T) {
	rb := setupRingBuffer(t, 5, []int64{5, 10, 15, 20})
	tests := []struct {
		name      string
		method    iterMethod
		pivot     int64
		stopAfter int
		want      []int64
	}{
		{
			name:      "find_first_match_ascending",
			method:    ascendGTE,
			pivot:     10,
			stopAfter: 1,
			want:      []int64{10},
		},
		{
			name:      "find_first_two_ascending_lt",
			method:    ascendLT,
			pivot:     20,
			stopAfter: 2,
			want:      []int64{5, 10},
		},
		{
			name:      "find_first_two_descending_gt",
			method:    descendGT,
			pivot:     5,
			stopAfter: 2,
			want:      []int64{20, 15},
		},
		{
			name:      "find_first_match_descending_lte",
			method:    descendLTE,
			pivot:     15,
			stopAfter: 1,
			want:      []int64{15},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var collected []int64
			callCount := 0

			rb.iterate(tt.method, tt.pivot, func(rev int64, events []*clientv3.Event) bool {
				collected = append(collected, rev)
				callCount++

				shouldContinue := callCount < tt.stopAfter

				if !shouldContinue {
					t.Logf("Stopping early after %d items (callback returned false)", callCount)
				}

				return shouldContinue
			})

			if diff := cmp.Diff(tt.want, collected); diff != "" {
				t.Fatalf("Early stop failed.\nExpected: \nDiff (-want +got):\n%s", diff)
			}

			if callCount != tt.stopAfter {
				t.Fatalf("Expected exactly %d callback calls, got %d", tt.stopAfter, callCount)
			}

			t.Logf("Successfully stopped early: collected %v after %d callbacks",
				collected, callCount)
		})
	}
}

type iterMethod string

const (
	ascendGTE  iterMethod = "AscendGreaterOrEqual"
	ascendLT   iterMethod = "AscendLessThan"
	descendGT  iterMethod = "DescendGreaterThan"
	descendLTE iterMethod = "DescendLessOrEqual"
)

func (r *ringBuffer[T]) iterate(method iterMethod, pivot int64, fn IterFunc[T]) {
	switch method {
	case ascendGTE:
		r.AscendGreaterOrEqual(pivot, fn)
	case ascendLT:
		r.AscendLessThan(pivot, fn)
	case descendGT:
		r.DescendGreaterThan(pivot, fn)
	case descendLTE:
		r.DescendLessOrEqual(pivot, fn)
	default:
		panic(fmt.Sprintf("unknown iteration method: %s", method))
	}
}

func TestAtomicOrdered(t *testing.T) {
	tests := []struct {
		name     string
		capacity int
		inputs   []struct {
			rev  int64
			key  string
			size int
		}
		wantRev  []int64
		wantSize []int
	}{
		{
			name:     "unfiltered",
			capacity: 5,
			inputs: []struct {
				rev  int64
				key  string
				size int
			}{
				{5, "a", 1},
				{10, "b", 3},
				{15, "c", 7},
				{20, "d", 11},
			},
			wantRev:  []int64{5, 10, 15, 20},
			wantSize: []int{1, 3, 7, 11},
		},
		{
			name:     "across_wrap",
			capacity: 3,
			inputs: []struct {
				rev  int64
				key  string
				size int
			}{
				{1, "a", 2},
				{2, "b", 1},
				{3, "c", 3},
				{4, "d", 7},
			},
			wantRev:  []int64{2, 3, 4},
			wantSize: []int{1, 3, 7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			rb := newRingBuffer(tt.capacity, func(batch []*clientv3.Event) int64 { return batch[0].Kv.ModRevision })
			for _, in := range tt.inputs {
				batch, err := makeEventBatch(in.rev, in.key, in.size)
				if err != nil {
					t.Fatalf("makeEventBatch(%d, k, 1) failed: %v", in.rev, err)
				}
				rb.Append(batch)
			}

			gotRevs := []int64{}
			var gotSizes []int
			rb.AscendGreaterOrEqual(0, func(rev int64, events []*clientv3.Event) bool {
				gotRevs = append(gotRevs, rev)
				gotSizes = append(gotSizes, len(events))
				return true
			})

			if len(gotRevs) != len(tt.wantRev) {
				t.Fatalf("len(got) = %d, want %d", len(gotRevs), len(tt.wantRev))
			}
			for i := range gotRevs {
				if gotRevs[i] != tt.wantRev[i] {
					t.Errorf("at idx %d: rev = %d, want %d", i, gotRevs[i], tt.wantRev[i])
				}
				if gotSizes[i] != tt.wantSize[i] {
					t.Errorf("at rev %d: events.len = %d, want %d", gotRevs[i], gotSizes[i], tt.wantSize[i])
				}
			}
		})
	}
}

func TestRebaseHistory(t *testing.T) {
	tests := []struct {
		name string
		revs []int64
	}{
		{
			name: "rebase_empty_buffer",
			revs: nil,
		},
		{
			name: "rebase_after_data",
			revs: []int64{7, 8, 9},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			rb := newRingBuffer(4, func(batch []*clientv3.Event) int64 { return batch[0].Kv.ModRevision })
			for _, r := range tt.revs {
				batch, err := makeEventBatch(r, "k", 1)
				if err != nil {
					t.Fatalf("makeEventBatch(%d, k, 1) failed: %v", r, err)
				}
				rb.Append(batch)
			}

			rb.RebaseHistory()

			oldestRev := rb.PeekOldest()
			latestRev := rb.PeekLatest()

			if oldestRev != 0 {
				t.Fatalf("PeekOldest()=%d, want=%d", oldestRev, 0)
			}
			if latestRev != 0 {
				t.Fatalf("PeekLatest()=%d, want=%d", latestRev, 0)
			}

			gotRevs := []int64{}
			rb.AscendGreaterOrEqual(0, func(rev int64, events []*clientv3.Event) bool {
				gotRevs = append(gotRevs, rev)
				return true
			})

			if len(gotRevs) != 0 {
				t.Fatalf("AscendGreaterOrEqual() len(events)=%d, want=%d", len(gotRevs), 0)
			}
		})
	}
}

func TestFull(t *testing.T) {
	tests := []struct {
		name         string
		capacity     int
		numAppends   int
		expectedFull bool
	}{
		{
			name:         "empty_buffer",
			capacity:     3,
			numAppends:   0,
			expectedFull: false,
		},
		{
			name:         "partially_filled",
			capacity:     5,
			numAppends:   3,
			expectedFull: false,
		},
		{
			name:         "exactly_at_capacity",
			capacity:     3,
			numAppends:   3,
			expectedFull: true,
		},
		{
			name:         "beyond_capacity_wrapping",
			capacity:     3,
			numAppends:   5,
			expectedFull: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rb := newRingBuffer(tt.capacity, func(batch []*clientv3.Event) int64 { return batch[0].Kv.ModRevision })

			for i := 1; i <= tt.numAppends; i++ {
				batch, err := makeEventBatch(int64(i), "k", 1)
				if err != nil {
					t.Fatalf("makeEventBatch(%d, k, 1) failed: %v", i, err)
				}
				rb.Append(batch)
			}

			if got := rb.full(); got != tt.expectedFull {
				t.Fatalf("full()=%t, want=%t (capacity=%d, appends=%d)",
					got, tt.expectedFull, tt.capacity, tt.numAppends)
			}
		})
	}
}

func setupRingBuffer(t *testing.T, capacity int, revs []int64) *ringBuffer[[]*clientv3.Event] {
	rb := newRingBuffer(capacity, func(batch []*clientv3.Event) int64 { return batch[0].Kv.ModRevision })
	for _, r := range revs {
		batch, err := makeEventBatch(r, "key", 1)
		if err != nil {
			t.Fatalf("makeEventBatch(%d, %s, %d) failed: %v", r, "key", 1, err)
		}
		rb.Append(batch)
	}
	return rb
}

func collectRevisions(rb *ringBuffer[[]*clientv3.Event], method iterMethod, pivot int64) []int64 {
	revs := []int64{}
	rb.iterate(method, pivot, func(rev int64, events []*clientv3.Event) bool {
		revs = append(revs, rev)
		return true
	})
	return revs
}

func makeEventBatch(rev int64, key string, batchSize int) ([]*clientv3.Event, error) {
	if batchSize < 0 {
		return nil, fmt.Errorf("invalid batchSize %d", batchSize)
	}
	events := make([]*clientv3.Event, batchSize)
	for i := range events {
		events[i] = &clientv3.Event{
			Kv: &mvccpb.KeyValue{
				Key:         fmt.Appendf(nil, "%s-%d", key, i),
				ModRevision: rev,
			},
		}
	}
	return events, nil
}

func TestLen(t *testing.T) {
	rb := setupRingBuffer(t, 3, []int64{1, 2, 3, 4}) // stored: 2, 3, 4

	if got := rb.Len(); got != 3 {
		t.Fatalf("Len()=%d, want=%d", got, 3)
	}

	rb.RemoveLess(4)
	if got := rb.Len(); got != 1 {
		t.Fatalf("Len() after RemoveLess=%d, want=%d", got, 1)
	}

	rb.RebaseHistory()
	if got := rb.Len(); got != 0 {
		t.Fatalf("Len() after RebaseHistory=%d, want=%d", got, 0)
	}
}

func TestReplaceLatest(t *testing.T) {
	t.Run("replace_latest_entry", func(t *testing.T) {
		rb := newRingBuffer(4, func(batch []*clientv3.Event) int64 { return batch[0].Kv.ModRevision })
		for _, rev := range []int64{5, 10} {
			batch, err := makeEventBatch(rev, "key", 1)
			if err != nil {
				t.Fatalf("makeEventBatch(%d, key, 1) failed: %v", rev, err)
			}
			rb.Append(batch)
		}

		replacement, err := makeEventBatch(12, "replacement", 2)
		if err != nil {
			t.Fatalf("makeEventBatch(12, replacement, 2) failed: %v", err)
		}
		rb.ReplaceLatest(replacement)

		if got := rb.Len(); got != 2 {
			t.Fatalf("Len()=%d, want=%d", got, 2)
		}
		if got := rb.PeekLatest(); got != 12 {
			t.Fatalf("PeekLatest()=%d, want=%d", got, 12)
		}

		gotRevs := collectRevisions(rb, ascendGTE, 0)
		if diff := cmp.Diff([]int64{5, 12}, gotRevs); diff != "" {
			t.Fatalf("revisions mismatch (-want +got):\n%s", diff)
		}

		var gotBatch [][]*clientv3.Event
		rb.AscendGreaterOrEqual(0, func(rev int64, events []*clientv3.Event) bool {
			gotBatch = append(gotBatch, events)
			return true
		})
		if diff := cmp.Diff(2, len(gotBatch[1])); diff != "" {
			t.Fatalf("replacement batch size mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("panic_on_empty_buffer", func(t *testing.T) {
		rb := newRingBuffer(2, func(batch []*clientv3.Event) int64 { return batch[0].Kv.ModRevision })
		replacement, err := makeEventBatch(1, "replacement", 1)
		if err != nil {
			t.Fatalf("makeEventBatch(1, replacement, 1) failed: %v", err)
		}

		defer func() {
			if r := recover(); r == nil {
				t.Fatal("ReplaceLatest() did not panic on empty buffer")
			}
		}()
		rb.ReplaceLatest(replacement)
	})
}

func TestRemoveLess(t *testing.T) {
	tests := []struct {
		name         string
		capacity     int
		setupRevs    []int64
		removeBefore int64
		wantRevs     []int64
		wantLen      int
		wantOldest   int64
		wantLatest   int64
	}{
		{
			name:         "empty_buffer",
			capacity:     4,
			removeBefore: 10,
			wantRevs:     []int64{},
			wantLen:      0,
			wantOldest:   0,
			wantLatest:   0,
		},
		{
			name:         "no_entries_removed",
			capacity:     4,
			setupRevs:    []int64{5, 10, 15},
			removeBefore: 5,
			wantRevs:     []int64{5, 10, 15},
			wantLen:      3,
			wantOldest:   5,
			wantLatest:   15,
		},
		{
			name:         "remove_prefix",
			capacity:     5,
			setupRevs:    []int64{5, 10, 15, 20},
			removeBefore: 15,
			wantRevs:     []int64{15, 20},
			wantLen:      2,
			wantOldest:   15,
			wantLatest:   20,
		},
		{
			name:         "remove_all_entries",
			capacity:     3,
			setupRevs:    []int64{10, 11, 12},
			removeBefore: 20,
			wantRevs:     []int64{},
			wantLen:      0,
			wantOldest:   0,
			wantLatest:   0,
		},
		{
			name:         "wrapped_buffer",
			capacity:     3,
			setupRevs:    []int64{20, 21, 22, 23, 24}, // stored: 22, 23, 24
			removeBefore: 24,
			wantRevs:     []int64{24},
			wantLen:      1,
			wantOldest:   24,
			wantLatest:   24,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rb := setupRingBuffer(t, tt.capacity, tt.setupRevs)

			rb.RemoveLess(tt.removeBefore)

			if got := rb.Len(); got != tt.wantLen {
				t.Fatalf("Len()=%d, want=%d", got, tt.wantLen)
			}
			if got := rb.PeekOldest(); got != tt.wantOldest {
				t.Fatalf("PeekOldest()=%d, want=%d", got, tt.wantOldest)
			}
			if got := rb.PeekLatest(); got != tt.wantLatest {
				t.Fatalf("PeekLatest()=%d, want=%d", got, tt.wantLatest)
			}

			gotRevs := collectRevisions(rb, ascendGTE, 0)
			if diff := cmp.Diff(tt.wantRevs, gotRevs); diff != "" {
				t.Fatalf("remaining revisions mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestEnsureCapacity(t *testing.T) {
	t.Run("grow_preserves_logical_order", func(t *testing.T) {
		rb := setupRingBuffer(t, 3, []int64{1, 2, 3, 4}) // wrapped: stored 2, 3, 4

		rb.ensureCapacity(5)

		if got := len(rb.buffer); got != 6 {
			t.Fatalf("len(buffer)=%d, want=%d", got, 6)
		}
		if rb.tail != 0 {
			t.Fatalf("tail=%d, want=%d", rb.tail, 0)
		}
		if rb.head != rb.size {
			t.Fatalf("head=%d, want size=%d", rb.head, rb.size)
		}

		gotRevs := collectRevisions(rb, ascendGTE, 0)
		if diff := cmp.Diff([]int64{2, 3, 4}, gotRevs); diff != "" {
			t.Fatalf("revisions mismatch after ensureCapacity (-want +got):\n%s", diff)
		}
	})

	t.Run("noop_when_capacity_is_sufficient", func(t *testing.T) {
		rb := setupRingBuffer(t, 4, []int64{1, 2})
		beforeCap := len(rb.buffer)
		beforeHead, beforeTail, beforeSize := rb.head, rb.tail, rb.size

		rb.ensureCapacity(4)

		if got := len(rb.buffer); got != beforeCap {
			t.Fatalf("len(buffer)=%d, want=%d", got, beforeCap)
		}
		if rb.head != beforeHead || rb.tail != beforeTail || rb.size != beforeSize {
			t.Fatalf("ringBuffer metadata changed unexpectedly: got head=%d tail=%d size=%d, want head=%d tail=%d size=%d",
				rb.head, rb.tail, rb.size, beforeHead, beforeTail, beforeSize)
		}
	})
}
