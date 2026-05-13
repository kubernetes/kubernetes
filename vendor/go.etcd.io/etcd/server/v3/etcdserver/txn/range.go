// Copyright 2025 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package txn

import (
	"bytes"
	"context"
	"math"
	"sort"
	"time"

	"go.uber.org/zap"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/api/v3/mvccpb"
	"go.etcd.io/etcd/pkg/v3/traceutil"
	"go.etcd.io/etcd/server/v3/storage/mvcc"
)

// Count returns the number of keys in [key, rangeEnd) at the given revision.
// A revision of 0 reads at the latest revision.
func Count(ctx context.Context, lg *zap.Logger, kv mvcc.KV, key, rangeEnd []byte, revision int64) (int64, error) {
	resp, _, err := Range(ctx, lg, kv, &pb.RangeRequest{
		Key:       key,
		RangeEnd:  rangeEnd,
		Revision:  revision,
		CountOnly: true,
	}, false)
	if err != nil {
		return 0, err
	}
	return resp.Count, nil
}

func Range(ctx context.Context, lg *zap.Logger, kv mvcc.KV, r *pb.RangeRequest, withTotalCount bool) (resp *pb.RangeResponse, trace *traceutil.Trace, err error) {
	ctx, trace = traceutil.EnsureTrace(ctx, lg, "range")
	defer func(start time.Time) {
		success := err == nil
		RangeSecObserve(success, time.Since(start))
	}(time.Now())
	txnRead := kv.Read(mvcc.ConcurrentReadTxMode, trace)
	defer txnRead.End()
	resp, err = executeRange(ctx, lg, txnRead, r, withTotalCount)
	return resp, trace, err
}

func executeRange(ctx context.Context, lg *zap.Logger, txnRead mvcc.TxnRead, r *pb.RangeRequest, withTotalCount bool) (*pb.RangeResponse, error) {
	trace := traceutil.Get(ctx)

	limit := rangeLimit(r)
	ro := mvcc.RangeOptions{
		Limit:          limit,
		Rev:            r.Revision,
		CountOnly:      r.CountOnly,
		WithTotalCount: withTotalCount,
	}

	rr, err := txnRead.Range(ctx, r.Key, mkGteRange(r.RangeEnd), ro)
	if err != nil {
		return nil, err
	}

	filterRangeResults(rr, r)
	sortRangeResults(rr, r, lg)
	trace.Step("filter and sort the key-value pairs")

	resp := asembleRangeResponse(rr, r)
	trace.Step("assemble the response")

	return resp, nil
}

func rangeLimit(r *pb.RangeRequest) int64 {
	limit := r.Limit
	if !IsDefaultOrdering(r.SortTarget, r.SortOrder) || HasRevisionFilters(r) {
		limit = 0
	}
	if limit > 0 && limit < math.MaxInt64 {
		limit = limit + 1
	}
	return limit
}

func IsDefaultOrdering(sortTarget pb.RangeRequest_SortTarget, sortOrder pb.RangeRequest_SortOrder) bool {
	// Since current mvcc.Range implementation returns results
	// sorted by keys in lexiographically ascending order,
	// don't re-sort when target is 'KEY' and order is ASCEND
	return sortOrder == pb.RangeRequest_NONE ||
		(sortTarget == pb.RangeRequest_KEY && sortOrder == pb.RangeRequest_ASCEND)
}

func HasRevisionFilters(r *pb.RangeRequest) bool {
	return r.MinModRevision != 0 || r.MaxModRevision != 0 ||
		r.MinCreateRevision != 0 || r.MaxCreateRevision != 0
}

func filterRangeResults(rr *mvcc.RangeResult, r *pb.RangeRequest) {
	if r.MaxModRevision != 0 {
		pruneKVs(rr, func(kv *mvccpb.KeyValue) bool { return kv.ModRevision > r.MaxModRevision })
	}
	if r.MinModRevision != 0 {
		pruneKVs(rr, func(kv *mvccpb.KeyValue) bool { return kv.ModRevision < r.MinModRevision })
	}
	if r.MaxCreateRevision != 0 {
		pruneKVs(rr, func(kv *mvccpb.KeyValue) bool { return kv.CreateRevision > r.MaxCreateRevision })
	}
	if r.MinCreateRevision != 0 {
		pruneKVs(rr, func(kv *mvccpb.KeyValue) bool { return kv.CreateRevision < r.MinCreateRevision })
	}
}

func sortRangeResults(rr *mvcc.RangeResult, r *pb.RangeRequest, lg *zap.Logger) {
	sortOrder := r.SortOrder
	if r.SortTarget != pb.RangeRequest_KEY && sortOrder == pb.RangeRequest_NONE {
		sortOrder = pb.RangeRequest_ASCEND
	}

	if !IsDefaultOrdering(r.SortTarget, sortOrder) {
		var sorter sort.Interface
		switch {
		case r.SortTarget == pb.RangeRequest_KEY:
			sorter = &kvSortByKey{&kvSort{rr.KVs}}
		case r.SortTarget == pb.RangeRequest_VERSION:
			sorter = &kvSortByVersion{&kvSort{rr.KVs}}
		case r.SortTarget == pb.RangeRequest_CREATE:
			sorter = &kvSortByCreate{&kvSort{rr.KVs}}
		case r.SortTarget == pb.RangeRequest_MOD:
			sorter = &kvSortByMod{&kvSort{rr.KVs}}
		case r.SortTarget == pb.RangeRequest_VALUE:
			sorter = &kvSortByValue{&kvSort{rr.KVs}}
		default:
			lg.Panic("unexpected sort target", zap.Int32("sort-target", int32(r.SortTarget)))
		}
		switch {
		case sortOrder == pb.RangeRequest_ASCEND:
			sort.Sort(sorter)
		case sortOrder == pb.RangeRequest_DESCEND:
			sort.Sort(sort.Reverse(sorter))
		}
	}
}

func asembleRangeResponse(rr *mvcc.RangeResult, r *pb.RangeRequest) *pb.RangeResponse {
	resp := &pb.RangeResponse{Header: &pb.ResponseHeader{}}
	if r.Limit > 0 && len(rr.KVs) > int(r.Limit) {
		rr.KVs = rr.KVs[:r.Limit]
		resp.More = true
	}
	resp.Header.Revision = rr.Rev
	resp.Count = int64(rr.Count)
	resp.Kvs = make([]*mvccpb.KeyValue, len(rr.KVs))
	for i := range rr.KVs {
		if r.KeysOnly {
			rr.KVs[i].Value = nil
		}
		resp.Kvs[i] = rr.KVs[i]
	}
	return resp
}

func checkRange(rv mvcc.ReadView, req *pb.RangeRequest) error {
	switch {
	case req.Revision == 0:
		return nil
	case req.Revision > rv.Rev():
		return mvcc.ErrFutureRev
	case req.Revision < rv.FirstRev():
		return mvcc.ErrCompacted
	}
	return nil
}

func pruneKVs(rr *mvcc.RangeResult, isPrunable func(*mvccpb.KeyValue) bool) {
	j := 0
	for i := range rr.KVs {
		rr.KVs[j] = rr.KVs[i]
		if !isPrunable(rr.KVs[i]) {
			j++
		}
	}
	rr.KVs = rr.KVs[:j]
}

type kvSort struct{ kvs []*mvccpb.KeyValue }

func (s *kvSort) Swap(i, j int) {
	t := s.kvs[i]
	s.kvs[i] = s.kvs[j]
	s.kvs[j] = t
}
func (s *kvSort) Len() int { return len(s.kvs) }

type kvSortByKey struct{ *kvSort }

func (s *kvSortByKey) Less(i, j int) bool {
	return bytes.Compare(s.kvs[i].Key, s.kvs[j].Key) < 0
}

type kvSortByVersion struct{ *kvSort }

func (s *kvSortByVersion) Less(i, j int) bool {
	return (s.kvs[i].Version - s.kvs[j].Version) < 0
}

type kvSortByCreate struct{ *kvSort }

func (s *kvSortByCreate) Less(i, j int) bool {
	return (s.kvs[i].CreateRevision - s.kvs[j].CreateRevision) < 0
}

type kvSortByMod struct{ *kvSort }

func (s *kvSortByMod) Less(i, j int) bool {
	return (s.kvs[i].ModRevision - s.kvs[j].ModRevision) < 0
}

type kvSortByValue struct{ *kvSort }

func (s *kvSortByValue) Less(i, j int) bool {
	return bytes.Compare(s.kvs[i].Value, s.kvs[j].Value) < 0
}
