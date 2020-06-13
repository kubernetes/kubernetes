// Copyright 2015 The etcd Authors
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

package etcdserver

import (
	"fmt"
	"reflect"
	"strings"
	"time"

	"github.com/golang/protobuf/proto"
	"go.etcd.io/etcd/etcdserver/api/membership"
	"go.etcd.io/etcd/etcdserver/api/rafthttp"
	pb "go.etcd.io/etcd/etcdserver/etcdserverpb"
	"go.etcd.io/etcd/pkg/types"

	"go.uber.org/zap"
)

// isConnectedToQuorumSince checks whether the local member is connected to the
// quorum of the cluster since the given time.
func isConnectedToQuorumSince(transport rafthttp.Transporter, since time.Time, self types.ID, members []*membership.Member) bool {
	return numConnectedSince(transport, since, self, members) >= (len(members)/2)+1
}

// isConnectedSince checks whether the local member is connected to the
// remote member since the given time.
func isConnectedSince(transport rafthttp.Transporter, since time.Time, remote types.ID) bool {
	t := transport.ActiveSince(remote)
	return !t.IsZero() && t.Before(since)
}

// isConnectedFullySince checks whether the local member is connected to all
// members in the cluster since the given time.
func isConnectedFullySince(transport rafthttp.Transporter, since time.Time, self types.ID, members []*membership.Member) bool {
	return numConnectedSince(transport, since, self, members) == len(members)
}

// numConnectedSince counts how many members are connected to the local member
// since the given time.
func numConnectedSince(transport rafthttp.Transporter, since time.Time, self types.ID, members []*membership.Member) int {
	connectedNum := 0
	for _, m := range members {
		if m.ID == self || isConnectedSince(transport, since, m.ID) {
			connectedNum++
		}
	}
	return connectedNum
}

// longestConnected chooses the member with longest active-since-time.
// It returns false, if nothing is active.
func longestConnected(tp rafthttp.Transporter, membs []types.ID) (types.ID, bool) {
	var longest types.ID
	var oldest time.Time
	for _, id := range membs {
		tm := tp.ActiveSince(id)
		if tm.IsZero() { // inactive
			continue
		}

		if oldest.IsZero() { // first longest candidate
			oldest = tm
			longest = id
		}

		if tm.Before(oldest) {
			oldest = tm
			longest = id
		}
	}
	if uint64(longest) == 0 {
		return longest, false
	}
	return longest, true
}

type notifier struct {
	c   chan struct{}
	err error
}

func newNotifier() *notifier {
	return &notifier{
		c: make(chan struct{}),
	}
}

func (nc *notifier) notify(err error) {
	nc.err = err
	close(nc.c)
}

func warnOfExpensiveRequest(lg *zap.Logger, now time.Time, reqStringer fmt.Stringer, respMsg proto.Message, err error) {
	var resp string
	if !isNil(respMsg) {
		resp = fmt.Sprintf("size:%d", proto.Size(respMsg))
	}
	warnOfExpensiveGenericRequest(lg, now, reqStringer, "", resp, err)
}

func warnOfFailedRequest(lg *zap.Logger, now time.Time, reqStringer fmt.Stringer, respMsg proto.Message, err error) {
	var resp string
	if !isNil(respMsg) {
		resp = fmt.Sprintf("size:%d", proto.Size(respMsg))
	}
	d := time.Since(now)
	if lg != nil {
		lg.Warn(
			"failed to apply request",
			zap.Duration("took", d),
			zap.String("request", reqStringer.String()),
			zap.String("response", resp),
			zap.Error(err),
		)
	} else {
		plog.Warningf("failed to apply request %q with response %q took (%v) to execute, err is %v", reqStringer.String(), resp, d, err)
	}
}

func warnOfExpensiveReadOnlyTxnRequest(lg *zap.Logger, now time.Time, r *pb.TxnRequest, txnResponse *pb.TxnResponse, err error) {
	reqStringer := pb.NewLoggableTxnRequest(r)
	var resp string
	if !isNil(txnResponse) {
		var resps []string
		for _, r := range txnResponse.Responses {
			switch op := r.Response.(type) {
			case *pb.ResponseOp_ResponseRange:
				resps = append(resps, fmt.Sprintf("range_response_count:%d", len(op.ResponseRange.Kvs)))
			default:
				// only range responses should be in a read only txn request
			}
		}
		resp = fmt.Sprintf("responses:<%s> size:%d", strings.Join(resps, " "), proto.Size(txnResponse))
	}
	warnOfExpensiveGenericRequest(lg, now, reqStringer, "read-only range ", resp, err)
}

func warnOfExpensiveReadOnlyRangeRequest(lg *zap.Logger, now time.Time, reqStringer fmt.Stringer, rangeResponse *pb.RangeResponse, err error) {
	var resp string
	if !isNil(rangeResponse) {
		resp = fmt.Sprintf("range_response_count:%d size:%d", len(rangeResponse.Kvs), proto.Size(rangeResponse))
	}
	warnOfExpensiveGenericRequest(lg, now, reqStringer, "read-only range ", resp, err)
}

func warnOfExpensiveGenericRequest(lg *zap.Logger, now time.Time, reqStringer fmt.Stringer, prefix string, resp string, err error) {
	d := time.Since(now)
	if d > warnApplyDuration {
		if lg != nil {
			lg.Warn(
				"apply request took too long",
				zap.Duration("took", d),
				zap.Duration("expected-duration", warnApplyDuration),
				zap.String("prefix", prefix),
				zap.String("request", reqStringer.String()),
				zap.String("response", resp),
				zap.Error(err),
			)
		} else {
			var result string
			if err != nil {
				result = fmt.Sprintf("error:%v", err)
			} else {
				result = resp
			}
			plog.Warningf("%srequest %q with result %q took too long (%v) to execute", prefix, reqStringer.String(), result, d)
		}
		slowApplies.Inc()
	}
}

func isNil(msg proto.Message) bool {
	return msg == nil || reflect.ValueOf(msg).IsNil()
}
