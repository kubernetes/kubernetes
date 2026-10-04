// Copyright 2026 The etcd Authors
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

package read

import (
	"context"
	"encoding/binary"
	errorspkg "errors"
	"strconv"
	"sync"
	"time"

	"go.uber.org/zap"

	"go.etcd.io/etcd/pkg/v3/traceutil"
	"go.etcd.io/etcd/server/v3/etcdserver/errors"
	"go.etcd.io/raft/v3"
)

var (
	readIndexRetryTime = 500 * time.Millisecond
)

func NewRead(server server, raft raftInterface) *Read {
	return &Read{
		server:   server,
		raft:     raft,
		waitC:    make(chan struct{}, 1),
		notifier: newNotifier(),
	}
}

type Read struct {
	server server
	raft   raftInterface
	mux    sync.RWMutex
	// read routine notifies etcd server that it waits for reading by sending an empty struct to
	// waitC
	waitC chan struct{}
	// notifier is used to notify the read routine that it can process the request
	// when there is no error
	notifier *notifier
}

type server interface {
	LeaderChanged() <-chan struct{}
	Stopping() <-chan struct{}
	Logger() *zap.Logger
	AppliedIndex() uint64
	ApplyWait(deadline uint64) <-chan struct{}
	NextRequestID() uint64
	RequestTimeout() time.Duration
	FirstCommitInTermNotify() <-chan struct{}
	Done() <-chan struct{}
}

type raftInterface interface {
	ReadState() <-chan raft.ReadState
	ReadIndex(ctx context.Context, rctx []byte) error
}

func (r *Read) LinearizableReadNotify(ctx context.Context) error {
	r.mux.RLock()
	nc := r.notifier
	r.mux.RUnlock()

	// signal linearizable loop for current notify if it hasn't been already
	select {
	case r.waitC <- struct{}{}:
	default:
	}

	// wait for read state notification
	select {
	case <-nc.c:
		return nc.err
	case <-ctx.Done():
		return ctx.Err()
	case <-r.server.Done():
		return errors.ErrStopped
	}
}

func (r *Read) LinearizableReadLoop() {
	for {
		leaderChangedNotifier := r.server.LeaderChanged()
		select {
		case <-leaderChangedNotifier:
			continue
		case <-r.waitC:
		case <-r.server.Stopping():
			return
		}

		// as a single loop is can unlock multiple reads, it is not very useful
		// to propagate the trace from Txn or Range.
		_, trace := traceutil.EnsureTrace(context.Background(), r.server.Logger(), "linearizableReadLoop")

		nextnr := newNotifier()
		r.mux.Lock()
		nr := r.notifier
		r.notifier = nextnr
		r.mux.Unlock()

		confirmedIndex, err := r.requestCurrentIndex(leaderChangedNotifier)
		if isStopped(err) {
			return
		}
		if err != nil {
			nr.notify(err)
			continue
		}

		trace.Step("read index received")

		trace.AddField(traceutil.Field{Key: "readStateIndex", Value: confirmedIndex})

		appliedIndex := r.server.AppliedIndex()
		trace.AddField(traceutil.Field{Key: "appliedIndex", Value: strconv.FormatUint(appliedIndex, 10)})

		if appliedIndex < confirmedIndex {
			select {
			case <-r.server.ApplyWait(confirmedIndex):
			case <-r.server.Stopping():
				return
			}
		}
		// unblock all l-reads requested at indices before confirmedIndex
		nr.notify(nil)
		trace.Step("applied index is now lower than readState.Index")

		trace.LogAllStepsIfLong(100 * time.Millisecond)
	}
}

func (r *Read) requestCurrentIndex(leaderChangedNotifier <-chan struct{}) (uint64, error) {
	requestIDs := map[uint64]struct{}{}
	requestID := r.server.NextRequestID()
	requestIDs[requestID] = struct{}{}
	err := r.sendReadIndex(requestID)
	if err != nil {
		return 0, err
	}

	lg := r.server.Logger()
	errorTimer := time.NewTimer(r.server.RequestTimeout())
	defer errorTimer.Stop()
	retryTimer := time.NewTimer(readIndexRetryTime)
	defer retryTimer.Stop()

	firstCommitInTermNotifier := r.server.FirstCommitInTermNotify()

	for {
		select {
		case rs := <-r.raft.ReadState():
			// Check again if leader changed as when multiple channels are ready, select picks randomly.
			select {
			case <-leaderChangedNotifier:
				readIndexFailed.Inc()
				return 0, errors.ErrLeaderChanged
			default:
			}
			responseID := uint64(0)
			if len(rs.RequestCtx) == 8 {
				responseID = binary.BigEndian.Uint64(rs.RequestCtx)
			}
			if _, ok := requestIDs[responseID]; !ok {
				// a previous request might time out. now we should ignore the response of it and
				// continue waiting for the response of the current requests.
				lg.Warn(
					"ignored out-of-date read index response; local node read indexes queueing up and waiting to be in sync with leader",
					zap.Uint64("received-request-id", responseID),
				)
				slowReadIndex.Inc()
				continue
			}
			return rs.Index, nil
		case <-leaderChangedNotifier:
			readIndexFailed.Inc()
			// return a retryable error.
			return 0, errors.ErrLeaderChanged
		case <-firstCommitInTermNotifier:
			firstCommitInTermNotifier = r.server.FirstCommitInTermNotify()
			lg.Info("first commit in current term: resending ReadIndex request")
			requestID = r.server.NextRequestID()
			requestIDs[requestID] = struct{}{}
			err := r.sendReadIndex(requestID)
			if err != nil {
				return 0, err
			}
			retryTimer.Reset(readIndexRetryTime)
			continue
		case <-retryTimer.C:
			lg.Warn(
				"waiting for ReadIndex response took too long, retrying",
				zap.Uint64("sent-request-id", requestID),
				zap.Duration("retry-timeout", readIndexRetryTime),
			)
			requestID = r.server.NextRequestID()
			requestIDs[requestID] = struct{}{}
			err := r.sendReadIndex(requestID)
			if err != nil {
				return 0, err
			}
			retryTimer.Reset(readIndexRetryTime)
			continue
		case <-errorTimer.C:
			lg.Warn(
				"timed out waiting for read index response (local node might have slow network)",
				zap.Duration("timeout", r.server.RequestTimeout()),
			)
			slowReadIndex.Inc()
			return 0, errors.ErrTimeout
		case <-r.server.Stopping():
			return 0, errors.ErrStopped
		}
	}
}

func (r *Read) sendReadIndex(requestIndex uint64) error {
	ctxToSend := uint64ToBigEndianBytes(requestIndex)

	cctx, cancel := context.WithTimeout(context.Background(), r.server.RequestTimeout())
	err := r.raft.ReadIndex(cctx, ctxToSend)
	cancel()
	if errorspkg.Is(err, raft.ErrStopped) {
		return err
	}
	if err != nil {
		lg := r.server.Logger()
		lg.Warn("failed to get read index from Raft", zap.Error(err))
		readIndexFailed.Inc()
		return err
	}
	return nil
}

func uint64ToBigEndianBytes(number uint64) []byte {
	byteResult := make([]byte, 8)
	binary.BigEndian.PutUint64(byteResult, number)
	return byteResult
}

func isStopped(err error) bool {
	return errorspkg.Is(err, raft.ErrStopped) || errorspkg.Is(err, errors.ErrStopped)
}
