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

package rafthttp

import (
	"bytes"
	"context"
	"io"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/coreos/etcd/pkg/httputil"
	pioutil "github.com/coreos/etcd/pkg/ioutil"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft"
	"github.com/coreos/etcd/snap"
)

var (
	// timeout for reading snapshot response body
	snapResponseReadTimeout = 5 * time.Second
)

type snapshotSender struct {
	from, to types.ID
	cid      types.ID

	tr     *Transport
	picker *urlPicker
	status *peerStatus
	r      Raft
	errorc chan error

	stopc chan struct{}
}

func newSnapshotSender(tr *Transport, picker *urlPicker, to types.ID, status *peerStatus) *snapshotSender {
	return &snapshotSender{
		from:   tr.ID,
		to:     to,
		cid:    tr.ClusterID,
		tr:     tr,
		picker: picker,
		status: status,
		r:      tr.Raft,
		errorc: tr.ErrorC,
		stopc:  make(chan struct{}),
	}
}

func (s *snapshotSender) stop() { close(s.stopc) }

func (s *snapshotSender) send(merged snap.Message) {
	start := time.Now()

	m := merged.Message
	to := types.ID(m.To).String()

	body := createSnapBody(merged)
	defer body.Close()

	u := s.picker.pick()
	req := createPostRequest(u, RaftSnapshotPrefix, body, "application/octet-stream", s.tr.URLs, s.from, s.cid)

	plog.Infof("start to send database snapshot [index: %d, to %s]...", m.Snapshot.Metadata.Index, types.ID(m.To))
	snapshotSendInflights.WithLabelValues(to).Inc()
	defer func() {
		snapshotSendInflights.WithLabelValues(to).Dec()
	}()

	err := s.post(req)
	defer merged.CloseWithError(err)
	if err != nil {
		plog.Warningf("database snapshot [index: %d, to: %s] failed to be sent out (%v)", m.Snapshot.Metadata.Index, types.ID(m.To), err)

		// errMemberRemoved is a critical error since a removed member should
		// always be stopped. So we use reportCriticalError to report it to errorc.
		if err == errMemberRemoved {
			reportCriticalError(err, s.errorc)
		}

		s.picker.unreachable(u)
		s.status.deactivate(failureType{source: sendSnap, action: "post"}, err.Error())
		s.r.ReportUnreachable(m.To)
		// report SnapshotFailure to raft state machine. After raft state
		// machine knows about it, it would pause a while and retry sending
		// new snapshot message.
		s.r.ReportSnapshot(m.To, raft.SnapshotFailure)
		sentFailures.WithLabelValues(to).Inc()
		snapshotSendFailures.WithLabelValues(to).Inc()
		return
	}
	s.status.activate()
	s.r.ReportSnapshot(m.To, raft.SnapshotFinish)
	plog.Infof("database snapshot [index: %d, to: %s] sent out successfully", m.Snapshot.Metadata.Index, types.ID(m.To))

	sentBytes.WithLabelValues(to).Add(float64(merged.TotalSize))

	snapshotSend.WithLabelValues(to).Inc()
	snapshotSendSeconds.WithLabelValues(to).Observe(time.Since(start).Seconds())
}

// post posts the given request.
// It returns nil when request is sent out and processed successfully.
func (s *snapshotSender) post(req *http.Request) (err error) {
	ctx, cancel := context.WithCancel(context.Background())
	req = req.WithContext(ctx)
	defer cancel()

	type responseAndError struct {
		resp *http.Response
		body []byte
		err  error
	}
	result := make(chan responseAndError, 1)

	go func() {
		resp, err := s.tr.pipelineRt.RoundTrip(req)
		if err != nil {
			result <- responseAndError{resp, nil, err}
			return
		}

		// close the response body when timeouts.
		// prevents from reading the body forever when the other side dies right after
		// successfully receives the request body.
		time.AfterFunc(snapResponseReadTimeout, func() { httputil.GracefulClose(resp) })
		body, err := ioutil.ReadAll(resp.Body)
		result <- responseAndError{resp, body, err}
	}()

	select {
	case <-s.stopc:
		return errStopped
	case r := <-result:
		if r.err != nil {
			return r.err
		}
		return checkPostResponse(r.resp, r.body, req, s.to)
	}
}

func createSnapBody(merged snap.Message) io.ReadCloser {
	buf := new(bytes.Buffer)
	enc := &messageEncoder{w: buf}
	// encode raft message
	if err := enc.encode(&merged.Message); err != nil {
		plog.Panicf("encode message error (%v)", err)
	}

	return &pioutil.ReaderAndCloser{
		Reader: io.MultiReader(buf, merged.ReadCloser),
		Closer: merged.ReadCloser,
	}
}
