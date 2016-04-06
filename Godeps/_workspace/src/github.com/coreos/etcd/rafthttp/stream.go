// Copyright 2015 CoreOS, Inc.
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
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/coreos/etcd/etcdserver/stats"
	"github.com/coreos/etcd/pkg/httputil"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/version"
	"github.com/coreos/go-semver/semver"
)

const (
	streamTypeMessage  streamType = "message"
	streamTypeMsgAppV2 streamType = "msgappv2"

	streamBufSize = 4096
)

var (
	errUnsupportedStreamType = fmt.Errorf("unsupported stream type")

	// the key is in string format "major.minor.patch"
	supportedStream = map[string][]streamType{
		"2.0.0": {},
		"2.1.0": {streamTypeMsgAppV2, streamTypeMessage},
		"2.2.0": {streamTypeMsgAppV2, streamTypeMessage},
		"2.3.0": {streamTypeMsgAppV2, streamTypeMessage},
	}
)

type streamType string

func (t streamType) endpoint() string {
	switch t {
	case streamTypeMsgAppV2:
		return path.Join(RaftStreamPrefix, "msgapp")
	case streamTypeMessage:
		return path.Join(RaftStreamPrefix, "message")
	default:
		plog.Panicf("unhandled stream type %v", t)
		return ""
	}
}

func (t streamType) String() string {
	switch t {
	case streamTypeMsgAppV2:
		return "stream MsgApp v2"
	case streamTypeMessage:
		return "stream Message"
	default:
		return "unknown stream"
	}
}

var (
	// linkHeartbeatMessage is a special message used as heartbeat message in
	// link layer. It never conflicts with messages from raft because raft
	// doesn't send out messages without From and To fields.
	linkHeartbeatMessage = raftpb.Message{Type: raftpb.MsgHeartbeat}
)

func isLinkHeartbeatMessage(m raftpb.Message) bool {
	return m.Type == raftpb.MsgHeartbeat && m.From == 0 && m.To == 0
}

type outgoingConn struct {
	t streamType
	io.Writer
	http.Flusher
	io.Closer
}

// streamWriter writes messages to the attached outgoingConn.
type streamWriter struct {
	id     types.ID
	status *peerStatus
	fs     *stats.FollowerStats
	r      Raft

	mu      sync.Mutex // guard field working and closer
	closer  io.Closer
	working bool

	msgc  chan raftpb.Message
	connc chan *outgoingConn
	stopc chan struct{}
	done  chan struct{}
}

// startStreamWriter creates a streamWrite and starts a long running go-routine that accepts
// messages and writes to the attached outgoing connection.
func startStreamWriter(id types.ID, status *peerStatus, fs *stats.FollowerStats, r Raft) *streamWriter {
	w := &streamWriter{
		id:     id,
		status: status,
		fs:     fs,
		r:      r,
		msgc:   make(chan raftpb.Message, streamBufSize),
		connc:  make(chan *outgoingConn),
		stopc:  make(chan struct{}),
		done:   make(chan struct{}),
	}
	go w.run()
	return w
}

func (cw *streamWriter) run() {
	var (
		msgc       chan raftpb.Message
		heartbeatc <-chan time.Time
		t          streamType
		enc        encoder
		flusher    http.Flusher
		batched    int
	)
	tickc := time.Tick(ConnReadTimeout / 3)

	for {
		select {
		case <-heartbeatc:
			start := time.Now()
			err := enc.encode(linkHeartbeatMessage)
			if err == nil {
				flusher.Flush()
				batched = 0
				reportSentDuration(string(t), linkHeartbeatMessage, time.Since(start))
				continue
			}

			reportSentFailure(string(t), linkHeartbeatMessage)
			cw.status.deactivate(failureType{source: t.String(), action: "heartbeat"}, err.Error())
			cw.close()
			heartbeatc, msgc = nil, nil

		case m := <-msgc:
			start := time.Now()
			err := enc.encode(m)
			if err == nil {
				if len(msgc) == 0 || batched > streamBufSize/2 {
					flusher.Flush()
					batched = 0
				} else {
					batched++
				}

				reportSentDuration(string(t), m, time.Since(start))
				continue
			}

			reportSentFailure(string(t), m)
			cw.status.deactivate(failureType{source: t.String(), action: "write"}, err.Error())
			cw.close()
			heartbeatc, msgc = nil, nil
			cw.r.ReportUnreachable(m.To)

		case conn := <-cw.connc:
			cw.close()
			t = conn.t
			switch conn.t {
			case streamTypeMsgAppV2:
				enc = newMsgAppV2Encoder(conn.Writer, cw.fs)
			case streamTypeMessage:
				enc = &messageEncoder{w: conn.Writer}
			default:
				plog.Panicf("unhandled stream type %s", conn.t)
			}
			flusher = conn.Flusher
			cw.mu.Lock()
			cw.status.activate()
			cw.closer = conn.Closer
			cw.working = true
			cw.mu.Unlock()
			heartbeatc, msgc = tickc, cw.msgc
		case <-cw.stopc:
			cw.close()
			close(cw.done)
			return
		}
	}
}

func (cw *streamWriter) writec() (chan<- raftpb.Message, bool) {
	cw.mu.Lock()
	defer cw.mu.Unlock()
	return cw.msgc, cw.working
}

func (cw *streamWriter) close() {
	cw.mu.Lock()
	defer cw.mu.Unlock()
	if !cw.working {
		return
	}
	cw.closer.Close()
	if len(cw.msgc) > 0 {
		cw.r.ReportUnreachable(uint64(cw.id))
	}
	cw.msgc = make(chan raftpb.Message, streamBufSize)
	cw.working = false
}

func (cw *streamWriter) attach(conn *outgoingConn) bool {
	select {
	case cw.connc <- conn:
		return true
	case <-cw.done:
		return false
	}
}

func (cw *streamWriter) stop() {
	close(cw.stopc)
	<-cw.done
}

// streamReader is a long-running go-routine that dials to the remote stream
// endpoint and reads messages from the response body returned.
type streamReader struct {
	tr            *Transport
	picker        *urlPicker
	t             streamType
	local, remote types.ID
	cid           types.ID
	status        *peerStatus
	recvc         chan<- raftpb.Message
	propc         chan<- raftpb.Message
	errorc        chan<- error

	mu     sync.Mutex
	paused bool
	cancel func()
	closer io.Closer
	stopc  chan struct{}
	done   chan struct{}
}

func startStreamReader(tr *Transport, picker *urlPicker, t streamType, local, remote, cid types.ID, status *peerStatus, recvc chan<- raftpb.Message, propc chan<- raftpb.Message, errorc chan<- error) *streamReader {
	r := &streamReader{
		tr:     tr,
		picker: picker,
		t:      t,
		local:  local,
		remote: remote,
		cid:    cid,
		status: status,
		recvc:  recvc,
		propc:  propc,
		errorc: errorc,
		stopc:  make(chan struct{}),
		done:   make(chan struct{}),
	}
	go r.run()
	return r
}

func (cr *streamReader) run() {
	for {
		t := cr.t
		rc, err := cr.dial(t)
		if err != nil {
			if err != errUnsupportedStreamType {
				cr.status.deactivate(failureType{source: t.String(), action: "dial"}, err.Error())
			}
		} else {
			cr.status.activate()
			err := cr.decodeLoop(rc, t)
			switch {
			// all data is read out
			case err == io.EOF:
			// connection is closed by the remote
			case isClosedConnectionError(err):
			default:
				cr.status.deactivate(failureType{source: t.String(), action: "read"}, err.Error())
			}
		}
		select {
		// Wait 100ms to create a new stream, so it doesn't bring too much
		// overhead when retry.
		case <-time.After(100 * time.Millisecond):
		case <-cr.stopc:
			close(cr.done)
			return
		}
	}
}

func (cr *streamReader) decodeLoop(rc io.ReadCloser, t streamType) error {
	var dec decoder
	cr.mu.Lock()
	switch t {
	case streamTypeMsgAppV2:
		dec = newMsgAppV2Decoder(rc, cr.local, cr.remote)
	case streamTypeMessage:
		dec = &messageDecoder{r: rc}
	default:
		plog.Panicf("unhandled stream type %s", t)
	}
	cr.closer = rc
	cr.mu.Unlock()

	for {
		m, err := dec.decode()
		if err != nil {
			cr.mu.Lock()
			cr.close()
			cr.mu.Unlock()
			return err
		}

		cr.mu.Lock()
		paused := cr.paused
		cr.mu.Unlock()

		if paused {
			continue
		}

		if isLinkHeartbeatMessage(m) {
			// raft is not interested in link layer
			// heartbeat message, so we should ignore
			// it.
			continue
		}

		recvc := cr.recvc
		if m.Type == raftpb.MsgProp {
			recvc = cr.propc
		}

		select {
		case recvc <- m:
		default:
			if cr.status.isActive() {
				plog.MergeWarningf("dropped internal raft message from %s since receiving buffer is full (overloaded network)", types.ID(m.From))
			}
			plog.Debugf("dropped %s from %s since receiving buffer is full", m.Type, types.ID(m.From))
		}
	}
}

func (cr *streamReader) stop() {
	close(cr.stopc)
	cr.mu.Lock()
	if cr.cancel != nil {
		cr.cancel()
	}
	cr.close()
	cr.mu.Unlock()
	<-cr.done
}

func (cr *streamReader) dial(t streamType) (io.ReadCloser, error) {
	u := cr.picker.pick()
	uu := u
	uu.Path = path.Join(t.endpoint(), cr.local.String())

	req, err := http.NewRequest("GET", uu.String(), nil)
	if err != nil {
		cr.picker.unreachable(u)
		return nil, fmt.Errorf("failed to make http request to %s (%v)", u, err)
	}
	req.Header.Set("X-Server-From", cr.local.String())
	req.Header.Set("X-Server-Version", version.Version)
	req.Header.Set("X-Min-Cluster-Version", version.MinClusterVersion)
	req.Header.Set("X-Etcd-Cluster-ID", cr.cid.String())
	req.Header.Set("X-Raft-To", cr.remote.String())

	setPeerURLsHeader(req, cr.tr.URLs)

	cr.mu.Lock()
	select {
	case <-cr.stopc:
		cr.mu.Unlock()
		return nil, fmt.Errorf("stream reader is stopped")
	default:
	}
	cr.cancel = httputil.RequestCanceler(cr.tr.streamRt, req)
	cr.mu.Unlock()

	resp, err := cr.tr.streamRt.RoundTrip(req)
	if err != nil {
		cr.picker.unreachable(u)
		return nil, err
	}

	rv := serverVersion(resp.Header)
	lv := semver.Must(semver.NewVersion(version.Version))
	if compareMajorMinorVersion(rv, lv) == -1 && !checkStreamSupport(rv, t) {
		httputil.GracefulClose(resp)
		cr.picker.unreachable(u)
		return nil, errUnsupportedStreamType
	}

	switch resp.StatusCode {
	case http.StatusGone:
		httputil.GracefulClose(resp)
		cr.picker.unreachable(u)
		err := fmt.Errorf("the member has been permanently removed from the cluster")
		select {
		case cr.errorc <- err:
		default:
		}
		return nil, err
	case http.StatusOK:
		return resp.Body, nil
	case http.StatusNotFound:
		httputil.GracefulClose(resp)
		cr.picker.unreachable(u)
		return nil, fmt.Errorf("remote member %s could not recognize local member", cr.remote)
	case http.StatusPreconditionFailed:
		b, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			cr.picker.unreachable(u)
			return nil, err
		}
		httputil.GracefulClose(resp)
		cr.picker.unreachable(u)

		switch strings.TrimSuffix(string(b), "\n") {
		case errIncompatibleVersion.Error():
			plog.Errorf("request sent was ignored by peer %s (server version incompatible)", cr.remote)
			return nil, errIncompatibleVersion
		case errClusterIDMismatch.Error():
			plog.Errorf("request sent was ignored (cluster ID mismatch: remote[%s]=%s, local=%s)",
				cr.remote, resp.Header.Get("X-Etcd-Cluster-ID"), cr.cid)
			return nil, errClusterIDMismatch
		default:
			return nil, fmt.Errorf("unhandled error %q when precondition failed", string(b))
		}
	default:
		httputil.GracefulClose(resp)
		cr.picker.unreachable(u)
		return nil, fmt.Errorf("unhandled http status %d", resp.StatusCode)
	}
}

func (cr *streamReader) close() {
	if cr.closer != nil {
		cr.closer.Close()
	}
	cr.closer = nil
}

func (cr *streamReader) pause() {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	cr.paused = true
}

func (cr *streamReader) resume() {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	cr.paused = false
}

func isClosedConnectionError(err error) bool {
	operr, ok := err.(*net.OpError)
	return ok && operr.Err.Error() == "use of closed network connection"
}

// checkStreamSupport checks whether the stream type is supported in the
// given version.
func checkStreamSupport(v *semver.Version, t streamType) bool {
	nv := &semver.Version{Major: v.Major, Minor: v.Minor}
	for _, s := range supportedStream[nv.String()] {
		if s == t {
			return true
		}
	}
	return false
}
