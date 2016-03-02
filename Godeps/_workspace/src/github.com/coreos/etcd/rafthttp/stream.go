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
	"strconv"
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
	streamTypeMsgApp   streamType = "msgapp"

	streamBufSize = 4096
)

var (
	errUnsupportedStreamType = fmt.Errorf("unsupported stream type")

	// the key is in string format "major.minor.patch"
	supportedStream = map[string][]streamType{
		"2.0.0": {streamTypeMsgApp},
		"2.1.0": {streamTypeMsgApp, streamTypeMsgAppV2, streamTypeMessage},
		"2.2.0": {streamTypeMsgApp, streamTypeMsgAppV2, streamTypeMessage},
	}
)

type streamType string

func (t streamType) endpoint() string {
	switch t {
	case streamTypeMsgApp: // for backward compatibility of v2.0
		return RaftStreamPrefix
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
	case streamTypeMsgApp:
		return "stream MsgApp"
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
	t       streamType
	termStr string
	io.Writer
	http.Flusher
	io.Closer
}

// streamWriter is a long-running go-routine that writes messages into the
// attached outgoingConn.
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
	var msgc chan raftpb.Message
	var heartbeatc <-chan time.Time
	var t streamType
	var msgAppTerm uint64
	var enc encoder
	var flusher http.Flusher
	tickc := time.Tick(ConnReadTimeout / 3)

	for {
		select {
		case <-heartbeatc:
			start := time.Now()
			if err := enc.encode(linkHeartbeatMessage); err != nil {
				reportSentFailure(string(t), linkHeartbeatMessage)

				cw.status.deactivate(failureType{source: t.String(), action: "heartbeat"}, err.Error())
				cw.close()
				heartbeatc, msgc = nil, nil
				continue
			}
			flusher.Flush()
			reportSentDuration(string(t), linkHeartbeatMessage, time.Since(start))
		case m := <-msgc:
			if t == streamTypeMsgApp && m.Term != msgAppTerm {
				// TODO: reasonable retry logic
				if m.Term > msgAppTerm {
					cw.close()
					heartbeatc, msgc = nil, nil
					// TODO: report to raft at peer level
					cw.r.ReportUnreachable(m.To)
				}
				continue
			}
			start := time.Now()
			if err := enc.encode(m); err != nil {
				reportSentFailure(string(t), m)

				cw.status.deactivate(failureType{source: t.String(), action: "write"}, err.Error())
				cw.close()
				heartbeatc, msgc = nil, nil
				cw.r.ReportUnreachable(m.To)
				continue
			}
			flusher.Flush()
			reportSentDuration(string(t), m, time.Since(start))
		case conn := <-cw.connc:
			cw.close()
			t = conn.t
			switch conn.t {
			case streamTypeMsgApp:
				var err error
				msgAppTerm, err = strconv.ParseUint(conn.termStr, 10, 64)
				if err != nil {
					plog.Panicf("could not parse term %s to uint (%v)", conn.termStr, err)
				}
				enc = &msgAppEncoder{w: conn.Writer, fs: cw.fs}
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
// endponit and reads messages from the response body returned.
type streamReader struct {
	tr            http.RoundTripper
	picker        *urlPicker
	t             streamType
	local, remote types.ID
	cid           types.ID
	status        *peerStatus
	recvc         chan<- raftpb.Message
	propc         chan<- raftpb.Message
	errorc        chan<- error

	mu         sync.Mutex
	msgAppTerm uint64
	cancel     func()
	closer     io.Closer
	stopc      chan struct{}
	done       chan struct{}
}

func startStreamReader(tr http.RoundTripper, picker *urlPicker, t streamType, local, remote, cid types.ID, status *peerStatus, recvc chan<- raftpb.Message, propc chan<- raftpb.Message, errorc chan<- error, term uint64) *streamReader {
	r := &streamReader{
		tr:         tr,
		picker:     picker,
		t:          t,
		local:      local,
		remote:     remote,
		cid:        cid,
		status:     status,
		recvc:      recvc,
		propc:      propc,
		errorc:     errorc,
		msgAppTerm: term,
		stopc:      make(chan struct{}),
		done:       make(chan struct{}),
	}
	go r.run()
	return r
}

func (cr *streamReader) run() {
	for {
		t := cr.t
		rc, err := cr.dial(t)
		// downgrade to streamTypeMsgApp if the remote doesn't support
		// streamTypeMsgAppV2
		if t == streamTypeMsgAppV2 && err == errUnsupportedStreamType {
			t = streamTypeMsgApp
			rc, err = cr.dial(t)
		}
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
			// stream msgapp is only used for etcd 2.0, and etcd 2.0 doesn't
			// heartbeat on the idle stream, so it is expected to time out.
			case t == streamTypeMsgApp && isNetworkTimeoutError(err):
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
	case streamTypeMsgApp:
		dec = &msgAppDecoder{r: rc, local: cr.local, remote: cr.remote, term: cr.msgAppTerm}
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
		switch {
		case err != nil:
			cr.mu.Lock()
			cr.close()
			cr.mu.Unlock()
			return err
		case isLinkHeartbeatMessage(m):
			// do nothing for linkHeartbeatMessage
		default:
			recvc := cr.recvc
			if m.Type == raftpb.MsgProp {
				recvc = cr.propc
			}
			select {
			case recvc <- m:
			default:
				if cr.status.isActive() {
					plog.Warningf("dropped %s from %s since receiving buffer is full", m.Type, types.ID(m.From))
				} else {
					plog.Debugf("dropped %s from %s since receiving buffer is full", m.Type, types.ID(m.From))
				}
			}
		}
	}
}

// updateMsgAppTerm updates the term for MsgApp stream, and closes
// the existing MsgApp stream if term is updated.
func (cr *streamReader) updateMsgAppTerm(term uint64) {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	if cr.msgAppTerm >= term {
		return
	}
	cr.msgAppTerm = term
	if cr.t == streamTypeMsgApp {
		cr.close()
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

func (cr *streamReader) isWorking() bool {
	cr.mu.Lock()
	defer cr.mu.Unlock()
	return cr.closer != nil
}

func (cr *streamReader) dial(t streamType) (io.ReadCloser, error) {
	u := cr.picker.pick()
	cr.mu.Lock()
	term := cr.msgAppTerm
	cr.mu.Unlock()

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
	if t == streamTypeMsgApp {
		req.Header.Set("X-Raft-Term", strconv.FormatUint(term, 10))
	}

	cr.mu.Lock()
	select {
	case <-cr.stopc:
		cr.mu.Unlock()
		return nil, fmt.Errorf("stream reader is stopped")
	default:
	}
	cr.cancel = httputil.RequestCanceler(cr.tr, req)
	cr.mu.Unlock()

	resp, err := cr.tr.RoundTrip(req)
	if err != nil {
		cr.picker.unreachable(u)
		return nil, err
	}

	rv := serverVersion(resp.Header)
	lv := semver.Must(semver.NewVersion(version.Version))
	if compareMajorMinorVersion(rv, lv) == -1 && !checkStreamSupport(rv, t) {
		resp.Body.Close()
		return nil, errUnsupportedStreamType
	}

	switch resp.StatusCode {
	case http.StatusGone:
		resp.Body.Close()
		err := fmt.Errorf("the member has been permanently removed from the cluster")
		select {
		case cr.errorc <- err:
		default:
		}
		return nil, err
	case http.StatusOK:
		return resp.Body, nil
	case http.StatusNotFound:
		resp.Body.Close()
		return nil, fmt.Errorf("remote member %s could not recognize local member", cr.remote)
	case http.StatusPreconditionFailed:
		b, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			cr.picker.unreachable(u)
			return nil, err
		}
		resp.Body.Close()

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
		resp.Body.Close()
		return nil, fmt.Errorf("unhandled http status %d", resp.StatusCode)
	}
}

func (cr *streamReader) close() {
	if cr.closer != nil {
		cr.closer.Close()
	}
	cr.closer = nil
}

func canUseMsgAppStream(m raftpb.Message) bool {
	return m.Type == raftpb.MsgApp && m.Term == m.LogTerm
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

func isNetworkTimeoutError(err error) bool {
	nerr, ok := err.(net.Error)
	return ok && nerr.Timeout()
}
