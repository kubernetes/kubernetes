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
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/coreos/etcd/etcdserver/stats"
	"github.com/coreos/etcd/pkg/testutil"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/version"
	"github.com/coreos/go-semver/semver"
)

// TestStreamWriterAttachOutgoingConn tests that outgoingConn can be attached
// to streamWriter. After that, streamWriter can use it to send messages
// continuously, and closes it when stopped.
func TestStreamWriterAttachOutgoingConn(t *testing.T) {
	sw := startStreamWriter(types.ID(1), newPeerStatus(types.ID(1)), &stats.FollowerStats{}, &fakeRaft{})
	// the expected initial state of streamWriter is not working
	if _, ok := sw.writec(); ok != false {
		t.Errorf("initial working status = %v, want false", ok)
	}

	// repeat tests to ensure streamWriter can use last attached connection
	var wfc *fakeWriteFlushCloser
	for i := 0; i < 3; i++ {
		prevwfc := wfc
		wfc = &fakeWriteFlushCloser{}
		sw.attach(&outgoingConn{t: streamTypeMessage, Writer: wfc, Flusher: wfc, Closer: wfc})

		// sw.attach happens asynchronously. Waits for its result in a for loop to make the
		// test more robust on slow CI.
		for j := 0; j < 3; j++ {
			testutil.WaitSchedule()
			// previous attached connection should be closed
			if prevwfc != nil && prevwfc.Closed() != true {
				continue
			}
			// write chan is available
			if _, ok := sw.writec(); ok != true {
				continue
			}
		}

		// previous attached connection should be closed
		if prevwfc != nil && prevwfc.Closed() != true {
			t.Errorf("#%d: close of previous connection = %v, want true", i, prevwfc.Closed())
		}
		// write chan is available
		if _, ok := sw.writec(); ok != true {
			t.Errorf("#%d: working status = %v, want true", i, ok)
		}

		sw.msgc <- raftpb.Message{}
		testutil.WaitSchedule()
		// write chan is available
		if _, ok := sw.writec(); ok != true {
			t.Errorf("#%d: working status = %v, want true", i, ok)
		}
		if wfc.Written() == 0 {
			t.Errorf("#%d: failed to write to the underlying connection", i)
		}
	}

	sw.stop()
	// write chan is unavailable since the writer is stopped.
	if _, ok := sw.writec(); ok != false {
		t.Errorf("working status after stop = %v, want false", ok)
	}
	if wfc.Closed() != true {
		t.Errorf("failed to close the underlying connection")
	}
}

// TestStreamWriterAttachBadOutgoingConn tests that streamWriter with bad
// outgoingConn will close the outgoingConn and fall back to non-working status.
func TestStreamWriterAttachBadOutgoingConn(t *testing.T) {
	sw := startStreamWriter(types.ID(1), newPeerStatus(types.ID(1)), &stats.FollowerStats{}, &fakeRaft{})
	defer sw.stop()
	wfc := &fakeWriteFlushCloser{err: errors.New("blah")}
	sw.attach(&outgoingConn{t: streamTypeMessage, Writer: wfc, Flusher: wfc, Closer: wfc})

	sw.msgc <- raftpb.Message{}
	testutil.WaitSchedule()
	// no longer working
	if _, ok := sw.writec(); ok != false {
		t.Errorf("working = %v, want false", ok)
	}
	if wfc.Closed() != true {
		t.Errorf("failed to close the underlying connection")
	}
}

func TestStreamReaderDialRequest(t *testing.T) {
	for i, tt := range []streamType{streamTypeMessage, streamTypeMsgAppV2} {
		tr := &roundTripperRecorder{}
		sr := &streamReader{
			tr:     &Transport{streamRt: tr},
			picker: mustNewURLPicker(t, []string{"http://localhost:2380"}),
			local:  types.ID(1),
			remote: types.ID(2),
			cid:    types.ID(1),
		}
		sr.dial(tt)

		req := tr.Request()
		wurl := fmt.Sprintf("http://localhost:2380" + tt.endpoint() + "/1")
		if req.URL.String() != wurl {
			t.Errorf("#%d: url = %s, want %s", i, req.URL.String(), wurl)
		}
		if w := "GET"; req.Method != w {
			t.Errorf("#%d: method = %s, want %s", i, req.Method, w)
		}
		if g := req.Header.Get("X-Etcd-Cluster-ID"); g != "1" {
			t.Errorf("#%d: header X-Etcd-Cluster-ID = %s, want 1", i, g)
		}
		if g := req.Header.Get("X-Raft-To"); g != "2" {
			t.Errorf("#%d: header X-Raft-To = %s, want 2", i, g)
		}
	}
}

// TestStreamReaderDialResult tests the result of the dial func call meets the
// HTTP response received.
func TestStreamReaderDialResult(t *testing.T) {
	tests := []struct {
		code  int
		err   error
		wok   bool
		whalt bool
	}{
		{0, errors.New("blah"), false, false},
		{http.StatusOK, nil, true, false},
		{http.StatusMethodNotAllowed, nil, false, false},
		{http.StatusNotFound, nil, false, false},
		{http.StatusPreconditionFailed, nil, false, false},
		{http.StatusGone, nil, false, true},
	}
	for i, tt := range tests {
		h := http.Header{}
		h.Add("X-Server-Version", version.Version)
		tr := &respRoundTripper{
			code:   tt.code,
			header: h,
			err:    tt.err,
		}
		sr := &streamReader{
			tr:     &Transport{streamRt: tr},
			picker: mustNewURLPicker(t, []string{"http://localhost:2380"}),
			local:  types.ID(1),
			remote: types.ID(2),
			cid:    types.ID(1),
			errorc: make(chan error, 1),
		}

		_, err := sr.dial(streamTypeMessage)
		if ok := err == nil; ok != tt.wok {
			t.Errorf("#%d: ok = %v, want %v", i, ok, tt.wok)
		}
		if halt := len(sr.errorc) > 0; halt != tt.whalt {
			t.Errorf("#%d: halt = %v, want %v", i, halt, tt.whalt)
		}
	}
}

// TestStreamReaderDialDetectUnsupport tests that dial func could find
// out that the stream type is not supported by the remote.
func TestStreamReaderDialDetectUnsupport(t *testing.T) {
	for i, typ := range []streamType{streamTypeMsgAppV2, streamTypeMessage} {
		// the response from etcd 2.0
		tr := &respRoundTripper{
			code:   http.StatusNotFound,
			header: http.Header{},
		}
		sr := &streamReader{
			tr:     &Transport{streamRt: tr},
			picker: mustNewURLPicker(t, []string{"http://localhost:2380"}),
			local:  types.ID(1),
			remote: types.ID(2),
			cid:    types.ID(1),
		}

		_, err := sr.dial(typ)
		if err != errUnsupportedStreamType {
			t.Errorf("#%d: error = %v, want %v", i, err, errUnsupportedStreamType)
		}
	}
}

// TestStream tests that streamReader and streamWriter can build stream to
// send messages between each other.
func TestStream(t *testing.T) {
	recvc := make(chan raftpb.Message, streamBufSize)
	propc := make(chan raftpb.Message, streamBufSize)
	msgapp := raftpb.Message{
		Type:    raftpb.MsgApp,
		From:    2,
		To:      1,
		Term:    1,
		LogTerm: 1,
		Index:   3,
		Entries: []raftpb.Entry{{Term: 1, Index: 4}},
	}

	tests := []struct {
		t  streamType
		m  raftpb.Message
		wc chan raftpb.Message
	}{
		{
			streamTypeMessage,
			raftpb.Message{Type: raftpb.MsgProp, To: 2},
			propc,
		},
		{
			streamTypeMessage,
			msgapp,
			recvc,
		},
		{
			streamTypeMsgAppV2,
			msgapp,
			recvc,
		},
	}
	for i, tt := range tests {
		h := &fakeStreamHandler{t: tt.t}
		srv := httptest.NewServer(h)
		defer srv.Close()

		sw := startStreamWriter(types.ID(1), newPeerStatus(types.ID(1)), &stats.FollowerStats{}, &fakeRaft{})
		defer sw.stop()
		h.sw = sw

		picker := mustNewURLPicker(t, []string{srv.URL})
		tr := &Transport{streamRt: &http.Transport{}}
		sr := startStreamReader(tr, picker, tt.t, types.ID(1), types.ID(2), types.ID(1), newPeerStatus(types.ID(1)), recvc, propc, nil)
		defer sr.stop()
		// wait for stream to work
		var writec chan<- raftpb.Message
		for {
			var ok bool
			if writec, ok = sw.writec(); ok {
				break
			}
			time.Sleep(time.Millisecond)
		}

		writec <- tt.m
		var m raftpb.Message
		select {
		case m = <-tt.wc:
		case <-time.After(time.Second):
			t.Fatalf("#%d: failed to receive message from the channel", i)
		}
		if !reflect.DeepEqual(m, tt.m) {
			t.Fatalf("#%d: message = %+v, want %+v", i, m, tt.m)
		}
	}
}

func TestCheckStreamSupport(t *testing.T) {
	tests := []struct {
		v *semver.Version
		t streamType
		w bool
	}{
		// support
		{
			semver.Must(semver.NewVersion("2.1.0")),
			streamTypeMsgAppV2,
			true,
		},
		// ignore patch
		{
			semver.Must(semver.NewVersion("2.1.9")),
			streamTypeMsgAppV2,
			true,
		},
		// ignore prerelease
		{
			semver.Must(semver.NewVersion("2.1.0-alpha")),
			streamTypeMsgAppV2,
			true,
		},
	}
	for i, tt := range tests {
		if g := checkStreamSupport(tt.v, tt.t); g != tt.w {
			t.Errorf("#%d: check = %v, want %v", i, g, tt.w)
		}
	}
}

type fakeWriteFlushCloser struct {
	mu      sync.Mutex
	err     error
	written int
	closed  bool
}

func (wfc *fakeWriteFlushCloser) Write(p []byte) (n int, err error) {
	wfc.mu.Lock()
	defer wfc.mu.Unlock()

	wfc.written += len(p)
	return len(p), wfc.err
}

func (wfc *fakeWriteFlushCloser) Flush() {}

func (wfc *fakeWriteFlushCloser) Close() error {
	wfc.mu.Lock()
	defer wfc.mu.Unlock()

	wfc.closed = true
	return wfc.err
}

func (wfc *fakeWriteFlushCloser) Written() int {
	wfc.mu.Lock()
	defer wfc.mu.Unlock()
	return wfc.written
}

func (wfc *fakeWriteFlushCloser) Closed() bool {
	wfc.mu.Lock()
	defer wfc.mu.Unlock()
	return wfc.closed
}

type fakeStreamHandler struct {
	t  streamType
	sw *streamWriter
}

func (h *fakeStreamHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("X-Server-Version", version.Version)
	w.(http.Flusher).Flush()
	c := newCloseNotifier()
	h.sw.attach(&outgoingConn{
		t:       h.t,
		Writer:  w,
		Flusher: w.(http.Flusher),
		Closer:  c,
	})
	<-c.closeNotify()
}
