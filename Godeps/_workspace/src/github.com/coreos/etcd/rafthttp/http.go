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
	"io/ioutil"
	"net/http"
	"path"

	pioutil "github.com/coreos/etcd/pkg/ioutil"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/version"
	"golang.org/x/net/context"
)

const (
	ConnReadLimitByte = 64 * 1024
)

var (
	RaftPrefix       = "/raft"
	ProbingPrefix    = path.Join(RaftPrefix, "probing")
	RaftStreamPrefix = path.Join(RaftPrefix, "stream")

	errIncompatibleVersion = errors.New("incompatible version")
	errClusterIDMismatch   = errors.New("cluster ID mismatch")
)

func NewHandler(r Raft, cid types.ID) http.Handler {
	return &handler{
		r:   r,
		cid: cid,
	}
}

type peerGetter interface {
	Get(id types.ID) Peer
}

func newStreamHandler(peerGetter peerGetter, r Raft, id, cid types.ID) http.Handler {
	return &streamHandler{
		peerGetter: peerGetter,
		r:          r,
		id:         id,
		cid:        cid,
	}
}

type writerToResponse interface {
	WriteTo(w http.ResponseWriter)
}

type handler struct {
	r   Raft
	cid types.ID
}

func (h *handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		w.Header().Set("Allow", "POST")
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	if err := checkVersionCompability(r.Header.Get("X-Server-From"), serverVersion(r.Header), minClusterVersion(r.Header)); err != nil {
		plog.Errorf("request received was ignored (%v)", err)
		http.Error(w, errIncompatibleVersion.Error(), http.StatusPreconditionFailed)
		return
	}

	wcid := h.cid.String()
	w.Header().Set("X-Etcd-Cluster-ID", wcid)

	gcid := r.Header.Get("X-Etcd-Cluster-ID")
	if gcid != wcid {
		plog.Errorf("request received was ignored (cluster ID mismatch got %s want %s)", gcid, wcid)
		http.Error(w, errClusterIDMismatch.Error(), http.StatusPreconditionFailed)
		return
	}

	// Limit the data size that could be read from the request body, which ensures that read from
	// connection will not time out accidentally due to possible block in underlying implementation.
	limitedr := pioutil.NewLimitedBufferReader(r.Body, ConnReadLimitByte)
	b, err := ioutil.ReadAll(limitedr)
	if err != nil {
		plog.Errorf("failed to read raft message (%v)", err)
		http.Error(w, "error reading raft message", http.StatusBadRequest)
		return
	}
	var m raftpb.Message
	if err := m.Unmarshal(b); err != nil {
		plog.Errorf("failed to unmarshal raft message (%v)", err)
		http.Error(w, "error unmarshaling raft message", http.StatusBadRequest)
		return
	}
	if err := h.r.Process(context.TODO(), m); err != nil {
		switch v := err.(type) {
		case writerToResponse:
			v.WriteTo(w)
		default:
			plog.Warningf("failed to process raft message (%v)", err)
			http.Error(w, "error processing raft message", http.StatusInternalServerError)
		}
		return
	}
	// Write StatusNoContet header after the message has been processed by
	// raft, which faciliates the client to report MsgSnap status.
	w.WriteHeader(http.StatusNoContent)
}

type streamHandler struct {
	peerGetter peerGetter
	r          Raft
	id         types.ID
	cid        types.ID
}

func (h *streamHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		w.Header().Set("Allow", "GET")
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("X-Server-Version", version.Version)

	if err := checkVersionCompability(r.Header.Get("X-Server-From"), serverVersion(r.Header), minClusterVersion(r.Header)); err != nil {
		plog.Errorf("request received was ignored (%v)", err)
		http.Error(w, errIncompatibleVersion.Error(), http.StatusPreconditionFailed)
		return
	}

	wcid := h.cid.String()
	w.Header().Set("X-Etcd-Cluster-ID", wcid)

	if gcid := r.Header.Get("X-Etcd-Cluster-ID"); gcid != wcid {
		plog.Errorf("streaming request ignored (cluster ID mismatch got %s want %s)", gcid, wcid)
		http.Error(w, errClusterIDMismatch.Error(), http.StatusPreconditionFailed)
		return
	}

	var t streamType
	switch path.Dir(r.URL.Path) {
	// backward compatibility
	case RaftStreamPrefix:
		t = streamTypeMsgApp
	case path.Join(RaftStreamPrefix, string(streamTypeMsgApp)):
		t = streamTypeMsgAppV2
	case path.Join(RaftStreamPrefix, string(streamTypeMessage)):
		t = streamTypeMessage
	default:
		plog.Debugf("ignored unexpected streaming request path %s", r.URL.Path)
		http.Error(w, "invalid path", http.StatusNotFound)
		return
	}

	fromStr := path.Base(r.URL.Path)
	from, err := types.IDFromString(fromStr)
	if err != nil {
		plog.Errorf("failed to parse from %s into ID (%v)", fromStr, err)
		http.Error(w, "invalid from", http.StatusNotFound)
		return
	}
	if h.r.IsIDRemoved(uint64(from)) {
		plog.Warningf("rejected the stream from peer %s since it was removed", from)
		http.Error(w, "removed member", http.StatusGone)
		return
	}
	p := h.peerGetter.Get(from)
	if p == nil {
		// This may happen in following cases:
		// 1. user starts a remote peer that belongs to a different cluster
		// with the same cluster ID.
		// 2. local etcd falls behind of the cluster, and cannot recognize
		// the members that joined after its current progress.
		plog.Errorf("failed to find member %s in cluster %s", from, wcid)
		http.Error(w, "error sender not found", http.StatusNotFound)
		return
	}

	wto := h.id.String()
	if gto := r.Header.Get("X-Raft-To"); gto != wto {
		plog.Errorf("streaming request ignored (ID mismatch got %s want %s)", gto, wto)
		http.Error(w, "to field mismatch", http.StatusPreconditionFailed)
		return
	}

	w.WriteHeader(http.StatusOK)
	w.(http.Flusher).Flush()

	c := newCloseNotifier()
	conn := &outgoingConn{
		t:       t,
		termStr: r.Header.Get("X-Raft-Term"),
		Writer:  w,
		Flusher: w.(http.Flusher),
		Closer:  c,
	}
	p.attachOutgoingConn(conn)
	<-c.closeNotify()
}

type closeNotifier struct {
	done chan struct{}
}

func newCloseNotifier() *closeNotifier {
	return &closeNotifier{
		done: make(chan struct{}),
	}
}

func (n *closeNotifier) Close() error {
	close(n.done)
	return nil
}

func (n *closeNotifier) closeNotify() <-chan struct{} { return n.done }
