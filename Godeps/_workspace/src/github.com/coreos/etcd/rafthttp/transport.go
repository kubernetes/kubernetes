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
	"net/http"
	"sync"
	"time"

	"github.com/coreos/etcd/etcdserver/stats"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft"
	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/pkg/capnslog"
	"github.com/xiang90/probing"
	"golang.org/x/net/context"
)

var plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "rafthttp")

type Raft interface {
	Process(ctx context.Context, m raftpb.Message) error
	IsIDRemoved(id uint64) bool
	ReportUnreachable(id uint64)
	ReportSnapshot(id uint64, status raft.SnapshotStatus)
}

type Transporter interface {
	// Handler returns the HTTP handler of the transporter.
	// A transporter HTTP handler handles the HTTP requests
	// from remote peers.
	// The handler MUST be used to handle RaftPrefix(/raft)
	// endpoint.
	Handler() http.Handler
	// Send sends out the given messages to the remote peers.
	// Each message has a To field, which is an id that maps
	// to an existing peer in the transport.
	// If the id cannot be found in the transport, the message
	// will be ignored.
	Send(m []raftpb.Message)
	// AddRemote adds a remote with given peer urls into the transport.
	// A remote helps newly joined member to catch up the progress of cluster,
	// and will not be used after that.
	// It is the caller's responsibility to ensure the urls are all valid,
	// or it panics.
	AddRemote(id types.ID, urls []string)
	// AddPeer adds a peer with given peer urls into the transport.
	// It is the caller's responsibility to ensure the urls are all valid,
	// or it panics.
	// Peer urls are used to connect to the remote peer.
	AddPeer(id types.ID, urls []string)
	// RemovePeer removes the peer with given id.
	RemovePeer(id types.ID)
	// RemoveAllPeers removes all the existing peers in the transport.
	RemoveAllPeers()
	// UpdatePeer updates the peer urls of the peer with the given id.
	// It is the caller's responsibility to ensure the urls are all valid,
	// or it panics.
	UpdatePeer(id types.ID, urls []string)
	// ActiveSince returns the time that the connection with the peer
	// of the given id becomes active.
	// If the connection is active since peer was added, it returns the adding time.
	// If the connection is currently inactive, it returns zero time.
	ActiveSince(id types.ID) time.Time
	// Stop closes the connections and stops the transporter.
	Stop()
}

type transport struct {
	roundTripper http.RoundTripper
	id           types.ID
	clusterID    types.ID
	raft         Raft
	serverStats  *stats.ServerStats
	leaderStats  *stats.LeaderStats

	mu      sync.RWMutex         // protect the term, remote and peer map
	term    uint64               // the latest term that has been observed
	remotes map[types.ID]*remote // remotes map that helps newly joined member to catch up
	peers   map[types.ID]Peer    // peers map

	prober probing.Prober
	errorc chan error
}

func NewTransporter(rt http.RoundTripper, id, cid types.ID, r Raft, errorc chan error, ss *stats.ServerStats, ls *stats.LeaderStats) Transporter {
	return &transport{
		roundTripper: rt,
		id:           id,
		clusterID:    cid,
		raft:         r,
		serverStats:  ss,
		leaderStats:  ls,
		remotes:      make(map[types.ID]*remote),
		peers:        make(map[types.ID]Peer),

		prober: probing.NewProber(rt),
		errorc: errorc,
	}
}

func (t *transport) Handler() http.Handler {
	pipelineHandler := NewHandler(t.raft, t.clusterID)
	streamHandler := newStreamHandler(t, t.raft, t.id, t.clusterID)
	mux := http.NewServeMux()
	mux.Handle(RaftPrefix, pipelineHandler)
	mux.Handle(RaftStreamPrefix+"/", streamHandler)
	mux.Handle(ProbingPrefix, probing.NewHandler())
	return mux
}

func (t *transport) Get(id types.ID) Peer {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.peers[id]
}

func (t *transport) maybeUpdatePeersTerm(term uint64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.term >= term {
		return
	}
	t.term = term
	for _, p := range t.peers {
		p.setTerm(term)
	}
}

func (t *transport) Send(msgs []raftpb.Message) {
	for _, m := range msgs {
		// intentionally dropped message
		if m.To == 0 {
			continue
		}
		to := types.ID(m.To)

		if m.Type != raftpb.MsgProp { // proposal message does not have a valid term
			t.maybeUpdatePeersTerm(m.Term)
		}

		p, ok := t.peers[to]
		if ok {
			if m.Type == raftpb.MsgApp {
				t.serverStats.SendAppendReq(m.Size())
			}
			p.Send(m)
			continue
		}

		g, ok := t.remotes[to]
		if ok {
			g.Send(m)
			continue
		}

		plog.Debugf("ignored message %s (sent to unknown peer %s)", m.Type, to)
	}
}

func (t *transport) Stop() {
	for _, r := range t.remotes {
		r.Stop()
	}
	for _, p := range t.peers {
		p.Stop()
	}
	t.prober.RemoveAll()
	if tr, ok := t.roundTripper.(*http.Transport); ok {
		tr.CloseIdleConnections()
	}
}

func (t *transport) AddRemote(id types.ID, us []string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if _, ok := t.remotes[id]; ok {
		return
	}
	urls, err := types.NewURLs(us)
	if err != nil {
		plog.Panicf("newURLs %+v should never fail: %+v", us, err)
	}
	t.remotes[id] = startRemote(t.roundTripper, urls, t.id, id, t.clusterID, t.raft, t.errorc)
}

func (t *transport) AddPeer(id types.ID, us []string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if _, ok := t.peers[id]; ok {
		return
	}
	urls, err := types.NewURLs(us)
	if err != nil {
		plog.Panicf("newURLs %+v should never fail: %+v", us, err)
	}
	fs := t.leaderStats.Follower(id.String())
	t.peers[id] = startPeer(t.roundTripper, urls, t.id, id, t.clusterID, t.raft, fs, t.errorc, t.term)
	addPeerToProber(t.prober, id.String(), us)
}

func (t *transport) RemovePeer(id types.ID) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.removePeer(id)
}

func (t *transport) RemoveAllPeers() {
	t.mu.Lock()
	defer t.mu.Unlock()
	for id := range t.peers {
		t.removePeer(id)
	}
}

// the caller of this function must have the peers mutex.
func (t *transport) removePeer(id types.ID) {
	if peer, ok := t.peers[id]; ok {
		peer.Stop()
	} else {
		plog.Panicf("unexpected removal of unknown peer '%d'", id)
	}
	delete(t.peers, id)
	delete(t.leaderStats.Followers, id.String())
	t.prober.Remove(id.String())
}

func (t *transport) UpdatePeer(id types.ID, us []string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	// TODO: return error or just panic?
	if _, ok := t.peers[id]; !ok {
		return
	}
	urls, err := types.NewURLs(us)
	if err != nil {
		plog.Panicf("newURLs %+v should never fail: %+v", us, err)
	}
	t.peers[id].Update(urls)

	t.prober.Remove(id.String())
	addPeerToProber(t.prober, id.String(), us)
}

func (t *transport) ActiveSince(id types.ID) time.Time {
	t.mu.Lock()
	defer t.mu.Unlock()
	if p, ok := t.peers[id]; ok {
		return p.activeSince()
	}
	return time.Time{}
}

type Pausable interface {
	Pause()
	Resume()
}

// for testing
func (t *transport) Pause() {
	for _, p := range t.peers {
		p.(Pausable).Pause()
	}
}

func (t *transport) Resume() {
	for _, p := range t.peers {
		p.(Pausable).Resume()
	}
}
