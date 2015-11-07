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
	"time"

	"github.com/coreos/etcd/etcdserver/stats"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft"
	"github.com/coreos/etcd/raft/raftpb"
	"golang.org/x/net/context"
)

const (
	// ConnRead/WriteTimeout is the i/o timeout set on each connection rafthttp pkg creates.
	// A 5 seconds timeout is good enough for recycling bad connections. Or we have to wait for
	// tcp keepalive failing to detect a bad connection, which is at minutes level.
	// For long term streaming connections, rafthttp pkg sends application level linkHeartbeat
	// to keep the connection alive.
	// For short term pipeline connections, the connection MUST be killed to avoid it being
	// put back to http pkg connection pool.
	ConnReadTimeout  = 5 * time.Second
	ConnWriteTimeout = 5 * time.Second

	recvBufSize = 4096
	// maxPendingProposals holds the proposals during one leader election process.
	// Generally one leader election takes at most 1 sec. It should have
	// 0-2 election conflicts, and each one takes 0.5 sec.
	// We assume the number of concurrent proposers is smaller than 4096.
	// One client blocks on its proposal for at least 1 sec, so 4096 is enough
	// to hold all proposals.
	maxPendingProposals = 4096

	streamApp   = "streamMsgApp"
	streamAppV2 = "streamMsgAppV2"
	streamMsg   = "streamMsg"
	pipelineMsg = "pipeline"
)

type Peer interface {
	// Send sends the message to the remote peer. The function is non-blocking
	// and has no promise that the message will be received by the remote.
	// When it fails to send message out, it will report the status to underlying
	// raft.
	Send(m raftpb.Message)
	// Update updates the urls of remote peer.
	Update(urls types.URLs)
	// setTerm sets the term of ongoing communication.
	setTerm(term uint64)
	// attachOutgoingConn attachs the outgoing connection to the peer for
	// stream usage. After the call, the ownership of the outgoing
	// connection hands over to the peer. The peer will close the connection
	// when it is no longer used.
	attachOutgoingConn(conn *outgoingConn)
	// activeSince returns the time that the connection with the
	// peer becomes active.
	activeSince() time.Time
	// Stop performs any necessary finalization and terminates the peer
	// elegantly.
	Stop()
}

// peer is the representative of a remote raft node. Local raft node sends
// messages to the remote through peer.
// Each peer has two underlying mechanisms to send out a message: stream and
// pipeline.
// A stream is a receiver initialized long-polling connection, which
// is always open to transfer messages. Besides general stream, peer also has
// a optimized stream for sending msgApp since msgApp accounts for large part
// of all messages. Only raft leader uses the optimized stream to send msgApp
// to the remote follower node.
// A pipeline is a series of http clients that send http requests to the remote.
// It is only used when the stream has not been established.
type peer struct {
	// id of the remote raft peer node
	id types.ID
	r  Raft

	status *peerStatus

	msgAppWriter *streamWriter
	writer       *streamWriter
	pipeline     *pipeline
	msgAppReader *streamReader

	sendc    chan raftpb.Message
	recvc    chan raftpb.Message
	propc    chan raftpb.Message
	newURLsC chan types.URLs
	termc    chan uint64

	// for testing
	pausec  chan struct{}
	resumec chan struct{}

	stopc chan struct{}
	done  chan struct{}
}

func startPeer(tr http.RoundTripper, urls types.URLs, local, to, cid types.ID, r Raft, fs *stats.FollowerStats, errorc chan error, term uint64) *peer {
	picker := newURLPicker(urls)
	status := newPeerStatus(to)
	p := &peer{
		id:           to,
		r:            r,
		status:       status,
		msgAppWriter: startStreamWriter(to, status, fs, r),
		writer:       startStreamWriter(to, status, fs, r),
		pipeline:     newPipeline(tr, picker, local, to, cid, status, fs, r, errorc),
		sendc:        make(chan raftpb.Message),
		recvc:        make(chan raftpb.Message, recvBufSize),
		propc:        make(chan raftpb.Message, maxPendingProposals),
		newURLsC:     make(chan types.URLs),
		termc:        make(chan uint64),
		pausec:       make(chan struct{}),
		resumec:      make(chan struct{}),
		stopc:        make(chan struct{}),
		done:         make(chan struct{}),
	}

	// Use go-routine for process of MsgProp because it is
	// blocking when there is no leader.
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		for {
			select {
			case mm := <-p.propc:
				if err := r.Process(ctx, mm); err != nil {
					plog.Warningf("failed to process raft message (%v)", err)
				}
			case <-p.stopc:
				return
			}
		}
	}()

	p.msgAppReader = startStreamReader(tr, picker, streamTypeMsgAppV2, local, to, cid, status, p.recvc, p.propc, errorc, term)
	reader := startStreamReader(tr, picker, streamTypeMessage, local, to, cid, status, p.recvc, p.propc, errorc, term)
	go func() {
		var paused bool
		for {
			select {
			case m := <-p.sendc:
				if paused {
					continue
				}
				writec, name := p.pick(m)
				select {
				case writec <- m:
				default:
					p.r.ReportUnreachable(m.To)
					if isMsgSnap(m) {
						p.r.ReportSnapshot(m.To, raft.SnapshotFailure)
					}
					if status.isActive() {
						plog.Warningf("dropped %s to %s since %s's sending buffer is full", m.Type, p.id, name)
					} else {
						plog.Debugf("dropped %s to %s since %s's sending buffer is full", m.Type, p.id, name)
					}
				}
			case mm := <-p.recvc:
				if err := r.Process(context.TODO(), mm); err != nil {
					plog.Warningf("failed to process raft message (%v)", err)
				}
			case urls := <-p.newURLsC:
				picker.update(urls)
			case <-p.pausec:
				paused = true
			case <-p.resumec:
				paused = false
			case <-p.stopc:
				cancel()
				p.msgAppWriter.stop()
				p.writer.stop()
				p.pipeline.stop()
				p.msgAppReader.stop()
				reader.stop()
				close(p.done)
				return
			}
		}
	}()

	return p
}

func (p *peer) Send(m raftpb.Message) {
	select {
	case p.sendc <- m:
	case <-p.done:
	}
}

func (p *peer) Update(urls types.URLs) {
	select {
	case p.newURLsC <- urls:
	case <-p.done:
	}
}

func (p *peer) setTerm(term uint64) { p.msgAppReader.updateMsgAppTerm(term) }

func (p *peer) attachOutgoingConn(conn *outgoingConn) {
	var ok bool
	switch conn.t {
	case streamTypeMsgApp, streamTypeMsgAppV2:
		ok = p.msgAppWriter.attach(conn)
	case streamTypeMessage:
		ok = p.writer.attach(conn)
	default:
		plog.Panicf("unhandled stream type %s", conn.t)
	}
	if !ok {
		conn.Close()
	}
}

func (p *peer) activeSince() time.Time { return p.status.activeSince }

// Pause pauses the peer. The peer will simply drops all incoming
// messages without retruning an error.
func (p *peer) Pause() {
	select {
	case p.pausec <- struct{}{}:
	case <-p.done:
	}
}

// Resume resumes a paused peer.
func (p *peer) Resume() {
	select {
	case p.resumec <- struct{}{}:
	case <-p.done:
	}
}

func (p *peer) Stop() {
	close(p.stopc)
	<-p.done
}

// pick picks a chan for sending the given message. The picked chan and the picked chan
// string name are returned.
func (p *peer) pick(m raftpb.Message) (writec chan<- raftpb.Message, picked string) {
	var ok bool
	// Considering MsgSnap may have a big size, e.g., 1G, and will block
	// stream for a long time, only use one of the N pipelines to send MsgSnap.
	if isMsgSnap(m) {
		return p.pipeline.msgc, pipelineMsg
	} else if writec, ok = p.msgAppWriter.writec(); ok && canUseMsgAppStream(m) {
		return writec, streamApp
	} else if writec, ok = p.writer.writec(); ok {
		return writec, streamMsg
	}
	return p.pipeline.msgc, pipelineMsg
}

func isMsgSnap(m raftpb.Message) bool { return m.Type == raftpb.MsgSnap }
