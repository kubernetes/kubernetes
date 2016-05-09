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
	"bytes"
	"errors"
	"io/ioutil"
	"sync"
	"time"

	"github.com/coreos/etcd/etcdserver/stats"
	"github.com/coreos/etcd/pkg/httputil"
	"github.com/coreos/etcd/pkg/pbutil"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft"
	"github.com/coreos/etcd/raft/raftpb"
)

const (
	connPerPipeline = 4
	// pipelineBufSize is the size of pipeline buffer, which helps hold the
	// temporary network latency.
	// The size ensures that pipeline does not drop messages when the network
	// is out of work for less than 1 second in good path.
	pipelineBufSize = 64
)

var errStopped = errors.New("stopped")

type pipeline struct {
	from, to types.ID
	cid      types.ID

	tr     *Transport
	picker *urlPicker
	status *peerStatus
	fs     *stats.FollowerStats
	r      Raft
	errorc chan error

	msgc chan raftpb.Message
	// wait for the handling routines
	wg    sync.WaitGroup
	stopc chan struct{}
}

func newPipeline(tr *Transport, picker *urlPicker, from, to, cid types.ID, status *peerStatus, fs *stats.FollowerStats, r Raft, errorc chan error) *pipeline {
	p := &pipeline{
		from:   from,
		to:     to,
		cid:    cid,
		tr:     tr,
		picker: picker,
		status: status,
		fs:     fs,
		r:      r,
		errorc: errorc,
		stopc:  make(chan struct{}),
		msgc:   make(chan raftpb.Message, pipelineBufSize),
	}
	p.wg.Add(connPerPipeline)
	for i := 0; i < connPerPipeline; i++ {
		go p.handle()
	}
	return p
}

func (p *pipeline) stop() {
	close(p.stopc)
	p.wg.Wait()
}

func (p *pipeline) handle() {
	defer p.wg.Done()

	for {
		select {
		case m := <-p.msgc:
			start := time.Now()
			err := p.post(pbutil.MustMarshal(&m))
			end := time.Now()

			if err != nil {
				p.status.deactivate(failureType{source: pipelineMsg, action: "write"}, err.Error())

				reportSentFailure(pipelineMsg, m)
				if m.Type == raftpb.MsgApp && p.fs != nil {
					p.fs.Fail()
				}
				p.r.ReportUnreachable(m.To)
				if isMsgSnap(m) {
					p.r.ReportSnapshot(m.To, raft.SnapshotFailure)
				}
				continue
			}

			p.status.activate()
			if m.Type == raftpb.MsgApp && p.fs != nil {
				p.fs.Succ(end.Sub(start))
			}
			if isMsgSnap(m) {
				p.r.ReportSnapshot(m.To, raft.SnapshotFinish)
			}
			reportSentDuration(pipelineMsg, m, time.Since(start))
		case <-p.stopc:
			return
		}
	}
}

// post POSTs a data payload to a url. Returns nil if the POST succeeds,
// error on any failure.
func (p *pipeline) post(data []byte) (err error) {
	u := p.picker.pick()
	req := createPostRequest(u, RaftPrefix, bytes.NewBuffer(data), "application/protobuf", p.tr.URLs, p.from, p.cid)

	done := make(chan struct{}, 1)
	cancel := httputil.RequestCanceler(p.tr.pipelineRt, req)
	go func() {
		select {
		case <-done:
		case <-p.stopc:
			waitSchedule()
			cancel()
		}
	}()

	resp, err := p.tr.pipelineRt.RoundTrip(req)
	done <- struct{}{}
	if err != nil {
		p.picker.unreachable(u)
		return err
	}
	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		p.picker.unreachable(u)
		return err
	}
	resp.Body.Close()

	err = checkPostResponse(resp, b, req, p.to)
	if err != nil {
		p.picker.unreachable(u)
		// errMemberRemoved is a critical error since a removed member should
		// always be stopped. So we use reportCriticalError to report it to errorc.
		if err == errMemberRemoved {
			reportCriticalError(err, p.errorc)
		}
		return err
	}

	return nil
}

// waitSchedule waits other goroutines to be scheduled for a while
func waitSchedule() { time.Sleep(time.Millisecond) }
