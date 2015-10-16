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
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/coreos/etcd/etcdserver/stats"
	"github.com/coreos/etcd/pkg/httputil"
	"github.com/coreos/etcd/pkg/pbutil"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft"
	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/version"
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

	tr     http.RoundTripper
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

func newPipeline(tr http.RoundTripper, picker *urlPicker, from, to, cid types.ID, status *peerStatus, fs *stats.FollowerStats, r Raft, errorc chan error) *pipeline {
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
	close(p.msgc)
	close(p.stopc)
	p.wg.Wait()
}

func (p *pipeline) handle() {
	defer p.wg.Done()
	for m := range p.msgc {
		start := time.Now()
		err := p.post(pbutil.MustMarshal(&m))
		if err == errStopped {
			return
		}
		end := time.Now()

		if err != nil {
			reportSentFailure(pipelineMsg, m)
			p.status.deactivate(failureType{source: pipelineMsg, action: "write"}, err.Error())
			if m.Type == raftpb.MsgApp && p.fs != nil {
				p.fs.Fail()
			}
			p.r.ReportUnreachable(m.To)
			if isMsgSnap(m) {
				p.r.ReportSnapshot(m.To, raft.SnapshotFailure)
			}
		} else {
			p.status.activate()
			if m.Type == raftpb.MsgApp && p.fs != nil {
				p.fs.Succ(end.Sub(start))
			}
			if isMsgSnap(m) {
				p.r.ReportSnapshot(m.To, raft.SnapshotFinish)
			}
			reportSentDuration(pipelineMsg, m, time.Since(start))
		}
	}
}

// post POSTs a data payload to a url. Returns nil if the POST succeeds,
// error on any failure.
func (p *pipeline) post(data []byte) (err error) {
	u := p.picker.pick()
	uu := u
	uu.Path = RaftPrefix
	req, err := http.NewRequest("POST", uu.String(), bytes.NewBuffer(data))
	if err != nil {
		p.picker.unreachable(u)
		return err
	}
	req.Header.Set("Content-Type", "application/protobuf")
	req.Header.Set("X-Server-From", p.from.String())
	req.Header.Set("X-Server-Version", version.Version)
	req.Header.Set("X-Min-Cluster-Version", version.MinClusterVersion)
	req.Header.Set("X-Etcd-Cluster-ID", p.cid.String())

	var stopped bool
	defer func() {
		if stopped {
			// rewrite to errStopped so the caller goroutine can stop itself
			err = errStopped
		}
	}()
	done := make(chan struct{}, 1)
	cancel := httputil.RequestCanceler(p.tr, req)
	go func() {
		select {
		case <-done:
		case <-p.stopc:
			waitSchedule()
			stopped = true
			cancel()
		}
	}()

	resp, err := p.tr.RoundTrip(req)
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

	switch resp.StatusCode {
	case http.StatusPreconditionFailed:
		switch strings.TrimSuffix(string(b), "\n") {
		case errIncompatibleVersion.Error():
			plog.Errorf("request sent was ignored by peer %s (server version incompatible)", p.to)
			return errIncompatibleVersion
		case errClusterIDMismatch.Error():
			plog.Errorf("request sent was ignored (cluster ID mismatch: remote[%s]=%s, local=%s)",
				p.to, resp.Header.Get("X-Etcd-Cluster-ID"), p.cid)
			return errClusterIDMismatch
		default:
			return fmt.Errorf("unhandled error %q when precondition failed", string(b))
		}
	case http.StatusForbidden:
		err := fmt.Errorf("the member has been permanently removed from the cluster")
		select {
		case p.errorc <- err:
		default:
		}
		return nil
	case http.StatusNoContent:
		return nil
	default:
		return fmt.Errorf("unexpected http status %s while posting to %q", http.StatusText(resp.StatusCode), req.URL.String())
	}
}

// waitSchedule waits other goroutines to be scheduled for a while
func waitSchedule() { time.Sleep(time.Millisecond) }
