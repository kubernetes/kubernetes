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

	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft/raftpb"
)

type remote struct {
	id       types.ID
	status   *peerStatus
	pipeline *pipeline
}

func startRemote(tr http.RoundTripper, urls types.URLs, local, to, cid types.ID, r Raft, errorc chan error) *remote {
	picker := newURLPicker(urls)
	status := newPeerStatus(to)
	return &remote{
		id:       to,
		status:   status,
		pipeline: newPipeline(tr, picker, local, to, cid, status, nil, r, errorc),
	}
}

func (g *remote) Send(m raftpb.Message) {
	select {
	case g.pipeline.msgc <- m:
	default:
		if g.status.isActive() {
			plog.Warningf("dropped %s to %s since sending buffer is full", m.Type, g.id)
		} else {
			plog.Debugf("dropped %s to %s since sending buffer is full", m.Type, g.id)
		}
	}
}

func (g *remote) Stop() {
	g.pipeline.stop()
}
