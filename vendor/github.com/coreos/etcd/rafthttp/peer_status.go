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
	"sync"
	"time"

	"github.com/coreos/etcd/pkg/types"
)

type failureType struct {
	source string
	action string
}

type peerStatus struct {
	id          types.ID
	mu          sync.Mutex // protect variables below
	active      bool
	activeSince time.Time
}

func newPeerStatus(id types.ID) *peerStatus {
	return &peerStatus{
		id: id,
	}
}

func (s *peerStatus) activate() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.active {
		plog.Infof("the connection with %s became active", s.id)
		s.active = true
		s.activeSince = time.Now()
	}
}

func (s *peerStatus) deactivate(failure failureType, reason string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	msg := fmt.Sprintf("failed to %s %s on %s (%s)", failure.action, s.id, failure.source, reason)
	if s.active {
		plog.Errorf(msg)
		plog.Infof("the connection with %s became inactive", s.id)
		s.active = false
		s.activeSince = time.Time{}
		return
	}
	plog.Debugf(msg)
}

func (s *peerStatus) isActive() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.active
}
