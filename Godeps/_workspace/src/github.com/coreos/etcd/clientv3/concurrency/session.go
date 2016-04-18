// Copyright 2016 CoreOS, Inc.
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

package concurrency

import (
	"sync"

	v3 "github.com/coreos/etcd/clientv3"
	"golang.org/x/net/context"
)

// only keep one ephemeral lease per client
var clientSessions clientSessionMgr = clientSessionMgr{sessions: make(map[*v3.Client]*Session)}

const sessionTTL = 60

type clientSessionMgr struct {
	sessions map[*v3.Client]*Session
	mu       sync.Mutex
}

// Session represents a lease kept alive for the lifetime of a client.
// Fault-tolerant applications may use sessions to reason about liveness.
type Session struct {
	client *v3.Client
	id     v3.LeaseID

	cancel context.CancelFunc
	donec  <-chan struct{}
}

// NewSession gets the leased session for a client.
func NewSession(client *v3.Client) (*Session, error) {
	clientSessions.mu.Lock()
	defer clientSessions.mu.Unlock()
	if s, ok := clientSessions.sessions[client]; ok {
		return s, nil
	}

	resp, err := client.Grant(client.Ctx(), sessionTTL)
	if err != nil {
		return nil, err
	}
	id := v3.LeaseID(resp.ID)

	ctx, cancel := context.WithCancel(client.Ctx())
	keepAlive, err := client.KeepAlive(ctx, id)
	if err != nil || keepAlive == nil {
		return nil, err
	}

	donec := make(chan struct{})
	s := &Session{client: client, id: id, cancel: cancel, donec: donec}
	clientSessions.sessions[client] = s

	// keep the lease alive until client error or cancelled context
	go func() {
		defer func() {
			clientSessions.mu.Lock()
			delete(clientSessions.sessions, client)
			clientSessions.mu.Unlock()
			close(donec)
		}()
		for range keepAlive {
			// eat messages until keep alive channel closes
		}
	}()

	return s, nil
}

// Lease is the lease ID for keys bound to the session.
func (s *Session) Lease() v3.LeaseID { return s.id }

// Done returns a channel that closes when the lease is orphaned, expires, or
// is otherwise no longer being refreshed.
func (s *Session) Done() <-chan struct{} { return s.donec }

// Orphan ends the refresh for the session lease. This is useful
// in case the state of the client connection is indeterminate (revoke
// would fail) or when transferring lease ownership.
func (s *Session) Orphan() {
	s.cancel()
	<-s.donec
}

// Close orphans the session and revokes the session lease.
func (s *Session) Close() error {
	s.Orphan()
	_, err := s.client.Revoke(s.client.Ctx(), s.id)
	return err
}
