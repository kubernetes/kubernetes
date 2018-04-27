// Copyright 2017 The etcd Authors
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

package grpcproxy

import (
	"math"
	"sync"

	"golang.org/x/net/context"
	"golang.org/x/time/rate"
	"google.golang.org/grpc"

	"github.com/coreos/etcd/clientv3"
)

const (
	lostLeaderKey  = "__lostleader" // watched to detect leader loss
	retryPerSecond = 10
)

type leader struct {
	ctx context.Context
	w   clientv3.Watcher
	// mu protects leaderc updates.
	mu       sync.RWMutex
	leaderc  chan struct{}
	disconnc chan struct{}
	donec    chan struct{}
}

func newLeader(ctx context.Context, w clientv3.Watcher) *leader {
	l := &leader{
		ctx:      clientv3.WithRequireLeader(ctx),
		w:        w,
		leaderc:  make(chan struct{}),
		disconnc: make(chan struct{}),
		donec:    make(chan struct{}),
	}
	// begin assuming leader is lost
	close(l.leaderc)
	go l.recvLoop()
	return l
}

func (l *leader) recvLoop() {
	defer close(l.donec)

	limiter := rate.NewLimiter(rate.Limit(retryPerSecond), retryPerSecond)
	rev := int64(math.MaxInt64 - 2)
	for limiter.Wait(l.ctx) == nil {
		wch := l.w.Watch(l.ctx, lostLeaderKey, clientv3.WithRev(rev), clientv3.WithCreatedNotify())
		cresp, ok := <-wch
		if !ok {
			l.loseLeader()
			continue
		}
		if cresp.Err() != nil {
			l.loseLeader()
			if grpc.ErrorDesc(cresp.Err()) == grpc.ErrClientConnClosing.Error() {
				close(l.disconnc)
				return
			}
			continue
		}
		l.gotLeader()
		<-wch
		l.loseLeader()
	}
}

func (l *leader) loseLeader() {
	l.mu.RLock()
	defer l.mu.RUnlock()
	select {
	case <-l.leaderc:
	default:
		close(l.leaderc)
	}
}

// gotLeader will force update the leadership status to having a leader.
func (l *leader) gotLeader() {
	l.mu.Lock()
	defer l.mu.Unlock()
	select {
	case <-l.leaderc:
		l.leaderc = make(chan struct{})
	default:
	}
}

func (l *leader) disconnectNotify() <-chan struct{} { return l.disconnc }

func (l *leader) stopNotify() <-chan struct{} { return l.donec }

// lostNotify returns a channel that is closed if there has been
// a leader loss not yet followed by a leader reacquire.
func (l *leader) lostNotify() <-chan struct{} {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.leaderc
}
