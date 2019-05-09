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

package compactor

import (
	"context"
	"sync"
	"time"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/mvcc"

	"github.com/jonboulle/clockwork"
)

// Revision compacts the log by purging revisions older than
// the configured reivison number. Compaction happens every 5 minutes.
type Revision struct {
	clock     clockwork.Clock
	retention int64

	rg RevGetter
	c  Compactable

	ctx    context.Context
	cancel context.CancelFunc

	mu     sync.Mutex
	paused bool
}

// NewRevision creates a new instance of Revisonal compactor that purges
// the log older than retention revisions from the current revision.
func NewRevision(retention int64, rg RevGetter, c Compactable) *Revision {
	return newRevision(clockwork.NewRealClock(), retention, rg, c)
}

func newRevision(clock clockwork.Clock, retention int64, rg RevGetter, c Compactable) *Revision {
	t := &Revision{
		clock:     clock,
		retention: retention,
		rg:        rg,
		c:         c,
	}
	t.ctx, t.cancel = context.WithCancel(context.Background())
	return t
}

const revInterval = 5 * time.Minute

// Run runs revision-based compactor.
func (t *Revision) Run() {
	prev := int64(0)
	go func() {
		for {
			select {
			case <-t.ctx.Done():
				return
			case <-t.clock.After(revInterval):
				t.mu.Lock()
				p := t.paused
				t.mu.Unlock()
				if p {
					continue
				}
			}

			rev := t.rg.Rev() - t.retention
			if rev <= 0 || rev == prev {
				continue
			}

			plog.Noticef("Starting auto-compaction at revision %d (retention: %d revisions)", rev, t.retention)
			_, err := t.c.Compact(t.ctx, &pb.CompactionRequest{Revision: rev})
			if err == nil || err == mvcc.ErrCompacted {
				prev = rev
				plog.Noticef("Finished auto-compaction at revision %d", rev)
			} else {
				plog.Noticef("Failed auto-compaction at revision %d (%v)", rev, err)
				plog.Noticef("Retry after %v", revInterval)
			}
		}
	}()
}

// Stop stops revision-based compactor.
func (t *Revision) Stop() {
	t.cancel()
}

// Pause pauses revision-based compactor.
func (t *Revision) Pause() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.paused = true
}

// Resume resumes revision-based compactor.
func (t *Revision) Resume() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.paused = false
}
