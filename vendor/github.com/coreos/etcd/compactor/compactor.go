// Copyright 2016 The etcd Authors
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
	"sync"
	"time"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/mvcc"
	"github.com/coreos/pkg/capnslog"
	"github.com/jonboulle/clockwork"
	"golang.org/x/net/context"
)

var (
	plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "compactor")
)

const (
	checkCompactionInterval   = 5 * time.Minute
	executeCompactionInterval = time.Hour
)

type Compactable interface {
	Compact(ctx context.Context, r *pb.CompactionRequest) (*pb.CompactionResponse, error)
}

type RevGetter interface {
	Rev() int64
}

// Periodic compacts the log by purging revisions older than
// the configured retention time. Compaction happens hourly.
type Periodic struct {
	clock        clockwork.Clock
	periodInHour int

	rg RevGetter
	c  Compactable

	revs   []int64
	ctx    context.Context
	cancel context.CancelFunc

	mu     sync.Mutex
	paused bool
}

func NewPeriodic(h int, rg RevGetter, c Compactable) *Periodic {
	return &Periodic{
		clock:        clockwork.NewRealClock(),
		periodInHour: h,
		rg:           rg,
		c:            c,
	}
}

func (t *Periodic) Run() {
	t.ctx, t.cancel = context.WithCancel(context.Background())
	t.revs = make([]int64, 0)
	clock := t.clock

	go func() {
		last := clock.Now()
		for {
			t.revs = append(t.revs, t.rg.Rev())
			select {
			case <-t.ctx.Done():
				return
			case <-clock.After(checkCompactionInterval):
				t.mu.Lock()
				p := t.paused
				t.mu.Unlock()
				if p {
					continue
				}
			}

			if clock.Now().Sub(last) < executeCompactionInterval {
				continue
			}

			rev, remaining := t.getRev(t.periodInHour)
			if rev < 0 {
				continue
			}

			plog.Noticef("Starting auto-compaction at revision %d", rev)
			_, err := t.c.Compact(t.ctx, &pb.CompactionRequest{Revision: rev})
			if err == nil || err == mvcc.ErrCompacted {
				t.revs = remaining
				last = clock.Now()
				plog.Noticef("Finished auto-compaction at revision %d", rev)
			} else {
				plog.Noticef("Failed auto-compaction at revision %d (%v)", rev, err)
				plog.Noticef("Retry after %v", checkCompactionInterval)
			}
		}
	}()
}

func (t *Periodic) Stop() {
	t.cancel()
}

func (t *Periodic) Pause() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.paused = true
}

func (t *Periodic) Resume() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.paused = false
}

func (t *Periodic) getRev(h int) (int64, []int64) {
	i := len(t.revs) - int(time.Duration(h)*time.Hour/checkCompactionInterval)
	if i < 0 {
		return -1, t.revs
	}
	return t.revs[i], t.revs[i+1:]
}
