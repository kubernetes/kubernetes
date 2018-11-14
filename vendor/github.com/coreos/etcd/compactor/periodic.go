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

// Periodic compacts the log by purging revisions older than
// the configured retention time.
type Periodic struct {
	clock  clockwork.Clock
	period time.Duration

	rg RevGetter
	c  Compactable

	revs   []int64
	ctx    context.Context
	cancel context.CancelFunc

	// mu protects paused
	mu     sync.RWMutex
	paused bool
}

// NewPeriodic creates a new instance of Periodic compactor that purges
// the log older than h Duration.
func NewPeriodic(h time.Duration, rg RevGetter, c Compactable) *Periodic {
	return newPeriodic(clockwork.NewRealClock(), h, rg, c)
}

func newPeriodic(clock clockwork.Clock, h time.Duration, rg RevGetter, c Compactable) *Periodic {
	t := &Periodic{
		clock:  clock,
		period: h,
		rg:     rg,
		c:      c,
		revs:   make([]int64, 0),
	}
	t.ctx, t.cancel = context.WithCancel(context.Background())
	return t
}

/*
Compaction period 1-hour:
  1. compute compaction period, which is 1-hour
  2. record revisions for every 1/10 of 1-hour (6-minute)
  3. keep recording revisions with no compaction for first 1-hour
  4. do compact with revs[0]
	- success? contiue on for-loop and move sliding window; revs = revs[1:]
	- failure? update revs, and retry after 1/10 of 1-hour (6-minute)

Compaction period 24-hour:
  1. compute compaction period, which is 1-hour
  2. record revisions for every 1/10 of 1-hour (6-minute)
  3. keep recording revisions with no compaction for first 24-hour
  4. do compact with revs[0]
	- success? contiue on for-loop and move sliding window; revs = revs[1:]
	- failure? update revs, and retry after 1/10 of 1-hour (6-minute)

Compaction period 59-min:
  1. compute compaction period, which is 59-min
  2. record revisions for every 1/10 of 59-min (5.9-min)
  3. keep recording revisions with no compaction for first 59-min
  4. do compact with revs[0]
	- success? contiue on for-loop and move sliding window; revs = revs[1:]
	- failure? update revs, and retry after 1/10 of 59-min (5.9-min)

Compaction period 5-sec:
  1. compute compaction period, which is 5-sec
  2. record revisions for every 1/10 of 5-sec (0.5-sec)
  3. keep recording revisions with no compaction for first 5-sec
  4. do compact with revs[0]
	- success? contiue on for-loop and move sliding window; revs = revs[1:]
	- failure? update revs, and retry after 1/10 of 5-sec (0.5-sec)
*/

// Run runs periodic compactor.
func (t *Periodic) Run() {
	compactInterval := t.getCompactInterval()
	retryInterval := t.getRetryInterval()
	retentions := t.getRetentions()

	go func() {
		lastSuccess := t.clock.Now()
		baseInterval := t.period
		for {
			t.revs = append(t.revs, t.rg.Rev())
			if len(t.revs) > retentions {
				t.revs = t.revs[1:] // t.revs[0] is always the rev at t.period ago
			}

			select {
			case <-t.ctx.Done():
				return
			case <-t.clock.After(retryInterval):
				t.mu.Lock()
				p := t.paused
				t.mu.Unlock()
				if p {
					continue
				}
			}

			if t.clock.Now().Sub(lastSuccess) < baseInterval {
				continue
			}

			// wait up to initial given period
			if baseInterval == t.period {
				baseInterval = compactInterval
			}
			rev := t.revs[0]

			plog.Noticef("Starting auto-compaction at revision %d (retention: %v)", rev, t.period)
			_, err := t.c.Compact(t.ctx, &pb.CompactionRequest{Revision: rev})
			if err == nil || err == mvcc.ErrCompacted {
				lastSuccess = t.clock.Now()
				plog.Noticef("Finished auto-compaction at revision %d", rev)
			} else {
				plog.Noticef("Failed auto-compaction at revision %d (%v)", rev, err)
				plog.Noticef("Retry after %v", retryInterval)
			}
		}
	}()
}

// if given compaction period x is <1-hour, compact every x duration.
// (e.g. --auto-compaction-mode 'periodic' --auto-compaction-retention='10m', then compact every 10-minute)
// if given compaction period x is >1-hour, compact every hour.
// (e.g. --auto-compaction-mode 'periodic' --auto-compaction-retention='2h', then compact every 1-hour)
func (t *Periodic) getCompactInterval() time.Duration {
	itv := t.period
	if itv > time.Hour {
		itv = time.Hour
	}
	return itv
}

func (t *Periodic) getRetentions() int {
	return int(t.period/t.getRetryInterval()) + 1
}

const retryDivisor = 10

func (t *Periodic) getRetryInterval() time.Duration {
	itv := t.period
	if itv > time.Hour {
		itv = time.Hour
	}
	return itv / retryDivisor
}

// Stop stops periodic compactor.
func (t *Periodic) Stop() {
	t.cancel()
}

// Pause pauses periodic compactor.
func (t *Periodic) Pause() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.paused = true
}

// Resume resumes periodic compactor.
func (t *Periodic) Resume() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.paused = false
}
