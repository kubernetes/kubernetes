/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package backoff

import (
	"math/rand"
	"sync"
	"time"

	log "github.com/golang/glog"
)

type clock interface {
	Now() time.Time
}

type realClock struct{}

func (realClock) Now() time.Time {
	return time.Now()
}

type backoffEntry struct {
	backoff    time.Duration
	lastUpdate time.Time
}

type Backoff struct {
	perItemBackoff  map[string]*backoffEntry
	lock            sync.Mutex
	clock           clock
	defaultDuration time.Duration
	maxDuration     time.Duration
}

func New(initial, max time.Duration) *Backoff {
	return &Backoff{
		perItemBackoff:  map[string]*backoffEntry{},
		clock:           realClock{},
		defaultDuration: initial,
		maxDuration:     max,
	}
}

func (p *Backoff) getEntry(id string) *backoffEntry {
	p.lock.Lock()
	defer p.lock.Unlock()
	entry, ok := p.perItemBackoff[id]
	if !ok {
		entry = &backoffEntry{backoff: p.defaultDuration}
		p.perItemBackoff[id] = entry
	}
	entry.lastUpdate = p.clock.Now()
	return entry
}

func (p *Backoff) Get(id string) time.Duration {
	entry := p.getEntry(id)
	duration := entry.backoff
	entry.backoff *= 2
	if entry.backoff > p.maxDuration {
		entry.backoff = p.maxDuration
	}
	//TODO(jdef) parameterize use of jitter?
	// add jitter, get better backoff distribution
	duration = time.Duration(rand.Int63n(int64(duration)))
	log.V(3).Infof("Backing off %v for pod %s", duration, id)
	return duration
}

// Garbage collect records that have aged past maxDuration. Backoff users are expected
// to invoke this periodically.
func (p *Backoff) GC() {
	p.lock.Lock()
	defer p.lock.Unlock()
	now := p.clock.Now()
	for id, entry := range p.perItemBackoff {
		if now.Sub(entry.lastUpdate) > p.maxDuration {
			delete(p.perItemBackoff, id)
		}
	}
}
