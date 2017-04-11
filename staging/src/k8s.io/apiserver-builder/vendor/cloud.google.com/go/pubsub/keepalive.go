// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pubsub

import (
	"sync"
	"time"

	"golang.org/x/net/context"
)

// keepAlive keeps track of which Messages need to have their deadline extended, and
// periodically extends them.
// Messages are tracked by Ack ID.
type keepAlive struct {
	s             service
	Ctx           context.Context  // The context to use when extending deadlines.
	Sub           string           // The full name of the subscription.
	ExtensionTick <-chan time.Time // ExtenstionTick supplies the frequency with which to make extension requests.
	Deadline      time.Duration    // How long to extend messages for each time they are extended. Should be greater than ExtensionTick frequency.
	MaxExtension  time.Duration    // How long to keep extending each message's ack deadline before automatically removing it.

	mu sync.Mutex
	// key: ackID; value: time at which ack deadline extension should cease.
	items map[string]time.Time
	dr    drain

	wg sync.WaitGroup
}

// Start initiates the deadline extension loop.  Stop must be called once keepAlive is no longer needed.
func (ka *keepAlive) Start() {
	ka.items = make(map[string]time.Time)
	ka.dr = drain{Drained: make(chan struct{})}
	ka.wg.Add(1)
	go func() {
		defer ka.wg.Done()
		for {
			select {
			case <-ka.Ctx.Done():
				// Don't bother waiting for items to be removed: we can't extend them any more.
				return
			case <-ka.dr.Drained:
				return
			case <-ka.ExtensionTick:
				live, expired := ka.getAckIDs()
				ka.wg.Add(1)
				go func() {
					defer ka.wg.Done()
					ka.extendDeadlines(live)
				}()

				for _, id := range expired {
					ka.Remove(id)
				}
			}
		}
	}()
}

// Add adds an ack id to be kept alive.
// It should not be called after Stop.
func (ka *keepAlive) Add(ackID string) {
	ka.mu.Lock()
	defer ka.mu.Unlock()

	ka.items[ackID] = time.Now().Add(ka.MaxExtension)
	ka.dr.SetPending(true)
}

// Remove removes ackID from the list to be kept alive.
func (ka *keepAlive) Remove(ackID string) {
	ka.mu.Lock()
	defer ka.mu.Unlock()

	// Note: If users NACKs a message after it has been removed due to
	// expiring, Remove will be called twice with same ack id.  This is OK.
	delete(ka.items, ackID)
	ka.dr.SetPending(len(ka.items) != 0)
}

// Stop waits until all added ackIDs have been removed, and cleans up resources.
// Stop may only be called once.
func (ka *keepAlive) Stop() {
	ka.mu.Lock()
	ka.dr.Drain()
	ka.mu.Unlock()

	ka.wg.Wait()
}

// getAckIDs returns the set of ackIDs that are being kept alive.
// The set is divided into two lists: one with IDs that should continue to be kept alive,
// and the other with IDs that should be dropped.
func (ka *keepAlive) getAckIDs() (live, expired []string) {
	ka.mu.Lock()
	defer ka.mu.Unlock()

	now := time.Now()
	for id, expiry := range ka.items {
		if expiry.Before(now) {
			expired = append(expired, id)
		} else {
			live = append(live, id)
		}
	}
	return live, expired
}

const maxExtensionAttempts = 2

func (ka *keepAlive) extendDeadlines(ackIDs []string) {
	head, tail := ka.s.splitAckIDs(ackIDs)
	for len(head) > 0 {
		for i := 0; i < maxExtensionAttempts; i++ {
			if ka.s.modifyAckDeadline(ka.Ctx, ka.Sub, ka.Deadline, head) == nil {
				break
			}
		}
		// NOTE: Messages whose deadlines we fail to extend will
		// eventually be redelivered and this is a documented behaviour
		// of the API.
		//
		// NOTE: If we fail to extend deadlines here, this
		// implementation will continue to attempt extending the
		// deadlines for those ack IDs the next time the extension
		// ticker ticks.  By then the deadline will have expired.
		// Re-extending them is harmless, however.
		//
		// TODO: call Remove for ids which fail to be extended.

		head, tail = ka.s.splitAckIDs(tail)
	}
}

// A drain (once started) indicates via a channel when there is no work pending.
type drain struct {
	started bool
	pending bool

	// Drained is closed once there are no items outstanding if Drain has been called.
	Drained chan struct{}
}

// Drain starts the drain process. This cannot be undone.
func (d *drain) Drain() {
	d.started = true
	d.closeIfDrained()
}

// SetPending sets whether there is work pending or not. It may be called multiple times before or after Drain.
func (d *drain) SetPending(pending bool) {
	d.pending = pending
	d.closeIfDrained()
}

func (d *drain) closeIfDrained() {
	if !d.pending && d.started {
		// Check to see if d.Drained is closed before closing it.
		// This allows SetPending(false) to be safely called multiple times.
		select {
		case <-d.Drained:
		default:
			close(d.Drained)
		}
	}
}
