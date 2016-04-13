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
	Client        *Client
	Ctx           context.Context  // The context to use when extending deadlines.
	Sub           string           // The full name of the subscription.
	ExtensionTick <-chan time.Time // ExtenstionTick supplies the frequency with which to make extension requests.
	Deadline      time.Duration    // How long to extend messages for each time they are extended. Should be greater than ExtensionTick frequency.
	MaxExtension  time.Duration    // How long to keep extending each message's ack deadline before automatically removing it.

	mu sync.Mutex
	// key: ackID; value: time at which ack deadline extension should cease.
	items map[string]time.Time

	wg   sync.WaitGroup
	done chan struct{}
}

// Start initiates the deadline extension loop.  Stop must be called once keepAlive is no longer needed.
func (ka *keepAlive) Start() {
	ka.items = make(map[string]time.Time)
	ka.done = make(chan struct{})
	ka.wg.Add(1)
	go func() {
		defer ka.wg.Done()
		done := false
		for {
			select {
			case <-ka.Ctx.Done():
				// Don't bother waiting for items to be removed: we can't extend them any more.
				return
			case <-ka.done:
				done = true
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
			live, _ := ka.getAckIDs()
			if done && len(live) == 0 {
				return
			}
		}
	}()
}

// Add adds an ack id to be kept alive.
func (ka *keepAlive) Add(ackID string) {
	ka.mu.Lock()
	defer ka.mu.Unlock()
	ka.items[ackID] = time.Now().Add(ka.MaxExtension)
}

// Remove removes ackID from the list to be kept alive.
func (ka *keepAlive) Remove(ackID string) {
	ka.mu.Lock()
	defer ka.mu.Unlock()
	delete(ka.items, ackID)
}

// Stop waits until all added ackIDs have been removed, and cleans up resources.
func (ka *keepAlive) Stop() {
	close(ka.done)
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

func (ka *keepAlive) extendDeadlines(ackIDs []string) {
	// TODO: split into separate requests if there are too many ackIDs.
	if len(ackIDs) > 0 {
		_ = ka.Client.s.modifyAckDeadline(ka.Ctx, ka.Sub, ka.Deadline, ackIDs)
	}
	// TODO: retry on error.  NOTE: if we ultimately fail to extend deadlines here, the messages will be redelivered, which is OK.
}
