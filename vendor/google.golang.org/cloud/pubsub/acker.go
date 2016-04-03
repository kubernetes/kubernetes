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

// acker acks messages in batches.
type acker struct {
	Client  *Client
	Ctx     context.Context  // The context to use when acknowledging messages.
	Sub     string           // The full name of the subscription.
	AckTick <-chan time.Time // AckTick supplies the frequency with which to make ack requests.

	// Notify is called with an ack ID after the message with that ack ID
	// has been processed.  An ackID is considered to have been processed
	// if at least one attempt has been made to acknowledge it.
	Notify func(string)

	mu      sync.Mutex
	pending []string

	wg   sync.WaitGroup
	done chan struct{}
}

// Start intiates processing of ackIDs which are added via Add.
// Notify is called with each ackID once it has been processed.
func (a *acker) Start() {
	a.done = make(chan struct{})

	a.wg.Add(1)
	go func() {
		defer a.wg.Done()
		for {
			select {
			case <-a.AckTick:
				// Launch Ack requests in a separate goroutine so that we don't
				// block the in channel while waiting for the ack request to run.
				a.launchAckRequest(a.batch())
			case <-a.done:
				a.launchAckRequest(a.batch())
				return
			}
		}

	}()
}

// Ack adds an ack id to be acked in the next batch.
func (a *acker) Ack(ackID string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.pending = append(a.pending, ackID)
}

// batch removes a batch of ackIDs from the pending list and returns them.
func (a *acker) batch() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	ret := a.pending
	a.pending = nil
	return ret
}

// launchAckRequest initiates an acknowledgement request in a separate goroutine.
// After the acknowledgement request has completed (regardless of its success
// or failure), ids will be passed to a.Notify.
// Calls to Wait on a.wg will block until this goroutine is done.
func (a *acker) launchAckRequest(ids []string) {
	a.wg.Add(1)
	go func() {
		defer a.wg.Done()
		a.ack(ids)
		for _, id := range ids {
			a.Notify(id)
		}
	}()
}

// Stop processes all pending messages, and releases resources before returning.
func (a *acker) Stop() {
	close(a.done)
	a.wg.Wait()
}

const maxAckRetries = 1

// ack acknowledges the supplied ackIDs.
func (a *acker) ack(ids []string) {
	var retries int
	head, tail := a.Client.s.splitAckIDs(ids)
	for len(head) > 0 {
		err := a.Client.s.acknowledge(a.Ctx, a.Sub, head)
		if err != nil && retries < maxAckRetries {
			// TODO(mcgreevy): more sophisticated retry on failure.
			// NOTE: it is not incorrect to drop acks if we decide not to retry; the messages
			//  will be redelievered, but this is a documented behaviour of the API.
			retries += 1
			continue
		}
		retries = 0
		head, tail = a.Client.s.splitAckIDs(tail)
	}
}
