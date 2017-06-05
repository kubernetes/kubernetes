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

	"golang.org/x/net/context"
)

// puller fetches messages from the server in a batch.
type puller struct {
	ctx    context.Context
	cancel context.CancelFunc

	// keepAlive takes ownership of the lifetime of the message identified
	// by ackID, ensuring that its ack deadline does not expire. It should
	// be called each time a new message is fetched from the server, even
	// if it is not yet returned from Next.
	keepAlive func(ackID string)

	// abandon should be called for each message which has previously been
	// passed to keepAlive, but will never be returned by Next.
	abandon func(ackID string)

	// fetch fetches a batch of messages from the server.
	fetch func() ([]*Message, error)

	mu  sync.Mutex
	buf []*Message
}

// newPuller constructs a new puller.
// batchSize is the maximum number of messages to fetch at once.
// No more than batchSize messages will be outstanding at any time.
func newPuller(s service, subName string, ctx context.Context, batchSize int64, keepAlive, abandon func(ackID string)) *puller {
	ctx, cancel := context.WithCancel(ctx)
	return &puller{
		cancel:    cancel,
		keepAlive: keepAlive,
		abandon:   abandon,
		ctx:       ctx,
		fetch:     func() ([]*Message, error) { return s.fetchMessages(ctx, subName, batchSize) },
	}
}

const maxPullAttempts = 2

// Next returns the next message from the server, fetching a new batch if necessary.
// keepAlive is called with the ackIDs of newly fetched messages.
// If p.Ctx has already been cancelled before Next is called, no new messages
// will be fetched.
func (p *puller) Next() (*Message, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// If ctx has been cancelled, return straight away (even if there are buffered messages available).
	select {
	case <-p.ctx.Done():
		return nil, p.ctx.Err()
	default:
	}

	for len(p.buf) == 0 {
		var buf []*Message
		var err error

		for i := 0; i < maxPullAttempts; i++ {
			// Once Stop has completed, all future calls to Next will immediately fail at this point.
			buf, err = p.fetch()
			if err == nil || err == context.Canceled || err == context.DeadlineExceeded {
				break
			}
		}
		if err != nil {
			return nil, err
		}

		for _, m := range buf {
			p.keepAlive(m.ackID)
		}
		p.buf = buf
	}

	m := p.buf[0]
	p.buf = p.buf[1:]
	return m, nil
}

// Stop aborts any pending calls to Next, and prevents any future ones from succeeding.
// Stop also abandons any messages that have been pre-fetched.
// Once Stop completes, no calls to Next will succeed.
func (p *puller) Stop() {
	// Next may be executing in another goroutine. Cancel it, and then wait until it terminates.
	p.cancel()
	p.mu.Lock()
	defer p.mu.Unlock()

	for _, m := range p.buf {
		p.abandon(m.ackID)
	}
	p.buf = nil
}
