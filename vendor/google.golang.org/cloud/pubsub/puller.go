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
	Client *Client
	Sub    string

	// The maximum number of messages to fetch at once.
	// No more than BatchSize messages will be outstanding at any time.
	BatchSize int64

	// A function to call when a new message is fetched from the server, but not yet returned from Next.
	Notify func(ackID string)

	mu  sync.Mutex
	buf []*Message
}

// Next returns the next message from the server, fetching a new batch if necessary.
// Notify is called with the ackIDs of newly fetched messages.
func (p *puller) Next(ctx context.Context) (*Message, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	for len(p.buf) == 0 {
		var err error
		p.buf, err = p.Client.s.fetchMessages(ctx, p.Sub, p.BatchSize)
		if err != nil {
			// TODO: retry before giving up.
			return nil, err
		}
		for _, m := range p.buf {
			p.Notify(m.AckID)
		}
	}
	m := p.buf[0]
	p.buf = p.buf[1:]
	return m, nil
}

// Pending returns the list of messages that have been fetched from the server
// but not yet returned via Next.
func (p *puller) Pending() []*Message {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.buf
}
