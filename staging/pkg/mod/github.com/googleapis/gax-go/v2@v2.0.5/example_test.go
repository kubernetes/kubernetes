// Copyright 2019, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package gax_test

import (
	"context"
	"time"

	"github.com/googleapis/gax-go/v2"
	"google.golang.org/grpc/codes"
)

const someRPCTimeout = 5 * time.Minute

// Some result that the client might return.
type fakeResponse struct{}

// Some client that can perform RPCs.
type fakeClient struct{}

// PerformSomeRPC is a fake RPC that a client might perform.
func (c *fakeClient) PerformSomeRPC(ctx context.Context) (*fakeResponse, error) {
	// An actual client would return something meaningful here.
	return nil, nil
}

func ExampleOnCodes() {
	ctx := context.Background()
	c := &fakeClient{}

	// UNKNOWN and UNAVAILABLE are typically safe to retry for idempotent RPCs.
	retryer := gax.OnCodes([]codes.Code{codes.Unknown, codes.Unavailable}, gax.Backoff{
		Initial:    time.Second,
		Max:        32 * time.Second,
		Multiplier: 2,
	})

	performSomeRPCWithRetry := func(ctx context.Context) (*fakeResponse, error) {
		for {
			resp, err := c.PerformSomeRPC(ctx)
			if err != nil {
				if delay, shouldRetry := retryer.Retry(err); shouldRetry {
					if err := gax.Sleep(ctx, delay); err != nil {
						return nil, err
					}
					continue
				}
				return nil, err
			}
			return resp, err
		}
	}

	// It's recommended to set deadlines on RPCs and around retrying. This is
	// also usually preferred over setting some fixed number of retries: one
	// advantage this has is that backoff settings can be changed independently
	// of the deadline, whereas with a fixed number of retries the deadline
	// would be a constantly-shifting goalpost.
	ctxWithTimeout, cancel := context.WithDeadline(ctx, time.Now().Add(someRPCTimeout))
	defer cancel()

	resp, err := performSomeRPCWithRetry(ctxWithTimeout)
	if err != nil {
		// TODO: handle err
	}
	_ = resp // TODO: use resp if err is nil
}
