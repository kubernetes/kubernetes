// Copyright 2016, Google Inc.
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

package gax

import (
	"context"
	"strings"
	"time"
)

// APICall is a user defined call stub.
type APICall func(context.Context, CallSettings) error

// Invoke calls the given APICall,
// performing retries as specified by opts, if any.
func Invoke(ctx context.Context, call APICall, opts ...CallOption) error {
	var settings CallSettings
	for _, opt := range opts {
		opt.Resolve(&settings)
	}
	return invoke(ctx, call, settings, Sleep)
}

// Sleep is similar to time.Sleep, but it can be interrupted by ctx.Done() closing.
// If interrupted, Sleep returns ctx.Err().
func Sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	select {
	case <-ctx.Done():
		t.Stop()
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

type sleeper func(ctx context.Context, d time.Duration) error

// invoke implements Invoke, taking an additional sleeper argument for testing.
func invoke(ctx context.Context, call APICall, settings CallSettings, sp sleeper) error {
	var retryer Retryer
	for {
		err := call(ctx, settings)
		if err == nil {
			return nil
		}
		if settings.Retry == nil {
			return err
		}
		// Never retry permanent certificate errors. (e.x. if ca-certificates
		// are not installed). We should only make very few, targeted
		// exceptions: many (other) status=Unavailable should be retried, such
		// as if there's a network hiccup, or the internet goes out for a
		// minute. This is also why here we are doing string parsing instead of
		// simply making Unavailable a non-retried code elsewhere.
		if strings.Contains(err.Error(), "x509: certificate signed by unknown authority") {
			return err
		}
		if retryer == nil {
			if r := settings.Retry(); r != nil {
				retryer = r
			} else {
				return err
			}
		}
		if d, ok := retryer.Retry(err); !ok {
			return err
		} else if err = sp(ctx, d); err != nil {
			return err
		}
	}
}
