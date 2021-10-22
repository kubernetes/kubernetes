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
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var _ Retryer = &boRetryer{}

func TestBackofDefault(t *testing.T) {
	backoff := Backoff{}

	max := []time.Duration{1, 2, 4, 8, 16, 30, 30, 30, 30, 30}
	for i, m := range max {
		max[i] = m * time.Second
	}

	for i, w := range max {
		if d := backoff.Pause(); d > w {
			t.Errorf("Backoff duration should be at most %s, got %s", w, d)
		} else if i < len(max)-1 && backoff.cur != max[i+1] {
			t.Errorf("current envelope is %s, want %s", backoff.cur, max[i+1])
		}
	}
}

func TestBackoffExponential(t *testing.T) {
	backoff := Backoff{Initial: 1, Max: 20, Multiplier: 2}
	want := []time.Duration{1, 2, 4, 8, 16, 20, 20, 20, 20, 20}
	for _, w := range want {
		if d := backoff.Pause(); d > w {
			t.Errorf("Backoff duration should be at most %s, got %s", w, d)
		}
	}
}

func TestOnCodes(t *testing.T) {
	// Lint errors grpc.Errorf in 1.6. It mistakenly expects the first arg to Errorf to be a string.
	errf := status.Errorf
	apiErr := errf(codes.Unavailable, "")
	tests := []struct {
		c     []codes.Code
		retry bool
	}{
		{nil, false},
		{[]codes.Code{codes.DeadlineExceeded}, false},
		{[]codes.Code{codes.DeadlineExceeded, codes.Unavailable}, true},
		{[]codes.Code{codes.Unavailable}, true},
	}
	for _, tst := range tests {
		b := OnCodes(tst.c, Backoff{})
		if _, retry := b.Retry(apiErr); retry != tst.retry {
			t.Errorf("retriable codes: %v, error: %s, retry: %t, want %t", tst.c, apiErr, retry, tst.retry)
		}
	}
}
