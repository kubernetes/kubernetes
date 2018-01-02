// Copyright 2017 Google Inc. All Rights Reserved.
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

package gensupport

import (
	"errors"
	"io"
	"net"
	"net/http"
	"testing"
	"time"

	"golang.org/x/net/context"
)

func TestRetry(t *testing.T) {
	testCases := []struct {
		desc       string
		respStatus []int // HTTP status codes returned (length indicates number of calls we expect).
		maxRetry   int   // Max number of calls allowed by the BackoffStrategy.
		wantStatus int   // StatusCode of returned response.
	}{
		{
			desc:       "First call successful",
			respStatus: []int{200},
			maxRetry:   3,
			wantStatus: 200,
		},
		{
			desc:       "Retry before success",
			respStatus: []int{500, 500, 500, 200},
			maxRetry:   3,
			wantStatus: 200,
		},
		{
			desc:       "Backoff strategy abandons after 3 retries",
			respStatus: []int{500, 500, 500, 500},
			maxRetry:   3,
			wantStatus: 500,
		},
		{
			desc:       "Backoff strategy abandons after 2 retries",
			respStatus: []int{500, 500, 500},
			maxRetry:   2,
			wantStatus: 500,
		},
	}
	for _, tt := range testCases {
		// Function consumes tt.respStatus
		f := func() (*http.Response, error) {
			if len(tt.respStatus) == 0 {
				return nil, errors.New("too many requests to function")
			}
			resp := &http.Response{StatusCode: tt.respStatus[0]}
			tt.respStatus = tt.respStatus[1:]
			return resp, nil
		}

		backoff := &LimitRetryStrategy{
			Max:      tt.maxRetry,
			Strategy: NoPauseStrategy,
		}

		resp, err := Retry(nil, f, backoff)
		if err != nil {
			t.Errorf("%s: Retry returned err %v", tt.desc, err)
		}
		if got := resp.StatusCode; got != tt.wantStatus {
			t.Errorf("%s: Retry returned response with StatusCode=%d; want %d", got, tt.wantStatus)
		}
		if len(tt.respStatus) != 0 {
			t.Errorf("%s: f was not called enough; status codes remaining: %v", tt.desc, tt.respStatus)
		}
	}
}

type checkCloseReader struct {
	closed bool
}

func (c *checkCloseReader) Read(p []byte) (n int, err error) { return 0, io.EOF }
func (c *checkCloseReader) Close() error {
	c.closed = true
	return nil
}

func TestRetryClosesBody(t *testing.T) {
	var i int
	responses := []*http.Response{
		{StatusCode: 500, Body: &checkCloseReader{}},
		{StatusCode: 500, Body: &checkCloseReader{}},
		{StatusCode: 200, Body: &checkCloseReader{}},
	}
	f := func() (*http.Response, error) {
		resp := responses[i]
		i++
		return resp, nil
	}

	resp, err := Retry(nil, f, NoPauseStrategy)
	if err != nil {
		t.Fatalf("Retry returned error: %v", err)
	}
	if resp != responses[2] {
		t.Errorf("Retry returned %v; want %v", resp, responses[2])
	}
	for i, resp := range responses {
		want := i != 2 // Only the last response should not be closed.
		got := resp.Body.(*checkCloseReader).closed
		if got != want {
			t.Errorf("response[%d].Body closed = %t, want %t", got, want)
		}
	}
}

func RetryReturnsOnContextCancel(t *testing.T) {
	f := func() (*http.Response, error) {
		return nil, io.ErrUnexpectedEOF
	}
	backoff := UniformPauseStrategy(time.Hour)
	ctx, cancel := context.WithCancel(context.Background())

	errc := make(chan error, 1)
	go func() {
		_, err := Retry(ctx, f, backoff)
		errc <- err
	}()

	cancel()
	select {
	case err := <-errc:
		if err != ctx.Err() {
			t.Errorf("Retry returned err: %v, want %v", err, ctx.Err())
		}
	case <-time.After(5 * time.Second):
		t.Errorf("Timed out waiting for Retry to return")
	}
}

func TestShouldRetry(t *testing.T) {
	testCases := []struct {
		status int
		err    error
		want   bool
	}{
		{status: 200, want: false},
		{status: 308, want: false},
		{status: 403, want: false},
		{status: 429, want: true},
		{status: 500, want: true},
		{status: 503, want: true},
		{status: 600, want: false},
		{err: io.EOF, want: false},
		{err: errors.New("random badness"), want: false},
		{err: io.ErrUnexpectedEOF, want: true},
		{err: &net.AddrError{}, want: false},              // Not temporary.
		{err: &net.DNSError{IsTimeout: true}, want: true}, // Temporary.
	}
	for _, tt := range testCases {
		if got := shouldRetry(tt.status, tt.err); got != tt.want {
			t.Errorf("shouldRetry(%d, %v) = %t; want %t", tt.status, tt.err, got, tt.want)
		}
	}
}
