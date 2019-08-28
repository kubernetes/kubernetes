/*
Copyright 2018 The Kubernetes Authors.

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

package transport

import (
	"fmt"
	"net/http"
	"reflect"
	"sync"
	"testing"
	"time"

	"golang.org/x/oauth2"
)

type testTokenSource struct {
	calls int
	tok   *oauth2.Token
	err   error
}

func (ts *testTokenSource) Token() (*oauth2.Token, error) {
	ts.calls++
	return ts.tok, ts.err
}

func TestCachingTokenSource(t *testing.T) {
	start := time.Now()
	tokA := &oauth2.Token{
		AccessToken: "a",
		Expiry:      start.Add(10 * time.Minute),
	}
	tokB := &oauth2.Token{
		AccessToken: "b",
		Expiry:      start.Add(20 * time.Minute),
	}
	tests := []struct {
		name string

		tok   *oauth2.Token
		tsTok *oauth2.Token
		tsErr error
		wait  time.Duration

		wantTok     *oauth2.Token
		wantErr     bool
		wantTSCalls int
	}{
		{
			name:    "valid token returned from cache",
			tok:     tokA,
			wantTok: tokA,
		},
		{
			name:    "valid token returned from cache 1 minute before scheduled refresh",
			tok:     tokA,
			wait:    8 * time.Minute,
			wantTok: tokA,
		},
		{
			name:        "new token created when cache is empty",
			tsTok:       tokA,
			wantTok:     tokA,
			wantTSCalls: 1,
		},
		{
			name:        "new token created 1 minute after scheduled refresh",
			tok:         tokA,
			tsTok:       tokB,
			wait:        10 * time.Minute,
			wantTok:     tokB,
			wantTSCalls: 1,
		},
		{
			name:        "error on create token returns error",
			tsErr:       fmt.Errorf("error"),
			wantErr:     true,
			wantTSCalls: 1,
		},
	}
	for _, c := range tests {
		t.Run(c.name, func(t *testing.T) {
			tts := &testTokenSource{
				tok: c.tsTok,
				err: c.tsErr,
			}

			ts := &cachingTokenSource{
				base:   tts,
				tok:    c.tok,
				leeway: 1 * time.Minute,
				now:    func() time.Time { return start.Add(c.wait) },
			}

			gotTok, gotErr := ts.Token()
			if got, want := gotTok, c.wantTok; !reflect.DeepEqual(got, want) {
				t.Errorf("unexpected token:\n\tgot:\t%#v\n\twant:\t%#v", got, want)
			}
			if got, want := tts.calls, c.wantTSCalls; got != want {
				t.Errorf("unexpected number of Token() calls: got %d, want %d", got, want)
			}
			if gotErr == nil && c.wantErr {
				t.Errorf("wanted error but got none")
			}
			if gotErr != nil && !c.wantErr {
				t.Errorf("unexpected error: %v", gotErr)
			}
		})
	}
}

func TestCachingTokenSourceRace(t *testing.T) {
	for i := 0; i < 100; i++ {
		tts := &testTokenSource{
			tok: &oauth2.Token{
				AccessToken: "a",
				Expiry:      time.Now().Add(1000 * time.Hour),
			},
		}

		ts := &cachingTokenSource{
			now:    time.Now,
			base:   tts,
			leeway: 1 * time.Minute,
		}

		var wg sync.WaitGroup
		wg.Add(100)

		for i := 0; i < 100; i++ {
			go func() {
				defer wg.Done()
				if _, err := ts.Token(); err != nil {
					t.Fatalf("err: %v", err)
				}
			}()
		}
		wg.Wait()
		if tts.calls != 1 {
			t.Errorf("expected one call to Token() but saw: %d", tts.calls)
		}
	}
}

type uncancellableRT struct {
	rt http.RoundTripper
}

func (urt *uncancellableRT) RoundTrip(req *http.Request) (*http.Response, error) {
	return urt.rt.RoundTrip(req)
}

func TestCancellation(t *testing.T) {
	tests := []struct {
		name          string
		header        http.Header
		wrapTransport func(http.RoundTripper) http.RoundTripper
		expectCancel  bool
	}{
		{
			name:         "cancel req with bearer token skips oauth rt",
			header:       map[string][]string{"Authorization": {"Bearer TOKEN"}},
			expectCancel: true,
		},
		{
			name:         "cancel req without bearer token hits both rts",
			expectCancel: true,
		},
		{
			name: "cancel req without bearer token hits both wrapped rts",
			wrapTransport: func(rt http.RoundTripper) http.RoundTripper {
				return NewUserAgentRoundTripper("testing testing", rt)
			},
			expectCancel: true,
		},
		{
			name: "can't cancel request with rts that doesn't implent unwrap or cancel",
			wrapTransport: func(rt http.RoundTripper) http.RoundTripper {
				return &uncancellableRT{rt: rt}
			},
			expectCancel: false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			baseRecorder := &recordCancelRoundTripper{}

			var base http.RoundTripper = baseRecorder
			if test.wrapTransport != nil {
				base = test.wrapTransport(base)
			}

			rt := &tokenSourceTransport{
				base: base,
				ort: &oauth2.Transport{
					Base: base,
				},
			}

			rt.CancelRequest(&http.Request{
				Header: test.header,
			})

			if baseRecorder.canceled != test.expectCancel {
				t.Errorf("unexpected cancel: got=%v, want=%v", baseRecorder.canceled, test.expectCancel)
			}
		})
	}
}

type recordCancelRoundTripper struct {
	canceled bool
	base     http.RoundTripper
}

func (rt *recordCancelRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return nil, nil
}

func (rt *recordCancelRoundTripper) CancelRequest(req *http.Request) {
	rt.canceled = true
	if rt.base != nil {
		tryCancelRequest(rt.base, req)
	}
}
