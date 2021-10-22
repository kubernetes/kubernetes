// Copyright 2018 Google LLC
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

// Package httpreplay provides an API for recording and replaying traffic
// from HTTP-based Google API clients.
//
// To record:
//  1.  Call NewRecorder to get a Recorder.
//  2.  Use its Client method to obtain an HTTP client to use when making API calls.
//  3.  Close the Recorder when you're done. That will save the log of interactions
//      to the file you provided to NewRecorder.
//
// To replay:
//  1.  Call NewReplayer with the same filename you used to record to get a Replayer.
//  2.  Call its Client method and use the client to make the same API calls.
//      You will get back the recorded responses.
//  3.  Close the Replayer when you're done.
//
// This package is EXPERIMENTAL and is subject to change or removal without notice.
// It requires Go version 1.8 or higher.
package httpreplay

// TODO(jba): add examples.

import (
	"context"
	"net/http"

	"cloud.google.com/go/httpreplay/internal/proxy"
	"google.golang.org/api/option"
	htransport "google.golang.org/api/transport/http"
)

// A Recorder records HTTP interactions.
type Recorder struct {
	proxy *proxy.Proxy
}

// NewRecorder creates a recorder that writes to filename. The file will
// also store initial state that can be retrieved to configure replay.
//
// You must call Close on the Recorder to ensure that all data is written.
func NewRecorder(filename string, initial []byte) (*Recorder, error) {
	p, err := proxy.ForRecording(filename, 0)
	if err != nil {
		return nil, err
	}
	p.Initial = initial
	return &Recorder{proxy: p}, nil
}

// RemoveRequestHeaders will remove request headers matching patterns from the log,
// and skip matching them during replay.
//
// Pattern is taken literally except for *, which matches any sequence of characters.
func (r *Recorder) RemoveRequestHeaders(patterns ...string) {
	r.proxy.RemoveRequestHeaders(patterns)
}

// ClearHeaders will replace the value of request and response headers that match
// any of the patterns with CLEARED, on both recording and replay.
// Use ClearHeaders when the header information is secret or may change from run to
// run, but you still want to verify that the headers are being sent and received.
//
// Pattern is taken literally except for *, which matches any sequence of characters.
func (r *Recorder) ClearHeaders(patterns ...string) {
	r.proxy.ClearHeaders(patterns)
}

// RemoveQueryParams will remove URL query parameters matching patterns from the log,
// and skip matching them during replay.
//
// Pattern is taken literally except for *, which matches any sequence of characters.
func (r *Recorder) RemoveQueryParams(patterns ...string) {
	r.proxy.RemoveQueryParams(patterns)
}

// ClearQueryParams will replace the value of URL query parametrs that match any of
// the patterns with CLEARED, on both recording and replay.
// Use ClearQueryParams when the parameter information is secret or may change from
// run to run, but you still want to verify that it are being sent.
//
// Pattern is taken literally except for *, which matches any sequence of characters.
func (r *Recorder) ClearQueryParams(patterns ...string) {
	r.proxy.ClearQueryParams(patterns)
}

// Client returns an http.Client to be used for recording. Provide authentication options
// like option.WithTokenSource as you normally would, or omit them to use Application Default
// Credentials.
func (r *Recorder) Client(ctx context.Context, opts ...option.ClientOption) (*http.Client, error) {
	return proxyClient(ctx, r.proxy, opts...)
}

func proxyClient(ctx context.Context, p *proxy.Proxy, opts ...option.ClientOption) (*http.Client, error) {
	trans, err := htransport.NewTransport(ctx, p.Transport(), opts...)
	if err != nil {
		return nil, err
	}
	return &http.Client{Transport: trans}, nil
}

// Close closes the Recorder and saves the log file.
func (r *Recorder) Close() error {
	return r.proxy.Close()
}

// A Replayer replays previously recorded HTTP interactions.
type Replayer struct {
	proxy *proxy.Proxy
}

// NewReplayer creates a replayer that reads from filename.
func NewReplayer(filename string) (*Replayer, error) {
	p, err := proxy.ForReplaying(filename, 0)
	if err != nil {
		return nil, err
	}
	return &Replayer{proxy: p}, nil
}

// Client returns an HTTP client for replaying. The client does not need to be
// configured with credentials for authenticating to a server, since it never
// contacts a real backend.
func (r *Replayer) Client(ctx context.Context) (*http.Client, error) {
	return proxyClient(ctx, r.proxy, option.WithoutAuthentication())
}

// Initial returns the initial state saved by the Recorder.
func (r *Replayer) Initial() []byte {
	return r.proxy.Initial
}

// IgnoreHeader will not use h when matching requests.
func (r *Replayer) IgnoreHeader(h string) {
	r.proxy.IgnoreHeader(h)
}

// Close closes the replayer.
func (r *Replayer) Close() error {
	return r.proxy.Close()
}

// DebugHeaders helps to determine whether a header should be ignored.
// When true, if requests have the same method, URL and body but differ
// in a header, the first mismatched header is logged.
func DebugHeaders() {
	proxy.DebugHeaders = true
}

// Supported reports whether httpreplay is supported in the current version of Go.
// For Go 1.8 and above, the answer is true.
func Supported() bool { return true }
