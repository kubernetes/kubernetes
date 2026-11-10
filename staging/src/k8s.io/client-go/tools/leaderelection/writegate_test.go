/*
Copyright The Kubernetes Authors.

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

package leaderelection

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/wait"
)

type roundTripperFunc func(req *http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func okResponse(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("ok")),
		Request:    req,
	}, nil
}

type closeTrackingBody struct {
	io.Reader
	closed bool
}

func (b *closeTrackingBody) Close() error {
	b.closed = true
	return nil
}

func newRequest(ctx context.Context, t *testing.T, method string) *http.Request {
	t.Helper()
	req, err := http.NewRequestWithContext(ctx, method, "https://localhost/api/v1/namespaces/default/pods", strings.NewReader("body"))
	if err != nil {
		t.Fatal(err)
	}
	return req
}

func TestWriteGateMethods(t *testing.T) {
	tests := []struct {
		method           string
		passesWhenClosed bool
	}{
		{http.MethodGet, true},
		{http.MethodHead, true},
		{http.MethodOptions, true},
		{http.MethodTrace, true},
		{http.MethodPost, false},
		{http.MethodPut, false},
		{http.MethodPatch, false},
		{http.MethodDelete, false},
		{http.MethodConnect, false},
		{"PROPFIND", false},
	}
	for _, tc := range tests {
		t.Run(tc.method, func(t *testing.T) {
			gate := &writeGate{}
			rt := gate.wrapper()(roundTripperFunc(okResponse))
			roundTrip := func() error {
				resp, err := rt.RoundTrip(newRequest(context.Background(), t, tc.method))
				if err == nil {
					_ = resp.Body.Close()
				}
				return err
			}
			expectClosed := func(step string) {
				t.Helper()
				err := roundTrip()
				if tc.passesWhenClosed && err != nil {
					t.Fatalf("%s: expected %s to pass a closed gate, got %v", step, tc.method, err)
				}
				if !tc.passesWhenClosed && !errors.Is(err, ErrWriteGateClosed) {
					t.Fatalf("%s: expected %s to be rejected by a closed gate with ErrWriteGateClosed, got %v", step, tc.method, err)
				}
			}
			expectOpen := func(step string) {
				t.Helper()
				if err := roundTrip(); err != nil {
					t.Fatalf("%s: expected %s to pass an open gate, got %v", step, tc.method, err)
				}
			}

			// The gate starts closed. Opening and closing are idempotent.
			expectClosed("initially")
			gate.open()
			gate.open()
			expectOpen("after open")
			gate.close()
			gate.close()
			expectClosed("after close")
			gate.open()
			expectOpen("after reopen")
		})
	}
}

func TestWriteGateRejectionClosesRequestBody(t *testing.T) {
	gate := &writeGate{}
	rt := gate.wrapper()(roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		t.Fatal("underlying transport must not be called")
		return nil, nil
	}))
	body := &closeTrackingBody{Reader: strings.NewReader("body")}
	req, err := http.NewRequest(http.MethodPost, "https://localhost/", body)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rt.RoundTrip(req); !errors.Is(err, ErrWriteGateClosed) {
		t.Fatalf("expected ErrWriteGateClosed, got %v", err)
	}
	if !body.closed {
		t.Error("request body was not closed on rejection")
	}
}

// TestWriteGateInFlightRequests covers requests that are in flight or just
// completed when the gate closes, and requests cancelled by the caller.
func TestWriteGateInFlightRequests(t *testing.T) {
	tests := []struct {
		name   string
		method string
		// cancelCaller cancels the caller's context before the request.
		cancelCaller bool
		// closeInFlight closes the gate while the request is in flight.
		closeInFlight bool
		// closeAfterReturn closes the gate once the round trip has returned.
		closeAfterReturn bool
		wantErr          error
	}{
		{name: "read in flight when the gate closes", method: http.MethodGet, closeInFlight: true},
		{name: "write in flight when the gate closes", method: http.MethodPut, closeInFlight: true, wantErr: ErrWriteGateClosed},
		{name: "write completed before the gate closes", method: http.MethodPost, closeAfterReturn: true},
		{name: "write cancelled by the caller", method: http.MethodPost, cancelCaller: true, wantErr: context.Canceled},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gate := &writeGate{}
			gate.open()

			inFlight := make(chan struct{})
			release := make(chan struct{})
			var cause error
			rt := gate.wrapper()(roundTripperFunc(func(req *http.Request) (*http.Response, error) {
				close(inFlight)
				select {
				case <-req.Context().Done():
					cause = context.Cause(req.Context())
					return nil, req.Context().Err()
				case <-release:
					return okResponse(req)
				}
			}))

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if tc.cancelCaller {
				cancel()
			}
			type result struct {
				resp *http.Response
				err  error
			}
			results := make(chan result, 1)
			go func() {
				resp, err := rt.RoundTrip(newRequest(ctx, t, tc.method))
				results <- result{resp, err}
			}()

			select {
			case <-inFlight:
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatal("request never reached the transport")
			}
			if tc.closeInFlight {
				gate.close()
			}
			if tc.wantErr == nil {
				// Successful requests complete through the transport; failing
				// ones must fail through their context, so don't release them.
				close(release)
			}
			var res result
			select {
			case res = <-results:
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatal("round trip did not return")
			}
			if tc.closeAfterReturn {
				gate.close()
			}

			if tc.wantErr == nil {
				if res.err != nil {
					t.Fatalf("unexpected error: %v", res.err)
				}
				body, err := io.ReadAll(res.resp.Body)
				if err != nil || string(body) != "ok" {
					t.Fatalf("reading response body after the gate closed: %q, %v", body, err)
				}
				if err := res.resp.Body.Close(); err != nil {
					t.Fatal(err)
				}
				return
			}
			if !errors.Is(res.err, tc.wantErr) {
				t.Fatalf("expected %v, got %v", tc.wantErr, res.err)
			}
			if errors.Is(tc.wantErr, ErrWriteGateClosed) && !errors.Is(cause, ErrWriteGateClosed) {
				t.Errorf("expected request context cause to be ErrWriteGateClosed, got %v", cause)
			}
			if errors.Is(tc.wantErr, context.Canceled) && errors.Is(res.err, ErrWriteGateClosed) {
				t.Fatalf("caller cancellation must not be reported as a gate rejection, got %v", res.err)
			}
		})
	}
}

func TestWriteGateWrappedRoundTripper(t *testing.T) {
	inner := roundTripperFunc(okResponse)
	rt := (&writeGate{}).wrapper()(inner)
	wrapper, ok := rt.(utilnet.RoundTripperWrapper)
	if !ok {
		t.Fatal("gated round tripper does not implement RoundTripperWrapper")
	}
	if _, ok := wrapper.WrappedRoundTripper().(roundTripperFunc); !ok {
		t.Fatalf("WrappedRoundTripper returned %T", wrapper.WrappedRoundTripper())
	}
}
