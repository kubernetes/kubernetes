/*
Copyright 2016 The Kubernetes Authors.

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

package remotecommand

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	remotecommandconsts "k8s.io/apimachinery/pkg/util/remotecommand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/streaming/pkg/httpstream"
	"k8s.io/streaming/pkg/httpstream/spdy"
	utilexec "k8s.io/utils/exec"
)

// An error stream that ends without any data is an error for v4 and v5, whose
// servers write the status before closing it, and a success for v2 and v3,
// whose error stream carries raw text.
func TestEmptyErrorStream(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	for _, tc := range []struct {
		name    string
		decoder errorStreamDecoder
		wantErr bool
	}{
		{name: "v2", decoder: &errorDecoderV2{}},
		{name: "v3", decoder: &errorDecoderV3{}},
		{name: "v4", decoder: &errorDecoderV4{}, wantErr: true},
	} {
		var err error
		select {
		case err = <-watchErrorStream(logger, strings.NewReader(""), tc.decoder):
		case <-time.After(wait.ForeverTestTimeout):
			t.Fatalf("%s: timed out waiting for the error stream result", tc.name)
		}
		switch {
		case tc.wantErr && err == nil:
			t.Errorf("%s: an empty error stream was read as success", tc.name)
		case tc.wantErr && !strings.Contains(err.Error(), "before the command's status was received"):
			t.Errorf("%s: unexpected error: %v", tc.name, err)
		case !tc.wantErr && err != nil:
			t.Errorf("%s: an empty error stream is success, got %v", tc.name, err)
		}
	}
}

// spdyServerWithoutStatus serves one SPDY session of the given protocol and
// closes the connection once the client's streams are open, without writing
// anything to them.
func spdyServerWithoutStatus(t *testing.T, protocol string) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		if _, err := httpstream.Handshake(req, w, []string{protocol}); err != nil {
			t.Errorf("error on handshake: %v", err)
			return
		}
		replies := make(chan (<-chan struct{}), 2)
		conn := spdy.NewResponseUpgrader().UpgradeResponse(w, req, func(_ httpstream.Stream, replySent <-chan struct{}) error {
			replies <- replySent
			return nil
		})
		if conn == nil {
			t.Error("error upgrading the connection")
			return
		}
		defer conn.Close()
		// The client opens the error stream and stdout, and waits for the
		// reply to each. Close once both replies are out.
		for range 2 {
			select {
			case replySent := <-replies:
				<-replySent
			case <-conn.CloseChan():
				return
			case <-time.After(wait.ForeverTestTimeout):
				return
			}
		}
	}
}

// A v4 or v5 server that closes the connection without writing a status makes
// StreamWithContext return an error rather than nil, over SPDY and over
// WebSocket. A v3 server that does the same ended the command successfully.
func TestStreamWithoutStatus(t *testing.T) {
	for _, tc := range []struct {
		name     string
		server   func(t *testing.T) http.HandlerFunc
		executor func(serverURL *url.URL) (Executor, error)
		wantErr  bool
	}{
		{
			name: "spdy v4",
			server: func(t *testing.T) http.HandlerFunc {
				return spdyServerWithoutStatus(t, remotecommandconsts.StreamProtocolV4Name)
			},
			executor: func(serverURL *url.URL) (Executor, error) {
				return NewSPDYExecutor(&rest.Config{Host: serverURL.Host}, "POST", serverURL)
			},
			wantErr: true,
		},
		{
			name: "spdy v3",
			server: func(t *testing.T) http.HandlerFunc {
				return spdyServerWithoutStatus(t, remotecommandconsts.StreamProtocolV3Name)
			},
			executor: func(serverURL *url.URL) (Executor, error) {
				return NewSPDYExecutor(&rest.Config{Host: serverURL.Host}, "POST", serverURL)
			},
		},
		{
			name: "websocket v5",
			server: func(t *testing.T) http.HandlerFunc {
				return func(w http.ResponseWriter, req *http.Request) {
					conns, err := webSocketServerStreams(req, w, &options{stdout: true})
					if err != nil {
						t.Errorf("error creating streams: %v", err)
						return
					}
					// A normal close (1000): the client reads an empty error
					// stream, as from a server that went away cleanly.
					_ = conns.conn.Close()
				}
			},
			executor: func(serverURL *url.URL) (Executor, error) {
				return NewWebSocketExecutor(&rest.Config{Host: serverURL.Host}, "GET", serverURL.String())
			},
			wantErr: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(tc.server(t))
			defer server.Close()
			serverURL, err := url.Parse(server.URL + "?stdout=true")
			if err != nil {
				t.Fatal(err)
			}
			executor, err := tc.executor(serverURL)
			if err != nil {
				t.Fatal(err)
			}

			ctx, cancel := context.WithTimeout(context.Background(), wait.ForeverTestTimeout)
			defer cancel()
			err = executor.StreamWithContext(ctx, StreamOptions{Stdout: &bytes.Buffer{}})
			if !tc.wantErr {
				if err != nil {
					t.Fatalf("StreamWithContext returned %v for a session that ended without a status", err)
				}
				return
			}
			if err == nil {
				t.Fatal("StreamWithContext returned nil for a session that ended without a status")
			}
			if !strings.Contains(err.Error(), "before the command's status was received") {
				t.Errorf("unexpected error: %v", err)
			}
			// kubectl turns an ExitError into the command's exit status; a
			// session that was cut must not pass for one.
			var exitErr utilexec.ExitError
			if errors.As(err, &exitErr) {
				t.Errorf("the error reads as the command's exit status: %v", err)
			}
		})
	}
}

func TestV4ErrorDecoder(t *testing.T) {
	dec := errorDecoderV4{}

	type Test struct {
		message string
		err     string
	}

	for _, test := range []Test{
		{
			message: "{}",
			err:     "error stream protocol error: unknown error",
		},
		{
			message: "{",
			err:     "unexpected end of JSON input in \"{\"",
		},
		{
			message: `{"status": "Success" }`,
			err:     "",
		},
		{
			message: `{"status": "Failure", "message": "foobar" }`,
			err:     "foobar",
		},
		{
			message: `{"status": "Failure", "message": "foobar", "reason": "NonZeroExitCode", "details": {"causes": [{"reason": "foo"}] } }`,
			err:     "error stream protocol error: no ExitCode cause given",
		},
		{
			message: `{"status": "Failure", "message": "foobar", "reason": "NonZeroExitCode", "details": {"causes": [{"reason": "ExitCode"}] } }`,
			err:     "error stream protocol error: invalid exit code value \"\"",
		},
		{
			message: `{"status": "Failure", "message": "foobar", "reason": "NonZeroExitCode", "details": {"causes": [{"reason": "ExitCode", "message": "42"}] } }`,
			err:     "command terminated with exit code 42",
		},
	} {
		err := dec.decode([]byte(test.message))
		want := test.err
		if want == "" {
			want = "<nil>"
		}
		if got := fmt.Sprintf("%v", err); !strings.Contains(got, want) {
			t.Errorf("wrong error for message %q: want=%q, got=%q", test.message, want, got)
		}
	}
}
