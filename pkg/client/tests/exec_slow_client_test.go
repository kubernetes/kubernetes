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

package tests

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/proxy"
	remotecommandconsts "k8s.io/apimachinery/pkg/util/remotecommand"
	translator "k8s.io/apiserver/pkg/util/proxy"
	restclient "k8s.io/client-go/rest"
	remoteclient "k8s.io/client-go/tools/remotecommand"
	clientspdy "k8s.io/client-go/transport/spdy"
	"k8s.io/cri-streaming/pkg/streaming/remotecommand"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/streaming/pkg/httpstream"
	"k8s.io/streaming/pkg/httpstream/spdy"
)

// payloadExecutor writes size bytes to stdout in 64 KiB chunks and returns.
type payloadExecutor struct {
	size int
}

func (e *payloadExecutor) ExecInContainer(_ context.Context, _ string, _ string, _ string, _ []string, _ io.Reader, out, _ io.WriteCloser, _ bool, _ <-chan remotecommand.TerminalSize, _ time.Duration) error {
	chunk := make([]byte, 64<<10)
	remaining := e.size
	for remaining > 0 {
		n := min(len(chunk), remaining)
		if _, err := out.Write(chunk[:n]); err != nil {
			return err
		}
		remaining -= n
	}
	return nil
}

// slowWriter counts what it receives and pauses after every write, like a
// client draining stdout into a slow pipe.
type slowWriter struct {
	n     int
	pause time.Duration
}

func (w *slowWriter) Write(p []byte) (int, error) {
	w.n += len(p)
	time.Sleep(w.pause)
	return len(p), nil
}

// trickle yields one byte every period until stopped.
type trickle struct {
	stop   chan struct{}
	period time.Duration
}

func (r *trickle) Read(p []byte) (int, error) {
	select {
	case <-r.stop:
		return 0, io.EOF
	case <-time.After(r.period):
		p[0] = '\n'
		return 1, nil
	}
}

type proxyResponder struct{ t *testing.T }

func (r proxyResponder) Error(_ http.ResponseWriter, _ *http.Request, err error) {
	r.t.Errorf("proxy error: %v", err)
}

func parseURL(t *testing.T, s string) *url.URL {
	t.Helper()
	u, err := url.Parse(s)
	if err != nil {
		t.Fatal(err)
	}
	return u
}

// A client that reads more slowly than the command writes must receive the
// whole output, and the command's status, through the apiserver and the
// kubelet: over SPDY both forward the upgraded connection, over WebSocket the
// apiserver translates to SPDY towards the kubelet.
func TestExecSlowClientThroughAPIServerAndKubelet(t *testing.T) {
	const payload = 16 << 20
	// The proxies forward to their configured location; the apiserver builds
	// it, query included, from the exec options it validated.
	const execPath = "/exec/pod/container?input=1&output=1&command=cmd"

	for _, tc := range []struct {
		name      string
		apiserver func(t *testing.T, kubelet string) http.Handler
		executor  func(t *testing.T, apiserver string) remoteclient.Executor
	}{
		{
			name: "spdy",
			apiserver: func(t *testing.T, kubelet string) http.Handler {
				return proxy.NewUpgradeAwareHandler(parseURL(t, kubelet+execPath), nil, false, true, proxyResponder{t})
			},
			executor: func(t *testing.T, apiserver string) remoteclient.Executor {
				// client-go pings every 5s; ping often so that the outcome does
				// not depend on where in the transfer a ping falls.
				rt, err := spdy.NewRoundTripperWithConfig(spdy.RoundTripperConfig{PingPeriod: 5 * time.Millisecond})
				if err != nil {
					t.Fatal(err)
				}
				rt.Dialer = smallReceiveBufferDialer()
				e, err := remoteclient.NewSPDYExecutorForTransports(rt, clientspdy.NewUpgraderForStreaming(rt), http.MethodPost, parseURL(t, apiserver+execPath))
				if err != nil {
					t.Fatal(err)
				}
				return e
			},
		},
		{
			name: "websocket",
			apiserver: func(t *testing.T, kubelet string) http.Handler {
				return translator.NewStreamTranslatorHandler(parseURL(t, kubelet+execPath), nil, 0, translator.Options{Stdin: true, Stdout: true})
			},
			executor: func(t *testing.T, apiserver string) remoteclient.Executor {
				// The WebSocket executor builds its own dialer, so this client's
				// receive buffer stays at the default; the input frames arriving
				// after the translator's close are enough to reset the connection.
				config := &restclient.Config{Host: apiserver}
				e, err := remoteclient.NewWebSocketExecutor(config, http.MethodPost, apiserver+execPath)
				if err != nil {
					t.Fatal(err)
				}
				return e
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backendDone := make(chan struct{})
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				defer close(backendDone)
				opts, err := remotecommand.NewOptions(req)
				if err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				remotecommand.ServeExec(w, req, &payloadExecutor{size: payload}, "pod", "uid", "container", []string{"cmd"}, opts, time.Minute, remotecommand.DefaultStreamCreationTimeout, remotecommand.SupportedStreamingProtocols)
			}))
			t.Cleanup(backend.Close)
			kubelet := httptest.NewServer(proxy.NewUpgradeAwareHandler(parseURL(t, backend.URL+execPath), nil, false, true, proxyResponder{t}))
			t.Cleanup(kubelet.Close)
			apiserverDone := make(chan struct{})
			apiserverHandler := tc.apiserver(t, kubelet.URL)
			apiserver := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				defer close(apiserverDone)
				apiserverHandler.ServeHTTP(w, req)
			}))
			t.Cleanup(apiserver.Close)

			executor := tc.executor(t, apiserver.URL)
			stdout := &slowWriter{pause: 2 * time.Millisecond}
			// Like a user typing into an interactive session, the client keeps
			// sending input while the output is still arriving.
			stdin := &trickle{stop: make(chan struct{}), period: 5 * time.Millisecond}
			defer close(stdin.stop)
			start := time.Now()
			err := executor.StreamWithContext(context.Background(), remoteclient.StreamOptions{Stdin: stdin, Stdout: stdout})
			elapsed := time.Since(start)

			t.Logf("received %d of %d bytes in %v, err %v", stdout.n, payload, elapsed.Round(time.Millisecond), err)
			if err != nil {
				t.Errorf("exec failed: %v", err)
			}
			if stdout.n != payload {
				t.Errorf("stdout truncated: got %d bytes, want %d (lost %d)", stdout.n, payload, payload-stdout.n)
			}
			for _, done := range []struct {
				name string
				ch   <-chan struct{}
			}{{"apiserver", apiserverDone}, {"streaming server", backendDone}} {
				select {
				case <-done.ch:
				case <-time.After(30 * time.Second):
					t.Fatalf("the %s handler did not return after the client closed the connection", done.name)
				}
			}
		})
	}
}

// statuslessExecServer is a v4 server that closes the connection without ever
// writing a status, which is what a streaming server from before this change
// produced for a client that had not read everything when the command exited.
// It negotiates v4, writes size bytes to stdout and returns.
func statuslessExecServer(t *testing.T, size int) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if _, err := httpstream.Handshake(req, w, []string{remotecommandconsts.StreamProtocolV4Name}); err != nil {
			t.Error(err)
			return
		}
		streams := make(chan httpstream.Stream, 4)
		conn := spdy.NewResponseUpgrader().UpgradeResponse(w, req, func(stream httpstream.Stream, replySent <-chan struct{}) error {
			streams <- stream
			return nil
		})
		if conn == nil {
			return
		}
		defer func() { _ = conn.Close() }()
		var stdout httpstream.Stream
		timeout := time.After(10 * time.Second)
		for stdout == nil {
			select {
			case s := <-streams:
				if s.Headers().Get(api.StreamType) == api.StreamTypeStdout {
					stdout = s
				}
			case <-timeout:
				t.Error("the client did not open a stdout stream")
				return
			}
		}
		chunk := make([]byte, 64<<10)
		for remaining := size; remaining > 0; {
			n := min(len(chunk), remaining)
			if _, err := stdout.Write(chunk[:n]); err != nil {
				return
			}
			remaining -= n
		}
	})
}

// A session that ends without a status is a failure, not a successful run: a
// server that closes the connection before the client has read everything
// (the servers this change fixes, still in place on nodes whose runtime has
// not picked it up) used to make client-go return nil for a truncated stream.
// The client reports it now, directly over SPDY and through the WebSocket
// translator, which is a SPDY client itself.
func TestExecWithoutStatusIsAnError(t *testing.T) {
	const payload = 1 << 20
	const execPath = "/exec/pod/container?output=1&command=cmd"

	for _, tc := range []struct {
		name     string
		front    func(t *testing.T, backend string) http.Handler
		executor func(t *testing.T, front string) remoteclient.Executor
	}{
		{
			name: "spdy",
			front: func(t *testing.T, backend string) http.Handler {
				return proxy.NewUpgradeAwareHandler(parseURL(t, backend+execPath), nil, false, true, proxyResponder{t})
			},
			executor: func(t *testing.T, front string) remoteclient.Executor {
				rt, err := spdy.NewRoundTripper(nil)
				if err != nil {
					t.Fatal(err)
				}
				e, err := remoteclient.NewSPDYExecutorForTransports(rt, clientspdy.NewUpgraderForStreaming(rt), http.MethodPost, parseURL(t, front+execPath))
				if err != nil {
					t.Fatal(err)
				}
				return e
			},
		},
		{
			name: "websocket through the translator",
			front: func(t *testing.T, backend string) http.Handler {
				return translator.NewStreamTranslatorHandler(parseURL(t, backend+execPath), nil, 0, translator.Options{Stdout: true})
			},
			executor: func(t *testing.T, front string) remoteclient.Executor {
				e, err := remoteclient.NewWebSocketExecutor(&restclient.Config{Host: front}, http.MethodPost, front+execPath)
				if err != nil {
					t.Fatal(err)
				}
				return e
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend := httptest.NewServer(statuslessExecServer(t, payload))
			t.Cleanup(backend.Close)
			front := httptest.NewServer(tc.front(t, backend.URL))
			t.Cleanup(front.Close)

			stdout := &slowWriter{}
			err := tc.executor(t, front.URL).StreamWithContext(context.Background(), remoteclient.StreamOptions{Stdout: stdout})
			t.Logf("received %d of %d bytes, err %v", stdout.n, payload, err)
			if err == nil {
				t.Fatal("StreamWithContext returned nil for a session that ended without a status")
			}
			if !strings.Contains(err.Error(), "before the command's status was received") {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
