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

package remotecommand

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	gwebsocket "github.com/gorilla/websocket"

	"k8s.io/streaming/pkg/httpstream"
	"k8s.io/streaming/pkg/httpstream/spdy"
)

const (
	// slowPayload is what the command writes in the slow-reader tests: well
	// above the send buffer of a socket, so that output is still in flight when
	// the command finishes.
	slowPayload = 16 << 20
	// slowReadPause is the pause after every 64 KiB the slow reader takes,
	// about 16 MiB/s.
	slowReadPause = 4 * time.Millisecond
	// fastPings is how often the clients in these tests ping. client-go pings
	// every 5s; pinging often makes the outcome independent of where in the
	// transfer a ping falls.
	fastPings = 5 * time.Millisecond
)

// payloadExecutor writes size bytes to stdout in 64 KiB chunks and returns
// without reading stdin.
type payloadExecutor struct {
	size int
}

func (e *payloadExecutor) ExecInContainer(_ context.Context, _ string, _ string, _ string, _ []string, _ io.Reader, out, _ io.WriteCloser, _ bool, _ <-chan TerminalSize, _ time.Duration) error {
	return e.write(out)
}

func (e *payloadExecutor) AttachContainer(_ context.Context, _ string, _ string, _ string, _ io.Reader, out, _ io.WriteCloser, _ bool, _ <-chan TerminalSize) error {
	return e.write(out)
}

func (e *payloadExecutor) write(out io.Writer) error {
	chunk := make([]byte, 64<<10)
	for i := range chunk {
		chunk[i] = byte(i)
	}
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

// streamServer serves exec (or attach) requests with the given executor and
// idle timeout. done is closed when the handler returns.
func streamServer(t *testing.T, executor *payloadExecutor, idleTimeout time.Duration, attach bool) (*httptest.Server, <-chan struct{}) {
	t.Helper()
	done := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer close(done)
		opts, err := NewOptions(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if attach {
			ServeAttach(w, req, executor, "pod", "uid", "container", opts, idleTimeout, DefaultStreamCreationTimeout, SupportedStreamingProtocols)
			return
		}
		ServeExec(w, req, executor, "pod", "uid", "container", []string{"cmd"}, opts, idleTimeout, DefaultStreamCreationTimeout, SupportedStreamingProtocols)
	}))
	t.Cleanup(server.Close)
	return server, done
}

func execServer(t *testing.T, executor *payloadExecutor, idleTimeout time.Duration) (*httptest.Server, <-chan struct{}) {
	return streamServer(t, executor, idleTimeout, false)
}

// spdyExecClient opens a session over SPDY the way client-go does, with the
// given ping period, and creates the error and stdout streams, plus stdin if
// requested.
func spdyExecClient(t *testing.T, url string, pingPeriod time.Duration, stdin bool) (conn httpstream.Connection, stdinStream, stdout, errorStream httpstream.Stream) {
	t.Helper()
	rt, err := spdy.NewRoundTripperWithConfig(spdy.RoundTripperConfig{PingPeriod: pingPeriod})
	if err != nil {
		t.Fatal(err)
	}
	rt.Dialer = smallReceiveBufferDialer()
	req, err := http.NewRequest(http.MethodPost, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set(httpstream.HeaderProtocolVersion, StreamProtocolV4Name)
	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	conn, err = rt.NewConnection(resp)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	headers := http.Header{}
	headers.Set(StreamType, StreamTypeError)
	if errorStream, err = conn.CreateStream(headers); err != nil {
		t.Fatal(err)
	}
	if stdin {
		headers.Set(StreamType, StreamTypeStdin)
		if stdinStream, err = conn.CreateStream(headers); err != nil {
			t.Fatal(err)
		}
	}
	headers.Set(StreamType, StreamTypeStdout)
	if stdout, err = conn.CreateStream(headers); err != nil {
		t.Fatal(err)
	}
	return conn, stdinStream, stdout, errorStream
}

// readSlowly reads r to EOF, pausing after every read, and returns the byte count.
func readSlowly(t *testing.T, r io.Reader, pause time.Duration) int {
	t.Helper()
	got := 0
	buf := make([]byte, 64<<10)
	for {
		n, err := r.Read(buf)
		got += n
		if err == io.EOF {
			return got
		}
		if err != nil {
			t.Fatalf("read error after %d bytes: %v", got, err)
		}
		time.Sleep(pause)
	}
}

func readStatus(t *testing.T, errorStream io.Reader) chan []byte {
	t.Helper()
	statusCh := make(chan []byte, 1)
	go func() {
		bs, _ := io.ReadAll(errorStream)
		statusCh <- bs
	}()
	return statusCh
}

func waitFor(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(30 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}

func expectSuccess(t *testing.T, status []byte) {
	t.Helper()
	var st streamStatus
	if err := json.Unmarshal(status, &st); err != nil {
		t.Errorf("error stream did not carry a status: %v (%q)", err, status)
	} else if st.Status != statusSuccess {
		t.Errorf("unexpected status: %+v", st)
	}
}

func closedChan(ch <-chan bool) <-chan struct{} {
	done := make(chan struct{})
	go func() {
		<-ch
		close(done)
	}()
	return done
}

// trickle yields one byte every period until stopped, like a user typing into
// an interactive session while output is still arriving. Unlike a SPDY ping,
// which waits for the previous pong to come back through the backlog, it keeps
// the client sending frames while the server drains.
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

// startTrickle copies a trickle of input into w until the returned stop
// function is called, which also waits for the copy to end.
func startTrickle(w io.Writer, period time.Duration) (stop func()) {
	r := &trickle{stop: make(chan struct{}), period: period}
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = io.Copy(w, r)
	}()
	return func() {
		close(r.stop)
		<-done
	}
}

// A client that reads more slowly than the command writes must still receive
// everything the command wrote, and the status. Closing the connection right
// after the status discarded the output the kernel had not delivered yet as
// soon as the client sent its next frame.
func TestServeExecSlowClientReceivesAllOutput(t *testing.T) {
	for _, tc := range []struct {
		name       string
		pingPeriod time.Duration
		input      bool
	}{
		{name: "client pings", pingPeriod: fastPings},
		{name: "client sends input", input: true},
		{name: "client sends nothing", pingPeriod: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server, serverDone := execServer(t, &payloadExecutor{size: slowPayload}, time.Minute)
			path := "/exec?output=1"
			if tc.input {
				path += "&input=1"
			}
			conn, stdin, stdout, errorStream := spdyExecClient(t, server.URL+path, tc.pingPeriod, tc.input)
			statusCh := readStatus(t, errorStream)
			if tc.input {
				defer startTrickle(stdin, fastPings)()
			}

			start := time.Now()
			got := readSlowly(t, stdout, slowReadPause)
			elapsed := time.Since(start)
			status := <-statusCh

			// The client closes the connection once it has read everything,
			// like client-go does.
			_ = conn.Close()
			waitFor(t, serverDone, "ServeExec to return after the client closed the connection")

			t.Logf("received %d of %d bytes in %v, status %q", got, slowPayload, elapsed.Round(time.Millisecond), status)
			if got != slowPayload {
				t.Errorf("stdout truncated: got %d bytes, want %d (lost %d)", got, slowPayload, slowPayload-got)
			}
			expectSuccess(t, status)
		})
	}
}

// The server must not close the connection before the client does, for exec
// and for attach.
func TestServeWaitsForClientToClose(t *testing.T) {
	for _, tc := range []struct {
		name   string
		attach bool
		path   string
	}{
		{name: "exec", path: "/exec?output=1"},
		{name: "attach", attach: true, path: "/attach?output=1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server, serverDone := streamServer(t, &payloadExecutor{size: 1 << 20}, time.Minute, tc.attach)
			conn, _, stdout, errorStream := spdyExecClient(t, server.URL+tc.path, 0, false)
			statusCh := readStatus(t, errorStream)

			got := readSlowly(t, stdout, 0)
			status := <-statusCh
			if got != 1<<20 {
				t.Errorf("got %d bytes, want %d", got, 1<<20)
			}
			expectSuccess(t, status)

			// The client has all the output and the status, but has not closed
			// the connection yet. The server must still be holding it open.
			select {
			case <-conn.CloseChan():
				t.Fatal("the server closed the connection before the client did")
			case <-serverDone:
				t.Fatal("the handler returned before the client closed the connection")
			case <-time.After(300 * time.Millisecond):
			}

			_ = conn.Close()
			waitFor(t, serverDone, "the handler to return after the client closed the connection")
		})
	}
}

// A client that goes quiet without closing the connection holds the handler
// only until the idle timeout expires.
func TestServeExecIdleTimeoutBoundsWaitForClient(t *testing.T) {
	const idleTimeout = 300 * time.Millisecond
	server, serverDone := execServer(t, &payloadExecutor{size: 1 << 20}, idleTimeout)
	// The server's timers start after the request arrives, so the handler
	// cannot return before start+idleTimeout; a server that closed as soon as
	// the command finished would return well before that.
	start := time.Now()
	conn, _, stdout, errorStream := spdyExecClient(t, server.URL+"/exec?output=1", 0, false)
	statusCh := readStatus(t, errorStream)

	got := readSlowly(t, stdout, 0)
	<-statusCh
	if got != 1<<20 {
		t.Errorf("got %d bytes, want %d", got, 1<<20)
	}

	waitFor(t, serverDone, "ServeExec to return after the idle timeout")
	elapsed := time.Since(start)
	if elapsed < idleTimeout {
		t.Errorf("ServeExec returned %v after the session started, before the idle timeout %v: it did not wait for the client", elapsed, idleTimeout)
	}
	if elapsed > 10*idleTimeout {
		t.Errorf("ServeExec returned %v after the session started, idle timeout is %v", elapsed, idleTimeout)
	}
	// The connection is closed for the client as well.
	waitFor(t, closedChan(conn.CloseChan()), "the client connection to be closed")
}

// A client that keeps pinging but never closes the connection holds the
// handler until finishTimeout: the idle timeout, configured here at a third
// of it, does not bound the wait on its own because the pings reset the SPDY
// idle timer. A silent client would be cut off by the idle timeout instead
// and make the elapsed-time check below fail.
func TestServeExecFinishTimeoutBoundsWaitForClient(t *testing.T) {
	const timeout = 300 * time.Millisecond
	previous := finishTimeout
	finishTimeout = timeout
	t.Cleanup(func() { finishTimeout = previous })

	server, serverDone := execServer(t, &payloadExecutor{size: 1 << 20}, timeout/3)
	start := time.Now()
	conn, _, stdout, errorStream := spdyExecClient(t, server.URL+"/exec?output=1", 20*time.Millisecond, false)
	statusCh := readStatus(t, errorStream)

	got := readSlowly(t, stdout, 0)
	<-statusCh
	if got != 1<<20 {
		t.Errorf("got %d bytes, want %d", got, 1<<20)
	}

	waitFor(t, serverDone, "ServeExec to return after finishTimeout")
	elapsed := time.Since(start)
	if elapsed < timeout {
		t.Errorf("ServeExec returned %v after the session started, before finishTimeout %v: it did not wait for the client", elapsed, timeout)
	}
	if elapsed > 10*timeout {
		t.Errorf("ServeExec returned %v after the session started, finishTimeout is %v", elapsed, timeout)
	}
	waitFor(t, closedChan(conn.CloseChan()), "the client connection to be closed")
}

// Input the client keeps sending after the command has exited is discarded
// and does not stall the connection.
func TestServeExecDiscardsInputAfterCommandExit(t *testing.T) {
	server, serverDone := execServer(t, &payloadExecutor{size: 1 << 20}, time.Minute)
	conn, stdin, stdout, errorStream := spdyExecClient(t, server.URL+"/exec?input=1&output=1", 0, true)
	statusCh := readStatus(t, errorStream)

	// Keep writing to stdin, which the command never reads, until the test ends.
	stop := make(chan struct{})
	var writer sync.WaitGroup
	writer.Go(func() {
		chunk := make([]byte, 64<<10)
		for {
			select {
			case <-stop:
				return
			default:
			}
			if _, err := stdin.Write(chunk); err != nil {
				return
			}
		}
	})

	got := readSlowly(t, stdout, 0)
	status := <-statusCh
	if got != 1<<20 {
		t.Errorf("got %d bytes, want %d", got, 1<<20)
	}
	expectSuccess(t, status)

	select {
	case <-serverDone:
		t.Fatal("ServeExec returned before the client closed the connection")
	case <-time.After(300 * time.Millisecond):
	}

	_ = conn.Close()
	close(stop)
	writer.Wait()
	waitFor(t, serverDone, "ServeExec to return after the client closed the connection")
}

// wsExecSession runs an exec session over WebSocket with a client that reads
// stdout slowly and pings the server like client-go does. With floodStdin it
// also keeps writing to stdin until the session ends. It returns the number of
// stdout bytes received, the status and the read error that ended the session.
func wsExecSession(t *testing.T, url string, pause time.Duration, floodStdin bool) (got int, status []byte, readErr error) {
	t.Helper()
	dialer := gwebsocket.Dialer{
		Subprotocols:   []string{v4BinaryWebsocketProtocol},
		NetDialContext: smallReceiveBufferDialer().DialContext,
	}
	conn, _, err := dialer.Dial("ws"+strings.TrimPrefix(url, "http"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = conn.Close() }()
	if conn.Subprotocol() != v4BinaryWebsocketProtocol {
		t.Fatalf("negotiated %q, want %q", conn.Subprotocol(), v4BinaryWebsocketProtocol)
	}

	// On return: stop the writers, close the connection so that a writer blocked
	// in a send returns, then wait for them.
	stop := make(chan struct{})
	var writers sync.WaitGroup
	defer writers.Wait()
	defer func() { _ = conn.Close() }()
	defer close(stop)
	writers.Go(func() {
		ticker := time.NewTicker(fastPings)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				_ = conn.WriteControl(gwebsocket.PingMessage, nil, time.Now().Add(time.Second))
			}
		}
	})
	if floodStdin {
		writers.Go(func() {
			message := make([]byte, 1+64<<10)
			message[0] = stdinChannel
			for {
				select {
				case <-stop:
					return
				default:
				}
				if err := conn.WriteMessage(gwebsocket.BinaryMessage, message); err != nil {
					return
				}
			}
		})
	}

	for {
		_, r, err := conn.NextReader()
		if err != nil {
			var closeErr *gwebsocket.CloseError
			if errors.As(err, &closeErr) && closeErr.Code == gwebsocket.CloseNormalClosure {
				return got, status, nil
			}
			return got, status, err
		}
		var id [1]byte
		if _, err := io.ReadFull(r, id[:]); err != nil {
			t.Fatalf("read channel id: %v", err)
		}
		data, err := io.ReadAll(r)
		if err != nil {
			t.Fatalf("read message: %v", err)
		}
		switch id[0] {
		case stdoutChannel:
			got += len(data)
			time.Sleep(pause)
		case errorChannel:
			status = append(status, data...)
		}
	}
}

func TestServeExecWebSocketSlowClientReceivesAllOutput(t *testing.T) {
	server, serverDone := execServer(t, &payloadExecutor{size: slowPayload}, time.Minute)

	start := time.Now()
	got, status, err := wsExecSession(t, server.URL+"/exec?output=1", slowReadPause, false)
	elapsed := time.Since(start)
	waitFor(t, serverDone, "ServeExec to return after the client closed the connection")

	t.Logf("received %d of %d bytes in %v, status %q, err %v", got, slowPayload, elapsed.Round(time.Millisecond), status, err)
	if err != nil {
		t.Errorf("session did not end with a normal closure: %v", err)
	}
	if got != slowPayload {
		t.Errorf("stdout truncated: got %d bytes, want %d (lost %d)", got, slowPayload, slowPayload-got)
	}
	expectSuccess(t, status)
}

func TestServeExecWebSocketDiscardsInputAfterCommandExit(t *testing.T) {
	server, serverDone := execServer(t, &payloadExecutor{size: 1 << 20}, time.Minute)

	got, status, err := wsExecSession(t, server.URL+"/exec?input=1&output=1", 0, true)
	waitFor(t, serverDone, "ServeExec to return after the client closed the connection")

	if err != nil {
		t.Errorf("session did not end with a normal closure: %v", err)
	}
	if got != 1<<20 {
		t.Errorf("got %d bytes, want %d", got, 1<<20)
	}
	expectSuccess(t, status)
}
