/*
Copyright 2023 The Kubernetes Authors.

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

package proxy

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"math"
	mrand "math/rand"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	rcconstants "k8s.io/apimachinery/pkg/util/remotecommand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/util/proxy/metrics"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/client-go/transport"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

// TestStreamTranslator_LoopbackStdinToStdout returns random data sent on the client's
// STDIN channel back onto the client's STDOUT channel.
func TestStreamTranslator_LoopbackStdinToStdout(t *testing.T) {
	metrics.ResetForTest()
	handlerPath := "/TestStreamTranslator_LoopbackStdinToStdout"

	// 1. Configure and register the upstream SPDY handler for this test
	spdyServerMux.HandleFunc(handlerPath, func(w http.ResponseWriter, req *http.Request) {
		opts := Options{Stdin: true, Stdout: true}
		ctx, err := createSPDYServerStreams(w, req, opts)
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.conn.Close()
		_, err = io.Copy(ctx.stdoutStream, ctx.stdinStream)
		if err != nil  {
			t.Errorf("error copying STDIN to STDOUT: %v", err)
		}
	})

	// 2. Configure and register the StreamTranslator handler for this test
	translatorOptions := Options{Stdin: true, Stdout: true}
	spdyTransport, err := fakeTransport()
	require.NoError(t, err)
	upstreamURL := *spdyServerURL
	upstreamURL.Path = handlerPath
	translatorHandler := NewStreamTranslatorHandler(&upstreamURL, spdyTransport, 0, translatorOptions)
	streamTranslatorServerMux.HandleFunc(handlerPath, translatorHandler.ServeHTTP)

	// 3. Configure the client to connect to the translator
	clientURL := *streamTranslatorServerURL
	clientURL.Path = handlerPath

	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: clientURL.Host}, "GET", clientURL.String())
	require.NoError(t, err)

	// 4. Execute the test logic
	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	_, err = rand.Read(randomData)
	require.NoError(t, err)

	var stdout bytes.Buffer
	options := &remotecommand.StreamOptions{
		Stdin:  bytes.NewReader(randomData),
		Stdout: &stdout,
	}
	err = exec.StreamWithContext(context.Background(), *options)
	require.NoError(t, err, "Internal error occurred")

	data, err := io.ReadAll(&stdout)
	require.NoError(t, err)
	if !bytes.Equal(randomData, data) {
		t.Errorf("unexpected data received: %d sent: %d", len(data), len(randomData))
	}

	// 5. Validate metrics
	metricName := "apiserver_stream_translator_requests_total"
	expected := `
# HELP apiserver_stream_translator_requests_total [ALPHA] Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5
# TYPE apiserver_stream_translator_requests_total counter
apiserver_stream_translator_requests_total{code="200"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricName); err != nil {
		t.Fatal(err)
	}
}

// TestStreamTranslator_LoopbackStdinToStderr returns random data sent on the client's
// STDIN channel back onto the client's STDERR channel.
func TestStreamTranslator_LoopbackStdinToStderr(t *testing.T) {
	metrics.ResetForTest()
	handlerPath := "/TestStreamTranslator_LoopbackStdinToStderr"

	spdyServerMux.HandleFunc(handlerPath, func(w http.ResponseWriter, req *http.Request) {
		opts := Options{Stdin: true, Stderr: true}
		ctx, err := createSPDYServerStreams(w, req, opts)
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.conn.Close()
		_, err = io.Copy(ctx.stderrStream, ctx.stdinStream)
		if err != nil  {
			t.Errorf("error copying STDIN to STDERR: %v", err)
		}
	})

	translatorOptions := Options{Stdin: true, Stderr: true}
	spdyTransport, err := fakeTransport()
	require.NoError(t, err)
	upstreamURL := *spdyServerURL
	upstreamURL.Path = handlerPath
	translatorHandler := NewStreamTranslatorHandler(&upstreamURL, spdyTransport, 0, translatorOptions)
	streamTranslatorServerMux.HandleFunc(handlerPath, translatorHandler.ServeHTTP)

	clientURL := *streamTranslatorServerURL
	clientURL.Path = handlerPath

	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: clientURL.Host}, "GET", clientURL.String())
	require.NoError(t, err)

	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	_, err = rand.Read(randomData)
	require.NoError(t, err)

	var stderr bytes.Buffer
	options := &remotecommand.StreamOptions{
		Stdin:  bytes.NewReader(randomData),
		Stderr: &stderr,
	}
	err = exec.StreamWithContext(context.Background(), *options)
	require.NoError(t, err)

	data, err := io.ReadAll(&stderr)
	require.NoError(t, err)
	if !bytes.Equal(randomData, data) {
		t.Errorf("unexpected data received: %d sent: %d", len(data), len(randomData))
	}

	metricName := "apiserver_stream_translator_requests_total"
	expected := `
# HELP apiserver_stream_translator_requests_total [ALPHA] Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5
# TYPE apiserver_stream_translator_requests_total counter
apiserver_stream_translator_requests_total{code="200"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricName); err != nil {
		t.Fatal(err)
	}
}

// Returns a random exit code in the range(1-127).
func randomExitCode() int {
	errorCode := mrand.Intn(127) // Range: (0 - 126)
	errorCode += 1               // Range: (1 - 127)
	return errorCode
}

// TestStreamTranslator_ErrorStream tests the error stream by sending an error with a random
// exit code, then validating the error arrives on the error stream.
func TestStreamTranslator_ErrorStream(t *testing.T) {
	metrics.ResetForTest()
	expectedExitCode := randomExitCode()
	handlerPath := "/TestStreamTranslator_ErrorStream"

	spdyServerMux.HandleFunc(handlerPath, func(w http.ResponseWriter, req *http.Request) {
		opts := Options{Stdin: true}
		ctx, err := createSPDYServerStreams(w, req, opts)
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.conn.Close()
		_, err = io.Copy(io.Discard, ctx.stdinStream)
		if err != nil  {
			t.Errorf("error copying STDIN to DISCARD: %v", err)
		}
		err = ctx.writeStatus(&apierrors.StatusError{ErrStatus: metav1.Status{
			Status: metav1.StatusFailure,
			Reason: rcconstants.NonZeroExitCodeReason,
			Details: &metav1.StatusDetails{
				Causes: []metav1.StatusCause{
					{
						Type:    rcconstants.ExitCodeCauseType,
						Message: fmt.Sprintf("%d", expectedExitCode),
					},
				},
			},
		}})
		if err != nil {
			t.Errorf("error writing status: %v", err)
		}
	})

	translatorOptions := Options{Stdin: true}
	spdyTransport, err := fakeTransport()
	require.NoError(t, err)
	upstreamURL := *spdyServerURL
	upstreamURL.Path = handlerPath
	translatorHandler := NewStreamTranslatorHandler(&upstreamURL, spdyTransport, 0, translatorOptions)
	streamTranslatorServerMux.HandleFunc(handlerPath, translatorHandler.ServeHTTP)

	clientURL := *streamTranslatorServerURL
	clientURL.Path = handlerPath

	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: clientURL.Host}, "GET", clientURL.String())
	require.NoError(t, err)

	options := &remotecommand.StreamOptions{
		Stdin: bytes.NewReader(make([]byte, 1024)),
	}
	err = exec.StreamWithContext(context.Background(), *options)
	if err == nil {
		t.Errorf("expected error, but received none")
	}
	expectedError := fmt.Sprintf("command terminated with exit code %d", expectedExitCode)
	if expectedError != err.Error() {
		t.Errorf("expected error (%s), got (%s)", expectedError, err)
	}

	metricName := "apiserver_stream_translator_requests_total"
	expected := `
# HELP apiserver_stream_translator_requests_total [ALPHA] Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5
# TYPE apiserver_stream_translator_requests_total counter
apiserver_stream_translator_requests_total{code="200"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricName); err != nil {
		t.Fatal(err)
	}
}

// TestStreamTranslator_MultipleReadChannels tests two streams (STDOUT, STDERR) reading from
// the connections at the same time.
func TestStreamTranslator_MultipleReadChannels(t *testing.T) {
	metrics.ResetForTest()
	handlerPath := "/TestStreamTranslator_MultipleReadChannels"

	spdyServerMux.HandleFunc(handlerPath, func(w http.ResponseWriter, req *http.Request) {
		opts := Options{Stdin: true, Stdout: true, Stderr: true}
		ctx, err := createSPDYServerStreams(w, req, opts)
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.conn.Close()
		stdinReader := io.TeeReader(ctx.stdinStream, ctx.stderrStream)
		_, err = io.Copy(ctx.stdoutStream, stdinReader)
		if err != nil  {
			t.Errorf("error copying STDIN to STDOUT: %v", err)
		}
	})

	translatorOptions := Options{Stdin: true, Stdout: true, Stderr: true}
	spdyTransport, err := fakeTransport()
	require.NoError(t, err)
	upstreamURL := *spdyServerURL
	upstreamURL.Path = handlerPath
	translatorHandler := NewStreamTranslatorHandler(&upstreamURL, spdyTransport, 0, translatorOptions)
	streamTranslatorServerMux.HandleFunc(handlerPath, translatorHandler.ServeHTTP)

	clientURL := *streamTranslatorServerURL
	clientURL.Path = handlerPath

	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: clientURL.Host}, "GET", clientURL.String())
	require.NoError(t, err)

	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	_, err = rand.Read(randomData)
	require.NoError(t, err)

	var stdout, stderr bytes.Buffer
	options := &remotecommand.StreamOptions{
		Stdin:  bytes.NewReader(randomData),
		Stdout: &stdout,
		Stderr: &stderr,
	}
	err = exec.StreamWithContext(context.Background(), *options)
	require.NoError(t, err)

	stdoutBytes, err := io.ReadAll(&stdout)
	require.NoError(t, err)
	if !bytes.Equal(stdoutBytes, randomData) {
		t.Errorf("unexpected data received: %d sent: %d", len(stdoutBytes), len(randomData))
	}
	stderrBytes, err := io.ReadAll(&stderr)
	require.NoError(t, err)
	if !bytes.Equal(stderrBytes, randomData) {
		t.Errorf("unexpected data received: %d sent: %d", len(stderrBytes), len(randomData))
	}

	metricName := "apiserver_stream_translator_requests_total"
	expected := `
# HELP apiserver_stream_translator_requests_total [ALPHA] Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5
# TYPE apiserver_stream_translator_requests_total counter
apiserver_stream_translator_requests_total{code="200"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricName); err != nil {
		t.Fatal(err)
	}
}

// TestStreamTranslator_ThrottleReadChannels tests two streams (STDOUT, STDERR) using rate limited streams.
func TestStreamTranslator_ThrottleReadChannels(t *testing.T) {
	t.Parallel()
	handlerPath := "/TestStreamTranslator_ThrottleReadChannels"

	spdyServerMux.HandleFunc(handlerPath, func(w http.ResponseWriter, req *http.Request) {
		opts := Options{Stdin: true, Stdout: true, Stderr: true}
		ctx, err := createSPDYServerStreams(w, req, opts)
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.conn.Close()
		stdinReader := io.TeeReader(ctx.stdinStream, ctx.stderrStream)
		_, err = io.Copy(ctx.stdoutStream, stdinReader)
		if err != nil  {
			t.Errorf("error copying STDIN to STDOUT: %v", err)
		}
	})

	translatorOptions := Options{Stdin: true, Stdout: true, Stderr: true}
	spdyTransport, err := fakeTransport()
	require.NoError(t, err)
	upstreamURL := *spdyServerURL
	upstreamURL.Path = handlerPath
	// Set a throttle limit just below the total data size to ensure throttling is exercised.
	translatorHandler := NewStreamTranslatorHandler(&upstreamURL, spdyTransport, 900*1024, translatorOptions)
	streamTranslatorServerMux.HandleFunc(handlerPath, translatorHandler.ServeHTTP)

	clientURL := *streamTranslatorServerURL
	clientURL.Path = handlerPath

	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: clientURL.Host}, "GET", clientURL.String())
	require.NoError(t, err)

	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	_, err = rand.Read(randomData)
	require.NoError(t, err)

	var stdout, stderr bytes.Buffer
	options := &remotecommand.StreamOptions{
		Stdin:  bytes.NewReader(randomData),
		Stdout: &stdout,
		Stderr: &stderr,
	}
	err = exec.StreamWithContext(context.Background(), *options)
	require.NoError(t, err)

	stdoutBytes, err := io.ReadAll(&stdout)
	require.NoError(t, err)
	if !bytes.Equal(stdoutBytes, randomData) {
		t.Errorf("unexpected data received: %d sent: %d", len(stdoutBytes), len(randomData))
	}
	stderrBytes, err := io.ReadAll(&stderr)
	require.NoError(t, err)
	if !bytes.Equal(stderrBytes, randomData) {
		t.Errorf("unexpected data received: %d sent: %d", len(stderrBytes), len(randomData))
	}
}

// fakeTerminalSizeQueue implements TerminalSizeQueue, returning a random set of
// "maxSizes" number of TerminalSizes, storing the TerminalSizes in "sizes" slice.
type fakeTerminalSizeQueue struct {
	maxSizes      int
	terminalSizes []remotecommand.TerminalSize
}

// newTerminalSizeQueue returns a pointer to a fakeTerminalSizeQueue passing
// "max" number of random TerminalSizes created.
func newTerminalSizeQueue(max int) *fakeTerminalSizeQueue {
	return &fakeTerminalSizeQueue{
		maxSizes:      max,
		terminalSizes: make([]remotecommand.TerminalSize, 0, max),
	}
}

// Next returns a pointer to the next random TerminalSize, or nil if we have
// already returned "maxSizes" TerminalSizes already. Stores the randomly
// created TerminalSize in "terminalSizes" field for later validation.
func (f *fakeTerminalSizeQueue) Next() *remotecommand.TerminalSize {
	if len(f.terminalSizes) >= f.maxSizes {
		return nil
	}
	size := randomTerminalSize()
	f.terminalSizes = append(f.terminalSizes, size)
	return &size
}

// randomTerminalSize returns a TerminalSize with random values in the
// range (0-65535) for the fields Width and Height.
func randomTerminalSize() remotecommand.TerminalSize {
	randWidth := uint16(mrand.Intn(int(math.Pow(2, 16))))
	randHeight := uint16(mrand.Intn(int(math.Pow(2, 16))))
	return remotecommand.TerminalSize{
		Width:  randWidth,
		Height: randHeight,
	}
}

func TestStreamTranslator_TTYResizeChannel(t *testing.T) {
	metrics.ResetForTest()
	handlerPath := "/TestStreamTranslator_TTYResizeChannel"

	numSizeQueue := 10000
	sizeQueue := newTerminalSizeQueue(numSizeQueue)
	actualTerminalSizes := make([]remotecommand.TerminalSize, 0, numSizeQueue)
	// The WaitGroup is used to synchronize the test. It ensures that the server
	// has processed all expected resize events before the test's main goroutine
	// proceeds to validation and teardown. This prevents a race condition where
	// the test could finish before all events are handled.
	var wg sync.WaitGroup
	wg.Add(numSizeQueue)

	spdyServerMux.HandleFunc(handlerPath, func(w http.ResponseWriter, req *http.Request) {
		opts := Options{Tty: true}
		ctx, err := createSPDYServerStreams(w, req, opts)
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.conn.Close()
		for i := 0; i < numSizeQueue; i++ {
			actualTerminalSize, ok := <-ctx.resizeChan
			if !ok {
				break
			}
			actualTerminalSizes = append(actualTerminalSizes, actualTerminalSize)
			// Signal that one resize event has been successfully processed by the server.
			wg.Done()
		}
	})

	translatorOptions := Options{Tty: true}
	spdyTransport, err := fakeTransport()
	require.NoError(t, err)
	upstreamURL := *spdyServerURL
	upstreamURL.Path = handlerPath
	translatorHandler := NewStreamTranslatorHandler(&upstreamURL, spdyTransport, 0, translatorOptions)
	streamTranslatorServerMux.HandleFunc(handlerPath, translatorHandler.ServeHTTP)

	clientURL := *streamTranslatorServerURL
	clientURL.Path = handlerPath

	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: clientURL.Host}, "GET", clientURL.String())
	require.NoError(t, err)

	options := &remotecommand.StreamOptions{
		Tty:               true,
		TerminalSizeQueue: sizeQueue,
	}

	errorChan := make(chan error, 1)
	go func() {
		// The client's stream execution is run in a separate goroutine. This is
		// necessary because the StreamWithContext call is blocking, but we need
		// the main test goroutine to be free to wait on the WaitGroup.
		errorChan <- exec.StreamWithContext(context.Background(), *options)
	}()

	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timed out waiting for client stream to complete")
	case err := <-errorChan:
		require.NoError(t, err)
	}

	// Block until the WaitGroup counter is zero, which signifies that the
	// server-side handler has received and processed all 10,000 resize events.
	// This is the critical synchronization point that prevents the test from
	// validating results prematurely.
	wg.Wait()

	if len(actualTerminalSizes) != numSizeQueue {
		t.Fatalf("expected to receive num terminal resizes (%d), got (%d)",
			numSizeQueue, len(actualTerminalSizes))
	}
	for i, actual := range actualTerminalSizes {
		expected := sizeQueue.terminalSizes[i]
		if !reflect.DeepEqual(expected, actual) {
			t.Errorf("expected terminal resize window %v, got %v", expected, actual)
		}
	}

	metricName := "apiserver_stream_translator_requests_total"
	expected := `
# HELP apiserver_stream_translator_requests_total [ALPHA] Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5
# TYPE apiserver_stream_translator_requests_total counter
apiserver_stream_translator_requests_total{code="200"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricName); err != nil {
		t.Fatal(err)
	}
}

// TestStreamTranslator_WebSocketServerErrors validates that when there is a problem creating
// the websocket server as the first step of the StreamTranslator an error is properly returned.
func TestStreamTranslator_WebSocketServerErrors(t *testing.T) {
	t.Parallel()
	handlerPath := "/TestStreamTranslator_WebSocketServerErrors"

	// Register a real translator handler that will fail the websocket handshake.
	spdyTransport, err := fakeTransport()
	require.NoError(t, err)
	// The upstream location is irrelevant as the handshake will fail before it's used.
	dummyURL, _ := url.Parse("http://localhost:12345")
	translatorHandler := NewStreamTranslatorHandler(dummyURL, spdyTransport, 0, Options{})
	streamTranslatorServerMux.HandleFunc(handlerPath, translatorHandler.ServeHTTP)

	clientURL := *streamTranslatorServerURL
	clientURL.Path = handlerPath

	exec, err := remotecommand.NewWebSocketExecutorForProtocols(
		&rest.Config{Host: clientURL.Host},
		"GET",
		clientURL.String(),
		rcconstants.StreamProtocolV4Name, // RemoteCommand V4 protocol is unsupported by the translator
	)
	require.NoError(t, err)

	err = exec.StreamWithContext(context.Background(), remotecommand.StreamOptions{})
	if err == nil {
		t.Fatalf("expected error, but received none")
	}
	if !strings.Contains(err.Error(), "unable to upgrade streaming request") {
		t.Errorf("expected websocket bad handshake error, got (%s)", err)
	}
}

// TestStreamTranslator_BlockRedirects verifies that the StreamTranslator will *not* follow
// redirects; it will thrown an error instead.
func TestStreamTranslator_BlockRedirects(t *testing.T) {
	for _, statusCode := range []int{
		http.StatusMovedPermanently,  // 301
		http.StatusFound,             // 302
		http.StatusSeeOther,          // 303
		http.StatusTemporaryRedirect, // 307
		http.StatusPermanentRedirect, // 308
	} {
		statusCode := statusCode
		t.Run(fmt.Sprintf("statusCode=%d", statusCode), func(t *testing.T) {
			metrics.ResetForTest()
			handlerPath := fmt.Sprintf("/TestStreamTranslator_BlockRedirects/%d", statusCode)

			spdyServerMux.HandleFunc(handlerPath, func(w http.ResponseWriter, req *http.Request) {
				w.Header().Set("Location", "/")
				w.WriteHeader(statusCode)
			})

			translatorOptions := Options{Stdout: true}
			spdyTransport, err := fakeTransport()
			require.NoError(t, err)
			upstreamURL := *spdyServerURL
			upstreamURL.Path = handlerPath
			translatorHandler := NewStreamTranslatorHandler(&upstreamURL, spdyTransport, 0, translatorOptions)
			streamTranslatorServerMux.HandleFunc(handlerPath, translatorHandler.ServeHTTP)

			clientURL := *streamTranslatorServerURL
			clientURL.Path = handlerPath

			exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: clientURL.Host}, "GET", clientURL.String())
			require.NoError(t, err)

			err = exec.StreamWithContext(context.Background(), remotecommand.StreamOptions{})
			if err == nil {
				t.Fatalf("expected error, but received none")
			}
			if !strings.Contains(err.Error(), "redirect not allowed") {
				t.Errorf("expected redirect not allowed error, got (%s)", err)
			}

			metricName := "apiserver_stream_translator_requests_total"
			expected := `
# HELP apiserver_stream_translator_requests_total [ALPHA] Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5
# TYPE apiserver_stream_translator_requests_total counter
apiserver_stream_translator_requests_total{code="500"} 1
`
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricName); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// streamContext encapsulates the structures necessary to communicate through
// a SPDY connection, including the Reader/Writer streams.
type streamContext struct {
	conn         io.Closer
	stdinStream  io.ReadCloser
	stdoutStream io.WriteCloser
	stderrStream io.WriteCloser
	resizeStream io.ReadCloser
	resizeChan   chan remotecommand.TerminalSize
	writeStatus  func(status *apierrors.StatusError) error
}

type streamAndReply struct {
	httpstream.Stream
	replySent <-chan struct{}
}

// CreateSPDYServerStreams upgrades the passed HTTP request to a SPDY bi-directional streaming
// connection with remote command streams defined in passed options. Returns a streamContext
// structure containing the Reader/Writer streams to communicate through the SDPY connection.
// Returns an error if unable to upgrade the HTTP connection to a SPDY connection.
func createSPDYServerStreams(w http.ResponseWriter, req *http.Request, opts Options) (*streamContext, error) {
	_, err := httpstream.Handshake(req, w, []string{rcconstants.StreamProtocolV4Name})
	if err != nil {
		return nil, err
	}

	upgrader := spdy.NewResponseUpgrader()
	streamCh := make(chan streamAndReply)
	conn := upgrader.UpgradeResponse(w, req, func(stream httpstream.Stream, replySent <-chan struct{}) error {
		streamCh <- streamAndReply{Stream: stream, replySent: replySent}
		return nil
	})
	ctx := &streamContext{
		conn: conn,
	}

	// wait for stream
	replyChan := make(chan struct{}, 5)
	defer close(replyChan)
	receivedStreams := 0
	expectedStreams := 1 // expect at least the error stream
	if opts.Stdout {
		expectedStreams++
	}
	if opts.Stdin {
		expectedStreams++
	}
	if opts.Stderr {
		expectedStreams++
	}
	if opts.Tty {
		expectedStreams++
	}
WaitForStreams:
	for {
		select {
		case stream := <-streamCh:
			streamType := stream.Headers().Get(v1.StreamType)
			streamHandled := false
			switch streamType {
			case v1.StreamTypeError:
				ctx.writeStatus = v4WriteStatusFunc(stream)
				streamHandled = true
			case v1.StreamTypeStdout:
				if opts.Stdout {
					ctx.stdoutStream = stream
					streamHandled = true
				}
			case v1.StreamTypeStdin:
				if opts.Stdin {
					ctx.stdinStream = stream
					streamHandled = true
				}
			case v1.StreamTypeStderr:
				if opts.Stderr {
					ctx.stderrStream = stream
					streamHandled = true
				}
			case v1.StreamTypeResize:
				if opts.Tty {
					ctx.resizeStream = stream
					streamHandled = true
				}
			}

			if streamHandled {
				replyChan <- struct{}{}
			} else {
				// This is a known but unexpected stream type, or an unknown one.
				// We must reset it to signal the client we won't be using it.
				stream.Reset()
			}
		case <-replyChan:
			receivedStreams++
			if receivedStreams == expectedStreams {
				break WaitForStreams
			}
		}
	}

	if ctx.resizeStream != nil {
		ctx.resizeChan = make(chan remotecommand.TerminalSize)
		go handleResizeEvents(req.Context(), ctx.resizeStream, ctx.resizeChan)
	}

	return ctx, nil
}

func v4WriteStatusFunc(stream io.Writer) func(status *apierrors.StatusError) error {
	return func(status *apierrors.StatusError) error {
		bs, err := json.Marshal(status.Status())
		if err != nil {
			return err
		}
		_, err = stream.Write(bs)
		return err
	}
}

func fakeTransport() (*http.Transport, error) {
	cfg := &transport.Config{
		TLS: transport.TLSConfig{
			Insecure: true,
			CAFile:   "",
		},
	}
	rt, err := transport.New(cfg)
	if err != nil {
		return nil, err
	}
	t, ok := rt.(*http.Transport)
	if !ok {
		return nil, fmt.Errorf("unknown transport type: %T", rt)
	}
	return t, nil
}
