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
	"errors"
	"fmt"
	"io"
	"math"
	mrand "math/rand"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"golang.org/x/net/websocket"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	rcconstants "k8s.io/apimachinery/pkg/util/remotecommand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/util/proxy/metrics"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/client-go/transport"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/streaming/pkg/httpstream"
	"k8s.io/streaming/pkg/httpstream/spdy"
)

// TestStreamTranslator_LoopbackStdinToStdout returns random data sent on the client's
// STDIN channel back onto the client's STDOUT channel. There are two servers in this test: the
// upstream fake SPDY server, and the StreamTranslator server. The StreamTranslator proxys the
// data received from the websocket client upstream to the SPDY server (by translating the
// websocket data into spdy). The returned data read on the websocket client STDOUT is then
// compared the random data sent on STDIN to ensure they are the same.
func TestStreamTranslator_LoopbackStdinToStdout(t *testing.T) {
	metrics.Register()
	metrics.ResetForTest()
	t.Cleanup(metrics.ResetForTest)
	// Create upstream fake SPDY server which copies STDIN back onto STDOUT stream.
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, err := createSPDYServerStreams(t, w, req, Options{
			Stdin:  true,
			Stdout: true,
		})
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.conn.Close()
		// Loopback STDIN data onto STDOUT stream.
		_, err = io.Copy(ctx.stdoutStream, ctx.stdinStream)
		if err != nil {
			t.Fatalf("error copying STDIN to STDOUT: %v", err)
		}
		writeSuccessStatus(t, ctx.writeStatus)
	}))
	defer spdyServer.Close()
	// Create StreamTranslatorHandler, which points upstream to fake SPDY server with
	// streams STDIN and STDOUT. Create test server from StreamTranslatorHandler.
	spdyLocation, err := url.Parse(spdyServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse spdy server URL: %s", spdyServer.URL)
	}
	spdyTransport, err := fakeTransport()
	if err != nil {
		t.Fatalf("Unexpected error creating transport: %v", err)
	}
	streams := Options{Stdin: true, Stdout: true}
	streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, 0, streams)
	streamTranslatorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		streamTranslator.ServeHTTP(w, req)
	}))
	defer streamTranslatorServer.Close()
	// Now create the websocket client (executor), and point it to the "streamTranslatorServer".
	streamTranslatorLocation, err := url.Parse(streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse StreamTranslator server URL: %s", streamTranslatorServer.URL)
	}
	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: streamTranslatorLocation.Host}, "GET", streamTranslatorServer.URL)
	if err != nil {
		t.Errorf("unexpected error creating websocket executor: %v", err)
	}
	// Generate random data, and set it up to stream on STDIN. The data will be
	// returned on the STDOUT buffer.
	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	if _, err := rand.Read(randomData); err != nil {
		t.Errorf("unexpected error reading random data: %v", err)
	}
	var stdout bytes.Buffer
	options := &remotecommand.StreamOptions{
		Stdin:  bytes.NewReader(randomData),
		Stdout: &stdout,
	}
	errorChan := make(chan error)
	go func() {
		// Start the streaming on the WebSocket "exec" client.
		errorChan <- exec.StreamWithContext(context.Background(), *options)
	}()

	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("expect stream to be closed after connection is closed.")
	case err := <-errorChan:
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
	data, err := io.ReadAll(bytes.NewReader(stdout.Bytes()))
	if err != nil {
		t.Errorf("error reading the stream: %v", err)
		return
	}
	// Check the random data sent on STDIN was the same returned on STDOUT.
	if !bytes.Equal(randomData, data) {
		t.Errorf("unexpected data received: %d sent: %d", len(data), len(randomData))
	}
	// Validate the streamtranslator metrics; should be one 200 success.
	metricNames := []string{"apiserver_stream_translator_requests_total"}
	expected := `
# HELP apiserver_stream_translator_requests_total [BETA] Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5
# TYPE apiserver_stream_translator_requests_total counter
apiserver_stream_translator_requests_total{code="200"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...); err != nil {
		t.Fatal(err)
	}
}

// TestStreamTranslator_LoopbackStdinToStderr returns random data sent on the client's
// STDIN channel back onto the client's STDERR channel. There are two servers in this test: the
// upstream fake SPDY server, and the StreamTranslator server. The StreamTranslator proxys the
// data received from the websocket client upstream to the SPDY server (by translating the
// websocket data into spdy). The returned data read on the websocket client STDERR is then
// compared the random data sent on STDIN to ensure they are the same.
func TestStreamTranslator_LoopbackStdinToStderr(t *testing.T) {
	metrics.Register()
	metrics.ResetForTest()
	t.Cleanup(metrics.ResetForTest)
	// Create upstream fake SPDY server which copies STDIN back onto STDERR stream.
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, err := createSPDYServerStreams(t, w, req, Options{
			Stdin:  true,
			Stderr: true,
		})
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.conn.Close()
		// Loopback STDIN data onto STDERR stream.
		_, err = io.Copy(ctx.stderrStream, ctx.stdinStream)
		if err != nil {
			t.Fatalf("error copying STDIN to STDERR: %v", err)
		}
		writeSuccessStatus(t, ctx.writeStatus)
	}))
	defer spdyServer.Close()
	// Create StreamTranslatorHandler, which points upstream to fake SPDY server with
	// streams STDIN and STDERR. Create test server from StreamTranslatorHandler.
	spdyLocation, err := url.Parse(spdyServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse spdy server URL: %s", spdyServer.URL)
	}
	spdyTransport, err := fakeTransport()
	if err != nil {
		t.Fatalf("Unexpected error creating transport: %v", err)
	}
	streams := Options{Stdin: true, Stderr: true}
	streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, 0, streams)
	streamTranslatorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		streamTranslator.ServeHTTP(w, req)
	}))
	defer streamTranslatorServer.Close()
	// Now create the websocket client (executor), and point it to the "streamTranslatorServer".
	streamTranslatorLocation, err := url.Parse(streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse StreamTranslator server URL: %s", streamTranslatorServer.URL)
	}
	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: streamTranslatorLocation.Host}, "GET", streamTranslatorServer.URL)
	if err != nil {
		t.Errorf("unexpected error creating websocket executor: %v", err)
	}
	// Generate random data, and set it up to stream on STDIN. The data will be
	// returned on the STDERR buffer.
	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	if _, err := rand.Read(randomData); err != nil {
		t.Errorf("unexpected error reading random data: %v", err)
	}
	var stderr bytes.Buffer
	options := &remotecommand.StreamOptions{
		Stdin:  bytes.NewReader(randomData),
		Stderr: &stderr,
	}
	errorChan := make(chan error)
	go func() {
		// Start the streaming on the WebSocket "exec" client.
		errorChan <- exec.StreamWithContext(context.Background(), *options)
	}()

	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("expect stream to be closed after connection is closed.")
	case err := <-errorChan:
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
	data, err := io.ReadAll(bytes.NewReader(stderr.Bytes()))
	if err != nil {
		t.Errorf("error reading the stream: %v", err)
		return
	}
	// Check the random data sent on STDIN was the same returned on STDERR.
	if !bytes.Equal(randomData, data) {
		t.Errorf("unexpected data received: %d sent: %d", len(data), len(randomData))
	}
	// Validate the streamtranslator metrics; should be one 200 success.
	metricNames := []string{"apiserver_stream_translator_requests_total"}
	expected := `
# HELP apiserver_stream_translator_requests_total [BETA] Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5
# TYPE apiserver_stream_translator_requests_total counter
apiserver_stream_translator_requests_total{code="200"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...); err != nil {
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
	metrics.Register()
	metrics.ResetForTest()
	t.Cleanup(metrics.ResetForTest)
	expectedExitCode := randomExitCode()
	// Create upstream fake SPDY server, returning a non-zero exit code
	// on error stream within the structured error.
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, err := createSPDYServerStreams(t, w, req, Options{
			Stdout: true,
		})
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.conn.Close()
		// Read/discard STDIN data before returning error on error stream.
		_, err = io.Copy(io.Discard, ctx.stdinStream)
		if err != nil {
			t.Fatalf("error copying STDIN to DISCARD: %v", err)
		}
		// Force an non-zero exit code error returned on the error stream.
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
			t.Fatalf("error writing status: %v", err)
		}
	}))
	defer spdyServer.Close()
	// Create StreamTranslatorHandler, which points upstream to fake SPDY server, and
	// create a test server using the  StreamTranslatorHandler.
	spdyLocation, err := url.Parse(spdyServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse spdy server URL: %s", spdyServer.URL)
	}
	spdyTransport, err := fakeTransport()
	if err != nil {
		t.Fatalf("Unexpected error creating transport: %v", err)
	}
	streams := Options{Stdin: true}
	streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, 0, streams)
	streamTranslatorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		streamTranslator.ServeHTTP(w, req)
	}))
	defer streamTranslatorServer.Close()
	// Now create the websocket client (executor), and point it to the "streamTranslatorServer".
	streamTranslatorLocation, err := url.Parse(streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse StreamTranslator server URL: %s", streamTranslatorServer.URL)
	}
	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: streamTranslatorLocation.Host}, "GET", streamTranslatorServer.URL)
	if err != nil {
		t.Errorf("unexpected error creating websocket executor: %v", err)
	}
	// Generate random data, and set it up to stream on STDIN. The data will be discarded at
	// upstream SDPY server.
	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	if _, err := rand.Read(randomData); err != nil {
		t.Errorf("unexpected error reading random data: %v", err)
	}
	options := &remotecommand.StreamOptions{
		Stdin: bytes.NewReader(randomData),
	}
	errorChan := make(chan error)
	go func() {
		// Start the streaming on the WebSocket "exec" client.
		errorChan <- exec.StreamWithContext(context.Background(), *options)
	}()

	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("expect stream to be closed after connection is closed.")
	case err := <-errorChan:
		// Expect exit code error on error stream.
		if err == nil {
			t.Errorf("expected error, but received none")
		}
		expectedError := fmt.Sprintf("command terminated with exit code %d", expectedExitCode)
		// Compare expected error with exit code to actual error.
		if expectedError != err.Error() {
			t.Errorf("expected error (%s), got (%s)", expectedError, err)
		}
	}
	// Validate the streamtranslator metrics; an exit code error is considered 200 success.
	metricNames := []string{"apiserver_stream_translator_requests_total"}
	expected := `
# HELP apiserver_stream_translator_requests_total [BETA] Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5
# TYPE apiserver_stream_translator_requests_total counter
apiserver_stream_translator_requests_total{code="200"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...); err != nil {
		t.Fatal(err)
	}
}

// TestStreamTranslator_UpstreamWithoutStatus checks that a session which the upstream
// SPDY server ends without writing a status reaches the websocket client as an error,
// and is counted as one, instead of as a successful run.
func TestStreamTranslator_UpstreamWithoutStatus(t *testing.T) {
	metrics.Register()
	metrics.ResetForTest()
	t.Cleanup(metrics.ResetForTest)
	// Create upstream fake SPDY server, which closes the connection without
	// writing a status.
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if _, err := httpstream.Handshake(req, w, []string{rcconstants.StreamProtocolV4Name}); err != nil {
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
		defer conn.Close() //nolint:errcheck
		// The translator opens the error stream and STDOUT, and waits for the
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
	}))
	defer spdyServer.Close()
	// Create StreamTranslatorHandler, which points upstream to fake SPDY server, and
	// create a test server using the StreamTranslatorHandler.
	spdyLocation, err := url.Parse(spdyServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse spdy server URL: %s", spdyServer.URL)
	}
	spdyTransport, err := fakeTransport()
	if err != nil {
		t.Fatalf("Unexpected error creating transport: %v", err)
	}
	streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, 0, Options{Stdout: true})
	streamTranslatorServer := httptest.NewServer(streamTranslator)
	defer streamTranslatorServer.Close()
	// Now create the websocket client (executor), and point it to the "streamTranslatorServer".
	streamTranslatorLocation, err := url.Parse(streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse StreamTranslator server URL: %s", streamTranslatorServer.URL)
	}
	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: streamTranslatorLocation.Host}, "GET", streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("unexpected error creating websocket executor: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), wait.ForeverTestTimeout)
	defer cancel()
	err = exec.StreamWithContext(ctx, remotecommand.StreamOptions{Stdout: io.Discard})
	if err == nil {
		t.Fatal("expected an error for a session that ended without a status, but received none")
	}
	// The translator's own SPDY client returns the error, and the translator
	// sends it to the websocket client as an internal error status. The prefix
	// tells that status apart from the websocket client finding its own error
	// stream empty.
	expectedError := "Internal error occurred: connection closed before the command's status was received; the output may be incomplete"
	if err.Error() != expectedError {
		t.Errorf("expected error (%s), got (%s)", expectedError, err)
	}
	// Validate the streamtranslator metrics; a session without a status is a 500.
	metricNames := []string{"apiserver_stream_translator_requests_total"}
	expected := `
# HELP apiserver_stream_translator_requests_total [BETA] Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5
# TYPE apiserver_stream_translator_requests_total counter
apiserver_stream_translator_requests_total{code="500"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...); err != nil {
		t.Fatal(err)
	}
}

// TestStreamTranslator_MultipleReadChannels tests two streams (STDOUT, STDERR) reading from
// the connections at the same time.
func TestStreamTranslator_MultipleReadChannels(t *testing.T) {
	metrics.Register()
	metrics.ResetForTest()
	t.Cleanup(metrics.ResetForTest)
	// Create upstream fake SPDY server which copies STDIN back onto STDOUT and STDERR stream.
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, err := createSPDYServerStreams(t, w, req, Options{
			Stdin:  true,
			Stdout: true,
			Stderr: true,
		})
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.conn.Close()
		// TeeReader copies data read on STDIN onto STDERR.
		stdinReader := io.TeeReader(ctx.stdinStream, ctx.stderrStream)
		// Also copy STDIN to STDOUT.
		_, err = io.Copy(ctx.stdoutStream, stdinReader)
		if err != nil {
			t.Errorf("error copying STDIN to STDOUT: %v", err)
		}
		writeSuccessStatus(t, ctx.writeStatus)
	}))
	defer spdyServer.Close()
	// Create StreamTranslatorHandler, which points upstream to fake SPDY server with
	// streams STDIN, STDOUT, and STDERR. Create test server from StreamTranslatorHandler.
	spdyLocation, err := url.Parse(spdyServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse spdy server URL: %s", spdyServer.URL)
	}
	spdyTransport, err := fakeTransport()
	if err != nil {
		t.Fatalf("Unexpected error creating transport: %v", err)
	}
	streams := Options{Stdin: true, Stdout: true, Stderr: true}
	streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, 0, streams)
	streamTranslatorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		streamTranslator.ServeHTTP(w, req)
	}))
	defer streamTranslatorServer.Close()
	// Now create the websocket client (executor), and point it to the "streamTranslatorServer".
	streamTranslatorLocation, err := url.Parse(streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse StreamTranslator server URL: %s", streamTranslatorServer.URL)
	}
	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: streamTranslatorLocation.Host}, "GET", streamTranslatorServer.URL)
	if err != nil {
		t.Errorf("unexpected error creating websocket executor: %v", err)
	}
	// Generate random data, and set it up to stream on STDIN. The data will be
	// returned on the STDOUT and STDERR buffer.
	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	if _, err := rand.Read(randomData); err != nil {
		t.Errorf("unexpected error reading random data: %v", err)
	}
	var stdout, stderr bytes.Buffer
	options := &remotecommand.StreamOptions{
		Stdin:  bytes.NewReader(randomData),
		Stdout: &stdout,
		Stderr: &stderr,
	}
	errorChan := make(chan error)
	go func() {
		// Start the streaming on the WebSocket "exec" client.
		errorChan <- exec.StreamWithContext(context.Background(), *options)
	}()

	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("expect stream to be closed after connection is closed.")
	case err := <-errorChan:
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
	stdoutBytes, err := io.ReadAll(bytes.NewReader(stdout.Bytes()))
	if err != nil {
		t.Errorf("error reading the stream: %v", err)
		return
	}
	// Check the random data sent on STDIN was the same returned on STDOUT.
	if !bytes.Equal(stdoutBytes, randomData) {
		t.Errorf("unexpected data received: %d sent: %d", len(stdoutBytes), len(randomData))
	}
	stderrBytes, err := io.ReadAll(bytes.NewReader(stderr.Bytes()))
	if err != nil {
		t.Errorf("error reading the stream: %v", err)
		return
	}
	// Check the random data sent on STDIN was the same returned on STDERR.
	if !bytes.Equal(stderrBytes, randomData) {
		t.Errorf("unexpected data received: %d sent: %d", len(stderrBytes), len(randomData))
	}
	// Validate the streamtranslator metrics; should have one 200 success.
	metricNames := []string{"apiserver_stream_translator_requests_total"}
	expected := `
# HELP apiserver_stream_translator_requests_total [BETA] Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5
# TYPE apiserver_stream_translator_requests_total counter
apiserver_stream_translator_requests_total{code="200"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...); err != nil {
		t.Fatal(err)
	}
}

// TestStreamTranslator_ThrottleReadChannels tests two streams (STDOUT, STDERR) using rate limited streams.
func TestStreamTranslator_ThrottleReadChannels(t *testing.T) {
	// Create upstream fake SPDY server which copies STDIN back onto STDOUT and STDERR stream.
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, err := createSPDYServerStreams(t, w, req, Options{
			Stdin:  true,
			Stdout: true,
			Stderr: true,
		})
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.conn.Close()
		// TeeReader copies data read on STDIN onto STDERR.
		stdinReader := io.TeeReader(ctx.stdinStream, ctx.stderrStream)
		// Also copy STDIN to STDOUT.
		_, err = io.Copy(ctx.stdoutStream, stdinReader)
		if err != nil {
			t.Errorf("error copying STDIN to STDOUT: %v", err)
		}
		writeSuccessStatus(t, ctx.writeStatus)
	}))
	defer spdyServer.Close()
	// Create StreamTranslatorHandler, which points upstream to fake SPDY server with
	// streams STDIN, STDOUT, and STDERR. Create test server from StreamTranslatorHandler.
	spdyLocation, err := url.Parse(spdyServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse spdy server URL: %s", spdyServer.URL)
	}
	spdyTransport, err := fakeTransport()
	if err != nil {
		t.Fatalf("Unexpected error creating transport: %v", err)
	}
	streams := Options{Stdin: true, Stdout: true, Stderr: true}
	maxBytesPerSec := 900 * 1024 // slightly less than the 1MB that is being transferred to exercise throttling.
	streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, int64(maxBytesPerSec), streams)
	streamTranslatorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		streamTranslator.ServeHTTP(w, req)
	}))
	defer streamTranslatorServer.Close()
	// Now create the websocket client (executor), and point it to the "streamTranslatorServer".
	streamTranslatorLocation, err := url.Parse(streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse StreamTranslator server URL: %s", streamTranslatorServer.URL)
	}
	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: streamTranslatorLocation.Host}, "GET", streamTranslatorServer.URL)
	if err != nil {
		t.Errorf("unexpected error creating websocket executor: %v", err)
	}
	// Generate random data, and set it up to stream on STDIN. The data will be
	// returned on the STDOUT and STDERR buffer.
	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	if _, err := rand.Read(randomData); err != nil {
		t.Errorf("unexpected error reading random data: %v", err)
	}
	var stdout, stderr bytes.Buffer
	options := &remotecommand.StreamOptions{
		Stdin:  bytes.NewReader(randomData),
		Stdout: &stdout,
		Stderr: &stderr,
	}
	errorChan := make(chan error)
	go func() {
		// Start the streaming on the WebSocket "exec" client.
		errorChan <- exec.StreamWithContext(context.Background(), *options)
	}()

	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("expect stream to be closed after connection is closed.")
	case err := <-errorChan:
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
	stdoutBytes, err := io.ReadAll(bytes.NewReader(stdout.Bytes()))
	if err != nil {
		t.Errorf("error reading the stream: %v", err)
		return
	}
	// Check the random data sent on STDIN was the same returned on STDOUT.
	if !bytes.Equal(stdoutBytes, randomData) {
		t.Errorf("unexpected data received: %d sent: %d", len(stdoutBytes), len(randomData))
	}
	stderrBytes, err := io.ReadAll(bytes.NewReader(stderr.Bytes()))
	if err != nil {
		t.Errorf("error reading the stream: %v", err)
		return
	}
	// Check the random data sent on STDIN was the same returned on STDERR.
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

// TestStreamTranslator_MultipleWriteChannels
func TestStreamTranslator_TTYResizeChannel(t *testing.T) {
	metrics.Register()
	metrics.ResetForTest()
	t.Cleanup(metrics.ResetForTest)
	// Create the fake terminal size queue and the actualTerminalSizes which
	// will be received at the opposite websocket endpoint.
	numSizeQueue := 10000
	sizeQueue := newTerminalSizeQueue(numSizeQueue)
	actualTerminalSizes := make([]remotecommand.TerminalSize, 0, numSizeQueue)
	// Create upstream fake SPDY server which copies STDIN back onto STDERR stream.
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, err := createSPDYServerStreams(t, w, req, Options{
			Tty: true,
		})
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.conn.Close()
		// Read the terminal resize requests, storing them in actualTerminalSizes
		for i := 0; i < numSizeQueue; i++ {
			actualTerminalSize := <-ctx.resizeChan
			actualTerminalSizes = append(actualTerminalSizes, actualTerminalSize)
		}
		writeSuccessStatus(t, ctx.writeStatus)
	}))
	defer spdyServer.Close()
	// Create StreamTranslatorHandler, which points upstream to fake SPDY server with
	// resize (TTY resize) stream. Create test server from StreamTranslatorHandler.
	spdyLocation, err := url.Parse(spdyServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse spdy server URL: %s", spdyServer.URL)
	}
	spdyTransport, err := fakeTransport()
	if err != nil {
		t.Fatalf("Unexpected error creating transport: %v", err)
	}
	streams := Options{Tty: true}
	streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, 0, streams)
	streamTranslatorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		streamTranslator.ServeHTTP(w, req)
	}))
	defer streamTranslatorServer.Close()
	// Now create the websocket client (executor), and point it to the "streamTranslatorServer".
	streamTranslatorLocation, err := url.Parse(streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse StreamTranslator server URL: %s", streamTranslatorServer.URL)
	}
	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: streamTranslatorLocation.Host}, "GET", streamTranslatorServer.URL)
	if err != nil {
		t.Errorf("unexpected error creating websocket executor: %v", err)
	}
	options := &remotecommand.StreamOptions{
		Tty:               true,
		TerminalSizeQueue: sizeQueue,
	}
	errorChan := make(chan error)
	go func() {
		// Start the streaming on the WebSocket "exec" client.
		errorChan <- exec.StreamWithContext(context.Background(), *options)
	}()

	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("expect stream to be closed after connection is closed.")
	case err := <-errorChan:
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
	// Validate the random TerminalSizes sent on the resize stream are the same
	// as the actual TerminalSizes received at the websocket server.
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
	// Validate the streamtranslator metrics; should have one 200 success.
	metricNames := []string{"apiserver_stream_translator_requests_total"}
	expected := `
# HELP apiserver_stream_translator_requests_total [BETA] Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5
# TYPE apiserver_stream_translator_requests_total counter
apiserver_stream_translator_requests_total{code="200"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...); err != nil {
		t.Fatal(err)
	}
}

// TestStreamTranslator_WebSocketServerErrors validates that when there is a problem creating
// the websocket server as the first step of the StreamTranslator an error is properly returned.
func TestStreamTranslator_WebSocketServerErrors(t *testing.T) {
	metrics.Register()
	metrics.ResetForTest()
	t.Cleanup(metrics.ResetForTest)
	spdyLocation, err := url.Parse("http://127.0.0.1")
	if err != nil {
		t.Fatalf("Unable to parse spdy server URL")
	}
	spdyTransport, err := fakeTransport()
	if err != nil {
		t.Fatalf("Unexpected error creating transport: %v", err)
	}
	streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, 0, Options{})
	streamTranslatorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		streamTranslator.ServeHTTP(w, req)
	}))
	defer streamTranslatorServer.Close()
	// Now create the websocket client (executor), and point it to the "streamTranslatorServer".
	streamTranslatorLocation, err := url.Parse(streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse StreamTranslator server URL: %s", streamTranslatorServer.URL)
	}
	exec, err := remotecommand.NewWebSocketExecutorForProtocols(
		&rest.Config{Host: streamTranslatorLocation.Host},
		"GET",
		streamTranslatorServer.URL,
		rcconstants.StreamProtocolV4Name, // RemoteCommand V4 protocol is unsupported
	)
	if err != nil {
		t.Errorf("unexpected error creating websocket executor: %v", err)
	}
	errorChan := make(chan error)
	go func() {
		// Start the streaming on the WebSocket "exec" client. The WebSocket server within the
		// StreamTranslator propagates an error here because the V4 protocol is not supported.
		errorChan <- exec.StreamWithContext(context.Background(), remotecommand.StreamOptions{})
	}()

	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("expect stream to be closed after connection is closed.")
	case err := <-errorChan:
		// Must return "websocket unable to upgrade" (bad handshake) error.
		if err == nil {
			t.Fatalf("expected error, but received none")
		}
		if !strings.Contains(err.Error(), "unable to upgrade streaming request") {
			t.Errorf("expected websocket bad handshake error, got (%s)", err)
		}
	}
	// Validate the streamtranslator metrics; should have one 400 failure.
	// Use polling to wait for the metric to be updated asynchronously.
	metricNames := []string{"apiserver_stream_translator_requests_total"}
	expected := `
# HELP apiserver_stream_translator_requests_total [BETA] Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5
# TYPE apiserver_stream_translator_requests_total counter
apiserver_stream_translator_requests_total{code="400"} 1
`
	if err := wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, 2*time.Second, true, func(ctx context.Context) (bool, error) {
		if testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...) == nil {
			return true, nil
		}
		return false, nil
	}); err != nil {
		t.Fatalf("Failed to observe metric after waiting 2 seconds: %v", err)
	}
}

// TestStreamTranslator_BlockRedirects verifies that the StreamTranslator will *not* follow
// redirects; it will thrown an error instead.
func TestStreamTranslator_BlockRedirects(t *testing.T) {
	metrics.Register()
	t.Cleanup(metrics.ResetForTest)

	for _, statusCode := range []int{
		http.StatusMovedPermanently,  // 301
		http.StatusFound,             // 302
		http.StatusSeeOther,          // 303
		http.StatusTemporaryRedirect, // 307
		http.StatusPermanentRedirect, // 308
	} {
		t.Run(fmt.Sprintf("statusCode=%d", statusCode), func(t *testing.T) {
			metrics.ResetForTest()
			// Create upstream fake SPDY server which returns a redirect.
			spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				w.Header().Set("Location", "/")
				w.WriteHeader(statusCode)
			}))
			defer spdyServer.Close()
			spdyLocation, err := url.Parse(spdyServer.URL)
			if err != nil {
				t.Fatalf("Unable to parse spdy server URL: %s", spdyServer.URL)
			}
			spdyTransport, err := fakeTransport()
			if err != nil {
				t.Fatalf("Unexpected error creating transport: %v", err)
			}
			streams := Options{Stdout: true}
			streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, 0, streams)
			streamTranslatorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				streamTranslator.ServeHTTP(w, req)
			}))
			defer streamTranslatorServer.Close()
			// Now create the websocket client (executor), and point it to the "streamTranslatorServer".
			streamTranslatorLocation, err := url.Parse(streamTranslatorServer.URL)
			if err != nil {
				t.Fatalf("Unable to parse StreamTranslator server URL: %s", streamTranslatorServer.URL)
			}
			exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: streamTranslatorLocation.Host}, "GET", streamTranslatorServer.URL)
			if err != nil {
				t.Errorf("unexpected error creating websocket executor: %v", err)
			}
			errorChan := make(chan error)
			go func() {
				// Start the streaming on the WebSocket "exec" client.
				// Should return "redirect not allowed" error.
				errorChan <- exec.StreamWithContext(context.Background(), remotecommand.StreamOptions{})
			}()

			select {
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatalf("expect stream to be closed after connection is closed.")
			case err := <-errorChan:
				// Must return "redirect now allowed" error.
				if err == nil {
					t.Fatalf("expected error, but received none")
				}
				if !strings.Contains(err.Error(), "redirect not allowed") {
					t.Errorf("expected redirect not allowed error, got (%s)", err)
				}
			}
			// Validate the streamtranslator metrics; should have one 500 failure each loop.
			metricNames := []string{"apiserver_stream_translator_requests_total"}
			expected := `
# HELP apiserver_stream_translator_requests_total [BETA] Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5
# TYPE apiserver_stream_translator_requests_total counter
apiserver_stream_translator_requests_total{code="500"} 1
`
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...); err != nil {
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
func createSPDYServerStreams(t *testing.T, w http.ResponseWriter, req *http.Request, opts Options) (*streamContext, error) {
	_, err := httpstream.Handshake(req, w, []string{rcconstants.StreamProtocolV4Name})
	if err != nil {
		t.Log("------------------- [DEBUG] HANDSHAKE FAILED -------------------")
		t.Logf("Method:        %s", req.Method)
		t.Logf("URL:           %s", req.URL.String())
		t.Logf("Proto:         %s", req.Proto)
		t.Logf("Header Count:  %d", len(req.Header))
		t.Logf("ContentLength: %d", req.ContentLength) // Should be zero for upgrade request
		// Dump the request in its entirety.
		dump, err := httputil.DumpRequest(req, true)
		if err != nil {
			t.Logf("DumpRequest failed: %v", err)
		} else {
			t.Logf("Full HTTP Request Dump:\n%s", dump)
		}

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
			switch streamType {
			case v1.StreamTypeError:
				replyChan <- struct{}{}
				ctx.writeStatus = v4WriteStatusFunc(stream)
			case v1.StreamTypeStdout:
				replyChan <- struct{}{}
				ctx.stdoutStream = stream
			case v1.StreamTypeStdin:
				replyChan <- struct{}{}
				ctx.stdinStream = stream
			case v1.StreamTypeStderr:
				replyChan <- struct{}{}
				ctx.stderrStream = stream
			case v1.StreamTypeResize:
				replyChan <- struct{}{}
				ctx.resizeStream = stream
			default:
				// add other stream ...
				return nil, errors.New("unimplemented stream type")
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

// writeSuccessStatus ends a fake v4 or v5 session the way a server does: with a
// status on the error stream. Without it the client reports the session as cut.
func writeSuccessStatus(t *testing.T, writeStatus func(*apierrors.StatusError) error) {
	t.Helper()
	if err := writeStatus(&apierrors.StatusError{ErrStatus: metav1.Status{Status: metav1.StatusSuccess}}); err != nil {
		t.Errorf("error writing status: %v", err)
	}
}

func fakeTransport() (http.RoundTripper, error) {
	cfg := &transport.Config{
		TLS: transport.TLSConfig{
			Insecure: true,
		},
	}
	return transport.New(cfg)
}

// rateLimitedWriter passes writes through to dst no faster than
// bytesPerSecond, to model a client that reads more slowly than the
// command writes.
type rateLimitedWriter struct {
	dst            io.Writer
	bytesPerSecond int
	chunkSize      int
}

func (w *rateLimitedWriter) Write(p []byte) (int, error) {
	written := 0
	for len(p) > 0 {
		chunk := min(len(p), w.chunkSize)
		if _, err := w.dst.Write(p[:chunk]); err != nil {
			return written, err
		}
		written += chunk
		p = p[chunk:]
		time.Sleep(time.Duration(chunk) * time.Second / time.Duration(w.bytesPerSecond))
	}
	return written, nil
}

// stdinTrickler produces one byte per interval, like a client that keeps
// sending input while it reads, and returns io.EOF once it has produced
// maxWrites bytes or been closed, so the client's copy of stdin ends.
type stdinTrickler struct {
	interval  time.Duration
	maxWrites int
	closed    chan struct{}
}

func newStdinTrickler(interval time.Duration, maxWrites int) *stdinTrickler {
	return &stdinTrickler{
		interval:  interval,
		maxWrites: maxWrites,
		closed:    make(chan struct{}),
	}
}

func (s *stdinTrickler) Read(p []byte) (int, error) {
	if s.maxWrites <= 0 {
		return 0, io.EOF
	}
	select {
	case <-s.closed:
		return 0, io.EOF
	case <-time.After(s.interval):
	}
	s.maxWrites--
	p[0] = 'x'
	return 1, nil
}

func (s *stdinTrickler) Close() error {
	select {
	case <-s.closed:
	default:
		close(s.closed)
	}
	return nil
}

// TestStreamTranslator_SlowClient checks that a client reading more slowly
// than the command writes still receives the complete output and a success
// status: the translator writes the status, signals the end of the session
// with a WebSocket close frame and waits for the client to close, instead of
// closing the connection with output still in flight (#142376). This test
// uses no API introduced by the change; on 555627c7b80 the end of the output
// is lost and the session is reported as an error.
func TestStreamTranslator_SlowClient(t *testing.T) {
	metrics.Register()
	metrics.ResetForTest()
	t.Cleanup(metrics.ResetForTest)
	// Well-behaved upstream: sends dataSize bytes of STDOUT faster than any
	// client reads, discards STDIN, and ends with the success status.
	const dataSize = 4 << 20
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, err := createSPDYServerStreams(t, w, req, Options{
			Stdin:  true,
			Stdout: true,
		})
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer func() { _ = ctx.conn.Close() }()
		go func() { _, _ = io.Copy(io.Discard, ctx.stdinStream) }()
		data := make([]byte, dataSize)
		if _, err := ctx.stdoutStream.Write(data); err != nil {
			t.Errorf("error writing stdout: %v", err)
			return
		}
		writeSuccessStatus(t, ctx.writeStatus)
	}))
	defer spdyServer.Close()
	spdyLocation, err := url.Parse(spdyServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse spdy server URL: %s", spdyServer.URL)
	}
	spdyTransport, err := fakeTransport()
	if err != nil {
		t.Fatalf("Unexpected error creating transport: %v", err)
	}
	streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, 0, Options{Stdin: true, Stdout: true})
	streamTranslatorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		streamTranslator.ServeHTTP(w, req)
	}))
	defer streamTranslatorServer.Close()
	streamTranslatorLocation, err := url.Parse(streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse StreamTranslator server URL: %s", streamTranslatorServer.URL)
	}
	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: streamTranslatorLocation.Host}, "GET", streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("unexpected error creating websocket executor: %v", err)
	}
	// A client reading 1 MiB/s while the command writes at memory speed, and
	// trickling STDIN: its next frame after the server closes the connection
	// makes the kernel reset it and drop the output not yet read.
	stdin := newStdinTrickler(50*time.Millisecond, 600)
	defer func() { _ = stdin.Close() }()
	var stdout bytes.Buffer
	err = exec.StreamWithContext(context.Background(), remotecommand.StreamOptions{
		Stdin: stdin,
		Stdout: &rateLimitedWriter{
			dst:            &stdout,
			bytesPerSecond: 1 << 20,
			chunkSize:      64 << 10,
		},
	})
	if err != nil {
		t.Errorf("unexpected error from a session with a slow but honest client: %v", err)
	}
	if stdout.Len() != dataSize {
		t.Errorf("output was truncated: got %d bytes, want %d", stdout.Len(), dataSize)
	}
}

// TestStreamTranslator_FinishBoundEnforcedForClientThatNeverCloses checks
// that the wait for the client to close after the status is bounded: a client
// that never reads, never answers the close frame and never closes holds the
// session for the configured FinishTimeout at most, instead of the 15 minutes
// of #142461 or the idle timeout.
func TestStreamTranslator_FinishBoundEnforcedForClientThatNeverCloses(t *testing.T) {
	const finishTimeout = 500 * time.Millisecond
	elapsed := runSessionWithClientThatNeverCloses(t, finishTimeout)
	if elapsed < finishTimeout-100*time.Millisecond {
		t.Errorf("the translator cut the client off after %v, before the configured bound of %v", elapsed, finishTimeout)
	}
	if max := 5 * time.Second; elapsed > max {
		t.Errorf("the translator waited %v for a client that never closes, want at most %v", elapsed, max)
	}
}

// TestStreamTranslator_FinishTimeoutConfigurable checks that the bound is
// actually taken from the handler's FinishTimeout field: two different values
// measurably change how long a client that never closes is held.
func TestStreamTranslator_FinishTimeoutConfigurable(t *testing.T) {
	short := runSessionWithClientThatNeverCloses(t, 200*time.Millisecond)
	long := runSessionWithClientThatNeverCloses(t, time.Second)
	if short > 700*time.Millisecond {
		t.Errorf("a bound of 200ms held the client for %v", short)
	}
	if long < 900*time.Millisecond {
		t.Errorf("a bound of 1s cut the client off after %v", long)
	}
}

// TestStreamTranslator_FinishClientClosesPromptly checks that the wait ends
// as soon as a well-behaved client has closed the connection: a fast client
// is held no longer than before the change.
func TestStreamTranslator_FinishClientClosesPromptly(t *testing.T) {
	metrics.Register()
	metrics.ResetForTest()
	t.Cleanup(metrics.ResetForTest)
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, err := createSPDYServerStreams(t, w, req, Options{Stdout: true})
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer func() { _ = ctx.conn.Close() }()
		if _, err := ctx.stdoutStream.Write([]byte("output")); err != nil {
			t.Errorf("error writing stdout: %v", err)
			return
		}
		writeSuccessStatus(t, ctx.writeStatus)
	}))
	defer spdyServer.Close()
	spdyLocation, err := url.Parse(spdyServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse spdy server URL: %s", spdyServer.URL)
	}
	spdyTransport, err := fakeTransport()
	if err != nil {
		t.Fatalf("Unexpected error creating transport: %v", err)
	}
	streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, 0, Options{Stdout: true})
	serveElapsed := make(chan time.Duration, 1)
	streamTranslatorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		start := time.Now()
		streamTranslator.ServeHTTP(w, req)
		serveElapsed <- time.Since(start)
	}))
	defer streamTranslatorServer.Close()
	streamTranslatorLocation, err := url.Parse(streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse StreamTranslator server URL: %s", streamTranslatorServer.URL)
	}
	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: streamTranslatorLocation.Host}, "GET", streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("unexpected error creating websocket executor: %v", err)
	}
	var stdout bytes.Buffer
	if err := exec.StreamWithContext(context.Background(), remotecommand.StreamOptions{Stdout: &stdout}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stdout.String() != "output" {
		t.Errorf("unexpected output: %q", stdout.String())
	}
	select {
	case elapsed := <-serveElapsed:
		if elapsed > 10*time.Second {
			t.Errorf("the translator held a session whose client closed promptly for %v", elapsed)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("the translator did not return after the client closed")
	}
}

// TestStreamTranslator_FinishClientStillSendingInput checks what closing the
// input streams in finish is for: a client that keeps sending stdin and resize
// events after the command has finished must not hold the session. That input
// is discarded, not queued, so the server's read loop stays free to see the
// client's close and the handler returns as soon as the client closes the
// connection, instead of at the finish bound.
func TestStreamTranslator_FinishClientStillSendingInput(t *testing.T) {
	metrics.Register()
	metrics.ResetForTest()
	t.Cleanup(metrics.ResetForTest)
	const dataSize = 4096
	// Well-behaved upstream: drains STDIN and resize events, sends the output
	// faster than any client reads it, and ends with the success status.
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, err := createSPDYServerStreams(t, w, req, Options{
			Stdin:  true,
			Stdout: true,
			Tty:    true,
		})
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer func() { _ = ctx.conn.Close() }()
		go func() { _, _ = io.Copy(io.Discard, ctx.stdinStream) }()
		go func() {
			for range ctx.resizeChan {
			}
		}()
		if _, err := ctx.stdoutStream.Write(make([]byte, dataSize)); err != nil {
			t.Errorf("error writing stdout: %v", err)
			return
		}
		writeSuccessStatus(t, ctx.writeStatus)
	}))
	defer spdyServer.Close()
	spdyLocation, err := url.Parse(spdyServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse spdy server URL: %s", spdyServer.URL)
	}
	spdyTransport, err := fakeTransport()
	if err != nil {
		t.Fatalf("Unexpected error creating transport: %v", err)
	}
	// The bound is far above the assertion below: if input sent after the end
	// were queued or blocked the server's read loop, the client's close would
	// never be seen and the handler would return only at this bound.
	streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, 0, Options{Stdin: true, Stdout: true, Tty: true})
	streamTranslator.FinishTimeout = 30 * time.Second
	serveElapsed := make(chan time.Duration, 1)
	streamTranslatorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		start := time.Now()
		streamTranslator.ServeHTTP(w, req)
		serveElapsed <- time.Since(start)
	}))
	defer streamTranslatorServer.Close()

	// A raw v5 client that keeps sending STDIN and resize events for the whole
	// session: unlike the client-go executor it does not stop once it has read
	// the status, so input is guaranteed to arrive after the command has ended.
	client := dialV5Websocket(t, streamTranslatorServer.URL)
	stdinFrame := []byte{rcconstants.StreamStdIn, 'x'}
	resizeJSON, err := json.Marshal(remotecommand.TerminalSize{Width: 80, Height: 24})
	if err != nil {
		t.Fatal(err)
	}
	resizeFrame := append([]byte{rcconstants.StreamResize}, resizeJSON...)
	stopWriting := make(chan struct{})
	writingDone := make(chan struct{})
	go func() {
		defer close(writingDone)
		stdin := time.NewTicker(10 * time.Millisecond)
		defer stdin.Stop()
		resize := time.NewTicker(15 * time.Millisecond)
		defer resize.Stop()
		for {
			select {
			case <-stopWriting:
				return
			case <-stdin.C:
				if err := websocket.Message.Send(client, stdinFrame); err != nil {
					return
				}
			case <-resize.C:
				if err := websocket.Message.Send(client, resizeFrame); err != nil {
					return
				}
			}
		}
	}()
	received := make(chan int, 1)
	statusSeen := make(chan struct{})
	go func() {
		var n int
		for {
			var msg []byte
			if err := websocket.Message.Receive(client, &msg); err != nil {
				received <- n
				return
			}
			if len(msg) < 1 {
				continue
			}
			switch msg[0] {
			case rcconstants.StreamStdOut:
				n += len(msg) - 1
			case rcconstants.StreamErr:
				close(statusSeen)
			}
		}
	}()

	// The status is written after the command has finished, when nothing reads
	// the input streams anymore. Keep the input flowing past that point, then
	// close like a client that has read everything.
	select {
	case <-statusSeen:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("the client never received the command's status")
	}
	time.Sleep(200 * time.Millisecond)
	close(stopWriting)
	<-writingDone
	if err := client.Close(); err != nil {
		t.Fatal(err)
	}
	if n := <-received; n != dataSize {
		t.Errorf("output was truncated: got %d bytes, want %d", n, dataSize)
	}
	select {
	case elapsed := <-serveElapsed:
		if elapsed > 10*time.Second {
			t.Errorf("the translator held a client that kept sending input after the end for %v", elapsed)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the translator did not return after the client closed: input sent after the command ended blocked or queued it")
	}
}

// TestStreamTranslator_FinishNoGoroutineLeak checks that sessions with
// clients that never close, ended by the finish bound, leave no goroutines
// behind.
func TestStreamTranslator_FinishNoGoroutineLeak(t *testing.T) {
	before := runtime.NumGoroutine()
	for range 5 {
		runSessionWithClientThatNeverCloses(t, 200*time.Millisecond)
	}
	// The bound ends the handler; the connection and its goroutines are
	// cleaned up shortly after.
	deadline := time.Now().Add(15 * time.Second)
	for runtime.NumGoroutine() > before+2 {
		if time.Now().After(deadline) {
			t.Fatalf("goroutines did not return to the baseline of %d: %d are left", before, runtime.NumGoroutine())
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// TestStreamTranslatorFinishTimeoutDefault checks how the handler resolves
// the bound on the wait for the client to close: the FinishTimeout field when
// set, DefaultFinishTimeout otherwise. The default itself is pinned: a slow
// but honest client needs seconds to drain the socket, and every client that
// never closes holds a session for exactly this long (#142461 review), so it
// must stay short. Update this expectation when the default is changed
// deliberately.
func TestStreamTranslatorFinishTimeoutDefault(t *testing.T) {
	if got, want := DefaultFinishTimeout, 30*time.Second; got != want {
		t.Errorf("DefaultFinishTimeout = %v, want %v", got, want)
	}
	if got := (&StreamTranslatorHandler{}).finishTimeout(); got != DefaultFinishTimeout {
		t.Errorf("unset FinishTimeout: got %v, want DefaultFinishTimeout (%v)", got, DefaultFinishTimeout)
	}
	if got := (&StreamTranslatorHandler{FinishTimeout: -time.Second}).finishTimeout(); got != DefaultFinishTimeout {
		t.Errorf("negative FinishTimeout: got %v, want DefaultFinishTimeout (%v)", got, DefaultFinishTimeout)
	}
	if got, want := (&StreamTranslatorHandler{FinishTimeout: 5 * time.Second}).finishTimeout(), 5*time.Second; got != want {
		t.Errorf("set FinishTimeout: got %v, want %v", got, want)
	}
}

// dialV5Websocket connects a raw v5 remote command WebSocket client to the
// passed HTTP server URL and returns it without reading from it: a client
// that neither reads the session's output nor answers the close frame.
func dialV5Websocket(t *testing.T, serverURL string) *websocket.Conn {
	t.Helper()
	config, err := websocket.NewConfig("ws"+strings.TrimPrefix(serverURL, "http"), "http://localhost/")
	if err != nil {
		t.Fatal(err)
	}
	config.Protocol = []string{rcconstants.StreamProtocolV5Name}
	client, err := websocket.DialConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	return client
}

// runSessionWithClientThatNeverCloses runs one exec session through the
// translator against a well-behaved upstream for a client that never reads
// and never closes, with the passed bound on the finish wait, and returns how
// long the translator held the session after the status was sent.
func runSessionWithClientThatNeverCloses(t *testing.T, finishTimeout time.Duration) time.Duration {
	t.Helper()
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, err := createSPDYServerStreams(t, w, req, Options{Stdout: true})
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer func() { _ = ctx.conn.Close() }()
		if _, err := ctx.stdoutStream.Write(make([]byte, 4096)); err != nil {
			t.Errorf("error writing stdout: %v", err)
			return
		}
		writeSuccessStatus(t, ctx.writeStatus)
	}))
	spdyLocation, err := url.Parse(spdyServer.URL)
	if err != nil {
		spdyServer.Close()
		t.Fatalf("Unable to parse spdy server URL: %s", spdyServer.URL)
	}
	spdyTransport, err := fakeTransport()
	if err != nil {
		spdyServer.Close()
		t.Fatalf("Unexpected error creating transport: %v", err)
	}
	streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, 0, Options{Stdout: true})
	streamTranslator.FinishTimeout = finishTimeout
	serveElapsed := make(chan time.Duration, 1)
	streamTranslatorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		start := time.Now()
		streamTranslator.ServeHTTP(w, req)
		serveElapsed <- time.Since(start)
	}))
	client := dialV5Websocket(t, streamTranslatorServer.URL)

	select {
	case elapsed := <-serveElapsed:
		// Everything the session started must be gone before the caller
		// looks at goroutine counts.
		_ = client.Close()
		streamTranslatorServer.Close()
		spdyServer.Close()
		return elapsed
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("the translator did not return for a client that never closes")
		return 0
	}
}
