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

package remotecommand

import (
	"bytes"
	"context"
	"crypto/rand"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/remotecommand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
)

func TestFallbackClient_WebSocketPrimarySucceeds(t *testing.T) {
	// Create fake WebSocket server. Copy received STDIN data back onto STDOUT stream.
	websocketServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		conns, err := webSocketServerStreams(req, w, streamOptionsFromRequest(req))
		if err != nil {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		defer conns.conn.Close()
		// Loopback the STDIN stream onto the STDOUT stream.
		_, err = io.Copy(conns.stdoutStream, conns.stdinStream)
		require.NoError(t, err)
	}))
	defer websocketServer.Close()

	// Now create the fallback client (executor), and point it to the "websocketServer".
	// Must add STDIN and STDOUT query params for the client request.
	websocketServer.URL = websocketServer.URL + "?" + "stdin=true" + "&" + "stdout=true"
	websocketLocation, err := url.Parse(websocketServer.URL)
	require.NoError(t, err)
	websocketExecutor, err := NewWebSocketExecutor(&rest.Config{Host: websocketLocation.Host}, "GET", websocketServer.URL)
	require.NoError(t, err)
	spdyExecutor, err := NewSPDYExecutor(&rest.Config{Host: websocketLocation.Host}, "POST", websocketLocation)
	require.NoError(t, err)
	// Never fallback, so always use the websocketExecutor, which succeeds against websocket server.
	exec, err := NewFallbackExecutor(websocketExecutor, spdyExecutor, func(error) bool { return false })
	require.NoError(t, err)
	// Generate random data, and set it up to stream on STDIN. The data will be
	// returned on the STDOUT buffer.
	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	if _, err := rand.Read(randomData); err != nil {
		t.Errorf("unexpected error reading random data: %v", err)
	}
	var stdout bytes.Buffer
	options := &StreamOptions{
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
			t.Errorf("unexpected error")
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
}

func TestFallbackClient_SPDYSecondarySucceeds(t *testing.T) {
	// Create fake SPDY server. Copy received STDIN data back onto STDOUT stream.
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var stdin, stdout bytes.Buffer
		ctx, err := createHTTPStreams(w, req, &StreamOptions{
			Stdin:  &stdin,
			Stdout: &stdout,
		})
		if err != nil {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		defer ctx.conn.Close()
		_, err = io.Copy(ctx.stdoutStream, ctx.stdinStream)
		if err != nil {
			t.Fatalf("error copying STDIN to STDOUT: %v", err)
		}
	}))
	defer spdyServer.Close()

	spdyLocation, err := url.Parse(spdyServer.URL)
	require.NoError(t, err)
	websocketExecutor, err := NewWebSocketExecutor(&rest.Config{Host: spdyLocation.Host}, "GET", spdyServer.URL)
	require.NoError(t, err)
	spdyExecutor, err := NewSPDYExecutor(&rest.Config{Host: spdyLocation.Host}, "POST", spdyLocation)
	require.NoError(t, err)
	// Always fallback to spdyExecutor, and spdyExecutor succeeds against fake spdy server.
	exec, err := NewFallbackExecutor(websocketExecutor, spdyExecutor, func(error) bool { return true })
	require.NoError(t, err)
	// Generate random data, and set it up to stream on STDIN. The data will be
	// returned on the STDOUT buffer.
	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	if _, err := rand.Read(randomData); err != nil {
		t.Errorf("unexpected error reading random data: %v", err)
	}
	var stdout bytes.Buffer
	options := &StreamOptions{
		Stdin:  bytes.NewReader(randomData),
		Stdout: &stdout,
	}
	errorChan := make(chan error)
	go func() {
		errorChan <- exec.StreamWithContext(context.Background(), *options)
	}()

	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("expect stream to be closed after connection is closed.")
	case err := <-errorChan:
		if err != nil {
			t.Errorf("unexpected error")
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
}

func TestFallbackClient_PrimaryAndSecondaryFail(t *testing.T) {
	// Create fake WebSocket server. Copy received STDIN data back onto STDOUT stream.
	websocketServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		conns, err := webSocketServerStreams(req, w, streamOptionsFromRequest(req))
		if err != nil {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		defer conns.conn.Close()
		// Loopback the STDIN stream onto the STDOUT stream.
		_, err = io.Copy(conns.stdoutStream, conns.stdinStream)
		require.NoError(t, err)
	}))
	defer websocketServer.Close()

	// Now create the fallback client (executor), and point it to the "websocketServer".
	// Must add STDIN and STDOUT query params for the client request.
	websocketServer.URL = websocketServer.URL + "?" + "stdin=true" + "&" + "stdout=true"
	websocketLocation, err := url.Parse(websocketServer.URL)
	require.NoError(t, err)
	websocketExecutor, err := NewWebSocketExecutor(&rest.Config{Host: websocketLocation.Host}, "GET", websocketServer.URL)
	require.NoError(t, err)
	spdyExecutor, err := NewSPDYExecutor(&rest.Config{Host: websocketLocation.Host}, "POST", websocketLocation)
	require.NoError(t, err)
	// Always fallback to spdyExecutor, but spdyExecutor fails against websocket server.
	exec, err := NewFallbackExecutor(websocketExecutor, spdyExecutor, func(error) bool { return true })
	require.NoError(t, err)
	// Update the websocket executor to request remote command v4, which is unsupported.
	fallbackExec, ok := exec.(*FallbackExecutor)
	assert.True(t, ok, "error casting executor as FallbackExecutor")
	websocketExec, ok := fallbackExec.primary.(*wsStreamExecutor)
	assert.True(t, ok, "error casting executor as websocket executor")
	// Set the attempted subprotocol version to V4; websocket server only accepts V5.
	websocketExec.protocols = []string{remotecommand.StreamProtocolV4Name}

	// Generate random data, and set it up to stream on STDIN. The data will be
	// returned on the STDOUT buffer.
	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	if _, err := rand.Read(randomData); err != nil {
		t.Errorf("unexpected error reading random data: %v", err)
	}
	var stdout bytes.Buffer
	options := &StreamOptions{
		Stdin:  bytes.NewReader(randomData),
		Stdout: &stdout,
	}
	errorChan := make(chan error)
	go func() {
		errorChan <- exec.StreamWithContext(context.Background(), *options)
	}()

	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("expect stream to be closed after connection is closed.")
	case err := <-errorChan:
		// Ensure secondary executor returned an error.
		require.Error(t, err)
	}
}
