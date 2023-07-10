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
	"crypto/tls"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
	rctesting "k8s.io/client-go/tools/remotecommand/testing"
)

// TestWebSocketStreamTranslator_LoopbackStdinToStdout returns random data sent on the client's
// STDIN channel back onto the client's STDOUT channel. There are two servers in this test: the
// upstream fake SPDY server, and the StreamTranslator server. The StreamTranslator proxys the
// data received from the websocket client upstream to the SPDY server (by translating the
// websocket data into spdy). The returned data read on the websocket client STDOUT is then
// compared the random data sent on STDIN to ensure they are the same.
func TestWebSocketStreamTranslator_LoopbackStdinToStdout(t *testing.T) {
	// Create upstream fake SPDY server which copies STDIN back onto STDOUT stream.
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, err := rctesting.CreateHTTPStreams(w, req, rctesting.Options{
			Stdin:  true,
			Stdout: true,
		})
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.Conn.Close()
		// Loopback STDIN data onto STDOUT stream.
		io.Copy(ctx.StdoutStream, ctx.StdinStream)
	}))
	defer spdyServer.Close()
	// Create StreamTranslatorHandler, which points upstream to fake SPDY server with
	// streams STDIN and STDOUT. Create test server from StreamTranslatorHandler.
	spdyLocation, err := url.Parse(spdyServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse spdy server URL: %s", spdyServer.URL)
	}
	spdyTransport := spdy.NewRoundTripper(&tls.Config{InsecureSkipVerify: true})
	streams := Options{Stdin: true, Stdout: true}
	streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, nil, streams)
	streamTranslatorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		streamTranslator.ServeHTTP(w, req)
	}))
	defer streamTranslatorServer.Close()
	// Now create the websocket client (executor), and point it to the "streamTranslatorServer".
	streamTranslatorLocation, err := url.Parse(streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse StreamTranslator server URL: %s", streamTranslatorServer.URL)
	}
	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: streamTranslatorLocation.Host}, "POST", streamTranslatorServer.URL)
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
			t.Errorf("unexpected error")
		}
	}
	data, err := ioutil.ReadAll(bytes.NewReader(stdout.Bytes()))
	if err != nil {
		t.Errorf("error reading the stream: %v", err)
		return
	}
	// Check the random data sent on STDIN was the same returned on STDOUT.
	if !bytes.Equal(randomData, data) {
		t.Errorf("unexpected data received: %d sent: %d", len(data), len(randomData))
	}
}

// TestWebSocketStreamTranslator_LoopbackStdinToStderr returns random data sent on the client's
// STDIN channel back onto the client's STDERR channel. There are two servers in this test: the
// upstream fake SPDY server, and the StreamTranslator server. The StreamTranslator proxys the
// data received from the websocket client upstream to the SPDY server (by translating the
// websocket data into spdy). The returned data read on the websocket client STDERR is then
// compared the random data sent on STDIN to ensure they are the same.
func TestWebSocketStreamTranslator_LoopbackStdinToStderr(t *testing.T) {
	// Create upstream fake SPDY server which copies STDIN back onto STDERR stream.
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, err := rctesting.CreateHTTPStreams(w, req, rctesting.Options{
			Stdin:  true,
			Stderr: true,
		})
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.Conn.Close()
		// Loopback STDIN data onto STDERR stream.
		io.Copy(ctx.StderrStream, ctx.StdinStream)
	}))
	defer spdyServer.Close()
	// Create StreamTranslatorHandler, which points upstream to fake SPDY server with
	// streams STDIN and STDERR. Create test server from StreamTranslatorHandler.
	spdyLocation, err := url.Parse(spdyServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse spdy server URL: %s", spdyServer.URL)
	}
	spdyTransport := spdy.NewRoundTripper(&tls.Config{InsecureSkipVerify: true})
	streams := Options{Stdin: true, Stderr: true}
	streamTranslator := NewStreamTranslatorHandler(spdyLocation, spdyTransport, nil, streams)
	streamTranslatorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		streamTranslator.ServeHTTP(w, req)
	}))
	defer streamTranslatorServer.Close()
	// Now create the websocket client (executor), and point it to the "streamTranslatorServer".
	streamTranslatorLocation, err := url.Parse(streamTranslatorServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse StreamTranslator server URL: %s", streamTranslatorServer.URL)
	}
	exec, err := remotecommand.NewWebSocketExecutor(&rest.Config{Host: streamTranslatorLocation.Host}, "POST", streamTranslatorServer.URL)
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
			t.Errorf("unexpected error")
		}
	}
	data, err := ioutil.ReadAll(bytes.NewReader(stderr.Bytes()))
	if err != nil {
		t.Errorf("error reading the stream: %v", err)
		return
	}
	// Check the random data sent on STDIN was the same returned on STDERR.
	if !bytes.Equal(randomData, data) {
		t.Errorf("unexpected data received: %d sent: %d", len(data), len(randomData))
	}
}
