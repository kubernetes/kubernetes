/*
Copyright 2024 The Kubernetes Authors.

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
	"crypto/rand"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	constants "k8s.io/apimachinery/pkg/util/portforward"
	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/registry/rest"
	restconfig "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/portforward"
)

func TestTunnelingHandler_UpgradeStreamingAndTunneling(t *testing.T) {
	// Create fake upstream SPDY server, with channel receiving SPDY streams.
	streamChan := make(chan httpstream.Stream)
	defer close(streamChan)
	stopServerChan := make(chan struct{})
	defer close(stopServerChan)
	// Create fake upstream SPDY server.
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		_, err := httpstream.Handshake(req, w, []string{constants.PortForwardV1Name})
		require.NoError(t, err)
		upgrader := spdy.NewResponseUpgrader()
		conn := upgrader.UpgradeResponse(w, req, justQueueStream(streamChan))
		require.NotNil(t, conn)
		defer conn.Close() //nolint:errcheck
		<-stopServerChan
	}))
	defer spdyServer.Close()
	// Create UpgradeAwareProxy handler, with url/transport pointing to upstream SPDY. Then
	// create TunnelingHandler by injecting upgrade handler. Create TunnelingServer.
	url, err := url.Parse(spdyServer.URL)
	require.NoError(t, err)
	transport, err := fakeTransport()
	require.NoError(t, err)
	upgradeHandler := proxy.NewUpgradeAwareHandler(url, transport, false, true, proxy.NewErrorResponder(&fakeResponder{}))
	tunnelingHandler := NewTunnelingHandler(upgradeHandler)
	tunnelingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		tunnelingHandler.ServeHTTP(w, req)
	}))
	defer tunnelingServer.Close()
	// Create SPDY client connection containing a TunnelingConnection by upgrading
	// a request to TunnelingHandler using new portforward version 2.
	tunnelingURL, err := url.Parse(tunnelingServer.URL)
	require.NoError(t, err)
	dialer, err := portforward.NewSPDYOverWebsocketDialer(tunnelingURL, &restconfig.Config{Host: tunnelingURL.Host})
	require.NoError(t, err)
	spdyClient, protocol, err := dialer.Dial(constants.PortForwardV1Name)
	require.NoError(t, err)
	assert.Equal(t, constants.PortForwardV1Name, protocol)
	defer spdyClient.Close() //nolint:errcheck
	// Create a SPDY client stream, which will queue a SPDY server stream
	// on the stream creation channel. Send random data on the client stream
	// reading off the SPDY server stream, and validating it was tunneled.
	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	_, err = rand.Read(randomData)
	require.NoError(t, err)
	var actual []byte
	go func() {
		clientStream, err := spdyClient.CreateStream(http.Header{})
		require.NoError(t, err)
		_, err = io.Copy(clientStream, bytes.NewReader(randomData))
		require.NoError(t, err)
		clientStream.Close() //nolint:errcheck
	}()
	select {
	case serverStream := <-streamChan:
		actual, err = io.ReadAll(serverStream)
		require.NoError(t, err)
		defer serverStream.Close() //nolint:errcheck
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout waiting for spdy stream to arrive on channel.")
	}
	assert.Equal(t, randomData, actual, "error validating tunneled random data")
}

var expectedContentLengthHeaders = http.Header{
	"Content-Length": []string{"25"},
	"Date":           []string{"Sun, 25 Feb 2024 08:09:25 GMT"},
	"Split-Point":    []string{"split"},
	"X-App-Protocol": []string{"portforward.k8s.io"},
}

const contentLengthHeaders = "HTTP/1.1 400 Error\r\n" +
	"Content-Length: 25\r\n" +
	"Date: Sun, 25 Feb 2024 08:09:25 GMT\r\n" +
	"Split-Point: split\r\n" +
	"X-App-Protocol: portforward.k8s.io\r\n" +
	"\r\n"

const contentLengthBody = "0123456789split0123456789"

var contentLengthHeadersAndBody = contentLengthHeaders + contentLengthBody

var expectedResponseHeaders = http.Header{
	"Date":           []string{"Sun, 25 Feb 2024 08:09:25 GMT"},
	"Split-Point":    []string{"split"},
	"X-App-Protocol": []string{"portforward.k8s.io"},
}

const responseHeaders = "HTTP/1.1 101 Switching Protocols\r\n" +
	"Date: Sun, 25 Feb 2024 08:09:25 GMT\r\n" +
	"Split-Point: split\r\n" +
	"X-App-Protocol: portforward.k8s.io\r\n" +
	"\r\n"

const responseBody = "This is extra split data.\n"

var responseHeadersAndBody = responseHeaders + responseBody

const invalidResponseData = "INVALID/1.1 101 Switching Protocols\r\n" +
	"Date: Sun, 25 Feb 2024 08:09:25 GMT\r\n" +
	"Split-Point: split\r\n" +
	"X-App-Protocol: portforward.k8s.io\r\n" +
	"\r\n"

func TestTunnelingHandler_HeaderInterceptingConn(t *testing.T) {
	// Basic http response is intercepted correctly; no extra data sent to net.Conn.
	t.Run("simple-no-body", func(t *testing.T) {
		testConnConstructor := &mockConnInitializer{mockConn: &mockConn{}, initializeWriteConn: &mockConn{}}
		hic := &headerInterceptingConn{initializableConn: testConnConstructor}
		_, err := hic.Write([]byte(responseHeaders))
		require.NoError(t, err)
		assert.True(t, hic.initialized, "successfully parsed http response headers")
		assert.Equal(t, expectedResponseHeaders, testConnConstructor.resp.Header)
		assert.Equal(t, "101 Switching Protocols", testConnConstructor.resp.Status)
		assert.Equal(t, "portforward.k8s.io", testConnConstructor.resp.Header.Get("X-App-Protocol"))
		assert.Equal(t, responseHeaders, string(testConnConstructor.initializeWriteConn.written), "only headers are written in initializeWrite")
		assert.Equal(t, "", string(testConnConstructor.mockConn.written))
	})

	// Extra data after response headers should be sent to net.Conn.
	t.Run("simple-single-write", func(t *testing.T) {
		testConnConstructor := &mockConnInitializer{mockConn: &mockConn{}, initializeWriteConn: &mockConn{}}
		hic := &headerInterceptingConn{initializableConn: testConnConstructor}
		_, err := hic.Write([]byte(responseHeadersAndBody))
		require.NoError(t, err)
		assert.True(t, hic.initialized)
		assert.Equal(t, expectedResponseHeaders, testConnConstructor.resp.Header)
		assert.Equal(t, "101 Switching Protocols", testConnConstructor.resp.Status)
		assert.Equal(t, responseHeaders, string(testConnConstructor.initializeWriteConn.written), "only headers are written in initializeWrite")
		assert.Equal(t, responseBody, string(testConnConstructor.mockConn.written), "extra data written to net.Conn")
	})

	// Partially written headers are buffered and decoded
	t.Run("simple-byte-by-byte", func(t *testing.T) {
		testConnConstructor := &mockConnInitializer{mockConn: &mockConn{}, initializeWriteConn: &mockConn{}}
		hic := &headerInterceptingConn{initializableConn: testConnConstructor}
		// write one byte at a time
		for _, b := range []byte(responseHeadersAndBody) {
			_, err := hic.Write([]byte{b})
			require.NoError(t, err)
		}
		assert.True(t, hic.initialized)
		assert.Equal(t, expectedResponseHeaders, testConnConstructor.resp.Header)
		assert.Equal(t, "101 Switching Protocols", testConnConstructor.resp.Status)
		assert.Equal(t, responseHeaders, string(testConnConstructor.initializeWriteConn.written), "only headers are written in initializeWrite")
		assert.Equal(t, responseBody, string(testConnConstructor.mockConn.written), "extra data written to net.Conn")
	})

	// Writes spanning the header/body breakpoint are buffered and decoded
	t.Run("simple-span-headerbody", func(t *testing.T) {
		testConnConstructor := &mockConnInitializer{mockConn: &mockConn{}, initializeWriteConn: &mockConn{}}
		hic := &headerInterceptingConn{initializableConn: testConnConstructor}
		// write one chunk at a time
		for i, chunk := range strings.Split(responseHeadersAndBody, "split") {
			if i > 0 {
				n, err := hic.Write([]byte("split"))
				require.Equal(t, n, len("split"))
				require.NoError(t, err)
			}
			n, err := hic.Write([]byte(chunk))
			require.Equal(t, n, len(chunk))
			require.NoError(t, err)
		}
		assert.True(t, hic.initialized)
		assert.Equal(t, expectedResponseHeaders, testConnConstructor.resp.Header)
		assert.Equal(t, "101 Switching Protocols", testConnConstructor.resp.Status)
		assert.Equal(t, responseHeaders, string(testConnConstructor.initializeWriteConn.written), "only headers are written in initializeWrite")
		assert.Equal(t, responseBody, string(testConnConstructor.mockConn.written), "extra data written to net.Conn")
	})

	// Tolerate header separators of \n instead of \r\n, and extra data after response headers should be sent to net.Conn.
	t.Run("simple-tolerate-lf", func(t *testing.T) {
		testConnConstructor := &mockConnInitializer{mockConn: &mockConn{}, initializeWriteConn: &mockConn{}}
		hic := &headerInterceptingConn{initializableConn: testConnConstructor}
		_, err := hic.Write([]byte(strings.ReplaceAll(responseHeadersAndBody, "\r", "")))
		require.NoError(t, err)
		assert.True(t, hic.initialized)
		assert.Equal(t, expectedResponseHeaders, testConnConstructor.resp.Header)
		assert.Equal(t, "101 Switching Protocols", testConnConstructor.resp.Status)
		assert.Equal(t, strings.ReplaceAll(responseHeaders, "\r", ""), string(testConnConstructor.initializeWriteConn.written), "only normalized headers are written in initializeWrite")
		assert.Equal(t, responseBody, string(testConnConstructor.mockConn.written), "extra data written to net.Conn")
	})

	// Content-Length handling
	t.Run("content-length-body", func(t *testing.T) {
		testConnConstructor := &mockConnInitializer{mockConn: &mockConn{}, initializeWriteConn: &mockConn{}}
		hic := &headerInterceptingConn{initializableConn: testConnConstructor}
		_, err := hic.Write([]byte(contentLengthHeadersAndBody))
		require.NoError(t, err)
		assert.True(t, hic.initialized, "successfully parsed http response headers")
		assert.Equal(t, expectedContentLengthHeaders, testConnConstructor.resp.Header)
		assert.Equal(t, "400 Error", testConnConstructor.resp.Status)
		assert.Equal(t, contentLengthHeaders, string(testConnConstructor.initializeWriteConn.written), "headers and content are written in initializeWrite")
		assert.Equal(t, contentLengthBody, string(testConnConstructor.mockConn.written))
	})

	// Content-Length separately written headers and body
	t.Run("content-length-headers-body", func(t *testing.T) {
		testConnConstructor := &mockConnInitializer{mockConn: &mockConn{}, initializeWriteConn: &mockConn{}}
		hic := &headerInterceptingConn{initializableConn: testConnConstructor}
		_, err := hic.Write([]byte(contentLengthHeaders))
		require.NoError(t, err)
		_, err = hic.Write([]byte(contentLengthBody))
		require.NoError(t, err)
		assert.True(t, hic.initialized, "successfully parsed http response headers")
		assert.Equal(t, expectedContentLengthHeaders, testConnConstructor.resp.Header)
		assert.Equal(t, "400 Error", testConnConstructor.resp.Status)
		assert.Equal(t, contentLengthHeaders, string(testConnConstructor.initializeWriteConn.written), "headers and content are written in initializeWrite")
		assert.Equal(t, contentLengthBody, string(testConnConstructor.mockConn.written))
	})

	// Content-Length accumulating byte-by-byte
	t.Run("content-length-byte-by-byte", func(t *testing.T) {
		testConnConstructor := &mockConnInitializer{mockConn: &mockConn{}, initializeWriteConn: &mockConn{}}
		hic := &headerInterceptingConn{initializableConn: testConnConstructor}
		for _, b := range []byte(contentLengthHeadersAndBody) {
			_, err := hic.Write([]byte{b})
			require.NoError(t, err)
		}
		assert.True(t, hic.initialized, "successfully parsed http response headers")
		assert.Equal(t, expectedContentLengthHeaders, testConnConstructor.resp.Header)
		assert.Equal(t, "400 Error", testConnConstructor.resp.Status)
		assert.Equal(t, contentLengthHeaders, string(testConnConstructor.initializeWriteConn.written), "headers and content are written in initializeWrite")
		assert.Equal(t, contentLengthBody, string(testConnConstructor.mockConn.written))
	})

	// Content-Length writes spanning headers / body
	t.Run("content-length-span-headerbody", func(t *testing.T) {
		testConnConstructor := &mockConnInitializer{mockConn: &mockConn{}, initializeWriteConn: &mockConn{}}
		hic := &headerInterceptingConn{initializableConn: testConnConstructor}
		// write one chunk at a time
		for i, chunk := range strings.Split(contentLengthHeadersAndBody, "split") {
			if i > 0 {
				n, err := hic.Write([]byte("split"))
				require.Equal(t, n, len("split"))
				require.NoError(t, err)
			}
			n, err := hic.Write([]byte(chunk))
			require.Equal(t, n, len(chunk))
			require.NoError(t, err)
		}
		assert.True(t, hic.initialized, "successfully parsed http response headers")
		assert.Equal(t, expectedContentLengthHeaders, testConnConstructor.resp.Header)
		assert.Equal(t, "400 Error", testConnConstructor.resp.Status)
		assert.Equal(t, contentLengthHeaders, string(testConnConstructor.initializeWriteConn.written), "headers and content are written in initializeWrite")
		assert.Equal(t, contentLengthBody, string(testConnConstructor.mockConn.written))
	})

	// Invalid response returns error.
	t.Run("invalid-single-write", func(t *testing.T) {
		testConnConstructor := &mockConnInitializer{mockConn: &mockConn{}, initializeWriteConn: &mockConn{}}
		hic := &headerInterceptingConn{initializableConn: testConnConstructor}
		_, err := hic.Write([]byte(invalidResponseData))
		assert.Error(t, err, "expected error from invalid http response")
	})

	// Invalid response written byte by byte returns error.
	t.Run("invalid-byte-by-byte", func(t *testing.T) {
		testConnConstructor := &mockConnInitializer{mockConn: &mockConn{}, initializeWriteConn: &mockConn{}}
		hic := &headerInterceptingConn{initializableConn: testConnConstructor}
		var err error
		for _, b := range []byte(invalidResponseData) {
			_, err = hic.Write([]byte{b})
			if err != nil {
				break
			}
		}
		assert.Error(t, err, "expected error from invalid http response")
	})
}

type mockConnInitializer struct {
	resp                *http.Response
	initializeWriteConn *mockConn
	*mockConn
}

func (m *mockConnInitializer) InitializeWrite(backendResponse *http.Response, backendResponseBytes []byte) error {
	m.resp = backendResponse
	_, err := m.initializeWriteConn.Write(backendResponseBytes)
	return err
}

// mockConn implements "net.Conn" interface.
var _ net.Conn = &mockConn{}

type mockConn struct {
	written []byte
}

func (mc *mockConn) Write(p []byte) (int, error) {
	mc.written = append(mc.written, p...)
	return len(p), nil
}

func (mc *mockConn) Read(p []byte) (int, error)         { return 0, nil }
func (mc *mockConn) Close() error                       { return nil }
func (mc *mockConn) LocalAddr() net.Addr                { return &net.TCPAddr{} }
func (mc *mockConn) RemoteAddr() net.Addr               { return &net.TCPAddr{} }
func (mc *mockConn) SetDeadline(t time.Time) error      { return nil }
func (mc *mockConn) SetReadDeadline(t time.Time) error  { return nil }
func (mc *mockConn) SetWriteDeadline(t time.Time) error { return nil }

// fakeResponder implements "rest.Responder" interface.
var _ rest.Responder = &fakeResponder{}

type fakeResponder struct{}

func (fr *fakeResponder) Object(statusCode int, obj runtime.Object) {}
func (fr *fakeResponder) Error(err error)                           {}

// justQueueStream skips the usual stream validation before
// queueing the stream on the stream channel.
func justQueueStream(streams chan httpstream.Stream) func(httpstream.Stream, <-chan struct{}) error {
	return func(stream httpstream.Stream, replySent <-chan struct{}) error {
		streams <- stream
		return nil
	}
}
