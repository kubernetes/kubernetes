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
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	constants "k8s.io/apimachinery/pkg/util/portforward"
	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/util/proxy/metrics"
	restconfig "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/portforward"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestTunnelingHandler_UpgradeStreamingAndTunneling(t *testing.T) {
	metrics.ResetForTest()
	t.Cleanup(metrics.ResetForTest)
	handlerPath := "/TestTunnelingHandler_UpgradeStreamingAndTunneling"

	// 1. Configure and register the upstream SPDY handler for this test.
	streamChan := make(chan httpstream.Stream)
	defer close(streamChan)
	stopSPDYServerChan := make(chan struct{})
	defer close(stopSPDYServerChan)
	spdyServerMux.HandleFunc(handlerPath, func(w http.ResponseWriter, req *http.Request) {
		_, err := httpstream.Handshake(req, w, []string{constants.PortForwardV1Name})
		if err != nil {
			t.Errorf("unexpected error %v", err)
			return
		}
		upgrader := spdy.NewResponseUpgrader()
		conn := upgrader.UpgradeResponse(w, req, justQueueStream(streamChan))
		if conn == nil {
			t.Error("connect is unexpected nil")
			return
		}
		defer conn.Close() //nolint:errcheck
		<-stopSPDYServerChan
	})

	// 2. Configure and register the TunnelingHandler for this test.
	upstreamURL := *spdyServerURL
	upstreamURL.Path = handlerPath
	transport, err := fakeTransport()
	require.NoError(t, err)
	upgradeHandler := proxy.NewUpgradeAwareHandler(&upstreamURL, transport, false, true, proxy.NewErrorResponder(&fakeResponder{}))
	tunnelingHandler := NewTunnelingHandler(upgradeHandler)
	tunnelingServerMux.HandleFunc(handlerPath, tunnelingHandler.ServeHTTP)

	// 3. Configure the client to connect to the tunneling server.
	clientURL := *tunnelingServerURL
	clientURL.Path = handlerPath
	dialer, err := portforward.NewSPDYOverWebsocketDialer(&clientURL, &restconfig.Config{Host: clientURL.Host})
	require.NoError(t, err)
	spdyClient, protocol, err := dialer.Dial(constants.PortForwardV1Name)
	require.NoError(t, err)
	assert.Equal(t, constants.PortForwardV1Name, protocol)
	defer spdyClient.Close() //nolint:errcheck

	// 4. Execute the test logic:
	// Create a SPDY client stream, which will queue a SPDY server stream
	// on the stream creation channel. Send random data on the client stream,
	// reading it off the SPDY server stream, and validating it was tunneled correctly.
	randomSize := 1024 * 1024
	randomData := make([]byte, randomSize)
	_, err = rand.Read(randomData)
	require.NoError(t, err)
	var actual []byte
	go func() {
		clientStream, err := spdyClient.CreateStream(http.Header{})
		if err != nil {
			t.Errorf("unexpected error %v", err)
			return
		}
		_, err = io.Copy(clientStream, bytes.NewReader(randomData))
		if err != nil {
			t.Errorf("unexpected error %v", err)
			return
		}
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

	// 5. Validate metrics.
	metricNames := []string{"apiserver_stream_tunnel_requests_total"}
	expected := `
# HELP apiserver_stream_tunnel_requests_total [ALPHA] Total number of requests that were handled by the StreamTunnelProxy, which processes streaming PortForward/V2
# TYPE apiserver_stream_tunnel_requests_total counter
apiserver_stream_tunnel_requests_total{code="101"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...); err != nil {
		t.Fatal(err)
	}
}

func TestTunnelingHandler_BadRequestWithoutProtcols(t *testing.T) {
	metrics.ResetForTest()
	t.Cleanup(metrics.ResetForTest)
	handlerPath := "/TestTunnelingHandler_BadRequestWithoutProtcols"

	// An error should be returned before proxying to an upstream SPDY server,
	// so a test SPDY server is not needed for this test.
	transport, err := fakeTransport()
	require.NoError(t, err)
	upgradeHandler := proxy.NewUpgradeAwareHandler(spdyServerURL, transport, false, true, proxy.NewErrorResponder(&fakeResponder{}))
	tunnelingHandler := NewTunnelingHandler(upgradeHandler)
	tunnelingServerMux.HandleFunc(handlerPath, tunnelingHandler.ServeHTTP)

	clientURL := *tunnelingServerURL
	clientURL.Path = handlerPath
	dialer, err := portforward.NewSPDYOverWebsocketDialer(&clientURL, &restconfig.Config{Host: clientURL.Host})
	require.NoError(t, err)

	// Request without subprotocols, causing a bad request to be returned.
	_, protocol, err := dialer.Dial("")
	require.Error(t, err)
	assert.Equal(t, "", protocol)

	// Validate the streamtunnel metrics; should be one 400 failure.
	metricNames := []string{"apiserver_stream_tunnel_requests_total"}
	expected := `
# HELP apiserver_stream_tunnel_requests_total [ALPHA] Total number of requests that were handled by the StreamTunnelProxy, which processes streaming PortForward/V2
# TYPE apiserver_stream_tunnel_requests_total counter
apiserver_stream_tunnel_requests_total{code="400"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...); err != nil {
		t.Fatal(err)
	}
}

func TestTunnelingHandler_BadHandshakeError(t *testing.T) {
	metrics.ResetForTest()
	t.Cleanup(metrics.ResetForTest)
	handlerPath := "/TestTunnelingHandler_BadHandshakeError"

	// Create fake upstream SPDY server that returns a forbidden error for a bad handshake.
	spdyServerMux.HandleFunc(handlerPath, func(w http.ResponseWriter, req *http.Request) {
		_, err := httpstream.Handshake(req, w, []string{constants.PortForwardV1Name})
		if err == nil {
			t.Errorf("handshake should have returned an error %v", err)
			return
		}
		assert.ErrorContains(t, err, "unable to negotiate protocol")
		w.WriteHeader(http.StatusForbidden)
	})

	upstreamURL := *spdyServerURL
	upstreamURL.Path = handlerPath
	transport, err := fakeTransport()
	require.NoError(t, err)
	upgradeHandler := proxy.NewUpgradeAwareHandler(&upstreamURL, transport, false, true, proxy.NewErrorResponder(&fakeResponder{}))
	tunnelingHandler := NewTunnelingHandler(upgradeHandler)
	tunnelingServerMux.HandleFunc(handlerPath, tunnelingHandler.ServeHTTP)

	clientURL := *tunnelingServerURL
	clientURL.Path = handlerPath
	dialer, err := portforward.NewSPDYOverWebsocketDialer(&clientURL, &restconfig.Config{Host: clientURL.Host})
	require.NoError(t, err)

	// Handshake will fail, returning a 400-level response.
	_, protocol, err := dialer.Dial("UNKNOWN_SUBPROTOCOL")
	require.Error(t, err)
	assert.Equal(t, "", protocol)

	// Validate the streamtunnel metrics; should be one 400 failure.
	metricNames := []string{"apiserver_stream_tunnel_requests_total"}
	expected := `
# HELP apiserver_stream_tunnel_requests_total [ALPHA] Total number of requests that were handled by the StreamTunnelProxy, which processes streaming PortForward/V2
# TYPE apiserver_stream_tunnel_requests_total counter
apiserver_stream_tunnel_requests_total{code="400"} 1
`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...); err != nil {
		t.Fatal(err)
	}
}

func TestTunnelingHandler_UpstreamSPDYServerErrorPropagated(t *testing.T) {
	metrics.ResetForTest()
	t.Cleanup(metrics.ResetForTest)

	// Validate that various 500-level errors are propagated and incremented in metrics.
	for statusCode, codeStr := range map[int]string{
		http.StatusInternalServerError: "500",
		http.StatusBadGateway:          "502",
		http.StatusServiceUnavailable:  "503",
	} {
		t.Run(fmt.Sprintf("statusCode=%d", statusCode), func(t *testing.T) {
			handlerPath := fmt.Sprintf("/TestTunnelingHandler_UpstreamSPDYServerErrorPropagated/%d", statusCode)

			// Create fake upstream SPDY server, which returns a 500-level error.
			spdyServerMux.HandleFunc(handlerPath, func(w http.ResponseWriter, req *http.Request) {
				_, err := httpstream.Handshake(req, w, []string{constants.PortForwardV1Name})
				if err != nil {
					t.Errorf("handshake should have succeeded %v", err)
					return
				}
				// Returned status code should be incremented in metrics.
				w.WriteHeader(statusCode)
			})

			upstreamURL := *spdyServerURL
			upstreamURL.Path = handlerPath
			transport, err := fakeTransport()
			require.NoError(t, err)
			upgradeHandler := proxy.NewUpgradeAwareHandler(&upstreamURL, transport, false, true, proxy.NewErrorResponder(&fakeResponder{}))
			tunnelingHandler := NewTunnelingHandler(upgradeHandler)
			tunnelingServerMux.HandleFunc(handlerPath, tunnelingHandler.ServeHTTP)

			clientURL := *tunnelingServerURL
			clientURL.Path = handlerPath
			dialer, err := portforward.NewSPDYOverWebsocketDialer(&clientURL, &restconfig.Config{Host: clientURL.Host})
			require.NoError(t, err)
			_, protocol, err := dialer.Dial(constants.PortForwardV1Name)
			require.Error(t, err)
			assert.Empty(t, protocol)

			// Validate the streamtunnel metrics are incrementing 500-level status codes.
			metricNames := []string{"apiserver_stream_tunnel_requests_total"}
			expected := `
# HELP apiserver_stream_tunnel_requests_total [ALPHA] Total number of requests that were handled by the StreamTunnelProxy, which processes streaming PortForward/V2
# TYPE apiserver_stream_tunnel_requests_total counter
apiserver_stream_tunnel_requests_total{code="` + codeStr + `"} 1
`
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...); err != nil {
				t.Fatal(err)
			}
			metrics.ResetForTest()
		})
	}
}

func TestTunnelingResponseWriter_Hijack(t *testing.T) {
	t.Parallel()
	// Regular hijack returns connection, nil bufio, and no error.
	trw := &tunnelingResponseWriter{conn: &mockConn{}}
	assert.False(t, trw.hijacked, "hijacked field starts false before Hijack()")
	assert.False(t, trw.written, "written field startes false before Hijack()")
	_, _, err := trw.Hijack()
	assert.NoError(t, err, "Hijack() does not return error")
	assert.True(t, trw.hijacked, "hijacked field becomes true after Hijack()")
	assert.False(t, trw.written, "written field stays false after Hijack()")
	// Hijacking after writing to response writer is an error.
	trw = &tunnelingResponseWriter{written: true}
	_, _, err = trw.Hijack()
	assert.Error(t, err, "Hijack after writing to response writer is error")
	assert.ErrorContains(t, err, "connection has already been written to")
	// Hijacking after already hijacked is an error.
	trw = &tunnelingResponseWriter{hijacked: true}
	_, _, err = trw.Hijack()
	assert.Error(t, err, "Hijack after writing to response writer is error")
	assert.ErrorContains(t, err, "connection has already been hijacked")
}

func TestTunnelingResponseWriter_DelegateResponseWriter(t *testing.T) {
	t.Parallel()
	// Validate Header() for delegate response writer.
	expectedHeader := http.Header{}
	expectedHeader.Set("foo", "bar")
	trw := &tunnelingResponseWriter{w: &mockResponseWriter{header: expectedHeader}}
	assert.Equal(t, expectedHeader, trw.Header(), "")
	// Validate Write() for delegate response writer.
	expectedWrite := []byte("this is a test write string")
	assert.False(t, trw.written, "written field is before Write()")
	_, err := trw.Write(expectedWrite)
	assert.NoError(t, err, "No error expected after Write() on tunneling response writer")
	assert.True(t, trw.written, "written field is set after writing to tunneling response writer")
	// Writing to response writer after hijacked is an error.
	trw.hijacked = true
	_, err = trw.Write(expectedWrite)
	assert.Error(t, err, "Writing to ResponseWriter after Hijack() is an error")
	require.ErrorIs(t, err, http.ErrHijacked, "Hijacked error returned if writing after hijacked")
	// Validate WriteHeader().
	trw = &tunnelingResponseWriter{w: &mockResponseWriter{}}
	expectedStatusCode := 201
	assert.False(t, trw.written, "Written field originally false in delegate response writer")
	trw.WriteHeader(expectedStatusCode)
	assert.Equal(t, expectedStatusCode, trw.w.(*mockResponseWriter).statusCode, "Expected written status code is correct")
	assert.True(t, trw.written, "Written field set to true after writing delegate response writer")
	// Response writer already written to does not write status.
	trw = &tunnelingResponseWriter{w: &mockResponseWriter{}}
	trw.written = true
	trw.WriteHeader(expectedStatusCode)
	assert.Equal(t, 0, trw.w.(*mockResponseWriter).statusCode, "No status code for previously written response writer")
	// Hijacked response writer does not write status.
	trw = &tunnelingResponseWriter{w: &mockResponseWriter{}}
	trw.hijacked = true
	trw.WriteHeader(expectedStatusCode)
	assert.Equal(t, 0, trw.w.(*mockResponseWriter).statusCode, "No status code written to hijacked response writer")
	assert.False(t, trw.written, "Hijacked response writer does not write status")
	// Writing "101 Switching Protocols" status is an error, since it should happen via hijacked connection.
	trw = &tunnelingResponseWriter{w: &mockResponseWriter{header: http.Header{}}}
	trw.WriteHeader(http.StatusSwitchingProtocols)
	assert.Equal(t, http.StatusInternalServerError, trw.w.(*mockResponseWriter).statusCode, "Internal server error written")
}

func TestTunnelingWebsocketUpgraderConn_LocalRemoteAddress(t *testing.T) {
	t.Parallel()
	expectedLocalAddr := &net.TCPAddr{
		IP:   net.IPv4(127, 0, 0, 1),
		Port: 80,
	}
	expectedRemoteAddr := &net.TCPAddr{
		IP:   net.IPv4(127, 0, 0, 2),
		Port: 443,
	}
	tc := &tunnelingWebsocketUpgraderConn{
		conn: &mockConn{
			localAddr:  expectedLocalAddr,
			remoteAddr: expectedRemoteAddr,
		},
	}
	assert.Equal(t, expectedLocalAddr, tc.LocalAddr(), "LocalAddr() returns expected TCPAddr")
	assert.Equal(t, expectedRemoteAddr, tc.RemoteAddr(), "RemoteAddr() returns expected TCPAddr")
	// Connection nil, returns empty address
	tc.conn = nil
	assert.Equal(t, noopAddr{}, tc.LocalAddr(), "nil connection, LocalAddr() returns noopAddr")
	assert.Equal(t, noopAddr{}, tc.RemoteAddr(), "nil connection, RemoteAddr() returns noopAddr")
	// Validate the empty strings from noopAddr
	assert.Equal(t, "", noopAddr{}.Network(), "noopAddr Network() returns empty string")
	assert.Equal(t, "", noopAddr{}.String(), "noopAddr String() returns empty string")
}

func TestTunnelingWebsocketUpgraderConn_SetDeadline(t *testing.T) {
	t.Parallel()
	tc := &tunnelingWebsocketUpgraderConn{conn: &mockConn{}}
	expected := time.Now()
	assert.NoError(t, tc.SetDeadline(expected), "SetDeadline does not return error")
	assert.Equal(t, expected, tc.conn.(*mockConn).readDeadline, "SetDeadline() sets read deadline")
	assert.Equal(t, expected, tc.conn.(*mockConn).writeDeadline, "SetDeadline() sets write deadline")
	expected = time.Now()
	assert.NoError(t, tc.SetWriteDeadline(expected), "SetWriteDeadline does not return error")
	assert.Equal(t, expected, tc.conn.(*mockConn).writeDeadline, "Expected write deadline set")
	expected = time.Now()
	assert.NoError(t, tc.SetReadDeadline(expected), "SetReadDeadline does not return error")
	assert.Equal(t, expected, tc.conn.(*mockConn).readDeadline, "Expected read deadline set")
	expectedErr := fmt.Errorf("deadline error")
	tc = &tunnelingWebsocketUpgraderConn{conn: &mockConn{deadlineErr: expectedErr}}
	expected = time.Now()
	actualErr := tc.SetDeadline(expected)
	assert.Equal(t, expectedErr, actualErr, "SetDeadline() expected error returned")
	// Connection nil, returns nil error.
	tc.conn = nil
	assert.NoError(t, tc.SetDeadline(expected), "SetDeadline() with nil connection always returns nil error")
	assert.NoError(t, tc.SetWriteDeadline(expected), "SetWriteDeadline() with nil connection always returns nil error")
	assert.NoError(t, tc.SetReadDeadline(expected), "SetReadDeadline() with nil connection always returns nil error")
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
	t.Parallel()
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
				require.Len(t, "split", n)
				require.NoError(t, err)
			}
			n, err := hic.Write([]byte(chunk))
			require.Len(t, chunk, n)
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
				require.Len(t, "split", n)
				require.NoError(t, err)
			}
			n, err := hic.Write([]byte(chunk))
			require.Len(t, chunk, n)
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
