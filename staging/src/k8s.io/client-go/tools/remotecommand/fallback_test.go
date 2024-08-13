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
	"crypto/tls"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/httpstream"
	utilnettesting "k8s.io/apimachinery/pkg/util/net/testing"
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

// localhostCert was generated from crypto/tls/generate_cert.go with the following command:
//
//	go run generate_cert.go  --rsa-bits 2048 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var localhostCert = []byte(`-----BEGIN CERTIFICATE-----
MIIDGTCCAgGgAwIBAgIRALL5AZcefF4kkYV1SEG6YrMwDQYJKoZIhvcNAQELBQAw
EjEQMA4GA1UEChMHQWNtZSBDbzAgFw03MDAxMDEwMDAwMDBaGA8yMDg0MDEyOTE2
MDAwMFowEjEQMA4GA1UEChMHQWNtZSBDbzCCASIwDQYJKoZIhvcNAQEBBQADggEP
ADCCAQoCggEBALQ/FHcyVwdFHxARbbD2KBtDUT7Eni+8ioNdjtGcmtXqBv45EC1C
JOqqGJTroFGJ6Q9kQIZ9FqH5IJR2fOOJD9kOTueG4Vt1JY1rj1Kbpjefu8XleZ5L
SBwIWVnN/lEsEbuKmj7N2gLt5AH3zMZiBI1mg1u9Z5ZZHYbCiTpBrwsq6cTlvR9g
dyo1YkM5hRESCzsrL0aUByoo0qRMD8ZsgANJwgsiO0/M6idbxDwv1BnGwGmRYvOE
Hxpy3v0Jg7GJYrvnpnifJTs4nw91N5X9pXxR7FFzi/6HTYDWRljvTb0w6XciKYAz
bWZ0+cJr5F7wB7ovlbm7HrQIR7z7EIIu2d8CAwEAAaNoMGYwDgYDVR0PAQH/BAQD
AgKkMBMGA1UdJQQMMAoGCCsGAQUFBwMBMA8GA1UdEwEB/wQFMAMBAf8wLgYDVR0R
BCcwJYILZXhhbXBsZS5jb22HBH8AAAGHEAAAAAAAAAAAAAAAAAAAAAEwDQYJKoZI
hvcNAQELBQADggEBAFPPWopNEJtIA2VFAQcqN6uJK+JVFOnjGRoCrM6Xgzdm0wxY
XCGjsxY5dl+V7KzdGqu858rCaq5osEBqypBpYAnS9C38VyCDA1vPS1PsN8SYv48z
DyBwj+7R2qar0ADBhnhWxvYO9M72lN/wuCqFKYMeFSnJdQLv3AsrrHe9lYqOa36s
8wxSwVTFTYXBzljPEnSaaJMPqFD8JXaZK1ryJPkO5OsCNQNGtatNiWAf3DcmwHAT
MGYMzP0u4nw47aRz9shB8w+taPKHx2BVwE1m/yp3nHVioOjXqA1fwRQVGclCJSH1
D2iq3hWVHRENgjTjANBPICLo9AZ4JfN6PH19mnU=
-----END CERTIFICATE-----`)

// localhostKey is the private key for localhostCert.
var localhostKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEAtD8UdzJXB0UfEBFtsPYoG0NRPsSeL7yKg12O0Zya1eoG/jkQ
LUIk6qoYlOugUYnpD2RAhn0WofkglHZ844kP2Q5O54bhW3UljWuPUpumN5+7xeV5
nktIHAhZWc3+USwRu4qaPs3aAu3kAffMxmIEjWaDW71nllkdhsKJOkGvCyrpxOW9
H2B3KjViQzmFERILOysvRpQHKijSpEwPxmyAA0nCCyI7T8zqJ1vEPC/UGcbAaZFi
84QfGnLe/QmDsYliu+emeJ8lOzifD3U3lf2lfFHsUXOL/odNgNZGWO9NvTDpdyIp
gDNtZnT5wmvkXvAHui+VubsetAhHvPsQgi7Z3wIDAQABAoIBAGmw93IxjYCQ0ncc
kSKMJNZfsdtJdaxuNRZ0nNNirhQzR2h403iGaZlEpmdkhzxozsWcto1l+gh+SdFk
bTUK4MUZM8FlgO2dEqkLYh5BcMT7ICMZvSfJ4v21E5eqR68XVUqQKoQbNvQyxFk3
EddeEGdNrkb0GDK8DKlBlzAW5ep4gjG85wSTjR+J+muUv3R0BgLBFSuQnIDM/IMB
LWqsja/QbtB7yppe7jL5u8UCFdZG8BBKT9fcvFIu5PRLO3MO0uOI7LTc8+W1Xm23
uv+j3SY0+v+6POjK0UlJFFi/wkSPTFIfrQO1qFBkTDQHhQ6q/7GnILYYOiGbIRg2
NNuP52ECgYEAzXEoy50wSYh8xfFaBuxbm3ruuG2W49jgop7ZfoFrPWwOQKAZS441
VIwV4+e5IcA6KkuYbtGSdTYqK1SMkgnUyD/VevwAqH5TJoEIGu0pDuKGwVuwqioZ
frCIAV5GllKyUJ55VZNbRr2vY2fCsWbaCSCHETn6C16DNuTCe5C0JBECgYEA4JqY
5GpNbMG8fOt4H7hU0Fbm2yd6SHJcQ3/9iimef7xG6ajxsYrIhg1ft+3IPHMjVI0+
9brwHDnWg4bOOx/VO4VJBt6Dm/F33bndnZRkuIjfSNpLM51P+EnRdaFVHOJHwKqx
uF69kihifCAG7YATgCveeXImzBUSyZUz9UrETu8CgYARNBimdFNG1RcdvEg9rC0/
p9u1tfecvNySwZqU7WF9kz7eSonTueTdX521qAHowaAdSpdJMGODTTXaywm6cPhQ
jIfj9JZZhbqQzt1O4+08Qdvm9TamCUB5S28YLjza+bHU7nBaqixKkDfPqzCyilpX
yVGGL8SwjwmN3zop/sQXAQKBgC0JMsESQ6YcDsRpnrOVjYQc+LtW5iEitTdfsaID
iGGKihmOI7B66IxgoCHMTws39wycKdSyADVYr5e97xpR3rrJlgQHmBIrz+Iow7Q2
LiAGaec8xjl6QK/DdXmFuQBKqyKJ14rljFODP4QuE9WJid94bGqjpf3j99ltznZP
4J8HAoGAJb4eb4lu4UGwifDzqfAPzLGCoi0fE1/hSx34lfuLcc1G+LEu9YDKoOVJ
9suOh0b5K/bfEy9KrVMBBriduvdaERSD8S3pkIQaitIz0B029AbE4FLFf9lKQpP2
KR8NJEkK99Vh/tew6jAMll70xFrE7aF8VLXJVE7w4sQzuvHxl9Q=
-----END RSA PRIVATE KEY-----
`)

// See (https://github.com/kubernetes/kubernetes/issues/126134).
func TestFallbackClient_WebSocketHTTPSProxyCausesSPDYFallback(t *testing.T) {
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Errorf("https (valid hostname): proxy_test: %v", err)
	}

	var proxyCalled atomic.Int64
	proxyHandler := utilnettesting.NewHTTPProxyHandler(t, func(req *http.Request) bool {
		proxyCalled.Add(1)
		return true
	})
	defer proxyHandler.Wait()

	proxyServer := httptest.NewUnstartedServer(proxyHandler)
	proxyServer.TLS = &tls.Config{Certificates: []tls.Certificate{cert}}
	proxyServer.StartTLS()
	defer proxyServer.Close() //nolint:errcheck

	proxyLocation, err := url.Parse(proxyServer.URL)
	require.NoError(t, err)

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
		defer ctx.conn.Close() //nolint:errcheck
		_, err = io.Copy(ctx.stdoutStream, ctx.stdinStream)
		if err != nil {
			t.Fatalf("error copying STDIN to STDOUT: %v", err)
		}
	}))
	defer spdyServer.Close() //nolint:errcheck

	backendLocation, err := url.Parse(spdyServer.URL)
	require.NoError(t, err)

	clientConfig := &rest.Config{
		Host:            spdyServer.URL,
		TLSClientConfig: rest.TLSClientConfig{CAData: localhostCert},
		Proxy: func(req *http.Request) (*url.URL, error) {
			return proxyLocation, nil
		},
	}

	// Websocket with https proxy will fail in dialing (falling back to SPDY).
	websocketExecutor, err := NewWebSocketExecutor(clientConfig, "GET", backendLocation.String())
	require.NoError(t, err)
	spdyExecutor, err := NewSPDYExecutor(clientConfig, "POST", backendLocation)
	require.NoError(t, err)
	// Fallback to spdyExecutor with websocket https proxy error; spdyExecutor succeeds against fake spdy server.
	sawHTTPSProxyError := false
	exec, err := NewFallbackExecutor(websocketExecutor, spdyExecutor, func(err error) bool {
		if httpstream.IsUpgradeFailure(err) {
			t.Errorf("saw upgrade failure: %v", err)
			return true
		}
		if httpstream.IsHTTPSProxyError(err) {
			sawHTTPSProxyError = true
			t.Logf("saw https proxy error: %v", err)
			return true
		}
		return false
	})
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

	// Ensure the https proxy error was observed
	if !sawHTTPSProxyError {
		t.Errorf("expected to see https proxy error")
	}
	// Ensure the proxy was called once
	if e, a := int64(1), proxyCalled.Load(); e != a {
		t.Errorf("expected %d proxy call, got %d", e, a)
	}
}
