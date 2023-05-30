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
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
)

func TestWebSocketStreamTranslator(t *testing.T) {
	// Final upstream server is SPDY server. This server only copies the STDIN to the STDOUT, so
	// the same data sent upstream on STDIN, should be returned on STDOUT.
	spdyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var stdin, stdout bytes.Buffer
		ctx, err := createHTTPStreams(w, req, &StreamOptions{
			Stdin:  &stdin,
			Stdout: &stdout,
		})
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.conn.Close()

		io.Copy(ctx.stdoutStream, ctx.stdinStream)
	}))
	defer spdyServer.Close()

	// Parse a URL to the SPDY server. Used as target of the StreamTranslator proxy.
	spdyLocation, err := url.Parse(spdyServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse SPDY server URL: %s", spdyServer.URL)
	}
	spdyRoundTripper := spdy.NewRoundTripper(nil)
	// Create the StreamTranslator proxy pointing upstream to the SPDY server, and have
	// it handle requests of the "proxyServer". NOTE: nil ErrorResponder not needed.
	proxy := proxy.NewStreamTranslatorHandler(spdyLocation, spdyRoundTripper, nil)
	proxyServer := httptest.NewServer(proxy)
	proxyUri, err := url.Parse(proxyServer.URL)
	if err != nil {
		t.Fatalf("Unable to parse proxy server URL: %s", proxyServer.URL)
	}
	// Now create the WebSocket client (executor), and point it to the "proxyServer".
	// Must add STDIN and STDOUT query params for the WebSocket client request.
	proxyServer.URL = proxyServer.URL + "?" + "stdin=true" + "&" + "stdout=true"
	exec := NewWebSocketExecutor(&rest.Config{Host: proxyUri.Host}, "POST", proxyServer.URL)
	// Generate random data, and set it up to stream on STDIN. The data will be
	// returned on the STDOUT buffer.
	randomData := make([]byte, 1024*1024)
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
	data, err := ioutil.ReadAll(bytes.NewReader(stdout.Bytes()))
	if err != nil {
		t.Errorf("error reading the stream: %v", err)
		return
	}
	// Check the data sent on STDIN was the same returned on STDOUT.
	if !bytes.Equal(randomData, data) {
		t.Errorf("unexpected data received: %d sent: %d", len(data), len(randomData))
	}

}
