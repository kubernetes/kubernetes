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

package oidc

import (
	"io"
	"net"
	"net/http"
	"sync/atomic"
	"testing"

	utilnet "k8s.io/apimachinery/pkg/util/net"
)

// NewHTTPConnectProxyHandler returns an http.Handler that implements an HTTP CONNECT proxy.
// When a CONNECT request is received, it dials the target, hijacks the client connection,
// and bidirectionally copies data between them. The called flag is set to true when a
// non-ready request is received.
func NewHTTPConnectProxyHandler(t testing.TB, called *atomic.Bool) http.Handler {
	t.Helper()

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/ready" {
			t.Log("egress proxy ready")
			w.WriteHeader(http.StatusOK)
			return
		}

		called.Store(true)

		if r.Method != http.MethodConnect {
			http.Error(w, "this proxy only supports CONNECT passthrough", http.StatusMethodNotAllowed)
			return
		}

		backendConn, err := (&net.Dialer{}).DialContext(r.Context(), "tcp", r.Host)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer func() { _ = backendConn.Close() }()

		hijacker, ok := w.(http.Hijacker)
		if !ok {
			http.Error(w, "hijacking not supported", http.StatusInternalServerError)
			return
		}

		clientConn, _, err := hijacker.Hijack()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer func() { _ = clientConn.Close() }()

		// use t.Errorf for all errors after this Write since the client may think the connection is good
		_, err = clientConn.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n"))
		if err != nil {
			t.Errorf("unexpected established error: %v", err)
			return
		}

		writerComplete := make(chan struct{})
		readerComplete := make(chan struct{})

		go func() {
			_, err := io.Copy(backendConn, clientConn)
			if err != nil && !utilnet.IsProbableEOF(err) {
				t.Logf("writer error: %v", err)
			}
			close(writerComplete)
		}()

		go func() {
			_, err := io.Copy(clientConn, backendConn)
			if err != nil && !utilnet.IsProbableEOF(err) {
				t.Logf("reader error: %v", err)
			}
			close(readerComplete)
		}()

		// Wait for one half the connection to exit. Once it does,
		// the defer will clean up the other half of the connection.
		select {
		case <-writerComplete:
		case <-readerComplete:
		}
	})
}
