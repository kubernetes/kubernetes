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

package server

import (
	"bufio"
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"net/http"
	"testing"
	"time"

	"golang.org/x/net/http2"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	"k8s.io/client-go/util/cert"
	netutils "k8s.io/utils/net"
)

// TestSecureServingWriteByteTimeoutDoesNotAffectStreamingResponses verifies that
// configuring HTTP2WriteByteTimeout does not break a watch-like streaming response
// whose bytes keep flowing. The write-byte timeout only runs while a write is
// actively blocked on the socket; periods with no pending write - the common case
// for a quiet watch - must not trip it. Here the handler streams several chunks
// separated by idle gaps that are deliberately longer than the timeout, and the
// client reads each chunk as it arrives, so every individual write completes well
// within the timeout. The connection must survive and deliver all chunks.
//
// This exercises the real SecureServingInfo.Serve path (and therefore the
// http2.Server wiring in secure_serving.go) rather than an httptest server, which
// would bypass that configuration.
func TestSecureServingWriteByteTimeoutDoesNotAffectStreamingResponses(t *testing.T) {
	const (
		writeByteTimeout = 250 * time.Millisecond
		// idle gap between chunks, larger than the timeout to prove that an idle
		// connection with no in-flight write is not closed.
		idleGap = 3 * writeByteTimeout
		chunks  = 3
	)

	certPEM, keyPEM, err := cert.GenerateSelfSignedCertKey("127.0.0.1", []net.IP{netutils.ParseIPSloppy("127.0.0.1")}, nil)
	if err != nil {
		t.Fatalf("failed to generate self-signed cert: %v", err)
	}
	servingCert, err := dynamiccertificates.NewStaticCertKeyContent("serving-cert", certPEM, keyPEM)
	if err != nil {
		t.Fatalf("failed to load serving cert: %v", err)
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}

	info := &SecureServingInfo{
		Listener:              ln,
		Cert:                  servingCert,
		HTTP2WriteByteTimeout: writeByteTimeout,
	}

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Errorf("expected the ResponseWriter to implement http.Flusher")
			return
		}
		w.WriteHeader(http.StatusOK)
		flusher.Flush()
		for i := range chunks {
			if i > 0 {
				time.Sleep(idleGap)
			}
			if _, err := fmt.Fprintf(w, "chunk-%d\n", i); err != nil {
				t.Errorf("unexpected error writing chunk %d: %v", i, err)
				return
			}
			flusher.Flush()
		}
	})

	stopCh := make(chan struct{})
	stoppedCh, _, err := info.Serve(handler, 10*time.Second, stopCh)
	if err != nil {
		t.Fatalf("failed to start serving: %v", err)
	}
	defer func() {
		close(stopCh)
		<-stoppedCh
	}()

	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(certPEM) {
		t.Fatal("failed to add serving cert to the client trust pool")
	}
	transport := &http.Transport{TLSClientConfig: &tls.Config{RootCAs: roots}}
	if err := http2.ConfigureTransport(transport); err != nil {
		t.Fatalf("failed to configure http/2 transport: %v", err)
	}
	client := &http.Client{Transport: transport}

	// A generous overall deadline (far above chunks*idleGap) keeps the test
	// flake-free under CI load while still failing fast if the stream stalls.
	ctx, cancel := context.WithTimeout(context.Background(), wait.ForeverTestTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("https://%s/watch", ln.Addr().String()), nil)
	if err != nil {
		t.Fatalf("failed to build request: %v", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.ProtoMajor != 2 {
		t.Errorf("expected an HTTP/2 response, got %s", resp.Proto)
	}

	// Reading every chunk to completion confirms the connection was not closed
	// by the write-byte timeout while the stream was idle between writes.
	scanner := bufio.NewScanner(resp.Body)
	got := 0
	for scanner.Scan() {
		got++
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("unexpected error reading the streamed response: %v", err)
	}
	if got != chunks {
		t.Errorf("expected to read %d streamed chunks, got %d", chunks, got)
	}
}
