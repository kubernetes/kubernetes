//go:build go1.27

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
	"strconv"
	"testing"
	"time"

	"golang.org/x/net/http2"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	certutil "k8s.io/client-go/util/cert"
	netutils "k8s.io/utils/net"
)

func TestSecureServingIdentityHeaders(t *testing.T) {
	certPEM, keyPEM, err := certutil.GenerateSelfSignedCertKey("localhost", []net.IP{netutils.ParseIPSloppy("127.0.0.1")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	cert, err := dynamiccertificates.NewStaticCertKeyContent("serving-cert", certPEM, keyPEM)
	if err != nil {
		t.Fatal(err)
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(certPEM) {
		t.Fatal("failed to add serving certificate to trust roots")
	}

	for _, tc := range []struct {
		name         string
		disableHTTP2 bool
		protocol     int
	}{
		{name: "HTTP/1.1", disableHTTP2: true, protocol: 1},
		{name: "HTTP/2", protocol: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			stopCh := make(chan struct{})
			serving := &SecureServingInfo{Listener: listener, Cert: cert, DisableHTTP2: tc.disableHTTP2}
			stoppedCh, listenerStoppedCh, err := serving.Serve(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Group-Count", strconv.Itoa(len(r.Header.Values("Impersonate-Group"))))
				w.WriteHeader(http.StatusOK)
			}), 5*time.Second, stopCh)
			if err != nil {
				_ = listener.Close()
				t.Fatal(err)
			}
			t.Cleanup(func() {
				close(stopCh)
				<-listenerStoppedCh
				<-stoppedCh
			})

			transport := &http.Transport{
				TLSClientConfig:   &tls.Config{RootCAs: roots},
				ForceAttemptHTTP2: !tc.disableHTTP2,
			}
			defer transport.CloseIdleConnections()
			client := &http.Client{Transport: transport, Timeout: 10 * time.Second}
			for _, count := range []int{1000, maxHeaderValueCount - 50, maxHeaderValueCount + 1} {
				req, err := http.NewRequest(http.MethodGet, "https://"+listener.Addr().String(), nil)
				if err != nil {
					t.Fatal(err)
				}
				for i := range count {
					req.Header.Add("Impersonate-Group", fmt.Sprintf("group-%d", i))
				}
				resp, err := client.Do(req)
				if err != nil {
					t.Fatal(err)
				}
				if err := resp.Body.Close(); err != nil {
					t.Fatal(err)
				}
				if resp.ProtoMajor != tc.protocol {
					t.Errorf("%d headers: protocol = %s, want HTTP/%d", count, resp.Proto, tc.protocol)
				}
				want := http.StatusOK
				if count > maxHeaderValueCount {
					want = http.StatusRequestHeaderFieldsTooLarge
				}
				if resp.StatusCode != want {
					t.Errorf("%d headers: status = %d, want %d", count, resp.StatusCode, want)
				}
				if want == http.StatusOK {
					if got := resp.Header.Get("Group-Count"); got != strconv.Itoa(count) {
						t.Errorf("%d headers: handler received %s group header values", count, got)
					}
				}
			}
		})
	}
}

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

	certPEM, keyPEM, err := certutil.GenerateSelfSignedCertKey("127.0.0.1", []net.IP{netutils.ParseIPSloppy("127.0.0.1")}, nil)
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
