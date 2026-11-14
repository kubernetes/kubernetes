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

package certificate

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"io"
	"math/big"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/client-go/rest"
	"k8s.io/klog/v2/ktesting"
)

type stubManager struct {
	cert  atomic.Pointer[tls.Certificate]
	reads chan struct{}
}

func (m *stubManager) Start()              {}
func (m *stubManager) Stop()               {}
func (m *stubManager) ServerHealthy() bool { return true }
func (m *stubManager) Current() *tls.Certificate {
	cert := m.cert.Load()
	select {
	case m.reads <- struct{}{}:
	default:
	}
	return cert
}

func newCertificate(t *testing.T, serial int64) *tls.Certificate {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{
		SerialNumber: big.NewInt(serial),
		Subject:      pkix.Name{CommonName: "system:node:bootstrap-repro"},
		NotBefore:    time.Now().Add(-time.Minute),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	leaf, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}
	return &tls.Certificate{Certificate: [][]byte{der}, PrivateKey: key, Leaf: leaf}
}

type observation struct {
	serial int64
	proto  int
}

type result struct {
	status int
	err    error
}

type fixture struct {
	manager  *stubManager
	config   *rest.Config
	client   *http.Client
	server   *httptest.Server
	stop     chan struct{}
	release  chan struct{}
	received chan observation
	canceled chan struct{}
}

func newFixture(t *testing.T, http2 bool, initial *tls.Certificate, period time.Duration) *fixture {
	t.Helper()
	f := &fixture{
		manager:  &stubManager{reads: make(chan struct{}, 20)},
		stop:     make(chan struct{}),
		release:  make(chan struct{}),
		received: make(chan observation, 20),
		canceled: make(chan struct{}, 20),
	}
	f.manager.cert.Store(initial)
	f.server = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Errorf("read request: %v", err)
			return
		}
		observed := observation{proto: r.ProtoMajor}
		if len(r.TLS.PeerCertificates) != 0 {
			observed.serial = r.TLS.PeerCertificates[0].SerialNumber.Int64()
		}
		f.received <- observed
		if r.URL.Path == "/hold" {
			select {
			case <-r.Context().Done():
				f.canceled <- struct{}{}
				return
			case <-f.release:
			}
		}
		if r.URL.Path == "/unavailable" {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusCreated)
	}))
	f.server.EnableHTTP2 = http2
	f.server.TLS = &tls.Config{ClientAuth: tls.RequestClientCert, MinVersion: tls.VersionTLS12}
	f.server.StartTLS()
	t.Cleanup(f.server.Close)
	f.config = &rest.Config{
		Host: f.server.URL,
		TLSClientConfig: rest.TLSClientConfig{
			CAData: pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: f.server.Certificate().Raw}),
		},
	}
	logger, _ := ktesting.NewTestContext(t)
	closeConnections, err := updateTransport(logger, f.stop, period, f.config, f.manager, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		select {
		case <-f.stop:
		default:
			close(f.stop)
		}
		closeConnections()
	})
	f.client = &http.Client{Transport: f.config.Transport, Timeout: 5 * time.Second}
	for i := 0; i < 2; i++ {
		select {
		case <-f.manager.reads:
		case <-time.After(5 * time.Second):
			t.Fatal("certificate watcher did not start")
		}
	}
	return f
}

func (f *fixture) post(ctx context.Context, path string) result {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, f.server.URL+path, strings.NewReader(`{"metadata":{"name":"bootstrap-repro"}}`))
	if err != nil {
		return result{err: err}
	}
	response, err := f.client.Do(request)
	if err != nil {
		return result{err: err}
	}
	_, readErr := io.Copy(io.Discard, response.Body)
	closeErr := response.Body.Close()
	if readErr != nil {
		return result{err: readErr}
	}
	return result{status: response.StatusCode, err: closeErr}
}

func receive[T any](t *testing.T, channel <-chan T) T {
	t.Helper()
	select {
	case value := <-channel:
		return value
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for test event")
		var zero T
		return zero
	}
}

func TestInitialCertificateDoesNotInterruptRegistration(t *testing.T) {
	for _, http2 := range []bool{false, true} {
		name := "HTTP1"
		if http2 {
			name = "HTTP2"
		}
		t.Run(name, func(t *testing.T) {
			f := newFixture(t, http2, nil, 10*time.Second)
			f.manager.cert.Store(newCertificate(t, 1))
			done := make(chan result, 1)
			go func() { done <- f.post(context.Background(), "/hold") }()
			observed := receive(t, f.received)
			wantProtocol := 1
			if http2 {
				wantProtocol = 2
			}
			if observed.serial != 1 || observed.proto != wantProtocol {
				t.Fatalf("unexpected registration: %+v", observed)
			}
			select {
			case result := <-done:
				t.Fatalf("initial certificate activation interrupted registration: %+v", result)
			case <-time.After(1100 * time.Millisecond):
			}
			close(f.release)
			got := receive(t, done)
			if got.err != nil || got.status != http.StatusCreated {
				t.Fatalf("registration failed: %+v", got)
			}
		})
	}
}

func TestExistingCertificateStartsImmediately(t *testing.T) {
	f := newFixture(t, true, newCertificate(t, 1), time.Hour)
	got := f.post(context.Background(), "/ok")
	if got.err != nil || got.status != http.StatusCreated {
		t.Fatalf("existing certificate was unnecessarily gated: %+v", got)
	}
	if got := receive(t, f.received); got.serial != 1 {
		t.Fatalf("wrong initial identity: %+v", got)
	}
}

func TestLaterRotationStillClosesOldConnections(t *testing.T) {
	f := newFixture(t, true, newCertificate(t, 1), 20*time.Millisecond)
	done := make(chan result, 1)
	go func() { done <- f.post(context.Background(), "/hold") }()
	if got := receive(t, f.received); got.serial != 1 {
		t.Fatalf("wrong original identity: %+v", got)
	}
	f.manager.cert.Store(newCertificate(t, 2))
	if got := receive(t, done); got.err == nil {
		t.Fatalf("old authenticated connection survived rotation: %+v", got)
	}
	receive(t, f.canceled)
	got := f.post(context.Background(), "/ok")
	if got.err != nil || got.status != http.StatusCreated {
		t.Fatalf("retry after ordinary rotation failed: %+v", got)
	}
	if got := receive(t, f.received); got.serial != 2 {
		t.Fatalf("retry did not use new identity: %+v", got)
	}
}

func TestMissingCertificateHonorsDialCancellation(t *testing.T) {
	f := newFixture(t, true, nil, time.Hour)
	transport, ok := f.config.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("unexpected transport %T", f.config.Transport)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		conn, err := transport.DialContext(ctx, "tcp", f.server.Listener.Addr().String())
		if conn != nil {
			conn.Close()
			done <- errors.New("dial succeeded before the initial certificate was available")
			return
		}
		done <- err
	}()
	cancel()
	if err := receive(t, done); !errors.Is(err, context.Canceled) {
		t.Fatalf("missing certificate did not preserve cancellation: %v", err)
	}
}

func TestStoppingReleasesDialWaitingForCertificate(t *testing.T) {
	f := newFixture(t, true, nil, time.Hour)
	transport, ok := f.config.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("unexpected transport %T", f.config.Transport)
	}
	done := make(chan error, 1)
	go func() {
		conn, err := transport.DialContext(context.Background(), "tcp", f.server.Listener.Addr().String())
		if conn != nil {
			conn.Close()
			done <- errors.New("dial succeeded before the initial certificate was available")
			return
		}
		done <- err
	}()
	close(f.stop)
	if err := receive(t, done); err == nil || !strings.Contains(err.Error(), "stopped") {
		t.Fatalf("stopping did not release pending dial: %v", err)
	}
}

func TestRealFailuresRemainVisible(t *testing.T) {
	for _, bootstrap := range []bool{false, true} {
		name := "existing-certificate"
		var initial *tls.Certificate
		if bootstrap {
			name = "new-certificate"
		} else {
			initial = newCertificate(t, 1)
		}
		t.Run(name, func(t *testing.T) {
			f := newFixture(t, true, initial, time.Hour)
			if bootstrap {
				f.manager.cert.Store(newCertificate(t, 1))
			}
			got := f.post(context.Background(), "/unavailable")
			if got.err != nil || got.status != http.StatusServiceUnavailable {
				t.Fatalf("real 503 was hidden: %+v", got)
			}
			receive(t, f.received)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan result, 1)
			go func() { done <- f.post(ctx, "/hold") }()
			receive(t, f.received)
			cancel()
			if got := receive(t, done); !errors.Is(got.err, context.Canceled) {
				t.Fatalf("unclassified cancellation was hidden: %+v", got)
			}
		})
	}
}

func TestMissingCertificateHonorsRequestDeadlineAndRecovers(t *testing.T) {
	f := newFixture(t, true, nil, time.Hour)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if got := f.post(ctx, "/ok"); !errors.Is(got.err, context.DeadlineExceeded) {
		t.Fatalf("pending initial certificate hid request deadline: %+v", got)
	}
	select {
	case got := <-f.received:
		t.Fatalf("request reached server without the initial certificate: %+v", got)
	default:
	}
	f.manager.cert.Store(newCertificate(t, 1))
	got := f.post(context.Background(), "/ok")
	if got.err != nil || got.status != http.StatusCreated {
		t.Fatalf("transport did not recover when certificate arrived: %+v", got)
	}
}

func TestNoCertificateManagerPreservesUnmanagedTransport(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
	}))
	defer server.Close()
	config := &rest.Config{Host: server.URL}
	stop := make(chan struct{})
	defer close(stop)
	logger, _ := ktesting.NewTestContext(t)
	closeConnections, err := UpdateTransport(logger, stop, config, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer closeConnections()
	client, err := rest.HTTPClientFor(config)
	if err != nil {
		t.Fatal(err)
	}
	client.Timeout = 5 * time.Second
	response, err := client.Get(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusCreated {
		t.Fatalf("unmanaged transport status = %d", response.StatusCode)
	}
}
