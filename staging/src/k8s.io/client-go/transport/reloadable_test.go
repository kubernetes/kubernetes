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

package transport

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
)

// func TestNewReloadableTransport(t *testing.T) {
// 	want := &http.Transport{}
// 	dt := newDynamicRootCATransport(want)
// 	if got := dt.container.Load(); want != got {
// 		t.Errorf("expected the original transport object: %p, but got: %p", want, got)
// 	}
// 	if got := dt.WrappedRoundTripper(); want != got {
// 		t.Errorf("expected the original transport object: %p, but got: %p", want, got)
// 	}

// 	want = &http.Transport{}
// 	dt.p.Store(want)
// 	if got := dt.WrappedRoundTripper(); want != got {
// 		t.Errorf("expected the original transport object: %p, but got: %p", want, got)
// 	}
// }

func TestReloadableTransportWithConfigAndCache(t *testing.T) {
	ca := setupCA(t)
	caFileName, removeFn := writeCACertToFile(t, ca.PEM)
	defer removeFn(t)

	getReloadableFn := func(rt http.RoundTripper) *dynamicRootCATransport {
		for {
			if rt == nil {
				return nil
			}
			switch transport := rt.(type) {
			case *dynamicRootCATransport:
				return transport
			case interface{ WrappedRoundTripper() http.RoundTripper }:
				rt = transport.WrappedRoundTripper()
			default:
				return nil
			}
		}
	}

	config1 := &Config{
		TLS: TLSConfig{
			CAFile: caFileName,
		},
	}
	key, _, err := tlsConfigKey(config1)
	if err != nil {
		t.Errorf("did not expect any error from tlsConfigKey: %v", err)
	}

	rt1, err := New(config1)
	if err != nil {
		t.Errorf("did not expect an error from New: %v", err)
		return
	}
	reloadable1 := getReloadableFn(rt1)
	if reloadable1 == nil {
		t.Errorf("expected the given RoundTripper object to have an embedded %T", &dynamicRootCATransport{})
		return
	}
	if cached, ok := tlsCache.transports[key]; !ok || cached != reloadable1 {
		t.Errorf("expected the reloadable transport to be cached")
	}

	config2 := config1
	rt2, err := New(config2)
	if err != nil {
		t.Errorf("did not expect an error from New: %v", err)
		return
	}
	reloadable2 := getReloadableFn(rt2)
	if reloadable2 == nil {
		t.Errorf("expected the given RoundTripper object to have an embedded %T", &dynamicRootCATransport{})
		return
	}
	if cached, ok := tlsCache.transports[key]; !ok || cached != reloadable2 {
		t.Errorf("expected the reloadable transport to be cached")
	}
	if reloadable1 != reloadable2 {
		t.Errorf("expected a single reloadable instance")
	}

	config3 := &Config{
		TLS: TLSConfig{
			CAFile: caFileName,
		},
	}
	key, _, err = tlsConfigKey(config3)
	if err != nil {
		t.Errorf("did not expect any error from tlsConfigKey: %v", err)
	}
	rt3, err := New(config1)
	if err != nil {
		t.Errorf("did not expect an error from New: %v", err)
		return
	}
	reloadable3 := getReloadableFn(rt3)
	if reloadable3 == nil {
		t.Errorf("expected the given RoundTripper object to have an embedded %T", &dynamicRootCATransport{})
		return
	}
	if cached, ok := tlsCache.transports[key]; !ok || cached != reloadable3 {
		t.Errorf("expected the reloadable transport to be cached")
	}
	if reloadable2 != reloadable3 {
		t.Errorf("expected a single reloadable instance")
	}
}

type fakeSyncer struct {
	invokedCh chan time.Time
}

func (fs *fakeSyncer) sync() error {
	fs.invokedCh <- time.Now()
	return errors.New("always fail")
}

func TestReloadableTransportControllerRetry(t *testing.T) {
	syncer := &fakeSyncer{invokedCh: make(chan time.Time, 1)}
	var once sync.Once
	keyAddedCh := make(chan struct{})
	controller := newDynamicRootCATransportController(syncer)
	// add a key once, and see how many attempt(s) are made
	controller.queueAdderFn = func(stopCtx context.Context) {
		once.Do(func() {
			defer close(keyAddedCh)
			controller.queue.Add(queueKey)
		})
		<-stopCtx.Done()
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	go controller.Run(ctx)

	select {
	case <-keyAddedCh:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("expected the sync to have completed")
		return
	}

	var attempts int
	var lastAt time.Time
	intervals := make([]time.Duration, 0)
	func() {
		for {
			select {
			case at, ok := <-syncer.invokedCh:
				if !ok {
					t.Errorf("the channel closed unexpectedly")
					return
				}
				attempts++
				if !lastAt.IsZero() {
					intervals = append(intervals, at.Sub(lastAt))
				}
				lastAt = at
			case <-ctx.Done():
				return
			}
		}
	}()

	if attempts < 2 {
		t.Errorf("expected sync attempts to be 2+, but got: %d", attempts)
	}
	t.Logf("attempts: %d, intervals: %v", attempts, intervals)
}

func TestSync(t *testing.T) {
	ca1 := setupCA(t)
	caFileName, removeFn := writeCACertToFile(t, ca1.PEM)
	defer removeFn(t)

	original := &http.Transport{TLSClientConfig: &tls.Config{}}
	rt := newDynamicRootCATransport(original)
	syncer := newRootCASyncer(context.Background(), rt, caFileName, ca1.PEM)

	// reloadable transport is a long lived object, so use a single
	// instance to exercise various aspects of sync operations.
	t.Run("no change in file content, no new transport", func(t *testing.T) {
		if err := syncer.sync(); err != nil {
			t.Errorf("expected no error from sync, but got: %v", err)
		}
		if want, got := original, rt.WrappedRoundTripper(); want != got {
			t.Errorf("expected the original transport object: %p, but got: %p", want, got)
		}
		if !bytes.Equal(syncer.caBytes, ca1.PEM) {
			t.Errorf("the CA data has changed unexpectedly")
		}
	})

	t.Run("bad bytes, now new transport", func(t *testing.T) {
		if err := os.WriteFile(caFileName, []byte("foo/bar"), 0664); err != nil {
			t.Fatalf("did not expect any error while writing to the CA file %q: %v", caFileName, err)
		}
		if err := syncer.sync(); err == nil {
			t.Errorf("expected an error from sync")
		}
		if want, got := original, rt.WrappedRoundTripper(); want != got {
			t.Errorf("expected the original transport object: %p, but got: %p", want, got)
		}
		if !bytes.Equal(syncer.caBytes, ca1.PEM) {
			t.Errorf("the CA data has changed unexpectedly")
		}
	})

	t.Run("empty file, no new transport", func(t *testing.T) {
		if err := os.WriteFile(caFileName, []byte{}, 0664); err != nil {
			t.Fatalf("did not expect any error while writing to the CA file %q: %v", caFileName, err)
		}
		if err := syncer.sync(); err != nil {
			t.Errorf("expected no error from sync, but got: %v", err)
		}
		if want, got := original, rt.WrappedRoundTripper(); want != got {
			t.Errorf("expected the original transport object: %p, but got: %p", want, got)
		}
		if !bytes.Equal(syncer.caBytes, ca1.PEM) {
			t.Errorf("the CA data has changed unexpectedly")
		}
	})

	t.Run("new and valid root CA cert has been written to the file, new transport", func(t *testing.T) {
		oldRootCAs := rt.container.transport.TLSClientConfig.RootCAs
		ca2 := setupCA(t)
		if err := os.WriteFile(caFileName, ca2.PEM, 0664); err != nil {
			t.Fatalf("did not expect any error while writing to the CA file %q: %v", caFileName, err)
		}
		if err := syncer.sync(); err != nil {
			t.Errorf("expected no error from sync, but got: %v", err)
		}
		if got := rt.WrappedRoundTripper(); original == got {
			t.Errorf("expected a new transport object: old: %p, new: %p", original, got)
		}
		if !bytes.Equal(syncer.caBytes, ca2.PEM) {
			t.Errorf("expected the new CA data to be stored")
		}
		if newTransort := rt.container.transport; newTransort.TLSClientConfig.RootCAs == oldRootCAs {
			t.Errorf("expected the new RootCAs to be in use")
		}
	})
}

func TestReloadableTransport(t *testing.T) {
	serverName := "s1.k8s.io"
	ca1, ca2, caUnknown := setupCA(t), setupCA(t), setupCA(t)
	serverCert1 := setupServerCertWithCA(t, ca1, serverName)
	serverCert2 := setupServerCertWithCA(t, ca2, serverName)

	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := w.Write([]byte("pong")); err != nil {
			t.Errorf("did not expect error from Write: %v", err)
		}
	}))
	serverSwitchCertCh := make(chan struct{})
	server.TLS = &tls.Config{
		// the server maintains two certificates:
		//  a) server cert 1: signed by ca1
		//  b) server cert 2: signed by ca2
		// the test dictates which one to send to the client
		GetCertificate: func(ch *tls.ClientHelloInfo) (*tls.Certificate, error) {
			select {
			case <-serverSwitchCertCh:
				return &serverCert2, nil
			default:
				return &serverCert1, nil
			}
		},
	}
	defer server.Close()
	server.StartTLS()

	caFileName, removeFn := writeCACertToFile(t, ca1.PEM)
	defer removeFn(t)
	config := &Config{
		TLS: TLSConfig{
			ServerName: serverName,
			NextProtos: []string{"http/1.1"},
			CAFile:     caFileName, // should point to ca1
		},
	}
	tlsConfig, err := TLSConfigFor(config)
	if err != nil {
		t.Errorf("did not expect TLSConfigFor to return an error: %v", err)
		return
	}

	// keep connection reuse enabled, so we can verify that the reloadable
	// transport works as expected with cached connections.
	transport := &http.Transport{TLSClientConfig: tlsConfig}
	rt := newDynamicRootCATransport(transport)
	syncer := newRootCASyncer(context.Background(), rt, caFileName, ca1.PEM)
	client := &http.Client{Transport: rt}

	// step 1: the server is configured to send cert 1 to the client, the
	// reloadable transport points to ca1
	resp, err := do(t, client, server.URL+"/ping", shouldUseNewConnection)
	expectStatusOK(t, resp, err)
	resp, err = do(t, client, server.URL+"/ping", shouldUseExistingConnection)
	expectStatusOK(t, resp, err)

	// step 2: overwrite the CA file with an unknown CA, reload the
	// transport, the server should return a TLS error.
	if err := os.WriteFile(caFileName, caUnknown.PEM, 0664); err != nil {
		t.Errorf("did not expect any error while writing to the CA file %q: %v", caFileName, err)
		return
	}
	if err := syncer.sync(); err != nil {
		t.Errorf("did not expect any error from ReloadCA: %v", err)
		return
	}
	resp, err = do(t, client, server.URL+"/ping", shouldNotGotConn)
	expectTLSError(t, resp, err, "x509: certificate signed by unknown authority")

	// step 3: overwrite ths CA file file with ca2 and sync, we expect the
	// server to return an error
	if err := os.WriteFile(caFileName, ca2.PEM, 0664); err != nil {
		t.Errorf("did not expect any error while writing to the CA file %q: %v", caFileName, err)
		return
	}
	if err := syncer.sync(); err != nil {
		t.Errorf("did not expect any error from ReloadCA: %v", err)
		return
	}

	resp, err = do(t, client, server.URL+"/ping", shouldNotGotConn)
	expectTLSError(t, resp, err, "x509: certificate signed by unknown authority")

	// switch the server to send server cert 2
	close(serverSwitchCertCh)

	resp, err = do(t, client, server.URL+"/ping", shouldUseNewConnection)
	expectStatusOK(t, resp, err)
	resp, err = do(t, client, server.URL+"/ping", shouldUseExistingConnection)
	expectStatusOK(t, resp, err)

	// let's revist the old transport, it should not be able to make new connectons
	// but the old connection should be able to talk to the server
	client = &http.Client{Transport: transport}
	resp, err = do(t, client, server.URL+"/ping", shouldNotGotConn)
	expectTLSError(t, resp, err, "x509: certificate signed by unknown authority")
}

func TestReloadableTransportWithController(t *testing.T) {
	serverName := "s1.k8s.io"
	ca1, ca2, caUnknown := setupCA(t), setupCA(t), setupCA(t)
	serverCert1 := setupServerCertWithCA(t, ca1, serverName)
	serverCert2 := setupServerCertWithCA(t, ca2, serverName)

	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := w.Write([]byte("pong")); err != nil {
			t.Errorf("did not expect error from Write: %v", err)
		}
	}))
	serverSwitchCertCh := make(chan struct{})
	server.TLS = &tls.Config{
		// the server maintains two certificates:
		//  a) server cert 1: signed by ca1
		//  b) server cert 2: signed by ca2
		// the test dictates which one to send to the client
		GetCertificate: func(ch *tls.ClientHelloInfo) (*tls.Certificate, error) {
			select {
			case <-serverSwitchCertCh:
				return &serverCert2, nil
			default:
				return &serverCert1, nil
			}
		},
	}
	defer server.Close()
	server.EnableHTTP2 = true
	server.StartTLS()

	caFileName, removeFn := writeCACertToFile(t, ca1.PEM)
	defer removeFn(t)
	config := &Config{
		TLS: TLSConfig{
			ServerName: serverName,
			CAFile:     caFileName, // should point to ca1
			NextProtos: []string{"h2"},
		},
	}
	tlsConfig, err := TLSConfigFor(config)
	if err != nil {
		t.Errorf("did not expect TLSConfigFor to return an error: %v", err)
		return
	}

	// keep connection reuse enabled, so we can verify that the reloadable
	// transport works as expected with cached connections.
	transport := &http.Transport{TLSClientConfig: tlsConfig, ForceAttemptHTTP2: true}

	reloadable := newDynamicRootCATransport(transport)
	syncer := &withSyncCount{
		rootCASyncer: newRootCASyncer(context.Background(), reloadable, caFileName, ca1.PEM),
		syncedCh:     make(chan error, 1),
	}
	client := &http.Client{Transport: reloadable}

	syncCh := make(chan struct{}, 1)
	controller := newDynamicRootCATransportController(syncer)
	controller.queueAdderFn = func(stopCtx context.Context) {
		for {
			select {
			case _, ok := <-syncCh:
				if !ok {
					return
				}
				controller.queue.Add(queueKey)
			case <-stopCtx.Done():
				return
			}
		}
	}
	triggerSyncAndWaitFn := func() error {
		syncCh <- struct{}{}
		select {
		case err, ok := <-syncer.syncedCh:
			if !ok {
				return errors.New("did not expect the channel to be closed")
			}
			return err
		case <-time.After(wait.ForeverTestTimeout):
			return errors.New("expected the sync to have completed")
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go controller.Run(ctx)

	// step 1: server is on serving cert 1, the reloadable transport points
	// to ca1, so we expect success.
	resp, err := do(t, client, server.URL+"/ping", shouldUseNewConnection)
	expectStatusOK(t, resp, err)
	resp, err = do(t, client, server.URL+"/ping", shouldUseExistingConnection)
	expectStatusOK(t, resp, err)

	// step 2: overwrite the CA file with an unknown CA, wait for the
	// transport to be reloaded; the server should return a TLS error.
	if err := os.WriteFile(caFileName, caUnknown.PEM, 0664); err != nil {
		t.Fatalf("did not expect any error while writing to the CA file %q: %v", caFileName, err)
	}
	if err := triggerSyncAndWaitFn(); err != nil {
		t.Fatalf("sync error: %v", err)
	}

	// reloadable has refreshed the underlying http.Transport object,
	// we expect TLS error, so no successful connection setup
	resp, err = do(t, client, server.URL+"/ping", shouldNotGotConn)
	expectTLSError(t, resp, err, "x509: certificate signed by unknown authority")

	// step 3: overwrite ths CA file file with ca2, and have the server be
	// serving on server cert 2 now.
	if err := os.WriteFile(caFileName, ca2.PEM, 0664); err != nil {
		t.Fatalf("did not expect any error while writing to the CA file %q: %v", caFileName, err)
	}
	if err := triggerSyncAndWaitFn(); err != nil {
		t.Fatalf("sync error: %v", err)
	}
	// switch the server to send server cert 2
	close(serverSwitchCertCh)

	// reloadable has refreshed the underlying http.Transport object, the
	resp, err = do(t, client, server.URL+"/ping", shouldUseNewConnection)
	expectStatusOK(t, resp, err)
	resp, err = do(t, client, server.URL+"/ping", shouldUseExistingConnection)
	expectStatusOK(t, resp, err)

	// let's revist the old transport, it should not be able to make new connectons
	// but the old connection should be able to talk to the server
	client = &http.Client{Transport: transport}
	resp, err = do(t, client, server.URL+"/ping", shouldNotGotConn)
	expectTLSError(t, resp, err, "x509: certificate signed by unknown authority")
}

func TestReloadableTransportWithOldTransportCleanup(t *testing.T) {
	serverName := "s1.k8s.io"
	ca1, ca2 := setupCA(t), setupCA(t)
	serverCert1 := setupServerCertWithCA(t, ca1, serverName)
	serverCert2 := setupServerCertWithCA(t, ca2, serverName)

	blockedCh := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(10)
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/block" {
			wg.Done()
			<-blockedCh
		}
		if _, err := w.Write([]byte("pong")); err != nil {
			t.Errorf("did not expect error from Write: %v", err)
		}
	}))
	serverSwitchCertCh := make(chan struct{})
	server.TLS = &tls.Config{
		GetCertificate: func(ch *tls.ClientHelloInfo) (*tls.Certificate, error) {
			select {
			case <-serverSwitchCertCh:
				return &serverCert2, nil
			default:
				return &serverCert1, nil
			}
		},
	}
	defer server.Close()
	server.StartTLS()

	caFileName, removeFn := writeCACertToFile(t, ca1.PEM)
	defer removeFn(t)
	config := &Config{
		TLS: TLSConfig{
			ServerName: serverName,
			CAFile:     caFileName, // should point to ca1
			NextProtos: []string{"http/1.1"},
		},
	}
	tlsConfig, err := TLSConfigFor(config)
	if err != nil {
		t.Errorf("did not expect TLSConfigFor to return an error: %v", err)
		return
	}
	// keep connection reuse enabled
	transport := &http.Transport{TLSClientConfig: tlsConfig, ForceAttemptHTTP2: false}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	reloadable := newDynamicRootCATransport(transport)
	syncer := &withSyncCount{
		rootCASyncer: newRootCASyncer(ctx, reloadable, caFileName, ca1.PEM),
		syncedCh:     make(chan error, 1),
	}
	client := &http.Client{Transport: reloadable}

	syncCh := make(chan struct{}, 1)
	controller := newDynamicRootCATransportController(syncer)
	controller.queueAdderFn = func(stopCtx context.Context) {
		for {
			select {
			case _, ok := <-syncCh:
				if !ok {
					return
				}
				controller.queue.Add(queueKey)
			case <-stopCtx.Done():
				return
			}
		}
	}
	triggerSyncAndWaitFn := func() error {
		syncCh <- struct{}{}
		select {
		case err, ok := <-syncer.syncedCh:
			if !ok {
				return errors.New("did not expect the channel to be closed")
			}
			return err
		case <-time.After(wait.ForeverTestTimeout):
			return errors.New("expected the sync to have completed")
		}
	}
	go controller.Run(ctx)

	useNewConn := func(t *testing.T, ci httptrace.GotConnInfo) {
		shouldUseNewConnection(t, ci)
	}
	// step 1: launch 4 concurrent requests that stay in flight
	inflightCh := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func() {
			resp, err := do(t, client, server.URL+"/block", useNewConn)
			expectStatusOK(t, resp, err)
			inflightCh <- true
		}()
	}

	wg.Wait()

	// step 2: overwrite ths CA file file with ca2
	if err := os.WriteFile(caFileName, ca2.PEM, 0664); err != nil {
		t.Fatalf("did not expect any error while writing to the CA file %q: %v", caFileName, err)
	}
	if err := triggerSyncAndWaitFn(); err != nil {
		t.Fatalf("sync error: %v", err)
	}
	close(serverSwitchCertCh)

	// reloadable should use the new transport
	resp, err := do(t, client, server.URL+"/ping", shouldUseNewConnection)
	expectStatusOK(t, resp, err)

	// the old requests should still be active
	select {
	case _, ok := <-inflightCh:
		if ok {
			t.Errorf("inflight request returned unexpectedly")
		}
	default:
	}

	close(blockedCh)
	count := 0
	for v := range inflightCh {
		if v {
			count++
		}
		if count == 10 {
			break
		}
	}
	if count != 10 {
		t.Errorf("inflight request returned unexpectedly")
	}
}

func do(t *testing.T, client *http.Client, url string, f func(*testing.T, httptrace.GotConnInfo)) (*http.Response, error) {
	t.Helper()
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		t.Fatalf("failed to create new request: %v", err)
	}

	trace := &httptrace.ClientTrace{
		GotConn: func(ci httptrace.GotConnInfo) {
			t.Logf("GotConnInfo: %+v", ci)
			f(t, ci)
		},
	}
	req = req.WithContext(httptrace.WithClientTrace(req.Context(), trace))
	resp, err := client.Do(req)
	// the content of the response body is not asserted on, so discard it here.
	if resp != nil {
		if _, err = io.Copy(io.Discard, resp.Body); err != nil {
			t.Errorf("unexpected error while reading the Response Body: %v", err)
		}
		if err := resp.Body.Close(); err != nil {
			t.Errorf("unexpected error while closing the Body of the Response object: %v", err)
		}
	}
	return resp, err
}

func shouldUseExistingConnection(t *testing.T, ci httptrace.GotConnInfo) {
	t.Helper()
	if !ci.Reused {
		t.Errorf("expected an existing TCP connection to be reused, but got: %+v", ci)
	}
}

func shouldUseNewConnection(t *testing.T, ci httptrace.GotConnInfo) {
	t.Helper()
	if ci.Reused {
		t.Errorf("expected a new connection, but got: %+v", ci)
	}
}

func shouldNotGotConn(t *testing.T, ci httptrace.GotConnInfo) {
	t.Helper()
	t.Errorf("unexpected GotConnInfo: %+v", ci)
}

type withSyncCount struct {
	*rootCASyncer
	syncedCh chan error
}

func (r *withSyncCount) sync() error {
	var err error
	defer func() {
		if r.syncedCh != nil {
			r.syncedCh <- err
		}
	}()
	err = r.rootCASyncer.sync()
	return err
}

type ca struct {
	PEM           []byte
	privateKeyPEM []byte
	certificate   *x509.Certificate
	privateKey    *rsa.PrivateKey
}

func writeCACertToFile(t *testing.T, bytes []byte) (string, func(*testing.T)) {
	t.Helper()
	f, err := os.CreateTemp("", "test-*-ca.crt")
	if err != nil {
		t.Fatalf("unexpected error while creating a temporary file: %v", err)
	}
	if _, err := f.Write(bytes); err != nil {
		t.Fatalf("unexpected error while writing to the ca file %q: %v", f.Name(), err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("unexpected error while closing the ca file %q: %v", f.Name(), err)
	}
	return f.Name(), func(t *testing.T) {
		if err := os.Remove(f.Name()); err != nil {
			t.Errorf("unexpected error while removing file: %q - %v", f.Name(), err)
		}
	}
}

func setupCA(t *testing.T) ca {
	t.Helper()
	caPrivateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("failed to generate private key: %v", err)
	}

	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: fmt.Sprintf("ca@%d", time.Now().Unix()),
		},
		NotBefore:             time.Unix(0, 0),
		NotAfter:              time.Now().Add(time.Hour * 24 * 365 * 100),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
	}

	caBytes, err := x509.CreateCertificate(rand.Reader, template, template, &caPrivateKey.PublicKey, caPrivateKey)
	if err != nil {
		t.Fatalf("failed to generate ca certificate: %v", err)
	}
	caPEM := bytes.Buffer{}
	if err := pem.Encode(&caPEM, &pem.Block{Type: "CERTIFICATE", Bytes: caBytes}); err != nil {
		t.Fatalf("failed to PEM encode the ca certificate: %v", err)
	}
	caPrivKeyPEM := bytes.Buffer{}
	if err := pem.Encode(&caPrivKeyPEM, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(caPrivateKey)}); err != nil {
		t.Fatalf("failed to PEM encode the ca key: %v", err)
	}

	return ca{PEM: caPEM.Bytes(), privateKeyPEM: caPrivKeyPEM.Bytes(), privateKey: caPrivateKey, certificate: template}
}

func setupServerCertWithCA(t *testing.T, ca ca, serverName string) tls.Certificate {
	t.Helper()
	template := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject: pkix.Name{
			CommonName: fmt.Sprintf("server@%d", time.Now().Unix()),
		},
		DNSNames:     []string{serverName},
		NotBefore:    time.Unix(0, 0),
		NotAfter:     time.Now().Add(time.Hour * 24 * 365 * 100),
		SubjectKeyId: []byte{1, 2, 3, 4, 6},
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		KeyUsage:     x509.KeyUsageDigitalSignature,
	}

	serverPrivateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("failed to generate private key: %v", err)
	}
	serverCertBytes, err := x509.CreateCertificate(rand.Reader, template, ca.certificate, &serverPrivateKey.PublicKey, ca.privateKey)
	if err != nil {
		t.Fatalf("failed to generate server certificate: %v", err)
	}

	serverCertPEM := bytes.Buffer{}
	if err := pem.Encode(&serverCertPEM, &pem.Block{Type: "CERTIFICATE", Bytes: serverCertBytes}); err != nil {
		t.Fatalf("failed to PEM encode the server certificate: %v", err)
	}
	serverPriKeyPEM := bytes.Buffer{}
	if err := pem.Encode(&serverPriKeyPEM, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(serverPrivateKey)}); err != nil {
		t.Fatalf("failed to PEM encode the server key: %v", err)
	}

	serverCertKP, err := tls.X509KeyPair(serverCertPEM.Bytes(), serverPriKeyPEM.Bytes())
	if err != nil {
		t.Fatalf("failed to generate server certificate: %v", err)
	}
	return serverCertKP
}

func expectStatusOK(t *testing.T, resp *http.Response, err error) {
	t.Helper()
	if err != nil {
		t.Errorf("expected no error, but got: %+v", err)
		return
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected status code: %d, but got: %v", http.StatusOK, resp)
	}
}

func expectTLSError(t *testing.T, resp *http.Response, err error, shouldContain string) {
	t.Helper()
	if resp != nil {
		t.Errorf("did not expect a response from the server, but got: %v", resp)
		return
	}
	var wantErr *tls.CertificateVerificationError
	if !errors.As(err, &wantErr) {
		t.Errorf("expected an error: %v, but got: %#v", tls.CertificateVerificationError{}, err)
		return
	}
	if !strings.Contains(wantErr.Error(), shouldContain) {
		t.Errorf("expected internal error to contain: %q, but got: %v", shouldContain, wantErr)
	}
}
