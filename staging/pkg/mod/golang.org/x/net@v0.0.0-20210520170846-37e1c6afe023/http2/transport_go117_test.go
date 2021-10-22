// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.17
// +build go1.17

package http2

import (
	"context"
	"crypto/tls"
	"errors"
	"net/http"
	"net/http/httptest"

	"testing"
)

func TestTransportDialTLSContext(t *testing.T) {
	blockCh := make(chan struct{})
	serverTLSConfigFunc := func(ts *httptest.Server) {
		ts.Config.TLSConfig = &tls.Config{
			// Triggers the server to request the clients certificate
			// during TLS handshake.
			ClientAuth: tls.RequestClientCert,
		}
	}
	ts := newServerTester(t,
		func(w http.ResponseWriter, r *http.Request) {},
		optOnlyServer,
		serverTLSConfigFunc,
	)
	defer ts.Close()
	tr := &Transport{
		TLSClientConfig: &tls.Config{
			GetClientCertificate: func(cri *tls.CertificateRequestInfo) (*tls.Certificate, error) {
				// Tests that the context provided to `req` is
				// passed into this function.
				close(blockCh)
				<-cri.Context().Done()
				return nil, cri.Context().Err()
			},
			InsecureSkipVerify: true,
		},
	}
	defer tr.CloseIdleConnections()
	req, err := http.NewRequest(http.MethodGet, ts.ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req = req.WithContext(ctx)
	errCh := make(chan error)
	go func() {
		defer close(errCh)
		res, err := tr.RoundTrip(req)
		if err != nil {
			errCh <- err
			return
		}
		res.Body.Close()
	}()
	// Wait for GetClientCertificate handler to be called
	<-blockCh
	// Cancel the context
	cancel()
	// Expect the cancellation error here
	err = <-errCh
	if err == nil {
		t.Fatal("cancelling context during client certificate fetch did not error as expected")
		return
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected error returned after cancellation: %v", err)
	}
}

// TestDialRaceResumesDial tests that, given two concurrent requests
// to the same address, when the first Dial is interrupted because
// the first request's context is cancelled, the second request
// resumes the dial automatically.
func TestDialRaceResumesDial(t *testing.T) {
	blockCh := make(chan struct{})
	serverTLSConfigFunc := func(ts *httptest.Server) {
		ts.Config.TLSConfig = &tls.Config{
			// Triggers the server to request the clients certificate
			// during TLS handshake.
			ClientAuth: tls.RequestClientCert,
		}
	}
	ts := newServerTester(t,
		func(w http.ResponseWriter, r *http.Request) {},
		optOnlyServer,
		serverTLSConfigFunc,
	)
	defer ts.Close()
	tr := &Transport{
		TLSClientConfig: &tls.Config{
			GetClientCertificate: func(cri *tls.CertificateRequestInfo) (*tls.Certificate, error) {
				select {
				case <-blockCh:
					// If we already errored, return without error.
					return &tls.Certificate{}, nil
				default:
				}
				close(blockCh)
				<-cri.Context().Done()
				return nil, cri.Context().Err()
			},
			InsecureSkipVerify: true,
		},
	}
	defer tr.CloseIdleConnections()
	req, err := http.NewRequest(http.MethodGet, ts.ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Create two requests with independent cancellation.
	ctx1, cancel1 := context.WithCancel(context.Background())
	defer cancel1()
	req1 := req.WithContext(ctx1)
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	req2 := req.WithContext(ctx2)
	errCh := make(chan error)
	go func() {
		res, err := tr.RoundTrip(req1)
		if err != nil {
			errCh <- err
			return
		}
		res.Body.Close()
	}()
	successCh := make(chan struct{})
	go func() {
		// Don't start request until first request
		// has initiated the handshake.
		<-blockCh
		res, err := tr.RoundTrip(req2)
		if err != nil {
			errCh <- err
			return
		}
		res.Body.Close()
		// Close successCh to indicate that the second request
		// made it to the server successfully.
		close(successCh)
	}()
	// Wait for GetClientCertificate handler to be called
	<-blockCh
	// Cancel the context first
	cancel1()
	// Expect the cancellation error here
	err = <-errCh
	if err == nil {
		t.Fatal("cancelling context during client certificate fetch did not error as expected")
		return
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected error returned after cancellation: %v", err)
	}
	select {
	case err := <-errCh:
		t.Fatalf("unexpected second error: %v", err)
	case <-successCh:
	}
}
