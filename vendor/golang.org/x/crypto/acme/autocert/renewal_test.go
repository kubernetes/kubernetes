// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package autocert

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"golang.org/x/crypto/acme"
)

func TestRenewalNext(t *testing.T) {
	now := time.Now()
	timeNow = func() time.Time { return now }
	defer func() { timeNow = time.Now }()

	man := &Manager{RenewBefore: 7 * 24 * time.Hour}
	defer man.stopRenew()
	tt := []struct {
		expiry   time.Time
		min, max time.Duration
	}{
		{now.Add(90 * 24 * time.Hour), 83*24*time.Hour - renewJitter, 83 * 24 * time.Hour},
		{now.Add(time.Hour), 0, 1},
		{now, 0, 1},
		{now.Add(-time.Hour), 0, 1},
	}

	dr := &domainRenewal{m: man}
	for i, test := range tt {
		next := dr.next(test.expiry)
		if next < test.min || test.max < next {
			t.Errorf("%d: next = %v; want between %v and %v", i, next, test.min, test.max)
		}
	}
}

func TestRenewFromCache(t *testing.T) {
	const domain = "example.org"

	// ACME CA server stub
	var ca *httptest.Server
	ca = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Replay-Nonce", "nonce")
		if r.Method == "HEAD" {
			// a nonce request
			return
		}

		switch r.URL.Path {
		// discovery
		case "/":
			if err := discoTmpl.Execute(w, ca.URL); err != nil {
				t.Fatalf("discoTmpl: %v", err)
			}
		// client key registration
		case "/new-reg":
			w.Write([]byte("{}"))
		// domain authorization
		case "/new-authz":
			w.Header().Set("Location", ca.URL+"/authz/1")
			w.WriteHeader(http.StatusCreated)
			w.Write([]byte(`{"status": "valid"}`))
		// cert request
		case "/new-cert":
			var req struct {
				CSR string `json:"csr"`
			}
			decodePayload(&req, r.Body)
			b, _ := base64.RawURLEncoding.DecodeString(req.CSR)
			csr, err := x509.ParseCertificateRequest(b)
			if err != nil {
				t.Fatalf("new-cert: CSR: %v", err)
			}
			der, err := dummyCert(csr.PublicKey, domain)
			if err != nil {
				t.Fatalf("new-cert: dummyCert: %v", err)
			}
			chainUp := fmt.Sprintf("<%s/ca-cert>; rel=up", ca.URL)
			w.Header().Set("Link", chainUp)
			w.WriteHeader(http.StatusCreated)
			w.Write(der)
		// CA chain cert
		case "/ca-cert":
			der, err := dummyCert(nil, "ca")
			if err != nil {
				t.Fatalf("ca-cert: dummyCert: %v", err)
			}
			w.Write(der)
		default:
			t.Errorf("unrecognized r.URL.Path: %s", r.URL.Path)
		}
	}))
	defer ca.Close()

	// use EC key to run faster on 386
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	man := &Manager{
		Prompt:      AcceptTOS,
		Cache:       newMemCache(),
		RenewBefore: 24 * time.Hour,
		Client: &acme.Client{
			Key:          key,
			DirectoryURL: ca.URL,
		},
	}
	defer man.stopRenew()

	// cache an almost expired cert
	now := time.Now()
	cert, err := dateDummyCert(key.Public(), now.Add(-2*time.Hour), now.Add(time.Minute), domain)
	if err != nil {
		t.Fatal(err)
	}
	tlscert := &tls.Certificate{PrivateKey: key, Certificate: [][]byte{cert}}
	if err := man.cachePut(context.Background(), domain, tlscert); err != nil {
		t.Fatal(err)
	}

	// veriy the renewal happened
	defer func() {
		testDidRenewLoop = func(next time.Duration, err error) {}
	}()
	done := make(chan struct{})
	testDidRenewLoop = func(next time.Duration, err error) {
		defer close(done)
		if err != nil {
			t.Errorf("testDidRenewLoop: %v", err)
		}
		// Next should be about 90 days:
		// dummyCert creates 90days expiry + account for man.RenewBefore.
		// Previous expiration was within 1 min.
		future := 88 * 24 * time.Hour
		if next < future {
			t.Errorf("testDidRenewLoop: next = %v; want >= %v", next, future)
		}

		// ensure the new cert is cached
		after := time.Now().Add(future)
		tlscert, err := man.cacheGet(context.Background(), domain)
		if err != nil {
			t.Fatalf("man.cacheGet: %v", err)
		}
		if !tlscert.Leaf.NotAfter.After(after) {
			t.Errorf("cache leaf.NotAfter = %v; want > %v", tlscert.Leaf.NotAfter, after)
		}

		// verify the old cert is also replaced in memory
		man.stateMu.Lock()
		defer man.stateMu.Unlock()
		s := man.state[domain]
		if s == nil {
			t.Fatalf("m.state[%q] is nil", domain)
		}
		tlscert, err = s.tlscert()
		if err != nil {
			t.Fatalf("s.tlscert: %v", err)
		}
		if !tlscert.Leaf.NotAfter.After(after) {
			t.Errorf("state leaf.NotAfter = %v; want > %v", tlscert.Leaf.NotAfter, after)
		}
	}

	// trigger renew
	hello := &tls.ClientHelloInfo{ServerName: domain}
	if _, err := man.GetCertificate(hello); err != nil {
		t.Fatal(err)
	}

	// wait for renew loop
	select {
	case <-time.After(10 * time.Second):
		t.Fatal("renew took too long to occur")
	case <-done:
	}
}
