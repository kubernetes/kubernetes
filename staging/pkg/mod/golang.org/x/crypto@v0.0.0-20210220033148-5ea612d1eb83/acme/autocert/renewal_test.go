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
	man := &Manager{
		RenewBefore: 7 * 24 * time.Hour,
		nowFunc:     func() time.Time { return now },
	}
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
		// authorization status request done by Manager's revokePendingAuthz.
		case "/authz/1":
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
			der, err := dummyCert(csr.PublicKey, exampleDomain)
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

	man := &Manager{
		Prompt:      AcceptTOS,
		Cache:       newMemCache(t),
		RenewBefore: 24 * time.Hour,
		Client: &acme.Client{
			DirectoryURL: ca.URL,
		},
	}
	defer man.stopRenew()

	// cache an almost expired cert
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	cert, err := dateDummyCert(key.Public(), now.Add(-2*time.Hour), now.Add(time.Minute), exampleDomain)
	if err != nil {
		t.Fatal(err)
	}
	tlscert := &tls.Certificate{PrivateKey: key, Certificate: [][]byte{cert}}
	if err := man.cachePut(context.Background(), exampleCertKey, tlscert); err != nil {
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
		tlscert, err := man.cacheGet(context.Background(), exampleCertKey)
		if err != nil {
			t.Fatalf("man.cacheGet: %v", err)
		}
		if !tlscert.Leaf.NotAfter.After(after) {
			t.Errorf("cache leaf.NotAfter = %v; want > %v", tlscert.Leaf.NotAfter, after)
		}

		// verify the old cert is also replaced in memory
		man.stateMu.Lock()
		defer man.stateMu.Unlock()
		s := man.state[exampleCertKey]
		if s == nil {
			t.Fatalf("m.state[%q] is nil", exampleCertKey)
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
	hello := clientHelloInfo(exampleDomain, algECDSA)
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

func TestRenewFromCacheAlreadyRenewed(t *testing.T) {
	man := &Manager{
		Prompt:      AcceptTOS,
		Cache:       newMemCache(t),
		RenewBefore: 24 * time.Hour,
		Client: &acme.Client{
			DirectoryURL: "invalid",
		},
	}
	defer man.stopRenew()

	// cache a recently renewed cert with a different private key
	newKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	newCert, err := dateDummyCert(newKey.Public(), now.Add(-2*time.Hour), now.Add(time.Hour*24*90), exampleDomain)
	if err != nil {
		t.Fatal(err)
	}
	newLeaf, err := validCert(exampleCertKey, [][]byte{newCert}, newKey, now)
	if err != nil {
		t.Fatal(err)
	}
	newTLSCert := &tls.Certificate{PrivateKey: newKey, Certificate: [][]byte{newCert}, Leaf: newLeaf}
	if err := man.cachePut(context.Background(), exampleCertKey, newTLSCert); err != nil {
		t.Fatal(err)
	}

	// set internal state to an almost expired cert
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	oldCert, err := dateDummyCert(key.Public(), now.Add(-2*time.Hour), now.Add(time.Minute), exampleDomain)
	if err != nil {
		t.Fatal(err)
	}
	oldLeaf, err := validCert(exampleCertKey, [][]byte{oldCert}, key, now)
	if err != nil {
		t.Fatal(err)
	}
	man.stateMu.Lock()
	if man.state == nil {
		man.state = make(map[certKey]*certState)
	}
	s := &certState{
		key:  key,
		cert: [][]byte{oldCert},
		leaf: oldLeaf,
	}
	man.state[exampleCertKey] = s
	man.stateMu.Unlock()

	// veriy the renewal accepted the newer cached cert
	defer func() {
		testDidRenewLoop = func(next time.Duration, err error) {}
	}()
	done := make(chan struct{})
	testDidRenewLoop = func(next time.Duration, err error) {
		defer close(done)
		if err != nil {
			t.Errorf("testDidRenewLoop: %v", err)
		}
		// Next should be about 90 days
		// Previous expiration was within 1 min.
		future := 88 * 24 * time.Hour
		if next < future {
			t.Errorf("testDidRenewLoop: next = %v; want >= %v", next, future)
		}

		// ensure the cached cert was not modified
		tlscert, err := man.cacheGet(context.Background(), exampleCertKey)
		if err != nil {
			t.Fatalf("man.cacheGet: %v", err)
		}
		if !tlscert.Leaf.NotAfter.Equal(newLeaf.NotAfter) {
			t.Errorf("cache leaf.NotAfter = %v; want == %v", tlscert.Leaf.NotAfter, newLeaf.NotAfter)
		}

		// verify the old cert is also replaced in memory
		man.stateMu.Lock()
		defer man.stateMu.Unlock()
		s := man.state[exampleCertKey]
		if s == nil {
			t.Fatalf("m.state[%q] is nil", exampleCertKey)
		}
		stateKey := s.key.Public().(*ecdsa.PublicKey)
		if stateKey.X.Cmp(newKey.X) != 0 || stateKey.Y.Cmp(newKey.Y) != 0 {
			t.Fatalf("state key was not updated from cache x: %v y: %v; want x: %v y: %v", stateKey.X, stateKey.Y, newKey.X, newKey.Y)
		}
		tlscert, err = s.tlscert()
		if err != nil {
			t.Fatalf("s.tlscert: %v", err)
		}
		if !tlscert.Leaf.NotAfter.Equal(newLeaf.NotAfter) {
			t.Errorf("state leaf.NotAfter = %v; want == %v", tlscert.Leaf.NotAfter, newLeaf.NotAfter)
		}

		// verify the private key is replaced in the renewal state
		r := man.renewal[exampleCertKey]
		if r == nil {
			t.Fatalf("m.renewal[%q] is nil", exampleCertKey)
		}
		renewalKey := r.key.Public().(*ecdsa.PublicKey)
		if renewalKey.X.Cmp(newKey.X) != 0 || renewalKey.Y.Cmp(newKey.Y) != 0 {
			t.Fatalf("renewal private key was not updated from cache x: %v y: %v; want x: %v y: %v", renewalKey.X, renewalKey.Y, newKey.X, newKey.Y)
		}

	}

	// assert the expiring cert is returned from state
	hello := clientHelloInfo(exampleDomain, algECDSA)
	tlscert, err := man.GetCertificate(hello)
	if err != nil {
		t.Fatal(err)
	}
	if !oldLeaf.NotAfter.Equal(tlscert.Leaf.NotAfter) {
		t.Errorf("state leaf.NotAfter = %v; want == %v", tlscert.Leaf.NotAfter, oldLeaf.NotAfter)
	}

	// trigger renew
	go man.renew(exampleCertKey, s.key, s.leaf.NotAfter)

	// wait for renew loop
	select {
	case <-time.After(10 * time.Second):
		t.Fatal("renew took too long to occur")
	case <-done:
		// assert the new cert is returned from state after renew
		hello := clientHelloInfo(exampleDomain, algECDSA)
		tlscert, err := man.GetCertificate(hello)
		if err != nil {
			t.Fatal(err)
		}
		if !newTLSCert.Leaf.NotAfter.Equal(tlscert.Leaf.NotAfter) {
			t.Errorf("state leaf.NotAfter = %v; want == %v", tlscert.Leaf.NotAfter, newTLSCert.Leaf.NotAfter)
		}
	}
}
