// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package acme

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"math/big"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"
	"time"
)

// While contents of this file is pertinent only to RFC8555,
// it is complementary to the tests in the other _test.go files
// many of which are valid for both pre- and RFC8555.
// This will make it easier to clean up the tests once non-RFC compliant
// code is removed.

func TestRFC_Discover(t *testing.T) {
	const (
		nonce       = "https://example.com/acme/new-nonce"
		reg         = "https://example.com/acme/new-acct"
		order       = "https://example.com/acme/new-order"
		authz       = "https://example.com/acme/new-authz"
		revoke      = "https://example.com/acme/revoke-cert"
		keychange   = "https://example.com/acme/key-change"
		metaTerms   = "https://example.com/acme/terms/2017-5-30"
		metaWebsite = "https://www.example.com/"
		metaCAA     = "example.com"
	)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{
			"newNonce": %q,
			"newAccount": %q,
			"newOrder": %q,
			"newAuthz": %q,
			"revokeCert": %q,
			"keyChange": %q,
			"meta": {
				"termsOfService": %q,
				"website": %q,
				"caaIdentities": [%q],
				"externalAccountRequired": true
			}
		}`, nonce, reg, order, authz, revoke, keychange, metaTerms, metaWebsite, metaCAA)
	}))
	defer ts.Close()
	c := Client{DirectoryURL: ts.URL}
	dir, err := c.Discover(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if dir.NonceURL != nonce {
		t.Errorf("dir.NonceURL = %q; want %q", dir.NonceURL, nonce)
	}
	if dir.RegURL != reg {
		t.Errorf("dir.RegURL = %q; want %q", dir.RegURL, reg)
	}
	if dir.OrderURL != order {
		t.Errorf("dir.OrderURL = %q; want %q", dir.OrderURL, order)
	}
	if dir.AuthzURL != authz {
		t.Errorf("dir.AuthzURL = %q; want %q", dir.AuthzURL, authz)
	}
	if dir.RevokeURL != revoke {
		t.Errorf("dir.RevokeURL = %q; want %q", dir.RevokeURL, revoke)
	}
	if dir.KeyChangeURL != keychange {
		t.Errorf("dir.KeyChangeURL = %q; want %q", dir.KeyChangeURL, keychange)
	}
	if dir.Terms != metaTerms {
		t.Errorf("dir.Terms = %q; want %q", dir.Terms, metaTerms)
	}
	if dir.Website != metaWebsite {
		t.Errorf("dir.Website = %q; want %q", dir.Website, metaWebsite)
	}
	if len(dir.CAA) == 0 || dir.CAA[0] != metaCAA {
		t.Errorf("dir.CAA = %q; want [%q]", dir.CAA, metaCAA)
	}
	if !dir.ExternalAccountRequired {
		t.Error("dir.Meta.ExternalAccountRequired is false")
	}
}

func TestRFC_popNonce(t *testing.T) {
	var count int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The Client uses only Directory.NonceURL when specified.
		// Expect no other URL paths.
		if r.URL.Path != "/new-nonce" {
			t.Errorf("r.URL.Path = %q; want /new-nonce", r.URL.Path)
		}
		if count > 0 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		count++
		w.Header().Set("Replay-Nonce", "second")
	}))
	cl := &Client{
		DirectoryURL: ts.URL,
		dir:          &Directory{NonceURL: ts.URL + "/new-nonce"},
	}
	cl.addNonce(http.Header{"Replay-Nonce": {"first"}})

	for i, nonce := range []string{"first", "second"} {
		v, err := cl.popNonce(context.Background(), "")
		if err != nil {
			t.Errorf("%d: cl.popNonce: %v", i, err)
		}
		if v != nonce {
			t.Errorf("%d: cl.popNonce = %q; want %q", i, v, nonce)
		}
	}
	// No more nonces and server replies with an error past first nonce fetch.
	// Expected to fail.
	if _, err := cl.popNonce(context.Background(), ""); err == nil {
		t.Error("last cl.popNonce returned nil error")
	}
}

func TestRFC_postKID(t *testing.T) {
	var ts *httptest.Server
	ts = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/new-nonce":
			w.Header().Set("Replay-Nonce", "nonce")
		case "/new-account":
			w.Header().Set("Location", "/account-1")
			w.Write([]byte(`{"status":"valid"}`))
		case "/post":
			b, _ := ioutil.ReadAll(r.Body) // check err later in decodeJWSxxx
			head, err := decodeJWSHead(bytes.NewReader(b))
			if err != nil {
				t.Errorf("decodeJWSHead: %v", err)
				return
			}
			if head.KID != "/account-1" {
				t.Errorf("head.KID = %q; want /account-1", head.KID)
			}
			if len(head.JWK) != 0 {
				t.Errorf("head.JWK = %q; want zero map", head.JWK)
			}
			if v := ts.URL + "/post"; head.URL != v {
				t.Errorf("head.URL = %q; want %q", head.URL, v)
			}

			var payload struct{ Msg string }
			decodeJWSRequest(t, &payload, bytes.NewReader(b))
			if payload.Msg != "ping" {
				t.Errorf("payload.Msg = %q; want ping", payload.Msg)
			}
			w.Write([]byte("pong"))
		default:
			t.Errorf("unhandled %s %s", r.Method, r.URL)
			w.WriteHeader(http.StatusBadRequest)
		}
	}))
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cl := &Client{
		Key:          testKey,
		DirectoryURL: ts.URL,
		dir: &Directory{
			NonceURL: ts.URL + "/new-nonce",
			RegURL:   ts.URL + "/new-account",
			OrderURL: "/force-rfc-mode",
		},
	}
	req := json.RawMessage(`{"msg":"ping"}`)
	res, err := cl.post(ctx, nil /* use kid */, ts.URL+"/post", req, wantStatus(http.StatusOK))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	b, _ := ioutil.ReadAll(res.Body) // don't care about err - just checking b
	if string(b) != "pong" {
		t.Errorf("res.Body = %q; want pong", b)
	}
}

// acmeServer simulates a subset of RFC8555 compliant CA.
//
// TODO: We also have x/crypto/acme/autocert/acmetest and startACMEServerStub in autocert_test.go.
// It feels like this acmeServer is a sweet spot between usefulness and added complexity.
// Also, acmetest and startACMEServerStub were both written for draft-02, no RFC support.
// The goal is to consolidate all into one ACME test server.
type acmeServer struct {
	ts      *httptest.Server
	handler map[string]http.HandlerFunc // keyed by r.URL.Path

	mu     sync.Mutex
	nnonce int
}

func newACMEServer() *acmeServer {
	return &acmeServer{handler: make(map[string]http.HandlerFunc)}
}

func (s *acmeServer) handle(path string, f func(http.ResponseWriter, *http.Request)) {
	s.handler[path] = http.HandlerFunc(f)
}

func (s *acmeServer) start() {
	s.ts = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		// Directory request.
		if r.URL.Path == "/" {
			fmt.Fprintf(w, `{
				"newNonce": %q,
				"newAccount": %q,
				"newOrder": %q,
				"newAuthz": %q,
				"revokeCert": %q,
				"meta": {"termsOfService": %q}
				}`,
				s.url("/acme/new-nonce"),
				s.url("/acme/new-account"),
				s.url("/acme/new-order"),
				s.url("/acme/new-authz"),
				s.url("/acme/revoke-cert"),
				s.url("/terms"),
			)
			return
		}

		// All other responses contain a nonce value unconditionally.
		w.Header().Set("Replay-Nonce", s.nonce())
		if r.URL.Path == "/acme/new-nonce" {
			return
		}

		h := s.handler[r.URL.Path]
		if h == nil {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprintf(w, "Unhandled %s", r.URL.Path)
			return
		}
		h.ServeHTTP(w, r)
	}))
}

func (s *acmeServer) close() {
	s.ts.Close()
}

func (s *acmeServer) url(path string) string {
	return s.ts.URL + path
}

func (s *acmeServer) nonce() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nnonce++
	return fmt.Sprintf("nonce%d", s.nnonce)
}

func (s *acmeServer) error(w http.ResponseWriter, e *wireError) {
	w.WriteHeader(e.Status)
	json.NewEncoder(w).Encode(e)
}

func TestRFC_Register(t *testing.T) {
	const email = "mailto:user@example.org"

	s := newACMEServer()
	s.handle("/acme/new-account", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", s.url("/accounts/1"))
		w.WriteHeader(http.StatusCreated) // 201 means new account created
		fmt.Fprintf(w, `{
			"status": "valid",
			"contact": [%q],
			"orders": %q
		}`, email, s.url("/accounts/1/orders"))

		b, _ := ioutil.ReadAll(r.Body) // check err later in decodeJWSxxx
		head, err := decodeJWSHead(bytes.NewReader(b))
		if err != nil {
			t.Errorf("decodeJWSHead: %v", err)
			return
		}
		if len(head.JWK) == 0 {
			t.Error("head.JWK is empty")
		}

		var req struct{ Contact []string }
		decodeJWSRequest(t, &req, bytes.NewReader(b))
		if len(req.Contact) != 1 || req.Contact[0] != email {
			t.Errorf("req.Contact = %q; want [%q]", req.Contact, email)
		}
	})
	s.start()
	defer s.close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cl := &Client{
		Key:          testKeyEC,
		DirectoryURL: s.url("/"),
	}

	var didPrompt bool
	a := &Account{Contact: []string{email}}
	acct, err := cl.Register(ctx, a, func(tos string) bool {
		didPrompt = true
		terms := s.url("/terms")
		if tos != terms {
			t.Errorf("tos = %q; want %q", tos, terms)
		}
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	okAccount := &Account{
		URI:       s.url("/accounts/1"),
		Status:    StatusValid,
		Contact:   []string{email},
		OrdersURL: s.url("/accounts/1/orders"),
	}
	if !reflect.DeepEqual(acct, okAccount) {
		t.Errorf("acct = %+v; want %+v", acct, okAccount)
	}
	if !didPrompt {
		t.Error("tos prompt wasn't called")
	}
	if v := cl.accountKID(ctx); v != keyID(okAccount.URI) {
		t.Errorf("account kid = %q; want %q", v, okAccount.URI)
	}
}

func TestRFC_RegisterExisting(t *testing.T) {
	s := newACMEServer()
	s.handle("/acme/new-account", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", s.url("/accounts/1"))
		w.WriteHeader(http.StatusOK) // 200 means account already exists
		w.Write([]byte(`{"status": "valid"}`))
	})
	s.start()
	defer s.close()

	cl := &Client{Key: testKeyEC, DirectoryURL: s.url("/")}
	_, err := cl.Register(context.Background(), &Account{}, AcceptTOS)
	if err != ErrAccountAlreadyExists {
		t.Errorf("err = %v; want %v", err, ErrAccountAlreadyExists)
	}
	kid := keyID(s.url("/accounts/1"))
	if v := cl.accountKID(context.Background()); v != kid {
		t.Errorf("account kid = %q; want %q", v, kid)
	}
}

func TestRFC_UpdateReg(t *testing.T) {
	const email = "mailto:user@example.org"

	s := newACMEServer()
	s.handle("/acme/new-account", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", s.url("/accounts/1"))
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status": "valid"}`))
	})
	var didUpdate bool
	s.handle("/accounts/1", func(w http.ResponseWriter, r *http.Request) {
		didUpdate = true
		w.Header().Set("Location", s.url("/accounts/1"))
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status": "valid"}`))

		b, _ := ioutil.ReadAll(r.Body) // check err later in decodeJWSxxx
		head, err := decodeJWSHead(bytes.NewReader(b))
		if err != nil {
			t.Errorf("decodeJWSHead: %v", err)
			return
		}
		if len(head.JWK) != 0 {
			t.Error("head.JWK is non-zero")
		}
		kid := s.url("/accounts/1")
		if head.KID != kid {
			t.Errorf("head.KID = %q; want %q", head.KID, kid)
		}

		var req struct{ Contact []string }
		decodeJWSRequest(t, &req, bytes.NewReader(b))
		if len(req.Contact) != 1 || req.Contact[0] != email {
			t.Errorf("req.Contact = %q; want [%q]", req.Contact, email)
		}
	})
	s.start()
	defer s.close()

	cl := &Client{Key: testKeyEC, DirectoryURL: s.url("/")}
	_, err := cl.UpdateReg(context.Background(), &Account{Contact: []string{email}})
	if err != nil {
		t.Error(err)
	}
	if !didUpdate {
		t.Error("UpdateReg didn't update the account")
	}
}

func TestRFC_GetReg(t *testing.T) {
	s := newACMEServer()
	s.handle("/acme/new-account", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", s.url("/accounts/1"))
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status": "valid"}`))

		head, err := decodeJWSHead(r.Body)
		if err != nil {
			t.Errorf("decodeJWSHead: %v", err)
			return
		}
		if len(head.JWK) == 0 {
			t.Error("head.JWK is empty")
		}
	})
	s.start()
	defer s.close()

	cl := &Client{Key: testKeyEC, DirectoryURL: s.url("/")}
	acct, err := cl.GetReg(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	okAccount := &Account{
		URI:    s.url("/accounts/1"),
		Status: StatusValid,
	}
	if !reflect.DeepEqual(acct, okAccount) {
		t.Errorf("acct = %+v; want %+v", acct, okAccount)
	}
}

func TestRFC_GetRegNoAccount(t *testing.T) {
	s := newACMEServer()
	s.handle("/acme/new-account", func(w http.ResponseWriter, r *http.Request) {
		s.error(w, &wireError{
			Status: http.StatusBadRequest,
			Type:   "urn:ietf:params:acme:error:accountDoesNotExist",
		})
	})
	s.start()
	defer s.close()

	cl := &Client{Key: testKeyEC, DirectoryURL: s.url("/")}
	if _, err := cl.GetReg(context.Background(), ""); err != ErrNoAccount {
		t.Errorf("err = %v; want %v", err, ErrNoAccount)
	}
}

func TestRFC_GetRegOtherError(t *testing.T) {
	s := newACMEServer()
	s.handle("/acme/new-account", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	})
	s.start()
	defer s.close()

	cl := &Client{Key: testKeyEC, DirectoryURL: s.url("/")}
	if _, err := cl.GetReg(context.Background(), ""); err == nil || err == ErrNoAccount {
		t.Errorf("GetReg: %v; want any other non-nil err", err)
	}
}

func TestRFC_AuthorizeOrder(t *testing.T) {
	s := newACMEServer()
	s.handle("/acme/new-account", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", s.url("/accounts/1"))
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status": "valid"}`))
	})
	s.handle("/acme/new-order", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", s.url("/orders/1"))
		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, `{
			"status": "pending",
			"expires": "2019-09-01T00:00:00Z",
			"notBefore": "2019-08-31T00:00:00Z",
			"notAfter": "2019-09-02T00:00:00Z",
			"identifiers": [{"type":"dns", "value":"example.org"}],
			"authorizations": [%q]
		}`, s.url("/authz/1"))
	})
	s.start()
	defer s.close()

	cl := &Client{Key: testKeyEC, DirectoryURL: s.url("/")}
	o, err := cl.AuthorizeOrder(context.Background(), DomainIDs("example.org"),
		WithOrderNotBefore(time.Date(2019, 8, 31, 0, 0, 0, 0, time.UTC)),
		WithOrderNotAfter(time.Date(2019, 9, 2, 0, 0, 0, 0, time.UTC)),
	)
	if err != nil {
		t.Fatal(err)
	}
	okOrder := &Order{
		URI:         s.url("/orders/1"),
		Status:      StatusPending,
		Expires:     time.Date(2019, 9, 1, 0, 0, 0, 0, time.UTC),
		NotBefore:   time.Date(2019, 8, 31, 0, 0, 0, 0, time.UTC),
		NotAfter:    time.Date(2019, 9, 2, 0, 0, 0, 0, time.UTC),
		Identifiers: []AuthzID{AuthzID{Type: "dns", Value: "example.org"}},
		AuthzURLs:   []string{s.url("/authz/1")},
	}
	if !reflect.DeepEqual(o, okOrder) {
		t.Errorf("AuthorizeOrder = %+v; want %+v", o, okOrder)
	}
}

func TestRFC_GetOrder(t *testing.T) {
	s := newACMEServer()
	s.handle("/acme/new-account", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", s.url("/accounts/1"))
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status": "valid"}`))
	})
	s.handle("/orders/1", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", s.url("/orders/1"))
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{
			"status": "invalid",
			"expires": "2019-09-01T00:00:00Z",
			"notBefore": "2019-08-31T00:00:00Z",
			"notAfter": "2019-09-02T00:00:00Z",
			"identifiers": [{"type":"dns", "value":"example.org"}],
			"authorizations": ["/authz/1"],
			"finalize": "/orders/1/fin",
			"certificate": "/orders/1/cert",
			"error": {"type": "badRequest"}
		}`))
	})
	s.start()
	defer s.close()

	cl := &Client{Key: testKeyEC, DirectoryURL: s.url("/")}
	o, err := cl.GetOrder(context.Background(), s.url("/orders/1"))
	if err != nil {
		t.Fatal(err)
	}
	okOrder := &Order{
		URI:         s.url("/orders/1"),
		Status:      StatusInvalid,
		Expires:     time.Date(2019, 9, 1, 0, 0, 0, 0, time.UTC),
		NotBefore:   time.Date(2019, 8, 31, 0, 0, 0, 0, time.UTC),
		NotAfter:    time.Date(2019, 9, 2, 0, 0, 0, 0, time.UTC),
		Identifiers: []AuthzID{AuthzID{Type: "dns", Value: "example.org"}},
		AuthzURLs:   []string{"/authz/1"},
		FinalizeURL: "/orders/1/fin",
		CertURL:     "/orders/1/cert",
		Error:       &Error{ProblemType: "badRequest"},
	}
	if !reflect.DeepEqual(o, okOrder) {
		t.Errorf("GetOrder = %+v\nwant %+v", o, okOrder)
	}
}

func TestRFC_WaitOrder(t *testing.T) {
	for _, st := range []string{StatusReady, StatusValid} {
		t.Run(st, func(t *testing.T) {
			testWaitOrderStatus(t, st)
		})
	}
}

func testWaitOrderStatus(t *testing.T, okStatus string) {
	s := newACMEServer()
	s.handle("/acme/new-account", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", s.url("/accounts/1"))
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status": "valid"}`))
	})
	var count int
	s.handle("/orders/1", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", s.url("/orders/1"))
		w.WriteHeader(http.StatusOK)
		s := StatusPending
		if count > 0 {
			s = okStatus
		}
		fmt.Fprintf(w, `{"status": %q}`, s)
		count++
	})
	s.start()
	defer s.close()

	var order *Order
	var err error
	done := make(chan struct{})
	go func() {
		cl := &Client{Key: testKeyEC, DirectoryURL: s.url("/")}
		order, err = cl.WaitOrder(context.Background(), s.url("/orders/1"))
		close(done)
	}()
	select {
	case <-time.After(3 * time.Second):
		t.Fatal("WaitOrder took too long to return")
	case <-done:
		if err != nil {
			t.Fatalf("WaitOrder: %v", err)
		}
		if order.Status != okStatus {
			t.Errorf("order.Status = %q; want %q", order.Status, okStatus)
		}
	}
}

func TestRFC_WaitOrderError(t *testing.T) {
	s := newACMEServer()
	s.handle("/acme/new-account", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", s.url("/accounts/1"))
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status": "valid"}`))
	})
	var count int
	s.handle("/orders/1", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", s.url("/orders/1"))
		w.WriteHeader(http.StatusOK)
		s := StatusPending
		if count > 0 {
			s = StatusInvalid
		}
		fmt.Fprintf(w, `{"status": %q}`, s)
		count++
	})
	s.start()
	defer s.close()

	var err error
	done := make(chan struct{})
	go func() {
		cl := &Client{Key: testKeyEC, DirectoryURL: s.url("/")}
		_, err = cl.WaitOrder(context.Background(), s.url("/orders/1"))
		close(done)
	}()
	select {
	case <-time.After(3 * time.Second):
		t.Fatal("WaitOrder took too long to return")
	case <-done:
		if err == nil {
			t.Fatal("WaitOrder returned nil error")
		}
		e, ok := err.(*OrderError)
		if !ok {
			t.Fatalf("err = %v (%T); want OrderError", err, err)
		}
		if e.OrderURL != s.url("/orders/1") {
			t.Errorf("e.OrderURL = %q; want %q", e.OrderURL, s.url("/orders/1"))
		}
		if e.Status != StatusInvalid {
			t.Errorf("e.Status = %q; want %q", e.Status, StatusInvalid)
		}
	}
}

func TestRFC_CreateOrderCert(t *testing.T) {
	q := &x509.CertificateRequest{
		Subject: pkix.Name{CommonName: "example.org"},
	}
	csr, err := x509.CreateCertificateRequest(rand.Reader, q, testKeyEC)
	if err != nil {
		t.Fatal(err)
	}

	tmpl := &x509.Certificate{SerialNumber: big.NewInt(1)}
	leaf, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &testKeyEC.PublicKey, testKeyEC)
	if err != nil {
		t.Fatal(err)
	}

	s := newACMEServer()
	s.handle("/acme/new-account", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", s.url("/accounts/1"))
		w.Write([]byte(`{"status": "valid"}`))
	})
	var count int
	s.handle("/pleaseissue", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", s.url("/pleaseissue"))
		st := StatusProcessing
		if count > 0 {
			st = StatusValid
		}
		fmt.Fprintf(w, `{"status":%q, "certificate":%q}`, st, s.url("/crt"))
		count++
	})
	s.handle("/crt", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/pem-certificate-chain")
		pem.Encode(w, &pem.Block{Type: "CERTIFICATE", Bytes: leaf})
	})
	s.start()
	defer s.close()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	cl := &Client{Key: testKeyEC, DirectoryURL: s.url("/")}
	cert, curl, err := cl.CreateOrderCert(ctx, s.url("/pleaseissue"), csr, true)
	if err != nil {
		t.Fatalf("CreateOrderCert: %v", err)
	}
	if _, err := x509.ParseCertificate(cert[0]); err != nil {
		t.Errorf("ParseCertificate: %v", err)
	}
	if !reflect.DeepEqual(cert[0], leaf) {
		t.Errorf("cert and leaf bytes don't match")
	}
	if u := s.url("/crt"); curl != u {
		t.Errorf("curl = %q; want %q", curl, u)
	}
}

func TestRFC_AlreadyRevokedCert(t *testing.T) {
	s := newACMEServer()
	s.handle("/acme/revoke-cert", func(w http.ResponseWriter, r *http.Request) {
		s.error(w, &wireError{
			Status: http.StatusBadRequest,
			Type:   "urn:ietf:params:acme:error:alreadyRevoked",
		})
	})
	s.start()
	defer s.close()

	cl := &Client{Key: testKeyEC, DirectoryURL: s.url("/")}
	err := cl.RevokeCert(context.Background(), testKeyEC, []byte{0}, CRLReasonUnspecified)
	if err != nil {
		t.Fatalf("RevokeCert: %v", err)
	}
}
