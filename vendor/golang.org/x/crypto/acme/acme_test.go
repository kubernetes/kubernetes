// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package acme

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/big"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"
)

// Decodes a JWS-encoded request and unmarshals the decoded JSON into a provided
// interface.
func decodeJWSRequest(t *testing.T, v interface{}, r *http.Request) {
	// Decode request
	var req struct{ Payload string }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		t.Fatal(err)
	}
	payload, err := base64.RawURLEncoding.DecodeString(req.Payload)
	if err != nil {
		t.Fatal(err)
	}
	err = json.Unmarshal(payload, v)
	if err != nil {
		t.Fatal(err)
	}
}

type jwsHead struct {
	Alg   string
	Nonce string
	JWK   map[string]string `json:"jwk"`
}

func decodeJWSHead(r *http.Request) (*jwsHead, error) {
	var req struct{ Protected string }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return nil, err
	}
	b, err := base64.RawURLEncoding.DecodeString(req.Protected)
	if err != nil {
		return nil, err
	}
	var head jwsHead
	if err := json.Unmarshal(b, &head); err != nil {
		return nil, err
	}
	return &head, nil
}

func TestDiscover(t *testing.T) {
	const (
		reg    = "https://example.com/acme/new-reg"
		authz  = "https://example.com/acme/new-authz"
		cert   = "https://example.com/acme/new-cert"
		revoke = "https://example.com/acme/revoke-cert"
	)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{
			"new-reg": %q,
			"new-authz": %q,
			"new-cert": %q,
			"revoke-cert": %q
		}`, reg, authz, cert, revoke)
	}))
	defer ts.Close()
	c := Client{DirectoryURL: ts.URL}
	dir, err := c.Discover(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if dir.RegURL != reg {
		t.Errorf("dir.RegURL = %q; want %q", dir.RegURL, reg)
	}
	if dir.AuthzURL != authz {
		t.Errorf("dir.AuthzURL = %q; want %q", dir.AuthzURL, authz)
	}
	if dir.CertURL != cert {
		t.Errorf("dir.CertURL = %q; want %q", dir.CertURL, cert)
	}
	if dir.RevokeURL != revoke {
		t.Errorf("dir.RevokeURL = %q; want %q", dir.RevokeURL, revoke)
	}
}

func TestRegister(t *testing.T) {
	contacts := []string{"mailto:admin@example.com"}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "HEAD" {
			w.Header().Set("Replay-Nonce", "test-nonce")
			return
		}
		if r.Method != "POST" {
			t.Errorf("r.Method = %q; want POST", r.Method)
		}

		var j struct {
			Resource  string
			Contact   []string
			Agreement string
		}
		decodeJWSRequest(t, &j, r)

		// Test request
		if j.Resource != "new-reg" {
			t.Errorf("j.Resource = %q; want new-reg", j.Resource)
		}
		if !reflect.DeepEqual(j.Contact, contacts) {
			t.Errorf("j.Contact = %v; want %v", j.Contact, contacts)
		}

		w.Header().Set("Location", "https://ca.tld/acme/reg/1")
		w.Header().Set("Link", `<https://ca.tld/acme/new-authz>;rel="next"`)
		w.Header().Add("Link", `<https://ca.tld/acme/recover-reg>;rel="recover"`)
		w.Header().Add("Link", `<https://ca.tld/acme/terms>;rel="terms-of-service"`)
		w.WriteHeader(http.StatusCreated)
		b, _ := json.Marshal(contacts)
		fmt.Fprintf(w, `{"contact": %s}`, b)
	}))
	defer ts.Close()

	prompt := func(url string) bool {
		const terms = "https://ca.tld/acme/terms"
		if url != terms {
			t.Errorf("prompt url = %q; want %q", url, terms)
		}
		return false
	}

	c := Client{Key: testKeyEC, dir: &Directory{RegURL: ts.URL}}
	a := &Account{Contact: contacts}
	var err error
	if a, err = c.Register(context.Background(), a, prompt); err != nil {
		t.Fatal(err)
	}
	if a.URI != "https://ca.tld/acme/reg/1" {
		t.Errorf("a.URI = %q; want https://ca.tld/acme/reg/1", a.URI)
	}
	if a.Authz != "https://ca.tld/acme/new-authz" {
		t.Errorf("a.Authz = %q; want https://ca.tld/acme/new-authz", a.Authz)
	}
	if a.CurrentTerms != "https://ca.tld/acme/terms" {
		t.Errorf("a.CurrentTerms = %q; want https://ca.tld/acme/terms", a.CurrentTerms)
	}
	if !reflect.DeepEqual(a.Contact, contacts) {
		t.Errorf("a.Contact = %v; want %v", a.Contact, contacts)
	}
}

func TestUpdateReg(t *testing.T) {
	const terms = "https://ca.tld/acme/terms"
	contacts := []string{"mailto:admin@example.com"}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "HEAD" {
			w.Header().Set("Replay-Nonce", "test-nonce")
			return
		}
		if r.Method != "POST" {
			t.Errorf("r.Method = %q; want POST", r.Method)
		}

		var j struct {
			Resource  string
			Contact   []string
			Agreement string
		}
		decodeJWSRequest(t, &j, r)

		// Test request
		if j.Resource != "reg" {
			t.Errorf("j.Resource = %q; want reg", j.Resource)
		}
		if j.Agreement != terms {
			t.Errorf("j.Agreement = %q; want %q", j.Agreement, terms)
		}
		if !reflect.DeepEqual(j.Contact, contacts) {
			t.Errorf("j.Contact = %v; want %v", j.Contact, contacts)
		}

		w.Header().Set("Link", `<https://ca.tld/acme/new-authz>;rel="next"`)
		w.Header().Add("Link", `<https://ca.tld/acme/recover-reg>;rel="recover"`)
		w.Header().Add("Link", fmt.Sprintf(`<%s>;rel="terms-of-service"`, terms))
		w.WriteHeader(http.StatusOK)
		b, _ := json.Marshal(contacts)
		fmt.Fprintf(w, `{"contact":%s, "agreement":%q}`, b, terms)
	}))
	defer ts.Close()

	c := Client{Key: testKeyEC}
	a := &Account{URI: ts.URL, Contact: contacts, AgreedTerms: terms}
	var err error
	if a, err = c.UpdateReg(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	if a.Authz != "https://ca.tld/acme/new-authz" {
		t.Errorf("a.Authz = %q; want https://ca.tld/acme/new-authz", a.Authz)
	}
	if a.AgreedTerms != terms {
		t.Errorf("a.AgreedTerms = %q; want %q", a.AgreedTerms, terms)
	}
	if a.CurrentTerms != terms {
		t.Errorf("a.CurrentTerms = %q; want %q", a.CurrentTerms, terms)
	}
	if a.URI != ts.URL {
		t.Errorf("a.URI = %q; want %q", a.URI, ts.URL)
	}
}

func TestGetReg(t *testing.T) {
	const terms = "https://ca.tld/acme/terms"
	const newTerms = "https://ca.tld/acme/new-terms"
	contacts := []string{"mailto:admin@example.com"}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "HEAD" {
			w.Header().Set("Replay-Nonce", "test-nonce")
			return
		}
		if r.Method != "POST" {
			t.Errorf("r.Method = %q; want POST", r.Method)
		}

		var j struct {
			Resource  string
			Contact   []string
			Agreement string
		}
		decodeJWSRequest(t, &j, r)

		// Test request
		if j.Resource != "reg" {
			t.Errorf("j.Resource = %q; want reg", j.Resource)
		}
		if len(j.Contact) != 0 {
			t.Errorf("j.Contact = %v", j.Contact)
		}
		if j.Agreement != "" {
			t.Errorf("j.Agreement = %q", j.Agreement)
		}

		w.Header().Set("Link", `<https://ca.tld/acme/new-authz>;rel="next"`)
		w.Header().Add("Link", `<https://ca.tld/acme/recover-reg>;rel="recover"`)
		w.Header().Add("Link", fmt.Sprintf(`<%s>;rel="terms-of-service"`, newTerms))
		w.WriteHeader(http.StatusOK)
		b, _ := json.Marshal(contacts)
		fmt.Fprintf(w, `{"contact":%s, "agreement":%q}`, b, terms)
	}))
	defer ts.Close()

	c := Client{Key: testKeyEC}
	a, err := c.GetReg(context.Background(), ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if a.Authz != "https://ca.tld/acme/new-authz" {
		t.Errorf("a.AuthzURL = %q; want https://ca.tld/acme/new-authz", a.Authz)
	}
	if a.AgreedTerms != terms {
		t.Errorf("a.AgreedTerms = %q; want %q", a.AgreedTerms, terms)
	}
	if a.CurrentTerms != newTerms {
		t.Errorf("a.CurrentTerms = %q; want %q", a.CurrentTerms, newTerms)
	}
	if a.URI != ts.URL {
		t.Errorf("a.URI = %q; want %q", a.URI, ts.URL)
	}
}

func TestAuthorize(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "HEAD" {
			w.Header().Set("Replay-Nonce", "test-nonce")
			return
		}
		if r.Method != "POST" {
			t.Errorf("r.Method = %q; want POST", r.Method)
		}

		var j struct {
			Resource   string
			Identifier struct {
				Type  string
				Value string
			}
		}
		decodeJWSRequest(t, &j, r)

		// Test request
		if j.Resource != "new-authz" {
			t.Errorf("j.Resource = %q; want new-authz", j.Resource)
		}
		if j.Identifier.Type != "dns" {
			t.Errorf("j.Identifier.Type = %q; want dns", j.Identifier.Type)
		}
		if j.Identifier.Value != "example.com" {
			t.Errorf("j.Identifier.Value = %q; want example.com", j.Identifier.Value)
		}

		w.Header().Set("Location", "https://ca.tld/acme/auth/1")
		w.WriteHeader(http.StatusCreated)
		fmt.Fprintf(w, `{
			"identifier": {"type":"dns","value":"example.com"},
			"status":"pending",
			"challenges":[
				{
					"type":"http-01",
					"status":"pending",
					"uri":"https://ca.tld/acme/challenge/publickey/id1",
					"token":"token1"
				},
				{
					"type":"tls-sni-01",
					"status":"pending",
					"uri":"https://ca.tld/acme/challenge/publickey/id2",
					"token":"token2"
				}
			],
			"combinations":[[0],[1]]}`)
	}))
	defer ts.Close()

	cl := Client{Key: testKeyEC, dir: &Directory{AuthzURL: ts.URL}}
	auth, err := cl.Authorize(context.Background(), "example.com")
	if err != nil {
		t.Fatal(err)
	}

	if auth.URI != "https://ca.tld/acme/auth/1" {
		t.Errorf("URI = %q; want https://ca.tld/acme/auth/1", auth.URI)
	}
	if auth.Status != "pending" {
		t.Errorf("Status = %q; want pending", auth.Status)
	}
	if auth.Identifier.Type != "dns" {
		t.Errorf("Identifier.Type = %q; want dns", auth.Identifier.Type)
	}
	if auth.Identifier.Value != "example.com" {
		t.Errorf("Identifier.Value = %q; want example.com", auth.Identifier.Value)
	}

	if n := len(auth.Challenges); n != 2 {
		t.Fatalf("len(auth.Challenges) = %d; want 2", n)
	}

	c := auth.Challenges[0]
	if c.Type != "http-01" {
		t.Errorf("c.Type = %q; want http-01", c.Type)
	}
	if c.URI != "https://ca.tld/acme/challenge/publickey/id1" {
		t.Errorf("c.URI = %q; want https://ca.tld/acme/challenge/publickey/id1", c.URI)
	}
	if c.Token != "token1" {
		t.Errorf("c.Token = %q; want token1", c.Token)
	}

	c = auth.Challenges[1]
	if c.Type != "tls-sni-01" {
		t.Errorf("c.Type = %q; want tls-sni-01", c.Type)
	}
	if c.URI != "https://ca.tld/acme/challenge/publickey/id2" {
		t.Errorf("c.URI = %q; want https://ca.tld/acme/challenge/publickey/id2", c.URI)
	}
	if c.Token != "token2" {
		t.Errorf("c.Token = %q; want token2", c.Token)
	}

	combs := [][]int{{0}, {1}}
	if !reflect.DeepEqual(auth.Combinations, combs) {
		t.Errorf("auth.Combinations: %+v\nwant: %+v\n", auth.Combinations, combs)
	}
}

func TestAuthorizeValid(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "HEAD" {
			w.Header().Set("Replay-Nonce", "nonce")
			return
		}
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte(`{"status":"valid"}`))
	}))
	defer ts.Close()
	client := Client{Key: testKey, dir: &Directory{AuthzURL: ts.URL}}
	_, err := client.Authorize(context.Background(), "example.com")
	if err != nil {
		t.Errorf("err = %v", err)
	}
}

func TestGetAuthorization(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			t.Errorf("r.Method = %q; want GET", r.Method)
		}

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{
			"identifier": {"type":"dns","value":"example.com"},
			"status":"pending",
			"challenges":[
				{
					"type":"http-01",
					"status":"pending",
					"uri":"https://ca.tld/acme/challenge/publickey/id1",
					"token":"token1"
				},
				{
					"type":"tls-sni-01",
					"status":"pending",
					"uri":"https://ca.tld/acme/challenge/publickey/id2",
					"token":"token2"
				}
			],
			"combinations":[[0],[1]]}`)
	}))
	defer ts.Close()

	cl := Client{Key: testKeyEC}
	auth, err := cl.GetAuthorization(context.Background(), ts.URL)
	if err != nil {
		t.Fatal(err)
	}

	if auth.Status != "pending" {
		t.Errorf("Status = %q; want pending", auth.Status)
	}
	if auth.Identifier.Type != "dns" {
		t.Errorf("Identifier.Type = %q; want dns", auth.Identifier.Type)
	}
	if auth.Identifier.Value != "example.com" {
		t.Errorf("Identifier.Value = %q; want example.com", auth.Identifier.Value)
	}

	if n := len(auth.Challenges); n != 2 {
		t.Fatalf("len(set.Challenges) = %d; want 2", n)
	}

	c := auth.Challenges[0]
	if c.Type != "http-01" {
		t.Errorf("c.Type = %q; want http-01", c.Type)
	}
	if c.URI != "https://ca.tld/acme/challenge/publickey/id1" {
		t.Errorf("c.URI = %q; want https://ca.tld/acme/challenge/publickey/id1", c.URI)
	}
	if c.Token != "token1" {
		t.Errorf("c.Token = %q; want token1", c.Token)
	}

	c = auth.Challenges[1]
	if c.Type != "tls-sni-01" {
		t.Errorf("c.Type = %q; want tls-sni-01", c.Type)
	}
	if c.URI != "https://ca.tld/acme/challenge/publickey/id2" {
		t.Errorf("c.URI = %q; want https://ca.tld/acme/challenge/publickey/id2", c.URI)
	}
	if c.Token != "token2" {
		t.Errorf("c.Token = %q; want token2", c.Token)
	}

	combs := [][]int{{0}, {1}}
	if !reflect.DeepEqual(auth.Combinations, combs) {
		t.Errorf("auth.Combinations: %+v\nwant: %+v\n", auth.Combinations, combs)
	}
}

func TestWaitAuthorization(t *testing.T) {
	var count int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count++
		w.Header().Set("Retry-After", "0")
		if count > 1 {
			fmt.Fprintf(w, `{"status":"valid"}`)
			return
		}
		fmt.Fprintf(w, `{"status":"pending"}`)
	}))
	defer ts.Close()

	type res struct {
		authz *Authorization
		err   error
	}
	done := make(chan res)
	defer close(done)
	go func() {
		var client Client
		a, err := client.WaitAuthorization(context.Background(), ts.URL)
		done <- res{a, err}
	}()

	select {
	case <-time.After(5 * time.Second):
		t.Fatal("WaitAuthz took too long to return")
	case res := <-done:
		if res.err != nil {
			t.Fatalf("res.err =  %v", res.err)
		}
		if res.authz == nil {
			t.Fatal("res.authz is nil")
		}
	}
}

func TestWaitAuthorizationInvalid(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `{"status":"invalid"}`)
	}))
	defer ts.Close()

	res := make(chan error)
	defer close(res)
	go func() {
		var client Client
		_, err := client.WaitAuthorization(context.Background(), ts.URL)
		res <- err
	}()

	select {
	case <-time.After(3 * time.Second):
		t.Fatal("WaitAuthz took too long to return")
	case err := <-res:
		if err == nil {
			t.Error("err is nil")
		}
		if _, ok := err.(*AuthorizationError); !ok {
			t.Errorf("err is %T; want *AuthorizationError", err)
		}
	}
}

func TestWaitAuthorizationCancel(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "60")
		fmt.Fprintf(w, `{"status":"pending"}`)
	}))
	defer ts.Close()

	res := make(chan error)
	defer close(res)
	go func() {
		var client Client
		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		defer cancel()
		_, err := client.WaitAuthorization(ctx, ts.URL)
		res <- err
	}()

	select {
	case <-time.After(time.Second):
		t.Fatal("WaitAuthz took too long to return")
	case err := <-res:
		if err == nil {
			t.Error("err is nil")
		}
	}
}

func TestRevokeAuthorization(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "HEAD" {
			w.Header().Set("Replay-Nonce", "nonce")
			return
		}
		switch r.URL.Path {
		case "/1":
			var req struct {
				Resource string
				Status   string
				Delete   bool
			}
			decodeJWSRequest(t, &req, r)
			if req.Resource != "authz" {
				t.Errorf("req.Resource = %q; want authz", req.Resource)
			}
			if req.Status != "deactivated" {
				t.Errorf("req.Status = %q; want deactivated", req.Status)
			}
			if !req.Delete {
				t.Errorf("req.Delete is false")
			}
		case "/2":
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer ts.Close()
	client := &Client{Key: testKey}
	ctx := context.Background()
	if err := client.RevokeAuthorization(ctx, ts.URL+"/1"); err != nil {
		t.Errorf("err = %v", err)
	}
	if client.RevokeAuthorization(ctx, ts.URL+"/2") == nil {
		t.Error("nil error")
	}
}

func TestPollChallenge(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			t.Errorf("r.Method = %q; want GET", r.Method)
		}

		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{
			"type":"http-01",
			"status":"pending",
			"uri":"https://ca.tld/acme/challenge/publickey/id1",
			"token":"token1"}`)
	}))
	defer ts.Close()

	cl := Client{Key: testKeyEC}
	chall, err := cl.GetChallenge(context.Background(), ts.URL)
	if err != nil {
		t.Fatal(err)
	}

	if chall.Status != "pending" {
		t.Errorf("Status = %q; want pending", chall.Status)
	}
	if chall.Type != "http-01" {
		t.Errorf("c.Type = %q; want http-01", chall.Type)
	}
	if chall.URI != "https://ca.tld/acme/challenge/publickey/id1" {
		t.Errorf("c.URI = %q; want https://ca.tld/acme/challenge/publickey/id1", chall.URI)
	}
	if chall.Token != "token1" {
		t.Errorf("c.Token = %q; want token1", chall.Token)
	}
}

func TestAcceptChallenge(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "HEAD" {
			w.Header().Set("Replay-Nonce", "test-nonce")
			return
		}
		if r.Method != "POST" {
			t.Errorf("r.Method = %q; want POST", r.Method)
		}

		var j struct {
			Resource string
			Type     string
			Auth     string `json:"keyAuthorization"`
		}
		decodeJWSRequest(t, &j, r)

		// Test request
		if j.Resource != "challenge" {
			t.Errorf(`resource = %q; want "challenge"`, j.Resource)
		}
		if j.Type != "http-01" {
			t.Errorf(`type = %q; want "http-01"`, j.Type)
		}
		keyAuth := "token1." + testKeyECThumbprint
		if j.Auth != keyAuth {
			t.Errorf(`keyAuthorization = %q; want %q`, j.Auth, keyAuth)
		}

		// Respond to request
		w.WriteHeader(http.StatusAccepted)
		fmt.Fprintf(w, `{
			"type":"http-01",
			"status":"pending",
			"uri":"https://ca.tld/acme/challenge/publickey/id1",
			"token":"token1",
			"keyAuthorization":%q
		}`, keyAuth)
	}))
	defer ts.Close()

	cl := Client{Key: testKeyEC}
	c, err := cl.Accept(context.Background(), &Challenge{
		URI:   ts.URL,
		Token: "token1",
		Type:  "http-01",
	})
	if err != nil {
		t.Fatal(err)
	}

	if c.Type != "http-01" {
		t.Errorf("c.Type = %q; want http-01", c.Type)
	}
	if c.URI != "https://ca.tld/acme/challenge/publickey/id1" {
		t.Errorf("c.URI = %q; want https://ca.tld/acme/challenge/publickey/id1", c.URI)
	}
	if c.Token != "token1" {
		t.Errorf("c.Token = %q; want token1", c.Token)
	}
}

func TestNewCert(t *testing.T) {
	notBefore := time.Now()
	notAfter := notBefore.AddDate(0, 2, 0)
	timeNow = func() time.Time { return notBefore }

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "HEAD" {
			w.Header().Set("Replay-Nonce", "test-nonce")
			return
		}
		if r.Method != "POST" {
			t.Errorf("r.Method = %q; want POST", r.Method)
		}

		var j struct {
			Resource  string `json:"resource"`
			CSR       string `json:"csr"`
			NotBefore string `json:"notBefore,omitempty"`
			NotAfter  string `json:"notAfter,omitempty"`
		}
		decodeJWSRequest(t, &j, r)

		// Test request
		if j.Resource != "new-cert" {
			t.Errorf(`resource = %q; want "new-cert"`, j.Resource)
		}
		if j.NotBefore != notBefore.Format(time.RFC3339) {
			t.Errorf(`notBefore = %q; wanted %q`, j.NotBefore, notBefore.Format(time.RFC3339))
		}
		if j.NotAfter != notAfter.Format(time.RFC3339) {
			t.Errorf(`notAfter = %q; wanted %q`, j.NotAfter, notAfter.Format(time.RFC3339))
		}

		// Respond to request
		template := x509.Certificate{
			SerialNumber: big.NewInt(int64(1)),
			Subject: pkix.Name{
				Organization: []string{"goacme"},
			},
			NotBefore: notBefore,
			NotAfter:  notAfter,

			KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
			ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
			BasicConstraintsValid: true,
		}

		sampleCert, err := x509.CreateCertificate(rand.Reader, &template, &template, &testKeyEC.PublicKey, testKeyEC)
		if err != nil {
			t.Fatalf("Error creating certificate: %v", err)
		}

		w.Header().Set("Location", "https://ca.tld/acme/cert/1")
		w.WriteHeader(http.StatusCreated)
		w.Write(sampleCert)
	}))
	defer ts.Close()

	csr := x509.CertificateRequest{
		Version: 0,
		Subject: pkix.Name{
			CommonName:   "example.com",
			Organization: []string{"goacme"},
		},
	}
	csrb, err := x509.CreateCertificateRequest(rand.Reader, &csr, testKeyEC)
	if err != nil {
		t.Fatal(err)
	}

	c := Client{Key: testKeyEC, dir: &Directory{CertURL: ts.URL}}
	cert, certURL, err := c.CreateCert(context.Background(), csrb, notAfter.Sub(notBefore), false)
	if err != nil {
		t.Fatal(err)
	}
	if cert == nil {
		t.Errorf("cert is nil")
	}
	if certURL != "https://ca.tld/acme/cert/1" {
		t.Errorf("certURL = %q; want https://ca.tld/acme/cert/1", certURL)
	}
}

func TestFetchCert(t *testing.T) {
	var count byte
	var ts *httptest.Server
	ts = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count++
		if count < 3 {
			up := fmt.Sprintf("<%s>;rel=up", ts.URL)
			w.Header().Set("Link", up)
		}
		w.Write([]byte{count})
	}))
	defer ts.Close()
	res, err := (&Client{}).FetchCert(context.Background(), ts.URL, true)
	if err != nil {
		t.Fatalf("FetchCert: %v", err)
	}
	cert := [][]byte{{1}, {2}, {3}}
	if !reflect.DeepEqual(res, cert) {
		t.Errorf("res = %v; want %v", res, cert)
	}
}

func TestFetchCertRetry(t *testing.T) {
	var count int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if count < 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusAccepted)
			count++
			return
		}
		w.Write([]byte{1})
	}))
	defer ts.Close()
	res, err := (&Client{}).FetchCert(context.Background(), ts.URL, false)
	if err != nil {
		t.Fatalf("FetchCert: %v", err)
	}
	cert := [][]byte{{1}}
	if !reflect.DeepEqual(res, cert) {
		t.Errorf("res = %v; want %v", res, cert)
	}
}

func TestFetchCertCancel(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "0")
		w.WriteHeader(http.StatusAccepted)
	}))
	defer ts.Close()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	var err error
	go func() {
		_, err = (&Client{}).FetchCert(ctx, ts.URL, false)
		close(done)
	}()
	cancel()
	<-done
	if err != context.Canceled {
		t.Errorf("err = %v; want %v", err, context.Canceled)
	}
}

func TestFetchCertDepth(t *testing.T) {
	var count byte
	var ts *httptest.Server
	ts = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count++
		if count > maxChainLen+1 {
			t.Errorf("count = %d; want at most %d", count, maxChainLen+1)
			w.WriteHeader(http.StatusInternalServerError)
		}
		w.Header().Set("Link", fmt.Sprintf("<%s>;rel=up", ts.URL))
		w.Write([]byte{count})
	}))
	defer ts.Close()
	_, err := (&Client{}).FetchCert(context.Background(), ts.URL, true)
	if err == nil {
		t.Errorf("err is nil")
	}
}

func TestFetchCertBreadth(t *testing.T) {
	var ts *httptest.Server
	ts = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for i := 0; i < maxChainLen+1; i++ {
			w.Header().Add("Link", fmt.Sprintf("<%s>;rel=up", ts.URL))
		}
		w.Write([]byte{1})
	}))
	defer ts.Close()
	_, err := (&Client{}).FetchCert(context.Background(), ts.URL, true)
	if err == nil {
		t.Errorf("err is nil")
	}
}

func TestFetchCertSize(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b := bytes.Repeat([]byte{1}, maxCertSize+1)
		w.Write(b)
	}))
	defer ts.Close()
	_, err := (&Client{}).FetchCert(context.Background(), ts.URL, false)
	if err == nil {
		t.Errorf("err is nil")
	}
}

func TestRevokeCert(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "HEAD" {
			w.Header().Set("Replay-Nonce", "nonce")
			return
		}

		var req struct {
			Resource    string
			Certificate string
			Reason      int
		}
		decodeJWSRequest(t, &req, r)
		if req.Resource != "revoke-cert" {
			t.Errorf("req.Resource = %q; want revoke-cert", req.Resource)
		}
		if req.Reason != 1 {
			t.Errorf("req.Reason = %d; want 1", req.Reason)
		}
		// echo -n cert | base64 | tr -d '=' | tr '/+' '_-'
		cert := "Y2VydA"
		if req.Certificate != cert {
			t.Errorf("req.Certificate = %q; want %q", req.Certificate, cert)
		}
	}))
	defer ts.Close()
	client := &Client{
		Key: testKeyEC,
		dir: &Directory{RevokeURL: ts.URL},
	}
	ctx := context.Background()
	if err := client.RevokeCert(ctx, nil, []byte("cert"), CRLReasonKeyCompromise); err != nil {
		t.Fatal(err)
	}
}

func TestNonce_add(t *testing.T) {
	var c Client
	c.addNonce(http.Header{"Replay-Nonce": {"nonce"}})
	c.addNonce(http.Header{"Replay-Nonce": {}})
	c.addNonce(http.Header{"Replay-Nonce": {"nonce"}})

	nonces := map[string]struct{}{"nonce": struct{}{}}
	if !reflect.DeepEqual(c.nonces, nonces) {
		t.Errorf("c.nonces = %q; want %q", c.nonces, nonces)
	}
}

func TestNonce_addMax(t *testing.T) {
	c := &Client{nonces: make(map[string]struct{})}
	for i := 0; i < maxNonces; i++ {
		c.nonces[fmt.Sprintf("%d", i)] = struct{}{}
	}
	c.addNonce(http.Header{"Replay-Nonce": {"nonce"}})
	if n := len(c.nonces); n != maxNonces {
		t.Errorf("len(c.nonces) = %d; want %d", n, maxNonces)
	}
}

func TestNonce_fetch(t *testing.T) {
	tests := []struct {
		code  int
		nonce string
	}{
		{http.StatusOK, "nonce1"},
		{http.StatusBadRequest, "nonce2"},
		{http.StatusOK, ""},
	}
	var i int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "HEAD" {
			t.Errorf("%d: r.Method = %q; want HEAD", i, r.Method)
		}
		w.Header().Set("Replay-Nonce", tests[i].nonce)
		w.WriteHeader(tests[i].code)
	}))
	defer ts.Close()
	for ; i < len(tests); i++ {
		test := tests[i]
		c := &Client{}
		n, err := c.fetchNonce(context.Background(), ts.URL)
		if n != test.nonce {
			t.Errorf("%d: n=%q; want %q", i, n, test.nonce)
		}
		switch {
		case err == nil && test.nonce == "":
			t.Errorf("%d: n=%q, err=%v; want non-nil error", i, n, err)
		case err != nil && test.nonce != "":
			t.Errorf("%d: n=%q, err=%v; want %q", i, n, err, test.nonce)
		}
	}
}

func TestNonce_fetchError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer ts.Close()
	c := &Client{}
	_, err := c.fetchNonce(context.Background(), ts.URL)
	e, ok := err.(*Error)
	if !ok {
		t.Fatalf("err is %T; want *Error", err)
	}
	if e.StatusCode != http.StatusTooManyRequests {
		t.Errorf("e.StatusCode = %d; want %d", e.StatusCode, http.StatusTooManyRequests)
	}
}

func TestNonce_postJWS(t *testing.T) {
	var count int
	seen := make(map[string]bool)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count++
		w.Header().Set("Replay-Nonce", fmt.Sprintf("nonce%d", count))
		if r.Method == "HEAD" {
			// We expect the client do a HEAD request
			// but only to fetch the first nonce.
			return
		}
		// Make client.Authorize happy; we're not testing its result.
		defer func() {
			w.WriteHeader(http.StatusCreated)
			w.Write([]byte(`{"status":"valid"}`))
		}()

		head, err := decodeJWSHead(r)
		if err != nil {
			t.Errorf("decodeJWSHead: %v", err)
			return
		}
		if head.Nonce == "" {
			t.Error("head.Nonce is empty")
			return
		}
		if seen[head.Nonce] {
			t.Errorf("nonce is already used: %q", head.Nonce)
		}
		seen[head.Nonce] = true
	}))
	defer ts.Close()

	client := Client{Key: testKey, dir: &Directory{AuthzURL: ts.URL}}
	if _, err := client.Authorize(context.Background(), "example.com"); err != nil {
		t.Errorf("client.Authorize 1: %v", err)
	}
	// The second call should not generate another extra HEAD request.
	if _, err := client.Authorize(context.Background(), "example.com"); err != nil {
		t.Errorf("client.Authorize 2: %v", err)
	}

	if count != 3 {
		t.Errorf("total requests count: %d; want 3", count)
	}
	if n := len(client.nonces); n != 1 {
		t.Errorf("len(client.nonces) = %d; want 1", n)
	}
	for k := range seen {
		if _, exist := client.nonces[k]; exist {
			t.Errorf("used nonce %q in client.nonces", k)
		}
	}
}

func TestRetryPostJWS(t *testing.T) {
	var count int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count++
		w.Header().Set("Replay-Nonce", fmt.Sprintf("nonce%d", count))
		if r.Method == "HEAD" {
			// We expect the client to do 2 head requests to fetch
			// nonces, one to start and another after getting badNonce
			return
		}

		head, err := decodeJWSHead(r)
		if err != nil {
			t.Errorf("decodeJWSHead: %v", err)
		} else if head.Nonce == "" {
			t.Error("head.Nonce is empty")
		} else if head.Nonce == "nonce1" {
			// return a badNonce error to force the call to retry
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"type":"urn:ietf:params:acme:error:badNonce"}`))
			return
		}
		// Make client.Authorize happy; we're not testing its result.
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte(`{"status":"valid"}`))
	}))
	defer ts.Close()

	client := Client{Key: testKey, dir: &Directory{AuthzURL: ts.URL}}
	// This call will fail with badNonce, causing a retry
	if _, err := client.Authorize(context.Background(), "example.com"); err != nil {
		t.Errorf("client.Authorize 1: %v", err)
	}
	if count != 4 {
		t.Errorf("total requests count: %d; want 4", count)
	}
}

func TestLinkHeader(t *testing.T) {
	h := http.Header{"Link": {
		`<https://example.com/acme/new-authz>;rel="next"`,
		`<https://example.com/acme/recover-reg>; rel=recover`,
		`<https://example.com/acme/terms>; foo=bar; rel="terms-of-service"`,
		`<dup>;rel="next"`,
	}}
	tests := []struct {
		rel string
		out []string
	}{
		{"next", []string{"https://example.com/acme/new-authz", "dup"}},
		{"recover", []string{"https://example.com/acme/recover-reg"}},
		{"terms-of-service", []string{"https://example.com/acme/terms"}},
		{"empty", nil},
	}
	for i, test := range tests {
		if v := linkHeader(h, test.rel); !reflect.DeepEqual(v, test.out) {
			t.Errorf("%d: linkHeader(%q): %v; want %v", i, test.rel, v, test.out)
		}
	}
}

func TestErrorResponse(t *testing.T) {
	s := `{
		"status": 400,
		"type": "urn:acme:error:xxx",
		"detail": "text"
	}`
	res := &http.Response{
		StatusCode: 400,
		Status:     "400 Bad Request",
		Body:       ioutil.NopCloser(strings.NewReader(s)),
		Header:     http.Header{"X-Foo": {"bar"}},
	}
	err := responseError(res)
	v, ok := err.(*Error)
	if !ok {
		t.Fatalf("err = %+v (%T); want *Error type", err, err)
	}
	if v.StatusCode != 400 {
		t.Errorf("v.StatusCode = %v; want 400", v.StatusCode)
	}
	if v.ProblemType != "urn:acme:error:xxx" {
		t.Errorf("v.ProblemType = %q; want urn:acme:error:xxx", v.ProblemType)
	}
	if v.Detail != "text" {
		t.Errorf("v.Detail = %q; want text", v.Detail)
	}
	if !reflect.DeepEqual(v.Header, res.Header) {
		t.Errorf("v.Header = %+v; want %+v", v.Header, res.Header)
	}
}

func TestTLSSNI01ChallengeCert(t *testing.T) {
	const (
		token = "evaGxfADs6pSRb2LAv9IZf17Dt3juxGJ-PCt92wr-oA"
		// echo -n <token.testKeyECThumbprint> | shasum -a 256
		san = "dbbd5eefe7b4d06eb9d1d9f5acb4c7cd.a27d320e4b30332f0b6cb441734ad7b0.acme.invalid"
	)

	client := &Client{Key: testKeyEC}
	tlscert, name, err := client.TLSSNI01ChallengeCert(token)
	if err != nil {
		t.Fatal(err)
	}

	if n := len(tlscert.Certificate); n != 1 {
		t.Fatalf("len(tlscert.Certificate) = %d; want 1", n)
	}
	cert, err := x509.ParseCertificate(tlscert.Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	if len(cert.DNSNames) != 1 || cert.DNSNames[0] != san {
		t.Fatalf("cert.DNSNames = %v; want %q", cert.DNSNames, san)
	}
	if cert.DNSNames[0] != name {
		t.Errorf("cert.DNSNames[0] != name: %q vs %q", cert.DNSNames[0], name)
	}
}

func TestTLSSNI02ChallengeCert(t *testing.T) {
	const (
		token = "evaGxfADs6pSRb2LAv9IZf17Dt3juxGJ-PCt92wr-oA"
		// echo -n evaGxfADs6pSRb2LAv9IZf17Dt3juxGJ-PCt92wr-oA | shasum -a 256
		sanA = "7ea0aaa69214e71e02cebb18bb867736.09b730209baabf60e43d4999979ff139.token.acme.invalid"
		// echo -n <token.testKeyECThumbprint> | shasum -a 256
		sanB = "dbbd5eefe7b4d06eb9d1d9f5acb4c7cd.a27d320e4b30332f0b6cb441734ad7b0.ka.acme.invalid"
	)

	client := &Client{Key: testKeyEC}
	tlscert, name, err := client.TLSSNI02ChallengeCert(token)
	if err != nil {
		t.Fatal(err)
	}

	if n := len(tlscert.Certificate); n != 1 {
		t.Fatalf("len(tlscert.Certificate) = %d; want 1", n)
	}
	cert, err := x509.ParseCertificate(tlscert.Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	names := []string{sanA, sanB}
	if !reflect.DeepEqual(cert.DNSNames, names) {
		t.Fatalf("cert.DNSNames = %v;\nwant %v", cert.DNSNames, names)
	}
	sort.Strings(cert.DNSNames)
	i := sort.SearchStrings(cert.DNSNames, name)
	if i >= len(cert.DNSNames) || cert.DNSNames[i] != name {
		t.Errorf("%v doesn't have %q", cert.DNSNames, name)
	}
}

func TestTLSChallengeCertOpt(t *testing.T) {
	key, err := rsa.GenerateKey(rand.Reader, 512)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject:      pkix.Name{Organization: []string{"Test"}},
		DNSNames:     []string{"should-be-overwritten"},
	}
	opts := []CertOption{WithKey(key), WithTemplate(tmpl)}

	client := &Client{Key: testKeyEC}
	cert1, _, err := client.TLSSNI01ChallengeCert("token", opts...)
	if err != nil {
		t.Fatal(err)
	}
	cert2, _, err := client.TLSSNI02ChallengeCert("token", opts...)
	if err != nil {
		t.Fatal(err)
	}

	for i, tlscert := range []tls.Certificate{cert1, cert2} {
		// verify generated cert private key
		tlskey, ok := tlscert.PrivateKey.(*rsa.PrivateKey)
		if !ok {
			t.Errorf("%d: tlscert.PrivateKey is %T; want *rsa.PrivateKey", i, tlscert.PrivateKey)
			continue
		}
		if tlskey.D.Cmp(key.D) != 0 {
			t.Errorf("%d: tlskey.D = %v; want %v", i, tlskey.D, key.D)
		}
		// verify generated cert public key
		x509Cert, err := x509.ParseCertificate(tlscert.Certificate[0])
		if err != nil {
			t.Errorf("%d: %v", i, err)
			continue
		}
		tlspub, ok := x509Cert.PublicKey.(*rsa.PublicKey)
		if !ok {
			t.Errorf("%d: x509Cert.PublicKey is %T; want *rsa.PublicKey", i, x509Cert.PublicKey)
			continue
		}
		if tlspub.N.Cmp(key.N) != 0 {
			t.Errorf("%d: tlspub.N = %v; want %v", i, tlspub.N, key.N)
		}
		// verify template option
		sn := big.NewInt(2)
		if x509Cert.SerialNumber.Cmp(sn) != 0 {
			t.Errorf("%d: SerialNumber = %v; want %v", i, x509Cert.SerialNumber, sn)
		}
		org := []string{"Test"}
		if !reflect.DeepEqual(x509Cert.Subject.Organization, org) {
			t.Errorf("%d: Subject.Organization = %+v; want %+v", i, x509Cert.Subject.Organization, org)
		}
		for _, v := range x509Cert.DNSNames {
			if !strings.HasSuffix(v, ".acme.invalid") {
				t.Errorf("%d: invalid DNSNames element: %q", i, v)
			}
		}
	}
}

func TestHTTP01Challenge(t *testing.T) {
	const (
		token = "xxx"
		// thumbprint is precomputed for testKeyEC in jws_test.go
		value   = token + "." + testKeyECThumbprint
		urlpath = "/.well-known/acme-challenge/" + token
	)
	client := &Client{Key: testKeyEC}
	val, err := client.HTTP01ChallengeResponse(token)
	if err != nil {
		t.Fatal(err)
	}
	if val != value {
		t.Errorf("val = %q; want %q", val, value)
	}
	if path := client.HTTP01ChallengePath(token); path != urlpath {
		t.Errorf("path = %q; want %q", path, urlpath)
	}
}

func TestDNS01ChallengeRecord(t *testing.T) {
	// echo -n xxx.<testKeyECThumbprint> | \
	//      openssl dgst -binary -sha256 | \
	//      base64 | tr -d '=' | tr '/+' '_-'
	const value = "8DERMexQ5VcdJ_prpPiA0mVdp7imgbCgjsG4SqqNMIo"

	client := &Client{Key: testKeyEC}
	val, err := client.DNS01ChallengeRecord("xxx")
	if err != nil {
		t.Fatal(err)
	}
	if val != value {
		t.Errorf("val = %q; want %q", val, value)
	}
}

func TestBackoff(t *testing.T) {
	tt := []struct{ min, max time.Duration }{
		{time.Second, 2 * time.Second},
		{2 * time.Second, 3 * time.Second},
		{4 * time.Second, 5 * time.Second},
		{8 * time.Second, 9 * time.Second},
	}
	for i, test := range tt {
		d := backoff(i, time.Minute)
		if d < test.min || test.max < d {
			t.Errorf("%d: d = %v; want between %v and %v", i, d, test.min, test.max)
		}
	}

	min, max := time.Second, 2*time.Second
	if d := backoff(-1, time.Minute); d < min || max < d {
		t.Errorf("d = %v; want between %v and %v", d, min, max)
	}

	bound := 10 * time.Second
	if d := backoff(100, bound); d != bound {
		t.Errorf("d = %v; want %v", d, bound)
	}
}
