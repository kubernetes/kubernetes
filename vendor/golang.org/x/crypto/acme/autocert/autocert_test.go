// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package autocert

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"math/big"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"
	"time"

	"golang.org/x/crypto/acme"
)

var discoTmpl = template.Must(template.New("disco").Parse(`{
	"new-reg": "{{.}}/new-reg",
	"new-authz": "{{.}}/new-authz",
	"new-cert": "{{.}}/new-cert"
}`))

var authzTmpl = template.Must(template.New("authz").Parse(`{
	"status": "pending",
	"challenges": [
		{
			"uri": "{{.}}/challenge/1",
			"type": "tls-sni-01",
			"token": "token-01"
		},
		{
			"uri": "{{.}}/challenge/2",
			"type": "tls-sni-02",
			"token": "token-02"
		}
	]
}`))

type memCache struct {
	mu      sync.Mutex
	keyData map[string][]byte
}

func (m *memCache) Get(ctx context.Context, key string) ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	v, ok := m.keyData[key]
	if !ok {
		return nil, ErrCacheMiss
	}
	return v, nil
}

func (m *memCache) Put(ctx context.Context, key string, data []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.keyData[key] = data
	return nil
}

func (m *memCache) Delete(ctx context.Context, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.keyData, key)
	return nil
}

func newMemCache() *memCache {
	return &memCache{
		keyData: make(map[string][]byte),
	}
}

func dummyCert(pub interface{}, san ...string) ([]byte, error) {
	return dateDummyCert(pub, time.Now(), time.Now().Add(90*24*time.Hour), san...)
}

func dateDummyCert(pub interface{}, start, end time.Time, san ...string) ([]byte, error) {
	// use EC key to run faster on 386
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, err
	}
	t := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		NotBefore:             start,
		NotAfter:              end,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageKeyEncipherment,
		DNSNames:              san,
	}
	if pub == nil {
		pub = &key.PublicKey
	}
	return x509.CreateCertificate(rand.Reader, t, t, pub, key)
}

func decodePayload(v interface{}, r io.Reader) error {
	var req struct{ Payload string }
	if err := json.NewDecoder(r).Decode(&req); err != nil {
		return err
	}
	payload, err := base64.RawURLEncoding.DecodeString(req.Payload)
	if err != nil {
		return err
	}
	return json.Unmarshal(payload, v)
}

func TestGetCertificate(t *testing.T) {
	man := &Manager{Prompt: AcceptTOS}
	defer man.stopRenew()
	hello := &tls.ClientHelloInfo{ServerName: "example.org"}
	testGetCertificate(t, man, "example.org", hello)
}

func TestGetCertificate_trailingDot(t *testing.T) {
	man := &Manager{Prompt: AcceptTOS}
	defer man.stopRenew()
	hello := &tls.ClientHelloInfo{ServerName: "example.org."}
	testGetCertificate(t, man, "example.org", hello)
}

func TestGetCertificate_ForceRSA(t *testing.T) {
	man := &Manager{
		Prompt:   AcceptTOS,
		Cache:    newMemCache(),
		ForceRSA: true,
	}
	defer man.stopRenew()
	hello := &tls.ClientHelloInfo{ServerName: "example.org"}
	testGetCertificate(t, man, "example.org", hello)

	cert, err := man.cacheGet(context.Background(), "example.org")
	if err != nil {
		t.Fatalf("man.cacheGet: %v", err)
	}
	if _, ok := cert.PrivateKey.(*rsa.PrivateKey); !ok {
		t.Errorf("cert.PrivateKey is %T; want *rsa.PrivateKey", cert.PrivateKey)
	}
}

func TestGetCertificate_nilPrompt(t *testing.T) {
	man := &Manager{}
	defer man.stopRenew()
	url, finish := startACMEServerStub(t, man, "example.org")
	defer finish()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	man.Client = &acme.Client{
		Key:          key,
		DirectoryURL: url,
	}
	hello := &tls.ClientHelloInfo{ServerName: "example.org"}
	if _, err := man.GetCertificate(hello); err == nil {
		t.Error("got certificate for example.org; wanted error")
	}
}

func TestGetCertificate_expiredCache(t *testing.T) {
	// Make an expired cert and cache it.
	pk, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "example.org"},
		NotAfter:     time.Now(),
	}
	pub, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &pk.PublicKey, pk)
	if err != nil {
		t.Fatal(err)
	}
	tlscert := &tls.Certificate{
		Certificate: [][]byte{pub},
		PrivateKey:  pk,
	}

	man := &Manager{Prompt: AcceptTOS, Cache: newMemCache()}
	defer man.stopRenew()
	if err := man.cachePut(context.Background(), "example.org", tlscert); err != nil {
		t.Fatalf("man.cachePut: %v", err)
	}

	// The expired cached cert should trigger a new cert issuance
	// and return without an error.
	hello := &tls.ClientHelloInfo{ServerName: "example.org"}
	testGetCertificate(t, man, "example.org", hello)
}

func TestGetCertificate_failedAttempt(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer ts.Close()

	const example = "example.org"
	d := createCertRetryAfter
	f := testDidRemoveState
	defer func() {
		createCertRetryAfter = d
		testDidRemoveState = f
	}()
	createCertRetryAfter = 0
	done := make(chan struct{})
	testDidRemoveState = func(domain string) {
		if domain != example {
			t.Errorf("testDidRemoveState: domain = %q; want %q", domain, example)
		}
		close(done)
	}

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	man := &Manager{
		Prompt: AcceptTOS,
		Client: &acme.Client{
			Key:          key,
			DirectoryURL: ts.URL,
		},
	}
	defer man.stopRenew()
	hello := &tls.ClientHelloInfo{ServerName: example}
	if _, err := man.GetCertificate(hello); err == nil {
		t.Error("GetCertificate: err is nil")
	}
	select {
	case <-time.After(5 * time.Second):
		t.Errorf("took too long to remove the %q state", example)
	case <-done:
		man.stateMu.Lock()
		defer man.stateMu.Unlock()
		if v, exist := man.state[example]; exist {
			t.Errorf("state exists for %q: %+v", example, v)
		}
	}
}

// startACMEServerStub runs an ACME server
// The domain argument is the expected domain name of a certificate request.
func startACMEServerStub(t *testing.T, man *Manager, domain string) (url string, finish func()) {
	// echo token-02 | shasum -a 256
	// then divide result in 2 parts separated by dot
	tokenCertName := "4e8eb87631187e9ff2153b56b13a4dec.13a35d002e485d60ff37354b32f665d9.token.acme.invalid"
	verifyTokenCert := func() {
		hello := &tls.ClientHelloInfo{ServerName: tokenCertName}
		_, err := man.GetCertificate(hello)
		if err != nil {
			t.Errorf("verifyTokenCert: GetCertificate(%q): %v", tokenCertName, err)
			return
		}
	}

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
				t.Errorf("discoTmpl: %v", err)
			}
		// client key registration
		case "/new-reg":
			w.Write([]byte("{}"))
		// domain authorization
		case "/new-authz":
			w.Header().Set("Location", ca.URL+"/authz/1")
			w.WriteHeader(http.StatusCreated)
			if err := authzTmpl.Execute(w, ca.URL); err != nil {
				t.Errorf("authzTmpl: %v", err)
			}
		// accept tls-sni-02 challenge
		case "/challenge/2":
			verifyTokenCert()
			w.Write([]byte("{}"))
		// authorization status
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
				t.Errorf("new-cert: CSR: %v", err)
			}
			if csr.Subject.CommonName != domain {
				t.Errorf("CommonName in CSR = %q; want %q", csr.Subject.CommonName, domain)
			}
			der, err := dummyCert(csr.PublicKey, domain)
			if err != nil {
				t.Errorf("new-cert: dummyCert: %v", err)
			}
			chainUp := fmt.Sprintf("<%s/ca-cert>; rel=up", ca.URL)
			w.Header().Set("Link", chainUp)
			w.WriteHeader(http.StatusCreated)
			w.Write(der)
		// CA chain cert
		case "/ca-cert":
			der, err := dummyCert(nil, "ca")
			if err != nil {
				t.Errorf("ca-cert: dummyCert: %v", err)
			}
			w.Write(der)
		default:
			t.Errorf("unrecognized r.URL.Path: %s", r.URL.Path)
		}
	}))
	finish = func() {
		ca.Close()

		// make sure token cert was removed
		cancel := make(chan struct{})
		done := make(chan struct{})
		go func() {
			defer close(done)
			tick := time.NewTicker(100 * time.Millisecond)
			defer tick.Stop()
			for {
				hello := &tls.ClientHelloInfo{ServerName: tokenCertName}
				if _, err := man.GetCertificate(hello); err != nil {
					return
				}
				select {
				case <-tick.C:
				case <-cancel:
					return
				}
			}
		}()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			close(cancel)
			t.Error("token cert was not removed")
			<-done
		}
	}
	return ca.URL, finish
}

// tests man.GetCertificate flow using the provided hello argument.
// The domain argument is the expected domain name of a certificate request.
func testGetCertificate(t *testing.T, man *Manager, domain string, hello *tls.ClientHelloInfo) {
	url, finish := startACMEServerStub(t, man, domain)
	defer finish()

	// use EC key to run faster on 386
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	man.Client = &acme.Client{
		Key:          key,
		DirectoryURL: url,
	}

	// simulate tls.Config.GetCertificate
	var tlscert *tls.Certificate
	done := make(chan struct{})
	go func() {
		tlscert, err = man.GetCertificate(hello)
		close(done)
	}()
	select {
	case <-time.After(time.Minute):
		t.Fatal("man.GetCertificate took too long to return")
	case <-done:
	}
	if err != nil {
		t.Fatalf("man.GetCertificate: %v", err)
	}

	// verify the tlscert is the same we responded with from the CA stub
	if len(tlscert.Certificate) == 0 {
		t.Fatal("len(tlscert.Certificate) is 0")
	}
	cert, err := x509.ParseCertificate(tlscert.Certificate[0])
	if err != nil {
		t.Fatalf("x509.ParseCertificate: %v", err)
	}
	if len(cert.DNSNames) == 0 || cert.DNSNames[0] != domain {
		t.Errorf("cert.DNSNames = %v; want %q", cert.DNSNames, domain)
	}

}

func TestAccountKeyCache(t *testing.T) {
	m := Manager{Cache: newMemCache()}
	ctx := context.Background()
	k1, err := m.accountKey(ctx)
	if err != nil {
		t.Fatal(err)
	}
	k2, err := m.accountKey(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(k1, k2) {
		t.Errorf("account keys don't match: k1 = %#v; k2 = %#v", k1, k2)
	}
}

func TestCache(t *testing.T) {
	privKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "example.org"},
		NotAfter:     time.Now().Add(time.Hour),
	}
	pub, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &privKey.PublicKey, privKey)
	if err != nil {
		t.Fatal(err)
	}
	tlscert := &tls.Certificate{
		Certificate: [][]byte{pub},
		PrivateKey:  privKey,
	}

	man := &Manager{Cache: newMemCache()}
	defer man.stopRenew()
	ctx := context.Background()
	if err := man.cachePut(ctx, "example.org", tlscert); err != nil {
		t.Fatalf("man.cachePut: %v", err)
	}
	res, err := man.cacheGet(ctx, "example.org")
	if err != nil {
		t.Fatalf("man.cacheGet: %v", err)
	}
	if res == nil {
		t.Fatal("res is nil")
	}
}

func TestHostWhitelist(t *testing.T) {
	policy := HostWhitelist("example.com", "example.org", "*.example.net")
	tt := []struct {
		host  string
		allow bool
	}{
		{"example.com", true},
		{"example.org", true},
		{"one.example.com", false},
		{"two.example.org", false},
		{"three.example.net", false},
		{"dummy", false},
	}
	for i, test := range tt {
		err := policy(nil, test.host)
		if err != nil && test.allow {
			t.Errorf("%d: policy(%q): %v; want nil", i, test.host, err)
		}
		if err == nil && !test.allow {
			t.Errorf("%d: policy(%q): nil; want an error", i, test.host)
		}
	}
}

func TestValidCert(t *testing.T) {
	key1, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	key2, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	key3, err := rsa.GenerateKey(rand.Reader, 512)
	if err != nil {
		t.Fatal(err)
	}
	cert1, err := dummyCert(key1.Public(), "example.org")
	if err != nil {
		t.Fatal(err)
	}
	cert2, err := dummyCert(key2.Public(), "example.org")
	if err != nil {
		t.Fatal(err)
	}
	cert3, err := dummyCert(key3.Public(), "example.org")
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	early, err := dateDummyCert(key1.Public(), now.Add(time.Hour), now.Add(2*time.Hour), "example.org")
	if err != nil {
		t.Fatal(err)
	}
	expired, err := dateDummyCert(key1.Public(), now.Add(-2*time.Hour), now.Add(-time.Hour), "example.org")
	if err != nil {
		t.Fatal(err)
	}

	tt := []struct {
		domain string
		key    crypto.Signer
		cert   [][]byte
		ok     bool
	}{
		{"example.org", key1, [][]byte{cert1}, true},
		{"example.org", key3, [][]byte{cert3}, true},
		{"example.org", key1, [][]byte{cert1, cert2, cert3}, true},
		{"example.org", key1, [][]byte{cert1, {1}}, false},
		{"example.org", key1, [][]byte{{1}}, false},
		{"example.org", key1, [][]byte{cert2}, false},
		{"example.org", key2, [][]byte{cert1}, false},
		{"example.org", key1, [][]byte{cert3}, false},
		{"example.org", key3, [][]byte{cert1}, false},
		{"example.net", key1, [][]byte{cert1}, false},
		{"example.org", key1, [][]byte{early}, false},
		{"example.org", key1, [][]byte{expired}, false},
	}
	for i, test := range tt {
		leaf, err := validCert(test.domain, test.cert, test.key)
		if err != nil && test.ok {
			t.Errorf("%d: err = %v", i, err)
		}
		if err == nil && !test.ok {
			t.Errorf("%d: err is nil", i)
		}
		if err == nil && test.ok && leaf == nil {
			t.Errorf("%d: leaf is nil", i)
		}
	}
}

type cacheGetFunc func(ctx context.Context, key string) ([]byte, error)

func (f cacheGetFunc) Get(ctx context.Context, key string) ([]byte, error) {
	return f(ctx, key)
}

func (f cacheGetFunc) Put(ctx context.Context, key string, data []byte) error {
	return fmt.Errorf("unsupported Put of %q = %q", key, data)
}

func (f cacheGetFunc) Delete(ctx context.Context, key string) error {
	return fmt.Errorf("unsupported Delete of %q", key)
}

func TestManagerGetCertificateBogusSNI(t *testing.T) {
	m := Manager{
		Prompt: AcceptTOS,
		Cache: cacheGetFunc(func(ctx context.Context, key string) ([]byte, error) {
			return nil, fmt.Errorf("cache.Get of %s", key)
		}),
	}
	tests := []struct {
		name    string
		wantErr string
	}{
		{"foo.com", "cache.Get of foo.com"},
		{"foo.com.", "cache.Get of foo.com"},
		{`a\b.com`, "acme/autocert: server name contains invalid character"},
		{`a/b.com`, "acme/autocert: server name contains invalid character"},
		{"", "acme/autocert: missing server name"},
		{"foo", "acme/autocert: server name component count invalid"},
		{".foo", "acme/autocert: server name component count invalid"},
		{"foo.", "acme/autocert: server name component count invalid"},
		{"fo.o", "cache.Get of fo.o"},
	}
	for _, tt := range tests {
		_, err := m.GetCertificate(&tls.ClientHelloInfo{ServerName: tt.name})
		got := fmt.Sprint(err)
		if got != tt.wantErr {
			t.Errorf("GetCertificate(SNI = %q) = %q; want %q", tt.name, got, tt.wantErr)
		}
	}
}
