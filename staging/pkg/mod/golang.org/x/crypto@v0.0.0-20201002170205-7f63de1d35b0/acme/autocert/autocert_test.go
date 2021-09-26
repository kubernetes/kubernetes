// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package autocert

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"io/ioutil"
	"math/big"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/crypto/acme"
	"golang.org/x/crypto/acme/autocert/internal/acmetest"
)

var (
	exampleDomain     = "example.org"
	exampleCertKey    = certKey{domain: exampleDomain}
	exampleCertKeyRSA = certKey{domain: exampleDomain, isRSA: true}
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
			"uri": "{{.}}/challenge/tls-alpn-01",
			"type": "tls-alpn-01",
			"token": "token-alpn"
		},
		{
			"uri": "{{.}}/challenge/dns-01",
			"type": "dns-01",
			"token": "token-dns-01"
		},
		{
			"uri": "{{.}}/challenge/http-01",
			"type": "http-01",
			"token": "token-http-01"
		}
	]
}`))

type memCache struct {
	t       *testing.T
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

// filenameSafe returns whether all characters in s are printable ASCII
// and safe to use in a filename on most filesystems.
func filenameSafe(s string) bool {
	for _, c := range s {
		if c < 0x20 || c > 0x7E {
			return false
		}
		switch c {
		case '\\', '/', ':', '*', '?', '"', '<', '>', '|':
			return false
		}
	}
	return true
}

func (m *memCache) Put(ctx context.Context, key string, data []byte) error {
	if !filenameSafe(key) {
		m.t.Errorf("invalid characters in cache key %q", key)
	}

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

func newMemCache(t *testing.T) *memCache {
	return &memCache{
		t:       t,
		keyData: make(map[string][]byte),
	}
}

func (m *memCache) numCerts() int {
	m.mu.Lock()
	defer m.mu.Unlock()

	res := 0
	for key := range m.keyData {
		if strings.HasSuffix(key, "+token") ||
			strings.HasSuffix(key, "+key") ||
			strings.HasSuffix(key, "+http-01") {
			continue
		}
		res++
	}
	return res
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

type algorithmSupport int

const (
	algRSA algorithmSupport = iota
	algECDSA
)

func clientHelloInfo(sni string, alg algorithmSupport) *tls.ClientHelloInfo {
	hello := &tls.ClientHelloInfo{
		ServerName:   sni,
		CipherSuites: []uint16{tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305},
	}
	if alg == algECDSA {
		hello.CipherSuites = append(hello.CipherSuites, tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305)
	}
	return hello
}

// tokenCertFn returns a function suitable for startACMEServerStub.
// The returned function simulates a TLS hello request from a CA
// during validation of a tls-alpn-01 challenge.
func tokenCertFn(man *Manager, alg algorithmSupport) getCertificateFunc {
	return func(sni string) (*tls.Certificate, error) {
		hello := clientHelloInfo(sni, alg)
		hello.SupportedProtos = []string{acme.ALPNProto}
		return man.GetCertificate(hello)
	}
}

func TestGetCertificate(t *testing.T) {
	man := &Manager{Prompt: AcceptTOS}
	defer man.stopRenew()
	hello := clientHelloInfo("example.org", algECDSA)
	testGetCertificate(t, man, "example.org", hello)
}

func TestGetCertificate_trailingDot(t *testing.T) {
	man := &Manager{Prompt: AcceptTOS}
	defer man.stopRenew()
	hello := clientHelloInfo("example.org.", algECDSA)
	testGetCertificate(t, man, "example.org", hello)
}

func TestGetCertificate_unicodeIDN(t *testing.T) {
	man := &Manager{Prompt: AcceptTOS}
	defer man.stopRenew()

	hello := clientHelloInfo("σσσ.com", algECDSA)
	testGetCertificate(t, man, "xn--4xaaa.com", hello)

	hello = clientHelloInfo("σςΣ.com", algECDSA)
	testGetCertificate(t, man, "xn--4xaaa.com", hello)
}

func TestGetCertificate_mixedcase(t *testing.T) {
	man := &Manager{Prompt: AcceptTOS}
	defer man.stopRenew()

	hello := clientHelloInfo("example.org", algECDSA)
	testGetCertificate(t, man, "example.org", hello)

	hello = clientHelloInfo("EXAMPLE.ORG", algECDSA)
	testGetCertificate(t, man, "example.org", hello)
}

func TestGetCertificate_ForceRSA(t *testing.T) {
	man := &Manager{
		Prompt:   AcceptTOS,
		Cache:    newMemCache(t),
		ForceRSA: true,
	}
	defer man.stopRenew()
	hello := clientHelloInfo(exampleDomain, algECDSA)
	testGetCertificate(t, man, exampleDomain, hello)

	// ForceRSA was deprecated and is now ignored.
	cert, err := man.cacheGet(context.Background(), exampleCertKey)
	if err != nil {
		t.Fatalf("man.cacheGet: %v", err)
	}
	if _, ok := cert.PrivateKey.(*ecdsa.PrivateKey); !ok {
		t.Errorf("cert.PrivateKey is %T; want *ecdsa.PrivateKey", cert.PrivateKey)
	}
}

func TestGetCertificate_nilPrompt(t *testing.T) {
	man := &Manager{}
	defer man.stopRenew()
	url, finish := startACMEServerStub(t, tokenCertFn(man, algECDSA), "example.org")
	defer finish()
	man.Client = &acme.Client{DirectoryURL: url}
	hello := clientHelloInfo("example.org", algECDSA)
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
		Subject:      pkix.Name{CommonName: exampleDomain},
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

	man := &Manager{Prompt: AcceptTOS, Cache: newMemCache(t)}
	defer man.stopRenew()
	if err := man.cachePut(context.Background(), exampleCertKey, tlscert); err != nil {
		t.Fatalf("man.cachePut: %v", err)
	}

	// The expired cached cert should trigger a new cert issuance
	// and return without an error.
	hello := clientHelloInfo(exampleDomain, algECDSA)
	testGetCertificate(t, man, exampleDomain, hello)
}

func TestGetCertificate_failedAttempt(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer ts.Close()

	d := createCertRetryAfter
	f := testDidRemoveState
	defer func() {
		createCertRetryAfter = d
		testDidRemoveState = f
	}()
	createCertRetryAfter = 0
	done := make(chan struct{})
	testDidRemoveState = func(ck certKey) {
		if ck != exampleCertKey {
			t.Errorf("testDidRemoveState: domain = %v; want %v", ck, exampleCertKey)
		}
		close(done)
	}

	man := &Manager{
		Prompt: AcceptTOS,
		Client: &acme.Client{
			DirectoryURL: ts.URL,
		},
	}
	defer man.stopRenew()
	hello := clientHelloInfo(exampleDomain, algECDSA)
	if _, err := man.GetCertificate(hello); err == nil {
		t.Error("GetCertificate: err is nil")
	}
	select {
	case <-time.After(5 * time.Second):
		t.Errorf("took too long to remove the %q state", exampleCertKey)
	case <-done:
		man.stateMu.Lock()
		defer man.stateMu.Unlock()
		if v, exist := man.state[exampleCertKey]; exist {
			t.Errorf("state exists for %v: %+v", exampleCertKey, v)
		}
	}
}

// testGetCertificate_tokenCache tests the fallback of token certificate fetches
// to cache when Manager.certTokens misses.
// algorithmSupport refers to the CA when verifying the certificate token.
func testGetCertificate_tokenCache(t *testing.T, tokenAlg algorithmSupport) {
	man1 := &Manager{
		Cache:  newMemCache(t),
		Prompt: AcceptTOS,
	}
	defer man1.stopRenew()
	man2 := &Manager{
		Cache:  man1.Cache,
		Prompt: AcceptTOS,
	}
	defer man2.stopRenew()

	// Send the verification request to a different Manager from the one that
	// initiated the authorization, when they share caches.
	url, finish := startACMEServerStub(t, tokenCertFn(man2, tokenAlg), "example.org")
	defer finish()
	man1.Client = &acme.Client{DirectoryURL: url}
	man2.Client = &acme.Client{DirectoryURL: url}
	hello := clientHelloInfo("example.org", algECDSA)
	if _, err := man1.GetCertificate(hello); err != nil {
		t.Error(err)
	}
	if _, err := man2.GetCertificate(hello); err != nil {
		t.Error(err)
	}
}

func TestGetCertificate_tokenCache(t *testing.T) {
	t.Run("ecdsaSupport=true", func(t *testing.T) {
		testGetCertificate_tokenCache(t, algECDSA)
	})
	t.Run("ecdsaSupport=false", func(t *testing.T) {
		testGetCertificate_tokenCache(t, algRSA)
	})
}

func TestGetCertificate_ecdsaVsRSA(t *testing.T) {
	cache := newMemCache(t)
	man := &Manager{Prompt: AcceptTOS, Cache: cache}
	defer man.stopRenew()
	url, finish := startACMEServerStub(t, tokenCertFn(man, algECDSA), "example.org")
	defer finish()
	man.Client = &acme.Client{DirectoryURL: url}

	cert, err := man.GetCertificate(clientHelloInfo("example.org", algECDSA))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := cert.Leaf.PublicKey.(*ecdsa.PublicKey); !ok {
		t.Error("an ECDSA client was served a non-ECDSA certificate")
	}

	cert, err = man.GetCertificate(clientHelloInfo("example.org", algRSA))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := cert.Leaf.PublicKey.(*rsa.PublicKey); !ok {
		t.Error("a RSA client was served a non-RSA certificate")
	}

	if _, err := man.GetCertificate(clientHelloInfo("example.org", algECDSA)); err != nil {
		t.Error(err)
	}
	if _, err := man.GetCertificate(clientHelloInfo("example.org", algRSA)); err != nil {
		t.Error(err)
	}
	if numCerts := cache.numCerts(); numCerts != 2 {
		t.Errorf("found %d certificates in cache; want %d", numCerts, 2)
	}
}

func TestGetCertificate_wrongCacheKeyType(t *testing.T) {
	cache := newMemCache(t)
	man := &Manager{Prompt: AcceptTOS, Cache: cache}
	defer man.stopRenew()
	url, finish := startACMEServerStub(t, tokenCertFn(man, algECDSA), exampleDomain)
	defer finish()
	man.Client = &acme.Client{DirectoryURL: url}

	// Make an RSA cert and cache it without suffix.
	pk, err := rsa.GenerateKey(rand.Reader, 512)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: exampleDomain},
		NotAfter:     time.Now().Add(90 * 24 * time.Hour),
	}
	pub, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &pk.PublicKey, pk)
	if err != nil {
		t.Fatal(err)
	}
	rsaCert := &tls.Certificate{
		Certificate: [][]byte{pub},
		PrivateKey:  pk,
	}
	if err := man.cachePut(context.Background(), exampleCertKey, rsaCert); err != nil {
		t.Fatalf("man.cachePut: %v", err)
	}

	// The RSA cached cert should be silently ignored and replaced.
	cert, err := man.GetCertificate(clientHelloInfo(exampleDomain, algECDSA))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := cert.Leaf.PublicKey.(*ecdsa.PublicKey); !ok {
		t.Error("an ECDSA client was served a non-ECDSA certificate")
	}
	if numCerts := cache.numCerts(); numCerts != 1 {
		t.Errorf("found %d certificates in cache; want %d", numCerts, 1)
	}
}

type getCertificateFunc func(domain string) (*tls.Certificate, error)

// startACMEServerStub runs an ACME server
// The domain argument is the expected domain name of a certificate request.
// TODO: Drop this in favour of x/crypto/acme/autocert/internal/acmetest.
func startACMEServerStub(t *testing.T, tokenCert getCertificateFunc, domain string) (url string, finish func()) {
	verifyTokenCert := func() {
		tlscert, err := tokenCert(domain)
		if err != nil {
			t.Errorf("verifyTokenCert: tokenCert(%q): %v", domain, err)
			return
		}
		crt, err := x509.ParseCertificate(tlscert.Certificate[0])
		if err != nil {
			t.Errorf("verifyTokenCert: x509.ParseCertificate: %v", err)
		}
		if err := crt.VerifyHostname(domain); err != nil {
			t.Errorf("verifyTokenCert: %v", err)
		}
		// See https://tools.ietf.org/html/draft-ietf-acme-tls-alpn-05#section-5.1
		oid := asn1.ObjectIdentifier{1, 3, 6, 1, 5, 5, 7, 1, 31}
		for _, x := range crt.Extensions {
			if x.Id.Equal(oid) {
				// No need to check the extension value here.
				// This is done in acme package tests.
				return
			}
		}
		t.Error("verifyTokenCert: no id-pe-acmeIdentifier extension found")
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
		// accept tls-alpn-01 challenge
		case "/challenge/tls-alpn-01":
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
				if _, err := tokenCert(domain); err != nil {
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
	url, finish := startACMEServerStub(t, tokenCertFn(man, algECDSA), domain)
	defer finish()
	man.Client = &acme.Client{DirectoryURL: url}

	// simulate tls.Config.GetCertificate
	var tlscert *tls.Certificate
	var err error
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

func TestVerifyHTTP01(t *testing.T) {
	var (
		http01 http.Handler

		authzCount      int // num. of created authorizations
		didAcceptHTTP01 bool
	)

	verifyHTTPToken := func() {
		r := httptest.NewRequest("GET", "/.well-known/acme-challenge/token-http-01", nil)
		w := httptest.NewRecorder()
		http01.ServeHTTP(w, r)
		if w.Code != http.StatusOK {
			t.Errorf("http token: w.Code = %d; want %d", w.Code, http.StatusOK)
		}
		if v := w.Body.String(); !strings.HasPrefix(v, "token-http-01.") {
			t.Errorf("http token value = %q; want 'token-http-01.' prefix", v)
		}
	}

	// ACME CA server stub, only the needed bits.
	// TODO: Replace this with x/crypto/acme/autocert/internal/acmetest.
	var ca *httptest.Server
	ca = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Replay-Nonce", "nonce")
		if r.Method == "HEAD" {
			// a nonce request
			return
		}

		switch r.URL.Path {
		// Discovery.
		case "/":
			if err := discoTmpl.Execute(w, ca.URL); err != nil {
				t.Errorf("discoTmpl: %v", err)
			}
		// Client key registration.
		case "/new-reg":
			w.Write([]byte("{}"))
		// New domain authorization.
		case "/new-authz":
			authzCount++
			w.Header().Set("Location", fmt.Sprintf("%s/authz/%d", ca.URL, authzCount))
			w.WriteHeader(http.StatusCreated)
			if err := authzTmpl.Execute(w, ca.URL); err != nil {
				t.Errorf("authzTmpl: %v", err)
			}
		// Reject tls-alpn-01.
		case "/challenge/tls-alpn-01":
			http.Error(w, "won't accept tls-sni-01", http.StatusBadRequest)
		// Should not accept dns-01.
		case "/challenge/dns-01":
			t.Errorf("dns-01 challenge was accepted")
			http.Error(w, "won't accept dns-01", http.StatusBadRequest)
		// Accept http-01.
		case "/challenge/http-01":
			didAcceptHTTP01 = true
			verifyHTTPToken()
			w.Write([]byte("{}"))
		// Authorization statuses.
		case "/authz/1": // tls-alpn-01
			w.Write([]byte(`{"status": "invalid"}`))
		case "/authz/2": // http-01
			w.Write([]byte(`{"status": "valid"}`))
		default:
			http.NotFound(w, r)
			t.Errorf("unrecognized r.URL.Path: %s", r.URL.Path)
		}
	}))
	defer ca.Close()

	m := &Manager{
		Client: &acme.Client{
			DirectoryURL: ca.URL,
		},
	}
	http01 = m.HTTPHandler(nil)
	ctx := context.Background()
	client, err := m.acmeClient(ctx)
	if err != nil {
		t.Fatalf("m.acmeClient: %v", err)
	}
	if err := m.verify(ctx, client, "example.org"); err != nil {
		t.Errorf("m.verify: %v", err)
	}
	// Only tls-alpn-01 and http-01 must be accepted.
	// The dns-01 challenge is unsupported.
	if authzCount != 2 {
		t.Errorf("authzCount = %d; want 2", authzCount)
	}
	if !didAcceptHTTP01 {
		t.Error("did not accept http-01 challenge")
	}
}

func TestRevokeFailedAuthz(t *testing.T) {
	// Prefill authorization URIs expected to be revoked.
	// The challenges are selected in a specific order,
	// each tried within a newly created authorization.
	// This means each authorization URI corresponds to a different challenge type.
	revokedAuthz := map[string]bool{
		"/authz/0": false, // tls-alpn-01
		"/authz/1": false, // http-01
		"/authz/2": false, // no viable challenge, but authz is created
	}

	var authzCount int          // num. of created authorizations
	var revokeCount int         // num. of revoked authorizations
	done := make(chan struct{}) // closed when revokeCount is 3

	// ACME CA server stub, only the needed bits.
	// TODO: Replace this with x/crypto/acme/autocert/internal/acmetest.
	var ca *httptest.Server
	ca = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Replay-Nonce", "nonce")
		if r.Method == "HEAD" {
			// a nonce request
			return
		}

		switch r.URL.Path {
		// Discovery.
		case "/":
			if err := discoTmpl.Execute(w, ca.URL); err != nil {
				t.Errorf("discoTmpl: %v", err)
			}
		// Client key registration.
		case "/new-reg":
			w.Write([]byte("{}"))
		// New domain authorization.
		case "/new-authz":
			w.Header().Set("Location", fmt.Sprintf("%s/authz/%d", ca.URL, authzCount))
			w.WriteHeader(http.StatusCreated)
			if err := authzTmpl.Execute(w, ca.URL); err != nil {
				t.Errorf("authzTmpl: %v", err)
			}
			authzCount++
		// tls-alpn-01 challenge "accept" request.
		case "/challenge/tls-alpn-01":
			// Refuse.
			http.Error(w, "won't accept tls-alpn-01 challenge", http.StatusBadRequest)
		// http-01 challenge "accept" request.
		case "/challenge/http-01":
			// Refuse.
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"status":"invalid"}`))
		// Authorization requests.
		case "/authz/0", "/authz/1", "/authz/2":
			// Revocation requests.
			if r.Method == "POST" {
				var req struct{ Status string }
				if err := decodePayload(&req, r.Body); err != nil {
					t.Errorf("%s: decodePayload: %v", r.URL, err)
				}
				switch req.Status {
				case "deactivated":
					revokedAuthz[r.URL.Path] = true
					revokeCount++
					if revokeCount >= 3 {
						// Last authorization is revoked.
						defer close(done)
					}
				default:
					t.Errorf("%s: req.Status = %q; want 'deactivated'", r.URL, req.Status)
				}
				w.Write([]byte(`{"status": "invalid"}`))
				return
			}
			// Authorization status requests.
			w.Write([]byte(`{"status":"pending"}`))
		default:
			http.NotFound(w, r)
			t.Errorf("unrecognized r.URL.Path: %s", r.URL.Path)
		}
	}))
	defer ca.Close()

	m := &Manager{
		Client: &acme.Client{DirectoryURL: ca.URL},
	}
	m.HTTPHandler(nil) // enable http-01 challenge type
	// Should fail and revoke 3 authorizations.
	// The first 2 are tls-alpn-01 and http-01 challenges.
	// The third time an authorization is created but no viable challenge is found.
	// See revokedAuthz above for more explanation.
	if _, err := m.createCert(context.Background(), exampleCertKey); err == nil {
		t.Errorf("m.createCert returned nil error")
	}
	select {
	case <-time.After(3 * time.Second):
		t.Error("revocations took too long")
	case <-done:
		// revokeCount is at least 3.
	}
	for uri, ok := range revokedAuthz {
		if !ok {
			t.Errorf("%q authorization was not revoked", uri)
		}
	}
}

func TestHTTPHandlerDefaultFallback(t *testing.T) {
	tt := []struct {
		method, url  string
		wantCode     int
		wantLocation string
	}{
		{"GET", "http://example.org", 302, "https://example.org/"},
		{"GET", "http://example.org/foo", 302, "https://example.org/foo"},
		{"GET", "http://example.org/foo/bar/", 302, "https://example.org/foo/bar/"},
		{"GET", "http://example.org/?a=b", 302, "https://example.org/?a=b"},
		{"GET", "http://example.org/foo?a=b", 302, "https://example.org/foo?a=b"},
		{"GET", "http://example.org:80/foo?a=b", 302, "https://example.org:443/foo?a=b"},
		{"GET", "http://example.org:80/foo%20bar", 302, "https://example.org:443/foo%20bar"},
		{"GET", "http://[2602:d1:xxxx::c60a]:1234", 302, "https://[2602:d1:xxxx::c60a]:443/"},
		{"GET", "http://[2602:d1:xxxx::c60a]", 302, "https://[2602:d1:xxxx::c60a]/"},
		{"GET", "http://[2602:d1:xxxx::c60a]/foo?a=b", 302, "https://[2602:d1:xxxx::c60a]/foo?a=b"},
		{"HEAD", "http://example.org", 302, "https://example.org/"},
		{"HEAD", "http://example.org/foo", 302, "https://example.org/foo"},
		{"HEAD", "http://example.org/foo/bar/", 302, "https://example.org/foo/bar/"},
		{"HEAD", "http://example.org/?a=b", 302, "https://example.org/?a=b"},
		{"HEAD", "http://example.org/foo?a=b", 302, "https://example.org/foo?a=b"},
		{"POST", "http://example.org", 400, ""},
		{"PUT", "http://example.org", 400, ""},
		{"GET", "http://example.org/.well-known/acme-challenge/x", 404, ""},
	}
	var m Manager
	h := m.HTTPHandler(nil)
	for i, test := range tt {
		r := httptest.NewRequest(test.method, test.url, nil)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != test.wantCode {
			t.Errorf("%d: w.Code = %d; want %d", i, w.Code, test.wantCode)
			t.Errorf("%d: body: %s", i, w.Body.Bytes())
		}
		if v := w.Header().Get("Location"); v != test.wantLocation {
			t.Errorf("%d: Location = %q; want %q", i, v, test.wantLocation)
		}
	}
}

func TestAccountKeyCache(t *testing.T) {
	m := Manager{Cache: newMemCache(t)}
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
	ecdsaKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	cert, err := dummyCert(ecdsaKey.Public(), exampleDomain)
	if err != nil {
		t.Fatal(err)
	}
	ecdsaCert := &tls.Certificate{
		Certificate: [][]byte{cert},
		PrivateKey:  ecdsaKey,
	}

	rsaKey, err := rsa.GenerateKey(rand.Reader, 512)
	if err != nil {
		t.Fatal(err)
	}
	cert, err = dummyCert(rsaKey.Public(), exampleDomain)
	if err != nil {
		t.Fatal(err)
	}
	rsaCert := &tls.Certificate{
		Certificate: [][]byte{cert},
		PrivateKey:  rsaKey,
	}

	man := &Manager{Cache: newMemCache(t)}
	defer man.stopRenew()
	ctx := context.Background()

	if err := man.cachePut(ctx, exampleCertKey, ecdsaCert); err != nil {
		t.Fatalf("man.cachePut: %v", err)
	}
	if err := man.cachePut(ctx, exampleCertKeyRSA, rsaCert); err != nil {
		t.Fatalf("man.cachePut: %v", err)
	}

	res, err := man.cacheGet(ctx, exampleCertKey)
	if err != nil {
		t.Fatalf("man.cacheGet: %v", err)
	}
	if res == nil || !bytes.Equal(res.Certificate[0], ecdsaCert.Certificate[0]) {
		t.Errorf("man.cacheGet = %+v; want %+v", res, ecdsaCert)
	}

	res, err = man.cacheGet(ctx, exampleCertKeyRSA)
	if err != nil {
		t.Fatalf("man.cacheGet: %v", err)
	}
	if res == nil || !bytes.Equal(res.Certificate[0], rsaCert.Certificate[0]) {
		t.Errorf("man.cacheGet = %+v; want %+v", res, rsaCert)
	}
}

func TestHostWhitelist(t *testing.T) {
	policy := HostWhitelist("example.com", "EXAMPLE.ORG", "*.example.net", "σςΣ.com")
	tt := []struct {
		host  string
		allow bool
	}{
		{"example.com", true},
		{"example.org", true},
		{"xn--4xaaa.com", true},
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
		ck   certKey
		key  crypto.Signer
		cert [][]byte
		ok   bool
	}{
		{certKey{domain: "example.org"}, key1, [][]byte{cert1}, true},
		{certKey{domain: "example.org", isRSA: true}, key3, [][]byte{cert3}, true},
		{certKey{domain: "example.org"}, key1, [][]byte{cert1, cert2, cert3}, true},
		{certKey{domain: "example.org"}, key1, [][]byte{cert1, {1}}, false},
		{certKey{domain: "example.org"}, key1, [][]byte{{1}}, false},
		{certKey{domain: "example.org"}, key1, [][]byte{cert2}, false},
		{certKey{domain: "example.org"}, key2, [][]byte{cert1}, false},
		{certKey{domain: "example.org"}, key1, [][]byte{cert3}, false},
		{certKey{domain: "example.org"}, key3, [][]byte{cert1}, false},
		{certKey{domain: "example.net"}, key1, [][]byte{cert1}, false},
		{certKey{domain: "example.org"}, key1, [][]byte{early}, false},
		{certKey{domain: "example.org"}, key1, [][]byte{expired}, false},
		{certKey{domain: "example.org", isRSA: true}, key1, [][]byte{cert1}, false},
		{certKey{domain: "example.org"}, key3, [][]byte{cert3}, false},
	}
	for i, test := range tt {
		leaf, err := validCert(test.ck, test.cert, test.key, now)
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
		_, err := m.GetCertificate(clientHelloInfo(tt.name, algECDSA))
		got := fmt.Sprint(err)
		if got != tt.wantErr {
			t.Errorf("GetCertificate(SNI = %q) = %q; want %q", tt.name, got, tt.wantErr)
		}
	}
}

func TestCertRequest(t *testing.T) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	// An extension from RFC7633. Any will do.
	ext := pkix.Extension{
		Id:    asn1.ObjectIdentifier{1, 3, 6, 1, 5, 5, 7, 1},
		Value: []byte("dummy"),
	}
	b, err := certRequest(key, "example.org", []pkix.Extension{ext}, "san.example.org")
	if err != nil {
		t.Fatalf("certRequest: %v", err)
	}
	r, err := x509.ParseCertificateRequest(b)
	if err != nil {
		t.Fatalf("ParseCertificateRequest: %v", err)
	}
	var found bool
	for _, v := range r.Extensions {
		if v.Id.Equal(ext.Id) {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("want %v in Extensions: %v", ext, r.Extensions)
	}
}

func TestSupportsECDSA(t *testing.T) {
	tests := []struct {
		CipherSuites     []uint16
		SignatureSchemes []tls.SignatureScheme
		SupportedCurves  []tls.CurveID
		ecdsaOk          bool
	}{
		{[]uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		}, nil, nil, false},
		{[]uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
		}, nil, nil, true},

		// SignatureSchemes limits, not extends, CipherSuites
		{[]uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		}, []tls.SignatureScheme{
			tls.PKCS1WithSHA256, tls.ECDSAWithP256AndSHA256,
		}, nil, false},
		{[]uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
		}, []tls.SignatureScheme{
			tls.PKCS1WithSHA256,
		}, nil, false},
		{[]uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
		}, []tls.SignatureScheme{
			tls.PKCS1WithSHA256, tls.ECDSAWithP256AndSHA256,
		}, nil, true},

		{[]uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
		}, []tls.SignatureScheme{
			tls.PKCS1WithSHA256, tls.ECDSAWithP256AndSHA256,
		}, []tls.CurveID{
			tls.CurveP521,
		}, false},
		{[]uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
		}, []tls.SignatureScheme{
			tls.PKCS1WithSHA256, tls.ECDSAWithP256AndSHA256,
		}, []tls.CurveID{
			tls.CurveP256,
			tls.CurveP521,
		}, true},
	}
	for i, tt := range tests {
		result := supportsECDSA(&tls.ClientHelloInfo{
			CipherSuites:     tt.CipherSuites,
			SignatureSchemes: tt.SignatureSchemes,
			SupportedCurves:  tt.SupportedCurves,
		})
		if result != tt.ecdsaOk {
			t.Errorf("%d: supportsECDSA = %v; want %v", i, result, tt.ecdsaOk)
		}
	}
}

// TODO: add same end-to-end for http-01 challenge type.
func TestEndToEnd(t *testing.T) {
	const domain = "example.org"

	// ACME CA server
	ca := acmetest.NewCAServer([]string{"tls-alpn-01"}, []string{domain})
	defer ca.Close()

	// User dummy server.
	m := &Manager{
		Prompt: AcceptTOS,
		Client: &acme.Client{DirectoryURL: ca.URL},
	}
	us := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("OK"))
	}))
	us.TLS = &tls.Config{
		NextProtos: []string{"http/1.1", acme.ALPNProto},
		GetCertificate: func(hello *tls.ClientHelloInfo) (*tls.Certificate, error) {
			cert, err := m.GetCertificate(hello)
			if err != nil {
				t.Errorf("m.GetCertificate: %v", err)
			}
			return cert, err
		},
	}
	us.StartTLS()
	defer us.Close()
	// In TLS-ALPN challenge verification, CA connects to the domain:443 in question.
	// Because the domain won't resolve in tests, we need to tell the CA
	// where to dial to instead.
	ca.Resolve(domain, strings.TrimPrefix(us.URL, "https://"))

	// A client visiting user dummy server.
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			RootCAs:    ca.Roots,
			ServerName: domain,
		},
	}
	client := &http.Client{Transport: tr}
	res, err := client.Get(us.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	b, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if v := string(b); v != "OK" {
		t.Errorf("user server response: %q; want 'OK'", v)
	}
}
