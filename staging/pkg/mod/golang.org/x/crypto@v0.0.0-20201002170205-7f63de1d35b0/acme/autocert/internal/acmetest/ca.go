// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package acmetest provides types for testing acme and autocert packages.
//
// TODO: Consider moving this to x/crypto/acme/internal/acmetest for acme tests as well.
package acmetest

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"log"
	"math/big"
	"net/http"
	"net/http/httptest"
	"path"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/acme"
)

// CAServer is a simple test server which implements ACME spec bits needed for testing.
type CAServer struct {
	URL   string         // server URL after it has been started
	Roots *x509.CertPool // CA root certificates; initialized in NewCAServer

	rootKey      crypto.Signer
	rootCert     []byte // DER encoding
	rootTemplate *x509.Certificate

	server           *httptest.Server
	challengeTypes   []string // supported challenge types
	domainsWhitelist []string // only these domains are valid for issuing, unless empty

	mu             sync.Mutex
	certCount      int                       // number of issued certs
	domainAddr     map[string]string         // domain name to addr:port resolution
	authorizations map[string]*authorization // keyed by domain name
	orders         []*order                  // index is used as order ID
	errors         []error                   // encountered client errors
}

// NewCAServer creates a new ACME test server and starts serving requests.
// The returned CAServer issues certs signed with the CA roots
// available in the Roots field.
//
// The challengeTypes argument defines the supported ACME challenge types
// sent to a client in a response for a domain authorization.
// If domainsWhitelist is non-empty, the certs will be issued only for the specified
// list of domains. Otherwise, any domain name is allowed.
func NewCAServer(challengeTypes []string, domainsWhitelist []string) *CAServer {
	var whitelist []string
	for _, name := range domainsWhitelist {
		whitelist = append(whitelist, name)
	}
	sort.Strings(whitelist)
	ca := &CAServer{
		challengeTypes:   challengeTypes,
		domainsWhitelist: whitelist,
		domainAddr:       make(map[string]string),
		authorizations:   make(map[string]*authorization),
	}

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		panic(fmt.Sprintf("ecdsa.GenerateKey: %v", err))
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			Organization: []string{"Test Acme Co"},
			CommonName:   "Root CA",
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		panic(fmt.Sprintf("x509.CreateCertificate: %v", err))
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		panic(fmt.Sprintf("x509.ParseCertificate: %v", err))
	}
	ca.Roots = x509.NewCertPool()
	ca.Roots.AddCert(cert)
	ca.rootKey = key
	ca.rootCert = der
	ca.rootTemplate = tmpl

	ca.server = httptest.NewServer(http.HandlerFunc(ca.handle))
	ca.URL = ca.server.URL
	return ca
}

// Close shuts down the server and blocks until all outstanding
// requests on this server have completed.
func (ca *CAServer) Close() {
	ca.server.Close()
}

func (ca *CAServer) serverURL(format string, arg ...interface{}) string {
	return ca.server.URL + fmt.Sprintf(format, arg...)
}

func (ca *CAServer) addr(domain string) (string, error) {
	ca.mu.Lock()
	defer ca.mu.Unlock()
	addr, ok := ca.domainAddr[domain]
	if !ok {
		return "", fmt.Errorf("CAServer: no addr resolution for %q", domain)
	}
	return addr, nil
}

func (ca *CAServer) httpErrorf(w http.ResponseWriter, code int, format string, a ...interface{}) {
	s := fmt.Sprintf(format, a...)
	log.Println(s)
	http.Error(w, s, code)
}

// Resolve adds a domain to address resolution for the ca to dial to
// when validating challenges for the domain authorization.
func (ca *CAServer) Resolve(domain, addr string) {
	ca.mu.Lock()
	defer ca.mu.Unlock()
	ca.domainAddr[domain] = addr
}

type discovery struct {
	NewNonce string `json:"newNonce"`
	NewReg   string `json:"newAccount"`
	NewOrder string `json:"newOrder"`
	NewAuthz string `json:"newAuthz"`
}

type challenge struct {
	URI   string `json:"uri"`
	Type  string `json:"type"`
	Token string `json:"token"`
}

type authorization struct {
	Status     string      `json:"status"`
	Challenges []challenge `json:"challenges"`

	domain string
}

type order struct {
	Status      string   `json:"status"`
	AuthzURLs   []string `json:"authorizations"`
	FinalizeURL string   `json:"finalize"`    // CSR submit URL
	CertURL     string   `json:"certificate"` // already issued cert

	leaf []byte // issued cert in DER format
}

func (ca *CAServer) handle(w http.ResponseWriter, r *http.Request) {
	log.Printf("%s %s", r.Method, r.URL)
	w.Header().Set("Replay-Nonce", "nonce")
	// TODO: Verify nonce header for all POST requests.

	switch {
	default:
		ca.httpErrorf(w, http.StatusBadRequest, "unrecognized r.URL.Path: %s", r.URL.Path)

	// Discovery request.
	case r.URL.Path == "/":
		resp := &discovery{
			NewNonce: ca.serverURL("/new-nonce"),
			NewReg:   ca.serverURL("/new-reg"),
			NewOrder: ca.serverURL("/new-order"),
			NewAuthz: ca.serverURL("/new-authz"),
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			panic(fmt.Sprintf("discovery response: %v", err))
		}

	// Nonce requests.
	case r.URL.Path == "/new-nonce":
		// Nonce values are always set. Nothing else to do.
		return

	// Client key registration request.
	case r.URL.Path == "/new-reg":
		// TODO: Check the user account key against a ca.accountKeys?
		w.Header().Set("Location", ca.serverURL("/accounts/1"))
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte("{}"))

	// New order request.
	case r.URL.Path == "/new-order":
		var req struct {
			Identifiers []struct{ Value string }
		}
		if err := decodePayload(&req, r.Body); err != nil {
			ca.httpErrorf(w, http.StatusBadRequest, err.Error())
			return
		}
		ca.mu.Lock()
		defer ca.mu.Unlock()
		o := &order{Status: acme.StatusPending}
		for _, id := range req.Identifiers {
			z := ca.authz(id.Value)
			o.AuthzURLs = append(o.AuthzURLs, ca.serverURL("/authz/%s", z.domain))
		}
		orderID := len(ca.orders)
		ca.orders = append(ca.orders, o)
		w.Header().Set("Location", ca.serverURL("/orders/%d", orderID))
		w.WriteHeader(http.StatusCreated)
		if err := json.NewEncoder(w).Encode(o); err != nil {
			panic(err)
		}

	// Existing order status requests.
	case strings.HasPrefix(r.URL.Path, "/orders/"):
		ca.mu.Lock()
		defer ca.mu.Unlock()
		o, err := ca.storedOrder(strings.TrimPrefix(r.URL.Path, "/orders/"))
		if err != nil {
			ca.httpErrorf(w, http.StatusBadRequest, err.Error())
			return
		}
		if err := json.NewEncoder(w).Encode(o); err != nil {
			panic(err)
		}

	// Identifier authorization request.
	case r.URL.Path == "/new-authz":
		var req struct {
			Identifier struct{ Value string }
		}
		if err := decodePayload(&req, r.Body); err != nil {
			ca.httpErrorf(w, http.StatusBadRequest, err.Error())
			return
		}
		ca.mu.Lock()
		defer ca.mu.Unlock()
		z := ca.authz(req.Identifier.Value)
		w.Header().Set("Location", ca.serverURL("/authz/%s", z.domain))
		w.WriteHeader(http.StatusCreated)
		if err := json.NewEncoder(w).Encode(z); err != nil {
			panic(fmt.Sprintf("new authz response: %v", err))
		}

	// Accept tls-alpn-01 challenge type requests.
	case strings.HasPrefix(r.URL.Path, "/challenge/tls-alpn-01/"):
		domain := strings.TrimPrefix(r.URL.Path, "/challenge/tls-alpn-01/")
		ca.mu.Lock()
		_, exist := ca.authorizations[domain]
		ca.mu.Unlock()
		if !exist {
			ca.httpErrorf(w, http.StatusBadRequest, "challenge accept: no authz for %q", domain)
			return
		}
		go ca.validateChallenge("tls-alpn-01", domain)
		w.Write([]byte("{}"))

	// Get authorization status requests.
	case strings.HasPrefix(r.URL.Path, "/authz/"):
		domain := strings.TrimPrefix(r.URL.Path, "/authz/")
		ca.mu.Lock()
		defer ca.mu.Unlock()
		authz, ok := ca.authorizations[domain]
		if !ok {
			ca.httpErrorf(w, http.StatusNotFound, "no authz for %q", domain)
			return
		}
		if err := json.NewEncoder(w).Encode(authz); err != nil {
			panic(fmt.Sprintf("get authz for %q response: %v", domain, err))
		}

	// Certificate issuance request.
	case strings.HasPrefix(r.URL.Path, "/new-cert/"):
		ca.mu.Lock()
		defer ca.mu.Unlock()
		orderID := strings.TrimPrefix(r.URL.Path, "/new-cert/")
		o, err := ca.storedOrder(orderID)
		if err != nil {
			ca.httpErrorf(w, http.StatusBadRequest, err.Error())
			return
		}
		if o.Status != acme.StatusReady {
			ca.httpErrorf(w, http.StatusForbidden, "order status: %s", o.Status)
			return
		}
		// Validate CSR request.
		var req struct {
			CSR string `json:"csr"`
		}
		decodePayload(&req, r.Body)
		b, _ := base64.RawURLEncoding.DecodeString(req.CSR)
		csr, err := x509.ParseCertificateRequest(b)
		if err != nil {
			ca.httpErrorf(w, http.StatusBadRequest, err.Error())
			return
		}
		names := unique(append(csr.DNSNames, csr.Subject.CommonName))
		if err := ca.matchWhitelist(names); err != nil {
			ca.httpErrorf(w, http.StatusUnauthorized, err.Error())
			return
		}
		if err := ca.authorized(names); err != nil {
			ca.httpErrorf(w, http.StatusUnauthorized, err.Error())
			return
		}
		// Issue the certificate.
		der, err := ca.leafCert(csr)
		if err != nil {
			ca.httpErrorf(w, http.StatusBadRequest, "new-cert response: ca.leafCert: %v", err)
			return
		}
		o.leaf = der
		o.CertURL = ca.serverURL("/issued-cert/%s", orderID)
		o.Status = acme.StatusValid
		if err := json.NewEncoder(w).Encode(o); err != nil {
			panic(err)
		}

	// Already issued cert download requests.
	case strings.HasPrefix(r.URL.Path, "/issued-cert/"):
		ca.mu.Lock()
		defer ca.mu.Unlock()
		o, err := ca.storedOrder(strings.TrimPrefix(r.URL.Path, "/issued-cert/"))
		if err != nil {
			ca.httpErrorf(w, http.StatusBadRequest, err.Error())
			return
		}
		if o.Status != acme.StatusValid {
			ca.httpErrorf(w, http.StatusForbidden, "order status: %s", o.Status)
			return
		}
		w.Header().Set("Content-Type", "application/pem-certificate-chain")
		pem.Encode(w, &pem.Block{Type: "CERTIFICATE", Bytes: o.leaf})
		pem.Encode(w, &pem.Block{Type: "CERTIFICATE", Bytes: ca.rootCert})
	}
}

// matchWhitelist reports whether all dnsNames are whitelisted.
// The whitelist is provided in NewCAServer.
func (ca *CAServer) matchWhitelist(dnsNames []string) error {
	if len(ca.domainsWhitelist) == 0 {
		return nil
	}
	var nomatch []string
	for _, name := range dnsNames {
		i := sort.SearchStrings(ca.domainsWhitelist, name)
		if i == len(ca.domainsWhitelist) || ca.domainsWhitelist[i] != name {
			nomatch = append(nomatch, name)
		}
	}
	if len(nomatch) > 0 {
		return fmt.Errorf("matchWhitelist: some domains don't match: %q", nomatch)
	}
	return nil
}

// storedOrder retrieves a previously created order at index i.
// It requires ca.mu to be locked.
func (ca *CAServer) storedOrder(i string) (*order, error) {
	idx, err := strconv.Atoi(i)
	if err != nil {
		return nil, fmt.Errorf("storedOrder: %v", err)
	}
	if idx < 0 {
		return nil, fmt.Errorf("storedOrder: invalid order index %d", idx)
	}
	if idx > len(ca.orders)-1 {
		return nil, fmt.Errorf("storedOrder: no such order %d", idx)
	}
	return ca.orders[idx], nil
}

// authz returns an existing authorization for the identifier or creates a new one.
// It requires ca.mu to be locked.
func (ca *CAServer) authz(identifier string) *authorization {
	authz, ok := ca.authorizations[identifier]
	if !ok {
		authz = &authorization{
			domain: identifier,
			Status: acme.StatusPending,
		}
		for _, typ := range ca.challengeTypes {
			authz.Challenges = append(authz.Challenges, challenge{
				Type:  typ,
				URI:   ca.serverURL("/challenge/%s/%s", typ, authz.domain),
				Token: challengeToken(authz.domain, typ),
			})
		}
		ca.authorizations[authz.domain] = authz
	}
	return authz
}

// authorized reports whether all authorizations for dnsNames have been satisfied.
// It requires ca.mu to be locked.
func (ca *CAServer) authorized(dnsNames []string) error {
	var noauthz []string
	for _, name := range dnsNames {
		authz, ok := ca.authorizations[name]
		if !ok || authz.Status != acme.StatusValid {
			noauthz = append(noauthz, name)
		}
	}
	if len(noauthz) > 0 {
		return fmt.Errorf("CAServer: no authz for %q", noauthz)
	}
	return nil
}

// leafCert issues a new certificate.
// It requires ca.mu to be locked.
func (ca *CAServer) leafCert(csr *x509.CertificateRequest) (der []byte, err error) {
	ca.certCount++ // next leaf cert serial number
	leaf := &x509.Certificate{
		SerialNumber:          big.NewInt(int64(ca.certCount)),
		Subject:               pkix.Name{Organization: []string{"Test Acme Co"}},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(90 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames:              csr.DNSNames,
		BasicConstraintsValid: true,
	}
	if len(csr.DNSNames) == 0 {
		leaf.DNSNames = []string{csr.Subject.CommonName}
	}
	return x509.CreateCertificate(rand.Reader, leaf, ca.rootTemplate, csr.PublicKey, ca.rootKey)
}

// TODO: Only tls-alpn-01 is currently supported: implement http-01 and dns-01.
func (ca *CAServer) validateChallenge(typ, identifier string) {
	var err error
	switch typ {
	case "tls-alpn-01":
		err = ca.verifyALPNChallenge(identifier)
	default:
		panic(fmt.Sprintf("validation of %q is not implemented", typ))
	}
	ca.mu.Lock()
	defer ca.mu.Unlock()
	authz := ca.authorizations[identifier]
	if err != nil {
		authz.Status = "invalid"
	} else {
		authz.Status = "valid"
	}
	log.Printf("validated %q for %q; authz status is now: %s", typ, identifier, authz.Status)
	// Update all pending orders.
	// An order becomes "ready" if all authorizations are "valid".
	// An order becomes "invalid" if any authorization is "invalid".
	// Status changes: https://tools.ietf.org/html/rfc8555#section-7.1.6
OrdersLoop:
	for i, o := range ca.orders {
		if o.Status != acme.StatusPending {
			continue
		}
		var countValid int
		for _, zurl := range o.AuthzURLs {
			z, ok := ca.authorizations[path.Base(zurl)]
			if !ok {
				log.Printf("no authz %q for order %d", zurl, i)
				continue OrdersLoop
			}
			if z.Status == acme.StatusInvalid {
				o.Status = acme.StatusInvalid
				log.Printf("order %d is now invalid", i)
				continue OrdersLoop
			}
			if z.Status == acme.StatusValid {
				countValid++
			}
		}
		if countValid == len(o.AuthzURLs) {
			o.Status = acme.StatusReady
			o.FinalizeURL = ca.serverURL("/new-cert/%d", i)
			log.Printf("order %d is now ready", i)
		}
	}
}

func (ca *CAServer) verifyALPNChallenge(domain string) error {
	const acmeALPNProto = "acme-tls/1"

	addr, err := ca.addr(domain)
	if err != nil {
		return err
	}
	conn, err := tls.Dial("tcp", addr, &tls.Config{
		ServerName:         domain,
		InsecureSkipVerify: true,
		NextProtos:         []string{acmeALPNProto},
	})
	if err != nil {
		return err
	}
	if v := conn.ConnectionState().NegotiatedProtocol; v != acmeALPNProto {
		return fmt.Errorf("CAServer: verifyALPNChallenge: negotiated proto is %q; want %q", v, acmeALPNProto)
	}
	if n := len(conn.ConnectionState().PeerCertificates); n != 1 {
		return fmt.Errorf("len(PeerCertificates) = %d; want 1", n)
	}
	// TODO: verify conn.ConnectionState().PeerCertificates[0]
	return nil
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

func challengeToken(domain, challType string) string {
	return fmt.Sprintf("token-%s-%s", domain, challType)
}

func unique(a []string) []string {
	seen := make(map[string]bool)
	var res []string
	for _, s := range a {
		if s != "" && !seen[s] {
			seen[s] = true
			res = append(res, s)
		}
	}
	return res
}
