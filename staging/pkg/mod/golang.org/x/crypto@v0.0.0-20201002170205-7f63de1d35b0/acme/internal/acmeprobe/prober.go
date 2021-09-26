// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The acmeprober program runs against an actual ACME CA implementation.
// It spins up an HTTP server to fulfill authorization challenges
// or execute a DNS script to provision a response to dns-01 challenge.
//
// For http-01 and tls-alpn-01 challenge types this requires the ACME CA
// to be able to reach the HTTP server.
//
// A usage example:
//
//     go run prober.go \
//       -d https://acme-staging-v02.api.letsencrypt.org/directory \
//       -f order \
//       -t http-01 \
//       -a :8080 \
//       -domain some.example.org
//
// The above assumes a TCP tunnel from some.example.org:80 to 0.0.0.0:8080
// in order for the test to be able to fulfill http-01 challenge.
// To test tls-alpn-01 challenge, 443 port would need to be tunneled
// to 0.0.0.0:8080.
// When running with dns-01 challenge type, use -s argument instead of -a.
package main

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"

	"golang.org/x/crypto/acme"
)

var (
	// ACME CA directory URL.
	// Let's Encrypt v1 prod: https://acme-v01.api.letsencrypt.org/directory
	// Let's Encrypt v2 prod: https://acme-v02.api.letsencrypt.org/directory
	// Let's Encrypt v2 staging: https://acme-staging-v02.api.letsencrypt.org/directory
	// See the following for more CAs implementing ACME protocol:
	// https://en.wikipedia.org/wiki/Automated_Certificate_Management_Environment#CAs_&_PKIs_that_offer_ACME_certificates
	directory = flag.String("d", "", "ACME directory URL.")
	reginfo   = flag.String("r", "", "ACME account registration info.")
	flow      = flag.String("f", "", "Flow to run: order, preauthz (RFC8555) or preauthz02 (draft-02).")
	chaltyp   = flag.String("t", "", "Challenge type: tls-alpn-01, http-01 or dns-01.")
	addr      = flag.String("a", "", "Local server address for tls-alpn-01 and http-01.")
	dnsscript = flag.String("s", "", "Script to run for provisioning dns-01 challenges.")
	domain    = flag.String("domain", "", "Space separate domain identifiers.")
	ipaddr    = flag.String("ip", "", "Space separate IP address identifiers.")
)

func main() {
	flag.Usage = func() {
		fmt.Fprintln(flag.CommandLine.Output(), `
The prober program runs against an actual ACME CA implementation.
It spins up an HTTP server to fulfill authorization challenges
or execute a DNS script to provision a response to dns-01 challenge.

For http-01 and tls-alpn-01 challenge types this requires the ACME CA
to be able to reach the HTTP server.

A usage example:

    go run prober.go \
      -d https://acme-staging-v02.api.letsencrypt.org/directory \
      -f order \
      -t http-01 \
      -a :8080 \
      -domain some.example.org

The above assumes a TCP tunnel from some.example.org:80 to 0.0.0.0:8080
in order for the test to be able to fulfill http-01 challenge.
To test tls-alpn-01 challenge, 443 port would need to be tunneled
to 0.0.0.0:8080.
When running with dns-01 challenge type, use -s argument instead of -a.
		`)
		flag.PrintDefaults()
	}
	flag.Parse()

	identifiers := acme.DomainIDs(strings.Fields(*domain)...)
	identifiers = append(identifiers, acme.IPIDs(strings.Fields(*ipaddr)...)...)
	if len(identifiers) == 0 {
		log.Fatal("at least one domain or IP addr identifier is required")
	}

	// Duration of the whole run.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	// Create and register a new account.
	akey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		log.Fatal(err)
	}
	cl := &acme.Client{Key: akey, DirectoryURL: *directory}
	a := &acme.Account{Contact: strings.Fields(*reginfo)}
	if _, err := cl.Register(ctx, a, acme.AcceptTOS); err != nil {
		log.Fatalf("Register: %v", err)
	}

	// Run the desired flow test.
	p := &prober{
		client:    cl,
		chalType:  *chaltyp,
		localAddr: *addr,
		dnsScript: *dnsscript,
	}
	switch *flow {
	case "order":
		p.runOrder(ctx, identifiers)
	case "preauthz":
		p.runPreauthz(ctx, identifiers)
	case "preauthz02":
		p.runPreauthzLegacy(ctx, identifiers)
	default:
		log.Fatalf("unknown flow: %q", *flow)
	}
	if len(p.errors) > 0 {
		os.Exit(1)
	}
}

type prober struct {
	client    *acme.Client
	chalType  string
	localAddr string
	dnsScript string

	errors []error
}

func (p *prober) errorf(format string, a ...interface{}) {
	err := fmt.Errorf(format, a...)
	log.Print(err)
	p.errors = append(p.errors, err)
}

func (p *prober) runOrder(ctx context.Context, identifiers []acme.AuthzID) {
	// Create a new order and pick a challenge.
	// Note that Let's Encrypt will reply with 400 error:malformed
	// "NotBefore and NotAfter are not supported" when providing a NotAfter
	// value like WithOrderNotAfter(time.Now().Add(24 * time.Hour)).
	o, err := p.client.AuthorizeOrder(ctx, identifiers)
	if err != nil {
		log.Fatalf("AuthorizeOrder: %v", err)
	}

	var zurls []string
	for _, u := range o.AuthzURLs {
		z, err := p.client.GetAuthorization(ctx, u)
		if err != nil {
			log.Fatalf("GetAuthorization(%q): %v", u, err)
		}
		log.Printf("%+v", z)
		if z.Status != acme.StatusPending {
			log.Printf("authz status is %q; skipping", z.Status)
			continue
		}
		if err := p.fulfill(ctx, z); err != nil {
			log.Fatalf("fulfill(%s): %v", z.URI, err)
		}
		zurls = append(zurls, z.URI)
		log.Printf("authorized for %+v", z.Identifier)
	}

	log.Print("all challenges are done")
	if _, err := p.client.WaitOrder(ctx, o.URI); err != nil {
		log.Fatalf("WaitOrder(%q): %v", o.URI, err)
	}
	csr, certkey := newCSR(identifiers)
	der, curl, err := p.client.CreateOrderCert(ctx, o.FinalizeURL, csr, true)
	if err != nil {
		log.Fatalf("CreateOrderCert: %v", err)
	}
	log.Printf("cert URL: %s", curl)
	if err := checkCert(der, identifiers); err != nil {
		p.errorf("invalid cert: %v", err)
	}

	// Deactivate all authorizations we satisfied earlier.
	for _, v := range zurls {
		if err := p.client.RevokeAuthorization(ctx, v); err != nil {
			p.errorf("RevokAuthorization(%q): %v", v, err)
			continue
		}
	}
	// Deactivate the account. We don't need it for any further calls.
	if err := p.client.DeactivateReg(ctx); err != nil {
		p.errorf("DeactivateReg: %v", err)
	}
	// Try revoking the issued cert using its private key.
	if err := p.client.RevokeCert(ctx, certkey, der[0], acme.CRLReasonCessationOfOperation); err != nil {
		p.errorf("RevokeCert: %v", err)
	}
}

func (p *prober) runPreauthz(ctx context.Context, identifiers []acme.AuthzID) {
	dir, err := p.client.Discover(ctx)
	if err != nil {
		log.Fatalf("Discover: %v", err)
	}
	if dir.AuthzURL == "" {
		log.Fatal("CA does not support pre-authorization")
	}

	var zurls []string
	for _, id := range identifiers {
		z, err := authorize(ctx, p.client, id)
		if err != nil {
			log.Fatalf("AuthorizeID(%+v): %v", z, err)
		}
		if z.Status == acme.StatusValid {
			log.Printf("authz %s is valid; skipping", z.URI)
			continue
		}
		if err := p.fulfill(ctx, z); err != nil {
			log.Fatalf("fulfill(%s): %v", z.URI, err)
		}
		zurls = append(zurls, z.URI)
		log.Printf("authorized for %+v", id)
	}

	// We should be all set now.
	// Expect all authorizations to be satisfied.
	log.Print("all challenges are done")
	o, err := p.client.AuthorizeOrder(ctx, identifiers)
	if err != nil {
		log.Fatalf("AuthorizeOrder: %v", err)
	}
	waitCtx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	if _, err := p.client.WaitOrder(waitCtx, o.URI); err != nil {
		log.Fatalf("WaitOrder(%q): %v", o.URI, err)
	}
	csr, certkey := newCSR(identifiers)
	der, curl, err := p.client.CreateOrderCert(ctx, o.FinalizeURL, csr, true)
	if err != nil {
		log.Fatalf("CreateOrderCert: %v", err)
	}
	log.Printf("cert URL: %s", curl)
	if err := checkCert(der, identifiers); err != nil {
		p.errorf("invalid cert: %v", err)
	}

	// Deactivate all authorizations we satisfied earlier.
	for _, v := range zurls {
		if err := p.client.RevokeAuthorization(ctx, v); err != nil {
			p.errorf("RevokeAuthorization(%q): %v", v, err)
			continue
		}
	}
	// Deactivate the account. We don't need it for any further calls.
	if err := p.client.DeactivateReg(ctx); err != nil {
		p.errorf("DeactivateReg: %v", err)
	}
	// Try revoking the issued cert using its private key.
	if err := p.client.RevokeCert(ctx, certkey, der[0], acme.CRLReasonCessationOfOperation); err != nil {
		p.errorf("RevokeCert: %v", err)
	}
}

func (p *prober) runPreauthzLegacy(ctx context.Context, identifiers []acme.AuthzID) {
	var zurls []string
	for _, id := range identifiers {
		z, err := authorize(ctx, p.client, id)
		if err != nil {
			log.Fatalf("AuthorizeID(%+v): %v", id, err)
		}
		if z.Status == acme.StatusValid {
			log.Printf("authz %s is valid; skipping", z.URI)
			continue
		}
		if err := p.fulfill(ctx, z); err != nil {
			log.Fatalf("fulfill(%s): %v", z.URI, err)
		}
		zurls = append(zurls, z.URI)
		log.Printf("authorized for %+v", id)
	}

	// We should be all set now.
	log.Print("all authorizations are done")
	csr, certkey := newCSR(identifiers)
	der, curl, err := p.client.CreateCert(ctx, csr, 48*time.Hour, true)
	if err != nil {
		log.Fatalf("CreateCert: %v", err)
	}
	log.Printf("cert URL: %s", curl)
	if err := checkCert(der, identifiers); err != nil {
		p.errorf("invalid cert: %v", err)
	}

	// Deactivate all authorizations we satisfied earlier.
	for _, v := range zurls {
		if err := p.client.RevokeAuthorization(ctx, v); err != nil {
			p.errorf("RevokAuthorization(%q): %v", v, err)
			continue
		}
	}
	// Try revoking the issued cert using its private key.
	if err := p.client.RevokeCert(ctx, certkey, der[0], acme.CRLReasonCessationOfOperation); err != nil {
		p.errorf("RevokeCert: %v", err)
	}

}

func (p *prober) fulfill(ctx context.Context, z *acme.Authorization) error {
	var chal *acme.Challenge
	for i, c := range z.Challenges {
		log.Printf("challenge %d: %+v", i, c)
		if c.Type == p.chalType {
			log.Printf("picked %s for authz %s", c.URI, z.URI)
			chal = c
		}
	}
	if chal == nil {
		return fmt.Errorf("challenge type %q wasn't offered for authz %s", p.chalType, z.URI)
	}

	switch chal.Type {
	case "tls-alpn-01":
		return p.runTLSALPN01(ctx, z, chal)
	case "http-01":
		return p.runHTTP01(ctx, z, chal)
	case "dns-01":
		return p.runDNS01(ctx, z, chal)
	default:
		return fmt.Errorf("unknown challenge type %q", chal.Type)
	}
}

func (p *prober) runTLSALPN01(ctx context.Context, z *acme.Authorization, chal *acme.Challenge) error {
	tokenCert, err := p.client.TLSALPN01ChallengeCert(chal.Token, z.Identifier.Value)
	if err != nil {
		return fmt.Errorf("TLSALPN01ChallengeCert: %v", err)
	}
	s := &http.Server{
		Addr: p.localAddr,
		TLSConfig: &tls.Config{
			NextProtos: []string{acme.ALPNProto},
			GetCertificate: func(hello *tls.ClientHelloInfo) (*tls.Certificate, error) {
				log.Printf("hello: %+v", hello)
				return &tokenCert, nil
			},
		},
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			log.Printf("%s %s", r.Method, r.URL)
			w.WriteHeader(http.StatusNotFound)
		}),
	}
	go s.ListenAndServeTLS("", "")
	defer s.Close()

	if _, err := p.client.Accept(ctx, chal); err != nil {
		return fmt.Errorf("Accept(%q): %v", chal.URI, err)
	}
	_, zerr := p.client.WaitAuthorization(ctx, z.URI)
	return zerr
}

func (p *prober) runHTTP01(ctx context.Context, z *acme.Authorization, chal *acme.Challenge) error {
	body, err := p.client.HTTP01ChallengeResponse(chal.Token)
	if err != nil {
		return fmt.Errorf("HTTP01ChallengeResponse: %v", err)
	}
	s := &http.Server{
		Addr: p.localAddr,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			log.Printf("%s %s", r.Method, r.URL)
			if r.URL.Path != p.client.HTTP01ChallengePath(chal.Token) {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			w.Write([]byte(body))
		}),
	}
	go s.ListenAndServe()
	defer s.Close()

	if _, err := p.client.Accept(ctx, chal); err != nil {
		return fmt.Errorf("Accept(%q): %v", chal.URI, err)
	}
	_, zerr := p.client.WaitAuthorization(ctx, z.URI)
	return zerr
}

func (p *prober) runDNS01(ctx context.Context, z *acme.Authorization, chal *acme.Challenge) error {
	token, err := p.client.DNS01ChallengeRecord(chal.Token)
	if err != nil {
		return fmt.Errorf("DNS01ChallengeRecord: %v", err)
	}

	name := fmt.Sprintf("_acme-challenge.%s", z.Identifier.Value)
	cmd := exec.CommandContext(ctx, p.dnsScript, name, token)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s: %v", p.dnsScript, err)
	}

	if _, err := p.client.Accept(ctx, chal); err != nil {
		return fmt.Errorf("Accept(%q): %v", chal.URI, err)
	}
	_, zerr := p.client.WaitAuthorization(ctx, z.URI)
	return zerr
}

func authorize(ctx context.Context, client *acme.Client, id acme.AuthzID) (*acme.Authorization, error) {
	if id.Type == "ip" {
		return client.AuthorizeIP(ctx, id.Value)
	}
	return client.Authorize(ctx, id.Value)
}

func checkCert(derChain [][]byte, id []acme.AuthzID) error {
	if len(derChain) == 0 {
		return errors.New("cert chain is zero bytes")
	}
	for i, b := range derChain {
		crt, err := x509.ParseCertificate(b)
		if err != nil {
			return fmt.Errorf("%d: ParseCertificate: %v", i, err)
		}
		log.Printf("%d: serial: 0x%s", i, crt.SerialNumber)
		log.Printf("%d: subject: %s", i, crt.Subject)
		log.Printf("%d: issuer: %s", i, crt.Issuer)
		log.Printf("%d: expires in %.1f day(s)", i, time.Until(crt.NotAfter).Hours()/24)
		if i > 0 { // not a leaf cert
			continue
		}
		p := &pem.Block{Type: "CERTIFICATE", Bytes: b}
		log.Printf("%d: leaf:\n%s", i, pem.EncodeToMemory(p))
		for _, v := range id {
			if err := crt.VerifyHostname(v.Value); err != nil {
				return err
			}
		}
	}
	return nil
}

func newCSR(identifiers []acme.AuthzID) ([]byte, crypto.Signer) {
	var csr x509.CertificateRequest
	for _, id := range identifiers {
		switch id.Type {
		case "dns":
			csr.DNSNames = append(csr.DNSNames, id.Value)
		case "ip":
			csr.IPAddresses = append(csr.IPAddresses, net.ParseIP(id.Value))
		default:
			panic(fmt.Sprintf("newCSR: unknown identifier type %q", id.Type))
		}
	}
	k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		panic(fmt.Sprintf("newCSR: ecdsa.GenerateKey for a cert: %v", err))
	}
	b, err := x509.CreateCertificateRequest(rand.Reader, &csr, k)
	if err != nil {
		panic(fmt.Sprintf("newCSR: x509.CreateCertificateRequest: %v", err))
	}
	return b, k
}
