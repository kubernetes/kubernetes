// Copyright 2020 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package httptest provides a method for testing a TLS server a la net/http/httptest.
package httptest

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"time"
)

// NewTLSServer returns an httptest server, with an http client that has been configured to
// send all requests to the returned server. The TLS certs are generated for the given domain.
// If you need a transport, Client().Transport is correctly configured.
func NewTLSServer(domain string, handler http.Handler) (*httptest.Server, error) {
	s := httptest.NewUnstartedServer(handler)

	template := x509.Certificate{
		SerialNumber: big.NewInt(1),
		NotBefore:    time.Now().Add(-1 * time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		IPAddresses: []net.IP{
			net.IPv4(127, 0, 0, 1),
			net.IPv6loopback,
		},
		DNSNames: []string{domain},

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
	}

	priv, err := ecdsa.GenerateKey(elliptic.P521(), rand.Reader)
	if err != nil {
		return nil, err
	}

	b, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		return nil, err
	}

	pc := &bytes.Buffer{}
	if err := pem.Encode(pc, &pem.Block{Type: "CERTIFICATE", Bytes: b}); err != nil {
		return nil, err
	}

	ek, err := x509.MarshalECPrivateKey(priv)
	if err != nil {
		return nil, err
	}

	pk := &bytes.Buffer{}
	if err := pem.Encode(pk, &pem.Block{Type: "EC PRIVATE KEY", Bytes: ek}); err != nil {
		return nil, err
	}

	c, err := tls.X509KeyPair(pc.Bytes(), pk.Bytes())
	if err != nil {
		return nil, err
	}
	s.TLS = &tls.Config{
		Certificates: []tls.Certificate{c},
	}
	s.StartTLS()

	certpool := x509.NewCertPool()
	certpool.AddCert(s.Certificate())

	t := &http.Transport{
		TLSClientConfig: &tls.Config{
			RootCAs: certpool,
		},
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			return net.Dial(s.Listener.Addr().Network(), s.Listener.Addr().String())
		},
	}
	s.Client().Transport = t

	return s, nil
}
