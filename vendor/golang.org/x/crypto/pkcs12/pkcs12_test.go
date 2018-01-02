// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkcs12

import (
	"crypto/rsa"
	"crypto/tls"
	"encoding/base64"
	"encoding/pem"
	"testing"
)

func TestPfx(t *testing.T) {
	for commonName, base64P12 := range testdata {
		p12, _ := base64.StdEncoding.DecodeString(base64P12)

		priv, cert, err := Decode(p12, "")
		if err != nil {
			t.Fatal(err)
		}

		if err := priv.(*rsa.PrivateKey).Validate(); err != nil {
			t.Errorf("error while validating private key: %v", err)
		}

		if cert.Subject.CommonName != commonName {
			t.Errorf("expected common name to be %q, but found %q", commonName, cert.Subject.CommonName)
		}
	}
}

func TestPEM(t *testing.T) {
	for commonName, base64P12 := range testdata {
		p12, _ := base64.StdEncoding.DecodeString(base64P12)

		blocks, err := ToPEM(p12, "")
		if err != nil {
			t.Fatalf("error while converting to PEM: %s", err)
		}

		var pemData []byte
		for _, b := range blocks {
			pemData = append(pemData, pem.EncodeToMemory(b)...)
		}

		cert, err := tls.X509KeyPair(pemData, pemData)
		if err != nil {
			t.Errorf("err while converting to key pair: %v", err)
		}
		config := tls.Config{
			Certificates: []tls.Certificate{cert},
		}
		config.BuildNameToCertificate()

		if _, exists := config.NameToCertificate[commonName]; !exists {
			t.Errorf("did not find our cert in PEM?: %v", config.NameToCertificate)
		}
	}
}

func ExampleToPEM() {
	p12, _ := base64.StdEncoding.DecodeString(`MIIJzgIBAzCCCZQGCS ... CA+gwggPk==`)

	blocks, err := ToPEM(p12, "password")
	if err != nil {
		panic(err)
	}

	var pemData []byte
	for _, b := range blocks {
		pemData = append(pemData, pem.EncodeToMemory(b)...)
	}

	// then use PEM data for tls to construct tls certificate:
	cert, err := tls.X509KeyPair(pemData, pemData)
	if err != nil {
		panic(err)
	}

	config := &tls.Config{
		Certificates: []tls.Certificate{cert},
	}

	_ = config
}

var testdata = map[string]string{
	// 'null' password test case
	"Windows Azure Tools": `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`,
	// empty string password test case
	"testing@example.com": `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
HQ8CAggA`,
}
