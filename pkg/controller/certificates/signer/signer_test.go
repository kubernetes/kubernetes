/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package signer

import (
	"crypto/x509"
	"crypto/x509/pkix"
	"io/ioutil"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	capi "k8s.io/api/certificates/v1beta1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/util/cert"
)

func TestSigner(t *testing.T) {
	clock := clock.FakeClock{}

	s, err := newSigner("./testdata/ca.crt", "./testdata/ca.key", nil, 1*time.Hour)
	if err != nil {
		t.Fatalf("failed to create signer: %v", err)
	}
	currCA, err := s.caProvider.currentCA()
	if err != nil {
		t.Fatal(err)
	}
	currCA.Now = clock.Now
	currCA.Backdate = 0
	s.caProvider.caValue.Store(currCA)

	csrb, err := ioutil.ReadFile("./testdata/kubelet.csr")
	if err != nil {
		t.Fatalf("failed to read CSR: %v", err)
	}

	csr := &capi.CertificateSigningRequest{
		Spec: capi.CertificateSigningRequestSpec{
			Request: []byte(csrb),
			Usages: []capi.KeyUsage{
				capi.UsageSigning,
				capi.UsageKeyEncipherment,
				capi.UsageServerAuth,
				capi.UsageClientAuth,
			},
		},
	}

	csr, err = s.sign(csr)
	if err != nil {
		t.Fatalf("failed to sign CSR: %v", err)
	}
	certData := csr.Status.Certificate
	if len(certData) == 0 {
		t.Fatalf("expected a certificate after signing")
	}

	certs, err := cert.ParseCertsPEM(certData)
	if err != nil {
		t.Fatalf("failed to parse certificate: %v", err)
	}
	if len(certs) != 1 {
		t.Fatalf("expected one certificate")
	}

	want := x509.Certificate{
		Version: 3,
		Subject: pkix.Name{
			CommonName:   "system:node:k-a-node-s36b",
			Organization: []string{"system:nodes"},
		},
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
		NotAfter:              clock.Now().Add(1 * time.Hour),
		PublicKeyAlgorithm:    x509.ECDSA,
		SignatureAlgorithm:    x509.SHA256WithRSA,
		MaxPathLen:            -1,
	}

	if !cmp.Equal(*certs[0], want, diff.IgnoreUnset()) {
		t.Errorf("unexpected diff: %v", cmp.Diff(certs[0], want, diff.IgnoreUnset()))
	}
}
