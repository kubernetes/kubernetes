/*
Copyright 2017 The Kubernetes Authors.

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
	"io/ioutil"
	"reflect"
	"strings"
	"testing"
	"time"

	capi "k8s.io/api/certificates/v1beta1"
	"k8s.io/client-go/util/cert"
)

func TestSigner(t *testing.T) {
	testNow := time.Now()
	testNowFn := func() time.Time {
		return testNow
	}

	s, err := newCFSSLSigner("./testdata/ca.crt", "./testdata/ca.key", nil, 1*time.Hour)
	if err != nil {
		t.Fatalf("failed to create signer: %v", err)
	}
	s.nowFn = testNowFn

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

	crt := certs[0]

	if crt.Subject.CommonName != "system:node:k-a-node-s36b" {
		t.Errorf("expected common name of 'system:node:k-a-node-s36b', but got: %v", certs[0].Subject.CommonName)
	}
	if !reflect.DeepEqual(crt.Subject.Organization, []string{"system:nodes"}) {
		t.Errorf("expected organization to be [system:nodes] but got: %v", crt.Subject.Organization)
	}
	if crt.KeyUsage != x509.KeyUsageDigitalSignature|x509.KeyUsageKeyEncipherment {
		t.Errorf("bad key usage")
	}
	if !reflect.DeepEqual(crt.ExtKeyUsage, []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth}) {
		t.Errorf("bad extended key usage")
	}

	expectedTime := testNow.Add(1 * time.Hour)
	// there is some jitter that we need to tolerate
	diff := expectedTime.Sub(crt.NotAfter)
	if diff > 10*time.Minute || diff < -10*time.Minute {
		t.Fatal(crt.NotAfter)
	}
}

func TestSignerExpired(t *testing.T) {
	hundredYearsFromNowFn := func() time.Time {
		return time.Now().Add(24 * time.Hour * 365 * 100)
	}
	s, err := newCFSSLSigner("./testdata/ca.crt", "./testdata/ca.key", nil, 1*time.Hour)
	if err != nil {
		t.Fatalf("failed to create signer: %v", err)
	}
	s.nowFn = hundredYearsFromNowFn

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

	_, err = s.sign(csr)
	if err == nil {
		t.Fatal("missing error")
	}
	if !strings.HasPrefix(err.Error(), "the signer has expired") {
		t.Fatal(err)
	}
}

func TestDurationLongerThanExpiry(t *testing.T) {
	testNow := time.Now()
	testNowFn := func() time.Time {
		return testNow
	}

	hundredYears := 24 * time.Hour * 365 * 100
	s, err := newCFSSLSigner("./testdata/ca.crt", "./testdata/ca.key", nil, hundredYears)
	if err != nil {
		t.Fatalf("failed to create signer: %v", err)
	}
	s.nowFn = testNowFn

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

	_, err = s.sign(csr)
	if err != nil {
		t.Fatalf("failed to sign CSR: %v", err)
	}

	// now we just need to verify that the expiry is based on the signing cert
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

	crt := certs[0]
	expected, err := time.Parse("2006-01-02 15:04:05.999999999 -0700 MST", "2044-05-09 00:20:11 +0000 UTC")
	if err != nil {
		t.Fatal(err)
	}
	// there is some jitter that we need to tolerate
	diff := expected.Sub(crt.NotAfter)
	if diff > 10*time.Minute || diff < -10*time.Minute {
		t.Fatal(crt.NotAfter)
	}
}
