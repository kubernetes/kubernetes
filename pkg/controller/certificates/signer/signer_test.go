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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"io/ioutil"
	"math/rand"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	capi "k8s.io/api/certificates/v1beta1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/kubernetes/fake"
	testclient "k8s.io/client-go/testing"
	"k8s.io/client-go/util/cert"

	capihelper "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
)

func TestSigner(t *testing.T) {
	clock := clock.FakeClock{}

	s, err := newSigner(capi.LegacyUnknownSignerName, "./testdata/ca.crt", "./testdata/ca.key", nil, 1*time.Hour)
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
	x509cr, err := capihelper.ParseCSR(csrb)
	if err != nil {
		t.Fatalf("failed to parse CSR: %v", err)
	}

	certData, err := s.sign(x509cr, []capi.KeyUsage{
		capi.UsageSigning,
		capi.UsageKeyEncipherment,
		capi.UsageServerAuth,
		capi.UsageClientAuth,
	})
	if err != nil {
		t.Fatalf("failed to sign CSR: %v", err)
	}
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

func TestHandle(t *testing.T) {
	cases := []struct {
		name string
		// parameters to be set on the generated CSR
		commonName string
		dnsNames   []string
		org        []string
		usages     []capi.KeyUsage
		// whether the generated CSR should be marked as approved
		approved bool
		// the signerName to be set on the generated CSR
		signerName string
		// if true, expect an error to be returned
		err bool
		// if true, expect an error to be returned during construction
		constructionErr bool
		// additional verification function
		verify func(*testing.T, []testclient.Action)
	}{
		{
			name:       "should sign if signerName is kubernetes.io/kube-apiserver-client",
			signerName: "kubernetes.io/kube-apiserver-client",
			commonName: "hello-world",
			org:        []string{"some-org"},
			usages:     []capi.KeyUsage{capi.UsageClientAuth, capi.UsageDigitalSignature, capi.UsageKeyEncipherment},
			approved:   true,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 1 {
					t.Errorf("expected one Update action but got %d", len(as))
					return
				}
				csr := as[0].(testclient.UpdateAction).GetObject().(*capi.CertificateSigningRequest)
				if len(csr.Status.Certificate) == 0 {
					t.Errorf("expected certificate to be issued but it was not")
				}
			},
		},
		{
			name:       "should refuse to sign if signerName is kubernetes.io/kube-apiserver-client and contains an unexpected usage",
			signerName: "kubernetes.io/kube-apiserver-client",
			commonName: "hello-world",
			org:        []string{"some-org"},
			usages:     []capi.KeyUsage{capi.UsageServerAuth, capi.UsageClientAuth, capi.UsageDigitalSignature, capi.UsageKeyEncipherment},
			approved:   true,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no Update action but got %d", len(as))
					return
				}
			},
		},
		{
			name:       "should sign if signerName is kubernetes.io/kube-apiserver-client-kubelet",
			signerName: "kubernetes.io/kube-apiserver-client-kubelet",
			commonName: "system:node:hello-world",
			org:        []string{"system:nodes"},
			usages:     []capi.KeyUsage{capi.UsageClientAuth, capi.UsageDigitalSignature, capi.UsageKeyEncipherment},
			approved:   true,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 1 {
					t.Errorf("expected one Update action but got %d", len(as))
					return
				}
				csr := as[0].(testclient.UpdateAction).GetObject().(*capi.CertificateSigningRequest)
				if len(csr.Status.Certificate) == 0 {
					t.Errorf("expected certificate to be issued but it was not")
				}
			},
		},
		{
			name:       "should sign if signerName is kubernetes.io/legacy-unknown",
			signerName: "kubernetes.io/legacy-unknown",
			approved:   true,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 1 {
					t.Errorf("expected one Update action but got %d", len(as))
					return
				}
				csr := as[0].(testclient.UpdateAction).GetObject().(*capi.CertificateSigningRequest)
				if len(csr.Status.Certificate) == 0 {
					t.Errorf("expected certificate to be issued but it was not")
				}
			},
		},
		{
			name:       "should sign if signerName is kubernetes.io/kubelet-serving",
			signerName: "kubernetes.io/kubelet-serving",
			commonName: "system:node:testnode",
			org:        []string{"system:nodes"},
			usages:     []capi.KeyUsage{capi.UsageServerAuth, capi.UsageDigitalSignature, capi.UsageKeyEncipherment},
			dnsNames:   []string{"example.com"},
			approved:   true,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 1 {
					t.Errorf("expected one Update action but got %d", len(as))
					return
				}
				csr := as[0].(testclient.UpdateAction).GetObject().(*capi.CertificateSigningRequest)
				if len(csr.Status.Certificate) == 0 {
					t.Errorf("expected certificate to be issued but it was not")
				}
			},
		},
		{
			name:            "should do nothing if an unrecognised signerName is used",
			signerName:      "kubernetes.io/not-recognised",
			constructionErr: true,
			approved:        true,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		{
			name:       "should do nothing if not approved",
			signerName: "kubernetes.io/kubelet-serving",
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		{
			name:            "should do nothing if signerName does not start with kubernetes.io",
			signerName:      "example.com/sample-name",
			constructionErr: true,
			approved:        true,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		{
			name:            "should do nothing if signerName starts with kubernetes.io but is unrecognised",
			signerName:      "kubernetes.io/not-a-real-signer",
			constructionErr: true,
			approved:        true,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			client := &fake.Clientset{}
			s, err := newSigner(c.signerName, "./testdata/ca.crt", "./testdata/ca.key", client, 1*time.Hour)
			switch {
			case c.constructionErr && err != nil:
				return
			case c.constructionErr && err == nil:
				t.Fatalf("expected failure during construction of controller")
			case !c.constructionErr && err != nil:
				t.Fatalf("failed to create signer: %v", err)

			case !c.constructionErr && err == nil:
				// continue with rest of test
			}

			csr := makeTestCSR(csrBuilder{cn: c.commonName, signerName: c.signerName, approved: c.approved, usages: c.usages, org: c.org, dnsNames: c.dnsNames})
			if err := s.handle(csr); err != nil && !c.err {
				t.Errorf("unexpected err: %v", err)
			}
			c.verify(t, client.Actions())
		})
	}
}

// noncryptographic for faster testing
// DO NOT COPY THIS CODE
var insecureRand = rand.New(rand.NewSource(0))

type csrBuilder struct {
	cn         string
	dnsNames   []string
	org        []string
	signerName string
	approved   bool
	usages     []capi.KeyUsage
}

func makeTestCSR(b csrBuilder) *capi.CertificateSigningRequest {
	pk, err := ecdsa.GenerateKey(elliptic.P256(), insecureRand)
	if err != nil {
		panic(err)
	}
	csrb, err := x509.CreateCertificateRequest(insecureRand, &x509.CertificateRequest{
		Subject: pkix.Name{
			CommonName:   b.cn,
			Organization: b.org,
		},
		DNSNames: b.dnsNames,
	}, pk)
	if err != nil {
		panic(err)
	}
	csr := &capi.CertificateSigningRequest{
		Spec: capi.CertificateSigningRequestSpec{
			Request: pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE REQUEST", Bytes: csrb}),
			Usages:  b.usages,
		},
	}
	if b.signerName != "" {
		csr.Spec.SignerName = &b.signerName
	}
	if b.approved {
		csr.Status.Conditions = append(csr.Status.Conditions, capi.CertificateSigningRequestCondition{
			Type: capi.CertificateApproved,
		})
	}
	return csr
}
