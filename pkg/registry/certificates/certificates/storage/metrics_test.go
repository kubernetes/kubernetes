/*
Copyright 2021 The Kubernetes Authors.

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

package storage

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/certificate/csr"
	"k8s.io/component-base/metrics"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/utils/pointer"
)

func Test_countCSRDurationMetric(t *testing.T) {
	caPrivateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	caCert, err := certutil.NewSelfSignedCACert(certutil.Config{CommonName: "test-ca"}, caPrivateKey)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name                       string
		success                    bool
		obj, old                   runtime.Object
		options                    *metav1.UpdateOptions
		wantSigner                 string
		wantRequested, wantHonored bool
	}{
		{
			name:    "cert parse failure",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: []byte("junk"),
				},
			},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "fancy",
					ExpirationSeconds: pointer.Int32(77),
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "other",
			wantRequested: true,
			wantHonored:   false,
		},
		{
			name:    "kube signer honors duration exactly",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: createCert(t, time.Hour, caPrivateKey, caCert),
				},
			},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "kubernetes.io/educate-dolphins",
					ExpirationSeconds: csr.DurationToExpirationSeconds(time.Hour),
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "kubernetes.io/educate-dolphins",
			wantRequested: true,
			wantHonored:   true,
		},
		{
			name:    "signer honors duration exactly",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: createCert(t, time.Hour, caPrivateKey, caCert),
				},
			},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "pandas",
					ExpirationSeconds: csr.DurationToExpirationSeconds(time.Hour),
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "other",
			wantRequested: true,
			wantHonored:   true,
		},
		{
			name:    "signer honors duration but just a little bit less",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: createCert(t, time.Hour-6*time.Minute, caPrivateKey, caCert),
				},
			},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "pandas",
					ExpirationSeconds: csr.DurationToExpirationSeconds(time.Hour),
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "other",
			wantRequested: true,
			wantHonored:   true,
		},
		{
			name:    "signer honors duration but just a little bit more",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: createCert(t, time.Hour+6*time.Minute, caPrivateKey, caCert),
				},
			},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "pandas",
					ExpirationSeconds: csr.DurationToExpirationSeconds(time.Hour),
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "other",
			wantRequested: true,
			wantHonored:   true,
		},
		{
			name:    "honors duration lower bound",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: createCert(t, 651*time.Second, caPrivateKey, caCert),
				},
			},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "kubernetes.io/educate-dolphins",
					ExpirationSeconds: csr.DurationToExpirationSeconds(1_000 * time.Second),
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "kubernetes.io/educate-dolphins",
			wantRequested: true,
			wantHonored:   true,
		},
		{
			name:    "does not honor duration just outside of lower bound",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: createCert(t, 650*time.Second, caPrivateKey, caCert),
				},
			},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "kubernetes.io/educate-dolphins",
					ExpirationSeconds: csr.DurationToExpirationSeconds(1_000 * time.Second),
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "kubernetes.io/educate-dolphins",
			wantRequested: true,
			wantHonored:   false,
		},
		{
			name:    "honors duration upper bound",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: createCert(t, 1349*time.Second, caPrivateKey, caCert),
				},
			},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "kubernetes.io/educate-dolphins",
					ExpirationSeconds: csr.DurationToExpirationSeconds(1_000 * time.Second),
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "kubernetes.io/educate-dolphins",
			wantRequested: true,
			wantHonored:   true,
		},
		{
			name:    "does not honor duration just outside of upper bound",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: createCert(t, 1350*time.Second, caPrivateKey, caCert),
				},
			},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "kubernetes.io/educate-dolphins",
					ExpirationSeconds: csr.DurationToExpirationSeconds(1_000 * time.Second),
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "kubernetes.io/educate-dolphins",
			wantRequested: true,
			wantHonored:   false,
		},
		{
			name:    "failed update is ignored",
			success: false,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: createCert(t, time.Hour, caPrivateKey, caCert),
				},
			},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "pandas",
					ExpirationSeconds: csr.DurationToExpirationSeconds(time.Hour),
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "",
			wantRequested: false,
			wantHonored:   false,
		},
		{
			name:    "dry run is ignored",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: createCert(t, time.Hour, caPrivateKey, caCert),
				},
			},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "pandas",
					ExpirationSeconds: csr.DurationToExpirationSeconds(time.Hour),
				},
			},
			options:       &metav1.UpdateOptions{DryRun: []string{"stuff"}},
			wantSigner:    "",
			wantRequested: false,
			wantHonored:   false,
		},
		{
			name:    "old CSR already has a cert so it is ignored",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: createCert(t, time.Hour, caPrivateKey, caCert),
				},
			},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "pandas",
					ExpirationSeconds: csr.DurationToExpirationSeconds(time.Hour),
				},
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: []byte("junk"),
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "",
			wantRequested: false,
			wantHonored:   false,
		},
		{
			name:    "CSRs with no duration are ignored",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: createCert(t, time.Hour, caPrivateKey, caCert),
				},
			},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "pandas",
					ExpirationSeconds: nil,
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "",
			wantRequested: false,
			wantHonored:   false,
		},
		{
			name:    "unissued CSRs are ignored",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: nil,
				},
			},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "pandas",
					ExpirationSeconds: csr.DurationToExpirationSeconds(time.Hour),
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "",
			wantRequested: false,
			wantHonored:   false,
		},
		{
			name:    "invalid data - nil old object",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: createCert(t, time.Hour, caPrivateKey, caCert),
				},
			},
			old:           nil,
			options:       &metav1.UpdateOptions{},
			wantSigner:    "",
			wantRequested: false,
			wantHonored:   false,
		},
		{
			name:    "invalid data - nil new object",
			success: true,
			obj:     nil,
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "pandas",
					ExpirationSeconds: csr.DurationToExpirationSeconds(time.Hour),
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "",
			wantRequested: false,
			wantHonored:   false,
		},
		{
			name:    "invalid data - junk old object",
			success: true,
			obj: &certificates.CertificateSigningRequest{
				Status: certificates.CertificateSigningRequestStatus{
					Certificate: createCert(t, time.Hour, caPrivateKey, caCert),
				},
			},
			old:           &corev1.Pod{},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "",
			wantRequested: false,
			wantHonored:   false,
		},
		{
			name:    "invalid data - junk new object",
			success: true,
			obj:     &corev1.Pod{},
			old: &certificates.CertificateSigningRequest{
				Spec: certificates.CertificateSigningRequestSpec{
					SignerName:        "pandas",
					ExpirationSeconds: csr.DurationToExpirationSeconds(time.Hour),
				},
			},
			options:       &metav1.UpdateOptions{},
			wantSigner:    "",
			wantRequested: false,
			wantHonored:   false,
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			testReq := &testCounterVecMetric{}
			testHon := &testCounterVecMetric{}

			finishFunc, err := countCSRDurationMetric(testReq, testHon)(nil, tt.obj, tt.old, tt.options)
			if err != nil {
				t.Fatal(err)
			}

			finishFunc(nil, tt.success)

			if got := testReq.signer; tt.wantRequested && tt.wantSigner != got {
				t.Errorf("requested signer: want %v, got %v", tt.wantSigner, got)
			}

			if got := testHon.signer; tt.wantHonored && tt.wantSigner != got {
				t.Errorf("honored signer: want %v, got %v", tt.wantSigner, got)
			}

			if got := testReq.called; tt.wantRequested != got {
				t.Errorf("requested inc: want %v, got %v", tt.wantRequested, got)
			}

			if got := testHon.called; tt.wantHonored != got {
				t.Errorf("honored inc: want %v, got %v", tt.wantHonored, got)
			}
		})
	}
}

func createCert(t *testing.T, duration time.Duration, caPrivateKey *ecdsa.PrivateKey, caCert *x509.Certificate) []byte {
	t.Helper()

	crPublicKey := &caPrivateKey.PublicKey // this is supposed to be public key of the signee but it does not matter for this test

	now := time.Now()
	tmpl := &x509.Certificate{Subject: pkix.Name{CommonName: "panda"}, SerialNumber: big.NewInt(1234), NotBefore: now, NotAfter: now.Add(duration)}

	der, err := x509.CreateCertificate(rand.Reader, tmpl, caCert, crPublicKey, caPrivateKey)
	if err != nil {
		t.Fatal(err)
	}

	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
}

type testCounterVecMetric struct {
	metrics.CounterMetric

	signer string
	called bool
}

func (m *testCounterVecMetric) WithLabelValues(lv ...string) metrics.CounterMetric {
	if len(lv) != 1 {
		panic(lv)
	}

	if len(m.signer) != 0 {
		panic("unexpected multiple WithLabelValues() calls")
	}

	signer := lv[0]

	if len(signer) == 0 {
		panic("invalid empty signer")
	}

	m.signer = signer
	return m
}

func (m *testCounterVecMetric) Inc() {
	if m.called {
		panic("unexpected multiple Inc() calls")
	}

	m.called = true
}
