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

package certificate

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	watch "k8s.io/apimachinery/pkg/watch"
	certificates "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
	certificatesclient "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/certificates/v1beta1"
)

type certificateData struct {
	keyPEM         []byte
	certificatePEM []byte
	certificate    *tls.Certificate
}

var storeCertData = newCertificateData(`-----BEGIN CERTIFICATE-----
MIICRzCCAfGgAwIBAgIJALMb7ecMIk3MMA0GCSqGSIb3DQEBCwUAMH4xCzAJBgNV
BAYTAkdCMQ8wDQYDVQQIDAZMb25kb24xDzANBgNVBAcMBkxvbmRvbjEYMBYGA1UE
CgwPR2xvYmFsIFNlY3VyaXR5MRYwFAYDVQQLDA1JVCBEZXBhcnRtZW50MRswGQYD
VQQDDBJ0ZXN0LWNlcnRpZmljYXRlLTAwIBcNMTcwNDI2MjMyNjUyWhgPMjExNzA0
MDIyMzI2NTJaMH4xCzAJBgNVBAYTAkdCMQ8wDQYDVQQIDAZMb25kb24xDzANBgNV
BAcMBkxvbmRvbjEYMBYGA1UECgwPR2xvYmFsIFNlY3VyaXR5MRYwFAYDVQQLDA1J
VCBEZXBhcnRtZW50MRswGQYDVQQDDBJ0ZXN0LWNlcnRpZmljYXRlLTAwXDANBgkq
hkiG9w0BAQEFAANLADBIAkEAtBMa7NWpv3BVlKTCPGO/LEsguKqWHBtKzweMY2CV
tAL1rQm913huhxF9w+ai76KQ3MHK5IVnLJjYYA5MzP2H5QIDAQABo1AwTjAdBgNV
HQ4EFgQU22iy8aWkNSxv0nBxFxerfsvnZVMwHwYDVR0jBBgwFoAU22iy8aWkNSxv
0nBxFxerfsvnZVMwDAYDVR0TBAUwAwEB/zANBgkqhkiG9w0BAQsFAANBAEOefGbV
NcHxklaW06w6OBYJPwpIhCVozC1qdxGX1dg8VkEKzjOzjgqVD30m59OFmSlBmHsl
nkVA6wyOSDYBf3o=
-----END CERTIFICATE-----`, `-----BEGIN RSA PRIVATE KEY-----
MIIBUwIBADANBgkqhkiG9w0BAQEFAASCAT0wggE5AgEAAkEAtBMa7NWpv3BVlKTC
PGO/LEsguKqWHBtKzweMY2CVtAL1rQm913huhxF9w+ai76KQ3MHK5IVnLJjYYA5M
zP2H5QIDAQABAkAS9BfXab3OKpK3bIgNNyp+DQJKrZnTJ4Q+OjsqkpXvNltPJosf
G8GsiKu/vAt4HGqI3eU77NvRI+mL4MnHRmXBAiEA3qM4FAtKSRBbcJzPxxLEUSwg
XSCcosCktbkXvpYrS30CIQDPDxgqlwDEJQ0uKuHkZI38/SPWWqfUmkecwlbpXABK
iQIgZX08DA8VfvcA5/Xj1Zjdey9FVY6POLXen6RPiabE97UCICp6eUW7ht+2jjar
e35EltCRCjoejRHTuN9TC0uCoVipAiAXaJIx/Q47vGwiw6Y8KXsNU6y54gTbOSxX
54LzHNk/+Q==
-----END RSA PRIVATE KEY-----`)
var bootstrapCertData = newCertificateData(
	`-----BEGIN CERTIFICATE-----
MIICRzCCAfGgAwIBAgIJANXr+UzRFq4TMA0GCSqGSIb3DQEBCwUAMH4xCzAJBgNV
BAYTAkdCMQ8wDQYDVQQIDAZMb25kb24xDzANBgNVBAcMBkxvbmRvbjEYMBYGA1UE
CgwPR2xvYmFsIFNlY3VyaXR5MRYwFAYDVQQLDA1JVCBEZXBhcnRtZW50MRswGQYD
VQQDDBJ0ZXN0LWNlcnRpZmljYXRlLTEwIBcNMTcwNDI2MjMyNzMyWhgPMjExNzA0
MDIyMzI3MzJaMH4xCzAJBgNVBAYTAkdCMQ8wDQYDVQQIDAZMb25kb24xDzANBgNV
BAcMBkxvbmRvbjEYMBYGA1UECgwPR2xvYmFsIFNlY3VyaXR5MRYwFAYDVQQLDA1J
VCBEZXBhcnRtZW50MRswGQYDVQQDDBJ0ZXN0LWNlcnRpZmljYXRlLTEwXDANBgkq
hkiG9w0BAQEFAANLADBIAkEAqvbkN4RShH1rL37JFp4fZPnn0JUhVWWsrP8NOomJ
pXdBDUMGWuEQIsZ1Gf9JrCQLu6ooRyHSKRFpAVbMQ3ABJwIDAQABo1AwTjAdBgNV
HQ4EFgQUEGBc6YYheEZ/5MhwqSUYYPYRj2MwHwYDVR0jBBgwFoAUEGBc6YYheEZ/
5MhwqSUYYPYRj2MwDAYDVR0TBAUwAwEB/zANBgkqhkiG9w0BAQsFAANBAIyNmznk
5dgJY52FppEEcfQRdS5k4XFPc22SHPcz77AHf5oWZ1WG9VezOZZPp8NCiFDDlDL8
yma33a5eMyTjLD8=
-----END CERTIFICATE-----`, `-----BEGIN RSA PRIVATE KEY-----
MIIBVAIBADANBgkqhkiG9w0BAQEFAASCAT4wggE6AgEAAkEAqvbkN4RShH1rL37J
Fp4fZPnn0JUhVWWsrP8NOomJpXdBDUMGWuEQIsZ1Gf9JrCQLu6ooRyHSKRFpAVbM
Q3ABJwIDAQABAkBC2OBpGLMPHN8BJijIUDFkURakBvuOoX+/8MYiYk7QxEmfLCk6
L6r+GLNFMfXwXcBmXtMKfZKAIKutKf098JaBAiEA10azfqt3G/5owrNA00plSyT6
ZmHPzY9Uq1p/QTR/uOcCIQDLTkfBkLHm0UKeobbO/fSm6ZflhyBRDINy4FvwmZMt
wQIgYV/tmQJeIh91q3wBepFQOClFykG8CTMoDUol/YyNqUkCIHfp6Rr7fGL3JIMq
QQgf9DCK8SPZqq8DYXjdan0kKBJBAiEAyDb+07o2gpggo8BYUKSaiRCiyXfaq87f
eVqgpBq/QN4=
-----END RSA PRIVATE KEY-----`)

func newCertificateData(certificatePEM string, keyPEM string) *certificateData {
	certificate, err := tls.X509KeyPair([]byte(certificatePEM), []byte(keyPEM))
	if err != nil {
		panic(fmt.Sprintf("Unable to initialize certificate: %v", err))
	}
	certs, err := x509.ParseCertificates(certificate.Certificate[0])
	if err != nil {
		panic(fmt.Sprintf("Unable to initialize certificate leaf: %v", err))
	}
	certificate.Leaf = certs[0]
	return &certificateData{
		keyPEM:         []byte(keyPEM),
		certificatePEM: []byte(certificatePEM),
		certificate:    &certificate,
	}
}

func TestNewManagerNoRotation(t *testing.T) {
	store := &fakeStore{
		cert: storeCertData.certificate,
	}
	if _, err := NewManager(&Config{
		Template:         &x509.CertificateRequest{},
		Usages:           []certificates.KeyUsage{},
		CertificateStore: store,
	}); err != nil {
		t.Fatalf("Failed to initialize the certificate manager: %v", err)
	}
}

func TestShouldRotate(t *testing.T) {
	now := time.Now()
	tests := []struct {
		name         string
		notBefore    time.Time
		notAfter     time.Time
		shouldRotate bool
	}{
		{"just issued, still good", now.Add(-1 * time.Hour), now.Add(99 * time.Hour), false},
		{"half way expired, still good", now.Add(-24 * time.Hour), now.Add(24 * time.Hour), false},
		{"mostly expired, still good", now.Add(-69 * time.Hour), now.Add(31 * time.Hour), false},
		{"just about expired, should rotate", now.Add(-91 * time.Hour), now.Add(9 * time.Hour), true},
		{"nearly expired, should rotate", now.Add(-99 * time.Hour), now.Add(1 * time.Hour), true},
		{"already expired, should rotate", now.Add(-10 * time.Hour), now.Add(-1 * time.Hour), true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			m := manager{
				cert: &tls.Certificate{
					Leaf: &x509.Certificate{
						NotAfter:  test.notAfter,
						NotBefore: test.notBefore,
					},
				},
				template: &x509.CertificateRequest{},
				usages:   []certificates.KeyUsage{},
			}
			if m.shouldRotate() != test.shouldRotate {
				t.Errorf("For time %v, a certificate issued for (%v, %v) should rotate should be %t.",
					now,
					m.cert.Leaf.NotBefore,
					m.cert.Leaf.NotAfter,
					test.shouldRotate)
			}
		})
	}
}

func TestRotateCertCreateCSRError(t *testing.T) {
	now := time.Now()
	m := manager{
		cert: &tls.Certificate{
			Leaf: &x509.Certificate{
				NotBefore: now.Add(-2 * time.Hour),
				NotAfter:  now.Add(-1 * time.Hour),
			},
		},
		template: &x509.CertificateRequest{},
		usages:   []certificates.KeyUsage{},
		certSigningRequestClient: fakeClient{
			failureType: createError,
		},
	}

	if err := m.rotateCerts(); err == nil {
		t.Errorf("Expected an error from 'rotateCerts'.")
	}
}

func TestRotateCertWaitingForResultError(t *testing.T) {
	now := time.Now()
	m := manager{
		cert: &tls.Certificate{
			Leaf: &x509.Certificate{
				NotBefore: now.Add(-2 * time.Hour),
				NotAfter:  now.Add(-1 * time.Hour),
			},
		},
		template: &x509.CertificateRequest{},
		usages:   []certificates.KeyUsage{},
		certSigningRequestClient: fakeClient{
			failureType: watchError,
		},
	}

	if err := m.rotateCerts(); err == nil {
		t.Errorf("Expected an error receiving results from the CSR request but nothing was received.")
	}
}

func TestNewManagerBootstrap(t *testing.T) {
	store := &fakeStore{}

	cm, err := NewManager(&Config{
		Template:                &x509.CertificateRequest{},
		Usages:                  []certificates.KeyUsage{},
		CertificateStore:        store,
		BootstrapCertificatePEM: bootstrapCertData.certificatePEM,
		BootstrapKeyPEM:         bootstrapCertData.keyPEM,
	})

	if err != nil {
		t.Fatalf("Failed to initialize the certificate manager: %v", err)
	}

	cert := cm.Current()

	if cert == nil {
		t.Errorf("Certificate was nil, expected something.")
	}

	if m, ok := cm.(*manager); !ok {
		t.Errorf("Expected a '*manager' from 'NewManager'")
	} else if !m.shouldRotate() {
		t.Errorf("Expected rotation should happen during bootstrap, but it won't.")
	}
}

func TestNewManagerNoBootstrap(t *testing.T) {
	now := time.Now()
	cert, err := tls.X509KeyPair(storeCertData.certificatePEM, storeCertData.keyPEM)
	if err != nil {
		t.Fatalf("Unable to initialize a certificate: %v", err)
	}
	cert.Leaf = &x509.Certificate{
		NotBefore: now.Add(-24 * time.Hour),
		NotAfter:  now.Add(24 * time.Hour),
	}
	store := &fakeStore{
		cert: &cert,
	}

	cm, err := NewManager(&Config{
		Template:                &x509.CertificateRequest{},
		Usages:                  []certificates.KeyUsage{},
		CertificateStore:        store,
		BootstrapCertificatePEM: bootstrapCertData.certificatePEM,
		BootstrapKeyPEM:         bootstrapCertData.keyPEM,
	})

	if err != nil {
		t.Fatalf("Failed to initialize the certificate manager: %v", err)
	}

	currentCert := cm.Current()

	if currentCert == nil {
		t.Errorf("Certificate was nil, expected something.")
	}

	if m, ok := cm.(*manager); !ok {
		t.Errorf("Expected a '*manager' from 'NewManager'")
	} else if m.shouldRotate() {
		t.Errorf("Expected rotation should happen during bootstrap, but it won't.")
	}
}

func TestGetCurrentCertificateOrBootstrap(t *testing.T) {
	testCases := []struct {
		description          string
		storeCert            *tls.Certificate
		bootstrapCertData    []byte
		bootstrapKeyData     []byte
		expectedCert         *tls.Certificate
		expectedShouldRotate bool
		expectedErrMsg       string
	}{
		{
			"return cert from store",
			storeCertData.certificate,
			nil,
			nil,
			storeCertData.certificate,
			false,
			"",
		},
		{
			"no cert in store and no bootstrap cert",
			nil,
			nil,
			nil,
			nil,
			false,
			"no cert/key available and no bootstrap cert/key to fall back to",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			store := &fakeStore{
				cert: tc.storeCert,
			}

			certResult, shouldRotate, err := getCurrentCertificateOrBootstrap(
				store,
				tc.bootstrapCertData,
				tc.bootstrapKeyData)
			if certResult == nil || tc.expectedCert == nil {
				if certResult != tc.expectedCert {
					t.Errorf("Got certificate %v, wanted %v", certResult, tc.expectedCert)
				}
			} else {
				if len(certResult.Certificate) != len(tc.expectedCert.Certificate) {
					t.Errorf("Got %d certificates, wanted %d", len(certResult.Certificate), len(tc.expectedCert.Certificate))
				}
				if !bytes.Equal(certResult.Certificate[0], tc.expectedCert.Certificate[0]) {
					t.Errorf("Got certificate %v, wanted %v", certResult, tc.expectedCert)
				}
			}
			if shouldRotate != tc.expectedShouldRotate {
				t.Errorf("Got shouldRotate %t, wanted %t", shouldRotate, tc.expectedShouldRotate)
			}
			if err == nil {
				if tc.expectedErrMsg != "" {
					t.Errorf("Got err %v, wanted %q", err, tc.expectedErrMsg)
				}
			} else {
				if tc.expectedErrMsg == "" || !strings.Contains(err.Error(), tc.expectedErrMsg) {
					t.Errorf("Got err %v, wanted %q", err, tc.expectedErrMsg)
				}
			}
		})
	}
}

type fakeClientFailureType int

const (
	none fakeClientFailureType = iota
	createError
	watchError
	certificateSigningRequestDenied
)

type fakeClient struct {
	certificatesclient.CertificateSigningRequestInterface
	failureType fakeClientFailureType
}

func (c fakeClient) Create(*certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error) {
	if c.failureType == createError {
		return nil, fmt.Errorf("Create error")
	}
	csr := certificates.CertificateSigningRequest{}
	csr.UID = "fake-uid"
	return &csr, nil
}

func (c fakeClient) Watch(opts v1.ListOptions) (watch.Interface, error) {
	if c.failureType == watchError {
		return nil, fmt.Errorf("Watch error")
	}
	return &fakeWatch{
		failureType: c.failureType,
	}, nil
}

type fakeWatch struct {
	failureType fakeClientFailureType
}

func (w *fakeWatch) Stop() {
}

func (w *fakeWatch) ResultChan() <-chan watch.Event {
	var condition certificates.CertificateSigningRequestCondition
	if w.failureType == certificateSigningRequestDenied {
		condition = certificates.CertificateSigningRequestCondition{
			Type: certificates.CertificateDenied,
		}
	} else {
		condition = certificates.CertificateSigningRequestCondition{
			Type: certificates.CertificateApproved,
		}
	}

	csr := certificates.CertificateSigningRequest{
		Status: certificates.CertificateSigningRequestStatus{
			Conditions: []certificates.CertificateSigningRequestCondition{
				condition,
			},
			Certificate: []byte(storeCertData.certificatePEM),
		},
	}
	csr.UID = "fake-uid"

	c := make(chan watch.Event, 1)
	c <- watch.Event{
		Type:   watch.Added,
		Object: &csr,
	}
	return c
}

type fakeStore struct {
	cert *tls.Certificate
}

func (s *fakeStore) Current() (*tls.Certificate, error) {
	if s.cert == nil {
		noKeyErr := NoCertKeyError("")
		return nil, &noKeyErr
	}
	return s.cert, nil
}

// Accepts the PEM data for the cert/key pair and makes the new cert/key
// pair the 'current' pair, that will be returned by future calls to
// Current().
func (s *fakeStore) Update(certPEM, keyPEM []byte) (*tls.Certificate, error) {
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		return nil, err
	}
	now := time.Now()
	s.cert = &cert
	s.cert.Leaf = &x509.Certificate{
		NotBefore: now.Add(-24 * time.Hour),
		NotAfter:  now.Add(24 * time.Hour),
	}
	return s.cert, nil
}
