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
	"crypto/x509/pkix"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	certificates "k8s.io/api/certificates/v1beta1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	watch "k8s.io/apimachinery/pkg/watch"
	certificatesclient "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
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
var apiServerCertData = newCertificateData(
	`-----BEGIN CERTIFICATE-----
MIICRzCCAfGgAwIBAgIJAIydTIADd+yqMA0GCSqGSIb3DQEBCwUAMH4xCzAJBgNV
BAYTAkdCMQ8wDQYDVQQIDAZMb25kb24xDzANBgNVBAcMBkxvbmRvbjEYMBYGA1UE
CgwPR2xvYmFsIFNlY3VyaXR5MRYwFAYDVQQLDA1JVCBEZXBhcnRtZW50MRswGQYD
VQQDDBJ0ZXN0LWNlcnRpZmljYXRlLTIwIBcNMTcwNDI2MjMyNDU4WhgPMjExNzA0
MDIyMzI0NThaMH4xCzAJBgNVBAYTAkdCMQ8wDQYDVQQIDAZMb25kb24xDzANBgNV
BAcMBkxvbmRvbjEYMBYGA1UECgwPR2xvYmFsIFNlY3VyaXR5MRYwFAYDVQQLDA1J
VCBEZXBhcnRtZW50MRswGQYDVQQDDBJ0ZXN0LWNlcnRpZmljYXRlLTIwXDANBgkq
hkiG9w0BAQEFAANLADBIAkEAuiRet28DV68Dk4A8eqCaqgXmymamUEjW/DxvIQqH
3lbhtm8BwSnS9wUAajSLSWiq3fci2RbRgaSPjUrnbOHCLQIDAQABo1AwTjAdBgNV
HQ4EFgQU0vhI4OPGEOqT+VAWwxdhVvcmgdIwHwYDVR0jBBgwFoAU0vhI4OPGEOqT
+VAWwxdhVvcmgdIwDAYDVR0TBAUwAwEB/zANBgkqhkiG9w0BAQsFAANBALNeJGDe
nV5cXbp9W1bC12Tc8nnNXn4ypLE2JTQAvyp51zoZ8hQoSnRVx/VCY55Yu+br8gQZ
+tW+O/PoE7B3tuY=
-----END CERTIFICATE-----`, `-----BEGIN RSA PRIVATE KEY-----
MIIBVgIBADANBgkqhkiG9w0BAQEFAASCAUAwggE8AgEAAkEAuiRet28DV68Dk4A8
eqCaqgXmymamUEjW/DxvIQqH3lbhtm8BwSnS9wUAajSLSWiq3fci2RbRgaSPjUrn
bOHCLQIDAQABAkEArDR1g9IqD3aUImNikDgAngbzqpAokOGyMoxeavzpEaFOgCzi
gi7HF7yHRmZkUt8CzdEvnHSqRjFuaaB0gGA+AQIhAOc8Z1h8ElLRSqaZGgI3jCTp
Izx9HNY//U5NGrXD2+ttAiEAzhOqkqI4+nDab7FpiD7MXI6fO549mEXeVBPvPtsS
OcECIQCIfkpOm+ZBBpO3JXaJynoqK4gGI6ALA/ik6LSUiIlfPQIhAISjd9hlfZME
bDQT1r8Q3Gx+h9LRqQeHgPBQ3F5ylqqBAiBaJ0hkYvrIdWxNlcLqD3065bJpHQ4S
WQkuZUQN1M/Xvg==
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
		Name:             "test_no_rotation",
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
						NotBefore: test.notBefore,
						NotAfter:  test.notAfter,
					},
				},
				template: &x509.CertificateRequest{},
				usages:   []certificates.KeyUsage{},
				certificateExpiration: prometheus.NewGauge(
					prometheus.GaugeOpts{
						Name: "test_gauge_name",
					},
				),
			}
			m.setRotationDeadline()
			if m.shouldRotate() != test.shouldRotate {
				t.Errorf("Time %v, a certificate issued for (%v, %v) should rotate should be %t.",
					now,
					m.cert.Leaf.NotBefore,
					m.cert.Leaf.NotAfter,
					test.shouldRotate)
			}
		})
	}
}

func TestSetRotationDeadline(t *testing.T) {
	now := time.Now()
	testCases := []struct {
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
		{"long duration", now.Add(-6 * 30 * 24 * time.Hour), now.Add(6 * 30 * 24 * time.Hour), true},
		{"short duration", now.Add(-30 * time.Second), now.Add(30 * time.Second), true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			m := manager{
				cert: &tls.Certificate{
					Leaf: &x509.Certificate{
						NotBefore: tc.notBefore,
						NotAfter:  tc.notAfter,
					},
				},
				template: &x509.CertificateRequest{},
				usages:   []certificates.KeyUsage{},
				certificateExpiration: prometheus.NewGauge(
					prometheus.GaugeOpts{
						Name: "test_gauge_name",
					},
				),
			}
			lowerBound := tc.notBefore.Add(time.Duration(float64(tc.notAfter.Sub(tc.notBefore)) * 0.7))
			upperBound := tc.notBefore.Add(time.Duration(float64(tc.notAfter.Sub(tc.notBefore)) * 0.9))
			for i := 0; i < 1000; i++ {
				// setRotationDeadline includes jitter, so this needs to run many times for validation.
				m.setRotationDeadline()
				if m.rotationDeadline.Before(lowerBound) || m.rotationDeadline.After(upperBound) {
					t.Errorf("For notBefore %v, notAfter %v, the rotationDeadline %v should be between %v and %v.",
						tc.notBefore,
						tc.notAfter,
						m.rotationDeadline,
						lowerBound,
						upperBound)
				}
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

	if success, err := m.rotateCerts(); success {
		t.Errorf("Got success from 'rotateCerts', wanted failure")
	} else if err != nil {
		t.Errorf("Got error %v from 'rotateCerts', wanted no error.", err)
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

	if success, err := m.rotateCerts(); success {
		t.Errorf("Got success from 'rotateCerts', wanted failure.")
	} else if err != nil {
		t.Errorf("Got error %v from 'rotateCerts', wanted no error.", err)
	}
}

func TestNewManagerBootstrap(t *testing.T) {
	store := &fakeStore{}

	var cm Manager
	cm, err := NewManager(&Config{
		Name:                    "test_bootstrap",
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
		Name:                    "test_no_bootstrap",
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
	} else {
		m.setRotationDeadline()
		if m.shouldRotate() {
			t.Errorf("Expected rotation should happen during bootstrap, but it won't.")
		}
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
			true,
			"",
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
			if certResult == nil || certResult.Certificate == nil || tc.expectedCert == nil {
				if certResult != nil && tc.expectedCert != nil {
					t.Errorf("Got certificate %v, wanted %v", certResult, tc.expectedCert)
				}
			} else {
				if !certificatesEqual(certResult, tc.expectedCert) {
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

func TestInitializeCertificateSigningRequestClient(t *testing.T) {
	var nilCertificate = &certificateData{}
	testCases := []struct {
		description             string
		storeCert               *certificateData
		bootstrapCert           *certificateData
		apiCert                 *certificateData
		expectedCertBeforeStart *certificateData
		expectedCertAfterStart  *certificateData
	}{
		{
			description:             "No current certificate, no bootstrap certificate",
			storeCert:               nilCertificate,
			bootstrapCert:           nilCertificate,
			apiCert:                 apiServerCertData,
			expectedCertBeforeStart: nilCertificate,
			expectedCertAfterStart:  apiServerCertData,
		},
		{
			description:             "No current certificate, bootstrap certificate",
			storeCert:               nilCertificate,
			bootstrapCert:           bootstrapCertData,
			apiCert:                 apiServerCertData,
			expectedCertBeforeStart: bootstrapCertData,
			expectedCertAfterStart:  apiServerCertData,
		},
		{
			description:             "Current certificate, no bootstrap certificate",
			storeCert:               storeCertData,
			bootstrapCert:           nilCertificate,
			apiCert:                 apiServerCertData,
			expectedCertBeforeStart: storeCertData,
			expectedCertAfterStart:  storeCertData,
		},
		{
			description:             "Current certificate, bootstrap certificate",
			storeCert:               storeCertData,
			bootstrapCert:           bootstrapCertData,
			apiCert:                 apiServerCertData,
			expectedCertBeforeStart: storeCertData,
			expectedCertAfterStart:  storeCertData,
		},
	}

	for i, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			certificateStore := &fakeStore{
				cert: tc.storeCert.certificate,
			}

			certificateManager, err := NewManager(&Config{
				Name: fmt.Sprintf("test_initialize_client_%d", i),
				Template: &x509.CertificateRequest{
					Subject: pkix.Name{
						Organization: []string{"system:nodes"},
						CommonName:   "system:node:fake-node-name",
					},
				},
				Usages: []certificates.KeyUsage{
					certificates.UsageDigitalSignature,
					certificates.UsageKeyEncipherment,
					certificates.UsageClientAuth,
				},
				CertificateStore:        certificateStore,
				BootstrapCertificatePEM: tc.bootstrapCert.certificatePEM,
				BootstrapKeyPEM:         tc.bootstrapCert.keyPEM,
			})
			if err != nil {
				t.Errorf("Got %v, wanted no error.", err)
			}

			certificate := certificateManager.Current()
			if !certificatesEqual(certificate, tc.expectedCertBeforeStart.certificate) {
				t.Errorf("Got %v, wanted %v", certificateString(certificate), certificateString(tc.expectedCertBeforeStart.certificate))
			}
			if err := certificateManager.SetCertificateSigningRequestClient(&fakeClient{
				certificatePEM: tc.apiCert.certificatePEM,
			}); err != nil {
				t.Errorf("Got error %v, expected none.", err)
			}

			if m, ok := certificateManager.(*manager); !ok {
				t.Errorf("Expected a '*manager' from 'NewManager'")
			} else {
				m.setRotationDeadline()
				if m.shouldRotate() {
					if success, err := m.rotateCerts(); !success {
						t.Errorf("Got failure from 'rotateCerts', wanted success.")
					} else if err != nil {
						t.Errorf("Got error %v, expected none.", err)
					}
				}
			}

			certificate = certificateManager.Current()
			if !certificatesEqual(certificate, tc.expectedCertAfterStart.certificate) {
				t.Errorf("Got %v, wanted %v", certificateString(certificate), certificateString(tc.expectedCertAfterStart.certificate))
			}
		})
	}
}

func TestInitializeOtherRESTClients(t *testing.T) {
	var nilCertificate = &certificateData{}
	testCases := []struct {
		description             string
		storeCert               *certificateData
		bootstrapCert           *certificateData
		apiCert                 *certificateData
		expectedCertBeforeStart *certificateData
		expectedCertAfterStart  *certificateData
	}{
		{
			description:             "No current certificate, no bootstrap certificate",
			storeCert:               nilCertificate,
			bootstrapCert:           nilCertificate,
			apiCert:                 apiServerCertData,
			expectedCertBeforeStart: nilCertificate,
			expectedCertAfterStart:  apiServerCertData,
		},
		{
			description:             "No current certificate, bootstrap certificate",
			storeCert:               nilCertificate,
			bootstrapCert:           bootstrapCertData,
			apiCert:                 apiServerCertData,
			expectedCertBeforeStart: bootstrapCertData,
			expectedCertAfterStart:  apiServerCertData,
		},
		{
			description:             "Current certificate, no bootstrap certificate",
			storeCert:               storeCertData,
			bootstrapCert:           nilCertificate,
			apiCert:                 apiServerCertData,
			expectedCertBeforeStart: storeCertData,
			expectedCertAfterStart:  storeCertData,
		},
		{
			description:             "Current certificate, bootstrap certificate",
			storeCert:               storeCertData,
			bootstrapCert:           bootstrapCertData,
			apiCert:                 apiServerCertData,
			expectedCertBeforeStart: storeCertData,
			expectedCertAfterStart:  storeCertData,
		},
	}

	for i, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			certificateStore := &fakeStore{
				cert: tc.storeCert.certificate,
			}

			certificateManager, err := NewManager(&Config{
				Name: fmt.Sprintf("test_initialize_other_rest_clients_%d", i),
				Template: &x509.CertificateRequest{
					Subject: pkix.Name{
						Organization: []string{"system:nodes"},
						CommonName:   "system:node:fake-node-name",
					},
				},
				Usages: []certificates.KeyUsage{
					certificates.UsageDigitalSignature,
					certificates.UsageKeyEncipherment,
					certificates.UsageClientAuth,
				},
				CertificateStore:        certificateStore,
				BootstrapCertificatePEM: tc.bootstrapCert.certificatePEM,
				BootstrapKeyPEM:         tc.bootstrapCert.keyPEM,
				CertificateSigningRequestClient: &fakeClient{
					certificatePEM: tc.apiCert.certificatePEM,
				},
			})
			if err != nil {
				t.Errorf("Got %v, wanted no error.", err)
			}

			certificate := certificateManager.Current()
			if !certificatesEqual(certificate, tc.expectedCertBeforeStart.certificate) {
				t.Errorf("Got %v, wanted %v", certificateString(certificate), certificateString(tc.expectedCertBeforeStart.certificate))
			}

			if m, ok := certificateManager.(*manager); !ok {
				t.Errorf("Expected a '*manager' from 'NewManager'")
			} else {
				m.setRotationDeadline()
				if m.shouldRotate() {
					if success, err := certificateManager.(*manager).rotateCerts(); !success {
						t.Errorf("Got failure from 'rotateCerts', expected success")
					} else if err != nil {
						t.Errorf("Got error %v, expected none.", err)
					}
				}
			}

			certificate = certificateManager.Current()
			if !certificatesEqual(certificate, tc.expectedCertAfterStart.certificate) {
				t.Errorf("Got %v, wanted %v", certificateString(certificate), certificateString(tc.expectedCertAfterStart.certificate))
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
	failureType    fakeClientFailureType
	certificatePEM []byte
}

func (c fakeClient) Create(*certificates.CertificateSigningRequest) (*certificates.CertificateSigningRequest, error) {
	if c.failureType == createError {
		return nil, fmt.Errorf("Create error")
	}
	csrReply := certificates.CertificateSigningRequest{}
	csrReply.UID = "fake-uid"
	return &csrReply, nil
}

func (c fakeClient) Watch(opts v1.ListOptions) (watch.Interface, error) {
	if c.failureType == watchError {
		return nil, fmt.Errorf("Watch error")
	}
	return &fakeWatch{
		failureType:    c.failureType,
		certificatePEM: c.certificatePEM,
	}, nil
}

type fakeWatch struct {
	failureType    fakeClientFailureType
	certificatePEM []byte
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
			Certificate: []byte(w.certificatePEM),
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
	// In order to make the mocking work, whenever a cert/key pair is passed in
	// to be updated in the mock store, assume that the certificate manager
	// generated the key, and then asked the mock CertificateSigningRequest API
	// to sign it, then the faked API returned a canned response. The canned
	// signing response will not match the generated key. In order to make
	// things work out, search here for the correct matching key and use that
	// instead of the passed in key. That way this file of test code doesn't
	// have to implement an actual certificate signing process.
	for _, tc := range []*certificateData{storeCertData, bootstrapCertData, apiServerCertData} {
		if bytes.Equal(tc.certificatePEM, certPEM) {
			keyPEM = tc.keyPEM
		}
	}
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

func certificatesEqual(c1 *tls.Certificate, c2 *tls.Certificate) bool {
	if c1 == nil || c2 == nil {
		return c1 == c2
	}
	if len(c1.Certificate) != len(c2.Certificate) {
		return false
	}
	for i := 0; i < len(c1.Certificate); i++ {
		if !bytes.Equal(c1.Certificate[i], c2.Certificate[i]) {
			return false
		}
	}
	return true
}

func certificateString(c *tls.Certificate) string {
	if c == nil {
		return "certificate == nil"
	}
	if c.Leaf == nil {
		return "certificate.Leaf == nil"
	}
	return c.Leaf.Subject.CommonName
}
