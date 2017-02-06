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
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	watch "k8s.io/apimachinery/pkg/watch"
	certificates "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
	certificatesclient "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/certificates/v1beta1"
)

const (
	privateKeyData = `-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA03ppJ1S3xK2UaXIatBPMbstHm8U9fwIFAj3a2WDV6FHo6zi2
YHVwCwSVnHL6D+Q5mmlbhnUpSD8SGTLk4EESAe2h203iBOBPBhymhTWA/gAEFk23
aP1/KlubjYN1+eyksA0lOVcO3sCuRZ64yjYJ369IfV1w8APZ4BXoFtU3uuYpjxyF
XlydkbLqQZLrBa1B5E8hEkDn4ywNDptGjRN3gT2GMQwnaCkWiLjGK6AxTCleXnjG
/JyEwbczv0zAE43utcYPW7qk1m5QsKMUAu4/K8y8oGBFy2ygpY1qckcgr5haehOS
IbFEvVd2oqW8NBicKNmSlh0OcAvQQZtaXhLg/QIDAQABAoIBAFkBmUZLerjVkbQ7
qQ+HkbBD8FSYVESjVfZWkEiTYBRSfSSbDu9UHh8VA97/6U1M8g2SMEpL/17/5J8k
c34LBQg4urmxcuI4gioBXviLx0mgOhglB3+xyZbLTZHm9X2F4t6R+cvDX2fTUsXM
gtvgmJFDlc/lxwXNqSKONct+W+FV/9D2H1Vzf8fQHfa+lltAy8e8MrbmGQTgev+5
vz/UR/bZz/CHRxXVA6txgvf4AL8BYibxgx6ihW9zKHy6GykqtQ2p0T5XCkObt41S
6KwUmIHP8CHY23MJ9BPIxYH2+lOXFLizB1VFuxRE1W+je7wVWxzQgFS4IMOLVYDD
LtprVQUCgYEA4g9ODbyW5vvyp8mmAWAvgeunOR1aP79IIyHiwefEIup4FNo+K2wZ
QhRPf0LsVvnthJXFWeW9arAWZRWKCFWwISq/cIIB6KXCIIsjiTUe8SYE/8bxAkvL
0lJhWugTpOnFd8oVuRivrsIWL+SXTNiO5JOP3/qfo+HFk3dqjDhXg4MCgYEA73y1
Cy+8vHweHKr8HTkPF13GAB1I43SvzTnGT2BT9q6Ia+zQDF1dHjnMrswD1v0+6Xmq
lKc5M69WBVuLIAfWfMQy0WANpsEMm5MYHShJ3YEYAqBiSTUWi23nLH/Poos4IUDV
nTAgFuoKFaG/9cLKA736zqJaiJCE/IR2/gqcYX8CgYA5PCjF/5axWt8ALmTyejjt
Cw4mvtDHzRVll8HC2HxnXrgSh4MwGUl32o6aKQaPqu3BIO57qVhA995jr4VoQNG8
RAd+Y9w53CX/eVsA9UslQTwIyoTg0PIFCUiO7K10lp+hia/gUmjAtXFKpPTNxxK+
usG1ss3Sf2o3wQdgAy/dIwKBgQCcHa1fZ3UfYcG3ancDDckasFR8ipqTO+PGYt01
rVPOwSPJRwywosQrCf62C+SM53V1eYyLbx9I5AmtYGmnLbTSjIucFYOQqtPvLspP
Z44PSTI/tBGeK29Q4QoL5h2SljK26q7V0yN4DIUaaODb8mkCW3v967QcxikK+8ce
AAjFPQKBgHnfVRX+00xSeNE0zya1FtQH3db9+fm3IYGK10NI/jTNF6RhUwHJ6X3+
TR6OhnTQ2j8eAo+6IlLqlDeC1X7GDvaxqstPvGi0lZjoQQGnQqw2m58AMJu3s9fW
2iddptVycNU0+187DIO39cM3o5s0822VUWDbmymD9cW4i8G6Yto9
-----END RSA PRIVATE KEY-----`
	certificateData = `-----BEGIN CERTIFICATE-----
MIIDEzCCAfugAwIBAgIBATANBgkqhkiG9w0BAQsFADAjMSEwHwYDVQQDDBhrLWEt
bm9kZS12YzFzQDE0ODYzMzM1NDgwHhcNMTcwMjA1MjIyNTQ4WhcNMTgwMjA1MjIy
NTQ4WjAjMSEwHwYDVQQDDBhrLWEtbm9kZS12YzFzQDE0ODYzMzM1NDgwggEiMA0G
CSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDTemknVLfErZRpchq0E8xuy0ebxT1/
AgUCPdrZYNXoUejrOLZgdXALBJWccvoP5DmaaVuGdSlIPxIZMuTgQRIB7aHbTeIE
4E8GHKaFNYD+AAQWTbdo/X8qW5uNg3X57KSwDSU5Vw7ewK5FnrjKNgnfr0h9XXDw
A9ngFegW1Te65imPHIVeXJ2RsupBkusFrUHkTyESQOfjLA0Om0aNE3eBPYYxDCdo
KRaIuMYroDFMKV5eeMb8nITBtzO/TMATje61xg9buqTWblCwoxQC7j8rzLygYEXL
bKCljWpyRyCvmFp6E5IhsUS9V3aipbw0GJwo2ZKWHQ5wC9BBm1peEuD9AgMBAAGj
UjBQMA4GA1UdDwEB/wQEAwICpDATBgNVHSUEDDAKBggrBgEFBQcDATAPBgNVHRMB
Af8EBTADAQH/MBgGA1UdEQQRMA+CDWstYS1ub2RlLXZjMXMwDQYJKoZIhvcNAQEL
BQADggEBAAHap+dwrAuejnIK8X/CA2kp2CNZgK8cQbTz6gHcAF7FESv5fL7BiYbJ
eljhZauh1MSU7hCeXNOK92I1ba7fa8gSdQoSblf9MOmeuNJ4tTwT0y5Cv0dE7anr
EEPWhp5BeHM10lvw/S2uPiN5CNo9pSniMamDcSC4JPXqfRbpqNQkeFOjByb/Y+ez
t+4mGQIouLdHDbx53xc0mmDXEfxwfE5K0gcF8T9EOE/azKlVA8Fk84vjMpVR2gka
O1eRCsCGPAnUCviFgNeH15ug+6N54DTTR6ZV/TTV64FDOcsox9nrhYcmH9sYuITi
0WC0XoXDL9tMOyzRR1ax/a26ks3Q3IY=
-----END CERTIFICATE-----`
)

func TestNewManagerNoRotation(t *testing.T) {
	cert, err := tls.X509KeyPair([]byte(certificateData), []byte(privateKeyData))
	if err != nil {
		t.Fatalf("Unable to initialize a certificate: %v", err)
	}

	store := &fakeStore{cert: &cert}
	if _, err := NewManager(nil, &x509.CertificateRequest{}, []certificates.KeyUsage{}, store, 0); err != nil {
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
		{"half way", now.Add(-24 * time.Hour), now.Add(24 * time.Hour), false},
		{"nearly there", now.Add(-100 * time.Hour), now.Add(1 * time.Hour), true},
		{"just started", now.Add(-1 * time.Hour), now.Add(100 * time.Hour), false},
	}

	for _, test := range tests {
		m := manager{
			cert: &tls.Certificate{
				Leaf: &x509.Certificate{
					NotAfter:  test.notAfter,
					NotBefore: test.notBefore,
				},
			},
			template:            &x509.CertificateRequest{},
			usages:              []certificates.KeyUsage{},
			shouldRotatePercent: 10,
		}

		if m.shouldRotate() != test.shouldRotate {
			t.Errorf("For test case %s, time %v, a certificate issued for (%v, %v) should rotate should be %t.",
				test.name,
				now,
				m.cert.Leaf.NotBefore,
				m.cert.Leaf.NotAfter,
				test.shouldRotate)
		}
	}
}

func TestRotateCertCreateCSRError(t *testing.T) {
	now := time.Now()
	m := manager{
		cert: &tls.Certificate{
			Leaf: &x509.Certificate{
				NotAfter:  now.Add(-1 * time.Hour),
				NotBefore: now.Add(-2 * time.Hour),
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
				NotAfter:  now.Add(-1 * time.Hour),
				NotBefore: now.Add(-2 * time.Hour),
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
			Certificate: []byte(certificateData),
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
	s.cert = &cert
	return s.cert, nil
}
