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

package certs

import (
	"crypto/rsa"
	"crypto/x509"
	"testing"

	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
)

// SetupCertificateAuthorithy is a utility function for kubeadm testing that creates a
// CertificateAuthorithy cert/key pair
func SetupCertificateAuthorithy(t *testing.T) (*x509.Certificate, *rsa.PrivateKey) {
	caCert, caKey, err := pkiutil.NewCertificateAuthority()
	if err != nil {
		t.Fatalf("failure while generating CA certificate and key: %v", err)
	}

	return caCert, caKey
}

// AssertCertificateIsSignedByCa is a utility function for kubeadm testing that asserts if a given certificate is signed
// by the expected CA
func AssertCertificateIsSignedByCa(t *testing.T, cert *x509.Certificate, signingCa *x509.Certificate) {
	if err := cert.CheckSignatureFrom(signingCa); err != nil {
		t.Error("cert is not signed by signing CA as expected")
	}
}

// AssertCertificateHasCommonName is a utility function for kubeadm testing that asserts if a given certificate has
// the expected SubjectCommonName
func AssertCertificateHasCommonName(t *testing.T, cert *x509.Certificate, commonName string) {
	if cert.Subject.CommonName != commonName {
		t.Errorf("cert has Subject.CommonName %s, expected %s", cert.Subject.CommonName, commonName)
	}
}

// AssertCertificateHasOrganizations is a utility function for kubeadm testing that asserts if a given certificate has
// the expected Subject.Organization
func AssertCertificateHasOrganizations(t *testing.T, cert *x509.Certificate, organizations ...string) {
	for _, organization := range organizations {
		found := false
		for i := range cert.Subject.Organization {
			if cert.Subject.Organization[i] == organization {
				found = true
			}
		}
		if !found {
			t.Errorf("cert does not contain Subject.Organization %s as expected", organization)
		}
	}
}

// AssertCertificateHasClientAuthUsage is a utility function for kubeadm testing that asserts if a given certificate has
// the expected ExtKeyUsageClientAuth
func AssertCertificateHasClientAuthUsage(t *testing.T, cert *x509.Certificate) {
	for i := range cert.ExtKeyUsage {
		if cert.ExtKeyUsage[i] == x509.ExtKeyUsageClientAuth {
			return
		}
	}
	t.Error("cert has not ClientAuth usage as expected")
}
