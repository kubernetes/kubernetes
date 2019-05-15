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

package renewal

import (
	"crypto"
	"crypto/x509"
	"net"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/client-go/tools/clientcmd"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	pkiutil "k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

func TestPKICertificateReadWriter(t *testing.T) {
	// creates a tmp folder
	dir := testutil.SetupTempDir(t)
	defer os.RemoveAll(dir)

	// creates a certificate
	cert := writeTestCertificate(t, dir, "test", testCACert, testCAKey)

	// Creates a pkiCertificateReadWriter
	pkiReadWriter := newPKICertificateReadWriter(dir, "test")

	// Reads the certificate
	readCert, err := pkiReadWriter.Read()
	if err != nil {
		t.Fatalf("couldn't read certificate: %v", err)
	}

	// Check if the certificate read from disk is equal to the original one
	if !cert.Equal(readCert) {
		t.Errorf("read cert does not match with expected cert")
	}

	// Create a new cert
	newCert, newkey, err := pkiutil.NewCertAndKey(testCACert, testCAKey, testCertCfg)
	if err != nil {
		t.Fatalf("couldn't generate certificate: %v", err)
	}

	// Writes the new certificate
	err = pkiReadWriter.Write(newCert, newkey)
	if err != nil {
		t.Fatalf("couldn't write new certificate: %v", err)
	}

	// Reads back the new certificate
	readCert, err = pkiReadWriter.Read()
	if err != nil {
		t.Fatalf("couldn't read new certificate: %v", err)
	}

	// Check if the new certificate read from disk is equal to the original one
	if !newCert.Equal(readCert) {
		t.Error("read cert does not match with expected new cert")
	}
}

func TestKubeconfigReadWriter(t *testing.T) {
	// creates a tmp folder
	dir := testutil.SetupTempDir(t)
	defer os.RemoveAll(dir)

	// creates a certificate and then embeds it into a kubeconfig file
	cert := writeTestKubeconfig(t, dir, "test", testCACert, testCAKey)

	// Creates a KubeconfigReadWriter
	kubeconfigReadWriter := newKubeconfigReadWriter(dir, "test")

	// Reads the certificate embedded in a kubeconfig
	readCert, err := kubeconfigReadWriter.Read()
	if err != nil {
		t.Fatalf("couldn't read embedded certificate: %v", err)
	}

	// Check if the certificate read from disk is equal to the original one
	if !cert.Equal(readCert) {
		t.Errorf("read cert does not match with expected cert")
	}

	// Create a new cert
	newCert, newkey, err := pkiutil.NewCertAndKey(testCACert, testCAKey, testCertCfg)
	if err != nil {
		t.Fatalf("couldn't generate certificate: %v", err)
	}

	// Writes the new certificate embedded in a kubeconfig
	err = kubeconfigReadWriter.Write(newCert, newkey)
	if err != nil {
		t.Fatalf("couldn't write new embedded certificate: %v", err)
	}

	// Reads back the new certificate embedded in a kubeconfig writer
	readCert, err = kubeconfigReadWriter.Read()
	if err != nil {
		t.Fatalf("couldn't read new embedded  certificate: %v", err)
	}

	// Check if the new certificate read from disk is equal to the original one
	if !newCert.Equal(readCert) {
		t.Errorf("read cert does not match with expected new cert")
	}
}

// writeTestCertificate is a utility for creating a test certificate
func writeTestCertificate(t *testing.T, dir, name string, caCert *x509.Certificate, caKey crypto.Signer) *x509.Certificate {
	cert, key, err := pkiutil.NewCertAndKey(caCert, caKey, testCertCfg)
	if err != nil {
		t.Fatalf("couldn't generate certificate: %v", err)
	}

	if err := pkiutil.WriteCertAndKey(dir, name, cert, key); err != nil {
		t.Fatalf("couldn't write out certificate %s to %s", name, dir)
	}

	return cert
}

// writeTestKubeconfig is a utility for creating a test kubeconfig with an embedded certificate
func writeTestKubeconfig(t *testing.T, dir, name string, caCert *x509.Certificate, caKey crypto.Signer) *x509.Certificate {

	cfg := &certutil.Config{
		CommonName:   "test-common-name",
		Organization: []string{"sig-cluster-lifecycle"},
		Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		AltNames: certutil.AltNames{
			IPs:      []net.IP{net.ParseIP("10.100.0.1")},
			DNSNames: []string{"test-domain.space"},
		},
	}
	cert, key, err := pkiutil.NewCertAndKey(caCert, caKey, cfg)
	if err != nil {
		t.Fatalf("couldn't generate certificate: %v", err)
	}

	encodedClientKey, err := keyutil.MarshalPrivateKeyToPEM(key)
	if err != nil {
		t.Fatalf("failed to marshal private key to PEM: %v", err)
	}

	certificateAuthorityData := pkiutil.EncodeCertPEM(caCert)

	config := kubeconfigutil.CreateWithCerts(
		"https://localhost:1234",
		"kubernetes-test",
		"user-test",
		certificateAuthorityData,
		encodedClientKey,
		pkiutil.EncodeCertPEM(cert),
	)

	if err := clientcmd.WriteToFile(*config, filepath.Join(dir, name)); err != nil {
		t.Fatalf("couldn't write out certificate")
	}

	return cert
}
