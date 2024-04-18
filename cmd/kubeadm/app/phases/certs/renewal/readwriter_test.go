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
	"fmt"
	"net"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/client-go/tools/clientcmd"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	netutils "k8s.io/utils/net"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

func TestPKICertificateReadWriter(t *testing.T) {
	// creates a tmp folder
	dir := testutil.SetupTempDir(t)
	defer os.RemoveAll(dir)

	// creates a certificate
	cert := writeTestCertificate(t, dir, "test", testCACert, testCAKey, testCertOrganization)

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
	// creates tmp folders
	dirKubernetes := testutil.SetupTempDir(t)
	defer os.RemoveAll(dirKubernetes)
	dirPKI := testutil.SetupTempDir(t)
	defer os.RemoveAll(dirPKI)

	// write the CA cert and key to the temporary PKI dir
	caName := kubeadmconstants.CACertAndKeyBaseName
	if err := pkiutil.WriteCertAndKey(
		dirPKI,
		caName,
		testCACert,
		testCAKey); err != nil {
		t.Fatalf("couldn't write out certificate %s to %s", caName, dirPKI)
	}

	// creates a certificate and then embeds it into a kubeconfig file
	cert := writeTestKubeconfig(t, dirKubernetes, "test", testCACert, testCAKey)

	// Creates a KubeconfigReadWriter
	kubeconfigReadWriter := newKubeconfigReadWriter(dirKubernetes, "test", dirPKI, caName)

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

	// Make sure that CA key is not present during Read() as it is not needed.
	// This covers testing when the CA is external and not present on the host.
	_, caKeyPath := pkiutil.PathsForCertAndKey(dirPKI, caName)
	os.Remove(caKeyPath)

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
func writeTestCertificate(t *testing.T, dir, name string, caCert *x509.Certificate, caKey crypto.Signer, organization []string) *x509.Certificate {
	cert, key, err := pkiutil.NewCertAndKey(caCert, caKey, makeTestCertConfig(organization))
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

	cfg := &pkiutil.CertConfig{
		Config: certutil.Config{
			CommonName:   "test-common-name",
			Organization: testCertOrganization,
			Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			AltNames: certutil.AltNames{
				IPs:      []net.IP{netutils.ParseIPSloppy("10.100.0.1")},
				DNSNames: []string{"test-domain.space"},
			},
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

func TestFileExists(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir: %v", err)
	}
	defer func() {
		err = os.RemoveAll(tmpdir)
		if err != nil {
			t.Fatalf("Fail to remove tmpdir: %v", err)
		}
	}()
	tmpfile, err := os.CreateTemp(tmpdir, "")
	if err != nil {
		t.Fatalf("Couldn't create tmpfile: %v", err)
	}
	if err := tmpfile.Close(); err != nil {
		t.Fatalf("Couldn't close tmpfile: %v", err)
	}

	tests := []struct {
		name     string
		filename string
		want     bool
	}{
		{
			name:     "file exist",
			filename: tmpfile.Name(),
			want:     true,
		},
		{
			name:     "file does not exist",
			filename: "foo",
			want:     false,
		},
		{
			name:     "file path is a dir",
			filename: tmpdir,
			want:     false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got, _ := fileExists(tt.filename); got != tt.want {
				t.Errorf("fileExists() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPKICertificateReadWriterExists(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir: %v", err)
	}
	defer func() {
		err = os.RemoveAll(tmpdir)
		if err != nil {
			t.Fatalf("Fail to remove tmpdir: %v", err)
		}
	}()
	filename := "testfile"
	tmpfilepath := filepath.Join(tmpdir, fmt.Sprintf(filename+".crt"))
	err = os.WriteFile(tmpfilepath, nil, 0644)
	if err != nil {
		t.Fatalf("Couldn't write file: %v", err)
	}
	type fields struct {
		baseName       string
		certificateDir string
	}
	tests := []struct {
		name   string
		fields fields
		want   bool
	}{
		{
			name: "cert file exists",
			fields: fields{
				baseName:       filename,
				certificateDir: tmpdir,
			},
			want: true,
		},
		{
			name: "cert file does not exist",
			fields: fields{
				baseName:       "foo",
				certificateDir: tmpdir,
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rw := &pkiCertificateReadWriter{
				baseName:       tt.fields.baseName,
				certificateDir: tt.fields.certificateDir,
			}
			if got, _ := rw.Exists(); got != tt.want {
				t.Errorf("pkiCertificateReadWriter.Exists() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestKubeConfigReadWriterExists(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir: %v", err)
	}
	defer func() {
		err = os.RemoveAll(tmpdir)
		if err != nil {
			t.Fatalf("Fail to remove tmpdir: %v", err)
		}
	}()
	tmpfile, err := os.CreateTemp(tmpdir, "")
	if err != nil {
		t.Fatalf("Couldn't create tmpfile: %v", err)
	}
	if err := tmpfile.Close(); err != nil {
		t.Fatalf("Couldn't close tmpfile: %v", err)
	}

	tests := []struct {
		name               string
		kubeConfigFilePath string
		want               bool
	}{
		{
			name:               "file exists",
			kubeConfigFilePath: tmpfile.Name(),
			want:               true,
		},
		{
			name:               "file does not exist",
			kubeConfigFilePath: "foo",
			want:               false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rw := &kubeConfigReadWriter{
				kubeConfigFilePath: tt.kubeConfigFilePath,
			}
			if got, _ := rw.Exists(); got != tt.want {
				t.Errorf("kubeConfigReadWriter.Exists() = %v, want %v", got, tt.want)
			}
		})
	}
}
