/*
Copyright 2016 The Kubernetes Authors.

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
	"bytes"
	"crypto"
	"crypto/sha256"
	"crypto/x509"
	"io/ioutil"
	"net"
	"os"
	"path"
	"path/filepath"
	"testing"

	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"

	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certstestutil "k8s.io/kubernetes/cmd/kubeadm/app/util/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

func createTestCSR(t *testing.T) (*x509.CertificateRequest, crypto.Signer) {
	csr, key, err := pkiutil.NewCSRAndKey(
		&pkiutil.CertConfig{
			Config: certutil.Config{
				CommonName: "testCert",
			},
		})
	if err != nil {
		t.Fatalf("couldn't create test cert: %v", err)
	}

	return csr, key
}

func TestWriteCertificateAuthorityFilesIfNotExist(t *testing.T) {
	setupCert, setupKey := certstestutil.CreateCACert(t)
	caCert, caKey := certstestutil.CreateCACert(t)

	var tests = []struct {
		setupFunc     func(pkiDir string) error
		expectedError bool
		expectedCa    *x509.Certificate
	}{
		{ // ca cert does not exists > ca written
			expectedCa: caCert,
		},
		{ // ca cert exists, is ca > existing ca used
			setupFunc: func(pkiDir string) error {
				return writeCertificateAuthorityFilesIfNotExist(pkiDir, "dummy", setupCert, setupKey)
			},
			expectedCa: setupCert,
		},
		{ // some file exists, but it is not a valid ca cert > err
			setupFunc: func(pkiDir string) error {
				testutil.SetupEmptyFiles(t, pkiDir, "dummy.crt")
				return nil
			},
			expectedError: true,
		},
		{ // cert exists, but it is not a ca > err
			setupFunc: func(pkiDir string) error {
				cert, key, config := certstestutil.CreateTestCert(t, setupCert, setupKey, certutil.AltNames{})
				return writeCertificateFilesIfNotExist(pkiDir, "dummy", setupCert, cert, key, config)
			},
			expectedError: true,
		},
	}

	for _, test := range tests {
		// Create temp folder for the test case
		tmpdir := testutil.SetupTempDir(t)
		defer os.RemoveAll(tmpdir)

		// executes setup func (if necessary)
		if test.setupFunc != nil {
			if err := test.setupFunc(tmpdir); err != nil {
				t.Errorf("error executing setupFunc: %v", err)
				continue
			}
		}

		// executes create func
		err := writeCertificateAuthorityFilesIfNotExist(tmpdir, "dummy", caCert, caKey)

		if !test.expectedError && err != nil {
			t.Errorf("error writeCertificateAuthorityFilesIfNotExist failed when not expected to fail: %v", err)
			continue
		} else if test.expectedError && err == nil {
			t.Error("error writeCertificateAuthorityFilesIfNotExist didn't failed when expected")
			continue
		} else if test.expectedError {
			continue
		}

		// asserts expected files are there
		testutil.AssertFileExists(t, tmpdir, "dummy.key", "dummy.crt")

		// check created cert
		resultingCaCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(tmpdir, "dummy")
		if err != nil {
			t.Errorf("failure reading created cert: %v", err)
			continue
		}
		if !resultingCaCert.Equal(test.expectedCa) {
			t.Error("created ca cert does not match expected ca cert")
		}
	}
}

func TestWriteCertificateFilesIfNotExist(t *testing.T) {
	altNames := certutil.AltNames{
		DNSNames: []string{"example.com"},
		IPs: []net.IP{
			net.IPv4(0, 0, 0, 0),
		},
	}

	caCert, caKey := certstestutil.CreateCACert(t)
	setupCert, setupKey, _ := certstestutil.CreateTestCert(t, caCert, caKey, altNames)
	cert, key, config := certstestutil.CreateTestCert(t, caCert, caKey, altNames)

	var tests = []struct {
		setupFunc     func(pkiDir string) error
		expectedError bool
		expectedCert  *x509.Certificate
	}{
		{ // cert does not exists > cert written
			expectedCert: cert,
		},
		{ // cert exists, is signed by the same ca, missing SANs (dns name) > err
			setupFunc: func(pkiDir string) error {
				setupCert, setupKey, setupConfig := certstestutil.CreateTestCert(t, caCert, caKey, certutil.AltNames{
					IPs: []net.IP{
						net.IPv4(0, 0, 0, 0),
					},
				})
				return writeCertificateFilesIfNotExist(pkiDir, "dummy", caCert, setupCert, setupKey, setupConfig)
			},
			expectedError: true,
		},
		{ // cert exists, is signed by the same ca, missing SANs (IP address) > err
			setupFunc: func(pkiDir string) error {
				setupCert, setupKey, setupConfig := certstestutil.CreateTestCert(t, caCert, caKey, certutil.AltNames{
					DNSNames: []string{"example.com"},
				})
				return writeCertificateFilesIfNotExist(pkiDir, "dummy", caCert, setupCert, setupKey, setupConfig)
			},
			expectedError: true,
		},
		{ // cert exists, is signed by the same ca, all SANs present > existing cert used
			setupFunc: func(pkiDir string) error {
				return writeCertificateFilesIfNotExist(pkiDir, "dummy", caCert, setupCert, setupKey, config)
			},
			expectedCert: setupCert,
		},
		{ // some file exists, but it is not a valid cert > err
			setupFunc: func(pkiDir string) error {
				testutil.SetupEmptyFiles(t, pkiDir, "dummy.crt")
				return nil
			},
			expectedError: true,
		},
		{ // cert exists, is signed by another ca > err
			setupFunc: func(pkiDir string) error {
				anotherCaCert, anotherCaKey := certstestutil.CreateCACert(t)
				anotherCert, anotherKey, config := certstestutil.CreateTestCert(t, anotherCaCert, anotherCaKey, certutil.AltNames{})

				return writeCertificateFilesIfNotExist(pkiDir, "dummy", anotherCaCert, anotherCert, anotherKey, config)
			},
			expectedError: true,
		},
	}

	for _, test := range tests {
		// Create temp folder for the test case
		tmpdir := testutil.SetupTempDir(t)
		defer os.RemoveAll(tmpdir)

		// executes setup func (if necessary)
		if test.setupFunc != nil {
			if err := test.setupFunc(tmpdir); err != nil {
				t.Errorf("error executing setupFunc: %v", err)
				continue
			}
		}

		// executes create func
		err := writeCertificateFilesIfNotExist(tmpdir, "dummy", caCert, cert, key, config)

		if !test.expectedError && err != nil {
			t.Errorf("error writeCertificateFilesIfNotExist failed when not expected to fail: %v", err)
			continue
		} else if test.expectedError && err == nil {
			t.Error("error writeCertificateFilesIfNotExist didn't fail when expected")
			continue
		} else if test.expectedError {
			continue
		}

		// asserts expected files are there
		testutil.AssertFileExists(t, tmpdir, "dummy.key", "dummy.crt")

		// check created cert
		resultingCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(tmpdir, "dummy")
		if err != nil {
			t.Errorf("failure reading created cert: %v", err)
			continue
		}
		if !resultingCert.Equal(test.expectedCert) {
			t.Error("created cert does not match expected cert")
		}
	}
}

func TestWriteCSRFilesIfNotExist(t *testing.T) {
	csr, key := createTestCSR(t)
	csr2, key2 := createTestCSR(t)

	var tests = []struct {
		name          string
		setupFunc     func(csrPath string) error
		expectedError bool
		expectedCSR   *x509.CertificateRequest
	}{
		{
			name:        "no files exist",
			expectedCSR: csr,
		},
		{
			name: "other key exists",
			setupFunc: func(csrPath string) error {
				if err := pkiutil.WriteCSR(csrPath, "dummy", csr2); err != nil {
					return err
				}
				return pkiutil.WriteKey(csrPath, "dummy", key2)
			},
			expectedCSR: csr2,
		},
		{
			name: "existing CSR is garbage",
			setupFunc: func(csrPath string) error {
				return ioutil.WriteFile(path.Join(csrPath, "dummy.csr"), []byte("a--bunch--of-garbage"), os.ModePerm)
			},
			expectedError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

			if test.setupFunc != nil {
				if err := test.setupFunc(tmpdir); err != nil {
					t.Fatalf("couldn't set up test: %v", err)
				}
			}

			if err := writeCSRFilesIfNotExist(tmpdir, "dummy", csr, key); err != nil {
				if test.expectedError {
					return
				}
				t.Fatalf("unexpected error %v: ", err)
			}

			if test.expectedError {
				t.Fatal("Expected error, but got none")
			}

			parsedCSR, _, err := pkiutil.TryLoadCSRAndKeyFromDisk(tmpdir, "dummy")
			if err != nil {
				t.Fatalf("couldn't load csr and key: %v", err)
			}

			if sha256.Sum256(test.expectedCSR.Raw) != sha256.Sum256(parsedCSR.Raw) {
				t.Error("expected csr's fingerprint does not match ")
			}

		})
	}

}

func TestCreateServiceAccountKeyAndPublicKeyFiles(t *testing.T) {
	setupKey, err := keyutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		t.Fatalf("Can't setup test: %v", err)
	}

	tcases := []struct {
		name        string
		setupFunc   func(pkiDir string) error
		expectedErr bool
		expectedKey []byte
	}{
		{ // key does not exists > key written
			name: "generate successfully",
		},
		{ // key exists > existing key used
			name: "use existing key",
			setupFunc: func(pkiDir string) error {
				err := keyutil.WriteKey(filepath.Join(pkiDir, kubeadmconstants.ServiceAccountPrivateKeyName), setupKey)
				return err
			},
			expectedKey: setupKey,
		},
		{ // some file exists, but it is not a valid key > err
			name: "empty key",
			setupFunc: func(pkiDir string) error {
				testutil.SetupEmptyFiles(t, pkiDir, kubeadmconstants.ServiceAccountPrivateKeyName)
				return nil
			},
			expectedErr: true,
		},
	}
	for _, tt := range tcases {
		t.Run(tt.name, func(t *testing.T) {
			dir := testutil.SetupTempDir(t)
			defer os.RemoveAll(dir)

			if tt.setupFunc != nil {
				if err := tt.setupFunc(dir); err != nil {
					t.Fatalf("error executing setupFunc: %v", err)
				}
			}

			err := CreateServiceAccountKeyAndPublicKeyFiles(dir, x509.RSA)
			if (err != nil) != tt.expectedErr {
				t.Fatalf("expected error: %v, got: %v, error: %v", tt.expectedErr, err != nil, err)
			} else if tt.expectedErr {
				return
			}

			resultingKeyPEM, wasGenerated, err := keyutil.LoadOrGenerateKeyFile(filepath.Join(dir, kubeadmconstants.ServiceAccountPrivateKeyName))
			if err != nil {
				t.Errorf("Can't load created key: %v", err)
			} else if wasGenerated {
				t.Error("The key was not created")
			} else if tt.expectedKey != nil && !bytes.Equal(resultingKeyPEM, tt.expectedKey) {
				t.Error("Non-existing key is used")
			}
		})
	}
}

func TestSharedCertificateExists(t *testing.T) {
	caCert, caKey := certstestutil.CreateCACert(t)
	_, key, _ := certstestutil.CreateTestCert(t, caCert, caKey, certutil.AltNames{})
	publicKey := key.Public()

	var tests = []struct {
		name          string
		files         certstestutil.PKIFiles
		expectedError bool
	}{
		{
			name: "success",
			files: certstestutil.PKIFiles{
				"ca.crt":             caCert,
				"ca.key":             caKey,
				"front-proxy-ca.crt": caCert,
				"front-proxy-ca.key": caKey,
				"sa.pub":             publicKey,
				"sa.key":             key,
				"etcd/ca.crt":        caCert,
				"etcd/ca.key":        caKey,
			},
		},
		{
			name: "missing ca.crt",
			files: certstestutil.PKIFiles{
				"ca.key":             caKey,
				"front-proxy-ca.crt": caCert,
				"front-proxy-ca.key": caKey,
				"sa.pub":             publicKey,
				"sa.key":             key,
				"etcd/ca.crt":        caCert,
				"etcd/ca.key":        caKey,
			},
			expectedError: true,
		},
		{
			name: "missing ca.key",
			files: certstestutil.PKIFiles{
				"ca.crt":             caCert,
				"front-proxy-ca.crt": caCert,
				"front-proxy-ca.key": caKey,
				"sa.pub":             publicKey,
				"sa.key":             key,
				"etcd/ca.crt":        caCert,
				"etcd/ca.key":        caKey,
			},
			expectedError: false,
		},
		{
			name: "missing sa.key",
			files: certstestutil.PKIFiles{
				"ca.crt":             caCert,
				"ca.key":             caKey,
				"front-proxy-ca.crt": caCert,
				"front-proxy-ca.key": caKey,
				"sa.pub":             publicKey,
				"etcd/ca.crt":        caCert,
				"etcd/ca.key":        caKey,
			},
			expectedError: true,
		},
		{
			name: "missing front-proxy.crt",
			files: certstestutil.PKIFiles{
				"ca.crt":             caCert,
				"ca.key":             caKey,
				"front-proxy-ca.key": caKey,
				"sa.pub":             publicKey,
				"sa.key":             key,
				"etcd/ca.crt":        caCert,
				"etcd/ca.key":        caKey,
			},
			expectedError: true,
		},
		{
			name: "missing etcd/ca.crt",
			files: certstestutil.PKIFiles{
				"ca.crt":             caCert,
				"ca.key":             caKey,
				"front-proxy-ca.key": caKey,
				"sa.pub":             publicKey,
				"sa.key":             key,
				"etcd/ca.crt":        caCert,
				"etcd/ca.key":        caKey,
			},
			expectedError: true,
		},
	}

	for _, test := range tests {
		t.Run("", func(t *testing.T) {
			tmpdir := testutil.SetupTempDir(t)
			os.MkdirAll(tmpdir+"/etcd", os.ModePerm)
			defer os.RemoveAll(tmpdir)

			cfg := &kubeadmapi.ClusterConfiguration{
				CertificatesDir: tmpdir,
			}

			// created expected keys
			certstestutil.WritePKIFiles(t, tmpdir, test.files)

			// executes create func
			ret, err := SharedCertificateExists(cfg)

			switch {
			case !test.expectedError && err != nil:
				t.Errorf("error SharedCertificateExists failed when not expected to fail: %v", err)
			case test.expectedError && err == nil:
				t.Errorf("error SharedCertificateExists didn't failed when expected")
			case ret != (err == nil):
				t.Errorf("error SharedCertificateExists returned %v when expected to return %v", ret, err == nil)
			}
		})
	}
}

func TestCreatePKIAssetsWithSparseCerts(t *testing.T) {
	for _, test := range certstestutil.GetSparseCertTestCases(t) {
		t.Run(test.Name, func(t *testing.T) {
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

			cfg := testutil.GetDefaultInternalConfig(t)
			cfg.ClusterConfiguration.CertificatesDir = tmpdir

			certstestutil.WritePKIFiles(t, tmpdir, test.Files)

			err := CreatePKIAssets(cfg)
			if err != nil {
				if test.ExpectError {
					return
				}
				t.Fatalf("Unexpected error: %v", err)
			}
			if test.ExpectError {
				t.Fatal("Expected error from CreatePKIAssets, got none")
			}
			assertCertsExist(t, tmpdir)
		})
	}

}

func TestUsingExternalCA(t *testing.T) {
	tests := []struct {
		name           string
		setupFuncs     []func(cfg *kubeadmapi.InitConfiguration) error
		externalCAFunc func(*kubeadmapi.ClusterConfiguration) (bool, error)
		expected       bool
		expectedErr    bool
	}{
		{
			name: "Test External CA, when complete PKI exists",
			setupFuncs: []func(cfg *kubeadmapi.InitConfiguration) error{
				CreatePKIAssets,
			},
			externalCAFunc: UsingExternalCA,
			expected:       false,
		},
		{
			name: "Test External CA, when ca.key missing",
			setupFuncs: []func(cfg *kubeadmapi.InitConfiguration) error{
				CreatePKIAssets,
				deleteCertOrKey(kubeadmconstants.CAKeyName),
			},
			externalCAFunc: UsingExternalCA,
			expected:       true,
		},
		{
			name: "Test External CA, when ca.key missing and signed certs are missing",
			setupFuncs: []func(cfg *kubeadmapi.InitConfiguration) error{
				CreatePKIAssets,
				deleteCertOrKey(kubeadmconstants.CAKeyName),
				deleteCertOrKey(kubeadmconstants.APIServerCertName),
			},
			externalCAFunc: UsingExternalCA,
			expected:       true,
			expectedErr:    true,
		},
		{
			name: "Test External CA, when ca.key missing",
			setupFuncs: []func(cfg *kubeadmapi.InitConfiguration) error{
				CreatePKIAssets,
				deleteCertOrKey(kubeadmconstants.CAKeyName),
			},
			externalCAFunc: UsingExternalCA,
			expected:       true,
		},
		{
			name: "Test External Front Proxy CA, when complete PKI exists",
			setupFuncs: []func(cfg *kubeadmapi.InitConfiguration) error{
				CreatePKIAssets,
			},
			externalCAFunc: UsingExternalFrontProxyCA,
			expected:       false,
		},
		{
			name: "Test External Front Proxy CA, when front-proxy-ca.key missing",
			setupFuncs: []func(cfg *kubeadmapi.InitConfiguration) error{
				CreatePKIAssets,
				deleteCertOrKey(kubeadmconstants.FrontProxyCAKeyName),
			},
			externalCAFunc: UsingExternalFrontProxyCA,
			expected:       true,
		},
		{
			name: "Test External Front Proxy CA, when front-proxy-.key missing and signed certs are missing",
			setupFuncs: []func(cfg *kubeadmapi.InitConfiguration) error{
				CreatePKIAssets,
				deleteCertOrKey(kubeadmconstants.FrontProxyCAKeyName),
				deleteCertOrKey(kubeadmconstants.FrontProxyClientCertName),
			},
			externalCAFunc: UsingExternalFrontProxyCA,
			expected:       true,
			expectedErr:    true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dir := testutil.SetupTempDir(t)
			defer os.RemoveAll(dir)

			cfg := &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					Networking:      kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
					CertificatesDir: dir,
				},
				NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: "valid-hostname"},
			}

			for _, f := range test.setupFuncs {
				if err := f(cfg); err != nil {
					t.Errorf("error executing setup function: %v", err)
				}
			}

			val, err := test.externalCAFunc(&cfg.ClusterConfiguration)
			if val != test.expected {
				t.Errorf("UsingExternalCA did not match expected: %v", test.expected)
			}

			if (err != nil) != test.expectedErr {
				t.Errorf("UsingExternalCA returned un expected err: %v", err)
			}
		})
	}
}

func TestValidateMethods(t *testing.T) {

	caCert, caKey := certstestutil.CreateCACert(t)
	cert, key, _ := certstestutil.CreateTestCert(t, caCert, caKey, certutil.AltNames{})

	tests := []struct {
		name            string
		files           certstestutil.PKIFiles
		validateFunc    func(l certKeyLocation) error
		loc             certKeyLocation
		expectedSuccess bool
	}{
		{
			name: "validateCACert",
			files: certstestutil.PKIFiles{
				"ca.crt": caCert,
			},
			validateFunc:    validateCACert,
			loc:             certKeyLocation{caBaseName: "ca", baseName: "", uxName: "CA"},
			expectedSuccess: true,
		},
		{
			name: "validateCACertAndKey (files present)",
			files: certstestutil.PKIFiles{
				"ca.crt": caCert,
				"ca.key": caKey,
			},
			validateFunc:    validateCACertAndKey,
			loc:             certKeyLocation{caBaseName: "ca", baseName: "", uxName: "CA"},
			expectedSuccess: true,
		},
		{
			files: certstestutil.PKIFiles{
				"ca.crt": caCert,
			},
			name:            "validateCACertAndKey (key missing)",
			validateFunc:    validateCACertAndKey,
			loc:             certKeyLocation{caBaseName: "ca", baseName: "", uxName: "CA"},
			expectedSuccess: true,
		},
		{
			name: "validateSignedCert",
			files: certstestutil.PKIFiles{
				"ca.crt":        caCert,
				"ca.key":        caKey,
				"apiserver.crt": cert,
				"apiserver.key": key,
			},
			validateFunc:    validateSignedCert,
			loc:             certKeyLocation{caBaseName: "ca", baseName: "apiserver", uxName: "apiserver"},
			expectedSuccess: true,
		},
		{
			name: "validatePrivatePublicKey",
			files: certstestutil.PKIFiles{
				"sa.pub": key.Public(),
				"sa.key": key,
			},
			validateFunc:    validatePrivatePublicKey,
			loc:             certKeyLocation{baseName: "sa", uxName: "service account"},
			expectedSuccess: true,
		},
		{
			name: "validatePrivatePublicKey (missing key)",
			files: certstestutil.PKIFiles{
				"sa.pub": key.Public(),
			},
			validateFunc:    validatePrivatePublicKey,
			loc:             certKeyLocation{baseName: "sa", uxName: "service account"},
			expectedSuccess: false,
		},
	}

	for _, test := range tests {
		dir := testutil.SetupTempDir(t)
		defer os.RemoveAll(dir)
		test.loc.pkiDir = dir

		certstestutil.WritePKIFiles(t, dir, test.files)

		err := test.validateFunc(test.loc)
		if test.expectedSuccess && err != nil {
			t.Errorf("expected success, error executing validateFunc: %v, %v", test.name, err)
		} else if !test.expectedSuccess && err == nil {
			t.Errorf("expected failure, no error executing validateFunc: %v", test.name)
		}
	}
}

func TestNewCSR(t *testing.T) {
	kubeadmCert := KubeadmCertAPIServer()
	cfg := testutil.GetDefaultInternalConfig(t)

	certConfig, err := kubeadmCert.GetConfig(cfg)
	if err != nil {
		t.Fatalf("couldn't get cert config: %v", err)
	}

	csr, _, err := NewCSR(kubeadmCert, cfg)

	if err != nil {
		t.Errorf("invalid signature on CSR: %v", err)
	}

	assert.ElementsMatch(t, certConfig.Organization, csr.Subject.Organization, "organizations not equal")

	if csr.Subject.CommonName != certConfig.CommonName {
		t.Errorf("expected common name %q, got %q", certConfig.CommonName, csr.Subject.CommonName)
	}

	assert.ElementsMatch(t, certConfig.AltNames.DNSNames, csr.DNSNames, "dns names not equal")

	assert.Len(t, csr.IPAddresses, len(certConfig.AltNames.IPs))

	for i, ip := range csr.IPAddresses {
		if !ip.Equal(certConfig.AltNames.IPs[i]) {
			t.Errorf("[%d]: %v != %v", i, ip, certConfig.AltNames.IPs[i])
		}
	}
}

func TestCreateCertificateFilesMethods(t *testing.T) {

	var tests = []struct {
		createFunc    func(cfg *kubeadmapi.InitConfiguration) error
		expectedFiles []string
		externalEtcd  bool
	}{
		{
			createFunc: CreatePKIAssets,
			expectedFiles: []string{
				kubeadmconstants.CACertName, kubeadmconstants.CAKeyName,
				kubeadmconstants.APIServerCertName, kubeadmconstants.APIServerKeyName,
				kubeadmconstants.APIServerKubeletClientCertName, kubeadmconstants.APIServerKubeletClientKeyName,
				kubeadmconstants.EtcdCACertName, kubeadmconstants.EtcdCAKeyName,
				kubeadmconstants.EtcdServerCertName, kubeadmconstants.EtcdServerKeyName,
				kubeadmconstants.EtcdPeerCertName, kubeadmconstants.EtcdPeerKeyName,
				kubeadmconstants.EtcdHealthcheckClientCertName, kubeadmconstants.EtcdHealthcheckClientKeyName,
				kubeadmconstants.APIServerEtcdClientCertName, kubeadmconstants.APIServerEtcdClientKeyName,
				kubeadmconstants.ServiceAccountPrivateKeyName, kubeadmconstants.ServiceAccountPublicKeyName,
				kubeadmconstants.FrontProxyCACertName, kubeadmconstants.FrontProxyCAKeyName,
				kubeadmconstants.FrontProxyClientCertName, kubeadmconstants.FrontProxyClientKeyName,
			},
		},
		{
			createFunc:   CreatePKIAssets,
			externalEtcd: true,
			expectedFiles: []string{
				kubeadmconstants.CACertName, kubeadmconstants.CAKeyName,
				kubeadmconstants.APIServerCertName, kubeadmconstants.APIServerKeyName,
				kubeadmconstants.APIServerKubeletClientCertName, kubeadmconstants.APIServerKubeletClientKeyName,
				kubeadmconstants.ServiceAccountPrivateKeyName, kubeadmconstants.ServiceAccountPublicKeyName,
				kubeadmconstants.FrontProxyCACertName, kubeadmconstants.FrontProxyCAKeyName,
				kubeadmconstants.FrontProxyClientCertName, kubeadmconstants.FrontProxyClientKeyName,
			},
		},
	}

	for _, test := range tests {
		// Create temp folder for the test case
		tmpdir := testutil.SetupTempDir(t)
		defer os.RemoveAll(tmpdir)

		cfg := &kubeadmapi.InitConfiguration{
			LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4"},
			ClusterConfiguration: kubeadmapi.ClusterConfiguration{
				Etcd:            kubeadmapi.Etcd{Local: &kubeadmapi.LocalEtcd{}},
				Networking:      kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
				CertificatesDir: tmpdir,
			},
			NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: "valid-hostname"},
		}

		if test.externalEtcd {
			if cfg.Etcd.External == nil {
				cfg.Etcd.External = &kubeadmapi.ExternalEtcd{}
			}
			cfg.Etcd.Local = nil
			cfg.Etcd.External.Endpoints = []string{"192.168.1.1:2379"}
		}

		// executes create func
		if err := test.createFunc(cfg); err != nil {
			t.Errorf("error executing createFunc: %v", err)
			continue
		}

		// asserts expected files are there
		testutil.AssertFileExists(t, tmpdir, test.expectedFiles...)
	}
}

func deleteCertOrKey(name string) func(*kubeadmapi.InitConfiguration) error {
	return func(cfg *kubeadmapi.InitConfiguration) error {
		if err := os.Remove(filepath.Join(cfg.CertificatesDir, name)); err != nil {
			return errors.Wrapf(err, "failed removing %s", name)
		}
		return nil
	}
}

func assertCertsExist(t *testing.T, dir string) {
	tree, err := GetDefaultCertList().AsMap().CertTree()
	if err != nil {
		t.Fatalf("unexpected error getting certificates: %v", err)
	}

	for caCert, certs := range tree {
		if err := validateCACert(certKeyLocation{dir, caCert.BaseName, "", caCert.Name}); err != nil {
			t.Errorf("couldn't validate CA certificate %v: %v", caCert.Name, err)
			// Don't bother validating child certs, but do try the other CAs
			continue
		}

		for _, cert := range certs {
			if err := validateSignedCert(certKeyLocation{dir, caCert.BaseName, cert.BaseName, cert.Name}); err != nil {
				t.Errorf("couldn't validate certificate %v: %v", cert.Name, err)
			}
		}
	}
}
