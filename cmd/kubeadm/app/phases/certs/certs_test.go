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
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"testing"

	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
	certstestutil "k8s.io/kubernetes/cmd/kubeadm/test/certs"
)

func createCACert(t *testing.T) (*x509.Certificate, *rsa.PrivateKey) {
	certCfg := &certutil.Config{CommonName: "kubernetes"}
	cert, key, err := NewCACertAndKey(certCfg)
	if err != nil {
		t.Fatalf("couldn't create CA: %v", err)
	}
	return cert, key
}

func createTestCert(t *testing.T, caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, *rsa.PrivateKey) {
	cert, key, err := pkiutil.NewCertAndKey(caCert, caKey,
		&certutil.Config{
			CommonName: "testCert",
			Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageAny},
		})
	if err != nil {
		t.Fatalf("couldn't create test cert: %v", err)
	}
	return cert, key
}

func TestWriteCertificateAuthorithyFilesIfNotExist(t *testing.T) {
	setupCert, setupKey := createCACert(t)
	caCert, caKey := createCACert(t)

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
				return writeCertificateAuthorithyFilesIfNotExist(pkiDir, "dummy", setupCert, setupKey)
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
				cert, key := createTestCert(t, setupCert, setupKey)
				return writeCertificateFilesIfNotExist(pkiDir, "dummy", setupCert, cert, key)
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
		err := writeCertificateAuthorithyFilesIfNotExist(tmpdir, "dummy", caCert, caKey)

		if !test.expectedError && err != nil {
			t.Errorf("error writeCertificateAuthorithyFilesIfNotExist failed when not expected to fail: %v", err)
			continue
		} else if test.expectedError && err == nil {
			t.Error("error writeCertificateAuthorithyFilesIfNotExist didn't failed when expected")
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

	caCert, caKey := createCACert(t)
	setupCert, setupKey := createTestCert(t, caCert, caKey)
	cert, key := createTestCert(t, caCert, caKey)

	var tests = []struct {
		setupFunc     func(pkiDir string) error
		expectedError bool
		expectedCert  *x509.Certificate
	}{
		{ // cert does not exists > cert written
			expectedCert: cert,
		},
		{ // cert exists, is signed by the same ca > existing cert used
			setupFunc: func(pkiDir string) error {
				return writeCertificateFilesIfNotExist(pkiDir, "dummy", caCert, setupCert, setupKey)
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
				anotherCaCert, anotherCaKey := createCACert(t)
				anotherCert, anotherKey := createTestCert(t, anotherCaCert, anotherCaKey)

				return writeCertificateFilesIfNotExist(pkiDir, "dummy", anotherCaCert, anotherCert, anotherKey)
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
		err := writeCertificateFilesIfNotExist(tmpdir, "dummy", caCert, cert, key)

		if !test.expectedError && err != nil {
			t.Errorf("error writeCertificateFilesIfNotExist failed when not expected to fail: %v", err)
			continue
		} else if test.expectedError && err == nil {
			t.Error("error writeCertificateFilesIfNotExist didn't failed when expected")
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

func TestWriteKeyFilesIfNotExist(t *testing.T) {

	setupKey, _ := NewServiceAccountSigningKey()
	key, _ := NewServiceAccountSigningKey()

	var tests = []struct {
		setupFunc     func(pkiDir string) error
		expectedError bool
		expectedKey   *rsa.PrivateKey
	}{
		{ // key does not exists > key written
			expectedKey: key,
		},
		{ // key exists > existing key used
			setupFunc: func(pkiDir string) error {
				return writeKeyFilesIfNotExist(pkiDir, "dummy", setupKey)
			},
			expectedKey: setupKey,
		},
		{ // some file exists, but it is not a valid key > err
			setupFunc: func(pkiDir string) error {
				testutil.SetupEmptyFiles(t, pkiDir, "dummy.key")
				return nil
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
		err := writeKeyFilesIfNotExist(tmpdir, "dummy", key)

		if !test.expectedError && err != nil {
			t.Errorf("error writeKeyFilesIfNotExist failed when not expected to fail: %v", err)
			continue
		} else if test.expectedError && err == nil {
			t.Error("error writeKeyFilesIfNotExist didn't failed when expected")
			continue
		} else if test.expectedError {
			continue
		}

		// asserts expected files are there
		testutil.AssertFileExists(t, tmpdir, "dummy.key", "dummy.pub")

		// check created key
		resultingKey, err := pkiutil.TryLoadKeyFromDisk(tmpdir, "dummy")
		if err != nil {
			t.Errorf("failure reading created key: %v", err)
			continue
		}

		//TODO: check if there is a better method to compare keys
		if resultingKey.D == key.D {
			t.Error("created key does not match expected key")
		}
	}
}

func TestNewCACertAndKey(t *testing.T) {
	certCfg := &certutil.Config{CommonName: "kubernetes"}
	caCert, _, err := NewCACertAndKey(certCfg)
	if err != nil {
		t.Fatalf("failed call NewCACertAndKey: %v", err)
	}

	certstestutil.AssertCertificateIsCa(t, caCert)
}

func TestSharedCertificateExists(t *testing.T) {
	caCert, caKey := createCACert(t)
	_, key := createTestCert(t, caCert, caKey)
	publicKey := &key.PublicKey

	var tests = []struct {
		name          string
		files         pkiFiles
		expectedError bool
	}{
		{
			name: "success",
			files: pkiFiles{
				"ca.crt":             caCert,
				"ca.key":             caKey,
				"front-proxy-ca.crt": caCert,
				"front-proxy-ca.key": caKey,
				"sa.pub":             publicKey,
				"sa.key":             key,
			},
		},
		{
			name: "missing ca.crt",
			files: pkiFiles{
				"ca.key":             caKey,
				"front-proxy-ca.crt": caCert,
				"front-proxy-ca.key": caKey,
				"sa.pub":             publicKey,
				"sa.key":             key,
			},
			expectedError: true,
		},
		{
			name: "missing sa.key",
			files: pkiFiles{
				"ca.crt":             caCert,
				"ca.key":             caKey,
				"front-proxy-ca.crt": caCert,
				"front-proxy-ca.key": caKey,
				"sa.pub":             publicKey,
			},
			expectedError: true,
		},
		{
			name: "expected front-proxy.crt missing",
			files: pkiFiles{
				"ca.crt":             caCert,
				"ca.key":             caKey,
				"front-proxy-ca.key": caKey,
				"sa.pub":             publicKey,
				"sa.key":             key,
			},
			expectedError: true,
		},
	}

	for _, test := range tests {
		t.Run("", func(t *testing.T) {
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

			cfg := &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					CertificatesDir: tmpdir,
				},
			}

			// created expected keys
			writePKIFiles(t, tmpdir, test.files)

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
	caCert, caKey := createCACert(t)
	fpCACert, fpCAKey := createCACert(t)
	etcdCACert, etcdCAKey := createCACert(t)

	fpCert, fpKey := createTestCert(t, fpCACert, fpCAKey)

	tests := []struct {
		name        string
		files       pkiFiles
		expectError bool
	}{
		{
			name: "nothing present",
		},
		{
			name: "CAs already exist",
			files: pkiFiles{
				"ca.crt":             caCert,
				"ca.key":             caKey,
				"front-proxy-ca.crt": fpCACert,
				"front-proxy-ca.key": fpCAKey,
				"etcd/ca.crt":        etcdCACert,
				"etcd/ca.key":        etcdCAKey,
			},
		},
		{
			name: "CA certs only",
			files: pkiFiles{
				"ca.crt":             caCert,
				"front-proxy-ca.crt": fpCACert,
				"etcd/ca.crt":        etcdCACert,
			},
			expectError: true,
		},
		{
			name: "FrontProxyCA with certs",
			files: pkiFiles{
				"ca.crt":                 caCert,
				"ca.key":                 caKey,
				"front-proxy-ca.crt":     fpCACert,
				"front-proxy-client.crt": fpCert,
				"front-proxy-client.key": fpKey,
				"etcd/ca.crt":            etcdCACert,
				"etcd/ca.key":            etcdCAKey,
			},
		},
		{
			name: "FrontProxy certs missing CA",
			files: pkiFiles{
				"front-proxy-client.crt": fpCert,
				"front-proxy-client.key": fpKey,
			},
			expectError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

			cfg := testutil.GetDefaultInternalConfig(t)
			cfg.ClusterConfiguration.CertificatesDir = tmpdir

			writePKIFiles(t, tmpdir, test.files)

			err := CreatePKIAssets(cfg)
			if err != nil {
				if test.expectError {
					return
				}
				t.Fatalf("Unexpected error: %v", err)
			}
			if test.expectError {
				t.Fatal("Expected error from CreatePKIAssets, got none")
			}
			assertCertsExist(t, tmpdir)
		})
	}

}

func TestUsingExternalCA(t *testing.T) {
	tests := []struct {
		setupFuncs []func(cfg *kubeadmapi.InitConfiguration) error
		expected   bool
	}{
		{
			setupFuncs: []func(cfg *kubeadmapi.InitConfiguration) error{
				CreatePKIAssets,
			},
			expected: false,
		},
		{
			setupFuncs: []func(cfg *kubeadmapi.InitConfiguration) error{
				CreatePKIAssets,
				deleteCAKey,
				deleteFrontProxyCAKey,
			},
			expected: true,
		},
	}

	for _, test := range tests {
		dir := testutil.SetupTempDir(t)
		defer os.RemoveAll(dir)

		cfg := &kubeadmapi.InitConfiguration{
			APIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4"},
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

		if val, _ := UsingExternalCA(cfg); val != test.expected {
			t.Errorf("UsingExternalCA did not match expected: %v", test.expected)
		}
	}
}

func TestValidateMethods(t *testing.T) {

	caCert, caKey := createCACert(t)
	cert, key := createTestCert(t, caCert, caKey)

	tests := []struct {
		name            string
		files           pkiFiles
		validateFunc    func(l certKeyLocation) error
		loc             certKeyLocation
		expectedSuccess bool
	}{
		{
			name: "validateCACert",
			files: pkiFiles{
				"ca.crt": caCert,
			},
			validateFunc:    validateCACert,
			loc:             certKeyLocation{caBaseName: "ca", baseName: "", uxName: "CA"},
			expectedSuccess: true,
		},
		{
			name: "validateCACertAndKey (files present)",
			files: pkiFiles{
				"ca.crt": caCert,
				"ca.key": caKey,
			},
			validateFunc:    validateCACertAndKey,
			loc:             certKeyLocation{caBaseName: "ca", baseName: "", uxName: "CA"},
			expectedSuccess: true,
		},
		{
			files: pkiFiles{
				"ca.crt": caCert,
			},
			name:            "validateCACertAndKey (key missing)",
			validateFunc:    validateCACertAndKey,
			loc:             certKeyLocation{caBaseName: "ca", baseName: "", uxName: "CA"},
			expectedSuccess: false,
		},
		{
			name: "validateSignedCert",
			files: pkiFiles{
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
			files: pkiFiles{
				"sa.pub": &key.PublicKey,
				"sa.key": key,
			},
			validateFunc:    validatePrivatePublicKey,
			loc:             certKeyLocation{baseName: "sa", uxName: "service account"},
			expectedSuccess: true,
		},
	}

	for _, test := range tests {
		dir := testutil.SetupTempDir(t)
		defer os.RemoveAll(dir)
		test.loc.pkiDir = dir

		writePKIFiles(t, dir, test.files)

		err := test.validateFunc(test.loc)
		if test.expectedSuccess && err != nil {
			t.Errorf("expected success, error executing validateFunc: %v, %v", test.name, err)
		} else if !test.expectedSuccess && err == nil {
			t.Errorf("expected failure, no error executing validateFunc: %v", test.name)
		}
	}
}

type pkiFiles map[string]interface{}

func writePKIFiles(t *testing.T, dir string, files pkiFiles) {
	for filename, body := range files {
		switch body := body.(type) {
		case *x509.Certificate:
			if err := certutil.WriteCert(path.Join(dir, filename), certutil.EncodeCertPEM(body)); err != nil {
				t.Errorf("unable to write certificate to file %q: [%v]", dir, err)
			}
		case *rsa.PublicKey:
			publicKeyBytes, err := certutil.EncodePublicKeyPEM(body)
			if err != nil {
				t.Errorf("unable to write public key to file %q: [%v]", filename, err)
			}
			if err := certutil.WriteKey(path.Join(dir, filename), publicKeyBytes); err != nil {
				t.Errorf("unable to write public key to file %q: [%v]", filename, err)
			}
		case *rsa.PrivateKey:
			if err := certutil.WriteKey(path.Join(dir, filename), certutil.EncodePrivateKeyPEM(body)); err != nil {
				t.Errorf("unable to write private key to file %q: [%v]", filename, err)
			}
		}
	}
}

func TestCreateCertificateFilesMethods(t *testing.T) {

	var tests = []struct {
		setupFunc     func(cfg *kubeadmapi.InitConfiguration) error
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
			APIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4"},
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

func deleteCAKey(cfg *kubeadmapi.InitConfiguration) error {
	if err := os.Remove(filepath.Join(cfg.CertificatesDir, kubeadmconstants.CAKeyName)); err != nil {
		return fmt.Errorf("failed removing %s: %v", kubeadmconstants.CAKeyName, err)
	}
	return nil
}

func deleteFrontProxyCAKey(cfg *kubeadmapi.InitConfiguration) error {
	if err := os.Remove(filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyCAKeyName)); err != nil {
		return fmt.Errorf("failed removing %s: %v", kubeadmconstants.FrontProxyCAKeyName, err)
	}
	return nil
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
