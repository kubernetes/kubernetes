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
	"net"
	"os"
	"path/filepath"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
	certstestutil "k8s.io/kubernetes/cmd/kubeadm/test/certs"
)

func TestWriteCertificateAuthorithyFilesIfNotExist(t *testing.T) {

	setupCert, setupKey, _ := NewCACertAndKey()
	caCert, caKey, _ := NewCACertAndKey()

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
				cert, key, _ := NewFrontProxyClientCertAndKey(setupCert, setupKey)
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

	caCert, caKey, _ := NewFrontProxyCACertAndKey()
	setupCert, setupKey, _ := NewFrontProxyClientCertAndKey(caCert, caKey)
	cert, key, _ := NewFrontProxyClientCertAndKey(caCert, caKey)

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
				anotherCaCert, anotherCaKey, _ := NewFrontProxyCACertAndKey()
				anotherCert, anotherKey, _ := NewFrontProxyClientCertAndKey(anotherCaCert, anotherCaKey)

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

func TestGetAltNames(t *testing.T) {
	hostname := "valid-hostname"
	advertiseIP := "1.2.3.4"
	cfg := &kubeadmapi.MasterConfiguration{
		API:        kubeadmapi.API{AdvertiseAddress: advertiseIP},
		Networking: kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
		NodeName:   hostname,
	}

	altNames, err := getAltNames(cfg)
	if err != nil {
		t.Fatalf("failed calling getAltNames: %v", err)
	}

	expectedDNSNames := []string{hostname, "kubernetes", "kubernetes.default", "kubernetes.default.svc", "kubernetes.default.svc.cluster.local"}
	for _, DNSName := range expectedDNSNames {
		found := false
		for _, val := range altNames.DNSNames {
			if val == DNSName {
				found = true
				break
			}
		}

		if !found {
			t.Errorf("altNames does not contain DNSName %s", DNSName)
		}
	}

	expectedIPAddresses := []string{"10.96.0.1", advertiseIP}
	for _, IPAddress := range expectedIPAddresses {
		found := false
		for _, val := range altNames.IPs {
			if val.Equal(net.ParseIP(IPAddress)) {
				found = true
				break
			}
		}

		if !found {
			t.Errorf("altNames does not contain IPAddress %s", IPAddress)
		}
	}
}

func TestNewCACertAndKey(t *testing.T) {
	caCert, _, err := NewCACertAndKey()
	if err != nil {
		t.Fatalf("failed call NewCACertAndKey: %v", err)
	}

	certstestutil.AssertCertificateIsCa(t, caCert)
}

func TestNewAPIServerCertAndKey(t *testing.T) {
	hostname := "valid-hostname"

	advertiseAddresses := []string{"1.2.3.4", "1:2:3::4"}
	for _, addr := range advertiseAddresses {
		cfg := &kubeadmapi.MasterConfiguration{
			API:        kubeadmapi.API{AdvertiseAddress: addr},
			Networking: kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
			NodeName:   "valid-hostname",
		}
		caCert, caKey, err := NewCACertAndKey()
		if err != nil {
			t.Fatalf("failed creation of ca cert and key: %v", err)
		}

		apiServerCert, _, err := NewAPIServerCertAndKey(cfg, caCert, caKey)
		if err != nil {
			t.Fatalf("failed creation of cert and key: %v", err)
		}

		certstestutil.AssertCertificateIsSignedByCa(t, apiServerCert, caCert)
		certstestutil.AssertCertificateHasServerAuthUsage(t, apiServerCert)
		certstestutil.AssertCertificateHasDNSNames(t, apiServerCert, hostname, "kubernetes", "kubernetes.default", "kubernetes.default.svc", "kubernetes.default.svc.cluster.local")
		certstestutil.AssertCertificateHasIPAddresses(t, apiServerCert, net.ParseIP("10.96.0.1"), net.ParseIP(addr))
	}
}

func TestNewAPIServerKubeletClientCertAndKey(t *testing.T) {
	caCert, caKey, err := NewCACertAndKey()
	if err != nil {
		t.Fatalf("failed creation of ca cert and key: %v", err)
	}

	apiClientCert, _, err := NewAPIServerKubeletClientCertAndKey(caCert, caKey)
	if err != nil {
		t.Fatalf("failed creation of cert and key: %v", err)
	}

	certstestutil.AssertCertificateIsSignedByCa(t, apiClientCert, caCert)
	certstestutil.AssertCertificateHasClientAuthUsage(t, apiClientCert)
	certstestutil.AssertCertificateHasOrganizations(t, apiClientCert, kubeadmconstants.MastersGroup)
}

func TestNewNewServiceAccountSigningKey(t *testing.T) {

	key, err := NewServiceAccountSigningKey()
	if err != nil {
		t.Fatalf("failed creation of key: %v", err)
	}

	if key.N.BitLen() < 2048 {
		t.Error("Service account signing key has less than 2048 bits size")
	}
}

func TestNewFrontProxyCACertAndKey(t *testing.T) {
	frontProxyCACert, _, err := NewFrontProxyCACertAndKey()
	if err != nil {
		t.Fatalf("failed creation of cert and key: %v", err)
	}

	certstestutil.AssertCertificateIsCa(t, frontProxyCACert)
}

func TestNewFrontProxyClientCertAndKey(t *testing.T) {
	frontProxyCACert, frontProxyCAKey, err := NewFrontProxyCACertAndKey()
	if err != nil {
		t.Fatalf("failed creation of ca cert and key: %v", err)
	}

	frontProxyClientCert, _, err := NewFrontProxyClientCertAndKey(frontProxyCACert, frontProxyCAKey)
	if err != nil {
		t.Fatalf("failed creation of cert and key: %v", err)
	}

	certstestutil.AssertCertificateIsSignedByCa(t, frontProxyClientCert, frontProxyCACert)
	certstestutil.AssertCertificateHasClientAuthUsage(t, frontProxyClientCert)
}

func TestUsingExternalCA(t *testing.T) {

	tests := []struct {
		setupFuncs []func(cfg *kubeadmapi.MasterConfiguration) error
		expected   bool
	}{
		{
			setupFuncs: []func(cfg *kubeadmapi.MasterConfiguration) error{
				CreatePKIAssets,
			},
			expected: false,
		},
		{
			setupFuncs: []func(cfg *kubeadmapi.MasterConfiguration) error{
				CreatePKIAssets,
				deleteCAKey,
			},
			expected: true,
		},
	}

	for _, test := range tests {
		dir := testutil.SetupTempDir(t)
		defer os.RemoveAll(dir)

		cfg := &kubeadmapi.MasterConfiguration{
			API:             kubeadmapi.API{AdvertiseAddress: "1.2.3.4"},
			Networking:      kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
			NodeName:        "valid-hostname",
			CertificatesDir: dir,
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

	tests := []struct {
		name            string
		setupFuncs      []func(cfg *kubeadmapi.MasterConfiguration) error
		validateFunc    func(l certKeyLocation) error
		loc             certKeyLocation
		expectedSuccess bool
	}{
		{
			name: "validateCACert",
			setupFuncs: []func(cfg *kubeadmapi.MasterConfiguration) error{
				CreateCACertAndKeyfiles,
			},
			validateFunc:    validateCACert,
			loc:             certKeyLocation{caBaseName: "ca", baseName: "", uxName: "CA"},
			expectedSuccess: true,
		},
		{
			name: "validateCACertAndKey (files present)",
			setupFuncs: []func(cfg *kubeadmapi.MasterConfiguration) error{
				CreateCACertAndKeyfiles,
			},
			validateFunc:    validateCACertAndKey,
			loc:             certKeyLocation{caBaseName: "ca", baseName: "", uxName: "CA"},
			expectedSuccess: true,
		},
		{
			name: "validateCACertAndKey (key missing)",
			setupFuncs: []func(cfg *kubeadmapi.MasterConfiguration) error{
				CreatePKIAssets,
				deleteCAKey,
			},
			validateFunc:    validateCACertAndKey,
			loc:             certKeyLocation{caBaseName: "ca", baseName: "", uxName: "CA"},
			expectedSuccess: false,
		},
		{
			name: "validateSignedCert",
			setupFuncs: []func(cfg *kubeadmapi.MasterConfiguration) error{
				CreateCACertAndKeyfiles,
				CreateAPIServerCertAndKeyFiles,
			},
			validateFunc:    validateSignedCert,
			loc:             certKeyLocation{caBaseName: "ca", baseName: "apiserver", uxName: "apiserver"},
			expectedSuccess: true,
		},
		{
			name: "validatePrivatePublicKey",
			setupFuncs: []func(cfg *kubeadmapi.MasterConfiguration) error{
				CreateServiceAccountKeyAndPublicKeyFiles,
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

		cfg := &kubeadmapi.MasterConfiguration{
			API:             kubeadmapi.API{AdvertiseAddress: "1.2.3.4"},
			Networking:      kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
			NodeName:        "valid-hostname",
			CertificatesDir: dir,
		}

		fmt.Println("Testing", test.name)

		for _, f := range test.setupFuncs {
			if err := f(cfg); err != nil {
				t.Errorf("error executing setup function: %v", err)
			}
		}

		err := test.validateFunc(test.loc)
		if test.expectedSuccess && err != nil {
			t.Errorf("expected success, error executing validateFunc: %v, %v", test.name, err)
		} else if !test.expectedSuccess && err == nil {
			t.Errorf("expected failure, no error executing validateFunc: %v", test.name)
		}
	}
}

func deleteCAKey(cfg *kubeadmapi.MasterConfiguration) error {
	if err := os.Remove(filepath.Join(cfg.CertificatesDir, "ca.key")); err != nil {
		return fmt.Errorf("failed removing ca.key: %v", err)
	}
	return nil
}

func assertIsCa(t *testing.T, cert *x509.Certificate) {
	if !cert.IsCA {
		t.Error("cert is not a valida CA")
	}
}

func TestCreateCertificateFilesMethods(t *testing.T) {

	var tests = []struct {
		setupFunc     func(cfg *kubeadmapi.MasterConfiguration) error
		createFunc    func(cfg *kubeadmapi.MasterConfiguration) error
		expectedFiles []string
	}{
		{
			createFunc: CreatePKIAssets,
			expectedFiles: []string{
				kubeadmconstants.CACertName, kubeadmconstants.CAKeyName,
				kubeadmconstants.APIServerCertName, kubeadmconstants.APIServerKeyName,
				kubeadmconstants.APIServerKubeletClientCertName, kubeadmconstants.APIServerKubeletClientKeyName,
				kubeadmconstants.ServiceAccountPrivateKeyName, kubeadmconstants.ServiceAccountPublicKeyName,
				kubeadmconstants.FrontProxyCACertName, kubeadmconstants.FrontProxyCAKeyName,
				kubeadmconstants.FrontProxyClientCertName, kubeadmconstants.FrontProxyClientKeyName,
			},
		},
		{
			createFunc:    CreateCACertAndKeyfiles,
			expectedFiles: []string{kubeadmconstants.CACertName, kubeadmconstants.CAKeyName},
		},
		{
			setupFunc:     CreateCACertAndKeyfiles,
			createFunc:    CreateAPIServerCertAndKeyFiles,
			expectedFiles: []string{kubeadmconstants.APIServerCertName, kubeadmconstants.APIServerKeyName},
		},
		{
			setupFunc:     CreateCACertAndKeyfiles,
			createFunc:    CreateAPIServerKubeletClientCertAndKeyFiles,
			expectedFiles: []string{kubeadmconstants.APIServerKubeletClientCertName, kubeadmconstants.APIServerKubeletClientKeyName},
		},
		{
			createFunc:    CreateServiceAccountKeyAndPublicKeyFiles,
			expectedFiles: []string{kubeadmconstants.ServiceAccountPrivateKeyName, kubeadmconstants.ServiceAccountPublicKeyName},
		},
		{
			createFunc:    CreateFrontProxyCACertAndKeyFiles,
			expectedFiles: []string{kubeadmconstants.FrontProxyCACertName, kubeadmconstants.FrontProxyCAKeyName},
		},
		{
			setupFunc:     CreateFrontProxyCACertAndKeyFiles,
			createFunc:    CreateFrontProxyClientCertAndKeyFiles,
			expectedFiles: []string{kubeadmconstants.FrontProxyCACertName, kubeadmconstants.FrontProxyCAKeyName},
		},
	}

	for _, test := range tests {
		// Create temp folder for the test case
		tmpdir := testutil.SetupTempDir(t)
		defer os.RemoveAll(tmpdir)

		cfg := &kubeadmapi.MasterConfiguration{
			API:             kubeadmapi.API{AdvertiseAddress: "1.2.3.4"},
			Networking:      kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
			NodeName:        "valid-hostname",
			CertificatesDir: tmpdir,
		}

		// executes setup func (if necessary)
		if test.setupFunc != nil {
			if err := test.setupFunc(cfg); err != nil {
				t.Errorf("error executing setupFunc: %v", err)
				continue
			}
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
