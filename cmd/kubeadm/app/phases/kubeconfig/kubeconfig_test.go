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

package kubeconfig

import (
	"bytes"
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"io"
	"os"
	"reflect"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"

	pkiutil "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"

	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
	certstestutil "k8s.io/kubernetes/cmd/kubeadm/test/certs"
	kubeconfigtestutil "k8s.io/kubernetes/cmd/kubeadm/test/kubeconfig"
)

func TestGetKubeConfigSpecsFailsIfCADoesntExists(t *testing.T) {
	// Create temp folder for the test case (without a CA)
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	// Creates a Master Configuration pointing to the pkidir folder
	cfg := &kubeadmapi.MasterConfiguration{
		CertificatesDir: tmpdir,
	}

	// Executes getKubeConfigSpecs
	if _, err := getKubeConfigSpecs(cfg); err == nil {
		t.Error("getKubeConfigSpecs didnt failed when expected")
	}
}

func TestGetKubeConfigSpecs(t *testing.T) {
	// Create temp folder for the test case
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	// Adds a pki folder with a ca certs to the temp folder
	pkidir := testutil.SetupPkiDirWithCertificateAuthorithy(t, tmpdir)

	// Creates a Master Configuration pointing to the pkidir folder
	cfg := &kubeadmapi.MasterConfiguration{
		API:             kubeadmapi.API{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
		CertificatesDir: pkidir,
		NodeName:        "valid-node-name",
	}

	// Executes getKubeConfigSpecs
	specs, err := getKubeConfigSpecs(cfg)
	if err != nil {
		t.Fatal("getKubeConfigSpecs failed!")
	}

	var assertions = []struct {
		kubeConfigFile string
		clientName     string
		organizations  []string
	}{
		{
			kubeConfigFile: kubeadmconstants.AdminKubeConfigFileName,
			clientName:     "kubernetes-admin",
			organizations:  []string{kubeadmconstants.MastersGroup},
		},
		{
			kubeConfigFile: kubeadmconstants.KubeletKubeConfigFileName,
			clientName:     fmt.Sprintf("system:node:%s", cfg.NodeName),
			organizations:  []string{kubeadmconstants.NodesGroup},
		},
		{
			kubeConfigFile: kubeadmconstants.ControllerManagerKubeConfigFileName,
			clientName:     kubeadmconstants.ControllerManagerUser,
		},
		{
			kubeConfigFile: kubeadmconstants.SchedulerKubeConfigFileName,
			clientName:     kubeadmconstants.SchedulerUser,
		},
	}

	for _, assertion := range assertions {

		// assert the spec for the kubeConfigFile exists
		if spec, ok := specs[assertion.kubeConfigFile]; ok {

			// Assert clientName
			if spec.ClientName != assertion.clientName {
				t.Errorf("getKubeConfigSpecs for %s clientName is %s, expected %s", assertion.kubeConfigFile, spec.ClientName, assertion.clientName)
			}

			// Assert Organizations
			if spec.ClientCertAuth == nil || !reflect.DeepEqual(spec.ClientCertAuth.Organizations, assertion.organizations) {
				t.Errorf("getKubeConfigSpecs for %s Organizations is %v, expected %v", assertion.kubeConfigFile, spec.ClientCertAuth.Organizations, assertion.organizations)
			}

			// Asserts MasterConfiguration values injected into spec
			if spec.APIServer != cfg.GetMasterEndpoint() {
				t.Errorf("getKubeConfigSpecs didn't injected cfg.APIServer address into spec for %s", assertion.kubeConfigFile)
			}

			// Asserts CA certs and CA keys loaded into specs
			if spec.CaCert == nil {
				t.Errorf("getKubeConfigSpecs didn't loaded CaCert into spec for %s!", assertion.kubeConfigFile)
			}
			if spec.ClientCertAuth == nil || spec.ClientCertAuth.CaKey == nil {
				t.Errorf("getKubeConfigSpecs didn't loaded CaKey into spec for %s!", assertion.kubeConfigFile)
			}
		} else {
			t.Errorf("getKubeConfigSpecs didn't create spec for %s ", assertion.kubeConfigFile)
		}
	}
}

func TestBuildKubeConfigFromSpecWithClientAuth(t *testing.T) {
	// Creates a CA
	caCert, caKey := certstestutil.SetupCertificateAuthorithy(t)

	// Executes buildKubeConfigFromSpec passing a KubeConfigSpec wiht a ClientAuth
	config := setupdKubeConfigWithClientAuth(t, caCert, caKey, "https://1.2.3.4:1234", "myClientName", "myOrg1", "myOrg2")

	// Asserts spec data are propagated to the kubeconfig
	kubeconfigtestutil.AssertKubeConfigCurrentCluster(t, config, "https://1.2.3.4:1234", caCert)
	kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithClientCert(t, config, caCert, "myClientName", "myOrg1", "myOrg2")
}

func TestBuildKubeConfigFromSpecWithTokenAuth(t *testing.T) {
	// Creates a CA
	caCert, _ := certstestutil.SetupCertificateAuthorithy(t)

	// Executes buildKubeConfigFromSpec passing a KubeConfigSpec wiht a Token
	config := setupdKubeConfigWithTokenAuth(t, caCert, "https://1.2.3.4:1234", "myClientName", "123456")

	// Asserts spec data are propagated to the kubeconfig
	kubeconfigtestutil.AssertKubeConfigCurrentCluster(t, config, "https://1.2.3.4:1234", caCert)
	kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithToken(t, config, "myClientName", "123456")
}

func TestCreateKubeConfigFileIfNotExists(t *testing.T) {

	// Creates a CAs
	caCert, caKey := certstestutil.SetupCertificateAuthorithy(t)
	anotherCaCert, anotherCaKey := certstestutil.SetupCertificateAuthorithy(t)

	// build kubeconfigs (to be used to test kubeconfigs equality/not equality)
	config := setupdKubeConfigWithClientAuth(t, caCert, caKey, "https://1.2.3.4:1234", "myOrg1", "myOrg2")
	configWithAnotherClusterCa := setupdKubeConfigWithClientAuth(t, anotherCaCert, anotherCaKey, "https://1.2.3.4:1234", "myOrg1", "myOrg2")
	configWithAnotherClusterAddress := setupdKubeConfigWithClientAuth(t, caCert, caKey, "https://3.4.5.6:3456", "myOrg1", "myOrg2")

	var tests = []struct {
		existingKubeConfig *clientcmdapi.Config
		kubeConfig         *clientcmdapi.Config
		expectedError      bool
	}{
		{ // if there is no existing KubeConfig, creates the kubeconfig
			kubeConfig: config,
		},
		{ // if KubeConfig is equal to the existingKubeConfig - refers to the same cluster -, use the existing (Test idempotency)
			existingKubeConfig: config,
			kubeConfig:         config,
		},
		{ // if KubeConfig is not equal to the existingKubeConfig - refers to the another cluster (a cluster with another Ca) -, raise error
			existingKubeConfig: config,
			kubeConfig:         configWithAnotherClusterCa,
			expectedError:      true,
		},
		{ // if KubeConfig is not equal to the existingKubeConfig - refers to the another cluster (a cluster with another address) -, raise error
			existingKubeConfig: config,
			kubeConfig:         configWithAnotherClusterAddress,
			expectedError:      true,
		},
	}

	for _, test := range tests {
		// Create temp folder for the test case
		tmpdir := testutil.SetupTempDir(t)
		defer os.RemoveAll(tmpdir)

		// Writes the existing kubeconfig file to disk
		if test.existingKubeConfig != nil {
			if err := createKubeConfigFileIfNotExists(tmpdir, "test.conf", test.existingKubeConfig); err != nil {
				t.Errorf("createKubeConfigFileIfNotExists failed")
			}
		}

		// Writes the KubeConfig file to disk
		err := createKubeConfigFileIfNotExists(tmpdir, "test.conf", test.kubeConfig)
		if test.expectedError && err == nil {
			t.Errorf("createKubeConfigFileIfNotExists didn't failed when expected to fail")
		}
		if !test.expectedError && err != nil {
			t.Errorf("createKubeConfigFileIfNotExists failed")
		}

		// Assert creted files is there
		testutil.AssertFileExists(t, tmpdir, "test.conf")
	}
}

func TestCreateKubeconfigFilesAndWrappers(t *testing.T) {
	var tests = []struct {
		createKubeConfigFunction func(outDir string, cfg *kubeadmapi.MasterConfiguration) error
		expectedFiles            []string
		expectedError            bool
	}{
		{ // Test createKubeConfigFiles fails for unknown kubeconfig is requested
			createKubeConfigFunction: func(outDir string, cfg *kubeadmapi.MasterConfiguration) error {
				return createKubeConfigFiles(outDir, cfg, "unknown.conf")
			},
			expectedError: true,
		},
		{ // Test CreateInitKubeConfigFiles (wrapper to createKubeConfigFile)
			createKubeConfigFunction: CreateInitKubeConfigFiles,
			expectedFiles: []string{
				kubeadmconstants.AdminKubeConfigFileName,
				kubeadmconstants.KubeletKubeConfigFileName,
				kubeadmconstants.ControllerManagerKubeConfigFileName,
				kubeadmconstants.SchedulerKubeConfigFileName,
			},
		},
		{ // Test CreateAdminKubeConfigFile (wrapper to createKubeConfigFile)
			createKubeConfigFunction: CreateAdminKubeConfigFile,
			expectedFiles:            []string{kubeadmconstants.AdminKubeConfigFileName},
		},
		{ // Test CreateKubeletKubeConfigFile (wrapper to createKubeConfigFile)
			createKubeConfigFunction: CreateKubeletKubeConfigFile,
			expectedFiles:            []string{kubeadmconstants.KubeletKubeConfigFileName},
		},
		{ // Test CreateControllerManagerKubeConfigFile (wrapper to createKubeConfigFile)
			createKubeConfigFunction: CreateControllerManagerKubeConfigFile,
			expectedFiles:            []string{kubeadmconstants.ControllerManagerKubeConfigFileName},
		},
		{ // Test createKubeConfigFile (wrapper to createKubeConfigFile)
			createKubeConfigFunction: CreateSchedulerKubeConfigFile,
			expectedFiles:            []string{kubeadmconstants.SchedulerKubeConfigFileName},
		},
	}

	for _, test := range tests {
		// Create temp folder for the test case
		tmpdir := testutil.SetupTempDir(t)
		defer os.RemoveAll(tmpdir)

		// Adds a pki folder with a ca certs to the temp folder
		pkidir := testutil.SetupPkiDirWithCertificateAuthorithy(t, tmpdir)

		// Creates a Master Configuration pointing to the pkidir folder
		cfg := &kubeadmapi.MasterConfiguration{
			API:             kubeadmapi.API{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
			CertificatesDir: pkidir,
		}

		// Execs the createKubeConfigFunction
		err := test.createKubeConfigFunction(tmpdir, cfg)
		if test.expectedError && err == nil {
			t.Errorf("createKubeConfigFunction didn't failed when expected to fail")
			continue
		}
		if !test.expectedError && err != nil {
			t.Errorf("createKubeConfigFunction failed")
			continue
		}

		// Assert expected files are there
		testutil.AssertFileExists(t, tmpdir, test.expectedFiles...)
	}
}

func TestWriteKubeConfigFailsIfCADoesntExists(t *testing.T) {

	// Temporary folders for the test case (without a CA)
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	// Creates a Master Configuration pointing to the tmpdir folder
	cfg := &kubeadmapi.MasterConfiguration{
		CertificatesDir: tmpdir,
	}

	var tests = []struct {
		writeKubeConfigFunction func(out io.Writer) error
	}{
		{ // Test WriteKubeConfigWithClientCert
			writeKubeConfigFunction: func(out io.Writer) error {
				return WriteKubeConfigWithClientCert(out, cfg, "myUser")
			},
		},
		{ // Test WriteKubeConfigWithToken
			writeKubeConfigFunction: func(out io.Writer) error {
				return WriteKubeConfigWithToken(out, cfg, "myUser", "12345")
			},
		},
	}

	for _, test := range tests {
		buf := new(bytes.Buffer)

		// executes writeKubeConfigFunction
		if err := test.writeKubeConfigFunction(buf); err == nil {
			t.Error("writeKubeConfigFunction didnt failed when expected")
		}
	}
}

func TestWriteKubeConfig(t *testing.T) {

	// Temporary folders for the test case
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	// Adds a pki folder with a ca cert to the temp folder
	pkidir := testutil.SetupPkiDirWithCertificateAuthorithy(t, tmpdir)

	// Retrives ca cert for assertions
	caCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(pkidir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		t.Fatalf("couldn't retrive ca cert: %v", err)
	}

	// Creates a Master Configuration pointing to the pkidir folder
	cfg := &kubeadmapi.MasterConfiguration{
		API:             kubeadmapi.API{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
		CertificatesDir: pkidir,
	}

	var tests = []struct {
		writeKubeConfigFunction func(out io.Writer) error
		withClientCert          bool
		withToken               bool
	}{
		{ // Test WriteKubeConfigWithClientCert
			writeKubeConfigFunction: func(out io.Writer) error {
				return WriteKubeConfigWithClientCert(out, cfg, "myUser")
			},
			withClientCert: true,
		},
		{ // Test WriteKubeConfigWithToken
			writeKubeConfigFunction: func(out io.Writer) error {
				return WriteKubeConfigWithToken(out, cfg, "myUser", "12345")
			},
			withToken: true,
		},
	}

	for _, test := range tests {
		buf := new(bytes.Buffer)

		// executes writeKubeConfigFunction
		if err := test.writeKubeConfigFunction(buf); err != nil {
			t.Error("writeKubeConfigFunction failed")
			continue
		}

		// reads kubeconfig written to stdout
		config, err := clientcmd.Load(buf.Bytes())
		if err != nil {
			t.Errorf("Couldn't read kubeconfig file from buffer: %v", err)
			continue
		}

		// checks that CLI flags are properly propagated
		kubeconfigtestutil.AssertKubeConfigCurrentCluster(t, config, "https://1.2.3.4:1234", caCert)

		if test.withClientCert {
			// checks that kubeconfig files have expected client cert
			kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithClientCert(t, config, caCert, "myUser")
		}

		if test.withToken {
			// checks that kubeconfig files have expected token
			kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithToken(t, config, "myUser", "12345")
		}
	}
}

// setupdKubeConfigWithClientAuth is a test utility function that wraps buildKubeConfigFromSpec for building a KubeConfig object With ClientAuth
func setupdKubeConfigWithClientAuth(t *testing.T, caCert *x509.Certificate, caKey *rsa.PrivateKey, APIServer, clientName string, organizations ...string) *clientcmdapi.Config {
	spec := &kubeConfigSpec{
		CaCert:     caCert,
		APIServer:  APIServer,
		ClientName: clientName,
		ClientCertAuth: &clientCertAuth{
			CaKey:         caKey,
			Organizations: organizations,
		},
	}

	config, err := buildKubeConfigFromSpec(spec)
	if err != nil {
		t.Fatal("buildKubeConfigFromSpec failed!")
	}

	return config
}

// setupdKubeConfigWithClientAuth is a test utility function that wraps buildKubeConfigFromSpec for building a KubeConfig object With Token
func setupdKubeConfigWithTokenAuth(t *testing.T, caCert *x509.Certificate, APIServer, clientName, token string) *clientcmdapi.Config {
	spec := &kubeConfigSpec{
		CaCert:     caCert,
		APIServer:  APIServer,
		ClientName: clientName,
		TokenAuth: &tokenAuth{
			Token: token,
		},
	}

	config, err := buildKubeConfigFromSpec(spec)
	if err != nil {
		t.Fatal("buildKubeConfigFromSpec failed!")
	}

	return config
}
