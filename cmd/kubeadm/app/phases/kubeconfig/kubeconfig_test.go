/*
Copyright 2018 The Kubernetes Authors.

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
	"crypto"
	"crypto/x509"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/lithammer/dedent"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	certstestutil "k8s.io/kubernetes/cmd/kubeadm/app/util/certs"
	pkiutil "k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
	kubeconfigtestutil "k8s.io/kubernetes/cmd/kubeadm/test/kubeconfig"
)

func TestGetKubeConfigSpecsFailsIfCADoesntExists(t *testing.T) {
	// Create temp folder for the test case (without a CA)
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	// Creates an InitConfiguration pointing to the pkidir folder
	cfg := &kubeadmapi.InitConfiguration{
		ClusterConfiguration: kubeadmapi.ClusterConfiguration{
			CertificatesDir: tmpdir,
		},
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
	pkidir := testutil.SetupPkiDirWithCertificateAuthority(t, tmpdir)

	// Creates InitConfigurations pointing to the pkidir folder
	cfgs := []*kubeadmapi.InitConfiguration{
		{
			LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
			ClusterConfiguration: kubeadmapi.ClusterConfiguration{
				CertificatesDir: pkidir,
			},
			NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: "valid-node-name"},
		},
		{
			LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
			ClusterConfiguration: kubeadmapi.ClusterConfiguration{
				ControlPlaneEndpoint: "api.k8s.io",
				CertificatesDir:      pkidir,
			},
			NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: "valid-node-name"},
		},
		{
			LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
			ClusterConfiguration: kubeadmapi.ClusterConfiguration{
				ControlPlaneEndpoint: "api.k8s.io:4321",
				CertificatesDir:      pkidir,
			},
			NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: "valid-node-name"},
		},
		{
			LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
			ClusterConfiguration: kubeadmapi.ClusterConfiguration{
				ControlPlaneEndpoint: "api.k8s.io",
				CertificatesDir:      pkidir,
			},
			NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: "valid-node-name"},
		},
		{
			LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
			ClusterConfiguration: kubeadmapi.ClusterConfiguration{
				ControlPlaneEndpoint: "api.k8s.io:4321",
				CertificatesDir:      pkidir,
			},
			NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: "valid-node-name"},
		},
	}

	for i, cfg := range cfgs {
		var assertions = []struct {
			kubeConfigFile string
			clientName     string
			organizations  []string
		}{
			{
				kubeConfigFile: kubeadmconstants.AdminKubeConfigFileName,
				clientName:     "kubernetes-admin",
				organizations:  []string{kubeadmconstants.SystemPrivilegedGroup},
			},
			{
				kubeConfigFile: kubeadmconstants.KubeletKubeConfigFileName,
				clientName:     fmt.Sprintf("%s%s", kubeadmconstants.NodesUserPrefix, cfg.NodeRegistration.Name),
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
			t.Run(fmt.Sprintf("%d-%s", i, assertion.clientName), func(t *testing.T) {
				// Executes getKubeConfigSpecs
				specs, err := getKubeConfigSpecs(cfg)
				if err != nil {
					t.Fatal("getKubeConfigSpecs failed!")
				}

				var spec *kubeConfigSpec
				var ok bool

				// assert the spec for the kubeConfigFile exists
				if spec, ok = specs[assertion.kubeConfigFile]; !ok {
					t.Errorf("getKubeConfigSpecs didn't create spec for %s ", assertion.kubeConfigFile)
					return
				}

				// Assert clientName
				if spec.ClientName != assertion.clientName {
					t.Errorf("getKubeConfigSpecs for %s clientName is %s, expected %s", assertion.kubeConfigFile, spec.ClientName, assertion.clientName)
				}

				// Assert Organizations
				if spec.ClientCertAuth == nil || !reflect.DeepEqual(spec.ClientCertAuth.Organizations, assertion.organizations) {
					t.Errorf("getKubeConfigSpecs for %s Organizations is %v, expected %v", assertion.kubeConfigFile, spec.ClientCertAuth.Organizations, assertion.organizations)
				}

				// Asserts InitConfiguration values injected into spec
				controlPlaneEndpoint, err := kubeadmutil.GetControlPlaneEndpoint(cfg.ControlPlaneEndpoint, &cfg.LocalAPIEndpoint)
				if err != nil {
					t.Error(err)
				}
				localAPIEndpoint, err := kubeadmutil.GetLocalAPIEndpoint(&cfg.LocalAPIEndpoint)
				if err != nil {
					t.Error(err)
				}

				switch assertion.kubeConfigFile {
				case kubeadmconstants.AdminKubeConfigFileName, kubeadmconstants.KubeletKubeConfigFileName:
					if spec.APIServer != controlPlaneEndpoint {
						t.Errorf("expected getKubeConfigSpecs for %s to set cfg.APIServer to %s, got %s",
							assertion.kubeConfigFile, controlPlaneEndpoint, spec.APIServer)
					}
				case kubeadmconstants.ControllerManagerKubeConfigFileName, kubeadmconstants.SchedulerKubeConfigFileName:
					if spec.APIServer != localAPIEndpoint {
						t.Errorf("expected getKubeConfigSpecs for %s to set cfg.APIServer to %s, got %s",
							assertion.kubeConfigFile, localAPIEndpoint, spec.APIServer)
					}
				}

				// Asserts CA certs and CA keys loaded into specs
				if spec.CACert == nil {
					t.Errorf("getKubeConfigSpecs didn't loaded CACert into spec for %s!", assertion.kubeConfigFile)
				}
				if spec.ClientCertAuth == nil || spec.ClientCertAuth.CAKey == nil {
					t.Errorf("getKubeConfigSpecs didn't loaded CAKey into spec for %s!", assertion.kubeConfigFile)
				}
			})
		}
	}
}

func TestBuildKubeConfigFromSpecWithClientAuth(t *testing.T) {
	// Creates a CA
	caCert, caKey := certstestutil.SetupCertificateAuthority(t)

	// Executes buildKubeConfigFromSpec passing a KubeConfigSpec with a ClientAuth
	config := setupdKubeConfigWithClientAuth(t, caCert, caKey, "https://1.2.3.4:1234", "myClientName", "test-cluster", "myOrg1", "myOrg2")

	// Asserts spec data are propagated to the kubeconfig
	kubeconfigtestutil.AssertKubeConfigCurrentCluster(t, config, "https://1.2.3.4:1234", caCert)
	kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithClientCert(t, config, caCert, "myClientName", "myOrg1", "myOrg2")
}

func TestBuildKubeConfigFromSpecWithTokenAuth(t *testing.T) {
	// Creates a CA
	caCert, _ := certstestutil.SetupCertificateAuthority(t)

	// Executes buildKubeConfigFromSpec passing a KubeConfigSpec with a Token
	config := setupdKubeConfigWithTokenAuth(t, caCert, "https://1.2.3.4:1234", "myClientName", "123456", "test-cluster")

	// Asserts spec data are propagated to the kubeconfig
	kubeconfigtestutil.AssertKubeConfigCurrentCluster(t, config, "https://1.2.3.4:1234", caCert)
	kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithToken(t, config, "myClientName", "123456")
}

func TestCreateKubeConfigFileIfNotExists(t *testing.T) {

	// Creates a CAs
	caCert, caKey := certstestutil.SetupCertificateAuthority(t)
	anotherCaCert, anotherCaKey := certstestutil.SetupCertificateAuthority(t)

	// build kubeconfigs (to be used to test kubeconfigs equality/not equality)
	config := setupdKubeConfigWithClientAuth(t, caCert, caKey, "https://1.2.3.4:1234", "test-cluster", "myOrg1", "myOrg2")
	configWithAnotherClusterCa := setupdKubeConfigWithClientAuth(t, anotherCaCert, anotherCaKey, "https://1.2.3.4:1234", "test-cluster", "myOrg1", "myOrg2")
	configWithAnotherClusterAddress := setupdKubeConfigWithClientAuth(t, caCert, caKey, "https://3.4.5.6:3456", "myOrg1", "test-cluster", "myOrg2")
	invalidConfig := setupdKubeConfigWithClientAuth(t, caCert, caKey, "https://1.2.3.4:1234", "test-cluster", "myOrg1", "myOrg2")
	invalidConfig.CurrentContext = "invalid context"

	var tests = []struct {
		name               string
		existingKubeConfig *clientcmdapi.Config
		kubeConfig         *clientcmdapi.Config
		expectedError      bool
	}{
		{ // if there is no existing KubeConfig, creates the kubeconfig
			name:       "KubeConfig doesn't exist",
			kubeConfig: config,
		},
		{ // if KubeConfig is invalid raise error
			name:               "KubeConfig is invalid",
			existingKubeConfig: invalidConfig,
			kubeConfig:         invalidConfig,
			expectedError:      true,
		},
		{ // if KubeConfig is equal to the existingKubeConfig - refers to the same cluster -, use the existing (Test idempotency)
			name:               "KubeConfig refers to the same cluster",
			existingKubeConfig: config,
			kubeConfig:         config,
		},
		{ // if KubeConfig is not equal to the existingKubeConfig - refers to the another cluster (a cluster with another Ca) -, raise error
			name:               "KubeConfig refers to the cluster with another CA",
			existingKubeConfig: config,
			kubeConfig:         configWithAnotherClusterCa,
			expectedError:      true,
		},
		{ // if KubeConfig is not equal to the existingKubeConfig - tolerate custom server addresses
			name:               "KubeConfig referst to the cluster with another address",
			existingKubeConfig: config,
			kubeConfig:         configWithAnotherClusterAddress,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Create temp folder for the test case
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

			// Writes the existing kubeconfig file to disk
			if test.existingKubeConfig != nil {
				if err := createKubeConfigFileIfNotExists(tmpdir, "test.conf", test.existingKubeConfig); err != nil {
					t.Errorf("createKubeConfigFileIfNotExists failed")
				}
			}

			// Writes the kubeconfig file to disk
			err := createKubeConfigFileIfNotExists(tmpdir, "test.conf", test.kubeConfig)
			if test.expectedError && err == nil {
				t.Errorf("createKubeConfigFileIfNotExists didn't failed when expected to fail")
			}
			if !test.expectedError && err != nil {
				t.Errorf("createKubeConfigFileIfNotExists failed")
			}

			// Assert that the created file is there
			testutil.AssertFileExists(t, tmpdir, "test.conf")
		})
	}
}

func TestCreateKubeconfigFilesAndWrappers(t *testing.T) {
	var tests = []struct {
		name                     string
		createKubeConfigFunction func(outDir string, cfg *kubeadmapi.InitConfiguration) error
		expectedFiles            []string
		expectedError            bool
	}{
		{ // Test createKubeConfigFiles fails for unknown kubeconfig is requested
			name: "createKubeConfigFiles",
			createKubeConfigFunction: func(outDir string, cfg *kubeadmapi.InitConfiguration) error {
				return createKubeConfigFiles(outDir, cfg, "unknown.conf")
			},
			expectedError: true,
		},
		{ // Test CreateJoinControlPlaneKubeConfigFiles (wrapper to createKubeConfigFile)
			name:                     "CreateJoinControlPlaneKubeConfigFiles",
			createKubeConfigFunction: CreateJoinControlPlaneKubeConfigFiles,
			expectedFiles: []string{
				kubeadmconstants.AdminKubeConfigFileName,
				kubeadmconstants.ControllerManagerKubeConfigFileName,
				kubeadmconstants.SchedulerKubeConfigFileName,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Create temp folder for the test case
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

			// Adds a pki folder with a ca certs to the temp folder
			pkidir := testutil.SetupPkiDirWithCertificateAuthority(t, tmpdir)

			// Creates an InitConfiguration pointing to the pkidir folder
			cfg := &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					CertificatesDir: pkidir,
				},
			}

			// Execs the createKubeConfigFunction
			err := test.createKubeConfigFunction(tmpdir, cfg)
			if test.expectedError && err == nil {
				t.Errorf("createKubeConfigFunction didn't failed when expected to fail")
				return
			}
			if !test.expectedError && err != nil {
				t.Errorf("createKubeConfigFunction failed")
				return
			}

			// Assert expected files are there
			testutil.AssertFileExists(t, tmpdir, test.expectedFiles...)
		})
	}
}

func TestWriteKubeConfigFailsIfCADoesntExists(t *testing.T) {
	// Temporary folders for the test case (without a CA)
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	// Creates an InitConfiguration pointing to the tmpdir folder
	cfg := &kubeadmapi.InitConfiguration{
		ClusterConfiguration: kubeadmapi.ClusterConfiguration{
			CertificatesDir: tmpdir,
		},
	}

	var tests = []struct {
		name                    string
		writeKubeConfigFunction func(out io.Writer) error
	}{
		{
			name: "WriteKubeConfigWithClientCert",
			writeKubeConfigFunction: func(out io.Writer) error {
				return WriteKubeConfigWithClientCert(out, cfg, "myUser", []string{"myOrg"}, nil)
			},
		},
		{
			name: "WriteKubeConfigWithToken",
			writeKubeConfigFunction: func(out io.Writer) error {
				return WriteKubeConfigWithToken(out, cfg, "myUser", "12345", nil)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			buf := new(bytes.Buffer)

			// executes writeKubeConfigFunction
			if err := test.writeKubeConfigFunction(buf); err == nil {
				t.Error("writeKubeConfigFunction didnt failed when expected")
			}
		})
	}
}

func TestWriteKubeConfig(t *testing.T) {
	// Temporary folders for the test case
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	// Adds a pki folder with a ca cert to the temp folder
	pkidir := testutil.SetupPkiDirWithCertificateAuthority(t, tmpdir)

	// Retrieves ca cert for assertions
	caCert, _, err := pkiutil.TryLoadCertAndKeyFromDisk(pkidir, kubeadmconstants.CACertAndKeyBaseName)
	if err != nil {
		t.Fatalf("couldn't retrieve ca cert: %v", err)
	}

	// Creates an InitConfiguration pointing to the pkidir folder
	cfg := &kubeadmapi.InitConfiguration{
		LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4", BindPort: 1234},
		ClusterConfiguration: kubeadmapi.ClusterConfiguration{
			CertificatesDir: pkidir,
		},
	}

	var tests = []struct {
		name                    string
		writeKubeConfigFunction func(out io.Writer) error
		withClientCert          bool
		withToken               bool
	}{
		{
			name: "WriteKubeConfigWithClientCert",
			writeKubeConfigFunction: func(out io.Writer) error {
				return WriteKubeConfigWithClientCert(out, cfg, "myUser", []string{"myOrg"}, nil)
			},
			withClientCert: true,
		},
		{
			name: "WriteKubeConfigWithToken",
			writeKubeConfigFunction: func(out io.Writer) error {
				return WriteKubeConfigWithToken(out, cfg, "myUser", "12345", nil)
			},
			withToken: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			buf := new(bytes.Buffer)

			// executes writeKubeConfigFunction
			if err := test.writeKubeConfigFunction(buf); err != nil {
				t.Error("writeKubeConfigFunction failed")
				return
			}

			// reads kubeconfig written to stdout
			config, err := clientcmd.Load(buf.Bytes())
			if err != nil {
				t.Errorf("Couldn't read kubeconfig file from buffer: %v", err)
				return
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
		})
	}
}

func TestValidateKubeConfig(t *testing.T) {
	caCert, caKey := certstestutil.SetupCertificateAuthority(t)
	anotherCaCert, anotherCaKey := certstestutil.SetupCertificateAuthority(t)

	config := setupdKubeConfigWithClientAuth(t, caCert, caKey, "https://1.2.3.4:1234", "test-cluster", "myOrg1")
	configWithAnotherClusterCa := setupdKubeConfigWithClientAuth(t, anotherCaCert, anotherCaKey, "https://1.2.3.4:1234", "test-cluster", "myOrg1")
	configWithAnotherServerURL := setupdKubeConfigWithClientAuth(t, caCert, caKey, "https://4.3.2.1:4321", "test-cluster", "myOrg1")

	configWithSameClusterCaByExternalFile := config.DeepCopy()
	currentCtx, exists := configWithSameClusterCaByExternalFile.Contexts[configWithSameClusterCaByExternalFile.CurrentContext]
	if !exists {
		t.Fatal("failed to find CurrentContext in Contexts of the kubeconfig")
	}
	if configWithSameClusterCaByExternalFile.Clusters[currentCtx.Cluster] == nil {
		t.Fatal("failed to find the given CurrentContext Cluster in Clusters of the kubeconfig")
	}
	tmpfile, err := ioutil.TempFile("", "external-ca.crt")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpfile.Name())
	if _, err := tmpfile.Write(pkiutil.EncodeCertPEM(caCert)); err != nil {
		t.Fatal(err)
	}
	configWithSameClusterCaByExternalFile.Clusters[currentCtx.Cluster].CertificateAuthorityData = nil
	configWithSameClusterCaByExternalFile.Clusters[currentCtx.Cluster].CertificateAuthority = tmpfile.Name()

	// create a valid config but with whitespace around the CA PEM.
	// validateKubeConfig() should tolerate that.
	configWhitespace := config.DeepCopy()
	configWhitespaceCtx := configWhitespace.Contexts[configWhitespace.CurrentContext]
	configWhitespaceCA := string(configWhitespace.Clusters[configWhitespaceCtx.Cluster].CertificateAuthorityData)
	configWhitespaceCA = "\n" + configWhitespaceCA + "\n"
	configWhitespace.Clusters[configWhitespaceCtx.Cluster].CertificateAuthorityData = []byte(configWhitespaceCA)

	tests := map[string]struct {
		existingKubeConfig *clientcmdapi.Config
		kubeConfig         *clientcmdapi.Config
		expectedError      bool
	}{
		"kubeconfig don't exist": {
			kubeConfig:    config,
			expectedError: true,
		},
		"kubeconfig exist and has invalid ca": {
			existingKubeConfig: configWithAnotherClusterCa,
			kubeConfig:         config,
			expectedError:      true,
		},
		"kubeconfig exist and has a different server url": {
			existingKubeConfig: configWithAnotherServerURL,
			kubeConfig:         config,
		},
		"kubeconfig exist and is valid": {
			existingKubeConfig: config,
			kubeConfig:         config,
			expectedError:      false,
		},
		"kubeconfig exist and is valid even if its CA contains whitespace": {
			existingKubeConfig: configWhitespace,
			kubeConfig:         config,
			expectedError:      false,
		},
		"kubeconfig exist and is valid even if its CA is provided as an external file": {
			existingKubeConfig: configWithSameClusterCaByExternalFile,
			kubeConfig:         config,
			expectedError:      false,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

			if test.existingKubeConfig != nil {
				if err := createKubeConfigFileIfNotExists(tmpdir, "test.conf", test.existingKubeConfig); err != nil {
					t.Errorf("createKubeConfigFileIfNotExists failed")
				}
			}

			err := validateKubeConfig(tmpdir, "test.conf", test.kubeConfig)
			if (err != nil) != test.expectedError {
				t.Fatalf(dedent.Dedent(
					"validateKubeConfig failed\n%s\nexpected error: %t\n\tgot: %t\nerror: %v"),
					name,
					test.expectedError,
					(err != nil),
					err,
				)
			}
		})
	}
}

func TestValidateKubeconfigsForExternalCA(t *testing.T) {
	tmpDir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpDir)
	pkiDir := filepath.Join(tmpDir, "pki")

	initConfig := &kubeadmapi.InitConfiguration{
		ClusterConfiguration: kubeadmapi.ClusterConfiguration{
			CertificatesDir: pkiDir,
		},
		LocalAPIEndpoint: kubeadmapi.APIEndpoint{
			BindPort:         1234,
			AdvertiseAddress: "1.2.3.4",
		},
	}

	// creates CA, write to pkiDir and remove ca.key to get into external CA condition
	caCert, caKey := certstestutil.SetupCertificateAuthority(t)
	if err := pkiutil.WriteCertAndKey(pkiDir, kubeadmconstants.CACertAndKeyBaseName, caCert, caKey); err != nil {
		t.Fatalf("failure while saving CA certificate and key: %v", err)
	}
	if err := os.Remove(filepath.Join(pkiDir, kubeadmconstants.CAKeyName)); err != nil {
		t.Fatalf("failure while deleting ca.key: %v", err)
	}

	// create a valid config
	config := setupdKubeConfigWithClientAuth(t, caCert, caKey, "https://1.2.3.4:1234", "test-cluster", "myOrg1")

	// create a config with another CA
	anotherCaCert, anotherCaKey := certstestutil.SetupCertificateAuthority(t)
	configWithAnotherClusterCa := setupdKubeConfigWithClientAuth(t, anotherCaCert, anotherCaKey, "https://1.2.3.4:1234", "test-cluster", "myOrg1")

	// create a config with another server URL
	configWithAnotherServerURL := setupdKubeConfigWithClientAuth(t, caCert, caKey, "https://4.3.2.1:4321", "test-cluster", "myOrg1")

	tests := map[string]struct {
		filesToWrite  map[string]*clientcmdapi.Config
		initConfig    *kubeadmapi.InitConfiguration
		expectedError bool
	}{
		"files don't exist": {
			initConfig:    initConfig,
			expectedError: true,
		},
		"some files don't exist": {
			filesToWrite: map[string]*clientcmdapi.Config{
				kubeadmconstants.AdminKubeConfigFileName:   config,
				kubeadmconstants.KubeletKubeConfigFileName: config,
			},
			initConfig:    initConfig,
			expectedError: true,
		},
		"some files have invalid CA": {
			filesToWrite: map[string]*clientcmdapi.Config{
				kubeadmconstants.AdminKubeConfigFileName:             config,
				kubeadmconstants.KubeletKubeConfigFileName:           config,
				kubeadmconstants.ControllerManagerKubeConfigFileName: configWithAnotherClusterCa,
				kubeadmconstants.SchedulerKubeConfigFileName:         config,
			},
			initConfig:    initConfig,
			expectedError: true,
		},
		"some files have a different Server URL": {
			filesToWrite: map[string]*clientcmdapi.Config{
				kubeadmconstants.AdminKubeConfigFileName:             config,
				kubeadmconstants.KubeletKubeConfigFileName:           config,
				kubeadmconstants.ControllerManagerKubeConfigFileName: config,
				kubeadmconstants.SchedulerKubeConfigFileName:         configWithAnotherServerURL,
			},
			initConfig: initConfig,
		},
		"all files are valid": {
			filesToWrite: map[string]*clientcmdapi.Config{
				kubeadmconstants.AdminKubeConfigFileName:             config,
				kubeadmconstants.KubeletKubeConfigFileName:           config,
				kubeadmconstants.ControllerManagerKubeConfigFileName: config,
				kubeadmconstants.SchedulerKubeConfigFileName:         config,
			},
			initConfig:    initConfig,
			expectedError: false,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

			for name, config := range test.filesToWrite {
				if err := createKubeConfigFileIfNotExists(tmpdir, name, config); err != nil {
					t.Errorf("createKubeConfigFileIfNotExists failed: %v", err)
				}
			}

			err := ValidateKubeconfigsForExternalCA(tmpdir, test.initConfig)
			if (err != nil) != test.expectedError {
				t.Fatalf(dedent.Dedent(
					"ValidateKubeconfigsForExternalCA failed\n%s\nexpected error: %t\n\tgot: %t\nerror: %v"),
					name,
					test.expectedError,
					(err != nil),
					err,
				)
			}
		})
	}
}

// setupdKubeConfigWithClientAuth is a test utility function that wraps buildKubeConfigFromSpec for building a KubeConfig object With ClientAuth
func setupdKubeConfigWithClientAuth(t *testing.T, caCert *x509.Certificate, caKey crypto.Signer, APIServer, clientName, clustername string, organizations ...string) *clientcmdapi.Config {
	spec := &kubeConfigSpec{
		CACert:     caCert,
		APIServer:  APIServer,
		ClientName: clientName,
		ClientCertAuth: &clientCertAuth{
			CAKey:         caKey,
			Organizations: organizations,
		},
	}

	config, err := buildKubeConfigFromSpec(spec, clustername, nil)
	if err != nil {
		t.Fatal("buildKubeConfigFromSpec failed!")
	}

	return config
}

// setupdKubeConfigWithClientAuth is a test utility function that wraps buildKubeConfigFromSpec for building a KubeConfig object With Token
func setupdKubeConfigWithTokenAuth(t *testing.T, caCert *x509.Certificate, APIServer, clientName, token, clustername string) *clientcmdapi.Config {
	spec := &kubeConfigSpec{
		CACert:     caCert,
		APIServer:  APIServer,
		ClientName: clientName,
		TokenAuth: &tokenAuth{
			Token: token,
		},
	}

	config, err := buildKubeConfigFromSpec(spec, clustername, nil)
	if err != nil {
		t.Fatal("buildKubeConfigFromSpec failed!")
	}

	return config
}
