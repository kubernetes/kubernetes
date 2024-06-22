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
	"context"
	"crypto"
	"crypto/x509"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"

	rbac "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clientgotesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	certstestutil "k8s.io/kubernetes/cmd/kubeadm/app/util/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
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
				CertificatesDir:     pkidir,
				EncryptionAlgorithm: kubeadmapi.EncryptionAlgorithmECDSAP256,
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
				organizations:  []string{kubeadmconstants.ClusterAdminsGroupAndClusterRoleBinding},
			},
			{
				kubeConfigFile: kubeadmconstants.SuperAdminKubeConfigFileName,
				clientName:     "kubernetes-super-admin",
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

				// Assert EncryptionAlgorithm
				if spec.EncryptionAlgorithm != cfg.EncryptionAlgorithm {
					t.Errorf("getKubeConfigSpecs for %s EncryptionAlgorithm is %s, expected %s", assertion.kubeConfigFile, spec.EncryptionAlgorithm, cfg.EncryptionAlgorithm)
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
				case kubeadmconstants.AdminKubeConfigFileName, kubeadmconstants.SuperAdminKubeConfigFileName, kubeadmconstants.KubeletKubeConfigFileName:
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

	notAfter, _ := time.Parse(time.RFC3339, "2026-01-02T15:04:05Z")

	// Executes buildKubeConfigFromSpec passing a KubeConfigSpec with a ClientAuth
	config := setupKubeConfigWithClientAuth(t, caCert, caKey, notAfter, "https://1.2.3.4:1234", "myClientName", "test-cluster", "myOrg1", "myOrg2")

	// Asserts spec data are propagated to the kubeconfig
	kubeconfigtestutil.AssertKubeConfigCurrentCluster(t, config, "https://1.2.3.4:1234", caCert)
	kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithClientCert(t, config, caCert, notAfter, "myClientName", "myOrg1", "myOrg2")
}

func TestBuildKubeConfigFromSpecWithTokenAuth(t *testing.T) {
	// Creates a CA
	caCert, _ := certstestutil.SetupCertificateAuthority(t)

	// Executes buildKubeConfigFromSpec passing a KubeConfigSpec with a Token
	config := setupKubeConfigWithTokenAuth(t, caCert, "https://1.2.3.4:1234", "myClientName", "123456", "test-cluster")

	// Asserts spec data are propagated to the kubeconfig
	kubeconfigtestutil.AssertKubeConfigCurrentCluster(t, config, "https://1.2.3.4:1234", caCert)
	kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithToken(t, config, "myClientName", "123456")
}

func TestCreateKubeConfigFileIfNotExists(t *testing.T) {

	// Creates a CAs
	caCert, caKey := certstestutil.SetupCertificateAuthority(t)
	anotherCaCert, anotherCaKey := certstestutil.SetupCertificateAuthority(t)

	notAfter, _ := time.Parse(time.RFC3339, "2026-01-02T15:04:05Z")

	// build kubeconfigs (to be used to test kubeconfigs equality/not equality)
	config := setupKubeConfigWithClientAuth(t, caCert, caKey, notAfter, "https://1.2.3.4:1234", "test-cluster", "myOrg1", "myOrg2")
	configWithAnotherClusterCa := setupKubeConfigWithClientAuth(t, anotherCaCert, anotherCaKey, notAfter, "https://1.2.3.4:1234", "test-cluster", "myOrg1", "myOrg2")
	configWithAnotherClusterAddress := setupKubeConfigWithClientAuth(t, caCert, caKey, notAfter, "https://3.4.5.6:3456", "myOrg1", "test-cluster", "myOrg2")
	invalidConfig := setupKubeConfigWithClientAuth(t, caCert, caKey, notAfter, "https://1.2.3.4:1234", "test-cluster", "myOrg1", "myOrg2")
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

	notAfter, _ := time.Parse(time.RFC3339, "2026-01-02T15:04:05Z")

	var tests = []struct {
		name                    string
		writeKubeConfigFunction func(out io.Writer) error
	}{
		{
			name: "WriteKubeConfigWithClientCert",
			writeKubeConfigFunction: func(out io.Writer) error {
				return WriteKubeConfigWithClientCert(out, cfg, "myUser", []string{"myOrg"}, notAfter)
			},
		},
		{
			name: "WriteKubeConfigWithToken",
			writeKubeConfigFunction: func(out io.Writer) error {
				return WriteKubeConfigWithToken(out, cfg, "myUser", "12345", notAfter)
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
			CertificateValidityPeriod: &metav1.Duration{
				Duration: time.Hour * 10,
			},
		},
	}

	notAfter, _ := time.Parse(time.RFC3339, "2026-01-02T15:04:05Z")

	var tests = []struct {
		name                    string
		writeKubeConfigFunction func(out io.Writer) error
		withClientCert          bool
		withToken               bool
	}{
		{
			name: "WriteKubeConfigWithClientCert",
			writeKubeConfigFunction: func(out io.Writer) error {
				return WriteKubeConfigWithClientCert(out, cfg, "myUser", []string{"myOrg"}, notAfter)
			},
			withClientCert: true,
		},
		{
			name: "WriteKubeConfigWithToken",
			writeKubeConfigFunction: func(out io.Writer) error {
				return WriteKubeConfigWithToken(out, cfg, "myUser", "12345", notAfter)
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
				kubeconfigtestutil.AssertKubeConfigCurrentAuthInfoWithClientCert(t, config, caCert, notAfter, "myUser", "myOrg")
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

	notAfter, _ := time.Parse(time.RFC3339, "2026-01-02T15:04:05Z")

	config := setupKubeConfigWithClientAuth(t, caCert, caKey, notAfter, "https://1.2.3.4:1234", "test-cluster", "myOrg1")
	configWithAnotherClusterCa := setupKubeConfigWithClientAuth(t, anotherCaCert, anotherCaKey, notAfter, "https://1.2.3.4:1234", "test-cluster", "myOrg1")
	configWithAnotherServerURL := setupKubeConfigWithClientAuth(t, caCert, caKey, notAfter, "https://4.3.2.1:4321", "test-cluster", "myOrg1")

	configWithSameClusterCaByExternalFile := config.DeepCopy()
	currentCtx, exists := configWithSameClusterCaByExternalFile.Contexts[configWithSameClusterCaByExternalFile.CurrentContext]
	if !exists {
		t.Fatal("failed to find CurrentContext in Contexts of the kubeconfig")
	}
	if configWithSameClusterCaByExternalFile.Clusters[currentCtx.Cluster] == nil {
		t.Fatal("failed to find the given CurrentContext Cluster in Clusters of the kubeconfig")
	}
	tmpfile, err := os.CreateTemp("", "external-ca.crt")
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

	notAfter, _ := time.Parse(time.RFC3339, "2026-01-02T15:04:05Z")

	// create a valid config
	config := setupKubeConfigWithClientAuth(t, caCert, caKey, notAfter, "https://1.2.3.4:1234", "test-cluster", "myOrg1")

	// create a config with another CA
	anotherCaCert, anotherCaKey := certstestutil.SetupCertificateAuthority(t)
	configWithAnotherClusterCa := setupKubeConfigWithClientAuth(t, anotherCaCert, anotherCaKey, notAfter, "https://1.2.3.4:1234", "test-cluster", "myOrg1")

	// create a config with another server URL
	configWithAnotherServerURL := setupKubeConfigWithClientAuth(t, caCert, caKey, notAfter, "https://4.3.2.1:4321", "test-cluster", "myOrg1")

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
				kubeadmconstants.AdminKubeConfigFileName:      config,
				kubeadmconstants.SuperAdminKubeConfigFileName: config,
				kubeadmconstants.KubeletKubeConfigFileName:    config,
			},
			initConfig:    initConfig,
			expectedError: true,
		},
		"some files have invalid CA": {
			filesToWrite: map[string]*clientcmdapi.Config{
				kubeadmconstants.AdminKubeConfigFileName:             config,
				kubeadmconstants.SuperAdminKubeConfigFileName:        config,
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
				kubeadmconstants.SuperAdminKubeConfigFileName:        config,
				kubeadmconstants.KubeletKubeConfigFileName:           config,
				kubeadmconstants.ControllerManagerKubeConfigFileName: config,
				kubeadmconstants.SchedulerKubeConfigFileName:         configWithAnotherServerURL,
			},
			initConfig: initConfig,
		},
		"all files are valid": {
			filesToWrite: map[string]*clientcmdapi.Config{
				kubeadmconstants.AdminKubeConfigFileName:             config,
				kubeadmconstants.SuperAdminKubeConfigFileName:        config,
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

// setupKubeConfigWithClientAuth is a test utility function that wraps buildKubeConfigFromSpec for building a KubeConfig object With ClientAuth
func setupKubeConfigWithClientAuth(t *testing.T, caCert *x509.Certificate, caKey crypto.Signer, notAfter time.Time, apiServer, clientName, clustername string, organizations ...string) *clientcmdapi.Config {
	spec := &kubeConfigSpec{
		CACert:     caCert,
		APIServer:  apiServer,
		ClientName: clientName,
		ClientCertAuth: &clientCertAuth{
			CAKey:         caKey,
			Organizations: organizations,
		},
		ClientCertNotAfter: notAfter,
	}

	config, err := buildKubeConfigFromSpec(spec, clustername)
	if err != nil {
		t.Fatal("buildKubeConfigFromSpec failed!")
	}

	return config
}

// setupKubeConfigWithClientAuth is a test utility function that wraps buildKubeConfigFromSpec for building a KubeConfig object With Token
func setupKubeConfigWithTokenAuth(t *testing.T, caCert *x509.Certificate, apiServer, clientName, token, clustername string) *clientcmdapi.Config {
	spec := &kubeConfigSpec{
		CACert:     caCert,
		APIServer:  apiServer,
		ClientName: clientName,
		TokenAuth: &tokenAuth{
			Token: token,
		},
	}

	config, err := buildKubeConfigFromSpec(spec, clustername)
	if err != nil {
		t.Fatal("buildKubeConfigFromSpec failed!")
	}

	return config
}

func TestEnsureAdminClusterRoleBinding(t *testing.T) {
	dir := testutil.SetupTempDir(t)
	defer os.RemoveAll(dir)

	cfg := testutil.GetDefaultInternalConfig(t)
	cfg.CertificatesDir = dir

	ca := certsphase.KubeadmCertRootCA()
	_, _, err := ca.CreateAsCA(cfg)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name                  string
		expectedRBACError     bool
		expectedError         bool
		missingAdminConf      bool
		missingSuperAdminConf bool
	}{
		{
			name: "no errors",
		},
		{
			name:              "expect RBAC error",
			expectedRBACError: true,
			expectedError:     true,
		},
		{
			name:             "admin.conf is missing",
			missingAdminConf: true,
			expectedError:    true,
		},
		{
			name:                  "super-admin.conf is missing",
			missingSuperAdminConf: true,
			expectedError:         false, // The file is optional.
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ensureRBACFunc := func(_ context.Context, adminClient clientset.Interface, superAdminClient clientset.Interface,
				_ time.Duration, _ time.Duration) (clientset.Interface, error) {

				if tc.expectedRBACError {
					return nil, errors.New("ensureRBACFunc error")
				}
				return adminClient, nil
			}

			// Create the admin.conf and super-admin.conf so that EnsureAdminClusterRoleBinding
			// can create clients from the files.
			os.Remove(filepath.Join(dir, kubeadmconstants.AdminKubeConfigFileName))
			if !tc.missingAdminConf {
				if err := CreateKubeConfigFile(kubeadmconstants.AdminKubeConfigFileName, dir, cfg); err != nil {
					t.Fatal(err)
				}
			}
			os.Remove(filepath.Join(dir, kubeadmconstants.SuperAdminKubeConfigFileName))
			if !tc.missingSuperAdminConf {
				if err := CreateKubeConfigFile(kubeadmconstants.SuperAdminKubeConfigFileName, dir, cfg); err != nil {
					t.Fatal(err)
				}
			}

			client, err := EnsureAdminClusterRoleBinding(dir, ensureRBACFunc)
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got: %v, error: %v", err != nil, tc.expectedError, err)
			}

			if err == nil && client == nil {
				t.Fatal("got nil client")
			}
		})
	}
}

func TestEnsureAdminClusterRoleBindingImpl(t *testing.T) {
	tests := []struct {
		name                  string
		setupAdminClient      func(*clientsetfake.Clientset)
		setupSuperAdminClient func(*clientsetfake.Clientset)
		expectedError         bool
	}{
		{
			name: "admin.conf: handle forbidden errors when the super-admin.conf client is nil",
			setupAdminClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(action clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewForbidden(
						schema.GroupResource{}, "name", errors.New(""))
				})
			},
			expectedError: true,
		},
		{
			// A "create" call against a real server can return a forbidden error and a non-nil CRB
			name: "admin.conf: handle forbidden error and returned CRBs, when the super-admin.conf client is nil",
			setupAdminClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(action clientgotesting.Action) (bool, runtime.Object, error) {
					return true, &rbac.ClusterRoleBinding{}, apierrors.NewForbidden(
						schema.GroupResource{}, "name", errors.New(""))
				})
			},
			expectedError: true,
		},
		{
			name: "admin.conf: CRB already exists, use the admin.conf client",
			setupAdminClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(action clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(
						schema.GroupResource{}, "name")
				})
			},
			setupSuperAdminClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(action clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(
						schema.GroupResource{}, "name")
				})
			},
			expectedError: false,
		},
		{
			name: "admin.conf: handle other errors, such as a server timeout",
			setupAdminClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(action clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewServerTimeout(
						schema.GroupResource{}, "create", 0)
				})
			},
			expectedError: true,
		},
		{
			name: "admin.conf: CRB exists, return a client from admin.conf",
			setupAdminClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(action clientgotesting.Action) (bool, runtime.Object, error) {
					return true, &rbac.ClusterRoleBinding{}, nil
				})
			},
			expectedError: false,
		},
		{
			name: "super-admin.conf: error while creating CRB",
			setupAdminClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(action clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewForbidden(
						schema.GroupResource{}, "name", errors.New(""))
				})
			},
			setupSuperAdminClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(action clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewServerTimeout(
						schema.GroupResource{}, "create", 0)
				})
			},
			expectedError: true,
		},
		{
			name: "super-admin.conf: admin.conf cannot create CRB, create CRB with super-admin.conf, return client from admin.conf",
			setupAdminClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(action clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewForbidden(
						schema.GroupResource{}, "name", errors.New(""))
				})
			},
			setupSuperAdminClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(action clientgotesting.Action) (bool, runtime.Object, error) {
					return true, &rbac.ClusterRoleBinding{}, nil
				})
			},
			expectedError: false,
		},
		{
			name: "super-admin.conf: admin.conf cannot create CRB, try to create CRB with super-admin.conf, encounter 'already exists' error",
			setupAdminClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(action clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewForbidden(
						schema.GroupResource{}, "name", errors.New(""))
				})
			},
			setupSuperAdminClient: func(client *clientsetfake.Clientset) {
				client.PrependReactor("create", "clusterrolebindings", func(action clientgotesting.Action) (bool, runtime.Object, error) {
					return true, nil, apierrors.NewAlreadyExists(
						schema.GroupResource{}, "name")
				})
			},
			expectedError: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			adminClient := clientsetfake.NewSimpleClientset()
			tc.setupAdminClient(adminClient)

			var superAdminClient clientset.Interface // ensure superAdminClient is nil by default
			if tc.setupSuperAdminClient != nil {
				fakeSuperAdminClient := clientsetfake.NewSimpleClientset()
				tc.setupSuperAdminClient(fakeSuperAdminClient)
				superAdminClient = fakeSuperAdminClient
			}

			client, err := EnsureAdminClusterRoleBindingImpl(
				context.Background(), adminClient, superAdminClient, 0, 0)
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got %v, error: %v", tc.expectedError, err != nil, err)
			}

			if err == nil && client == nil {
				t.Fatal("got nil client")
			}
		})
	}
}

func TestCreateKubeConfigAndCSR(t *testing.T) {
	tmpDir := testutil.SetupTempDir(t)
	testutil.SetupEmptyFiles(t, tmpDir, "testfile", "bar.csr", "bar.key")
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			t.Error(err)
		}
	}()
	caCert, caKey := certstestutil.SetupCertificateAuthority(t)

	type args struct {
		kubeConfigDir string
		kubeadmConfig *kubeadmapi.InitConfiguration
		name          string
		spec          *kubeConfigSpec
	}
	tests := []struct {
		name          string
		args          args
		expectedError bool
	}{
		{
			name: "kubeadmConfig is nil",
			args: args{
				kubeConfigDir: tmpDir,
				kubeadmConfig: nil,
				name:          "foo",
				spec: &kubeConfigSpec{
					CACert:         caCert,
					APIServer:      "10.0.0.1",
					ClientName:     "foo",
					TokenAuth:      &tokenAuth{Token: "test"},
					ClientCertAuth: &clientCertAuth{CAKey: caKey},
				},
			},
			expectedError: true,
		},
		{
			name: "The kubeConfigDir is empty",
			args: args{
				kubeConfigDir: "",
				kubeadmConfig: &kubeadmapi.InitConfiguration{},
				name:          "foo",
				spec: &kubeConfigSpec{
					CACert:         caCert,
					APIServer:      "10.0.0.1",
					ClientName:     "foo",
					TokenAuth:      &tokenAuth{Token: "test"},
					ClientCertAuth: &clientCertAuth{CAKey: caKey},
				},
			},
			expectedError: true,
		},
		{
			name: "The name is empty",
			args: args{
				kubeConfigDir: tmpDir,
				kubeadmConfig: &kubeadmapi.InitConfiguration{},
				name:          "",
				spec: &kubeConfigSpec{
					CACert:         caCert,
					APIServer:      "10.0.0.1",
					ClientName:     "foo",
					TokenAuth:      &tokenAuth{Token: "test"},
					ClientCertAuth: &clientCertAuth{CAKey: caKey},
				},
			},
			expectedError: true,
		},
		{
			name: "The spec is empty",
			args: args{
				kubeConfigDir: tmpDir,
				kubeadmConfig: &kubeadmapi.InitConfiguration{},
				name:          "foo",
				spec:          nil,
			},
			expectedError: true,
		},
		{
			name: "The kubeconfig file already exists",
			args: args{
				kubeConfigDir: tmpDir,
				kubeadmConfig: &kubeadmapi.InitConfiguration{},
				name:          "testfile",
				spec: &kubeConfigSpec{
					CACert:         caCert,
					APIServer:      "10.0.0.1",
					ClientName:     "foo",
					TokenAuth:      &tokenAuth{Token: "test"},
					ClientCertAuth: &clientCertAuth{CAKey: caKey},
				},
			},
			expectedError: true,
		},
		{
			name: "The CSR or key files already exists",
			args: args{
				kubeConfigDir: tmpDir,
				kubeadmConfig: &kubeadmapi.InitConfiguration{},
				name:          "bar",
				spec: &kubeConfigSpec{
					CACert:         caCert,
					APIServer:      "10.0.0.1",
					ClientName:     "foo",
					TokenAuth:      &tokenAuth{Token: "test"},
					ClientCertAuth: &clientCertAuth{CAKey: caKey},
				},
			},
			expectedError: true,
		},
		{
			name: "configuration is valid, expect no errors",
			args: args{
				kubeConfigDir: tmpDir,
				kubeadmConfig: &kubeadmapi.InitConfiguration{},
				name:          "test",
				spec: &kubeConfigSpec{
					CACert:         caCert,
					APIServer:      "10.0.0.1",
					ClientName:     "foo",
					TokenAuth:      &tokenAuth{Token: "test"},
					ClientCertAuth: &clientCertAuth{CAKey: caKey},
				},
			},
			expectedError: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if err := createKubeConfigAndCSR(tc.args.kubeConfigDir, tc.args.kubeadmConfig, tc.args.name, tc.args.spec); (err != nil) != tc.expectedError {
				t.Errorf("createKubeConfigAndCSR() error = %v, wantErr %v", err, tc.expectedError)
			}
		})
	}
}

func TestCreateDefaultKubeConfigsAndCSRFiles(t *testing.T) {
	tmpDir := testutil.SetupTempDir(t)
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			t.Error(err)
		}
	}()
	type args struct {
		kubeConfigDir string
		kubeadmConfig *kubeadmapi.InitConfiguration
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "kubeadmConfig is empty",
			args: args{
				kubeConfigDir: tmpDir,
				kubeadmConfig: &kubeadmapi.InitConfiguration{},
			},
			wantErr: true,
		},
		{
			name: "The APIEndpoint is invalid",
			args: args{
				kubeConfigDir: tmpDir,
				kubeadmConfig: &kubeadmapi.InitConfiguration{
					LocalAPIEndpoint: kubeadmapi.APIEndpoint{
						AdvertiseAddress: "x.12.FOo.1",
						BindPort:         6443,
					},
				},
			},
			wantErr: true,
		},
		{
			name: "The APIEndpoint is valid",
			args: args{
				kubeConfigDir: tmpDir,
				kubeadmConfig: &kubeadmapi.InitConfiguration{
					LocalAPIEndpoint: kubeadmapi.APIEndpoint{
						AdvertiseAddress: "127.0.0.1",
						BindPort:         6443,
					},
				},
			},
			wantErr: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out := &bytes.Buffer{}
			if err := CreateDefaultKubeConfigsAndCSRFiles(out, tc.args.kubeConfigDir, tc.args.kubeadmConfig); (err != nil) != tc.wantErr {
				t.Errorf("CreateDefaultKubeConfigsAndCSRFiles() error = %v, wantErr %v", err, tc.wantErr)
				return
			}
		})
	}
}
