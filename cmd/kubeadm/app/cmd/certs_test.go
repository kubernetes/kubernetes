//go:build !windows
// +build !windows

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

package cmd

import (
	"bytes"
	"crypto"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/lithammer/dedent"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	clientgotesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/clientcmd"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	outputapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/output/v1alpha3"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
	cmdtestutil "k8s.io/kubernetes/cmd/kubeadm/test/cmd"
)

func TestCommandsGenerated(t *testing.T) {
	expectedFlags := []string{
		"cert-dir",
		"config",
	}

	expectedCommands := []string{
		"renew all",

		"renew apiserver",
		"renew apiserver-kubelet-client",
		"renew apiserver-etcd-client",

		"renew front-proxy-client",

		"renew etcd-server",
		"renew etcd-peer",
		"renew etcd-healthcheck-client",

		"renew admin.conf",
		"renew scheduler.conf",
		"renew controller-manager.conf",
	}

	renewCmd := newCmdCertsRenewal(os.Stdout)

	fakeRoot := &cobra.Command{}
	fakeRoot.AddCommand(renewCmd)

	for _, cmdPath := range expectedCommands {
		t.Run(cmdPath, func(t *testing.T) {
			cmd, rem, _ := fakeRoot.Find(strings.Split(cmdPath, " "))
			if cmd == nil || len(rem) != 0 {
				t.Fatalf("couldn't locate command %q (%v)", cmdPath, rem)
			}

			for _, flag := range expectedFlags {
				if cmd.Flags().Lookup(flag) == nil {
					t.Errorf("couldn't find expected flag --%s", flag)
				}
			}
		})
	}
}

func TestRunRenewCommands(t *testing.T) {
	tmpDir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpDir)

	cfg := testutil.GetDefaultInternalConfig(t)
	cfg.CertificatesDir = tmpDir

	// Generate all the CA
	CACerts := map[string]*x509.Certificate{}
	CAKeys := map[string]crypto.Signer{}
	for _, ca := range []*certsphase.KubeadmCert{
		certsphase.KubeadmCertRootCA(),
		certsphase.KubeadmCertFrontProxyCA(),
		certsphase.KubeadmCertEtcdCA(),
	} {
		caCert, caKey, err := ca.CreateAsCA(cfg)
		if err != nil {
			t.Fatalf("couldn't write out CA %s: %v", ca.Name, err)
		}
		CACerts[ca.Name] = caCert
		CAKeys[ca.Name] = caKey
	}

	// Generate all the signed certificates
	for _, cert := range []*certsphase.KubeadmCert{
		certsphase.KubeadmCertAPIServer(),
		certsphase.KubeadmCertKubeletClient(),
		certsphase.KubeadmCertFrontProxyClient(),
		certsphase.KubeadmCertEtcdAPIClient(),
		certsphase.KubeadmCertEtcdServer(),
		certsphase.KubeadmCertEtcdPeer(),
		certsphase.KubeadmCertEtcdHealthcheck(),
	} {
		caCert := CACerts[cert.CAName]
		caKey := CAKeys[cert.CAName]
		if err := cert.CreateFromCA(cfg, caCert, caKey); err != nil {
			t.Fatalf("couldn't write certificate %s: %v", cert.Name, err)
		}
	}

	// Generate all the kubeconfig files with embedded certs
	for _, kubeConfig := range []string{
		kubeadmconstants.AdminKubeConfigFileName,
		kubeadmconstants.SuperAdminKubeConfigFileName,
		kubeadmconstants.SchedulerKubeConfigFileName,
		kubeadmconstants.ControllerManagerKubeConfigFileName,
	} {
		if err := kubeconfigphase.CreateKubeConfigFile(kubeConfig, tmpDir, cfg); err != nil {
			t.Fatalf("couldn't create kubeconfig %q: %v", kubeConfig, err)
		}
	}

	tests := []struct {
		command         string
		Certs           []*certsphase.KubeadmCert
		KubeconfigFiles []string
		Args            string
		expectedError   bool
	}{
		{
			command: "all",
			Certs: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertAPIServer(),
				certsphase.KubeadmCertKubeletClient(),
				certsphase.KubeadmCertFrontProxyClient(),
				certsphase.KubeadmCertEtcdAPIClient(),
				certsphase.KubeadmCertEtcdServer(),
				certsphase.KubeadmCertEtcdPeer(),
				certsphase.KubeadmCertEtcdHealthcheck(),
			},
			KubeconfigFiles: []string{
				kubeadmconstants.AdminKubeConfigFileName,
				kubeadmconstants.SuperAdminKubeConfigFileName,
				kubeadmconstants.SchedulerKubeConfigFileName,
				kubeadmconstants.ControllerManagerKubeConfigFileName,
			},
		},
		{
			command: "apiserver",
			Certs: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertAPIServer(),
			},
		},
		{
			command: "apiserver-kubelet-client",
			Certs: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertKubeletClient(),
			},
		},
		{
			command: "apiserver-etcd-client",
			Certs: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertEtcdAPIClient(),
			},
		},
		{
			command: "front-proxy-client",
			Certs: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertFrontProxyClient(),
			},
		},
		{
			command: "etcd-server",
			Certs: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertEtcdServer(),
			},
		},
		{
			command: "etcd-peer",
			Certs: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertEtcdPeer(),
			},
		},
		{
			command: "etcd-healthcheck-client",
			Certs: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertEtcdHealthcheck(),
			},
		},
		{
			command: "admin.conf",
			KubeconfigFiles: []string{
				kubeadmconstants.AdminKubeConfigFileName,
			},
		},
		{
			command: "super-admin.conf",
			KubeconfigFiles: []string{
				kubeadmconstants.SuperAdminKubeConfigFileName,
			},
		},
		{
			command: "scheduler.conf",
			KubeconfigFiles: []string{
				kubeadmconstants.SchedulerKubeConfigFileName,
			},
		},
		{
			command: "controller-manager.conf",
			KubeconfigFiles: []string{
				kubeadmconstants.ControllerManagerKubeConfigFileName,
			},
		},
		{
			command: "apiserver",
			Certs: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertAPIServer(),
			},
			Args:          "args",
			expectedError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.command, func(t *testing.T) {
			// Get file ModTime before renew
			ModTime := map[string]time.Time{}
			for _, cert := range test.Certs {
				file, err := os.Stat(filepath.Join(tmpDir, fmt.Sprintf("%s.crt", cert.BaseName)))
				if err != nil {
					t.Fatalf("couldn't get certificate %s: %v", cert.Name, err)
				}
				ModTime[cert.Name] = file.ModTime()
			}
			for _, kubeConfig := range test.KubeconfigFiles {
				file, err := os.Stat(filepath.Join(tmpDir, kubeConfig))
				if err != nil {
					t.Fatalf("couldn't get kubeconfig %s: %v", kubeConfig, err)
				}
				ModTime[kubeConfig] = file.ModTime()
			}

			// exec renew
			renewCmds := getRenewSubCommands(os.Stdout, tmpDir)
			args := fmt.Sprintf("--cert-dir=%s", tmpDir)
			if len(test.Args) > 0 {
				args = test.Args + " " + args
			}
			err := cmdtestutil.RunSubCommand(t, renewCmds, test.command, io.Discard, args)
			// certs renew doesn't support positional Args
			if (err != nil) != test.expectedError {
				t.Errorf("failed to run renew commands, expected error: %t, actual error: %v", test.expectedError, err)
			}
			if !test.expectedError {
				// check the file is modified
				for _, cert := range test.Certs {
					file, err := os.Stat(filepath.Join(tmpDir, fmt.Sprintf("%s.crt", cert.BaseName)))
					if err != nil {
						t.Fatalf("couldn't get certificate %s: %v", cert.Name, err)
					}
					if ModTime[cert.Name] == file.ModTime() {
						t.Errorf("certificate %s was not renewed as expected", cert.Name)
					}
				}
				for _, kubeConfig := range test.KubeconfigFiles {
					file, err := os.Stat(filepath.Join(tmpDir, kubeConfig))
					if err != nil {
						t.Fatalf("couldn't get kubeconfig %s: %v", kubeConfig, err)
					}
					if ModTime[kubeConfig] == file.ModTime() {
						t.Errorf("kubeconfig %s was not renewed as expected", kubeConfig)
					}
				}
			}
		})
	}
}

func TestRunGenCSR(t *testing.T) {
	tmpDir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpDir)

	kubeConfigDir := filepath.Join(tmpDir, "kubernetes")
	certDir := kubeConfigDir + "/pki"

	expectedCertificates := []string{
		"apiserver",
		"apiserver-etcd-client",
		"apiserver-kubelet-client",
		"front-proxy-client",
		"etcd/healthcheck-client",
		"etcd/peer",
		"etcd/server",
	}

	expectedKubeConfigs := []string{
		"admin",
		"kubelet",
		"controller-manager",
		"scheduler",
	}

	config := genCSRConfig{
		kubeConfigDir: kubeConfigDir,
		kubeadmConfig: &kubeadmapi.InitConfiguration{
			LocalAPIEndpoint: kubeadmapi.APIEndpoint{
				AdvertiseAddress: "192.0.2.1",
				BindPort:         443,
			},
			ClusterConfiguration: kubeadmapi.ClusterConfiguration{
				Networking: kubeadmapi.Networking{
					ServiceSubnet: "192.0.2.0/24",
				},
				CertificatesDir:   certDir,
				KubernetesVersion: kubeadmconstants.MinimumControlPlaneVersion.String(),
			},
		},
	}

	err := runGenCSR(nil, &config)
	require.NoError(t, err, "expected runGenCSR to not fail")

	t.Log("The command generates key and CSR files in the configured --cert-dir")
	for _, name := range expectedCertificates {
		_, err = pkiutil.TryLoadKeyFromDisk(certDir, name)
		assert.NoErrorf(t, err, "failed to load key file: %s", name)

		_, err = pkiutil.TryLoadCSRFromDisk(certDir, name)
		assert.NoError(t, err, "failed to load CSR file: %s", name)
	}

	t.Log("The command generates kubeconfig files in the configured --kubeconfig-dir")
	for _, name := range expectedKubeConfigs {
		_, err = clientcmd.LoadFromFile(kubeConfigDir + "/" + name + ".conf")
		assert.NoErrorf(t, err, "failed to load kubeconfig file: %s", name)

		_, err = pkiutil.TryLoadCSRFromDisk(kubeConfigDir, name+".conf")
		assert.NoError(t, err, "failed to load kubeconfig CSR file: %s", name)
	}
}

func TestGenCSRConfig(t *testing.T) {
	type assertion func(*testing.T, *genCSRConfig)

	hasCertDir := func(expected string) assertion {
		return func(t *testing.T, config *genCSRConfig) {
			assert.Equal(t, expected, config.kubeadmConfig.CertificatesDir)
		}
	}
	hasKubeConfigDir := func(expected string) assertion {
		return func(t *testing.T, config *genCSRConfig) {
			assert.Equal(t, expected, config.kubeConfigDir)
		}
	}
	hasAdvertiseAddress := func(expected string) assertion {
		return func(t *testing.T, config *genCSRConfig) {
			assert.Equal(t, expected, config.kubeadmConfig.LocalAPIEndpoint.AdvertiseAddress)
		}
	}

	// A minimal kubeadm config with just enough values to avoid triggering
	// auto-detection of config values at runtime.
	var kubeadmConfig = fmt.Sprintf(`
apiVersion: %s
kind: InitConfiguration
localAPIEndpoint:
  advertiseAddress: 192.0.2.1
nodeRegistration:
  criSocket: "unix:///var/run/containerd/containerd.sock"
---
apiVersion: %[1]s
kind: ClusterConfiguration
certificatesDir: /custom/config/certificates-dir
kubernetesVersion: %s`,
		kubeadmapiv1.SchemeGroupVersion.String(),
		kubeadmconstants.MinimumControlPlaneVersion.String())

	tmpDir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpDir)

	customConfigPath := tmpDir + "/kubeadm.conf"

	f, err := os.Create(customConfigPath)
	require.NoError(t, err)
	_, err = f.Write([]byte(kubeadmConfig))
	require.NoError(t, err)

	tests := []struct {
		name       string
		flags      []string
		assertions []assertion
		expectErr  bool
	}{
		{
			name: "default",
			assertions: []assertion{
				hasCertDir(kubeadmapiv1.DefaultCertificatesDir),
				hasKubeConfigDir(kubeadmconstants.KubernetesDir),
			},
		},
		{
			name:  "--cert-dir overrides default",
			flags: []string{"--cert-dir", "/foo/bar/pki"},
			assertions: []assertion{
				hasCertDir("/foo/bar/pki"),
			},
		},
		{
			name:  "--config is loaded",
			flags: []string{"--config", customConfigPath},
			assertions: []assertion{
				hasCertDir("/custom/config/certificates-dir"),
				hasAdvertiseAddress("192.0.2.1"),
			},
		},
		{
			name:      "--config not found",
			flags:     []string{"--config", "/does/not/exist"},
			expectErr: true,
		},
		{
			name: "--cert-dir overrides --config certificatesDir",
			flags: []string{
				"--config", customConfigPath,
				"--cert-dir", "/foo/bar/pki",
			},
			assertions: []assertion{
				hasCertDir("/foo/bar/pki"),
				hasAdvertiseAddress("192.0.2.1"),
			},
		},
		{
			name: "--kubeconfig-dir overrides default",
			flags: []string{
				"--kubeconfig-dir", "/foo/bar/kubernetes",
			},
			assertions: []assertion{
				hasKubeConfigDir("/foo/bar/kubernetes"),
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			flagset := pflag.NewFlagSet("flags-for-gencsr", pflag.ContinueOnError)
			config := newGenCSRConfig()
			config.addFlagSet(flagset)
			require.NoError(t, flagset.Parse(test.flags))

			err := config.load()
			if test.expectErr {
				assert.Error(t, err)
			}
			if !test.expectErr && assert.NoError(t, err) {
				for _, assertFunc := range test.assertions {
					assertFunc(t, config)
				}
			}
		})
	}
}

func TestRunCmdCertsExpiration(t *testing.T) {
	kdir := testutil.SetupTempDir(t)
	defer func() {
		if err := os.RemoveAll(kdir); err != nil {
			t.Fatalf("Failed to teardown: %s", err)
		}
		clientSetFromFile = kubeconfigutil.ClientSetFromFile
	}()

	cfg := testutil.GetDefaultInternalConfig(t)
	cfg.CertificatesDir = kdir

	// Generate all the CA
	caCerts := map[string]*x509.Certificate{}
	caKeys := map[string]crypto.Signer{}
	for _, ca := range []*certsphase.KubeadmCert{
		certsphase.KubeadmCertRootCA(),
		certsphase.KubeadmCertFrontProxyCA(),
		certsphase.KubeadmCertEtcdCA(),
	} {
		caCert, caKey, err := ca.CreateAsCA(cfg)
		if err != nil {
			t.Fatalf("couldn't write out CA %s: %v", ca.Name, err)
		}
		caCerts[ca.Name] = caCert
		caKeys[ca.Name] = caKey
	}

	// Generate all the signed certificates
	kubeadmCerts := []*certsphase.KubeadmCert{
		certsphase.KubeadmCertAPIServer(),
		certsphase.KubeadmCertKubeletClient(),
		certsphase.KubeadmCertFrontProxyClient(),
		certsphase.KubeadmCertEtcdAPIClient(),
		certsphase.KubeadmCertEtcdServer(),
		certsphase.KubeadmCertEtcdPeer(),
		certsphase.KubeadmCertEtcdHealthcheck(),
	}
	for _, cert := range kubeadmCerts {
		caCert := caCerts[cert.CAName]
		caKey := caKeys[cert.CAName]
		if err := cert.CreateFromCA(cfg, caCert, caKey); err != nil {
			t.Fatalf("couldn't write certificate %s: %v", cert.Name, err)
		}
	}

	// Generate all the kubeconfig files with embedded certs
	kubeConfigs := []string{
		kubeadmconstants.AdminKubeConfigFileName,
		kubeadmconstants.SuperAdminKubeConfigFileName,
		kubeadmconstants.SchedulerKubeConfigFileName,
		kubeadmconstants.ControllerManagerKubeConfigFileName,
	}
	for _, kubeConfig := range kubeConfigs {
		if err := kubeconfigphase.CreateKubeConfigFile(kubeConfig, kdir, cfg); err != nil {
			t.Fatalf("couldn't create kubeconfig %q: %v", kubeConfig, err)
		}
	}

	// A minimal kubeadm config with just enough values to avoid triggering
	// auto-detection of config values at runtime.
	var kubeadmConfig = fmt.Sprintf(`
apiVersion: %[1]s
kind: ClusterConfiguration
certificatesDir: %s
kubernetesVersion: %s`,
		kubeadmapiv1.SchemeGroupVersion.String(),
		cfg.CertificatesDir,
		kubeadmconstants.MinimumControlPlaneVersion.String())

	// Write the minimal kubeadm config to a file
	customConfigPath := kdir + "/kubeadm.conf"
	f, err := os.Create(customConfigPath)
	require.NoError(t, err)
	_, err = f.Write([]byte(kubeadmConfig))
	require.NoError(t, err)

	// fakeClientSetFromFile returns a fake clientset with kubeadm config map
	var fakeClientSetFromFile = func(_ string) (kubernetes.Interface, error) {
		client := fakeclientset.NewSimpleClientset()
		client.PrependReactor("get", "configmaps", func(action clientgotesting.Action) (bool, runtime.Object, error) {
			getAction := action.(clientgotesting.GetAction)
			if getAction.GetNamespace() == metav1.NamespaceSystem && getAction.GetName() == kubeadmconstants.KubeadmConfigConfigMap {
				cm := &v1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Name:      kubeadmconstants.KubeadmConfigConfigMap,
						Namespace: metav1.NamespaceSystem,
					},
					Data: map[string]string{
						kubeadmconstants.ClusterConfigurationConfigMapKey: dedent.Dedent(kubeadmConfig),
					},
				}
				return true, cm, nil
			}
			// Return an empty config map for kubelet and kube-proxy components.
			return true, &v1.ConfigMap{}, nil
		})
		return client, nil
	}

	// Select a certificate used to simulate a missing certificate file
	brokenCertName := kubeadmconstants.APIServerCertAndKeyBaseName
	brokenCertPath, _ := pkiutil.PathsForCertAndKey(cfg.CertificatesDir, brokenCertName)

	type testCase struct {
		name           string
		config         string
		output         string
		brokenCertName string
	}

	var runTestCase = func(t *testing.T, tc testCase) {
		var output bytes.Buffer
		cmd := newCmdCertsExpiration(&output, kdir)
		args := []string{
			fmt.Sprintf("--cert-dir=%s", cfg.CertificatesDir),
		}
		if tc.config != "" {
			args = append(args, fmt.Sprintf("--config=%s", tc.config))
		} else {
			clientSetFromFile = fakeClientSetFromFile
			defer func() {
				clientSetFromFile = kubeconfigutil.ClientSetFromFile
			}()
		}
		if tc.output != "" {
			args = append(args, fmt.Sprintf("-o=%s", tc.output))
		}
		cmd.SetArgs(args)
		require.NoError(t, cmd.Execute())

		switch tc.output {
		case "json":
			var info outputapiv1alpha3.CertificateExpirationInfo
			require.NoError(t, json.Unmarshal(output.Bytes(), &info))
			assert.Len(t, info.Certificates, len(kubeadmCerts)+len(kubeConfigs))
			assert.Len(t, info.CertificateAuthorities, len(caCerts))
			for _, cert := range info.Certificates {
				if tc.brokenCertName == cert.Name {
					assert.True(t, cert.Missing, "expected certificate to be missing")
				} else {
					assert.False(t, cert.Missing, "expected certificate to be present")
				}
			}
		default:
			outputStr := output.String()
			if tc.brokenCertName != "" {
				assert.Contains(t, outputStr, "!MISSING!")
			} else {
				assert.NotContains(t, outputStr, "!MISSING!")
			}

			var lines []string
			for _, line := range strings.SplitAfter(outputStr, "\n") {
				if strings.TrimSpace(line) != "" {
					lines = append(lines, line)
				}
			}
			// 2 lines for the column headers.
			expectedLineCount := len(caCerts) + len(kubeadmCerts) + len(kubeConfigs) + 2
			assert.Lenf(t, lines, expectedLineCount, "expected %d non-blank lines in output", expectedLineCount)
		}
	}

	testCases := []testCase{
		{
			name:   "fetch kubeadm config from file and print columns and no missing certs",
			config: customConfigPath,
			output: "",
		},
		{
			name:   "fetch kubeadm config from server and print columns and no missing certs",
			output: "",
		},
		{
			name:   "fetch kubeadm config from file and print json and no missing certs",
			config: customConfigPath,
			output: "json",
		},
		{
			name:   "fetch kubeadm config from server and print json and no missing certs",
			output: "json",
		},
		// all broken cases must be at the end of the list.
		{
			name:           "fetch kubeadm config from file and print columns and missing certs",
			config:         customConfigPath,
			output:         "",
			brokenCertName: brokenCertName,
		},
		{
			name:           "fetch kubeadm config from server and print columns and missing certs",
			output:         "",
			brokenCertName: brokenCertName,
		},
		{
			name:           "fetch kubeadm config from file and print json and missing certs",
			config:         customConfigPath,
			output:         "json",
			brokenCertName: brokenCertName,
		},
		{
			name:           "fetch kubeadm config from server and print json and missing certs",
			output:         "json",
			brokenCertName: brokenCertName,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			if tc.brokenCertName != "" {
				// remove the file to simulate a missing certificate
				_ = os.Remove(brokenCertPath)
			}
			runTestCase(t, tc)
		})
	}
}
