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
	"crypto"
	"crypto/x509"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/client-go/tools/clientcmd"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
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
			err := cmdtestutil.RunSubCommand(t, renewCmds, test.command, args)
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
			} else {
				assert.NoError(t, err)
			}
			for _, assertFunc := range test.assertions {
				assertFunc(t, config)
			}
		})
	}
}
