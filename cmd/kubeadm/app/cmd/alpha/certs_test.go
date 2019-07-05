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

package alpha

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
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
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
		"use-api",
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

	renewCmd := newCmdCertsRenewal()

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
		&certsphase.KubeadmCertRootCA,
		&certsphase.KubeadmCertFrontProxyCA,
		&certsphase.KubeadmCertEtcdCA,
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
		&certsphase.KubeadmCertAPIServer,
		&certsphase.KubeadmCertKubeletClient,
		&certsphase.KubeadmCertFrontProxyClient,
		&certsphase.KubeadmCertEtcdAPIClient,
		&certsphase.KubeadmCertEtcdServer,
		&certsphase.KubeadmCertEtcdPeer,
		&certsphase.KubeadmCertEtcdHealthcheck,
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
	}{
		{
			command: "all",
			Certs: []*certsphase.KubeadmCert{
				&certsphase.KubeadmCertAPIServer,
				&certsphase.KubeadmCertKubeletClient,
				&certsphase.KubeadmCertFrontProxyClient,
				&certsphase.KubeadmCertEtcdAPIClient,
				&certsphase.KubeadmCertEtcdServer,
				&certsphase.KubeadmCertEtcdPeer,
				&certsphase.KubeadmCertEtcdHealthcheck,
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
				&certsphase.KubeadmCertAPIServer,
			},
		},
		{
			command: "apiserver-kubelet-client",
			Certs: []*certsphase.KubeadmCert{
				&certsphase.KubeadmCertKubeletClient,
			},
		},
		{
			command: "apiserver-etcd-client",
			Certs: []*certsphase.KubeadmCert{
				&certsphase.KubeadmCertEtcdAPIClient,
			},
		},
		{
			command: "front-proxy-client",
			Certs: []*certsphase.KubeadmCert{
				&certsphase.KubeadmCertFrontProxyClient,
			},
		},
		{
			command: "etcd-server",
			Certs: []*certsphase.KubeadmCert{
				&certsphase.KubeadmCertEtcdServer,
			},
		},
		{
			command: "etcd-peer",
			Certs: []*certsphase.KubeadmCert{
				&certsphase.KubeadmCertEtcdPeer,
			},
		},
		{
			command: "etcd-healthcheck-client",
			Certs: []*certsphase.KubeadmCert{
				&certsphase.KubeadmCertEtcdHealthcheck,
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
			renewCmds := getRenewSubCommands(tmpDir)
			cmdtestutil.RunSubCommand(t, renewCmds, test.command, fmt.Sprintf("--cert-dir=%s", tmpDir))

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
		})
	}
}

func TestRenewUsingCSR(t *testing.T) {
	tmpDir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpDir)
	cert := &certs.KubeadmCertEtcdServer

	cfg := testutil.GetDefaultInternalConfig(t)
	cfg.CertificatesDir = tmpDir

	caCert, caKey, err := certsphase.KubeadmCertEtcdCA.CreateAsCA(cfg)
	if err != nil {
		t.Fatalf("couldn't write out CA %s: %v", certsphase.KubeadmCertEtcdCA.Name, err)
	}

	if err := cert.CreateFromCA(cfg, caCert, caKey); err != nil {
		t.Fatalf("couldn't write certificate %s: %v", cert.Name, err)
	}

	renewCmds := getRenewSubCommands(tmpDir)
	cmdtestutil.RunSubCommand(t, renewCmds, cert.Name, "--csr-only", "--csr-dir="+tmpDir, fmt.Sprintf("--cert-dir=%s", tmpDir))

	if _, _, err := pkiutil.TryLoadCSRAndKeyFromDisk(tmpDir, cert.Name); err != nil {
		t.Fatalf("couldn't load certificate %q: %v", cert.Name, err)
	}
}
