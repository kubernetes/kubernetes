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

package renew

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"fmt"
	"math/big"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
	certstestutil "k8s.io/kubernetes/cmd/kubeadm/test/certs"
	cmdtestutil "k8s.io/kubernetes/cmd/kubeadm/test/cmd"
)

func TestCommandsGenerated(t *testing.T) {
	expectedFlags := []string{
		"cert-dir",
		"config",
		"use-api",
	}

	expectedCommands := []string{
		// TODO(EKF): add `renew all`
		// "renew",

		"renew apiserver",
		"renew apiserver-kubelet-client",
		"renew apiserver-etcd-client",

		"renew front-proxy-client",

		"renew etcd-server",
		"renew etcd-peer",
		"renew etcd-healthcheck-client",
	}

	renewCmd := NewCmdCertsRenewal()

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
	tests := []struct {
		command    string
		baseName   string
		caBaseName string
	}{
		{
			command:    "apiserver",
			baseName:   kubeadmconstants.APIServerCertAndKeyBaseName,
			caBaseName: kubeadmconstants.CACertAndKeyBaseName,
		},
		{
			command:    "apiserver-kubelet-client",
			baseName:   kubeadmconstants.APIServerKubeletClientCertAndKeyBaseName,
			caBaseName: kubeadmconstants.CACertAndKeyBaseName,
		},
		{
			command:    "apiserver-etcd-client",
			baseName:   kubeadmconstants.APIServerEtcdClientCertAndKeyBaseName,
			caBaseName: kubeadmconstants.EtcdCACertAndKeyBaseName,
		},
		{
			command:    "front-proxy-client",
			baseName:   kubeadmconstants.FrontProxyClientCertAndKeyBaseName,
			caBaseName: kubeadmconstants.FrontProxyCACertAndKeyBaseName,
		},
		{
			command:    "etcd-server",
			baseName:   kubeadmconstants.EtcdServerCertAndKeyBaseName,
			caBaseName: kubeadmconstants.EtcdCACertAndKeyBaseName,
		},
		{
			command:    "etcd-peer",
			baseName:   kubeadmconstants.EtcdPeerCertAndKeyBaseName,
			caBaseName: kubeadmconstants.EtcdCACertAndKeyBaseName,
		},
		{
			command:    "etcd-healthcheck-client",
			baseName:   kubeadmconstants.EtcdHealthcheckClientCertAndKeyBaseName,
			caBaseName: kubeadmconstants.EtcdCACertAndKeyBaseName,
		},
	}

	renewCmds := getRenewSubCommands()

	for _, test := range tests {
		t.Run(test.command, func(t *testing.T) {
			tmpDir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpDir)

			caCert, caKey := certstestutil.SetupCertificateAuthorithy(t)

			if err := pkiutil.WriteCertAndKey(tmpDir, test.caBaseName, caCert, caKey); err != nil {
				t.Fatalf("couldn't write out CA: %v", err)
			}

			certTmpl := x509.Certificate{
				Subject: pkix.Name{
					CommonName:   "test-cert",
					Organization: []string{"sig-cluster-lifecycle"},
				},
				DNSNames:     []string{"test-domain.space"},
				SerialNumber: new(big.Int).SetInt64(0),
				NotBefore:    time.Now().Add(-time.Hour * 24 * 365),
				NotAfter:     time.Now().Add(-time.Hour),
				KeyUsage:     x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
				ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			}

			key, err := rsa.GenerateKey(rand.Reader, 2048)
			if err != nil {
				t.Fatalf("couldn't generate private key: %v", err)
			}

			certDERBytes, err := x509.CreateCertificate(rand.Reader, &certTmpl, caCert, key.Public(), caKey)
			if err != nil {
				t.Fatalf("couldn't generate private key: %v", err)
			}
			cert, err := x509.ParseCertificate(certDERBytes)
			if err != nil {
				t.Fatalf("couldn't generate private key: %v", err)
			}

			if err := pkiutil.WriteCertAndKey(tmpDir, test.baseName, cert, key); err != nil {
				t.Fatalf("couldn't write out initial certificate")
			}

			cmdtestutil.RunSubCommand(t, renewCmds, test.command, fmt.Sprintf("--cert-dir=%s", tmpDir))

			newCert, newKey, err := pkiutil.TryLoadCertAndKeyFromDisk(tmpDir, test.baseName)
			if err != nil {
				t.Fatalf("couldn't load renewed certificate: %v", err)
			}

			certstestutil.AssertCertificateIsSignedByCa(t, newCert, caCert)

			pool := x509.NewCertPool()
			pool.AddCert(caCert)

			_, err = newCert.Verify(x509.VerifyOptions{
				DNSName:   "test-domain.space",
				Roots:     pool,
				KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			})
			if err != nil {
				t.Errorf("couldn't verify renewed cert: %v", err)
			}

			pubKey, ok := newCert.PublicKey.(*rsa.PublicKey)
			if !ok {
				t.Errorf("unknown public key type %T", newCert.PublicKey)
			} else if pubKey.N.Cmp(newKey.N) != 0 {
				t.Error("private key does not match public key")
			}

		})
	}
}
