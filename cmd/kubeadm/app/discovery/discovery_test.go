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

package discovery

import (
	"crypto/x509"
	"crypto/x509/pkix"
	_ "embed"
	"math/big"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

//go:embed testdata/ca.crt
var testCACert []byte

func TestFor(t *testing.T) {
	tests := []struct {
		name   string
		d      kubeadm.JoinConfiguration
		expect bool
	}{
		{
			name:   "default Discovery",
			d:      kubeadm.JoinConfiguration{},
			expect: false,
		},
		{
			name: "file Discovery with a path",
			d: kubeadm.JoinConfiguration{
				Discovery: kubeadm.Discovery{
					File: &kubeadm.FileDiscovery{
						KubeConfigPath: "notnil",
					},
				},
			},
			expect: false,
		},
		{
			name: "file Discovery with an url",
			d: kubeadm.JoinConfiguration{
				Discovery: kubeadm.Discovery{
					File: &kubeadm.FileDiscovery{
						KubeConfigPath: "https://localhost",
					},
				},
			},
			expect: false,
		},
		{
			name: "BootstrapTokenDiscovery",
			d: kubeadm.JoinConfiguration{
				Discovery: kubeadm.Discovery{
					BootstrapToken: &kubeadm.BootstrapTokenDiscovery{
						Token: "foo.bar@foobar",
					},
				},
			},
			expect: false,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			config := rt.d
			config.Timeouts = &kubeadm.Timeouts{
				Discovery: &metav1.Duration{Duration: 1 * time.Minute},
			}
			client := fakeclient.NewSimpleClientset()
			_, actual := For(client, &config)
			if (actual == nil) != rt.expect {
				t.Errorf(
					"failed For:\n\texpected: %t\n\t  actual: %t",
					rt.expect,
					(actual == nil),
				)
			}
		})
	}
}

func TestGetCACertFromKubeconfig(t *testing.T) {
	tests := []struct {
		name          string
		config        *clientcmdapi.Config
		expectedError bool
	}{
		{
			name:          "empty kubeconfig",
			config:        &clientcmdapi.Config{},
			expectedError: true,
		},
		{
			name: "kubeconfig with invalid cert",
			config: &clientcmdapi.Config{
				CurrentContext: "cluster",
				Contexts: map[string]*clientcmdapi.Context{
					"cluster": {
						Cluster: "cluster",
					},
				},
				Clusters: map[string]*clientcmdapi.Cluster{
					"cluster": {
						CertificateAuthorityData: []byte("foo"),
					},
				},
			},
			expectedError: true,
		},
		{
			name: "kubeconfig with CA cert",
			config: &clientcmdapi.Config{
				CurrentContext: "cluster",
				Contexts: map[string]*clientcmdapi.Context{
					"cluster": {
						Cluster: "cluster",
					},
				},
				Clusters: map[string]*clientcmdapi.Cluster{
					"cluster": {
						CertificateAuthorityData: testCACert,
					},
				},
			},
			expectedError: false,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			_, err := getCACertFromKubeconfig(rt.config)
			if (err != nil) != rt.expectedError {
				t.Errorf("Expected error: %v, got: %v, error: %v", rt.expectedError, err != nil, err)
			}
		})
	}
}

func TestFormatCACertInfo(t *testing.T) {
	certInfo := formatCACertInfo(&x509.Certificate{
		Subject:            pkix.Name{CommonName: "test-subject"},
		Issuer:             pkix.Name{CommonName: "test-issuer"},
		SerialNumber:       big.NewInt(12345),
		NotBefore:          time.Date(2020, time.January, 1, 0, 0, 0, 0, time.UTC),
		NotAfter:           time.Date(2030, time.January, 1, 0, 0, 0, 0, time.UTC),
		SignatureAlgorithm: x509.SHA256WithRSA,
		PublicKeyAlgorithm: x509.RSA,
	})
	expected := "CA Certificate:\n" +
		"\tSubject: CN=test-subject\n" +
		"\tIssuer: CN=test-issuer\n" +
		"\tSerialNumber: 12345\n" +
		"\tNotBefore: 2020-01-01 00:00:00 +0000 UTC\n" +
		"\tNotAfter: 2030-01-01 00:00:00 +0000 UTC\n" +
		"\tSignatureAlgorithm: SHA256-RSA\n" +
		"\tPublicKeyAlgorithm: RSA\n" +
		"\tHash: sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	if certInfo != expected {
		t.Errorf("expected:\n%s\ngot:\n%s\n", expected, certInfo)
	}
}
