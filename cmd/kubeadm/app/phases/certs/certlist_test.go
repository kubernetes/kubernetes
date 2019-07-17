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

package certs

import (
	"crypto"
	"crypto/tls"
	"crypto/x509"
	"io/ioutil"
	"os"
	"path"
	"testing"

	certutil "k8s.io/client-go/util/cert"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestCertListOrder(t *testing.T) {
	tests := []struct {
		certs Certificates
		name  string
	}{
		{
			name:  "Default Certificate List",
			certs: GetDefaultCertList(),
		},
		{
			name:  "Cert list less etcd",
			certs: GetCertsWithoutEtcd(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var lastCA *KubeadmCert
			for i, cert := range test.certs {
				if i > 0 && lastCA == nil {
					t.Fatalf("CA not present in list before certificate %q", cert.Name)
				}
				if cert.CAName == "" {
					lastCA = cert
				} else {
					if cert.CAName != lastCA.Name {
						t.Fatalf("expected CA name %q, got %q, for certificate %q", lastCA.Name, cert.CAName, cert.Name)
					}
				}
			}
		})
	}
}

func TestCAPointersValid(t *testing.T) {
	tests := []struct {
		certs Certificates
		name  string
	}{
		{
			name:  "Default Certificate List",
			certs: GetDefaultCertList(),
		},
		{
			name:  "Cert list less etcd",
			certs: GetCertsWithoutEtcd(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			certMap := test.certs.AsMap()

			for _, cert := range test.certs {
				if cert.CAName != "" && certMap[cert.CAName] == nil {
					t.Errorf("Certificate %q references nonexistent CA %q", cert.Name, cert.CAName)
				}
			}
		})
	}
}

func TestMakeCertTree(t *testing.T) {
	rootCert := &KubeadmCert{
		Name: "root",
	}
	leaf0 := &KubeadmCert{
		Name:   "leaf0",
		CAName: "root",
	}
	leaf1 := &KubeadmCert{
		Name:   "leaf1",
		CAName: "root",
	}
	selfSigned := &KubeadmCert{
		Name: "self-signed",
	}

	certMap := CertificateMap{
		"root":        rootCert,
		"leaf0":       leaf0,
		"leaf1":       leaf1,
		"self-signed": selfSigned,
	}

	orphanCertMap := CertificateMap{
		"leaf0": leaf0,
	}

	if _, err := orphanCertMap.CertTree(); err == nil {
		t.Error("expected orphan cert map to error, but got nil")
	}

	certTree, err := certMap.CertTree()
	t.Logf("cert tree: %v", certTree)
	if err != nil {
		t.Errorf("expected no error, but got %v", err)
	}

	if len(certTree) != 2 {
		t.Errorf("Expected tree to have 2 roots, got %d", len(certTree))
	}

	if len(certTree[rootCert]) != 2 {
		t.Errorf("Expected root to have 2 leaves, got %d", len(certTree[rootCert]))
	}

	if _, ok := certTree[selfSigned]; !ok {
		t.Error("Expected selfSigned to be present in tree, but missing")
	}
}

func TestCreateCertificateChain(t *testing.T) {
	dir, err := ioutil.TempDir("", t.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	ic := &kubeadmapi.InitConfiguration{
		NodeRegistration: kubeadmapi.NodeRegistrationOptions{
			Name: "test-node",
		},
		ClusterConfiguration: kubeadmapi.ClusterConfiguration{
			CertificatesDir: dir,
		},
	}

	caCfg := Certificates{
		{
			config:   certutil.Config{},
			Name:     "test-ca",
			BaseName: "test-ca",
		},
		{
			config: certutil.Config{
				AltNames: certutil.AltNames{
					DNSNames: []string{"test-domain.space"},
				},
				Usages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
			configMutators: []configMutatorsFunc{
				setCommonNameToNodeName(),
			},
			CAName:   "test-ca",
			Name:     "test-daughter",
			BaseName: "test-daughter",
		},
	}

	certTree, err := caCfg.AsMap().CertTree()
	if err != nil {
		t.Fatalf("unexpected error getting tree: %v", err)
	}

	if certTree.CreateTree(ic); err != nil {
		t.Fatal(err)
	}

	caCert, _ := parseCertAndKey(path.Join(dir, "test-ca"), t)
	daughterCert, _ := parseCertAndKey(path.Join(dir, "test-daughter"), t)

	pool := x509.NewCertPool()
	pool.AddCert(caCert)

	_, err = daughterCert.Verify(x509.VerifyOptions{
		DNSName:   "test-domain.space",
		Roots:     pool,
		KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	})
	if err != nil {
		t.Errorf("couldn't verify daughter cert: %v", err)
	}

}

func parseCertAndKey(basePath string, t *testing.T) (*x509.Certificate, crypto.PrivateKey) {
	certPair, err := tls.LoadX509KeyPair(basePath+".crt", basePath+".key")
	if err != nil {
		t.Fatalf("couldn't parse certificate and key: %v", err)
	}

	parsedCert, err := x509.ParseCertificate(certPair.Certificate[0])
	if err != nil {
		t.Fatalf("couldn't parse certificate: %v", err)
	}

	return parsedCert, certPair.PrivateKey
}
