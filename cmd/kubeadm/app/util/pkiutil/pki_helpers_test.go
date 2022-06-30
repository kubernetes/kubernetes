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

package pkiutil

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"fmt"
	"net"
	"os"
	"reflect"
	"testing"

	certutil "k8s.io/client-go/util/cert"
	netutils "k8s.io/utils/net"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

var (
	// TestMain generates the bellow certs and keys so that
	// they are reused in tests whenever possible

	rootCACert, servCert *x509.Certificate
	rootCAKey, servKey   crypto.Signer

	ecdsaKey *ecdsa.PrivateKey
)

func TestMain(m *testing.M) {
	var err error

	rootCACert, rootCAKey, err = NewCertificateAuthority(&CertConfig{
		Config: certutil.Config{
			CommonName: "Root CA 1",
		},
		PublicKeyAlgorithm: x509.RSA,
	})
	if err != nil {
		panic(fmt.Sprintf("Failed generating Root CA: %v", err))
	}
	if !rootCACert.IsCA {
		panic("rootCACert is not a valid CA")
	}

	servCert, servKey, err = NewCertAndKey(rootCACert, rootCAKey, &CertConfig{
		Config: certutil.Config{
			CommonName: "kubernetes",
			Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		},
	})
	if err != nil {
		panic(fmt.Sprintf("Failed generating serving cert/key: %v", err))
	}

	ecdsaKey, err = ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		panic("Could not generate ECDSA key")
	}

	os.Exit(m.Run())
}

func TestNewCertAndKey(t *testing.T) {
	var tests = []struct {
		name string
		key  crypto.Signer
	}{
		{
			name: "ECDSA should succeed",
			key:  ecdsaKey,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			caCert := &x509.Certificate{}
			config := &CertConfig{
				Config: certutil.Config{
					CommonName:   "test",
					Organization: []string{"test"},
					Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
				},
			}
			_, _, err := NewCertAndKey(caCert, rt.key, config)
			if err != nil {
				t.Errorf("failed NewCertAndKey: %v", err)
			}
		})
	}
}

func TestHasServerAuth(t *testing.T) {
	// Override NewPrivateKey to reuse the same key for all certs
	// since this test is only checking cert.ExtKeyUsage
	privateKeyFunc := NewPrivateKey
	NewPrivateKey = func(x509.PublicKeyAlgorithm) (crypto.Signer, error) {
		return rootCAKey, nil
	}
	defer func() {
		NewPrivateKey = privateKeyFunc
	}()

	var tests = []struct {
		name     string
		config   CertConfig
		expected bool
	}{
		{
			name: "has ServerAuth",
			config: CertConfig{
				Config: certutil.Config{
					CommonName: "test",
					Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
				},
			},
			expected: true,
		},
		{
			name: "has ServerAuth ECDSA",
			config: CertConfig{
				Config: certutil.Config{
					CommonName: "test",
					Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
				},
				PublicKeyAlgorithm: x509.ECDSA,
			},
			expected: true,
		},
		{
			name: "doesn't have ServerAuth",
			config: CertConfig{
				Config: certutil.Config{
					CommonName: "test",
					Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
				},
			},
			expected: false,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			cert, _, err := NewCertAndKey(rootCACert, rootCAKey, &rt.config)
			if err != nil {
				t.Fatalf("Couldn't create cert: %v", err)
			}
			actual := HasServerAuth(cert)
			if actual != rt.expected {
				t.Errorf(
					"failed HasServerAuth:\n\texpected: %t\n\t  actual: %t",
					rt.expected,
					actual,
				)
			}
		})
	}
}

func TestWriteCertAndKey(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	caCert := &x509.Certificate{}
	actual := WriteCertAndKey(tmpdir, "foo", caCert, rootCAKey)
	if actual != nil {
		t.Errorf(
			"failed WriteCertAndKey with an error: %v",
			actual,
		)
	}
}

func TestWriteCert(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	caCert := &x509.Certificate{}
	actual := WriteCert(tmpdir, "foo", caCert)
	if actual != nil {
		t.Errorf(
			"failed WriteCert with an error: %v",
			actual,
		)
	}
}

func TestWriteCertBundle(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	certs := []*x509.Certificate{{}, {}}

	actual := WriteCertBundle(tmpdir, "foo", certs)
	if actual != nil {
		t.Errorf("failed WriteCertBundle with an error: %v", actual)
	}
}

func TestWriteKey(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	actual := WriteKey(tmpdir, "foo", rootCAKey)
	if actual != nil {
		t.Errorf(
			"failed WriteCertAndKey with an error: %v",
			actual,
		)
	}
}

func TestWritePublicKey(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	actual := WritePublicKey(tmpdir, "foo", rootCAKey.Public())
	if actual != nil {
		t.Errorf(
			"failed WriteCertAndKey with an error: %v",
			actual,
		)
	}
}

func TestCertOrKeyExist(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	if err = WriteCertAndKey(tmpdir, "foo-0", rootCACert, rootCAKey); err != nil {
		t.Errorf(
			"failed WriteCertAndKey with an error: %v",
			err,
		)
	}
	if err = WriteCert(tmpdir, "foo-1", rootCACert); err != nil {
		t.Errorf(
			"failed WriteCert with an error: %v",
			err,
		)
	}

	var tests = []struct {
		desc     string
		path     string
		name     string
		expected bool
	}{
		{
			desc:     "empty path and name",
			path:     "",
			name:     "",
			expected: false,
		},
		{
			desc:     "valid path and name, both cert and key exist",
			path:     tmpdir,
			name:     "foo-0",
			expected: true,
		},
		{
			desc:     "valid path and name, only cert exist",
			path:     tmpdir,
			name:     "foo-1",
			expected: true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actual := CertOrKeyExist(rt.path, rt.name)
			if actual != rt.expected {
				t.Errorf(
					"failed CertOrKeyExist:\n\texpected: %t\n\t  actual: %t",
					rt.expected,
					actual,
				)
			}
		})
	}
}

func TestTryLoadCertAndKeyFromDisk(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	err = WriteCertAndKey(tmpdir, "foo", rootCACert, rootCAKey)
	if err != nil {
		t.Fatalf(
			"failed to write cert and key with an error: %v",
			err,
		)
	}

	var tests = []struct {
		desc     string
		path     string
		name     string
		expected bool
	}{
		{
			desc:     "empty path and name",
			path:     "",
			name:     "",
			expected: false,
		},
		{
			desc:     "valid path and name",
			path:     tmpdir,
			name:     "foo",
			expected: true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.desc, func(t *testing.T) {
			_, _, actual := TryLoadCertAndKeyFromDisk(rt.path, rt.name)
			if (actual == nil) != rt.expected {
				t.Errorf(
					"failed TryLoadCertAndKeyFromDisk:\n\texpected: %t\n\t  actual: %t",
					rt.expected,
					(actual == nil),
				)
			}
		})
	}
}

func TestTryLoadCertFromDisk(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	err = WriteCert(tmpdir, "foo", rootCACert)
	if err != nil {
		t.Fatalf(
			"failed to write cert and key with an error: %v",
			err,
		)
	}

	var tests = []struct {
		desc     string
		path     string
		name     string
		expected bool
	}{
		{
			desc:     "empty path and name",
			path:     "",
			name:     "",
			expected: false,
		},
		{
			desc:     "valid path and name",
			path:     tmpdir,
			name:     "foo",
			expected: true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.desc, func(t *testing.T) {
			_, actual := TryLoadCertFromDisk(rt.path, rt.name)
			if (actual == nil) != rt.expected {
				t.Errorf(
					"failed TryLoadCertAndKeyFromDisk:\n\texpected: %t\n\t  actual: %t",
					rt.expected,
					(actual == nil),
				)
			}
		})
	}
}

func TestTryLoadCertChainFromDisk(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	err = WriteCert(tmpdir, "leaf", servCert)
	if err != nil {
		t.Fatalf("failed to write cert: %v", err)
	}

	// rootCACert is treated as an intermediate CA here
	bundle := []*x509.Certificate{servCert, rootCACert}
	err = WriteCertBundle(tmpdir, "bundle", bundle)
	if err != nil {
		t.Fatalf("failed to write cert bundle: %v", err)
	}

	var tests = []struct {
		desc          string
		path          string
		name          string
		expected      bool
		intermediates int
	}{
		{
			desc:          "empty path and name",
			path:          "",
			name:          "",
			expected:      false,
			intermediates: 0,
		},
		{
			desc:          "leaf certificate",
			path:          tmpdir,
			name:          "leaf",
			expected:      true,
			intermediates: 0,
		},
		{
			desc:          "certificate bundle",
			path:          tmpdir,
			name:          "bundle",
			expected:      true,
			intermediates: 1,
		},
	}
	for _, rt := range tests {
		t.Run(rt.desc, func(t *testing.T) {
			_, intermediates, actual := TryLoadCertChainFromDisk(rt.path, rt.name)
			if (actual == nil) != rt.expected {
				t.Errorf(
					"failed TryLoadCertChainFromDisk:\n\texpected: %t\n\t  actual: %t",
					rt.expected,
					(actual == nil),
				)
			}
			if len(intermediates) != rt.intermediates {
				t.Errorf(
					"TryLoadCertChainFromDisk returned the wrong number of intermediate certificates:\n\texpected: %d\n\t  actual: %d",
					rt.intermediates,
					len(intermediates),
				)
			}
		})
	}
}

func TestTryLoadKeyFromDisk(t *testing.T) {
	var tests = []struct {
		desc       string
		pathSuffix string
		name       string
		caKey      crypto.Signer
		expected   bool
	}{
		{
			desc:       "empty path and name",
			pathSuffix: "somegarbage",
			name:       "",
			caKey:      rootCAKey,
			expected:   false,
		},
		{
			desc:       "RSA valid path and name",
			pathSuffix: "",
			name:       "foo",
			caKey:      rootCAKey,
			expected:   true,
		},
		{
			desc:       "ECDSA valid path and name",
			pathSuffix: "",
			name:       "foo",
			caKey:      ecdsaKey,
			expected:   true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.desc, func(t *testing.T) {
			tmpdir, err := os.MkdirTemp("", "")
			if err != nil {
				t.Fatalf("Couldn't create tmpdir")
			}
			defer os.RemoveAll(tmpdir)

			err = WriteKey(tmpdir, "foo", rt.caKey)
			if err != nil {
				t.Errorf(
					"failed to write key with an error: %v",
					err,
				)
			}
			_, actual := TryLoadKeyFromDisk(tmpdir+rt.pathSuffix, rt.name)
			if (actual == nil) != rt.expected {
				t.Errorf(
					"failed TryLoadCertAndKeyFromDisk:\n\texpected: %t\n\t  actual: %t",
					rt.expected,
					(actual == nil),
				)
			}
		})
	}
}

func TestPathsForCertAndKey(t *testing.T) {
	crtPath, keyPath := PathsForCertAndKey("/foo", "bar")
	if crtPath != "/foo/bar.crt" {
		t.Errorf("unexpected certificate path: %s", crtPath)
	}
	if keyPath != "/foo/bar.key" {
		t.Errorf("unexpected key path: %s", keyPath)
	}
}

func TestPathForCert(t *testing.T) {
	crtPath := pathForCert("/foo", "bar")
	if crtPath != "/foo/bar.crt" {
		t.Errorf("unexpected certificate path: %s", crtPath)
	}
}

func TestPathForKey(t *testing.T) {
	keyPath := pathForKey("/foo", "bar")
	if keyPath != "/foo/bar.key" {
		t.Errorf("unexpected certificate path: %s", keyPath)
	}
}

func TestPathForPublicKey(t *testing.T) {
	pubPath := pathForPublicKey("/foo", "bar")
	if pubPath != "/foo/bar.pub" {
		t.Errorf("unexpected certificate path: %s", pubPath)
	}
}

func TestPathForCSR(t *testing.T) {
	csrPath := pathForCSR("/foo", "bar")
	if csrPath != "/foo/bar.csr" {
		t.Errorf("unexpected certificate path: %s", csrPath)
	}
}

func TestGetAPIServerAltNames(t *testing.T) {

	var tests = []struct {
		desc                string
		name                string
		cfg                 *kubeadmapi.InitConfiguration
		expectedDNSNames    []string
		expectedIPAddresses []string
	}{
		{
			desc: "empty name",
			name: "",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ControlPlaneEndpoint: "api.k8s.io:6443",
					Networking:           kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
					APIServer: kubeadmapi.APIServer{
						CertSANs: []string{"10.1.245.94", "10.1.245.95", "1.2.3.L", "invalid,commas,in,DNS"},
					},
				},
				NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: "valid-hostname"},
			},
			expectedDNSNames:    []string{"valid-hostname", "kubernetes", "kubernetes.default", "kubernetes.default.svc", "kubernetes.default.svc.cluster.local", "api.k8s.io"},
			expectedIPAddresses: []string{"10.96.0.1", "1.2.3.4", "10.1.245.94", "10.1.245.95"},
		},
		{
			desc: "ControlPlaneEndpoint IP",
			name: "ControlPlaneEndpoint IP",
			cfg: &kubeadmapi.InitConfiguration{
				LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4"},
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ControlPlaneEndpoint: "4.5.6.7:6443",
					Networking:           kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
					APIServer: kubeadmapi.APIServer{
						CertSANs: []string{"10.1.245.94", "10.1.245.95", "1.2.3.L", "invalid,commas,in,DNS"},
					},
				},
				NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: "valid-hostname"},
			},
			expectedDNSNames:    []string{"valid-hostname", "kubernetes", "kubernetes.default", "kubernetes.default.svc", "kubernetes.default.svc.cluster.local"},
			expectedIPAddresses: []string{"10.96.0.1", "1.2.3.4", "10.1.245.94", "10.1.245.95", "4.5.6.7"},
		},
	}

	for _, rt := range tests {
		t.Run(rt.desc, func(t *testing.T) {
			altNames, err := GetAPIServerAltNames(rt.cfg)
			if err != nil {
				t.Fatalf("failed calling GetAPIServerAltNames: %s: %v", rt.name, err)
			}

			for _, DNSName := range rt.expectedDNSNames {
				found := false
				for _, val := range altNames.DNSNames {
					if val == DNSName {
						found = true
						break
					}
				}

				if !found {
					t.Errorf("%s: altNames does not contain DNSName %s but %v", rt.name, DNSName, altNames.DNSNames)
				}
			}

			for _, IPAddress := range rt.expectedIPAddresses {
				found := false
				for _, val := range altNames.IPs {
					if val.Equal(netutils.ParseIPSloppy(IPAddress)) {
						found = true
						break
					}
				}

				if !found {
					t.Errorf("%s: altNames does not contain IPAddress %s but %v", rt.name, IPAddress, altNames.IPs)
				}
			}
		})
	}
}

func TestGetEtcdAltNames(t *testing.T) {
	proxy := "user-etcd-proxy"
	proxyIP := "10.10.10.100"
	cfg := &kubeadmapi.InitConfiguration{
		LocalAPIEndpoint: kubeadmapi.APIEndpoint{
			AdvertiseAddress: "1.2.3.4",
		},
		NodeRegistration: kubeadmapi.NodeRegistrationOptions{
			Name: "myNode",
		},
		ClusterConfiguration: kubeadmapi.ClusterConfiguration{
			Etcd: kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					ServerCertSANs: []string{
						proxy,
						proxyIP,
						"1.2.3.L",
						"invalid,commas,in,DNS",
					},
				},
			},
		},
	}

	altNames, err := GetEtcdAltNames(cfg)
	if err != nil {
		t.Fatalf("failed calling GetEtcdAltNames: %v", err)
	}

	expectedDNSNames := []string{"myNode", "localhost", proxy}
	for _, DNSName := range expectedDNSNames {
		t.Run(DNSName, func(t *testing.T) {
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
		})
	}

	expectedIPAddresses := []string{"1.2.3.4", "127.0.0.1", net.IPv6loopback.String(), proxyIP}
	for _, IPAddress := range expectedIPAddresses {
		t.Run(IPAddress, func(t *testing.T) {
			found := false
			for _, val := range altNames.IPs {
				if val.Equal(netutils.ParseIPSloppy(IPAddress)) {
					found = true
					break
				}
			}

			if !found {
				t.Errorf("altNames does not contain IPAddress %s", IPAddress)
			}
		})
	}
}

func TestGetEtcdPeerAltNames(t *testing.T) {
	hostname := "valid-hostname"
	proxy := "user-etcd-proxy"
	proxyIP := "10.10.10.100"
	advertiseIP := "1.2.3.4"
	cfg := &kubeadmapi.InitConfiguration{
		LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: advertiseIP},
		ClusterConfiguration: kubeadmapi.ClusterConfiguration{
			Etcd: kubeadmapi.Etcd{
				Local: &kubeadmapi.LocalEtcd{
					PeerCertSANs: []string{
						proxy,
						proxyIP,
						"1.2.3.L",
						"invalid,commas,in,DNS",
					},
				},
			},
		},
		NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: hostname},
	}

	altNames, err := GetEtcdPeerAltNames(cfg)
	if err != nil {
		t.Fatalf("failed calling GetEtcdPeerAltNames: %v", err)
	}

	expectedDNSNames := []string{hostname, proxy}
	for _, DNSName := range expectedDNSNames {
		t.Run(DNSName, func(t *testing.T) {
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

			expectedIPAddresses := []string{advertiseIP, proxyIP}
			for _, IPAddress := range expectedIPAddresses {
				found := false
				for _, val := range altNames.IPs {
					if val.Equal(netutils.ParseIPSloppy(IPAddress)) {
						found = true
						break
					}
				}

				if !found {
					t.Errorf("altNames does not contain IPAddress %s", IPAddress)
				}
			}
		})
	}
}

func TestAppendSANsToAltNames(t *testing.T) {
	var tests = []struct {
		sans     []string
		expected int
	}{
		{[]string{}, 0},
		{[]string{"abc"}, 1},
		{[]string{"*.abc"}, 1},
		{[]string{"**.abc"}, 0},
		{[]string{"a.*.bc"}, 0},
		{[]string{"a.*.bc", "abc.def"}, 1},
		{[]string{"a*.bc", "abc.def"}, 1},
	}
	for _, rt := range tests {
		altNames := certutil.AltNames{}
		appendSANsToAltNames(&altNames, rt.sans, "foo")
		actual := len(altNames.DNSNames)
		if actual != rt.expected {
			t.Errorf(
				"failed AppendSANsToAltNames Numbers:\n\texpected: %d\n\t  actual: %d",
				rt.expected,
				actual,
			)
		}
	}

}

func TestRemoveDuplicateAltNames(t *testing.T) {
	tests := []struct {
		args *certutil.AltNames
		want *certutil.AltNames
	}{
		{
			&certutil.AltNames{},
			&certutil.AltNames{},
		},
		{
			&certutil.AltNames{
				DNSNames: []string{"a", "a"},
				IPs:      []net.IP{{127, 0, 0, 1}},
			},
			&certutil.AltNames{
				DNSNames: []string{"a"},
				IPs:      []net.IP{{127, 0, 0, 1}},
			},
		},
		{
			&certutil.AltNames{
				DNSNames: []string{"a"},
				IPs:      []net.IP{{127, 0, 0, 1}, {127, 0, 0, 1}},
			},
			&certutil.AltNames{
				DNSNames: []string{"a"},
				IPs:      []net.IP{{127, 0, 0, 1}},
			},
		},
		{
			&certutil.AltNames{
				DNSNames: []string{"a", "a"},
				IPs:      []net.IP{{127, 0, 0, 1}, {127, 0, 0, 1}},
			},
			&certutil.AltNames{
				DNSNames: []string{"a"},
				IPs:      []net.IP{{127, 0, 0, 1}},
			},
		},
	}
	for _, tt := range tests {
		RemoveDuplicateAltNames(tt.args)
		if !reflect.DeepEqual(tt.args, tt.want) {
			t.Errorf("Wanted %v, got %v", tt.want, tt.args)
		}
	}
}

func TestVerifyCertChain(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	rootCert2, rootKey2, err := NewCertificateAuthority(&CertConfig{
		Config: certutil.Config{CommonName: "Root CA 2"},
	})
	if err != nil {
		t.Errorf("failed to create root CA cert and key with an error: %v", err)
	}

	intCert2, intKey2, err := NewIntermediateCertificateAuthority(rootCert2, rootKey2, &CertConfig{
		Config: certutil.Config{
			CommonName: "Intermediate CA 2",
			Usages:     []x509.ExtKeyUsage{},
		},
	})
	if err != nil {
		t.Errorf("failed to create intermediate CA cert and key with an error: %v", err)
	}

	leafCert2, _, err := NewCertAndKey(intCert2, intKey2, &CertConfig{
		Config: certutil.Config{
			CommonName: "Leaf Certificate 2",
			Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		},
	})
	if err != nil {
		t.Errorf("failed to create leaf cert and key with an error: %v", err)
	}

	var tests = []struct {
		desc          string
		leaf          *x509.Certificate
		intermediates []*x509.Certificate
		root          *x509.Certificate
		expected      bool
	}{
		{
			desc:          "without any intermediate CAs",
			leaf:          servCert,
			intermediates: []*x509.Certificate{},
			root:          rootCACert,
			expected:      true,
		},
		{
			desc:          "missing intermediate CA",
			leaf:          leafCert2,
			intermediates: []*x509.Certificate{},
			root:          rootCert2,
			expected:      false,
		},
		{
			desc:          "with one intermediate CA",
			leaf:          leafCert2,
			intermediates: []*x509.Certificate{intCert2},
			root:          rootCert2,
			expected:      true,
		},
	}
	for _, rt := range tests {
		t.Run(rt.desc, func(t *testing.T) {
			actual := VerifyCertChain(rt.leaf, rt.intermediates, rt.root)
			if (actual == nil) != rt.expected {
				t.Errorf(
					"failed VerifyCertChain:\n\texpected: %t\n\t  actual: %t",
					rt.expected,
					(actual == nil),
				)
			}
		})
	}
}
