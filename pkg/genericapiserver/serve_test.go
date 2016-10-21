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

package genericapiserver

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"testing"

	utilcert "k8s.io/kubernetes/pkg/util/cert"

	"github.com/stretchr/testify/assert"
)

type TestCertSpec struct {
	host       string
	names, ips []string // in certificate
}

type NamedTestCertSpec struct {
	TestCertSpec
	explicitNames []string // as --tls-sni-cert-key explicit names
}

func createTestCerts(spec TestCertSpec) (certFilePath, keyFilePath string, err error) {
	var ips []net.IP
	for _, ip := range spec.ips {
		ips = append(ips, net.ParseIP(ip))
	}

	certPem, keyPem, err := utilcert.GenerateSelfSignedCertKey(spec.host, ips, spec.names)
	if err != nil {
		return "", "", err
	}

	certFile, err := ioutil.TempFile(os.TempDir(), "cert")
	if err != nil {
		return "", "", err
	}

	keyFile, err := ioutil.TempFile(os.TempDir(), "key")
	if err != nil {
		os.Remove(certFile.Name())
		return "", "", err
	}

	_, err = certFile.Write(certPem)
	if err != nil {
		os.Remove(certFile.Name())
		os.Remove(keyFile.Name())
		return "", "", err
	}
	certFile.Close()

	_, err = keyFile.Write(keyPem)
	if err != nil {
		os.Remove(certFile.Name())
		os.Remove(keyFile.Name())
		return "", "", err
	}
	keyFile.Close()

	return certFile.Name(), keyFile.Name(), nil
}

func TestGetNamedCertificateMap(t *testing.T) {
	tests := []struct {
		certs         []NamedTestCertSpec
		explicitNames []string
		expected      map[string]int // name to certs[*] index
		errorString   string
	}{
		{
			// empty certs
			expected: map[string]int{},
		},
		{
			// only one cert
			certs: []NamedTestCertSpec{
				{
					TestCertSpec: TestCertSpec{
						host: "test.com",
					},
				},
			},
			expected: map[string]int{
				"test.com": 0,
			},
		},
		{
			// ips are ignored
			certs: []NamedTestCertSpec{
				{
					TestCertSpec: TestCertSpec{
						host: "test.com",
						ips:  []string{"1.2.3.4"},
					},
				},
			},
			expected: map[string]int{
				"test.com": 0,
			},
		},
		{
			// two certs with the same name
			certs: []NamedTestCertSpec{
				{
					TestCertSpec: TestCertSpec{
						host: "test.com",
					},
				},
				{
					TestCertSpec: TestCertSpec{
						host: "test.com",
					},
				},
			},
			expected: map[string]int{
				"test.com": 0,
			},
		},
		{
			// two certs with different names
			certs: []NamedTestCertSpec{
				{
					TestCertSpec: TestCertSpec{
						host: "test2.com",
					},
				},
				{
					TestCertSpec: TestCertSpec{
						host: "test1.com",
					},
				},
			},
			expected: map[string]int{
				"test1.com": 1,
				"test2.com": 0,
			},
		},
		{
			// two certs with the same name, explicit trumps
			certs: []NamedTestCertSpec{
				{
					TestCertSpec: TestCertSpec{
						host: "test.com",
					},
				},
				{
					TestCertSpec: TestCertSpec{
						host: "test.com",
					},
					explicitNames: []string{"test.com"},
				},
			},
			expected: map[string]int{
				"test.com": 1,
			},
		},
		{
			// certs with partial overlap; ips are ignored
			certs: []NamedTestCertSpec{
				{
					TestCertSpec: TestCertSpec{
						host:  "a",
						names: []string{"a.test.com", "test.com"},
					},
				},
				{
					TestCertSpec: TestCertSpec{
						host:  "b",
						names: []string{"b.test.com", "test.com"},
					},
				},
			},
			expected: map[string]int{
				"a": 0, "b": 1,
				"a.test.com": 0, "b.test.com": 1,
				"test.com": 0,
			},
		},
		{
			// wildcards
			certs: []NamedTestCertSpec{
				{
					TestCertSpec: TestCertSpec{
						host:  "a",
						names: []string{"a.test.com", "test.com"},
					},
					explicitNames: []string{"*.test.com", "test.com"},
				},
				{
					TestCertSpec: TestCertSpec{
						host:  "b",
						names: []string{"b.test.com", "test.com"},
					},
					explicitNames: []string{"dev.test.com", "test.com"},
				}},
			expected: map[string]int{
				"test.com":     0,
				"*.test.com":   0,
				"dev.test.com": 1,
			},
		},
	}

NextTest:
	for i, test := range tests {
		var namedCertKeys []NamedCertKey
		bySignature := map[string]int{} // index in test.certs by cert signature
		for j, c := range test.certs {
			certFile, keyFile, err := createTestCerts(c.TestCertSpec)
			if err != nil {
				t.Errorf("%d - failed to create cert %d: %v", i, j, err)
				continue NextTest
			}
			defer os.Remove(certFile)
			defer os.Remove(keyFile)

			namedCertKeys = append(namedCertKeys, NamedCertKey{
				CertKey: CertKey{
					KeyFile:  keyFile,
					CertFile: certFile,
				},
				Names: c.explicitNames,
			})

			sig, err := certFileSignature(certFile, keyFile)
			if err != nil {
				t.Errorf("%d - failed to get signature for %d: %v", i, j, err)
				continue NextTest
			}
			bySignature[sig] = j
		}

		certMap, err := getNamedCertificateMap(namedCertKeys)
		if err == nil && len(test.errorString) != 0 {
			t.Errorf("%d - expected no error, got: %v", i, err)
		} else if err != nil && err.Error() != test.errorString {
			t.Errorf("%d - expected error %q, got: %v", i, test.errorString, err)
		} else {
			got := map[string]int{}
			for name, cert := range certMap {
				x509Certs, err := x509.ParseCertificates(cert.Certificate[0])
				assert.NoError(t, err, "%d - invalid certificate for %q", i, name)
				assert.True(t, len(x509Certs) > 0, "%d - expected at least one x509 cert in tls cert for %q", i, name)
				got[name] = bySignature[x509CertSignature(x509Certs[0])]
			}

			assert.EqualValues(t, test.expected, got, "%d - wrong certificate map", i)
		}
	}
}

func TestServerRunWithSNI(t *testing.T) {
	tests := []struct {
		Cert              TestCertSpec
		SNICerts          []NamedTestCertSpec
		ExpectedCertIndex int

		// passed in the client hello info, "localhost" if unset
		ServerName string
	}{
		{
			// only one cert
			Cert: TestCertSpec{
				host: "localhost",
			},
			ExpectedCertIndex: -1,
		},
		{
			// cert with multiple alternate names
			Cert: TestCertSpec{
				host:  "localhost",
				names: []string{"test.com"},
				ips:   []string{"127.0.0.1"},
			},
			ExpectedCertIndex: -1,
			ServerName:        "test.com",
		},
		{
			// one SNI and the default cert with the same name
			Cert: TestCertSpec{
				host: "localhost",
			},
			SNICerts: []NamedTestCertSpec{
				{
					TestCertSpec: TestCertSpec{
						host: "localhost",
					},
				},
			},
			ExpectedCertIndex: 0,
		},
		{
			// matching SNI cert
			Cert: TestCertSpec{
				host: "localhost",
			},
			SNICerts: []NamedTestCertSpec{
				{
					TestCertSpec: TestCertSpec{
						host: "test.com",
					},
				},
			},
			ExpectedCertIndex: 0,
			ServerName:        "test.com",
		},
		{
			// matching IP in SNI cert and the server cert. But IPs must not be
			// passed via SNI. Hence, the ServerName in the HELLO packet is empty
			// and the server should select the non-SNI cert.
			Cert: TestCertSpec{
				host: "localhost",
				ips:  []string{"10.0.0.1"},
			},
			SNICerts: []NamedTestCertSpec{
				{
					TestCertSpec: TestCertSpec{
						host: "test.com",
						ips:  []string{"10.0.0.1"},
					},
				},
			},
			ExpectedCertIndex: -1,
			ServerName:        "10.0.0.1",
		},
		{
			// wildcards
			Cert: TestCertSpec{
				host: "localhost",
			},
			SNICerts: []NamedTestCertSpec{
				{
					TestCertSpec: TestCertSpec{
						host:  "test.com",
						names: []string{"*.test.com"},
					},
				},
			},
			ExpectedCertIndex: 0,
			ServerName:        "www.test.com",
		},
	}

NextTest:
	for i, test := range tests {
		// create server cert
		serverCertFile, serverKeyFile, err := createTestCerts(test.Cert)
		if err != nil {
			t.Errorf("%d - failed to create server cert: %v", i, err)
		}
		defer os.Remove(serverCertFile)
		defer os.Remove(serverKeyFile)

		// create SNI certs
		var namedCertKeys []NamedCertKey
		serverSig, err := certFileSignature(serverCertFile, serverKeyFile)
		if err != nil {
			t.Errorf("%d - failed to get server cert signature: %v", i, err)
			continue NextTest
		}
		signatures := map[string]int{
			serverSig: -1,
		}
		for j, c := range test.SNICerts {
			certFile, keyFile, err := createTestCerts(c.TestCertSpec)
			if err != nil {
				t.Errorf("%d - failed to create SNI cert %d: %v", i, j, err)
				continue NextTest
			}
			defer os.Remove(certFile)
			defer os.Remove(keyFile)

			namedCertKeys = append(namedCertKeys, NamedCertKey{
				CertKey: CertKey{
					KeyFile:  keyFile,
					CertFile: certFile,
				},
				Names: c.explicitNames,
			})

			// store index in namedCertKeys with the signature as the key
			sig, err := certFileSignature(certFile, keyFile)
			if err != nil {
				t.Errorf("%d - failed get SNI cert %d signature: %v", i, j, err)
				continue NextTest
			}
			signatures[sig] = j
		}

		stopCh := make(chan struct{})

		// launch server
		etcdserver, config, _ := setUp(t)
		defer etcdserver.Terminate(t)

		config.EnableIndex = true
		config.SecureServingInfo = &SecureServingInfo{
			ServingInfo: ServingInfo{
				BindAddress: "localhost:0",
			},
			ServerCert: GeneratableKeyCert{
				CertKey: CertKey{
					CertFile: serverCertFile,
					KeyFile:  serverKeyFile,
				},
			},
			SNICerts: namedCertKeys,
		}
		config.InsecureServingInfo = nil

		s, err := config.Complete().New()
		if err != nil {
			t.Errorf("%d - failed creating the server: %v", i, err)
			continue NextTest
		}

		if err := s.serveSecurely(stopCh); err != nil {
			t.Errorf("%d - failed running the server: %v", i, err)
			continue NextTest
		}

		// load certificates into a pool
		roots := x509.NewCertPool()
		certFiles := []string{serverCertFile}
		for _, c := range namedCertKeys {
			certFiles = append(certFiles, c.CertFile)
		}
		for _, certFile := range certFiles {
			bs, err := ioutil.ReadFile(certFile)
			if err != nil {
				t.Errorf("%d - error reading %q: %v", i, certFile, err)
				continue NextTest
			}
			if ok := roots.AppendCertsFromPEM(bs); !ok {
				t.Errorf("%d - error adding cert %q to the pool", i, certFile)
				continue NextTest
			}
		}

		// try to dial
		addr := fmt.Sprintf("localhost:%d", s.effectiveSecurePort)
		t.Logf("Dialing %s as %q", addr, test.ServerName)
		conn, err := tls.Dial("tcp", addr, &tls.Config{
			RootCAs:    roots,
			ServerName: test.ServerName, // used for SNI in the client HELLO packet
		})
		if err != nil {
			t.Errorf("%d - failed to connect: %v", i, err)
			continue NextTest
		}

		// check returned server certificate
		sig := x509CertSignature(conn.ConnectionState().PeerCertificates[0])
		gotCertIndex, found := signatures[sig]
		if !found {
			t.Errorf("%d - unknown signature returned from server: %s", i, sig)
		}
		if gotCertIndex != test.ExpectedCertIndex {
			t.Errorf("%d - expected cert index %d, got cert index %d", i, test.ExpectedCertIndex, gotCertIndex)
		}

		conn.Close()
	}
}

func x509CertSignature(cert *x509.Certificate) string {
	return base64.StdEncoding.EncodeToString(cert.Signature)
}

func certFileSignature(certFile, keyFile string) (string, error) {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return "", err
	}

	x509Certs, err := x509.ParseCertificates(cert.Certificate[0])
	if err != nil {
		return "", err
	}
	if len(x509Certs) == 0 {
		return "", fmt.Errorf("expected at least one cert after reparsing cert %q", certFile)
	}
	return x509CertSignature(x509Certs[0]), nil
}
