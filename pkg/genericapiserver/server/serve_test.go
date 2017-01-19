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

package server

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"reflect"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/genericapiserver/server/options"
	utilcert "k8s.io/kubernetes/pkg/util/cert"
	"k8s.io/kubernetes/pkg/util/config"
)

type TestCertSpec struct {
	host       string
	names, ips []string // in certificate
}

type NamedTestCertSpec struct {
	TestCertSpec
	explicitNames []string // as --tls-sni-cert-key explicit names
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
		var namedTLSCerts []namedTlsCert
		bySignature := map[string]int{} // index in test.certs by cert signature
		for j, c := range test.certs {
			cert, err := createTestTLSCerts(c.TestCertSpec)
			if err != nil {
				t.Errorf("%d - failed to create cert %d: %v", i, j, err)
				continue NextTest
			}

			namedTLSCerts = append(namedTLSCerts, namedTlsCert{
				tlsCert: cert,
				names:   c.explicitNames,
			})

			sig, err := certSignature(cert)
			if err != nil {
				t.Errorf("%d - failed to get signature for %d: %v", i, j, err)
				continue NextTest
			}
			bySignature[sig] = j
		}

		certMap, err := getNamedCertificateMap(namedTLSCerts)
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
	tests := map[string]struct {
		Cert              TestCertSpec
		SNICerts          []NamedTestCertSpec
		ExpectedCertIndex int

		// passed in the client hello info, "localhost" if unset
		ServerName string

		// optional ip or hostname to pass to NewSelfClientConfig
		SelfClientBindAddressOverride string
		ExpectSelfClientError         bool
	}{
		"only one cert": {
			Cert: TestCertSpec{
				host: "localhost",
				ips:  []string{"127.0.0.1"},
			},
			ExpectedCertIndex: -1,
		},
		"cert with multiple alternate names": {
			Cert: TestCertSpec{
				host:  "localhost",
				names: []string{"test.com"},
				ips:   []string{"127.0.0.1"},
			},
			ExpectedCertIndex: -1,
			ServerName:        "test.com",
		},
		"one SNI and the default cert with the same name": {
			Cert: TestCertSpec{
				host: "localhost",
				ips:  []string{"127.0.0.1"},
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
		"matching SNI cert": {
			Cert: TestCertSpec{
				host: "localhost",
				ips:  []string{"127.0.0.1"},
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
		"matching IP in SNI cert and the server cert": {
			// IPs must not be passed via SNI. Hence, the ServerName in the
			// HELLO packet is empty and the server should select the non-SNI cert.
			Cert: TestCertSpec{
				host: "localhost",
				ips:  []string{"10.0.0.1", "127.0.0.1"},
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
		"wildcards": {
			Cert: TestCertSpec{
				host: "localhost",
				ips:  []string{"127.0.0.1"},
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

		"loopback: IP for loopback client on SNI cert": {
			Cert: TestCertSpec{
				host: "localhost",
			},
			SNICerts: []NamedTestCertSpec{
				{
					TestCertSpec: TestCertSpec{
						host: "test.com",
						ips:  []string{"127.0.0.1"},
					},
				},
			},
			ExpectedCertIndex:     -1,
			ExpectSelfClientError: true,
		},
		"loopback: IP for loopback client on server and SNI cert": {
			Cert: TestCertSpec{
				ips:  []string{"127.0.0.1"},
				host: "localhost",
			},
			SNICerts: []NamedTestCertSpec{
				{
					TestCertSpec: TestCertSpec{
						host: "test.com",
						ips:  []string{"127.0.0.1"},
					},
				},
			},
			ExpectedCertIndex: -1,
		},
		"loopback: bind to 0.0.0.0 => loopback uses localhost; localhost on server cert": {
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
			ExpectedCertIndex:             -1,
			SelfClientBindAddressOverride: "0.0.0.0",
		},
		"loopback: bind to 0.0.0.0 => loopback uses localhost; localhost on SNI cert": {
			Cert: TestCertSpec{
				host: "test.com",
			},
			SNICerts: []NamedTestCertSpec{
				{
					TestCertSpec: TestCertSpec{
						host: "localhost",
					},
				},
			},
			ExpectedCertIndex:             0,
			SelfClientBindAddressOverride: "0.0.0.0",
		},
		"loopback: bind to 0.0.0.0 => loopback uses localhost; localhost on server and SNI cert": {
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
			ExpectedCertIndex:             0,
			SelfClientBindAddressOverride: "0.0.0.0",
		},
	}

	tempDir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

NextTest:
	for title, test := range tests {
		// create server cert
		serverCertBundleFile, serverKeyFile, err := createTestCertFiles(tempDir, test.Cert)
		if err != nil {
			t.Errorf("%q - failed to create server cert: %v", title, err)
			continue NextTest
		}
		ca, err := caCertFromBundle(serverCertBundleFile)
		if err != nil {
			t.Errorf("%q - failed to extract ca cert from server cert bundle: %v", title, err)
			continue NextTest
		}
		caCerts := []*x509.Certificate{ca}

		// create SNI certs
		var namedCertKeys []config.NamedCertKey
		serverSig, err := certFileSignature(serverCertBundleFile, serverKeyFile)
		if err != nil {
			t.Errorf("%q - failed to get server cert signature: %v", title, err)
			continue NextTest
		}
		signatures := map[string]int{
			serverSig: -1,
		}
		for j, c := range test.SNICerts {
			certBundleFile, keyFile, err := createTestCertFiles(tempDir, c.TestCertSpec)
			if err != nil {
				t.Errorf("%q - failed to create SNI cert %d: %v", title, j, err)
				continue NextTest
			}

			namedCertKeys = append(namedCertKeys, config.NamedCertKey{
				KeyFile:  keyFile,
				CertFile: certBundleFile,
				Names:    c.explicitNames,
			})

			ca, err := caCertFromBundle(certBundleFile)
			if err != nil {
				t.Errorf("%q - failed to extract ca cert from SNI cert %d: %v", title, j, err)
				continue NextTest
			}
			caCerts = append(caCerts, ca)

			// store index in namedCertKeys with the signature as the key
			sig, err := certFileSignature(certBundleFile, keyFile)
			if err != nil {
				t.Errorf("%q - failed get SNI cert %d signature: %v", title, j, err)
				continue NextTest
			}
			signatures[sig] = j
		}

		stopCh := make(chan struct{})

		// launch server
		etcdserver, config, _ := setUp(t)
		defer etcdserver.Terminate(t)

		v := fakeVersion()
		config.Version = &v

		config.EnableIndex = true
		_, err = config.ApplySecureServingOptions(&options.SecureServingOptions{
			ServingOptions: options.ServingOptions{
				BindAddress: net.ParseIP("127.0.0.1"),
				BindPort:    6443,
			},
			ServerCert: options.GeneratableKeyCert{
				CertKey: options.CertKey{
					CertFile: serverCertBundleFile,
					KeyFile:  serverKeyFile,
				},
			},
			SNICertKeys: namedCertKeys,
		})
		if err != nil {
			t.Errorf("%q - failed applying the SecureServingOptions: %v", title, err)
			continue NextTest
		}
		config.InsecureServingInfo = nil

		s, err := config.Complete().New()
		if err != nil {
			t.Errorf("%q - failed creating the server: %v", title, err)
			continue NextTest
		}

		// patch in a 0-port to enable auto port allocation
		s.SecureServingInfo.BindAddress = "127.0.0.1:0"

		if err := s.serveSecurely(stopCh); err != nil {
			t.Errorf("%q - failed running the server: %v", title, err)
			continue NextTest
		}

		// load ca certificates into a pool
		roots := x509.NewCertPool()
		for _, caCert := range caCerts {
			roots.AddCert(caCert)
		}

		// try to dial
		addr := fmt.Sprintf("localhost:%d", s.effectiveSecurePort)
		t.Logf("Dialing %s as %q", addr, test.ServerName)
		conn, err := tls.Dial("tcp", addr, &tls.Config{
			RootCAs:    roots,
			ServerName: test.ServerName, // used for SNI in the client HELLO packet
		})
		if err != nil {
			t.Errorf("%q - failed to connect: %v", title, err)
			continue NextTest
		}

		// check returned server certificate
		sig := x509CertSignature(conn.ConnectionState().PeerCertificates[0])
		gotCertIndex, found := signatures[sig]
		if !found {
			t.Errorf("%q - unknown signature returned from server: %s", title, sig)
		}
		if gotCertIndex != test.ExpectedCertIndex {
			t.Errorf("%q - expected cert index %d, got cert index %d", title, test.ExpectedCertIndex, gotCertIndex)
		}

		conn.Close()

		// check that the loopback client can connect
		host := "127.0.0.1"
		if len(test.SelfClientBindAddressOverride) != 0 {
			host = test.SelfClientBindAddressOverride
		}
		config.SecureServingInfo.ServingInfo.BindAddress = net.JoinHostPort(host, strconv.Itoa(s.effectiveSecurePort))
		cfg, err := config.SecureServingInfo.NewSelfClientConfig("some-token")
		if test.ExpectSelfClientError {
			if err == nil {
				t.Errorf("%q - expected error creating loopback client config", title)
			}
			continue NextTest
		}
		if err != nil {
			t.Errorf("%q - failed creating loopback client config: %v", title, err)
			continue NextTest
		}
		client, err := clientset.NewForConfig(cfg)
		if err != nil {
			t.Errorf("%q - failed to create loopback client: %v", title, err)
			continue NextTest
		}
		got, err := client.ServerVersion()
		if err != nil {
			t.Errorf("%q - failed to connect with loopback client: %v", title, err)
			continue NextTest
		}
		if expected := &v; !reflect.DeepEqual(got, expected) {
			t.Errorf("%q - loopback client didn't get correct version info: expected=%v got=%v", title, expected, got)
		}
	}
}

func TestBasicIpLimit(t *testing.T) {
	var ipLimit ipBasedLimit
	validateIpLimitState(t, &ipLimit, "Basic/raw IpBasedLimit", 0, 0, 0, 0)
	ipLimit.incrementIp("1.2.3.4")
	validateIpLimitState(t, &ipLimit, "Basic/raw IpBasedLimit increment", 0, 0, 0, 0)
	ipLimit.decrementIp("1.2.3.4")
	validateIpLimitState(t, &ipLimit, "Basic/raw IpBasedLimit decrement", 0, 0, 0, 0)
	ipLimit.incrementIp("127.0.0.1")
	validateIpLimitState(t, &ipLimit, "Basic/raw IpBasedLimit increment localhost", 0, 0, 0, 0)
	ipLimit.decrementIp("127.0.0.1")
	validateIpLimitState(t, &ipLimit, "Basic/raw IpBasedLimit decrement localhost", 0, 0, 0, 0)
}

func TestIpLimitInitialization(t *testing.T) {
	ipLimit := ipBasedLimit{limits: make(map[string]int), ipMax: 10, max: 20}
	validateIpLimitState(t, &ipLimit, "IpBasedLimit explicit initialization", 0, 10, 0, 20)
}

func TestIpLimitWithLocalHost(t *testing.T) {
	ipLimit := ipBasedLimit{limits: make(map[string]int), ipMax: 10, max: 20}
	ipLimit.incrementIp("127.0.0.1")
	validateIpLimitState(t, &ipLimit, "IPv4 localhost increment call for ipBasedLimit", 0, 10, 0, 20)
	ipLimit.decrementIp("127.0.0.1")
	validateIpLimitState(t, &ipLimit, "IPv4 localhost decrement call for ipBasedLimit", 0, 10, 0, 20)
	ipv6loopback := net.IPv6loopback.String()
	ipLimit.incrementIp(ipv6loopback)
	validateIpLimitState(t, &ipLimit, "IPv6 localhost increment call for ipBasedLimit", 0, 10, 0, 20)
	ipLimit.decrementIp(ipv6loopback)
	validateIpLimitState(t, &ipLimit, "IPv6 localhost decrement call for ipBasedLimit", 0, 10, 0, 20)
}

func TestIpLimitIncrementAndDecrement(t *testing.T) {
	ipLimit := ipBasedLimit{limits: make(map[string]int), ipMax: 10, max: 20}
	ipLimit.incrementIp("1.2.3.4")
	validateIpLimitState(t, &ipLimit, "First increment call for ipBasedLimit", 1, 10, 1, 20)
	val, ok := ipLimit.limits["1.2.3.4"]
	assert.True(t, ok, "Increment call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment ipBasedLimit has a max of %d", val)
	ipLimit.incrementIp("1.2.3.4")
	validateIpLimitState(t, &ipLimit, "Second increment call for ipBasedLimit", 1, 10, 2, 20)
	val, ok = ipLimit.limits["1.2.3.4"]
	assert.True(t, ok, "Increment call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 2, "Post increment ipBasedLimit has a max of %d", val)

	ipLimit.incrementIp("4.3.2.1")
	validateIpLimitState(t, &ipLimit, "First increment call for ipBasedLimit", 2, 10, 3, 20)
	val, ok = ipLimit.limits["4.3.2.1"]
	assert.True(t, ok, "Increment call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment ipBasedLimit has a max of %d", val)
	ipLimit.incrementIp("4.3.2.1")
	validateIpLimitState(t, &ipLimit, "Second increment call for ipBasedLimit", 2, 10, 4, 20)
	val, ok = ipLimit.limits["4.3.2.1"]
	assert.True(t, ok, "Increment call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 2, "Post increment ipBasedLimit has a max of %d", val)

	ipLimit.decrementIp("4.3.2.1")
	validateIpLimitState(t, &ipLimit, "First decrement call for ipBasedLimit", 2, 10, 3, 20)
	val, ok = ipLimit.limits["4.3.2.1"]
	assert.True(t, ok, "Decrement call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment ipBasedLimit has a max of %d", val)

	ipLimit.decrementIp("4.3.2.1")
	validateIpLimitState(t, &ipLimit, "Deleting decrement call for ipBasedLimit", 1, 10, 2, 20)
	val, ok = ipLimit.limits["4.3.2.1"]
	assert.EqualValues(t, val, 0, "Deleting decrement call for ipBasedLimit has a val of %d", 0)
	assert.False(t, ok, "Deleting decrement call for ipBasedLimit had a val")
	val, ok = ipLimit.limits["1.2.3.4"]
	assert.True(t, ok, "Unmodified ip in ipBasedLimit had no val")
	assert.EqualValues(t, val, 2, "Post increment ipBasedLimit has a max of %d", val)

	ipLimit.decrementIp("1.2.3.4")
	validateIpLimitState(t, &ipLimit, "First decrement call for ipBasedLimit", 1, 10, 1, 20)
	val, ok = ipLimit.limits["1.2.3.4"]
	assert.True(t, ok, "Decrement call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment ipBasedLimit has a max of %d", val)

	ipLimit.decrementIp("1.2.3.4")
	validateIpLimitState(t, &ipLimit, "Deleting decrement call for ipBasedLimit", 0, 10, 0, 20)
	val, ok = ipLimit.limits["1.2.3.4"]
	assert.False(t, ok, "Deleting decrement call for ipBasedLimit had a val")
	assert.EqualValues(t, val, 0, "Deleting decrement call for ipBasedLimit has a val of %d", 0)
}

func TestIpLimitIncrementFailOnIpLimit(t *testing.T) {
	ipLimit := ipBasedLimit{limits: make(map[string]int), ipMax: 3, max: 6}
	err := ipLimit.incrementIp("1.2.3.4")
	assert.NoError(t, err)
	validateIpLimitState(t, &ipLimit, "First increment call for ipBasedLimit", 1, 3, 1, 6)
	val, ok := ipLimit.limits["1.2.3.4"]
	assert.True(t, ok, "Increment call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment ipBasedLimit has a max of %d", val)

	err = ipLimit.incrementIp("1.2.3.4")
	assert.NoError(t, err)
	validateIpLimitState(t, &ipLimit, "Second increment call for ipBasedLimit", 1, 3, 2, 6)
	val, ok = ipLimit.limits["1.2.3.4"]
	assert.EqualValues(t, val, 2, "Post increment ipBasedLimit has a max of %d", val)

	err = ipLimit.incrementIp("1.2.3.4")
	assert.NoError(t, err)
	validateIpLimitState(t, &ipLimit, "Third increment call for ipBasedLimit", 1, 3, 3, 6)
	val, ok = ipLimit.limits["1.2.3.4"]
	assert.True(t, ok, "Increment call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 3, "Post increment ipBasedLimit has a max of %d", val)

	err = ipLimit.incrementIp("1.2.3.4")
	assert.Error(t, err)
	if limErr, ok := err.(limitedError); ok {
		assert.True(t, limErr.Timeout(), "IpLimit limitedError should return timeout true")
		assert.True(t, limErr.Temporary(), "IpLimit limitedError should return temporary true")
	} else {
		assert.Fail(t, "Expected a limitedError from incrementIp fail but got %T", err)
	}
	validateIpLimitState(t, &ipLimit, "Second increment call for ipBasedLimit", 1, 3, 3, 6)
	val, ok = ipLimit.limits["1.2.3.4"]
	assert.True(t, ok, "Increment call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 3, "Post increment ipBasedLimit has a max of %d", val)

	ipLimit.decrementIp("1.2.3.4")
	validateIpLimitState(t, &ipLimit, "Decrement call for ipBasedLimit", 1, 3, 2, 6)
	val, ok = ipLimit.limits["1.2.3.4"]
	assert.True(t, ok, "Decrement call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 2, "Post increment ipBasedLimit has a max of %d", val)

	ipLimit.decrementIp("1.2.3.4")
	validateIpLimitState(t, &ipLimit, "Decrement call for ipBasedLimit", 1, 3, 1, 6)
	val, ok = ipLimit.limits["1.2.3.4"]
	assert.True(t, ok, "Decrement call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment ipBasedLimit has a max of %d", val)

	ipLimit.decrementIp("1.2.3.4")
	validateIpLimitState(t, &ipLimit, "Deleting decrement call for ipBasedLimit", 0, 3, 0, 6)
	val, ok = ipLimit.limits["1.2.3.4"]
	assert.False(t, ok, "Deleting decrement call for ipBasedLimit had a val")
	assert.EqualValues(t, val, 0, "Deleting decrement call for ipBasedLimit has a val of %d", 0)
}

func TestIpLimitIncrementFailOnTotalLimit(t *testing.T) {
	ipLimit := ipBasedLimit{limits: make(map[string]int), ipMax: 2, max: 3}
	err := ipLimit.incrementIp("1.2.3.4")
	assert.NoError(t, err)
	validateIpLimitState(t, &ipLimit, "First increment call for ipBasedLimit", 1, 2, 1, 3)
	val, ok := ipLimit.limits["1.2.3.4"]
	assert.True(t, ok, "Increment call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment ipBasedLimit has a max of %d", val)

	err = ipLimit.incrementIp("1.2.3.4")
	assert.NoError(t, err)
	validateIpLimitState(t, &ipLimit, "Second increment call for ipBasedLimit", 1, 2, 2, 3)
	val, ok = ipLimit.limits["1.2.3.4"]
	assert.True(t, ok, "Increment call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 2, "Post increment ipBasedLimit has a max of %d", val)

	err = ipLimit.incrementIp("4.3.2.1")
	assert.NoError(t, err)
	validateIpLimitState(t, &ipLimit, "First increment call for second ip ipBasedLimit", 2, 2, 3, 3)
	val, ok = ipLimit.limits["4.3.2.1"]
	assert.True(t, ok, "Increment call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment ipBasedLimit has a max of %d", val)

	err = ipLimit.incrementIp("4.3.2.1")
	assert.Error(t, err)
	if limErr, ok := err.(limitedError); ok {
		assert.True(t, limErr.Timeout(), "IpLimit limitedError should return timeout true")
		assert.True(t, limErr.Temporary(), "IpLimit limitedError should return temporary true")
	} else {
		assert.Fail(t, "Expected a limitedError from incrementIp fail but got %T", err)
	}
	validateIpLimitState(t, &ipLimit, "Second increment call for ipBasedLimit", 2, 2, 3, 3)
	val, ok = ipLimit.limits["4.3.2.1"]
	assert.True(t, ok, "Increment call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment ipBasedLimit has a max of %d", val)

	ipLimit.decrementIp("1.2.3.4")
	validateIpLimitState(t, &ipLimit, "Decrement call for ipBasedLimit", 2, 2, 2, 3)
	val, ok = ipLimit.limits["1.2.3.4"]
	assert.True(t, ok, "Decrement call for ipBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Post increment ipBasedLimit has a max of %d", val)

	ipLimit.decrementIp("1.2.3.4")
	validateIpLimitState(t, &ipLimit, "Deleting decrement call for ipBasedLimit", 1, 2, 1, 3)
	val, ok = ipLimit.limits["1.2.3.4"]
	assert.False(t, ok, "Deleting decrement call for ipBasedLimit had a val")
	assert.EqualValues(t, val, 0, "Deleting decrement call for ipBasedLimit has a val of %d", 0)
	val, ok = ipLimit.limits["4.3.2.1"]
	assert.True(t, ok, "Unmodified IP for ipBasedLimit had no val")
	assert.EqualValues(t, val, 1, "Unmodified IP for ipBasedLimit has a max of %d", val)

	ipLimit.decrementIp("4.3.2.1")
	validateIpLimitState(t, &ipLimit, "Deleting decrement call for ipBasedLimit", 0, 2, 0, 3)
	val, ok = ipLimit.limits["4.3.2.1"]
	assert.False(t, ok, "Deleting decrement call for ipBasedLimit had a val")
	assert.EqualValues(t, val, 0, "Deleting decrement call for ipBasedLimit has a val of %d", 0)
}

func TestParrallelIpLimitAccessNoLimit(t *testing.T) {
	ipLimit := ipBasedLimit{limits: make(map[string]int), ipMax: 10, max: 100}
	stop := time.Now().Add(time.Second * 10) // Long enough to prevent flaky ???
	stateHolder := parrallelStateHolder{ipLimit: &ipLimit, waitSize: 100,
		stop: stop, good: 0, limit: 0, count: 0}
	ips := []string{"1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4", "5.5.5.5",
		"6.6.6.6", "7.7.7.7", "8.8.8.8", "9.9.9.9",
		"1973:dead:beef:0121:1973:dead:beef:0121"}
	done := make(chan struct{})
	defer close(done)
	for _, ip := range ips {
		for i := 0; i < 10; i++ {
			go stateHolder.accessIpLimit(ip, done)
		}
	}
	for i := 0; i < 100; i++ {
		<-done
	}
	validateIpLimitState(t, &ipLimit, "Deleting decrement call for ipBasedLimit", 0, 10, 0, 100)
	assert.EqualValues(t, stateHolder.good, 100, "Concurrent increment count for ipBasedLimit %d successes", 0)
	assert.EqualValues(t, stateHolder.limit, 0, "Concurrent increment count for ipBasedLimit %d limits", 0)
}

func TestParrallelIpLimitHitIpLimit(t *testing.T) {
	ipLimit := ipBasedLimit{limits: make(map[string]int), ipMax: 8, max: 100}
	stop := time.Now().Add(time.Second * 10) // Long enough to prevent flaky ???
	stateHolder := parrallelStateHolder{ipLimit: &ipLimit, waitSize: 100,
		stop: stop, good: 0, limit: 0, count: 0}
	ips := []string{"1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4", "5.5.5.5",
		"6.6.6.6", "7.7.7.7", "8.8.8.8", "9.9.9.9",
		"1973:dead:beef:0121:1973:dead:beef:0121"}
	done := make(chan struct{})
	defer close(done)
	for _, ip := range ips {
		for i := 0; i < 10; i++ {
			go stateHolder.accessIpLimit(ip, done)
		}
	}
	for i := 0; i < 100; i++ {
		<-done
	}
	validateIpLimitState(t, &ipLimit, "Deleting decrement call for ipBasedLimit", 0, 8, 0, 100)
	assert.EqualValues(t, stateHolder.good, 80, "Concurrent increment count for ipBasedLimit %d successes", 0)
	assert.EqualValues(t, stateHolder.limit, 20, "Concurrent increment count for ipBasedLimit %d limits", 0)
}

func TestParrallelIpLimitHitTotalLimit(t *testing.T) {
	ipLimit := ipBasedLimit{limits: make(map[string]int), ipMax: 10, max: 79}
	stop := time.Now().Add(time.Second * 10) // Long enough to prevent flaky ???
	stateHolder := parrallelStateHolder{ipLimit: &ipLimit, waitSize: 100,
		stop: stop, good: 0, limit: 0, count: 0}
	ips := []string{"1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4", "5.5.5.5",
		"6.6.6.6", "7.7.7.7", "8.8.8.8", "9.9.9.9",
		"1977:dead:beef:0525:2015:dead:beef:1218"}
	done := make(chan struct{})
	defer close(done)
	for _, ip := range ips {
		for i := 0; i < 10; i++ {
			go stateHolder.accessIpLimit(ip, done)
		}
	}
	for i := 0; i < 100; i++ {
		<-done
	}
	validateIpLimitState(t, &ipLimit, "Deleting decrement call for ipBasedLimit", 0, 10, 0, 79)
	assert.EqualValues(t, stateHolder.good, 79, "Concurrent increment count for ipBasedLimit %d successes", 0)
	assert.EqualValues(t, stateHolder.limit, 21, "Concurrent increment count for ipBasedLimit %d limits", 0)
}

func (sh *parrallelStateHolder) accessIpLimit(ip string, done chan<- struct{}) {
	defer func() { done <- struct{}{} }()
	err := sh.ipLimit.incrementIp(ip)
	if err == nil {
		atomic.AddUint64(&sh.good, 1)
	} else {
		atomic.AddUint64(&sh.limit, 1)
	}
	atomic.AddUint64(&sh.count, 1)
	if err == nil {
		sh.wait()
		sh.ipLimit.decrementIp(ip)
	}
}

func (sh *parrallelStateHolder) wait() {
	var current uint64
	for current < sh.waitSize {
		time.Sleep(time.Millisecond * 100)
		if time.Now().After(sh.stop) {
			break
		}
		current = atomic.LoadUint64(&sh.count)
	}
}

type parrallelStateHolder struct {
	ipLimit  *ipBasedLimit
	waitSize uint64
	stop     time.Time
	good     uint64
	limit    uint64
	count    uint64
}

func validateIpLimitState(t *testing.T, ipLimit *ipBasedLimit, base string, expMapSize, expIpMax, expTtl, expMax int) {
	assert.EqualValues(t, len(ipLimit.limits), expMapSize, "%s map is size %d not %d", base, len(ipLimit.limits), expMapSize)
	assert.EqualValues(t, ipLimit.ipMax, expIpMax, "%s ipMax is %d not %d", base, ipLimit.ipMax, expIpMax)
	assert.EqualValues(t, ipLimit.total, expTtl, "%s count is %d not %d", base, ipLimit.total, expTtl)
	assert.EqualValues(t, ipLimit.max, expMax, "%s max is %d not %d", base, ipLimit.max, expMax)
}

func parseIPList(ips []string) []net.IP {
	var netIPs []net.IP
	for _, ip := range ips {
		netIPs = append(netIPs, net.ParseIP(ip))
	}
	return netIPs
}

func createTestTLSCerts(spec TestCertSpec) (tlsCert tls.Certificate, err error) {
	certPem, keyPem, err := utilcert.GenerateSelfSignedCertKey(spec.host, parseIPList(spec.ips), spec.names)
	if err != nil {
		return tlsCert, err
	}

	tlsCert, err = tls.X509KeyPair(certPem, keyPem)
	return tlsCert, err
}

func createTestCertFiles(dir string, spec TestCertSpec) (certFilePath, keyFilePath string, err error) {
	certPem, keyPem, err := utilcert.GenerateSelfSignedCertKey(spec.host, parseIPList(spec.ips), spec.names)
	if err != nil {
		return "", "", err
	}

	certFile, err := ioutil.TempFile(dir, "cert")
	if err != nil {
		return "", "", err
	}

	keyFile, err := ioutil.TempFile(dir, "key")
	if err != nil {
		return "", "", err
	}

	_, err = certFile.Write(certPem)
	if err != nil {
		return "", "", err
	}
	certFile.Close()

	_, err = keyFile.Write(keyPem)
	if err != nil {
		return "", "", err
	}
	keyFile.Close()

	return certFile.Name(), keyFile.Name(), nil
}

func caCertFromBundle(bundlePath string) (*x509.Certificate, error) {
	pemData, err := ioutil.ReadFile(bundlePath)
	if err != nil {
		return nil, err
	}

	// fetch last block
	var block *pem.Block
	for {
		var nextBlock *pem.Block
		nextBlock, pemData = pem.Decode(pemData)
		if nextBlock == nil {
			if block == nil {
				return nil, fmt.Errorf("no certificate found in %q", bundlePath)

			}
			return x509.ParseCertificate(block.Bytes)
		}
		block = nextBlock
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
	return certSignature(cert)
}

func certSignature(cert tls.Certificate) (string, error) {
	x509Certs, err := x509.ParseCertificates(cert.Certificate[0])
	if err != nil {
		return "", err
	}
	return x509CertSignature(x509Certs[0]), nil
}
