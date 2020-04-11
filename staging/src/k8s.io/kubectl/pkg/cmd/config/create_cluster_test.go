/*
Copyright 2017 The Kubernetes Authors.

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

package config

import (
	"bytes"
	"encoding/base64"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
)

type createClusterTest struct {
	description    string
	config         clientcmdapi.Config
	args           []string
	flags          []string
	expected       string
	expectedError  string
	expectedConfig clientcmdapi.Config
}

// This is a certificate generated for testing and has no other significance
const testCertificate string = `
-----BEGIN CERTIFICATE-----
MIIFazCCA1OgAwIBAgIUenSLpTUWjs+okVvbEx4YRX0rV1MwDQYJKoZIhvcNAQEL
BQAwRTELMAkGA1UEBhMCQVUxEzARBgNVBAgMClNvbWUtU3RhdGUxITAfBgNVBAoM
GEludGVybmV0IFdpZGdpdHMgUHR5IEx0ZDAeFw0yMDA0MTAxNjQyMTlaFw0zMDA0
MDgxNjQyMTlaMEUxCzAJBgNVBAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEw
HwYDVQQKDBhJbnRlcm5ldCBXaWRnaXRzIFB0eSBMdGQwggIiMA0GCSqGSIb3DQEB
AQUAA4ICDwAwggIKAoICAQC4qiGIPgOkrDAFVAilBwAGWrcAdcYRvc4fU9/dafwd
XM4IAlOa8qtMXVw9rdomSlKXaxfaIRt7ycswLrSoVimy55g7DN7lNqDFjE6agARH
EkzbYQ+WrDR3TgJf8vCqDgBwsCzz3/HWHMWXueN3YIFgKKDBxPsiA6XdqJv215Q6
PETxcb6h4ay2Xo+MavSA7F5onrEvRGLJfnSOsd8HOZ3vATgKLPGLjew0HF57T/fR
C8WLlq4TDbQknn4Hsou25tkIVANN7fYyyL7gXWKui+Kjm2+uCMacBN6kmfuCrRTW
4CTIUrFR8iUIGcRvccND9JujgrzuRHpBNwov8yVXP9xbnEtoGhbyVZpTWPPr9rdz
HXOJaMcSvffemA95ImSQ3qxnSuVnJc8jJKUiUFZIwga26k6BFfIIKxSvX8sBFLPU
gTbFoAbf9Z24lHpL4od3e7reuWmhIOEFC7uzQFOor6/yfiIXss6xRDO356Zizfh6
c7sy/MGLIm9jk7w7X/l+ND75lI2tuGQxD9Gy116f6ZLnqXFmLc7G+QMzzmXREHlm
M3xDdayrcxR5hpZNPLzISbSF6484JWImvnYqCaLP7VFg9vWKVB0o5NgEdbAb0AXM
rMKC+6Q9SFGMpwu6HwvQwQUseCqWCUjmiuZCPKzmVf5oqmuI4iQclf4RhNrGHn93
VQIDAQABo1MwUTAdBgNVHQ4EFgQUkM9FpYk9oMPNEbGHdpn8PNxOWUUwHwYDVR0j
BBgwFoAUkM9FpYk9oMPNEbGHdpn8PNxOWUUwDwYDVR0TAQH/BAUwAwEB/zANBgkq
hkiG9w0BAQsFAAOCAgEAagMpJAFzeZ2U1mZkDZkJrIcuWOFP2rTzjnBMELjqzqcs
8sqboRK9ef2XpHRXtvlDuBcFBwXaB/dVxb7pjVRxCbG7QVN1bhztrtTUmGoRa4aS
3IOW9WOxOq+Oywj9GPud/dapfuk43y/RLT3m0A32AtborxgoSiLAKKbZ8RqoGqih
hukwLhgpKjOrp6Ibqv5OfK93WE/RCAWz/H7bPCaLoMK/DFtQiZMK+PRnvyEWvpnt
pdcKqgSVKvBKwnJaRICl/vR+evsQHXYe0B+N27aMcJ4nU3Dxw2bsuma7c5Gy00FH
U9ZpR4+XPqdjHtqFkw7m/UPomb2OPfwSiEfE7yoX139Kqk9159otyOzue+C86gJ+
hos631VEM/WWr6wXEZe7j9sQJneZ9eflvXlbtAWEpxxX/VLOzOzPALFIpPzOFigG
pJQftZWJY4Hrj93x+tJwxGB61QMjM8OPdvvpj8+8vQ1Ue8YRiD86+b6IqaX0uBM9
g7vtDs8oizDnwUomr5K5nQRixIJO3CA3Tp8wJxRYmy92e+oKxO3gdGfQ+1vUUAmn
811Kkf+41cHP99Is9Km6bKTI0ttsh90OAhx4fz1DfiuX8zdXdxGpvcGyyxltDROv
rSCNAJXLEWEAHZz8UNCqr53mxldt9dOkpkmZYX3KZY3/xibcBcALp5CQHlQcRsY=
-----END CERTIFICATE-----
`

func TestCreateCluster(t *testing.T) {
	conf := clientcmdapi.Config{}
	test := createClusterTest{
		description: "Testing 'kubectl config set-cluster' with a new cluster",
		config:      conf,
		args:        []string{"my-cluster"},
		flags: []string{
			"--server=http://192.168.0.1",
			"--tls-server-name=my-cluster-name",
		},
		expected: `Cluster "my-cluster" set.` + "\n",
		expectedConfig: clientcmdapi.Config{
			Clusters: map[string]*clientcmdapi.Cluster{
				"my-cluster": {Server: "http://192.168.0.1", TLSServerName: "my-cluster-name"},
			},
		},
	}
	test.run(t)
}

func TestModifyCluster(t *testing.T) {
	conf := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"my-cluster": {Server: "https://192.168.0.1", TLSServerName: "to-be-cleared"},
		},
	}
	test := createClusterTest{
		description: "Testing 'kubectl config set-cluster' with an existing cluster",
		config:      conf,
		args:        []string{"my-cluster"},
		flags: []string{
			"--server=https://192.168.0.99",
		},
		expected: `Cluster "my-cluster" set.` + "\n",
		expectedConfig: clientcmdapi.Config{
			Clusters: map[string]*clientcmdapi.Cluster{
				"my-cluster": {Server: "https://192.168.0.99"},
			},
		},
	}
	test.run(t)
}

func TestModifyClusterServerAndTLS(t *testing.T) {
	conf := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"my-cluster": {Server: "https://192.168.0.1"},
		},
	}
	test := createClusterTest{
		description: "Testing 'kubectl config set-cluster' with an existing cluster",
		config:      conf,
		args:        []string{"my-cluster"},
		flags: []string{
			"--server=https://192.168.0.99",
			"--tls-server-name=my-cluster-name",
		},
		expected: `Cluster "my-cluster" set.` + "\n",
		expectedConfig: clientcmdapi.Config{
			Clusters: map[string]*clientcmdapi.Cluster{
				"my-cluster": {Server: "https://192.168.0.99", TLSServerName: "my-cluster-name"},
			},
		},
	}
	test.run(t)
}

func TestCertificateAuthorityFlags(t *testing.T) {

	// Create a temp file containing the ca for testing
	d, err := ioutil.TempDir("", "kubectl-test")
	if err != nil {
		t.Fatalf("tempdir: %v", err)
	}
	defer os.RemoveAll(d)
	caPath := filepath.Join(d, "test-ca.crt")
	ioutil.WriteFile(caPath, []byte(testCertificate), 0644)

	conf := clientcmdapi.Config{}
	tests := []createClusterTest{
		{
			description: "CertificateAuthority should contain ca filename",
			config:      conf,
			args:        []string{"my-cluster"},
			flags: []string{
				"--server=http://192.168.0.1",
				"--tls-server-name=my-tls-server-name",
				"--certificate-authority=" + caPath,
			},
			expected: "Cluster \"my-cluster\" set.\n",
			expectedConfig: clientcmdapi.Config{
				Clusters: map[string]*clientcmdapi.Cluster{
					"my-cluster": {
						Server:                   "http://192.168.0.1",
						TLSServerName:            "my-tls-server-name",
						CertificateAuthority:     caPath,
						CertificateAuthorityData: nil,
					},
				},
			},
		},
		{
			description: "CertificateAuthorityData should contain data from ca file if --embed-certs flag is specified",
			config:      conf,
			args:        []string{"my-cluster"},
			flags: []string{
				"--server=http://192.168.0.1",
				"--tls-server-name=my-tls-server-name",
				"--certificate-authority=" + caPath,
				"--embed-certs",
			},
			expected: "Cluster \"my-cluster\" set.\n",
			expectedConfig: clientcmdapi.Config{
				Clusters: map[string]*clientcmdapi.Cluster{
					"my-cluster": {
						Server:                   "http://192.168.0.1",
						TLSServerName:            "my-tls-server-name",
						CertificateAuthority:     "",
						CertificateAuthorityData: []byte(testCertificate),
					},
				},
			},
		},
		{
			description: "CertificateAuthorityData should contain base64 decoded data --certificate-authority-data flag",
			config:      conf,
			args:        []string{"my-cluster"},
			flags: []string{
				"--server=http://192.168.0.1",
				"--tls-server-name=my-tls-server-name",
				"--certificate-authority-data=" + base64.StdEncoding.EncodeToString([]byte(testCertificate)),
			},
			expected: "Cluster \"my-cluster\" set.\n",
			expectedConfig: clientcmdapi.Config{
				Clusters: map[string]*clientcmdapi.Cluster{
					"my-cluster": {
						Server:                   "http://192.168.0.1",
						TLSServerName:            "my-tls-server-name",
						CertificateAuthority:     "",
						CertificateAuthorityData: []byte(testCertificate),
					},
				},
			},
		},
		{
			description: "SetCluster should fail if --embed-certs is specified without --certificate-authority or --certificate-authority-data",
			config:      conf,
			args:        []string{"my-cluster"},
			flags: []string{
				"--server=http://192.168.0.1",
				"--tls-server-name=my-tls-server-name",
				"--embed-certs",
			},
			expectedError: "error: you must specify a --certificate-authority to embed",
		},
		{
			description: "SetCluster should fail if both --certificate-authority and --certificate-authority-data are specified",
			config:      conf,
			args:        []string{"my-cluster"},
			flags: []string{
				"--server=http://192.168.0.1",
				"--tls-server-name=my-tls-server-name",
				"--certificate-authority=" + caPath,
				"--certificate-authority-data=" + base64.StdEncoding.EncodeToString([]byte(testCertificate)),
			},
			expectedError: "error: you cannot specify a certificate authority file and certificate authority data at the same time",
		},
		{
			description: "SetCluster should fail if --certificate-authority and --insecure-skip-tls-verify are specified",
			config:      conf,
			args:        []string{"my-cluster"},
			flags: []string{
				"--server=http://192.168.0.1",
				"--tls-server-name=my-tls-server-name",
				"--certificate-authority=" + caPath,
				"--insecure-skip-tls-verify",
			},
			expectedError: "error: you cannot specify a certificate authority and insecure mode at the same time",
		},
		{
			description: "SetCluster should fail if --certificate-authority-data and --insecure-skip-tls-verify are specified",
			config:      conf,
			args:        []string{"my-cluster"},
			flags: []string{
				"--server=http://192.168.0.1",
				"--tls-server-name=my-tls-server-name",
				"--certificate-authority-data=" + base64.StdEncoding.EncodeToString([]byte(testCertificate)),
				"--insecure-skip-tls-verify",
			},
			expectedError: "error: you cannot specify a certificate authority and insecure mode at the same time",
		},
		{
			description: "SetCluster should fail if --certificate-authority-data does not contain base64 encoded data",
			config:      conf,
			args:        []string{"my-cluster"},
			flags: []string{
				"--server=http://192.168.0.1",
				"--tls-server-name=my-tls-server-name",
				"--certificate-authority-data=foo",
			},
			expectedError: "error: could not decode certificate authority data: illegal base64 data at input byte 0",
		},
		{
			description: "SetCluster should fail if --embed-certs is specified and --certificate-authority file does not exist",
			config:      conf,
			args:        []string{"my-cluster"},
			flags: []string{
				"--server=http://192.168.0.1",
				"--tls-server-name=my-tls-server-name",
				"--certificate-authority=/file/does/not/exist",
				"--embed-certs",
			},
			expectedError: "error: could not read certificate authority data from /file/does/not/exist: open /file/does/not/exist: no such file or directory",
		},
	}
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			test.run(t)
		})
	}
}

func (test createClusterTest) run(t *testing.T) {
	fakeKubeFile, err := ioutil.TempFile(os.TempDir(), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(fakeKubeFile.Name())
	err = clientcmd.WriteToFile(test.config, fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = fakeKubeFile.Name()
	pathOptions.EnvVar = ""
	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdConfigSetCluster(buf, pathOptions)
	cmd.SetArgs(test.args)
	cmd.Flags().Parse(test.flags)

	var actualError string
	if len(test.expectedError) > 0 {
		cmdutil.BehaviorOnFatal(func(str string, code int) {
			actualError = str
		})
	}

	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v, args: %v, flags: %v", err, test.args, test.flags)
	}

	if len(test.expectedError) > 0 {
		if actualError == test.expectedError {
			return
		}
		t.Fatalf("Expected error %s but got %s", test.expectedError, actualError)
	}

	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}
	if len(test.expected) != 0 {
		if buf.String() != test.expected {
			t.Errorf("Failed in %q\n expected %v\n but got %v", test.description, test.expected, buf.String())
		}
	}
	if len(test.args) > 0 {
		cluster, ok := config.Clusters[test.args[0]]
		if !ok {
			t.Errorf("expected cluster %v, but got nil", test.args[0])
			return
		}
		expectedCluster := test.expectedConfig.Clusters[test.args[0]]
		if cluster.Server != expectedCluster.Server {
			t.Errorf("Fail in %q\n expectedCluster cluster server %v\n but got %v\n ", test.description, expectedCluster.Server, cluster.Server)
		}
		if cluster.TLSServerName != expectedCluster.TLSServerName {
			t.Errorf("Fail in %q\n expectedCluster cluster TLS server name %q\n but got %q\n ", test.description, expectedCluster.TLSServerName, cluster.TLSServerName)
		}

		// The actual cluster has a location of origin in the temp folder.
		// In order to compare the expected and actual paths, we need to also relativize the expected cluster's paths.
		expectedCluster.LocationOfOrigin = cluster.LocationOfOrigin
		clientcmd.RelativizeClusterLocalPaths(expectedCluster)

		if cluster.CertificateAuthority != expectedCluster.CertificateAuthority {
			t.Errorf("Fail in %q\n expectedCluster cluster certificate authority %q\n but got %q\n ", test.description, expectedCluster.CertificateAuthority, cluster.CertificateAuthority)
		}
		if !reflect.DeepEqual(cluster.CertificateAuthorityData, expectedCluster.CertificateAuthorityData) {
			t.Errorf("Fail in %q\n expectedCluster cluster certificate authority data %q\n but got %q\n ", test.description, expectedCluster.CertificateAuthorityData, cluster.CertificateAuthorityData)
		}
	}
}
