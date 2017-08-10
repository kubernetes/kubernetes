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

package token

import (
	"strconv"
	"testing"
	"time"

	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

// testCertPEM is a simple self-signed test certificate issued with the openssl CLI:
// openssl req -new -newkey rsa:2048 -days 36500 -nodes -x509 -keyout /dev/null -out test.crt
const testCertPEM = `
-----BEGIN CERTIFICATE-----
MIIDRDCCAiygAwIBAgIJAJgVaCXvC6HkMA0GCSqGSIb3DQEBBQUAMB8xHTAbBgNV
BAMTFGt1YmVhZG0ta2V5cGlucy10ZXN0MCAXDTE3MDcwNTE3NDMxMFoYDzIxMTcw
NjExMTc0MzEwWjAfMR0wGwYDVQQDExRrdWJlYWRtLWtleXBpbnMtdGVzdDCCASIw
DQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAK0ba8mHU9UtYlzM1Own2Fk/XGjR
J4uJQvSeGLtz1hID1IA0dLwruvgLCPadXEOw/f/IWIWcmT+ZmvIHZKa/woq2iHi5
+HLhXs7aG4tjKGLYhag1hLjBI7icqV7ovkjdGAt9pWkxEzhIYClFMXDjKpMSynu+
YX6nZ9tic1cOkHmx2yiZdMkuriRQnpTOa7bb03OC1VfGl7gHlOAIYaj4539WCOr8
+ACTUMJUFEHcRZ2o8a/v6F9GMK+7SC8SJUI+GuroXqlMAdhEv4lX5Co52enYaClN
+D9FJLRpBv2YfiCQdJRaiTvCBSxEFz6BN+PtP5l2Hs703ZWEkOqCByM6HV8CAwEA
AaOBgDB+MB0GA1UdDgQWBBRQgUX8MhK2rWBWQiPHWcKzoWDH5DBPBgNVHSMESDBG
gBRQgUX8MhK2rWBWQiPHWcKzoWDH5KEjpCEwHzEdMBsGA1UEAxMUa3ViZWFkbS1r
ZXlwaW5zLXRlc3SCCQCYFWgl7wuh5DAMBgNVHRMEBTADAQH/MA0GCSqGSIb3DQEB
BQUAA4IBAQCaAUif7Pfx3X0F08cxhx8/Hdx4jcJw6MCq6iq6rsXM32ge43t8OHKC
pJW08dk58a3O1YQSMMvD6GJDAiAfXzfwcwY6j258b1ZlI9Ag0VokvhMl/XfdCsdh
AWImnL1t4hvU5jLaImUUMlYxMcSfHBGAm7WJIZ2LdEfg6YWfZh+WGbg1W7uxLxk6
y4h5rWdNnzBHWAGf7zJ0oEDV6W6RSwNXtC0JNnLaeIUm/6xdSddJlQPwUv8YH4jX
c1vuFqTnJBPcb7W//R/GI2Paicm1cmns9NLnPR35exHxFTy+D1yxmGokpoPMdife
aH+sfuxT8xeTPb3kjzF9eJTlnEquUDLM
-----END CERTIFICATE-----`

func TestRunForEndpointsAndReturnFirst(t *testing.T) {
	tests := []struct {
		endpoints        []string
		expectedEndpoint string
	}{
		{
			endpoints:        []string{"1", "2", "3"},
			expectedEndpoint: "1",
		},
		{
			endpoints:        []string{"6", "5"},
			expectedEndpoint: "5",
		},
		{
			endpoints:        []string{"10", "4"},
			expectedEndpoint: "4",
		},
	}
	for _, rt := range tests {
		returnKubeConfig := runForEndpointsAndReturnFirst(rt.endpoints, func(endpoint string) (*clientcmdapi.Config, error) {
			timeout, _ := strconv.Atoi(endpoint)
			time.Sleep(time.Second * time.Duration(timeout))
			return kubeconfigutil.CreateBasic(endpoint, "foo", "foo", []byte{}), nil
		})
		endpoint := returnKubeConfig.Clusters[returnKubeConfig.Contexts[returnKubeConfig.CurrentContext].Cluster].Server
		if endpoint != rt.expectedEndpoint {
			t.Errorf(
				"failed TestRunForEndpointsAndReturnFirst:\n\texpected: %s\n\t  actual: %s",
				endpoint,
				rt.expectedEndpoint,
			)
		}
	}
}

func TestParsePEMCert(t *testing.T) {
	for _, testCase := range []struct {
		name        string
		input       []byte
		expectValid bool
	}{
		{"invalid certificate data", []byte{0}, false},
		{"certificate with junk appended", []byte(testCertPEM + "\nABC"), false},
		{"multiple certificates", []byte(testCertPEM + "\n" + testCertPEM), false},
		{"valid", []byte(testCertPEM), true},
	} {
		cert, err := parsePEMCert(testCase.input)
		if testCase.expectValid {
			if err != nil {
				t.Errorf("failed TestParsePEMCert(%s): unexpected error %v", testCase.name, err)
			}
			if cert == nil {
				t.Errorf("failed TestParsePEMCert(%s): returned nil", testCase.name)
			}
		} else {
			if err == nil {
				t.Errorf("failed TestParsePEMCert(%s): expected an error", testCase.name)
			}
			if cert != nil {
				t.Errorf("failed TestParsePEMCert(%s): expected not to get a certificate back, but got one", testCase.name)
			}
		}
	}
}
