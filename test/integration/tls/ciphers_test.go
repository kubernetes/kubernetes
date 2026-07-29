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

package tls

import (
	"crypto/tls"
	"fmt"
	"net/http"
	"strings"
	"testing"

	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func runBasicSecureAPIServer(t *testing.T, ciphers []string) (kubeapiservertesting.TearDownFunc, int) {
	flags := []string{"--tls-cipher-suites", strings.Join(ciphers, ",")}
	testServer := kubeapiservertesting.StartTestServerOrDie(t, nil, flags, framework.SharedEtcd())
	return testServer.TearDownFn, testServer.ServerOpts.SecureServing.BindPort
}

func TestAPICiphers(t *testing.T) {

	basicServerCiphers := []string{"TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256", "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256", "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384", "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384", "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305", "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305", "TLS_RSA_WITH_AES_128_CBC_SHA", "TLS_RSA_WITH_AES_256_CBC_SHA", "TLS_RSA_WITH_AES_128_GCM_SHA256", "TLS_RSA_WITH_AES_256_GCM_SHA384", "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA", "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA", "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA", "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA"}

	tearDown, port := runBasicSecureAPIServer(t, basicServerCiphers)
	defer tearDown()
	tests := []struct {
		clientCiphers []uint16
		expectedError bool
	}{
		{
			// Not supported cipher
			clientCiphers: []uint16{tls.TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA},
			expectedError: true,
		},
		{
			// Supported cipher
			clientCiphers: []uint16{tls.TLS_RSA_WITH_AES_256_CBC_SHA},
			expectedError: false,
		},
	}

	for i, test := range tests {
		runTestAPICiphers(t, i, port, test.clientCiphers, test.expectedError)
	}
}

func runTestAPICiphers(t *testing.T, testID int, kubePort int, clientCiphers []uint16, expectedError bool) {

	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			MaxVersion:         tls.VersionTLS12, // Limit to TLS1.2 to allow cipher configuration
			InsecureSkipVerify: true,
			CipherSuites:       clientCiphers,
		},
	}
	client := &http.Client{Transport: tr}
	req, err := http.NewRequest("GET", fmt.Sprintf("https://127.0.0.1:%d", kubePort), nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.Do(req)
	if err == nil {
		defer resp.Body.Close()
	}

	if expectedError && err == nil {
		t.Fatalf("%d: expecting error for cipher test, client cipher is supported and it should't", testID)
	} else if err != nil && !expectedError {
		t.Fatalf("%d: not expecting error by client with cipher failed: %+v", testID, err)
	}
}
