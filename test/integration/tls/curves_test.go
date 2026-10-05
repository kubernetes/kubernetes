/*
Copyright The Kubernetes Authors.

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

func runSecureAPIServerWithCurves(t *testing.T, curves []string) (kubeapiservertesting.TearDownFunc, int) {
	flags := []string{"--tls-curve-preferences", strings.Join(curves, ",")}
	testServer := kubeapiservertesting.StartTestServerOrDie(t, nil, flags, framework.SharedEtcd())
	return testServer.TearDownFn, testServer.ServerOpts.SecureServing.BindPort
}

func TestAPICurvePreferences(t *testing.T) {
	serverCurves := []string{"23", "24", "25", "29", "4588"}

	tearDown, port := runSecureAPIServerWithCurves(t, serverCurves)
	defer tearDown()

	tests := []struct {
		name          string
		clientCurves  []tls.CurveID
		minTLSVersion uint16
		expectError   bool
	}{
		{
			name:          "supported curve",
			clientCurves:  []tls.CurveID{tls.CurveP256},
			minTLSVersion: tls.VersionTLS12,
			expectError:   false,
		},
		{
			name:          "supported curve",
			clientCurves:  []tls.CurveID{tls.CurveP384},
			minTLSVersion: tls.VersionTLS12,
			expectError:   false,
		},
		{
			name:          "supported curve",
			clientCurves:  []tls.CurveID{tls.CurveP521},
			minTLSVersion: tls.VersionTLS12,
			expectError:   false,
		},
		{
			name:          "supported curve",
			clientCurves:  []tls.CurveID{tls.X25519MLKEM768},
			minTLSVersion: tls.VersionTLS13,
			expectError:   false,
		},
		{
			name:          "supported curve",
			clientCurves:  []tls.CurveID{tls.X25519},
			minTLSVersion: tls.VersionTLS13,
			expectError:   false,
		},
	}

	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runTestAPICurves(t, i, port, test.clientCurves, test.minTLSVersion, test.expectError)
		})
	}
}

func runTestAPICurves(t *testing.T, testID int, kubePort int, clientCurves []tls.CurveID, minTLSVersion uint16, expectedError bool) {
	t.Helper()

	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			MinVersion:         minTLSVersion,
			InsecureSkipVerify: true,
			CurvePreferences:   clientCurves,
		},
	}
	client := &http.Client{Transport: tr}
	req, err := http.NewRequest(http.MethodGet, fmt.Sprintf("https://127.0.0.1:%d", kubePort), nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.Do(req)
	if err == nil {
		defer func() { _ = resp.Body.Close() }()
	}

	if expectedError && err == nil {
		t.Fatalf("%d: expecting error for curve test, but client curve was accepted and it shouldn't have been", testID)
	} else if err != nil && !expectedError {
		t.Fatalf("%d: not expecting error but client with curve failed: %+v", testID, err)
	}
}
