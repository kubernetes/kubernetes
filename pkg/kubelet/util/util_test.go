/*
Copyright 2020 The Kubernetes Authors.

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

package util

import (
	"net"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetNodenameForKernel(t *testing.T) {
	testcases := []struct {
		description       string
		hostname          string
		hostDomain        string
		setHostnameAsFQDN bool
		expectedHostname  string
		expectError       bool
	}{{
		description:       "no hostDomain, setHostnameAsFQDN false",
		hostname:          "test.pod.hostname",
		hostDomain:        "",
		setHostnameAsFQDN: false,
		expectedHostname:  "test.pod.hostname",
		expectError:       false,
	}, {
		description:       "no hostDomain, setHostnameAsFQDN true",
		hostname:          "test.pod.hostname",
		hostDomain:        "",
		setHostnameAsFQDN: true,
		expectedHostname:  "test.pod.hostname",
		expectError:       false,
	}, {
		description:       "valid hostDomain, setHostnameAsFQDN false",
		hostname:          "test.pod.hostname",
		hostDomain:        "svc.subdomain.local",
		setHostnameAsFQDN: false,
		expectedHostname:  "test.pod.hostname",
		expectError:       false,
	}, {
		description:       "valid hostDomain, setHostnameAsFQDN true",
		hostname:          "test.pod.hostname",
		hostDomain:        "svc.subdomain.local",
		setHostnameAsFQDN: true,
		expectedHostname:  "test.pod.hostname.svc.subdomain.local",
		expectError:       false,
	}, {
		description:       "FQDN is too long, setHostnameAsFQDN false",
		hostname:          "1234567.1234567",                                         //8*2-1=15 chars
		hostDomain:        "1234567.1234567.1234567.1234567.1234567.1234567.1234567", //8*7-1=55 chars
		setHostnameAsFQDN: false,                                                     //FQDN=15 + 1(dot) + 55 = 71 chars
		expectedHostname:  "1234567.1234567",
		expectError:       false,
	}, {
		description:       "FQDN is too long, setHostnameAsFQDN true",
		hostname:          "1234567.1234567",                                         //8*2-1=15 chars
		hostDomain:        "1234567.1234567.1234567.1234567.1234567.1234567.1234567", //8*7-1=55 chars
		setHostnameAsFQDN: true,                                                      //FQDN=15 + 1(dot) + 55 = 71 chars
		expectedHostname:  "",
		expectError:       true,
	}}

	for _, tc := range testcases {
		t.Logf("TestCase: %q", tc.description)
		outputHostname, err := GetNodenameForKernel(tc.hostname, tc.hostDomain, &tc.setHostnameAsFQDN)
		if tc.expectError {
			assert.Error(t, err)
		} else {
			assert.NoError(t, err)
		}
		assert.Equal(t, tc.expectedHostname, outputHostname)
	}

}

func TestIsUnixDomainSocket(t *testing.T) {
	tests := []struct {
		label          string
		listenOnSocket bool
		expectSocket   bool
		expectError    bool
		invalidFile    bool
	}{
		{
			label:          "Domain Socket file",
			listenOnSocket: true,
			expectSocket:   true,
			expectError:    false,
		},
		{
			label:       "Non Existent file",
			invalidFile: true,
			expectError: true,
		},
		{
			label:          "Regular file",
			listenOnSocket: false,
			expectSocket:   false,
			expectError:    false,
		},
	}
	for _, test := range tests {
		f, err := os.CreateTemp("", "test-domain-socket")
		require.NoErrorf(t, err, "Failed to create file for test purposes: %v while setting up: %s", err, test.label)
		addr := f.Name()
		f.Close()
		var ln *net.UnixListener
		if test.listenOnSocket {
			os.Remove(addr)
			ta, err := net.ResolveUnixAddr("unix", addr)
			require.NoErrorf(t, err, "Failed to ResolveUnixAddr: %v while setting up: %s", err, test.label)
			ln, err = net.ListenUnix("unix", ta)
			require.NoErrorf(t, err, "Failed to ListenUnix: %v while setting up: %s", err, test.label)
		}
		fileToTest := addr
		if test.invalidFile {
			fileToTest = fileToTest + ".invalid"
		}
		result, err := IsUnixDomainSocket(fileToTest)
		if test.listenOnSocket {
			// this takes care of removing the file associated with the domain socket
			ln.Close()
		} else {
			// explicitly remove regular file
			os.Remove(addr)
		}
		if test.expectError {
			assert.Errorf(t, err, "Unexpected nil error from IsUnixDomainSocket for %s", test.label)
		} else {
			assert.NoErrorf(t, err, "Unexpected error invoking IsUnixDomainSocket for %s", test.label)
		}
		assert.Equal(t, result, test.expectSocket, "Unexpected result from IsUnixDomainSocket: %v for %s", result, test.label)
	}
}
