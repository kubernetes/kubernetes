/*
Copyright 2023 The Kubernetes Authors.

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

package filesystem

import (
	"net"
	"os"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
		{
			label:       "Empty file path",
			invalidFile: true,
			expectError: true,
		},
		{
			label:          "Socket on different path",
			listenOnSocket: true,
			expectSocket:   true,
			expectError:    false,
		},
		{
			label:       "Directory instead of file",
			invalidFile: true,
			expectError: true,
		},
		{
			label:       "Regular file with socket name",
			invalidFile: true,
			expectError: true,
		},
		{
			label:          "Multiple Unix Domain Sockets",
			listenOnSocket: true,
			expectSocket:   true,
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

func TestIsCleanPath(t *testing.T) {
	type Case struct {
		path     string
		expected bool
	}

	// Credits https://github.com/kubernetes/kubernetes/pull/124084/files#r1557566941
	cases := []Case{
		{path: "/logs", expected: true},
		{path: "/var/lib/something", expected: true},
		{path: "var/lib/something", expected: true},
		{path: "var\\lib\\something", expected: true},
		{path: "/", expected: true},
		{path: ".", expected: true},
		{path: "/var/../something", expected: false},
		{path: "/var//lib/something", expected: false},
		{path: "/var/./lib/something", expected: false},
	}

	// Additional cases applicable on Windows
	if runtime.GOOS == "windows" {
		cases = append(cases, []Case{
			{path: "\\", expected: true},
			{path: "C:/var/lib/something", expected: true},
			{path: "C:\\var\\lib\\something", expected: true},
			{path: "C:/", expected: true},
			{path: "C:\\", expected: true},
			{path: "C:/var//lib/something", expected: false},
			{path: "\\var\\\\lib\\something", expected: false},
			{path: "C:\\var\\\\lib\\something", expected: false},
			{path: "C:\\var\\..\\something", expected: false},
			{path: "\\var\\.\\lib\\something", expected: false},
			{path: "C:\\var\\.\\lib\\something", expected: false},
		}...)
	}

	for _, tc := range cases {
		actual := IsPathClean(tc.path)
		if actual != tc.expected {
			t.Errorf("actual: %t, expected: %t, for path: %s\n", actual, tc.expected, tc.path)
		}
	}

}
