//go:build freebsd || linux || darwin
// +build freebsd linux darwin

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

package util

import (
	"io/ioutil"
	"net"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseEndpoint(t *testing.T) {
	tests := []struct {
		endpoint         string
		expectError      bool
		expectedProtocol string
		expectedAddr     string
	}{
		{
			endpoint:         "unix:///tmp/s1.sock",
			expectedProtocol: "unix",
			expectedAddr:     "/tmp/s1.sock",
		},
		{
			endpoint:         "tcp://localhost:15880",
			expectedProtocol: "tcp",
			expectedAddr:     "localhost:15880",
		},
		{
			endpoint:         "npipe://./pipe/mypipe",
			expectedProtocol: "npipe",
			expectError:      true,
		},
		{
			endpoint:         "tcp1://abc",
			expectedProtocol: "tcp1",
			expectError:      true,
		},
		{
			endpoint:    "a b c",
			expectError: true,
		},
	}

	for _, test := range tests {
		protocol, addr, err := parseEndpoint(test.endpoint)
		assert.Equal(t, test.expectedProtocol, protocol)
		if test.expectError {
			assert.NotNil(t, err, "Expect error during parsing %q", test.endpoint)
			continue
		}
		assert.Nil(t, err, "Expect no error during parsing %q", test.endpoint)
		assert.Equal(t, test.expectedAddr, addr)
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
		f, err := ioutil.TempFile("", "test-domain-socket")
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
			assert.NotNil(t, err, "Unexpected nil error from IsUnixDomainSocket for %s", test.label)
		} else {
			assert.Nil(t, err, "Unexpected error invoking IsUnixDomainSocket for %s", test.label)
		}
		assert.Equal(t, result, test.expectSocket, "Unexpected result from IsUnixDomainSocket: %v for %s", result, test.label)
	}
}

func TestGetAddressAndDialer(t *testing.T) {
	tests := []struct {
		endpoint     string
		expectError  bool
		expectedAddr string
	}{
		{
			endpoint:     "unix:///tmp/s1.sock",
			expectError:  false,
			expectedAddr: "/tmp/s1.sock",
		},
		{
			endpoint:     "unix:///tmp/f6.sock",
			expectError:  false,
			expectedAddr: "/tmp/f6.sock",
		},
		{
			endpoint:    "tcp://localhost:9090",
			expectError: true,
		},
		{
			// The misspelling is intentional to make it error
			endpoint:    "htta://free-test.com",
			expectError: true,
		},
		{
			endpoint:    "https://www.youtube.com/",
			expectError: true,
		},
		{
			endpoint:    "http://www.baidu.com/",
			expectError: true,
		},
	}
	for _, test := range tests {
		// just test addr and err
		addr, _, err := GetAddressAndDialer(test.endpoint)
		if test.expectError {
			assert.NotNil(t, err, "expected error during parsing %s", test.endpoint)
			continue
		}
		assert.Nil(t, err, "expected no error during parsing %s", test.endpoint)
		assert.Equal(t, test.expectedAddr, addr)
	}
}

func TestLocalEndpoint(t *testing.T) {
	tests := []struct {
		path             string
		file             string
		expectError      bool
		expectedFullPath string
	}{
		{
			path:             "path",
			file:             "file",
			expectError:      false,
			expectedFullPath: "unix:/path/file.sock",
		},
	}
	for _, test := range tests {
		fullPath, err := LocalEndpoint(test.path, test.file)
		if test.expectError {
			assert.NotNil(t, err, "expected error")
			continue
		}
		assert.Nil(t, err, "expected no error")
		assert.Equal(t, test.expectedFullPath, fullPath)
	}
}
