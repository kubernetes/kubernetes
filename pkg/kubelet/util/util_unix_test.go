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
	"testing"

	"github.com/stretchr/testify/assert"
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
