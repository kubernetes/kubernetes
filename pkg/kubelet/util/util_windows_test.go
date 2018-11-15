// +build windows

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
			expectError:      true,
		},
		{
			endpoint:         "tcp://localhost:15880",
			expectedProtocol: "tcp",
			expectedAddr:     "localhost:15880",
		},
		{
			endpoint:         "npipe://./pipe/mypipe",
			expectedProtocol: "npipe",
			expectedAddr:     "//./pipe/mypipe",
		},
		{
			endpoint:         "npipe:////./pipe/mypipe2",
			expectedProtocol: "npipe",
			expectedAddr:     "//./pipe/mypipe2",
		},
		{
			endpoint:         "npipe:/pipe/mypipe3",
			expectedProtocol: "npipe",
			expectedAddr:     "//./pipe/mypipe3",
		},
		{
			endpoint:         "npipe:\\\\.\\pipe\\mypipe4",
			expectedProtocol: "npipe",
			expectedAddr:     "//./pipe/mypipe4",
		},
		{
			endpoint:         "npipe:\\pipe\\mypipe5",
			expectedProtocol: "npipe",
			expectedAddr:     "//./pipe/mypipe5",
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
		require.Nil(t, err, "Expect no error during parsing %q", test.endpoint)
		assert.Equal(t, test.expectedAddr, addr)
	}

}
