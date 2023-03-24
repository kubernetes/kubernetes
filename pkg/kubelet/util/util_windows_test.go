//go:build windows
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
	"fmt"
	"math/rand"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"

	winio "github.com/Microsoft/go-winio"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetAddressAndDialer(t *testing.T) {

	// Compare dialer function by pointer
	tcpDialPointer := reflect.ValueOf(tcpDial).Pointer()
	npipeDialPointer := reflect.ValueOf(npipeDial).Pointer()
	var nilDialPointer uintptr = 0x0

	tests := []struct {
		endpoint      string
		expectedAddr  string
		expectedDial  uintptr
		expectedError bool
	}{
		{
			endpoint:      "tcp://localhost:15880",
			expectedAddr:  "localhost:15880",
			expectedDial:  tcpDialPointer,
			expectedError: false,
		},
		{
			endpoint:      "npipe://./pipe/mypipe",
			expectedAddr:  "//./pipe/mypipe",
			expectedDial:  npipeDialPointer,
			expectedError: false,
		},
		{
			endpoint:      "npipe:\\\\.\\pipe\\mypipe",
			expectedAddr:  "//./pipe/mypipe",
			expectedDial:  npipeDialPointer,
			expectedError: false,
		},
		{
			endpoint:      "unix:///tmp/s1.sock",
			expectedAddr:  "",
			expectedDial:  nilDialPointer,
			expectedError: true,
		},
		{
			endpoint:      "tcp1://abc",
			expectedAddr:  "",
			expectedDial:  nilDialPointer,
			expectedError: true,
		},
		{
			endpoint:      "a b c",
			expectedAddr:  "",
			expectedDial:  nilDialPointer,
			expectedError: true,
		},
	}

	for _, test := range tests {
		expectedDialerName := runtime.FuncForPC(test.expectedDial).Name()
		if expectedDialerName == "" {
			expectedDialerName = "nilDial"
		}
		t.Run(fmt.Sprintf("Endpoint is %s, addr is %s and dialer is %s",
			test.endpoint, test.expectedAddr, expectedDialerName),
			func(t *testing.T) {
				address, dialer, err := GetAddressAndDialer(test.endpoint)

				dialerPointer := reflect.ValueOf(dialer).Pointer()
				actualDialerName := runtime.FuncForPC(dialerPointer).Name()
				if actualDialerName == "" {
					actualDialerName = "nilDial"
				}

				assert.Equalf(t, test.expectedDial, dialerPointer,
					"Expected dialer %s, but get %s", expectedDialerName, actualDialerName)

				assert.Equal(t, test.expectedAddr, address)

				if test.expectedError {
					assert.NotNil(t, err, "Expect error during parsing %q", test.endpoint)
				} else {
					assert.Nil(t, err, "Expect no error during parsing %q", test.endpoint)
				}
			})
	}
}

func TestParseEndpoint(t *testing.T) {
	tests := []struct {
		endpoint         string
		expectedError    bool
		expectedProtocol string
		expectedAddr     string
	}{
		{
			endpoint:         "unix:///tmp/s1.sock",
			expectedProtocol: "unix",
			expectedError:    true,
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
			expectedError:    true,
		},
		{
			endpoint:      "a b c",
			expectedError: true,
		},
	}

	for _, test := range tests {
		protocol, addr, err := parseEndpoint(test.endpoint)
		assert.Equal(t, test.expectedProtocol, protocol)
		if test.expectedError {
			assert.NotNil(t, err, "Expect error during parsing %q", test.endpoint)
			continue
		}
		require.Nil(t, err, "Expect no error during parsing %q", test.endpoint)
		assert.Equal(t, test.expectedAddr, addr)
	}

}

func TestIsUnixDomainSocketPipe(t *testing.T) {
	generatePipeName := func(suffixLen int) string {
		rand.Seed(time.Now().UnixNano())
		letter := []rune("abcdef0123456789")
		b := make([]rune, suffixLen)
		for i := range b {
			b[i] = letter[rand.Intn(len(letter))]
		}
		return "\\\\.\\pipe\\test-pipe" + string(b)
	}
	testFile := generatePipeName(4)
	pipeln, err := winio.ListenPipe(testFile, &winio.PipeConfig{SecurityDescriptor: "D:P(A;;GA;;;BA)(A;;GA;;;SY)"})
	defer pipeln.Close()

	require.NoErrorf(t, err, "Failed to listen on named pipe for test purposes: %v", err)
	result, err := IsUnixDomainSocket(testFile)
	assert.NoError(t, err, "Unexpected error from IsUnixDomainSocket.")
	assert.False(t, result, "Unexpected result: true from IsUnixDomainSocket.")
}

// This is required as on Windows it's possible for the socket file backing a Unix domain socket to
// exist but not be ready for socket communications yet as per
// https://github.com/kubernetes/kubernetes/issues/104584
func TestPendingUnixDomainSocket(t *testing.T) {
	// Create a temporary file that will simulate the Unix domain socket file in a
	// not-yet-ready state. We need this because the Kubelet keeps an eye on file
	// changes and acts on them, leading to potential race issues as described in
	// the referenced issue above
	f, err := os.CreateTemp("", "test-domain-socket")
	require.NoErrorf(t, err, "Failed to create file for test purposes: %v", err)
	testFile := f.Name()
	f.Close()

	// Start the check at this point
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		result, err := IsUnixDomainSocket(testFile)
		assert.Nil(t, err, "Unexpected error from IsUnixDomainSocket: %v", err)
		assert.True(t, result, "Unexpected result: false from IsUnixDomainSocket.")
		wg.Done()
	}()

	// Wait a sufficient amount of time to make sure the retry logic kicks in
	time.Sleep(socketDialRetryPeriod)

	// Replace the temporary file with an actual Unix domain socket file
	os.Remove(testFile)
	ta, err := net.ResolveUnixAddr("unix", testFile)
	require.NoError(t, err, "Failed to ResolveUnixAddr.")
	unixln, err := net.ListenUnix("unix", ta)
	require.NoError(t, err, "Failed to ListenUnix.")

	// Wait for the goroutine to finish, then close the socket
	wg.Wait()
	unixln.Close()
}

func TestNormalizePath(t *testing.T) {
	tests := []struct {
		originalpath   string
		normalizedPath string
	}{
		{
			originalpath:   "\\path\\to\\file",
			normalizedPath: "c:\\path\\to\\file",
		},
		{
			originalpath:   "/path/to/file",
			normalizedPath: "c:\\path\\to\\file",
		},
		{
			originalpath:   "/path/to/dir/",
			normalizedPath: "c:\\path\\to\\dir\\",
		},
		{
			originalpath:   "\\path\\to\\dir\\",
			normalizedPath: "c:\\path\\to\\dir\\",
		},
		{
			originalpath:   "/file",
			normalizedPath: "c:\\file",
		},
		{
			originalpath:   "\\file",
			normalizedPath: "c:\\file",
		},
		{
			originalpath:   "fileonly",
			normalizedPath: "fileonly",
		},
	}

	for _, test := range tests {
		assert.Equal(t, test.normalizedPath, NormalizePath(test.originalpath))
	}
}
