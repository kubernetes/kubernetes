//go:build windows
// +build windows

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

package hostutil

import (
	"net"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnixDomainSocket(t *testing.T) {
	f, err := os.CreateTemp("", "test-domain-socket")
	require.NoErrorf(t, err, "Failed to create file for test purposes: %v", err)
	testFile := f.Name()
	f.Close()
	os.Remove(testFile)
	ta, err := net.ResolveUnixAddr("unix", testFile)
	require.NoErrorf(t, err, "Failed to ResolveUnixAddr: %v", err)
	unixln, err := net.ListenUnix("unix", ta)
	require.NoErrorf(t, err, "Failed to ListenUnix: %v", err)
	result, err := IsUnixDomainSocket(testFile)
	unixln.Close()
	assert.Nil(t, err, "Unexpected error from IsUnixDomainSocket: %v", err)
	assert.True(t, result, "Unexpected result: false from IsUnixDomainSocket")
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
		assert.Nil(t, err, "Unexpected error: %v from IsUnixDomainSocket", err)
		assert.True(t, result, "Unexpected result: false from IsUnixDomainSocket")
		wg.Done()
	}()

	// Wait a sufficient amount of time to make sure the retry logic kicks in
	time.Sleep(socketDialRetryPeriod)

	// Replace the temporary file with an actual Unix domain socket file
	os.Remove(testFile)
	ta, err := net.ResolveUnixAddr("unix", testFile)
	require.NoErrorf(t, err, "Failed to ResolveUnixAddr: %v", err)
	unixln, err := net.ListenUnix("unix", ta)
	require.NoErrorf(t, err, "Failed to ListenUnix: %v", err)

	// Wait for the goroutine to finish, then close the socket
	wg.Wait()
	unixln.Close()
}
