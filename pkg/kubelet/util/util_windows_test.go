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
	"context"
	"net"
	"reflect"
	"runtime"
	"testing"

	"github.com/Microsoft/go-winio"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/cri-client/pkg/util"
)

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

func TestLocalEndpoint(t *testing.T) {
	tests := []struct {
		path             string
		file             string
		expectError      bool
		expectedFullPath string
	}{
		{
			path:             "/var/lib/kubelet/pod-resources",
			file:             "kube.sock", // this is not the default, but it's not relevant here
			expectError:      false,
			expectedFullPath: `npipe://\\.\pipe\kubelet-pod-resources`,
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

func npipeDial(ctx context.Context, addr string) (net.Conn, error) {
	return winio.DialPipeContext(ctx, addr)
}

func TestLocalEndpointRoundTrip(t *testing.T) {
	npipeDialPointer := reflect.ValueOf(npipeDial).Pointer()
	expectedDialerName := runtime.FuncForPC(npipeDialPointer).Name()
	expectedAddress := "//./pipe/kubelet-pod-resources"

	fullPath, err := LocalEndpoint(`pod-resources`, "kubelet")
	require.NoErrorf(t, err, "Failed to create the local endpoint path")

	address, dialer, err := util.GetAddressAndDialer(fullPath)
	require.NoErrorf(t, err, "Failed to parse the endpoint path and get back address and dialer (path=%q)", fullPath)

	dialerPointer := reflect.ValueOf(dialer).Pointer()
	actualDialerName := runtime.FuncForPC(dialerPointer).Name()

	assert.Equalf(t, npipeDialPointer, dialerPointer,
		"Expected dialer %s, but get %s", expectedDialerName, actualDialerName)

	assert.Equal(t, expectedAddress, address)
}
