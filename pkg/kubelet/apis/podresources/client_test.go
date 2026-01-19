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

package podresources

import (
	"net"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	v1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	"k8s.io/kubelet/pkg/apis/podresources/v1alpha1"
)

// clientTestCase defines a test case for client creation with different API versions.
// These unit tests focus on boundary conditions that are difficult to test in e2e:
// - Client connection logic with invalid paths
// - Timeout handling with short durations
// - Error scenarios that require specific network conditions
type clientTestCase struct {
	version     string
	getClientFn func(string, time.Duration, int) (interface{}, *grpc.ClientConn, error)
	registerFn  func(*grpc.Server)
}

func getClientTestCases() []clientTestCase {
	return []clientTestCase{
		{
			version: "v1alpha1",
			getClientFn: func(socket string, timeout time.Duration, maxSize int) (interface{}, *grpc.ClientConn, error) {
				return GetV1alpha1Client(socket, timeout, maxSize)
			},
			registerFn: func(server *grpc.Server) {
				v1alpha1.RegisterPodResourcesListerServer(server, &v1alpha1.UnimplementedPodResourcesListerServer{})
			},
		},
		{
			version: "v1",
			getClientFn: func(socket string, timeout time.Duration, maxSize int) (interface{}, *grpc.ClientConn, error) {
				return GetV1Client(socket, timeout, maxSize)
			},
			registerFn: func(server *grpc.Server) {
				v1.RegisterPodResourcesListerServer(server, &v1.UnimplementedPodResourcesListerServer{})
			},
		},
	}
}

func TestGetClient(t *testing.T) {
	testCases := getClientTestCases()

	for _, tc := range testCases {
		t.Run(tc.version, func(t *testing.T) {
			socketPath, cleanup, err := startTestServer(t, tc.registerFn)
			require.NoError(t, err)
			defer cleanup()

			client, conn, err := tc.getClientFn(socketPath, 10*time.Second, 1024*1024)
			require.NoError(t, err)
			defer func() { _ = conn.Close() }()

			require.NotNil(t, client)
		})
	}
}

// TestGetClientError tests client creation with invalid socket paths.
// This unit test validates error handling for connection failures that are
// difficult to reproduce reliably in e2e tests (invalid paths, null bytes).
func TestGetClientError(t *testing.T) {
	testCases := getClientTestCases()

	for _, tc := range testCases {
		t.Run(tc.version, func(t *testing.T) {
			client, conn, err := tc.getClientFn("unix:///invalid\x00path", 100*time.Millisecond, 1024*1024)
			require.Error(t, err)
			require.Nil(t, client)
			require.Nil(t, conn)
		})
	}
}

func startTestServer(t *testing.T, registerFn func(*grpc.Server)) (string, func(), error) {
	socketPath := filepath.Join(t.TempDir(), "podresources.sock")
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		return "", nil, err
	}

	server := grpc.NewServer()
	registerFn(server)

	go func() {
		if err := server.Serve(listener); err != nil {
			t.Logf("server stopped: %v", err)
		}
	}()

	return socketPath, server.Stop, nil
}
