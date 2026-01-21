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
	"context"
	"net"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	v1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	"k8s.io/kubelet/pkg/apis/podresources/v1alpha1"
)

func TestGetClient(t *testing.T) {
	tests := []struct {
		name      string
		setupFunc func(*grpc.Server)
		getClient func(string, time.Duration, int) (interface{}, *grpc.ClientConn, error)
	}{
		{
			name: "v1alpha1",
			setupFunc: func(server *grpc.Server) {
				v1alpha1.RegisterPodResourcesListerServer(server, &mockV1alpha1Server{})
			},
			getClient: func(socket string, timeout time.Duration, maxSize int) (interface{}, *grpc.ClientConn, error) {
				return GetV1alpha1Client(socket, timeout, maxSize)
			},
		},
		{
			name: "v1",
			setupFunc: func(server *grpc.Server) {
				v1.RegisterPodResourcesListerServer(server, &mockV1Server{})
			},
			getClient: func(socket string, timeout time.Duration, maxSize int) (interface{}, *grpc.ClientConn, error) {
				return GetV1Client(socket, timeout, maxSize)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			socketPath := filepath.Join(t.TempDir(), "podresources.sock")
			listener, err := net.Listen("unix", socketPath)
			require.NoError(t, err)
			defer listener.Close()

			cleanup := startTestServer(t, listener, tt.setupFunc)
			defer cleanup()

			client, conn, err := tt.getClient(socketPath, 10*time.Second, 1024*1024)
			require.NoError(t, err)
			defer conn.Close()

			require.NotNil(t, client)
		})
	}
}

func TestGetV1alpha1ClientError(t *testing.T) {
	_, _, err := GetV1alpha1Client("unix:///invalid\x00path", 100*time.Millisecond, 1024*1024)
	if err == nil {
		t.Fatal("expected error for invalid socket path")
	}
}

func TestGetV1ClientError(t *testing.T) {
	_, _, err := GetV1Client("unix:///invalid\x00path", 100*time.Millisecond, 1024*1024)
	if err == nil {
		t.Fatal("expected error for invalid socket path")
	}
}

type mockV1alpha1Server struct {
	v1alpha1.UnimplementedPodResourcesListerServer
}

func (m *mockV1alpha1Server) List(context.Context, *v1alpha1.ListPodResourcesRequest) (*v1alpha1.ListPodResourcesResponse, error) {
	return &v1alpha1.ListPodResourcesResponse{}, nil
}

type mockV1Server struct {
	v1.UnimplementedPodResourcesListerServer
}

func (m *mockV1Server) List(context.Context, *v1.ListPodResourcesRequest) (*v1.ListPodResourcesResponse, error) {
	return &v1.ListPodResourcesResponse{}, nil
}

func (m *mockV1Server) GetAllocatableResources(context.Context, *v1.AllocatableResourcesRequest) (*v1.AllocatableResourcesResponse, error) {
	return &v1.AllocatableResourcesResponse{}, nil
}

func (m *mockV1Server) Get(context.Context, *v1.GetPodResourcesRequest) (*v1.GetPodResourcesResponse, error) {
	return &v1.GetPodResourcesResponse{}, nil
}

func startTestServer(t *testing.T, listener net.Listener, registerFunc func(server *grpc.Server)) func() {
	server := grpc.NewServer()
	registerFunc(server)
	go func() {
		if err := server.Serve(listener); err != nil {
			t.Logf("server stopped: %v", err)
		}
	}()

	time.Sleep(100 * time.Millisecond)

	return func() {
		server.Stop()
	}
}
