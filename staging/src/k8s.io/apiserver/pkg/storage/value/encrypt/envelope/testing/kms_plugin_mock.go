// +build !windows

/*
Copyright 2017 The Kubernetes Authors.

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

package testing

import (
	"context"
	"encoding/base64"
	"fmt"
	"net"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"k8s.io/apimachinery/pkg/util/wait"
	kmsapi "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1"
	"k8s.io/klog"
)

const (
	// Now only supported unix domain socket.
	unixProtocol = "unix"

	// Current version for the protocol interface definition.
	kmsapiVersion = "v1beta1"
)

// Base64Plugin gRPC sever for a mock KMS provider.
// Uses base64 to simulate encrypt and decrypt.
type Base64Plugin struct {
	grpcServer         *grpc.Server
	listener           net.Listener
	mu                 *sync.Mutex
	lastEncryptRequest *kmsapi.EncryptRequest
	inFailedState      bool
	ver                string
	socketPath         string
}

// NewBase64Plugin is a constructor for Base64Plugin.
func NewBase64Plugin(socketPath string) (*Base64Plugin, error) {
	server := grpc.NewServer()
	result := &Base64Plugin{
		grpcServer: server,
		mu:         &sync.Mutex{},
		ver:        kmsapiVersion,
		socketPath: socketPath,
	}

	kmsapi.RegisterKeyManagementServiceServer(server, result)
	return result, nil
}

// WaitForBase64PluginToBeUp waits until the plugin is ready to serve requests.
func WaitForBase64PluginToBeUp(plugin *Base64Plugin) error {
	var gRPCErr error
	pollErr := wait.PollImmediate(1*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		_, gRPCErr = plugin.Encrypt(context.Background(), &kmsapi.EncryptRequest{Plain: []byte("foo")})
		return gRPCErr == nil, nil
	})

	if pollErr == wait.ErrWaitTimeout {
		return fmt.Errorf("failed to start kms-plugin, error: %v", gRPCErr)
	}

	return nil
}

// LastEncryptRequest returns the last EncryptRequest.Plain sent to the plugin.
func (s *Base64Plugin) LastEncryptRequest() []byte {
	return s.lastEncryptRequest.Plain
}

// SetVersion sets the version of kms-plugin.
func (s *Base64Plugin) SetVersion(ver string) {
	s.ver = ver
}

// Start starts plugin's gRPC service.
func (s *Base64Plugin) Start() error {
	var err error
	s.listener, err = net.Listen(unixProtocol, s.socketPath)
	if err != nil {
		return fmt.Errorf("failed to listen on the unix socket, error: %v", err)
	}
	klog.Infof("Listening on %s", s.socketPath)

	go s.grpcServer.Serve(s.listener)
	return nil
}

// CleanUp stops gRPC server and the underlying listener.
func (s *Base64Plugin) CleanUp() {
	s.grpcServer.Stop()
	s.listener.Close()
	if !strings.HasPrefix(s.socketPath, "@") || runtime.GOOS != "linux" {
		os.Remove(s.socketPath)
	}
}

// EnterFailedState places the plugin into failed state.
func (s *Base64Plugin) EnterFailedState() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inFailedState = true
}

// ExitFailedState removes the plugin from the failed state.
func (s *Base64Plugin) ExitFailedState() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inFailedState = false
}

// Version returns the version of the kms-plugin.
func (s *Base64Plugin) Version(ctx context.Context, request *kmsapi.VersionRequest) (*kmsapi.VersionResponse, error) {
	klog.Infof("Received request for Version: %v", request)
	return &kmsapi.VersionResponse{Version: s.ver, RuntimeName: "testKMS", RuntimeVersion: "0.0.1"}, nil
}

// Decrypt performs base64 decoding of the payload of kms.DecryptRequest.
func (s *Base64Plugin) Decrypt(ctx context.Context, request *kmsapi.DecryptRequest) (*kmsapi.DecryptResponse, error) {
	klog.V(3).Infof("Received Decrypt Request for DEK: %s", string(request.Cipher))

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.inFailedState {
		return nil, status.Error(codes.FailedPrecondition, "failed precondition - key disabled")
	}

	buf := make([]byte, base64.StdEncoding.DecodedLen(len(request.Cipher)))
	n, err := base64.StdEncoding.Decode(buf, request.Cipher)
	if err != nil {
		return nil, err
	}

	return &kmsapi.DecryptResponse{Plain: buf[:n]}, nil
}

// Encrypt performs base64 encoding of the payload of kms.EncryptRequest.
func (s *Base64Plugin) Encrypt(ctx context.Context, request *kmsapi.EncryptRequest) (*kmsapi.EncryptResponse, error) {
	klog.V(3).Infof("Received Encrypt Request for DEK: %x", request.Plain)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastEncryptRequest = request

	if s.inFailedState {
		return nil, status.Error(codes.FailedPrecondition, "failed precondition - key disabled")
	}

	buf := make([]byte, base64.StdEncoding.EncodedLen(len(request.Plain)))
	base64.StdEncoding.Encode(buf, request.Plain)

	return &kmsapi.EncryptResponse{Cipher: buf}, nil
}
