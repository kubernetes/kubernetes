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

package master

import (
	"context"
	"encoding/base64"
	"fmt"
	"net"
	"sync"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	kmsapi "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1"
	"k8s.io/klog"
)

const (
	kmsAPIVersion = "v1beta1"
	unixProtocol  = "unix"
)

// base64Plugin gRPC sever for a mock KMS provider.
// Uses base64 to simulate encrypt and decrypt.
type base64Plugin struct {
	grpcServer         *grpc.Server
	listener           net.Listener
	mu                 *sync.Mutex
	lastEncryptRequest *kmsapi.EncryptRequest
	inFailedState      bool
}

func newBase64Plugin(socketPath string) (*base64Plugin, error) {
	listener, err := net.Listen(unixProtocol, socketPath)
	if err != nil {
		return nil, fmt.Errorf("failed to listen on the unix socket, error: %v", err)
	}
	klog.Infof("Listening on %s", socketPath)

	server := grpc.NewServer()

	result := &base64Plugin{
		grpcServer: server,
		listener:   listener,
		mu:         &sync.Mutex{},
	}

	kmsapi.RegisterKeyManagementServiceServer(server, result)

	return result, nil
}

func (s *base64Plugin) cleanUp() {
	s.grpcServer.Stop()
	s.listener.Close()
}

var testProviderAPIVersion = kmsAPIVersion

func (s *base64Plugin) enterFailedState() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inFailedState = true
}

func (s *base64Plugin) exitFailedState() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inFailedState = false
}

func (s *base64Plugin) Version(ctx context.Context, request *kmsapi.VersionRequest) (*kmsapi.VersionResponse, error) {
	return &kmsapi.VersionResponse{Version: testProviderAPIVersion, RuntimeName: "testKMS", RuntimeVersion: "0.0.1"}, nil
}

func (s *base64Plugin) Decrypt(ctx context.Context, request *kmsapi.DecryptRequest) (*kmsapi.DecryptResponse, error) {
	klog.Infof("Received Decrypt Request for DEK: %s", string(request.Cipher))

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

func (s *base64Plugin) Encrypt(ctx context.Context, request *kmsapi.EncryptRequest) (*kmsapi.EncryptResponse, error) {
	klog.Infof("Received Encrypt Request for DEK: %x", request.Plain)
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
