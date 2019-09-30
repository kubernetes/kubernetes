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

// Package envelope transforms values for storage at rest using a Envelope provider
package envelope

import (
	"context"
	"fmt"
	"net"
	"sync"
	"time"

	"k8s.io/klog"

	"google.golang.org/grpc"

	kmsapi "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1"
)

const (
	// Now only supported unix domain socket.
	unixProtocol = "unix"

	// Current version for the protocol interface definition.
	kmsapiVersion = "v1beta1"

	versionErrorf = "KMS provider api version %s is not supported, only %s is supported now"
)

// The gRPC implementation for envelope.Service.
type gRPCService struct {
	kmsClient      kmsapi.KeyManagementServiceClient
	connection     *grpc.ClientConn
	callTimeout    time.Duration
	mux            sync.RWMutex
	versionChecked bool
}

// NewGRPCService returns an envelope.Service which use gRPC to communicate the remote KMS provider.
func NewGRPCService(addr string, callTimeout time.Duration) (Service, error) {
	klog.V(4).Infof("Configure KMS provider with addr: %s", addr)

	connection, err := grpc.Dial(addr, grpc.WithInsecure(), grpc.WithDefaultCallOptions(grpc.FailFast(false)), grpc.WithDialer(
		func(string, time.Duration) (net.Conn, error) {
			// Ignoring addr and timeout arguments:
			// addr - comes from the closure
			// timeout - is ignored since we are connecting in a non-blocking configuration
			c, err := net.DialTimeout(unixProtocol, addr, 0)
			if err != nil {
				klog.Errorf("failed to create connection to unix socket: %s, error: %v", addr, err)
			}
			return c, err
		}))

	if err != nil {
		return nil, fmt.Errorf("failed to create connection to %s, error: %v", addr, err)
	}

	kmsClient := kmsapi.NewKeyManagementServiceClient(connection)
	return &gRPCService{
		kmsClient:   kmsClient,
		connection:  connection,
		callTimeout: callTimeout,
	}, nil
}

func (g *gRPCService) checkAPIVersion(ctx context.Context) error {
	g.mux.Lock()
	defer g.mux.Unlock()

	if g.versionChecked {
		return nil
	}

	request := &kmsapi.VersionRequest{Version: kmsapiVersion}
	response, err := g.kmsClient.Version(ctx, request)
	if err != nil {
		return fmt.Errorf("failed get version from remote KMS provider: %v", err)
	}
	if response.Version != kmsapiVersion {
		return fmt.Errorf(versionErrorf, response.Version, kmsapiVersion)
	}
	g.versionChecked = true

	klog.V(4).Infof("Version of KMS provider is %s", response.Version)
	return nil
}

// Decrypt a given data string to obtain the original byte data.
func (g *gRPCService) Decrypt(cipher []byte) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), g.callTimeout)
	defer cancel()

	if err := g.checkAPIVersion(ctx); err != nil {
		return nil, err
	}

	request := &kmsapi.DecryptRequest{Cipher: cipher, Version: kmsapiVersion}
	response, err := g.kmsClient.Decrypt(ctx, request)
	if err != nil {
		return nil, err
	}
	return response.Plain, nil
}

// Encrypt bytes to a string ciphertext.
func (g *gRPCService) Encrypt(plain []byte) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), g.callTimeout)
	defer cancel()
	if err := g.checkAPIVersion(ctx); err != nil {
		return nil, err
	}

	request := &kmsapi.EncryptRequest{Plain: plain, Version: kmsapiVersion}
	response, err := g.kmsClient.Encrypt(ctx, request)
	if err != nil {
		return nil, err
	}
	return response.Cipher, nil
}
