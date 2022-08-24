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

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/util"
	kmsapi "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1"
	"k8s.io/klog/v2"
)

const (
	// unixProtocol is the only supported protocol for remote KMS provider.
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
func NewGRPCService(ctx context.Context, endpoint string, callTimeout time.Duration) (Service, error) {
	klog.V(4).Infof("Configure KMS provider with endpoint: %s", endpoint)

	addr, err := util.ParseEndpoint(endpoint)
	if err != nil {
		return nil, err
	}

	s := &gRPCService{callTimeout: callTimeout}
	s.connection, err = grpc.Dial(
		addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithUnaryInterceptor(s.interceptor),
		grpc.WithDefaultCallOptions(grpc.WaitForReady(true)),
		grpc.WithContextDialer(
			func(context.Context, string) (net.Conn, error) {
				// Ignoring addr and timeout arguments:
				// addr - comes from the closure
				c, err := net.DialUnix(unixProtocol, nil, &net.UnixAddr{Name: addr})
				if err != nil {
					klog.Errorf("failed to create connection to unix socket: %s, error: %v", addr, err)
				} else {
					klog.V(4).Infof("Successfully dialed Unix socket %v", addr)
				}
				return c, err
			}))

	if err != nil {
		return nil, fmt.Errorf("failed to create connection to %s, error: %v", endpoint, err)
	}

	s.kmsClient = kmsapi.NewKeyManagementServiceClient(s.connection)

	go func() {
		defer utilruntime.HandleCrash()

		<-ctx.Done()
		_ = s.connection.Close()
	}()

	return s, nil
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

	request := &kmsapi.EncryptRequest{Plain: plain, Version: kmsapiVersion}
	response, err := g.kmsClient.Encrypt(ctx, request)
	if err != nil {
		return nil, err
	}
	return response.Cipher, nil
}

func (g *gRPCService) interceptor(
	ctx context.Context,
	method string,
	req interface{},
	reply interface{},
	cc *grpc.ClientConn,
	invoker grpc.UnaryInvoker,
	opts ...grpc.CallOption,
) error {
	if !kmsapi.IsVersionCheckMethod(method) {
		if err := g.checkAPIVersion(ctx); err != nil {
			return err
		}
	}

	return invoker(ctx, method, req, reply, cc, opts...)
}
