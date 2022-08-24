/*
Copyright 2022 The Kubernetes Authors.

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

// Package kmsv2 transforms values for storage at rest using a Envelope provider
package kmsv2

import (
	"context"
	"fmt"
	"net"
	"time"

	"google.golang.org/grpc"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/util"
	kmsapi "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v2alpha1"
	"k8s.io/klog/v2"
)

const (
	// unixProtocol is the only supported protocol for remote KMS provider.
	unixProtocol = "unix"
)

// The gRPC implementation for envelope.Service.
type gRPCService struct {
	kmsClient   kmsapi.KeyManagementServiceClient
	connection  *grpc.ClientConn
	callTimeout time.Duration
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
		grpc.WithInsecure(),
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

// Decrypt a given data string to obtain the original byte data.
func (g *gRPCService) Decrypt(ctx context.Context, uid string, req *DecryptRequest) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, g.callTimeout)
	defer cancel()

	request := &kmsapi.DecryptRequest{
		Ciphertext:  req.Ciphertext,
		Uid:         uid,
		KeyId:       req.KeyID,
		Annotations: req.Annotations,
	}
	response, err := g.kmsClient.Decrypt(ctx, request)
	if err != nil {
		return nil, err
	}
	return response.Plaintext, nil
}

// Encrypt bytes to a string ciphertext.
func (g *gRPCService) Encrypt(ctx context.Context, uid string, plaintext []byte) (*EncryptResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, g.callTimeout)
	defer cancel()

	request := &kmsapi.EncryptRequest{
		Plaintext: plaintext,
		Uid:       uid,
	}
	response, err := g.kmsClient.Encrypt(ctx, request)
	if err != nil {
		return nil, err
	}
	return &EncryptResponse{
		Ciphertext:  response.Ciphertext,
		KeyID:       response.KeyId,
		Annotations: response.Annotations,
	}, nil
}

// Status returns the status of the KMSv2 provider.
func (g *gRPCService) Status(ctx context.Context) (*StatusResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, g.callTimeout)
	defer cancel()

	request := &kmsapi.StatusRequest{}
	response, err := g.kmsClient.Status(ctx, request)
	if err != nil {
		return nil, err
	}
	return &StatusResponse{Version: response.Version, Healthz: response.Healthz, KeyID: response.KeyId}, nil
}
