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
	"net/url"
	"strings"
	"time"

	"github.com/golang/glog"

	"google.golang.org/grpc"

	kmsapi "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1"
)

const (
	// Now only supported unix domain socket.
	unixProtocol = "unix"

	// Current version for the protocol interface definition.
	kmsapiVersion = "v1beta1"

	// The timeout that communicate with KMS server.
	timeout = 30 * time.Second
)

// The gRPC implementation for envelope.Service.
type gRPCService struct {
	// gRPC client instance
	kmsClient  kmsapi.KeyManagementServiceClient
	connection *grpc.ClientConn
}

// NewGRPCService returns an envelope.Service which use gRPC to communicate the remote KMS provider.
func NewGRPCService(endpoint string) (Service, error) {
	glog.V(4).Infof("Configure KMS provider with endpoint: %s", endpoint)

	addr, err := parseEndpoint(endpoint)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	connection, err := grpc.DialContext(ctx, addr, grpc.WithInsecure(), grpc.WithDialer(unixDial))
	if err != nil {
		return nil, fmt.Errorf("connect remote KMS provider %q failed, error: %v", addr, err)
	}

	kmsClient := kmsapi.NewKeyManagementServiceClient(connection)

	err = checkAPIVersion(kmsClient)
	if err != nil {
		connection.Close()
		return nil, fmt.Errorf("failed check version for %q, error: %v", addr, err)
	}

	return &gRPCService{kmsClient: kmsClient, connection: connection}, nil
}

// This dialer explicitly ask gRPC to use unix socket as network.
func unixDial(addr string, timeout time.Duration) (net.Conn, error) {
	return net.DialTimeout(unixProtocol, addr, timeout)
}

// Parse the endpoint to extract schema, host or path.
func parseEndpoint(endpoint string) (string, error) {
	if len(endpoint) == 0 {
		return "", fmt.Errorf("remote KMS provider can't use empty string as endpoint")
	}

	u, err := url.Parse(endpoint)
	if err != nil {
		return "", fmt.Errorf("invalid endpoint %q for remote KMS provider, error: %v", endpoint, err)
	}

	if u.Scheme != unixProtocol {
		return "", fmt.Errorf("unsupported scheme %q for remote KMS provider", u.Scheme)
	}

	// Linux abstract namespace socket - no physical file required
	// Warning: Linux Abstract sockets have not concept of ACL (unlike traditional file based sockets).
	// However, Linux Abstract sockets are subject to Linux networking namespace, so will only be accessible to
	// containers within the same pod (unless host networking is used).
	if strings.HasPrefix(u.Path, "/@") {
		return strings.TrimPrefix(u.Path, "/"), nil
	}

	return u.Path, nil
}

// Check the KMS provider API version.
// Only matching kmsapiVersion is supported now.
func checkAPIVersion(kmsClient kmsapi.KeyManagementServiceClient) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	request := &kmsapi.VersionRequest{Version: kmsapiVersion}
	response, err := kmsClient.Version(ctx, request)
	if err != nil {
		return fmt.Errorf("failed get version from remote KMS provider: %v", err)
	}
	if response.Version != kmsapiVersion {
		return fmt.Errorf("KMS provider api version %s is not supported, only %s is supported now",
			response.Version, kmsapiVersion)
	}

	glog.V(4).Infof("KMS provider %s initialized, version: %s", response.RuntimeName, response.RuntimeVersion)
	return nil
}

// Decrypt a given data string to obtain the original byte data.
func (g *gRPCService) Decrypt(cipher []byte) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
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
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	request := &kmsapi.EncryptRequest{Plain: plain, Version: kmsapiVersion}
	response, err := g.kmsClient.Encrypt(ctx, request)
	if err != nil {
		return nil, err
	}
	return response.Cipher, nil
}
