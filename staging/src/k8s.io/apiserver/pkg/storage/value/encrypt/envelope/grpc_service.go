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
	"fmt"
	"net"
	"net/url"
	"time"

	"github.com/golang/glog"

	"google.golang.org/grpc"

	"golang.org/x/net/context"

	kmsapi "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1"
)

const (
	// Now only supportied unix domain socket.
	unixProtocol = "unix"

	// Current version for the protocal interface definition.
	kmsapiVersion = "v1beta1"
)

// The gRPC implementation for envelope.Service.
type gRPCService struct {
	// gRPC client instance
	kmsClient  kmsapi.KMSServiceClient
	connection *grpc.ClientConn
}

// NewGRPCService returns an envelope.Service which use gRPC to communicate the remote KMS provider.
func NewGRPCService(endpoint string) (Service, error) {
	glog.V(4).Infof("Configure KMS provider with endpoint: %s", endpoint)

	protocol, addr, err := parseEndpoint(endpoint)
	if err != nil {
		return nil, err
	}

	dialer := func(addr string, timeout time.Duration) (net.Conn, error) {
		return net.DialTimeout(protocol, addr, timeout)
	}

	connection, err := grpc.Dial(addr, grpc.WithInsecure(), grpc.WithDialer(dialer))
	if err != nil {
		return nil, fmt.Errorf("connect remote KMS provider %q failed, error: %v", addr, err)
	}

	kmsClient := kmsapi.NewKMSServiceClient(connection)

	err = checkAPIVersion(kmsClient)
	if err != nil {
		connection.Close()
		return nil, fmt.Errorf("failed check version for %q, error: %v", addr, err)
	}

	return &gRPCService{kmsClient: kmsClient, connection: connection}, nil
}

// Parse the endpoint to extract schema, host or path.
func parseEndpoint(endpoint string) (string, string, error) {
	if len(endpoint) == 0 {
		return "", "", fmt.Errorf("remote KMS provider can't use empty string as endpoint")
	}

	u, err := url.Parse(endpoint)
	if err != nil {
		return "", "", fmt.Errorf("invalid endpoint %q for remote KMS provider, error: %v", endpoint, err)
	}

	if u.Scheme != unixProtocol {
		return "", "", fmt.Errorf("unsupported scheme %q for remote KMS provider", u.Scheme)
	}
	return unixProtocol, u.Path, nil
}

// Check the KMS provider API version.
// Only matching kubeRuntimeAPIVersion is supported now.
func checkAPIVersion(kmsClient kmsapi.KMSServiceClient) error {
	request := &kmsapi.VersionRequest{Version: kmsapiVersion}
	response, err := kmsClient.Version(context.Background(), request)
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
	request := &kmsapi.DecryptRequest{Cipher: cipher, Version: kmsapiVersion}
	response, err := g.kmsClient.Decrypt(context.Background(), request)
	if err != nil {
		return nil, err
	}
	return response.Plain, nil
}

// Encrypt bytes to a string ciphertext.
func (g *gRPCService) Encrypt(plain []byte) ([]byte, error) {
	request := &kmsapi.EncryptRequest{Plain: plain, Version: kmsapiVersion}
	response, err := g.kmsClient.Encrypt(context.Background(), request)
	if err != nil {
		return nil, err
	}
	return response.Cipher, nil
}
