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

// Package envelope transforms values for storage at rest using a Envelope provider
package envelope

import (
	"context"
	"encoding/base64"
	"fmt"
	"net"
	"reflect"
	"testing"

	"google.golang.org/grpc"

	kmsapi "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1"
)

const (
	endpoint = "unix:///@kms-socket.sock"
)

// Normal encryption and decryption operation.
func TestGRPCService(t *testing.T) {
	// Start a test gRPC server.
	server, err := startTestKMSProvider()
	if err != nil {
		t.Fatalf("failed to start test KMS provider server, error: %v", err)
	}
	defer stopTestKMSProvider(server)

	// Create the gRPC client service.
	service, err := NewGRPCService(endpoint)
	if err != nil {
		t.Fatalf("failed to create envelope service, error: %v", err)
	}
	defer destroyService(service)

	// Call service to encrypt data.
	data := []byte("test data")
	cipher, err := service.Encrypt(data)
	if err != nil {
		t.Fatalf("failed when execute encrypt, error: %v", err)
	}

	// Call service to decrypt data.
	result, err := service.Decrypt(cipher)
	if err != nil {
		t.Fatalf("failed when execute decrypt, error: %v", err)
	}

	if !reflect.DeepEqual(data, result) {
		t.Errorf("expect: %v, but: %v", data, result)
	}
}

func destroyService(service Service) {
	s := service.(*gRPCService)
	s.connection.Close()
}

// Test all those invalid configuration for KMS provider.
func TestInvalidConfiguration(t *testing.T) {
	// Start a test gRPC server.
	server, err := startTestKMSProvider()
	if err != nil {
		t.Fatalf("failed to start test KMS provider server, error: %v", err)
	}
	defer stopTestKMSProvider(server)

	invalidConfigs := []struct {
		name       string
		apiVersion string
		endpoint   string
	}{
		{"emptyConfiguration", kmsapiVersion, ""},
		{"invalidScheme", kmsapiVersion, "tcp://localhost:6060"},
		{"unavailableEndpoint", kmsapiVersion, unixProtocol + ":///kms-socket.nonexist"},
		{"invalidAPIVersion", "invalidVersion", endpoint},
	}

	for _, testCase := range invalidConfigs {
		t.Run(testCase.name, func(t *testing.T) {
			setAPIVersion(testCase.apiVersion)
			defer setAPIVersion(kmsapiVersion)

			_, err := NewGRPCService(testCase.endpoint)
			if err == nil {
				t.Fatalf("should fail to create envelope service for %s.", testCase.name)
			}
		})
	}
}

// Start the gRPC server that listens on unix socket.
func startTestKMSProvider() (*grpc.Server, error) {
	sockFile, err := parseEndpoint(endpoint)
	if err != nil {
		return nil, fmt.Errorf("failed to parse endpoint:%q, error %v", endpoint, err)
	}
	listener, err := net.Listen(unixProtocol, sockFile)
	if err != nil {
		return nil, fmt.Errorf("failed to listen on the unix socket, error: %v", err)
	}

	server := grpc.NewServer()
	kmsapi.RegisterKeyManagementServiceServer(server, &base64Server{})
	go server.Serve(listener)
	return server, nil
}

func stopTestKMSProvider(server *grpc.Server) {
	server.Stop()
}

// Fake gRPC sever for remote KMS provider.
// Use base64 to simulate encrypt and decrypt.
type base64Server struct{}

var testProviderAPIVersion = kmsapiVersion

func setAPIVersion(apiVersion string) {
	testProviderAPIVersion = apiVersion
}

func (s *base64Server) Version(ctx context.Context, request *kmsapi.VersionRequest) (*kmsapi.VersionResponse, error) {
	return &kmsapi.VersionResponse{Version: testProviderAPIVersion, RuntimeName: "testKMS", RuntimeVersion: "0.0.1"}, nil
}

func (s *base64Server) Decrypt(ctx context.Context, request *kmsapi.DecryptRequest) (*kmsapi.DecryptResponse, error) {
	buf := make([]byte, base64.StdEncoding.DecodedLen(len(request.Cipher)))
	n, err := base64.StdEncoding.Decode(buf, request.Cipher)
	if err != nil {
		return nil, err
	}

	return &kmsapi.DecryptResponse{Plain: buf[:n]}, nil
}

func (s *base64Server) Encrypt(ctx context.Context, request *kmsapi.EncryptRequest) (*kmsapi.EncryptResponse, error) {
	buf := make([]byte, base64.StdEncoding.EncodedLen(len(request.Plain)))
	base64.StdEncoding.Encode(buf, request.Plain)
	return &kmsapi.EncryptResponse{Cipher: buf}, nil
}
