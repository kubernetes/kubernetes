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
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"

	kmsapi "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1"
)

const (
	endpoint = "unix:///@kms-socket.sock"
)

// TestKMSPluginLateStart tests the scenario where kms-plugin pod/container starts after kube-apiserver pod/container.
// Since the Dial to kms-plugin is non-blocking we expect the construction of gRPC service to succeed even when
// kms-plugin is not yet up - dialing happens in the background.
func TestKMSPluginLateStart(t *testing.T) {
	callTimeout := 3 * time.Second

	service, err := NewGRPCService(endpoint, callTimeout)
	if err != nil {
		t.Fatalf("failed to create envelope service, error: %v", err)
	}
	defer destroyService(service)

	time.Sleep(callTimeout / 2)
	f, err := startFakeKMSProvider(kmsapiVersion)
	if err != nil {
		t.Fatalf("failed to start test KMS provider server, error: %v", err)
	}
	defer f.server.Stop()

	data := []byte("test data")
	_, err = service.Encrypt(data)
	if err != nil {
		t.Fatalf("failed when execute encrypt, error: %v", err)
	}
}

// TestIntermittentConnectionLoss tests the scenario where the connection with kms-plugin is intermittently lost.
func TestIntermittentConnectionLoss(t *testing.T) {
	var (
		wg1      sync.WaitGroup
		wg2      sync.WaitGroup
		timeout  = 30 * time.Second
		blackOut = 1 * time.Second
		data     = []byte("test data")
	)
	// Start KMS Plugin
	f, err := startFakeKMSProvider(kmsapiVersion)
	if err != nil {
		t.Fatalf("failed to start test KMS provider server, error: %v", err)
	}

	//  connect to kms plugin
	service, err := NewGRPCService(endpoint, timeout)
	if err != nil {
		t.Fatalf("failed to create envelope service, error: %v", err)
	}
	defer destroyService(service)

	_, err = service.Encrypt(data)
	if err != nil {
		t.Fatalf("failed when execute encrypt, error: %v", err)
	}
	t.Log("Connected to KMSPlugin")

	// Stop KMS Plugin - simulating connection loss
	f.server.Stop()
	t.Log("KMS Plugin is stopped")

	wg1.Add(1)
	wg2.Add(1)
	go func() {
		defer wg2.Done()
		// Call service to encrypt data.
		t.Log("Sending encrypt request")
		wg1.Done()
		_, err := service.Encrypt(data)
		if err != nil {
			t.Fatalf("failed when executing encrypt, error: %v", err)
		}
	}()

	wg1.Wait()
	time.Sleep(blackOut)
	// Start KMS Plugin
	f, err = startFakeKMSProvider(kmsapiVersion)
	if err != nil {
		t.Fatalf("failed to start test KMS provider server, error: %v", err)
	}
	defer f.server.Stop()
	t.Log("Restarted KMS Plugin")

	wg2.Wait()
}

func TestUnsupportedVersion(t *testing.T) {
	ver := "invalid"
	data := []byte("test data")
	wantErr := fmt.Errorf(versionErrorf, ver, kmsapiVersion)

	f, err := startFakeKMSProvider(ver)
	if err != nil {
		t.Fatalf("failed to start test KMS provider server, error: %ver", err)
	}
	defer f.server.Stop()

	s, err := NewGRPCService(endpoint, 1*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer destroyService(s)

	// Encrypt
	_, err = s.Encrypt(data)
	if err == nil || err.Error() != wantErr.Error() {
		t.Errorf("got err: %ver, want: %ver", err, wantErr)
	}

	destroyService(s)

	s, err = NewGRPCService(endpoint, 1*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer destroyService(s)

	// Decrypt
	_, err = s.Decrypt(data)
	if err == nil || err.Error() != wantErr.Error() {
		t.Errorf("got err: %ver, want: %ver", err, wantErr)
	}
}

func TestConcurrentAccess(t *testing.T) {

}

// Normal encryption and decryption operation.
func TestGRPCService(t *testing.T) {
	// Start a test gRPC server.
	f, err := startFakeKMSProvider(kmsapiVersion)
	if err != nil {
		t.Fatalf("failed to start test KMS provider server, error: %v", err)
	}
	defer f.server.Stop()

	// Create the gRPC client service.
	service, err := NewGRPCService(endpoint, 1*time.Second)
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

// Normal encryption and decryption operation by multiple go-routines.
func TestGRPCServiceConcurrentAccess(t *testing.T) {
	// Start a test gRPC server.
	f, err := startFakeKMSProvider(kmsapiVersion)
	if err != nil {
		t.Fatalf("failed to start test KMS provider server, error: %v", err)
	}
	defer f.server.Stop()

	// Create the gRPC client service.
	service, err := NewGRPCService(endpoint, 1*time.Second)
	if err != nil {
		t.Fatalf("failed to create envelope service, error: %v", err)
	}
	defer destroyService(service)

	var wg sync.WaitGroup
	n := 1000
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			// Call service to encrypt data.
			data := []byte("test data")
			cipher, err := service.Encrypt(data)
			if err != nil {
				t.Errorf("failed when execute encrypt, error: %v", err)
			}

			// Call service to decrypt data.
			result, err := service.Decrypt(cipher)
			if err != nil {
				t.Errorf("failed when execute decrypt, error: %v", err)
			}

			if !reflect.DeepEqual(data, result) {
				t.Errorf("expect: %v, but: %v", data, result)
			}
		}()
	}

	wg.Wait()
}

func destroyService(service Service) {
	if service != nil {
		s := service.(*gRPCService)
		s.connection.Close()
	}
}

// Test all those invalid configuration for KMS provider.
func TestInvalidConfiguration(t *testing.T) {
	// Start a test gRPC server.
	f, err := startFakeKMSProvider(kmsapiVersion)
	if err != nil {
		t.Fatalf("failed to start test KMS provider server, error: %v", err)
	}
	defer f.server.Stop()

	invalidConfigs := []struct {
		name       string
		apiVersion string
		endpoint   string
	}{
		{"emptyConfiguration", kmsapiVersion, ""},
		{"invalidScheme", kmsapiVersion, "tcp://localhost:6060"},
	}

	for _, testCase := range invalidConfigs {
		t.Run(testCase.name, func(t *testing.T) {
			f.apiVersion = testCase.apiVersion
			_, err := NewGRPCService(testCase.endpoint, 1*time.Second)
			if err == nil {
				t.Fatalf("should fail to create envelope service for %s.", testCase.name)
			}
		})
	}
}

// Start the gRPC server that listens on unix socket.
func startFakeKMSProvider(version string) (*fakeKMSPlugin, error) {
	sockFile, err := parseEndpoint(endpoint)
	if err != nil {
		return nil, fmt.Errorf("failed to parse endpoint:%q, error %v", endpoint, err)
	}
	listener, err := net.Listen(unixProtocol, sockFile)
	if err != nil {
		return nil, fmt.Errorf("failed to listen on the unix socket, error: %v", err)
	}

	s := grpc.NewServer()
	f := &fakeKMSPlugin{apiVersion: version, server: s}
	kmsapi.RegisterKeyManagementServiceServer(s, f)
	go s.Serve(listener)
	return f, nil
}

// Fake gRPC sever for remote KMS provider.
// Use base64 to simulate encrypt and decrypt.
type fakeKMSPlugin struct {
	apiVersion string
	server     *grpc.Server
}

func (s *fakeKMSPlugin) Version(ctx context.Context, request *kmsapi.VersionRequest) (*kmsapi.VersionResponse, error) {
	return &kmsapi.VersionResponse{Version: s.apiVersion, RuntimeName: "testKMS", RuntimeVersion: "0.0.1"}, nil
}

func (s *fakeKMSPlugin) Decrypt(ctx context.Context, request *kmsapi.DecryptRequest) (*kmsapi.DecryptResponse, error) {
	buf := make([]byte, base64.StdEncoding.DecodedLen(len(request.Cipher)))
	n, err := base64.StdEncoding.Decode(buf, request.Cipher)
	if err != nil {
		return nil, err
	}

	return &kmsapi.DecryptResponse{Plain: buf[:n]}, nil
}

func (s *fakeKMSPlugin) Encrypt(ctx context.Context, request *kmsapi.EncryptRequest) (*kmsapi.EncryptResponse, error) {
	buf := make([]byte, base64.StdEncoding.EncodedLen(len(request.Plain)))
	base64.StdEncoding.Encode(buf, request.Plain)
	return &kmsapi.EncryptResponse{Cipher: buf}, nil
}
