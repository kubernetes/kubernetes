/*
Copyright 2020 The Kubernetes Authors.

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
	"path/filepath"
	"reflect"
	"testing"
	"time"

	apiserverconfig "k8s.io/apiserver/pkg/apis/config"
	mock "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing"
)

// Normal encryption and decryption operation use TLS gRPC server.
func TestTLSGRPCService(t *testing.T) {
	t.Parallel()
	f, err := mock.NewTLSBase64Plugin(filepath.Join("testdata", "ca.crt"), filepath.Join("testdata", "server.crt"),
		filepath.Join("testdata", "server.key"))
	if err != nil {
		t.Fatalf("failed to construct test KMS provider server, error: %v", err)
	}
	addr, err := f.StartTLS()
	if err != nil {
		t.Fatalf("Failed to start kms-plugin, err: %v", err)
	}
	defer f.CleanUp()

	// Create the gRPC client service.
	service, err := NewGRPCService(addr, 1*time.Second, &apiserverconfig.KMSTLSClientConfig{
		CertFile: filepath.Join("testdata", "client.crt"),
		KeyFile:  filepath.Join("testdata", "client.key"),
		CAFile:   filepath.Join("testdata", "ca.crt"),
	})
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
