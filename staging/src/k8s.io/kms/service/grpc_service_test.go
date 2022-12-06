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

package service

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	kmsapi "k8s.io/kms/apis/v2alpha1"
	"k8s.io/kms/encryption"
)

func TestGRPCService(t *testing.T) {
	t.Parallel()

	id, err := uuid.NewRandom()
	if err != nil {
		t.Fatal(err)
	}
	plaintext := []byte("lorem ipsum dolor sit amet")
	ctx := context.Background()
	address := filepath.Join(os.TempDir(), fmt.Sprintf("kmsv2-%s.sock", id))
	kmsUpstream, err := encryption.NewInMemory()
	if err != nil {
		t.Fatal(err)
	}

	server, err := NewGRPCService(address, time.Second, kmsUpstream)
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		if err := server.ListenAndServe(); err != nil {
			t.Error(err)
		}
	}()
	defer server.Shutdown()

	client, close, err := newClient(address)
	if err != nil {
		t.Fatal(err)
	}
	defer close()

	t.Run("should be able to encrypt and decrypt through unix domain sockets", func(t *testing.T) {
		encRes, err := client.Encrypt(ctx, &kmsapi.EncryptRequest{
			Plaintext: plaintext,
			Uid:       id.String(),
		})
		if err != nil {
			t.Fatal(err)
		}

		decRes, err := client.Decrypt(ctx, &kmsapi.DecryptRequest{
			Ciphertext:  encRes.Ciphertext,
			KeyId:       encRes.KeyId,
			Annotations: encRes.Annotations,
			Uid:         id.String(),
		})
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(decRes.Plaintext, plaintext) {
			t.Errorf("want: %q, have: %q", plaintext, decRes.Plaintext)
		}
	})

	t.Run("should return status data", func(t *testing.T) {
		status, err := client.Status(ctx, &kmsapi.StatusRequest{})
		if err != nil {
			t.Fatal(err)
		}

		if status.Healthz != "ok" {
			t.Errorf("want: %q, have: %q", "ok", status.Healthz)
		}
		if _, err := uuid.Parse(status.KeyId); err != nil {
			t.Error(err)
		}
		if status.Version != "v2alpha1" {
			t.Errorf("want %q, have: %q", "v2alpha1", status.Version)
		}
	})
}

func newClient(address string) (kmsapi.KeyManagementServiceClient, func() error, error) {
	cnn, err := grpc.Dial(
		address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDialer(func(addr string, t time.Duration) (net.Conn, error) {
			return net.Dial("unix", addr)
		}),
	)
	if err != nil {
		return nil, func() error { return nil }, err
	}

	return kmsapi.NewKeyManagementServiceClient(cnn), cnn.Close, nil
}
