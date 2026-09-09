/*
Copyright 2023 The Kubernetes Authors.

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
	"encoding/base64"
	"fmt"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	kmsapi "k8s.io/kms/apis/v2"
)

const version = "v2"

func TestGRPCService(t *testing.T) {
	t.Parallel()

	defaultTimeout := 30 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	t.Cleanup(cancel)

	address := filepath.Join(os.TempDir(), "kmsv2.sock")
	plaintext := []byte("lorem ipsum dolor sit amet")
	r := rand.New(rand.NewSource(time.Now().Unix()))
	id, err := makeID(r.Read)
	if err != nil {
		t.Fatal(err)
	}

	kmsService := newBase64Service(id)
	server := NewGRPCService(address, defaultTimeout, kmsService)
	go func() {
		if err := server.ListenAndServe(); err != nil {
			panic(err)
		}
	}()
	t.Cleanup(server.Shutdown)

	client := newClient(t, address)

	// make sure the gRPC server is up before running tests
ready:
	for {
		select {
		case <-ctx.Done():
			t.Fatalf("server failed to start in time: %v", ctx.Err())

		default:
			if done := func() bool {
				ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
				defer cancel()

				_, err := client.Status(ctx, &kmsapi.StatusRequest{})
				if err != nil {
					t.Logf("failed to get kms status: %v", err)
				}

				return err == nil
			}(); done {
				break ready
			}
			time.Sleep(time.Second)
		}
	}

	t.Run("should be able to encrypt and decrypt through unix domain sockets", func(t *testing.T) {
		t.Parallel()

		encRes, err := client.Encrypt(ctx, &kmsapi.EncryptRequest{
			Plaintext: plaintext,
			Uid:       id,
		})
		if err != nil {
			t.Fatal(err)
		}

		if bytes.Equal(plaintext, encRes.Ciphertext) {
			t.Fatal("plaintext and ciphertext shouldn't be equal!")
		}

		decRes, err := client.Decrypt(ctx, &kmsapi.DecryptRequest{
			Ciphertext:  encRes.Ciphertext,
			KeyId:       encRes.KeyId,
			Annotations: encRes.Annotations,
			Uid:         id,
		})
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(decRes.Plaintext, plaintext) {
			t.Errorf("want: %q, have: %q", plaintext, decRes.Plaintext)
		}
	})

	t.Run("should return status data", func(t *testing.T) {
		t.Parallel()

		status, err := client.Status(ctx, &kmsapi.StatusRequest{})
		if err != nil {
			t.Fatal(err)
		}

		if status.Healthz != "ok" {
			t.Errorf("want: %q, have: %q", "ok", status.Healthz)
		}
		if len(status.KeyId) == 0 {
			t.Errorf("want: len(keyID) > 0, have: %d", len(status.KeyId))
		}
		if status.Version != version {
			t.Errorf("want %q, have: %q", version, status.Version)
		}
	})
}

func newClient(t *testing.T, address string) kmsapi.KeyManagementServiceClient {
	t.Helper()

	cnn, err := grpc.Dial(
		address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDialer(func(addr string, t time.Duration) (net.Conn, error) {
			return net.Dial("unix", addr)
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = cnn.Close() })

	return kmsapi.NewKeyManagementServiceClient(cnn)
}

type testService struct {
	decrypt func(ctx context.Context, uid string, req *DecryptRequest) ([]byte, error)
	encrypt func(ctx context.Context, uid string, data []byte) (*EncryptResponse, error)
	status  func(ctx context.Context) (*StatusResponse, error)
}

var _ Service = (*testService)(nil)

func (s *testService) Decrypt(ctx context.Context, uid string, req *DecryptRequest) ([]byte, error) {
	return s.decrypt(ctx, uid, req)
}

func (s *testService) Encrypt(ctx context.Context, uid string, data []byte) (*EncryptResponse, error) {
	return s.encrypt(ctx, uid, data)
}

func (s *testService) Status(ctx context.Context) (*StatusResponse, error) {
	return s.status(ctx)
}

func makeID(rand func([]byte) (int, error)) (string, error) {
	b := make([]byte, 10)
	if _, err := rand(b); err != nil {
		return "", err
	}

	return base64.StdEncoding.EncodeToString(b), nil
}

func newBase64Service(keyID string) *testService {
	decrypt := func(_ context.Context, _ string, req *DecryptRequest) ([]byte, error) {
		if req.KeyID != keyID {
			return nil, fmt.Errorf("keyID mismatch. want: %q, have: %q", keyID, req.KeyID)
		}

		return base64.StdEncoding.DecodeString(string(req.Ciphertext))
	}

	encrypt := func(_ context.Context, _ string, data []byte) (*EncryptResponse, error) {
		return &EncryptResponse{
			Ciphertext: []byte(base64.StdEncoding.EncodeToString(data)),
			KeyID:      keyID,
		}, nil
	}

	status := func(_ context.Context) (*StatusResponse, error) {
		return &StatusResponse{
			Version: version,
			Healthz: "ok",
			KeyID:   keyID,
		}, nil
	}

	return &testService{
		decrypt: decrypt,
		encrypt: encrypt,
		status:  status,
	}
}
