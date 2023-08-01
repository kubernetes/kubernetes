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

package internal

import (
	"bytes"
	"context"
	"testing"

	"k8s.io/kms/pkg/service"
)

const (
	version       = "v2beta1"
	testAESKey    = "abcdefghijklmnop"
	testKeyID     = "test-key-id"
	testPlaintext = "lorem ipsum dolor sit amet"
)

func testContext(t *testing.T) context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	return ctx
}

func TestMockAESRemoteService(t *testing.T) {
	t.Parallel()
	ctx := testContext(t)

	plaintext := []byte(testPlaintext)

	kmsService, err := NewMockAESService(testAESKey, testKeyID)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("should be able to encrypt and decrypt", func(t *testing.T) {
		t.Parallel()

		encRes, err := kmsService.Encrypt(ctx, "", plaintext)
		if err != nil {
			t.Fatal(err)
		}

		if bytes.Equal(plaintext, encRes.Ciphertext) {
			t.Fatal("plaintext and ciphertext shouldn't be equal!")
		}

		decRes, err := kmsService.Decrypt(ctx, "", &service.DecryptRequest{
			Ciphertext:  encRes.Ciphertext,
			KeyID:       encRes.KeyID,
			Annotations: encRes.Annotations,
		})
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(decRes, plaintext) {
			t.Errorf("want: %q, have: %q", plaintext, decRes)
		}
	})

	t.Run("should return error when decrypt with an invalid keyID", func(t *testing.T) {
		t.Parallel()

		encRes, err := kmsService.Encrypt(ctx, "", plaintext)
		if err != nil {
			t.Fatal(err)
		}

		if bytes.Equal(plaintext, encRes.Ciphertext) {
			t.Fatal("plaintext and ciphertext shouldn't be equal!")
		}

		_, err = kmsService.Decrypt(ctx, "", &service.DecryptRequest{
			Ciphertext:  encRes.Ciphertext,
			KeyID:       encRes.KeyID + "1",
			Annotations: encRes.Annotations,
		})
		if err.Error() != "invalid keyID" {
			t.Errorf("should have returned an invalid keyID error. Got %v, requested keyID: %q, remote service keyID: %q", err, encRes.KeyID+"1", testKeyID)
		}
	})

	t.Run("should return status data", func(t *testing.T) {
		t.Parallel()

		status, err := kmsService.Status(ctx)
		if err != nil {
			t.Fatal(err)
		}

		if status.Healthz != "ok" {
			t.Errorf("want: %q, have: %q", "ok", status.Healthz)
		}
		if len(status.KeyID) == 0 {
			t.Errorf("want: len(keyID) > 0, have: %d", len(status.KeyID))
		}
		if status.Version != version {
			t.Errorf("want %q, have: %q", version, status.Version)
		}
	})
}
