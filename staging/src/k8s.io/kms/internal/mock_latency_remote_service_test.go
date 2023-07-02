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
	"testing"
	"time"

	"k8s.io/kms/pkg/service"
)

const (
	testLatencyInMillisecond = 100 * time.Millisecond
)

func TestMockLatencyRemoteService(t *testing.T) {
	t.Parallel()
	ctx := testContext(t)

	plaintext := []byte(testPlaintext)
	aesService, err := NewMockAESService(testAESKey, testKeyID)
	if err != nil {
		t.Fatal(err)
	}
	kmsService := NewMockLatencyService(aesService, testLatencyInMillisecond)

	t.Run("should be able to encrypt and decrypt with some known latency", func(t *testing.T) {
		t.Parallel()
		start := time.Now()
		encRes, err := kmsService.Encrypt(ctx, "", plaintext)
		if err != nil {
			t.Fatal(err)
		}

		duration := time.Since(start)

		if bytes.Equal(plaintext, encRes.Ciphertext) {
			t.Fatal("plaintext and ciphertext shouldn't be equal!")
		}
		// Max is set to 3s to limit the risk of a CPU limited CI node taking a long time to do encryption.
		if duration < testLatencyInMillisecond || duration > 3*time.Second {
			t.Errorf("duration for encrypt should be around: %q, have: %q", testLatencyInMillisecond, duration)
		}
		start = time.Now()
		decRes, err := kmsService.Decrypt(ctx, "", &service.DecryptRequest{
			Ciphertext:  encRes.Ciphertext,
			KeyID:       encRes.KeyID,
			Annotations: encRes.Annotations,
		})
		if err != nil {
			t.Fatal(err)
		}
		duration = time.Since(start)

		if !bytes.Equal(decRes, plaintext) {
			t.Errorf("want: %q, have: %q", plaintext, decRes)
		}
		if duration < testLatencyInMillisecond || duration > 3*time.Second {
			t.Errorf("duration decrypt should be around: %q, have: %q", testLatencyInMillisecond, duration)
		}
	})

	t.Run("should return status data", func(t *testing.T) {
		t.Parallel()
		start := time.Now()
		status, err := kmsService.Status(ctx)
		if err != nil {
			t.Fatal(err)
		}
		duration := time.Since(start)
		if status.Healthz != "ok" {
			t.Errorf("want: %q, have: %q", "ok", status.Healthz)
		}
		if len(status.KeyID) == 0 {
			t.Errorf("want: len(keyID) > 0, have: %d", len(status.KeyID))
		}
		if status.Version != version {
			t.Errorf("want %q, have: %q", version, status.Version)
		}
		if duration > 3*time.Second {
			t.Errorf("duration status should be less than: 3s, have: %q", duration)
		}
	})
}
