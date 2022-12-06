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

package encryption

import (
	"bytes"
	"context"
	"testing"
	"time"

	"github.com/google/uuid"

	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2"
)

func TestInMemory(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	plaintext := []byte("lorem ipsum dolor")
	m, err := NewInMemory()
	if err != nil {
		t.Fatal(err)
	}

	t.Run("should have a healthy state after initialization", func(t *testing.T) {
		res, err := m.Status(ctx)
		if err != nil {
			t.Fatal(err)
		}

		if res.Version != version {
			t.Errorf("want: %q, have: %q", version, res.Version)
		}

		if res.Healthz != "ok" {
			t.Errorf("want 'ok', have: %q", res.Healthz)
		}

		if _, err := uuid.Parse(res.KeyID); err != nil {
			t.Error(err)

		}
	})

	t.Run("should be able to de/encrypt", func(t *testing.T) {
		uid, err := uuid.NewRandom()
		if err != nil {
			t.Fatal(err)
		}

		res, err := m.Encrypt(ctx, uid.String(), plaintext)
		if err != nil {
			t.Fatal(err)
		}

		pt, err := m.Decrypt(ctx, uid.String(), &kmsv2.DecryptRequest{
			Ciphertext: res.Ciphertext,
			KeyID:      res.KeyID,
		})
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(pt, plaintext) {
			t.Errorf("want: %q, have: %s", plaintext, pt)
		}
	})

	t.Run("should be able to decrypt, but not encrypt on expired key", func(t *testing.T) {
		key, err := randomBytes(keySize)
		if err != nil {
			t.Fatal(err)
		}

		counter := maxUsage - 1
		aesgcm, err := newAESGCM(key, counter, time.Now().Add(week))
		if err != nil {
			t.Fatal(err)
		}

		m, err := newInMemory(aesgcm)
		if err != nil {
			t.Fatal(err)
		}

		id, err := uuid.NewRandom()
		if err != nil {
			t.Fatal(err)
		}

		res, err := m.Encrypt(ctx, id.String(), plaintext)
		if err != nil {
			t.Fatal(err)
		}

		decrypted, err := m.Decrypt(ctx, id.String(), &kmsv2.DecryptRequest{
			Ciphertext: res.Ciphertext,
			KeyID:      res.KeyID,
		})
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(decrypted, plaintext) {
			t.Errorf("want: %q, have: %q", plaintext, decrypted)
		}

		if _, err := m.Encrypt(ctx, id.String(), plaintext); err != ErrKeyExpired {
			t.Errorf("want: %q, have: %q", ErrKeyExpired, err)
		}
	})

	t.Run("shouldn't attempt to decrypt on different keyIDs", func(t *testing.T) {
		id, err := uuid.NewRandom()
		if err != nil {
			t.Fatal(err)
		}

		res, err := m.Encrypt(ctx, id.String(), plaintext)
		if err != nil {
			t.Fatal(err)
		}

		n, err := NewInMemory()
		if err != nil {
			t.Fatal(err)
		}

		if _, err := n.Decrypt(ctx, id.String(), &kmsv2.DecryptRequest{
			Ciphertext: res.Ciphertext,
			KeyID:      res.KeyID,
		}); err != ErrKeyIDMismatch {
			t.Errorf("want: %q, have: %q", ErrKeyIDMismatch, err)
		}
	})
}
