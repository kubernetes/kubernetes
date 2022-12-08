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
	"testing"
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
		t.Parallel()
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

		if len(res.KeyID) == 0 {
			t.Error("keyID should consist of 10 chars")
		}
	})

	t.Run("should be able to de/encrypt", func(t *testing.T) {
		t.Parallel()
		uid, err := makeID(10)
		if err != nil {
			t.Fatal(err)
		}

		res, err := m.Encrypt(ctx, string(uid), plaintext)
		if err != nil {
			t.Fatal(err)
		}

		pt, err := m.Decrypt(ctx, string(uid), &DecryptRequest{
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

	t.Run("shouldn't attempt to decrypt on different keyIDs", func(t *testing.T) {
		t.Parallel()
		id, err := makeID(10)
		if err != nil {
			t.Fatal(err)
		}

		res, err := m.Encrypt(ctx, id, plaintext)
		if err != nil {
			t.Fatal(err)
		}

		n, err := NewInMemory()
		if err != nil {
			t.Fatal(err)
		}

		if _, err := n.Decrypt(ctx, id, &DecryptRequest{
			Ciphertext: res.Ciphertext,
			KeyID:      res.KeyID,
		}); err != ErrKeyIDMismatch {
			t.Errorf("want: %q, have: %q", ErrKeyIDMismatch, err)
		}
	})
}
