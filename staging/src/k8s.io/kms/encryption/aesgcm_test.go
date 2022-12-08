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
)

func TestAESGCM(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	aesgcm, err := NewAESGCM()
	if err != nil {
		t.Fatal(err)
	}

	plaintext := []byte("lorem ipsum")
	t.Run("should be able to encrypt and decrypt from scratch", func(t *testing.T) {
		t.Parallel()
		ciphertext, err := aesgcm.TransformToStorage(ctx, plaintext, DefaultContext{})
		if err != nil {
			t.Fatal(err)
		}

		if bytes.Equal(plaintext, ciphertext) {
			t.Fatalf("plaintext (%q) and ciphertext (%q) shouldn't be equal", plaintext, ciphertext)
		}

		decrypted, _, err := aesgcm.TransformFromStorage(ctx, ciphertext, DefaultContext{})
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(plaintext, decrypted) {
			t.Fatalf("want: %q,\nhave: %q", plaintext, decrypted)
		}
	})

	t.Run("should not create the same cipher with same key and same plaintext", func(t *testing.T) {
		t.Parallel()
		ciphertextA, err := aesgcm.TransformToStorage(ctx, plaintext, DefaultContext{})
		if err != nil {
			t.Fatal(err)
		}

		ciphertextB, err := aesgcm.TransformToStorage(ctx, plaintext, DefaultContext{})
		if err != nil {
			t.Fatal(err)
		}

		if bytes.Equal(ciphertextA, ciphertextB) {
			t.Error("encrypted plaintext twice and ciphertext are identic.")
		}
	})

	t.Run("should not be able to decrypt non-sense", func(t *testing.T) {
		t.Parallel()
		pt, _, err := aesgcm.TransformFromStorage(ctx, []byte("lorem ipsum dolor"), DefaultContext{})
		if err == nil {
			t.Errorf("non-sense got decrypted: %q", pt)
		}
	})

	t.Run("should be able to use cipher on high counter below max usage", func(t *testing.T) {
		t.Parallel()
		key, err := RandomBytes(keySize)
		if err != nil {
			t.Fatal(err)
		}

		counter := safeUsage + 1
		aesgcm, err := newAESGCM(key, counter, time.Now().Add(week))
		if err != nil {
			t.Fatal(err)
		}

		if _, err := aesgcm.TransformToStorage(ctx, plaintext, DefaultContext{}); err != nil {
			t.Errorf("want: nil, have: %q", err)
		}
	})

	t.Run("should not be able to use cipher on counter beyond max usage", func(t *testing.T) {
		t.Parallel()
		key, err := RandomBytes(keySize)
		if err != nil {
			t.Fatal(err)
		}

		counter := maxUsage + 1
		aesgcm, err := newAESGCM(key, counter, time.Now().Add(week))
		if err != nil {
			t.Fatal(err)
		}

		if _, err := aesgcm.TransformToStorage(ctx, plaintext, DefaultContext{}); err != ErrKeyExpired {
			t.Error("key should be expired by counter")
		}
	})

	t.Run("should not be able to use cipher on expiry by time", func(t *testing.T) {
		t.Parallel()
		key, err := RandomBytes(keySize)
		if err != nil {
			t.Fatal(err)
		}

		aesgcm, err := newAESGCM(key, 0, time.Now())
		if err != nil {
			t.Fatal(err)
		}

		if _, err := aesgcm.TransformToStorage(ctx, plaintext, DefaultContext{}); err != ErrKeyExpired {
			t.Error("key should be expired by counter")
		}
	})
}
