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

package encryption_test

import (
	"bytes"
	"context"
	"testing"

	"k8s.io/kms/encryption"
)

func TestAESGCM(t *testing.T) {
	ctx := context.Background()
	aesgcmNew, err := encryption.NewAESGCM()
	if err != nil {
		t.Fatal(err)
	}

	plaintext := []byte("lorem ipsum")
	t.Run("should be able to encrypt and decrypt from scratch", func(t *testing.T) {
		ciphertext, err := aesgcmNew.Encrypt(ctx, plaintext)
		if err != nil {
			t.Fatal(err)
		}

		decrypted, err := aesgcmNew.Decrypt(ctx, ciphertext)
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(plaintext, decrypted) {
			t.Fatalf(
				"want: %q,\nhave: %q", plaintext, decrypted,
			)
		}
	})

	for _, tt := range []struct {
		name       string
		key        []byte
		shouldFail bool
	}{
		{
			name: "should be able to re-use given key",
			key:  aesgcmNew.Key(),
		},
		{
			name:       "shouldn't be able to use non-sense key",
			key:        []byte("lorem ipsum dolor"),
			shouldFail: true,
		},
	} {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			aesgcmOld, err := encryption.FromKey(tt.key)
			if err != nil && !tt.shouldFail {
				t.Fatal(err)
			}
			if tt.shouldFail {
				return
			}

			encrypted, err := aesgcmOld.Encrypt(ctx, plaintext)
			if err != nil {
				t.Fatal(err)
			}

			decrypted, err := aesgcmNew.Decrypt(ctx, encrypted)
			if err != nil {
				t.Fatal(err)
			}

			if !bytes.Equal(plaintext, decrypted) {
				t.Fatalf(
					"want: %q,\nhave: %q", plaintext, decrypted,
				)
			}
		})
	}

	t.Run("should not be able to encrypt on empty byte slice", func(t *testing.T) {
		var plaintext []byte
		nonceKey, err := aesgcmNew.Encrypt(ctx, plaintext)
		if err == nil {
			t.Errorf("empty plaintext could've been used: %q", nonceKey)
		}
	})

	t.Run("should not be able to decrypt non-sense", func(t *testing.T) {
		pt, err := aesgcmNew.Decrypt(ctx, []byte("lorem ipsum dolor"))
		if err == nil {
			t.Errorf("non-sense got decrypted: %q", pt)
		}
	})
}
