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
)

func TestAESGCM(t *testing.T) {
	ctx := context.Background()
	key, err := NewKey()
	if err != nil {
		t.Fatal(err)
	}
	aesgcmNew, err := NewAESGCM(key)
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
			t.Fatalf("want: %q,\nhave: %q", plaintext, decrypted)
		}
	})

	t.Run("should not be able to decrypt non-sense", func(t *testing.T) {
		pt, err := aesgcmNew.Decrypt(ctx, []byte("lorem ipsum dolor"))
		if err == nil {
			t.Errorf("non-sense got decrypted: %q", pt)
		}
	})
}
