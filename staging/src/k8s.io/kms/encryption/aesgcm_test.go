package encryption_test

import (
	"bytes"
	"testing"

	"k8s.io/kms/encryption"
)

func TestAESGCM(t *testing.T) {
	aesgcmNew, err := encryption.NewAESGCM()
	if err != nil {
		t.Fatal(err)
	}

	plaintext := []byte("lorem ipsum")
	t.Run("should be able to encrypt and decrypt from scratch", func(t *testing.T) {
		ciphertext, err := aesgcmNew.Encrypt(plaintext)
		if err != nil {
			t.Fatal(err)
		}

		decrypted, err := aesgcmNew.Decrypt(ciphertext)
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

			encrypted, err := aesgcmOld.Encrypt(plaintext)
			if err != nil {
				t.Fatal(err)
			}

			decrypted, err := aesgcmNew.Decrypt(encrypted)
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
		nonceKey, err := aesgcmNew.Encrypt(plaintext)
		if err == nil {
			t.Errorf("empty plaintext could've been used: %q", nonceKey)
		}
	})

	t.Run("should not be able to decrypt non-sense", func(t *testing.T) {
		pt, err := aesgcmNew.Decrypt([]byte("lorem ipsum dolor"))
		if err == nil {
			t.Errorf("non-sense got decrypted: %q", pt)
		}
	})
}
