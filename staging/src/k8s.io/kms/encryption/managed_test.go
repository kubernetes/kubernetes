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
	"encoding/base64"
	"sync"
	"testing"
	"time"
)

func TestManagedCipher(t *testing.T) {
	var id string
	var encryptedLocalKEK, ct []byte
	plaintext := []byte("lorem ipsum")
	ctx := context.Background()
	remoteKMS, err := newRemoteKMS("remoteKMSID")
	if err != nil {
		t.Fatal(err)
	}

	t.Run("encrypt with ManagedCipher", func(t *testing.T) {
		mc, err := NewManagedCipher(ctx, remoteKMS)
		if err != nil {
			t.Fatal(err)
		}

		_, encryptedLocalKEK, ct, err = mc.Encrypt(ctx, plaintext)
		if err != nil {
			t.Fatal(err)
		}

		pt, err := mc.Decrypt(ctx, id, encryptedLocalKEK, ct)
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(pt, plaintext) {
			t.Fatalf(
				"want: %q,\nhave %q",
				plaintext, pt,
			)
		}
	})

	t.Run("decrypt with another ManagedCipher", func(t *testing.T) {
		mc, err := NewManagedCipher(ctx, remoteKMS)
		if err != nil {
			t.Fatal(err)
		}

		pt, err := mc.Decrypt(ctx, id, encryptedLocalKEK, ct)
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(pt, plaintext) {
			t.Fatalf(
				"want: %q,\nhave %q",
				plaintext, pt,
			)
		}
	})
}

func TodoTestExpiry(t *testing.T) {
	// Set up remoteKMS and managedCipher
	sleepDuration := 15 * time.Minute
	_, ok := t.Deadline()
	if !ok {
		t.Logf("Please consider using -timeout of %s", sleepDuration)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	key, err := NewKey()
	if err != nil {
		t.Fatal(err)
	}

	cipher, err := NewAESGCM(key)
	if err != nil {
		t.Fatal(err)
	}

	counter := 0
	remoteKMS := remoteKMS{
		encrypt: func(ctx context.Context, plaintext []byte) (string, []byte, error) {
			counter = counter + 1
			if counter > 2 {
				time.Sleep(sleepDuration)
				t.Fatal("local kek rotation shouldn't lock the main thread")
			}

			ct, err := cipher.Encrypt(ctx, plaintext)
			if err != nil {
				return "", nil, err
			}

			return "112358", ct, nil
		},
		decrypt: func(ctx context.Context, keyID string, ciphertext []byte) ([]byte, error) {
			return cipher.Decrypt(ctx, ciphertext)
		},
	}

	mc, err := NewManagedCipher(ctx, &remoteKMS)
	if err != nil {
		t.Fatal(err)
	}
	mc.Run(ctx)

	t.Run("should work to use a different local kek (cipher) after expiry", func(t *testing.T) {
		// Encrypt 2 million times to reach the state we are looking for. Use multi-
		// threading. TODO consider injecting the cipher into constructor.
		ids := make(map[string]struct{})
		plaintext := []byte("lorem ipsum")
		ctx := context.Background()

		beyondCollision := usage + 5
		var wg sync.WaitGroup
		var m safeMap

		// This might take a couple of seconds
		var i uint32
		for ; i < beyondCollision; i++ {
			wg.Add(1)

			go func(t *testing.T, ids map[string]struct{}) {
				defer wg.Done()

				_, encKey, _, err := mc.Encrypt(ctx, plaintext)
				if err != nil {
					t.Error(err)
				}

				id := base64.StdEncoding.EncodeToString(encKey)
				m.Add(id)
			}(t, ids)
		}

		wg.Wait()

		if m.Len() != 2 {
			t.Errorf("Expected 2 encrypted keys, have: %d", len(ids))
		}
	})
}

type remoteKMS struct {
	encrypt func(context.Context, []byte) (string, []byte, error)
	decrypt func(context.Context, string, []byte) ([]byte, error)
}

var (
	_ EncrypterDecrypter = (*remoteKMS)(nil)
)

func newRemoteKMS(keyID string) (*remoteKMS, error) {
	key, err := NewKey()
	if err != nil {
		return nil, err
	}
	cipher, err := NewAESGCM(key)
	if err != nil {
		return nil, err
	}

	return &remoteKMS{
		encrypt: func(ctx context.Context, pt []byte) (string, []byte, error) {
			ct, err := cipher.Encrypt(ctx, pt)
			if err != nil {
				return "", nil, err
			}

			return keyID, ct, nil
		},
		decrypt: func(ctx context.Context, keyID string, encryptedKey []byte) ([]byte, error) {
			pt, err := cipher.Decrypt(ctx, encryptedKey)
			if err != nil {
				return nil, err
			}

			return pt, nil
		},
	}, nil
}

func (k *remoteKMS) Encrypt(ctx context.Context, plaintext []byte) (string, []byte, error) {
	return k.encrypt(ctx, plaintext)
}

func (k *remoteKMS) Decrypt(ctx context.Context, keyID string, ciphertext []byte) ([]byte, error) {
	return k.decrypt(ctx, keyID, ciphertext)
}

type safeMap struct {
	ma map[string]struct{}
	mu sync.Mutex
}

func (m *safeMap) Add(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.ma == nil {
		m.ma = make(map[string]struct{})
	}

	m.ma[id] = struct{}{}
}

func (m *safeMap) Len() int {
	return len(m.ma)
}
