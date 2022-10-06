package encryption_test

import (
	"bytes"
	"encoding/base64"
	"sync"
	"testing"
	"time"

	"k8s.io/kms/encryption"
)

func TestManagedCipher(t *testing.T) {
	var id, encryptedLocalKEK, ct []byte
	plaintext := []byte("lorem ipsum")
	remoteKMS, err := newRemoteKMS([]byte("helloworld"))
	if err != nil {
		t.Fatal(err)
	}

	t.Run("encrypt with ManagedCipher", func(t *testing.T) {
		mc, err := encryption.NewManagedCipher(remoteKMS)
		if err != nil {
			t.Fatal(err)
		}

		_, encryptedLocalKEK, ct, err = mc.Encrypt(plaintext)
		if err != nil {
			t.Fatal(err)
		}

		pt, err := mc.Decrypt(id, encryptedLocalKEK, ct)
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
		mc, err := encryption.NewManagedCipher(remoteKMS)
		if err != nil {
			t.Fatal(err)
		}

		pt, err := mc.Decrypt(id, encryptedLocalKEK, ct)
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

func TestExpiry(t *testing.T) {
	// Set up remoteKMS and managedCipher
	sleepDuration := 15 * time.Minute
	_, ok := t.Deadline()
	if !ok {
		t.Logf("Please consider using -timeout of %s", sleepDuration)
	}

	cipher, err := encryption.NewAESGCM()
	if err != nil {
		t.Fatal(err)
	}

	counter := 0
	remoteKMS := remoteKMS{
		encrypt: func(plaintext []byte) ([]byte, []byte, error) {
			counter = counter + 1
			if counter > 2 {
				time.Sleep(sleepDuration)
				t.Fatal("local kek rotation shouldn't lock the main thread")
			}

			ct, err := cipher.Encrypt(plaintext)
			if err != nil {
				return nil, nil, err
			}

			return []byte("112358"), ct, nil
		},
		decrypt: func(keyID, ciphertext []byte) ([]byte, error) {
			return cipher.Decrypt(ciphertext)
		},
	}

	mc, err := encryption.NewManagedCipher(&remoteKMS)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("should work to use a different local kek (cipher) after expiry", func(t *testing.T) {
		// Encrypt 2 million times to reach the state we are looking for. Use multi-
		// threading. TODO consider injecting the cipher into constructor.
		ids := make(map[string]struct{})
		plaintext := []byte("lorem ipsum")

		beyondCollision := encryption.MaxUsage + 5
		var wg sync.WaitGroup
		var m safeMap

		// This might take a couple of seconds
		for i := 0; i < beyondCollision; i++ {
			wg.Add(1)

			go func(t *testing.T, ids map[string]struct{}) {
				defer wg.Done()

				_, encKey, _, err := mc.Encrypt(plaintext)
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
	encrypt func([]byte) ([]byte, []byte, error)
	decrypt func([]byte, []byte) ([]byte, error)
}

var (
	_ encryption.EncrypterDecrypter = (*remoteKMS)(nil)
)

func newRemoteKMS(keyID []byte) (*remoteKMS, error) {
	cipher, err := encryption.NewAESGCM()
	if err != nil {
		return nil, err
	}

	return &remoteKMS{
		encrypt: func(pt []byte) ([]byte, []byte, error) {
			ct, err := cipher.Encrypt(pt)
			if err != nil {
				return nil, nil, err
			}

			return keyID, ct, nil
		},
		decrypt: func(keyID []byte, encryptedKey []byte) ([]byte, error) {
			pt, err := cipher.Decrypt(encryptedKey)
			if err != nil {
				return nil, err
			}

			return pt, nil
		},
	}, nil
}

func (k *remoteKMS) Encrypt(plaintext []byte) ([]byte, []byte, error) {
	return k.encrypt(plaintext)
}

func (k *remoteKMS) Decrypt(keyID, ciphertext []byte) ([]byte, error) {
	return k.decrypt(keyID, ciphertext)
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
