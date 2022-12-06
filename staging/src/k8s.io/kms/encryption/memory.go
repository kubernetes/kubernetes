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
	"context"
	"errors"

	"github.com/google/uuid"

	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2"
)

// InMemory is a kmsv2.Service wrapper around AESGCM. It can be used for testing
// or experimenting. It isn't meant to be used on production as the key is only
// in memory.
type InMemory struct {
	keyID  string
	cipher *AESGCM
}

var (
	_ kmsv2.Service = (*InMemory)(nil)

	version = "v2alpha1"

	// ErrKeyIDMismatch is returned, when the expected KeyID doesn't match the
	// existing one. This is an indicator that the ciphertext was encrypted with
	// a different key.
	ErrKeyIDMismatch = errors.New("KeyID doesn't match")
)

// NewInMemory creates a in-memory kmsv2.Service that encrypts with a cipher.
func NewInMemory() (*InMemory, error) {
	aesgcm, err := NewAESGCM()
	if err != nil {
		return nil, err
	}

	return newInMemory(aesgcm)
}

func newInMemory(aesgcm *AESGCM) (*InMemory, error) {
	id, err := uuid.NewRandom()
	if err != nil {
		return nil, err
	}

	return &InMemory{keyID: id.String(), cipher: aesgcm}, nil
}

// Status returns the KeyID (UUID), the current API version and a health status
// message. The health status returns "ok" if everything is working, and returns
// a more detailed version, if there are limitations on its usage.
func (m *InMemory) Status(ctx context.Context) (*kmsv2.StatusResponse, error) {
	health := "ok"
	if !m.cipher.IsValid() {
		health = "decrypt: ok, encrypt: ErrKeyExpired"
	}

	return &kmsv2.StatusResponse{
		Version: version,
		KeyID:   m.keyID,
		Healthz: health,
	}, nil
}

// Encrypt encrypts given data with its cipher. Will fail, on exhausted cipher.
func (m *InMemory) Encrypt(ctx context.Context, uid string, data []byte) (*kmsv2.EncryptResponse, error) {
	ciphertext, err := m.cipher.Encrypt(ctx, data)
	if err != nil {
		return nil, err
	}

	return &kmsv2.EncryptResponse{
		Ciphertext: ciphertext,
		KeyID:      m.keyID,
	}, nil
}

// Decrypt decrypts given kmsv2.DecryptRequest with its cipher. Will continue to
// work on exhausted cipher.
func (m *InMemory) Decrypt(ctx context.Context, uid string, req *kmsv2.DecryptRequest) ([]byte, error) {
	if m.keyID != req.KeyID {
		return nil, ErrKeyIDMismatch
	}

	plaintext, err := m.cipher.Decrypt(ctx, req.Ciphertext)
	if err != nil {
		return nil, err
	}

	return plaintext, nil
}
