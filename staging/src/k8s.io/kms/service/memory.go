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
	"context"
	"encoding/base64"
	"errors"

	"k8s.io/kms/encryption"
)

// InMemory is a kmsv2.wrapper around AESGCM. It can be used for testing
// or experimenting. It isn't meant to be used on production as the key is only
// in memory.
type InMemory struct {
	keyID       string
	transformer encryption.Transformer
}

const version = "v2alpha1"

var (
	_ = (*InMemory)(nil)

	// ErrKeyIDMismatch is returned, when the expected KeyID doesn't match the
	// existing one. This is an indicator that the ciphertext was encrypted with
	// a different key.
	ErrKeyIDMismatch = errors.New("KeyID doesn't match")
)

// NewInMemory creates a in-memory kmsv2.that encrypts with a cipher.
func NewInMemory() (*InMemory, error) {
	aesgcm, err := encryption.NewAESGCM()
	if err != nil {
		return nil, err
	}

	return newInMemory(aesgcm)
}

func newInMemory(transformer encryption.Transformer) (*InMemory, error) {
	id, err := makeID(10)
	if err != nil {
		return nil, err
	}

	return &InMemory{keyID: id, transformer: transformer}, nil
}

// Status returns the KeyID (UUID), the current API version and a health status
// message. The health status returns "ok" if everything is working, and returns
// a more detailed version, if there are limitations on its usage.
func (m *InMemory) Status(ctx context.Context) (*StatusResponse, error) {
	health := "ok"
	if _, err := m.transformer.TransformToStorage(ctx, []byte(health), encryption.DefaultContext{}); err != nil {
		health = err.Error()
	}

	return &StatusResponse{
		Version: version,
		KeyID:   m.keyID,
		Healthz: health,
	}, nil
}

// Encrypt encrypts given data with its cipher. Will fail, on exhausted cipher.
func (m *InMemory) Encrypt(ctx context.Context, uid string, data []byte) (*EncryptResponse, error) {
	ciphertext, err := m.transformer.TransformToStorage(ctx, data, encryption.DefaultContext{})
	if err != nil {
		return nil, err
	}

	return &EncryptResponse{
		Ciphertext: ciphertext,
		KeyID:      m.keyID,
	}, nil
}

// Decrypt decrypts given kmsv2.DecryptRequest with its cipher. Will continue to
// work on exhausted cipher.
func (m *InMemory) Decrypt(ctx context.Context, uid string, req *DecryptRequest) ([]byte, error) {
	if m.keyID != req.KeyID {
		return nil, ErrKeyIDMismatch
	}

	plaintext, _, err := m.transformer.TransformFromStorage(ctx, req.Ciphertext, encryption.DefaultContext{})
	if err != nil {
		return nil, err
	}

	return plaintext, nil
}

func makeID(length int) (string, error) {
	random, err := encryption.RandomBytes(length)
	if err != nil {
		return "", err
	}

	return base64.StdEncoding.EncodeToString(random), nil
}
