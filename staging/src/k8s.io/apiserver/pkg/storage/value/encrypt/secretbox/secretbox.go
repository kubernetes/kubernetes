/*
Copyright 2017 The Kubernetes Authors.

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

// Package secretbox transforms values for storage at rest using XSalsa20 and Poly1305.
package secretbox

import (
	"context"
	"crypto/rand"
	"fmt"

	"golang.org/x/crypto/nacl/secretbox"

	"k8s.io/apiserver/pkg/storage/value"
)

// secretbox implements at rest encryption of the provided values given a 32 byte secret key.
// Uses a standard 24 byte nonce (placed at the beginning of the cipher text) generated
// from crypto/rand. Does not perform authentication of the data at rest.
type secretboxTransformer struct {
	key [32]byte
}

const nonceSize = 24

// NewSecretboxTransformer takes the given key and performs encryption and decryption on the given
// data.
func NewSecretboxTransformer(key [32]byte) value.Transformer {
	return &secretboxTransformer{key: key}
}

func (t *secretboxTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	if len(data) < (secretbox.Overhead + nonceSize) {
		return nil, false, fmt.Errorf("the stored data was shorter than the required size")
	}
	var nonce [nonceSize]byte
	copy(nonce[:], data[:nonceSize])
	data = data[nonceSize:]
	out := make([]byte, 0, len(data)-secretbox.Overhead)
	result, ok := secretbox.Open(out, data, &nonce, &t.key)
	if !ok {
		return nil, false, fmt.Errorf("output array was not large enough for encryption")
	}
	return result, false, nil
}

func (t *secretboxTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	var nonce [nonceSize]byte
	n, err := rand.Read(nonce[:])
	if err != nil {
		return nil, err
	}
	if n != nonceSize {
		return nil, fmt.Errorf("unable to read sufficient random bytes")
	}
	return secretbox.Seal(nonce[:], data, &nonce, &t.key), nil
}
