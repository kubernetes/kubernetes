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

// Package aes transforms values for storage at rest.
package aes

import (
	"crypto/cipher"
	"crypto/rand"
	"fmt"

	"k8s.io/apiserver/pkg/storage/value"
)

type gcm struct {
	block cipher.Block
}

// NewGCMTransformer takes the given block cipher and performs encryption and decryption on the given
// data.
func NewGCMTransformer(block cipher.Block) value.Transformer {
	return &gcm{block: block}
}

func (t *gcm) TransformFromStorage(data []byte, context value.Context) ([]byte, bool, error) {
	aead, err := cipher.NewGCM(t.block)
	if err != nil {
		return nil, false, err
	}
	nonceSize := aead.NonceSize()
	if len(data) < nonceSize {
		return nil, false, fmt.Errorf("the stored data was shorter than the required size")
	}
	result, err := aead.Open(nil, data[:nonceSize], data[nonceSize:], context.AuthenticatedData())
	return result, false, err
}

func (t *gcm) TransformToStorage(data []byte, context value.Context) ([]byte, error) {
	aead, err := cipher.NewGCM(t.block)
	if err != nil {
		return nil, err
	}
	nonceSize := aead.NonceSize()
	result := make([]byte, nonceSize+aead.Overhead()+len(data))
	n, err := rand.Read(result[:nonceSize])
	if err != nil {
		return nil, err
	}
	if n != nonceSize {
		return nil, fmt.Errorf("unable to read sufficient random bytes")
	}
	cipherText := aead.Seal(result[nonceSize:nonceSize], result[:nonceSize], data, context.AuthenticatedData())
	return result[:nonceSize+len(cipherText)], nil
}
