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

// Package kms transforms values for storage at rest using a KMS provider
package kms

import (
	"bytes"
	"crypto/aes"
	"crypto/rand"
	"fmt"
	"strings"
	"sync"
	"time"

	randutil "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
)

const (
	keyNameLength    = 5
	primaryKeyPrefix = "-"
)

type kmsTransformer struct {
	kmsService value.KMSService

	transformers   []value.PrefixTransformer
	primaryKeyName string

	storage value.KMSStorage

	rotateLock  sync.RWMutex
	refreshLock sync.RWMutex
}

// NewKMSTransformer returns a transformer which implements a KEK-DEK based envelope encryption scheme.
// It uses kmsService to communicate with the KEK store, and storage to communicate with the DEK store.
func NewKMSTransformer(kmsService value.KMSService, storage value.KMSStorage) (value.Transformer, error) {
	err := storage.Setup()
	if err != nil {
		return nil, err
	}
	transformer := &kmsTransformer{
		kmsService: kmsService,
		storage:    storage,
	}

	deks, err := transformer.storage.GetAllDEKs()
	if err != nil {
		return nil, err
	}
	if len(deks) == 0 {
		// Create new DEK in that case
		// If there are no keys, rotate(false) will create one.
		if err = transformer.Rotate(); err != nil {
			return nil, err
		}
	}

	return transformer, nil
}

// Rotate creates a new key and makes it the default for writing to disk. It refreshes the transformer
// once the new key has been written to disk.
func (t *kmsTransformer) Rotate() error {
	t.rotateLock.Lock()
	defer t.rotateLock.Unlock()

	deks, err := t.storage.GetAllDEKs()
	if err != nil {
		return err
	}

	newDEKs := map[string]string{}
	for keyname, dek := range deks {
		// Remove the identifying prefix in front of the primary key.
		if strings.HasPrefix(keyname, "-") {
			keyname = keyname[1:]
		}
		newDEKs[keyname] = dek
	}

	// Now ensure the new primary key also has the identifying marker.
	keyname := "-" + generateName(newDEKs)
	dekBytes, err := generateKey(32)
	if err != nil {
		return err
	}

	newDEKs[keyname], err = t.kmsService.Encrypt(dekBytes)
	if err != nil {
		return err
	}
	t.storage.StoreNewDEKs(newDEKs)

	return t.Refresh()
}

// Refresh reads the DEKs from disk and recreates the transformer.
func (t *kmsTransformer) Refresh() error {
	t.refreshLock.Lock()
	defer t.refreshLock.Unlock()

	deks, err := t.storage.GetAllDEKs()
	if err != nil {
		return err
	}
	transformers := []value.PrefixTransformer{}

	primary := false
	for keyname, encDek := range deks {
		// Check if the current key is the primary key. Necessary because maps are unordered.
		if strings.HasPrefix(keyname, "-") {
			keyname = keyname[1:]
			primary = true
		} else {
			primary = false
		}

		dekBytes, err := t.kmsService.Decrypt(encDek)
		if err != nil {
			return err
		}
		block, err := aes.NewCipher(dekBytes)
		if err != nil {
			return err
		}
		prefixTransformer := value.PrefixTransformer{
			Prefix:      []byte(keyname + ":"),
			Transformer: aestransformer.NewCBCTransformer(block),
		}

		if primary {
			// The primary key has to be at the beginning of the list
			transformers = append([]value.PrefixTransformer{prefixTransformer}, transformers...)
		} else {
			transformers = append(transformers, prefixTransformer)
		}
	}

	// TODO(sakshams): Confirm that this can be done safely without any race conditions.
	t.transformers = transformers

	return nil
}

// TransformFromStorage implements value.Transformer
func (t *kmsTransformer) TransformFromStorage(data []byte, context value.Context) ([]byte, bool, error) {
	for attempt := 0; attempt < 5; attempt++ {
		for i, transformer := range t.transformers {
			if bytes.HasPrefix(data, transformer.Prefix) {
				result, stale, err := transformer.Transformer.TransformFromStorage(data[len(transformer.Prefix):], context)
				if len(transformer.Prefix) == 0 && err != nil {
					continue
				}
				return result, stale || i != 0, err
			}
		}
		// A new key may have been added.
		// TODO(saksham): Do we need the iterative back off?
		time.Sleep(50 * time.Duration(attempt) * time.Millisecond)
		t.Refresh()
	}
	return nil, false, fmt.Errorf("did not find a transformer to read key")
}

// TransformToStorage implements value.Transformer
func (t *kmsTransformer) TransformToStorage(data []byte, context value.Context) ([]byte, error) {
	transformer := t.transformers[0]
	prefixedData := make([]byte, len(transformer.Prefix), len(data)+len(transformer.Prefix))
	copy(prefixedData, transformer.Prefix)
	result, err := transformer.Transformer.TransformToStorage(data, context)
	if err != nil {
		return nil, err
	}
	prefixedData = append(prefixedData, result...)
	return prefixedData, nil
}

// generateName generates a unique new name for the new DEK.
func generateName(existingNames map[string]string) string {
	name := randutil.String(keyNameLength)

	_, ok := existingNames[name]
	for ok {
		name := randutil.String(keyNameLength)
		_, ok = existingNames[name]
	}

	return name
}

// generateKey generates a random key using system randomness.
func generateKey(length int) ([]byte, error) {
	key := make([]byte, length)
	_, err := rand.Read(key)
	if err != nil {
		return []byte{}, err
	}

	return key, nil
}
