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

// Package kmsv2 transforms values for storage at rest using a Envelope v2 provider
package kmsv2

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"time"

	"github.com/gogo/protobuf/proto"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/value"
	kmstypes "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2/v2alpha1"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/metrics"
	"k8s.io/klog/v2"
	"k8s.io/utils/lru"
)

const (
	// KMSAPIVersion is the version of the KMS API.
	KMSAPIVersion = "v2alpha1"
	// annotationsMaxSize is the maximum size of the annotations.
	annotationsMaxSize = 32 * 1024 // 32 kB
	// keyIDMaxSize is the maximum size of the keyID.
	keyIDMaxSize = 1 * 1024 // 1 kB
	// encryptedDEKMaxSize is the maximum size of the encrypted DEK.
	encryptedDEKMaxSize = 1 * 1024 // 1 kB
)

// Service allows encrypting and decrypting data using an external Key Management Service.
type Service interface {
	// Decrypt a given bytearray to obtain the original data as bytes.
	Decrypt(ctx context.Context, uid string, req *DecryptRequest) ([]byte, error)
	// Encrypt bytes to a ciphertext.
	Encrypt(ctx context.Context, uid string, data []byte) (*EncryptResponse, error)
	// Status returns the status of the KMS.
	Status(ctx context.Context) (*StatusResponse, error)
}

type envelopeTransformer struct {
	envelopeService Service

	// transformers is a thread-safe LRU cache which caches decrypted DEKs indexed by their encrypted form.
	transformers *lru.Cache

	// baseTransformerFunc creates a new transformer for encrypting the data with the DEK.
	baseTransformerFunc func(cipher.Block) value.Transformer

	cacheSize    int
	cacheEnabled bool
}

// EncryptResponse is the response from the Envelope service when encrypting data.
type EncryptResponse struct {
	Ciphertext  []byte
	KeyID       string
	Annotations map[string][]byte
}

// DecryptRequest is the request to the Envelope service when decrypting data.
type DecryptRequest struct {
	Ciphertext  []byte
	KeyID       string
	Annotations map[string][]byte
}

// StatusResponse is the response from the Envelope service when getting the status of the service.
type StatusResponse struct {
	Version string
	Healthz string
	KeyID   string
}

// NewEnvelopeTransformer returns a transformer which implements a KEK-DEK based envelope encryption scheme.
// It uses envelopeService to encrypt and decrypt DEKs. Respective DEKs (in encrypted form) are prepended to
// the data items they encrypt. A cache (of size cacheSize) is maintained to store the most recently
// used decrypted DEKs in memory.
func NewEnvelopeTransformer(envelopeService Service, cacheSize int, baseTransformerFunc func(cipher.Block) value.Transformer) value.Transformer {
	var cache *lru.Cache

	if cacheSize > 0 {
		// TODO(aramase): Switch to using expiring cache: kubernetes/kubernetes/staging/src/k8s.io/apimachinery/pkg/util/cache/expiring.go.
		// It handles scans a lot better, doesn't have to be right sized, and don't have a global lock on reads.
		cache = lru.New(cacheSize)
	}

	return &envelopeTransformer{
		envelopeService:     envelopeService,
		transformers:        cache,
		baseTransformerFunc: baseTransformerFunc,
		cacheEnabled:        cacheSize > 0,
		cacheSize:           cacheSize,
	}
}

// TransformFromStorage decrypts data encrypted by this transformer using envelope encryption.
func (t *envelopeTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	metrics.RecordArrival(metrics.FromStorageLabel, time.Now())

	// Deserialize the EncryptedObject from the data.
	encryptedObject, err := t.doDecode(data)
	if err != nil {
		return nil, false, err
	}

	// Look up the decrypted DEK from cache or Envelope.
	transformer := t.getTransformer(encryptedObject.EncryptedDEK)
	if transformer == nil {
		if t.cacheEnabled {
			value.RecordCacheMiss()
		}
		uid := string(uuid.NewUUID())
		klog.V(6).InfoS("Decrypting content using envelope service", "uid", uid, "key", string(dataCtx.AuthenticatedData()))
		key, err := t.envelopeService.Decrypt(ctx, uid, &DecryptRequest{
			Ciphertext:  encryptedObject.EncryptedDEK,
			KeyID:       encryptedObject.KeyID,
			Annotations: encryptedObject.Annotations,
		})
		if err != nil {
			return nil, false, fmt.Errorf("failed to decrypt DEK, error: %w", err)
		}

		transformer, err = t.addTransformer(encryptedObject.EncryptedDEK, key)
		if err != nil {
			return nil, false, err
		}
	}

	return transformer.TransformFromStorage(ctx, encryptedObject.EncryptedData, dataCtx)
}

// TransformToStorage encrypts data to be written to disk using envelope encryption.
func (t *envelopeTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	metrics.RecordArrival(metrics.ToStorageLabel, time.Now())
	newKey, err := generateKey(32)
	if err != nil {
		return nil, err
	}

	uid := string(uuid.NewUUID())
	klog.V(6).InfoS("Encrypting content using envelope service", "uid", uid, "key", string(dataCtx.AuthenticatedData()))
	resp, err := t.envelopeService.Encrypt(ctx, uid, newKey)
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt DEK, error: %w", err)
	}

	transformer, err := t.addTransformer(resp.Ciphertext, newKey)
	if err != nil {
		return nil, err
	}

	result, err := transformer.TransformToStorage(ctx, data, dataCtx)
	if err != nil {
		return nil, err
	}

	encObject := &kmstypes.EncryptedObject{
		KeyID:         resp.KeyID,
		EncryptedDEK:  resp.Ciphertext,
		EncryptedData: result,
		Annotations:   resp.Annotations,
	}

	// Serialize the EncryptedObject to a byte array.
	return t.doEncode(encObject)
}

// addTransformer inserts a new transformer to the Envelope cache of DEKs for future reads.
func (t *envelopeTransformer) addTransformer(encKey []byte, key []byte) (value.Transformer, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	transformer := t.baseTransformerFunc(block)
	// Use base64 of encKey as the key into the cache because hashicorp/golang-lru
	// cannot hash []uint8.
	if t.cacheEnabled {
		t.transformers.Add(base64.StdEncoding.EncodeToString(encKey), transformer)
		metrics.RecordDekCacheFillPercent(float64(t.transformers.Len()) / float64(t.cacheSize))
	}
	return transformer, nil
}

// getTransformer fetches the transformer corresponding to encKey from cache, if it exists.
func (t *envelopeTransformer) getTransformer(encKey []byte) value.Transformer {
	if !t.cacheEnabled {
		return nil
	}

	_transformer, found := t.transformers.Get(base64.StdEncoding.EncodeToString(encKey))
	if found {
		return _transformer.(value.Transformer)
	}
	return nil
}

// doEncode encodes the EncryptedObject to a byte array.
func (t *envelopeTransformer) doEncode(request *kmstypes.EncryptedObject) ([]byte, error) {
	if err := validateEncryptedObject(request); err != nil {
		return nil, err
	}
	return proto.Marshal(request)
}

// doDecode decodes the byte array to an EncryptedObject.
func (t *envelopeTransformer) doDecode(originalData []byte) (*kmstypes.EncryptedObject, error) {
	o := &kmstypes.EncryptedObject{}
	if err := proto.Unmarshal(originalData, o); err != nil {
		return nil, err
	}
	// validate the EncryptedObject
	if err := validateEncryptedObject(o); err != nil {
		return nil, err
	}

	return o, nil
}

// generateKey generates a random key using system randomness.
func generateKey(length int) (key []byte, err error) {
	defer func(start time.Time) {
		value.RecordDataKeyGeneration(start, err)
	}(time.Now())
	key = make([]byte, length)
	if _, err = rand.Read(key); err != nil {
		return nil, err
	}

	return key, nil
}

func validateEncryptedObject(o *kmstypes.EncryptedObject) error {
	if o == nil {
		return fmt.Errorf("encrypted object is nil")
	}
	if len(o.EncryptedData) == 0 {
		return fmt.Errorf("encrypted data is empty")
	}
	if err := validateEncryptedDEK(o.EncryptedDEK); err != nil {
		return fmt.Errorf("failed to validate encrypted DEK: %w", err)
	}
	if err := validateKeyID(o.KeyID); err != nil {
		return fmt.Errorf("failed to validate key id: %w", err)
	}
	if err := validateAnnotations(o.Annotations); err != nil {
		return fmt.Errorf("failed to validate annotations: %w", err)
	}
	return nil
}

// validateEncryptedDEK tests the following:
// 1. The encrypted DEK is not empty.
// 2. The size of encrypted DEK is less than 1 kB.
func validateEncryptedDEK(encryptedDEK []byte) error {
	if len(encryptedDEK) == 0 {
		return fmt.Errorf("encrypted DEK is empty")
	}
	if len(encryptedDEK) > encryptedDEKMaxSize {
		return fmt.Errorf("encrypted DEK is %d bytes, which exceeds the max size of %d", len(encryptedDEK), encryptedDEKMaxSize)
	}
	return nil
}

// validateAnnotations tests the following:
//  1. checks if the annotation key is fully qualified
//  2. The size of annotations keys + values is less than 32 kB.
func validateAnnotations(annotations map[string][]byte) error {
	var errs []error
	var totalSize uint64
	for k, v := range annotations {
		if fieldErr := validation.IsFullyQualifiedDomainName(field.NewPath("annotations"), k); fieldErr != nil {
			errs = append(errs, fieldErr.ToAggregate())
		}
		totalSize += uint64(len(k)) + uint64(len(v))
	}
	if totalSize > annotationsMaxSize {
		errs = append(errs, fmt.Errorf("total size of annotations is %d, which exceeds the max size of %d", totalSize, annotationsMaxSize))
	}
	return utilerrors.NewAggregate(errs)
}

// validateKeyID tests the following:
// 1. The keyID is not empty.
// 2. The size of keyID is less than 1 kB.
func validateKeyID(keyID string) error {
	if len(keyID) == 0 {
		return fmt.Errorf("keyID is empty")
	}
	if len(keyID) > keyIDMaxSize {
		return fmt.Errorf("keyID is %d bytes, which exceeds the max size of %d", len(keyID), keyIDMaxSize)
	}
	return nil
}
