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
	"crypto/rand"
	"crypto/subtle"
	"fmt"
	"time"

	"github.com/gogo/protobuf/proto"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	kmstypes "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2/v2alpha1"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/metrics"
	"k8s.io/klog/v2"
	kmsservice "k8s.io/kms/pkg/service"
	"k8s.io/utils/clock"
)

func init() {
	value.RegisterMetrics()
	metrics.RegisterMetrics()
}

const (
	// KMSAPIVersion is the version of the KMS API.
	KMSAPIVersion = "v2alpha1"
	// annotationsMaxSize is the maximum size of the annotations.
	annotationsMaxSize = 32 * 1024 // 32 kB
	// KeyIDMaxSize is the maximum size of the keyID.
	KeyIDMaxSize = 1 * 1024 // 1 kB
	// encryptedDEKMaxSize is the maximum size of the encrypted DEK.
	encryptedDEKMaxSize = 1 * 1024 // 1 kB
	// cacheTTL is the default time-to-live for the cache entry.
	cacheTTL = 1 * time.Hour
	// error code
	errKeyIDOKCode      ErrCodeKeyID = "ok"
	errKeyIDEmptyCode   ErrCodeKeyID = "empty"
	errKeyIDTooLongCode ErrCodeKeyID = "too_long"
)

// ValidateEncryptCapabilityNowFunc is exported so integration tests can override it.
var ValidateEncryptCapabilityNowFunc = time.Now

type StateFunc func() (State, error)
type ErrCodeKeyID string

type State struct {
	Transformer  value.Transformer
	EncryptedDEK []byte
	KeyID        string
	Annotations  map[string][]byte

	UID string

	ExpirationTimestamp time.Time
}

func (s *State) ValidateEncryptCapability() error {
	if now := ValidateEncryptCapabilityNowFunc(); now.After(s.ExpirationTimestamp) {
		return fmt.Errorf("EDEK with keyID %q expired at %s (current time is %s)",
			s.KeyID, s.ExpirationTimestamp.Format(time.RFC3339), now.Format(time.RFC3339))
	}
	return nil
}

type envelopeTransformer struct {
	envelopeService kmsservice.Service
	providerName    string
	stateFunc       StateFunc

	// cache is a thread-safe expiring lru cache which caches decrypted DEKs indexed by their encrypted form.
	cache *simpleCache
}

// NewEnvelopeTransformer returns a transformer which implements a KEK-DEK based envelope encryption scheme.
// It uses envelopeService to encrypt and decrypt DEKs. Respective DEKs (in encrypted form) are prepended to
// the data items they encrypt.
func NewEnvelopeTransformer(envelopeService kmsservice.Service, providerName string, stateFunc StateFunc) value.Transformer {
	return newEnvelopeTransformerWithClock(envelopeService, providerName, stateFunc, cacheTTL, clock.RealClock{})
}

func newEnvelopeTransformerWithClock(envelopeService kmsservice.Service, providerName string, stateFunc StateFunc, cacheTTL time.Duration, clock clock.Clock) value.Transformer {
	return &envelopeTransformer{
		envelopeService: envelopeService,
		providerName:    providerName,
		stateFunc:       stateFunc,
		cache:           newSimpleCache(clock, cacheTTL),
	}
}

// TransformFromStorage decrypts data encrypted by this transformer using envelope encryption.
func (t *envelopeTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	// Deserialize the EncryptedObject from the data.
	encryptedObject, err := t.doDecode(data)
	if err != nil {
		return nil, false, err
	}

	state, err := t.stateFunc() // no need to call state.ValidateEncryptCapability on reads
	if err != nil {
		return nil, false, err
	}

	// start with the assumption that the current write transformer is also the read transformer
	transformer := state.Transformer

	// if the current write transformer is not what was used to encrypt this data, check in the cache
	if subtle.ConstantTimeCompare(state.EncryptedDEK, encryptedObject.EncryptedDEK) != 1 {
		// TODO value.RecordStateMiss() metric
		transformer = t.cache.get(encryptedObject.EncryptedDEK)
	}

	// fallback to the envelope service if we do not have the transformer locally
	if transformer == nil {
		value.RecordCacheMiss()

		requestInfo := getRequestInfoFromContext(ctx)
		uid := string(uuid.NewUUID())
		klog.V(6).InfoS("decrypting content using envelope service", "uid", uid, "key", string(dataCtx.AuthenticatedData()),
			"group", requestInfo.APIGroup, "version", requestInfo.APIVersion, "resource", requestInfo.Resource, "subresource", requestInfo.Subresource,
			"verb", requestInfo.Verb, "namespace", requestInfo.Namespace, "name", requestInfo.Name)

		key, err := t.envelopeService.Decrypt(ctx, uid, &kmsservice.DecryptRequest{
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
	metrics.RecordKeyID(metrics.FromStorageLabel, t.providerName, encryptedObject.KeyID)

	out, stale, err := transformer.TransformFromStorage(ctx, encryptedObject.EncryptedData, dataCtx)
	if err != nil {
		return nil, false, err
	}

	// data is considered stale if the key ID does not match our current write transformer
	return out, stale || encryptedObject.KeyID != state.KeyID, nil

}

// TransformToStorage encrypts data to be written to disk using envelope encryption.
func (t *envelopeTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	state, err := t.stateFunc()
	if err != nil {
		return nil, err
	}
	if err := state.ValidateEncryptCapability(); err != nil {
		return nil, err
	}

	// this prevents a cache miss every time the DEK rotates
	// TODO see if we can do this inside the stateFunc control loop
	t.cache.set(state.EncryptedDEK, state.Transformer)

	requestInfo := getRequestInfoFromContext(ctx)
	klog.V(6).InfoS("encrypting content using cached DEK", "uid", state.UID, "key", string(dataCtx.AuthenticatedData()),
		"group", requestInfo.APIGroup, "version", requestInfo.APIVersion, "resource", requestInfo.Resource, "subresource", requestInfo.Subresource,
		"verb", requestInfo.Verb, "namespace", requestInfo.Namespace, "name", requestInfo.Name)

	result, err := state.Transformer.TransformToStorage(ctx, data, dataCtx)
	if err != nil {
		return nil, err
	}

	metrics.RecordKeyID(metrics.ToStorageLabel, t.providerName, state.KeyID)

	encObject := &kmstypes.EncryptedObject{
		KeyID:         state.KeyID,
		EncryptedDEK:  state.EncryptedDEK,
		EncryptedData: result,
		Annotations:   state.Annotations,
	}

	// Serialize the EncryptedObject to a byte array.
	return t.doEncode(encObject)
}

// addTransformer inserts a new transformer to the Envelope cache of DEKs for future reads.
func (t *envelopeTransformer) addTransformer(encKey []byte, key []byte) (value.Transformer, error) {
	transformer, err := generateAESTransformer(key)
	if err != nil {
		return nil, err
	}
	// TODO(aramase): Add metrics for cache fill percentage with custom cache implementation.
	t.cache.set(encKey, transformer)
	return transformer, nil
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

func GenerateTransformer(ctx context.Context, uid string, envelopeService kmsservice.Service) (value.Transformer, *kmsservice.EncryptResponse, error) {
	newKey, err := generateKey(32)
	if err != nil {
		return nil, nil, err
	}

	klog.V(6).InfoS("encrypting content using envelope service", "uid", uid)

	resp, err := envelopeService.Encrypt(ctx, uid, newKey)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to encrypt DEK, error: %w", err)
	}

	transformer, err := generateAESTransformer(newKey)
	if err != nil {
		return nil, nil, err
	}

	return transformer, resp, nil
}

func generateAESTransformer(key []byte) (value.Transformer, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	return aestransformer.NewGCMTransformer(block), nil
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
	if _, err := ValidateKeyID(o.KeyID); err != nil {
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

// ValidateKeyID tests the following:
// 1. The keyID is not empty.
// 2. The size of keyID is less than 1 kB.
func ValidateKeyID(keyID string) (ErrCodeKeyID, error) {
	if len(keyID) == 0 {
		return errKeyIDEmptyCode, fmt.Errorf("keyID is empty")
	}
	if len(keyID) > KeyIDMaxSize {
		return errKeyIDTooLongCode, fmt.Errorf("keyID is %d bytes, which exceeds the max size of %d", len(keyID), KeyIDMaxSize)
	}
	return errKeyIDOKCode, nil
}

func getRequestInfoFromContext(ctx context.Context) *genericapirequest.RequestInfo {
	if reqInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		return reqInfo
	}
	return &genericapirequest.RequestInfo{}
}
