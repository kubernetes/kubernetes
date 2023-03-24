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
	"fmt"
	"sort"
	"time"
	"unsafe"

	"github.com/gogo/protobuf/proto"
	"golang.org/x/crypto/cryptobyte"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	kmstypes "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/kmsv2/v2"
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
	KMSAPIVersion = "v2beta1"
	// annotationsMaxSize is the maximum size of the annotations.
	annotationsMaxSize = 32 * 1024 // 32 kB
	// KeyIDMaxSize is the maximum size of the keyID.
	KeyIDMaxSize = 1 * 1024 // 1 kB
	// encryptedDEKMaxSize is the maximum size of the encrypted DEK.
	encryptedDEKMaxSize = 1 * 1024 // 1 kB
	// cacheTTL is the default time-to-live for the cache entry.
	// this allows the cache to grow to an infinite size for up to a day.
	// this is meant as a temporary solution until the cache is re-written to not have a TTL.
	// there is unlikely to be any meaningful memory impact on the server
	// because the cache will likely never have more than a few thousand entries
	// and each entry is roughly ~200 bytes in size.  with DEK reuse
	// and no storage migration, the number of entries in this cache
	// would be approximated by unique key IDs used by the KMS plugin
	// combined with the number of server restarts.  If storage migration
	// is performed after key ID changes, and the number of restarts
	// is limited, this cache size may be as small as the number of API
	// servers in use (once old entries expire out from the TTL).
	cacheTTL = 24 * time.Hour
	// error code
	errKeyIDOKCode      ErrCodeKeyID = "ok"
	errKeyIDEmptyCode   ErrCodeKeyID = "empty"
	errKeyIDTooLongCode ErrCodeKeyID = "too_long"
)

// NowFunc is exported so tests can override it.
var NowFunc = time.Now

type StateFunc func() (State, error)
type ErrCodeKeyID string

type State struct {
	Transformer  value.Transformer
	EncryptedDEK []byte
	KeyID        string
	Annotations  map[string][]byte

	UID string

	ExpirationTimestamp time.Time

	// CacheKey is the key used to cache the DEK in transformer.cache.
	CacheKey []byte
}

func (s *State) ValidateEncryptCapability() error {
	if now := NowFunc(); now.After(s.ExpirationTimestamp) {
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

	// TODO: consider marking state.EncryptedDEK != encryptedObject.EncryptedDEK as a stale read to support DEK defragmentation
	//  at a minimum we should have a metric that helps the user understand if DEK fragmentation is high
	state, err := t.stateFunc() // no need to call state.ValidateEncryptCapability on reads
	if err != nil {
		return nil, false, err
	}

	encryptedObjectCacheKey, err := generateCacheKey(encryptedObject.EncryptedDEK, encryptedObject.KeyID, encryptedObject.Annotations)
	if err != nil {
		return nil, false, err
	}

	// Look up the decrypted DEK from cache first
	transformer := t.cache.get(encryptedObjectCacheKey)

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

		transformer, err = t.addTransformerForDecryption(encryptedObjectCacheKey, key)
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
	// this has the side benefit of causing the cache to perform a GC
	// TODO see if we can do this inside the stateFunc control loop
	// TODO(aramase): Add metrics for cache fill percentage with custom cache implementation.
	t.cache.set(state.CacheKey, state.Transformer)

	requestInfo := getRequestInfoFromContext(ctx)
	klog.V(6).InfoS("encrypting content using DEK", "uid", state.UID, "key", string(dataCtx.AuthenticatedData()),
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

// addTransformerForDecryption inserts a new transformer to the Envelope cache of DEKs for future reads.
func (t *envelopeTransformer) addTransformerForDecryption(cacheKey []byte, key []byte) (decryptTransformer, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	// this is compatible with NewGCMTransformerWithUniqueKeyUnsafe for decryption
	// it would use random nonces for encryption but we never do that
	transformer, err := aestransformer.NewGCMTransformer(block)
	if err != nil {
		return nil, err
	}
	// TODO(aramase): Add metrics for cache fill percentage with custom cache implementation.
	t.cache.set(cacheKey, transformer)
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

func GenerateTransformer(ctx context.Context, uid string, envelopeService kmsservice.Service) (value.Transformer, *kmsservice.EncryptResponse, []byte, error) {
	transformer, newKey, err := aestransformer.NewGCMTransformerWithUniqueKeyUnsafe()
	if err != nil {
		return nil, nil, nil, err
	}

	klog.V(6).InfoS("encrypting content using envelope service", "uid", uid)

	resp, err := envelopeService.Encrypt(ctx, uid, newKey)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to encrypt DEK, error: %w", err)
	}

	if err := validateEncryptedObject(&kmstypes.EncryptedObject{
		KeyID:         resp.KeyID,
		EncryptedDEK:  resp.Ciphertext,
		EncryptedData: []byte{0}, // any non-empty value to pass validation
		Annotations:   resp.Annotations,
	}); err != nil {
		return nil, nil, nil, err
	}

	cacheKey, err := generateCacheKey(resp.Ciphertext, resp.KeyID, resp.Annotations)
	if err != nil {
		return nil, nil, nil, err
	}

	return transformer, resp, cacheKey, nil
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

// generateCacheKey returns a key for the cache.
// The key is a concatenation of:
//  1. encryptedDEK
//  2. keyID
//  3. length of annotations
//  4. annotations (sorted by key) - each annotation is a concatenation of:
//     a. annotation key
//     b. annotation value
func generateCacheKey(encryptedDEK []byte, keyID string, annotations map[string][]byte) ([]byte, error) {
	// TODO(aramase): use sync pool buffer to avoid allocations
	b := cryptobyte.NewBuilder(nil)
	b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes(encryptedDEK)
	})
	b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes(toBytes(keyID))
	})
	if len(annotations) == 0 {
		return b.Bytes()
	}

	// add the length of annotations to the cache key
	b.AddUint32(uint32(len(annotations)))

	// Sort the annotations by key.
	keys := make([]string, 0, len(annotations))
	for k := range annotations {
		k := k
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		// The maximum size of annotations is annotationsMaxSize (32 kB) so we can safely
		// assume that the length of the key and value will fit in a uint16.
		b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
			b.AddBytes(toBytes(k))
		})
		b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
			b.AddBytes(annotations[k])
		})
	}

	return b.Bytes()
}

// toBytes performs unholy acts to avoid allocations
func toBytes(s string) []byte {
	// unsafe.StringData is unspecified for the empty string, so we provide a strict interpretation
	if len(s) == 0 {
		return nil
	}
	// Copied from go 1.20.1 os.File.WriteString
	// https://github.com/golang/go/blob/202a1a57064127c3f19d96df57b9f9586145e21c/src/os/file.go#L246
	return unsafe.Slice(unsafe.StringData(s), len(s))
}
