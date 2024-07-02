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
	"crypto/sha256"
	"fmt"
	"sort"
	"time"

	"github.com/gogo/protobuf/proto"
	"go.opentelemetry.io/otel/attribute"
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
	"k8s.io/component-base/tracing"
	"k8s.io/klog/v2"
	kmsservice "k8s.io/kms/pkg/service"
	"k8s.io/utils/clock"
)

func init() {
	value.RegisterMetrics()
	metrics.RegisterMetrics()
}

const (
	// KMSAPIVersionv2 is a version of the KMS API.
	KMSAPIVersionv2 = "v2"
	// KMSAPIVersionv2beta1 is a version of the KMS API.
	KMSAPIVersionv2beta1 = "v2beta1"
	// annotationsMaxSize is the maximum size of the annotations.
	annotationsMaxSize = 32 * 1024 // 32 kB
	// KeyIDMaxSize is the maximum size of the keyID.
	KeyIDMaxSize = 1 * 1024 // 1 kB
	// encryptedDEKSourceMaxSize is the maximum size of the encrypted DEK source.
	encryptedDEKSourceMaxSize = 1 * 1024 // 1 kB
	// cacheTTL is the default time-to-live for the cache entry.
	// this allows the cache to grow to an infinite size for up to a day.
	// there is unlikely to be any meaningful memory impact on the server
	// because the cache will likely never have more than a few thousand entries.
	// each entry can be large due to an internal cache that maps the DEK seed to individual
	// DEK entries, but that cache has an aggressive TTL to keep the size under control.
	// with DEK/seed reuse and no storage migration, the number of entries in this cache
	// would be approximated by unique key IDs used by the KMS plugin
	// combined with the number of server restarts.  If storage migration
	// is performed after key ID changes, and the number of restarts
	// is limited, this cache size may be as small as the number of API
	// servers in use (once old entries expire out from the TTL).
	cacheTTL = 24 * time.Hour
	// key ID related error codes for metrics
	errKeyIDOKCode      ErrCodeKeyID = "ok"
	errKeyIDEmptyCode   ErrCodeKeyID = "empty"
	errKeyIDTooLongCode ErrCodeKeyID = "too_long"
)

// NowFunc is exported so tests can override it.
var NowFunc = time.Now

type StateFunc func() (State, error)
type ErrCodeKeyID string

type State struct {
	Transformer value.Transformer

	EncryptedObject kmstypes.EncryptedObject

	UID string

	ExpirationTimestamp time.Time

	// CacheKey is the key used to cache the DEK/seed in envelopeTransformer.cache.
	CacheKey []byte
}

func (s *State) ValidateEncryptCapability() error {
	if now := NowFunc(); now.After(s.ExpirationTimestamp) {
		return fmt.Errorf("encryptedDEKSource with keyID hash %q expired at %s (current time is %s)",
			GetHashIfNotEmpty(s.EncryptedObject.KeyID), s.ExpirationTimestamp.Format(time.RFC3339), now.Format(time.RFC3339))
	}
	return nil
}

type envelopeTransformer struct {
	envelopeService kmsservice.Service
	providerName    string
	stateFunc       StateFunc

	// cache is a thread-safe expiring lru cache which caches decrypted DEKs indexed by their encrypted form.
	cache       *simpleCache
	apiServerID string
}

// NewEnvelopeTransformer returns a transformer which implements a KEK-DEK based envelope encryption scheme.
// It uses envelopeService to encrypt and decrypt DEKs. Respective DEKs (in encrypted form) are prepended to
// the data items they encrypt.
func NewEnvelopeTransformer(envelopeService kmsservice.Service, providerName string, stateFunc StateFunc, apiServerID string) value.Transformer {
	return newEnvelopeTransformerWithClock(envelopeService, providerName, stateFunc, apiServerID, cacheTTL, clock.RealClock{})
}

func newEnvelopeTransformerWithClock(envelopeService kmsservice.Service, providerName string, stateFunc StateFunc, apiServerID string, cacheTTL time.Duration, clock clock.Clock) value.Transformer {
	return &envelopeTransformer{
		envelopeService: envelopeService,
		providerName:    providerName,
		stateFunc:       stateFunc,
		cache:           newSimpleCache(clock, cacheTTL, providerName),
		apiServerID:     apiServerID,
	}
}

// TransformFromStorage decrypts data encrypted by this transformer using envelope encryption.
func (t *envelopeTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	ctx, span := tracing.Start(ctx, "TransformFromStorage with envelopeTransformer",
		attribute.String("transformer.provider.name", t.providerName),
		// The service.instance_id of the apiserver is already available in the trace
		/*
			{
			"key": "service.instance.id",
			"type": "string",
			"value": "apiserver-zsteyir5lyrtdcmqqmd5kzze6m"
			}
		*/
	)
	defer span.End(500 * time.Millisecond)

	span.AddEvent("About to decode encrypted object")
	// Deserialize the EncryptedObject from the data.
	encryptedObject, err := t.doDecode(data)
	if err != nil {
		span.AddEvent("Decoding encrypted object failed")
		span.RecordError(err)
		return nil, false, err
	}
	span.AddEvent("Decoded encrypted object")

	useSeed := encryptedObject.EncryptedDEKSourceType == kmstypes.EncryptedDEKSourceType_HKDF_SHA256_XNONCE_AES_GCM_SEED

	// TODO: consider marking state.EncryptedDEK != encryptedObject.EncryptedDEK as a stale read to support DEK defragmentation
	//  at a minimum we should have a metric that helps the user understand if DEK fragmentation is high
	state, err := t.stateFunc() // no need to call state.ValidateEncryptCapability on reads
	if err != nil {
		return nil, false, err
	}

	encryptedObjectCacheKey, err := generateCacheKey(encryptedObject.EncryptedDEKSourceType, encryptedObject.EncryptedDEKSource, encryptedObject.KeyID, encryptedObject.Annotations)
	if err != nil {
		return nil, false, err
	}

	// Look up the decrypted DEK from cache first
	transformer := t.cache.get(encryptedObjectCacheKey)

	// fallback to the envelope service if we do not have the transformer locally
	if transformer == nil {
		span.AddEvent("About to decrypt DEK using remote service")
		value.RecordCacheMiss()

		requestInfo := getRequestInfoFromContext(ctx)
		uid := string(uuid.NewUUID())
		klog.V(6).InfoS("decrypting content using envelope service", "uid", uid, "key", string(dataCtx.AuthenticatedData()),
			"group", requestInfo.APIGroup, "version", requestInfo.APIVersion, "resource", requestInfo.Resource, "subresource", requestInfo.Subresource,
			"verb", requestInfo.Verb, "namespace", requestInfo.Namespace, "name", requestInfo.Name)

		key, err := t.envelopeService.Decrypt(ctx, uid, &kmsservice.DecryptRequest{
			Ciphertext:  encryptedObject.EncryptedDEKSource,
			KeyID:       encryptedObject.KeyID,
			Annotations: encryptedObject.Annotations,
		})
		if err != nil {
			span.AddEvent("DEK decryption failed")
			span.RecordError(err)
			return nil, false, fmt.Errorf("failed to decrypt DEK, error: %w", err)
		}
		span.AddEvent("DEK decryption succeeded")

		transformer, err = t.addTransformerForDecryption(encryptedObjectCacheKey, key, useSeed)
		if err != nil {
			return nil, false, err
		}
	}
	metrics.RecordKeyID(metrics.FromStorageLabel, t.providerName, encryptedObject.KeyID, t.apiServerID)

	span.AddEvent("About to decrypt data using DEK")
	out, stale, err := transformer.TransformFromStorage(ctx, encryptedObject.EncryptedData, dataCtx)
	if err != nil {
		span.AddEvent("Data decryption failed")
		span.RecordError(err)
		return nil, false, err
	}

	span.AddEvent("Data decryption succeeded")
	// data is considered stale if the key ID does not match our current write transformer
	return out,
		stale ||
			encryptedObject.KeyID != state.EncryptedObject.KeyID ||
			encryptedObject.EncryptedDEKSourceType != state.EncryptedObject.EncryptedDEKSourceType,
		nil
}

// TransformToStorage encrypts data to be written to disk using envelope encryption.
func (t *envelopeTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	ctx, span := tracing.Start(ctx, "TransformToStorage with envelopeTransformer",
		attribute.String("transformer.provider.name", t.providerName),
		// The service.instance_id of the apiserver is already available in the trace
		/*
			{
			"key": "service.instance.id",
			"type": "string",
			"value": "apiserver-zsteyir5lyrtdcmqqmd5kzze6m"
			}
		*/
	)
	defer span.End(500 * time.Millisecond)

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
	t.cache.set(state.CacheKey, state.Transformer)

	requestInfo := getRequestInfoFromContext(ctx)
	klog.V(6).InfoS("encrypting content using DEK", "uid", state.UID, "key", string(dataCtx.AuthenticatedData()),
		"group", requestInfo.APIGroup, "version", requestInfo.APIVersion, "resource", requestInfo.Resource, "subresource", requestInfo.Subresource,
		"verb", requestInfo.Verb, "namespace", requestInfo.Namespace, "name", requestInfo.Name)

	span.AddEvent("About to encrypt data using DEK")
	result, err := state.Transformer.TransformToStorage(ctx, data, dataCtx)
	if err != nil {
		span.AddEvent("Data encryption failed")
		span.RecordError(err)
		return nil, err
	}
	span.AddEvent("Data encryption succeeded")

	metrics.RecordKeyID(metrics.ToStorageLabel, t.providerName, state.EncryptedObject.KeyID, t.apiServerID)

	encObjectCopy := state.EncryptedObject
	encObjectCopy.EncryptedData = result

	span.AddEvent("About to encode encrypted object")
	// Serialize the EncryptedObject to a byte array.
	out, err := t.doEncode(&encObjectCopy)
	if err != nil {
		span.AddEvent("Encoding encrypted object failed")
		span.RecordError(err)
		return nil, err
	}
	span.AddEvent("Encoded encrypted object")

	return out, nil
}

// addTransformerForDecryption inserts a new transformer to the Envelope cache of DEKs for future reads.
func (t *envelopeTransformer) addTransformerForDecryption(cacheKey []byte, key []byte, useSeed bool) (value.Read, error) {
	var transformer value.Read
	var err error
	if useSeed {
		// the input key is considered safe to use here because it is coming from the KMS plugin / etcd
		transformer, err = aestransformer.NewHKDFExtendedNonceGCMTransformer(key)
	} else {
		var block cipher.Block
		block, err = aes.NewCipher(key)
		if err != nil {
			return nil, err
		}
		// this is compatible with NewGCMTransformerWithUniqueKeyUnsafe for decryption
		// it would use random nonces for encryption but we never do that
		transformer, err = aestransformer.NewGCMTransformer(block)
	}
	if err != nil {
		return nil, err
	}
	t.cache.set(cacheKey, transformer)
	return transformer, nil
}

// doEncode encodes the EncryptedObject to a byte array.
func (t *envelopeTransformer) doEncode(request *kmstypes.EncryptedObject) ([]byte, error) {
	if err := ValidateEncryptedObject(request); err != nil {
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
	if err := ValidateEncryptedObject(o); err != nil {
		return nil, err
	}

	return o, nil
}

// GenerateTransformer generates a new transformer and encrypts the DEK/seed using the envelope service.
// It returns the transformer, the encrypted DEK/seed, cache key and error.
func GenerateTransformer(ctx context.Context, uid string, envelopeService kmsservice.Service, useSeed bool) (value.Transformer, *kmstypes.EncryptedObject, []byte, error) {
	newTransformerFunc := func() (value.Transformer, []byte, error) {
		seed, err := aestransformer.GenerateKey(aestransformer.MinSeedSizeExtendedNonceGCM)
		if err != nil {
			return nil, nil, err
		}
		transformer, err := aestransformer.NewHKDFExtendedNonceGCMTransformer(seed)
		if err != nil {
			return nil, nil, err
		}
		return transformer, seed, nil
	}
	if !useSeed {
		newTransformerFunc = aestransformer.NewGCMTransformerWithUniqueKeyUnsafe
	}
	transformer, newKey, err := newTransformerFunc()
	if err != nil {
		return nil, nil, nil, err
	}

	klog.V(6).InfoS("encrypting content using envelope service", "uid", uid)

	resp, err := envelopeService.Encrypt(ctx, uid, newKey)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to encrypt DEK, error: %w", err)
	}

	o := &kmstypes.EncryptedObject{
		KeyID:              resp.KeyID,
		EncryptedDEKSource: resp.Ciphertext,
		EncryptedData:      []byte{0}, // any non-empty value to pass validation
		Annotations:        resp.Annotations,
	}

	if useSeed {
		o.EncryptedDEKSourceType = kmstypes.EncryptedDEKSourceType_HKDF_SHA256_XNONCE_AES_GCM_SEED
	} else {
		o.EncryptedDEKSourceType = kmstypes.EncryptedDEKSourceType_AES_GCM_KEY
	}

	if err := ValidateEncryptedObject(o); err != nil {
		return nil, nil, nil, err
	}

	cacheKey, err := generateCacheKey(o.EncryptedDEKSourceType, resp.Ciphertext, resp.KeyID, resp.Annotations)
	if err != nil {
		return nil, nil, nil, err
	}

	o.EncryptedData = nil // make sure that later code that uses this encrypted object sets this field

	return transformer, o, cacheKey, nil
}

func ValidateEncryptedObject(o *kmstypes.EncryptedObject) error {
	if o == nil {
		return fmt.Errorf("encrypted object is nil")
	}
	switch t := o.EncryptedDEKSourceType; t {
	case kmstypes.EncryptedDEKSourceType_AES_GCM_KEY:
	case kmstypes.EncryptedDEKSourceType_HKDF_SHA256_XNONCE_AES_GCM_SEED:
	default:
		return fmt.Errorf("unknown encryptedDEKSourceType: %d", t)
	}
	if len(o.EncryptedData) == 0 {
		return fmt.Errorf("encrypted data is empty")
	}
	if err := validateEncryptedDEKSource(o.EncryptedDEKSource); err != nil {
		return fmt.Errorf("failed to validate encrypted DEK source: %w", err)
	}
	if _, err := ValidateKeyID(o.KeyID); err != nil {
		return fmt.Errorf("failed to validate key id: %w", err)
	}
	if err := validateAnnotations(o.Annotations); err != nil {
		return fmt.Errorf("failed to validate annotations: %w", err)
	}
	return nil
}

// validateEncryptedDEKSource tests the following:
// 1. The encrypted DEK source is not empty.
// 2. The size of encrypted DEK source is less than 1 kB.
func validateEncryptedDEKSource(encryptedDEKSource []byte) error {
	if len(encryptedDEKSource) == 0 {
		return fmt.Errorf("encrypted DEK source is empty")
	}
	if len(encryptedDEKSource) > encryptedDEKSourceMaxSize {
		return fmt.Errorf("encrypted DEK source is %d bytes, which exceeds the max size of %d", len(encryptedDEKSource), encryptedDEKSourceMaxSize)
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
//  0. encryptedDEKSourceType
//  1. encryptedDEKSource
//  2. keyID
//  3. length of annotations
//  4. annotations (sorted by key) - each annotation is a concatenation of:
//     a. annotation key
//     b. annotation value
func generateCacheKey(encryptedDEKSourceType kmstypes.EncryptedDEKSourceType, encryptedDEKSource []byte, keyID string, annotations map[string][]byte) ([]byte, error) {
	// TODO(aramase): use sync pool buffer to avoid allocations
	b := cryptobyte.NewBuilder(nil)
	b.AddUint32(uint32(encryptedDEKSourceType))
	b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
		b.AddBytes(encryptedDEKSource)
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
	//As of go 1.22, string to bytes conversion []bytes(str) is faster than using the unsafe package.
	return []byte(s)
}

// GetHashIfNotEmpty returns the sha256 hash of the data if it is not empty.
func GetHashIfNotEmpty(data string) string {
	if len(data) > 0 {
		return fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(data)))
	}
	return ""
}
