/*
Copyright 2023 The Kubernetes Authors.

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

package aes

import (
	"bytes"
	"context"
	"crypto/aes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"time"

	"golang.org/x/crypto/hkdf"

	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/utils/clock"
)

const (
	// cacheTTL is the TTL of KDF cache entries.  We assume that the value.Context.AuthenticatedData
	// for every call is the etcd storage path of the associated resource, and use that as the primary
	// cache key (with a secondary check that confirms that the info matches).  Thus if a client
	// is constantly creating resources with new names (and thus new paths), they will keep adding new
	// entries to the cache for up to this TTL before the GC logic starts deleting old entries.  Each
	// entry is ~300 bytes in size, so even a malicious client will be bounded in the overall memory
	// it can consume.
	cacheTTL = 10 * time.Minute

	derivedKeySizeExtendedNonceGCM = commonSize
	infoSizeExtendedNonceGCM
	MinSeedSizeExtendedNonceGCM
)

// NewHKDFExtendedNonceGCMTransformer is the same as NewGCMTransformer but trades storage,
// memory and CPU to work around the limitations of AES-GCM's 12 byte nonce size.  The input seed
// is assumed to be a cryptographically strong slice of MinSeedSizeExtendedNonceGCM+ random bytes.
// Unlike NewGCMTransformer, this function is immune to the birthday attack because a new key is generated
// per encryption via a key derivation function: KDF(seed, random_bytes) -> key.  The derived key is
// only used once as an AES-GCM key with a random 12 byte nonce.  This avoids any concerns around
// cryptographic wear out (by either number of encryptions or the amount of data being encrypted).
// Speaking on the cryptographic safety, the limit on the number of operations that can be preformed
// with a single seed with derived keys and randomly generated nonces is not practically reachable.
// Thus, the scheme does not impose any specific requirements on the seed rotation schedule.
// Reusing the same seed is safe to do over time and across process restarts.  Whenever a new
// seed is needed, the caller should generate it via GenerateKey(MinSeedSizeExtendedNonceGCM).
// In regard to KMSv2, organization standards or compliance policies around rotation may require
// that the seed be rotated at some interval.  This can be implemented externally by rotating
// the key encryption key via a key ID change.
func NewHKDFExtendedNonceGCMTransformer(seed []byte) (value.Transformer, error) {
	if seedLen := len(seed); seedLen < MinSeedSizeExtendedNonceGCM {
		return nil, fmt.Errorf("invalid seed length %d used for key generation", seedLen)
	}
	return &extendedNonceGCM{
		seed:  seed,
		cache: newSimpleCache(clock.RealClock{}, cacheTTL),
	}, nil
}

type extendedNonceGCM struct {
	seed  []byte
	cache *simpleCache
}

func (e *extendedNonceGCM) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	if len(data) < infoSizeExtendedNonceGCM {
		return nil, false, errors.New("the stored data was shorter than the required size")
	}

	info := data[:infoSizeExtendedNonceGCM]

	transformer, err := e.derivedKeyTransformer(info, dataCtx, false)
	if err != nil {
		return nil, false, fmt.Errorf("failed to derive read key from KDF: %w", err)
	}

	return transformer.TransformFromStorage(ctx, data, dataCtx)
}

func (e *extendedNonceGCM) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	info := make([]byte, infoSizeExtendedNonceGCM)
	if err := randomNonce(info); err != nil {
		return nil, fmt.Errorf("failed to generate info for KDF: %w", err)
	}

	transformer, err := e.derivedKeyTransformer(info, dataCtx, true)
	if err != nil {
		return nil, fmt.Errorf("failed to derive write key from KDF: %w", err)
	}

	return transformer.TransformToStorage(ctx, data, dataCtx)
}

func (e *extendedNonceGCM) derivedKeyTransformer(info []byte, dataCtx value.Context, write bool) (value.Transformer, error) {
	if !write { // no need to check cache on write since we always generate a new transformer
		if transformer := e.cache.get(info, dataCtx); transformer != nil {
			return transformer, nil
		}

		// on read, this is a subslice of a much larger slice and we do not want to hold onto that larger slice
		info = bytes.Clone(info)
	}

	key, err := e.sha256KDFExpandOnly(info)
	if err != nil {
		return nil, fmt.Errorf("failed to KDF expand seed with info: %w", err)
	}

	transformer, err := newGCMTransformerWithInfo(key, info)
	if err != nil {
		return nil, fmt.Errorf("failed to build transformer with KDF derived key: %w", err)
	}

	e.cache.set(dataCtx, transformer)

	return transformer, nil
}

func (e *extendedNonceGCM) sha256KDFExpandOnly(info []byte) ([]byte, error) {
	kdf := hkdf.Expand(sha256.New, e.seed, info)

	derivedKey := make([]byte, derivedKeySizeExtendedNonceGCM)
	if _, err := io.ReadFull(kdf, derivedKey); err != nil {
		return nil, fmt.Errorf("failed to read a derived key from KDF: %w", err)
	}

	return derivedKey, nil
}

func newGCMTransformerWithInfo(key, info []byte) (*transformerWithInfo, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	transformer, err := NewGCMTransformer(block)
	if err != nil {
		return nil, err
	}

	return &transformerWithInfo{transformer: transformer, info: info}, nil
}

type transformerWithInfo struct {
	transformer value.Transformer
	// info are extra opaque bytes prepended to the writes from transformer and stripped from reads.
	// currently info is used to generate a key via KDF(seed, info) -> key
	// and transformer is the output of NewGCMTransformer(aes.NewCipher(key))
	info []byte
}

func (t *transformerWithInfo) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	if !bytes.HasPrefix(data, t.info) {
		return nil, false, errors.New("the stored data is missing the required info prefix")
	}

	return t.transformer.TransformFromStorage(ctx, data[len(t.info):], dataCtx)
}

func (t *transformerWithInfo) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	out, err := t.transformer.TransformToStorage(ctx, data, dataCtx)
	if err != nil {
		return nil, err
	}

	outWithInfo := make([]byte, 0, len(out)+len(t.info))
	outWithInfo = append(outWithInfo, t.info...)
	outWithInfo = append(outWithInfo, out...)

	return outWithInfo, nil
}
