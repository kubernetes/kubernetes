/*
Copyright 2024 The Kubernetes Authors.

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

package plugin

import (
	"context"
	"crypto/x509"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/singleflight"

	externaljwtv1 "k8s.io/externaljwt/apis/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/serviceaccount"
	externaljwtmetrics "k8s.io/kubernetes/pkg/serviceaccount/externaljwt/metrics"
)

const fallbackRefreshDuration = 10 * time.Second

type keyCache struct {
	client externaljwtv1.ExternalJWTSignerClient

	syncGroup     singleflight.Group
	listenersLock sync.Mutex
	listeners     []serviceaccount.Listener

	verificationKeys atomic.Pointer[VerificationKeys]
}

// newKeyCache constructs an implementation of KeyCache.
func newKeyCache(client externaljwtv1.ExternalJWTSignerClient) *keyCache {
	cache := &keyCache{
		client: client,
	}
	cache.verificationKeys.Store(&VerificationKeys{})
	return cache
}

// InitialFill can be used to perform an initial fetch for keys get the
// refresh interval as recommended by external signer.
func (p *keyCache) initialFill(ctx context.Context) error {
	if err := p.syncKeys(ctx); err != nil {
		return fmt.Errorf("while performing initial cache fill: %w", err)
	}
	return nil
}

func (p *keyCache) scheduleSync(ctx context.Context, keySyncTimeout time.Duration) {
	timer := time.NewTimer(p.verificationKeys.Load().NextRefreshHint.Sub(time.Now()))
	defer timer.Stop()

	for {
		select {
		case <-ctx.Done():
			klog.InfoS("Key cache shutting down")
			return
		case <-timer.C:
		}

		timedCtx, cancel := context.WithTimeout(ctx, keySyncTimeout)
		if err := p.syncKeys(timedCtx); err != nil {
			klog.Errorf("when syncing supported public keys(Stale set of keys will be supported): %v", err)
			timer.Reset(fallbackRefreshDuration)
		} else {
			timer.Reset(p.verificationKeys.Load().NextRefreshHint.Sub(time.Now()))
		}
		cancel()
	}
}

func (p *keyCache) AddListener(listener serviceaccount.Listener) {
	p.listenersLock.Lock()
	defer p.listenersLock.Unlock()

	p.listeners = append(p.listeners, listener)
}

func (p *keyCache) GetCacheAgeMaxSeconds() int {
	val := int(p.verificationKeys.Load().NextRefreshHint.Sub(time.Now()).Seconds())
	if val < 0 {
		return 0
	}
	return val
}

// GetPublicKeys returns the public key corresponding to requested keyID.
// Getter is expected to return All keys for keyID ""
func (p *keyCache) GetPublicKeys(ctx context.Context, keyID string) []serviceaccount.PublicKey {
	pubKeys, ok := p.findKeyForKeyID(keyID)
	if ok {
		return pubKeys
	}

	// If we didn't find it, trigger a sync.
	if err := p.syncKeys(ctx); err != nil {
		klog.ErrorS(err, "Error while syncing keys")
		return []serviceaccount.PublicKey{}
	}

	pubKeys, ok = p.findKeyForKeyID(keyID)
	if ok {
		return pubKeys
	}

	// If we still didn't find it, then it's an unknown keyID.
	klog.Errorf("Key id %q not found after refresh", keyID)
	return []serviceaccount.PublicKey{}
}

func (p *keyCache) findKeyForKeyID(keyID string) ([]serviceaccount.PublicKey, bool) {
	if len(p.verificationKeys.Load().Keys) == 0 {
		klog.Error("No keys currently in cache. Initial fill has not completed")
		return nil, false
	}

	if keyID == "" {
		return p.verificationKeys.Load().Keys, true
	}

	keysToReturn := []serviceaccount.PublicKey{}
	for _, key := range p.verificationKeys.Load().Keys {
		if key.KeyID == keyID {
			keysToReturn = append(keysToReturn, key)
		}
	}

	return keysToReturn, len(keysToReturn) > 0
}

// sync supported external keys.
// completely re-writes the set of supported keys.
func (p *keyCache) syncKeys(ctx context.Context) error {
	_, err, _ := p.syncGroup.Do("", func() (any, error) {
		oldPublicKeys := p.verificationKeys.Load()
		newPublicKeys, err := p.getTokenVerificationKeys(ctx)
		externaljwtmetrics.RecordFetchKeysAttempt(err)
		if err != nil {
			return nil, fmt.Errorf("while fetching token verification keys: %w", err)
		}

		p.verificationKeys.Store(newPublicKeys)
		externaljwtmetrics.RecordKeyDataTimeStamp(float64(newPublicKeys.DataTimestamp.UnixNano()) / float64(1000000000))

		if keysChanged(oldPublicKeys, newPublicKeys) {
			p.broadcastUpdate()
		}

		return nil, nil
	})
	return err
}

// keysChanged returns true if the data timestamp, key count, order of key ids or excludeFromOIDCDiscovery indicators
func keysChanged(oldPublicKeys, newPublicKeys *VerificationKeys) bool {
	// If the timestamp changed, we changed
	if !oldPublicKeys.DataTimestamp.Equal(newPublicKeys.DataTimestamp) {
		return true
	}
	// Avoid deepequal checks on key content itself.
	// If the number of keys changed, we changed
	if len(oldPublicKeys.Keys) != len(newPublicKeys.Keys) {
		return true
	}
	// If the order, key id, or oidc discovery flag changed, we changed.
	for i := range oldPublicKeys.Keys {
		if oldPublicKeys.Keys[i].KeyID != newPublicKeys.Keys[i].KeyID {
			return true
		}
		if oldPublicKeys.Keys[i].ExcludeFromOIDCDiscovery != newPublicKeys.Keys[i].ExcludeFromOIDCDiscovery {
			return true
		}
	}
	return false
}

func (p *keyCache) broadcastUpdate() {
	p.listenersLock.Lock()
	defer p.listenersLock.Unlock()

	for _, l := range p.listeners {
		// don't block on a slow listener
		go l.Enqueue()
	}
}

// GetTokenVerificationKeys returns a map of supported external keyIDs to keys
// the keys are PKIX-serialized. It calls external-jwt-signer with a timeout of keySyncTimeoutSec.
func (p *keyCache) getTokenVerificationKeys(ctx context.Context) (*VerificationKeys, error) {
	req := &externaljwtv1.FetchKeysRequest{}
	resp, err := p.client.FetchKeys(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("while getting externally supported jwt signing keys: %w", err)
	}
	// Validate the refresh hint.
	if resp.RefreshHintSeconds <= 0 {
		return nil, fmt.Errorf("found invalid refresh hint (%ds)", resp.RefreshHintSeconds)
	}

	if len(resp.Keys) == 0 {
		return nil, fmt.Errorf("found no keys")
	}
	if err := resp.DataTimestamp.CheckValid(); err != nil {
		return nil, fmt.Errorf("invalid data timestamp: %w", err)
	}
	keys := make([]serviceaccount.PublicKey, 0, len(resp.Keys))
	for _, protoKey := range resp.Keys {
		if protoKey == nil {
			return nil, fmt.Errorf("found nil public key")
		}
		if len(protoKey.KeyId) == 0 || len(protoKey.KeyId) > 1024 {
			return nil, fmt.Errorf("found invalid public key id %q", protoKey.KeyId)
		}
		if len(protoKey.Key) == 0 {
			return nil, fmt.Errorf("found empty public key")
		}
		parsedPublicKey, err := x509.ParsePKIXPublicKey(protoKey.Key)
		if err != nil {
			return nil, fmt.Errorf("while parsing external public keys: %w", err)
		}

		keys = append(keys, serviceaccount.PublicKey{
			KeyID:                    protoKey.KeyId,
			PublicKey:                parsedPublicKey,
			ExcludeFromOIDCDiscovery: protoKey.ExcludeFromOidcDiscovery,
		})
	}

	vk := &VerificationKeys{
		Keys:            keys,
		DataTimestamp:   resp.DataTimestamp.AsTime(),
		NextRefreshHint: time.Now().Add(time.Duration(resp.RefreshHintSeconds) * time.Second),
	}

	return vk, nil
}
