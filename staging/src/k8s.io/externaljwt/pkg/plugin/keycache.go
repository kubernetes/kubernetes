/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"sync"
	"time"

	"golang.org/x/sync/singleflight"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

type KeyCache struct {
	plugin *Plugin

	syncGroup singleflight.Group

	publicKeysLock sync.RWMutex
	publicKeys     *VerificationKeys

	listenersLock sync.Mutex
	listeners     []serviceaccount.Listener

	ticker             *time.Ticker
	refreshIntervalSec int
}

// NewKeyCache constructs an implementation of KeyCache that wraps a GKE
// external JWT signer plugin.
func NewKeyCache(plugin *Plugin) *KeyCache {
	cache := &KeyCache{
		plugin:             plugin,
		refreshIntervalSec: 600, // Default refresh interval is 10 mins
	}
	return cache
}

// InitialFill can be used to performs an initial fetch for keys get the
// refresh interval as recommended by external signer.
func (p *KeyCache) InitialFill(ctx context.Context) error {
	if err := p.syncKeys(ctx); err != nil {
		return fmt.Errorf("while performing initial cache fill: %w", err)
	}
	return nil
}

func (p *KeyCache) StartPeriodicSync(ctx context.Context) {
	p.ticker = time.NewTicker(time.Second * time.Duration(p.refreshIntervalSec))
	defer p.ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			klog.InfoS("Key cache shutting down")
			return
		case <-p.ticker.C:
		}

		if err := p.syncKeys(ctx); err != nil {
			klog.Errorf("when syncing supported public keys(Stale set of keys will be supported): %v", err)
			continue
		}

		p.broadcastUpdate()
	}
}

func (p *KeyCache) AddListener(listener serviceaccount.Listener) {
	p.listenersLock.Lock()
	defer p.listenersLock.Unlock()

	p.listeners = append(p.listeners, listener)
}

func (p *KeyCache) GetCacheAgeMaxSeconds() int {
	// consumer shall support the keys for no longer than 10 mins before refreshing.
	return p.refreshIntervalSec
}

// GetPublicKeys returns the public key corresponding to requested keyID.
// Getter is expected to return All keys for keyID ""
func (p *KeyCache) GetPublicKeys(keyID string) []serviceaccount.PublicKey {
	ctx := context.Background()

	pubKeys, ok := p.findKeyForKeyID(keyID)
	if ok {
		return pubKeys
	}

	// If we didn't find it, trigger a sync.
	if err := p.syncKeys(ctx); err != nil {
		klog.ErrorS(err, "Error while syncing keys")
	}

	pubKeys, ok = p.findKeyForKeyID(keyID)
	if ok {
		return pubKeys
	}

	// If we still didn't find it, then it's an unknown keyID.
	klog.ErrorS(nil, "Key id not found after refresh", "keyID", keyID)
	return []serviceaccount.PublicKey{}
}

func (p *KeyCache) findKeyForKeyID(keyID string) ([]serviceaccount.PublicKey, bool) {
	p.publicKeysLock.RLock()
	defer p.publicKeysLock.RUnlock()

	if len(p.publicKeys.Keys) == 0 {
		klog.ErrorS(nil, "No keys currently in cache.  Initial fill has not completed")
		return nil, false
	}

	if keyID == "" {
		return p.publicKeys.Keys, true
	}

	for _, key := range p.publicKeys.Keys {
		if key.KeyID == keyID {
			return []serviceaccount.PublicKey{key}, true
		}
	}

	return nil, false
}

// sync supported external keys.
// completely re-writes the set of supported keys.
func (p *KeyCache) syncKeys(ctx context.Context) error {
	_, err, _ := p.syncGroup.Do("", func() (any, error) {
		newPublicKeys, refreshHint, err := p.plugin.GetTokenVerificationKeys(ctx)
		if err != nil {
			return nil, fmt.Errorf("while fetching token verification keys: %w", err)
		}

		p.publicKeysLock.Lock()
		defer p.publicKeysLock.Unlock()

		p.publicKeys = newPublicKeys
		p.updateRefreshInterval(refreshHint)

		return newPublicKeys, nil
	})
	if err != nil {
		return fmt.Errorf("in singleflight: %w", err)
	}

	return nil
}

func (p *KeyCache) metrics() (float64, []string) {
	p.publicKeysLock.RLock()
	defer p.publicKeysLock.RUnlock()

	if p.publicKeys == nil {
		return -1.0, nil
	}

	keyIDs := make([]string, len(p.publicKeys.Keys))
	for i := 0; i < len(p.publicKeys.Keys); i++ {
		keyIDs[i] = p.publicKeys.Keys[i].KeyID
	}
	return time.Since(p.publicKeys.DataTimestamp).Seconds(), keyIDs
}

func (p *KeyCache) broadcastUpdate() {
	p.listenersLock.Lock()
	defer p.listenersLock.Unlock()

	for _, l := range p.listeners {
		l.Enqueue()
	}
}

func (p *KeyCache) updateRefreshInterval(refreshHint int) {
	if refreshHint > 0 && p.refreshIntervalSec != refreshHint {
		p.refreshIntervalSec = refreshHint
		if p.ticker != nil {
			p.ticker.Reset(time.Second * time.Duration(p.refreshIntervalSec))
		}
	}
}
