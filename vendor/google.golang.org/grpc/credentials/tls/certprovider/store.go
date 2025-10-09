/*
 *
 * Copyright 2020 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package certprovider

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
)

// provStore is the global singleton certificate provider store.
var provStore = &store{
	providers: make(map[storeKey]*wrappedProvider),
}

// storeKey acts as the key to the map of providers maintained by the store. A
// combination of provider name and configuration is used to uniquely identify
// every provider instance in the store. Go maps need to be indexed by
// comparable types, so the provider configuration is converted from `any` to
// `string` using the ParseConfig method while creating this key.
type storeKey struct {
	// name of the certificate provider.
	name string
	// configuration of the certificate provider in string form.
	config string
	// opts contains the certificate name and other keyMaterial options.
	opts BuildOptions
}

// wrappedProvider wraps a provider instance with a reference count.
type wrappedProvider struct {
	Provider
	refCount int

	// A reference to the key and store are also kept here to override the
	// Close method on the provider.
	storeKey storeKey
	store    *store
}

// closedProvider always returns errProviderClosed error.
type closedProvider struct{}

func (c closedProvider) KeyMaterial(context.Context) (*KeyMaterial, error) {
	return nil, errProviderClosed
}

func (c closedProvider) Close() {
}

// singleCloseWrappedProvider wraps a provider instance with a reference count
// to properly handle multiple calls to Close.
type singleCloseWrappedProvider struct {
	provider atomic.Pointer[Provider]
}

// store is a collection of provider instances, safe for concurrent access.
type store struct {
	mu        sync.Mutex
	providers map[storeKey]*wrappedProvider
}

// Close overrides the Close method of the embedded provider. It releases the
// reference held by the caller on the underlying provider and if the
// provider's reference count reaches zero, it is removed from the store, and
// its Close method is also invoked.
func (wp *wrappedProvider) Close() {
	ps := wp.store
	ps.mu.Lock()
	defer ps.mu.Unlock()

	wp.refCount--
	if wp.refCount == 0 {
		wp.Provider.Close()
		delete(ps.providers, wp.storeKey)
	}
}

// Close overrides the Close method of the embedded provider to avoid release the
// already released reference.
func (w *singleCloseWrappedProvider) Close() {
	newProvider := Provider(closedProvider{})
	oldProvider := w.provider.Swap(&newProvider)
	(*oldProvider).Close()
}

// KeyMaterial returns the key material sourced by the Provider.
// Callers are expected to use the returned value as read-only.
func (w *singleCloseWrappedProvider) KeyMaterial(ctx context.Context) (*KeyMaterial, error) {
	return (*w.provider.Load()).KeyMaterial(ctx)
}

// newSingleCloseWrappedProvider create wrapper a provider instance with a reference count
// to properly handle multiple calls to Close.
func newSingleCloseWrappedProvider(provider Provider) *singleCloseWrappedProvider {
	w := &singleCloseWrappedProvider{}
	w.provider.Store(&provider)
	return w
}

// BuildableConfig wraps parsed provider configuration and functionality to
// instantiate provider instances.
type BuildableConfig struct {
	name    string
	config  []byte
	starter func(BuildOptions) Provider
	pStore  *store
}

// NewBuildableConfig creates a new BuildableConfig with the given arguments.
// Provider implementations are expected to invoke this function after parsing
// the given configuration as part of their ParseConfig() method.
// Equivalent configurations are expected to invoke this function with the same
// config argument.
func NewBuildableConfig(name string, config []byte, starter func(BuildOptions) Provider) *BuildableConfig {
	return &BuildableConfig{
		name:    name,
		config:  config,
		starter: starter,
		pStore:  provStore,
	}
}

// Build kicks off a provider instance with the wrapped configuration. Multiple
// invocations of this method with the same opts will result in provider
// instances being reused.
func (bc *BuildableConfig) Build(opts BuildOptions) (Provider, error) {
	provStore.mu.Lock()
	defer provStore.mu.Unlock()

	sk := storeKey{
		name:   bc.name,
		config: string(bc.config),
		opts:   opts,
	}
	if wp, ok := provStore.providers[sk]; ok {
		wp.refCount++
		return newSingleCloseWrappedProvider(wp), nil
	}

	provider := bc.starter(opts)
	if provider == nil {
		return nil, fmt.Errorf("provider(%q, %q).Build(%v) failed", sk.name, sk.config, opts)
	}
	wp := &wrappedProvider{
		Provider: provider,
		refCount: 1,
		storeKey: sk,
		store:    provStore,
	}
	provStore.providers[sk] = wp
	return newSingleCloseWrappedProvider(wp), nil
}

// String returns the provider name and config as a colon separated string.
func (bc *BuildableConfig) String() string {
	return fmt.Sprintf("%s:%s", bc.name, string(bc.config))
}

// ParseConfig is a convenience function to create a BuildableConfig given a
// provider name and configuration. Returns an error if there is no registered
// builder for the given name or if the config parsing fails.
func ParseConfig(name string, config any) (*BuildableConfig, error) {
	parser := getBuilder(name)
	if parser == nil {
		return nil, fmt.Errorf("no certificate provider builder found for %q", name)
	}
	return parser.ParseConfig(config)
}

// GetProvider is a convenience function to create a provider given the name,
// config and build options.
func GetProvider(name string, config any, opts BuildOptions) (Provider, error) {
	bc, err := ParseConfig(name, config)
	if err != nil {
		return nil, err
	}
	return bc.Build(opts)
}
