// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package adapters

import (
	"fmt"
	"reflect"
	"slices"
	"sync"

	"github.com/go-openapi/swag/jsonutils/adapters/ifaces"
	stdlib "github.com/go-openapi/swag/jsonutils/adapters/stdlib/json"
)

// Registry holds the global registry for registered adapters.
var Registry = NewRegistrar()

var (
	defaultRegistered = stdlib.Register

	_ ifaces.Registrar = &Registrar{}
)

type registryError string

func (e registryError) Error() string {
	return string(e)
}

// ErrRegistry indicates an error returned by the [Registrar].
var ErrRegistry registryError = "JSON adapters registry error"

type registry []*ifaces.RegistryEntry

// Registrar holds registered [ifaces.Adapters] for different serialization capabilities.
//
// Internally, it maintains a cache for data types that favor a given adapter.
type Registrar struct {
	marshalerRegistry          registry
	unmarshalerRegistry        registry
	orderedMarshalerRegistry   registry
	orderedUnmarshalerRegistry registry
	orderedMapRegistry         registry

	gmx sync.RWMutex

	// cache indexed by value type, so we don't have to lookup
	marshalerCache          map[reflect.Type]*ifaces.RegistryEntry
	unmarshalerCache        map[reflect.Type]*ifaces.RegistryEntry
	orderedMarshalerCache   map[reflect.Type]*ifaces.RegistryEntry
	orderedUnmarshalerCache map[reflect.Type]*ifaces.RegistryEntry
	orderedMapCache         map[reflect.Type]*ifaces.RegistryEntry
}

func NewRegistrar() *Registrar {
	r := &Registrar{}

	r.marshalerRegistry = make(registry, 0, 1)
	r.unmarshalerRegistry = make(registry, 0, 1)
	r.orderedMarshalerRegistry = make(registry, 0, 1)
	r.orderedUnmarshalerRegistry = make(registry, 0, 1)
	r.orderedMapRegistry = make(registry, 0, 1)

	r.marshalerCache = make(map[reflect.Type]*ifaces.RegistryEntry)
	r.unmarshalerCache = make(map[reflect.Type]*ifaces.RegistryEntry)
	r.orderedMarshalerCache = make(map[reflect.Type]*ifaces.RegistryEntry)
	r.orderedUnmarshalerCache = make(map[reflect.Type]*ifaces.RegistryEntry)
	r.orderedMapCache = make(map[reflect.Type]*ifaces.RegistryEntry)

	defaultRegistered(r)

	return r
}

// ClearCache resets the internal type cache.
func (r *Registrar) ClearCache() {
	r.gmx.Lock()
	r.clearCache()
	r.gmx.Unlock()
}

// Reset the [Registrar] to its defaults.
func (r *Registrar) Reset() {
	r.gmx.Lock()
	r.clearCache()
	r.marshalerRegistry = r.marshalerRegistry[:0]
	r.unmarshalerRegistry = r.unmarshalerRegistry[:0]
	r.orderedMarshalerRegistry = r.orderedMarshalerRegistry[:0]
	r.orderedUnmarshalerRegistry = r.orderedUnmarshalerRegistry[:0]
	r.orderedMapRegistry = r.orderedMapRegistry[:0]
	r.gmx.Unlock()

	defaultRegistered(r)
}

// RegisterFor registers an adapter for some JSON capabilities.
func (r *Registrar) RegisterFor(entry ifaces.RegistryEntry) {
	r.gmx.Lock()
	if entry.What.Has(ifaces.CapabilityMarshalJSON) {
		e := entry
		e.What &= ifaces.Capabilities(ifaces.CapabilityMarshalJSON)
		r.marshalerRegistry = slices.Insert(r.marshalerRegistry, 0, &e)
	}
	if entry.What.Has(ifaces.CapabilityUnmarshalJSON) {
		e := entry
		e.What &= ifaces.Capabilities(ifaces.CapabilityUnmarshalJSON)
		r.unmarshalerRegistry = slices.Insert(r.unmarshalerRegistry, 0, &e)
	}
	if entry.What.Has(ifaces.CapabilityOrderedMarshalJSON) {
		e := entry
		e.What &= ifaces.Capabilities(ifaces.CapabilityOrderedMarshalJSON)
		r.orderedMarshalerRegistry = slices.Insert(r.orderedMarshalerRegistry, 0, &e)
	}
	if entry.What.Has(ifaces.CapabilityOrderedUnmarshalJSON) {
		e := entry
		e.What &= ifaces.Capabilities(ifaces.CapabilityOrderedUnmarshalJSON)
		r.orderedUnmarshalerRegistry = slices.Insert(r.orderedUnmarshalerRegistry, 0, &e)
	}
	if entry.What.Has(ifaces.CapabilityOrderedMap) {
		e := entry
		e.What &= ifaces.Capabilities(ifaces.CapabilityOrderedMap)
		r.orderedMapRegistry = slices.Insert(r.orderedMapRegistry, 0, &e)
	}
	r.gmx.Unlock()
}

// AdapterFor returns an [ifaces.Adapter] that supports this capability for this type of value.
//
// The [ifaces.Adapter] may be redeemed to its pool using its Redeem() method, for adapters that support global
// pooling. When this is not the case, the redeem function is just a no-operation.
func (r *Registrar) AdapterFor(capability ifaces.Capability, value any) ifaces.Adapter {
	entry := r.findFirstFor(capability, value)
	if entry == nil {
		return nil
	}

	return entry.Constructor()
}

func (r *Registrar) clearCache() {
	clear(r.marshalerCache)
	clear(r.unmarshalerCache)
	clear(r.orderedMarshalerCache)
	clear(r.orderedUnmarshalerCache)
	clear(r.orderedMapCache)
}

func (r *Registrar) findFirstFor(capability ifaces.Capability, value any) *ifaces.RegistryEntry {
	switch capability {
	case ifaces.CapabilityMarshalJSON:
		return r.findFirstInRegistryFor(r.marshalerRegistry, r.marshalerCache, capability, value)
	case ifaces.CapabilityUnmarshalJSON:
		return r.findFirstInRegistryFor(r.unmarshalerRegistry, r.unmarshalerCache, capability, value)
	case ifaces.CapabilityOrderedMarshalJSON:
		return r.findFirstInRegistryFor(r.orderedMarshalerRegistry, r.orderedMarshalerCache, capability, value)
	case ifaces.CapabilityOrderedUnmarshalJSON:
		return r.findFirstInRegistryFor(r.orderedUnmarshalerRegistry, r.orderedUnmarshalerCache, capability, value)
	case ifaces.CapabilityOrderedMap:
		return r.findFirstInRegistryFor(r.orderedMapRegistry, r.orderedMapCache, capability, value)
	default:
		panic(fmt.Errorf("unsupported capability %d: %w", capability, ErrRegistry))
	}
}

func (r *Registrar) findFirstInRegistryFor(reg registry, cache map[reflect.Type]*ifaces.RegistryEntry, capability ifaces.Capability, value any) *ifaces.RegistryEntry {
	r.gmx.RLock()
	if len(reg) > 1 {
		if entry, ok := cache[reflect.TypeOf(value)]; ok {
			// cache hit
			r.gmx.RUnlock()
			return entry
		}
	}

	for _, entry := range reg {
		if !entry.Support(capability, value) {
			continue
		}

		r.gmx.RUnlock()

		// update the internal cache
		r.gmx.Lock()
		cache[reflect.TypeOf(value)] = entry
		r.gmx.Unlock()

		return entry
	}

	// no adapter found
	r.gmx.RUnlock()

	return nil
}

// MarshalAdapterFor returns the first adapter that knows how to Marshal this type of value.
func MarshalAdapterFor(value any) ifaces.MarshalAdapter {
	return Registry.AdapterFor(ifaces.CapabilityMarshalJSON, value)
}

// OrderedMarshalAdapterFor returns the first adapter that knows how to OrderedMarshal this type of value.
func OrderedMarshalAdapterFor(value ifaces.Ordered) ifaces.OrderedMarshalAdapter {
	return Registry.AdapterFor(ifaces.CapabilityOrderedMarshalJSON, value)
}

// UnmarshalAdapterFor returns the first adapter that knows how to Unmarshal this type of value.
func UnmarshalAdapterFor(value any) ifaces.UnmarshalAdapter {
	return Registry.AdapterFor(ifaces.CapabilityUnmarshalJSON, value)
}

// OrderedUnmarshalAdapterFor provides the first adapter that knows how to OrderedUnmarshal this type of value.
func OrderedUnmarshalAdapterFor(value ifaces.SetOrdered) ifaces.OrderedUnmarshalAdapter {
	return Registry.AdapterFor(ifaces.CapabilityOrderedUnmarshalJSON, value)
}

// NewOrderedMap provides the "ordered map" implementation provided by the registry.
func NewOrderedMap(capacity int) ifaces.OrderedMap {
	var v any
	adapter := Registry.AdapterFor(ifaces.CapabilityOrderedUnmarshalJSON, v)
	if adapter == nil {
		return nil
	}

	defer adapter.Redeem()
	return adapter.NewOrderedMap(capacity)
}

func noopRedeemer() {}
