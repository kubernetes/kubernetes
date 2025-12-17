/*
Copyright 2025 The Kubernetes Authors.

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

// Package cache is a client-side caching mechanism. It is useful for
// reducing the number of server calls you'd otherwise need to make.
// Reflector watches a server and updates a Store. Two stores are provided;
// one that simply caches objects (for example, to allow a scheduler to
// list currently available nodes), and one that additionally acts as
// a FIFO queue (for example, to allow a scheduler to process incoming
// pods).
package cache

import (
	"reflect"
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
)

// identifierRegistry tracks all registered identifier keys to detect collisions.
// Keys are composed of name+itemType. Only explicitly named identifiers are registered.
var identifierRegistry = struct {
	sync.Mutex
	keys map[string]bool
}{
	keys: make(map[string]bool),
}

// Identifier is used to identify of informers and FIFO for metrics and logging purposes.
//
// Metrics are only published for identifiers that are:
// 1. Explicitly named (non-empty name)
// 2. Unique (no other identifier has the same name+itemType combination)
//
// This ensures that metrics labels are consistent across restarts and don't collide.
// Unnamed FIFOs will not have metrics published - it's the responsibility of
// client authors to name all FIFOs they care about observing.
type Identifier struct {
	// Name is the name of proposed name to reference.
	name string
	// ItemType is the type of item type like "v1.Pod".
	itemType string
	// unique indicates whether this identifier was successfully registered
	// as unique (true) or if it collided with an existing name+itemType (false).
	unique bool
}

// NewIdentifier creates a new Identifier with the given name and example object.
// If name is non-empty, it will be registered for uniqueness tracking using
// both name and itemType as the composite key.
// If the name+itemType collides with an existing identifier, IsUnique() will return false
// and metrics will not be published for this identifier.
func NewIdentifier(name string, obj runtime.Object) *Identifier {
	id := &Identifier{name: name, itemType: itemType(obj)}
	if name != "" {
		id.unique = registerIdentifier(id.name, id.itemType)
	}
	return id
}

// registerIdentifier attempts to register a name+itemType key and returns true if the key
// was unique (not previously registered), false if it collides.
func registerIdentifier(name, itemType string) bool {
	key := name + "/" + itemType

	identifierRegistry.Lock()
	defer identifierRegistry.Unlock()

	if identifierRegistry.keys[key] {
		klog.Warningf("FIFO identifier %q (itemType=%s) is not unique - metrics will not be published for this FIFO", name, itemType)
		return false
	}

	identifierRegistry.keys[key] = true
	return true
}

func (id *Identifier) Name() string {
	if id == nil {
		return ""
	}
	return id.name
}

func (id *Identifier) ItemType() string {
	if id == nil {
		return ""
	}
	return id.itemType
}

// IsUnique returns true if this identifier has an explicit name+itemType that is unique
// across all identifiers. Metrics are only published for unique identifiers.
//
// Returns false if:
// - The identifier is nil
// - The identifier has no name (unnamed FIFOs don't get metrics)
// - The identifier's name+itemType collides with another identifier
func (id *Identifier) IsUnique() bool {
	if id == nil {
		return false
	}
	return id.unique
}

func itemType(exampleObject runtime.Object) string {
	return reflect.TypeOf(exampleObject).Elem().String()
}

func init() {
	resetIdentity()
}

// resetIdentity clears the identifier registry.
func resetIdentity() {
	identifierRegistry.Lock()
	defer identifierRegistry.Unlock()
	identifierRegistry.keys = make(map[string]bool)
}
