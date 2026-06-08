/*
Copyright The Kubernetes Authors.

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
	"fmt"
	"sync"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/klog/v2"
)

// informerNameRegistry tracks all registered InformerName instances to detect collisions.
// Names must be globally unique across a process.
var informerNameRegistry = struct {
	sync.Mutex
	names map[string]*InformerName
}{
	names: make(map[string]*InformerName),
}

// InformerName represents a named informer identity used for metrics.
// It is created once at startup (e.g., in cmd/kube-controller-manager) and passed to
// the SharedInformerFactory. The name must be globally unique within a process.
//
// InformerName tracks which GVRs have been registered under
// this name. When an informer requests a name+GVR combination, the first one wins
// and gets metrics enabled. Subsequent requests for the same GVR silently get
// metrics disabled.
type InformerName struct {
	name string
	// lock protects gvrs map modifications
	lock sync.Mutex
	// reserved is flipped to false when Release() is called
	reserved *atomic.Bool
	// gvrs maps each registered GVR to its atomic bool for lock-free Reserved() checks
	gvrs map[schema.GroupVersionResource]*atomic.Bool
}

// NewInformerName creates a new InformerName with the given name.
// The name must be globally unique within the process. If a name collision
// is detected, an error is returned.
//
// The caller should call Release() when the informer name is no longer needed
// (eg. at shutdown) to allow the name to be reused.
func NewInformerName(name string) (*InformerName, error) {
	if name == "" {
		return nil, fmt.Errorf("informer name cannot be empty")
	}

	informerNameRegistry.Lock()
	defer informerNameRegistry.Unlock()

	if existing, ok := informerNameRegistry.names[name]; ok {
		// Check if the existing one is still reserved
		if existing.reserved.Load() {
			return nil, fmt.Errorf("informer name %q is already registered", name)
		}
		// Previous one was released, we can reuse the name
		delete(informerNameRegistry.names, name)
	}

	reserved := &atomic.Bool{}
	reserved.Store(true)

	n := &InformerName{
		name:     name,
		reserved: reserved,
		gvrs:     make(map[schema.GroupVersionResource]*atomic.Bool),
	}

	informerNameRegistry.names[name] = n
	return n, nil
}

// WithResource registers a GVR under this InformerName and returns an
// InformerNameAndResource that can be passed to FIFO/SharedIndexInformer.
//
// If this is the first time this GVR is registered under this name, the
// returned InformerNameAndResource will have Reserved() return true.
// If the GVR was already registered, the returned InformerNameAndResource
// will have Reserved() return false to prevent duplicate metrics.
func (n *InformerName) WithResource(gvr schema.GroupVersionResource) InformerNameAndResource {
	if n == nil {
		return InformerNameAndResource{gvr: gvr}
	}

	n.lock.Lock()
	defer n.lock.Unlock()

	retval := InformerNameAndResource{name: n.name, gvr: gvr, reserved: &atomic.Bool{}}
	if n.reserved.Load() {
		if _, gvrExists := n.gvrs[gvr]; !gvrExists {
			retval.reserved.Store(true)
			n.gvrs[gvr] = retval.reserved
		} else {
			// WithResource is called by generated informer code and probably
			// not worth converting to contextual logging, which would require
			// changing all those generated APIs.
			klog.TODO().Error(nil, "Duplicate informer registration - metrics will not be published", "informerName", n.name, "group", gvr.Group, "version", gvr.Version, "resource", gvr.Resource)
		}
	}
	return retval
}

// Release marks this InformerName as no longer in use.
// All InformerNameAndResource instances created from this InformerName
// will have their Reserved() return false after this call.
// The name can be reused by a subsequent NewInformerName call.
func (n *InformerName) Release() {
	if n == nil {
		return
	}

	n.lock.Lock()
	defer n.lock.Unlock()

	// Flip all GVR-specific flags so that any InformerNameAndResource
	// instances that were returned from WithResource() will have
	// Reserved() return false. These instances hold pointers to the
	// same atomic bools, so we must flip them before clearing the map.
	for _, reserved := range n.gvrs {
		reserved.Store(false)
	}

	// Clear the map
	n.gvrs = make(map[schema.GroupVersionResource]*atomic.Bool)

	// Flip the main reserved flag
	n.reserved.Store(false)

	// Remove from global registry
	informerNameRegistry.Lock()
	defer informerNameRegistry.Unlock()
	delete(informerNameRegistry.names, n.name)
}

// Name returns the name of this InformerName.
func (n *InformerName) Name() string {
	if n == nil {
		return ""
	}
	return n.name
}

// InformerNameAndResource represents a specific informer identity with both
// a name and a GVR. This is passed to FIFO and SharedIndexInformer for metrics.
//
// The Reserved() method provides a lock-free check to determine
// if metrics should be published. This is called on every queue operation
// so it must be fast.
type InformerNameAndResource struct {
	name     string
	gvr      schema.GroupVersionResource
	reserved *atomic.Bool
}

// Reserved returns true if this informer identity is reserved for metrics.
// This is a lock-free atomic load, safe and fast for hot-path usage.
//
// Returns false if:
// - The InformerNameAndResource is zero-valued (no name was configured)
// - The parent InformerName was released
// - This was a duplicate GVR registration
func (n InformerNameAndResource) Reserved() bool {
	if n.reserved == nil {
		return false
	}
	return n.reserved.Load()
}

// Name returns the informer name.
func (n InformerNameAndResource) Name() string {
	return n.name
}

// GroupVersionResource returns the GVR for this informer identity.
func (n InformerNameAndResource) GroupVersionResource() schema.GroupVersionResource {
	return n.gvr
}

// ResetInformerNamesForTesting clears the informer name registry.
// This is exported for testing purposes only.
func ResetInformerNamesForTesting() {
	informerNameRegistry.Lock()
	names := make([]*InformerName, 0, len(informerNameRegistry.names))
	for _, name := range informerNameRegistry.names {
		names = append(names, name)
	}
	informerNameRegistry.Unlock()

	for _, name := range names {
		name.Release()
	}
}
