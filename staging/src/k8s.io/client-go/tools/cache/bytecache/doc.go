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

// Package bytecache is an EXPERIMENTAL storage backend for informer caches.
//
// Objects at rest are relocated into a file-backed, read-only-mapped memory
// arena: they cost no Go heap objects, are invisible to the GC and the GOGC
// pacer, and cold pages can be evicted by the kernel (no swap required).
// Get materializes objects into self-contained heap []byte blocks via a
// recorded relocation list (memcpy + pointer patch, ~3µs for a typical pod),
// fronted by a bounded LRU so hot reads are a pointer return. Consumers hold
// ordinary GC-traced pointers; block lifetime is managed by the GC through
// interior-pointer reachability.
//
// The implementation mirrors Go runtime internals (Swiss map layout,
// interface representation) and is gated to amd64, non-race builds; any
// object it cannot encode is stored as a plain heap pointer (passthrough),
// so behavior degrades to the status quo rather than failing.
//
// Enabled via KUBE_BYTECACHE=1. See KUBE_BYTECACHE_HOT_BYTES and
// KUBE_BYTECACHE_DIR to tune the per-store hot budget and arena location.
package bytecache
