/*
Copyright 2019 The Kubernetes Authors.

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

package memory

import (
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/discovery/cached/memory"
)

// NewMemCacheClient is DEPRECATED. Use memory.NewMemCacheClient directly.
func NewMemCacheClient(delegate discovery.DiscoveryInterface) discovery.CachedDiscoveryInterface {
	return memory.NewMemCacheClient(delegate)
}

// ErrCacheNotFound is DEPRECATED. Use memory.ErrCacheNotFound directly.
var ErrCacheNotFound = memory.ErrCacheNotFound
