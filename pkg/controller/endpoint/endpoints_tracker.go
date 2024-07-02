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

package endpoint

import (
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// staleEndpointsTracker tracks Endpoints and their stale resource versions to
// help determine if an Endpoints is stale.
type staleEndpointsTracker struct {
	// lock protects staleResourceVersionByEndpoints.
	lock sync.RWMutex
	// staleResourceVersionByEndpoints tracks the stale resource version of Endpoints.
	staleResourceVersionByEndpoints map[types.NamespacedName]string
}

func newStaleEndpointsTracker() *staleEndpointsTracker {
	return &staleEndpointsTracker{
		staleResourceVersionByEndpoints: map[types.NamespacedName]string{},
	}
}

func (t *staleEndpointsTracker) Stale(endpoints *v1.Endpoints) {
	t.lock.Lock()
	defer t.lock.Unlock()
	nn := types.NamespacedName{Name: endpoints.Name, Namespace: endpoints.Namespace}
	t.staleResourceVersionByEndpoints[nn] = endpoints.ResourceVersion
}

func (t *staleEndpointsTracker) IsStale(endpoints *v1.Endpoints) bool {
	t.lock.RLock()
	defer t.lock.RUnlock()
	nn := types.NamespacedName{Name: endpoints.Name, Namespace: endpoints.Namespace}
	staleResourceVersion, exists := t.staleResourceVersionByEndpoints[nn]
	if exists && staleResourceVersion == endpoints.ResourceVersion {
		return true
	}
	return false
}

func (t *staleEndpointsTracker) Delete(namespace, name string) {
	t.lock.Lock()
	defer t.lock.Unlock()
	nn := types.NamespacedName{Namespace: namespace, Name: name}
	delete(t.staleResourceVersionByEndpoints, nn)
}
