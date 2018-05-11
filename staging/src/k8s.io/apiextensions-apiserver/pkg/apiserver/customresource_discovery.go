/*
Copyright 2017 The Kubernetes Authors.

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

package apiserver

import (
	"net/http"
	"strings"
	"sync"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/discovery"
)

type versionDiscoveryMap map[schema.GroupVersion]*discovery.APIVersionHandler

type versionDiscoveryHandler struct {
	discoveryLock    sync.RWMutex
	discoveryStorage atomic.Value // writing is infrequent

	delegate http.Handler
}

func (r *versionDiscoveryHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	pathParts := splitPath(req.URL.Path)
	// only match /apis/<group>/<version>
	if len(pathParts) != 3 || pathParts[0] != "apis" {
		r.delegate.ServeHTTP(w, req)
		return
	}
	discovery, ok := r.getDiscovery(schema.GroupVersion{Group: pathParts[1], Version: pathParts[2]})
	if !ok {
		r.delegate.ServeHTTP(w, req)
		return
	}

	discovery.ServeHTTP(w, req)
}

func (r *versionDiscoveryHandler) getDiscovery(gv schema.GroupVersion) (*discovery.APIVersionHandler, bool) {
	discoveryMap := r.discoveryStorage.Load().(versionDiscoveryMap)
	ret, ok := discoveryMap[gv]
	return ret, ok
}

func (r *versionDiscoveryHandler) setDiscovery(gv schema.GroupVersion, discovery *discovery.APIVersionHandler) {
	r.discoveryLock.Lock()
	defer r.discoveryLock.Unlock()

	discoveryMap := r.discoveryStorage.Load().(versionDiscoveryMap)
	discoveryMap2 := clone(discoveryMap).(versionDiscoveryMap)
	discoveryMap2[gv] = discovery
	r.discoveryStorage.Store(discoveryMap2)
}

func (r *versionDiscoveryHandler) unsetDiscovery(gv schema.GroupVersion) {
	r.discoveryLock.Lock()
	defer r.discoveryLock.Unlock()

	discoveryMap := r.discoveryStorage.Load().(versionDiscoveryMap)
	discoveryMap2 := clone(discoveryMap).(versionDiscoveryMap)
	delete(discoveryMap2, gv)
	r.discoveryStorage.Store(discoveryMap2)
}

type groupDiscoveryMap map[string]*discovery.APIGroupHandler

type groupDiscoveryHandler struct {
	discoveryLock    sync.RWMutex
	discoveryStorage atomic.Value // writing is infrequent

	delegate http.Handler
}

func (r *groupDiscoveryHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	pathParts := splitPath(req.URL.Path)
	// only match /apis/<group>
	if len(pathParts) != 2 || pathParts[0] != "apis" {
		r.delegate.ServeHTTP(w, req)
		return
	}
	discovery, ok := r.getDiscovery(pathParts[1])
	if !ok {
		r.delegate.ServeHTTP(w, req)
		return
	}

	discovery.ServeHTTP(w, req)
}

func (r *groupDiscoveryHandler) getDiscovery(group string) (*discovery.APIGroupHandler, bool) {
	discoveryMap := r.discoveryStorage.Load().(groupDiscoveryMap)
	ret, ok := discoveryMap[group]
	return ret, ok
}

func (r *groupDiscoveryHandler) setDiscovery(group string, discovery *discovery.APIGroupHandler) {
	r.discoveryLock.Lock()
	defer r.discoveryLock.Unlock()

	discoveryMap := r.discoveryStorage.Load().(groupDiscoveryMap)
	discoveryMap2 := clone(discoveryMap).(groupDiscoveryMap)
	discoveryMap2[group] = discovery
	r.discoveryStorage.Store(discoveryMap2)
}

func (r *groupDiscoveryHandler) unsetDiscovery(group string) {
	r.discoveryLock.Lock()
	defer r.discoveryLock.Unlock()

	discoveryMap := r.discoveryStorage.Load().(groupDiscoveryMap)
	discoveryMap2 := clone(discoveryMap).(groupDiscoveryMap)
	delete(discoveryMap2, group)
	r.discoveryStorage.Store(discoveryMap2)
}

// splitPath returns the segments for a URL path.
func splitPath(path string) []string {
	path = strings.Trim(path, "/")
	if path == "" {
		return []string{}
	}
	return strings.Split(path, "/")
}

// clone returns a shallow copy of the provided object.
func clone(original interface{}) interface{} {
	if original == nil {
		return nil
	}

	switch original := original.(type) {
	case versionDiscoveryMap:
		clone := make(versionDiscoveryMap, len(original))
		for k, v := range original {
			clone[k] = v
		}
		return clone
	case groupDiscoveryMap:
		clone := make(groupDiscoveryMap, len(original))
		for k, v := range original {
			clone[k] = v
		}
		return clone
	case crdStorageMap:
		clone := make(crdStorageMap, len(original))
		for k, v := range original {
			clone[k] = v
		}
		return clone
	default:
		return original
	}
}
