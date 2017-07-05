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

package cached

import (
	"errors"
	"fmt"
	"sync"

	"github.com/emicklei/go-restful-swagger12"
	"github.com/googleapis/gnostic/OpenAPIv2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/discovery"
	restclient "k8s.io/client-go/rest"
)

// memCacheClient can Invalidate() to stay up-to-date with discovery
// information.
//
// TODO: Switch to a watch interface. Right now it will poll anytime
// Invalidate() is called.
type memCacheClient struct {
	delegate discovery.DiscoveryInterface

	lock                   sync.RWMutex
	groupToServerResources map[string]*metav1.APIResourceList
	groupList              *metav1.APIGroupList
	cacheValid             bool
}

var (
	ErrCacheEmpty    = errors.New("the cache has not been filled yet")
	ErrCacheNotFound = errors.New("not found")
)

var _ discovery.CachedDiscoveryInterface = &memCacheClient{}

// ServerResourcesForGroupVersion returns the supported resources for a group and version.
func (d *memCacheClient) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	d.lock.RLock()
	defer d.lock.RUnlock()
	if !d.cacheValid {
		return nil, ErrCacheEmpty
	}
	cachedVal, ok := d.groupToServerResources[groupVersion]
	if !ok {
		return nil, ErrCacheNotFound
	}
	return cachedVal, nil
}

// ServerResources returns the supported resources for all groups and versions.
func (d *memCacheClient) ServerResources() ([]*metav1.APIResourceList, error) {
	apiGroups, err := d.ServerGroups()
	if err != nil {
		return nil, err
	}
	groupVersions := metav1.ExtractGroupVersions(apiGroups)
	result := []*metav1.APIResourceList{}
	for _, groupVersion := range groupVersions {
		resources, err := d.ServerResourcesForGroupVersion(groupVersion)
		if err != nil {
			return nil, err
		}
		result = append(result, resources)
	}
	return result, nil
}

func (d *memCacheClient) ServerGroups() (*metav1.APIGroupList, error) {
	d.lock.RLock()
	defer d.lock.RUnlock()
	if d.groupList == nil {
		return nil, ErrCacheEmpty
	}
	return d.groupList, nil
}

func (d *memCacheClient) RESTClient() restclient.Interface {
	return d.delegate.RESTClient()
}

// TODO: Should this also be cached? The results seem more likely to be
// inconsistent with ServerGroups and ServerResources given the requirement to
// actively Invalidate.
func (d *memCacheClient) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	return d.delegate.ServerPreferredResources()
}

// TODO: Should this also be cached? The results seem more likely to be
// inconsistent with ServerGroups and ServerResources given the requirement to
// actively Invalidate.
func (d *memCacheClient) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	return d.delegate.ServerPreferredNamespacedResources()
}

func (d *memCacheClient) ServerVersion() (*version.Info, error) {
	return d.delegate.ServerVersion()
}

func (d *memCacheClient) SwaggerSchema(version schema.GroupVersion) (*swagger.ApiDeclaration, error) {
	return d.delegate.SwaggerSchema(version)
}

func (d *memCacheClient) OpenAPISchema() (*openapi_v2.Document, error) {
	return d.delegate.OpenAPISchema()
}

func (d *memCacheClient) Fresh() bool {
	d.lock.RLock()
	defer d.lock.RUnlock()
	// Fresh is supposed to tell the caller whether or not to retry if the cache
	// fails to find something. The idea here is that Invalidate will be called
	// periodically and therefore we'll always be returning the latest data. (And
	// in the future we can watch and stay even more up-to-date.) So we only
	// return false if the cache has never been filled.
	return d.cacheValid
}

// Invalidate refreshes the cache, blocking calls until the cache has been
// refreshed. It would be trivial to make a version that does this in the
// background while continuing to respond to requests if needed.
func (d *memCacheClient) Invalidate() {
	d.lock.Lock()
	defer d.lock.Unlock()

	// TODO: Could this multiplicative set of calls be replaced by a single call
	// to ServerResources? If it's possible for more than one resulting
	// APIResourceList to have the same GroupVersion, the lists would need merged.
	gl, err := d.delegate.ServerGroups()
	if err != nil || len(gl.Groups) == 0 {
		utilruntime.HandleError(fmt.Errorf("couldn't get current server API group list; will keep using cached value. (%v)", err))
		return
	}

	rl := map[string]*metav1.APIResourceList{}
	for _, g := range gl.Groups {
		for _, v := range g.Versions {
			r, err := d.delegate.ServerResourcesForGroupVersion(v.GroupVersion)
			if err != nil || len(r.APIResources) == 0 {
				utilruntime.HandleError(fmt.Errorf("couldn't get resource list for %v: %v", v.GroupVersion, err))
				if cur, ok := d.groupToServerResources[v.GroupVersion]; ok {
					// retain the existing list, if we had it.
					r = cur
				} else {
					continue
				}
			}
			rl[v.GroupVersion] = r
		}
	}

	d.groupToServerResources, d.groupList = rl, gl
	d.cacheValid = true
}

// NewMemCacheClient creates a new CachedDiscoveryInterface which caches
// discovery information in memory and will stay up-to-date if Invalidate is
// called with regularity.
//
// NOTE: The client will NOT resort to live lookups on cache misses.
func NewMemCacheClient(delegate discovery.DiscoveryInterface) discovery.CachedDiscoveryInterface {
	return &memCacheClient{
		delegate:               delegate,
		groupToServerResources: map[string]*metav1.APIResourceList{},
	}
}
