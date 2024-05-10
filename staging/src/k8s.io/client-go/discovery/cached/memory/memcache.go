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

package memory

import (
	"errors"
	"fmt"
	"sync"
	"syscall"

	openapi_v2 "github.com/google/gnostic-models/openapiv2"

	errorsutil "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/openapi"
	cachedopenapi "k8s.io/client-go/openapi/cached"
	restclient "k8s.io/client-go/rest"
	"k8s.io/klog/v2"
)

type cacheEntry struct {
	resourceList *metav1.APIResourceList
	err          error
}

// memCacheClient can Invalidate() to stay up-to-date with discovery
// information.
//
// TODO: Switch to a watch interface. Right now it will poll after each
// Invalidate() call.
type memCacheClient struct {
	delegate discovery.DiscoveryInterface

	lock                        sync.RWMutex
	groupToServerResources      map[string]*cacheEntry
	groupList                   *metav1.APIGroupList
	cacheValid                  bool
	openapiClient               openapi.Client
	receivedAggregatedDiscovery bool
}

// Error Constants
var (
	ErrCacheNotFound = errors.New("not found")
)

// Server returning empty ResourceList for Group/Version.
type emptyResponseError struct {
	gv string
}

func (e *emptyResponseError) Error() string {
	return fmt.Sprintf("received empty response for: %s", e.gv)
}

var _ discovery.CachedDiscoveryInterface = &memCacheClient{}

// isTransientConnectionError checks whether given error is "Connection refused" or
// "Connection reset" error which usually means that apiserver is temporarily
// unavailable.
func isTransientConnectionError(err error) bool {
	var errno syscall.Errno
	if errors.As(err, &errno) {
		return errno == syscall.ECONNREFUSED || errno == syscall.ECONNRESET
	}
	return false
}

func isTransientError(err error) bool {
	if isTransientConnectionError(err) {
		return true
	}

	if t, ok := err.(errorsutil.APIStatus); ok && t.Status().Code >= 500 {
		return true
	}

	return errorsutil.IsTooManyRequests(err)
}

// ServerResourcesForGroupVersion returns the supported resources for a group and version.
func (d *memCacheClient) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	d.lock.Lock()
	defer d.lock.Unlock()
	if !d.cacheValid {
		if err := d.refreshLocked(); err != nil {
			return nil, err
		}
	}
	cachedVal, ok := d.groupToServerResources[groupVersion]
	if !ok {
		return nil, ErrCacheNotFound
	}

	if cachedVal.err != nil && isTransientError(cachedVal.err) {
		r, err := d.serverResourcesForGroupVersion(groupVersion)
		if err != nil {
			// Don't log "empty response" as an error; it is a common response for metrics.
			if _, emptyErr := err.(*emptyResponseError); emptyErr {
				// Log at same verbosity as disk cache.
				klog.V(3).InfoS("Received empty response for GroupVersion", "groupVersion", groupVersion)
			} else {
				utilruntime.HandleError(fmt.Errorf("couldn't get resource list for %v: %v", groupVersion, err))
			}
		}
		cachedVal = &cacheEntry{r, err}
		d.groupToServerResources[groupVersion] = cachedVal
	}

	return cachedVal.resourceList, cachedVal.err
}

// ServerGroupsAndResources returns the groups and supported resources for all groups and versions.
func (d *memCacheClient) ServerGroupsAndResources() ([]*metav1.APIGroup, []*metav1.APIResourceList, error) {
	return discovery.ServerGroupsAndResources(d)
}

// GroupsAndMaybeResources returns the list of APIGroups, and possibly the map of group/version
// to resources. The returned groups will never be nil, but the resources map can be nil
// if there are no cached resources.
func (d *memCacheClient) GroupsAndMaybeResources() (*metav1.APIGroupList, map[schema.GroupVersion]*metav1.APIResourceList, map[schema.GroupVersion]error, error) {
	d.lock.Lock()
	defer d.lock.Unlock()

	if !d.cacheValid {
		if err := d.refreshLocked(); err != nil {
			return nil, nil, nil, err
		}
	}
	// Build the resourceList from the cache?
	var resourcesMap map[schema.GroupVersion]*metav1.APIResourceList
	var failedGVs map[schema.GroupVersion]error
	if d.receivedAggregatedDiscovery && len(d.groupToServerResources) > 0 {
		resourcesMap = map[schema.GroupVersion]*metav1.APIResourceList{}
		failedGVs = map[schema.GroupVersion]error{}
		for gv, cacheEntry := range d.groupToServerResources {
			groupVersion, err := schema.ParseGroupVersion(gv)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("failed to parse group version (%v): %v", gv, err)
			}
			if cacheEntry.err != nil {
				failedGVs[groupVersion] = cacheEntry.err
			} else {
				resourcesMap[groupVersion] = cacheEntry.resourceList
			}
		}
	}
	return d.groupList, resourcesMap, failedGVs, nil
}

func (d *memCacheClient) ServerGroups() (*metav1.APIGroupList, error) {
	groups, _, _, err := d.GroupsAndMaybeResources()
	if err != nil {
		return nil, err
	}
	return groups, nil
}

func (d *memCacheClient) RESTClient() restclient.Interface {
	return d.delegate.RESTClient()
}

func (d *memCacheClient) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	return discovery.ServerPreferredResources(d)
}

func (d *memCacheClient) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	return discovery.ServerPreferredNamespacedResources(d)
}

func (d *memCacheClient) ServerVersion() (*version.Info, error) {
	return d.delegate.ServerVersion()
}

func (d *memCacheClient) OpenAPISchema() (*openapi_v2.Document, error) {
	return d.delegate.OpenAPISchema()
}

func (d *memCacheClient) OpenAPIV3() openapi.Client {
	// Must take lock since Invalidate call may modify openapiClient
	d.lock.Lock()
	defer d.lock.Unlock()

	if d.openapiClient == nil {
		d.openapiClient = cachedopenapi.NewClient(d.delegate.OpenAPIV3())
	}

	return d.openapiClient
}

func (d *memCacheClient) Fresh() bool {
	d.lock.RLock()
	defer d.lock.RUnlock()
	// Return whether the cache is populated at all. It is still possible that
	// a single entry is missing due to transient errors and the attempt to read
	// that entry will trigger retry.
	return d.cacheValid
}

// Invalidate enforces that no cached data that is older than the current time
// is used.
func (d *memCacheClient) Invalidate() {
	d.lock.Lock()
	defer d.lock.Unlock()
	d.cacheValid = false
	d.groupToServerResources = nil
	d.groupList = nil
	d.openapiClient = nil
	d.receivedAggregatedDiscovery = false
	if ad, ok := d.delegate.(discovery.CachedDiscoveryInterface); ok {
		ad.Invalidate()
	}
}

// refreshLocked refreshes the state of cache. The caller must hold d.lock for
// writing.
func (d *memCacheClient) refreshLocked() error {
	// TODO: Could this multiplicative set of calls be replaced by a single call
	// to ServerResources? If it's possible for more than one resulting
	// APIResourceList to have the same GroupVersion, the lists would need merged.
	var gl *metav1.APIGroupList
	var err error

	if ad, ok := d.delegate.(discovery.AggregatedDiscoveryInterface); ok {
		var resources map[schema.GroupVersion]*metav1.APIResourceList
		var failedGVs map[schema.GroupVersion]error
		gl, resources, failedGVs, err = ad.GroupsAndMaybeResources()
		if resources != nil && err == nil {
			// Cache the resources.
			d.groupToServerResources = map[string]*cacheEntry{}
			d.groupList = gl
			for gv, resources := range resources {
				d.groupToServerResources[gv.String()] = &cacheEntry{resources, nil}
			}
			// Cache GroupVersion discovery errors
			for gv, err := range failedGVs {
				d.groupToServerResources[gv.String()] = &cacheEntry{nil, err}
			}
			d.receivedAggregatedDiscovery = true
			d.cacheValid = true
			return nil
		}
	} else {
		gl, err = d.delegate.ServerGroups()
	}
	if err != nil || len(gl.Groups) == 0 {
		utilruntime.HandleError(fmt.Errorf("couldn't get current server API group list: %v", err))
		return err
	}

	wg := &sync.WaitGroup{}
	resultLock := &sync.Mutex{}
	rl := map[string]*cacheEntry{}
	for _, g := range gl.Groups {
		for _, v := range g.Versions {
			gv := v.GroupVersion
			wg.Add(1)
			go func() {
				defer wg.Done()
				defer utilruntime.HandleCrash()

				r, err := d.serverResourcesForGroupVersion(gv)
				if err != nil {
					// Don't log "empty response" as an error; it is a common response for metrics.
					if _, emptyErr := err.(*emptyResponseError); emptyErr {
						// Log at same verbosity as disk cache.
						klog.V(3).InfoS("Received empty response for GroupVersion", "groupVersion", gv)
					} else {
						utilruntime.HandleError(fmt.Errorf("couldn't get resource list for %v: %v", gv, err))
					}
				}

				resultLock.Lock()
				defer resultLock.Unlock()
				rl[gv] = &cacheEntry{r, err}
			}()
		}
	}
	wg.Wait()

	d.groupToServerResources, d.groupList = rl, gl
	d.cacheValid = true
	return nil
}

func (d *memCacheClient) serverResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	r, err := d.delegate.ServerResourcesForGroupVersion(groupVersion)
	if err != nil {
		return r, err
	}
	if len(r.APIResources) == 0 {
		return r, &emptyResponseError{gv: groupVersion}
	}
	return r, nil
}

// WithLegacy returns current memory-cached discovery client;
// current client does not support legacy-only discovery.
func (d *memCacheClient) WithLegacy() discovery.DiscoveryInterface {
	return d
}

// NewMemCacheClient creates a new CachedDiscoveryInterface which caches
// discovery information in memory and will stay up-to-date if Invalidate is
// called with regularity.
//
// NOTE: The client will NOT resort to live lookups on cache misses.
func NewMemCacheClient(delegate discovery.DiscoveryInterface) discovery.CachedDiscoveryInterface {
	return &memCacheClient{
		delegate:                    delegate,
		groupToServerResources:      map[string]*cacheEntry{},
		receivedAggregatedDiscovery: false,
	}
}
