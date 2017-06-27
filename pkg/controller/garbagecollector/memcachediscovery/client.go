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

// Package memcachediscovery includes a Client which is a CachedDiscoveryInterface.
package memcachediscovery

import (
	"errors"
	"sync"

	"github.com/emicklei/go-restful-swagger12"
	"github.com/go-openapi/spec"
	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/discovery"
	restclient "k8s.io/client-go/rest"
)

// Client can Refresh() to stay up-to-date with discovery information. Before
// this is moved some place where it's easier to call, it needs to switch to a
// watch interface. Right now it will poll anytime Refresh() is called.
type Client struct {
	delegate discovery.DiscoveryInterface

	lock                   sync.RWMutex
	groupToServerResources map[string]*metav1.APIResourceList
	groupList              *metav1.APIGroupList
	cacheValid             bool
}

var (
	ErrCacheEmpty = errors.New("the cache has not been filled yet")
)

var _ discovery.CachedDiscoveryInterface = &Client{}

// ServerResourcesForGroupVersion returns the supported resources for a group and version.
func (d *Client) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	d.lock.RLock()
	defer d.lock.RUnlock()
	cachedVal, ok := d.groupToServerResources[groupVersion]
	if !ok {
		return nil, ErrCacheEmpty
	}
	return cachedVal, nil
}

// ServerResources returns the supported resources for all groups and versions.
func (d *Client) ServerResources() ([]*metav1.APIResourceList, error) {
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

func (d *Client) ServerGroups() (*metav1.APIGroupList, error) {
	d.lock.RLock()
	defer d.lock.RUnlock()
	if d.groupList == nil {
		return nil, ErrCacheEmpty
	}
	return d.groupList, nil
}

func (d *Client) RESTClient() restclient.Interface {
	return d.delegate.RESTClient()
}

func (d *Client) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	return d.delegate.ServerPreferredResources()
}

func (d *Client) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	return d.delegate.ServerPreferredNamespacedResources()
}

func (d *Client) ServerVersion() (*version.Info, error) {
	return d.delegate.ServerVersion()
}

func (d *Client) SwaggerSchema(version schema.GroupVersion) (*swagger.ApiDeclaration, error) {
	return d.delegate.SwaggerSchema(version)
}

func (d *Client) OpenAPISchema() (*spec.Swagger, error) {
	return d.delegate.OpenAPISchema()
}

func (d *Client) Fresh() bool {
	d.lock.RLock()
	defer d.lock.RUnlock()
	// Fresh is supposed to tell the caller whether or not to retry if the
	// cache fails to find something. The idea here is that Refresh and/or
	// Invalidate will be called periodically and therefore we'll always be
	// returning the latest data. (And in the future we can watch and stay
	// even more up-to-date.) So we only return false if the cache has
	// never been filled.
	return d.cacheValid
}

// Invalidate refreshes the cache, blocking calls until the cache has been
// refreshed. It would be trivial to make a version that does this in the
// background while continuing to respond to requests if needed.
func (d *Client) Invalidate() {
	d.lock.Lock()
	defer d.lock.Unlock()
	gl, err := d.delegate.ServerGroups()
	if err != nil || len(gl.Groups) == 0 {
		glog.V(2).Infof("Error getting current server API group list, will keep using cached value. (%v)", err)
		return
	}

	rl := map[string]*metav1.APIResourceList{}
	for _, g := range gl.Groups {
		for _, v := range g.Versions {
			r, err := d.delegate.ServerResourcesForGroupVersion(v.GroupVersion)
			if err != nil || len(r.APIResources) == 0 {
				glog.V(2).Infof("Error getting resource list for %v: %v", v.GroupVersion, err)
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

// NewClient creates a new Client which caches discovery information in memory
// and will stay up-to-date if Invalidate is called with regularity.
func NewClient(delegate discovery.DiscoveryInterface) *Client {
	return &Client{
		delegate:               delegate,
		groupToServerResources: map[string]*metav1.APIResourceList{},
	}
}
