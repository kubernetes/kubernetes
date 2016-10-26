/*
Copyright 2016 The Kubernetes Authors.

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

package util

import (
	"errors"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"

	"github.com/emicklei/go-restful/swagger"
	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/version"
)

// CachedDiscoveryClient implements the functions that discovery server-supported API groups,
// versions and resources.
type CachedDiscoveryClient struct {
	delegate discovery.DiscoveryInterface

	// cacheDirectory is the directory where discovery docs are held.  It must be unique per host:port combination to work well.
	cacheDirectory string

	// ttl is how long the cache should be considered valid
	ttl time.Duration

	// startedAt is the time this client was created
	startedAt time.Time

	// age is the time of the oldest used cache file, zero if no cache was used
	age time.Time

	// every cache file before this timestamp will be ignored
	lastInvalidation time.Time
}

var _ discovery.CachedDiscoveryInterface = &CachedDiscoveryClient{}

// ServerResourcesForGroupVersion returns the supported resources for a group and version.
func (d *CachedDiscoveryClient) ServerResourcesForGroupVersion(groupVersion string) (*unversioned.APIResourceList, error) {
	filename := filepath.Join(d.cacheDirectory, groupVersion, "serverresources.json")
	cachedBytes, err := d.getCachedFile(filename)
	// don't fail on errors, we either don't have a file or won't be able to run the cached check. Either way we can fallback.
	if err == nil {
		cachedResources := &unversioned.APIResourceList{}
		if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), cachedBytes, cachedResources); err == nil {
			glog.V(6).Infof("returning cached discovery info from %v", filename)
			return cachedResources, nil
		}
	}

	liveResources, err := d.delegate.ServerResourcesForGroupVersion(groupVersion)
	if err != nil {
		return liveResources, err
	}

	if err := d.writeCachedFile(filename, liveResources); err != nil {
		glog.V(3).Infof("failed to write cache to %v due to %v", filename, err)
	}

	return liveResources, nil
}

// ServerResources returns the supported resources for all groups and versions.
func (d *CachedDiscoveryClient) ServerResources() (map[string]*unversioned.APIResourceList, error) {
	apiGroups, err := d.ServerGroups()
	if err != nil {
		return nil, err
	}
	groupVersions := unversioned.ExtractGroupVersions(apiGroups)
	result := map[string]*unversioned.APIResourceList{}
	for _, groupVersion := range groupVersions {
		resources, err := d.ServerResourcesForGroupVersion(groupVersion)
		if err != nil {
			return nil, err
		}
		result[groupVersion] = resources
	}
	return result, nil
}

func (d *CachedDiscoveryClient) ServerGroups() (*unversioned.APIGroupList, error) {
	filename := filepath.Join(d.cacheDirectory, "servergroups.json")
	cachedBytes, err := d.getCachedFile(filename)
	// don't fail on errors, we either don't have a file or won't be able to run the cached check. Either way we can fallback.
	if err == nil {
		cachedGroups := &unversioned.APIGroupList{}
		if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), cachedBytes, cachedGroups); err == nil {
			glog.V(6).Infof("returning cached discovery info from %v", filename)
			return cachedGroups, nil
		}
	}

	liveGroups, err := d.delegate.ServerGroups()
	if err != nil {
		return liveGroups, err
	}

	if err := d.writeCachedFile(filename, liveGroups); err != nil {
		glog.V(3).Infof("failed to write cache to %v due to %v", filename, err)
	}

	return liveGroups, nil
}

func (d *CachedDiscoveryClient) getCachedFile(filename string) ([]byte, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	fileInfo, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if fileInfo.ModTime().Before(d.lastInvalidation) {
		return nil, errors.New("cache invalidated")
	}
	if time.Now().After(fileInfo.ModTime().Add(d.ttl)) {
		return nil, errors.New("cache expired")
	}

	if d.age.IsZero() || d.age.After(fileInfo.ModTime()) {
		d.age = fileInfo.ModTime()
	}

	// the cache is present and its valid.  Try to read and use it.
	cachedBytes, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	return cachedBytes, nil
}

func (d *CachedDiscoveryClient) writeCachedFile(filename string, obj runtime.Object) error {
	if err := os.MkdirAll(filepath.Dir(filename), 0755); err != nil {
		return err
	}

	bytes, err := runtime.Encode(api.Codecs.LegacyCodec(), obj)
	if err != nil {
		return err
	}

	return ioutil.WriteFile(filename, bytes, 0755)
}

func (d *CachedDiscoveryClient) RESTClient() restclient.Interface {
	return d.delegate.RESTClient()
}

func (d *CachedDiscoveryClient) ServerPreferredResources() ([]unversioned.GroupVersionResource, error) {
	return d.delegate.ServerPreferredResources()
}

func (d *CachedDiscoveryClient) ServerPreferredNamespacedResources() ([]unversioned.GroupVersionResource, error) {
	return d.delegate.ServerPreferredNamespacedResources()
}

func (d *CachedDiscoveryClient) ServerVersion() (*version.Info, error) {
	return d.delegate.ServerVersion()
}

func (d *CachedDiscoveryClient) SwaggerSchema(version unversioned.GroupVersion) (*swagger.ApiDeclaration, error) {
	return d.delegate.SwaggerSchema(version)
}

func (d* CachedDiscoveryClient) Fresh() bool {
	return d.age.IsZero() || d.age.After(d.startedAt)
}

func (d *CachedDiscoveryClient) Invalidate() {
	d.lastInvalidation = time.Now()
	d.age = time.Time{}
}

// NewCachedDiscoveryClient creates a new DiscoveryClient.  cacheDirectory is the directory where discovery docs are held.  It must be unique per host:port combination to work well.
func NewCachedDiscoveryClient(delegate discovery.DiscoveryInterface, cacheDirectory string, ttl time.Duration) *CachedDiscoveryClient {
	return &CachedDiscoveryClient{delegate: delegate, cacheDirectory: cacheDirectory, ttl: ttl, startedAt: time.Now()}
}
