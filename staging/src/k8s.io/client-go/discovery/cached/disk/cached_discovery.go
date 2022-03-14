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

package disk

import (
	"context"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	openapi_v2 "github.com/googleapis/gnostic/openapiv2"
	"k8s.io/klog/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
)

// CachedDiscoveryClient implements the functions that discovery server-supported API groups,
// versions and resources.
type CachedDiscoveryClient struct {
	delegate discovery.DiscoveryInterface

	// cacheDirectory is the directory where discovery docs are held.  It must be unique per host:port combination to work well.
	cacheDirectory string

	// mutex protects the variables below
	mutex sync.Mutex

	// ourFiles are all filenames of cache files created by this process
	ourFiles map[string]struct{}
	// invalidated is true if all cache files should be ignored that are not ours (e.g. after Invalidate() was called)
	invalidated bool
	// fresh is true if all used cache files were ours
	fresh bool
	// etag is the calculated etag value of whole openAPI spec.
	// If something is changed in openAPI spec, this will be used to invalidate cache.
	etag string
	// openAPIEndpoint is used for discriminate between openapi/v2 or openapi/v3 endpoints are enabled in cluster.
	// Eventually after moving to openapi/v3, there will be no more need for this.
	openAPIEndpoint string
}

var _ discovery.CachedDiscoveryInterface = &CachedDiscoveryClient{}

// ServerResourcesForGroupVersion returns the supported resources for a group and version.
func (d *CachedDiscoveryClient) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	filename := filepath.Join(d.cacheDirectory, groupVersion, "serverresources.json")
	if change, err := d.checkETagChange(); !change && err == nil {
		cachedBytes, err := d.getCachedFile(filename)
		// don't fail on errors, we either don't have a file or won't be able to run the cached check. Either way we can fallback.
		if err == nil {
			cachedResources := &metav1.APIResourceList{}
			if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), cachedBytes, cachedResources); err == nil {
				klog.V(10).Infof("returning cached discovery info from %v", filename)
				return cachedResources, nil
			}
		}
	}

	liveResources, err := d.delegate.ServerResourcesForGroupVersion(groupVersion)
	if err != nil {
		klog.V(3).Infof("skipped caching discovery info due to %v", err)
		return liveResources, err
	}
	if liveResources == nil || len(liveResources.APIResources) == 0 {
		klog.V(3).Infof("skipped caching discovery info, no resources found")
		return liveResources, err
	}

	if err := d.writeCachedFile(filename, liveResources); err != nil {
		klog.V(1).Infof("failed to write cache to %v due to %v", filename, err)
	}

	return liveResources, nil
}

// ServerResources returns the supported resources for all groups and versions.
// Deprecated: use ServerGroupsAndResources instead.
func (d *CachedDiscoveryClient) ServerResources() ([]*metav1.APIResourceList, error) {
	_, rs, err := discovery.ServerGroupsAndResources(d)
	return rs, err
}

// ServerGroupsAndResources returns the supported groups and resources for all groups and versions.
func (d *CachedDiscoveryClient) ServerGroupsAndResources() ([]*metav1.APIGroup, []*metav1.APIResourceList, error) {
	return discovery.ServerGroupsAndResources(d)
}

// ServerGroups returns the supported groups, with information like supported versions and the
// preferred version.
func (d *CachedDiscoveryClient) ServerGroups() (*metav1.APIGroupList, error) {
	filename := filepath.Join(d.cacheDirectory, "servergroups.json")
	if change, err := d.checkETagChange(); !change && err == nil {
		cachedBytes, err := d.getCachedFile(filename)
		// don't fail on errors, we either don't have a file or won't be able to run the cached check. Either way we can fallback.
		if err == nil {
			cachedGroups := &metav1.APIGroupList{}
			if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), cachedBytes, cachedGroups); err == nil {
				klog.V(10).Infof("returning cached discovery info from %v", filename)
				return cachedGroups, nil
			}
		}
	}

	liveGroups, err := d.delegate.ServerGroups()
	if err != nil {
		klog.V(3).Infof("skipped caching discovery info due to %v", err)
		return liveGroups, err
	}
	if liveGroups == nil || len(liveGroups.Groups) == 0 {
		klog.V(3).Infof("skipped caching discovery info, no groups found")
		return liveGroups, err
	}

	if err := d.writeCachedFile(filename, liveGroups); err != nil {
		klog.V(1).Infof("failed to write cache to %v due to %v", filename, err)
	}

	return liveGroups, nil
}

func (d *CachedDiscoveryClient) getCachedFile(filename string) ([]byte, error) {
	// after invalidation ignore cache files not created by this process
	d.mutex.Lock()
	_, ourFile := d.ourFiles[filename]
	if d.invalidated && !ourFile {
		d.mutex.Unlock()
		return nil, errors.New("cache invalidated")
	}
	d.mutex.Unlock()

	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// the cache is present and its valid.  Try to read and use it.
	cachedBytes, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	d.mutex.Lock()
	defer d.mutex.Unlock()
	d.fresh = d.fresh && ourFile

	return cachedBytes, nil
}

func (d *CachedDiscoveryClient) writeCachedFile(filename string, obj runtime.Object) error {
	if err := os.MkdirAll(filepath.Dir(filename), 0750); err != nil {
		return err
	}

	bytes, err := runtime.Encode(scheme.Codecs.LegacyCodec(), obj)
	if err != nil {
		return err
	}

	f, err := ioutil.TempFile(filepath.Dir(filename), filepath.Base(filename)+".")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	_, err = f.Write(bytes)
	if err != nil {
		return err
	}

	err = os.Chmod(f.Name(), 0660)
	if err != nil {
		return err
	}

	name := f.Name()
	err = f.Close()
	if err != nil {
		return err
	}

	// atomic rename
	d.mutex.Lock()
	defer d.mutex.Unlock()
	err = os.Rename(name, filename)
	if err == nil {
		d.ourFiles[filename] = struct{}{}
	}
	return err
}

func (d *CachedDiscoveryClient) writeEtag(filename, data string) error {
	if data == "" {
		return nil
	}

	if err := os.MkdirAll(filepath.Dir(filename), 0750); err != nil {
		return err
	}

	f, err := ioutil.TempFile(filepath.Dir(filename), filepath.Base(filename)+".")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	_, err = f.Write([]byte(data))
	if err != nil {
		return err
	}

	err = os.Chmod(f.Name(), 0660)
	if err != nil {
		return err
	}

	name := f.Name()
	err = f.Close()
	if err != nil {
		return err
	}

	// atomic rename
	d.mutex.Lock()
	defer d.mutex.Unlock()
	err = os.Rename(name, filename)
	d.etag = data

	return err
}

func (d *CachedDiscoveryClient) getEtag(filename string) string {
	if d.etag != "" {
		return d.etag
	}

	if d.invalidated {
		return ""
	}

	d.mutex.Lock()
	defer d.mutex.Unlock()
	file, err := os.Open(filename)
	if err != nil {
		return ""
	}
	defer file.Close()

	cachedEtagBytes, err := ioutil.ReadAll(file)
	if err != nil {
		return ""
	}
	d.fresh = true

	return string(cachedEtagBytes)
}

func (d *CachedDiscoveryClient) setOpenAPIEndpoint(e string) {
	d.mutex.Lock()
	defer d.mutex.Unlock()
	d.openAPIEndpoint = e
}

func (d *CachedDiscoveryClient) setEtag(etag string) {
	d.mutex.Lock()
	defer d.mutex.Unlock()
	d.etag = etag
}

// checkETagChange checks the stored in memory or in local file etag
// with the up to date etag which is gotten from open API endpoint.
// If the etag is not changed, openapi endpoint will return 304 instead
// all data. Thus, we can safely rely on our local files.
// If the etag is changed, openAPI endpoint will return 200 status code
// and we need to invalidate all local cache to re-fetch everything from scratch.
func (d *CachedDiscoveryClient) checkETagChange() (bool, error) {
	if d.openAPIEndpoint == "" {
		openapiEndpointFile := filepath.Join(d.cacheDirectory, "openapi_endpoint.json")
		d.setOpenAPIEndpoint(d.getEtag(openapiEndpointFile))

		if d.openAPIEndpoint == "" {
			res, err := d.RESTClient().Get().AbsPath("/").Do(context.TODO()).Raw()
			if err != nil {
				return true, err
			}
			if strings.Contains(string(res), "/openapi/v3") {
				d.setOpenAPIEndpoint("v3")
			} else if strings.Contains(string(res), "/openapi/v2") {
				d.setOpenAPIEndpoint("v2")
			} else {
				klog.V(10).Infof("openapi endpoint is not enabled for this cluster")
				return true, nil
			}
			err = d.writeEtag(openapiEndpointFile, d.openAPIEndpoint)
			if err != nil {
				klog.V(10).Infof("openapi endpoint file write error", err)
				return true, err
			}
		}
	}

	client := d.RESTClient().(*restclient.RESTClient).Client
	u := d.RESTClient().Get().AbsPath("openapi", d.openAPIEndpoint).URL()
	req, err := http.NewRequest("GET", u.String(), nil)
	if err != nil {
		return true, err
	}

	etags := filepath.Join(d.cacheDirectory, "etags.json")
	d.setEtag(d.getEtag(etags))

	if d.etag != "" {
		req.Header.Set("If-None-Match", fmt.Sprintf(`"%s"`, d.etag))
	}

	resp, err := client.Do(req)
	if err != nil {
		klog.Infof("openapi spec retrieval error", err)
		return true, err
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusNotModified:
		return false, nil
	case http.StatusOK:
		{
			if newEtag := resp.Header.Get("Etag"); newEtag != "" {
				err := d.writeEtag(etags, strings.Trim(newEtag, `"`))
				if err != nil {
					klog.V(10).Infof("etag file write error", err)
					return true, err
				}
			}
			return true, nil
		}
	default:
		klog.V(10).Infof("%s endpoint returned unknown status %d", d.openAPIEndpoint, resp.StatusCode)
		return true, nil
	}
}

// RESTClient returns a RESTClient that is used to communicate with API server
// by this client implementation.
func (d *CachedDiscoveryClient) RESTClient() restclient.Interface {
	return d.delegate.RESTClient()
}

// ServerPreferredResources returns the supported resources with the version preferred by the
// server.
func (d *CachedDiscoveryClient) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	return discovery.ServerPreferredResources(d)
}

// ServerPreferredNamespacedResources returns the supported namespaced resources with the
// version preferred by the server.
func (d *CachedDiscoveryClient) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	return discovery.ServerPreferredNamespacedResources(d)
}

// ServerVersion retrieves and parses the server's version (git version).
func (d *CachedDiscoveryClient) ServerVersion() (*version.Info, error) {
	return d.delegate.ServerVersion()
}

// OpenAPISchema retrieves and parses the swagger API schema the server supports.
func (d *CachedDiscoveryClient) OpenAPISchema() (*openapi_v2.Document, error) {
	return d.delegate.OpenAPISchema()
}

// Fresh is supposed to tell the caller whether or not to retry if the cache
// fails to find something (false = retry, true = no need to retry).
func (d *CachedDiscoveryClient) Fresh() bool {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	return d.fresh
}

// Invalidate enforces that no cached data is used in the future that is older than the current time.
func (d *CachedDiscoveryClient) Invalidate() {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	d.ourFiles = map[string]struct{}{}
	d.fresh = true
	d.invalidated = true
	d.etag = ""
	d.openAPIEndpoint = ""
}

// NewCachedDiscoveryClientForConfig creates a new DiscoveryClient for the given config, and wraps
// the created client in a CachedDiscoveryClient. The provided configuration is updated with a
// custom transport that understands cache responses.
// We receive two distinct cache directories for now, in order to preserve old behavior
// which makes use of the --cache-dir flag value for storing cache data from the CacheRoundTripper,
// and makes use of the hardcoded destination (~/.kube/cache/discovery/...) for storing
// CachedDiscoveryClient cache data. If httpCacheDir is empty, the restconfig's transport will not
// be updated with a roundtripper that understands cache responses.
// If discoveryCacheDir is empty, cached server resource data will be looked up in the current directory.
func NewCachedDiscoveryClientForConfig(config *restclient.Config, discoveryCacheDir string) (*CachedDiscoveryClient, error) {
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		return nil, err
	}

	return newCachedDiscoveryClient(discoveryClient, discoveryCacheDir), nil
}

// NewCachedDiscoveryClient creates a new DiscoveryClient.  cacheDirectory is the directory where discovery docs are held.  It must be unique per host:port combination to work well.
func newCachedDiscoveryClient(delegate discovery.DiscoveryInterface, cacheDirectory string) *CachedDiscoveryClient {
	return &CachedDiscoveryClient{
		delegate:       delegate,
		cacheDirectory: cacheDirectory,
		ourFiles:       map[string]struct{}{},
		fresh:          true,
	}
}
