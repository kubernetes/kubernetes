/*
Copyright 2015 The Kubernetes Authors.

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

package discovery

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/golang/protobuf/proto"
	openapi_v2 "github.com/googleapis/gnostic/openapiv2"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
)

const (
	// defaultRetries is the number of times a resource discovery is repeated if an api group disappears on the fly (e.g. ThirdPartyResources).
	defaultRetries = 2
	// protobuf mime type
	mimePb = "application/com.github.proto-openapi.spec.v2@v1.0+protobuf"
	// defaultTimeout is the maximum amount of time per request when no timeout has been set on a RESTClient.
	// Defaults to 32s in order to have a distinguishable length of time, relative to other timeouts that exist.
	defaultTimeout = 32 * time.Second
)

// DiscoveryInterface holds the methods that discover server-supported API groups,
// versions and resources.
type DiscoveryInterface interface {
	RESTClient() restclient.Interface
	ServerGroupsInterface
	ServerResourcesInterface
	ServerVersionInterface
	OpenAPISchemaInterface
}

// CachedDiscoveryInterface is a DiscoveryInterface with cache invalidation and freshness.
// Note that If the ServerResourcesForGroupVersion method returns a cache miss
// error, the user needs to explicitly call Invalidate to clear the cache,
// otherwise the same cache miss error will be returned next time.
type CachedDiscoveryInterface interface {
	DiscoveryInterface
	// Fresh is supposed to tell the caller whether or not to retry if the cache
	// fails to find something (false = retry, true = no need to retry).
	//
	// TODO: this needs to be revisited, this interface can't be locked properly
	// and doesn't make a lot of sense.
	Fresh() bool
	// Invalidate enforces that no cached data that is older than the current time
	// is used.
	Invalidate()
}

// ServerGroupsInterface has methods for obtaining supported groups on the API server
type ServerGroupsInterface interface {
	// ServerGroups returns the supported groups, with information like supported versions and the
	// preferred version.
	ServerGroups() (*metav1.APIGroupList, error)
}

// ServerResourcesInterface has methods for obtaining supported resources on the API server
type ServerResourcesInterface interface {
	// ServerResourcesForGroupVersion returns the supported resources for a group and version.
	ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error)
	// ServerResources returns the supported resources for all groups and versions.
	//
	// The returned resource list might be non-nil with partial results even in the case of
	// non-nil error.
	//
	// Deprecated: use ServerGroupsAndResources instead.
	ServerResources() ([]*metav1.APIResourceList, error)
	// ServerResources returns the supported groups and resources for all groups and versions.
	//
	// The returned group and resource lists might be non-nil with partial results even in the
	// case of non-nil error.
	ServerGroupsAndResources() ([]*metav1.APIGroup, []*metav1.APIResourceList, error)
	// ServerPreferredResources returns the supported resources with the version preferred by the
	// server.
	//
	// The returned group and resource lists might be non-nil with partial results even in the
	// case of non-nil error.
	ServerPreferredResources() ([]*metav1.APIResourceList, error)
	// ServerPreferredNamespacedResources returns the supported namespaced resources with the
	// version preferred by the server.
	//
	// The returned resource list might be non-nil with partial results even in the case of
	// non-nil error.
	ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error)
}

// ServerVersionInterface has a method for retrieving the server's version.
type ServerVersionInterface interface {
	// ServerVersion retrieves and parses the server's version (git version).
	ServerVersion() (*version.Info, error)
}

// OpenAPISchemaInterface has a method to retrieve the open API schema.
type OpenAPISchemaInterface interface {
	// OpenAPISchema retrieves and parses the swagger API schema the server supports.
	OpenAPISchema() (*openapi_v2.Document, error)
}

// DiscoveryClient implements the functions that discover server-supported API groups,
// versions and resources.
type DiscoveryClient struct {
	restClient restclient.Interface

	LegacyPrefix string
}

// Convert metav1.APIVersions to metav1.APIGroup. APIVersions is used by legacy v1, so
// group would be "".
func apiVersionsToAPIGroup(apiVersions *metav1.APIVersions) (apiGroup metav1.APIGroup) {
	groupVersions := []metav1.GroupVersionForDiscovery{}
	for _, version := range apiVersions.Versions {
		groupVersion := metav1.GroupVersionForDiscovery{
			GroupVersion: version,
			Version:      version,
		}
		groupVersions = append(groupVersions, groupVersion)
	}
	apiGroup.Versions = groupVersions
	// There should be only one groupVersion returned at /api
	apiGroup.PreferredVersion = groupVersions[0]
	return
}

// ServerGroups returns the supported groups, with information like supported versions and the
// preferred version.
func (d *DiscoveryClient) ServerGroups() (apiGroupList *metav1.APIGroupList, err error) {
	// Get the groupVersions exposed at /api
	v := &metav1.APIVersions{}
	err = d.restClient.Get().AbsPath(d.LegacyPrefix).Do(context.TODO()).Into(v)
	apiGroup := metav1.APIGroup{}
	if err == nil && len(v.Versions) != 0 {
		apiGroup = apiVersionsToAPIGroup(v)
	}
	if err != nil && !errors.IsNotFound(err) && !errors.IsForbidden(err) {
		return nil, err
	}

	// Get the groupVersions exposed at /apis
	apiGroupList = &metav1.APIGroupList{}
	err = d.restClient.Get().AbsPath("/apis").Do(context.TODO()).Into(apiGroupList)
	if err != nil && !errors.IsNotFound(err) && !errors.IsForbidden(err) {
		return nil, err
	}
	// to be compatible with a v1.0 server, if it's a 403 or 404, ignore and return whatever we got from /api
	if err != nil && (errors.IsNotFound(err) || errors.IsForbidden(err)) {
		apiGroupList = &metav1.APIGroupList{}
	}

	// prepend the group retrieved from /api to the list if not empty
	if len(v.Versions) != 0 {
		apiGroupList.Groups = append([]metav1.APIGroup{apiGroup}, apiGroupList.Groups...)
	}
	return apiGroupList, nil
}

// ServerResourcesForGroupVersion returns the supported resources for a group and version.
func (d *DiscoveryClient) ServerResourcesForGroupVersion(groupVersion string) (resources *metav1.APIResourceList, err error) {
	url := url.URL{}
	if len(groupVersion) == 0 {
		return nil, fmt.Errorf("groupVersion shouldn't be empty")
	}
	if len(d.LegacyPrefix) > 0 && groupVersion == "v1" {
		url.Path = d.LegacyPrefix + "/" + groupVersion
	} else {
		url.Path = "/apis/" + groupVersion
	}
	resources = &metav1.APIResourceList{
		GroupVersion: groupVersion,
	}
	err = d.restClient.Get().AbsPath(url.String()).Do(context.TODO()).Into(resources)
	if err != nil {
		// ignore 403 or 404 error to be compatible with an v1.0 server.
		if groupVersion == "v1" && (errors.IsNotFound(err) || errors.IsForbidden(err)) {
			return resources, nil
		}
		return nil, err
	}
	return resources, nil
}

// ServerResources returns the supported resources for all groups and versions.
// Deprecated: use ServerGroupsAndResources instead.
func (d *DiscoveryClient) ServerResources() ([]*metav1.APIResourceList, error) {
	_, rs, err := d.ServerGroupsAndResources()
	return rs, err
}

// ServerGroupsAndResources returns the supported resources for all groups and versions.
func (d *DiscoveryClient) ServerGroupsAndResources() ([]*metav1.APIGroup, []*metav1.APIResourceList, error) {
	return withRetries(defaultRetries, func() ([]*metav1.APIGroup, []*metav1.APIResourceList, error) {
		return ServerGroupsAndResources(d)
	})
}

// ErrGroupDiscoveryFailed is returned if one or more API groups fail to load.
type ErrGroupDiscoveryFailed struct {
	// Groups is a list of the groups that failed to load and the error cause
	Groups map[schema.GroupVersion]error
}

// Error implements the error interface
func (e *ErrGroupDiscoveryFailed) Error() string {
	var groups []string
	for k, v := range e.Groups {
		groups = append(groups, fmt.Sprintf("%s: %v", k, v))
	}
	sort.Strings(groups)
	return fmt.Sprintf("unable to retrieve the complete list of server APIs: %s", strings.Join(groups, ", "))
}

// IsGroupDiscoveryFailedError returns true if the provided error indicates the server was unable to discover
// a complete list of APIs for the client to use.
func IsGroupDiscoveryFailedError(err error) bool {
	_, ok := err.(*ErrGroupDiscoveryFailed)
	return err != nil && ok
}

// ServerResources uses the provided discovery interface to look up supported resources for all groups and versions.
// Deprecated: use ServerGroupsAndResources instead.
func ServerResources(d DiscoveryInterface) ([]*metav1.APIResourceList, error) {
	_, rs, err := ServerGroupsAndResources(d)
	return rs, err
}

func ServerGroupsAndResources(d DiscoveryInterface) ([]*metav1.APIGroup, []*metav1.APIResourceList, error) {
	sgs, err := d.ServerGroups()
	if sgs == nil {
		return nil, nil, err
	}
	resultGroups := []*metav1.APIGroup{}
	for i := range sgs.Groups {
		resultGroups = append(resultGroups, &sgs.Groups[i])
	}

	groupVersionResources, failedGroups := fetchGroupVersionResources(d, sgs)

	// order results by group/version discovery order
	result := []*metav1.APIResourceList{}
	for _, apiGroup := range sgs.Groups {
		for _, version := range apiGroup.Versions {
			gv := schema.GroupVersion{Group: apiGroup.Name, Version: version.Version}
			if resources, ok := groupVersionResources[gv]; ok {
				result = append(result, resources)
			}
		}
	}

	if len(failedGroups) == 0 {
		return resultGroups, result, nil
	}

	return resultGroups, result, &ErrGroupDiscoveryFailed{Groups: failedGroups}
}

// ServerPreferredResources uses the provided discovery interface to look up preferred resources
func ServerPreferredResources(d DiscoveryInterface) ([]*metav1.APIResourceList, error) {
	serverGroupList, err := d.ServerGroups()
	if err != nil {
		return nil, err
	}

	groupVersionResources, failedGroups := fetchGroupVersionResources(d, serverGroupList)

	result := []*metav1.APIResourceList{}
	grVersions := map[schema.GroupResource]string{}                         // selected version of a GroupResource
	grAPIResources := map[schema.GroupResource]*metav1.APIResource{}        // selected APIResource for a GroupResource
	gvAPIResourceLists := map[schema.GroupVersion]*metav1.APIResourceList{} // blueprint for a APIResourceList for later grouping

	for _, apiGroup := range serverGroupList.Groups {
		for _, version := range apiGroup.Versions {
			groupVersion := schema.GroupVersion{Group: apiGroup.Name, Version: version.Version}

			apiResourceList, ok := groupVersionResources[groupVersion]
			if !ok {
				continue
			}

			// create empty list which is filled later in another loop
			emptyAPIResourceList := metav1.APIResourceList{
				GroupVersion: version.GroupVersion,
			}
			gvAPIResourceLists[groupVersion] = &emptyAPIResourceList
			result = append(result, &emptyAPIResourceList)

			for i := range apiResourceList.APIResources {
				apiResource := &apiResourceList.APIResources[i]
				if strings.Contains(apiResource.Name, "/") {
					continue
				}
				gv := schema.GroupResource{Group: apiGroup.Name, Resource: apiResource.Name}
				if _, ok := grAPIResources[gv]; ok && version.Version != apiGroup.PreferredVersion.Version {
					// only override with preferred version
					continue
				}
				grVersions[gv] = version.Version
				grAPIResources[gv] = apiResource
			}
		}
	}

	// group selected APIResources according to GroupVersion into APIResourceLists
	for groupResource, apiResource := range grAPIResources {
		version := grVersions[groupResource]
		groupVersion := schema.GroupVersion{Group: groupResource.Group, Version: version}
		apiResourceList := gvAPIResourceLists[groupVersion]
		apiResourceList.APIResources = append(apiResourceList.APIResources, *apiResource)
	}

	if len(failedGroups) == 0 {
		return result, nil
	}

	return result, &ErrGroupDiscoveryFailed{Groups: failedGroups}
}

// fetchServerResourcesForGroupVersions uses the discovery client to fetch the resources for the specified groups in parallel.
func fetchGroupVersionResources(d DiscoveryInterface, apiGroups *metav1.APIGroupList) (map[schema.GroupVersion]*metav1.APIResourceList, map[schema.GroupVersion]error) {
	groupVersionResources := make(map[schema.GroupVersion]*metav1.APIResourceList)
	failedGroups := make(map[schema.GroupVersion]error)

	wg := &sync.WaitGroup{}
	resultLock := &sync.Mutex{}
	for _, apiGroup := range apiGroups.Groups {
		for _, version := range apiGroup.Versions {
			groupVersion := schema.GroupVersion{Group: apiGroup.Name, Version: version.Version}
			wg.Add(1)
			go func() {
				defer wg.Done()
				defer utilruntime.HandleCrash()

				apiResourceList, err := d.ServerResourcesForGroupVersion(groupVersion.String())

				// lock to record results
				resultLock.Lock()
				defer resultLock.Unlock()

				if err != nil {
					// TODO: maybe restrict this to NotFound errors
					failedGroups[groupVersion] = err
				}
				if apiResourceList != nil {
					// even in case of error, some fallback might have been returned
					groupVersionResources[groupVersion] = apiResourceList
				}
			}()
		}
	}
	wg.Wait()

	return groupVersionResources, failedGroups
}

// ServerPreferredResources returns the supported resources with the version preferred by the
// server.
func (d *DiscoveryClient) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	_, rs, err := withRetries(defaultRetries, func() ([]*metav1.APIGroup, []*metav1.APIResourceList, error) {
		rs, err := ServerPreferredResources(d)
		return nil, rs, err
	})
	return rs, err
}

// ServerPreferredNamespacedResources returns the supported namespaced resources with the
// version preferred by the server.
func (d *DiscoveryClient) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	return ServerPreferredNamespacedResources(d)
}

// ServerPreferredNamespacedResources uses the provided discovery interface to look up preferred namespaced resources
func ServerPreferredNamespacedResources(d DiscoveryInterface) ([]*metav1.APIResourceList, error) {
	all, err := ServerPreferredResources(d)
	return FilteredBy(ResourcePredicateFunc(func(groupVersion string, r *metav1.APIResource) bool {
		return r.Namespaced
	}), all), err
}

// ServerVersion retrieves and parses the server's version (git version).
func (d *DiscoveryClient) ServerVersion() (*version.Info, error) {
	body, err := d.restClient.Get().AbsPath("/version").Do(context.TODO()).Raw()
	if err != nil {
		return nil, err
	}
	var info version.Info
	err = json.Unmarshal(body, &info)
	if err != nil {
		return nil, fmt.Errorf("unable to parse the server version: %v", err)
	}
	return &info, nil
}

// OpenAPISchema fetches the open api schema using a rest client and parses the proto.
func (d *DiscoveryClient) OpenAPISchema() (*openapi_v2.Document, error) {
	data, err := d.restClient.Get().AbsPath("/openapi/v2").SetHeader("Accept", mimePb).Do(context.TODO()).Raw()
	if err != nil {
		if errors.IsForbidden(err) || errors.IsNotFound(err) || errors.IsNotAcceptable(err) {
			// single endpoint not found/registered in old server, try to fetch old endpoint
			// TODO: remove this when kubectl/client-go don't work with 1.9 server
			data, err = d.restClient.Get().AbsPath("/swagger-2.0.0.pb-v1").Do(context.TODO()).Raw()
			if err != nil {
				return nil, err
			}
		} else {
			return nil, err
		}
	}
	document := &openapi_v2.Document{}
	err = proto.Unmarshal(data, document)
	if err != nil {
		return nil, err
	}
	return document, nil
}

// withRetries retries the given recovery function in case the groups supported by the server change after ServerGroup() returns.
func withRetries(maxRetries int, f func() ([]*metav1.APIGroup, []*metav1.APIResourceList, error)) ([]*metav1.APIGroup, []*metav1.APIResourceList, error) {
	var result []*metav1.APIResourceList
	var resultGroups []*metav1.APIGroup
	var err error
	for i := 0; i < maxRetries; i++ {
		resultGroups, result, err = f()
		if err == nil {
			return resultGroups, result, nil
		}
		if _, ok := err.(*ErrGroupDiscoveryFailed); !ok {
			return nil, nil, err
		}
	}
	return resultGroups, result, err
}

func setDiscoveryDefaults(config *restclient.Config) error {
	config.APIPath = ""
	config.GroupVersion = nil
	if config.Timeout == 0 {
		config.Timeout = defaultTimeout
	}
	if config.Burst == 0 && config.QPS < 100 {
		// discovery is expected to be bursty, increase the default burst
		// to accommodate looking up resource info for many API groups.
		// matches burst set by ConfigFlags#ToDiscoveryClient().
		// see https://issue.k8s.io/86149
		config.Burst = 100
	}
	codec := runtime.NoopEncoder{Decoder: scheme.Codecs.UniversalDecoder()}
	config.NegotiatedSerializer = serializer.NegotiatedSerializerWrapper(runtime.SerializerInfo{Serializer: codec})
	if len(config.UserAgent) == 0 {
		config.UserAgent = restclient.DefaultKubernetesUserAgent()
	}
	return nil
}

// NewDiscoveryClientForConfig creates a new DiscoveryClient for the given config. This client
// can be used to discover supported resources in the API server.
func NewDiscoveryClientForConfig(c *restclient.Config) (*DiscoveryClient, error) {
	config := *c
	if err := setDiscoveryDefaults(&config); err != nil {
		return nil, err
	}
	client, err := restclient.UnversionedRESTClientFor(&config)
	return &DiscoveryClient{restClient: client, LegacyPrefix: "/api"}, err
}

// NewDiscoveryClientForConfigOrDie creates a new DiscoveryClient for the given config. If
// there is an error, it panics.
func NewDiscoveryClientForConfigOrDie(c *restclient.Config) *DiscoveryClient {
	client, err := NewDiscoveryClientForConfig(c)
	if err != nil {
		panic(err)
	}
	return client

}

// NewDiscoveryClient returns a new DiscoveryClient for the given RESTClient.
func NewDiscoveryClient(c restclient.Interface) *DiscoveryClient {
	return &DiscoveryClient{restClient: c, LegacyPrefix: "/api"}
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (d *DiscoveryClient) RESTClient() restclient.Interface {
	if d == nil {
		return nil
	}
	return d.restClient
}
