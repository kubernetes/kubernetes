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
	"encoding/json"
	"fmt"
	"net/url"
	"sort"
	"strings"

	"github.com/emicklei/go-restful-swagger12"
	"github.com/golang/protobuf/proto"
	"github.com/googleapis/gnostic/OpenAPIv2"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
)

// defaultRetries is the number of times a resource discovery is repeated if an api group disappears on the fly (e.g. ThirdPartyResources).
const defaultRetries = 2

// DiscoveryInterface holds the methods that discover server-supported API groups,
// versions and resources.
type DiscoveryInterface interface {
	RESTClient() restclient.Interface
	ServerGroupsInterface
	ServerResourcesInterface
	ServerVersionInterface
	SwaggerSchemaInterface
	OpenAPISchemaInterface
}

// CachedDiscoveryInterface is a DiscoveryInterface with cache invalidation and freshness.
type CachedDiscoveryInterface interface {
	DiscoveryInterface
	// Fresh is supposed to tell the caller whether or not to retry if the cache
	// fails to find something (false = retry, true = no need to retry).
	//
	// TODO: this needs to be revisited, this interface can't be locked properly
	// and doesn't make a lot of sense.
	Fresh() bool
	// Invalidate enforces that no cached data is used in the future that is older than the current time.
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
	ServerResources() ([]*metav1.APIResourceList, error)
	// ServerPreferredResources returns the supported resources with the version preferred by the
	// server.
	ServerPreferredResources() ([]*metav1.APIResourceList, error)
	// ServerPreferredNamespacedResources returns the supported namespaced resources with the
	// version preferred by the server.
	ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error)
}

// ServerVersionInterface has a method for retrieving the server's version.
type ServerVersionInterface interface {
	// ServerVersion retrieves and parses the server's version (git version).
	ServerVersion() (*version.Info, error)
}

// SwaggerSchemaInterface has a method to retrieve the swagger schema.
type SwaggerSchemaInterface interface {
	// SwaggerSchema retrieves and parses the swagger API schema the server supports.
	SwaggerSchema(version schema.GroupVersion) (*swagger.ApiDeclaration, error)
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
	err = d.restClient.Get().AbsPath(d.LegacyPrefix).Do().Into(v)
	apiGroup := metav1.APIGroup{}
	if err == nil && len(v.Versions) != 0 {
		apiGroup = apiVersionsToAPIGroup(v)
	}
	if err != nil && !errors.IsNotFound(err) && !errors.IsForbidden(err) {
		return nil, err
	}

	// Get the groupVersions exposed at /apis
	apiGroupList = &metav1.APIGroupList{}
	err = d.restClient.Get().AbsPath("/apis").Do().Into(apiGroupList)
	if err != nil && !errors.IsNotFound(err) && !errors.IsForbidden(err) {
		return nil, err
	}
	// to be compatible with a v1.0 server, if it's a 403 or 404, ignore and return whatever we got from /api
	if err != nil && (errors.IsNotFound(err) || errors.IsForbidden(err)) {
		apiGroupList = &metav1.APIGroupList{}
	}

	// append the group retrieved from /api to the list if not empty
	if len(v.Versions) != 0 {
		apiGroupList.Groups = append(apiGroupList.Groups, apiGroup)
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
	err = d.restClient.Get().AbsPath(url.String()).Do().Into(resources)
	if err != nil {
		// ignore 403 or 404 error to be compatible with an v1.0 server.
		if groupVersion == "v1" && (errors.IsNotFound(err) || errors.IsForbidden(err)) {
			return resources, nil
		}
		return nil, err
	}
	return resources, nil
}

// serverResources returns the supported resources for all groups and versions.
func (d *DiscoveryClient) serverResources() ([]*metav1.APIResourceList, error) {
	apiGroups, err := d.ServerGroups()
	if err != nil {
		return nil, err
	}

	result := []*metav1.APIResourceList{}
	failedGroups := make(map[schema.GroupVersion]error)

	for _, apiGroup := range apiGroups.Groups {
		for _, version := range apiGroup.Versions {
			gv := schema.GroupVersion{Group: apiGroup.Name, Version: version.Version}
			resources, err := d.ServerResourcesForGroupVersion(version.GroupVersion)
			if err != nil {
				// TODO: maybe restrict this to NotFound errors
				failedGroups[gv] = err
				continue
			}

			result = append(result, resources)
		}
	}

	if len(failedGroups) == 0 {
		return result, nil
	}

	return result, &ErrGroupDiscoveryFailed{Groups: failedGroups}
}

// ServerResources returns the supported resources for all groups and versions.
func (d *DiscoveryClient) ServerResources() ([]*metav1.APIResourceList, error) {
	return withRetries(defaultRetries, d.serverResources)
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

// serverPreferredResources returns the supported resources with the version preferred by the server.
func (d *DiscoveryClient) serverPreferredResources() ([]*metav1.APIResourceList, error) {
	serverGroupList, err := d.ServerGroups()
	if err != nil {
		return nil, err
	}

	result := []*metav1.APIResourceList{}
	failedGroups := make(map[schema.GroupVersion]error)

	grVersions := map[schema.GroupResource]string{}                         // selected version of a GroupResource
	grApiResources := map[schema.GroupResource]*metav1.APIResource{}        // selected APIResource for a GroupResource
	gvApiResourceLists := map[schema.GroupVersion]*metav1.APIResourceList{} // blueprint for a APIResourceList for later grouping

	for _, apiGroup := range serverGroupList.Groups {
		for _, version := range apiGroup.Versions {
			groupVersion := schema.GroupVersion{Group: apiGroup.Name, Version: version.Version}
			apiResourceList, err := d.ServerResourcesForGroupVersion(version.GroupVersion)
			if err != nil {
				// TODO: maybe restrict this to NotFound errors
				failedGroups[groupVersion] = err
				continue
			}

			// create empty list which is filled later in another loop
			emptyApiResourceList := metav1.APIResourceList{
				GroupVersion: version.GroupVersion,
			}
			gvApiResourceLists[groupVersion] = &emptyApiResourceList
			result = append(result, &emptyApiResourceList)

			for i := range apiResourceList.APIResources {
				apiResource := &apiResourceList.APIResources[i]
				if strings.Contains(apiResource.Name, "/") {
					continue
				}
				gv := schema.GroupResource{Group: apiGroup.Name, Resource: apiResource.Name}
				if _, ok := grApiResources[gv]; ok && version.Version != apiGroup.PreferredVersion.Version {
					// only override with preferred version
					continue
				}
				grVersions[gv] = version.Version
				grApiResources[gv] = apiResource
			}
		}
	}

	// group selected APIResources according to GroupVersion into APIResourceLists
	for groupResource, apiResource := range grApiResources {
		version := grVersions[groupResource]
		groupVersion := schema.GroupVersion{Group: groupResource.Group, Version: version}
		apiResourceList := gvApiResourceLists[groupVersion]
		apiResourceList.APIResources = append(apiResourceList.APIResources, *apiResource)
	}

	if len(failedGroups) == 0 {
		return result, nil
	}

	return result, &ErrGroupDiscoveryFailed{Groups: failedGroups}
}

// ServerPreferredResources returns the supported resources with the version preferred by the
// server.
func (d *DiscoveryClient) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	return withRetries(defaultRetries, d.serverPreferredResources)
}

// ServerPreferredNamespacedResources returns the supported namespaced resources with the
// version preferred by the server.
func (d *DiscoveryClient) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	all, err := d.ServerPreferredResources()
	return FilteredBy(ResourcePredicateFunc(func(groupVersion string, r *metav1.APIResource) bool {
		return r.Namespaced
	}), all), err
}

// ServerVersion retrieves and parses the server's version (git version).
func (d *DiscoveryClient) ServerVersion() (*version.Info, error) {
	body, err := d.restClient.Get().AbsPath("/version").Do().Raw()
	if err != nil {
		return nil, err
	}
	var info version.Info
	err = json.Unmarshal(body, &info)
	if err != nil {
		return nil, fmt.Errorf("got '%s': %v", string(body), err)
	}
	return &info, nil
}

// SwaggerSchema retrieves and parses the swagger API schema the server supports.
// TODO: Replace usages with Open API.  Tracked in https://github.com/kubernetes/kubernetes/issues/44589
func (d *DiscoveryClient) SwaggerSchema(version schema.GroupVersion) (*swagger.ApiDeclaration, error) {
	if version.Empty() {
		return nil, fmt.Errorf("groupVersion cannot be empty")
	}

	groupList, err := d.ServerGroups()
	if err != nil {
		return nil, err
	}
	groupVersions := metav1.ExtractGroupVersions(groupList)
	// This check also takes care the case that kubectl is newer than the running endpoint
	if stringDoesntExistIn(version.String(), groupVersions) {
		return nil, fmt.Errorf("API version: %v is not supported by the server. Use one of: %v", version, groupVersions)
	}
	var path string
	if len(d.LegacyPrefix) > 0 && version == v1.SchemeGroupVersion {
		path = "/swaggerapi" + d.LegacyPrefix + "/" + version.Version
	} else {
		path = "/swaggerapi/apis/" + version.Group + "/" + version.Version
	}

	body, err := d.restClient.Get().AbsPath(path).Do().Raw()
	if err != nil {
		return nil, err
	}
	var schema swagger.ApiDeclaration
	err = json.Unmarshal(body, &schema)
	if err != nil {
		return nil, fmt.Errorf("got '%s': %v", string(body), err)
	}
	return &schema, nil
}

// OpenAPISchema fetches the open api schema using a rest client and parses the proto.
func (d *DiscoveryClient) OpenAPISchema() (*openapi_v2.Document, error) {
	data, err := d.restClient.Get().AbsPath("/swagger-2.0.0.pb-v1").Do().Raw()
	if err != nil {
		return nil, err
	}
	document := &openapi_v2.Document{}
	err = proto.Unmarshal(data, document)
	if err != nil {
		return nil, err
	}
	return document, nil
}

// withRetries retries the given recovery function in case the groups supported by the server change after ServerGroup() returns.
func withRetries(maxRetries int, f func() ([]*metav1.APIResourceList, error)) ([]*metav1.APIResourceList, error) {
	var result []*metav1.APIResourceList
	var err error
	for i := 0; i < maxRetries; i++ {
		result, err = f()
		if err == nil {
			return result, nil
		}
		if _, ok := err.(*ErrGroupDiscoveryFailed); !ok {
			return nil, err
		}
	}
	return result, err
}

func setDiscoveryDefaults(config *restclient.Config) error {
	config.APIPath = ""
	config.GroupVersion = nil
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

// NewDiscoveryClient returns  a new DiscoveryClient for the given RESTClient.
func NewDiscoveryClient(c restclient.Interface) *DiscoveryClient {
	return &DiscoveryClient{restClient: c, LegacyPrefix: "/api"}
}

func stringDoesntExistIn(str string, slice []string) bool {
	for _, s := range slice {
		if s == str {
			return false
		}
	}
	return true
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *DiscoveryClient) RESTClient() restclient.Interface {
	if c == nil {
		return nil
	}
	return c.restClient
}
